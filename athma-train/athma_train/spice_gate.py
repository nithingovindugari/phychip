"""Strict ngspice gate — single source of truth.

Used by:
  - athma_train.env.SpiceEnv (RL reward at training time)
  - scripts/analogcoder_baseline.py (eval gate)
  - scripts/stage5_rlvr_train.py (RL reward at training time)

The train-time gate and eval-time gate MUST be identical, otherwise:
  - Model reward-hacks the training gate
  - Model gets surprising eval numbers
This module enforces that invariant.

v2 (2026-05-15) updates over v1:
  - Accept `.control / op / .endc` block syntax (Humphery7 dataset uses this)
  - Strip Humphery7-style tokenization (<NETLIST>, <NL>, <END_NETLIST>)
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path

# --- ngspice version guard (bug 2026-06-12) ---------------------------------
# AnalogCoder pass@1 is highly sensitive to the ngspice version: the SAME
# adapter scored 0/24 on ngspice-36 (Ubuntu apt default) and 16/24 on
# ngspice-46. Comparing scores across versions produced a FALSE "GRPO degrades
# OOD" conclusion. All evals and GRPO rewards MUST run on one pinned version.
# Set PHYCHIP_REQUIRE_NGSPICE (e.g. "46") to hard-enforce; otherwise warn once.
_NGSPICE_VERSION_CHECKED = False


def _check_ngspice_version() -> None:
    global _NGSPICE_VERSION_CHECKED
    if _NGSPICE_VERSION_CHECKED:
        return
    _NGSPICE_VERSION_CHECKED = True
    want = os.environ.get("PHYCHIP_REQUIRE_NGSPICE")
    try:
        out = subprocess.run(["ngspice", "--version"], capture_output=True,
                             text=True, timeout=10).stdout
        m = re.search(r"ngspice-(\d+)", out)
        got = m.group(1) if m else "unknown"
    except Exception:
        got = "unavailable"
    if want and got != want:
        raise RuntimeError(
            f"ngspice version mismatch: found {got}, PHYCHIP_REQUIRE_NGSPICE={want}. "
            f"Eval scores are version-sensitive (see EVAL_ENV_BUG_2026-06-12.md). "
            f"Build ngspice {want} from source; apt ships an old version.")
    if got in ("36", "unknown", "unavailable") and not want:
        warnings.warn(
            f"ngspice version is {got}; AnalogCoder scores are version-sensitive. "
            f"Pin PHYCHIP_REQUIRE_NGSPICE=46 for comparable results "
            f"(see EVAL_ENV_BUG_2026-06-12.md).", RuntimeWarning)


DEVICE_PREFIXES = ("r", "c", "l", "m", "q", "d", "v", "i", "x", "j")
ANALYSIS_DIRECTIVES = (".op", ".tran", ".dc", ".ac", ".noise", ".disto")
HARD_FAILURES = (
    "run simulation(s) aborted",
    "singular matrix", "segmentation fault", "fatal error",
    "no such device", "doanalyses: ", "timestep too small",
    "convergence problem",
)
# "no simulations run" is BENIGN when analysis actually ran but no .print
# directive consumed the output. Handled separately in grade_netlist().
ANALYSIS_RAN_MARKERS = (
    "doing analysis at temp",
    "no. of data rows",
    "transient analysis",
    "dc analysis",
    "ac analysis",
    "operating point",
)


def normalize_netlist(text: str) -> str:
    """Strip tokenization markers (Humphery7 / other tokenized formats) +
    markdown fences. Returns clean ngspice text."""
    s = text
    # Humphery7 tokenization
    s = s.replace("<NL>", "\n").replace("<NETLIST>", "").replace("<END_NETLIST>", "")
    # Markdown fences
    if "```" in s:
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1]
            if s.startswith("spice\n") or s.startswith("ngspice\n"):
                s = s.split("\n", 1)[1]
    # Truncate at .end (last occurrence)
    if ".end" in s.lower():
        idx = s.lower().rfind(".end")
        s = s[: idx + len(".end")]
    return s.strip()


def has_analysis_directive(netlist_lower: str) -> bool:
    """True if netlist contains an analysis directive OR a .control block
    with an analysis command inside.

    Standalone form: lines starting with .op / .tran / .dc / .ac / .noise / .disto
    Block form:      .control ... op / tran / dc / ac ... .endc
    """
    for line in netlist_lower.split("\n"):
        line = line.strip()
        for d in ANALYSIS_DIRECTIVES:
            if line.startswith(d):
                return True
    m = re.search(r"\.control\b(.*?)\.endc\b", netlist_lower, re.DOTALL)
    if m:
        body = m.group(1)
        for cmd in ("op", "tran", "dc ", "ac ", "ac\n", "noise", "disto"):
            if re.search(rf"^\s*{cmd}", body, re.MULTILINE):
                return True
    return False


def has_device_line(netlist: str) -> bool:
    """True if at least one non-comment, non-directive line starts with a
    device prefix (R/C/L/M/Q/D/V/I/X/J)."""
    for ln in netlist.split("\n"):
        ln = ln.strip()
        if not ln or ln.startswith("*") or ln.startswith("."):
            continue
        if ln[0].lower() in DEVICE_PREFIXES:
            return True
    return False


def grade_netlist(text: str, timeout_s: int = 20) -> tuple[bool, str]:
    """Strict ngspice gate.

    Returns (passed, log_or_reason). Passes iff:
      1. After normalization, has ≥1 device line.
      2. Has an analysis directive (standalone or in .control block).
      3. ngspice exits 0 (no warnings-only, no signal).
      4. Output does NOT contain any HARD_FAILURES marker.
    """
    _check_ngspice_version()
    netlist = normalize_netlist(text)
    if not has_device_line(netlist):
        return False, "no device lines in netlist"
    if not has_analysis_directive(netlist.lower()):
        return False, "no analysis directive (.op/.tran/.dc/.ac or .control block)"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cir", delete=False, dir="/tmp"
    ) as f:
        f.write(netlist)
        path = f.name
    try:
        proc = subprocess.run(
            ["ngspice", "-b", "-n", path],
            capture_output=True, text=True, timeout=timeout_s,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        out_lower = out.lower()
        # Hard failures first — these are always real
        for bad in HARD_FAILURES:
            if bad in out_lower:
                return False, f"{bad}\n{out[-300:]}"
        # Negative returncode = signal (segfault, OOM) → real fail
        if proc.returncode < 0:
            return False, f"signal={proc.returncode}\n{out[-300:]}"
        # Did the analysis actually run? (this is what we care about — not whether
        # the model wrote a .print directive)
        analysis_ran = any(m in out_lower for m in ANALYSIS_RAN_MARKERS)
        if analysis_ran:
            return True, out
        # No analysis ran AND no hard failure → likely "no simulations run" alone
        if "no simulations run" in out_lower:
            return False, f"no_simulations_run (and no analysis marker)\n{out[-300:]}"
        if proc.returncode == 0:
            return True, out  # exit clean, no markers — give benefit of doubt
        return False, f"exit={proc.returncode} (no analysis ran)\n{out[-300:]}"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        Path(path).unlink(missing_ok=True)
