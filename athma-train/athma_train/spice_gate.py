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
import pwd
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import warnings
from pathlib import Path

# --- ngspice version guard (bug 2026-06-12) ---------------------------------
# AnalogCoder pass@1 is highly sensitive to the ngspice version: the SAME
# adapter scored 0/24 on ngspice-36 (Ubuntu apt default) and 16/24 on
# ngspice-46. Comparing scores across versions produced a FALSE "GRPO degrades
# OOD" conclusion. All evals and GRPO rewards MUST run on one pinned version.
# Set PHYCHIP_REQUIRE_NGSPICE (e.g. "46") to hard-enforce; otherwise warn once.
_NGSPICE_VERSION_CHECKED = False


def _check_ngspice_version(executable: str = "ngspice") -> None:
    global _NGSPICE_VERSION_CHECKED
    if _NGSPICE_VERSION_CHECKED:
        return
    _NGSPICE_VERSION_CHECKED = True
    want = os.environ.get("PHYCHIP_REQUIRE_NGSPICE")
    try:
        out = subprocess.run([executable, "--version"], capture_output=True,
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

# ngspice's control language includes commands that can execute programs and
# read/write arbitrary host files.  Model output is untrusted, so accepted
# control blocks are deliberately smaller than the full ngspice language.
SAFE_CONTROL_COMMANDS = frozenset(
    {
        "ac",
        "alter",
        "dc",
        "disto",
        "echo",
        "let",
        "meas",
        "measure",
        "noise",
        "op",
        "plot",
        "print",
        "quit",
        "reset",
        "run",
        "save",
        "setplot",
        "tran",
    }
)

# This is an allowlist, not merely a denylist: an unfamiliar dotcard must be
# reviewed before model-generated decks may use it.  In particular, file
# loading directives (.include/.lib/.osdi) are intentionally absent.
SAFE_DOTCARDS = frozenset(
    {
        "ac",
        "control",
        "dc",
        "disto",
        "end",
        "endc",
        "ends",
        "four",
        "func",
        "global",
        "ic",
        "meas",
        "measure",
        "model",
        "nodeset",
        "noise",
        "op",
        "option",
        "options",
        "param",
        "plot",
        "print",
        "probe",
        "pz",
        "save",
        "sens",
        "subckt",
        "temp",
        "tf",
        "title",
        "tran",
        "width",
    }
)

DANGEROUS_DOTCARDS = frozenset(
    {
        "cd",
        "chdir",
        "exec",
        "hdl",
        "inc",
        "include",
        "lib",
        "load",
        "osdi",
        "pre_osdi",
        "shell",
        "source",
        "verilog",
        "write",
        "wrdata",
    }
)

_CONTROL_META_RE = re.compile(r";|`|\$\(|&&|\|\||[<>]")
_DOTCARD_RE = re.compile(r"^\.([a-z][a-z0-9_]*)\b", re.IGNORECASE)
_CONTROL_COMMAND_RE = re.compile(r"^([a-z][a-z0-9_-]*)\b", re.IGNORECASE)
_EXTERNAL_FILE_RE = re.compile(r"\bfile\s*=", re.IGNORECASE)

_DEFAULT_MEMORY_LIMIT_MB = 1024
_OUTPUT_FILE_LIMIT_BYTES = 16 * 1024 * 1024
_OPEN_FILE_LIMIT = 32

# RLIMIT_NPROC is enforced per real UID, not per child process. Every
# root-launched sandbox drops to the shared ``nobody`` account; concurrent
# launches would therefore make all but one child fail execve with EAGAIN when
# each child sets RLIMIT_NPROC=0. Serialize the short ngspice subprocesses so
# the no-fork boundary remains intact and threaded reward grading stays exact.
_NGSPICE_EXECUTION_LOCK = threading.Lock()

# A fixed child program applies limits after subprocess creation and then
# replaces itself with ngspice.  This avoids preexec_fn, which can deadlock when
# grade_netlist is called from the threaded GRPO reward pool.
_SANDBOX_LAUNCHER = r"""
import math
import os
import resource
import sys

cpu_seconds = max(1, int(math.ceil(float(sys.argv[1]))))
memory_bytes = int(sys.argv[2])
file_bytes = int(sys.argv[3])
open_files = int(sys.argv[4])
uid = int(sys.argv[5])
gid = int(sys.argv[6])
executable = sys.argv[7]
child_args = sys.argv[7:]

def set_limit(name, value):
    limit = getattr(resource, name, None)
    if limit is None:
        return
    _, hard = resource.getrlimit(limit)
    bounded = value if hard == resource.RLIM_INFINITY else min(value, hard)
    resource.setrlimit(limit, (bounded, bounded))

def try_memory(name):
    try:
        set_limit(name, memory_bytes)
    except (OSError, ValueError):
        return False
    return True

os.umask(0o077)
set_limit("RLIMIT_CPU", cpu_seconds)
if not try_memory("RLIMIT_AS"):
    try_memory("RLIMIT_DATA")
set_limit("RLIMIT_FSIZE", file_bytes)
set_limit("RLIMIT_NOFILE", open_files)
set_limit("RLIMIT_NPROC", 0)
set_limit("RLIMIT_CORE", 0)
if uid >= 0:
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)
os.execv(executable, child_args)
"""


class UnsafeNetlistError(ValueError):
    """Raised when an untrusted deck fails static safety validation."""


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
    # Truncate only at an exact .end card.  The old substring search confused
    # .endc/.ends with .end and could hide malformed control blocks.
    kept: list[str] = []
    for line in s.splitlines():
        kept.append(line)
        if re.match(r"^\s*\.end\s*(?:\$.*)?$", line, re.IGNORECASE):
            break
    s = "\n".join(kept)
    return s.strip()


def validate_netlist_safety(netlist: str) -> tuple[bool, str]:
    """Fail-closed validation for model-generated ngspice input.

    This gate prevents access to ngspice's host-command and file-I/O features
    before any subprocess is created.  It also rejects malformed/nested
    control blocks so parser ambiguity cannot turn data into commands.
    """
    if "\x00" in netlist:
        return False, "NUL byte is not allowed"

    in_control = False
    for lineno, raw_line in enumerate(netlist.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("*"):
            continue
        lower = line.lower()

        if lower == ".control":
            if in_control:
                return False, f"line {lineno}: nested .control block"
            in_control = True
            continue
        if lower == ".endc":
            if not in_control:
                return False, f"line {lineno}: .endc without .control"
            in_control = False
            continue

        if in_control:
            if lower == ".end":
                return False, f"line {lineno}: unterminated .control block before .end"
            if line.startswith("."):
                return False, f"line {lineno}: dotcard inside .control block"
            if _CONTROL_META_RE.search(line):
                return False, f"line {lineno}: control command chaining/redirection is not allowed"
            match = _CONTROL_COMMAND_RE.match(line)
            if not match:
                return False, f"line {lineno}: malformed control command"
            command = match.group(1).lower()
            if command == "set":
                # Keep the one setting used by the canonical harnesses.  Other
                # set variables can redirect output or change file paths.
                if not re.fullmatch(r"set\s+noaskquit", lower):
                    return (
                        False,
                        f"line {lineno}: control command 'set' is not allowed with these arguments",
                    )
                continue
            if command not in SAFE_CONTROL_COMMANDS:
                return False, f"line {lineno}: control command '{command}' is not allowed"
            continue

        if not line.startswith("."):
            if _EXTERNAL_FILE_RE.search(line):
                return False, f"line {lineno}: external file references are not allowed"
            continue

        match = _DOTCARD_RE.match(line)
        if not match:
            return False, f"line {lineno}: malformed dotcard"
        dotcard = match.group(1).lower()
        if dotcard in ("control", "endc"):
            # Markers with trailing tokens are deliberately invalid.
            return False, f"line {lineno}: malformed .{dotcard} marker"
        if dotcard in DANGEROUS_DOTCARDS:
            return False, f"line {lineno}: dangerous dotcard '.{dotcard}' is not allowed"
        if dotcard not in SAFE_DOTCARDS:
            return False, f"line {lineno}: dotcard '.{dotcard}' is not allowlisted"
        if _EXTERNAL_FILE_RE.search(line):
            return False, f"line {lineno}: external file references are not allowed"

    if in_control:
        return False, "unterminated .control block"
    return True, "ok"


def has_analysis_directive(netlist_lower: str) -> bool:
    """True if netlist contains an analysis directive OR a .control block
    with an analysis command inside.

    Standalone form: lines starting with .op / .tran / .dc / .ac / .noise / .disto
    Block form:      .control ... op / tran / dc / ac ... .endc
    """
    in_control = False
    analysis_commands = {directive[1:] for directive in ANALYSIS_DIRECTIVES}
    for raw_line in netlist_lower.splitlines():
        line = raw_line.strip()
        if line == ".control":
            in_control = True
            continue
        if line == ".endc":
            in_control = False
            continue
        if not line or line.startswith("*"):
            continue
        token = line.split(None, 1)[0].lower()
        if in_control and token in analysis_commands:
            return True
        if not in_control and token in ANALYSIS_DIRECTIVES:
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


def _nobody_ids() -> tuple[int, int] | None:
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return None
    try:
        account = pwd.getpwnam("nobody")
    except KeyError:
        return None
    return account.pw_uid, account.pw_gid


def _sandbox_environment(sandbox_dir: Path, executable: str) -> dict[str, str]:
    executable_dir = str(Path(executable).parent) if os.path.sep in executable else ""
    path_parts = [part for part in (executable_dir, os.defpath) if part]
    return {
        "HOME": str(sandbox_dir),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.pathsep.join(path_parts),
        "TMPDIR": str(sandbox_dir),
    }


def _read_limited(path: Path, limit: int = _OUTPUT_FILE_LIMIT_BYTES) -> str:
    data = path.read_bytes()[:limit]
    return data.decode("utf-8", errors="replace")


def run_ngspice_sandboxed(
    text: str,
    timeout_s: int = 20,
    memory_limit_mb: int = _DEFAULT_MEMORY_LIMIT_MB,
    *,
    ngspice_bin: str = "ngspice",
    scratch_root: str | Path = "/tmp",
) -> subprocess.CompletedProcess[str]:
    """Run one untrusted deck inside the canonical constrained boundary.

    Static validation happens before *any* subprocess.  The child receives an
    isolated working directory and environment, bounded CPU/address-space/file
    resources, no ability to fork, and (when the caller is root) the ``nobody``
    identity.  Stdout/stderr go to size-limited files rather than unbounded
    in-memory pipes.
    """
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if memory_limit_mb <= 0:
        raise ValueError("memory_limit_mb must be positive")

    netlist = normalize_netlist(text)
    safe, reason = validate_netlist_safety(netlist)
    if not safe:
        raise UnsafeNetlistError(reason)

    executable = shutil.which(ngspice_bin) or ngspice_bin
    _check_ngspice_version(executable)
    root = Path(scratch_root)
    root.mkdir(parents=True, exist_ok=True)
    with _NGSPICE_EXECUTION_LOCK, tempfile.TemporaryDirectory(
        prefix="phychip-ngspice-", dir=root
    ) as tmp:
        sandbox_dir = Path(tmp)
        deck_path = sandbox_dir / "circuit.cir"
        stdout_path = sandbox_dir / "stdout.log"
        stderr_path = sandbox_dir / "stderr.log"
        deck_path.write_text(netlist, encoding="utf-8")
        deck_path.chmod(0o400)
        stdout_path.touch(mode=0o600)
        stderr_path.touch(mode=0o600)

        drop_ids = _nobody_ids()
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            if drop_ids is None:
                raise OSError("sandbox setup failed: cannot resolve nobody account")
            uid, gid = drop_ids
            for owned_path in (sandbox_dir, deck_path, stdout_path, stderr_path):
                os.chown(owned_path, uid, gid)
            sandbox_dir.chmod(0o700)

        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            uid, gid = drop_ids if drop_ids is not None else (-1, -1)
            command = [
                sys.executable,
                "-I",
                "-S",
                "-c",
                _SANDBOX_LAUNCHER,
                str(timeout_s),
                str(memory_limit_mb * 1024 * 1024),
                str(_OUTPUT_FILE_LIMIT_BYTES),
                str(_OPEN_FILE_LIMIT),
                str(uid),
                str(gid),
                executable,
                "-b",
                "-n",
                str(deck_path),
            ]
            raw = subprocess.run(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=timeout_s,
                cwd=sandbox_dir,
                env=_sandbox_environment(sandbox_dir, executable),
                start_new_session=True,
            )

        # Tests and alternate subprocess implementations may return captured
        # strings directly; real execution writes into the bounded files.
        stdout = raw.stdout if isinstance(raw.stdout, str) else _read_limited(stdout_path)
        stderr = raw.stderr if isinstance(raw.stderr, str) else _read_limited(stderr_path)
        return subprocess.CompletedProcess(raw.args, raw.returncode, stdout, stderr)


def grade_netlist(
    text: str,
    timeout_s: int = 20,
    memory_limit_mb: int = _DEFAULT_MEMORY_LIMIT_MB,
    *,
    ngspice_bin: str = "ngspice",
    scratch_root: str | Path = "/tmp",
) -> tuple[bool, str]:
    """Strict ngspice gate.

    Returns (passed, log_or_reason). Passes iff:
      1. After normalization, has ≥1 device line.
      2. Has an analysis directive (standalone or in .control block).
      3. ngspice exits 0 (no warnings-only, no signal).
      4. Output does NOT contain any HARD_FAILURES marker.
    """
    netlist = normalize_netlist(text)
    safe, reason = validate_netlist_safety(netlist)
    if not safe:
        return False, f"unsafe netlist: {reason}"
    if not has_device_line(netlist):
        return False, "no device lines in netlist"
    if not has_analysis_directive(netlist.lower()):
        return False, "no analysis directive (.op/.tran/.dc/.ac or .control block)"
    try:
        proc = run_ngspice_sandboxed(
            netlist,
            timeout_s=timeout_s,
            memory_limit_mb=memory_limit_mb,
            ngspice_bin=ngspice_bin,
            scratch_root=scratch_root,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        out_lower = out.lower()
        # Hard failures first — these are always real
        for bad in HARD_FAILURES:
            if bad in out_lower:
                return False, f"{bad}\n{out[-300:]}"
        # The release contract requires a clean process exit. Output markers
        # cannot turn a failed ngspice process into a successful simulation.
        if proc.returncode != 0:
            kind = "signal" if proc.returncode < 0 else "exit"
            return False, f"{kind}={proc.returncode}\n{out[-300:]}"
        # Did the analysis actually run? (this is what we care about — not whether
        # the model wrote a .print directive)
        analysis_ran = any(m in out_lower for m in ANALYSIS_RAN_MARKERS)
        if analysis_ran:
            return True, out
        # No analysis ran AND no hard failure → likely "no simulations run" alone
        if "no simulations run" in out_lower:
            return False, f"no_simulations_run (and no analysis marker)\n{out[-300:]}"
        return True, out  # exit clean, no hard failure
    except UnsafeNetlistError as exc:
        return False, f"unsafe netlist: {exc}"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"sandbox execution failed: {type(exc).__name__}"
