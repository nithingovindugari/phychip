"""Thin pre-scorer sanitizer for phy_chip_env.score@v11  ("score@v11+sanitize-v12").

Pure string processing, no dependencies besides the unchanged scorer.

Why: 14 harnesses inject their testbench *after* the user deck and parse the first
``name = value`` line of ngspice stdout, but only 9 of them strip user analysis /
.meas cards.  A user ``.meas ac gain_db param='19.9'`` therefore forges the scored
figure of merit.  The AC magnitude of declared input sources is also user-chosen
while several harnesses measure absolute vdb(Vout)/vm(Vout).

What this layer does (everything else about scoring is unchanged):
  * unfold '+' continuation lines, drop comments, drop the user title;
  * keep ONLY circuit content: device cards, .model, .subckt/.ends, .param, .func, .global;
  * drop every other dotcard: .meas/.measure .print .plot .probe .save .four .noise .tf
    .ac .dc .tran .op .pz .sens .disto .ic .nodeset .option(s) .temp .width .lib .include ...
    and the whole .control ... .endc block (echo/let/print tricks), also when nested in a subckt;
  * force the AC magnitude of every declared input-port source that has a non-zero AC drive
    to magnitude 1 (sign / phase tokens preserved); strip AC from every other independent
    V/I source.  AC-zero / AC-absent input sources are left alone (e.g. a DC-only Vcm).
    Harness check: diff-amp, diff-pair, ina, summing-amp, gain-trim, buffer, tia, bandgap, ldo
    re-`alter` the magnitudes themselves, current-sense divides by the differential input, sallen-key
    divides by Vin; inv-amp, noninv-amp, cs-stage, differentiator, integrator rely on the
    user AC magnitude, which is therefore pinned to 1 here;
  * drop user devices that collide with harness-injected names;
  * append a single ``.control / op / .endc / .end``.
"""
from __future__ import annotations

import re

SEMANTICS = "phy_chip_env.score@v11+sanitize-v12"

# From phy-chip/poc_experiments/generate_rlvr_candidates_v7.py FAMILY_SPECS (input ports)
FAMILY_INPUTS: dict[str, tuple[str, ...]] = {
    "bandgap": ("Vdd",), "boost": ("Vin",), "bridge-ina": ("Vex",), "buck": ("Vin",),
    "buffer": ("Vin",), "cs-stage": ("Vin",), "current-mirror": ("Iref",),
    "current-sense": ("Vsense_p", "Vsense_n", "Vcm"), "diff-amp": ("Vp", "Vn"),
    "diff-pair": ("Vp", "Vn"), "differentiator": ("Vin",), "gain-trim": ("Vin", "Vtrim"),
    "h-bridge": ("Vctrl_a", "Vctrl_b"), "ina": ("Vp", "Vn"), "integrator": ("Vin",),
    "inv-amp": ("Vin",), "ldo": ("Vin",), "level-shift": ("Vin_lv",), "noninv-amp": ("Vin",),
    "summing-amp": ("V1", "V2", "V3"), "sallen-key": ("Vin",), "schmitt": ("Vin",),
    "tia": ("I_pd",),
}

_KEEP_DOT = {".model", ".subckt", ".ends", ".param", ".func", ".global"}
_DEVICE_LETTERS = set("rclkvidmqjzxegfhbswtuoanyp")
# names the harnesses inject themselves
_RESERVED = re.compile(r"^(vtb_|ildo_bench|ibuf_z|i_tia_harness|iref_harn|vsweep$)", re.I)
_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?(meg|mil|[tgkmunpf])?[a-z]*$", re.I)
_INLINE = re.compile(r"\s\$.*$|;.*$")


def _logical_lines(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.replace("\r", "").split("\n"):
        s = raw.strip()
        if s.startswith("+") and out:
            out[-1] += " " + s[1:].strip()
        else:
            out.append(s)
    return out


def _fix_ac(tokens: list[str], is_input: bool) -> list[str]:
    low = [t.lower() for t in tokens]
    if "ac" not in low[3:]:
        return tokens
    i = low.index("ac", 3)
    mag = tokens[i + 1] if i + 1 < len(tokens) else ""
    try:
        val = float(mag)
        zero = val == 0.0
        neg = val < 0
    except ValueError:
        zero, neg = False, mag.startswith("-")   # parameterised AC: cannot know -> treat as non-zero
    has_phase = i + 2 < len(tokens) and _NUM.match(tokens[i + 2]) is not None
    end = i + 2 + (1 if has_phase else 0)
    if is_input:
        if zero:
            return tokens
        return tokens[:i] + ["AC", "-1" if neg else "1"] + tokens[i + 2:]
    return tokens[:i] + tokens[end:]


def sanitize_netlist(netlist: str, circuit_id: str, input_ports: list[str] | None = None) -> str:
    ports = {p.lower() for p in (input_ports if input_ports is not None else FAMILY_INPUTS.get(circuit_id, ()))}
    kept: list[str] = []
    control_depth = 0
    for line in _logical_lines(netlist or ""):
        if not line or line.startswith("*"):
            continue
        line = _INLINE.sub("", line).strip()
        if not line:
            continue
        low = line.lower()
        head = low.split(None, 1)[0]
        if head == ".control":
            control_depth += 1
            continue
        if head == ".endc":
            control_depth = max(0, control_depth - 1)
            continue
        if control_depth:
            continue
        if head == ".end":
            break
        if head.startswith("."):
            if head in _KEEP_DOT:
                kept.append(line)
            continue
        if head[0] not in _DEVICE_LETTERS or _RESERVED.match(head):
            continue
        tokens = line.split()
        if head[0] in "vi" and len(tokens) >= 4:
            tokens = _fix_ac(tokens, head in ports)
            line = " ".join(tokens)
        kept.append(line)
    return "* phychip sanitize-v12 deck\n" + "\n".join(kept) + "\n.control\nop\n.endc\n.end"


def _extract(completion: str) -> str | None:
    m = re.search(r"```spice\s*(.*?)```", completion or "", re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else None


_IC_TOKEN = re.compile(r"\s+ic\s*=\s*[^\s]+", re.IGNORECASE)
_PRECHARGE_FAMILIES = {"boost", "buck"}


def _strip_precharge(net: str) -> str:
    """Remove initial conditions (IC= on elements, .ic/.nodeset cards, uic) so the circuit must reach its operating point itself."""
    out = []
    for line in net.splitlines():
        low = line.strip().lower()
        if low.startswith((".ic", ".nodeset")):
            continue
        if low.startswith(".tran"):
            line = re.sub(r"\s+uic\b", "", line, flags=re.IGNORECASE)
        elif low and not low.startswith(("*", ".")):
            line = _IC_TOKEN.sub("", line)
        out.append(line)
    return "\n".join(out)


def v14_check(circuit_id: str, target_specs: dict, completion: str, result, timeout_s: float = 25):
    """Scorer v14 = v13 plus two physics checks for shortcuts the wave-5 audit found (audit/wave5_hacks/REPORT.md):
    1. boost/buck: a pass must survive removing capacitor/inductor pre-charge (IC=, .ic, uic); RL learned to hold
       the boost output with a pre-charged 4.7 mF cap (efficiency ~14%) instead of converting power.
    2. tia: no current-controlled sources (H/F); an ideal CCVS supplied the transimpedance in 0.8% of passes.
    Only passes are re-examined, so v14 can only turn a v13 pass into a fail."""
    import dataclasses
    from phy_chip_env import score as _score
    if not result.all_specs_met:
        return result
    net = _extract(completion) or ""
    if circuit_id == "tia" and re.search(r"^\s*[hf]\w*\s", net, re.IGNORECASE | re.MULTILINE):
        return dataclasses.replace(result, all_specs_met=False, anticheat_blocked=True, reward=0.0)
    if circuit_id in _PRECHARGE_FAMILIES:
        stripped = _strip_precharge(net)
        if stripped != net:
            again = _score(circuit_id, target_specs, "```spice\n" + stripped + "\n```", timeout_s=timeout_s)
            if not again.all_specs_met:
                return dataclasses.replace(result, all_specs_met=False, reward=again.reward, spec_score=again.spec_score,
                                           met_specs=again.met_specs)
    return result


def score_v12(circuit_id: str, target_specs: dict, completion: str, timeout_s: float = 25):
    import os as _os
    if _os.environ.get("PHYCHIP_SCORER") in ("v13", "v14", "v15"):  # v13 sanitises inside the canonical scorer
        from phy_chip_env import score as _score
        result = _score(circuit_id, target_specs, completion, timeout_s=timeout_s)
        if _os.environ.get("PHYCHIP_SCORER") in ("v14", "v15"):  # v15 = v14 checks on the J14-patched env
            result = v14_check(circuit_id, target_specs, completion, result, timeout_s=timeout_s)
        return result
    from phy_chip_env import score
    net = _extract(completion)
    if net is None:               # unfenced -> v11 behaviour (no score)
        return score(circuit_id, target_specs, completion, timeout_s=timeout_s)
    from phy_chip_env.sandbox import normalize_netlist, validate_netlist_safety
    safe, _why = validate_netlist_safety(normalize_netlist(net))
    if not safe:                  # fail closed: file/shell access attempts are rejected, not silently stripped
        return score(circuit_id, target_specs, completion, timeout_s=timeout_s)
    clean = sanitize_netlist(net, circuit_id)
    return score(circuit_id, target_specs, "```spice\n" + clean + "\n```", timeout_s=timeout_s)
