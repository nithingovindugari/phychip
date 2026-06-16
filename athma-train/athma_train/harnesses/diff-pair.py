from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from athma_train.harnesses._measure_utils import parse_numeric_rows
from athma_train.spice_gate import grade_netlist, normalize_netlist


CIRCUIT_ID: str = "diff-pair"
TIER: int = 0
FOM_NAMES: tuple[str, ...] = (
    "differential_gain_db",
    "cmrr_db",
    "tail_current_a",
    "input_offset_v",
)
SPEC_UNITS: dict[str, str] = {
    "differential_gain_db": "dB",
    "cmrr_db": "dB",
    "tail_current_a": "A",
    "input_offset_v": "V",
}

_TESTBENCH_HEADER = """\
* AUTO-INJECTED TESTBENCH FOR {circuit_id}
* Do not collapse; user netlist appended below verbatim.
"""

_CONTROL_BLOCK = """\
.control
{analyses}
{measurements}
print {prints}
.endc
.end
"""


def _none_result() -> dict[str, float | None]:
    return {fom: None for fom in FOM_NAMES}


def _source_for_node(netlist: str, node: str) -> tuple[str, float] | None:
    pattern = re.compile(
        rf"^\s*(v[\w$.-]*)\s+{re.escape(node)}\s+0\b(.*)$",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(netlist)
    if match is None:
        return None
    value_match = re.search(
        r"\bdc\s*=?\s*([-+0-9.eE]+)|\s([-+0-9.eE]+)(?:\s|$)",
        match.group(2),
        re.IGNORECASE,
    )
    value = float(value_match.group(1) or value_match.group(2)) if value_match else 0.0
    return match.group(1), value


def _supply_source(netlist: str) -> tuple[str, float] | None:
    pattern = re.compile(
        r"^\s*(v[\w$.-]*)\s+(\S+)\s+0\b(.*)$",
        re.IGNORECASE | re.MULTILINE,
    )
    for match in pattern.finditer(netlist):
        if match.group(2).lower() not in {"vdd", "vcc", "vsup"}:
            continue
        value_match = re.search(
            r"\bdc\s*=?\s*([-+0-9.eE]+)|\s([-+0-9.eE]+)(?:\s|$)",
            match.group(3),
            re.IGNORECASE,
        )
        value = (
            float(value_match.group(1) or value_match.group(2))
            if value_match
            else 5.0
        )
        return match.group(1), value
    return None


def _pick_analyses(spec: dict[str, float] | None) -> str:
    return "set noaskquit"


def _pick_measurements(
    spec: dict[str, float] | None,
    vp_source: str | None,
    vn_source: str | None,
    vcm: float,
    supply: tuple[str, float] | None,
) -> str:
    if vp_source is None or vn_source is None:
        return ""

    sweep = max(abs(vcm) * 0.01, 0.02)
    lines = [
        f"alter @{vp_source}[acmag]=0.5",
        f"alter @{vp_source}[acphase]=0",
        f"alter @{vn_source}[acmag]=0.5",
        f"alter @{vn_source}[acphase]=180",
        "ac dec 100 1 1e8",
        "meas ac differential_gain_db max vdb(Vout) from=1 to=1e3",
        f"alter @{vp_source}[acmag]=1",
        f"alter @{vp_source}[acphase]=0",
        f"alter @{vn_source}[acmag]=1",
        f"alter @{vn_source}[acphase]=0",
        "ac dec 100 1 1e8",
        "meas ac common_gain_db max vdb(Vout) from=1 to=1e3",
        f"alter @{vp_source}[dc]={vcm:.12g}",
        f"alter @{vn_source}[dc]={vcm:.12g}",
        "op",
        "let vout_mid = v(Vout)",
        "print vout_mid",
        f"dc {vp_source} {vcm - sweep:.12g} {vcm + sweep:.12g} {sweep / 20:.12g}",
        "echo __offset_sweep__",
        "print v(Vp) v(Vout)",
    ]
    if supply is not None:
        supply_name, supply_v = supply
        step = max(abs(supply_v) * 1e-4, 1e-4)
        lines.extend(
            (
                f"dc {supply_name} {supply_v:.12g} {supply_v + step:.12g} {step:.12g}",
                f"meas dc tail_current_a avg i({supply_name})",
            )
        )
    return "\n".join(lines)


def _pick_prints() -> str:
    return "vdb(Vout) vm(Vout)"


def _input_common_mode(user_netlist: str) -> float:
    nl = normalize_netlist(user_netlist)
    nl = re.sub(r"\.control\b.*?\.endc\b", "", nl, flags=re.DOTALL | re.IGNORECASE)
    nl = re.sub(r"\.end\s*$", "", nl, flags=re.IGNORECASE).rstrip()
    vp_source = _source_for_node(nl, "Vp")
    vn_source = _source_for_node(nl, "Vn")
    vp_dc = vp_source[1] if vp_source else 0.0
    vn_dc = vn_source[1] if vn_source else 0.0
    return (vp_dc + vn_dc) / 2.0


def _build_deck(user_netlist: str, spec: dict[str, float] | None) -> str:
    nl = normalize_netlist(user_netlist)
    nl = re.sub(r"\.control\b.*?\.endc\b", "", nl, flags=re.DOTALL | re.IGNORECASE)
    nl = re.sub(r"\.end\s*$", "", nl, flags=re.IGNORECASE).rstrip()
    vp_source = _source_for_node(nl, "Vp")
    vn_source = _source_for_node(nl, "Vn")
    vcm = _input_common_mode(user_netlist)
    return (
        _TESTBENCH_HEADER.format(circuit_id=CIRCUIT_ID)
        + nl
        + "\n"
        + _CONTROL_BLOCK.format(
            analyses=_pick_analyses(spec),
            measurements=_pick_measurements(
                spec,
                vp_source[0] if vp_source else None,
                vn_source[0] if vn_source else None,
                vcm,
                _supply_source(nl),
            ),
            prints=_pick_prints(),
        )
    )


def _run_ngspice(deck: str, timeout_s: int) -> str:
    path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cir", delete=False, dir="/tmp"
        ) as f:
            f.write(deck)
            path = f.name
        proc = subprocess.run(
            ["ngspice", "-b", "-n", path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return (proc.stdout or "") + "\n" + (proc.stderr or "")
    except (OSError, subprocess.TimeoutExpired):
        return "__FAILED__"
    finally:
        if path:
            Path(path).unlink(missing_ok=True)


def _parse_meas(stdout: str, name: str) -> float | None:
    match = re.search(
        rf"^\s*{re.escape(name.lower())}\s*=\s*([-+0-9.eE]+)\b",
        stdout.lower(),
        re.MULTILINE,
    )
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _postprocess(stdout: str, input_common_mode: float) -> dict[str, float | None]:
    result = {fom: _parse_meas(stdout, fom) for fom in FOM_NAMES}
    common_gain_db = _parse_meas(stdout, "common_gain_db")
    diff_gain_db = result["differential_gain_db"]
    if diff_gain_db is not None and common_gain_db is not None:
        result["cmrr_db"] = diff_gain_db - common_gain_db
    if result["tail_current_a"] is not None:
        result["tail_current_a"] = abs(result["tail_current_a"])
    result["input_offset_v"] = _parse_input_offset(stdout, input_common_mode)
    return result


def _parse_input_offset(stdout: str, input_common_mode: float) -> float | None:
    vout_mid = _parse_meas(stdout, "vout_mid")
    marker = stdout.find("__offset_sweep__")
    section = stdout[marker:] if marker >= 0 else stdout
    rows = parse_numeric_rows(section, columns=3)
    if vout_mid is None or len(rows) < 2:
        return None
    points = [(row[1], row[2]) for row in rows]
    for (vp0, out0), (vp1, out1) in zip(points, points[1:]):
        if out0 == out1:
            continue
        if (out0 - vout_mid) * (out1 - vout_mid) <= 0.0:
            ratio = (vout_mid - out0) / (out1 - out0)
            vp_cross = vp0 + ratio * (vp1 - vp0)
            return abs(vp_cross - input_common_mode)
    return None


def measure(
    netlist: str,
    spec: dict[str, float] | None = None,
    timeout_s: int = 30,
) -> dict[str, float | None]:
    """Simulate a differential pair netlist and extract Tier-0 FoMs."""
    try:
        passed, _log = grade_netlist(netlist, timeout_s=20)
        if not passed:
            return _none_result()
        input_common_mode = _input_common_mode(netlist)
        stdout = _run_ngspice(_build_deck(netlist, spec), timeout_s)
        if stdout == "__FAILED__":
            return _none_result()
        return _postprocess(stdout, input_common_mode)
    except Exception:
        return _none_result()
