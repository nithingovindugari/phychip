from __future__ import annotations

import math
import re
import subprocess
import tempfile
from pathlib import Path

from athma_train.harnesses._measure_utils import (
    has_prepared_device_line,
    prepare_injected_netlist,
    three_db_frequency_hz,
)


CIRCUIT_ID: str = "bridge-ina"
TIER: int = 2
FOM_NAMES: tuple[str, ...] = (
    "sensitivity_v_per_strain", "cmrr_db", "output_offset_v", "bandwidth_hz"
)
SPEC_UNITS: dict[str, str] = {
    "sensitivity_v_per_strain": "V/strain", "cmrr_db": "dB",
    "output_offset_v": "V", "bandwidth_hz": "Hz",
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

_STRAIN_STEP = 1e-3
_CM_STEP = 1e-3
_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _none_result() -> dict[str, float | None]:
    return {fom: None for fom in FOM_NAMES}


def _parse_params(nl: str) -> dict[str, float]:
    params: dict[str, float] = {}
    for line in nl.splitlines():
        stripped = line.strip()
        if not stripped.lower().startswith(".param"):
            continue
        for name, value in re.findall(r"(\w+)\s*=\s*([^\s;]+)", stripped[6:]):
            parsed = _parse_value(value, params)
            if parsed is not None:
                params[name.lower()] = parsed
    return params


def _safe_eval(expr: str, params: dict[str, float]) -> float | None:
    body = expr.strip()
    if body.startswith("{") and body.endswith("}"):
        body = body[1:-1]
    body = re.sub(
        r"\b[A-Za-z_]\w*\b",
        lambda m: str(params.get(m.group(0).lower(), float("nan"))),
        body,
    )
    if not re.fullmatch(r"[0-9eE+\-*/().\s]+|nan", body):
        return None
    try:
        value = eval(body, {"__builtins__": {}}, {})
    except Exception:
        return None
    return float(value) if math.isfinite(value) else None


def _parse_value(token: str, params: dict[str, float] | None = None) -> float | None:
    if params is None:
        params = {}
    if "{" in token or "}" in token:
        return _safe_eval(token, params)
    multipliers = {
        "t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "m": 1e-3,
        "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15,
    }
    m = re.fullmatch(rf"({_NUM})([a-zA-Z]*)", token)
    if not m:
        return params.get(token.lower())
    suffix = m.group(2).lower()
    if suffix and suffix not in multipliers:
        return None
    return float(m.group(1)) * multipliers.get(suffix, 1.0)


def _find_bridge_arm(nl: str) -> tuple[str, str, float, float] | None:
    params = _parse_params(nl)
    lines = [
        line.strip()
        for line in nl.splitlines()
        if line.strip() and not line.lstrip().startswith(("*", "."))
    ]
    sources: list[tuple[str, str, float]] = []
    for line in lines:
        parts = line.split(";", 1)[0].split()
        if len(parts) < 4 or not parts[0].lower().startswith("v"):
            continue
        value = None
        for index, token in enumerate(parts[3:]):
            if token.lower() == "dc" and index + 4 < len(parts):
                value = _parse_value(parts[index + 4], params)
                break
        value = _parse_value(parts[3], params) if value is None else value
        if value is not None and parts[2].lower() in ("0", "gnd"):
            sources.append((parts[0], parts[1], value))
    for source_name, excitation_node, excitation_v in sources:
        midpoint_to_bottom: dict[str, tuple[str, float]] = {}
        midpoint_has_top: set[str] = set()
        for line in lines:
            parts = line.split(";", 1)[0].split()
            if len(parts) < 4 or not parts[0].lower().startswith("r"):
                continue
            value = _parse_value(parts[3], params)
            if value is None:
                continue
            n1, n2 = parts[1], parts[2]
            if excitation_node.lower() in (n1.lower(), n2.lower()):
                midpoint = n2 if n1.lower() == excitation_node.lower() else n1
                midpoint_has_top.add(midpoint.lower())
            if n1.lower() in ("0", "gnd") or n2.lower() in ("0", "gnd"):
                midpoint = n2 if n1.lower() in ("0", "gnd") else n1
                midpoint_to_bottom[midpoint.lower()] = (parts[0], value)
        for midpoint, bottom in midpoint_to_bottom.items():
            if midpoint in midpoint_has_top:
                resistor_name, resistor_value = bottom
                return source_name, resistor_name, resistor_value, excitation_v
    return None


def _pick_measurements(nl: str, spec: dict[str, float] | None) -> str:
    bridge = _find_bridge_arm(nl)
    if bridge is None:
        return "\n".join(("op", "echo __bridge_offset__", "print v(Vout)",
                          "ac dec 50 1 1e6",
                          "echo __ac_bw__",
                          "print frequency vdb(Vout)",
                          "echo __end_ac_bw__"))
    source_name, resistor_name, resistor_value, excitation_v = bridge
    r_low = resistor_value * (1.0 - _STRAIN_STEP)
    r_high = resistor_value * (1.0 + _STRAIN_STEP)
    v_low = excitation_v * (1.0 - _CM_STEP)
    v_high = excitation_v * (1.0 + _CM_STEP)
    return "\n".join((
        "op", "echo __bridge_offset__", "print v(Vout)",
        f"alter @{resistor_name}[resistance] = {r_high:.12g}",
        "op", "echo __bridge_strain_high__", "print v(Vout)",
        f"alter @{resistor_name}[resistance] = {r_low:.12g}",
        "op", "echo __bridge_strain_low__", "print v(Vout)",
        f"alter @{resistor_name}[resistance] = {resistor_value:.12g}",
        f"alter @{source_name}[dc] = {v_high:.12g}",
        "op", "echo __bridge_cm_high__", "print v(Vout)",
        f"alter @{source_name}[dc] = {v_low:.12g}",
        "op", "echo __bridge_cm_low__", "print v(Vout)",
        f"alter @{source_name}[dc] = {excitation_v:.12g}",
        f"alter @{source_name}[acmag] = 1",
        "ac dec 50 1 1e6",
        "echo __ac_bw__",
        "print frequency vdb(Vout)",
        "echo __end_ac_bw__",
    ))


def _build_deck(user_netlist: str, spec: dict[str, float] | None) -> str:
    nl = prepare_injected_netlist(user_netlist)
    return (
        _TESTBENCH_HEADER.format(circuit_id=CIRCUIT_ID)
        + nl
        + "\n"
        + _CONTROL_BLOCK.format(
            analyses="set noaskquit",
            measurements=_pick_measurements(nl, spec),
            prints="vdb(Vout)",
        )
    )


def _run_ngspice(deck: str, timeout_s: int) -> str:
    path = ""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cir", delete=False, dir="/tmp") as f:
            f.write(deck)
            path = f.name
        proc = subprocess.run(
            ["ngspice", "-b", "-n", path], capture_output=True,
            text=True, timeout=timeout_s,
        )
        return (proc.stdout or "") + "\n" + (proc.stderr or "")
    except (OSError, subprocess.TimeoutExpired):
        return "__FAILED__"
    finally:
        if path:
            Path(path).unlink(missing_ok=True)


def _parse_meas(stdout: str, name: str) -> float | None:
    m = re.search(
        rf"^\s*{re.escape(name.lower())}\s*=\s*([-+0-9.eE]+)\b",
        stdout.lower(),
        re.MULTILINE,
    )
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _parse_probe(stdout: str, marker: str) -> float | None:
    m = re.search(
        rf"{re.escape(marker)}.*?^\s*v\(vout\)\s*=\s*([-+0-9.eE]+)\b",
        stdout.lower(),
        re.DOTALL | re.MULTILINE,
    )
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _derived_foms(nl: str, stdout: str) -> dict[str, float | None]:
    bridge = _find_bridge_arm(nl)
    if bridge is None:
        return {
            "sensitivity_v_per_strain": None,
            "cmrr_db": None,
            "output_offset_v": _parse_probe(stdout, "__bridge_offset__"),
        }
    _source_name, _resistor_name, _resistor_value, excitation_v = bridge
    offset = _parse_probe(stdout, "__bridge_offset__")
    strain_high = _parse_probe(stdout, "__bridge_strain_high__")
    strain_low = _parse_probe(stdout, "__bridge_strain_low__")
    cm_high = _parse_probe(stdout, "__bridge_cm_high__")
    cm_low = _parse_probe(stdout, "__bridge_cm_low__")
    sensitivity = None
    cmrr = None
    if strain_high is not None and strain_low is not None:
        sensitivity = (strain_high - strain_low) / (2.0 * _STRAIN_STEP)
    if (
        sensitivity is not None
        and cm_high is not None
        and cm_low is not None
        and abs(excitation_v) > 0.0
    ):
        common_gain = (cm_high - cm_low) / (
            excitation_v * _CM_STEP
        )
        diff_gain = sensitivity / (abs(excitation_v) / 4.0)
        cmrr = 20.0 * math.log10(abs(diff_gain) / (abs(common_gain) + 1e-30))
    return {
        "sensitivity_v_per_strain": sensitivity,
        "cmrr_db": cmrr,
        "output_offset_v": offset,
    }


def _has_analysis_data(stdout: str) -> bool:
    lower = stdout.lower()
    return "no. of data rows" in lower or "__bridge_offset__" in lower


def measure(netlist: str, spec: dict[str, float] | None = None,
            timeout_s: int = 30) -> dict[str, float | None]:
    try:
        if not has_prepared_device_line(netlist):
            return _none_result()
        nl = prepare_injected_netlist(netlist)
        stdout = _run_ngspice(_build_deck(netlist, spec), timeout_s)
        if stdout == "__FAILED__":
            return _none_result()
        derived = _derived_foms(nl, stdout)
        derived["bandwidth_hz"] = three_db_frequency_hz(
            stdout,
            marker="__ac_bw__",
        )
        if derived["bandwidth_hz"] is None and _has_analysis_data(stdout):
            derived["bandwidth_hz"] = 1.0e6
        return {
            fom: derived.get(fom, _parse_meas(stdout, fom))
            for fom in FOM_NAMES
        }
    except Exception:
        return _none_result()
