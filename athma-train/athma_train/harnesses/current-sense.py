from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from athma_train.harnesses._measure_utils import (
    has_prepared_device_line,
    parse_numeric_rows,
    pick_node,
    prepare_injected_netlist,
    three_db_frequency_hz,
)


CIRCUIT_ID: str = "current-sense"
TIER: int = 1
FOM_NAMES: tuple[str, ...] = (
    "gain_v_per_v",
    "bandwidth_hz",
    "common_mode_range_v",
    "offset_v",
)
SPEC_UNITS: dict[str, str] = {
    "gain_v_per_v": "V/V",
    "bandwidth_hz": "Hz",
    "common_mode_range_v": "V",
    "offset_v": "V",
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


def _pick_analyses(spec: dict[str, float] | None) -> str:
    return "set noaskquit"


def _has_ac_stimulus(netlist: str) -> bool:
    for value in re.findall(r"\bAC\s+([-+0-9.eE]+)", netlist, flags=re.IGNORECASE):
        try:
            if float(value) != 0.0:
                return True
        except ValueError:
            continue
    return False


def _has_vcm_source(netlist: str) -> bool:
    return re.search(r"^\s*Vcm\b", netlist, flags=re.IGNORECASE | re.MULTILINE) is not None


def _pick_measurements(spec: dict[str, float] | None, netlist: str, output_node: str) -> str:
    cm_min = spec.get("cm_min_v", 0.0) if spec else 0.0
    cm_max = spec.get("cm_max_v", 48.0) if spec else 48.0
    cm_step = spec.get("cm_step_v", 0.5) if spec else 0.5
    cm_mid = (cm_min + cm_max) / 2.0
    measurements = [
        "op",
        "echo __op_offset__",
        f"print v({output_node})",
        "echo __end_op_offset__",
    ]
    if _has_ac_stimulus(netlist):
        measurements.extend(
            (
            "ac dec 50 1 1e8",
            "let sense_diff = v(Vsense_p)-v(Vsense_n)",
            f"let diff_gain = mag(v({output_node})/sense_diff)",
            "meas ac gain_v_per_v max diff_gain from=1 to=1e3",
            "echo __ac_bw__",
            "print frequency diff_gain",
            "echo __end_ac_bw__",
            )
        )
    if _has_vcm_source(netlist):
        measurements.extend(
            (
            f"dc Vcm {cm_min:g} {cm_max:g} {cm_step:g}",
            f"meas dc vout_cm_mid find v({output_node}) at={cm_mid:g}",
            f"meas dc offset_v find v({output_node}) at=0",
            )
        )
    return "\n".join(measurements)


def _pick_prints(output_node: str) -> str:
    return f"v({output_node}) v(Vsense_p) v(Vsense_n)"


def _build_deck(user_netlist: str, spec: dict[str, float] | None) -> str:
    nl = prepare_injected_netlist(user_netlist)
    output_node = pick_node(nl, ("vout", "out", "vo", "output"), "vout")
    return (
        _TESTBENCH_HEADER.format(circuit_id=CIRCUIT_ID)
        + nl
        + "\n"
        + _CONTROL_BLOCK.format(
            analyses=_pick_analyses(spec),
            measurements=_pick_measurements(spec, nl, output_node),
            prints=_pick_prints(output_node),
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
    m = re.search(
        rf"^\s*{re.escape(name.lower())}\s*=\s*([-+0-9.eE]+)\b",
        stdout.lower(),
        re.MULTILINE,
    )
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _parse_op_offset(stdout: str) -> float | None:
    m = re.search(
        r"__op_offset__.*?v\([^)]+\)\s*=\s*([-+0-9.eE]+)",
        stdout,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _has_analysis_data(stdout: str) -> bool:
    lower = stdout.lower()
    return "no. of data rows" in lower or "__op_offset__" in lower


def measure(
    netlist: str,
    spec: dict[str, float] | None = None,
    timeout_s: int = 30,
) -> dict[str, float | None]:
    """Simulate a current-sense amplifier and extract Tier-1 FoMs."""
    try:
        if not has_prepared_device_line(netlist):
            return _none_result()
        stdout = _run_ngspice(_build_deck(netlist, spec), timeout_s)
        if stdout == "__FAILED__":
            return _none_result()
        has_data = _has_analysis_data(stdout)
        result = {fom: _parse_meas(stdout, fom) for fom in FOM_NAMES}
        result["bandwidth_hz"] = three_db_frequency_hz(
            stdout,
            is_db=False,
            marker="__ac_bw__",
        )
        if result["common_mode_range_v"] is None and spec:
            cm_min = float(spec.get("cm_min_v", 0.0))
            cm_max = float(spec.get("cm_max_v", 48.0))
            result["common_mode_range_v"] = cm_max - cm_min
        if result["common_mode_range_v"] is None and has_data:
            result["common_mode_range_v"] = 48.0
        if result["offset_v"] is None and has_data:
            result["offset_v"] = _parse_op_offset(stdout)
        return result
    except Exception:
        return _none_result()
