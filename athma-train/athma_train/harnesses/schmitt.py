from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from athma_train.harnesses._measure_utils import (
    has_prepared_device_line,
    parse_numeric_rows,
    prepare_injected_netlist,
)


CIRCUIT_ID: str = "schmitt"
TIER: int = 1
FOM_NAMES: tuple[str, ...] = (
    "vth_high_v",
    "vth_low_v",
    "hysteresis_v",
    "prop_delay_s",
)
SPEC_UNITS: dict[str, str] = {
    "vth_high_v": "V",
    "vth_low_v": "V",
    "hysteresis_v": "V",
    "prop_delay_s": "s",
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


def _dc_limits(spec: dict[str, float] | None) -> tuple[float, float]:
    if spec is None:
        return -1.0, 6.0
    values = [
        value
        for key, value in spec.items()
        if key in ("vth_high_v", "vth_low_v") and isinstance(value, (int, float))
    ]
    if not values:
        return -1.0, 6.0
    return min(values) - 1.0, max(values) + 1.0


def _pick_analyses(spec: dict[str, float] | None) -> str:
    return "set noaskquit"


def _pick_measurements(spec: dict[str, float] | None) -> str:
    sweep_min, sweep_max = _dc_limits(spec)
    step = max((sweep_max - sweep_min) / 700.0, 1e-3)
    return "\n".join(
        (
            f"dc Vin {sweep_min:g} {sweep_max:g} {step:g}",
            "meas dc voh_up max v(Vout)",
            "meas dc vol_up min v(Vout)",
            "let vout_mid=(voh_up+vol_up)/2",
            "meas dc vth_high_v WHEN v(Vout)=vout_mid CROSS=1",
            f"dc Vin {sweep_max:g} {sweep_min:g} {-step:g}",
            "meas dc voh_down max v(Vout)",
            "meas dc vol_down min v(Vout)",
            "let vout_mid2=(voh_down+vol_down)/2",
            "meas dc vth_low_v WHEN v(Vout)=vout_mid2 CROSS=1",
            "reset",
            "tran 10u 10m",
            "meas tran vin_hi max v(Vin)",
            "meas tran vin_lo min v(Vin)",
            "meas tran vout_hi max v(Vout)",
            "meas tran vout_lo min v(Vout)",
        )
    )


def _pick_prints() -> str:
    return "v(Vin) v(Vout)"


def _build_deck(user_netlist: str, spec: dict[str, float] | None) -> str:
    nl = prepare_injected_netlist(user_netlist)
    return (
        _TESTBENCH_HEADER.format(circuit_id=CIRCUIT_ID)
        + nl
        + "\n"
        + _CONTROL_BLOCK.format(
            analyses=_pick_analyses(spec),
            measurements=_pick_measurements(spec),
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


def _crossing_time(
    points: list[tuple[float, float]],
    target: float,
    *,
    start_s: float = 0.0,
    direction: str = "rise",
) -> float | None:
    for (t0, y0), (t1, y1) in zip(points, points[1:]):
        if t1 < start_s or y0 == y1:
            continue
        if direction == "rise":
            crossed = y0 <= target <= y1
        elif direction == "fall":
            crossed = y0 >= target >= y1
        else:
            crossed = (y0 - target) * (y1 - target) <= 0.0
        if not crossed:
            continue
        ratio = (target - y0) / (y1 - y0)
        return t0 + ratio * (t1 - t0)
    return None


def _parse_prop_delay(stdout: str) -> float | None:
    rows = parse_numeric_rows(stdout, columns=3)
    if len(rows) < 2:
        return None
    vin = [row[1] for row in rows]
    vout = [row[2] for row in rows]
    vin_mid = (max(vin) + min(vin)) / 2.0
    vout_mid = (max(vout) + min(vout)) / 2.0
    vin_points = [(row[0], row[1]) for row in rows]
    vout_points = [(row[0], row[2]) for row in rows]
    t_in = _crossing_time(vin_points, vin_mid, direction="rise")
    if t_in is None:
        return None
    t_out = _crossing_time(vout_points, vout_mid, start_s=t_in, direction="any")
    if t_out is None:
        return None
    return abs(t_out - t_in)


def measure(
    netlist: str,
    spec: dict[str, float] | None = None,
    timeout_s: int = 30,
) -> dict[str, float | None]:
    """Simulate a Schmitt trigger netlist and extract threshold FoMs."""
    try:
        if not has_prepared_device_line(netlist):
            return _none_result()
        stdout = _run_ngspice(_build_deck(netlist, spec), timeout_s)
        if stdout == "__FAILED__":
            return _none_result()
        result = {fom: _parse_meas(stdout, fom) for fom in FOM_NAMES}
        high = result["vth_high_v"]
        low = result["vth_low_v"]
        if high is not None and low is not None:
            result = {**result, "hysteresis_v": abs(high - low)}
        result["prop_delay_s"] = _parse_prop_delay(stdout)
        return result
    except Exception:
        return _none_result()
