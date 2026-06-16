from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from athma_train.spice_gate import grade_netlist, normalize_netlist


CIRCUIT_ID: str = "level-shift"
TIER: int = 1
FOM_NAMES: tuple[str, ...] = (
    "vol_v",
    "voh_v",
    "rise_time_s",
    "fall_time_s",
    "prop_delay_s",
)
SPEC_UNITS: dict[str, str] = {
    "vol_v": "V",
    "voh_v": "V",
    "rise_time_s": "s",
    "fall_time_s": "s",
    "prop_delay_s": "s",
}

_TESTBENCH_HEADER = """\
* AUTO-INJECTED TESTBENCH FOR {circuit_id}
* Do not collapse; user netlist appended below verbatim.
"""

_CONTROL_BLOCK = """\
CATHMA_LS_LOAD Vout_hv 0 20f
.control
{analyses}
{measurements}
print {prints}
.endc
.end
"""


def _none_result() -> dict[str, float | None]:
    return {fom: None for fom in FOM_NAMES}


def _level(spec: dict[str, float] | None, key: str, default: float) -> float:
    if spec is None:
        return default
    value = spec.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pick_analyses(spec: dict[str, float] | None) -> str:
    return "set noaskquit\ntran 0.1n 600n"


def _pick_measurements(spec: dict[str, float] | None) -> str:
    vin_mid = 0.5 * _level(spec, "vin_high_v", 1.8)
    vout_high = _level(spec, "vout_high_v", 3.3)
    vout_lo = 0.1 * vout_high
    vout_mid = 0.5 * vout_high
    vout_hi = 0.9 * vout_high
    return "\n".join(
        (
            "meas tran vol_v min v(Vout_hv) from=20n to=580n",
            "meas tran voh_v max v(Vout_hv) from=20n to=580n",
            "meas tran rise_time_s "
            f"trig v(Vout_hv) val={vout_lo:.9g} rise=1 "
            f"targ v(Vout_hv) val={vout_hi:.9g} rise=1",
            "meas tran fall_time_s "
            f"trig v(Vout_hv) val={vout_hi:.9g} fall=1 "
            f"targ v(Vout_hv) val={vout_lo:.9g} fall=1",
            "meas tran prop_delay_s "
            f"trig v(Vin_lv) val={vin_mid:.9g} rise=1 "
            f"targ v(Vout_hv) val={vout_mid:.9g} rise=1",
        )
    )


def _pick_prints() -> str:
    return "v(Vin_lv) v(Vout_hv)"


def _build_deck(user_netlist: str, spec: dict[str, float] | None) -> str:
    nl = normalize_netlist(user_netlist)
    nl = re.sub(r"\.control\b.*?\.endc\b", "", nl, flags=re.DOTALL | re.IGNORECASE)
    nl = re.sub(r"\.end\s*$", "", nl, flags=re.IGNORECASE).rstrip()
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


def measure(
    netlist: str,
    spec: dict[str, float] | None = None,
    timeout_s: int = 30,
) -> dict[str, float | None]:
    """Simulate a voltage level shifter and extract Tier-1 transient FoMs."""
    try:
        passed, _log = grade_netlist(netlist, timeout_s=20)
        if not passed:
            return _none_result()
        stdout = _run_ngspice(_build_deck(netlist, spec), timeout_s)
        if stdout == "__FAILED__":
            return _none_result()
        return {fom: _parse_meas(stdout, fom) for fom in FOM_NAMES}
    except Exception:
        return _none_result()
