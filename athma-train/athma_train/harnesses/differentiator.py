from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from athma_train.harnesses._measure_utils import (
    interpolate_crossing_hz,
    has_prepared_device_line,
    parse_ac_points,
    prepare_injected_netlist,
)


CIRCUIT_ID: str = "differentiator"
TIER: int = 0
FOM_NAMES: tuple[str, ...] = (
    "unity_gain_freq_hz",
    "hf_gain_db",
    "output_swing_v",
)
SPEC_UNITS: dict[str, str] = {
    "unity_gain_freq_hz": "Hz",
    "hf_gain_db": "dB",
    "output_swing_v": "V",
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

_VIN_AC_RE = re.compile(
    r"^(\s*v\w+\s+vin\s+\S+\s+)(?!.*\bac\b)(.*)$",
    re.IGNORECASE | re.MULTILINE,
)


def _none_result() -> dict[str, float | None]:
    return {fom: None for fom in FOM_NAMES}


def _pick_analyses(spec: dict[str, float] | None) -> str:
    return "set noaskquit"


def _pick_measurements(spec: dict[str, float] | None) -> str:
    return "\n".join(
        (
            "ac dec 50 10 10Meg",
            "echo __unity_ac__",
            "print frequency vm(Vout)",
            "echo __end_unity_ac__",
            "meas ac hf_gain_db find vdb(Vout) at=1Meg",
            "tran 1u 5m",
            "meas tran output_swing_v pp v(Vout) from=4m to=5m",
        )
    )


def _pick_prints() -> str:
    return "vdb(Vout) vm(Vout) v(Vout)"


def _build_deck(user_netlist: str, spec: dict[str, float] | None) -> str:
    nl = prepare_injected_netlist(user_netlist)
    nl = _VIN_AC_RE.sub(r"\1AC 1 \2", nl, count=1)
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
    """Simulate a differentiator netlist and extract Tier-0 FoMs."""
    try:
        if not has_prepared_device_line(netlist):
            return _none_result()
        stdout = _run_ngspice(_build_deck(netlist, spec), timeout_s)
        if stdout == "__FAILED__":
            return _none_result()
        result = {fom: _parse_meas(stdout, fom) for fom in FOM_NAMES}
        result["unity_gain_freq_hz"] = interpolate_crossing_hz(
            parse_ac_points(stdout, marker="__unity_ac__"),
            1.0,
            direction="rise",
        )
        return result
    except Exception:
        return _none_result()
