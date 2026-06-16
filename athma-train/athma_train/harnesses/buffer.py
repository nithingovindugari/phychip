from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from athma_train.harnesses._measure_utils import (
    has_prepared_device_line,
    prepare_injected_netlist,
    three_db_frequency_hz,
)


CIRCUIT_ID: str = "buffer"
TIER: int = 0
FOM_NAMES: tuple[str, ...] = (
    "gain_db",
    "bandwidth_hz",
    "output_z_ohm",
    "input_offset_v",
)
SPEC_UNITS: dict[str, str] = {
    "gain_db": "dB",
    "bandwidth_hz": "Hz",
    "output_z_ohm": "ohm",
    "input_offset_v": "V",
}

_TESTBENCH_HEADER = """\
* AUTO-INJECTED TESTBENCH FOR {circuit_id}
* Do not collapse; user netlist appended below verbatim.
"""

_CONTROL_BLOCK = """\
Ibuf_z Vout 0 DC 0 AC 0
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


def _pick_measurements(spec: dict[str, float] | None) -> str:
    return "\n".join(
        (
            "dc Vin 0 1m 1m",
            "meas dc input_offset_v find v(Vout) at=0",
            "alter @Vin[acmag]=1",
            "alter @Ibuf_z[acmag]=0",
            "ac dec 50 1 1e8",
            "meas ac gain_db max vdb(Vout) from=1 to=1e3",
            "echo __ac_bw__",
            "print frequency vdb(Vout)",
            "echo __end_ac_bw__",
            "alter @Vin[acmag]=0",
            "alter @Ibuf_z[acmag]=1",
            "ac dec 10 1e3 2e3",
            "meas ac output_z_ohm max vm(Vout) from=1e3 to=2e3",
        )
    )


def _pick_prints() -> str:
    return "vdb(Vout) vm(Vout)"


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


def measure(
    netlist: str,
    spec: dict[str, float] | None = None,
    timeout_s: int = 30,
) -> dict[str, float | None]:
    """Simulate a unity-gain buffer netlist and extract Tier-0 FoMs."""
    try:
        if not has_prepared_device_line(netlist):
            return _none_result()
        stdout = _run_ngspice(_build_deck(netlist, spec), timeout_s)
        if stdout == "__FAILED__":
            return _none_result()
        result = {fom: _parse_meas(stdout, fom) for fom in FOM_NAMES}
        result["bandwidth_hz"] = three_db_frequency_hz(
            stdout,
            fallback_hz=1.0e8,
            marker="__ac_bw__",
        )
        return result
    except Exception:
        return _none_result()
