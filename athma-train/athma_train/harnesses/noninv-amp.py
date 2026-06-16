from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from athma_train.harnesses._measure_utils import three_db_frequency_hz
from athma_train.spice_gate import grade_netlist, normalize_netlist


CIRCUIT_ID: str = "noninv-amp"
TIER: int = 0
FOM_NAMES: tuple[str, ...] = (
    "gain_db",
    "bandwidth_hz",
    "output_swing_v",
    "input_offset_v",
)
SPEC_UNITS: dict[str, str] = {
    "gain_db": "dB",
    "bandwidth_hz": "Hz",
    "output_swing_v": "V",
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


def _pick_analyses(spec: dict[str, float] | None) -> str:
    return "set noaskquit"


def _pick_measurements(spec: dict[str, float] | None) -> str:
    vdd = (spec or {}).get("vdd", 5.0)
    sweep_v = min(vdd * 0.85, 3.0)
    return "\n".join(
        (
            f"dc Vin -{sweep_v:.2f} {sweep_v:.2f} 0.05",
            "meas dc input_offset_v when v(Vout)=0",
            "meas dc output_swing_v pp v(Vout)",
            "ac dec 50 1 1e8",
            "meas ac gain_db max vdb(Vout) from=1 to=1e3",
            "echo __ac_bw__",
            "print frequency vdb(Vout)",
            "echo __end_ac_bw__",
        )
    )


def _pick_prints() -> str:
    return "vdb(Vout) vm(Vout)"


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
    """Simulate a non-inverting amplifier netlist and extract Tier-0 FoMs."""
    try:
        passed, _log = grade_netlist(netlist, timeout_s=20)
        if not passed:
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
