from __future__ import annotations

import re

from phy_chip_env.harnesses._measure_utils import three_db_frequency_hz
from phy_chip_env.sandbox import grade_netlist, normalize_netlist

CIRCUIT_ID: str = "gain-trim"
TIER: int = 1
FOM_NAMES: tuple[str, ...] = (
    "gain_db",
    "offset_v",
    "trim_range_v",
    "bandwidth_hz",
    "gain_modulation_db",
)
SPEC_UNITS: dict[str, str] = {
    "gain_db": "dB",
    "offset_v": "V",
    "trim_range_v": "V",
    "bandwidth_hz": "Hz",
    "gain_modulation_db": "dB",
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


def _trim_span(spec: dict[str, float] | None) -> float:
    if not spec:
        return 0.1
    raw_value = spec.get("trim_span_v", spec.get("vtrim_span_v", 0.1))
    try:
        value = abs(float(raw_value))
    except (TypeError, ValueError):
        return 0.1
    return value if value > 0 else 0.1


def _pick_analyses(spec: dict[str, float] | None) -> str:
    return "set noaskquit"


def _pick_measurements(spec: dict[str, float] | None) -> str:
    trim_span = _trim_span(spec)
    return "\n".join(
        (
            "alter Vin dc 0",
            "alter Vin ac 1",
            "alter Vtrim dc 0",
            "alter Vtrim ac 0",
            "dc Vin -1u 1u 1u",
            "meas dc offset_v avg v(Vout) from=-1u to=1u",
            f"dc Vtrim {-trim_span:.12g} {trim_span:.12g} {trim_span / 20:.12g}",
            "meas dc trim_range_v pp v(Vout)",
            "alter Vin dc 0",
            "alter Vin ac 1",
            "alter Vtrim ac 0",
            f"alter Vtrim dc {-trim_span:.12g}",
            "ac lin 2 10 11",
            "meas ac gain_trim_low_db max vdb(Vout) from=10 to=11",
            f"alter Vtrim dc {trim_span:.12g}",
            "ac lin 2 10 11",
            "meas ac gain_trim_high_db max vdb(Vout) from=10 to=11",
            "alter Vin dc 0",
            "alter Vin ac 1",
            "alter Vtrim dc 0",
            "alter Vtrim ac 0",
            "ac dec 50 1 1e8",
            "meas ac gain_db max vdb(Vout) from=1 to=1e3",
            "echo __ac_bw__",
            "print frequency vdb(Vout)",
            "echo __end_ac_bw__",
        )
    )


def _pick_prints() -> str:
    return "v(Vout) vdb(Vout) vm(Vout)"


def _build_deck(user_netlist: str, spec: dict[str, float] | None) -> str:
    nl = normalize_netlist(user_netlist)
    nl = re.sub(
        r"(?m)^[ \t]*\.control\b.*?^[ \t]*\.endc\b[^\r\n]*(?:\r?\n|$)",
        "",
        nl,
        flags=re.DOTALL | re.IGNORECASE,
    )
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
    from phy_chip_env.sandbox import run_ngspice_sandboxed

    try:
        proc = run_ngspice_sandboxed(deck, timeout_s=timeout_s)
    except Exception:
        return "__FAILED__"
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


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
    """Simulate a gain-trim stage and extract DC trim plus AC response FoMs."""
    try:
        passed, _log = grade_netlist(netlist, timeout_s=20)
        if not passed:
            return _none_result()
        stdout = _run_ngspice(_build_deck(netlist, spec), timeout_s)
        if stdout == "__FAILED__":
            return _none_result()
        result = {
            fom: _parse_meas(stdout, fom)
            for fom in FOM_NAMES
            if fom != "gain_modulation_db"
        }
        gain_low = _parse_meas(stdout, "gain_trim_low_db")
        gain_high = _parse_meas(stdout, "gain_trim_high_db")
        result["gain_modulation_db"] = (
            abs(gain_high - gain_low)
            if gain_low is not None and gain_high is not None
            else None
        )
        result["bandwidth_hz"] = three_db_frequency_hz(stdout, marker="__ac_bw__")
        return result
    except Exception:
        return _none_result()
