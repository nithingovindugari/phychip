from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from athma_train.harnesses._measure_utils import three_db_frequency_hz
from athma_train.spice_gate import grade_netlist, normalize_netlist


CIRCUIT_ID: str = "ina"
TIER: int = 1
FOM_NAMES: tuple[str, ...] = (
    "differential_gain_db",
    "cmrr_db",
    "bandwidth_hz",
    "input_offset_v",
)
SPEC_UNITS: dict[str, str] = {
    "differential_gain_db": "dB",
    "cmrr_db": "dB",
    "bandwidth_hz": "Hz",
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

_AC_START_HZ = 1.0
_AC_STOP_HZ = 1.0e8
_DIFF_HALF_AC_V = 0.5


def _none_result() -> dict[str, float | None]:
    return {fom: None for fom in FOM_NAMES}


def _pick_analyses(spec: dict[str, float] | None) -> str:
    return "set noaskquit"


def _source_dc(netlist: str, source: str) -> float:
    pattern = re.compile(
        rf"^\s*{re.escape(source)}\s+\S+\s+\S+\b(.*)$",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(netlist)
    if match is None:
        return 0.0
    value_match = re.search(
        r"\bdc\s*=?\s*([-+0-9.eE]+)|\s([-+0-9.eE]+)(?:\s|$)",
        match.group(1),
        re.IGNORECASE,
    )
    return float(value_match.group(1) or value_match.group(2)) if value_match else 0.0


def _pick_measurements(spec: dict[str, float] | None, vp_dc: float, vn_dc: float) -> str:
    return "\n".join(
        (
            "alter @Vp[acmag]=0.5",
            "alter @Vp[acphase]=0",
            "alter @Vn[acmag]=0.5",
            "alter @Vn[acphase]=180",
            f"ac dec 50 {_AC_START_HZ:g} {_AC_STOP_HZ:g}",
            "meas ac differential_gain_db max vdb(Vout) from=1 to=1e3",
            "echo __ac_bw__",
            "print frequency vdb(Vout)",
            "echo __end_ac_bw__",
            "alter @Vp[acmag]=1",
            "alter @Vp[acphase]=0",
            "alter @Vn[acmag]=1",
            "alter @Vn[acphase]=0",
            f"ac dec 100 {_AC_START_HZ:g} {_AC_STOP_HZ:g}",
            "meas ac common_gain_db max vdb(Vout) from=1 to=1e3",
            f"alter @Vp[dc]={vp_dc:.12g}",
            f"alter @Vn[dc]={vn_dc:.12g}",
            "op",
            "let vout_q = v(Vout)",
            "let vp_q = v(Vp)",
            "print vp_q",
            f"dc Vp {vp_dc - 0.05:.12g} {vp_dc + 0.05:.12g} 50u",
            "meas dc vp_cross find v(Vp) when v(Vout)=vout_q cross=1",
        )
    )


def _pick_prints() -> str:
    return "vdb(Vout) v(Vout)"


def _build_deck(user_netlist: str, spec: dict[str, float] | None) -> str:
    nl = normalize_netlist(user_netlist)
    nl = re.sub(r"\.control\b.*?\.endc\b", "", nl, flags=re.DOTALL | re.IGNORECASE)
    nl = re.sub(r"\.end\s*$", "", nl, flags=re.IGNORECASE).rstrip()
    vp_dc = _source_dc(nl, "Vp")
    vn_dc = _source_dc(nl, "Vn")
    return (
        _TESTBENCH_HEADER.format(circuit_id=CIRCUIT_ID)
        + nl
        + "\n"
        + _CONTROL_BLOCK.format(
            analyses=_pick_analyses(spec),
            measurements=_pick_measurements(spec, vp_dc, vn_dc),
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


def _parse_cmrr(stdout: str, differential_gain_db: float | None) -> float | None:
    common_gain_db = _parse_meas(stdout, "common_gain_db")
    if differential_gain_db is None or common_gain_db is None:
        return None
    return differential_gain_db - common_gain_db


def _parse_input_offset(stdout: str) -> float | None:
    vp_q = _parse_meas(stdout, "vp_q")
    vp_cross = _parse_meas(stdout, "vp_cross")
    if vp_q is None:
        return None
    if vp_cross is None:
        return 0.0
    return abs(vp_cross - vp_q)


def measure(
    netlist: str,
    spec: dict[str, float] | None = None,
    timeout_s: int = 30,
) -> dict[str, float | None]:
    """Simulate a 3-op-amp instrumentation amplifier and extract Tier-1 FoMs."""
    try:
        passed, _log = grade_netlist(netlist, timeout_s=20)
        if not passed:
            return _none_result()
        stdout = _run_ngspice(_build_deck(netlist, spec), timeout_s)
        if stdout == "__FAILED__":
            return _none_result()
        differential_gain_db = _parse_meas(stdout, "differential_gain_db")
        return {
            "differential_gain_db": differential_gain_db,
            "cmrr_db": _parse_cmrr(stdout, differential_gain_db),
            "bandwidth_hz": three_db_frequency_hz(
                stdout,
                fallback_hz=_AC_STOP_HZ,
                marker="__ac_bw__",
            ),
            "input_offset_v": _parse_input_offset(stdout),
        }
    except Exception:
        return _none_result()
