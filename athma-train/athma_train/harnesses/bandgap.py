from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from athma_train.harnesses._measure_utils import parse_ac_points, parse_numeric_rows
from athma_train.spice_gate import grade_netlist, normalize_netlist


CIRCUIT_ID: str = "bandgap"
TIER: int = 1
FOM_NAMES: tuple[str, ...] = (
    "vref_v",
    "tempco_ppm_per_c",
    "line_reg_v_per_v",
    "psrr_db",
)
SPEC_UNITS: dict[str, str] = {
    "vref_v": "V",
    "tempco_ppm_per_c": "ppm/C",
    "line_reg_v_per_v": "V/V",
    "psrr_db": "dB",
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
    vdd_nom = (spec or {}).get("vdd_v", 5.0)
    vdd_lo = (spec or {}).get("vdd_min_v", vdd_nom - 0.5)
    vdd_hi = (spec or {}).get("vdd_max_v", vdd_nom + 0.5)
    vdd_step = max((vdd_hi - vdd_lo) / 20.0, 0.01)
    return "\n".join(
        (
            "dc temp -40 85 1",
            "meas dc vref_v find v(Vref) at=27",
            "echo __temp_sweep__",
            "print v(Vref)",
            f"dc Vdd {vdd_lo:.12g} {vdd_hi:.12g} {vdd_step:.12g}",
            f"let line_vec = v(Vref)/({vdd_hi:.12g}-{vdd_lo:.12g})",
            "meas dc line_reg_v_per_v pp line_vec",
            "alter @vdd[acmag]=1",
            "ac lin 1 1k 1k",
            "echo __psrr_ac__",
            "print frequency vdb(Vref)",
        )
    )


def _pick_prints() -> str:
    return "v(Vref)"


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


def _interp_linear(points: list[tuple[float, float]], x: float) -> float | None:
    if not points:
        return None
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1 and x0 != x1:
            return y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return points[-1][1]


def _parse_tempco(stdout: str) -> float | None:
    marker = stdout.find("__temp_sweep__")
    section = stdout[marker:] if marker >= 0 else stdout
    points = [(row[0], row[1]) for row in parse_numeric_rows(section, columns=2)]
    if len(points) < 2:
        return None
    vref_27 = _interp_linear(points, 27.0)
    if vref_27 in (None, 0.0):
        return None
    values = [value for _temp, value in points]
    return (max(values) - min(values)) / vref_27 / 125.0 * 1.0e6


def _parse_psrr(stdout: str) -> float | None:
    points = parse_ac_points(stdout, marker="__psrr_ac__")
    if points:
        return -points[0][1]
    marker = stdout.find("__psrr_ac__")
    section = stdout[marker:] if marker >= 0 else stdout
    match = re.search(r"vdb\(vref\)\s*=\s*([-+0-9.eE]+)", section, re.IGNORECASE)
    if match is None:
        return None
    try:
        return -float(match.group(1))
    except ValueError:
        return None


def measure(
    netlist: str,
    spec: dict[str, float] | None = None,
    timeout_s: int = 30,
) -> dict[str, float | None]:
    """Simulate a bandgap reference netlist and extract Tier-1 FoMs."""
    try:
        passed, _log = grade_netlist(netlist, timeout_s=20)
        if not passed:
            return _none_result()
        stdout = _run_ngspice(_build_deck(netlist, spec), timeout_s)
        if stdout == "__FAILED__":
            return _none_result()
        result = {fom: _parse_meas(stdout, fom) for fom in FOM_NAMES}
        result["tempco_ppm_per_c"] = _parse_tempco(stdout)
        result["psrr_db"] = _parse_psrr(stdout)
        return result
    except Exception:
        return _none_result()
