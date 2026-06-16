from __future__ import annotations

import math
import re
import subprocess
import tempfile
from pathlib import Path

from athma_train.harnesses._measure_utils import interpolate_crossing_hz, parse_numeric_rows
from athma_train.harnesses._measure_utils import has_prepared_device_line, prepare_injected_netlist


CIRCUIT_ID: str = "sallen-key"
TIER: int = 1
FOM_NAMES: tuple[str, ...] = (
    "cutoff_hz",
    "dc_gain_db",
    "q_factor",
    "stopband_atten_db",
)
SPEC_UNITS: dict[str, str] = {
    "cutoff_hz": "Hz",
    "dc_gain_db": "dB",
    "q_factor": "ratio",
    "stopband_atten_db": "dB",
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
    return "set noaskquit\nac dec 50 1 1e6"


def _pick_measurements(spec: dict[str, float] | None) -> str:
    stopband_hz = 100_000.0
    if spec and spec.get("cutoff_hz"):
        cutoff_spec = spec["cutoff_hz"]
        if isinstance(cutoff_spec, (list, tuple)) and len(cutoff_spec) == 2:
            cutoff = (float(cutoff_spec[0]) + float(cutoff_spec[1])) / 2.0
        else:
            cutoff = float(cutoff_spec)
        stopband_hz = min(max(cutoff * 100.0, 1.0), 1e6)
    return "\n".join(
        (
            "let gain_db_trace = vdb(Vout) - vdb(Vin)",
            "let phase_trace = 180 / PI * cph(Vout / Vin)",
            f"let stopband_hz_marker = {stopband_hz:g}",
            "echo __ac_filter__",
            "print frequency gain_db_trace phase_trace",
        )
    )


def _pick_prints() -> str:
    return "cutoff_hz dc_gain_db q_factor stopband_atten_db"


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


def _value_at(points: list[tuple[float, float]], frequency_hz: float) -> float | None:
    if not points:
        return None
    if frequency_hz <= points[0][0]:
        return points[0][1]
    for (f0, y0), (f1, y1) in zip(points, points[1:]):
        if f0 <= frequency_hz <= f1 and f0 > 0.0 and f1 > 0.0:
            ratio = (math.log10(frequency_hz) - math.log10(f0)) / (
                math.log10(f1) - math.log10(f0)
            )
            return y0 + ratio * (y1 - y0)
    return points[-1][1]


def _parse_filter(stdout: str, spec: dict[str, float] | None) -> dict[str, float | None]:
    marker = stdout.find("__ac_filter__")
    section = stdout[marker:] if marker >= 0 else stdout
    rows = parse_numeric_rows(section, columns=3)
    gain_points = [(row[0], row[1]) for row in rows if row[0] > 0.0]
    phase_points = [(row[0], row[2]) for row in rows if row[0] > 0.0]
    dc_gain = gain_points[0][1] if gain_points else None
    cutoff = None
    natural = None
    q_factor = None
    stopband = None
    if dc_gain is not None:
        cutoff = interpolate_crossing_hz(gain_points, dc_gain - 3.0, direction="fall")
    natural = interpolate_crossing_hz(phase_points, -90.0, direction="fall")
    if cutoff is not None and natural not in (None, 0.0):
        cutoff_ratio = cutoff / natural
        denom = 2.0 - (1.0 - cutoff_ratio * cutoff_ratio) ** 2
        if denom > 0.0:
            q_factor = cutoff_ratio / math.sqrt(denom)
    stopband_hz = 100_000.0
    if spec and spec.get("cutoff_hz"):
        cutoff_spec = spec["cutoff_hz"]
        if isinstance(cutoff_spec, (list, tuple)) and len(cutoff_spec) == 2:
            cutoff = (float(cutoff_spec[0]) + float(cutoff_spec[1])) / 2.0
        else:
            cutoff = float(cutoff_spec)
        stopband_hz = min(max(cutoff * 100.0, 1.0), 1e6)
    stopband_gain = _value_at(gain_points, stopband_hz)
    if dc_gain is not None and stopband_gain is not None:
        stopband = dc_gain - stopband_gain
    return {
        "cutoff_hz": cutoff,
        "dc_gain_db": dc_gain,
        "q_factor": q_factor,
        "stopband_atten_db": stopband,
    }


def measure(
    netlist: str,
    spec: dict[str, float] | None = None,
    timeout_s: int = 30,
) -> dict[str, float | None]:
    """Simulate a Sallen-Key low-pass netlist and extract Tier-1 FoMs."""
    try:
        if not has_prepared_device_line(netlist):
            return _none_result()
        stdout = _run_ngspice(_build_deck(netlist, spec), timeout_s)
        if stdout == "__FAILED__":
            return _none_result()
        return _parse_filter(stdout, spec)
    except Exception:
        return _none_result()
