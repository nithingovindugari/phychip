from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from athma_train.harnesses._measure_utils import (
    has_prepared_device_line,
    pick_node,
    prepare_injected_netlist,
    three_db_frequency_hz,
)


CIRCUIT_ID: str = "tia"
TIER: int = 2
FOM_NAMES: tuple[str, ...] = (
    "transimpedance_v_per_a",
    "bandwidth_hz",
    "input_noise_a_per_rthz",
    "stability_phase_margin_deg",
)
SPEC_UNITS: dict[str, str] = {
    "transimpedance_v_per_a": "V/A",
    "bandwidth_hz": "Hz",
    "input_noise_a_per_rthz": "A/sqrt(Hz)",
    "stability_phase_margin_deg": "deg",
}

_TESTBENCH_HEADER = """\
* AUTO-INJECTED TESTBENCH FOR {circuit_id}
* Do not collapse; user netlist appended below verbatim.
"""

_CONTROL_BLOCK = """\
{stimulus}
.control
{analyses}
{measurements}
print {prints}
.endc
.end
"""


def _none_result() -> dict[str, float | None]:
    return {fom: None for fom in FOM_NAMES}


def _find_input_current_source(nl: str, input_node: str) -> str | None:
    for line in nl.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].lower().startswith("i"):
            if input_node.lower() in (parts[1].lower(), parts[2].lower()):
                return parts[0]
    return None


def _pick_analyses(spec: dict[str, float] | None, source_name: str | None) -> str:
    alter = f"alter @{source_name}[acmag]=1" if source_name else ""
    return "\n".join(
        (
            "set noaskquit",
            alter,
            "ac dec 50 1 1e8",
        )
    )


def _pick_measurements(spec: dict[str, float] | None, source_name: str) -> str:
    return "\n".join(
        (
            "meas ac transimpedance_v_per_a find vm(Vout) at=10",
            "meas ac tia_peak max vm(Vout) from=1 to=1e8",
            "echo __ac_bw__",
            "print frequency vm(Vout)",
            "echo __end_ac_bw__",
            f"noise v(Vout) {source_name} dec 40 10 1e6",
            "setplot noise1",
        )
    )


def _pick_prints() -> str:
    return "inoise_spectrum"


def _build_deck(user_netlist: str, spec: dict[str, float] | None) -> str:
    nl = prepare_injected_netlist(user_netlist)
    input_node = pick_node(nl, ("i_pd", "inn", "in", "input"), "i_pd")
    source_name = _find_input_current_source(nl, input_node)
    stimulus = "" if source_name else f"I_TIA_HARNESS {input_node} 0 DC 0 AC 1"
    noise_source = source_name or "I_TIA_HARNESS"
    return (
        _TESTBENCH_HEADER.format(circuit_id=CIRCUIT_ID)
        + nl
        + "\n"
        + _CONTROL_BLOCK.format(
            stimulus=stimulus,
            analyses=_pick_analyses(spec, source_name),
            measurements=_pick_measurements(spec, noise_source),
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


def _parse_first_noise(stdout: str) -> float | None:
    in_noise = False
    for line in stdout.splitlines():
        lower = line.lower()
        if "noise spectral density curves" in lower:
            in_noise = True
            continue
        if in_noise and "integrated noise" in lower:
            break
        if in_noise and re.match(r"\s*\d+\s+", line):
            values = re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", line)
            if len(values) >= 3:
                try:
                    return float(values[2])
                except ValueError:
                    return None
    return None


def _parse_phase_margin(stdout: str) -> float | None:
    flatband = _parse_meas(stdout, "transimpedance_v_per_a")
    peak = _parse_meas(stdout, "tia_peak")
    if flatband is None or peak is None or flatband <= 0 or peak <= 0:
        return None
    ratio = peak / flatband
    if ratio <= 1.0:
        return 65.0
    zeta_sq = (1.0 - (1.0 - (1.0 / (ratio * ratio))) ** 0.5) / 2.0
    zeta = zeta_sq**0.5
    denom = ((1.0 + 4.0 * zeta**4) ** 0.5 - 2.0 * zeta_sq) ** 0.5
    if denom <= 0:
        return None
    x = (2.0 * zeta) / denom
    if x <= 0:
        return None
    if x > 1.0:
        return 90.0 - _atan_deg(1.0 / x)
    return _atan_deg(x)


def _atan_deg(x: float) -> float:
    return x * (0.7853981633974483 + 0.273 * (1.0 - x)) * 57.29577951308232


def _has_analysis_data(stdout: str) -> bool:
    lower = stdout.lower()
    return "no. of data rows" in lower or "transimpedance_v_per_a" in lower


def measure(
    netlist: str,
    spec: dict[str, float] | None = None,
    timeout_s: int = 30,
) -> dict[str, float | None]:
    """Simulate a TIA netlist and extract Tier-2 FoMs."""
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
        if result["bandwidth_hz"] is None and has_data:
            result["bandwidth_hz"] = 1.0e8
        result["input_noise_a_per_rthz"] = _parse_first_noise(stdout)
        result["stability_phase_margin_deg"] = _parse_phase_margin(stdout)
        return result
    except Exception:
        return _none_result()
