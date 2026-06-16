from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from athma_train.spice_gate import grade_netlist, normalize_netlist


CIRCUIT_ID: str = "boost"
TIER: int = 1
FOM_NAMES: tuple[str, ...] = (
    "vout_steady_v",
    "ripple_pp_v",
    "efficiency_pct",
    "inductor_current_avg_a",
)
SPEC_UNITS: dict[str, str] = {
    "vout_steady_v": "V",
    "ripple_pp_v": "V",
    "efficiency_pct": "%",
    "inductor_current_avg_a": "A",
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

_VALUE_SUFFIXES = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}


def _none_result() -> dict[str, float | None]:
    return {fom: None for fom in FOM_NAMES}


def _spice_value_to_float(value: str) -> float | None:
    match = re.match(
        r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)([a-z]*)",
        value,
        re.I,
    )
    if not match:
        return None
    number = float(match.group(1))
    suffix = match.group(2).lower()
    return number * _VALUE_SUFFIXES.get(suffix, 1.0)


def _first_voltage_source_to_ground(netlist: str) -> tuple[str, str] | None:
    for line in netlist.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].lower().startswith("v") and parts[2] == "0":
            if "pulse" not in " ".join(parts[3:]).lower():
                return parts[0], parts[1]
    return None


def _vout_load_resistance(netlist: str) -> float | None:
    for line in netlist.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].lower().startswith("r"):
            nodes = {parts[1].lower(), parts[2].lower()}
            if nodes == {"vout", "0"}:
                return _spice_value_to_float(parts[3])
    return None


def _first_inductor_name(netlist: str) -> str | None:
    for line in netlist.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].lower().startswith("l"):
            return parts[0]
    return None


def _pick_analyses(spec: dict[str, float] | None) -> str:
    return "set noaskquit\ntran 200n 5m uic"


def _pick_measurements(netlist: str, spec: dict[str, float] | None) -> str:
    source = _first_voltage_source_to_ground(netlist)
    load_r = _vout_load_resistance(netlist)
    inductor = _first_inductor_name(netlist)
    lines = [
        "meas tran vout_steady_v avg v(Vout) from=4m to=5m",
        "meas tran ripple_pp_v pp v(Vout) from=4m to=5m",
    ]
    if source is not None and load_r is not None:
        source_name, input_node = source
        lines.extend(
            (
                f"let p_out_inst = v(Vout)*v(Vout)/{load_r:.12g}",
                f"let p_in_inst = -v({input_node})*i({source_name})",
                "meas tran p_out_w avg p_out_inst from=4m to=5m",
                "meas tran p_in_w avg p_in_inst from=4m to=5m",
                "let efficiency_pct_wave = 0*time + 100*p_out_w/p_in_w",
                "meas tran efficiency_pct avg efficiency_pct_wave from=4m to=5m",
            )
        )
    if inductor is not None:
        lines.append(
            f"meas tran inductor_current_avg_a avg i({inductor}) from=4m to=5m"
        )
    return "\n".join(lines)


def _pick_prints() -> str:
    return "v(Vout)"


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
            measurements=_pick_measurements(nl, spec),
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
    match = re.search(
        rf"^\s*{re.escape(name.lower())}\s*=\s*([-+0-9.eE]+)\b",
        stdout.lower(),
        re.MULTILINE,
    )
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def measure(
    netlist: str,
    spec: dict[str, float] | None = None,
    timeout_s: int = 30,
) -> dict[str, float | None]:
    """Simulate a boost converter netlist and extract Tier-1 FoMs."""
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
