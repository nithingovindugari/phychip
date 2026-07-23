from __future__ import annotations

import re

from phy_chip_env.harnesses._measure_utils import parse_numeric_rows
from phy_chip_env.sandbox import grade_netlist, normalize_netlist

CIRCUIT_ID: str = "buck"
TIER: int = 1
FOM_NAMES: tuple[str, ...] = (
    "vout_steady_v",
    "ripple_pp_v",
    "efficiency_pct",
    "settling_time_s",
)
SPEC_UNITS: dict[str, str] = {
    "vout_steady_v": "V",
    "ripple_pp_v": "V",
    "efficiency_pct": "%",
    "settling_time_s": "s",
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


def _spice_float(value: str) -> float | None:
    match = re.fullmatch(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)([a-z]*)",
        value.strip().lower(),
    )
    if not match:
        return None
    scale_by_suffix = {
        "": 1.0,
        "f": 1e-15,
        "p": 1e-12,
        "n": 1e-9,
        "u": 1e-6,
        "m": 1e-3,
        "k": 1e3,
        "meg": 1e6,
        "g": 1e9,
        "t": 1e12,
    }
    suffix = match.group(2)
    if suffix not in scale_by_suffix:
        return None
    return float(match.group(1)) * scale_by_suffix[suffix]


def _find_vin_source(nl: str) -> tuple[str, float]:
    for line in nl.splitlines():
        parts = line.split()
        if len(parts) < 4 or not parts[0].lower().startswith("v"):
            continue
        node_a, node_b = parts[1].lower(), parts[2].lower()
        if {node_a, node_b} != {"vin", "0"}:
            continue
        sign = -1.0 if node_a == "vin" else 1.0
        return parts[0], sign
    return "Vin", -1.0


def _find_load_ohm(nl: str, spec: dict[str, float] | None = None) -> float | None:
    # `spec` is accepted for compatibility but intentionally ignored. An
    # efficiency denominator must describe a branch that is actually present
    # in the simulated circuit, not a requested load the harness did not inject.
    for line in nl.splitlines():
        parts = line.split()
        if len(parts) < 4 or not parts[0].lower().startswith("r"):
            continue
        node_a, node_b = parts[1].lower(), parts[2].lower()
        if {node_a, node_b} != {"vout", "0"}:
            continue
        value = _spice_float(parts[3])
        if value and value > 0:
            return value
    return None


def _pick_analyses(spec: dict[str, float] | None) -> str:
    return "set noaskquit\ntran 100n 5m 0 100n"


def _pick_measurements(nl: str, spec: dict[str, float] | None) -> str:
    source_name, input_power_sign = _find_vin_source(nl)
    load_ohm = _find_load_ohm(nl, spec)
    measurements = [
        "meas tran vout_steady_v avg v(Vout) from=4m to=5m",
        "meas tran ripple_pp_v pp v(Vout) from=4m to=5m",
    ]
    if load_ohm is not None:
        measurements.extend(
            (
                f"let pin_w_trace = {input_power_sign:g} * v(Vin) * i({source_name})",
                f"let pout_w_trace = (v(Vout) * v(Vout)) / {load_ohm:.12g}",
                "meas tran pin_w avg pin_w_trace from=4m to=5m",
                "meas tran pout_w avg pout_w_trace from=4m to=5m",
                "let efficiency_pct_trace = 100 * pout_w / pin_w + 0 * time",
                "meas tran efficiency_pct avg efficiency_pct_trace from=4m to=5m",
            )
        )
    measurements.extend(("echo __settling_tran__", "print time v(Vout)"))
    return "\n".join(measurements)


def _pick_prints() -> str:
    return "v(Vin) v(Vout)"


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
            measurements=_pick_measurements(nl, spec),
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


def _parse_settling_time(stdout: str, steady_v: float | None) -> float | None:
    if steady_v is None:
        return None
    marker = stdout.find("__settling_tran__")
    section = stdout[marker:] if marker >= 0 else stdout
    rows = parse_numeric_rows(section, columns=2)
    if len(rows) < 2:
        return None
    target = 0.98 * steady_v
    for (t0, v0), (t1, v1) in zip(rows, rows[1:]):
        if v0 == v1:
            continue
        if v0 <= target <= v1:
            return t0 + (target - v0) / (v1 - v0) * (t1 - t0)
    return None


def measure(
    netlist: str,
    spec: dict[str, float] | None = None,
    timeout_s: int = 30,
) -> dict[str, float | None]:
    """Simulate a buck converter netlist and extract Tier-1 FoMs."""
    try:
        passed, _log = grade_netlist(netlist, timeout_s=20)
        if not passed:
            return _none_result()
        stdout = _run_ngspice(_build_deck(netlist, spec), timeout_s)
        if stdout == "__FAILED__":
            return _none_result()
        result = {fom: _parse_meas(stdout, fom) for fom in FOM_NAMES}
        result["settling_time_s"] = _parse_settling_time(
            stdout,
            result["vout_steady_v"],
        )
        return result
    except Exception:
        return _none_result()
