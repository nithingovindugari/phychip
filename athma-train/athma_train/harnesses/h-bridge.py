from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from athma_train.harnesses._measure_utils import (
    has_prepared_device_line,
    pick_node,
    prepare_injected_netlist,
)


CIRCUIT_ID: str = "h-bridge"
TIER: int = 2
FOM_NAMES: tuple[str, ...] = (
    "vmotor_pp_v",
    "rds_on_ohm",
    "switching_loss_w",
    "deadtime_s",
)
SPEC_UNITS: dict[str, str] = {
    "vmotor_pp_v": "V",
    "rds_on_ohm": "ohm",
    "switching_loss_w": "W",
    "deadtime_s": "s",
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


def _as_float(spec: dict[str, float] | None, key: str, default: float) -> float:
    if spec is None:
        return default
    try:
        return float(spec.get(key, default))
    except (TypeError, ValueError):
        return default


def _mos_switches(netlist: str) -> list[tuple[str, str, str]]:
    switches: list[tuple[str, str, str]] = []
    for line in netlist.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("*", ".")):
            continue
        parts = stripped.split()
        if len(parts) >= 6 and parts[0].lower().startswith("m"):
            switches.append((parts[0].lower(), parts[1].lower(), parts[3].lower()))
    return switches


def _remove_duplicate_behavioral_gate_drives(nl: str) -> str:
    bsource_nodes = set()
    for line in nl.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].lower().startswith("b"):
            bsource_nodes.add((parts[1].lower(), parts[2].lower()))
    kept: list[str] = []
    for line in nl.splitlines():
        parts = line.split()
        if (
            len(parts) >= 3
            and parts[0].lower().startswith("e")
            and (parts[1].lower(), parts[2].lower()) in bsource_nodes
        ):
            continue
        kept.append(line)
    return "\n".join(kept)


def _node_vdrop_expr(a: str, b: str) -> str:
    if a == "0":
        return f"abs(v({b}))"
    if b == "0":
        return f"abs(v({a}))"
    return f"abs(v({a})-v({b}))"


def _pick_analyses(switches: list[tuple[str, str, str]]) -> str:
    saved = " ".join(f"@{name}[id]" for name, _drain, _source in switches)
    if saved:
        return f"set noaskquit\nsave all {saved}\ntran 0.1u 400u 0 0.1u uic"
    return "set noaskquit\ntran 0.1u 400u 0 0.1u uic"


def _pick_measurements(
    spec: dict[str, float] | None,
    switches: list[tuple[str, str, str]],
    ctrl_a: str,
    ctrl_b: str,
) -> str:
    start_s = _as_float(spec, "measure_start_s", 50e-6)
    stop_s = _as_float(spec, "measure_stop_s", 400e-6)
    threshold_v = _as_float(spec, "ctrl_threshold_v", 2.5)
    if switches:
        power_terms = [
            f"{_node_vdrop_expr(drain, source)}*abs(@{name}[id])"
            for name, drain, source in switches
        ]
        current_terms = [f"@{name}[id]*@{name}[id]" for name, _d, _s in switches]
        psw = " + ".join(power_terms)
        isw_sq = " + ".join(current_terms)
    else:
        psw = "0"
        isw_sq = "0"
    return "\n".join(
        (
            "let vdiff = v(vmotor_p) - v(vmotor_n)",
            f"let psw = {psw}",
            f"let isw_sq = {isw_sq}",
            "let rinst = psw / isw_sq",
            f"meas tran vmotor_pp_v pp vdiff from={start_s} to={stop_s}",
            f"meas tran rds_on_ohm avg rinst from={start_s} to={stop_s}",
            f"meas tran switching_loss_w avg psw from={start_s} to={stop_s}",
            (
                "meas tran deadtime_s trig v(vctrl_a) "
                f"val={threshold_v} fall=1 targ v({ctrl_b}) "
                f"val={threshold_v} rise=2"
            ).replace("v(vctrl_a)", f"v({ctrl_a})"),
        )
    )


def _pick_prints(ctrl_a: str, ctrl_b: str) -> str:
    return f"vdiff psw rinst v({ctrl_a}) v({ctrl_b})"


def _build_deck(user_netlist: str, spec: dict[str, float] | None) -> str:
    nl = _remove_duplicate_behavioral_gate_drives(prepare_injected_netlist(user_netlist))
    switches = _mos_switches(nl)
    ctrl_a = pick_node(nl, ("vctrl_a", "ga", "ctrl_a", "ina"), "vctrl_a")
    ctrl_b = pick_node(nl, ("vctrl_b", "gb", "ctrl_b", "inb"), "vctrl_b")
    return (
        _TESTBENCH_HEADER.format(circuit_id=CIRCUIT_ID)
        + nl
        + "\n"
        + _CONTROL_BLOCK.format(
            analyses=_pick_analyses(switches),
            measurements=_pick_measurements(spec, switches, ctrl_a, ctrl_b),
            prints=_pick_prints(ctrl_a, ctrl_b),
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


def _has_analysis_data(stdout: str) -> bool:
    lower = stdout.lower()
    return "no. of data rows" in lower or "vmotor_pp_v" in lower


def measure(
    netlist: str,
    spec: dict[str, float] | None = None,
    timeout_s: int = 30,
) -> dict[str, float | None]:
    """Simulate an H-bridge netlist and extract Tier-2 transient FoMs."""
    try:
        if not has_prepared_device_line(netlist):
            return _none_result()
        stdout = _run_ngspice(_build_deck(netlist, spec), timeout_s)
        if stdout == "__FAILED__":
            return _none_result()
        result = {fom: _parse_meas(stdout, fom) for fom in FOM_NAMES}
        if result["deadtime_s"] is None and _has_analysis_data(stdout):
            result["deadtime_s"] = 0.0
        return result
    except Exception:
        return _none_result()
