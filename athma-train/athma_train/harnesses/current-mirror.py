from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from athma_train.harnesses._measure_utils import parse_numeric_rows
from athma_train.spice_gate import grade_netlist, normalize_netlist


CIRCUIT_ID: str = "current-mirror"
TIER: int = 0
FOM_NAMES: tuple[str, ...] = (
    "current_ratio",
    "output_resistance_ohm",
    "compliance_v",
)
SPEC_UNITS: dict[str, str] = {
    "current_ratio": "ratio",
    "output_resistance_ohm": "ohm",
    "compliance_v": "V",
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

_HARNESS_SOURCES = """\
Iref_harn 0 {ref_node} dc {iref_a:g}
Vsweep iout 0 dc {vout_high_v:g}
"""


def _none_result() -> dict[str, float | None]:
    return {fom: None for fom in FOM_NAMES}


def _pick_analyses(spec: dict[str, float] | None) -> str:
    return "set noaskquit"


def _pick_measurements(spec: dict[str, float] | None) -> str:
    iref_a = 100e-6 if spec is None else spec.get("iref_a", 100e-6)
    vout_low_v = 1.0 if spec is None else spec.get("vout_low_v", 1.0)
    vout_high_v = 2.0 if spec is None else spec.get("vout_high_v", 2.0)
    vout_step_v = 0.02 if spec is None else spec.get("vout_step_v", 0.02)
    return "\n".join(
        (
            f"dc Iref_harn {iref_a / 10:g} {iref_a:g} {iref_a / 10:g}",
            f"meas dc iout_nom find i(Vsweep) at={iref_a:g}",
            f"let current_ratio = abs(iout_nom / {iref_a:g})",
            "print current_ratio",
            f"dc Vsweep 0 {vout_high_v:g} {vout_step_v:g}",
            f"meas dc iout_low find i(Vsweep) at={vout_low_v:g}",
            f"meas dc iout_high find i(Vsweep) at={vout_high_v:g}",
            (
                "let output_resistance_ohm = "
                f"abs(({vout_high_v:g} - {vout_low_v:g}) / "
                "(iout_high - iout_low))"
            ),
            "print output_resistance_ohm",
            "echo __compliance_sweep__",
            "print i(Vsweep)",
        )
    )


def _pick_prints() -> str:
    return "current_ratio output_resistance_ohm compliance_v"


def _ref_node(nl: str) -> str:
    nodes = {
        token.lower()
        for line in nl.splitlines()
        for token in line.split()[1:3]
        if line.strip() and not line.lstrip().startswith(("*", "."))
    }
    return "iref" if "iref" in nodes else "ref"


def _strip_user_bias_and_load(nl: str, ref_node: str) -> str:
    drop_nodes = {ref_node.lower(), "iout"}
    keep = []
    for line in nl.splitlines():
        m = re.match(r"^\s*[RIV]\w*\s+(\S+)\s+(\S+)\b", line, re.IGNORECASE)
        if m and ({m.group(1).lower(), m.group(2).lower()} & drop_nodes):
            continue
        keep.append(line)
    return "\n".join(keep)


def _build_deck(user_netlist: str, spec: dict[str, float] | None) -> str:
    nl = normalize_netlist(user_netlist)
    nl = re.sub(r"\.control\b.*?\.endc\b", "", nl, flags=re.DOTALL | re.IGNORECASE)
    nl = re.sub(r"\.end\s*$", "", nl, flags=re.IGNORECASE).rstrip()
    ref_node = _ref_node(nl)
    nl = _strip_user_bias_and_load(nl, ref_node)
    iref_a = 100e-6 if spec is None else spec.get("iref_a", 100e-6)
    vout_high_v = 2.0 if spec is None else spec.get("vout_high_v", 2.0)
    return (
        _TESTBENCH_HEADER.format(circuit_id=CIRCUIT_ID)
        + nl
        + "\n"
        + _HARNESS_SOURCES.format(
            ref_node=ref_node,
            iref_a=iref_a,
            vout_high_v=vout_high_v,
        )
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


def _parse_compliance(stdout: str) -> float | None:
    marker = stdout.find("__compliance_sweep__")
    section = stdout[marker:] if marker >= 0 else stdout
    points = [(row[0], abs(row[1])) for row in parse_numeric_rows(section, columns=2)]
    if len(points) < 2:
        return None
    target = 0.9 * points[-1][1]
    for (v0, i0), (v1, i1) in zip(points, points[1:]):
        if i0 == i1:
            continue
        if i0 <= target <= i1 or i1 <= target <= i0:
            return v0 + (target - i0) / (i1 - i0) * (v1 - v0)
    return None


def measure(
    netlist: str,
    spec: dict[str, float] | None = None,
    timeout_s: int = 30,
) -> dict[str, float | None]:
    """Simulate a current mirror and extract DC ratio, Rout, and compliance."""
    try:
        passed, _log = grade_netlist(netlist, timeout_s=20)
        if not passed:
            return _none_result()
        stdout = _run_ngspice(_build_deck(netlist, spec), timeout_s)
        if stdout == "__FAILED__":
            return _none_result()
        result = {fom: _parse_meas(stdout, fom) for fom in FOM_NAMES}
        result["compliance_v"] = _parse_compliance(stdout)
        return result
    except Exception:
        return _none_result()
