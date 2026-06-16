from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from athma_train.harnesses._measure_utils import (
    interpolate_crossing_hz,
    has_prepared_device_line,
    parse_ac_points,
    prepare_injected_netlist,
)


CIRCUIT_ID: str = "integrator"
TIER: int = 0
FOM_NAMES: tuple[str, ...] = (
    "unity_gain_freq_hz",
    "dc_gain_db",
    "output_swing_v",
)
SPEC_UNITS: dict[str, str] = {
    "unity_gain_freq_hz": "Hz",
    "dc_gain_db": "dB",
    "output_swing_v": "V",
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

_OUTPUT_NODE_CANDIDATES: tuple[str, ...] = ("vout", "out", "vo", "output")
_INPUT_NODE_CANDIDATES: tuple[str, ...] = ("vin", "in", "input")
_VSRC_LINE_RE = re.compile(
    r"^(?P<name>V\w*)\s+(?P<n1>\S+)\s+(?P<n2>\S+)\s+(?P<rest>.+)$",
    re.IGNORECASE,
)
_AC_FIELD_RE = re.compile(r"\bAC\s+[-+0-9.eE]+", re.IGNORECASE)


def _none_result() -> dict[str, float | None]:
    return {fom: None for fom in FOM_NAMES}


def _detect_node(nl: str, candidates: tuple[str, ...], fallback: str) -> str:
    nodes: set[str] = set()
    for line in nl.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("*", ".")):
            continue
        nodes.update(token.lower() for token in stripped.split()[1:])
    for candidate in candidates:
        if candidate in nodes:
            return candidate
    return fallback


def _ensure_ac_on_input(nl: str, input_node: str) -> str:
    lines: list[str] = []
    fixed = False
    for line in nl.splitlines():
        match = _VSRC_LINE_RE.match(line.strip())
        if fixed or not match:
            lines.append(line)
            continue
        if input_node not in (match["n1"].lower(), match["n2"].lower()):
            lines.append(line)
            continue
        rest = match["rest"]
        if _AC_FIELD_RE.search(rest):
            rest = _AC_FIELD_RE.sub("AC 1", rest, count=1)
        else:
            rest = "AC 1 " + rest
        lines.append(f"{match['name']} {match['n1']} {match['n2']} {rest}")
        fixed = True
    return "\n".join(lines)


def _pick_analyses(spec: dict[str, float] | None, output_node: str) -> str:
    return "\n".join(
        (
            "set noaskquit",
            "ac dec 50 1 1e8",
            "echo __unity_ac__",
            f"print frequency vm({output_node})",
            "echo __end_unity_ac__",
            f"meas ac dc_gain_db find vdb({output_node}) at=1",
            "tran 1u 5m",
        )
    )


def _pick_measurements(spec: dict[str, float] | None, output_node: str) -> str:
    return f"meas tran output_swing_v pp v({output_node}) from=1m to=5m"


def _pick_prints(output_node: str) -> str:
    return f"v({output_node})"


def _build_deck(user_netlist: str, spec: dict[str, float] | None) -> str:
    nl = prepare_injected_netlist(user_netlist)
    output_node = _detect_node(nl, _OUTPUT_NODE_CANDIDATES, "vout")
    input_node = _detect_node(nl, _INPUT_NODE_CANDIDATES, "vin")
    nl = _ensure_ac_on_input(nl, input_node)
    return (
        _TESTBENCH_HEADER.format(circuit_id=CIRCUIT_ID)
        + nl
        + "\n"
        + _CONTROL_BLOCK.format(
            analyses=_pick_analyses(spec, output_node),
            measurements=_pick_measurements(spec, output_node),
            prints=_pick_prints(output_node),
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
    """Simulate an op-amp integrator netlist and extract Tier-0 FoMs."""
    try:
        if not has_prepared_device_line(netlist):
            return _none_result()
        stdout = _run_ngspice(_build_deck(netlist, spec), timeout_s)
        if stdout == "__FAILED__":
            return _none_result()
        result = {fom: _parse_meas(stdout, fom) for fom in FOM_NAMES}
        result["unity_gain_freq_hz"] = interpolate_crossing_hz(
            parse_ac_points(stdout, marker="__unity_ac__"),
            1.0,
            direction="fall",
        )
        return result
    except Exception:
        return _none_result()
