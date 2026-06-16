from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from athma_train.harnesses._measure_utils import parse_numeric_rows
from athma_train.spice_gate import grade_netlist, normalize_netlist


CIRCUIT_ID: str = "ldo"
TIER: int = 1
FOM_NAMES: tuple[str, ...] = (
    "vout_dc_v",
    "dropout_v",
    "line_reg_v_per_v",
    "load_reg_ohm",
    "psrr_1khz_db",
)
SPEC_UNITS: dict[str, str] = {
    "vout_dc_v": "V",
    "dropout_v": "V",
    "line_reg_v_per_v": "V/V",
    "load_reg_ohm": "ohm",
    "psrr_1khz_db": "dB",
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

_BENCH_SUPPLY = "Vtb_supply Vin 0 DC 5 AC 0"
_BENCH_LOAD = "Ildo_bench_load Vout 0 DC 0"


def _none_result() -> dict[str, float | None]:
    return {fom: None for fom in FOM_NAMES}


def _target_vout(spec: dict[str, float] | None) -> float:
    if not spec:
        return 3.3
    value = spec.get("vout_dc_v", spec.get("vout_v", 3.3))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]) + float(value[1])) / 2.0
    return float(value)


def _load_current(spec: dict[str, float] | None) -> float:
    if not spec:
        return 0.05
    value = spec.get("load_current_a", spec.get("i_load_a", 0.01))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]) + float(value[1])) / 2.0
    return float(value)


def _pick_analyses(spec: dict[str, float] | None) -> str:
    return "set noaskquit"


def _pick_measurements(spec: dict[str, float] | None) -> str:
    target = _target_vout(spec)
    load_current = _load_current(spec)
    dropout_trip = 0.99 * target
    return "\n".join(
        (
            "dc Vtb_supply 5.0 5.01 0.01",
            "meas dc vout_dc_v find v(Vout) at=5.0",
            "dc Vtb_supply 4.0 5.0 0.05",
            "echo __line_reg__",
            "print v(Vout)",
            "dc Vtb_supply 1.0 5.0 0.01",
            f"meas dc vin_at_drop when v(Vout)={dropout_trip:.9g} rise=1",
            "alter @Vtb_supply[dc]=5",
            f"dc Ildo_bench_load 0 {load_current:.9g} {load_current / 50:.9g}",
            "meas dc vout_noload find v(Vout) at=0",
            f"meas dc vout_fullload find v(Vout) at={load_current:.9g}",
            f"let load_reg_ohm=(vout_noload-vout_fullload)/{load_current:.9g}",
            "echo load_reg_ohm = $&load_reg_ohm",
            "alter @Ildo_bench_load[dc]=0",
            "alter @Vtb_supply[dc]=5",
            "alter @Vtb_supply[acmag]=1",
            "ac dec 10 1 10k",
            "meas ac vout_1khz_db find vdb(Vout) at=1k",
            "let psrr_1khz_db=-vout_1khz_db",
            "echo psrr_1khz_db = $&psrr_1khz_db",
        )
    )


def _pick_prints() -> str:
    return "v(Vout) v(Vfb) vdb(Vout)"


def _build_deck(user_netlist: str, spec: dict[str, float] | None) -> str:
    nl = normalize_netlist(user_netlist)
    nl = re.sub(r"\.control\b.*?\.endc\b", "", nl, flags=re.DOTALL | re.IGNORECASE)
    nl = re.sub(r"\.end\s*$", "", nl, flags=re.IGNORECASE).rstrip()
    nl = re.sub(r"(?im)^\s*V\S+\s+Vin\s+0\s+.*$", "", nl).rstrip()
    return (
        _TESTBENCH_HEADER.format(circuit_id=CIRCUIT_ID)
        + nl
        + "\n"
        + _BENCH_SUPPLY
        + "\n"
        + _BENCH_LOAD
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


def _parse_line_reg(stdout: str) -> float | None:
    marker = stdout.find("__line_reg__")
    section = stdout[marker:] if marker >= 0 else stdout
    rows = parse_numeric_rows(section, columns=2)
    if len(rows) < 2:
        return None
    vin0, vout0 = rows[0]
    vin1, vout1 = rows[-1]
    if vin0 == vin1:
        return None
    return (vout1 - vout0) / (vin1 - vin0)


def measure(
    netlist: str,
    spec: dict[str, float] | None = None,
    timeout_s: int = 30,
) -> dict[str, float | None]:
    """Simulate an LDO netlist and extract Tier-1 FoMs."""
    try:
        passed, _log = grade_netlist(netlist, timeout_s=20)
        if not passed:
            return _none_result()
        stdout = _run_ngspice(_build_deck(netlist, spec), timeout_s)
        if stdout == "__FAILED__":
            return _none_result()
        result = {fom: _parse_meas(stdout, fom) for fom in FOM_NAMES}
        result["line_reg_v_per_v"] = _parse_line_reg(stdout)
        vin_at_drop = _parse_meas(stdout, "vin_at_drop")
        if vin_at_drop is not None:
            result["dropout_v"] = vin_at_drop - (0.99 * _target_vout(spec))
        return result
    except Exception:
        return _none_result()
