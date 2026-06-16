from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "athma_train"
    / "harnesses"
    / "bandgap.py"
)

spec = importlib.util.spec_from_file_location("bandgap_harness", MODULE_PATH)
assert spec is not None
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

CIRCUIT_ID = module.CIRCUIT_ID
FOM_NAMES = module.FOM_NAMES
SPEC_UNITS = module.SPEC_UNITS
TIER = module.TIER
measure = module.measure


GOOD_NETLIST = """
* 1.2V reference from 5V Vdd using diode-connected NMOS
Vdd Vdd 0 DC 5.0

R1 Vdd Vref 75k
M1 Vref Vref 0 0 NMOS_MODEL W=2u L=1u
C1 Vref 0 10p

.model NMOS_MODEL nmos level=1 kp=100u vto=0.5
.model PMOS_MODEL pmos level=1 kp=50u vto=-0.5

.control
op
print v(Vref) i(Vdd)
.endc

.end
"""

MARGINAL_NETLIST = """
* Vref = 1.2V Bandgap-style resistor divider from Vdd
Vdd vdd 0 DC 5.0

R1 vdd vref 380k
R2 vref 0 120k

M1 vref_out vref vdd vdd PMOS_MODEL W=10u L=1u
Rbias vref_out 0 1Meg

.model NMOS_MODEL nmos level=1 kp=100u vto=0.5
.model PMOS_MODEL pmos level=1 kp=50u vto=-0.5

.control
op
print v(vref) v(vref_out)
.endc

.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "bandgap"
    assert TIER in (0, 1, 2)
    assert len(FOM_NAMES) >= 3
    for fom in FOM_NAMES:
        assert fom in SPEC_UNITS


def test_good_netlist_extracts_all_foms():
    result = measure(GOOD_NETLIST)
    assert set(result.keys()) == set(FOM_NAMES)
    non_none = [value for value in result.values() if value is not None]
    assert len(non_none) >= 2


def test_broken_netlist_returns_all_none():
    result = measure(BROKEN_NETLIST)
    assert set(result.keys()) == set(FOM_NAMES)
    assert all(value is None for value in result.values())


def test_idempotent():
    a = measure(GOOD_NETLIST)
    b = measure(GOOD_NETLIST)
    for key in FOM_NAMES:
        if a[key] is None and b[key] is None:
            continue
        assert abs((a[key] or 0) - (b[key] or 0)) < 1e-9, key
