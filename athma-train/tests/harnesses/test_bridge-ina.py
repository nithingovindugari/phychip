from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "athma_train"
    / "harnesses"
    / "bridge-ina.py"
)

spec = importlib.util.spec_from_file_location("bridge_ina_harness", MODULE_PATH)
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
.title bridge instrumentation amplifier b01
Vex Vex 0 DC 5 AC 1
Rb01a Vex bpb01 350
Rb01b bpb01 0 350
Rb01c Vex bnb01 350
Rb01d bnb01 0 350.8
Rb01fp bpb01 ipb01 200
Rb01fn bnb01 inb01 200
Cb01fp ipb01 0 22n
Cb01fn inb01 0 22n
Einab01 Vout 0 ipb01 inb01 100
Rloadb01 Vout 0 100k
.control
op
ac dec 10 1 10k
print v(Vout)
.endc
.end
"""

MARGINAL_NETLIST = """
.title bridge instrumentation amplifier 1
VEXC Vex 0 DC 5.0
RBL1 Vex bp 3012
RBL2 bp 0 2988
RBR1 Vex bn 3000
RBR2 bn 0 3000
ECORE amp 0 bp bn 110
ROUT amp Vout 120
COUT Vout 0 1e-06
RLOAD Vout 0 20000
.op
.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "bridge-ina"
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
