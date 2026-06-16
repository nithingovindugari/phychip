from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "athma_train"
    / "harnesses"
    / "level-shift.py"
)

spec = importlib.util.spec_from_file_location("level_shift_harness", MODULE_PATH)
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
* Level shifter: Vin_lv (1.8V logic) -> Vout_hv (3.3V logic)
* Cross-coupled PMOS latch with NMOS pull-downs

.model NMOS_MODEL nmos level=1 kp=100u vto=0.5
.model PMOS_MODEL pmos level=1 kp=50u vto=-0.5

VDDL vddl 0 DC 1.8
VDDH vddh 0 DC 3.3
Vin_lv Vin_lv 0 PULSE(0 1.8 5n 1n 1n 20n 40n)

MP3 Vin_lv_b Vin_lv vddl vddl PMOS_MODEL W=4u L=1u
MN3 Vin_lv_b Vin_lv 0    0    NMOS_MODEL W=2u L=1u

MP1 nA Vout_hv vddh vddh PMOS_MODEL W=4u L=1u
MP2 Vout_hv nA   vddh vddh PMOS_MODEL W=4u L=1u

MN1 nA      Vin_lv   0 0 NMOS_MODEL W=10u L=1u
MN2 Vout_hv Vin_lv_b 0 0 NMOS_MODEL W=10u L=1u

CL Vout_hv 0 20f

.control
tran 0.1n 120n
print v(Vin_lv) v(Vout_hv)
.endc

.end
"""

MARGINAL_NETLIST = """
* Weak-pulldown level shifter: may switch slowly but remains measurable.
.model NMOS_MODEL nmos level=1 kp=100u vto=0.5
.model PMOS_MODEL pmos level=1 kp=50u vto=-0.5

Vdd_lv Vdd_lv 0 DC 1.8
Vdd_hv Vdd_hv 0 DC 3.3
Vin Vin_lv 0 PULSE(0 1.8 10n 1n 1n 50n 100n)

M_inv_n Vin_lv_b Vin_lv 0 0 NMOS_MODEL W=2u L=0.18u
M_inv_p Vin_lv_b Vin_lv Vdd_lv Vdd_lv PMOS_MODEL W=4u L=0.18u

M_p1 Vout_hv_b Vout_hv Vdd_hv Vdd_hv PMOS_MODEL W=4u L=0.18u
M_p2 Vout_hv Vout_hv_b Vdd_hv Vdd_hv PMOS_MODEL W=4u L=0.18u

M_n1 Vout_hv_b Vin_lv 0 0 NMOS_MODEL W=2u L=0.18u
M_n2 Vout_hv Vin_lv_b 0 0 NMOS_MODEL W=2u L=0.18u

.control
tran 0.1n 300n
print v(Vin_lv) v(Vout_hv)
.endc

.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "level-shift"
    assert TIER in (0, 1, 2)
    assert len(FOM_NAMES) >= 3
    for fom in FOM_NAMES:
        assert fom in SPEC_UNITS


def test_good_netlist_extracts_all_foms():
    for netlist in (GOOD_NETLIST, MARGINAL_NETLIST):
        result = measure(netlist)
        assert set(result.keys()) == set(FOM_NAMES)
        non_none = [v for v in result.values() if v is not None]
        assert len(non_none) >= 2


def test_broken_netlist_returns_all_none():
    result = measure(BROKEN_NETLIST)
    assert set(result.keys()) == set(FOM_NAMES)
    assert all(v is None for v in result.values())


def test_idempotent():
    a = measure(GOOD_NETLIST)
    b = measure(GOOD_NETLIST)
    for key in FOM_NAMES:
        if a[key] is None and b[key] is None:
            continue
        assert abs((a[key] or 0) - (b[key] or 0)) < 1e-9, key
