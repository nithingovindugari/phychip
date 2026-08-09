from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "athma_train"
    / "harnesses"
    / "cs-stage.py"
)

spec = importlib.util.spec_from_file_location("cs_stage_harness", MODULE_PATH)
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
* Common-Source NMOS Amplifier - BW = 10 MHz
* Av ~= -8.7, Vout_DC ~= 2.5 V, ID ~= 167 uA

.model NMOS_MODEL nmos level=1 kp=100u vto=0.5
.model PMOS_MODEL pmos level=1 kp=50u vto=-0.5

Vdd vdd 0 DC 5.0
Vbias gate_dc 0 DC 1.08
Rbias gate gate_dc 1Meg

Vin Vin 0 AC 1 SIN(0 100m 1Meg)
Cin Vin gate 1u

M1 Vout gate 0 0 NMOS_MODEL W=10u L=1u

RD  vdd Vout 15k
CL  Vout 0   1p

.control
  op
  print v(Vout) v(gate) i(Vdd)
  ac dec 100 1k 1G
  plot vdb(Vout)
  tran 5n 300n
  plot v(Vin) v(Vout)
.endc

.end
"""

MARGINAL_NETLIST = """
* Common-Source Amplifier Av=-5
* Simple gate-divider bias without explicit output capacitance.

Vdd Vdd 0 DC 5.0
Vin Vin 0 DC 0.9 AC 1

Rg1 Vdd Vg 100k
Rg2 Vg 0 100k
Rd Vdd Vout 2k
Rs Vs 0 400
Cs Vs 0 100u
Cin Vin Vg 10u

M1 Vout Vg Vs 0 NMOS_MODEL W=10u L=1u

.model NMOS_MODEL nmos level=1 kp=100u vto=0.5

.control
op
print v(Vout) v(Vg) v(Vs)
ac dec 100 1 1G
plot vdb(Vout)
.endc

.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "cs-stage"
    assert TIER in (0, 1, 2)
    assert len(FOM_NAMES) >= 3
    for fom in FOM_NAMES:
        assert fom in SPEC_UNITS


def test_good_netlist_extracts_all_foms():
    result = measure(GOOD_NETLIST)
    assert set(result.keys()) == set(FOM_NAMES)
    non_none = [v for v in result.values() if v is not None]
    assert len(non_none) >= 2


def test_marginal_netlist_returns_all_fom_keys():
    result = measure(MARGINAL_NETLIST)
    assert set(result.keys()) == set(FOM_NAMES)


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
