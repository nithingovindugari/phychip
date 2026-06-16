from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "athma_train"
    / "harnesses"
    / "inv-amp.py"
)

spec = importlib.util.spec_from_file_location("inv_amp_harness", MODULE_PATH)
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
* Inverting Amplifier: Gain = -50, BW >= 1 MHz
* Macromodel GBW = 100 MHz, A0 = 120 dB

.subckt opamp inp inn out vdd vss
Gm    0   nd  inp inn  10m
Rp    nd  0   100Meg
Cp    nd  0   16p
Eout  buf 0   nd  0    1
Rout  buf out 10
.ends opamp

Vdd  vdd 0  DC  5.0
Vss  vss 0  DC -5.0
Vin  Vin 0  AC 1

Rin  Vin  inv  1k
Rf   inv  Vout 50k

X1   0    inv  Vout  vdd vss  opamp

.control
  ac dec 100 1k 200Meg
  plot vdb(Vout)
.endc

.end
"""

MARGINAL_NETLIST = """
* -20 V/V Inverting Amplifier
* Gain = -Rf/Rin = -20

Vdd VDD 0 DC 5.0
Vee VEE 0 DC -5.0
Vin Vin 0 DC 0 AC 1

Rin Vin inv_in 1k
Rf  Vout inv_in 20k

Eout Vout 0 VALUE={LIMIT(-20*V(Vin), -4.5, 4.5)}

.control
  ac dec 100 1 10Meg
  plot vdb(Vout)
  op
.endc

.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "inv-amp"
    assert TIER in (0, 1, 2)
    assert len(FOM_NAMES) >= 3
    for fom in FOM_NAMES:
        assert fom in SPEC_UNITS


def test_good_netlist_extracts_all_foms():
    result = measure(GOOD_NETLIST)
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
