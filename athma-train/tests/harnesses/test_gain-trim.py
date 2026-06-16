from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "athma_train"
    / "harnesses"
    / "gain-trim.py"
)

spec = importlib.util.spec_from_file_location("gain_trim_harness", MODULE_PATH)
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
.title gain-trim long_s1 01 inverting summing servo
* topology: dual rail inverting summer with a weak trim injection resistor
* public pins: Vin Vtrim Vout
Vin Vin 0 DC 0 AC 1 SIN(0 2m 1k)
Vtrim Vtrim 0 DC 0 AC 0
VCC vcc 0 DC 5
VEE vee 0 DC -5
Rin Vin nsum 2k
Rf Vout nsum 100k
Rg nsum 0 2k
RtrimA Vtrim ntrim 240k
RtrimB ntrim nsum 20k
RtrimLeak ntrim 0 2Meg
Rbias nsum 0 10Meg
Cfb Vout nsum 2p
Ctrim ntrim 0 4.7n
Bdrv ndrv 0 V = {-50*V(Vin)+5*V(Vtrim)}
Rout ndrv Vout 35
Rload Vout 0 20k
Cload Vout 0 15p
Riso Vout nload 10
Csnub nload 0 100p
Rguard nload 0 100Meg
.control
op
ac dec 30 1 10Meg
tran 1u 5m
noise V(Vout) Vin dec 20 1 1Meg
dc Vtrim -0.10 0.10 0.005
.endc
.end
"""

MARGINAL_NETLIST = """
.title gain trim 01 inverting sum
Vin Vin 0 DC 0 AC 1
Vtrim Vtrim 0 DC 0.01
Rin Vin nsum 1k
Rf Vout nsum 49k
Rtrim Vtrim nsum 220k
Rbias nsum 0 10Meg
Eamp Vout 0 0 nsum 1e6
Rload Vout 0 100k
Cload Vout 0 8p
.control
op
ac dec 20 10 1Meg
.endc
.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "gain-trim"
    assert TIER == 1
    assert FOM_NAMES == ("gain_db", "offset_v", "trim_range_v", "bandwidth_hz")
    assert set(SPEC_UNITS) == set(FOM_NAMES)


def test_good_netlist_extracts_all_foms():
    result = measure(GOOD_NETLIST)
    assert set(result.keys()) == set(FOM_NAMES)
    assert all(result[fom] is not None for fom in FOM_NAMES)
    assert result["gain_db"] > 30
    assert result["trim_range_v"] > 0.5
    assert result["bandwidth_hz"] > 1e6


def test_marginal_netlist_extracts_partial_foms():
    result = measure(MARGINAL_NETLIST)
    assert set(result.keys()) == set(FOM_NAMES)
    assert result["gain_db"] is not None
    assert result["offset_v"] is not None
    assert result["trim_range_v"] is not None


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
