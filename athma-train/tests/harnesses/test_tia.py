from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "athma_train" / "harnesses" / "tia.py"
)

spec = importlib.util.spec_from_file_location("tia_harness", MODULE_PATH)
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
.title tia_s1_01_plain_shunt_feedback
Iphoto I_pd 0 DC 1u AC 1u
Vref vref 0 DC 0
Eol ndrv 0 vref I_pd 1e5
Rout ndrv Vout 37
Rf Vout I_pd 100k
Cf Vout I_pd 1.6p
Rbias I_pd vref 10G
Cl Vout 0 8p
Rguard I_pd guard 10Meg
Eguard guard 0 vref I_pd 0.98
.op
.ac dec 30 1 20meg
.print ac vm(Vout) vp(Vout)
.end
"""

MARGINAL_NETLIST = """
.title tia_s3_03_dual_rail_fast_photodiode
Iphoto I_pd 0 DC 1.8u AC 1.8u
Vref vref 0 DC 0
Eol ndrv 0 vref I_pd 3.9e4
Rout ndrv Vout 22
Rf Vout I_pd 39k
Cf Vout I_pd 0.6p
Rbias I_pd vref 15G
Cl Vout 0 5p
Vpos vpwr 0 DC 2.5
Vneg vnwr 0 DC -2.5
Cpd I_pd vref 2.2p
.op
.ac dec 40 100 100meg
.print ac vm(Vout) vp(Vout)
.end
"""

LOW_BW_NETLIST = """
.title tia_s1_03_input_clamp_protection
Iphoto I_pd 0 DC 2u AC 2u
Vref vref 0 DC 0
Eol ndrv 0 vref I_pd 1e5
Rout ndrv Vout 51
Rf Vout I_pd 47k
Cf Vout I_pd 3.3p
Rbias I_pd vref 10G
Cl Vout 0 20p
D1 I_pd vref DLIM
D2 vref I_pd DLIM
.model DLIM D(Is=1e-14 Rs=5 Cjo=0.1p)
.op
.ac dec 30 1 20meg
.print ac vm(Vout) vp(Vout)
.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "tia"
    assert TIER == 2
    assert FOM_NAMES == (
        "transimpedance_v_per_a",
        "bandwidth_hz",
        "input_noise_a_per_rthz",
        "stability_phase_margin_deg",
    )
    for fom in FOM_NAMES:
        assert fom in SPEC_UNITS


def test_good_netlist_extracts_all_foms():
    result = measure(GOOD_NETLIST)
    assert set(result.keys()) == set(FOM_NAMES)
    non_none = [value for value in result.values() if value is not None]
    assert len(non_none) >= 2
    assert result["transimpedance_v_per_a"] is not None
    assert result["transimpedance_v_per_a"] > 1e4


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


def test_marginal_and_low_bw_netlists_extract_transimpedance():
    marginal = measure(MARGINAL_NETLIST)
    low_bw = measure(LOW_BW_NETLIST)
    assert marginal["transimpedance_v_per_a"] is not None
    assert low_bw["transimpedance_v_per_a"] is not None
