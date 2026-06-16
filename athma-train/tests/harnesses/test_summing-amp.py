from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "athma_train"
    / "harnesses"
    / "summing-amp.py"
)

spec = importlib.util.spec_from_file_location("summing_amp_harness", MODULE_PATH)
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
* 3-Input Unity-Gain Inverting Summing Amplifier
* Vout = -(V1 + V2 + V3)

Vdd vdd 0 DC  5.0
Vss vss 0 DC -5.0

V1 v1 0 DC 1.0
V2 v2 0 DC 1.5
V3 v3 0 DC 0.5

R1  v1  inv  10k
R2  v2  inv  10k
R3  v3  inv  10k
Rf  inv vout 10k

Eamp vout 0 0 inv 1e5

.control
op
print v(v1) v(v2) v(v3) v(inv) v(vout)
.endc

.end
"""

MARGINAL_NETLIST = """
* Weighted Summing Amplifier + Output Buffer
* Vout = -(1.0*V1 + 2.0*V2 + 0.5*V3)

Vdd vdd 0 DC  5.0
Vss vss 0 DC -5.0

V1 v1 0 SIN(0 1.0 1000)
V2 v2 0 SIN(0 1.0 500)
V3 v3 0 SIN(0 0.5 250)

R1  v1  vsum 10k
R2  v2  vsum  5k
R3  v3  vsum 20k
Rf  vsum vout_sum 10k
Rbal inp1 0 2.4k

Xoa1 inp1 vsum vout_sum vdd vss opamp
Xoa2 vout_sum vout vout vdd vss opamp

.subckt opamp inp inn out vdd vss
Rin   inp inn 10Meg
Egain outx 0  inp inn 1e6
Rout  outx out 50
.ends opamp

.tran 1u 5m

.control
run
print v(vout)
.endc

.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "summing-amp"
    assert TIER in (0, 1, 2)
    assert len(FOM_NAMES) >= 3
    for fom in FOM_NAMES:
        assert fom in SPEC_UNITS


def test_good_netlist_extracts_per_input_gains():
    result = measure(GOOD_NETLIST)

    assert set(result.keys()) == set(FOM_NAMES)
    assert result["gain_v1_db"] is not None
    assert result["gain_v2_db"] is not None
    assert result["gain_v3_db"] is not None
    assert abs(result["gain_v1_db"] or 0) < 0.01
    assert abs(result["gain_v2_db"] or 0) < 0.01
    assert abs(result["gain_v3_db"] or 0) < 0.01


def test_marginal_netlist_extracts_weighted_gains():
    result = measure(MARGINAL_NETLIST)

    assert set(result.keys()) == set(FOM_NAMES)
    assert result["gain_v1_db"] is not None
    assert result["gain_v2_db"] is not None
    assert result["gain_v3_db"] is not None
    assert abs((result["gain_v1_db"] or 0) - 0.0) < 0.05
    assert abs((result["gain_v2_db"] or 0) - 6.02) < 0.05
    assert abs((result["gain_v3_db"] or 0) - -6.02) < 0.05


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
