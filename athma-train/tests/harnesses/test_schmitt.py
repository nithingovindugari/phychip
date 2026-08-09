from __future__ import annotations

from athma_train.harnesses.schmitt import (
    CIRCUIT_ID,
    FOM_NAMES,
    SPEC_UNITS,
    TIER,
    measure,
)


GOOD_NETLIST = """
* Non-inverting op-amp Schmitt trigger, 100 mV hysteresis
* Vin -> R1 -> Vplus; Vout -> R2 -> Vplus; Vminus tied to ground

Vdd vdd 0 DC 5.0
Vss vss 0 DC -5.0

Vin Vin 0 PWL(0 -0.2  1m 0.2  2m -0.2  3m 0.2  4m -0.2)

R1 Vin   np   1k
R2 Vout  np   100k

Eopamp nout 0 VALUE = { 5 * tanh( 1e6 * (V(np) - V(nn)) ) }
Rnn nn 0 1Meg
Vnn nn 0 DC 0
Eout Vout 0 nout 0 1.0

.control
tran 1u 4m
plot V(Vin) V(Vout)
.endc
.end
"""

MARGINAL_NETLIST = """
* Wider-hysteresis Schmitt trigger, useful as a marginal reference

Vdd vdd 0 DC 5.0
Vss vss 0 DC -5.0

Vin Vin 0 PWL(0 -0.5  1m 0.5  2m -0.5  3m 0.5  4m -0.5)

R1 Vin   np   5k
R2 Vout  np   100k

Eopamp nout 0 VALUE = { 5 * tanh( 1e6 * (V(np) - V(nn)) ) }
Rnn nn 0 1Meg
Vnn nn 0 DC 0
Eout Vout 0 nout 0 1.0

.control
tran 1u 4m
plot V(Vin) V(Vout)
.endc
.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "schmitt"
    assert TIER in (0, 1, 2)
    assert len(FOM_NAMES) >= 3
    for fom in FOM_NAMES:
        assert fom in SPEC_UNITS


def test_good_netlist_extracts_all_foms():
    result = measure(GOOD_NETLIST)
    assert set(result.keys()) == set(FOM_NAMES)
    non_none = [value for value in result.values() if value is not None]
    assert len(non_none) >= 2


def test_marginal_netlist_extracts_thresholds():
    result = measure(MARGINAL_NETLIST)
    assert set(result.keys()) == set(FOM_NAMES)
    assert result["vth_high_v"] is not None
    assert result["vth_low_v"] is not None
    assert result["hysteresis_v"] is not None


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
