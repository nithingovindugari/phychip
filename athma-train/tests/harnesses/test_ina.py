from __future__ import annotations

from athma_train.harnesses.ina import CIRCUIT_ID, FOM_NAMES, SPEC_UNITS, TIER, measure


GOOD_NETLIST = """
* 3-op-amp Instrumentation Amplifier, differential gain = 100
* Inputs: Vp, Vn   Output: Vout

.param R1val=49.5k
.param Rgval=1k
.param R2val=10k
.param R3val=10k

Vdd  Vdd  0  DC  5.0
Vss  Vss  0  DC -5.0

Vp   Vp   0  DC 0 SIN(0 1m 1k)
Vn   Vn   0  DC 0

R1a  A    NA  {R1val}
Rg   NA   NB  {Rgval}
R1b  B    NB  {R1val}

EU1  A   0   Vp  NA   1e6
EU2  B   0   Vn  NB   1e6

R2b  B    NP   {R2val}
R3b  NP   0    {R3val}
R2a  A    NM   {R2val}
R3a  NM   Vout {R3val}

EU3  Vout 0    NP  NM   1e6

.control
tran 5u 3m
print v(Vp) v(Vn) v(Vout)
.endc

.end
"""

MARGINAL_NETLIST = """
* Instrumentation amplifier, G = 50, single 5V supply
* Ports: Vp (non-inverting in), Vn (inverting in), Vout

Vdd  vdd 0 DC 5.0
Vref vref 0 DC 2.5

Vp   Vp  vref DC 0
Vn   Vn  vref DC 0

XU1 Vp  n1  na  opamp
XU2 Vn  n2  nb  opamp

R1a na n1 24.5k
Rg  n1 n2 1k
R1b nb n2 24.5k

XU3 np nm Vout opamp

R2 nb nm   10k
R3 nm Vout 10k
R4 na np   10k
R5 np vref 10k

.subckt opamp inp inn out
Eop out 0 inp inn 1e6
.ends opamp

.control
dc Vp -40m 40m 1m
print v(Vout)
op
print v(na) v(nb) v(Vout)
.endc

.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "ina"
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
