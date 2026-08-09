from __future__ import annotations

from athma_train.harnesses.boost import (
    CIRCUIT_ID,
    FOM_NAMES,
    SPEC_UNITS,
    TIER,
    measure,
)


GOOD_NETLIST = """
* Boost converter: Vin (5V) -> Vout (~12V), step-up, fsw=100kHz, D~0.583
* Nodes: Vin = input rail, Vout = regulated output, sw = switch node, gate = PWM drive

Vdd  Vin 0   DC 5.0

* Power stage
L1   Vin sw   220u
RL   sw  swx  50m
D1   swx Vout DMOD
Cout Vout 0  47u  IC=5
Rload Vout 0 50

* Switch: large NMOS to keep Rds(on) low
M1   swx gate 0 0 NMOS_MODEL W=10m L=1u

* PWM gate drive: 100 kHz, D = 0.583, 0 -> 5 V, fast edges
Vpwm gate 0 PULSE(0 5 0 20n 20n 5.83u 10u)

* Device models
.model NMOS_MODEL nmos level=1 kp=100u vto=0.5
.model PMOS_MODEL pmos level=1 kp=50u  vto=-0.5
.model DMOD       d    is=1e-7 n=1 rs=0.02 cjo=50p

.ic V(Vout)=5

.control
tran 200n 4m uic
print v(Vout)
.endc

.end
"""

MARGINAL_NETLIST = """
* Boost Converter: Vin to Vout
* Continuous conduction mode boost, D=0.5, Fsw=100kHz

* --- Power Stage ---
Vin Vin 0 DC 5.0
L1  Vin sw 10u
M1  sw  gate 0 0 NMOS_MODEL W=1000u L=1u
D1  sw  Vout DBOOST
C1  Vout 0 100u IC=10
Rload Vout 0 10

* --- Gate Drive (PWM ~100kHz, 50% duty) ---
Vgate gate 0 PULSE(0 5 0 10n 10n 5u 10u)

* --- Models ---
.model NMOS_MODEL nmos level=1 kp=100u vto=0.5
.model PMOS_MODEL pmos level=1 kp=50u vto=-0.5
.model DBOOST d IS=1e-14 RS=0.01 N=1

* --- Analysis ---
.control
tran 10n 500u
plot v(Vout) v(Vin)
.endc

.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "boost"
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
    first = measure(GOOD_NETLIST)
    second = measure(GOOD_NETLIST)
    for key in FOM_NAMES:
        if first[key] is None and second[key] is None:
            continue
        assert abs((first[key] or 0) - (second[key] or 0)) < 1e-9, key
