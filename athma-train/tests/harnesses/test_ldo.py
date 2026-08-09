from __future__ import annotations

from athma_train.harnesses.ldo import CIRCUIT_ID, FOM_NAMES, SPEC_UNITS, TIER, measure


GOOD_NETLIST = """
* LDO regulator: PMOS pass + NMOS-diffpair error amp
* Vin = 5V, Vout = 3.3V, Vfb internal divider node
.model NMOS_MODEL nmos level=1 kp=100u vto=0.5
.model PMOS_MODEL pmos level=1 kp=50u vto=-0.5

Vin   Vin   0 DC 5.0
Vref  vref  0 DC 1.0

Mp    Vout  egout Vin Vin PMOS_MODEL W=1600u L=1u

M1    n1    Vfb   tail 0   NMOS_MODEL W=10u L=2u
M2    egout vref  tail 0   NMOS_MODEL W=10u L=2u
M3    n1    n1    Vin  Vin PMOS_MODEL W=10u L=1u
M4    egout n1    Vin  Vin PMOS_MODEL W=10u L=1u
Itail tail  0     DC 20u

R1    Vout  Vfb   23k
R2    Vfb   0     10k

RL    Vout  0     330
Cout  Vout  0     1u
Cc    egout Vout  5p

.control
tran 1u 2m
.endc
.end
"""

MARGINAL_NETLIST = """
* 3.3V LDO with lighter pass device and resistive load
Vdd Vdd 0 DC 5.0
Vin  Vin  0 DC 3.5
Vref Vref 0 DC 1.25
Vbias Vbias 0 DC 0.9

Mtail Ntail Vbias 0 0 NMOS_MODEL W=5u L=1u
M1 Nd1 Vfb Ntail 0 NMOS_MODEL W=40u L=1u
M2 Vea Vref Ntail 0 NMOS_MODEL W=40u L=1u
Mp1 Nd1 Nd1 Vin Vin PMOS_MODEL W=40u L=1u
Mp2 Vea Nd1 Vin Vin PMOS_MODEL W=40u L=1u
Mpass Vout Vea Vin Vin PMOS_MODEL W=2000u L=0.5u

R1 Vout Vfb 164k
R2 Vfb  0   100k
RL Vout 0   330
CL Vout 0   1u
Cc Vea Vout 10p

.model NMOS_MODEL nmos level=1 kp=100u vto=0.5
.model PMOS_MODEL pmos level=1 kp=50u  vto=-0.5

.control
tran 10u 3m
.endc
.end
"""

DROPOUT_NETLIST = """
* Quick LDO: target Vout=3.3V, dropout-oriented pass device
.model NMOS_MODEL nmos level=1 kp=100u vto=0.5
.model PMOS_MODEL pmos level=1 kp=50u vto=-0.5

Vin   Vin 0 PWL(0 5.0  1m 5.0  1.001m 3.45)
Vref  Vref 0 DC 1.0

M1    Vout Vea Vin Vin PMOS_MODEL W=5000u L=1u
Rbias Vin   Vbn 270k
M6b   Vbn   Vbn 0 0  NMOS_MODEL W=5u L=1u
M6    Vtail Vbn 0 0  NMOS_MODEL W=5u L=1u
M2    Vd2 Vfb  Vtail 0 NMOS_MODEL W=10u L=1u
M3    Vea Vref Vtail 0 NMOS_MODEL W=10u L=1u
M4    Vd2 Vd2 Vin Vin PMOS_MODEL W=4u L=1u
M5    Vea Vd2 Vin Vin PMOS_MODEL W=4u L=1u

R1    Vout Vfb 230k
R2    Vfb  0   100k
Cc    Vea Vout 10p
RL    Vout 0 66
CL    Vout 0 1u

.control
tran 2u 3m
.endc
.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "ldo"
    assert TIER == 1
    assert len(FOM_NAMES) == 5
    for fom in FOM_NAMES:
        assert fom in SPEC_UNITS


def test_good_netlist_extracts_all_foms():
    result = measure(GOOD_NETLIST)
    assert set(result.keys()) == set(FOM_NAMES)
    non_none = [value for value in result.values() if value is not None]
    assert len(non_none) >= 2


def test_reference_variants_extract_some_foms():
    marginal = measure(MARGINAL_NETLIST)
    dropout = measure(DROPOUT_NETLIST)
    assert set(marginal.keys()) == set(FOM_NAMES)
    assert set(dropout.keys()) == set(FOM_NAMES)
    assert any(value is not None for value in marginal.values())
    assert any(value is not None for value in dropout.values())


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
