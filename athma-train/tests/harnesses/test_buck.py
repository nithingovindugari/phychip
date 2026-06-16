from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "athma_train" / "harnesses" / "buck.py"
)

spec = importlib.util.spec_from_file_location("buck_harness", MODULE_PATH)
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
* Open-loop asynchronous buck converter: Vin (12V) -> Vout (~5V)
Vin     Vin   0      DC 12
Vgate   gate  0      PULSE(0 5 0 10n 10n 4.3u 10u)

S1      Vin   sw     gate 0   SW_IDEAL
D1      0     sw     DSCH

L1      sw    Vout   100u
Cout    Vout  0      10u    IC=0
Rload   Vout  0      5

.model SW_IDEAL SW(Ron=10m Roff=1Meg Vt=2.5 Vh=0.1)
.model DSCH    D(Is=1e-7 N=1 Rs=10m Cjo=100p)
.model NMOS_MODEL nmos level=1 kp=100u vto=0.5
.model PMOS_MODEL pmos level=1 kp=50u  vto=-0.5

.control
tran 100n 4m uic
print v(Vout)
.endc

.end
"""

MARGINAL_NETLIST = """
* Synchronous Buck Converter: 12V -> 3.3V @ 2A, fsw = 500 kHz
Vin   Vin    0    DC 12.0
Cin   Vin    0    10u

SHS   Vin    SW   PWM_H 0   SW_MODEL
SLS   SW     0    PWM_L 0   SW_MODEL

L1    SW     Vout 8u
Cout  Vout   0    22u
Rload Vout   0    1.65

Vpwmh PWM_H 0   PULSE(0 5 0n     5n 5n 540n 2000n)
Vpwml PWM_L 0   PULSE(5 0 0n     5n 5n 600n 2000n)

.model SW_MODEL SW(Ron=10m Roff=1Meg Vt=2.5 Vh=0.2)
.model NMOS_MODEL nmos level=1 kp=100u vto=0.5
.model PMOS_MODEL pmos level=1 kp=50u vto=-0.5

.control
tran 100n 500u
print v(Vin) v(Vout) v(SW)
.endc

.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "buck"
    assert TIER == 1
    assert FOM_NAMES == (
        "vout_steady_v",
        "ripple_pp_v",
        "efficiency_pct",
        "settling_time_s",
    )
    for fom in FOM_NAMES:
        assert fom in SPEC_UNITS


def test_good_netlist_extracts_all_foms():
    result = measure(GOOD_NETLIST)
    assert set(result.keys()) == set(FOM_NAMES)
    non_none = [value for value in result.values() if value is not None]
    assert len(non_none) >= 3
    assert result["vout_steady_v"] is not None
    assert result["vout_steady_v"] > 1.0
    assert result["efficiency_pct"] is not None
    assert result["efficiency_pct"] > 0


def test_marginal_netlist_extracts_partial_foms():
    result = measure(MARGINAL_NETLIST)
    assert set(result.keys()) == set(FOM_NAMES)
    non_none = [value for value in result.values() if value is not None]
    assert len(non_none) >= 2


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
