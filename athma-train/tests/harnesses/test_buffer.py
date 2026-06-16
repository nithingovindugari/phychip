from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "athma_train" / "harnesses" / "buffer.py"
)

spec = importlib.util.spec_from_file_location("buffer_harness", MODULE_PATH)
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
* Unity-Gain Buffer
* Vin to non-inverting input, output fed back to inverting input

.subckt OPAMP inp inn out vdd vss
Rin    inp inn 1Meg
Egain  int_out 0 inp inn 1e5
Rout   int_out out 75
.ends OPAMP

Vdd vdd 0 DC  5.0
Vss vss 0 DC -5.0
Vin Vin 0 SIN(0 1 1k)

Xbuf Vin Vout Vout vdd vss OPAMP
Rload Vout 0 10k

.control
tran 1u 3m
plot v(Vin) v(Vout)
op
print v(Vin) v(Vout)
.endc

.end
"""

MARGINAL_NETLIST = """
* Marginal voltage follower with lower loop gain

Vdd vdd 0 DC 5.0
Vss vss 0 DC 0.0
Vin vin 0 SIN(2.5 1.0 1k)

Rin   vin vout 1Meg
Eout  vout_int 0 vin vout 100k
Rout  vout_int vout 100

.control
op
tran 1u 3m
print v(vin) v(vout)
.endc

.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "buffer"
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
