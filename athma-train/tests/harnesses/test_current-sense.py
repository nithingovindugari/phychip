from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "athma_train"
    / "harnesses"
    / "current-sense.py"
)

spec = importlib.util.spec_from_file_location("current_sense_harness", MODULE_PATH)
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
* High-CMRR instrumentation current-sense amplifier
Vdd Vdd 0 DC 5.0
Vss Vss 0 DC -5.0

Vcm   Ncm 0 DC 2.5 AC 0
Vdiff Nd  0 DC 0   AC 1
Ep Vsense_p Ncm Nd 0  0.5
En Vsense_n Ncm Nd 0 -0.5

XU1 Vsense_p N1 N2 OPAMP
XU2 Vsense_n N4 N3 OPAMP
R1a N2 N1 10k
Rg  N1 N4 2.222k
R1b N4 N3 10k

R3 N2 Nm   10k
R4 Nm Vout 10k
R5 N3 Np   10k
R6 Np 0    10k
XU3 Np Nm Vout OPAMP

.subckt OPAMP vp vn vout
E1 vint 0 vp vn 1e6
R1 vint vout 100
C1 vout 0 1.59u
.ends

.control
ac dec 20 1 1Meg
print v(Vout)
.endc
.end
"""

MARGINAL_NETLIST = """
* Current-sense amplifier with lower bandwidth and gain
Vdd vdd 0 DC 5.0
Vref vref 0 DC 2.5

Vcm  vcm   0      DC 12.0
Vdif Vsense_p vcm DC 0       AC 1
Vneg Vsense_n vcm DC 0

R1 Vsense_p n_inp  1k
R2 n_inp    Vout   20k
R3 Vsense_n n_inn  1k
R4 n_inn    vref   20k

Eop  n_oa  0    n_inp n_inn 1e5
Rp   n_oa  n_p  1k
Cp   n_p   0    159n
Eout Vout  0    n_p   0       1.0
Rout Vout  0    1Meg

.control
ac dec 20 1 100Meg
print v(Vout)
.endc
.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "current-sense"
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
    a = measure(GOOD_NETLIST)
    b = measure(GOOD_NETLIST)
    for key in FOM_NAMES:
        if a[key] is None and b[key] is None:
            continue
        assert abs((a[key] or 0) - (b[key] or 0)) < 1e-9, key
