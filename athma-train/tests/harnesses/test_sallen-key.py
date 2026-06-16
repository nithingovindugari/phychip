from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "athma_train"
    / "harnesses"
    / "sallen-key.py"
)

spec = importlib.util.spec_from_file_location("sallen_key_harness", MODULE_PATH)
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
* Sallen-Key Butterworth 2nd-order LPF, fc = 1 kHz, Q = 0.707
* Equal-R, C1 = 2*C2 design

Vin Vin 0 AC 1

R1 Vin N1   11.3k
R2 N1  Np   11.3k
C1 N1  Vout 20n
C2 Np  0    10n

* Ideal unity-gain op-amp (VCVS follower)
Eopa Vout 0 Np Vout 1e6

.control
ac dec 20 10 100k
.endc

.end
"""

MARGINAL_NETLIST = """
* Sallen-Key low-pass, overdamped Q near 0.5, fc near 5 kHz

Vin Vin 0 AC 1 DC 0
Vdd vdd 0 DC 5.0
Vss vss 0 DC -5.0

R1 Vin n1 3.18k
R2 n1 n2 3.18k
C1 n1 Vout 10n
C2 n2 0 10n

Eopamp Vout 0 n2 Vout 1e6

.control
ac dec 100 10 1Meg
let vdb = db(v(Vout)/v(Vin))
print vdb
.endc

.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "sallen-key"
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
