from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "athma_train"
    / "harnesses"
    / "noninv-amp.py"
)

spec = importlib.util.spec_from_file_location("noninv_amp_harness", MODULE_PATH)
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
* Non-Inverting Op-Amp (Gain = 10)

Vdd vdd 0 DC 5.0
Vss vss 0 DC -5.0
Vin Vin 0 DC 0 AC 1

R1 vfb 0 1k
R2 Vout vfb 9k

Eamp Vout 0 Vin vfb 1e6

.control
ac dec 100 1 100Meg
plot vdb(Vout)
.endc

.end
"""

MARGINAL_NETLIST = """
* Non-Inverting Amplifier, Gain = 2

Vdd vdd 0 DC 5.0
Vin Vin 0 DC 1.0 AC 1.0

Rf Vout Vmid 10k
R1 Vmid 0  10k
Eamp Vout 0 Vin Vmid 1e6
Rload Vout 0 100k

.control
op
print v(Vin) v(Vout)
ac dec 100 1 100Meg
plot vdb(Vout)
.endc

.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "noninv-amp"
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
