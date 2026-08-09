from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "athma_train"
    / "harnesses"
    / "current-mirror.py"
)

spec = importlib.util.spec_from_file_location("current_mirror_harness", MODULE_PATH)
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
* 1:1 NMOS Current Mirror
* Ports: iref (reference input), iout (mirror output)

Vdd vdd 0 DC 5.0
Iref vdd iref DC 100u

M1 iref iref 0 0 NMOS_MODEL W=10u L=1u
M2 iout iref 0 0 NMOS_MODEL W=10u L=1u

Vout iout 0 DC 1.0

.model NMOS_MODEL nmos level=1 kp=100u vto=0.5 lambda=0.01

.control
  op
  print i(Iref) i(Vout)
.endc

.end
"""

MARGINAL_NETLIST = """
* Mismatched NMOS Current Mirror

Vdd vdd 0 DC 5.0
Iref vdd ref DC 100u

M1 ref ref 0 0 NMOS_MODEL W=10u L=1u
M2 iout ref 0 0 NMOS_MODEL W=5u L=1u

Vout iout 0 DC 1.0

.model NMOS_MODEL nmos level=1 kp=100u vto=0.5 lambda=0.02

.control
  op
  print i(Iref) i(Vout)
.endc

.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "current-mirror"
    assert TIER in (0, 1, 2)
    assert len(FOM_NAMES) >= 3
    for fom in FOM_NAMES:
        assert fom in SPEC_UNITS


def test_good_netlist_extracts_all_foms():
    result = measure(GOOD_NETLIST)
    assert set(result.keys()) == set(FOM_NAMES)
    non_none = [value for value in result.values() if value is not None]
    assert len(non_none) >= 2


def test_marginal_netlist_extracts_some_foms():
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
