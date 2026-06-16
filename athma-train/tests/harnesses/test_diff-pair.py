from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "athma_train"
    / "harnesses"
    / "diff-pair.py"
)

spec = importlib.util.spec_from_file_location("diff_pair_harness", MODULE_PATH)
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
* Differential Pair with PMOS Active Load
* Inputs: Vp, Vn | Output: Vout

.model NMOS_MODEL nmos level=1 kp=100u vto=0.5 lambda=0.01
.model PMOS_MODEL pmos level=1 kp=50u  vto=-0.5 lambda=0.02

Vdd vdd 0 DC 5.0
Vp_src Vp 0 DC 2.5 AC 1
Vn_src Vn 0 DC 2.5 AC 0

Vbias vbias 0 DC 1.2
M5 vtail vbias 0 0 NMOS_MODEL W=20u L=1u

M1 vmirror Vp vtail 0 NMOS_MODEL W=20u L=1u
M2 Vout Vn vtail 0 NMOS_MODEL W=20u L=1u

M3 vmirror vmirror vdd vdd PMOS_MODEL W=25u L=1u
M4 Vout vmirror vdd vdd PMOS_MODEL W=25u L=1u

.control
  op
  ac dec 100 1 1G
.endc

.end
"""

MARGINAL_NETLIST = """
* Differential pair with active load, lower intrinsic gain

.model NMOS_MODEL nmos level=1 kp=100u vto=0.5 lambda=0.08
.model PMOS_MODEL pmos level=1 kp=50u  vto=-0.5 lambda=0.08

Vdd vdd 0 DC 5.0
Vbias Vbias 0 DC 2.5
Vn Vn 0 DC 2.0
Vp Vp 0 DC 2.0

M1 d1 Vp tail 0 NMOS_MODEL W=100u L=10u
M2 Vout Vn tail 0 NMOS_MODEL W=100u L=10u

M3 d1 d1 Vdd Vdd PMOS_MODEL W=200u L=10u
M4 Vout d1 Vdd Vdd PMOS_MODEL W=200u L=10u

M5 tail Vbias 0 0 NMOS_MODEL W=50u L=10u

CL Vout 0 1p

.control
  dc Vp 1.5 2.5 5m
  op
.endc

.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "diff-pair"
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
