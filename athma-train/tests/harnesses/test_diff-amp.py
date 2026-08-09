from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "athma_train"
    / "harnesses"
    / "diff-amp.py"
)

spec = importlib.util.spec_from_file_location("diff_amp_harness", MODULE_PATH)
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
* Difference amplifier, gain=2, finite bandwidth
Vp vp 0 DC 0 AC 0.5
Vn vn 0 DC 0 AC -0.5
Vdd vdd 0 DC 5
Vss vss 0 DC -5

R1 vn n_inv 10k
Rf n_inv vout 20k
R2 vp n_pos 10k
Rg n_pos 0 20k
CL vout 0 10p

.subckt opamp inp inn out vdd vss
Egain g 0 inp inn 200k
Rp g p 1k
Cp p 0 3.183u
Ebuf ob 0 p 0 1
Rout ob out 10
.ends opamp

X1 n_pos n_inv vout vdd vss opamp

.control
op
print v(vout) v(vp) v(vn)
.endc

.end
"""

MARGINAL_NETLIST = """
* Unity-gain difference amplifier with lower open-loop gain
Vp vp 0 DC 0 AC 0.5
Vn vn 0 DC 0 AC -0.5
Vdd vdd 0 DC 5
Vss vss 0 DC -5

R1 vn n_inv 10k
Rf n_inv vout 10k
R2 vp n_pos 10k
Rg n_pos 0 10k
CL vout 0 20p

Eop x 0 n_pos n_inv 10000
Rout x vout 100

.control
op
print v(vout)
.endc

.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "diff-amp"
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
