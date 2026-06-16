from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "athma_train"
    / "harnesses"
    / "h-bridge.py"
)

spec = importlib.util.spec_from_file_location("h_bridge_harness", MODULE_PATH)
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
.title h bridge plain-coast
Vbus Vbus 0 9.6
VctrlA Vctrl_a 0 PULSE(0 9.6 0 50n 50n 0.75m 1.5m)
VctrlB Vctrl_b 0 PULSE(9.6 0 0 50n 50n 0.75m 1.5m)
Bgha gha 0 v={V(Vbus) - V(Vctrl_a)}
Bgla gla 0 v={V(Vctrl_a)}
Bghb ghb 0 v={V(Vbus) - V(Vctrl_b)}
Bglb glb 0 v={V(Vctrl_b)}
MHA Vmotor_p gha Vbus Vbus PMOD L=1u W=785u
MLA Vmotor_p gla 0 0 NMOD L=1u W=1005u
MHB Vmotor_n ghb Vbus Vbus PMOD L=1u W=790u
MLB Vmotor_n glb 0 0 NMOD L=1u W=1010u
Rmot Vmotor_p mot_mid 2.2
Lmot mot_mid Vmotor_n 160u
Dpa Vmotor_p Vbus DCLAMP
Dna 0 Vmotor_p DCLAMP
Dpb Vmotor_n Vbus DCLAMP
Dnb 0 Vmotor_n DCLAMP
Rterm Vmotor_p Vmotor_n 100k
.model NMOD nmos (LEVEL=1 VTO=2 KP=0.08 LAMBDA=0.02)
.model PMOD pmos (LEVEL=1 VTO=-2 KP=0.04 LAMBDA=0.02)
.model DCLAMP D (IS=1n RS=0.05 CJO=60p TT=20n)
.control
op
tran 1u 3m
.endc
.end
"""

MARGINAL_NETLIST = """
.title h bridge dual-snubber
Vbus Vbus 0 13.2
VctrlA Vctrl_a 0 PULSE(0 10.0 0 50n 50n 1.25m 2.5m)
VctrlB Vctrl_b 0 PULSE(10.0 0 0 50n 50n 1.25m 2.5m)
Bgha gha 0 v={V(Vbus) - V(Vctrl_a)}
Bgla gla 0 v={V(Vctrl_a)}
Bghb ghb 0 v={V(Vbus) - V(Vctrl_b)}
Bglb glb 0 v={V(Vctrl_b)}
MHA Vmotor_p gha Vbus Vbus PMOD L=1u W=735u
MLA Vmotor_p gla 0 0 NMOD L=1u W=955u
MHB Vmotor_n ghb Vbus Vbus PMOD L=1u W=740u
MLB Vmotor_n glb 0 0 NMOD L=1u W=960u
Rmot Vmotor_p mot_mid 3.7
Lmot mot_mid Vmotor_n 290u
Dpa Vmotor_p Vbus DCLAMP
Dna 0 Vmotor_p DCLAMP
Dpb Vmotor_n Vbus DCLAMP
Dnb 0 Vmotor_n DCLAMP
Rspa Vmotor_p spa 4.7
Cspa spa Vmotor_n 18n
.model NMOD nmos (LEVEL=1 VTO=2 KP=0.08 LAMBDA=0.02)
.model PMOD pmos (LEVEL=1 VTO=-2 KP=0.04 LAMBDA=0.02)
.model DCLAMP D (IS=1n RS=0.05 CJO=60p TT=20n)
.control
op
tran 2u 5m
.endc
.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "h-bridge"
    assert TIER == 2
    assert len(FOM_NAMES) == 4
    for fom in FOM_NAMES:
        assert fom in SPEC_UNITS


def test_good_netlist_extracts_all_foms():
    for netlist in (GOOD_NETLIST, MARGINAL_NETLIST):
        result = measure(netlist)
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
