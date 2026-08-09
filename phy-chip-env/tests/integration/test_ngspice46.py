from __future__ import annotations

import pytest

from phy_chip_env import runtime_fingerprint, score
from phy_chip_env.sandbox import grade_netlist

REFERENCE = """```spice
* authored non-inverting amplifier reference
Vin Vin 0 DC 0 AC 1
Rg Vfb 0 10k
Rf Vout Vfb 90k
XU1 Vin Vfb Vout ideal_opamp
.subckt ideal_opamp inp inn out
EGAIN out 0 inp inn 1e6
.ends ideal_opamp
.control
op
.endc
.end
```"""


@pytest.mark.integration
def test_authored_gold_reference_meets_requested_gain() -> None:
    result = score("noninv-amp", {"gain_db": [19.5, 20.5]}, REFERENCE)

    assert result.simulated
    assert not result.anticheat_blocked
    assert result.all_specs_met
    assert result.reward == pytest.approx(1.3)


@pytest.mark.integration
def test_actual_runtime_timeout_fails_closed() -> None:
    deck = """V1 in 0 1
R1 in 0 1k
.tran 1p 1000
.end
"""

    passed, reason = grade_netlist(deck, timeout_s=0.001)

    assert not passed
    assert reason == "timeout"


@pytest.mark.integration
def test_runtime_fingerprint_requires_ngspice46() -> None:
    fingerprint = runtime_fingerprint()

    assert "ngspice-46" in fingerprint["ngspice"]["version_output"]
    assert len(fingerprint["ngspice"]["sha256"]) == 64
    assert len(fingerprint["verifier_hash"]) == 64
    source_names = {source["name"] for source in fingerprint["sources"]}
    assert {"batching.py", "environment.py", "registry.py"} <= source_names
    assert len(fingerprint["sources"]) == 29
