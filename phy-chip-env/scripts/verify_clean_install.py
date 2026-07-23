#!/usr/bin/env python3
"""Installed-wheel smoke gate for the v0.1 release candidate."""

from __future__ import annotations

import json
import math

from phy_chip_env import list_harnesses, runtime_fingerprint, score
from phy_chip_env.sandbox import grade_netlist

GOLD = """```spice
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

ATTACK = """```spice
V1 in 0 1
R1 in 0 1k
.control
op
shell touch /tmp/phychip-clean-install-pwned
.endc
.end
```"""


def main() -> None:
    harnesses = list_harnesses()
    assert len(harnesses) == 23

    gold = score("noninv-amp", {"gain_db": [19.5, 20.5]}, GOLD)
    assert gold.all_specs_met
    assert math.isclose(gold.reward, 1.3, rel_tol=0.0, abs_tol=1e-12)

    attack = score("noninv-amp", {"gain_db": [19.5, 20.5]}, ATTACK)
    assert not attack.simulated
    assert not attack.all_specs_met

    timeout_deck = "V1 in 0 1\nR1 in 0 1k\n.tran 1p 1000\n.end\n"
    passed, reason = grade_netlist(timeout_deck, timeout_s=0.001)
    assert not passed and reason == "timeout"

    fingerprint = runtime_fingerprint()
    assert "ngspice-46" in fingerprint["ngspice"]["version_output"]

    print(
        json.dumps(
            {
                "status": "pass",
                "package_version": fingerprint["package_version"],
                "harnesses": len(harnesses),
                "gold_reward": gold.reward,
                "attack_simulated": attack.simulated,
                "timeout_reason": reason,
                "verifier_hash": fingerprint["verifier_hash"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
