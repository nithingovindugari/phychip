"""SPICE Reward Card — structured per-rollout diagnostics (paper P0).

Every rollout gets one JSON record separating syntax, simulation, analysis,
measurement, and spec failures, so negative RL results become analyzable
science instead of an opaque scalar (lesson #122: reward curves alone are
uninterpretable; lesson #121: artifacts must be auditable).

Schema (one dict per rollout):
    completion_has_block  - fenced ```spice block present
    syntax_ok             - block contains device lines + .end
    analysis_ran          - analysis directive present (standalone or .control)
    ngspice_converged     - SimResult.success
    measurements_extracted- number of voltage/current measurements parsed
    spec_results          - {spec_key: {target, measured, passed, margin}}
    pass_count            - specs passed
    spec_total            - specs requested
    passed                - all specs passed
    reward                - scalar (same shaping ladder as train_lora_grpo)
    exploit_flags         - reward without analysis / >2 specs missing measurement
"""
from __future__ import annotations

import re
from typing import Any

SPICE_BLOCK_RE = re.compile(r"```spice\s*(.*?)```", re.DOTALL | re.IGNORECASE)
DEVICE_RE = re.compile(r"^\s*[RCLMQVIXDEJ]\w*\s+\S+\s+\S+", re.MULTILINE | re.IGNORECASE)
ANALYSIS_RE = re.compile(r"\.(op|tran|ac|dc|noise)\b|\.control\b", re.IGNORECASE)


def extract_block(completion: str) -> str:
    m = SPICE_BLOCK_RE.search(completion)
    if m:
        return m.group(1).strip()
    # continuation mode: prompt opened the fence
    head = completion.split("```")[0]
    return head.strip() if DEVICE_RE.search(head) else ""


def build_reward_card(completion: str, spec: dict[str, Any], env) -> dict[str, Any]:
    """Run a completion through ngspice and emit the structured card."""
    netlist = extract_block(completion)
    card: dict[str, Any] = {
        "completion_has_block": bool(netlist),
        "syntax_ok": False,
        "analysis_ran": False,
        "ngspice_converged": False,
        "measurements_extracted": 0,
        "spec_results": {},
        "pass_count": 0,
        "spec_total": len(spec),
        "passed": False,
        "reward": 0.0,
        "exploit_flags": [],
    }
    if not netlist:
        return card

    card["syntax_ok"] = bool(DEVICE_RE.search(netlist)) and ".end" in netlist.lower()
    card["analysis_ran"] = bool(ANALYSIS_RE.search(netlist))

    # shaping ladder (matches train_lora_grpo)
    reward = 0.1
    if card["syntax_ok"]:
        reward = 0.2
    if card["syntax_ok"] and card["analysis_ran"]:
        reward = 0.5
    if not card["syntax_ok"]:
        card["reward"] = reward
        return card

    # ngspice consumes the first line as the title; protect device lines
    sim = env.simulate("* rollout\n" + netlist, spec)
    card["ngspice_converged"] = sim.success
    card["measurements_extracted"] = len(sim.measurements)
    if not sim.success:
        card["reward"] = reward
        return card

    reward = 1.0
    missing = 0
    for key, target in spec.items():
        measured = sim.measurements.get(key)
        if measured is None:
            missing += 1
            card["spec_results"][key] = {"target": target, "measured": None,
                                         "passed": False, "margin": None}
            continue
        tgt = float(target if not isinstance(target, dict) else target.get("value", 0))
        tol = 0.10 if not isinstance(target, dict) else float(target.get("tol", 0.10))
        margin = abs(measured - tgt) / max(abs(tgt), 1e-12)
        ok = margin <= tol
        card["spec_results"][key] = {"target": tgt, "measured": measured,
                                     "passed": ok, "margin": round(margin, 4)}
        card["pass_count"] += int(ok)

    card["passed"] = spec and card["pass_count"] == len(spec)
    if card["passed"]:
        reward += 0.5
    card["reward"] = reward

    if reward >= 1.0 and not card["analysis_ran"]:
        card["exploit_flags"].append("reward_without_analysis")
    if missing > 2:
        card["exploit_flags"].append("specs_unmeasurable")
    return card
