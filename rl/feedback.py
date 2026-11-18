"""Simulator feedback as text, and the closed-loop (design -> measure -> revise) message format.

One function builds the feedback message from a scorer event, so training data and the
evaluation loop render byte-identical feedback.
"""
from __future__ import annotations


def _fmt(value: float) -> str:
    return f"{value:.5g}"


def feedback_text(event: dict, targets: dict) -> str:
    """Plain-language verifier result for one attempt. `event` may carry `measured`."""
    if not (event.get("has_spice_block", True) and event.get("has_end", True)):
        return ("Verifier result: no complete netlist was found (it needs one ```spice fence ending with .end). "
                "Return the full corrected netlist.")
    if not event.get("simulated", False):
        return ("Verifier result: the netlist did not simulate in ngspice (syntax error, missing model, floating node "
                "or no convergence). Return the full corrected netlist.")
    if event.get("anticheat_blocked", False):
        return ("Verifier result: the netlist simulates but was rejected by the circuit-family check: the measured "
                "behaviour must come from a real instance of this circuit (correct topology, load-bearing devices, "
                "no ideal source producing the figure). Return the full corrected netlist.")
    measured = event.get("measured") or {}
    lines = []
    for key, bounds in targets.items():
        lo, hi = min(bounds), max(bounds)
        if key not in measured:
            lines.append(f"{key}: could not be measured (target [{_fmt(lo)}, {_fmt(hi)}])")
            continue
        value = measured[key]
        verdict = "in range" if lo <= value <= hi else ("too low" if value < lo else "too high")
        lines.append(f"{key} = {_fmt(value)} (target [{_fmt(lo)}, {_fmt(hi)}]): {verdict}")
    return ("Verifier result, measured by ngspice on your netlist:\n" + "\n".join(lines)
            + "\nRevise the circuit so every target is in range and return the full corrected netlist.")


def repair_messages(base_messages: list[dict], attempts: list[tuple[str, str]]) -> list[dict]:
    """system + user, then (assistant attempt, user feedback) for each earlier attempt."""
    msgs = list(base_messages[:2])
    for completion, feedback in attempts:
        msgs.append({"role": "assistant", "content": completion})
        msgs.append({"role": "user", "content": feedback})
    return msgs
