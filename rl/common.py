"""Shared prompt contract and task helpers for the PhyChip wrap-up campaign.

The prompt is the repository's GRPO chat contract (system prompt + user
specification + grader interface contract), reproduced here so the GPU box,
the frontier baselines and the data builders render byte-identical prompts.
"""
from __future__ import annotations

import json

SYS_PROMPT = (
    "You are an expert analog IC designer. Given a specification, output a complete "
    "ngspice netlist that meets EVERY target measurement. Use exactly the named ports. "
    "Return exactly one ```spice fenced netlist ending with .end and no text outside it. "
    "Do not include reasoning, design notes, or explanatory comments; executable circuit "
    "lines are the answer."
)

# The SFT-v1 corpus was written under this system prompt (SmolLM3 "custom instructions").
SFT_NATIVE_SYS = (
    "You are an expert analog circuit designer who emits ngspice netlists. "
    "Use the user's exact input/output node names."
)

def interface_contract(inputs, outputs) -> str:
    input_lines = "; ".join(f"`{port} {port} 0 DC 0 AC 1`" for port in inputs) or "(none)"
    return (
        "\n\nGRADER INTERFACE CONTRACT (follow exactly or the circuit cannot be measured):\n"
        "  - For each input port, include an independent voltage source whose NAME equals the "
        f"port name, driving a node of that name, with an AC magnitude: {input_lines}. The grader "
        "injects/overrides the stimulus by altering these sources, so a DC bias + `AC 1` is "
        "enough.\n"
        "  - Drive nothing else; the output(s) MUST be node(s) named exactly: "
        f"{', '.join(outputs) or '(see prompt)'}.\n"
        "  - End the netlist with a MINIMAL self-test block so it simulates standalone: a "
        "`.control` / `op` / `.endc` block (or a bare `.op`) then `.end`. The grader STRIPS this "
        "and injects its own AC/DC/transient measurement testbench — so do NOT add any "
        "`.ac/.tran/.dc/.meas/.print/.plot` analysis of your own; just the `op`.\n"
        "  - Otherwise provide only: a title comment, the circuit devices, and any behavioral "
        "op-amp `.subckt` you need.\n"
        "  - Do NOT use an ideal dependent source that directly synthesizes a measured FOM."
    )


def spec_prompt(family: str, inputs, outputs, targets: dict) -> str:
    return (
        f"Design a {family} circuit using input port(s) {', '.join(inputs)} and output "
        f"port(s) {', '.join(outputs)}. Meet these simulator-measured targets: "
        + "; ".join(f"{key} in [{lo:.8g}, {hi:.8g}]" for key, (lo, hi) in targets.items())
        + ". Return one self-contained ngspice netlist in a closed spice fence ending with .end."
    )


def messages(prompt: str, inputs, outputs, style: str = "grpo") -> list[dict]:
    user = prompt + interface_contract(inputs, outputs)
    if style == "grpo":
        return [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user}]
    if style == "sft":
        return [{"role": "system", "content": SFT_NATIVE_SYS}, {"role": "user", "content": user}]
    if style == "bare":
        return [{"role": "user", "content": SYS_PROMPT + "\n\n" + user}]
    raise ValueError(style)


def read_jsonl(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str, rows) -> None:
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
