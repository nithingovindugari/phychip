"""Single prompt and termination contract shared by PhyChip GRPO train/eval.

The policy is deployed through chat, so RL rollouts and evaluation must render
the same system message, user specification, grader interface, and reasoning
mode.  Keeping the renderer here prevents an apparently harmless prompt edit
from silently changing only one side of the experiment.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any


SYS_PROMPT = (
    "You are an expert analog IC designer. Given a specification, output a complete "
    "ngspice netlist that meets EVERY target measurement. Use exactly the named ports. "
    "Return exactly one ```spice fenced netlist ending with .end and no text outside it. "
    "Do not include reasoning, design notes, or explanatory comments; executable circuit "
    "lines are the answer."
)
PROMPT_CONTRACT_VERSION = "grpo-shared-chat-v2-netlist-only"


def interface_contract(spec: dict[str, Any]) -> str:
    """Return the executable I/O contract used by both rollout and evaluation."""

    inputs = spec.get("input_ports") or []
    outputs = spec.get("output_ports") or []
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


def make_grpo_messages(spec: dict[str, Any]) -> list[dict[str, str]]:
    """Build the exact two-message prompt contract for one specification."""

    prompt = spec.get("spec_prompt") or spec.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("GRPO spec requires a non-empty spec_prompt")
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": prompt + interface_contract(spec)},
    ]


def render_grpo_prompt(tokenizer: Any, spec: dict[str, Any]) -> str:
    """Render the shared chat prompt with thinking disabled."""

    rendered = tokenizer.apply_chat_template(
        make_grpo_messages(spec),
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("chat template returned an empty/non-text GRPO prompt")
    return rendered


def prompt_contract_hash_from_template(chat_template: str) -> str:
    """Hash the exact template, prompt builders, and generation-mode contract."""

    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("cannot fingerprint an empty chat template")
    payload = {
        "version": PROMPT_CONTRACT_VERSION,
        "system_prompt": SYS_PROMPT,
        "interface_builder_source": inspect.getsource(interface_contract),
        "message_builder_source": inspect.getsource(make_grpo_messages),
        "chat_template": chat_template,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prompt_contract_hash(tokenizer: Any) -> str:
    """Hash the prompt contract used by a tokenizer."""

    return prompt_contract_hash_from_template(getattr(tokenizer, "chat_template", None))


def configure_chat_termination(tokenizer: Any, model: Any | None = None) -> dict[str, int]:
    """Make ``<|im_end|>`` the generation EOS and keep padding distinct.

    SmolLM3 assistant turns end with ``<|im_end|>`` while the base tokenizer's
    default EOS is ``<|end_of_text|>``.  TRL builds its own generation config
    from ``tokenizer.eos_token_id``, so changing only ``model.generation_config``
    is insufficient.
    """

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(im_end_id, int) or im_end_id < 0:
        raise ValueError("tokenizer does not contain <|im_end|>")
    try:
        roundtrip = tokenizer.convert_ids_to_tokens(im_end_id)
    except AttributeError:
        roundtrip = "<|im_end|>"
    if roundtrip != "<|im_end|>":
        raise ValueError("tokenizer does not contain an exact <|im_end|> token")

    original_eos_id = tokenizer.eos_token_id
    if original_eos_id is None:
        raise ValueError("tokenizer has no original EOS token for safe padding")
    original_eos_id = int(original_eos_id)
    tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == im_end_id:
        tokenizer.pad_token_id = original_eos_id
    if tokenizer.pad_token_id == im_end_id:
        raise ValueError("padding token must differ from assistant EOS")

    if model is not None and getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = im_end_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return {
        "assistant_eos_token_id": im_end_id,
        "pad_token_id": int(tokenizer.pad_token_id),
        "original_eos_token_id": original_eos_id,
    }
