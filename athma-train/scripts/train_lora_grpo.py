#!/usr/bin/env python3
"""LoRA GRPO trainer for ablation condition D (and E's chained GRPO step).

100-step canary RLVR over LoRA on top of SFT+DPO. Reward = hierarchical
ngspice gate + per-spec margin from the appropriate harness module.

Usage:
    ABLATION_CONDITION=D ABLATION_STAGE=grpo \\
    .venv/bin/python athma-train/scripts/train_lora_grpo.py \\
        --base HuggingFaceTB/SmolLM3-3B-Base \\
        --sft-dpo-adapter checkpoints/ablation_C_sft_dpo_lora \\
        --specs data/rlvr_specs_v1/codex_session_1.jsonl \\
        --specs-limit 50 \\
        --out checkpoints/ablation_D_grpo_lora
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from athma_train.grpo_contract import (  # noqa: E402
    configure_chat_termination,
    prompt_contract_hash,
    render_grpo_prompt,
)
from athma_train.experiment_provenance import (  # noqa: E402
    ngspice_version,
    policy_hash as compute_policy_hash,
    verifier_hash as compute_verifier_hash,
)
from athma_train.env.rlvr_scorer import (  # noqa: E402
    SCORER_SEMANTICS,
    anticheat_ok as canonical_anticheat_ok,
    score_completion,
)
from athma_train.rlvr_data_gate import validate_trainability_label  # noqa: E402
from athma_train.spice_gate import grade_netlist, normalize_netlist  # noqa: E402

HARNESSES_DIR = Path(__file__).resolve().parent.parent / "athma_train" / "harnesses"
_HARNESS_CACHE: dict[str, object] = {}


def load_harness(circuit_id: str):
    """Dynamically import athma_train.harnesses.<circuit_id>."""
    if circuit_id in _HARNESS_CACHE:
        return _HARNESS_CACHE[circuit_id]
    path = HARNESSES_DIR / f"{circuit_id}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"harness_{circuit_id}", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"  warn: failed to load harness {circuit_id}: {e}", file=sys.stderr)
        return None
    _HARNESS_CACHE[circuit_id] = mod
    return mod


def extract_netlist(text: str) -> str:
    """Pull the spice block from a completion."""
    m = re.search(r"```spice\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    # fallback: whole text
    return normalize_netlist(text)


def completion_text(completion) -> str:
    """Normalize TRL conversational/plain completions for local audit output."""
    if isinstance(completion, list):
        if not completion:
            return ""
        value = completion[-1]
        if isinstance(value, dict):
            return str(value.get("content", ""))
    return str(completion or "")


def executable_policy_token_mask(tokenizer, token_ids: list[int]) -> list[int]:
    """Mask prose and SPICE comment tokens out of the GRPO policy loss.

    The verifier grades the executable deck, while the SFT policy often emits
    long, stale ``*`` design-note comments. Sequence reward applied to every
    response token would positively reinforce those false explanations. Keep
    attention on the complete on-policy trajectory, but apply policy gradient
    only to the fence, executable non-comment lines, and termination token.

    Malformed/unfenced completions deliberately retain an all-token mask so a
    negative advantage can still suppress the failure mode.
    """

    if not token_ids:
        return []
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    match = re.search(
        r"```(?:spice|ngspice)[^\S\r\n]*(?:\r?\n)?(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return [1] * len(token_ids)

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    roundtrip_ids = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    if len(offsets) != len(roundtrip_ids):
        raise RuntimeError("completion tokenizer returned misaligned offset mappings")

    allowed = [False] * len(text)
    for index in range(match.start(), match.start(1)):
        allowed[index] = True
    for index in range(match.end(1), match.end()):
        allowed[index] = True

    body = match.group(1)
    cursor = match.start(1)
    for line in body.splitlines(keepends=True):
        stripped = line.strip()
        is_title = stripped.lower().startswith(".title")
        executable = line.split(";", 1)[0]
        if executable.strip() and not stripped.startswith("*") and not is_title:
            for index in range(cursor, cursor + len(executable)):
                allowed[index] = True
            # Retain the line break that separates executable statements, but
            # not a trailing semicolon comment on the same line.
            line_ending = len(line) - len(line.rstrip("\r\n"))
            for index in range(cursor + len(line) - line_ending, cursor + len(line)):
                allowed[index] = True
        cursor += len(line)

    termination_ids = {getattr(tokenizer, "eos_token_id", None)}
    try:
        termination_ids.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    except AttributeError:
        pass
    canonical_mask: list[int] = []
    for token_id, (start, end) in zip(roundtrip_ids, offsets):
        # Fast tokenizers use (0, 0) for added/special termination tokens.
        if token_id in termination_ids or start == end:
            canonical_mask.append(1)
        else:
            canonical_mask.append(int(any(allowed[start:end])))

    if roundtrip_ids == list(token_ids):
        mask = canonical_mask
    else:
        # BPE decoding is many-to-one: a generated sequence can use adjacent
        # tokens whose decoded text retokenizes as one larger merge token.
        # Align the canonical offset-bearing tokenization back to the actual
        # generated IDs instead of requiring impossible ID-level round trips.
        canonical_text = tokenizer.decode(roundtrip_ids, skip_special_tokens=False)
        if canonical_text != text:
            raise RuntimeError(
                "completion text did not round-trip for policy-loss masking"
            )
        mask = [0] * len(token_ids)
        matcher = difflib.SequenceMatcher(
            a=list(token_ids),
            b=roundtrip_ids,
            autojunk=False,
        )
        for tag, original_start, original_end, canonical_start, canonical_end in matcher.get_opcodes():
            if tag == "equal":
                mask[original_start:original_end] = canonical_mask[
                    canonical_start:canonical_end
                ]
                continue
            # Fail closed across ambiguous alignment blocks. If one replacement
            # spans both masked prose/comment text and executable text, do not
            # let the executable portion make the whole block trainable.
            canonical_block = canonical_mask[canonical_start:canonical_end]
            block_allowed = int(bool(canonical_block) and all(canonical_block))
            for index in range(original_start, original_end):
                mask[index] = int(
                    token_ids[index] in termination_ids or block_allowed
                )
    if not any(mask):
        raise RuntimeError("policy-loss mask removed every completion token")
    return mask


def deduplicate_reward_items(
    items: list[tuple],
) -> tuple[list[tuple[str, object, object]], list[int]]:
    """Deduplicate exact deterministic verifier calls within one rollout batch."""

    unique: list[tuple[str, object, str]] = []
    inverse: list[int] = []
    indices: dict[tuple[str, str, str], int] = {}
    for completion, circuit_id, targets in items:
        normalized_completion = completion_text(completion)
        if isinstance(targets, str):
            targets_key = targets
        else:
            targets_key = json.dumps(
                targets or {},
                sort_keys=True,
                separators=(",", ":"),
            )
        key = (str(circuit_id), targets_key, normalized_completion)
        index = indices.get(key)
        if index is None:
            index = len(unique)
            indices[key] = index
            unique.append((normalized_completion, circuit_id, targets))
        inverse.append(index)
    return unique, inverse


def reward_fn(circuit_id: str, target_specs: dict, completion: str) -> float:
    """Reward-shaped hierarchical reward.

    Layers (each provides gradient signal so rare-pass rollouts aren't the only
    way to learn):

        0.00 → empty / no spice block / no .end
        0.10 → has ```spice``` block with .end
        0.20 → has at least one device line (R/C/L/M/Q/V/I/X)
        0.30 → has analysis directive (.op/.tran/.ac/.dc or .control block)
        0.50 → has both device lines AND analysis directive
        1.00 → ngspice simulation passes the gate
        1.0 + mean(per-spec margin) → harness-measured spec satisfaction (cap 2.0)

    This breaks the 0-reward floor that killed 20-step GRPO at sparse-pass
    rates. Now every rollout has variance even when none pass.
    """
    import re

    if not completion or not isinstance(completion, str):
        return 0.0

    # --- Layer 1: syntactic structure (cheap regex, no ngspice) ---
    has_spice_block = bool(re.search(r"```spice\s*.*?```", completion, re.DOTALL))
    netlist = extract_netlist(completion)
    has_end = bool(re.search(r"\.end\b", netlist, re.IGNORECASE))
    has_device = bool(
        re.search(r"^\s*[RCLMQVIX]\w*\s+\S+\s+\S+", netlist, re.MULTILINE | re.IGNORECASE)
    )
    has_analysis = bool(re.search(r"\.(op|tran|ac|dc)\b|\.control\b", netlist, re.IGNORECASE))

    if not (has_spice_block and has_end):
        return 0.0

    base = 0.1
    if has_device:
        base = 0.2
    if has_analysis:
        base = 0.3
    if has_device and has_analysis:
        base = 0.5

    # --- Layer 2: ngspice simulation gate ---
    passed, _ = grade_netlist(netlist, timeout_s=15)
    if not passed:
        return base
    base = 1.0

    # --- Layer 3: harness-measured spec satisfaction ---
    harness = load_harness(circuit_id)
    if harness is None or not target_specs:
        return base
    try:
        measured = harness.measure(netlist, spec=target_specs, timeout_s=15)
    except Exception:
        return base

    n_total = 0
    margin_sum = 0.0
    for k, v_target in target_specs.items():
        v_meas = measured.get(k)
        if v_meas is None:
            continue
        n_total += 1
        try:
            v_meas = float(v_meas)
            # Window target [lo, hi] (codex spec format): full credit inside,
            # graded penalty by distance outside relative to window scale.
            if isinstance(v_target, (list, tuple)) and len(v_target) == 2:
                lo, hi = float(v_target[0]), float(v_target[1])
                if lo > hi:
                    lo, hi = hi, lo
                if lo <= v_meas <= hi:
                    margin_sum += 1.0
                else:
                    d = (lo - v_meas) if v_meas < lo else (v_meas - hi)
                    scale = max(abs(hi - lo), abs((lo + hi) / 2.0), 1e-9)
                    margin_sum += max(0.0, 1.0 - d / scale)
            else:
                # Scalar target: relative-error credit.
                tgt = float(v_target)
                rel_err = abs(v_meas - tgt) / max(abs(tgt), 1e-9)
                margin_sum += 1.0 if rel_err < 0.1 else max(0.0, 1.0 - rel_err)
        except (TypeError, ValueError):
            pass
    if n_total == 0:
        return base
    return base + (margin_sum / n_total)


_anticheat_ok = canonical_anticheat_ok


# --- reward-decomposition instrument -----------------------------------------
# Record verifier components separately. Raw pass fractions are not gradient
# contributions under GRPO: only within-prompt variation creates an advantage.
# `_log_reward_decomp` therefore reports both rollout fractions and the fraction
# of K-sized groups with reward/spec/objective variance.
_DECOMP_STEP = [0]
_DECOMP_HISTORY: list[dict[str, float]] = []


def _has_variance(values: list[float | bool], atol: float = 1e-12) -> bool:
    return bool(values) and (max(values) - min(values)) > atol


def _log_reward_decomp(sink, group_size: int, unique_scores: int | None = None) -> None:
    if not sink:
        return
    n = len(sink)
    nosim = sum(1 for event in sink if not event["simulated"])
    met = sum(1 for event in sink if event["all_specs_met"])
    blocked = sum(1 for event in sink if event["anticheat_blocked"])
    mean_ss = sum(event["spec_score"] for event in sink) / n
    mean_r = sum(event["reward"] for event in sink) / n
    requested = sum(event["requested_specs"] for event in sink)
    measured = sum(event["measured_specs"] for event in sink)
    metric_coverage = measured / requested if requested else 0.0

    complete_group_count = n // group_size if group_size > 0 else 0
    groups = [
        sink[index * group_size : (index + 1) * group_size] for index in range(complete_group_count)
    ]
    reward_var = sum(_has_variance([event["reward"] for event in group]) for group in groups)
    spec_var = sum(_has_variance([event["spec_score"] for event in group]) for group in groups)
    objective_var = sum(
        _has_variance([event["all_specs_met"] for event in group]) for group in groups
    )
    winner_counts = {
        "exact": 0,
        "partial": 0,
        "gate_only": 0,
        "nosim": 0,
        "blocked": 0,
    }
    for group in groups:
        group_mean = sum(float(event["reward"]) for event in group) / len(group)
        for event in group:
            if float(event["reward"]) <= group_mean + 1e-12:
                continue
            if event["anticheat_blocked"]:
                winner_counts["blocked"] += 1
            elif not event["simulated"]:
                winner_counts["nosim"] += 1
            elif event["all_specs_met"]:
                winner_counts["exact"] += 1
            elif float(event["spec_score"]) > 0:
                winner_counts["partial"] += 1
            else:
                winner_counts["gate_only"] += 1
    winner_total = sum(winner_counts.values())
    denominator = len(groups) or 1
    _DECOMP_STEP[0] += 1
    unique_scores = n if unique_scores is None else unique_scores
    metrics = {
        "frac_nosim": nosim / n,
        "frac_spec_met": met / n,
        "frac_anticheat_blocked": blocked / n,
        "mean_spec_score": mean_ss,
        "metric_coverage": metric_coverage,
        "frac_group_reward_var": reward_var / denominator,
        "frac_group_spec_var": spec_var / denominator,
        "frac_group_objective_var": objective_var / denominator,
        "winner_exact_fraction": winner_counts["exact"] / (winner_total or 1),
        "winner_partial_fraction": winner_counts["partial"] / (winner_total or 1),
        "winner_gate_only_fraction": winner_counts["gate_only"] / (winner_total or 1),
        "winner_nosim_fraction": winner_counts["nosim"] / (winner_total or 1),
        "winner_blocked_fraction": winner_counts["blocked"] / (winner_total or 1),
    }
    _DECOMP_HISTORY.append(metrics)
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(
                {
                    "reward_decomp/generation_step": _DECOMP_STEP[0],
                    **{f"reward_decomp/{key}": value for key, value in metrics.items()},
                }
            )
    except Exception as exc:
        print(f"  warn: W&B reward-decomposition logging failed: {exc}", file=sys.stderr)
    print(
        f"  [reward-decomp ~{_DECOMP_STEP[0]}] n={n} mean_r={mean_r:.3f} "
        f"frac_nosim={nosim / n:.2f} frac_sim={(n - nosim) / n:.2f} "
        f"frac_spec_met={met / n:.2f} frac_anticheat_blocked={blocked / n:.2f} "
        f"mean_spec_score={mean_ss:.3f} metric_coverage={metric_coverage:.2f} "
        f"frac_group_reward_var={reward_var / denominator:.2f} "
        f"frac_group_spec_var={spec_var / denominator:.2f} "
        f"frac_group_objective_var={objective_var / denominator:.2f} "
        f"complete_groups={len(groups)} remainder={n - len(groups) * group_size} "
        f"unique_scores={unique_scores} deduplicated_scores={n - unique_scores} "
        f"winner_exact={metrics['winner_exact_fraction']:.2f} "
        f"winner_partial={metrics['winner_partial_fraction']:.2f} "
        f"winner_gate_only={metrics['winner_gate_only_fraction']:.2f} "
        f"winner_nosim={metrics['winner_nosim_fraction']:.2f} "
        f"winner_blocked={metrics['winner_blocked_fraction']:.2f}",
        file=sys.stderr,
    )


def trainable_parameter_sha256(model) -> str:
    """Fingerprint the exact trainable LoRA state without retaining a second copy."""

    import torch as _torch

    digest = hashlib.sha256()
    trainable = 0
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        trainable += parameter.numel()
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(_torch.uint8).numpy().tobytes())
    if trainable == 0:
        raise RuntimeError("model has no trainable parameters to fingerprint")
    return digest.hexdigest()


def reward_fn_v2(
    circuit_id: str,
    target_specs: dict,
    completion: str,
    eval_tol: float = 0.30,
    _sink: list | None = None,
) -> float:
    """TRL scalar adapter around the canonical environment scorer.

    ``score_completion`` is the sole reward implementation used by training,
    exact-regime labeling, evaluation, and auditable rescoring.  This wrapper is
    retained because TRL expects a float-returning callable.
    """
    result = score_completion(
        circuit_id,
        target_specs,
        completion,
        eval_tol=eval_tol,
        grade_fn=grade_netlist,
        harness_loader=load_harness,
        anticheat_fn=_anticheat_ok,
    )
    if _sink is not None:
        _sink.append(result.to_event())
    return result.reward


def build_training_rows(specs: list[dict], tokenizer) -> list[dict]:
    """Render the exact deployment/eval prompt for every GRPO training row."""

    return [
        {
            "prompt": render_grpo_prompt(tokenizer, spec),
            "spec_id": spec.get("spec_id") or (spec.get("data_quality") or {}).get("spec_id"),
            "circuit_id": spec["circuit_id"],
            "targets": json.dumps(spec["target_measurements"], sort_keys=True),
        }
        for spec in specs
    ]


def gradient_checkpointing_config(environ) -> dict:
    """Return the fail-closed gradient-checkpointing portion of GRPOConfig."""

    config = {"gradient_checkpointing": False}
    if environ.get("PHYCHIP_GRAD_CKPT") == "1":
        config["gradient_checkpointing"] = True
        config["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return config


def select_reward_function(environ):
    """Return the canonical scorer adapter and reject legacy reward modes."""

    requested = environ.get("PHYCHIP_REWARD", "canonical")
    if requested not in {"canonical", "v2"}:
        raise RuntimeError("legacy PHYCHIP_REWARD modes are retired; use the canonical RLVR scorer")
    return reward_fn_v2


def require_matching_trainability_evidence(
    row: dict,
    *,
    expected_k: int,
    expected_hashes: dict[str, str],
) -> dict:
    """Fail unless one row's exact labels match the launch regime exactly."""

    derived, reasons = validate_trainability_label(row)
    if reasons or not derived.get("trainable"):
        detail = ", ".join(reasons) if reasons else str(derived.get("status"))
        raise ValueError(f"row lacks valid mixed-success evidence: {detail}")
    if derived.get("K") != expected_k:
        raise ValueError(f"trainability K={derived.get('K')} does not match launch K={expected_k}")
    for name, expected in expected_hashes.items():
        if derived.get(name) != expected:
            raise ValueError(f"trainability {name} does not match this launch")
    return derived


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM3-3B-Base")
    ap.add_argument(
        "--base-revision",
        default="d78a42f79198603e614095753484a04c10c2b940",
    )
    ap.add_argument("--template-model", default="HuggingFaceTB/SmolLM3-3B")
    ap.add_argument(
        "--template-revision",
        default="a07cc9a04f16550a088caea529712d1d335b0ac1",
    )
    ap.add_argument(
        "--sft-dpo-adapter",
        required=True,
        help="Stacked SFT+DPO LoRA adapter — will be merged before GRPO LoRA",
    )
    ap.add_argument("--specs", type=Path, default=Path("data/rlvr_specs_v1/codex_session_1.jsonl"))
    ap.add_argument("--specs-limit", type=int, default=50)
    ap.add_argument(
        "--require-trainable-label",
        action="store_true",
        help="accept only rows that the RLVR gate derived as trainable_mixed_success",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--resume-from",
        default=None,
        help="path to a checkpoint-N dir to resume GRPO from (restores optimizer/scheduler/step)",
    )
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument(
        "--steps", type=int, default=100, help="Number of GRPO update steps (canary = 100)"
    )
    ap.add_argument(
        "--rollouts-per-step", type=int, default=4, help="K in GRPO (group size per prompt)"
    )
    # generation_batch_size = bs * world_size must be divisible by num_generations (K).
    # With 2 GPUs and K=4, bs=2 -> gen_batch=4 ✓ (bs=1 -> gen_batch=2 ✗).
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument(
        "--lr-scheduler-type",
        choices=("linear", "cosine"),
        default="linear",
    )
    ap.add_argument("--warmup-ratio", type=float, default=0.0)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--max-prompt-len", type=int, default=1024)
    ap.add_argument("--max-completion-len", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--num-iterations", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--save-total-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    condition = os.environ.get("ABLATION_CONDITION", "?")
    stage = os.environ.get("ABLATION_STAGE", "grpo")
    args.out.mkdir(parents=True, exist_ok=True)
    rollout_audit_path_value = os.environ.get("PHYCHIP_ROLLOUT_AUDIT_PATH", "").strip()
    rollout_audit_limit = max(0, int(os.environ.get("PHYCHIP_ROLLOUT_AUDIT_LIMIT", "64")))
    rollout_audit_handle = None
    rollout_audit_written = 0
    if rollout_audit_path_value and rollout_audit_limit:
        rollout_audit_path = Path(rollout_audit_path_value)
        rollout_audit_path.parent.mkdir(parents=True, exist_ok=True)
        rollout_audit_handle = rollout_audit_path.open("w", encoding="utf-8")
        print(
            f"  local rollout audit: {rollout_audit_path} "
            f"(limit={rollout_audit_limit}; never sent to W&B)",
            file=sys.stderr,
        )

    if not args.no_wandb:
        from athma_train.tracking import (
            configure_wandb_env,
            wandb_init_kwargs,
            assert_wandb_token_present,
        )

        assert_wandb_token_present()
        configure_wandb_env("ablation")
        import wandb

        if condition.startswith("POWERED_SEED"):
            adapter_chain = "clean-sft→grpo"
        elif condition == "D":
            adapter_chain = "sft→dpo→grpo"
        else:
            adapter_chain = "cpt→sft→dpo→grpo"
        wandb.init(
            **wandb_init_kwargs(
                stage="ablation",
                base=args.base,
                descriptor=f"{condition}-grpo-loraR{args.lora_rank}-K{args.rollouts_per_step}",
                framework="trl",
                extra_tags=[
                    f"condition:{condition}",
                    f"stage:{stage}",
                    f"lora_rank:{args.lora_rank}",
                    "adapter_chain:" + adapter_chain,
                ],
                config={**vars(args), "condition": condition},
            )
        )
        wandb.define_metric("reward_decomp/generation_step")
        wandb.define_metric(
            "reward_decomp/*", step_metric="reward_decomp/generation_step"
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    from trl import GRPOConfig, GRPOTrainer

    class ExecutableNetlistGRPOTrainer(GRPOTrainer):
        """Use TRL's tool mask as a policy-loss-only executable-token mask."""

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            completion_ids = output["completion_ids"]
            completion_mask = output["completion_mask"]
            executable_mask = torch.zeros_like(completion_mask)
            for row_index in range(completion_ids.size(0)):
                active_tokens = int(completion_mask[row_index].sum().item())
                if active_tokens == 0:
                    continue
                ids = completion_ids[row_index, :active_tokens].tolist()
                row_mask = executable_policy_token_mask(self._tokenizer, ids)
                executable_mask[row_index, :active_tokens] = torch.tensor(
                    row_mask,
                    dtype=executable_mask.dtype,
                    device=executable_mask.device,
                )
            if "tool_mask" in output:
                executable_mask = executable_mask * output["tool_mask"]
            output["tool_mask"] = executable_mask
            local_active = executable_mask.sum()
            local_total = completion_mask.sum()
            global_active = self.accelerator.gather(local_active).sum()
            global_total = self.accelerator.gather(local_total).sum().clamp(min=1)
            output["num_items_in_batch"] = global_active.clamp(min=1)
            fraction = (global_active / global_total).detach()
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["policy/executable_token_fraction"].append(fraction.item())
            return output

    print(f"Loading base: {args.base}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.base, revision=args.base_revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Base tokenizer has no chat_template; pull from Instruct variant so that
    # GRPO's conversational-prompt path generates chat-formatted text matching
    # the SFT adapter's training distribution. Without this, raw prompts hit a
    # chat-trained model and it emits a single EOS token (mean_length=1, reward=0).
    if tok.chat_template is None:
        try:
            from transformers import AutoTokenizer as _ATK

            adapter_template = Path(args.sft_dpo_adapter) / "chat_template.jinja"
            if adapter_template.is_file():
                tok.chat_template = adapter_template.read_text(encoding="utf-8")
                print("  loaded chat_template from the pinned SFT adapter", file=sys.stderr)
            else:
                inst = _ATK.from_pretrained(args.template_model, revision=args.template_revision)
                tok.chat_template = inst.chat_template
                print("  loaded pinned chat_template from SmolLM3-3B", file=sys.stderr)
        except Exception as e:
            print(f"  warn: couldn't load instruct chat_template: {e}", file=sys.stderr)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = {"": local_rank} if local_rank >= 0 else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        revision=args.base_revision,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
    )
    print(f"Loading SFT+DPO adapter: {args.sft_dpo_adapter}", file=sys.stderr)
    model = PeftModel.from_pretrained(model, args.sft_dpo_adapter, is_trainable=False)
    model = model.merge_and_unload()

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if os.environ.get("PHYCHIP_GRAD_CKPT") == "1":
        # Required only for the explicit checkpointing opt-in on a PEFT model.
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    initial_trainable_sha256 = trainable_parameter_sha256(model)
    print(f"  initial trainable-parameter SHA-256: {initial_trainable_sha256}", file=sys.stderr)

    # TRL constructs its own generation config from the tokenizer. Fail closed
    # unless the assistant-turn token itself is the EOS and padding is distinct.
    termination = configure_chat_termination(tok, model)
    print(f"  chat termination: {termination}", file=sys.stderr)

    expected_label_hashes = {
        "policy_hash": compute_policy_hash(
            args.base, args.base_revision, [Path(args.sft_dpo_adapter)]
        ),
        "verifier_hash": compute_verifier_hash(
            args.specs,
            Path(__file__).resolve().parent.parent,
            ngspice_version(),
        ),
        "prompt_contract_hash": prompt_contract_hash(tok),
    }

    # Load RLVR specs — expect schema {circuit_id, spec_prompt, target_measurements}
    from datasets import Dataset

    print(f"Loading RLVR specs: {args.specs}", file=sys.stderr)
    specs = []
    rejected_unlabeled = 0
    with args.specs.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = row.get("circuit_id")
            prompt = row.get("spec_prompt") or row.get("prompt")
            targets = row.get("target_measurements") or row.get("targets") or {}
            if args.require_trainable_label:
                quality = row.get("data_quality")
                evidence = quality.get("trainability") if isinstance(quality, dict) else None
                if not (
                    row.get("trainable") is True
                    and isinstance(evidence, dict)
                    and evidence.get("status") == "trainable_mixed_success"
                    and evidence.get("trainable") is True
                ):
                    rejected_unlabeled += 1
                    continue
                try:
                    require_matching_trainability_evidence(
                        row,
                        expected_k=args.rollouts_per_step,
                        expected_hashes=expected_label_hashes,
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        f"stale/incompatible trainability evidence at input line: {exc}"
                    ) from exc
            if cid and prompt:
                specs.append(
                    {
                        **row,
                        "circuit_id": cid,
                        "spec_prompt": prompt,
                        "target_measurements": targets,
                    }
                )
    specs = specs[: args.specs_limit]
    print(
        f"  using {len(specs)} specs; rejected_unlabeled={rejected_unlabeled}",
        file=sys.stderr,
    )
    if not specs:
        raise RuntimeError("no eligible RLVR specs remain after input validation")
    # Raw completion prompts caused the July train/eval distribution mismatch
    # and never emitted the assistant EOS. They are no longer a supported mode.
    if os.environ.get("PHYCHIP_GRPO_COMPLETION_FMT") == "1":
        raise RuntimeError(
            "PHYCHIP_GRPO_COMPLETION_FMT=1 is unsafe: train/eval must use the shared chat prompt"
        )
    ds = Dataset.from_list(build_training_rows(specs, tok))
    print("  prompt format: SHARED CHAT (system + spec + interface, no-think)", file=sys.stderr)

    # Reward function (TRL signature). The packaged scorer is unconditional;
    # silently falling back to legacy reward-v1 would invalidate train/eval parity.
    import os as _os

    _reward = select_reward_function(_os.environ)
    # MFU lever: the reward is bs*K independent ngspice sims/step (CPU-bound). Serial
    # scoring strands the GPU at ~28% util (lesson #152/#154). PHYCHIP_REWARD_WORKERS>1
    # scores the group on a thread pool — grade_netlist uses per-call NamedTemporaryFile
    # + subprocess, so it is thread-safe and the GIL releases during the ngspice wait.
    # Default 1 = serial (production-unchanged). Set ~= min(vCPUs, bs*K) on the pod.
    _RW = max(1, int(_os.environ.get("PHYCHIP_REWARD_WORKERS", "1")))
    _POOL = None
    if _RW > 1:
        from concurrent.futures import ThreadPoolExecutor

        _POOL = ThreadPoolExecutor(max_workers=_RW)
    print(f"  reward fn: {_reward.__name__} | reward workers: {_RW}", file=sys.stderr)

    def _score_one(item, capture_event=False):
        comp, cid, tgt_json = item
        comp = completion_text(comp)
        try:
            tgt = json.loads(tgt_json) if isinstance(tgt_json, str) else (tgt_json or {})
        except json.JSONDecodeError:
            tgt = {}
        event_sink = [] if capture_event else None
        reward = _reward(cid, tgt, comp, _sink=event_sink)
        if capture_event:
            if len(event_sink) != 1:
                raise RuntimeError("reward_fn_v2 must emit exactly one decomposition event")
            return reward, event_sink[0]
        return reward

    # RANDOM-REWARD CONTROL (PHYCHIP_RANDOM_REWARD=1): rewards uncorrelated with the
    # completion. If GRPO "improves" held-out pass@1 under this, the gain is spurious
    # (contamination/leakage), not real RL signal. Run alongside the real GRPO before
    # claiming any lift (the control that separates signal from luck).
    _RANDOM_REWARD = _os.environ.get("PHYCHIP_RANDOM_REWARD") == "1"
    if _RANDOM_REWARD:
        import random as _rnd_ctrl

        print("  !! RANDOM-REWARD CONTROL active — rewards are random U(0,1)", file=sys.stderr)

    def reward_func(completions, **kwargs):
        nonlocal rollout_audit_written
        if _RANDOM_REWARD:
            return [_rnd_ctrl.random() for _ in completions]
        # kwargs gives full row passthrough via dataset columns
        circuit_ids = kwargs.get("circuit_id", [None] * len(completions))
        spec_ids = kwargs.get("spec_id", [None] * len(completions))
        targets_list = kwargs.get("targets", ["{}"] * len(completions))
        items = list(zip(completions, circuit_ids, targets_list))
        unique_items, inverse = deduplicate_reward_items(items)
        capture = True
        if _POOL is not None:
            unique_scored = list(
                _POOL.map(lambda item: _score_one(item, capture), unique_items)
            )
        else:
            unique_scored = [_score_one(item, capture) for item in unique_items]
        scored = [unique_scored[index] for index in inverse]
        if capture:
            rewards = [reward for reward, _event in scored]
            events = [event for _reward_value, event in scored]
            _log_reward_decomp(
                events,
                args.rollouts_per_step,
                unique_scores=len(unique_items),
            )
            if rollout_audit_handle is not None and rollout_audit_written < rollout_audit_limit:
                for item, spec_id, reward_value, event in zip(
                    items, spec_ids, rewards, events
                ):
                    if rollout_audit_written >= rollout_audit_limit:
                        break
                    comp, cid, tgt_json = item
                    try:
                        targets = json.loads(tgt_json) if isinstance(tgt_json, str) else tgt_json
                    except json.JSONDecodeError:
                        targets = None
                    rollout_audit_handle.write(
                        json.dumps(
                            {
                                "schema": "phychip-grpo-rollout-audit-v1",
                                "index": rollout_audit_written,
                                "generation_group": _DECOMP_STEP[0],
                                "spec_id": spec_id,
                                "circuit_id": cid,
                                "targets": targets,
                                "completion": completion_text(comp),
                                "completion_char_count": len(completion_text(comp)),
                                "reward": float(reward_value),
                                "reward_decomposition": event,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    rollout_audit_handle.flush()
                    rollout_audit_written += 1
        else:
            rewards = scored
        return rewards

    # TRL 1.5.1: GRPOConfig has max_completion_length but not max_prompt_length.
    cfg_kwargs = dict(
        output_dir=str(args.out),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # Keep one complete K-sized rollout group even when optimizer
        # microbatching is reduced for memory. TRL then reuses that generation
        # batch across the required microsteps instead of forcing bs >= K.
        generation_batch_size=args.rollouts_per_step,
        num_generations=args.rollouts_per_step,
        num_iterations=args.num_iterations,
        max_steps=args.steps,
        max_completion_length=args.max_completion_len,
        temperature=args.temperature,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        beta=args.kl_coef,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=1,  # every step for canary
        bf16=True,
        report_to="wandb" if not args.no_wandb else "none",
        seed=args.seed,
    )
    # GRPOConfig defaults this to True in installed TRL 1.6. Set it explicitly:
    # this PEFT+merge setup previously produced grad_norm=0 with checkpointing.
    # The memory-pressure opt-in uses the safer non-reentrant implementation.
    cfg_kwargs.update(gradient_checkpointing_config(_os.environ))
    # --- 2026 RLVR recipe, env-gated so default behavior is unchanged ---
    # PHYCHIP_GRPO_RECIPE=modern enables Dr.GRPO reward scaling plus DAPO loss.
    # It does NOT implement DAPO dynamic sampling; trainability filtering belongs
    # in the versioned input-data gate.
    import inspect as _inspect

    _gc_params = set(_inspect.signature(GRPOConfig.__init__).parameters)
    if _os.environ.get("PHYCHIP_GRPO_RECIPE") == "modern":
        _wanted = dict(
            scale_rewards=False,  # Dr.GRPO: drop std-normalization bias
            loss_type="dapo",  # token-level loss aggregation
            epsilon_high=0.28,  # DAPO asymmetric clip (preserve exploration)
            mask_truncated_completions=True,
        )
        missing_recipe_fields = sorted(set(_wanted) - _gc_params)
        if missing_recipe_fields:
            raise RuntimeError(
                "installed TRL cannot realize the requested modern recipe: missing "
                + ", ".join(missing_recipe_fields)
            )
        for _k, _v in _wanted.items():
            cfg_kwargs[_k] = _v
    # individual overrides (e.g. PHYCHIP_GRPO_SCALE_REWARDS=0)
    if "PHYCHIP_GRPO_SCALE_REWARDS" in _os.environ and "scale_rewards" in _gc_params:
        cfg_kwargs["scale_rewards"] = _os.environ["PHYCHIP_GRPO_SCALE_REWARDS"] not in (
            "0",
            "false",
            "False",
        )
    cfg = GRPOConfig(**cfg_kwargs)
    if _os.environ.get("PHYCHIP_GRPO_RECIPE") == "modern":
        if cfg.scale_rewards != "none" or cfg.loss_type != "dapo":
            raise RuntimeError("GRPOConfig did not preserve the requested Dr.GRPO + DAPO recipe")
        if not cfg.mask_truncated_completions:
            raise RuntimeError("modern GRPO must mask truncated completions")
    requested_grad_ckpt = _os.environ.get("PHYCHIP_GRAD_CKPT") == "1"
    if bool(cfg.gradient_checkpointing) != requested_grad_ckpt:
        raise RuntimeError("GRPOConfig did not preserve the gradient-checkpointing contract")
    if requested_grad_ckpt and cfg_kwargs.get("gradient_checkpointing_kwargs") != {
        "use_reentrant": False
    }:
        raise RuntimeError("gradient checkpointing must use the non-reentrant implementation")
    if tok.eos_token != "<|im_end|>" or tok.pad_token_id == tok.eos_token_id:
        raise RuntimeError("chat termination contract is not active")
    if not args.no_wandb:
        import wandb

        wandb.config.update(
            {
                "resolved_prompt_contract_hash": expected_label_hashes["prompt_contract_hash"],
                "resolved_policy_hash": expected_label_hashes["policy_hash"],
                "resolved_verifier_hash": expected_label_hashes["verifier_hash"],
                "resolved_ngspice_version": ngspice_version(),
                "resolved_reward_semantics": SCORER_SEMANTICS,
                "resolved_reward_workers": _RW,
                "resolved_scale_rewards": cfg.scale_rewards,
                "resolved_loss_type": cfg.loss_type,
                "resolved_epsilon_high": getattr(cfg, "epsilon_high", None),
                "resolved_mask_truncated_completions": cfg.mask_truncated_completions,
                "resolved_gradient_checkpointing": cfg.gradient_checkpointing,
                "initial_trainable_sha256": initial_trainable_sha256,
            },
            allow_val_change=True,
        )
    print(
        f"  GRPOConfig recipe: scale_rewards={cfg_kwargs.get('scale_rewards', True)} "
        f"loss_type={cfg_kwargs.get('loss_type', 'grpo')} beta(KL)={args.kl_coef} "
        f"K={args.rollouts_per_step} grad_accum={args.gradient_accumulation_steps} "
        f"num_iterations={args.num_iterations} mask_truncated="
        f"{cfg_kwargs.get('mask_truncated_completions', False)}",
        file=sys.stderr,
    )
    # Optional: Microsoft post-training-toolkit diagnostics (GRPO group
    # rewards / advantages / KL, crash postmortem). Never breaks the run.
    callbacks = []
    try:
        from post_training_toolkit import DiagnosticsCallback

        callbacks.append(DiagnosticsCallback())
        print("  + post-training-toolkit DiagnosticsCallback enabled", file=sys.stderr)
    except Exception as e:
        print(f"  (post-training-toolkit not active: {e})", file=sys.stderr)

    trainer = ExecutableNetlistGRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        reward_funcs=[reward_func],
        processing_class=tok,
        callbacks=callbacks or None,
    )
    if args.resume_from:
        print(f"  RESUMING GRPO from {args.resume_from}", file=sys.stderr)
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()
    final_trainable_sha256 = trainable_parameter_sha256(model)
    if final_trainable_sha256 == initial_trainable_sha256:
        raise RuntimeError("GRPO completed without changing any trainable LoRA parameter")
    print(f"  final trainable-parameter SHA-256: {final_trainable_sha256}", file=sys.stderr)
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))
    if rollout_audit_handle is not None:
        rollout_audit_handle.close()

    (args.out / "ablation_manifest.json").write_text(
        json.dumps(
            {
                "condition": condition,
                "stage": stage,
                "base": args.base,
                "base_revision": args.base_revision,
                "template_model": args.template_model,
                "template_revision": args.template_revision,
                "sft_dpo_adapter": args.sft_dpo_adapter,
                "specs": str(args.specs),
                "specs_used": len(specs),
                "lora_rank": args.lora_rank,
                "steps": args.steps,
                "rollouts_per_step": args.rollouts_per_step,
                "lr": args.lr,
                "kl_coef": args.kl_coef,
                "lr_scheduler_type": args.lr_scheduler_type,
                "warmup_ratio": args.warmup_ratio,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "generation_batch_size": args.rollouts_per_step,
                "num_iterations": args.num_iterations,
                "max_completion_length": args.max_completion_len,
                "prompt_contract": "grpo-shared-chat-v2-netlist-only",
                "policy_loss_mask": "spice-fence-executable-noncomment-v2",
                "label_provenance": expected_label_hashes,
                "assistant_eos_token_id": termination["assistant_eos_token_id"],
                "mask_truncated_completions": cfg_kwargs.get("mask_truncated_completions", False),
                "save_total_limit": args.save_total_limit,
                "local_rollout_audit_rows": rollout_audit_written,
                "initial_trainable_sha256": initial_trainable_sha256,
                "final_trainable_sha256": final_trainable_sha256,
            },
            indent=2,
        )
    )
    print(f"DONE: GRPO adapter saved to {args.out}", file=sys.stderr)
    if not args.no_wandb:
        import wandb

        wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
