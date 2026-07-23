#!/usr/bin/env python3
"""LoRA SFT trainer for ablation condition B (and E's chained SFT step).

Trains a LoRA adapter on top of SmolLM3-3B-Base (or another base) using
data/final_data/sft_v0/*.jsonl. Logs to W&B project phy-chip-ablation-2026-06.

Usage:
    # Condition B: base → LoRA SFT
    ABLATION_CONDITION=B ABLATION_STAGE=sft \\
    .venv/bin/python athma-train/scripts/train_lora_sft.py \\
        --base HuggingFaceTB/SmolLM3-3B-Base \\
        --data data/final_data/sft_v0 \\
        --out checkpoints/ablation_B_sft_lora

    # Condition E's SFT: CPT-adapter → SFT-adapter (stacked)
    ABLATION_CONDITION=E ABLATION_STAGE=sft \\
    .venv/bin/python athma-train/scripts/train_lora_sft.py \\
        --base HuggingFaceTB/SmolLM3-3B-Base \\
        --base-adapter checkpoints/ablation_E_cpt_lora \\
        --data data/final_data/sft_v0 \\
        --out checkpoints/ablation_E_cpt_sft_lora
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def resolve_data_files(data_path: Path) -> list[Path]:
    """Resolve the exact SFT membership without mixing gate artifacts.

    A dynamically verified corpus is passed as its explicit
    ``sft_training_ready.jsonl`` file.  If a gate output directory is supplied,
    select that file before the static-only ``sft_clean.jsonl`` fallback and
    never glob quarantine/manifest JSONLs into training.
    """

    if data_path.is_file():
        if data_path.suffix != ".jsonl":
            raise ValueError(f"SFT data file must be JSONL: {data_path}")
        return [data_path]
    if not data_path.is_dir():
        raise ValueError(f"SFT data path does not exist: {data_path}")
    for verified_name in ("sft_training_ready.jsonl", "sft_clean.jsonl"):
        verified = data_path / verified_name
        if verified.is_file():
            return [verified]
    files = sorted(data_path.glob("*.jsonl"))
    if not files:
        raise ValueError(f"no SFT JSONL files found in {data_path}")
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def training_data_hash_fields(
    data_file_identities: list[dict[str, str]],
) -> dict[str, object]:
    """Name raw-file and corpus-manifest hashes without conflating them.

    ``corpus_sha256`` is retained as a backward-compatible alias for the
    canonical hash of the ordered file-identity list.  It has never been the
    SHA-256 of the JSONL bytes themselves.
    """

    from athma_train.experiment_provenance import canonical_hash

    corpus_manifest_sha256 = canonical_hash(data_file_identities)
    return {
        "training_file_sha256": (
            data_file_identities[0]["sha256"]
            if len(data_file_identities) == 1
            else None
        ),
        "training_file_sha256s": {
            identity["name"]: identity["sha256"]
            for identity in data_file_identities
        },
        "corpus_manifest_sha256": corpus_manifest_sha256,
        # Backward compatibility for existing consumers and manifests.
        "corpus_sha256": corpus_manifest_sha256,
    }


def training_batch_topology(
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
) -> dict[str, int]:
    """Return the unambiguous process and effective global batch sizes."""

    if min(per_device_batch_size, gradient_accumulation_steps, world_size) <= 0:
        raise ValueError("batch topology values must be positive")
    return {
        "world_size": int(world_size),
        "global_batch_size": int(
            per_device_batch_size * gradient_accumulation_steps * world_size
        ),
    }


def nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def estimate_random_batch_padding(
    token_lengths: list[int], batch_size: int, sequence_length: int, seed: int
) -> dict[str, float | int]:
    """Estimate padding cost for random micro-batches across fixed seeds.

    TRL 1.6 does not expose ``group_by_length`` in ``SFTConfig``.  A larger
    micro-batch can therefore gain kernel throughput while losing work to
    padding.  Report both effects instead of pretending raw-token throughput
    transfers unchanged to the full shuffled corpus.
    """

    effective = [min(int(length), int(sequence_length)) for length in token_lengths]
    unpadded_tokens = sum(effective)
    if not effective or batch_size <= 1:
        return {
            "batch_size": int(batch_size),
            "unpadded_tokens": int(unpadded_tokens),
            "seed_count": 1,
            "seed_padding_multiplier": 1.0,
            "mean_padding_multiplier": 1.0,
            "p95_padding_multiplier": 1.0,
            "max_padding_multiplier": 1.0,
        }
    multipliers: list[float] = []
    for offset in range(32):
        shuffled = list(effective)
        random.Random(seed + offset).shuffle(shuffled)
        padded_tokens = 0
        for start in range(0, len(shuffled), batch_size):
            group = shuffled[start : start + batch_size]
            padded_tokens += max(group) * len(group)
        multipliers.append(padded_tokens / max(unpadded_tokens, 1))
    return {
        "batch_size": int(batch_size),
        "unpadded_tokens": int(unpadded_tokens),
        "seed_count": len(multipliers),
        "seed_padding_multiplier": round(multipliers[0], 6),
        "mean_padding_multiplier": round(sum(multipliers) / len(multipliers), 6),
        "p95_padding_multiplier": round(
            float(nearest_rank([round(value * 1_000_000) for value in multipliers], 0.95))
            / 1_000_000,
            6,
        ),
        "max_padding_multiplier": round(max(multipliers), 6),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM3-3B-Base")
    ap.add_argument(
        "--base-revision",
        default="d78a42f79198603e614095753484a04c10c2b940",
        help="Pinned Hugging Face commit for the base model.",
    )
    ap.add_argument("--template-model", default="HuggingFaceTB/SmolLM3-3B")
    ap.add_argument(
        "--template-revision",
        default="a07cc9a04f16550a088caea529712d1d335b0ac1",
        help="Pinned commit supplying the assistant-mask-aware chat template.",
    )
    ap.add_argument("--base-adapter", default=None,
                    help="Optional: load this LoRA adapter first (stacks on base) before SFT")
    ap.add_argument("--data", type=Path, default=Path("data/final_data/sft_v0"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument(
        "--max-steps", type=int, default=-1,
        help="Optional optimizer-step cap; use 1 for a paid-pod smoke test.",
    )
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--lr-scheduler-type", default="linear")
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument(
        "--smoke-longest-example",
        action="store_true",
        help="Train only on the longest tokenized row (paid memory/throughput smoke).",
    )
    args = ap.parse_args()

    condition = os.environ.get("ABLATION_CONDITION", "?")
    stage = os.environ.get("ABLATION_STAGE", "sft")
    args.out.mkdir(parents=True, exist_ok=True)

    # --- Tracking ---
    wandb_run = None
    if not args.no_wandb:
        from athma_train.tracking import (
            configure_wandb_env, wandb_init_kwargs, assert_wandb_token_present,
        )
        assert_wandb_token_present()
        configure_wandb_env("ablation")
        import wandb
        wandb_kwargs = wandb_init_kwargs(
            stage="ablation",
            base=args.base,
            descriptor=f"{condition}-sft-loraR{args.lora_rank}",
            framework="trl",
            extra_tags=[
                f"condition:{condition}",
                f"stage:{stage}",
                f"lora_rank:{args.lora_rank}",
                "adapter_chain:" + ("cpt→sft" if args.base_adapter else "sft"),
            ],
            config={
                "experiment/type": "sft_stronger_prior_challenger",
                "experiment/condition": condition,
                "model/base_repo": args.base,
                "model/base_revision": args.base_revision,
                "model/template_repo": args.template_model,
                "model/template_revision": args.template_revision,
                "lora/rank": args.lora_rank,
                "lora/alpha": args.lora_alpha,
                "lora/dropout": args.lora_dropout,
                "lora/target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                    "out_proj",
                ],
                "train/epochs": args.epochs,
                "train/max_steps": args.max_steps,
                "train/per_device_batch_size": args.bs,
                "train/gradient_accumulation_steps": args.grad_accum,
                "train/learning_rate": args.lr,
                "train/lr_scheduler_type": args.lr_scheduler_type,
                "train/optimizer": args.optim,
                "train/weight_decay": args.weight_decay,
                "train/warmup_ratio": args.warmup_ratio,
                "train/max_sequence_length": args.seq_len,
                "train/seed": args.seed,
                "train/assistant_only_loss": True,
                "train/packing": False,
                "train/bf16": True,
                "train/num_tokens_semantics": (
                    "cumulative_nonpadding_input_tokens_prompt_plus_assistant"
                ),
            },
        )
        wandb_kwargs["settings"] = wandb.Settings(
            disable_code=True,
            disable_git=True,
            console="off",
            x_save_requirements=False,
        )
        wandb_run = wandb.init(**wandb_kwargs)

    # --- Load model + tokenizer ---
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )
    from peft import LoraConfig, PeftModel, get_peft_model
    from trl import SFTConfig, SFTTrainer

    print(f"Loading base: {args.base}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.base, revision=args.base_revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Base models often lack chat_template. Pull it from the Instruct variant.
    if tok.chat_template is None:
        try:
            from transformers import AutoTokenizer as _ATK
            inst = _ATK.from_pretrained(
                args.template_model, revision=args.template_revision
            )
            tok.chat_template = inst.chat_template
            print("  loaded chat_template from SmolLM3-3B (instruct)", file=sys.stderr)
        except Exception as e:
            print(f"  warn: couldn't load instruct chat_template: {e}; falling back to simple format", file=sys.stderr)
            # NOTE: includes a {% generation %} block so assistant_only_loss works
            # (the loss mask is derived from these keywords).
            tok.chat_template = (
                "{% for m in messages %}"
                "{% if m['role'] == 'system' %}<|system|>\n{{ m['content'] }}<|end|>\n"
                "{% elif m['role'] == 'user' %}<|user|>\n{{ m['content'] }}<|end|>\n"
                "{% elif m['role'] == 'assistant' %}<|assistant|>\n"
                "{% generation %}{{ m['content'] }}<|end|>{% endgeneration %}\n"
                "{% endif %}{% endfor %}"
            )
    from athma_train.eval_contract import (
        ensure_assistant_generation_block,
        tokenized_sequence_length,
    )
    tok.chat_template = ensure_assistant_generation_block(str(tok.chat_template))
    # DDP-safe device map: under torchrun, each rank loads to its assigned GPU.
    # Single-process launch falls back to device_map="auto".
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = {"": local_rank} if local_rank >= 0 else "auto"
    base_config = AutoConfig.from_pretrained(
        args.base, revision=args.base_revision, trust_remote_code=True
    )
    model_loader = (
        AutoModelForImageTextToText
        if base_config.model_type == "qwen3_5"
        else AutoModelForCausalLM
    )
    model = model_loader.from_pretrained(
        args.base, revision=args.base_revision,
        dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True,
        attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
    )

    if args.base_adapter:
        print(f"Loading base adapter (CPT): {args.base_adapter}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, args.base_adapter, is_trainable=False)
        # Merge so we have a single base on which SFT-LoRA stacks
        model = model.merge_and_unload()

    lora_targets: list[str] | str
    matched_target_names: list[str]
    if base_config.model_type == "qwen3_5":
        # Qwen3.5-4B includes a vision tower. Adapt only the language model;
        # suffix-only matching would silently train visual modules on text.
        target_suffixes = (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
        )
        lora_targets = (
            r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|"
            r"gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|"
            r"in_proj_b|in_proj_a|out_proj)$"
        )
        matched_target_names = [
            name for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and "language_model" in name
            and name.endswith(target_suffixes)
        ]
        if len(matched_target_names) != 248:
            raise RuntimeError(
                "Qwen3.5 LoRA target contract expected 248 language projection "
                f"modules, found {len(matched_target_names)}"
            )
        if any("visual" in name for name in matched_target_names):
            raise RuntimeError("Qwen3.5 LoRA target contract selected visual modules")
    else:
        lora_targets = [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
            "up_proj", "down_proj",
        ]
        matched_target_names = [
            name for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and name.endswith(tuple(lora_targets))
        ]
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable_parameter_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if base_config.model_type == "qwen3_5" and any(
        "visual" in name for name in trainable_parameter_names
    ):
        raise RuntimeError("Qwen3.5 visual parameters became trainable")
    model.print_trainable_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    # --- Load data ---
    # Manual JSON load avoids datasets-library schema-unification crash
    # when JSONL files have heterogeneous fields (e.g. hardcase rows have
    # 'hardcase_mode' but original SFT rows don't).
    from datasets import Dataset
    files = resolve_data_files(args.data)
    print(f"Loading SFT data: {len(files)} files (manual JSON load)", file=sys.stderr)
    rows = []
    n_text_only = 0
    for f in files:
        with open(f) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = row.get("messages")
                if msgs:
                    # Keep conversational structure so SFTTrainer can apply the
                    # chat template AND compute the assistant-only loss mask.
                    # (Previously we pre-rendered to {"text": ...}, which threw the
                    # mask away and trained on prompt tokens too — see
                    # scripts/verify_sft_masking.py.)
                    rows.append({"messages": msgs})
                elif row.get("text"):
                    n_text_only += 1  # legacy text-only rows can't be assistant-masked; skipped
    corpus_row_count = len(rows)
    token_lengths: list[int] = []
    if wandb_run is not None or args.smoke_longest_example:
        for row in rows:
            encoded = tok.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=False,
                return_dict=True,
            )
            token_lengths.append(tokenized_sequence_length(encoded))
    data_token_stats = None
    batch_padding_stats = None
    if token_lengths:
        effective_lengths = [min(length, args.seq_len) for length in token_lengths]
        data_token_stats = {
            "mean": round(sum(token_lengths) / len(token_lengths), 4),
            "p95": nearest_rank(token_lengths, 0.95),
            "p99": nearest_rank(token_lengths, 0.99),
            "max": max(token_lengths),
            "rows_over_sequence_length": sum(
                length > args.seq_len for length in token_lengths
            ),
            "effective_tokens_at_sequence_length": sum(effective_lengths),
        }
        batch_padding_stats = estimate_random_batch_padding(
            token_lengths, args.bs, args.seq_len, args.seed
        )
    if args.smoke_longest_example:
        if not token_lengths:
            raise RuntimeError("longest-example smoke requires token lengths")
        longest_index = max(range(len(token_lengths)), key=token_lengths.__getitem__)
        # Repeat the worst-case row so --bs N actually exercises N simultaneous
        # 4,096-token sequences. A one-row dataset silently reduces every smoke
        # to micro-batch 1 and cannot prove larger-batch memory safety.
        rows = [rows[longest_index] for _ in range(args.bs)]
    ds = Dataset.from_list(rows)
    print(f"  loaded {len(ds):,} conversational examples"
          f" ({n_text_only} text-only rows skipped — not assistant-maskable)", file=sys.stderr)

    data_file_identities = [
        {"name": path.name, "sha256": sha256_file(path)} for path in files
    ]
    train_root = Path(__file__).resolve().parent.parent
    training_source_identities = [
        {
            "name": path.relative_to(train_root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in (
            Path(__file__).resolve(),
            train_root / "athma_train" / "tracking.py",
            train_root / "athma_train" / "eval_contract.py",
        )
    ]
    data_hash_fields = training_data_hash_fields(data_file_identities)
    corpus_sha256 = str(data_hash_fields["corpus_manifest_sha256"])
    chat_template_sha256 = hashlib.sha256(
        str(tok.chat_template).encode("utf-8")
    ).hexdigest()
    runtime_packages = package_versions(
        (
            "torch", "transformers", "trl", "peft", "datasets", "accelerate",
            "wandb", "flash-linear-attention", "causal-conv1d",
        )
    )
    gpu_config: dict[str, object] = {"runtime/cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu_config.update({
            "runtime/gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "runtime/gpu_memory_bytes": int(props.total_memory),
            "runtime/cuda_capability": list(torch.cuda.get_device_capability()),
            "runtime/cuda_version": torch.version.cuda,
        })
    if wandb_run is not None:
        wandb_run.config.update({
            "model/model_type": getattr(model.config, "model_type", None),
            "model/architectures": getattr(model.config, "architectures", None),
            "model/trainable_parameters": trainable_params,
            "model/total_parameters_with_adapter": total_params,
            "model/trainable_fraction": trainable_params / max(total_params, 1),
            "lora/matched_linear_module_count": len(matched_target_names),
            "lora/trainable_parameter_tensor_count": len(trainable_parameter_names),
            "data/corpus_label": args.data.name,
            "data/training_file_sha256": data_hash_fields["training_file_sha256"],
            "data/training_file_sha256s": data_hash_fields["training_file_sha256s"],
            "data/corpus_manifest_sha256": data_hash_fields["corpus_manifest_sha256"],
            # Backward-compatible alias for the ordered file-identity-list hash.
            "data/corpus_sha256": corpus_sha256,
            "data/file_count": len(files),
            "data/file_identities": data_file_identities,
            "data/conversational_rows": len(rows),
            "data/corpus_conversational_rows": corpus_row_count,
            "data/text_only_rows_skipped": n_text_only,
            "data/token_length_stats": data_token_stats,
            "data/batch_padding_projection": batch_padding_stats,
            "prompt/chat_template_sha256": chat_template_sha256,
            "runtime/package_versions": runtime_packages,
            "code/training_source_identities": training_source_identities,
            **gpu_config,
        })

    # --- Train ---
    cfg = SFTConfig(
        output_dir=str(args.out),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        max_length=args.seq_len,
        save_steps=args.save_steps,
        logging_steps=10,
        include_num_input_tokens_seen=True,
        bf16=True,
        # W&B is initialized manually above and receives only an allowlisted
        # scalar set through the callback below.
        report_to="none",
        run_name=os.environ.get("WANDB_RUN_NAME"),
        seed=args.seed,
        # --- assistant-only loss (the masking fix) ---
        # Train ONLY on assistant tokens (the netlist + reasoning); mask the
        # system+user prompt. Requires the chat template's {% generation %} block
        # (SmolLM3's instruct template has it). packing=False because per-example
        # masking is incompatible with sequence packing in TRL.
        assistant_only_loss=True,
        packing=False,
    )
    callbacks = []
    if wandb_run is not None:
        from transformers import TrainerCallback
        from athma_train.tracking import filter_training_metrics

        class PrivacySafeWandbCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if not state.is_world_process_zero:
                    return
                payload = filter_training_metrics(logs or {})
                if not payload:
                    return
                payload["train/global_step"] = int(state.global_step)
                wandb_run.log(payload, step=int(state.global_step))

        callbacks.append(PrivacySafeWandbCallback())

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        callbacks=callbacks,
    )
    batch_topology = training_batch_topology(
        args.bs,
        args.grad_accum,
        int(trainer.accelerator.num_processes),
    )
    if wandb_run is not None:
        wandb_run.config.update({
            "train/world_size": batch_topology["world_size"],
            "train/global_batch_size": batch_topology["global_batch_size"],
        })
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    peak_memory = {}
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        peak_memory = {
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
            "gpu_total_memory_bytes": int(
                torch.cuda.get_device_properties(device_index).total_memory
            ),
        }
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))

    # --- Manifest ---
    from athma_train.experiment_provenance import policy_hash
    adapter_policy_hash = policy_hash(args.base, args.base_revision, [args.out])
    manifest = {
        "condition": condition, "stage": stage,
        "base": args.base, "base_revision": args.base_revision,
        "template_model": args.template_model,
        "template_revision": args.template_revision,
        "base_adapter": args.base_adapter,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": lora_targets,
        "matched_linear_module_count": len(matched_target_names),
        "trainable_parameter_tensor_count": len(trainable_parameter_names),
        "epochs": args.epochs, "lr": args.lr, "bs": args.bs,
        "max_steps": args.max_steps,
        "grad_accum": args.grad_accum, "seq_len": args.seq_len,
        "assistant_only_loss": True, "packing": False, "bf16": True,
        **batch_topology,
        "lr_scheduler_type": args.lr_scheduler_type,
        "optim": args.optim, "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio, "seed": args.seed,
        "trainable_parameters": trainable_params,
        "total_parameters_with_adapter": total_params,
        "data_files": data_file_identities,
        **data_hash_fields,
        "corpus_conversational_rows": corpus_row_count,
        "training_conversational_rows": len(rows),
        "text_only_rows_skipped": n_text_only,
        "data_token_length_stats": data_token_stats,
        "batch_padding_projection": batch_padding_stats,
        "chat_template_sha256": chat_template_sha256,
        "adapter_policy_hash": adapter_policy_hash,
        "package_versions": runtime_packages,
        "training_source_identities": training_source_identities,
        "train_metric_semantics": {
            "num_tokens": "cumulative non-padding input tokens (prompt + assistant)",
            "num_input_tokens_seen": "cumulative padded input-token slots seen",
            "loss": "assistant-label tokens only",
            "mean_token_accuracy": "assistant-label tokens only",
        },
        "train_metrics": train_result.metrics,
        "cuda_memory": peak_memory,
    }
    (args.out / "ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"DONE: adapter saved to {args.out}", file=sys.stderr)
    if wandb_run is not None:
        from athma_train.tracking import filter_training_metrics
        final_metrics = filter_training_metrics(train_result.metrics)
        for key, value in peak_memory.items():
            final_metrics[f"runtime/{key}"] = value
        wandb_run.log(final_metrics)
        wandb_run.summary.update({
            **final_metrics,
            "model/adapter_policy_hash": adapter_policy_hash,
        })
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
