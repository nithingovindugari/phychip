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
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM3-3B-Base")
    ap.add_argument("--base-adapter", default=None,
                    help="Optional: load this LoRA adapter first (stacks on base) before SFT")
    ap.add_argument("--data", type=Path, default=Path("data/final_data/sft_v0"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    condition = os.environ.get("ABLATION_CONDITION", "?")
    stage = os.environ.get("ABLATION_STAGE", "sft")
    args.out.mkdir(parents=True, exist_ok=True)

    # --- Tracking ---
    if not args.no_wandb:
        from athma_train.tracking import (
            configure_wandb_env, wandb_init_kwargs, assert_wandb_token_present,
        )
        assert_wandb_token_present()
        configure_wandb_env("ablation")
        import wandb
        wandb.init(**wandb_init_kwargs(
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
            config={**vars(args), "condition": condition},
        ))

    # --- Load model + tokenizer ---
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    from trl import SFTConfig, SFTTrainer

    print(f"Loading base: {args.base}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Base models often lack chat_template. Pull it from the Instruct variant.
    if tok.chat_template is None:
        try:
            from transformers import AutoTokenizer as _ATK
            inst = _ATK.from_pretrained("HuggingFaceTB/SmolLM3-3B")
            tok.chat_template = inst.chat_template
            print(f"  loaded chat_template from SmolLM3-3B (instruct)", file=sys.stderr)
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
    # DDP-safe device map: under torchrun, each rank loads to its assigned GPU.
    # Single-process launch falls back to device_map="auto".
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = {"": local_rank} if local_rank >= 0 else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map=device_map,
        attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
    )

    if args.base_adapter:
        print(f"Loading base adapter (CPT): {args.base_adapter}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, args.base_adapter, is_trainable=False)
        # Merge so we have a single base on which SFT-LoRA stacks
        model = model.merge_and_unload()

    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --- Load data ---
    # Manual JSON load avoids datasets-library schema-unification crash
    # when JSONL files have heterogeneous fields (e.g. hardcase rows have
    # 'hardcase_mode' but original SFT rows don't).
    from datasets import Dataset
    files = sorted(glob.glob(str(args.data / "*.jsonl")))
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
    ds = Dataset.from_list(rows)
    print(f"  loaded {len(ds):,} conversational examples"
          f" ({n_text_only} text-only rows skipped — not assistant-maskable)", file=sys.stderr)

    # --- Train ---
    cfg = SFTConfig(
        output_dir=str(args.out),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        max_length=args.seq_len,
        save_steps=args.save_steps,
        logging_steps=10,
        bf16=True,
        report_to="wandb" if not args.no_wandb else "none",
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
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))

    # --- Manifest ---
    (args.out / "ablation_manifest.json").write_text(json.dumps({
        "condition": condition, "stage": stage,
        "base": args.base, "base_adapter": args.base_adapter,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "epochs": args.epochs, "lr": args.lr, "bs": args.bs,
    }, indent=2))
    print(f"DONE: adapter saved to {args.out}", file=sys.stderr)
    if not args.no_wandb:
        import wandb; wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
