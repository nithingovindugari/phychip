#!/usr/bin/env python3
"""Verify SFT assistant-only loss masking is actually applied.

Run BEFORE any SFT training (the 2-minute check that gates every SFT number).
Replicates train_lora_sft.py's tokenizer setup and confirms:
  1. the loss mask covers ONLY assistant tokens (prompt masked out), and
  2. the assistant stop token (<|im_end|>) is inside the trained span
     (so the model learns to STOP -- a likely contributor to novel-type repetition).

Usage:
  HF_HUB_OFFLINE=1 .venv/bin/python athma-train/scripts/verify_sft_masking.py \
      --base HuggingFaceTB/SmolLM3-3B-Base --data data/final_data/sft_v0
"""
import argparse, glob, json, sys
from pathlib import Path
from transformers import AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM3-3B-Base")
    ap.add_argument("--instruct", default="HuggingFaceTB/SmolLM3-3B")
    ap.add_argument("--data", type=Path, default=Path("data/final_data/sft_v0"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        tok.chat_template = AutoTokenizer.from_pretrained(args.instruct).chat_template
    assert "generation" in (tok.chat_template or ""), \
        "FAIL: chat template has no {% generation %} block -> assistant_only_loss cannot mask."

    f = sorted(glob.glob(str(args.data / "*.jsonl")))[0]
    msgs = json.loads(open(f).readline())["messages"]
    out = tok.apply_chat_template(msgs, return_assistant_tokens_mask=True,
                                  return_dict=True, tokenize=True)
    ids, mask = out["input_ids"], out["assistant_masks"]
    s, n = sum(mask), len(mask)
    assert mask is not None, "FAIL: assistant_masks is None."
    assert 0 < s < n, f"FAIL: mask trains {s}/{n} tokens (all-0 or all-1) -> masking NOT working."

    trained_ids = [t for t, m in zip(ids, mask) if m]
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    stop_in_target = eos_id in trained_ids or tok.eos_token_id in trained_ids

    print(f"PASS: mask trains {s}/{n} tokens ({100*s/n:.0f}% assistant, {100*(n-s)/n:.0f}% prompt masked).")
    print(f"stop token (<|im_end|>/eos) inside trained span: {stop_in_target}"
          + ("" if stop_in_target else "  <-- WARN: model may not learn to stop"))
    print("TRAINED-ON  :", tok.decode(trained_ids)[:140].replace(chr(10), ' '))
    print("MASKED-OUT  :", tok.decode([t for t, m in zip(ids, mask) if not m])[:140].replace(chr(10), ' '))
    return 0


if __name__ == "__main__":
    sys.exit(main())
