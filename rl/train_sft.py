"""Assistant-only LoRA SFT with exact train/inference prompt parity.

Single GPU:  python train_sft.py --base B --data D.jsonl --out OUT
Multi GPU:   torchrun --nproc_per_node 4 train_sft.py ... (HF Trainer runs DDP)

Every row is {"messages": [system, user, assistant]}. The prompt is rendered with
the same chat template call the evaluator uses (generation prompt, thinking off);
loss is applied only to the assistant netlist plus the end-of-turn token.
"""
import argparse
import json
import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--init-adapter", default=None)
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--alpha", type=int, default=128)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--grad-accum", type=int, default=1)
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max-steps", type=int, default=-1)
ap.add_argument("--chat-template", default=None, help="jinja file for base models that ship without a template")
ap.add_argument("--grad-ckpt", action="store_true")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"
if args.chat_template:
    tok.chat_template = open(args.chat_template).read()
EOT = "<|im_end|>" if tok.convert_tokens_to_ids("<|im_end|>") not in (None, tok.unk_token_id) else tok.eos_token


def render(msgs):
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


class Rows(torch.utils.data.Dataset):
    def __init__(self, path):
        self.items = []
        dropped = 0
        for line in open(path):
            row = json.loads(line)
            msgs = row["messages"]
            prompt = tok(render(msgs[:-1]), add_special_tokens=False)["input_ids"]
            target = tok(msgs[-1]["content"] + EOT, add_special_tokens=False)["input_ids"]
            if len(prompt) + len(target) > args.max_len:
                dropped += 1
                continue
            self.items.append((prompt, target))
            if args.limit and len(self.items) >= args.limit:
                break
        self.lengths = [len(p) + len(t) for p, t in self.items]
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"rows {len(self.items)} dropped_overlength {dropped} tokens {sum(self.lengths)} "
                  f"target_tokens {sum(len(t) for _, t in self.items)}", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        prompt, target = self.items[index]
        return {"input_ids": prompt + target, "labels": [-100] * len(prompt) + target}


def collate(batch):
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


data = Rows(args.data)
import random as _r
_r.Random(args.seed).shuffle(data.items)
_chunk = args.bs * 64
data.items = [x for i in range(0, len(data.items), _chunk) for x in sorted(data.items[i:i + _chunk], key=lambda pt: len(pt[0]) + len(pt[1]))]
cfg = AutoConfig.from_pretrained(args.base, trust_remote_code=True)
is_vl = getattr(cfg, "vision_config", None) is not None
loader = AutoModelForImageTextToText if is_vl else AutoModelForCausalLM
model = loader.from_pretrained(args.base, dtype=torch.bfloat16, trust_remote_code=True)
if args.init_adapter:
    model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
else:
    if is_vl:  # same module set as the project's earlier Qwen3.5 adapters
        targets = (r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|"
                   r"in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)$")
    elif str(getattr(cfg, "model_type", "")).startswith("lfm"):
        targets = "all-linear"  # LFM blocks name their projections differently; adapt every linear layer except the head
    else:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
                                             target_modules=targets, task_type="CAUSAL_LM"))
if args.grad_ckpt:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
if int(os.environ.get("RANK", "0")) == 0:
    model.print_trainable_parameters()

targs = TrainingArguments(
    output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps,
    per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr, lr_scheduler_type="cosine", weight_decay=0.0,
    warmup_steps=max(1, int(0.03 * len(data) * max(args.epochs, 1) / (args.bs * args.grad_accum))),
    bf16=True, logging_steps=10, save_strategy="no", report_to=[], seed=args.seed,
    dataloader_num_workers=2, ddp_find_unused_parameters=False,
    optim="adamw_torch_fused", max_grad_norm=1.0,
)
trainer = Trainer(model=model, args=targs, train_dataset=data, data_collator=collate)
result = trainer.train()
if trainer.is_world_process_zero():
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    json.dump({"args": vars(args), "metrics": result.metrics, "rows": len(data),
               "log": trainer.state.log_history}, open(os.path.join(args.out, "train_summary.json"), "w"), indent=1)
    print("TRAIN_DONE", result.metrics, flush=True)
