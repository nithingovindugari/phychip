"""Batched sampling from a base model (+ optional LoRA adapter) with HF transformers.

python gen.py --base ./base_models/Qwen3.5-4B --adapter DIR --tasks tasks.jsonl --out out.jsonl \
    --k 8 --temperature 0.8 [--greedy] [--max-new 1536] [--bs 16]
tasks rows: {"id":..., "messages":[...]}; out rows: {"id","sample","completion","n_tokens","finished"}
"""
import argparse, json, os, time
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", default=None)
ap.add_argument("--tasks", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=1)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--greedy", action="store_true")
ap.add_argument("--max-new", type=int, default=1536)
ap.add_argument("--bs", type=int, default=64, help="sequences per generate() call")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed)
tok = AutoTokenizer.from_pretrained(args.adapter if args.adapter and os.path.exists(os.path.join(args.adapter, "tokenizer.json")) else args.base, trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained(args.base, trust_remote_code=True)
# Qwen3.5 is a vision-language checkpoint; the project's adapters target `language_model.*` modules
loader = AutoModelForImageTextToText if getattr(cfg, "vision_config", None) is not None else AutoModelForCausalLM
model = loader.from_pretrained(args.base, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
model.eval()

eos_ids = {tok.eos_token_id}
for name in ("<|im_end|>", "<|endoftext|>"):
    tid = tok.convert_tokens_to_ids(name)
    if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
        eos_ids.add(tid)
eos_ids = sorted(i for i in eos_ids if i is not None)

tasks = [json.loads(l) for l in open(args.tasks)]
if args.limit:
    tasks = tasks[: args.limit]
done = set()
if os.path.exists(args.out):
    done = {(r["id"], r["sample"]) for r in map(json.loads, open(args.out))}
k = 1 if args.greedy else args.k
jobs = [(t, s) for t in tasks for s in range(k) if (t["id"], s) not in done]


def render(msgs):
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


prompts = {t["id"]: render(t["messages"]) for t in tasks}
jobs.sort(key=lambda j: len(prompts[j[0]["id"]]))
start = time.time(); ntok = 0
with open(args.out, "a") as sink:
    for i in range(0, len(jobs), args.bs):
        chunk = jobs[i : i + args.bs]
        enc = tok([prompts[t["id"]] for t, _ in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(0)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new, do_sample=not args.greedy,
                temperature=None if args.greedy else args.temperature,
                top_p=None if args.greedy else args.top_p,
                eos_token_id=eos_ids, pad_token_id=tok.pad_token_id,
            )
        new = out[:, enc["input_ids"].shape[1]:]
        for (t, s), ids in zip(chunk, new):
            ids = ids.tolist()
            finished = any(e in ids for e in eos_ids)
            cut = min([ids.index(e) for e in eos_ids if e in ids] or [len(ids)])
            text = tok.decode(ids[:cut], skip_special_tokens=True)
            ntok += cut
            sink.write(json.dumps({"id": t["id"], "sample": s, "completion": text, "n_tokens": cut, "finished": finished}) + "\n")
        sink.flush()
        el = time.time() - start
        print(f"{i+len(chunk)}/{len(jobs)} {el:.0f}s {ntok/el:.0f} tok/s", flush=True)
print("GEN_DONE", len(jobs), round(time.time() - start))
