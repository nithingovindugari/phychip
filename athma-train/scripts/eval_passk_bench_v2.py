#!/usr/bin/env python3
"""pass@k evaluation with bootstrap CIs for phy-chip-bench-v2.

Main-track statistical rigor: instead of a single greedy pass@1, draw K samples
per problem at temperature T, grade each with the EXACT same gate + harness used
by eval_on_bench_v1.py, then report:
  - pass@1  (mean correctness over all K·N samples)
  - pass@k  (unbiased Chen et al. estimator: 1 - C(n-c,k)/C(n,k), averaged over problems)
  - 95% bootstrap CIs over the N problems (B resamples)
per level (L1/L2/L3) and overall.

Usage:
  PHYCHIP_REQUIRE_NGSPICE=46 athma-train/.venv/bin/python \
    athma-train/scripts/eval_passk_bench_v2.py \
    --base HuggingFaceTB/SmolLM3-3B-Base \
    [--pre-adapters ...] [--adapter ...] \
    --bench eval_sets/phychip_bench_v2/bench_v2.jsonl \
    --k 5 --temperature 0.8 --output eval_results/passk_<tag>.json
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str((ROOT / "athma-train").resolve()))
sys.path.insert(0, str((ROOT / "athma-train" / "scripts").resolve()))

import eval_on_bench_v1 as ev  # reuse extract_netlist, grade_netlist, harness_spec_check, etc.


def passk_estimator(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021): 1 - C(n-c,k)/C(n,k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--pre-adapters", default=None)
    p.add_argument("--bench", type=Path, default=Path("eval_sets/phychip_bench_v2/bench_v2.jsonl"))
    p.add_argument("--k", type=int, default=5, help="samples per problem")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    rows = [json.loads(l) for l in a.bench.open()]
    problems = [r for r in rows if r.get("verify_passed")]
    print(f"=== pass@k eval ({len(problems)} problems, K={a.k}, T={a.temperature}) ===", flush=True)
    print(f"  base={a.base} pre={a.pre_adapters} adapter={a.adapter}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(a.seed)

    device = ev.pick_device()
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dtype, trust_remote_code=True).to(device)
    if a.pre_adapters:
        from peft import PeftModel
        for pre in a.pre_adapters.split(","):
            pre = pre.strip()
            if pre:
                model = PeftModel.from_pretrained(model, pre); model = model.merge_and_unload()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter); model = model.merge_and_unload()
    model.eval()

    # build K copies of each prompt
    tasks = []  # (prob_idx, prompt)
    for i, prob in enumerate(problems):
        for _ in range(a.k):
            tasks.append((i, ev.PROMPT_TEMPLATE.format(spec=prob["prompt"])))

    print(f"\n=== generating {len(tasks)} samples (batch {a.batch_size}) ===", flush=True)
    completions = [None] * len(tasks)
    t1 = time.time()
    for b in range(0, len(tasks), a.batch_size):
        chunk = tasks[b:b + a.batch_size]
        prompts = [c[1] for c in chunk]
        inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=a.max_new_tokens,
                do_sample=True, temperature=a.temperature, top_p=a.top_p,
                pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for j, g in enumerate(gen):
            completions[b + j] = g
        done = b + len(chunk); el = time.time() - t1
        print(f"  [{done}/{len(tasks)}] {el:.0f}s ETA {(len(tasks)-done)/(done/el):.0f}s", flush=True)

    print("\n=== grading ===", flush=True)
    # correctness[i] = list of bool over K samples for problem i
    correct = [[] for _ in problems]
    for (i, _), comp in zip(tasks, completions):
        prob = problems[i]
        netlist = ev.extract_netlist(comp or "")
        gate_ok, _ = ev.grade_netlist(netlist, timeout_s=20)
        if not gate_ok:
            ok = False
        elif prob["level"] == 1:
            ok = True
        else:
            ok, _ = ev.harness_spec_check(prob, ev.normalize_netlist(netlist))
        correct[i].append(bool(ok))

    # metrics per level + overall, with bootstrap CIs over problems
    import random
    rng = random.Random(a.seed)
    levels = sorted({prob["level"] for prob in problems})

    def metrics(idxs):
        n = a.k
        p1 = [sum(correct[i]) / n for i in idxs]
        pk = [passk_estimator(n, sum(correct[i]), a.k) for i in idxs]
        return p1, pk

    def boot_ci(vals, idxs):
        if not idxs:
            return [0, 0]
        means = []
        for _ in range(a.bootstrap):
            samp = [vals[rng.randrange(len(vals))] for _ in idxs]
            means.append(sum(samp) / len(samp))
        means.sort()
        lo = means[int(0.025 * len(means))]; hi = means[int(0.975 * len(means))]
        return [round(lo, 4), round(hi, 4)]

    report = {"base": a.base, "pre_adapters": a.pre_adapters, "adapter": a.adapter,
              "k": a.k, "temperature": a.temperature, "n_problems": len(problems),
              "device": device, "by_level": {}, "overall": {}}

    all_idx = list(range(len(problems)))
    for scope_name, idxs in [("overall", all_idx)] + [(f"L{l}", [i for i in all_idx if problems[i]["level"] == l]) for l in levels]:
        if not idxs:
            continue
        p1, pk = metrics(idxs)
        mean_p1 = sum(p1) / len(p1)
        mean_pk = sum(pk) / len(pk)
        entry = {
            "n": len(idxs),
            f"pass@1": round(mean_p1, 4),
            f"pass@1_ci95": boot_ci(p1, idxs),
            f"pass@{a.k}": round(mean_pk, 4),
            f"pass@{a.k}_ci95": boot_ci(pk, idxs),
            "any_solved": sum(1 for i in idxs if any(correct[i])),
        }
        if scope_name == "overall":
            report["overall"] = entry
        else:
            report["by_level"][scope_name] = entry
        print(f"  {scope_name:8s} n={len(idxs):2d}  pass@1={entry['pass@1']:.3f} "
              f"CI{entry['pass@1_ci95']}  pass@{a.k}={entry[f'pass@{a.k}']:.3f} "
              f"CI{entry[f'pass@{a.k}_ci95']}  any_solved={entry['any_solved']}/{len(idxs)}", flush=True)

    report["per_problem"] = [
        {"id": problems[i]["id"], "level": problems[i]["level"],
         "correct": sum(correct[i]), "k": a.k} for i in all_idx]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.output, "w"), indent=1)
    print(f"\nwrote {a.output}", flush=True)


if __name__ == "__main__":
    main()
