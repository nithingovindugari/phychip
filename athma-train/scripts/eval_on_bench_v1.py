#!/usr/bin/env python3
"""Eval a model (base or base+adapter) against phy-chip-bench-v1.

For each verified bench problem:
  - feed the natural-language `prompt` to the student model with a thin
    "wrap your netlist in ```spice ... ``` fences" instruction
  - extract the generated netlist from the completion
  - grade L1 with strict ngspice gate; L2/L3 with strict gate + spec_check

Auto-picks device: cuda → mps (Apple Silicon) → cpu.

Usage:
    # base SmolLM3-3B-Base (control)
    .venv/bin/python athma-train/scripts/eval_on_bench_v1.py \\
        --base HuggingFaceTB/SmolLM3-3B-Base \\
        --output-dir eval_results/bench_v1_smollm3_base

    # PhyChip-3B-v0 (treatment)
    .venv/bin/python athma-train/scripts/eval_on_bench_v1.py \\
        --base HuggingFaceTB/SmolLM3-3B-Base \\
        --adapter NithinReddyG/PhyChip-3B-v0 \\
        --output-dir eval_results/bench_v1_phychip_v0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from athma_train.spice_gate import grade_netlist, normalize_netlist  # noqa: E402

PROMPT_TEMPLATE = """{spec}

Output your complete ngspice netlist between ```spice and ``` fences.
The netlist must include device lines, an analysis directive
(.op or .tran or .ac inside a .control/.endc block), and end with .end.

```spice
"""


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_netlist(completion: str) -> str:
    """Output begins inside the ```spice fence — close at ``` or .end."""
    s = completion
    if "```" in s:
        s = s.split("```")[0]
    if ".end" in s.lower():
        idx = s.lower().rfind(".end")
        s = s[: idx + len(".end")]
    return s.strip()


import importlib.util as _ilu
_HARNESS_DIR = Path(__file__).resolve().parent.parent / "athma_train" / "harnesses"
_HCACHE: dict = {}


def _load_harness(circuit_id: str):
    if circuit_id in _HCACHE:
        return _HCACHE[circuit_id]
    p = _HARNESS_DIR / f"{circuit_id}.py"
    mod = None
    if p.exists():
        try:
            spec = _ilu.spec_from_file_location(f"h_{circuit_id}", p)
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception:
            mod = None
    _HCACHE[circuit_id] = mod
    return mod


def _circuit_id_from_bench_id(bench_id: str):
    m = re.match(r"tier\d+-(.+?)-l[23]", bench_id or "")
    return m.group(1) if m else None


def _within(mv, rv, tol=0.30) -> bool:
    """Model metric mv reproduces reference metric rv within tolerance.
    Sign must agree; magnitude within tol (relative), with an absolute floor."""
    if mv is None or rv is None:
        return False
    try:
        mv = float(mv); rv = float(rv)
    except (TypeError, ValueError):
        return False
    if (mv < 0) != (rv < 0) and abs(rv) > 1e-9:
        return False
    return abs(mv - rv) <= tol * max(abs(rv), 1e-6) + 1e-9


def harness_spec_check(prob: dict, netlist: str, timeout_s: int = 25) -> tuple[bool, str]:
    """FIX : grade L2/L3 by MEASURING the model's circuit with the same
    harness used in training (topology-agnostic, via the prompt-mandated ports),
    and require it to reproduce the verified reference's behavior within tolerance.
    Replaces the broken label-grep grader that failed all correct circuits.
    """
    cid = _circuit_id_from_bench_id(prob.get("id", ""))
    if not cid:
        return False, "no circuit_id"
    h = _load_harness(cid)
    if h is None or not hasattr(h, "measure"):
        return False, f"no harness:{cid}"
    ref_nl = prob.get("reference_netlist", "")
    try:
        ref = h.measure(normalize_netlist(ref_nl), None, timeout_s)
        mod = h.measure(netlist, None, timeout_s)
    except Exception as e:
        return False, f"measure err: {str(e)[:40]}"
    ref_metrics = {k: v for k, v in (ref or {}).items() if v is not None}
    if not ref_metrics:
        return False, f"ref unmeasurable:{cid}"
    ok = 0
    for k, rv in ref_metrics.items():
        if _within(mod.get(k), rv):
            ok += 1
    # pass iff the model reproduces ALL of the reference's measurable metrics
    passed = ok == len(ref_metrics)
    return passed, f"harness {cid}: {ok}/{len(ref_metrics)} metrics match ref"


def run_spec_check(spec_check_code: str, netlist: str, timeout_s: int = 20) -> bool:
    """Re-run ngspice to capture stdout, then call the bench's spec_check fn."""
    if not spec_check_code or "def check" not in spec_check_code:
        return True  # L1 / no spec → already passed by gate
    try:
        ns: dict = {}
        exec(spec_check_code, ns)
        check_fn = ns["check"]
    except Exception:
        return False
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cir", delete=False, dir="/tmp") as f:
        f.write(netlist)
        path = f.name
    try:
        proc = subprocess.run(
            ["ngspice", "-b", "-n", path],
            capture_output=True, text=True, timeout=timeout_s,
        )
        stdout = proc.stdout or ""
    finally:
        Path(path).unlink(missing_ok=True)
    try:
        return bool(check_fn(stdout))
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--pre-adapters", default=None,
                   help="Comma-separated list of LoRA adapters to merge into the "
                        "base before loading --adapter. Required when the eval "
                        "adapter was trained on top of a chain (e.g. SFT+DPO+GRPO).")
    p.add_argument("--bench", type=Path,
                   default=Path("eval_sets/bench_v1/bench_v1.jsonl"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="HF generate repetition_penalty (1.0 = off). 1.2-1.3 "
                        "breaks greedy degenerate-repetition loops.")
    p.add_argument("--no-repeat-ngram-size", type=int, default=0,
                   help="HF generate no_repeat_ngram_size (0 = off).")
    p.add_argument("--chat", action="store_true",
                   help="Apply the model's chat template (for off-the-shelf INSTRUCT "
                        "baselines — fair eval vs raw-completion base models).")
    p.add_argument("--no-think", action="store_true",
                   help="Append /no_think to suppress hybrid-reasoning blocks that eat "
                        "the token budget before the netlist (use with --chat).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Mac MPS: 2-4 safe. CUDA: 8-16.")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in args.bench.open()]
    problems = [r for r in rows if r.get("verify_passed")]
    print(f"=== eval on bench v1 ===", flush=True)
    print(f"  base:      {args.base}", flush=True)
    print(f"  adapter:   {args.adapter or '(none — base only)'}", flush=True)
    print(f"  bench:     {args.bench} ({len(problems)} verified problems)", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device()
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    print(f"  device:    {device} ({dtype})", flush=True)

    print("\n=== loading model ===", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=dtype, trust_remote_code=True,
    ).to(device)
    if args.pre_adapters:
        from peft import PeftModel
        for pre in args.pre_adapters.split(","):
            pre = pre.strip()
            if not pre: continue
            print(f"  pre-adapter merge: {pre}", flush=True)
            model = PeftModel.from_pretrained(model, pre)
            model = model.merge_and_unload()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        print(f"  adapter merged into base", flush=True)
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    print(f"\n=== generating in batches of {args.batch_size} ===", flush=True)
    all_completions: list[str] = []
    t1 = time.time()
    for i in range(0, len(problems), args.batch_size):
        batch = problems[i : i + args.batch_size]
        if args.chat:
            # Fair eval for off-the-shelf INSTRUCT models: use their chat template.
            instr = ("{spec}\n\nDesign the circuit and output your complete ngspice "
                     "netlist between ```spice and ``` fences. Include device lines, an "
                     "analysis directive (.op/.tran/.ac inside .control/.endc), and .end.")
            # enable_thinking=False prefills an empty <think></think> so hybrid models
            # (SmolLM3/Qwen3) answer directly instead of spending the budget reasoning.
            ct_kw = {"enable_thinking": False} if args.no_think else {}
            def _ct(spec):
                try:
                    return tokenizer.apply_chat_template(
                        [{"role": "user", "content": instr.format(spec=spec)}],
                        tokenize=False, add_generation_prompt=True, **ct_kw)
                except TypeError:
                    return tokenizer.apply_chat_template(
                        [{"role": "user", "content": instr.format(spec=spec)}],
                        tokenize=False, add_generation_prompt=True)
            prompts = [_ct(p["prompt"]) for p in batch]
        else:
            prompts = [PROMPT_TEMPLATE.format(spec=p["prompt"]) for p in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=2048).to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else 1.0,
                top_p=0.95 if args.temperature > 0 else 1.0,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = out[:, inputs.input_ids.shape[1] :]
        completions = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        all_completions.extend(completions)
        elapsed = time.time() - t1
        n_done = len(all_completions)
        rate = n_done / elapsed if elapsed > 0 else 0
        eta = (len(problems) - n_done) / rate if rate > 0 else 0
        print(f"  [{n_done}/{len(problems)}] {elapsed:.0f}s elapsed, "
              f"{rate*60:.1f} prob/min, ETA {eta:.0f}s", flush=True)

    gen_seconds = time.time() - t1
    print(f"\ngeneration done in {gen_seconds:.0f}s "
          f"({gen_seconds/len(problems):.1f}s/prompt)", flush=True)

    print("\n=== grading ===", flush=True)
    results = []
    n_pass = 0
    for prob, completion in zip(problems, all_completions):
        netlist = extract_netlist(completion)
        gate_ok, gate_log = grade_netlist(netlist, timeout_s=20)
        if not gate_ok:
            passed = False
            why = f"gate: {gate_log[:100]}"
        elif prob["level"] == 1:
            passed = True
            why = "L1 gate ok"
        elif os.environ.get("PHYCHIP_L2L3_GRADER", "harness") == "legacy":
            spec_ok = run_spec_check(
                prob.get("spec_check_python", ""),
                normalize_netlist(netlist),
            )
            passed = spec_ok
            why = "L2/L3 spec ok (legacy)" if spec_ok else "L2/L3 spec FAIL (legacy)"
        else:
            spec_ok, why = harness_spec_check(prob, normalize_netlist(netlist))
            passed = spec_ok
        n_pass += int(passed)
        results.append({
            "id": prob["id"], "tier": prob["tier"], "level": prob["level"],
            "passed": passed, "why": why,
            "completion_head": completion[:600],
            "extracted_netlist_head": netlist[:600],
        })
        flag = "PASS" if passed else "FAIL"
        print(f"  {flag}  {prob['id']:30s}  ({why})", flush=True)

    pass_rate = n_pass / len(problems) if problems else 0
    print(f"\n=== pass@1: {n_pass}/{len(problems)} = {pass_rate:.1%} ===",
          flush=True)

    by_level = {}
    by_tier = {}
    for r in results:
        by_level.setdefault(r["level"], []).append(r["passed"])
        by_tier.setdefault(r["tier"], []).append(r["passed"])
    print("\nby level:")
    for lvl in sorted(by_level):
        lst = by_level[lvl]
        print(f"  L{lvl}: {sum(lst):2d}/{len(lst):2d} = {sum(lst)/len(lst):.0%}")
    print("by tier:")
    for tier in sorted(by_tier):
        lst = by_tier[tier]
        print(f"  T{tier}: {sum(lst):2d}/{len(lst):2d} = {sum(lst)/len(lst):.0%}")

    summary = {
        "model_base": args.base,
        "adapter": args.adapter,
        "device": device,
        "n_problems": len(problems),
        "n_passed": n_pass,
        "pass@1": round(pass_rate, 4),
        "generation_seconds": round(gen_seconds, 1),
        "by_level": {lvl: {"n": len(by_level[lvl]),
                           "passed": sum(by_level[lvl]),
                           "rate": round(sum(by_level[lvl]) / len(by_level[lvl]), 4)}
                     for lvl in sorted(by_level)},
        "by_tier": {tier: {"n": len(by_tier[tier]),
                           "passed": sum(by_tier[tier]),
                           "rate": round(sum(by_tier[tier]) / len(by_tier[tier]), 4)}
                    for tier in sorted(by_tier)},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "details.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.output_dir}/summary.json + details.json", flush=True)


if __name__ == "__main__":
    main()
