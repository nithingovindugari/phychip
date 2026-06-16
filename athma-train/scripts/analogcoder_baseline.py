#!/usr/bin/env python3
"""AnalogCoder baseline runner — pass@1 on locked SmolLM3-3B-Base.

Anchors every future "+X pp over base" claim. Cost: ~$0.30, ~30-60 min on
1×H100 PCIe. Target metric: AnalogCoder 23-task pass@1 (ngspice converges
+ no syntax error). Published baselines: GPT-4 ~37%, Llama-3.1-8B ~21%,
SmolLM3-3B-Base unknown — that's what we're measuring.

Usage:
    python scripts/analogcoder_baseline.py \\
        --base HuggingFaceTB/SmolLM3-3B-Base \\
        --output-dir eval_results/analogcoder_baseline_smollm3
"""

from __future__ import annotations

import argparse
import csv
import re
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from athma_train.tracking import (
    STAGE_WANDB_PROJECTS,
    assert_wandb_token_present,
    configure_wandb_env,
    wandb_run_name,
)

ANALOGCODER_TSV_URL = (
    "https://raw.githubusercontent.com/laiyao1/AnalogCoder/main/problem_set.tsv"
)


def fetch_problem_set(cache_dir: Path) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = cache_dir / "analogcoder_problem_set.tsv"
    if not tsv_path.exists():
        urlretrieve(ANALOGCODER_TSV_URL, tsv_path)
    with tsv_path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return list(reader)


def build_prompt(row: dict) -> str:
    return (
        "You are an analog circuit designer. Write a complete SPICE netlist "
        "for the following circuit using standard ngspice syntax.\n\n"
        f"Circuit: {row['Circuit']}\n"
        f"Input port: {row['Input']}\n"
        f"Output port: {row['Output']}\n"
        f"Circuit type: {row['Type']}\n"
        f"Subcircuit name: {row['Submodule Name']}\n\n"
        "Output only the SPICE netlist. Include device models, biasing, "
        "and supply nodes. End the netlist with .end.\n\n"
        "```spice\n"
    )


def extract_netlist(generation: str) -> str:
    """Extract the netlist from a completion.

    The prompt ends with an OPEN ```spice fence, so the completion is the
    netlist body followed by a closing fence — take what precedes the first
    fence. Fallback (prompt-echo mode): take what's between the first pair.
    """
    text = generation
    if "```" in text:
        parts = text.split("```")
        # continuation mode: netlist precedes the first fence
        if parts[0].strip() and re.search(
            r"^\s*[RCLMQVIXDE]\w*\s+\S+\s+\S+", parts[0],
            re.MULTILINE | re.IGNORECASE,
        ):
            text = parts[0]
        elif len(parts) >= 2:
            text = parts[1]
            if text.startswith("spice\n") or text.startswith("ngspice\n"):
                text = text.split("\n", 1)[1]
    if ".end" in text.lower():
        idx = text.lower().rfind(".end")
        text = text[: idx + len(".end")]
    return text.strip()


def run_ngspice(netlist: str, timeout_s: int = 30) -> tuple[bool, str]:
    """Returns (passed, log). STRICT gate — requires:
      1. Process exited 0 (not 1, which is warnings; not negative which is signal)
      2. Netlist contains at least one device line (R/C/L/M/Q/D/V/I/X)
      3. Netlist contains a simulation analysis line (.op/.tran/.dc/.ac)
      4. ngspice did NOT print "no simulations run"
      5. ngspice did NOT print "aborted"
      6. stderr doesn't contain SEGFAULT-like signals
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cir", delete=False, dir="/tmp"
    ) as f:
        f.write(netlist)
        path = f.name
    try:
        # Pre-check: netlist must have actual content
        non_comment_lines = [
            ln.strip() for ln in netlist.split("\n")
            if ln.strip() and not ln.strip().startswith("*")
        ]
        device_chars = ("r", "c", "l", "m", "q", "d", "v", "i", "x", "j")
        has_device = any(
            ln[0].lower() in device_chars
            for ln in non_comment_lines
            if ln and not ln.startswith(".")
        )
        analysis_dirs = (".op", ".tran", ".dc", ".ac", ".noise", ".disto")
        has_analysis = any(
            ln.lower().startswith(d) for ln in non_comment_lines for d in analysis_dirs
        )
        # analyses inside .control blocks count too (op / ac dec ... / tran ...)
        if not has_analysis and ".control" in netlist.lower():
            ctl_dirs = ("op", "tran", "dc", "ac", "noise", "disto")
            has_analysis = any(
                ln.lower().split()[0] in ctl_dirs
                for ln in non_comment_lines
                if ln and not ln.startswith(".")
            )
        if not has_device:
            return False, f"no device lines in netlist"
        if not has_analysis:
            return False, f"no .op/.tran/.dc/.ac analysis directive"

        proc = subprocess.run(
            ["ngspice", "-b", "-n", path],
            capture_output=True, text=True, timeout=timeout_s,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        out_lower = out.lower()

        if proc.returncode != 0:
            return False, f"exit={proc.returncode}\n{out}"
        if "run simulation(s) aborted" in out_lower:
            return False, f"aborted\n{out}"
        if "no simulations run" in out_lower:
            return False, f"no_simulations_run\n{out}"
        # Hard failures
        for bad in ["singular matrix", "segmentation fault",
                    "fatal error", "no such device", "doanalyses: ",
                    "timestep too small", "convergence problem"]:
            if bad in out_lower:
                return False, f"{bad}\n{out}"
        return True, out
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        Path(path).unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="HuggingFaceTB/SmolLM3-3B-Base")
    p.add_argument("--adapter", default=None, help="Optional PEFT LoRA adapter path/repo to apply on top of --base")
    p.add_argument("--pre-adapters", default=None,
                   help="Comma-separated list of LoRA adapters to merge into base BEFORE loading --adapter. Required for chained eval (e.g. SFT+DPO+GRPO).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "phychip")
    p.add_argument("--max-new-tokens", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy (pass@1); >0 = sampling (use --k for pass@k)")
    p.add_argument("--k", type=int, default=1, help="Generations per problem (pass@k)")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--ngspice-timeout", type=int, default=30)
    p.add_argument("--chat", action="store_true",
                   help="Apply chat template (fair eval for off-the-shelf INSTRUCT models).")
    p.add_argument("--no-think", action="store_true",
                   help="enable_thinking=False to suppress hybrid-reasoning blocks (with --chat).")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # W&B setup
    wandb_run = None
    if not args.no_wandb:
        assert_wandb_token_present()
        configure_wandb_env("phase0")
        import wandb
        base_short = args.base.split("/")[-1]
        wandb_run = wandb.init(
            project=STAGE_WANDB_PROJECTS["phase0"],
            name=wandb_run_name(
                stage="phase0", base=base_short,
                descriptor=f"analogcoder-pass@{args.k}-T{args.temperature}",
            ),
            tags=["phase0", base_short, "framework:transformers",
                  "benchmark:analogcoder", f"pass@{args.k}"],
            group="analogcoder-baseline",
            config={
                "base": args.base, "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature, "k": args.k,
                "ngspice_timeout_s": args.ngspice_timeout,
            },
        )

    print(f"=== AnalogCoder baseline: {args.base} ===", file=sys.stderr)
    problems = fetch_problem_set(args.cache_dir)
    print(f"  {len(problems)} problems", file=sys.stderr)

    print("\n=== Loading model ===", file=sys.stderr)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Pick device — cuda if available, else mps (Mac), else cpu
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    )
    if args.pre_adapters:
        from peft import PeftModel
        for pre in args.pre_adapters.split(","):
            pre = pre.strip()
            if not pre: continue
            print(f"  pre-adapter merge: {pre}", file=sys.stderr)
            model = PeftModel.from_pretrained(model, pre)
            model = model.merge_and_unload()
    if args.adapter:
        from peft import PeftModel
        print(f"  loading adapter: {args.adapter}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)",
          file=sys.stderr)

    do_sample = args.temperature > 0
    results = []
    n_pass = 0

    # Per-problem serial generation. Slow (~30s/problem on H100) but
    # robust — transformers batched generate hangs on SmolLM3 architecture.
    # vLLM was attempted but hit transformers 5.x incompatibility.
    # Time budget: 23 problems × ~30s = ~12 min total.
    print(f"\n=== Serial generation: {len(problems)} problems × k={args.k} ===",
          file=sys.stderr, flush=True)
    t_total = time.time()
    for i, prob in enumerate(problems, 1):
        if args.chat:
            # Fair instruct eval: chat template, instruction without the ```spice prime.
            instr = build_prompt(prob).rsplit("```spice", 1)[0].rstrip()
            ct_kw = {"enable_thinking": False} if args.no_think else {}
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": instr}], tokenize=False,
                    add_generation_prompt=True, **ct_kw)
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": instr}], tokenize=False,
                    add_generation_prompt=True)
        else:
            prompt = build_prompt(prob)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        gens = []
        gens_passed = 0
        for k in range(args.k):
            t1 = time.time()
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature if do_sample else 1.0,
                    top_p=0.95 if do_sample else 1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_text = tokenizer.decode(
                out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
            )
            netlist = extract_netlist(gen_text)
            passed, log = run_ngspice(netlist, timeout_s=args.ngspice_timeout)
            gens_passed += int(passed)
            gens.append({
                "k": k, "passed": passed, "netlist": netlist,
                "ngspice_log_tail": log[-300:],
                "gen_seconds": round(time.time() - t1, 2),
            })
        any_passed = gens_passed > 0
        n_pass += int(any_passed)
        results.append({
            "id": prob["Id"], "level": prob["Level"], "type": prob["Type"],
            "submodule": prob["Submodule Name"],
            "k": args.k, "n_passed": gens_passed,
            "any_passed": any_passed, "generations": gens,
        })
        status = "PASS" if any_passed else "FAIL"
        elapsed = time.time() - t_total
        rate = i / elapsed * 60  # problems per minute
        eta_min = (len(problems) - i) / max(rate, 0.001)
        print(f"  [{i:2d}/{len(problems)}] {status} {prob['Submodule Name']} "
              f"({gens_passed}/{args.k} passed) "
              f"[rate={rate:.1f}/min ETA={eta_min:.1f}min]",
              file=sys.stderr, flush=True)
        if wandb_run:
            wandb_run.log({
                "problem_idx": i, "running_pass_rate": n_pass / i,
                f"problem/{prob['Submodule Name']}/passed": int(any_passed),
            })

        # Save partial results every problem in case of crash
        partial_path = args.output_dir / "partial_results.json"
        partial_path.write_text(json.dumps({
            "n_done": i, "n_passed": n_pass,
            "running_pass_rate": n_pass / i, "results": results,
        }, indent=2))

    pass_rate = n_pass / len(problems)
    print(f"\n=== AnalogCoder pass@{args.k}: {n_pass}/{len(problems)} = "
          f"{pass_rate:.1%} ===", file=sys.stderr)

    summary = {
        "base": args.base, "n_problems": len(problems), "k": args.k,
        "temperature": args.temperature, "n_passed": n_pass,
        f"pass@{args.k}": round(pass_rate, 4),
        "by_level": {},
    }
    by_level: dict[str, list[bool]] = {}
    for r in results:
        by_level.setdefault(r["level"], []).append(r["any_passed"])
    for lvl, lst in by_level.items():
        summary["by_level"][lvl] = {
            "n": len(lst), "passed": sum(lst),
            "pass_rate": round(sum(lst) / len(lst), 4),
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {summary_path}", file=sys.stderr)
    detail_path = args.output_dir / "details.json"
    detail_path.write_text(json.dumps(results, indent=2))
    print(f"  wrote {detail_path}", file=sys.stderr)

    if wandb_run:
        wandb_run.log({f"final/pass@{args.k}": pass_rate, "final/n_passed": n_pass})
        wandb_run.summary[f"pass@{args.k}"] = pass_rate
        wandb_run.summary["n_passed"] = n_pass
        wandb_run.finish()


if __name__ == "__main__":
    main()
