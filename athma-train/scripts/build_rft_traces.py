#!/usr/bin/env python3
"""Build RFT (rejection-sampling / expert-iteration) multi-turn SFT traces.

Pipeline (the cheap unlock proven by closed_loop_probe.py):
  strong expert (claude / codex)  +  simulator-in-the-loop  ->  in-spec trajectory

For each spec we run the closed loop:
    user:      design spec (target FOM windows)
    assistant: reasoning + netlist                 <- expert turn (LOSS)
    user:      ngspice-measured FOMs vs target      <- "tool" observation (no loss)
    assistant: revised netlist                      <- expert turn (LOSS)
    ... until every FOM is inside its window (or we give up)

REJECTION SAMPLING: a trajectory is KEPT only if it ends fully in-spec (verified
by the SAME harness that grades GRPO).  The kept conversation teaches SmolLM3 the
*revision policy* — read the measured gap, adjust component values — which is the
skill single-pass generation provably lacks.

Output: JSONL in the existing SFT schema ({"messages":[...], + metadata}) so it
is drop-in for train_lora_sft.py (assistant_only_loss masks user/system, trains
on every assistant turn => TITO-correct multi-turn loss).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys


def spec_key(spec) -> str:
    """Stable cross-process key (builtin hash() is randomized per process)."""
    h = hashlib.md5(spec["spec_prompt"].encode()).hexdigest()[:12]
    return f"{spec['circuit_id']}:{h}"
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closed_loop_probe import (  # noqa: E402
    extract_netlist, build_feedback, interface_contract, in_window, load_harness,
)

ROOT = Path(__file__).resolve().parent.parent.parent

SYSTEM_PROMPT = (
    "You are an expert analog circuit designer. You design ngspice netlists to meet "
    "measured figure-of-merit (FOM) targets. After each attempt a SPICE simulator "
    "reports the measured FOMs; you read the gap to target and revise the component "
    "values until every FOM lands inside its window. Reason briefly about which values "
    "to change and by how much, then output the netlist in a spice-tagged code fence "
    "ending in .end."
)


_HF = {}  # lazy-loaded {model, tok} for the trained-policy eval


def _load_hf(base: str, adapters: str):
    """Load base + chained LoRA adapters (merged) for the multi-turn EVAL of the
    trained policy. adapters = comma-separated, merged in order (e.g. SFT,RFT)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(base if not adapters else adapters.split(",")[-1])
    if tok.chat_template is None:
        tok.chat_template = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B").chat_template
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
    for a in [x for x in adapters.split(",") if x]:
        m = PeftModel.from_pretrained(m, a); m = m.merge_and_unload()
    m.eval()
    _HF.update(model=m, tok=tok)


def gen_clean(generator: str, prompt: str, messages=None, timeout_s: int = 420) -> str:
    """Single-shot expert call; returns the clean final message text.

    Never raises: a slow/failed call returns "" so one bad turn skips a round
    instead of killing the whole batch.  generator='hf' runs the trained policy
    (multi-turn EVAL) — chat-template on `messages`, decode only new tokens.
    """
    try:
        if generator == "claude":
            p = subprocess.run(["claude", "-p", "--allowedTools", ""],
                               input=prompt, capture_output=True, text=True, timeout=timeout_s)
            return p.stdout or ""
        elif generator == "codex":
            import tempfile, os
            tf = tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False); tf.close()
            try:
                subprocess.run(["codex", "exec", "--sandbox", "read-only",
                                "--output-last-message", tf.name, prompt],
                               capture_output=True, text=True, timeout=timeout_s)
                return open(tf.name).read()
            finally:
                try: os.unlink(tf.name)
                except OSError: pass
        elif generator == "hf":
            import torch
            tok, m = _HF["tok"], _HF["model"]
            enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True)
            ids = enc["input_ids"].to(m.device)
            am = enc.get("attention_mask")
            am = am.to(m.device) if am is not None else None
            with torch.no_grad():
                out = m.generate(ids, attention_mask=am, max_new_tokens=_HF.get("max_new", 2048),
                                 do_sample=False, pad_token_id=tok.eos_token_id)
            return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        raise ValueError(generator)
    except subprocess.TimeoutExpired:
        return ""


def build_user0(spec, contract: bool) -> str:
    """First user turn — deterministic, so it doubles as the resume key."""
    contract_txt = interface_contract(spec) if contract else ""
    return spec["spec_prompt"] + contract_txt + (
        "\n\nReason briefly about component values, then output the netlist wrapped in a "
        "Markdown code fence tagged spice, ending with .end."
    )


def run_trajectory(spec, generator, max_rounds, contract, log):
    """Run the closed loop; return (messages, solved_bool, rounds_used, best_in)."""
    cid = spec["circuit_id"]
    targets = {k: tuple(v) for k, v in spec["target_measurements"].items()}
    harness = load_harness(cid)
    contract_txt = interface_contract(spec) if contract else ""
    user0 = build_user0(spec, contract)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user0}]
    best_in = 0
    n_tot = len(targets)
    for r in range(max_rounds):
        resp = gen_clean(generator, _render(messages), messages=messages)
        netlist = extract_netlist(resp)
        if not netlist:
            # don't store an assistant turn with no netlist; retry same context
            log(f"      round {r}: no netlist parsed; retry")
            continue
        messages.append({"role": "assistant", "content": resp.strip()})
        try:
            measured = harness.measure(netlist, spec=targets, timeout_s=15)
        except Exception as e:  # noqa: BLE001
            measured = {}
            log(f"      round {r}: measure error {str(e)[:60]}")
        n_in = sum(1 for k, (lo, hi) in targets.items() if in_window(measured.get(k), lo, hi))
        best_in = max(best_in, n_in)
        log(f"      round {r}: {n_in}/{n_tot} in-window")
        if n_in == n_tot:
            return messages, True, r + 1, best_in
        # append simulator observation as a user turn (the "tool" message)
        messages.append({"role": "user",
                         "content": build_feedback(netlist, measured, targets, contract_txt)})
    return messages, False, max_rounds, best_in


def _render(messages):
    """Flatten the running conversation into a single prompt for a stateless CLI.

    We pass the whole conversation as text (the expert is stateless per call), which
    mirrors the closed_loop_probe behavior.  This is the DATA-GENERATION expert, not
    the trained policy, so re-rendering here is fine (no gradient)."""
    parts = []
    for m in messages:
        if m["role"] == "system":
            parts.append(m["content"])
        elif m["role"] == "user":
            parts.append("\n[DESIGN REQUEST / SIMULATOR FEEDBACK]\n" + m["content"])
        else:
            parts.append("\n[YOUR PREVIOUS ATTEMPT]\n" + m["content"])
    parts.append("\nNow give your next attempt. Output the netlist in a spice-tagged fence ending in .end.")
    return "\n".join(parts)


def select_specs(measurable, difficulty, per_family, offset=0):
    specs = [json.loads(l) for l in open(ROOT / "data/rlvr_specs_v2_verified/specs.jsonl")]
    byf = defaultdict(list)
    for s in specs:
        if s["circuit_id"] in measurable and s["difficulty"] == difficulty:
            byf[s["circuit_id"]].append(s)
    sel = []
    for cid in measurable:
        sel.extend(byf[cid][offset:offset + per_family])  # offset => HELD-OUT eval split
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", choices=["claude", "codex", "hf"], default="claude")
    ap.add_argument("--hf-base", default="HuggingFaceTB/SmolLM3-3B-Base",
                    help="(generator=hf) base model for the trained-policy multi-turn EVAL")
    ap.add_argument("--hf-adapters", default="",
                    help="(generator=hf) comma-separated LoRA adapters merged in order, e.g. "
                         "checkpoints/sft_masked,checkpoints/rft_sft_v0")
    ap.add_argument("--difficulty", default="medium", choices=["loose", "medium", "tight"])
    ap.add_argument("--per-family", type=int, default=3)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip first N specs/family (use a HELD-OUT split for eval, e.g. --offset 6)")
    ap.add_argument("--hf-max-new-tokens", type=int, default=1536)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--contract", action="store_true")
    ap.add_argument("--measurable", default="bridge-ina,h-bridge,integrator,sallen-key,schmitt,tia")
    ap.add_argument("--out", default="final_artifacts/datasets/rft_v0/rft_traces.jsonl")
    args = ap.parse_args()

    measurable = args.measurable.split(",")
    specs = select_specs(measurable, args.difficulty, args.per_family, offset=args.offset)
    print(f"RFT trace gen: {len(specs)} specs ({args.difficulty}, {args.per_family}/family), "
          f"expert={args.generator}, max_rounds={args.max_rounds}\n", flush=True)
    if args.generator == "hf":
        print(f"loading trained policy: base={args.hf_base} adapters={args.hf_adapters}", flush=True)
        _load_hf(args.hf_base, args.hf_adapters)
        _HF["max_new"] = args.hf_max_new_tokens

    def log(m): print(m, flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resume: key on the (deterministic) first user turn, recomputed identically
    # for stored traces and candidates — robust to Python's randomized hash().
    def _ukey(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    done_keys = set()
    if out_path.exists():
        for l in open(out_path):
            try:
                r = json.loads(l)
                done_keys.add(_ukey(r["messages"][1]["content"]))  # messages[1] = user spec turn
            except Exception:  # noqa: BLE001
                pass
    kept_n, attempted = len(done_keys), []
    out_f = open(out_path, "a")  # APPEND: every kept trace is flushed immediately
    for i, s in enumerate(specs):
        key = _ukey(build_user0(s, args.contract))
        if key in done_keys:
            log(f"[{i+1}/{len(specs)}] {s['circuit_id']} -- already done, skip"); continue
        log(f"[{i+1}/{len(specs)}] {s['circuit_id']} ({s['difficulty']})")
        msgs, solved, rounds, best = run_trajectory(s, args.generator, args.max_rounds,
                                                    args.contract, log)
        attempted.append({"cid": s["circuit_id"], "solved": solved, "rounds": rounds,
                          "best_in": best, "n_tot": len(s["target_measurements"])})
        if solved:
            out_f.write(json.dumps({
                "messages": msgs,
                "circuit_id": s["circuit_id"],
                "spec_hash": hash(s["spec_prompt"]),
                "tier": s["tier"],
                "difficulty": s["difficulty"],
                "rounds": rounds,
                "n_assistant_turns": sum(1 for m in msgs if m["role"] == "assistant"),
                "source_teacher": args.generator,
                "format_version": "rft_v0_multiturn",
            }) + "\n")
            out_f.flush()
            kept_n += 1
        log(f"    -> {'KEPT' if solved else 'dropped'} (solved={solved}, rounds={rounds}, best {best}/{len(s['target_measurements'])})\n")
    out_f.close()
    n_solved = sum(a["solved"] for a in attempted)
    total_kept = sum(1 for _ in open(out_path)) if out_path.exists() else 0
    log("=" * 60)
    log(f"attempted this run: {len(attempted)}  | solved this run: {n_solved}  "
        f"({100*n_solved/max(1,len(attempted)):.0f}% yield)")
    log(f"total kept traces in file: {total_kept}  -> {out_path}")
    json.dump({"attempted": attempted, "n_kept_total": total_kept, "generator": args.generator,
               "difficulty": args.difficulty},
              open(out_path.with_suffix(".report.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
