"""Closed-loop evaluation: design -> ngspice measures -> the model sees the measurement -> revise.

Each task gets up to --turns verifier calls. Turn 1 is the usual one-shot answer (greedy). A failed
attempt is answered with the verifier's measured values (feedback.py) and the model revises.
A budget-matched blind control answers the same task --turns times independently (T=0.8) with no
feedback, so "used the measurement" can be told apart from "got more tries".

python loop_eval.py --model DIR --tasks tasks.jsonl --out DIR --turns 3 --name NAME [--ledger F]
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import grpo  # scoring worker with measurements
from feedback import feedback_text, repair_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--name", default=None)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    grpo.NEED_MEASURE = True
    pool = mp.get_context("fork").Pool(args.procs)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=args.gpu_frac, seed=0,
              trust_remote_code=True)
    tok = llm.get_tokenizer()

    def render(msgs):
        try:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    stop = ["<|im_end|>", "<|endoftext|>"]
    tasks = [json.loads(line) for line in open(args.tasks)]
    start = time.time()

    def score(pairs):
        return pool.map(grpo._score, [(t["circuit_id"], t["target_measurements"], text) for t, text in pairs], chunksize=2)

    # ---- closed loop
    history = {t["id"]: [] for t in tasks}          # list of (completion, feedback)
    solved_at = {}
    trace = {t["id"]: [] for t in tasks}
    active = list(tasks)
    for turn in range(1, args.turns + 1):
        if not active:
            break
        prompts = [render(repair_messages(t["messages"], history[t["id"]])) for t in active]
        outs = llm.generate(prompts, SamplingParams(n=1, temperature=0.0, max_tokens=args.max_new, stop=stop))
        texts = [o.outputs[0].text for o in outs]
        events = score(list(zip(active, texts)))
        nxt = []
        for t, text, ev in zip(active, texts, events):
            trace[t["id"]].append({"turn": turn, "completion": text, "all_specs_met": ev["all_specs_met"],
                                   "simulated": ev.get("simulated"), "blocked": ev.get("anticheat_blocked"),
                                   "measured": ev.get("measured")})
            if ev["all_specs_met"]:
                solved_at[t["id"]] = turn
            else:
                history[t["id"]].append((text, feedback_text(ev, t["target_measurements"])))
                nxt.append(t)
        active = nxt
    # ---- blind control: the same number of verifier calls, no feedback
    prompts = [render(t["messages"][:2]) for t in tasks]
    outs = llm.generate(prompts, SamplingParams(n=args.turns, temperature=0.8, top_p=0.95, max_tokens=args.max_new, stop=stop, seed=0))
    pairs = [(t, c.text) for t, o in zip(tasks, outs) for c in o.outputs]
    events = score(pairs)
    blind = {}
    for i, t in enumerate(tasks):
        evs = events[i * args.turns:(i + 1) * args.turns]
        first = next((j + 1 for j, e in enumerate(evs) if e["all_specs_met"]), None)
        blind[t["id"]] = first
    n = len(tasks)
    summary = {"name": args.name, "tasks": n, "turns": args.turns,
               "closed_loop_solved_by_turn": [sum(1 for v in solved_at.values() if v <= k) for k in range(1, args.turns + 1)],
               "blind_resample_solved_by_try": [sum(1 for v in blind.values() if v is not None and v <= k) for k in range(1, args.turns + 1)],
               "seconds": round(time.time() - start), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(args.out, "loop_trace.jsonl"), "w") as sink:
        for t in tasks:
            sink.write(json.dumps({"id": t["id"], "circuit_id": t["circuit_id"], "solved_at": solved_at.get(t["id"]),
                                   "blind_solved_at": blind[t["id"]], "turns": trace[t["id"]]}) + "\n")
    json.dump(summary, open(os.path.join(args.out, "loop_summary.json"), "w"), indent=1)
    if args.ledger:
        with open(args.ledger, "a") as led:
            led.write(json.dumps({"kind": "closed_loop", **summary}) + "\n")
    print("LOOP", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
