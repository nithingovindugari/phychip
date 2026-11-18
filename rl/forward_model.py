"""Forward-model task: predict what the harness will measure for a netlist (the inverse of design).

build : python forward_model.py build OUT.jsonl SFT_ROWS.jsonl [--max-rows N]
        turns verified design rows (netlist + measured value) into prediction rows.
tasks : python forward_model.py tasks OUT.jsonl HL.jsonl [--max-rows N]
        prediction tasks with known answers from hindsight rows (for evaluation).
eval  : python forward_model.py eval --model DIR --tasks TASKS.jsonl --out O.json
        accuracy = prediction inside the +/-30% (or +/-3 dB) window around the simulator's value.
pick  : python forward_model.py pick --model DIR --design-tasks T.jsonl --gen K8.jsonl --scored K8_SCORED.jsonl --out O.json
        simulator-free selection: among K sampled designs, choose the one whose PREDICTED measurement
        sits best inside the requested window; report how often that choice truly passes.
"""
import argparse
import json
import os
import random
import re
import sys

FWD_SYS = ("You are an expert analog IC designer. Read the ngspice netlist and predict what the measurement "
           "harness will report for the listed figures. Answer with one line per figure in the form name = value "
           "and nothing else.")


def fwd_messages(family, keys, netlist):
    user = (f"Circuit family: {family}\nFigures to predict: {', '.join(keys)}\n\n```spice\n{netlist.strip()}\n```")
    return [{"role": "system", "content": FWD_SYS}, {"role": "user", "content": user}]


def fmt_answer(values):
    return "\n".join(f"{k} = {v:.4g}" for k, v in values.items())


def parse_answer(text):
    out = {}
    for m in re.finditer(r"([A-Za-z_0-9]+)\s*=\s*([-+0-9.eE]+)", text or ""):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return out


def netlist_of(row):
    m = re.search(r"```spice\s*(.*?)```", row["messages"][-1]["content"], re.S)
    return m.group(1) if m else None


def in_window(key, pred, true):
    if key.endswith("_db"):
        return abs(pred - true) <= 3.0
    if key.startswith("vth_"):
        return abs(pred - true) <= max(abs(true), 1.0) * 0.30
    return abs(pred - true) <= abs(true) * 0.30


def cmd_build(args):
    rng = random.Random(0)
    rows = [json.loads(l) for l in open(args.src)]
    rng.shuffle(rows)
    n = 0
    with open(args.out, "w") as sink:
        for r in rows:
            net = netlist_of(r)
            if not net or r.get("kind") in ("repair", "equation") or not r.get("measured"):
                continue
            keys = list(r["measured"])
            sink.write(json.dumps({"id": "fwd-" + r["id"], "circuit_id": r["circuit_id"], "kind": "forward",
                                   "messages": fwd_messages(r["circuit_id"], keys, net)
                                   + [{"role": "assistant", "content": fmt_answer(r["measured"])}]}) + "\n")
            n += 1
            if args.max_rows and n >= args.max_rows:
                break
    print("forward rows", n)


def cmd_tasks(args):
    sys.path.insert(0, os.path.join(os.environ.get("PHYCHIP_REPO", "./phy-chip"), "poc_experiments"))
    import generate_rlvr_candidates_v7 as gen
    rng = random.Random(1)
    rows = [json.loads(l) for l in open(args.src)]
    rows = [r for r in rows if r.get("stage") == "ok"]
    rng.shuffle(rows)
    with open(args.out, "w") as sink:
        for r in rows[: args.max_rows or len(rows)]:
            keys = list(gen.FAMILY_SPECS[r["circuit_id"]][2])
            sink.write(json.dumps({"id": "fwdtask-" + r["id"], "circuit_id": r["circuit_id"],
                                   "true": {k: r["measured"][k] for k in keys},
                                   "messages": fwd_messages(r["circuit_id"], keys, r["netlist"])}) + "\n")
    print("forward tasks", min(len(rows), args.max_rows or len(rows)))


def load_llm(model):
    from vllm import LLM
    llm = LLM(model=model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, seed=0, trust_remote_code=True)
    tok = llm.get_tokenizer()

    def render(msgs):
        try:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return llm, render


def predict(llm, render, msg_lists):
    from vllm import SamplingParams
    outs = llm.generate([render(m) for m in msg_lists],
                        SamplingParams(n=1, temperature=0.0, max_tokens=64, stop=["<|im_end|>", "<|endoftext|>"]))
    return [parse_answer(o.outputs[0].text) for o in outs]


def cmd_eval(args):
    llm, render = load_llm(args.model)
    tasks = [json.loads(l) for l in open(args.tasks)]
    preds = predict(llm, render, [t["messages"] for t in tasks])
    fam = {}
    ok = 0
    for t, p in zip(tasks, preds):
        good = all(k in p and in_window(k, p[k], v) for k, v in t["true"].items())
        ok += good
        f = fam.setdefault(t["circuit_id"], [0, 0]); f[0] += good; f[1] += 1
    summary = {"tasks": len(tasks), "within_window": ok, "accuracy": round(ok / len(tasks), 4),
               "family_accuracy": {k: round(a / b, 3) for k, (a, b) in sorted(fam.items())}}
    json.dump(summary, open(args.out, "w"), indent=1)
    print("FWD", json.dumps(summary))


def cmd_pick(args):
    sys.path.insert(0, os.path.join(os.environ.get("PHYCHIP_REPO", "./phy-chip"), "poc_experiments"))
    llm, render = load_llm(args.model)
    tasks = {t["id"]: t for t in map(json.loads, open(args.design_tasks))}
    gens = [json.loads(l) for l in open(args.gen)]
    scored = {(s["id"], s["sample"]): s for s in map(json.loads, open(args.scored))}
    msgs, meta = [], []
    for g in gens:
        t = tasks[g["id"]]
        m = re.search(r"```spice\s*(.*?)```", g["completion"] or "", re.S)
        if not m:
            continue
        msgs.append(fwd_messages(t["circuit_id"], list(t["target_measurements"]), m.group(1)))
        meta.append((g["id"], g["sample"]))
    preds = predict(llm, render, msgs)
    by = {}
    for (tid, sample), p in zip(meta, preds):
        t = tasks[tid]
        badness = 0.0
        for k, (lo, hi) in t["target_measurements"].items():
            lo, hi = min(lo, hi), max(lo, hi)
            if k not in p:
                badness += 10
            else:
                mid, half = (lo + hi) / 2, max((hi - lo) / 2, 1e-12)
                badness += abs(p[k] - mid) / half
        by.setdefault(tid, []).append((badness, sample))
    n = len(tasks)
    picked = sum(bool(scored[(tid, min(v)[1])]["all_specs_met"]) for tid, v in by.items())
    first = sum(bool(scored.get((tid, 0), {}).get("all_specs_met")) for tid in tasks)
    anyk = sum(any(scored.get((tid, s), {}).get("all_specs_met") for s in range(64)) for tid in tasks)
    # agreement of the predicted verdict with the simulator's verdict
    agree = tot = 0
    for tid, v in by.items():
        for badness, sample in v:
            n_spec = len(tasks[tid]["target_measurements"])
            tot += 1
            agree += (badness <= n_spec) == bool(scored[(tid, sample)]["all_specs_met"])
    summary = {"tasks": n, "solved_first_sample": first, "solved_forward_model_pick": picked, "solved_oracle_any": anyk,
               "verdict_agreement": round(agree / max(tot, 1), 4)}
    json.dump(summary, open(args.out, "w"), indent=1)
    print("PICK", json.dumps(summary))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("out"); b.add_argument("src"); b.add_argument("--max-rows", type=int, default=0)
    t = sub.add_parser("tasks"); t.add_argument("out"); t.add_argument("src"); t.add_argument("--max-rows", type=int, default=0)
    e = sub.add_parser("eval"); e.add_argument("--model", required=True); e.add_argument("--tasks", required=True); e.add_argument("--out", required=True)
    p = sub.add_parser("pick"); p.add_argument("--model", required=True); p.add_argument("--design-tasks", required=True)
    p.add_argument("--gen", required=True); p.add_argument("--scored", required=True); p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"build": cmd_build, "tasks": cmd_tasks, "eval": cmd_eval, "pick": cmd_pick}[args.cmd](args)
