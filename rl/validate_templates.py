"""Law-check equation-carrying templates with the simulator, then emit training rows.

A template is a netlist whose component values are brace expressions of target parameters
`T_<figure>`. For a grid of target values the template is instantiated, sanitised, admitted and
measured by the family harness. A grid point "tracks" when the canonical scorer accepts the
instantiated netlist for the task window centred on that target. Templates that track on most of
the grid are kept, with the range where they track.

python validate_templates.py TEMPLATE_DIR OUT_PREFIX [--grid 9] [--rows-per-template 40] [--procs 3]
writes OUT_PREFIX.report.jsonl (per template) and OUT_PREFIX.sft.jsonl (training rows)
"""
import argparse
import collections
import glob
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.environ.get("PHYCHIP_REPO", "./phy-chip"), "poc_experiments"))
import generate_rlvr_candidates_v7 as gen  # noqa: E402
from common import messages, spec_prompt  # noqa: E402
from phychip_v12 import sanitize_netlist, score_v12  # noqa: E402

ROOTS = os.environ.get("HL_ROOTS", "./work/hl_roots.jsonl")


def family_ranges():
    seen = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(ROOTS):
        r = json.loads(line)
        if r.get("stage") == "ok":
            for k in gen.FAMILY_SPECS[r["circuit_id"]][2]:
                seen[r["circuit_id"]][k].append(r["measured"][k])
    return {f: {k: (min(v), max(v)) for k, v in d.items()} for f, d in seen.items()}


def grid(lo, hi, n):
    if lo > 0 and hi / lo > 20:
        return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def instantiate(text, values):
    out = text
    for key, value in values.items():
        out, count = re.subn(rf"(?im)^(\s*\.param\s+T_{re.escape(key)}\s*=\s*)\S+", lambda m: m.group(1) + f"{value:.6g}", out)
        if count == 0:
            return None
    return out


def check(job):
    family, text, values = job
    deck = instantiate(text, values)
    if deck is None:
        return {"ok": False, "why": "no T_ param"}
    targets = {k: gen.interval(k, v, 0.30) for k, v in values.items()}
    try:
        res = score_v12(family, targets, f"```spice\n{deck}\n```", timeout_s=25)
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "why": repr(error)[:80]}
    return {"ok": bool(res.all_specs_met), "blocked": bool(res.anticheat_blocked), "simulated": bool(res.simulated),
            "values": values, "targets": targets, "deck": sanitize_netlist(deck, family) if res.all_specs_met else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("template_dir")
    ap.add_argument("out_prefix")
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--rows-per-template", type=int, default=40)
    ap.add_argument("--min-track", type=float, default=0.5)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    ranges = family_ranges()
    templates = []
    if args.template_dir.endswith(".jsonl"):
        named = [(r["name"], r["text"]) for r in map(json.loads, open(args.template_dir))]
    else:
        named = [(os.path.basename(p)[:-4], open(p).read())
                 for p in sorted(glob.glob(os.path.join(args.template_dir, "**", "*.cir"), recursive=True))]
    for name, text in named:
        family = re.sub(r"_T\d+$", "", name)
        if family in gen.FAMILY_SPECS:
            templates.append((name, family, text))
    jobs, index = [], []
    for t_idx, (name, family, text) in enumerate(templates):
        keys = gen.FAMILY_SPECS[family][2]
        axes = [grid(*ranges[family][k], args.grid) for k in keys]
        points = [dict(zip(keys, combo)) for combo in zip(*axes)] if len(keys) > 1 else [{keys[0]: v} for v in axes[0]]
        if family == "schmitt":  # ordered thresholds: sweep the high threshold, keep a fixed hysteresis fraction
            points = [{"vth_high_v": v, "vth_low_v": v * 0.55} for v in axes[0]]
        for p in points:
            jobs.append((family, text, p)); index.append(t_idx)
    with mp.Pool(args.procs) as pool:
        results = pool.map(check, jobs, chunksize=2)
    per = collections.defaultdict(list)
    for t_idx, res in zip(index, results):
        per[t_idx].append(res)
    kept = []
    with open(args.out_prefix + ".report.jsonl", "w") as rep:
        for t_idx, (name, family, text) in enumerate(templates):
            rs = per[t_idx]
            track = sum(r["ok"] for r in rs) / max(len(rs), 1)
            good = [r["values"] for r in rs if r["ok"]]
            rep.write(json.dumps({"template": name, "family": family, "track": track, "points": len(rs),
                                  "blocked": sum(bool(r.get("blocked")) for r in rs),
                                  "not_simulated": sum(not r.get("simulated", False) for r in rs),
                                  "good_values": good}) + "\n")
            if track >= args.min_track and good:
                kept.append((name, family, text, good))
    # training rows: fresh targets inside the tracked range, verified one by one
    jobs, meta = [], []
    for name, family, text, good in kept:
        keys = gen.FAMILY_SPECS[family][2]
        lo = {k: min(g[k] for g in good) for k in keys}
        hi = {k: max(g[k] for g in good) for k in keys}
        for i in range(args.rows_per_template):
            u = rng.random()
            vals = {}
            for k in keys:
                a, b = lo[k], hi[k]
                vals[k] = a * (b / a) ** u if a > 0 and b / a > 20 else a + (b - a) * u
            jobs.append((family, text, vals)); meta.append((name, family, i))
    with mp.Pool(args.procs) as pool:
        results = pool.map(check, jobs, chunksize=2)
    rows = 0
    with open(args.out_prefix + ".sft.jsonl", "w") as sink:
        for (name, family, i), res in zip(meta, results):
            if not res["ok"]:
                continue
            inputs, outputs, _ = gen.FAMILY_SPECS[family]
            prompt = spec_prompt(family, inputs, outputs, res["targets"])
            sink.write(json.dumps({"id": f"eq-{name}-{i}", "circuit_id": family, "kind": "equation", "src": f"template:{name}",
                                   "target_measurements": res["targets"], "measured": res["values"],
                                   "messages": messages(prompt, inputs, outputs, "grpo") + [{"role": "assistant", "content": f"```spice\n{res['deck']}```"}]}) + "\n")
            rows += 1
    fam = collections.Counter(f for _, f, _, _ in kept)
    print(json.dumps({"templates": len(templates), "kept": len(kept), "rows": rows, "kept_by_family": dict(sorted(fam.items()))}))


if __name__ == "__main__":
    main()
