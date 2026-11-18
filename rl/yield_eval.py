"""Tolerance yield: does a passing design still pass when every passive component drifts?

For each answer that meets its spec, draw N variants with every R/C/L value multiplied by an
independent log-uniform factor in [1/(1+tol), 1+tol] (default tol = 10%), re-score each variant
against the same task, and report the fraction that still pass. A design that sits in the middle
of its window survives; one that scrapes the edge does not.

python yield_eval.py --tasks T.jsonl --gen G.jsonl --out O.json [--n 8] [--tol 0.10] [--procs 3]
"""
import argparse
import json
import math
import multiprocessing as mp
import random
import re

from amplify import fmt, parse
from phychip_v12 import score_v12

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", required=True)
ap.add_argument("--gen", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--tol", type=float, default=0.10)
ap.add_argument("--procs", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
tasks = {t["id"]: t for t in map(json.loads, open(args.tasks))}
SPAN = math.log(1 + args.tol)


def drift(netlist, rng):
    out, touched, in_control = [], 0, False
    for line in netlist.split("\n"):
        low = line.strip().lower()
        if low.startswith(".control"):
            in_control = True
        if low.startswith(".endc"):
            in_control = False
        if not in_control and low[:1] in ("r", "c", "l"):
            parts = line.split()
            if len(parts) >= 4:
                value = parse(parts[3])
                if value is not None and value > 0:
                    parts[3] = fmt(value * math.exp(rng.uniform(-SPAN, SPAN)))
                    line = " ".join(parts)
                    touched += 1
        out.append(line)
    return "\n".join(out), touched


def one(row):
    t = tasks[row["id"]]
    text = re.sub(r"<think>.*?</think>", "", row["completion"] or "", flags=re.S)
    try:
        base = score_v12(t["circuit_id"], t["target_measurements"], text, timeout_s=25)
    except Exception:  # noqa: BLE001
        return {"id": row["id"], "circuit_id": t["circuit_id"], "pass": False}
    if not base.all_specs_met:
        return {"id": row["id"], "circuit_id": t["circuit_id"], "pass": False}
    match = re.search(r"```spice\s*(.*?)```", text, re.S | re.I)
    rng = random.Random(f"{args.seed}-{row['id']}")
    ok, n, touched = 0, 0, 0
    for _ in range(args.n):
        variant, touched = drift(match.group(1), rng)
        if not touched:
            break
        n += 1
        try:
            ok += bool(score_v12(t["circuit_id"], t["target_measurements"], f"```spice\n{variant}\n```", timeout_s=25).all_specs_met)
        except Exception:  # noqa: BLE001
            pass
    return {"id": row["id"], "circuit_id": t["circuit_id"], "pass": True, "draws": n, "survive": ok, "passives_drifted": touched}


if __name__ == "__main__":
    rows = [r for r in map(json.loads, open(args.gen)) if r.get("sample", 0) == 0]
    with mp.Pool(args.procs) as pool:
        res = pool.map(one, rows, chunksize=1)
    passed = [r for r in res if r["pass"]]
    with_draws = [r for r in passed if r.get("draws")]
    summary = {
        "tasks": len(res), "solved": len(passed), "tolerance": args.tol, "draws_per_design": args.n,
        "mean_yield": round(sum(r["survive"] / r["draws"] for r in with_draws) / max(len(with_draws), 1), 4),
        "robust_solved_yield_ge_0.75": sum(1 for r in with_draws if r["survive"] / r["draws"] >= 0.75),
        "fully_robust": sum(1 for r in with_draws if r["survive"] == r["draws"]),
        "no_literal_passives": len(passed) - len(with_draws),
    }
    json.dump({"summary": summary, "rows": res}, open(args.out, "w"), indent=1)
    print("YIELD", json.dumps(summary))
