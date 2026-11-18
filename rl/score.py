"""Score completions with the canonical scorer (phy_chip_env.score@v11) in a process pool.

python score.py --tasks tasks.jsonl --gen out.jsonl --out scored.jsonl [--procs 20]
Success is always ScoreResult.all_specs_met.
"""
import argparse, collections, json, multiprocessing as mp, re, time
from phy_chip_env import SCORER_SEMANTICS, score
from phychip_v12 import score_v12

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", required=True)
ap.add_argument("--gen", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--procs", type=int, default=20)
ap.add_argument("--timeout", type=float, default=25)
ap.add_argument("--scorer", default="v11", choices=["v11", "v12"], help="v12 = v11 behind the sanitize-v12 layer")
ap.add_argument("--name", default=None)
ap.add_argument("--ledger", default=None)
ap.add_argument("--note", default="")
args = ap.parse_args()
tasks = {t["id"]: t for t in map(json.loads, open(args.tasks))}
rows = [json.loads(l) for l in open(args.gen)]


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)


def one(row):
    t = tasks[row["id"]]
    try:
        fn = score_v12 if args.scorer == "v12" else score
        r = fn(t["circuit_id"], t["target_measurements"], strip_think(row["completion"]), timeout_s=args.timeout).to_event()
    except Exception as e:  # noqa: BLE001
        r = {"error": repr(e)[:200], "all_specs_met": False, "reward": 0.0, "simulated": False, "anticheat_blocked": False}
    return {"id": row["id"], "sample": row["sample"], "circuit_id": t["circuit_id"], "n_tokens": row.get("n_tokens"), "finished": row.get("finished"), **r}


if __name__ == "__main__":
    start = time.time()
    with mp.Pool(args.procs) as pool:
        scored = list(pool.imap(one, rows, chunksize=2))
    with open(args.out, "w") as sink:
        for s in scored:
            sink.write(json.dumps(s) + "\n")
    by_task = collections.defaultdict(list)
    for s in scored:
        by_task[s["id"]].append(s)
    n = len(by_task)
    rate = sum(sum(x["all_specs_met"] for x in v) / len(v) for v in by_task.values()) / n
    any_ = sum(any(x["all_specs_met"] for x in v) for v in by_task.values())
    first = sum(v[0]["all_specs_met"] for v in by_task.values())
    fam = collections.defaultdict(lambda: [0, 0])
    for v in by_task.values():
        f = fam[v[0]["circuit_id"]]; f[1] += 1; f[0] += sum(x["all_specs_met"] for x in v) / len(v)
    summary = {
        "scorer": SCORER_SEMANTICS + ("+sanitize-v12" if args.scorer == "v12" else ""), "tasks": n, "samples": len(scored),
        "mean_pass_rate": round(rate, 4), "pass_any": any_, "pass_first": first,
        "simulated": round(sum(s["simulated"] for s in scored) / len(scored), 4),
        "anticheat_blocked": round(sum(s["anticheat_blocked"] for s in scored) / len(scored), 4),
        "unfinished": round(sum(1 for s in scored if s.get("finished") is False) / len(scored), 4),
        "mean_reward": round(sum(s["reward"] for s in scored) / len(scored), 4),
        "family_pass_rate": {k: round(a / b, 3) for k, (a, b) in sorted(fam.items())},
        "seconds": round(time.time() - start),
    }
    json.dump(summary, open(args.out.replace(".jsonl", "") + ".summary.json", "w"), indent=1)
    if args.ledger:
        with open(args.ledger, "a") as led:
            led.write(json.dumps({"name": args.name, "tasks_file": args.tasks, "gen_file": args.gen, "note": args.note,
                                  "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **summary}) + "\n")
    print(json.dumps(summary))
