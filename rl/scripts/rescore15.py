"""Scorer v15 = v14 checks on the v15 env (J14 structural-admission fixes). v15 only relaxes admission, so only rows
that v14 marked anticheat-blocked are re-simulated. usage: rescore15.py <rescore14 summary.json>"""
import json, os, sys
from multiprocessing import Pool
os.environ["PHYCHIP_SCORER"] = "v14"
sys.path.insert(0, "./work")


def run(job):
    from phychip_v12 import score_v12
    fam, targets, text = job
    try:
        return bool(score_v12(fam, targets, text).all_specs_met)
    except Exception:  # noqa: BLE001
        return False


def main():
    s = json.load(open(sys.argv[1])); name = s["name"]
    rows = [json.loads(l) for l in open(sys.argv[1].replace(".summary.json", ".jsonl"))]
    gen_path = s["gen"] if os.path.isabs(s["gen"]) else os.path.join(os.path.dirname(os.path.dirname(sys.argv[1])), s["gen"])
    if not os.path.exists(gen_path):
        gen_path = os.path.join("./codexdata", s["gen"])
    gen = [json.loads(l) for l in open(gen_path)]
    tasks_path = sys.argv[2]
    tasks = {t["id"]: t for t in map(json.loads, open(tasks_path))}
    todo = [i for i, r in enumerate(rows) if r["anticheat_blocked"] and not r["all_specs_met"]]
    with Pool(3) as p:
        res = p.map(run, [(tasks[gen[i]["id"]]["circuit_id"], tasks[gen[i]["id"]]["target_measurements"], gen[i].get("completion", "")) for i in todo])
    gained = 0
    for i, ok in zip(todo, res):
        if ok:
            rows[i]["all_specs_met"] = True; gained += 1
    by = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)
    out = {"scorer": "v15", "name": name, "tasks": len(by), "rescored_rows": len(todo), "gained_rows": gained,
           "pass_first": sum(1 for v in by.values() if sorted(v, key=lambda r: r["sample"])[0]["all_specs_met"]),
           "pass_any": sum(1 for v in by.values() if any(r["all_specs_met"] for r in v)),
           "mean_pass_rate": round(sum(r["all_specs_met"] for r in rows) / len(rows), 4), "v14_pass_first": s["pass_first"]}
    json.dump(out, open(f"./rescore15/out/{name}.summary.json", "w"), indent=1)
    with open(f"./rescore15/out/{name}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
