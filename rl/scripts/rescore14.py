"""Replay saved generations under scorer v14 (= v13 + precharge and H/F checks; see work/phychip_v12.py v14_check).

v14 can only turn a v13 pass into a fail, and only for boost, buck and tia. So when a v13-scored file from the same
generations exists, only those rows are re-simulated; otherwise every row is scored.
usage: rescore14.py NAME TASKS GEN [V13_SCORED]  -> out/NAME.summary.json, out/NAME.jsonl
"""
import json, os, sys
from multiprocessing import Pool
os.environ["PHYCHIP_SCORER"] = "v14"
sys.path.insert(0, "./work")
FAMS = {"boost", "buck", "tia"}


def run(job):
    from phychip_v12 import score_v12
    fam, targets, text = job
    try:
        return score_v12(fam, targets, text).to_event()
    except Exception as error:  # noqa: BLE001
        return {"all_specs_met": False, "anticheat_blocked": False, "error": repr(error)[:100]}


def main():
    name, tasks_path, gen_path = sys.argv[1:4]
    v13 = [json.loads(l) for l in open(sys.argv[4])] if len(sys.argv) > 4 else None
    tasks = {t["id"]: t for t in map(json.loads, open(tasks_path))}
    gen = [json.loads(l) for l in open(gen_path)]
    if v13 is not None:
        assert len(v13) == len(gen) and all(a["id"] == b["id"] for a, b in zip(v13, gen)), "scored file does not match generations"
    rows, todo = [], []
    for i, g in enumerate(gen):
        t = tasks[g["id"]]
        base = v13[i] if v13 is not None else None
        if base is None or (base["all_specs_met"] and (t["circuit_id"] in FAMS or os.environ.get("ALLPASS"))):
            todo.append(i)
        rows.append({"id": g["id"], "sample": g.get("sample", 0), "circuit_id": t["circuit_id"],
                     "all_specs_met": bool(base and base["all_specs_met"]), "anticheat_blocked": bool(base and base.get("anticheat_blocked")),
                     "v13_pass": None if base is None else bool(base["all_specs_met"])})
    with Pool(int(os.environ.get("PROCS", "4"))) as pool:
        for i, ev in zip(todo, pool.map(run, [(tasks[gen[i]["id"]]["circuit_id"], tasks[gen[i]["id"]]["target_measurements"],
                                                gen[i].get("completion", "")) for i in todo], chunksize=1)):
            rows[i]["all_specs_met"] = bool(ev.get("all_specs_met"))
            rows[i]["anticheat_blocked"] = bool(ev.get("anticheat_blocked"))
    by = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)
    first = sum(1 for v in by.values() if sorted(v, key=lambda r: r["sample"])[0]["all_specs_met"])
    summ = {"scorer": "v14", "name": name, "gen": gen_path, "tasks": len(by), "samples": len(rows), "rescored_rows": len(todo),
            "pass_first": first, "pass_any": sum(1 for v in by.values() if any(r["all_specs_met"] for r in v)),
            "mean_pass_rate": round(sum(r["all_specs_met"] for r in rows) / len(rows), 4),
            "v13_pass_rows": None if v13 is None else sum(1 for r in rows if r["v13_pass"]),
            "flipped_to_fail": None if v13 is None else sum(1 for r in rows if r["v13_pass"] and not r["all_specs_met"])}
    os.makedirs("./rescore14/out", exist_ok=True)
    with open(f"./rescore14/out/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump(summ, open(f"./rescore14/out/{name}.summary.json", "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
