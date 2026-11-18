"""Offline dynamic sampling before RL: keep prompts where the current policy sometimes passes and sometimes fails.

python label_pool.py TASKS.jsonl SCORED.jsonl OUT_POOL.jsonl  (SCORED = score.py output for K samples per task)
"""
import collections, json, sys
tasks = {t["id"]: t for t in map(json.loads, open(sys.argv[1]))}
by = collections.defaultdict(list)
for s in map(json.loads, open(sys.argv[2])):
    by[s["id"]].append(bool(s["all_specs_met"]))
fam = collections.defaultdict(lambda: collections.Counter())
with open(sys.argv[3], "w") as out:
    for tid, passes in by.items():
        t = tasks[tid]; k = len(passes); p = sum(passes)
        kind = "zero" if p == 0 else "all" if p == k else "mixed"
        fam[t["circuit_id"]][kind] += 1
        if kind == "mixed":
            out.write(json.dumps({**t, "label_pass_count": p, "label_k": k}) + "\n")
tot = collections.Counter()
for c in fam.values(): tot.update(c)
json.dump({"total": dict(tot), "families": {f: dict(c) for f, c in sorted(fam.items())}}, open(sys.argv[3].replace(".jsonl", "") + ".stats.json", "w"), indent=1)
print("pool", dict(tot))
