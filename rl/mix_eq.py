"""Build the equation-preferred set: for families with law-checked templates use only equation rows; plain rows elsewhere.
python mix_eq.py PLAIN.jsonl EQ.jsonl OUT.jsonl [--keep-plain-frac 0.0]"""
import collections, json, random, sys
plain, eq, out = sys.argv[1:4]
keep = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[4] == "--keep-plain-frac" else 0.0
rng = random.Random(0)
eq_rows = [json.loads(l) for l in open(eq)]
fams = collections.Counter(r["circuit_id"] for r in eq_rows)
rows = [r for r in map(json.loads, open(plain)) if r["circuit_id"] not in fams or rng.random() < keep] + eq_rows
rng.shuffle(rows)
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(json.dumps({"rows": len(rows), "equation_rows": len(eq_rows), "equation_families": dict(sorted(fams.items()))}))
