"""Structural diversity and novelty of passing answers: signature = per-family multiset of device types
(element letter + MOS/BJT polarity), insensitive to values and node names. Novel = signature absent from hv7 SFT data."""
import json, re, collections, sys
def sig(text):
    m = re.search(r"```spice\s*(.*?)```", text or "", re.S | re.I); net = m.group(1) if m else (text or "")
    c = collections.Counter()
    for ln in net.splitlines():
        s = ln.strip()
        if not s or s[0] in "*.+": continue
        t = s[0].upper()
        if t in "MQ":
            low = s.lower(); t += "p" if re.search(r"\b(pmos|pnp|p\w*mod|pch)\b", low) else "n"
        if t in "VI": continue  # sources are fixed by the interface contract
        c[t] += 1
    return tuple(sorted(c.items()))
sft = collections.defaultdict(set)
for l in open("./work/data/hv7.jsonl"):
    r = json.loads(l); m = r["messages"]; m = m if isinstance(m, list) else eval(m)
    sft[r["circuit_id"]].add(sig(m[-1]["content"]))
tasks = {}
for p in ("./work/tasks_sealed.jsonl", "./work/tasks_topo_heldout.jsonl"):
    for t in map(json.loads, open(p)): tasks[t["id"]] = t
for d in sys.argv[1:]:
    gen = [json.loads(l) for l in open(d + "/k8.jsonl")]; sc = [json.loads(l) for l in open(d + "/k8_scored.jsonl")]
    fam = collections.defaultdict(set); passes = novel = 0
    for g, s in zip(gen, sc):
        if not s["all_specs_met"]: continue
        f = tasks[g["id"]]["circuit_id"]; k = sig(g["completion"]); fam[f].add(k); passes += 1; novel += k not in sft[f]
    print(f"{d.split('/')[-1]:34s} passes {passes:5d} distinct signatures {sum(len(v) for v in fam.values()):4d} novel-pass share {novel/passes:.3f}")
