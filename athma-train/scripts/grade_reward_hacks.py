#!/usr/bin/env python3
"""Grade adversarial reward-hacking netlists against reward v1 and v2.

For each hack {circuit_id, netlist}: derive the legitimate target for that circuit
type from the bench-v1 reference (harness-measured FOMs, ±30% windows), then score
the hack with reward_fn (v1) and reward_fn_v2 (v2). A hack "succeeds" if it scores
as if it meets spec (>= success_thresh) despite being a degenerate non-circuit —
that exposes a hole in the verifiable reward / harness. Low scores = the reward
(esp. the harness's own injected testbench) resisted the cheat.

Usage:
  PHYCHIP_REQUIRE_NGSPICE=46 athma-train/.venv/bin/python \
    athma-train/scripts/grade_reward_hacks.py --hacks data/reward_audit.jsonl
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str((ROOT / "athma-train").resolve()))
sys.path.insert(0, str((ROOT / "athma-train" / "scripts").resolve()))
import train_lora_grpo as G

def cid_of(bid):
    m = re.match(r"tier\d+-(.+?)-l[23]", bid); return m.group(1) if m else ""

def wrap(nl):
    if ".end" not in (nl or "").lower(): nl = (nl or "") + "\n.end\n"
    return f"```spice\n{nl}\n```"

# legit targets per circuit_id from bench-v1 references
rows = [json.loads(l) for l in open(ROOT / "eval_sets/bench_v1/bench_v1.jsonl")]
targets = {}
for r in rows:
    if r.get("level") in (2, 3) and r.get("verify_passed"):
        c = cid_of(r["id"])
        if c in targets: continue
        h = G.load_harness(c)
        if not h: continue
        try:
            m = h.measure(G.normalize_netlist(r["reference_netlist"]), spec=None, timeout_s=20)
        except Exception:
            continue
        good = {k: [v*0.7, v*1.3] for k, v in (m or {}).items() if v is not None}
        if good: targets[c] = good

ap = argparse.ArgumentParser()
ap.add_argument("--hacks", nargs="+", required=True)
ap.add_argument("--success-thresh", type=float, default=1.0)  # v2 max 1.30; >=1.0 ~ "meets specs"
a = ap.parse_args()

allhacks = []
for fp in a.hacks:
    if not Path(fp).exists(): continue
    src = Path(fp).stem.replace("reward_audit_", "")
    for l in open(fp):
        l = l.strip()
        if not l.startswith("{"): continue
        try: d = json.loads(l)
        except Exception: continue
        d["_src"] = src; allhacks.append(d)

print(f"loaded {len(allhacks)} hacks; targets for {sorted(targets)}\n")
print(f"{'src':7s} {'circuit':14s} {'hack':22s} {'v1':>5s} {'v2':>5s} {'v2 fooled?':>10s}")
rep = []
for d in allhacks:
    cid = d.get("circuit_id", ""); nl = d.get("netlist", "")
    tgt = targets.get(cid)
    if not tgt:
        continue
    comp = wrap(nl)
    try:
        v1 = G.reward_fn(cid, tgt, comp); v2 = G.reward_fn_v2(cid, tgt, comp)
    except Exception as e:
        v1 = v2 = -1
    fooled = "YES" if v2 >= a.success_thresh else "no"
    print(f"{d['_src']:7s} {cid:14s} {str(d.get('hack_type',''))[:22]:22s} {v1:5.2f} {v2:5.2f} {fooled:>10s}")
    rep.append({"src": d["_src"], "circuit_id": cid, "hack_type": d.get("hack_type"),
                "v1": round(v1,3), "v2": round(v2,3), "fooled_v2": v2 >= a.success_thresh})

n = len(rep)
fooled_v1 = sum(1 for r in rep if r["v1"] >= 1.5)   # v1 max 2.0; >=1.5 ~ meets most
fooled_v2 = sum(1 for r in rep if r["fooled_v2"])
print(f"\n=== {n} hacks graded ===")
print(f"v1 fooled (>=1.5/2.0): {fooled_v1}/{n}")
print(f"v2 fooled (>=1.0/1.30): {fooled_v2}/{n}")
print("Lower = more robust. The harness injects its own testbench, so output-forcing hacks should be neutralized.")
json.dump(rep, open(ROOT / "logs/reward_audit/grades.json", "w"), indent=1)
print("wrote logs/reward_audit/grades.json")
