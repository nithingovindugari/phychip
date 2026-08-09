#!/usr/bin/env python3
"""Well-posedness gate for phy-chip-bench-v2 (post-repair, reviewer-found).

The malformed-numeric repair (l2l3_repaired.jsonl) fixes 17/22 corrupted L2/L3
references. A residual set still encodes DEGENERATE gold references: the harness
measures a non-physical headline FOM (gain << 0 dB, gain == 0 V/V, efficiency
outside [1,100] %, or a None headline). Such a reference would unfairly FAIL a
correct implementation of the named circuit, so it is not a valid grading target.

This gate runs every L2/L3 reference through its harness and DROPS rows whose
headline FOM is degenerate. L1 rows (compile-only) pass through untouched. The
rule is objective and uniform (no per-row tuning):

  DROP if any of:
    * headline FOM (first measured key) is None
    * every *gain*_db key is < -60 dB        (near-zero output, not unity 0 dB)
    * any gain in V/V (*_v_per_v) == 0        (no transfer for a named amp)
    * efficiency_pct present and not 1<=x<=100 (non-physical converter)

Usage:
  python athma-train/scripts/wellposed_gate_bench_v2.py \
    --in  eval_sets/phychip_bench_v2/bench_v2_clean.jsonl \
    --out eval_sets/phychip_bench_v2/bench_v2.jsonl \
    --report logs/bench_v2/wellposed.json
"""
from __future__ import annotations
import argparse, importlib.util, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str((ROOT / "athma-train").resolve()))
from athma_train.spice_gate import normalize_netlist  # noqa: E402


def load_harness(cid: str):
    fp = ROOT / "athma-train" / "athma_train" / "harnesses" / f"{cid}.py"
    if not fp.exists():
        return None
    spec = importlib.util.spec_from_file_location(cid.replace("-", "_"), fp)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    return h


def cid_of(bid: str) -> str:
    m = re.match(r"tier\d+-(.+?)-l[123]", bid)
    return m.group(1) if m else ""


def degenerate(fom: dict) -> str | None:
    """Return a reason string if the reference FOMs are degenerate, else None."""
    if not fom:
        return "no FOMs"
    items = list(fom.items())
    if items[0][1] is None:
        return f"headline FOM {items[0][0]} is None"
    gdb = {k: v for k, v in fom.items() if "gain" in k and k.endswith("_db") and v is not None}
    if gdb and all(v < -60 for v in gdb.values()):
        return f"all dB gains <-60 ({gdb})"
    # linear gain keys: degenerate if |gain| < 1e-3 V/V (= the -60 dB floor above,
    # so a 1.68 nV/V "amplifier" is caught the same way a -291 dB one is).
    for k, v in fom.items():
        if "gain" in k and k.endswith("_v_per_v") and v is not None and abs(v) < 1e-3:
            return f"{k}={v:g} (<1e-3 V/V, no transfer)"
    eff = fom.get("efficiency_pct")
    if eff is not None and not (1.0 <= eff <= 100.0):
        return f"efficiency_pct={eff} non-physical"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    kept, drops = [], []
    for r in rows:
        if r.get("level") == 1 or r.get("compile_only_ok"):
            kept.append(r)
            continue
        h = load_harness(cid_of(r["id"]))
        if h is None or not hasattr(h, "measure"):
            drops.append({"id": r["id"], "why": "no harness"})
            continue
        try:
            fom = h.measure(normalize_netlist(r["reference_netlist"]), None, 25)
        except Exception as e:
            drops.append({"id": r["id"], "why": f"measure-err:{str(e)[:50]}"})
            continue
        why = degenerate({k: v for k, v in (fom or {}).items()})
        if why:
            drops.append({"id": r["id"], "why": why,
                          "fom": {k: (round(v, 3) if isinstance(v, (int, float)) else v)
                                  for k, v in (fom or {}).items()}})
            continue
        kept.append(r)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    klvl = Counter(r["level"] for r in kept)
    report = {"in": len(rows), "kept": len(kept), "dropped": len(drops),
              "kept_by_level": dict(klvl), "drops": drops}
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.report, "w"), indent=1)
    print(f"KEPT {len(kept)}/{len(rows)} (L1={klvl.get(1,0)} L2={klvl.get(2,0)} L3={klvl.get(3,0)})")
    for d in drops:
        print(f"  DROP {d['id']:30s} {d['why']}")


if __name__ == "__main__":
    main()
