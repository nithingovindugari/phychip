"""Turn hindsight-verified rows into chat SFT rows in the evaluation prompt format.

python build_sft.py OUT.jsonl HL1.jsonl [HL2.jsonl ...] [--cap-per-bin N] [--seed S]
Each kept row: the netlist cleared the canonical scorer on windows around its own measurement.
Excludes any netlist whose hash is a dev-46 or sealed reference.
"""
import argparse, collections, hashlib, json, math, os, random, sys

sys.path.insert(0, os.path.join(os.environ.get("PHYCHIP_REPO", "./phy-chip"), "poc_experiments"))
import generate_rlvr_candidates_v7 as gen  # noqa: E402
from common import messages, spec_prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("hl", nargs="+")
ap.add_argument("--cap-per-bin", type=int, default=0, help="max rows per (family, target bin); 0 = no cap")
ap.add_argument("--exclude", nargs="*", default=[], help="jsonl files whose reference_netlist hashes are held out")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-rows", type=int, default=0)
ap.add_argument("--ids-from", default=None, help="restrict to ids listed in this jsonl (for matched ablations)")
ap.add_argument("--variants", default="loose:0.7,medium:0.15,offset:0.15")
args = ap.parse_args()
rng = random.Random(args.seed)
mix = [(k, float(v)) for k, v in (x.split(":") for x in args.variants.split(","))]

held = set()
for path in args.exclude:
    for line in open(path):
        row = json.loads(line)
        ref = row.get("reference_netlist")
        if ref:
            from phy_chip_env.sandbox import normalize_netlist
            held.add(hashlib.sha256(normalize_netlist(ref).encode()).hexdigest())
            held.add(row.get("reference_netlist_sha256"))

rows, seen = [], set()
stats = collections.Counter()
for path in args.hl:
    for line in open(path):
        r = json.loads(line)
        if r.get("stage") != "ok":
            continue
        # mechanism gates from the red-team audit (an unregulated pass device is not an LDO; a dead bridge is not a driver)
        m = r.get("measured", {})
        if r["circuit_id"] == "ldo" and abs(m.get("line_reg_v_per_v", 0.0)) >= 0.5:
            stats["gate_ldo_unregulated"] += 1; continue
        if r["circuit_id"] == "h-bridge" and m.get("vmotor_pp_v", 0.0) < 1.0:
            stats["gate_hbridge_dead"] += 1; continue
        sha = hashlib.sha256(r["netlist"].encode()).hexdigest()
        if sha in seen:
            stats["dup"] += 1; continue
        if sha in held or r.get("netlist_sha") in held:
            stats["heldout"] += 1; continue
        seen.add(sha)
        rows.append(r)


def bin_of(key, value):
    if key.endswith("_db"):
        return round(value / 3.0)
    return round(math.log10(abs(value) + 1e-30) * 4)  # quarter-decade bins


rng.shuffle(rows)
if args.ids_from:
    keep = {json.loads(l)["id"] for l in open(args.ids_from)}
    rows = [r for r in rows if r["id"] in keep]
if args.max_rows:
    rows = rows[: args.max_rows]
per_bin = collections.Counter()
out = []
for r in rows:
    family = r["circuit_id"]
    inputs, outputs, keys = gen.FAMILY_SPECS[family]
    primary = {k: r["measured"][k] for k in keys}
    b = (family,) + tuple(bin_of(k, v) for k, v in primary.items())
    if args.cap_per_bin and per_bin[b] >= args.cap_per_bin:
        stats["capped"] += 1; continue
    per_bin[b] += 1
    kind = rng.choices([k for k, _ in mix], [w for _, w in mix])[0]
    band = 0.15 if kind == "medium" else 0.30
    targets = {}
    for k, v in primary.items():
        lo, hi = gen.interval(k, v, band)
        if kind == "offset":  # same width, measurement anywhere inside
            shift = rng.uniform(-0.8, 0.8) * (hi - lo) / 2
            lo, hi = lo + shift, hi + shift
        targets[k] = [lo, hi]
    prompt = spec_prompt(family, inputs, outputs, targets)
    out.append({"id": r["id"], "circuit_id": family, "kind": kind, "src": r.get("src"),
                "target_measurements": targets, "measured": primary,
                "messages": messages(prompt, inputs, outputs, "grpo") + [{"role": "assistant", "content": f"```spice\n{r['netlist']}```"}]})
    stats[kind] += 1
with open(args.out, "w") as sink:
    for row in out:
        sink.write(json.dumps(row) + "\n")
fam = collections.Counter(r["circuit_id"] for r in out)
print(json.dumps({"rows": len(out), "bins": len(per_bin), "stats": dict(stats), "families": dict(sorted(fam.items()))}))
