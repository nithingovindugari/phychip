"""Build closed-loop repair training rows from verified sibling circuits.

Two verified circuits that share a parent (same topology, different component values) give a free
repair demonstration: "you were asked for windows around sibling B's measurement, you answered with
sibling A, the simulator reported A's measurement, the corrected answer is B". Only the final answer
carries loss (train_sft.py trains on the last message).

python build_repair.py OUT.jsonl HL1.jsonl [HL2.jsonl ...] --pairs-per-parent 6
"""
import argparse, collections, json, os, random, re, sys

sys.path.insert(0, os.path.join(os.environ.get("PHYCHIP_REPO", "./phy-chip"), "poc_experiments"))
import generate_rlvr_candidates_v7 as gen  # noqa: E402
from common import messages, spec_prompt  # noqa: E402
from feedback import feedback_text, repair_messages  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("hl", nargs="+")
ap.add_argument("--pairs-per-parent", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-rows", type=int, default=0)
args = ap.parse_args()
rng = random.Random(args.seed)


def parent_of(row_id: str) -> str:
    m = re.match(r"^amp[A-Za-z0-9]*?\d*-(.+)-\d+$", row_id)
    return m.group(1) if m else row_id


groups = collections.defaultdict(list)
for path in args.hl:
    for line in open(path):
        r = json.loads(line)
        if r.get("stage") != "ok":
            continue
        m = r.get("measured", {})
        if r["circuit_id"] == "ldo" and abs(m.get("line_reg_v_per_v", 0.0)) >= 0.5:
            continue
        if r["circuit_id"] == "h-bridge" and m.get("vmotor_pp_v", 0.0) < 1.0:
            continue
        groups[(r["circuit_id"], parent_of(r["id"]))].append(r)

out = []
for (family, parent), rows in groups.items():
    if len(rows) < 2:
        continue
    inputs, outputs, keys = gen.FAMILY_SPECS[family]
    made = 0
    tries = 0
    while made < args.pairs_per_parent and tries < args.pairs_per_parent * 6:
        tries += 1
        a, b = rng.sample(rows, 2)
        targets = {k: gen.interval(k, b["measured"][k], 0.30) for k in keys}
        if all(min(t) <= a["measured"][k] <= max(t) for k, t in targets.items()):
            continue  # A already satisfies B's request: not a repair example
        base = messages(spec_prompt(family, inputs, outputs, targets), inputs, outputs, "grpo")
        event = {"simulated": True, "anticheat_blocked": False, "measured": {k: a["measured"][k] for k in keys}}
        attempt = f"```spice\n{a['netlist']}```"
        msgs = repair_messages(base, [(attempt, feedback_text(event, targets))])
        msgs.append({"role": "assistant", "content": f"```spice\n{b['netlist']}```"})
        out.append({"id": f"repair-{a['id']}>{b['id']}", "circuit_id": family, "kind": "repair",
                    "target_measurements": targets, "messages": msgs})
        made += 1
rng.shuffle(out)
if args.max_rows:
    out = out[: args.max_rows]
with open(args.out, "w") as sink:
    for row in out:
        sink.write(json.dumps(row) + "\n")
print(json.dumps({"rows": len(out), "parents": len(groups),
                  "families": dict(sorted(collections.Counter(r["circuit_id"] for r in out).items()))}))
