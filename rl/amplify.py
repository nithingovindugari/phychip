"""Verifier-driven data amplification.

Take netlists that already clear the hindsight gate and perturb their passive
component values (R/C/L cards and numeric .param assignments). Each variant is a
new circuit with different physics; hindsight.py then measures it and keeps it
only if the canonical scorer accepts it. No language model is involved: the
simulator is the teacher.

python amplify.py OK.jsonl OUT_IN.jsonl --per 6 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re

SUFFIX = {"t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}
NUM = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)(meg|[tgkmunpf])?[a-z]*$", re.I)


def parse(token: str) -> float | None:
    match = NUM.match(token)
    if not match:
        return None
    value = float(match.group(1))
    if match.group(2):
        value *= SUFFIX[match.group(2).lower()]
    return value


def fmt(value: float) -> str:
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    value = abs(value)
    for suffix, scale in (("G", 1e9), ("Meg", 1e6), ("k", 1e3), ("", 1.0), ("m", 1e-3), ("u", 1e-6), ("n", 1e-9), ("p", 1e-12), ("f", 1e-15)):
        if value >= scale:
            return f"{sign}{value / scale:.3g}{suffix}"
    return f"{sign}{value:.3g}"


def perturb(netlist: str, rng: random.Random, span: float, p_touch: float) -> tuple[str, int]:
    lines = netlist.split("\n")
    touched = 0
    out = []
    in_control = False
    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith(".control"):
            in_control = True
        if low.startswith(".endc"):
            in_control = False
        if in_control or not stripped or stripped.startswith("*"):
            out.append(line)
            continue
        head = low[0]
        if head in "rcl":
            parts = line.split()
            if len(parts) >= 4:
                value = parse(parts[3])
                if value is not None and value > 0 and rng.random() < p_touch:
                    parts[3] = fmt(value * math.exp(rng.uniform(-span, span)))
                    touched += 1
                    line = " ".join(parts)
        elif low.startswith(".param"):
            def repl(match):
                nonlocal touched
                value = parse(match.group(2))
                if value is None or value <= 0 or rng.random() > p_touch * 0.6:
                    return match.group(0)
                touched += 1
                return f"{match.group(1)}{fmt(value * math.exp(rng.uniform(-span, span)))}"
            line = re.sub(r"(\b\w+\s*=\s*)([+-]?[\d.]+(?:e[+-]?\d+)?(?:meg|[tgkmunpf])?)\b", repl, line, flags=re.I)
        out.append(line)
    return "\n".join(out), touched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ok")
    ap.add_argument("out")
    ap.add_argument("--per", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="amp")
    ap.add_argument("--spans", default="0.4,0.8,1.2,1.8", help="log-uniform half-widths (natural log) to draw from")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    spans = [float(x) for x in args.spans.split(",")]
    rows = [r for r in map(json.loads, open(args.ok)) if r.get("stage") == "ok"]
    seen = set()
    count = 0
    with open(args.out, "w") as sink:
        for row in rows:
            for index in range(args.per):
                span = rng.choice(spans)
                p_touch = rng.choice([0.3, 0.6, 1.0])
                variant, touched = perturb(row["netlist"], rng, span, p_touch)
                if not touched or variant in seen:
                    continue
                seen.add(variant)
                sink.write(json.dumps({"id": f"{args.tag}{args.seed}-{row['id']}-{index}", "circuit_id": row["circuit_id"],
                                       "netlist": variant, "src": f"{args.tag}:{row['id']}"}) + "\n")
                count += 1
    print("variants", count, "from", len(rows))


if __name__ == "__main__":
    main()
