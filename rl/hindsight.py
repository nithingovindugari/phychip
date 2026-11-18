"""Hindsight labelling: measure any netlist with the scorer-v11 harness and
re-express it as a verified (specification -> netlist) pair.

A netlist that simulates, clears the family anti-cheat admission and produces
finite, plausible measurements is, by construction, a correct answer to the
task "meet windows centred on what it measured". The canonical `score()` is
re-run on the relabelled task, so the scorer stays the only judge.

Usage: python hindsight.py IN.jsonl OUT.jsonl [--procs 4]
IN rows: {"id":..., "circuit_id":..., "netlist":...}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import sys
import time

import os  # noqa: E402

sys.path.insert(0, os.path.join(os.environ.get("PHYCHIP_REPO", "./phy-chip"), "poc_experiments"))

from phy_chip_env import get_harness, score  # noqa: E402
from phy_chip_env.sandbox import grade_netlist, normalize_netlist  # noqa: E402
from phy_chip_env.scorer import anticheat_ok  # noqa: E402

import generate_rlvr_candidates_v7 as gen  # noqa: E402
from phychip_v12 import sanitize_netlist, score_v12  # noqa: E402

FAMILY_SPECS = gen.FAMILY_SPECS
TIMEOUT = 25


def clean(netlist: str, family: str) -> str:
    """Drop prose so the stored deck is executable lines under a neutral title.

    Comments in teacher/model decks often state stale numbers; a policy trained on
    them learns to write claims the simulator never checked.
    """
    kept = []
    for line in netlist.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("*") or stripped.lower().startswith(".title"):
            continue
        if ";" in line and not stripped.lower().startswith((".control", "let", "meas", "echo", "print")):
            line = line.split(";", 1)[0].rstrip()
            if not line.strip():
                continue
        kept.append(line.rstrip())
    return f"* {family}\n" + "\n".join(kept) + "\n"


def label(row: dict) -> dict:
    out = {"id": row["id"], "circuit_id": row["circuit_id"], "src": row.get("src")}
    family = row["circuit_id"]
    try:
        netlist = normalize_netlist(row["netlist"])
        # sanitize-v12: circuit content only, neutral title, minimal op block, input AC magnitude pinned to 1
        netlist = sanitize_netlist(netlist, family)
        out["netlist_sha"] = hashlib.sha256(netlist.encode()).hexdigest()
        if family not in FAMILY_SPECS:
            return {**out, "stage": "family"}
        if ".end" not in netlist.casefold():
            return {**out, "stage": "no_end"}
        passed, _ = grade_netlist(netlist, timeout_s=TIMEOUT)
        if not passed:
            return {**out, "stage": "simulate"}
        if not anticheat_ok(netlist, family):
            return {**out, "stage": "anticheat"}
        harness = get_harness(family)
        raw = harness.measure(netlist, spec=None, timeout_s=TIMEOUT) or {}
        measured = {}
        for key, value in raw.items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                measured[key] = value
        out["measured"] = measured
        keys = FAMILY_SPECS[family][2]
        if any(key not in measured for key in keys):
            return {**out, "stage": "measure"}
        primary = {key: measured[key] for key in keys}
        ok, why = gen.plausible(family, primary)
        if not ok:
            return {**out, "stage": "plausibility", "why": why}
        targets = {key: gen.interval(key, primary[key], 0.30) for key in keys}
        result = score_v12(family, targets, f"```spice\n{netlist}\n```", timeout_s=TIMEOUT)
        if not result.all_specs_met or result.anticheat_blocked:
            return {**out, "stage": "score", "blocked": result.anticheat_blocked}
        return {**out, "stage": "ok", "netlist": netlist}
    except Exception as error:  # noqa: BLE001 - one bad deck must not stop the batch
        return {**out, "stage": "error", "why": repr(error)[:200]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inp")
    parser.add_argument("out")
    parser.add_argument("--procs", type=int, default=4)
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.inp)]
    done = set()
    try:
        done = {json.loads(line)["id"] for line in open(args.out)}
    except FileNotFoundError:
        pass
    rows = [row for row in rows if row["id"] not in done]
    start = time.time()
    counts: dict[str, int] = {}
    with mp.Pool(args.procs) as pool, open(args.out, "a") as sink:
        for index, result in enumerate(pool.imap_unordered(label, rows, chunksize=4), 1):
            sink.write(json.dumps(result) + "\n")
            counts[result["stage"]] = counts.get(result["stage"], 0) + 1
            if index % 500 == 0:
                sink.flush()
                print(index, len(rows), round(time.time() - start), counts, flush=True)
    print("done", len(rows), round(time.time() - start), counts, flush=True)


if __name__ == "__main__":
    main()
