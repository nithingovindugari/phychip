#!/usr/bin/env python3
"""Audit a private K-sample RLVR rollout file and build a manual-review queue."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phy-chip-env" / "src"))

from phy_chip_env.scorer import SCORER_SEMANTICS, score  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--review-queue-out", type=Path, required=True)
    parser.add_argument("--rescored-rollouts-out", type=Path)
    parser.add_argument("--expected-specs", type=int, default=48)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--high-reward", type=float, default=0.85)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--rescore", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rescored_rollouts_out is not None and not args.rescore:
        raise SystemExit("--rescored-rollouts-out requires --rescore")
    specs = {row["spec_id"]: row for row in load_jsonl(args.specs)}
    rollouts = load_jsonl(args.rollouts)
    failures: list[str] = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rollouts:
        grouped[row.get("spec_id", "")].append(row)

    if len(specs) != args.expected_specs:
        failures.append(f"expected {args.expected_specs} specs, found {len(specs)}")
    if set(grouped) != set(specs):
        failures.append("rollout spec IDs do not exactly match the frozen reference set")
    if len(rollouts) != args.expected_specs * args.k:
        failures.append(
            f"expected {args.expected_specs * args.k} rollouts, found {len(rollouts)}"
        )

    family_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()
    review_reasons: Counter[str] = Counter()
    mixed_families: Counter[str] = Counter()
    review_rows: list[dict] = []
    rescored_rows: list[dict] = []
    local_mismatches: list[dict] = []
    exact_successes = 0
    capped = 0
    ended = 0
    anticheat = 0

    for spec_id, spec in specs.items():
        rows = sorted(grouped.get(spec_id, []), key=lambda row: row.get("sample_index", -1))
        family = spec["circuit_id"]
        family_counts[family] += 1
        if len(rows) != args.k:
            failures.append(f"{spec_id}: expected {args.k} rollouts, found {len(rows)}")
        if [row.get("sample_index") for row in rows] != list(range(args.k)):
            failures.append(f"{spec_id}: sample indexes are incomplete or duplicated")

        effective_events: list[dict] = []
        for row in rows:
            remote = row.get("score", {})
            effective = remote
            if args.rescore:
                local = score(
                    family,
                    spec.get("target_measurements") or {},
                    row.get("completion", ""),
                    timeout_s=args.timeout_s,
                ).to_event()
                comparable = ("reward", "spec_score", "all_specs_met", "anticheat_blocked")
                agrees = all(
                    close(remote.get(key, 0.0), local.get(key, 0.0))
                    if key in {"reward", "spec_score"}
                    else remote.get(key) == local.get(key)
                    for key in comparable
                )
                if not agrees:
                    local_mismatches.append(
                        {
                            "spec_id": spec_id,
                            "sample_index": row.get("sample_index"),
                            "remote": {key: remote.get(key) for key in comparable},
                            "local": {key: local.get(key) for key in comparable},
                        }
                    )
                effective = local
            effective_events.append(effective)

        max_reward = max(
            (float(event.get("reward", 0.0)) for event in effective_events),
            default=0.0,
        )
        pass_count = sum(bool(event.get("all_specs_met")) for event in effective_events)
        if pass_count == 0:
            stratum_counts["zero_0"] += 1
        elif pass_count == args.k:
            stratum_counts["easy_k"] += 1
        else:
            stratum_counts["mixed"] += 1
            mixed_families[family] += 1

        for row, effective in zip(rows, effective_events, strict=True):
            if args.rescore:
                rescored_rows.append({**row, "score": effective})
            reasons: list[str] = []
            if effective.get("all_specs_met"):
                reasons.append("exact_success")
                exact_successes += 1
            if float(effective.get("reward", 0.0)) >= args.high_reward:
                reasons.append("high_reward")
            if effective.get("anticheat_blocked"):
                reasons.append("anticheat_blocked")
                anticheat += 1
            if close(float(effective.get("reward", 0.0)), max_reward) and max_reward > 0.0:
                reasons.append("group_winner")
            capped += int(bool(row.get("capped")))
            ended += int(bool(row.get("ended_with_assistant_eos")))
            if reasons:
                for reason in set(reasons):
                    review_reasons[reason] += 1
                review_rows.append(
                    {
                        **row,
                        "manual_review_reasons": sorted(set(reasons)),
                        "local_rescore": effective if args.rescore else None,
                    }
                )

    if local_mismatches and args.rescored_rollouts_out is None:
        failures.append(f"{len(local_mismatches)} local rescoring mismatches")

    summary = {
        "schema": "phychip-rlvr-rollout-audit-v1",
        "passed_machine_checks": not failures,
        "manual_review_complete": False,
        "scorer_semantics": SCORER_SEMANTICS,
        "spec_count": len(specs),
        "rollout_count": len(rollouts),
        "K": args.k,
        "family_counts": dict(sorted(family_counts.items())),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "mixed_family_counts": dict(sorted(mixed_families.items())),
        "exact_success_count": exact_successes,
        "anticheat_blocked_count": anticheat,
        "capped_count": capped,
        "ended_with_assistant_eos_count": ended,
        "local_rescore_mismatch_count": len(local_mismatches),
        "local_rescore_mismatches": local_mismatches,
        "manual_review_queue_count": len(review_rows),
        "manual_review_reason_counts": dict(sorted(review_reasons.items())),
        "high_reward_threshold": args.high_reward,
        "timeout_s": args.timeout_s,
        "failures": failures,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.review_queue_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.review_queue_out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in review_rows)
    )
    if args.rescored_rollouts_out is not None:
        args.rescored_rollouts_out.parent.mkdir(parents=True, exist_ok=True)
        args.rescored_rollouts_out.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rescored_rows)
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
