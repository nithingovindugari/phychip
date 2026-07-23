#!/usr/bin/env python3
"""Fail-closed health gate for a completed GRPO training log."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path


MAPPING_RE = re.compile(r"\{.*\}")
REWARD_DECOMP_RE = re.compile(
    r"frac_anticheat_blocked=(?P<blocked>[0-9.]+).*"
    r"frac_group_reward_var=(?P<reward_var>[0-9.]+).*"
    r"frac_group_objective_var=(?P<objective_var>[0-9.]+).*"
    r"winner_exact=(?P<winner_exact>[0-9.]+).*"
    r"winner_partial=(?P<winner_partial>[0-9.]+).*"
    r"winner_gate_only=(?P<winner_gate>[0-9.]+).*"
    r"winner_nosim=(?P<winner_nosim>[0-9.]+).*"
    r"winner_blocked=(?P<winner_blocked>[0-9.]+)"
)


def _numeric(records: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"non-finite {key}: {value}")
        values.append(converted)
    return values


def inspect_log(path: Path) -> dict:
    records: list[dict] = []
    decompositions: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = MAPPING_RE.search(line)
        if match:
            try:
                value = ast.literal_eval(match.group(0))
            except (SyntaxError, ValueError):
                value = None
            if isinstance(value, dict) and "entropy" in value:
                records.append(value)
        decomp = REWARD_DECOMP_RE.search(line)
        if decomp:
            decompositions.append({key: float(value) for key, value in decomp.groupdict().items()})

    generation = [record for record in records if "completions/clipped_ratio" in record]
    entropy = _numeric(records, "entropy")
    grad_norm = _numeric(records, "grad_norm")
    clip = _numeric(records, "clip_ratio/region_mean")
    clipped = _numeric(generation, "completions/clipped_ratio")
    zero_std = _numeric(generation, "frac_reward_zero_std")
    if not generation or not entropy or not grad_norm or not decompositions:
        raise ValueError("training log is missing required health metrics")

    blocked = [value["blocked"] for value in decompositions]
    observations = {
        "optimizer_log_rows": len(records),
        "generation_groups": len(generation),
        "reward_decomposition_groups": len(decompositions),
        "entropy_min": min(entropy),
        "entropy_final": entropy[-1],
        "grad_norm_max": max(grad_norm),
        "positive_grad_fraction": sum(value > 1e-8 for value in grad_norm) / len(grad_norm),
        "clip_region_max": max(clip),
        "completion_clipped_ratio_mean": sum(clipped) / len(clipped),
        "completion_clipped_ratio_max": max(clipped),
        "completion_majority_clipped_group_fraction": sum(value > 0.50 for value in clipped)
        / len(clipped),
        "zero_reward_std_group_fraction": sum(value >= 1.0 for value in zero_std) / len(zero_std),
        "anticheat_blocked_mean": sum(blocked) / len(blocked),
        "anticheat_blocked_max": max(blocked),
        "anticheat_majority_blocked_group_fraction": sum(value > 0.50 for value in blocked)
        / len(blocked),
        "reward_variance_group_fraction": sum(value["reward_var"] > 0 for value in decompositions)
        / len(decompositions),
        "objective_variance_group_fraction": sum(
            value["objective_var"] > 0 for value in decompositions
        )
        / len(decompositions),
        "winner_exact_mean": sum(value["winner_exact"] for value in decompositions)
        / len(decompositions),
        "winner_partial_mean": sum(value["winner_partial"] for value in decompositions)
        / len(decompositions),
        "winner_gate_only_mean": sum(value["winner_gate"] for value in decompositions)
        / len(decompositions),
        "winner_nosim_max": max(value["winner_nosim"] for value in decompositions),
        "winner_blocked_max": max(value["winner_blocked"] for value in decompositions),
    }
    checks = {
        "generation_groups_present": observations["generation_groups"] >= 20,
        "entropy_above_floor": observations["entropy_min"] >= 0.15,
        "gradients_present_finite_and_bounded": (
            observations["grad_norm_max"] > 1e-8
            and observations["grad_norm_max"] <= 5.0
            and observations["positive_grad_fraction"] >= 0.05
        ),
        "clip_region_bounded": observations["clip_region_max"] <= 0.25,
        "completion_clipping_bounded": (
            observations["completion_clipped_ratio_mean"] <= 0.05
            and observations["completion_majority_clipped_group_fraction"] <= 0.10
        ),
        "zero_variance_groups_bounded": observations["zero_reward_std_group_fraction"] <= 0.25,
        "reward_variance_present": observations["reward_variance_group_fraction"] >= 0.75,
        "objective_variance_present": observations["objective_variance_group_fraction"] >= 0.50,
        "invalid_rollouts_never_win_advantage": (
            observations["winner_nosim_max"] == 0.0
            and observations["winner_blocked_max"] == 0.0
        ),
        "spec_signal_dominates_advantage_winners": (
            observations["winner_exact_mean"] + observations["winner_partial_mean"] >= 0.50
            and observations["winner_gate_only_mean"] <= 0.25
        ),
        "anticheat_self_blocking_bounded": (
            observations["anticheat_blocked_mean"] <= 0.35
            and observations["anticheat_majority_blocked_group_fraction"] <= 0.20
        ),
    }
    return {
        "schema": "phychip-grpo-train-health-v1",
        "log": path.as_posix(),
        "passed": all(checks.values()),
        "checks": checks,
        "observations": observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_log(args.log)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
