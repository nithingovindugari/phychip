"""Paired, specification-clustered statistics for PhyChip GRPO evaluation."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable


def spec_identity(row: dict[str, Any]) -> str:
    """Stable identity for pairing the same specification across policies."""

    if row.get("spec_id"):
        return str(row["spec_id"])
    identity = {
        "circuit_id": row.get("circuit_id"),
        "spec_prompt": row.get("spec_prompt") or row.get("prompt"),
        "target_measurements": row.get("target_measurements") or row.get("targets") or {},
        "input_ports": row.get("input_ports") or [],
        "output_ports": row.get("output_ports") or [],
    }
    payload = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_policy_results(path: Path) -> dict[str, dict[str, Any]]:
    """Load one policy's per-spec results, rejecting duplicates and malformed rows."""

    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = spec_identity(row)
            if identity in rows:
                raise ValueError(f"{path}:{line_number}: duplicate spec identity {identity}")
            pass_count = int(row["pass_count"])
            k = int(row["k"])
            if k < 1 or not 0 <= pass_count <= k:
                raise ValueError(f"{path}:{line_number}: invalid pass_count={pass_count}, k={k}")
            rows[identity] = row
    if not rows:
        raise ValueError(f"no evaluation rows in {path}")
    return rows


def _pass1(row: dict[str, Any]) -> float:
    return int(row["pass_count"]) / int(row["k"])


def _passk(row: dict[str, Any]) -> float:
    return float(int(row["pass_count"]) > 0)


def _interval(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    ordered = sorted(values)
    lower = ordered[int((alpha / 2) * len(ordered))]
    upper = ordered[min(len(ordered) - 1, int((1 - alpha / 2) * len(ordered)))]
    return lower, upper


def _paired_metric(
    baseline: dict[str, dict[str, Any]],
    treatments: list[dict[str, dict[str, Any]]],
    random_control: dict[str, dict[str, Any]],
    metric: Callable[[dict[str, Any]], float],
    *,
    bootstrap_samples: int,
    rng: random.Random,
) -> dict[str, Any]:
    identities = sorted(baseline)
    expected = set(identities)
    for index, policy in enumerate(treatments):
        if set(policy) != expected:
            raise ValueError(f"treatment seed {index} does not contain the baseline spec set")
    if set(random_control) != expected:
        raise ValueError("random control does not contain the baseline spec set")

    base = [metric(baseline[identity]) for identity in identities]
    treatment_by_seed = [
        [metric(policy[identity]) for identity in identities] for policy in treatments
    ]
    treatment_seed_means = [
        sum(seed_values) / len(seed_values) for seed_values in treatment_by_seed
    ]
    treatment = [
        sum(seed_values[index] for seed_values in treatment_by_seed) / len(treatment_by_seed)
        for index in range(len(identities))
    ]
    random_values = [metric(random_control[identity]) for identity in identities]
    delta = [treatment[index] - base[index] for index in range(len(identities))]
    random_delta = [random_values[index] - base[index] for index in range(len(identities))]

    boot_base: list[float] = []
    boot_treatment: list[float] = []
    boot_delta: list[float] = []
    boot_random_delta: list[float] = []
    n = len(identities)
    for _ in range(bootstrap_samples):
        sampled = [rng.randrange(n) for _ in range(n)]
        boot_base.append(sum(base[index] for index in sampled) / n)
        boot_treatment.append(sum(treatment[index] for index in sampled) / n)
        boot_delta.append(sum(delta[index] for index in sampled) / n)
        boot_random_delta.append(sum(random_delta[index] for index in sampled) / n)

    delta_low, delta_high = _interval(boot_delta)
    random_low, random_high = _interval(boot_random_delta)
    random_control_not_positive = random_low <= 0.0
    return {
        "n_specs": n,
        "n_treatment_seeds": len(treatments),
        "baseline_mean": sum(base) / n,
        "baseline_ci": list(_interval(boot_base)),
        "treatment_mean_across_seeds": sum(treatment) / n,
        "treatment_seed_means": treatment_seed_means,
        "treatment_seed_mean_range": [
            min(treatment_seed_means),
            max(treatment_seed_means),
        ],
        "treatment_ci_clustered_by_spec": list(_interval(boot_treatment)),
        "paired_delta": sum(delta) / n,
        "paired_delta_ci": [delta_low, delta_high],
        "random_control_mean": sum(random_values) / n,
        "random_paired_delta": sum(random_delta) / n,
        "random_paired_delta_ci": [random_low, random_high],
        "random_control_not_positive": random_control_not_positive,
        "grpo_lift": delta_low > 0.0 and random_control_not_positive,
    }


def paired_cluster_verdict(
    baseline: dict[str, dict[str, Any]],
    treatments: list[dict[str, dict[str, Any]]],
    random_control: dict[str, dict[str, Any]],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Average treatment seeds within spec, then bootstrap paired specs.

    Seeds are treated as fixed experimental replicates; the confidence interval
    clusters on the independent unit (specification) instead of pretending each
    seed×spec value is an independent observation.
    """

    if not treatments:
        raise ValueError("at least one treatment seed is required")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    rng = random.Random(seed)
    return {
        "method": "paired specification-cluster bootstrap; treatment seeds averaged within spec",
        "bootstrap_samples": bootstrap_samples,
        "pass_at_1": _paired_metric(
            baseline,
            treatments,
            random_control,
            _pass1,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        ),
        "pass_at_k": _paired_metric(
            baseline,
            treatments,
            random_control,
            _passk,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        ),
    }
