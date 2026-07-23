"""Ordered concurrent scoring for rollout batches."""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from .scorer import ScoreRequest, ScoreResult, score


def score_batch(
    requests: Sequence[ScoreRequest],
    *,
    max_workers: int | None = None,
) -> list[ScoreResult]:
    """Score requests concurrently while preserving their input order."""

    if not requests:
        return []
    workers = max_workers or min(os.cpu_count() or 1, len(requests))
    if workers <= 0:
        raise ValueError("max_workers must be positive")

    def run(request: ScoreRequest) -> ScoreResult:
        return score(
            request.circuit_id,
            request.target_specs,
            request.completion,
            eval_tol=request.eval_tol,
            timeout_s=request.timeout_s,
        )

    with ThreadPoolExecutor(max_workers=min(workers, len(requests))) as pool:
        return list(pool.map(run, requests))
