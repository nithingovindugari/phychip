"""Small configured wrapper around the canonical functional scorer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .batching import score_batch
from .scorer import ScoreRequest, ScoreResult, score


@dataclass(frozen=True)
class PhyChipEnv:
    """One-shot physics verifier suitable for RLVR reward callbacks."""

    eval_tol: float = 0.30
    timeout_s: float = 15.0
    max_workers: int | None = None

    def score(
        self,
        circuit_id: str,
        target_specs: dict,
        completion: str,
    ) -> ScoreResult:
        return score(
            circuit_id,
            target_specs,
            completion,
            eval_tol=self.eval_tol,
            timeout_s=self.timeout_s,
        )

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[ScoreResult]:
        normalized = [
            ScoreRequest(
                circuit_id=request.circuit_id,
                target_specs=request.target_specs,
                completion=request.completion,
                eval_tol=request.eval_tol,
                timeout_s=request.timeout_s,
            )
            for request in requests
        ]
        return score_batch(normalized, max_workers=self.max_workers)
