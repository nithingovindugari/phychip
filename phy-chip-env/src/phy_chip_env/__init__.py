"""Public API for the PhyChip verifiable-reward environment."""

from .batching import score_batch
from .environment import PhyChipEnv
from .provenance import runtime_fingerprint
from .registry import HarnessInfo, get_harness, list_harnesses
from .scorer import (
    SCORER_SCHEMA,
    SCORER_SEMANTICS,
    ScoreRequest,
    ScoreResult,
    score,
    score_completion,
)

__all__ = [
    "HarnessInfo",
    "PhyChipEnv",
    "SCORER_SCHEMA",
    "SCORER_SEMANTICS",
    "ScoreRequest",
    "ScoreResult",
    "get_harness",
    "list_harnesses",
    "runtime_fingerprint",
    "score",
    "score_batch",
    "score_completion",
]

__version__ = "0.1.0"
