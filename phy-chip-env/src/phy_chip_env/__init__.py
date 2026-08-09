"""Public API for the PhyChip verifiable-reward environment."""

from .batching import score_batch
from .campaign import (
    SKY130_CAMPAIGN_SCHEMA,
    load_sky130_campaign,
    preflight_sky130_campaign,
)
from .environment import PhyChipEnv
from .observation import (
    OBSERVATION_SCHEMA,
    Observation,
    ObservationFailure,
    observe_completion,
)
from .provenance import runtime_fingerprint
from .readiness import (
    TOKEN_DISTRIBUTION_MANIFEST_SCHEMA,
    TRAJECTORY_APPROVAL_SCHEMA,
    assess_training_readiness,
)
from .registry import HarnessInfo, get_harness, list_harnesses
from .research import (
    compare_benchmark_runs,
    compare_feedback_runs,
    load_benchmark,
    run_benchmark,
    run_paired_feedback_benchmark,
)
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
    "OBSERVATION_SCHEMA",
    "Observation",
    "ObservationFailure",
    "PhyChipEnv",
    "SCORER_SCHEMA",
    "SCORER_SEMANTICS",
    "ScoreRequest",
    "ScoreResult",
    "TRAJECTORY_APPROVAL_SCHEMA",
    "TOKEN_DISTRIBUTION_MANIFEST_SCHEMA",
    "SKY130_CAMPAIGN_SCHEMA",
    "assess_training_readiness",
    "compare_benchmark_runs",
    "compare_feedback_runs",
    "get_harness",
    "list_harnesses",
    "load_benchmark",
    "load_sky130_campaign",
    "observe_completion",
    "preflight_sky130_campaign",
    "runtime_fingerprint",
    "run_benchmark",
    "run_paired_feedback_benchmark",
    "score",
    "score_batch",
    "score_completion",
]

__version__ = "0.1.0"
