from pathlib import Path

from audit_rlvr_candidate_pilot import is_quarantined_candidate_path


def test_quarantined_candidate_pools_are_not_treated_as_training_data() -> None:
    assert is_quarantined_candidate_path(
        Path("data/rlvr_candidates_v7_20260722/candidates.jsonl")
    )
    assert is_quarantined_candidate_path(
        Path("data/rlvr_candidates_v8_20260722/rollouts.jsonl")
    )
    assert not is_quarantined_candidate_path(
        Path("data/rlvr_trainable_v6/train.jsonl")
    )
