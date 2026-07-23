from __future__ import annotations

from phy_chip_env import ScoreRequest, batching
from phy_chip_env.scorer import ScoreResult


def _result(value: float) -> ScoreResult:
    return ScoreResult(
        reward=value,
        simulated=True,
        anticheat_blocked=False,
        requested_specs=1,
        measured_specs=1,
        met_specs=1,
        spec_score=1.0,
        all_specs_met=True,
        has_spice_block=True,
        has_end=True,
        has_device=True,
        has_analysis=True,
    )


def test_score_batch_preserves_input_order(monkeypatch) -> None:
    monkeypatch.setattr(
        batching,
        "score",
        lambda circuit_id, *_args, **_kwargs: _result(float(circuit_id)),
    )
    requests = [ScoreRequest(str(index), {}, "completion") for index in (3, 1, 4, 2)]

    results = batching.score_batch(requests, max_workers=4)

    assert [result.reward for result in results] == [3.0, 1.0, 4.0, 2.0]


def test_empty_batch_is_empty() -> None:
    assert batching.score_batch([]) == []
