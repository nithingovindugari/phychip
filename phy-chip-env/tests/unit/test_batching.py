from __future__ import annotations

import pytest

from phy_chip_env import PhyChipEnv, ScoreRequest, batching, environment
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


def test_functional_batch_uses_each_requests_configuration(monkeypatch) -> None:
    captured: list[tuple[str, float, float]] = []

    def fake_score(circuit_id, _targets, _completion, *, eval_tol, timeout_s):
        captured.append((circuit_id, eval_tol, timeout_s))
        return _result(eval_tol)

    monkeypatch.setattr(batching, "score", fake_score)
    requests = [
        ScoreRequest("buffer", {}, "first", eval_tol=0.07, timeout_s=1.25),
        ScoreRequest("inv-amp", {}, "second", eval_tol=0.41, timeout_s=9.5),
    ]

    results = batching.score_batch(requests, max_workers=1)

    assert captured == [
        ("buffer", 0.07, 1.25),
        ("inv-amp", 0.41, 9.5),
    ]
    assert [result.reward for result in results] == [0.07, 0.41]


def test_configured_environment_overrides_batch_request_defaults(monkeypatch) -> None:
    captured: list[tuple[float, float]] = []

    def fake_score_batch(requests, *, max_workers=None):
        del max_workers
        captured.extend((request.eval_tol, request.timeout_s) for request in requests)
        return [_result(1.0) for _request in requests]

    monkeypatch.setattr(environment, "score_batch", fake_score_batch)
    configured = PhyChipEnv(eval_tol=0.10, timeout_s=5.0, max_workers=2)
    configured.score_batch(
        [ScoreRequest("buffer", {}, "completion", eval_tol=0.30, timeout_s=15.0)]
    )
    assert captured == [(0.10, 5.0)]


def test_configured_score_and_batch_are_equivalent(monkeypatch) -> None:
    calls: list[tuple[str, float, float]] = []

    def fake_score(circuit_id, _targets, _completion, *, eval_tol, timeout_s):
        calls.append((circuit_id, eval_tol, timeout_s))
        return _result(float(len(calls)))

    monkeypatch.setattr(environment, "score", fake_score)
    monkeypatch.setattr(batching, "score", fake_score)
    configured = PhyChipEnv(eval_tol=0.12, timeout_s=4.5, max_workers=1)
    request = ScoreRequest("buffer", {}, "completion", eval_tol=0.30, timeout_s=15.0)

    scalar = configured.score(request.circuit_id, request.target_specs, request.completion)
    batch = configured.score_batch([request])[0]

    assert scalar.to_event() | {"reward": 0.0} == batch.to_event() | {"reward": 0.0}
    assert calls == [("buffer", 0.12, 4.5), ("buffer", 0.12, 4.5)]


def test_batch_propagates_scoring_exception(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise TimeoutError("scorer timed out")

    monkeypatch.setattr(batching, "score", fail)
    with pytest.raises(TimeoutError, match="scorer timed out"):
        batching.score_batch([ScoreRequest("buffer", {}, "completion")], max_workers=1)
