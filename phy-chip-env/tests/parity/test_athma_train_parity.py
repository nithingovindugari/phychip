from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from phy_chip_env import score as packaged_score

REPO_ROOT = Path(__file__).resolve().parents[3]
ATHMA_ROOT = REPO_ROOT / "athma-train"
if str(ATHMA_ROOT) not in sys.path:
    sys.path.insert(0, str(ATHMA_ROOT))

from athma_train.env.rlvr_scorer import score_completion as athma_score  # noqa: E402

ATTACK_TARGETS = runpy.run_path(
    str(REPO_ROOT / "phy-chip-env" / "tests" / "integration" / "test_known_reward_attacks.py")
)["TARGETS"]


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parity
@pytest.mark.integration
def test_public_attack_corpus_matches_athma() -> None:
    attacks = _rows(REPO_ROOT / "phy-chip-env" / "tests" / "fixtures" / "known_reward_attacks.jsonl")
    assert len(attacks) == 22
    for row in attacks:
        completion = f"```spice\n{row['netlist']}\n```"
        targets = ATTACK_TARGETS[row["circuit_id"]]
        expected = athma_score(row["circuit_id"], targets, completion)
        actual = packaged_score(row["circuit_id"], targets, completion)
        assert actual.to_event() == expected.to_event(), row["hack_type"]


@pytest.mark.parity
@pytest.mark.integration
def test_53_reference_rows_and_106_retained_completions_match_athma() -> None:
    holdout_path = REPO_ROOT / "data" / "rlvr_rank64_smoke_v1" / "holdout.jsonl"
    if not holdout_path.exists():
        pytest.skip("private retained-completion corpus is not distributed publicly")
    holdout = _rows(holdout_path)
    assert len(holdout) == 53

    passing_reference_ids: list[str] = []
    blocked_reference_ids: list[str] = []
    for row in holdout:
        row_id = (row.get("data_quality") or {}).get("spec_id", "missing-spec-id")
        completion = f"```spice\n{row['reference_netlist']}\n```"
        expected = athma_score(row["circuit_id"], row["target_measurements"], completion)
        actual = packaged_score(row["circuit_id"], row["target_measurements"], completion)
        assert actual.to_event() == expected.to_event(), row_id
        (passing_reference_ids if actual.all_specs_met else blocked_reference_ids).append(row_id)

    assert len(passing_reference_ids) == 44
    assert blocked_reference_ids == [
        "spec_0ce284c31e124387f937",
        "spec_0d22fb5d0c7f1e6fae21",
        "spec_0e603bce43bde456d32d",
        "spec_0f6c66bc6d78e183e14e",
        "spec_0f7cf88cfb66478e7e2d",
        "spec_13c46a8235651f98368c",
        "spec_17402d9b8e850fd5eb50",
        "spec_1b35575ded91f6c853eb",
        "spec_1d567bd47c8ef6a22d27",
    ]

    for filename in (
        "baseline_holdout_current_auditable.jsonl",
        "trained_holdout_current_auditable.jsonl",
    ):
        rows = _rows(REPO_ROOT / "eval_results" / "grpo_rank64_smoke_v1" / filename)
        assert len(rows) == 53
        for index, row in enumerate(rows):
            expected = athma_score(
                row["circuit_id"], row["target_measurements"], row["completion_vector"][0]
            )
            actual = packaged_score(
                row["circuit_id"], row["target_measurements"], row["completion_vector"][0]
            )
            assert actual.to_event() == expected.to_event(), (filename, index)
