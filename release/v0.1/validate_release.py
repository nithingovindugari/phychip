#!/usr/bin/env python3
"""Fail closed when release claims drift from their source artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RELEASE = Path(__file__).resolve().parent


def load(relative: str) -> dict:
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_score(relative: str, passed: int, total: int) -> None:
    value = load(relative)
    assert value["n_passed"] == passed, relative
    assert value["n_problems"] == total, relative


def main() -> None:
    assert_score("eval_results/sft_clean_v1/analogcoder_sft_chat/summary.json", 21, 24)
    assert_score("eval_results/sft_clean_v1/bench_v1_sft_chat/summary.json", 23, 40)
    assert_score("eval_results/sft_clean_v1/bench_v2_sft_chat/summary.json", 25, 46)
    assert_score(
        "eval_results/grpo_v3_corrected/analogcoder_sft_chat/summary.json", 16, 24
    )
    assert_score(
        "eval_results/grpo_v3_corrected/bench_v1_sft_chat/summary.json", 19, 40
    )
    assert_score(
        "eval_results/grpo_v3_corrected/bench_v2_sft_chat/summary.json", 10, 46
    )

    physics = load("eval_results/sft_clean_v1/physics_mmlu_comparison.json")
    assert physics["n"] == 785
    assert physics["noninferior"] is False
    assert physics["paired_bootstrap_ci95"] == [
        -0.033121019108280254,
        0.008917197452229299,
    ]

    matched = load(
        "eval_results/grpo_rank64_smoke_v1/matched_current_rescore_v2_comparison.json"
    )
    assert matched["n"] == 53
    assert matched["baseline"]["passes"] == 3
    assert matched["trained"]["passes"] == 6
    assert matched["mcnemar_exact_p"] == 0.375
    assert matched["paired_bootstrap_95_ci"] == [
        -0.018867924528301886,
        0.1320754716981132,
    ]

    pilot = load("poc_experiments/results/rlvr_scorer_v9_pilot_summary.json")
    assert pilot["scorer_semantics"] == "phy_chip_env.score@v9"
    assert pilot["contract"] == {
        "k": 8,
        "spec_count": 48,
        "rollout_count": 384,
        "evaluation_blind_references": True,
        "optimizer_steps": 0,
    }
    assert pilot["results"]["mixed_prompts"] == 11
    assert pilot["results"]["mixed_families"] == 7
    assert pilot["results"]["exact_successes"] == 18
    assert pilot["manual_audit"]["known_invalid_v8_rows_on_v9_frontier"] == 0
    assert pilot["decision"]["sufficient_for_decisive_grpo_training"] is False

    corpus = REPO / "data/final_data/sft_v0_verified_v1/sft_training_ready.jsonl"
    assert sum(1 for _ in corpus.open(encoding="utf-8")) == 15660
    assert (
        sha256(corpus)
        == "02af3e6c58b668acec130850ea31d468fb8fd49f938272854d4d857b0fd39b9b"
    )

    manifest = load("release/v0.1/evidence_manifest.json")
    for artifact in manifest["artifacts"].values():
        path = REPO / artifact["path"]
        assert path.is_file(), path
        assert sha256(path) == artifact["sha256"], path

    model_card = (RELEASE / "MODEL_CARD.md").read_text(encoding="utf-8")
    required_phrases = (
        "limited research preview",
        "not established",
        "does not establish broad RL generalization",
        "CC-BY-NC-SA-4.0",
    )
    assert all(phrase in model_card for phrase in required_phrases)
    assert not re.search(r"(?i)official AnalogCoder score", model_card)

    print("release validation: PASS")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, OSError, ValueError) as exc:
        print(f"release validation: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
