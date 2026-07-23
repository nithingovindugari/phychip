#!/usr/bin/env python3
"""Build content-addressed v0.1 evidence and release checksums."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RELEASE = Path(__file__).resolve().parent
SOURCE = REPO / "phy-chip-env" / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from phy_chip_env import runtime_fingerprint  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


EVIDENCE = {
    "clean_sft_adapter": "checkpoints/sft_clean_v1/adapter_model.safetensors",
    "clean_sft_adapter_config": "checkpoints/sft_clean_v1/adapter_config.json",
    "clean_sft_chat_template": "checkpoints/sft_clean_v1/chat_template.jinja",
    "clean_sft_corpus": "data/final_data/sft_v0_verified_v1/sft_training_ready.jsonl",
    "decontamination_report": "data/final_data/sft_v0_verified_v1/decontam_report.json",
    "dynamic_data_report": "data/final_data/sft_v0_verified_v1/sft_dynamic_report.json",
    "analogcoder_clean_sft_summary": "eval_results/sft_clean_v1/analogcoder_sft_chat/summary.json",
    "analogcoder_clean_sft_details": "eval_results/sft_clean_v1/analogcoder_sft_chat/details.json",
    "bench_v1_clean_sft_summary": "eval_results/sft_clean_v1/bench_v1_sft_chat/summary.json",
    "bench_v1_clean_sft_details": "eval_results/sft_clean_v1/bench_v1_sft_chat/details.json",
    "bench_v2_clean_sft_summary": "eval_results/sft_clean_v1/bench_v2_sft_chat/summary.json",
    "bench_v2_clean_sft_details": "eval_results/sft_clean_v1/bench_v2_sft_chat/details.json",
    "physics_mmlu_clean_sft_comparison": "eval_results/sft_clean_v1/physics_mmlu_comparison.json",
    "analogcoder_grpo_v3_summary": "eval_results/grpo_v3_corrected/analogcoder_sft_chat/summary.json",
    "bench_v1_grpo_v3_summary": "eval_results/grpo_v3_corrected/bench_v1_sft_chat/summary.json",
    "bench_v2_grpo_v3_summary": "eval_results/grpo_v3_corrected/bench_v2_sft_chat/summary.json",
    "rank64_matched_comparison": "eval_results/grpo_rank64_smoke_v1/matched_current_rescore_v2_comparison.json",
    "rank64_baseline_completions": "eval_results/grpo_rank64_smoke_v1/baseline_holdout_current_rescore_v2.jsonl",
    "rank64_trained_completions": "eval_results/grpo_rank64_smoke_v1/trained_holdout_current_rescore_v2.jsonl",
    "rank64_manual_audit": "eval_results/grpo_rank64_smoke_v1/MANUAL_AUDIT_CURRENT_V2.md",
    "scorer_v9_pilot_audit": "poc_experiments/results/rlvr_candidate_v8_audit_2026-07-22.md",
    "scorer_v9_diagnosis": "poc_experiments/results/rlvr_diagnosis_2026-07-22.md",
    "scorer_v9_pilot_summary": "poc_experiments/results/rlvr_scorer_v9_pilot_summary.json",
    "phy_chip_env_wheel": "phy-chip-env/dist/phy_chip_env-0.1.0-py3-none-any.whl",
}


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    missing = [
        relative for relative in EVIDENCE.values() if not (REPO / relative).is_file()
    ]
    if missing:
        raise SystemExit(f"missing release evidence: {missing}")

    dirty_paths = git_output("status", "--short").splitlines()
    artifacts = {
        name: {"path": relative, "sha256": sha256(REPO / relative)}
        for name, relative in EVIDENCE.items()
    }
    manifest = {
        "schema": "phychip-v0.1-evidence-manifest-v1",
        "release_status": "prepared_uncommitted"
        if dirty_paths
        else "clean_commit_candidate",
        "git": {
            "branch": git_output("branch", "--show-current"),
            "head": git_output("rev-parse", "HEAD"),
            "working_tree_dirty": bool(dirty_paths),
            "dirty_path_count": len(dirty_paths),
        },
        "claims": {
            "clean_sft": {
                "analogcoder_raw_spice_proxy": {"passed": 21, "n": 24},
                "bench_v1": {"passed": 23, "n": 40},
                "bench_v2": {"passed": 25, "n": 46},
                "physics_mmlu_noninferiority_established": False,
            },
            "corrected_grpo_v3": {
                "analogcoder_raw_spice_proxy": {"passed": 16, "n": 24},
                "bench_v1": {"passed": 19, "n": 40},
                "bench_v2": {"passed": 10, "n": 46},
                "release_candidate_accepted": False,
            },
            "matched_rank64": {
                "baseline": {"passed": 3, "n": 53},
                "trained": {"passed": 6, "n": 53},
                "delta_percentage_points": 5.660377358490566,
                "paired_ci95_percentage_points": [
                    -1.8867924528301887,
                    13.20754716981132,
                ],
                "mcnemar_exact_p": 0.375,
                "rl_generalization_established": False,
            },
            "scorer_v9_pilot": {
                "specs": 48,
                "rollouts": 384,
                "k": 8,
                "mixed_prompts": 11,
                "mixed_families": 7,
                "exact_successes": 18,
                "easy_prompts": 0,
                "known_invalid_frontier_rows": 0,
                "optimizer_steps": 0,
            },
        },
        "artifacts": artifacts,
    }
    (RELEASE / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fingerprint = runtime_fingerprint()
    (RELEASE / "phy_chip_env_runtime_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    checksum_paths = [
        path
        for path in sorted(RELEASE.iterdir())
        if path.is_file() and path.name not in {"SHA256SUMS"}
    ]
    checksum_paths.append(REPO / EVIDENCE["phy_chip_env_wheel"])
    lines = []
    for path in checksum_paths:
        display = path.relative_to(REPO) if path.is_relative_to(REPO) else path.name
        lines.append(f"{sha256(path)}  {display}")
    (RELEASE / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
