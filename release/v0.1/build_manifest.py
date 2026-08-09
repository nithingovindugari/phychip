#!/usr/bin/env python3
"""Build content-addressed v0.1 evidence and repository checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RELEASE = Path(__file__).resolve().parent
SOURCE = REPO / "phy-chip-env" / "src"
SOURCE_STATE_EXCLUDED_PATHS = {
    "release/v0.1/RELEASE_CHECKLIST.md",
    "release/v0.1/SHA256SUMS",
    "release/v0.1/evidence_manifest.json",
    "release/v0.1/phy_chip_env_runtime_fingerprint.json",
}
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


ARTIFACTS = {
    "clean_sft_adapter": (
        "checkpoints/sft_clean_v1/adapter_model.safetensors",
        "external-release",
    ),
    "clean_sft_adapter_config": (
        "checkpoints/sft_clean_v1/adapter_config.json",
        "external-release",
    ),
    "clean_sft_chat_template": (
        "checkpoints/sft_clean_v1/chat_template.jinja",
        "external-release",
    ),
    "clean_sft_corpus": (
        "data/final_data/sft_v0_verified_v1/sft_training_ready.jsonl",
        "external-release",
    ),
    "decontamination_report": (
        "data/final_data/sft_v0_verified_v1/decontam_report.json",
        "external-release",
    ),
    "dynamic_data_report": (
        "data/final_data/sft_v0_verified_v1/sft_dynamic_report.json",
        "external-release",
    ),
    "analogcoder_clean_sft_summary": (
        "eval_results/sft_clean_v1/analogcoder_sft_chat/summary.json",
        "maintainer-evidence",
    ),
    "analogcoder_clean_sft_details": (
        "eval_results/sft_clean_v1/analogcoder_sft_chat/details.json",
        "maintainer-evidence",
    ),
    "bench_v1_clean_sft_summary": (
        "eval_results/sft_clean_v1/bench_v1_sft_chat/summary.json",
        "maintainer-evidence",
    ),
    "bench_v1_clean_sft_details": (
        "eval_results/sft_clean_v1/bench_v1_sft_chat/details.json",
        "maintainer-evidence",
    ),
    "bench_v2_clean_sft_summary": (
        "eval_results/sft_clean_v1/bench_v2_sft_chat/summary.json",
        "maintainer-evidence",
    ),
    "bench_v2_clean_sft_details": (
        "eval_results/sft_clean_v1/bench_v2_sft_chat/details.json",
        "maintainer-evidence",
    ),
    "physics_mmlu_clean_sft_comparison": (
        "eval_results/sft_clean_v1/physics_mmlu_comparison.json",
        "maintainer-evidence",
    ),
    "analogcoder_grpo_v3_summary": (
        "eval_results/grpo_v3_corrected/analogcoder_sft_chat/summary.json",
        "maintainer-evidence",
    ),
    "bench_v1_grpo_v3_summary": (
        "eval_results/grpo_v3_corrected/bench_v1_sft_chat/summary.json",
        "maintainer-evidence",
    ),
    "bench_v2_grpo_v3_summary": (
        "eval_results/grpo_v3_corrected/bench_v2_sft_chat/summary.json",
        "maintainer-evidence",
    ),
    "rank64_matched_comparison": (
        "eval_results/grpo_rank64_smoke_v1/matched_current_rescore_v2_comparison.json",
        "maintainer-evidence",
    ),
    "rank64_baseline_completions": (
        "eval_results/grpo_rank64_smoke_v1/baseline_holdout_current_rescore_v2.jsonl",
        "maintainer-evidence",
    ),
    "rank64_trained_completions": (
        "eval_results/grpo_rank64_smoke_v1/trained_holdout_current_rescore_v2.jsonl",
        "maintainer-evidence",
    ),
    "rank64_manual_audit": (
        "eval_results/grpo_rank64_smoke_v1/MANUAL_AUDIT_CURRENT_V2.md",
        "maintainer-evidence",
    ),
    "scorer_v9_pilot_audit": (
        "poc_experiments/results/rlvr_candidate_v8_audit_2026-07-22.md",
        "repository",
    ),
    "scorer_v9_diagnosis": (
        "poc_experiments/results/rlvr_diagnosis_2026-07-22.md",
        "repository",
    ),
    "scorer_v9_pilot_summary": (
        "poc_experiments/results/rlvr_scorer_v9_pilot_summary.json",
        "repository",
    ),
    "feedback_v3_benchmark": (
        "phy-chip-env/benchmarks/feedback_use_v3.json",
        "repository",
    ),
    "feedback_v3_summary": (
        "poc_experiments/results/feedback_use_v3_summary_2026-08-03.json",
        "repository",
    ),
    "feedback_v3_manual_audit": (
        "poc_experiments/results/feedback_use_v3_manual_audit_2026-08-03.json",
        "repository",
    ),
    "parameter_search_v1_result": (
        "phy-chip-env/benchmarks/parameter_search_v1_result.json",
        "repository",
    ),
    "phy_chip_env_wheel": (
        "phy-chip-env/dist/phy_chip_env-0.1.0-py3-none-any.whl",
        "external-release",
    ),
}


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def git_paths(*args: str) -> set[str]:
    output = subprocess.run(
        ["git", *args, "-z"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    return {
        value.decode("utf-8", errors="surrogateescape")
        for value in output.split(b"\0")
        if value
    }


def working_tree_state() -> tuple[int, str]:
    paths = (
        git_paths("diff", "--name-only", "HEAD")
        | git_paths("ls-files", "--others", "--exclude-standard")
    ) - SOURCE_STATE_EXCLUDED_PATHS
    records = []
    for relative in sorted(paths):
        path = REPO / relative
        if path.is_symlink():
            kind = "symlink"
            content_sha256 = hashlib.sha256(
                os.readlink(path).encode("utf-8", errors="surrogateescape")
            ).hexdigest()
            mode = None
        elif path.is_file():
            kind = "file"
            content_sha256 = sha256(path)
            mode = path.stat().st_mode & 0o777
        else:
            kind = "deleted"
            content_sha256 = None
            mode = None
        records.append(
            {
                "path": relative,
                "kind": kind,
                "mode": mode,
                "sha256": content_sha256,
            }
        )
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8", errors="surrogateescape")
    return len(paths), hashlib.sha256(encoded).hexdigest()


def artifact_path(relative: str, distribution: str, evidence_root: Path) -> Path:
    repository_path = REPO / relative
    if distribution == "repository" or repository_path.is_file():
        return repository_path
    return evidence_root / relative


def main() -> None:
    from phy_chip_env import runtime_fingerprint

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--draft",
        action="store_true",
        help="build review-only metadata from a dirty local feature branch",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=REPO,
        help="private source tree containing non-repository evidence",
    )
    args = parser.parse_args()
    evidence_root = args.evidence_root.resolve()

    dirty_path_count, source_state_sha256 = working_tree_state()
    if dirty_path_count and not args.draft:
        raise SystemExit(
            "release source tree must be clean before manifest generation: "
            f"{dirty_path_count} dirty paths"
        )

    resolved = {
        name: artifact_path(relative, distribution, evidence_root)
        for name, (relative, distribution) in ARTIFACTS.items()
    }
    missing = [str(path) for path in resolved.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing release evidence: {missing}")

    artifacts = {}
    for name, (relative, distribution) in ARTIFACTS.items():
        artifacts[name] = {
            "distribution": distribution,
            "path": relative,
            "sha256": sha256(resolved[name]),
        }

    manifest = {
        "schema": "phychip-v0.1-evidence-manifest-v2",
        "release_status": (
            "draft_from_dirty_feature_branch"
            if dirty_path_count
            else "built_from_clean_source_commit"
        ),
        "source_git": {
            "branch": git_output("branch", "--show-current"),
            "head": git_output("rev-parse", "HEAD"),
            "working_tree_dirty": bool(dirty_path_count),
            "dirty_path_count": dirty_path_count,
            "working_tree_state_sha256": source_state_sha256,
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
            "feedback_use_v3": {
                "paired_tasks": 24,
                "circuit_families": 8,
                "full_feedback_successes": 19,
                "feedback_blind_successes": 10,
                "family_positive": 5,
                "family_tied": 3,
                "family_exact_sign_p": 0.0625,
                "controller_error_arms": 10,
                "topology_plausible_successes": 28,
                "component_practical_successes": 27,
                "promotion_gate_passed": False,
                "sealed_or_training_eligible": False,
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

    # SHA256SUMS covers immutable files present in a public checkout. The
    # operational checklist remains mutable through tag/publication, and
    # external assets are committed separately in evidence_manifest.json.
    checksum_paths = [
        path
        for path in sorted(RELEASE.iterdir())
        if path.is_file()
        and path.name not in {"SHA256SUMS", "RELEASE_CHECKLIST.md"}
    ]
    lines = [
        f"{sha256(path)}  {path.relative_to(REPO)}" for path in checksum_paths
    ]
    (RELEASE / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
