#!/usr/bin/env python3
"""Fail closed when public or maintainer v0.1 evidence drifts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RELEASE = Path(__file__).resolve().parent
SOURCE = REPO / "phy-chip-env" / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from phy_chip_env.research import fixed_search_controller_id, load_benchmark

ALLOWED_POST_SOURCE_PATHS = {
    "release/v0.1/RELEASE_CHECKLIST.md",
    "release/v0.1/SHA256SUMS",
    "release/v0.1/evidence_manifest.json",
    "release/v0.1/phy_chip_env_runtime_fingerprint.json",
}
REQUIRED_RELEASE_FILES = {
    "BENCHMARK_CONTAMINATION.md",
    "DATA_AND_LICENSE_NOTICE.md",
    "EVIDENCE.md",
    "LICENSE_WEIGHTS_DATA.txt",
    "MODEL_CARD.md",
    "RELEASE_CHECKLIST.md",
    "REPRODUCE.md",
    "SHA256SUMS",
    "TECHNICAL_NOTE.md",
    "build_manifest.py",
    "eval-environment.lock.json",
    "evidence_manifest.json",
    "phy_chip_env_runtime_fingerprint.json",
    "validate_release.py",
}
UNCHECKSUMMED_RELEASE_FILES = {
    "RELEASE_CHECKLIST.md",
    "SHA256SUMS",
}
SOURCE_STATE_EXCLUDED_PATHS = ALLOWED_POST_SOURCE_PATHS
REQUIRED_ARTIFACT_KEYS = {
    "analogcoder_clean_sft_details",
    "analogcoder_clean_sft_summary",
    "analogcoder_grpo_v3_summary",
    "bench_v1_clean_sft_details",
    "bench_v1_clean_sft_summary",
    "bench_v1_grpo_v3_summary",
    "bench_v2_clean_sft_details",
    "bench_v2_clean_sft_summary",
    "bench_v2_grpo_v3_summary",
    "clean_sft_adapter",
    "clean_sft_adapter_config",
    "clean_sft_chat_template",
    "clean_sft_corpus",
    "decontamination_report",
    "dynamic_data_report",
    "feedback_v3_benchmark",
    "feedback_v3_manual_audit",
    "feedback_v3_summary",
    "parameter_search_v1_result",
    "phy_chip_env_wheel",
    "physics_mmlu_clean_sft_comparison",
    "rank64_baseline_completions",
    "rank64_manual_audit",
    "rank64_matched_comparison",
    "rank64_trained_completions",
    "scorer_v9_diagnosis",
    "scorer_v9_pilot_audit",
    "scorer_v9_pilot_summary",
}
PUBLIC_FORBIDDEN_PATH_PREFIXES = (
    "athma-frontend/",
    "content/",
    "docs/files/",
    "docs/interview-prep/",
    "final_report/",
    "local/",
    "paper/",
    "plan/",
    "private/",
)
PUBLIC_FORBIDDEN_FILENAMES = {
    ".env",
    ".prime_pod_id",
    ".runpod_pod_v3",
    "datasets.md",
    "learning.md",
    "me.md",
    "pending.md",
    "plan_2.0.md",
    "progress.md",
    "research.md",
    "someday.md",
    "teaching.md",
    "temp.md",
}
PUBLIC_FORBIDDEN_SUFFIXES = {".key", ".pdf", ".pptx"}
PUBLIC_FORBIDDEN_CONTENT = {
    "absolute user-home path": re.compile(rb"/(?:Users|home)/[^/\s]+/"),
    "SSH wildcard host": re.compile(rb"(?mi)^\s*Host\s+\*\s*(?:#.*)?$"),
    "SSH real-IP host": re.compile(
        rb"(?mi)^\s*HostName\s+(?:\d{1,3}\.){3}\d{1,3}\s*(?:#.*)?$"
    ),
    "private key": re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----"),
    "credential assignment": re.compile(
        rb"(?i)(?:PI_API_KEY|WANDB_API_KEY|HF_TOKEN|OPENAI_API_KEY)\s*=\s*[^\s]+"
    ),
    "absent private runner": re.compile(rb"plan/sft_clean_eval_pod_run\.sh"),
}


class ReleaseValidationError(RuntimeError):
    """Raised when any release invariant is false."""


def require(condition: bool, detail: object) -> None:
    if not condition:
        raise ReleaseValidationError(str(detail))


def load(root: Path, relative: str) -> dict:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def public_boundary_violations(entries: dict[str, bytes]) -> list[str]:
    """Return deterministic reasons a tracked entry cannot ship publicly."""

    violations: list[str] = []
    for relative, content in sorted(entries.items()):
        path = Path(relative)
        if any(relative.startswith(prefix) for prefix in PUBLIC_FORBIDDEN_PATH_PREFIXES):
            violations.append(f"forbidden public path: {relative}")
        if path.name in PUBLIC_FORBIDDEN_FILENAMES:
            violations.append(f"forbidden public filename: {relative}")
        if path.suffix.lower() in PUBLIC_FORBIDDEN_SUFFIXES:
            violations.append(f"forbidden public artifact type: {relative}")
        for label, pattern in PUBLIC_FORBIDDEN_CONTENT.items():
            if pattern.search(content):
                violations.append(f"{label}: {relative}")
    return violations


def assert_public_boundary() -> None:
    tracked = git_paths("ls-files")
    entries = {
        relative: (REPO / relative).read_bytes()
        for relative in tracked
        if (REPO / relative).is_file()
    }
    violations = public_boundary_violations(entries)
    require(not violations, "; ".join(violations))


def assert_score(root: Path, relative: str, passed: int, total: int) -> None:
    value = load(root, relative)
    require(value["n_passed"] == passed, f"{relative}: n_passed")
    require(value["n_problems"] == total, f"{relative}: n_problems")


def assert_manifest_claims(manifest: dict, draft: bool) -> None:
    require(
        manifest["schema"] == "phychip-v0.1-evidence-manifest-v2",
        "evidence manifest schema",
    )
    expected_status = (
        "draft_from_dirty_feature_branch"
        if draft
        else "built_from_clean_source_commit"
    )
    require(manifest["release_status"] == expected_status, "release status")
    require(
        manifest["source_git"]["working_tree_dirty"] is draft,
        "recorded working-tree state",
    )
    if draft:
        current_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        current_dirty_count, current_state_sha256 = working_tree_state()
        require(manifest["source_git"]["head"] == current_head, "draft source HEAD")
        require(
            manifest["source_git"]["dirty_path_count"] == current_dirty_count,
            "draft dirty-path count",
        )
        require(
            manifest["source_git"]["working_tree_state_sha256"]
            == current_state_sha256,
            "draft working-tree content state",
        )
    else:
        require(
            manifest["source_git"]["dirty_path_count"] == 0,
            "clean source dirty-path count",
        )
        empty_state_sha256 = hashlib.sha256(b"[]").hexdigest()
        require(
            manifest["source_git"]["working_tree_state_sha256"]
            == empty_state_sha256,
            "clean source state hash",
        )

    claims = manifest["claims"]
    require(
        claims["clean_sft"]
        == {
            "analogcoder_raw_spice_proxy": {"passed": 21, "n": 24},
            "bench_v1": {"passed": 23, "n": 40},
            "bench_v2": {"passed": 25, "n": 46},
            "physics_mmlu_noninferiority_established": False,
        },
        "clean-SFT claims",
    )
    require(
        claims["corrected_grpo_v3"]
        == {
            "analogcoder_raw_spice_proxy": {"passed": 16, "n": 24},
            "bench_v1": {"passed": 19, "n": 40},
            "bench_v2": {"passed": 10, "n": 46},
            "release_candidate_accepted": False,
        },
        "corrected-GRPO claims",
    )
    require(
        claims["matched_rank64"]
        == {
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
        "matched rank-64 claims",
    )
    require(
        claims["scorer_v9_pilot"]
        == {
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
        "scorer-v9 pilot claims",
    )
    require(
        claims["feedback_use_v3"]
        == {
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
        "feedback-use-v3 claims",
    )


def assert_source_commit(manifest: dict) -> None:
    source_head = manifest["source_git"]["head"]
    subprocess.run(
        ["git", "cat-file", "-e", f"{source_head}^{{commit}}"],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{source_head}..HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    require(
        set(changed) <= ALLOWED_POST_SOURCE_PATHS,
        f"committed post-source changes: {changed}",
    )
    working_changes: set[str] = set()
    for command in (
        ["git", "diff", "--name-only", "HEAD"],
        ["git", "diff", "--cached", "--name-only", "HEAD"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ):
        working_changes.update(
            subprocess.run(
                command,
                cwd=REPO,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        )
    require(
        working_changes <= ALLOWED_POST_SOURCE_PATHS,
        f"uncommitted post-source changes: {sorted(working_changes)}",
    )


def assert_repository_checksums() -> None:
    actual_release_files = {
        path.name for path in RELEASE.iterdir() if path.is_file()
    }
    require(
        actual_release_files == REQUIRED_RELEASE_FILES,
        "release file inventory mismatch: "
        f"missing={sorted(REQUIRED_RELEASE_FILES - actual_release_files)}, "
        f"extra={sorted(actual_release_files - REQUIRED_RELEASE_FILES)}",
    )
    entries: dict[str, str] = {}
    for line in (RELEASE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        require(relative not in entries, f"duplicate checksum entry: {relative}")
        entries[relative] = expected
    expected_paths = {
        str((RELEASE / name).relative_to(REPO))
        for name in REQUIRED_RELEASE_FILES - UNCHECKSUMMED_RELEASE_FILES
    }
    require(set(entries) == expected_paths, "SHA256SUMS path inventory mismatch")
    for relative, expected in entries.items():
        path = REPO / relative
        require(path.is_file(), relative)
        require(sha256(path) == expected, relative)


def assert_artifacts(manifest: dict, evidence_root: Path, full: bool) -> None:
    valid_distributions = {
        "repository",
        "external-release",
        "maintainer-evidence",
    }
    artifacts = manifest["artifacts"]
    require(set(artifacts) == REQUIRED_ARTIFACT_KEYS, "artifact inventory mismatch")
    artifact_paths: set[str] = set()
    for artifact in artifacts.values():
        require(
            set(artifact) == {"distribution", "path", "sha256"},
            "artifact record shape",
        )
        distribution = artifact["distribution"]
        require(distribution in valid_distributions, distribution)
        relative = Path(artifact["path"])
        require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and bool(relative.parts),
            f"unsafe artifact path: {relative}",
        )
        require(str(relative) not in artifact_paths, f"duplicate artifact path: {relative}")
        artifact_paths.add(str(relative))
        require(
            isinstance(artifact["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]) is not None,
            f"invalid artifact SHA-256: {relative}",
        )
        repository_path = REPO / artifact["path"]
        evidence_path = evidence_root / artifact["path"]
        if distribution == "repository" or repository_path.is_file():
            path = repository_path
        else:
            path = evidence_path

        if full or distribution == "repository" or path.is_file():
            require(path.is_file(), path)
            require(sha256(path) == artifact["sha256"], path)


def assert_public_pilot() -> None:
    pilot = load(
        REPO,
        "poc_experiments/results/rlvr_scorer_v9_pilot_summary.json",
    )
    require(pilot["scorer_semantics"] == "phy_chip_env.score@v9", "pilot semantics")
    require(
        pilot["contract"]
        == {
            "k": 8,
            "spec_count": 48,
            "rollout_count": 384,
            "evaluation_blind_references": True,
            "optimizer_steps": 0,
        },
        "pilot contract",
    )
    require(pilot["results"]["mixed_prompts"] == 11, "pilot mixed prompts")
    require(pilot["results"]["mixed_families"] == 7, "pilot mixed families")
    require(pilot["results"]["exact_successes"] == 18, "pilot exact successes")
    require(
        pilot["manual_audit"]["known_invalid_v8_rows_on_v9_frontier"] == 0,
        "pilot known-invalid frontier",
    )
    attestation = pilot["manual_audit"]["attestation"]
    require(
        attestation["schema"] == "phychip-circuit-review-attestation-v1",
        "pilot attestation schema",
    )
    require(attestation["status"] == "recorded", "pilot attestation status")
    require(
        "not independently reproducible" in attestation["limitation"],
        "pilot attestation limitation",
    )
    require(
        pilot["decision"]["sufficient_for_decisive_grpo_training"] is False,
        "pilot decisive-training decision",
    )


def assert_feedback_v3() -> None:
    benchmark = load(REPO, "phy-chip-env/benchmarks/feedback_use_v3.json")
    summary = load(
        REPO,
        "poc_experiments/results/feedback_use_v3_summary_2026-08-03.json",
    )
    audit = load(
        REPO,
        "poc_experiments/results/feedback_use_v3_manual_audit_2026-08-03.json",
    )
    require(len(benchmark["tasks"]) == 24, "feedback-v3 paired-task count")
    require(
        len({task["circuit_id"] for task in benchmark["tasks"]}) == 8,
        "feedback-v3 circuit-family count",
    )
    require(summary["full_feedback"]["successes"] == 19, "feedback-v3 full successes")
    require(summary["feedback_blind"]["successes"] == 10, "feedback-v3 blind successes")
    controller_errors = sum(
        reason == "controller_error"
        for arm in ("full_feedback", "feedback_blind")
        for reason in summary[arm]["stop_reasons"].values()
    )
    require(controller_errors == 10, "feedback-v3 controller-error count")
    require(
        summary["family_exact_sign_test"]["two_sided_p"] == 0.0625,
        "feedback-v3 family exact test",
    )
    require(summary["feedback_advantage_gate"]["passed"] is False, "feedback-v3 gate")
    require(
        audit["task_set_sha256"] == benchmark["task_set_sha256"]
        and audit["protocol_sha256"] == benchmark["protocol_sha256"],
        "feedback-v3 audit binding",
    )
    require(audit["topology_plausible_successes"] == 28, "feedback-v3 topology audit")
    require(audit["component_practical_successes"] == 27, "feedback-v3 component audit")


def assert_parameter_search() -> None:
    result = load(REPO, "phy-chip-env/benchmarks/parameter_search_v1_result.json")
    benchmark = REPO / "phy-chip-env/benchmarks/closed_loop_v1.json"
    metadata, tasks = load_benchmark(benchmark)
    task_by_id = {task.task_id: task for task in tasks}
    runtime = load(REPO, "release/v0.1/phy_chip_env_runtime_fingerprint.json")
    source_root = SOURCE / "phy_chip_env"

    require(
        result["schema"] == "phychip-parameter-search-development-result-v2",
        "parameter-search schema",
    )
    require(
        result["status"]
        == "development-only-not-sealed-not-a-generalization-result",
        "parameter-search claim boundary",
    )
    require(result["task_set_sha256"] == metadata["task_set_sha256"], "parameter-search task set")
    require(result["protocol_sha256"] == metadata["protocol_sha256"], "parameter-search protocol")
    require(result["tasks"] == len(tasks) == 4, "parameter-search task count")
    require(result["maximum_simulator_calls_per_task"] == 8, "parameter-search budget")
    require(result["raw_episode_logs_distributed"] is False, "parameter-search raw-log boundary")
    require(
        result["research_source_sha256"] == sha256(source_root / "research.py"),
        "parameter-search research source",
    )
    require(
        result["cli_source_sha256"] == sha256(source_root / "cli.py"),
        "parameter-search CLI source",
    )
    require(result["verifier_hash"] == runtime["verifier_hash"], "parameter-search verifier")
    require(
        result["ngspice"]
        == {"major": 46, "binary_sha256": runtime["ngspice"]["sha256"]},
        "parameter-search ngspice",
    )

    expected_arms = {
        "random": ("random", 3, 18),
        "latin_hypercube": ("lhs", 3, 17),
        "tpe": ("tpe", 4, 21),
    }
    require(set(result["arms"]) == set(expected_arms), "parameter-search arm inventory")
    for arm_name, (strategy, successes, calls) in expected_arms.items():
        arm = result["arms"][arm_name]
        outcomes = arm["outcomes"]
        require(set(outcomes) == set(task_by_id), f"parameter-search {arm_name} tasks")
        require(
            arm["successes"] == successes == sum(row["success"] for row in outcomes.values()),
            f"parameter-search {arm_name} successes",
        )
        require(
            arm["simulator_calls"] == calls == sum(row["calls"] for row in outcomes.values()),
            f"parameter-search {arm_name} calls",
        )
        for task_id, outcome in outcomes.items():
            require(
                isinstance(outcome["success"], bool)
                and 1 <= outcome["calls"] <= result["maximum_simulator_calls_per_task"],
                f"parameter-search {arm_name}/{task_id} outcome",
            )
            require(
                outcome["controller_id"]
                == fixed_search_controller_id(task_by_id[task_id], strategy),
                f"parameter-search {arm_name}/{task_id} controller",
            )


def assert_full_evidence(evidence_root: Path) -> None:
    assert_score(
        evidence_root,
        "eval_results/sft_clean_v1/analogcoder_sft_chat/summary.json",
        21,
        24,
    )
    assert_score(
        evidence_root,
        "eval_results/sft_clean_v1/bench_v1_sft_chat/summary.json",
        23,
        40,
    )
    assert_score(
        evidence_root,
        "eval_results/sft_clean_v1/bench_v2_sft_chat/summary.json",
        25,
        46,
    )
    assert_score(
        evidence_root,
        "eval_results/grpo_v3_corrected/analogcoder_sft_chat/summary.json",
        16,
        24,
    )
    assert_score(
        evidence_root,
        "eval_results/grpo_v3_corrected/bench_v1_sft_chat/summary.json",
        19,
        40,
    )
    assert_score(
        evidence_root,
        "eval_results/grpo_v3_corrected/bench_v2_sft_chat/summary.json",
        10,
        46,
    )

    physics = load(
        evidence_root,
        "eval_results/sft_clean_v1/physics_mmlu_comparison.json",
    )
    require(physics["n"] == 785, "physics-MMLU sample count")
    require(physics["noninferior"] is False, "physics-MMLU noninferiority")
    require(
        physics["paired_bootstrap_ci95"]
        == [
            -0.033121019108280254,
            0.008917197452229299,
        ],
        "physics-MMLU confidence interval",
    )

    matched = load(
        evidence_root,
        "eval_results/grpo_rank64_smoke_v1/"
        "matched_current_rescore_v2_comparison.json",
    )
    require(matched["n"] == 53, "matched sample count")
    require(matched["baseline"]["passes"] == 3, "matched baseline passes")
    require(matched["trained"]["passes"] == 6, "matched trained passes")
    require(matched["mcnemar_exact_p"] == 0.375, "matched McNemar p")
    require(
        matched["paired_bootstrap_95_ci"]
        == [
            -0.018867924528301886,
            0.1320754716981132,
        ],
        "matched confidence interval",
    )

    corpus = (
        evidence_root
        / "data/final_data/sft_v0_verified_v1/sft_training_ready.jsonl"
    )
    require(
        sum(1 for _ in corpus.open(encoding="utf-8")) == 15660,
        "clean-SFT corpus row count",
    )
    require(
        sha256(corpus)
        == "02af3e6c58b668acec130850ea31d468fb8fd49f938272854d4d857b0fd39b9b",
        "clean-SFT corpus SHA-256",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--draft",
        action="store_true",
        help="validate review-only metadata from a dirty local feature branch",
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=REPO,
        help="private source tree containing non-repository evidence",
    )
    args = parser.parse_args()
    evidence_root = args.evidence_root.resolve()

    manifest = load(REPO, "release/v0.1/evidence_manifest.json")
    assert_public_boundary()
    assert_manifest_claims(manifest, args.draft)
    if not args.draft:
        assert_source_commit(manifest)
    assert_repository_checksums()
    assert_artifacts(manifest, evidence_root, args.full)
    assert_public_pilot()
    assert_feedback_v3()
    assert_parameter_search()
    if args.full:
        assert_full_evidence(evidence_root)

    model_card = (RELEASE / "MODEL_CARD.md").read_text(encoding="utf-8")
    required_phrases = (
        "limited research preview",
        "not established",
        "does not establish broad RL generalization",
        "CC-BY-NC-SA-4.0",
    )
    require(
        all(phrase in model_card for phrase in required_phrases),
        "model-card claim-boundary phrases",
    )
    require(
        not re.search(r"(?i)official AnalogCoder score", model_card),
        "model card must not claim an official AnalogCoder score",
    )

    scope = "full" if args.full else "public"
    if args.draft:
        scope = f"draft-{scope}"
    print(f"release validation ({scope}): PASS")


if __name__ == "__main__":
    try:
        main()
    except (
        AssertionError,
        KeyError,
        OSError,
        ReleaseValidationError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        print(f"release validation: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
