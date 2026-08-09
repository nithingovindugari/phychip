"""Command-line interface for scoring, inventory, and runtime diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .campaign import preflight_sky130_campaign
from .observation import observe_completion
from .provenance import runtime_fingerprint
from .readiness import assess_training_readiness
from .registry import list_harnesses
from .research import (
    CodexCLIController,
    EpisodeLog,
    FeedbackBlindCodexCLIController,
    ResearchRunner,
    compare_benchmark_runs,
    compare_feedback_runs,
    load_benchmark,
    run_benchmark,
    run_fixed_topology,
    run_paired_feedback_benchmark,
)
from .scorer import score


def _targets(value: str) -> dict:
    candidate = Path(value)
    raw = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("targets must decode to a JSON object")
    return parsed


def _emit_json(result: dict, output: Path | None = None) -> None:
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(encoded, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _add_score_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--circuit-id", required=True)
    parser.add_argument("--targets", required=True, type=_targets)
    parser.add_argument("--completion", required=True, type=Path)
    parser.add_argument("--eval-tol", type=float, default=0.30)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--require-success", action="store_true")


def _run_score(args: argparse.Namespace) -> int:
    result = score(
        args.circuit_id,
        args.targets,
        args.completion.read_text(encoding="utf-8"),
        eval_tol=args.eval_tol,
        timeout_s=args.timeout,
    )
    print(json.dumps(result.to_event(), indent=2, sort_keys=True))
    return 1 if args.require_success and not result.all_specs_met else 0


def _run_observe(args: argparse.Namespace) -> int:
    result = observe_completion(
        args.circuit_id,
        args.targets,
        args.completion.read_text(encoding="utf-8"),
        eval_tol=args.eval_tol,
        timeout_s=args.timeout,
    )
    print(json.dumps(result.to_event(), indent=2, sort_keys=True))
    return 1 if args.require_success and not result.score.all_specs_met else 0


def score_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phy-chip-score")
    _add_score_arguments(parser)
    return _run_score(parser.parse_args(argv))


def _run_research(args: argparse.Namespace) -> int:
    metadata, tasks = load_benchmark(args.benchmark)
    matches = [task for task in tasks if task.task_id == args.task_id]
    if len(matches) != 1:
        raise ValueError(f"task_id {args.task_id!r} not found exactly once")
    task = matches[0]
    log = EpisodeLog(args.log)
    if args.controller in {"fixed", "tpe", "random", "lhs"}:
        strategy = "tpe" if args.controller == "fixed" else args.controller
        result = run_fixed_topology(
            task,
            log,
            strategy=strategy,
            rollout_id=args.rollout_id,
        )
    else:
        controller_class = (
            FeedbackBlindCodexCLIController
            if args.controller == "codex-blind"
            else CodexCLIController
        )
        controller = controller_class(model=args.model, timeout_s=args.codex_timeout)
        result = ResearchRunner().run(
            task,
            controller,
            log,
            rollout_id=args.rollout_id,
        )
    print(
        json.dumps(
            {
                "benchmark_id": metadata["benchmark_id"],
                "task_set_sha256": metadata["task_set_sha256"],
                "result": result,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["stop_reason"] == "all_specs_met" else 1


def _run_training_readiness(args: argparse.Namespace) -> int:
    result = assess_training_readiness(
        args.benchmark,
        args.episode_log,
        approval_manifest=args.approvals,
        teacher_token_distributions_artifact=args.teacher_token_distributions_artifact,
        required_episodes=args.required_episodes,
        required_families=args.required_families,
        required_scale_episodes=args.required_scale_episodes,
        required_scale_families=args.required_scale_families,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gates"][args.require_gate]["passed"] else 1


def _run_sky130_preflight(args: argparse.Namespace) -> int:
    result = preflight_sky130_campaign(
        args.campaign,
        pdk_root=args.pdk_root,
        sealed_eval_manifest=args.sealed_eval_manifest,
        release_evidence=args.release_evidence,
        design_m1_evidence=args.design_m1_evidence,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ready"] else 1


def _run_research_batch(args: argparse.Namespace) -> int:
    result = run_benchmark(
        args.benchmark,
        args.output_directory,
        controller=args.controller,
        model=args.model,
        codex_timeout_s=args.codex_timeout,
        max_workers=args.max_workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["tasks"] <= 0:
        return 1
    infrastructure_failures = sum(
        item.get("stop_reason") == "controller_error"
        for item in result.get("results", {}).values()
    )
    return 1 if infrastructure_failures else 0


def _run_paired_feedback_batch(args: argparse.Namespace) -> int:
    result = run_paired_feedback_benchmark(
        args.benchmark,
        args.output_directory,
        model=args.model,
        codex_timeout_s=args.codex_timeout,
        schedule_seed=args.schedule_seed,
        max_workers=args.max_workers,
        rollout_id=args.rollout_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    infrastructure_failures = sum(
        arm.get("stop_reason") == "controller_error"
        for item in result.get("results", {}).values()
        for arm in (item.get("full_feedback", {}), item.get("feedback_blind", {}))
    )
    return 1 if infrastructure_failures else 0


def _run_research_compare(args: argparse.Namespace) -> int:
    result = compare_benchmark_runs(
        args.benchmark,
        args.model_directory,
        args.baseline_directory,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _run_feedback_compare(args: argparse.Namespace) -> int:
    result = compare_feedback_runs(
        args.benchmark,
        args.full_feedback_directory,
        args.feedback_blind_directory,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        manual_audit_path=args.manual_audit,
    )
    _emit_json(result, args.output)
    if args.require_gate and not result["feedback_advantage_gate"]["passed"]:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phy-chip-env")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="verify and fingerprint ngspice-46")
    doctor.add_argument("--ngspice-bin", default="ngspice")
    subparsers.add_parser("capabilities", help="print packaged harness metadata")
    score_parser = subparsers.add_parser("score", help="score one completion")
    _add_score_arguments(score_parser)
    observe_parser = subparsers.add_parser(
        "observe",
        help="score once and print raw diagnostic measurements",
    )
    _add_score_arguments(observe_parser)
    research_parser = subparsers.add_parser(
        "research",
        help="run or resume one verifier-bound fixed or Codex-CLI episode",
    )
    research_parser.add_argument("--benchmark", required=True, type=Path)
    research_parser.add_argument("--task-id", required=True)
    research_parser.add_argument(
        "--controller",
        choices=("fixed", "tpe", "random", "lhs", "codex", "codex-blind"),
        required=True,
    )
    research_parser.add_argument("--log", required=True, type=Path)
    research_parser.add_argument("--rollout-id", default="0")
    research_parser.add_argument("--model", default="gpt-5.6-sol")
    research_parser.add_argument("--codex-timeout", type=float, default=300.0)
    research_batch_parser = subparsers.add_parser(
        "research-batch",
        help="run or resume every frozen benchmark task",
    )
    research_batch_parser.add_argument("--benchmark", required=True, type=Path)
    research_batch_parser.add_argument(
        "--controller",
        choices=("fixed", "tpe", "random", "lhs", "codex", "codex-blind"),
        required=True,
    )
    research_batch_parser.add_argument(
        "--output-directory",
        required=True,
        type=Path,
    )
    research_batch_parser.add_argument("--model", default="gpt-5.6-sol")
    research_batch_parser.add_argument("--codex-timeout", type=float, default=300.0)
    research_batch_parser.add_argument("--max-workers", type=int, default=1)
    paired_batch_parser = subparsers.add_parser("feedback-paired-batch")
    paired_batch_parser.add_argument("--benchmark", required=True, type=Path)
    paired_batch_parser.add_argument("--output-directory", required=True, type=Path)
    paired_batch_parser.add_argument("--model", default="gpt-5.6-sol")
    paired_batch_parser.add_argument("--codex-timeout", type=float, default=300.0)
    paired_batch_parser.add_argument("--schedule-seed", type=int, default=20260801)
    paired_batch_parser.add_argument("--max-workers", type=int, default=1)
    paired_batch_parser.add_argument("--rollout-id", default="0")
    compare_parser = subparsers.add_parser(
        "research-compare",
        help="compare complete model and fixed-topology benchmark directories",
    )
    compare_parser.add_argument("--benchmark", required=True, type=Path)
    compare_parser.add_argument("--model-directory", required=True, type=Path)
    compare_parser.add_argument("--baseline-directory", required=True, type=Path)
    compare_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    compare_parser.add_argument("--seed", type=int, default=20260729)
    feedback_compare_parser = subparsers.add_parser(
        "feedback-compare",
        help="compare paired full-feedback and feedback-blind Codex runs",
    )
    feedback_compare_parser.add_argument("--benchmark", required=True, type=Path)
    feedback_compare_parser.add_argument(
        "--full-feedback-directory",
        required=True,
        type=Path,
    )
    feedback_compare_parser.add_argument(
        "--feedback-blind-directory",
        required=True,
        type=Path,
    )
    feedback_compare_parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
    )
    feedback_compare_parser.add_argument("--seed", type=int, default=20260801)
    feedback_compare_parser.add_argument(
        "--manual-audit",
        type=Path,
        help="hash-bound human review required before a promotion gate can pass",
    )
    feedback_compare_parser.add_argument("--output", type=Path)
    feedback_compare_parser.add_argument("--require-gate", action="store_true")
    readiness_parser = subparsers.add_parser(
        "training-readiness",
        help="fail closed on trajectory, distillation, and multi-turn RLVR gates",
    )
    readiness_parser.add_argument("--benchmark", required=True, type=Path)
    readiness_parser.add_argument(
        "--episode-log",
        required=True,
        action="append",
        type=Path,
    )
    readiness_parser.add_argument("--approvals", type=Path)
    readiness_parser.add_argument(
        "--teacher-token-distributions-artifact",
        type=Path,
    )
    readiness_parser.add_argument("--required-episodes", type=int, default=100)
    readiness_parser.add_argument("--required-families", type=int, default=8)
    readiness_parser.add_argument("--required-scale-episodes", type=int, default=500)
    readiness_parser.add_argument("--required-scale-families", type=int, default=23)
    readiness_parser.add_argument(
        "--require-gate",
        choices=(
            "trajectory_audit_pilot",
            "trajectory_scale",
            "trajectory_pilot",
            "agentic_rft_rejection_sft_sequence_distillation",
            "on_policy_distillation",
            "multi_turn_rlvr",
        ),
        default="trajectory_pilot",
        help="gate that controls the process exit status",
    )
    sky130_parser = subparsers.add_parser(
        "sky130-preflight",
        help="verify the pinned PDK and prerequisites before a research campaign",
    )
    sky130_parser.add_argument("--campaign", required=True, type=Path)
    sky130_parser.add_argument("--pdk-root", type=Path)
    sky130_parser.add_argument("--sealed-eval-manifest", type=Path)
    sky130_parser.add_argument("--release-evidence", type=Path)
    sky130_parser.add_argument("--design-m1-evidence", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.command == "doctor":
            print(json.dumps(runtime_fingerprint(args.ngspice_bin), indent=2, sort_keys=True))
            return 0
        if args.command == "capabilities":
            print(
                json.dumps(
                    [
                        {
                            "circuit_id": info.circuit_id,
                            "tier": info.tier,
                            "fom_names": info.fom_names,
                            "spec_units": info.spec_units,
                        }
                        for info in list_harnesses()
                    ],
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "observe":
            return _run_observe(args)
        if args.command == "research":
            return _run_research(args)
        if args.command == "research-batch":
            return _run_research_batch(args)
        if args.command == "feedback-paired-batch":
            return _run_paired_feedback_batch(args)
        if args.command == "research-compare":
            return _run_research_compare(args)
        if args.command == "feedback-compare":
            return _run_feedback_compare(args)
        if args.command == "training-readiness":
            return _run_training_readiness(args)
        if args.command == "sky130-preflight":
            return _run_sky130_preflight(args)
        return _run_score(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
