"""Fail-closed admission gates for trajectory training and multi-turn RLVR."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from .observation import observe_completion
from .research import (
    EpisodeLog,
    load_benchmark,
    research_episode_id,
    validated_episode_outcome,
    verifier_completion,
)

TRAJECTORY_APPROVAL_SCHEMA = "phychip-trajectory-approvals-v1"
TOKEN_DISTRIBUTION_MANIFEST_SCHEMA = "phychip-token-distributions-v1"
MIN_TRAJECTORY_AUDIT_EPISODES = 100
MIN_TRAJECTORY_AUDIT_FAMILIES = 8
MIN_TRAJECTORY_SCALE_EPISODES = 500
MIN_TRAJECTORY_SCALE_FAMILIES = 23
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_approvals(
    path: str | Path | None,
    *,
    task_set_sha256: str,
    protocol_sha256: str,
) -> tuple[dict[str, dict[str, Any]], str | None]:
    if path is None:
        return {}, None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != TRAJECTORY_APPROVAL_SCHEMA:
        raise ValueError("unsupported trajectory approval schema")
    if payload.get("benchmark_task_set_sha256") != task_set_sha256:
        raise ValueError("trajectory approvals do not bind the frozen benchmark")
    if payload.get("benchmark_protocol_sha256") != protocol_sha256:
        raise ValueError("trajectory approvals do not bind the frozen protocol")
    approval_values = payload.get("approvals")
    if not isinstance(approval_values, list):
        raise ValueError("trajectory approvals must be a list")
    approvals: dict[str, dict[str, Any]] = {}
    for value in approval_values:
        if not isinstance(value, dict):
            raise ValueError("each trajectory approval must be an object")
        event_hash = str(value.get("episode_end_hash", ""))
        if not _SHA256.fullmatch(event_hash):
            raise ValueError("approval episode_end_hash must be a SHA-256 digest")
        if event_hash in approvals:
            raise ValueError(f"duplicate approval for {event_hash}")
        approvals[event_hash] = value
    token_distribution_hash = payload.get("teacher_token_distributions_sha256")
    token_distributions = (
        token_distribution_hash
        if (
            payload.get("teacher_token_distributions_preserved") is True
            and isinstance(token_distribution_hash, str)
            and _SHA256.fullmatch(token_distribution_hash)
        )
        else None
    )
    return approvals, token_distributions


def _approved(value: dict[str, Any] | None, outcome: dict[str, Any]) -> bool:
    if value is None:
        return False
    reviewed_at = str(value.get("reviewed_at", "")).strip()
    try:
        timestamp = datetime.fromisoformat(reviewed_at)
    except ValueError:
        return False
    controller_ids = outcome["controller_ids"]
    expected_outcome = "success" if outcome["success"] else "failure"
    common = bool(
        value.get("decision") == "approve"
        and value.get("outcome_verified") is True
        and str(value.get("reviewer", "")).strip()
        and timestamp.tzinfo is not None
        and timestamp.utcoffset() is not None
        and value.get("episode_end_hash") == outcome["episode_end_hash"]
        and value.get("task_id") == outcome["task_id"]
        and value.get("controller_id") == controller_ids[0]
        and value.get("outcome") == expected_outcome
    )
    if not common:
        return False
    if outcome["success"]:
        return bool(
            value.get("topology_plausible") is True
            and value.get("winning_event_hash") == outcome["winning_event_hash"]
        )
    return value.get("winning_event_hash") in {None, ""}


def _verify_token_distribution_manifest(
    path: str | Path | None,
    *,
    expected_sha256: str | None,
    task_set_sha256: str,
    protocol_sha256: str,
    controller_ids: set[str],
    episode_end_hashes: set[str],
) -> tuple[bool, list[str]]:
    if expected_sha256 is None:
        return False, ["approval manifest does not pin token-distribution evidence"]
    if path is None:
        return False, ["no token-distribution manifest was supplied"]
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        return False, ["token-distribution manifest is missing"]
    if manifest_path.stat().st_size > 1_000_000:
        return False, ["token-distribution manifest exceeds 1,000,000 bytes"]
    if _sha256_file(manifest_path) != expected_sha256:
        return False, ["token-distribution manifest SHA-256 mismatch"]
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, ["token-distribution manifest is not valid JSON"]
    if payload.get("schema") != TOKEN_DISTRIBUTION_MANIFEST_SCHEMA:
        return False, ["unsupported token-distribution manifest schema"]
    if payload.get("benchmark_task_set_sha256") != task_set_sha256:
        return False, ["token distributions do not bind the frozen benchmark"]
    if payload.get("benchmark_protocol_sha256") != protocol_sha256:
        return False, ["token distributions do not bind the frozen protocol"]
    if len(controller_ids) != 1 or payload.get("controller_id") not in controller_ids:
        return False, ["token distributions do not bind one admitted controller"]
    vocabulary_sha256 = payload.get("vocabulary_sha256")
    capture_revision_sha256 = payload.get("capture_revision_sha256")
    if (
        not isinstance(vocabulary_sha256, str)
        or not _SHA256.fullmatch(vocabulary_sha256)
        or not isinstance(capture_revision_sha256, str)
        or not _SHA256.fullmatch(capture_revision_sha256)
    ):
        return False, ["token distributions lack vocabulary/capture commitments"]
    declared_episodes = payload.get("episode_end_hashes")
    if (
        not isinstance(declared_episodes, list)
        or set(declared_episodes) != episode_end_hashes
        or len(declared_episodes) != len(episode_end_hashes)
    ):
        return False, ["token distributions do not bind every admitted episode"]
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        return False, ["token-distribution manifest has no shards"]
    root = manifest_path.parent
    seen_paths: set[str] = set()
    total_examples = 0
    covered_positions: dict[str, set[int]] = defaultdict(set)
    expected_row_fields = {
        "schema",
        "episode_end_hash",
        "token_index",
        "chosen_token_id",
        "token_ids",
        "probabilities",
        "vocabulary_sha256",
        "controller_id",
    }
    for shard in shards:
        if not isinstance(shard, dict):
            return False, ["token-distribution shard entry is not an object"]
        relative_text = str(shard.get("path", ""))
        relative = Path(relative_text)
        expected = shard.get("sha256")
        examples = shard.get("examples")
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or relative_text in seen_paths
        ):
            return False, ["token-distribution shard path is unsafe or duplicated"]
        seen_paths.add(relative_text)
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root):
            return False, ["token-distribution shard escapes its manifest directory"]
        if (
            not isinstance(expected, str)
            or not _SHA256.fullmatch(expected)
            or not candidate.is_file()
            or _sha256_file(candidate) != expected
        ):
            return False, [f"token-distribution shard failed verification: {relative}"]
        if (
            not isinstance(examples, int)
            or isinstance(examples, bool)
            or examples <= 0
        ):
            return False, [f"token-distribution shard has invalid count: {relative}"]
        actual_examples = 0
        try:
            with candidate.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    if len(line.encode("utf-8")) > 1_000_000:
                        return False, [f"token-distribution row is too large: {relative}"]
                    row = json.loads(line)
                    if not isinstance(row, dict) or set(row) != expected_row_fields:
                        return False, [
                            f"token-distribution row schema mismatch: {relative}"
                        ]
                    if row.get("schema") != "phychip-token-distribution-row-v1":
                        return False, [
                            f"unsupported token-distribution row schema: {relative}"
                        ]
                    episode_hash = row.get("episode_end_hash")
                    token_index = row.get("token_index")
                    token_ids = row.get("token_ids")
                    probabilities = row.get("probabilities")
                    chosen_token_id = row.get("chosen_token_id")
                    if episode_hash not in episode_end_hashes:
                        return False, [
                            f"token-distribution row names an unknown episode: {relative}"
                        ]
                    if (
                        not isinstance(token_index, int)
                        or isinstance(token_index, bool)
                        or token_index < 0
                        or token_index in covered_positions[episode_hash]
                    ):
                        return False, [
                            f"token-distribution token index is invalid: {relative}"
                        ]
                    if (
                        not isinstance(token_ids, list)
                        or not token_ids
                        or any(
                            not isinstance(token_id, int)
                            or isinstance(token_id, bool)
                            or token_id < 0
                            for token_id in token_ids
                        )
                        or len(token_ids) != len(set(token_ids))
                        or not isinstance(probabilities, list)
                        or len(probabilities) != len(token_ids)
                        or any(
                            not isinstance(probability, int | float)
                            or isinstance(probability, bool)
                            or not math.isfinite(float(probability))
                            or not 0.0 <= float(probability) <= 1.0
                            for probability in probabilities
                        )
                        or not math.isclose(
                            sum(map(float, probabilities)),
                            1.0,
                            rel_tol=1e-6,
                            abs_tol=1e-6,
                        )
                        or chosen_token_id not in token_ids
                    ):
                        return False, [
                            f"token-distribution probabilities are invalid: {relative}"
                        ]
                    if (
                        row.get("vocabulary_sha256") != vocabulary_sha256
                        or row.get("controller_id") not in controller_ids
                    ):
                        return False, [
                            f"token-distribution row identity mismatch: {relative}"
                        ]
                    covered_positions[episode_hash].add(token_index)
                    actual_examples += 1
        except (OSError, json.JSONDecodeError):
            return False, [f"token-distribution shard is not valid JSONL: {relative}"]
        if actual_examples != examples:
            return False, [f"token-distribution shard row count mismatch: {relative}"]
        total_examples += actual_examples
    if set(covered_positions) != episode_end_hashes:
        return False, ["token distributions do not cover every admitted episode"]
    for episode_hash, positions in covered_positions.items():
        if positions != set(range(len(positions))):
            return False, [
                f"token positions are not contiguous for episode {episode_hash}"
            ]
    if total_examples < len(episode_end_hashes):
        return False, ["token distributions do not cover every admitted episode"]
    return True, []


def _trajectory_content_hash(
    circuit_id: str,
    target_specs: dict[str, list[float]],
    verifier_augmentation: str | None,
    initial_observation: dict[str, Any] | None,
    trials: list[dict[str, Any]],
) -> str:
    def normalized_netlist(value: str) -> str:
        return "\n".join(
            " ".join(line.lower().split())
            for line in value.splitlines()
            if line.strip() and not line.lstrip().startswith("*")
        )

    payload = {
        "circuit_id": circuit_id,
        "target_specs": target_specs,
        "verifier_augmentation": verifier_augmentation,
        "initial_observation": (
            {
                "topology": initial_observation.get("topology"),
                "netlist": normalized_netlist(
                    str(initial_observation.get("netlist", ""))
                ),
                "score": initial_observation["observation"]["score"],
            }
            if initial_observation is not None
            else None
        ),
        "trials": [
            {
                "topology": trial["proposal"].get("topology"),
                "netlist": normalized_netlist(str(trial["proposal"].get("netlist", ""))),
                "score": trial["observation"]["score"],
            }
            for trial in trials
        ],
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_observation_evidence(value: dict[str, Any]) -> dict[str, Any]:
    evidence = dict(value)
    failure = evidence.get("failure")
    if isinstance(failure, dict):
        evidence["failure"] = {
            "stage": failure.get("stage"),
            "code": failure.get("code"),
        }
    return evidence


def assess_training_readiness(
    benchmark_path: str | Path,
    episode_logs: Iterable[str | Path],
    *,
    approval_manifest: str | Path | None = None,
    teacher_token_distributions_artifact: str | Path | None = None,
    required_episodes: int = 100,
    required_families: int = 8,
    required_scale_episodes: int = 500,
    required_scale_families: int = 23,
    reverify_fn=None,
) -> dict[str, Any]:
    """Assess evidence without treating filenames or reward as approval."""

    if required_episodes < MIN_TRAJECTORY_AUDIT_EPISODES:
        raise ValueError(
            f"required_episodes cannot be below {MIN_TRAJECTORY_AUDIT_EPISODES}"
        )
    if required_families < MIN_TRAJECTORY_AUDIT_FAMILIES:
        raise ValueError(
            f"required_families cannot be below {MIN_TRAJECTORY_AUDIT_FAMILIES}"
        )
    if required_scale_episodes < MIN_TRAJECTORY_SCALE_EPISODES:
        raise ValueError(
            "required_scale_episodes cannot be below "
            f"{MIN_TRAJECTORY_SCALE_EPISODES}"
        )
    if required_scale_families < MIN_TRAJECTORY_SCALE_FAMILIES:
        raise ValueError(
            "required_scale_families cannot be below "
            f"{MIN_TRAJECTORY_SCALE_FAMILIES}"
        )
    metadata, tasks = load_benchmark(benchmark_path)
    task_by_id = {task.task_id: task for task in tasks}
    approvals, token_distribution_hash = _load_approvals(
        approval_manifest,
        task_set_sha256=metadata["task_set_sha256"],
        protocol_sha256=metadata["protocol_sha256"],
    )
    verifier = reverify_fn or observe_completion
    outcomes_by_task: dict[str, set[bool]] = defaultdict(set)
    controllers: set[str] = set()
    families: set[str] = set()
    episode_hashes: set[str] = set()
    episode_ids: set[str] = set()
    all_event_hashes: set[str] = set()
    trajectory_content_hashes: set[str] = set()
    outcomes: list[dict[str, Any]] = []
    successes = 0
    episodes = 0
    feedback_episodes = 0
    feedback_successes = 0
    reverified_trials = 0

    for log_path in episode_logs:
        events = EpisodeLog(log_path).read()
        outcome = validated_episode_outcome(log_path)
        end = events[-1]
        task_id = outcome["task_id"]
        if task_id not in task_by_id:
            raise ValueError(f"{log_path}: task {task_id!r} is outside the benchmark")
        task = task_by_id[task_id]
        episode_hash = str(end["event_hash"])
        if episode_hash in episode_hashes:
            raise ValueError(f"duplicate episode_end event {episode_hash}")
        episode_hashes.add(episode_hash)
        if outcome["episode_id"] in episode_ids:
            raise ValueError(f"{log_path}: duplicate episode identity")
        episode_ids.add(outcome["episode_id"])
        for event in events:
            event_hash = str(event["event_hash"])
            if event_hash in all_event_hashes:
                raise ValueError(f"{log_path}: replayed event evidence {event_hash}")
            all_event_hashes.add(event_hash)
        if len(outcome["controller_ids"]) != 1 or outcome["verifier_calls"] == 0:
            raise ValueError(f"{log_path}: trajectory lacks one scored model controller")
        controller_id = outcome["controller_ids"][0]
        if outcome["episode_id"] != research_episode_id(
            task,
            controller_id,
            outcome["rollout_id"],
        ):
            raise ValueError(f"{log_path}: episode identity is not reproducible")
        initial_events = [
            event
            for event in events
            if event.get("event_type") == "initial_observation"
        ]
        initial_event = initial_events[0] if initial_events else None
        if initial_event is not None:
            if task.initial_netlist is None or initial_event["netlist"] != task.initial_netlist:
                raise ValueError(
                    f"{log_path}: initial observation does not match the frozen task"
                )
            replayed_initial = verifier(
                task.circuit_id,
                task.target_specs,
                verifier_completion(task, task.initial_netlist),
                timeout_s=task.timeout_s,
            ).to_event()
            if _canonical_observation_evidence(
                replayed_initial
            ) != _canonical_observation_evidence(initial_event["observation"]):
                raise ValueError(
                    f"{log_path}: initial observation does not match scorer replay"
                )
        trials = [event for event in events if event.get("event_type") == "trial"]
        if any("proposal" not in event for event in trials):
            raise ValueError(f"{log_path}: trajectory is not model-proposal evidence")
        for trial in trials:
            proposal = trial["proposal"]
            replayed = verifier(
                task.circuit_id,
                task.target_specs,
                verifier_completion(task, str(proposal.get("netlist", ""))),
                timeout_s=task.timeout_s,
            ).to_event()
            if _canonical_observation_evidence(
                replayed
            ) != _canonical_observation_evidence(trial["observation"]):
                raise ValueError(
                    f"{log_path}: logged observation does not match scorer replay"
                )
        reverified_trials += len(trials) + len(initial_events)
        content_hash = _trajectory_content_hash(
            task.circuit_id,
            task.target_specs,
            task.verifier_augmentation,
            initial_event,
            trials,
        )
        if content_hash in trajectory_content_hashes:
            raise ValueError(f"{log_path}: duplicate trajectory content")
        trajectory_content_hashes.add(content_hash)
        if outcome["verifier_calls"] > task.max_verifier_calls:
            raise ValueError(f"{log_path}: verifier-call budget was exceeded")
        episodes += 1
        if outcome["feedback_episode"]:
            feedback_episodes += 1
        families.add(task.circuit_id)
        success = outcome["success"]
        outcomes_by_task[task_id].add(success)
        controllers.add(controller_id)
        outcomes.append(outcome)
        if success:
            successes += 1
            if outcome["feedback_episode"]:
                feedback_successes += 1
            winning_trial = outcome["winning_trial"]
            if winning_trial is None or "proposal" not in winning_trial:
                raise ValueError(
                    f"{log_path}: winning trial lacks canonical model-trajectory evidence"
                )

    approved_outcomes = [
        outcome
        for outcome in outcomes
        if _approved(approvals.get(outcome["episode_end_hash"]), outcome)
    ]
    approved_episodes = len(approved_outcomes)
    approved_successes = sum(outcome["success"] for outcome in approved_outcomes)
    unapproved_episodes = episodes - approved_episodes
    mixed_tasks = sorted(
        task_id
        for task_id, task_outcomes in outcomes_by_task.items()
        if task_outcomes == {False, True}
    )

    audit_reasons = []
    if episodes < required_episodes:
        audit_reasons.append(
            f"{episodes}/{required_episodes} required audited trajectories are present"
        )
    if len(families) < required_families:
        audit_reasons.append(
            f"{len(families)}/{required_families} audit circuit families are covered"
        )
    if unapproved_episodes:
        audit_reasons.append(
            f"{unapproved_episodes} trajectories lack exact-hash outcome approval"
        )
    if not successes:
        audit_reasons.append("no scorer-v9-successful trajectory is present")
    if feedback_episodes != episodes:
        audit_reasons.append(
            f"{episodes - feedback_episodes} trajectories lack frozen initial feedback"
        )
    split_role = metadata.get("split_policy", {}).get("role")
    if split_role != "trajectory_collection":
        audit_reasons.append(
            f"benchmark split role {split_role!r} is not training-eligible trajectory_collection"
        )
    if len(controllers) != 1:
        audit_reasons.append("audited trajectories do not use exactly one controller")
    if any(controller.startswith("codex-cli-feedback-blind:") for controller in controllers):
        audit_reasons.append("feedback-blind control trajectories are not training eligible")
    if episodes and successes == episodes:
        audit_reasons.append("trajectory corpus has no failed revision episodes")

    audit_ready = not audit_reasons
    scale_reasons = list(audit_reasons)
    if episodes < required_scale_episodes:
        scale_reasons.append(
            f"{episodes}/{required_scale_episodes} scale trajectories are present"
        )
    if len(families) < required_scale_families:
        scale_reasons.append(
            f"{len(families)}/{required_scale_families} scale circuit families are covered"
        )
    scale_ready = not scale_reasons
    token_distributions, token_distribution_reasons = (
        _verify_token_distribution_manifest(
            teacher_token_distributions_artifact,
            expected_sha256=token_distribution_hash,
            task_set_sha256=metadata["task_set_sha256"],
            protocol_sha256=metadata["protocol_sha256"],
            controller_ids=controllers,
            episode_end_hashes=episode_hashes,
        )
    )
    rlvr_reasons = []
    if not mixed_tasks:
        rlvr_reasons.append(
            "no task has both success and failure under the current controller"
        )
    if len(controllers) != 1:
        rlvr_reasons.append(
            "episodes do not bind to exactly one current-policy controller"
        )
    if not scale_ready:
        rlvr_reasons.append("the trajectory-scale admission gate has not passed")
    rlvr_reasons.append(
        "authenticated current-policy rollout provenance is not implemented in v0.1"
    )

    return {
        "schema": "phychip-training-readiness-v1",
        "benchmark_id": metadata["benchmark_id"],
        "task_set_sha256": metadata["task_set_sha256"],
        "protocol_sha256": metadata["protocol_sha256"],
        "counts": {
            "episodes": episodes,
            "successes": successes,
            "approved_successes": approved_successes,
            "approved_episodes": approved_episodes,
            "feedback_episodes": feedback_episodes,
            "feedback_successes": feedback_successes,
            "reverified_trials": reverified_trials,
            "families": len(families),
            "controllers": sorted(controllers),
            "mixed_current_policy_tasks": mixed_tasks,
        },
        "gates": {
            "trajectory_audit_pilot": {
                "passed": audit_ready,
                "reasons": audit_reasons,
            },
            "trajectory_scale": {
                "passed": scale_ready,
                "reasons": scale_reasons,
            },
            "trajectory_pilot": {
                "passed": scale_ready,
                "reasons": scale_reasons,
            },
            "agentic_rft_rejection_sft_sequence_distillation": {
                "passed": audit_ready,
                "reasons": (
                    []
                    if audit_ready
                    else ["the trajectory-audit admission gate has not passed"]
                ),
            },
            "on_policy_distillation": {
                "passed": audit_ready and token_distributions,
                "reasons": (
                    []
                    if audit_ready and token_distributions
                    else (
                        ([] if audit_ready else [
                            "the trajectory-audit admission gate has not passed"
                        ])
                        + token_distribution_reasons
                    )
                ),
            },
            "multi_turn_rlvr": {
                "passed": False,
                "reasons": rlvr_reasons,
            },
        },
    }
