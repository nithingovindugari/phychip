"""Preflight contracts for PDK-backed autonomous-research campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

SKY130_CAMPAIGN_SCHEMA = "phychip-sky130-opamp-campaign-v1"
SEALED_EVAL_SCHEMA = "phychip-sealed-sky130-eval-v1"
DESIGN_M1_SCHEMA = "phychip-design-agent-m1-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _campaign_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _content_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_candidate(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("campaign artifact paths must be safe relative paths")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"campaign artifact escapes its root: {relative}")
    return candidate


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive_number(value: Any) -> bool:
    return _finite_number(value) and float(value) > 0.0


def _valid_target_specs(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    for target in value.values():
        if _finite_number(target):
            continue
        if (
            not isinstance(target, list)
            or len(target) != 2
            or not all(_finite_number(bound) for bound in target)
            or float(target[0]) > float(target[1])
        ):
            return False
    return True


def _valid_pvt(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"corner", "vdd_v", "temperature_c"}
        and value["corner"] in {"tt", "ss", "ff", "sf", "fs"}
        and _positive_number(value["vdd_v"])
        and _finite_number(value["temperature_c"])
    )


def load_sky130_campaign(path: str | Path) -> dict[str, Any]:
    """Load the preregistration and reject ambiguous campaign contracts."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != SKY130_CAMPAIGN_SCHEMA:
        raise ValueError("unsupported SKY130 campaign schema")
    topologies = payload.get("topologies")
    if not isinstance(topologies, list) or len(topologies) != 3:
        raise ValueError("the first SKY130 campaign must define exactly three topologies")
    topology_ids = [str(value.get("topology_id", "")) for value in topologies]
    if any(not value for value in topology_ids) or len(set(topology_ids)) != 3:
        raise ValueError("campaign topology_id values must be unique and non-empty")
    for topology in topologies:
        parameters = topology.get("continuous_parameters")
        if (
            not isinstance(parameters, list)
            or not parameters
            or any(not isinstance(value, str) or not value for value in parameters)
            or len(parameters) != len(set(parameters))
        ):
            raise ValueError("each topology needs unique continuous parameters")

    budgets = payload.get("budgets", {})
    for name in (
        "fixed_optimizer_calls_per_region",
        "model_guided_calls_per_region",
        "pvt_replays_per_finalist",
    ):
        if not isinstance(budgets.get(name), int) or budgets[name] <= 0:
            raise ValueError(f"campaign budget {name!r} must be a positive integer")
    if (
        budgets["fixed_optimizer_calls_per_region"]
        != budgets["model_guided_calls_per_region"]
    ):
        raise ValueError("fixed and model-guided arms must have equal call budgets")

    constraints = payload.get("constraints", {})
    if not _positive_number(constraints.get("vdd_nominal_v")):
        raise ValueError("vdd_nominal_v must be positive")
    if not _positive_number(constraints.get("quiescent_current_max_a")):
        raise ValueError("quiescent_current_max_a must be positive")
    metrics = payload.get("metrics")
    if (
        not isinstance(metrics, list)
        or not metrics
        or any(not isinstance(metric, str) or not metric for metric in metrics)
        or len(metrics) != len(set(metrics))
    ):
        raise ValueError("campaign metrics cannot be empty")
    discovery = payload.get("discovery_grid")
    if not isinstance(discovery, dict):
        raise ValueError("campaign discovery_grid is required")
    for name in (
        "load_capacitance_f",
        "minimum_open_loop_gain_db",
        "minimum_unity_gain_bandwidth_hz",
    ):
        values = discovery.get(name)
        if (
            not isinstance(values, list)
            or not values
            or any(
                not _positive_number(value)
                for value in values
            )
        ):
            raise ValueError(f"discovery_grid {name!r} must be positive")
    if not _positive_number(discovery.get("minimum_phase_margin_deg")):
        raise ValueError("minimum_phase_margin_deg must be positive")
    pvt = payload.get("pvt_replay_grid")
    if (
        not isinstance(pvt, dict)
        or set(pvt.get("corners", [])) != {"tt", "ss", "ff", "sf", "fs"}
        or not isinstance(pvt.get("vdd_v"), list)
        or not pvt["vdd_v"]
        or not all(_positive_number(value) for value in pvt["vdd_v"])
        or not isinstance(pvt.get("temperature_c"), list)
        or not pvt["temperature_c"]
        or not all(_finite_number(value) for value in pvt["temperature_c"])
    ):
        raise ValueError("pvt_replay_grid must define five corners, voltage, and temperature")
    if set(payload.get("arms", [])) != {
        "fixed_topology_deterministic_sweep_tpe",
        "model_guided_topology_and_experiment_selection",
    }:
        raise ValueError("campaign arms must define the frozen paired comparison")
    promotion = payload.get("promotion_gates")
    required_promotion_fields = {
        "all_designs_replay_under_pinned_environment_and_pdk",
        "known_reward_attacks_accepted",
        "paired_success_or_hypervolume_ci95_lower_bound_gt",
        "held_out_topology_rule_beats_majority_baseline",
        "manual_topology_plausibility_required",
    }
    if not isinstance(promotion, dict) or set(promotion) != required_promotion_fields:
        raise ValueError("campaign promotion_gates are incomplete")
    if promotion != {
        "all_designs_replay_under_pinned_environment_and_pdk": True,
        "known_reward_attacks_accepted": 0,
        "paired_success_or_hypervolume_ci95_lower_bound_gt": 0.0,
        "held_out_topology_rule_beats_majority_baseline": True,
        "manual_topology_plausibility_required": True,
    }:
        raise ValueError("campaign promotion_gates do not match the frozen protocol")
    return payload


def _verified_json(
    path: str | Path | None,
    expected_sha256: Any,
    *,
    label: str,
    blockers: list[str],
) -> tuple[Path | None, dict[str, Any] | None]:
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        blockers.append(f"{label} SHA-256 is not pinned")
        return None, None
    if path is None:
        blockers.append(f"no {label} was supplied")
        return None, None
    candidate = Path(path).resolve()
    if not candidate.is_file():
        blockers.append(f"{label} is missing")
        return candidate, None
    if candidate.stat().st_size > 2_000_000:
        blockers.append(f"{label} exceeds 2,000,000 bytes")
        return candidate, None
    if _sha256_file(candidate) != expected_sha256:
        blockers.append(f"{label} SHA-256 mismatch")
        return candidate, None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        blockers.append(f"{label} is not valid JSON")
        return candidate, None
    if not isinstance(payload, dict):
        blockers.append(f"{label} must be a JSON object")
        return candidate, None
    return candidate, payload


def _validate_template(path: Path, parameters: list[str]) -> bool:
    if not path.is_file() or path.stat().st_size > 200_000:
        return False
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    lower = text.lower()
    if (
        not any(line.strip().lower() == ".end" for line in lines)
        or not any(card in lower for card in (".op", ".ac", ".tran"))
        or any(card in lower for card in (".include", ".lib", ".control", "shell"))
        or sum(
            line.lstrip().lower().startswith("m")
            for line in lines
            if line.strip() and not line.lstrip().startswith("*")
        )
        < 4
    ):
        return False
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", lower))
    if not {"vinp", "vinn", "vout", "vdd", "vss"} <= tokens:
        return False
    return all(
        f"${parameter}" in text or f"${{{parameter}}}" in text
        for parameter in parameters
    )


def preflight_sky130_campaign(
    campaign_path: str | Path,
    *,
    pdk_root: str | Path | None = None,
    sealed_eval_manifest: str | Path | None = None,
    release_evidence: str | Path | None = None,
    design_m1_evidence: str | Path | None = None,
) -> dict[str, Any]:
    """Verify immutable inputs and refuse an unfrozen or uninstalled campaign."""

    payload = load_sky130_campaign(campaign_path)
    blockers: list[str] = []
    prerequisites = payload.get("prerequisites", {})
    for name in (
        "v0_1_release_locally_validated",
        "design_agent_clears_m1",
        "sealed_eval_manifest_frozen",
    ):
        if prerequisites.get(name) is not True:
            blockers.append(f"prerequisite {name!r} is not satisfied")

    evidence_hashes = payload.get("prerequisite_evidence_sha256", {})
    release_path, release_payload = _verified_json(
        release_evidence,
        evidence_hashes.get("v0_1_release"),
        label="v0.1 release evidence",
        blockers=blockers,
    )
    m1_path, m1_payload = _verified_json(
        design_m1_evidence,
        evidence_hashes.get("design_agent_m1"),
        label="design-agent M1 evidence",
        blockers=blockers,
    )
    sealed_hash = payload.get("sealed_eval_manifest_sha256")
    sealed_path, sealed_payload = _verified_json(
        sealed_eval_manifest,
        sealed_hash,
        label="sealed evaluation manifest",
        blockers=blockers,
    )

    pdk = payload.get("pdk", {})
    revision = pdk.get("source_revision")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        blockers.append("SKY130 source_revision is not pinned to a 40-hex commit")
    if release_payload is not None and (
        release_payload.get("schema") != "phychip-v0.1-evidence-manifest-v2"
        or release_payload.get("release_status") != "built_from_clean_source_commit"
        or release_payload.get("source_git", {}).get("working_tree_dirty") is not False
    ):
        blockers.append("v0.1 release evidence is not a clean validated release")
    if m1_payload is not None and (
        m1_payload.get("schema") != DESIGN_M1_SCHEMA
        or m1_payload.get("campaign_id") != payload.get("campaign_id")
        or m1_payload.get("decision") != "pass"
        or m1_payload.get("pdk_source_revision") != revision
        or m1_payload.get("sealed_eval_manifest_sha256") != sealed_hash
    ):
        blockers.append("design-agent M1 evidence does not bind this campaign")
    if sealed_payload is not None:
        tasks = sealed_payload.get("tasks")
        declared_task_hash = sealed_payload.get("task_set_sha256")
        valid_tasks = (
            isinstance(tasks, list)
            and bool(tasks)
            and all(
                isinstance(task, dict)
                and isinstance(task.get("task_id"), str)
                and bool(task["task_id"])
                and task.get("split") == "held_out"
                and _valid_target_specs(task.get("target_specs"))
                and _valid_pvt(task.get("pvt"))
                for task in tasks
            )
            and len({task["task_id"] for task in tasks}) == len(tasks)
        )
        if (
            sealed_payload.get("schema") != SEALED_EVAL_SCHEMA
            or sealed_payload.get("campaign_id") != payload.get("campaign_id")
            or sealed_payload.get("pdk_source_revision") != revision
            or not valid_tasks
            or declared_task_hash != _content_hash(tasks)
        ):
            blockers.append("sealed evaluation manifest is empty or contract-invalid")

    required_models = pdk.get("required_models")
    if not isinstance(required_models, list) or not required_models:
        blockers.append("no required SKY130 model files are declared")
        required_models = []

    resolved_models = []
    root = Path(pdk_root).resolve() if pdk_root is not None else None
    if root is None:
        blockers.append("no local PDK root was supplied")
    for model in required_models:
        relative = Path(str(model.get("path", "")))
        expected = model.get("sha256")
        if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
            blockers.append(f"{relative}: expected SHA-256 is not pinned")
            continue
        candidate = _safe_candidate(root, relative) if root is not None else None
        if candidate is None or not candidate.is_file():
            blockers.append(f"{relative}: required model file is missing")
            continue
        actual = _sha256_file(candidate)
        if actual != expected:
            blockers.append(f"{relative}: SHA-256 mismatch")
            continue
        resolved_models.append(
            {
                "path": str(relative),
                "sha256": actual,
            }
        )

    campaign_root = Path(campaign_path).resolve().parent
    resolved_templates = []
    unfinished = []
    for topology in payload["topologies"]:
        topology_id = str(topology["topology_id"])
        template_path = Path(str(topology.get("template_path", "")))
        template_hash = topology.get("template_sha256")
        if (
            topology.get("template_status") != "verified"
            or not isinstance(template_hash, str)
            or not _SHA256.fullmatch(template_hash)
        ):
            unfinished.append(topology_id)
            continue
        candidate = _safe_candidate(campaign_root, template_path)
        if (
            not candidate.is_file()
            or _sha256_file(candidate) != template_hash
            or not _validate_template(
                candidate,
                list(map(str, topology["continuous_parameters"])),
            )
        ):
            unfinished.append(topology_id)
            continue
        resolved_templates.append(
            {
                "topology_id": topology_id,
                "path": str(template_path),
                "sha256": template_hash,
            }
        )
    if unfinished:
        blockers.append(
            "unverified topology templates: " + ", ".join(sorted(unfinished))
        )

    return {
        "schema": "phychip-sky130-preflight-v1",
        "campaign_id": payload["campaign_id"],
        "campaign_sha256": _campaign_hash(payload),
        "ready": not blockers,
        "blockers": blockers,
        "resolved_models": resolved_models,
        "resolved_templates": resolved_templates,
        "sealed_eval_manifest": (
            {
                "path": str(sealed_path),
                "sha256": sealed_hash,
            }
            if sealed_path is not None
            and sealed_path.is_file()
            and isinstance(sealed_hash, str)
            and _SHA256.fullmatch(sealed_hash)
            and _sha256_file(sealed_path) == sealed_hash
            else None
        ),
        "release_evidence": (
            {"path": str(release_path), "sha256": evidence_hashes.get("v0_1_release")}
            if release_payload is not None
            and not any("v0.1 release evidence" in blocker for blocker in blockers)
            else None
        ),
        "design_m1_evidence": (
            {"path": str(m1_path), "sha256": evidence_hashes.get("design_agent_m1")}
            if m1_payload is not None
            and not any("design-agent M1 evidence" in blocker for blocker in blockers)
            else None
        ),
        "pdk_source_revision": revision,
        "topology_ids": [
            topology["topology_id"] for topology in payload["topologies"]
        ],
    }
