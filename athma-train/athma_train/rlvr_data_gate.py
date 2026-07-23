"""Deterministic, fail-closed data-quality gate for PhyChip RLVR specs.

This module deliberately does *not* run a model or ngspice.  It validates the
evidence already attached to a row, assigns stable identities, removes exact
prompt duplicates, and creates a prompt/topology-disjoint split.  Simulator
execution and exact-regime rollout labeling belong to separate, sandboxed
pipelines; this gate only accepts their versioned evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from athma_train.env.rlvr_scorer import SCORER_SEMANTICS, anticheat_ok


GATE_SCHEMA_VERSION = "rlvr-data-gate-v1"
TRAINABILITY_SCHEMA = "exact-regime-v2"
SEMANTIC_POLICY_VERSION = "phychip-v3-conservative-v1"
REWARD_CONTRACT_POLICY_VERSION = "canonical-rlvr-anticheat-v1"
REWARD_CONTRACT_RULES = {
    "scorer_semantics": SCORER_SEMANTICS,
    "callable": "athma_train.env.rlvr_scorer.anticheat_ok",
}

_HEX_256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SPACE_RE = re.compile(r"\s+")
_PARAM_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?"
    r"(?:t|g|meg|k|m|u|n|p|f)?(?:[A-Za-z]+)?"
    r"(?![A-Za-z0-9_.])",
    re.IGNORECASE,
)
_SPICE_NUMBER_RE = re.compile(
    r"^([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)"
    r"(t|g|meg|k|m|u|n|p|f)?(?:[A-Za-z]+)?$",
    re.IGNORECASE,
)
_SPICE_MULTIPLIERS = {
    "": 1.0,
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation used by all content hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(prompt: str) -> str:
    """Normalize presentation-only differences without erasing numeric specs."""
    text = unicodedata.normalize("NFKC", prompt)
    text = text.replace("\u2192", "->").replace("\u2013", "-").replace("\u2014", "-")
    return _SPACE_RE.sub(" ", text).strip().casefold()


def _normalized_tail(tokens: Sequence[str]) -> list[str]:
    tail = " ".join(tokens).casefold()
    tail = _PARAM_NUMBER_RE.sub("<num>", tail)
    return tail.split()


def topology_canonical_form(netlist: str) -> str:
    """Create a numeric-insensitive structural representation of a netlist.

    Node labels remain intact so shorted and non-shorted circuits do not
    collapse together.  Instance names and component values are normalized;
    analyses and ``.control`` programs are excluded because they are not the
    circuit topology.
    """
    node_counts = {
        "r": 2,
        "c": 2,
        "l": 2,
        "d": 2,
        "v": 2,
        "i": 2,
        "e": 4,
        "g": 4,
        "f": 2,
        "h": 2,
        "b": 2,
        "m": 4,
        "q": 3,
        "j": 3,
        "s": 4,
        "w": 2,
    }
    canonical: list[str] = []
    in_control = False
    for raw in netlist.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("*"):
            continue
        lowered = stripped.casefold()
        if lowered.startswith(".control"):
            in_control = True
            continue
        if lowered.startswith(".endc"):
            in_control = False
            continue
        if in_control:
            continue
        if lowered.startswith(".subckt"):
            tokens = lowered.split()
            canonical.append(" ".join(tokens))
            continue
        if lowered.startswith(".ends"):
            canonical.append(".ends")
            continue
        if lowered.startswith("."):
            continue

        tokens = stripped.split()
        if len(tokens) < 2:
            continue
        device = tokens[0][0].casefold()
        if not device.isalpha():
            continue
        if device == "x":
            # Xname node... subckt-name [params...].  Parameter assignments
            # begin the value tail; otherwise the final token is the subckt.
            assignment_at = next(
                (i for i, token in enumerate(tokens[1:], 1) if "=" in token), len(tokens)
            )
            body = tokens[1:assignment_at]
            if not body:
                continue
            nodes, model = body[:-1], body[-1].casefold()
            rest = _normalized_tail(tokens[assignment_at:])
            canonical.append(" ".join(["x", *(n.casefold() for n in nodes), model, *rest]))
            continue

        count = node_counts.get(device)
        if count is None or len(tokens) < count + 1:
            # Unknown devices are retained, with numeric literals normalized.
            canonical.append(" ".join([device, *_normalized_tail(tokens[1:])]))
            continue
        nodes = [token.casefold() for token in tokens[1 : count + 1]]
        tail = _normalized_tail(tokens[count + 1 :])
        canonical.append(" ".join([device, *nodes, *tail]))
    return "\n".join(sorted(canonical))


def topology_fingerprint(netlist: str) -> str:
    return sha256_text(topology_canonical_form(netlist))


def reward_contract_policy_hash() -> str:
    return sha256_text(SCORER_SEMANTICS + "\n" + inspect.getsource(anticheat_ok))


def _device_line_count(netlist: str, prefixes: str) -> int:
    return len(re.findall(rf"(?mi)^\s*[{re.escape(prefixes)}]\w*\s", netlist or ""))


def reward_contract_rejection_reasons(row: Mapping[str, Any]) -> list[str]:
    """Check a reference against the current GRPO anti-cheat contract.

    This is intentionally simulator-free and calls the exact packaged
    anti-cheat gate used by training/evaluation.  Maintaining a partial static
    copy previously allowed data admission and runtime reward to diverge.
    """
    reasons: list[str] = []
    family = str(row.get("circuit_id", "")).casefold().replace("_", "-")
    netlist = str(row.get("reference_netlist", ""))
    if not anticheat_ok(netlist, family):
        reasons.append(f"reward_contract:canonical_anticheat:{family}")
    return reasons


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _parse_spice_number(token: str) -> float | None:
    match = _SPICE_NUMBER_RE.match(token.strip().casefold())
    if not match:
        return None
    try:
        return float(match.group(1)) * _SPICE_MULTIPLIERS[match.group(2) or ""]
    except (ValueError, OverflowError):
        return None


def _dc_source_values(netlist: str) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    for raw in netlist.splitlines():
        tokens = raw.strip().split()
        if len(tokens) < 4 or not tokens[0].casefold().startswith("v"):
            continue
        lower = [token.casefold() for token in tokens]
        try:
            dc_at = lower.index("dc")
        except ValueError:
            dc_at = -1
        value_token = tokens[dc_at + 1] if dc_at >= 0 and dc_at + 1 < len(tokens) else tokens[3]
        value = _parse_spice_number(value_token)
        if value is not None and math.isfinite(value):
            values.append((tokens[0].casefold(), value))
    return values


def _target(row: Mapping[str, Any], name: str) -> tuple[float, float] | None:
    value = row.get("target_measurements", {}).get(name)
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(_is_number(v) for v in value):
        return float(value[0]), float(value[1])
    return None


def _measured(row: Mapping[str, Any], name: str) -> float | None:
    value = row.get("gen_measured", {}).get(name)
    return float(value) if _is_number(value) else None


def _require_metric(row: Mapping[str, Any], name: str, reasons: list[str]) -> bool:
    if _target(row, name) is None or _measured(row, name) is None:
        reasons.append(f"semantic:missing_required_metric:{name}")
        return False
    return True


def _rail_span(netlist: str) -> tuple[float, float] | None:
    sources = _dc_source_values(netlist)
    rail_values = [
        value
        for name, value in sources
        if any(tag in name for tag in ("vdd", "vss", "vcc", "vee", "vddh", "vssh", "vin"))
    ]
    values = rail_values or [value for _, value in sources]
    if not values:
        return None
    # Independent voltage sources in these generated netlists are referenced
    # to node 0.  Include ground explicitly so a single positive Vin/VDD still
    # resolves to a meaningful 0..VDD rail span.
    values = [0.0, *values]
    low, high = min(values), max(values)
    return (low, high) if high > low else None


def semantic_rejection_reasons(row: Mapping[str, Any]) -> list[str]:
    """Apply conservative family-specific physical bounds to dataset-v3 rows."""
    reasons: list[str] = []
    family = str(row.get("circuit_id", "")).casefold().replace("_", "-")
    netlist = str(row.get("reference_netlist", ""))

    if family == "bandgap" and _require_metric(row, "vref_v", reasons):
        lo, hi = _target(row, "vref_v")  # type: ignore[misc]
        measured = _measured(row, "vref_v")
        if lo < 0.05 or hi > 5.0 or measured is None or not (0.05 <= measured <= 5.0):
            reasons.append("semantic:bandgap:vref_out_of_conservative_range_0.05_to_5V")

    elif family == "schmitt":
        have_high = _require_metric(row, "vth_high_v", reasons)
        have_low = _require_metric(row, "vth_low_v", reasons)
        if have_high and have_low:
            high_target = _target(row, "vth_high_v")
            low_target = _target(row, "vth_low_v")
            high_measured = _measured(row, "vth_high_v")
            low_measured = _measured(row, "vth_low_v")
            assert high_target and low_target and high_measured is not None and low_measured is not None
            if high_measured <= low_measured:
                reasons.append("semantic:schmitt:nonpositive_measured_hysteresis")
            if high_target[0] <= low_target[1]:
                reasons.append("semantic:schmitt:threshold_windows_overlap_or_reverse")
            rails = _rail_span(netlist)
            if rails is None:
                reasons.append("semantic:schmitt:supply_rails_unresolved")
            else:
                tolerance = max(1e-6, 0.05 * (rails[1] - rails[0]))
                if low_target[0] < rails[0] - tolerance or high_target[1] > rails[1] + tolerance:
                    reasons.append("semantic:schmitt:threshold_outside_supply_rails")

    elif family == "ldo" and _require_metric(row, "vout_dc_v", reasons):
        lo, hi = _target(row, "vout_dc_v")  # type: ignore[misc]
        measured = _measured(row, "vout_dc_v")
        if lo < 0.05 or measured is None or measured < 0.05:
            reasons.append("semantic:ldo:nonpositive_or_near_zero_output")
        rails = _rail_span(netlist)
        if rails is None or rails[1] <= 0:
            reasons.append("semantic:ldo:positive_input_supply_unresolved")
        elif hi > rails[1] * 1.05:
            reasons.append("semantic:ldo:target_exceeds_input_supply")

    elif family == "current-mirror" and _require_metric(row, "current_ratio", reasons):
        lo, hi = _target(row, "current_ratio")  # type: ignore[misc]
        measured = _measured(row, "current_ratio")
        if lo < 0.01 or hi > 100.0 or measured is None or not (0.01 <= measured <= 100.0):
            reasons.append("semantic:current-mirror:ratio_out_of_conservative_range_0.01_to_100")

    elif family == "level-shift" and _require_metric(row, "voh_v", reasons):
        lo, hi = _target(row, "voh_v")  # type: ignore[misc]
        rails = _rail_span(netlist)
        if rails is None:
            reasons.append("semantic:level-shift:supply_rails_unresolved")
        else:
            tolerance = max(1e-6, 0.05 * (rails[1] - rails[0]))
            if lo < rails[0] - tolerance or hi > rails[1] + tolerance:
                reasons.append("semantic:level-shift:voh_outside_supply_rails")
    return reasons


def _claimed_trainable(row: Mapping[str, Any]) -> bool:
    bucket = str(row.get("bucket", "")).casefold()
    return row.get("trainable") is True or bucket in {"train", "trainable"}


def validate_trainability_label(row: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate exact-regime rollout evidence and derive trainability.

    Trainability is derived from mixed exact-objective outcomes
    (``0 < success_count < K``); shaped-reward variance is diagnostic only.
    Top-level ``bucket`` or ``trainable`` values are never trusted.  An
    old-style positive claim with no exact-regime evidence is a quarantine
    error.
    """
    label = row.get("trainability_label")
    if label is None:
        if _claimed_trainable(row):
            return {}, ["trainability:positive_claim_without_exact_regime_metadata"]
        return {"status": "unlabeled", "trainable": False}, []
    if not isinstance(label, Mapping):
        return {}, ["trainability:label_not_object"]

    reasons: list[str] = []
    if label.get("label_schema") != TRAINABILITY_SCHEMA:
        reasons.append(f"trainability:label_schema_must_equal:{TRAINABILITY_SCHEMA}")
    k = label.get("K")
    if not isinstance(k, int) or isinstance(k, bool) or k < 2:
        reasons.append("trainability:K_must_be_integer_at_least_2")
    rewards = label.get("reward_vector")
    if not isinstance(rewards, list) or not rewards or not all(_is_number(v) for v in rewards):
        reasons.append("trainability:reward_vector_must_be_nonempty_finite_number_list")
    elif isinstance(k, int) and len(rewards) != k:
        reasons.append("trainability:reward_vector_length_must_equal_K")
    successes = label.get("success_vector")
    if (
        not isinstance(successes, list)
        or not successes
        or not all(isinstance(value, bool) for value in successes)
    ):
        reasons.append("trainability:success_vector_must_be_nonempty_boolean_list")
    elif isinstance(k, int) and len(successes) != k:
        reasons.append("trainability:success_vector_length_must_equal_K")
    hashes: dict[str, str] = {}
    for name in ("policy_hash", "verifier_hash", "prompt_contract_hash"):
        value = label.get(name)
        if not isinstance(value, str) or not _HEX_256_RE.fullmatch(value):
            reasons.append(f"trainability:{name}_must_be_sha256_hex")
        else:
            hashes[name] = value.casefold()
    if reasons:
        return {}, reasons

    numeric_rewards = [float(value) for value in rewards]
    success_count = sum(successes)
    trainable = 0 < success_count < k
    if _claimed_trainable(row) and not trainable:
        return {}, ["trainability:positive_source_claim_conflicts_with_success_vector"]
    evidence_hash = sha256_text(canonical_json(dict(label)))
    if trainable:
        status = "trainable_mixed_success"
    elif success_count == 0:
        status = "too_hard_zero_success"
    else:
        status = "too_easy_all_success"
    return {
        "status": status,
        "trainable": trainable,
        "K": k,
        "success_count": success_count,
        "success_rate": success_count / k,
        "success_vector_hash": sha256_text(canonical_json(successes)),
        "reward_has_variance": max(numeric_rewards) > min(numeric_rewards),
        "reward_min": min(numeric_rewards),
        "reward_max": max(numeric_rewards),
        "reward_vector_hash": sha256_text(canonical_json(numeric_rewards)),
        "evidence_hash": evidence_hash,
        **hashes,
    }, []


def schema_rejection_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    for name in ("circuit_id", "spec_prompt", "reference_netlist"):
        if not isinstance(row.get(name), str) or not str(row.get(name)).strip():
            reasons.append(f"schema:{name}_must_be_nonempty_string")
    for name in ("input_ports", "output_ports"):
        value = row.get(name)
        if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
            reasons.append(f"schema:{name}_must_be_nonempty_string_list")

    targets = row.get("target_measurements")
    measured = row.get("gen_measured")
    if not isinstance(targets, Mapping) or not targets:
        reasons.append("schema:target_measurements_must_be_nonempty_object")
        targets = {}
    if not isinstance(measured, Mapping):
        reasons.append("schema:gen_measured_must_be_object")
        measured = {}
    for metric, interval in targets.items():
        if not isinstance(metric, str) or not metric:
            reasons.append("schema:target_metric_name_must_be_nonempty_string")
            continue
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            reasons.append(f"schema:target_interval_must_have_two_values:{metric}")
            continue
        if not all(_is_number(value) for value in interval):
            reasons.append(f"schema:target_interval_must_be_finite:{metric}")
            continue
        lo, hi = float(interval[0]), float(interval[1])
        if lo >= hi:
            reasons.append(f"schema:target_interval_must_have_positive_width:{metric}")
        value = measured.get(metric)
        if not _is_number(value):
            reasons.append(f"schema:gen_measured_missing_or_nonfinite:{metric}")
        elif not (lo <= float(value) <= hi):
            reasons.append(f"schema:gen_measured_outside_target:{metric}")
    n_objectives = row.get("n_objectives")
    if not isinstance(n_objectives, int) or isinstance(n_objectives, bool) or n_objectives != len(targets):
        reasons.append("schema:n_objectives_must_equal_target_count")
    if row.get("verified") is not True:
        reasons.append("schema:verified_must_be_true")

    if isinstance(row.get("reference_netlist"), str):
        canonical = topology_canonical_form(str(row["reference_netlist"]))
        if not canonical:
            reasons.append("schema:reference_netlist_has_no_topology")
    return reasons


def _spec_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: row.get(name)
        for name in (
            "circuit_id",
            "input_ports",
            "output_ports",
            "spec_prompt",
            "target_measurements",
            "reference_netlist",
        )
    }


@dataclass
class GateRecord:
    row: dict[str, Any] | None
    source_path: str
    source_line: int
    raw_sha256: str
    row_hash: str
    spec_hash: str = ""
    spec_id: str = ""
    normalized_prompt: str = ""
    prompt_hash: str = ""
    topology_hash: str = ""
    trainability: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    disposition: str = "pending"
    duplicate_of: str | None = None

    @property
    def record_id(self) -> str:
        return sha256_text(f"{self.source_path}:{self.source_line}:{self.raw_sha256}")

    def manifest(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_path": self.source_path,
            "source_line": self.source_line,
            "raw_sha256": self.raw_sha256,
            "row_hash": self.row_hash,
            "spec_hash": self.spec_hash or None,
            "spec_id": self.spec_id or None,
            "normalized_prompt_sha256": self.prompt_hash or None,
            "topology_sha256": self.topology_hash or None,
            "disposition": self.disposition,
            "rejection_reasons": sorted(set(self.reasons)),
            "duplicate_of": self.duplicate_of,
            "trainability": self.trainability or None,
        }

    def enriched_row(self) -> dict[str, Any]:
        assert self.row is not None
        row = dict(self.row)
        row["data_quality"] = {
            "gate_schema": GATE_SCHEMA_VERSION,
            "semantic_policy": SEMANTIC_POLICY_VERSION,
            "row_hash": self.row_hash,
            "spec_hash": self.spec_hash,
            "spec_id": self.spec_id,
            "normalized_prompt_sha256": self.prompt_hash,
            "topology_sha256": self.topology_hash,
            "trainability": self.trainability,
        }
        # Derived truth overrides stale top-level labels.  Preserve the source
        # values as provenance without leaving a misleading active ``bucket``.
        source_claim = {
            name: row.pop(name)
            for name in ("trainable", "bucket")
            if name in row
        }
        if source_claim:
            row["source_trainability_claim"] = source_claim
        row["trainable"] = bool(self.trainability.get("trainable", False))
        row["trainability_status"] = self.trainability.get("status", "unlabeled")
        return row


def load_records(paths: Iterable[Path]) -> list[GateRecord]:
    records: list[GateRecord] = []
    for path in sorted(paths, key=lambda value: value.as_posix()):
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                stripped = raw.strip()
                if not stripped:
                    continue
                raw_hash = sha256_text(stripped)
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    records.append(
                        GateRecord(
                            row=None,
                            source_path=path.as_posix(),
                            source_line=line_number,
                            raw_sha256=raw_hash,
                            row_hash=raw_hash,
                            reasons=[f"schema:invalid_json:{exc.msg}"],
                            disposition="quarantine",
                        )
                    )
                    continue
                if not isinstance(value, dict):
                    records.append(
                        GateRecord(
                            row=None,
                            source_path=path.as_posix(),
                            source_line=line_number,
                            raw_sha256=raw_hash,
                            row_hash=sha256_text(canonical_json(value)),
                            reasons=["schema:row_must_be_object"],
                            disposition="quarantine",
                        )
                    )
                    continue
                clean_value = dict(value)
                clean_value.pop("data_quality", None)
                row_hash = sha256_text(canonical_json(clean_value))
                spec_hash = sha256_text(canonical_json(_spec_payload(clean_value)))
                prompt = normalize_prompt(str(clean_value.get("spec_prompt", "")))
                netlist = str(clean_value.get("reference_netlist", ""))
                trainability, label_reasons = validate_trainability_label(clean_value)
                reasons = schema_rejection_reasons(clean_value)
                if not reasons:
                    reasons.extend(semantic_rejection_reasons(clean_value))
                reasons.extend(reward_contract_rejection_reasons(clean_value))
                reasons.extend(label_reasons)
                records.append(
                    GateRecord(
                        row=clean_value,
                        source_path=path.as_posix(),
                        source_line=line_number,
                        raw_sha256=raw_hash,
                        row_hash=row_hash,
                        spec_hash=spec_hash,
                        spec_id=f"spec_{spec_hash[:20]}",
                        normalized_prompt=prompt,
                        prompt_hash=sha256_text(prompt),
                        topology_hash=topology_fingerprint(netlist),
                        trainability=trainability,
                        reasons=sorted(set(reasons)),
                        disposition="quarantine" if reasons else "validated",
                    )
                )
    return records


def _source_data_accounting(records: Sequence[GateRecord]) -> dict[str, Any]:
    """Describe raw duplication and cross-source leakage at explicit grains."""
    by_source: dict[str, list[GateRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source_path].append(record)

    def keys(source_records: Sequence[GateRecord]) -> dict[str, list[str]]:
        object_records = [record for record in source_records if record.row is not None]
        exact_prompts = [
            str(record.row.get("spec_prompt"))
            for record in object_records
            if isinstance(record.row.get("spec_prompt"), str)
        ]
        normalized_prompts = [
            record.prompt_hash for record in object_records if record.normalized_prompt
        ]
        topology_hashes = [
            record.topology_hash
            for record in object_records
            if topology_canonical_form(str(record.row.get("reference_netlist", "")))
        ]
        return {
            "row": [record.row_hash for record in source_records],
            "exact_prompt": exact_prompts,
            "normalized_prompt": normalized_prompts,
            "topology": topology_hashes,
        }

    def summary(source_records: Sequence[GateRecord]) -> dict[str, Any]:
        value_keys = keys(source_records)
        return {
            "rows": len(source_records),
            "exact_row_unique": len(set(value_keys["row"])),
            "exact_row_duplicate_excess": len(value_keys["row"]) - len(set(value_keys["row"])),
            "exact_prompt_unique": len(set(value_keys["exact_prompt"])),
            "exact_prompt_duplicate_excess": len(value_keys["exact_prompt"])
            - len(set(value_keys["exact_prompt"])),
            "normalized_prompt_unique": len(set(value_keys["normalized_prompt"])),
            "normalized_prompt_duplicate_excess": len(value_keys["normalized_prompt"])
            - len(set(value_keys["normalized_prompt"])),
            "topology_unique": len(set(value_keys["topology"])),
            "topology_duplicate_excess": len(value_keys["topology"])
            - len(set(value_keys["topology"])),
        }

    pairwise: list[dict[str, Any]] = []
    for left_path, right_path in combinations(sorted(by_source), 2):
        # Give train->holdout direction stable semantics when filenames expose
        # those roles; otherwise preserve lexical order and report both paths.
        if "holdout" in Path(left_path).name.casefold() and "train" in Path(right_path).name.casefold():
            left_path, right_path = right_path, left_path
        left_keys, right_keys = keys(by_source[left_path]), keys(by_source[right_path])
        exact_rows = set(left_keys["row"])
        exact_prompts = set(left_keys["exact_prompt"])
        normalized_prompts = set(left_keys["normalized_prompt"])
        topologies = set(left_keys["topology"])
        pairwise.append(
            {
                "left_source": left_path,
                "right_source": right_path,
                "exact_row_unique_overlap": len(exact_rows & set(right_keys["row"])),
                "right_rows_with_exact_row_in_left": sum(
                    value in exact_rows for value in right_keys["row"]
                ),
                "exact_prompt_unique_overlap": len(
                    exact_prompts & set(right_keys["exact_prompt"])
                ),
                "right_rows_with_exact_prompt_in_left": sum(
                    value in exact_prompts for value in right_keys["exact_prompt"]
                ),
                "normalized_prompt_unique_overlap": len(
                    normalized_prompts & set(right_keys["normalized_prompt"])
                ),
                "right_rows_with_normalized_prompt_in_left": sum(
                    value in normalized_prompts for value in right_keys["normalized_prompt"]
                ),
                "topology_unique_overlap": len(topologies & set(right_keys["topology"])),
                "right_rows_with_topology_in_left": sum(
                    value in topologies for value in right_keys["topology"]
                ),
            }
        )

    combined = [record for source_records in by_source.values() for record in source_records]
    return {
        "combined": summary(combined),
        "by_source": {
            source: summary(source_records)
            for source, source_records in sorted(by_source.items())
        },
        "pairwise_source_overlap": pairwise,
    }


def deduplicate_prompts(records: Sequence[GateRecord]) -> list[GateRecord]:
    valid = [record for record in records if record.disposition == "validated"]
    by_prompt: dict[str, list[GateRecord]] = defaultdict(list)
    for record in valid:
        by_prompt[record.prompt_hash].append(record)
    kept: list[GateRecord] = []
    for group in by_prompt.values():
        ordered = sorted(group, key=lambda record: (record.row_hash, record.record_id))
        winner = ordered[0]
        kept.append(winner)
        for duplicate in ordered[1:]:
            duplicate.disposition = "quarantine"
            duplicate.reasons.append("dedup:duplicate_normalized_prompt")
            duplicate.duplicate_of = winner.spec_id
    return sorted(kept, key=lambda record: (record.spec_id, record.record_id))


def assign_group_disjoint_split(
    records: Sequence[GateRecord], holdout_fraction: float, split_seed: str
) -> tuple[list[GateRecord], list[GateRecord]]:
    if not 0 <= holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in [0, 1)")
    groups: dict[str, list[GateRecord]] = defaultdict(list)
    for record in records:
        # Prompt duplicates are removed first.  Grouping on both hashes keeps
        # this safe if a caller bypasses that convenience in future.
        group_key = sha256_text(f"{record.prompt_hash}:{record.topology_hash}")
        groups[group_key].append(record)

    # Union groups sharing either prompt or topology.  In normal operation
    # prompt groups are singletons after deduplication, but the transitive
    # closure makes the isolation invariant explicit.
    parent = {key: key for key in groups}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    prompt_owner: dict[str, str] = {}
    topology_owner: dict[str, str] = {}
    for key, members in groups.items():
        for record in members:
            if record.prompt_hash in prompt_owner:
                union(key, prompt_owner[record.prompt_hash])
            else:
                prompt_owner[record.prompt_hash] = key
            if record.topology_hash in topology_owner:
                union(key, topology_owner[record.topology_hash])
            else:
                topology_owner[record.topology_hash] = key

    components: dict[str, list[GateRecord]] = defaultdict(list)
    for key, members in groups.items():
        components[find(key)].extend(members)

    train: list[GateRecord] = []
    holdout: list[GateRecord] = []
    cutoff = int(holdout_fraction * (1 << 256))
    for component in components.values():
        component_id = sha256_text(
            split_seed + ":" + ":".join(sorted(record.spec_hash for record in component))
        )
        destination = holdout if int(component_id, 16) < cutoff else train
        disposition = "holdout" if destination is holdout else "verified_candidate"
        for record in component:
            record.disposition = disposition
            destination.append(record)
    return (
        sorted(train, key=lambda record: (record.spec_id, record.record_id)),
        sorted(holdout, key=lambda record: (record.spec_id, record.record_id)),
    )


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(canonical_json(value) + "\n")


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            values.append(value)
    return values


def check_verified_artifacts(output_dir: Path) -> dict[str, Any]:
    """Run simulator-free acceptance checks on generated RLVR artifacts."""
    candidate_path = output_dir / "verified_candidates.jsonl"
    holdout_path = output_dir / "holdout.jsonl"
    trainable_path = output_dir / "trainable_candidates.jsonl"
    candidates = _read_jsonl_objects(candidate_path)
    holdout = _read_jsonl_objects(holdout_path)
    trainable_rows = _read_jsonl_objects(trainable_path)
    all_rows = candidates + holdout

    schema_invalid = sum(bool(schema_rejection_reasons(row)) for row in all_rows)
    semantic_invalid = sum(bool(semantic_rejection_reasons(row)) for row in all_rows)
    reward_contract_invalid = sum(
        bool(reward_contract_rejection_reasons(row)) for row in all_rows
    )
    identity_invalid = 0
    trainability_invalid = 0
    for row in all_rows:
        quality = row.get("data_quality")
        if not isinstance(quality, Mapping):
            identity_invalid += 1
        else:
            expected_spec_hash = sha256_text(canonical_json(_spec_payload(row)))
            expected_prompt_hash = sha256_text(normalize_prompt(str(row.get("spec_prompt", ""))))
            expected_topology_hash = topology_fingerprint(str(row.get("reference_netlist", "")))
            if (
                quality.get("spec_hash") != expected_spec_hash
                or quality.get("spec_id") != f"spec_{expected_spec_hash[:20]}"
                or quality.get("normalized_prompt_sha256") != expected_prompt_hash
                or quality.get("topology_sha256") != expected_topology_hash
            ):
                identity_invalid += 1
        if row.get("trainable") is True:
            evidence = quality.get("trainability") if isinstance(quality, Mapping) else None
            if (
                not isinstance(evidence, Mapping)
                or evidence.get("status") != "trainable_mixed_success"
                or not isinstance(evidence.get("success_count"), int)
                or not 0 < evidence["success_count"] < evidence.get("K", 0)
                or not all(_HEX_256_RE.fullmatch(str(evidence.get(name, ""))) for name in (
                    "policy_hash",
                    "verifier_hash",
                    "prompt_contract_hash",
                    "reward_vector_hash",
                    "success_vector_hash",
                    "evidence_hash",
                ))
            ):
                trainability_invalid += 1

    candidate_prompts = [normalize_prompt(str(row.get("spec_prompt", ""))) for row in candidates]
    holdout_prompts = [normalize_prompt(str(row.get("spec_prompt", ""))) for row in holdout]
    candidate_topologies = [
        topology_fingerprint(str(row.get("reference_netlist", ""))) for row in candidates
    ]
    holdout_topologies = [
        topology_fingerprint(str(row.get("reference_netlist", ""))) for row in holdout
    ]
    candidate_ids = [
        str(row.get("data_quality", {}).get("spec_id", "")) for row in candidates
    ]
    expected_trainable_ids = {
        str(row.get("data_quality", {}).get("spec_id", ""))
        for row in candidates
        if row.get("trainable") is True
    }
    trainable_ids: list[str] = []
    trainable_rows_invalid = 0
    for row in trainable_rows:
        quality = row.get("data_quality")
        spec_id = str(quality.get("spec_id", "")) if isinstance(quality, Mapping) else ""
        trainable_ids.append(spec_id)
        derived, reasons = validate_trainability_label(row)
        stored = quality.get("trainability") if isinstance(quality, Mapping) else None
        if (
            reasons
            or row.get("trainable") is not True
            or not derived.get("trainable")
            or derived.get("status") != "trainable_mixed_success"
            or not isinstance(stored, Mapping)
            or stored.get("evidence_hash") != derived.get("evidence_hash")
        ):
            trainable_rows_invalid += 1
    observations = {
        "candidate_rows": len(candidates),
        "holdout_rows": len(holdout),
        "trainable_candidate_rows": len(trainable_rows),
        "candidate_normalized_prompt_duplicate_excess": len(candidate_prompts)
        - len(set(candidate_prompts)),
        "holdout_normalized_prompt_duplicate_excess": len(holdout_prompts)
        - len(set(holdout_prompts)),
        "normalized_prompt_overlap": len(set(candidate_prompts) & set(holdout_prompts)),
        "topology_overlap": len(set(candidate_topologies) & set(holdout_topologies)),
        "schema_invalid_rows": schema_invalid,
        "semantic_invalid_rows": semantic_invalid,
        "reward_contract_invalid_rows": reward_contract_invalid,
        "identity_invalid_rows": identity_invalid,
        "trainability_evidence_invalid_rows": trainability_invalid,
        "trainable_artifact_invalid_rows": trainable_rows_invalid,
        "trainable_artifact_duplicate_spec_ids": len(trainable_ids) - len(set(trainable_ids)),
        "trainable_artifact_unknown_spec_ids": len(set(trainable_ids) - set(candidate_ids)),
        "trainable_artifact_unexpected_spec_ids": len(
            set(trainable_ids) - expected_trainable_ids
        ),
        "trainable_artifact_missing_spec_ids": len(expected_trainable_ids - set(trainable_ids)),
    }
    checks = {
        "candidate_prompts_unique": observations[
            "candidate_normalized_prompt_duplicate_excess"
        ]
        == 0,
        "holdout_prompts_unique": observations[
            "holdout_normalized_prompt_duplicate_excess"
        ]
        == 0,
        "prompt_disjoint": observations["normalized_prompt_overlap"] == 0,
        "topology_disjoint": observations["topology_overlap"] == 0,
        "schema_valid": schema_invalid == 0,
        "semantic_policy_valid": semantic_invalid == 0,
        "reward_contract_valid": reward_contract_invalid == 0,
        "identity_hashes_valid": identity_invalid == 0,
        "trainability_evidence_valid": trainability_invalid == 0,
        "trainable_artifact_rows_valid": trainable_rows_invalid == 0,
        "trainable_artifact_unique": observations[
            "trainable_artifact_duplicate_spec_ids"
        ]
        == 0,
        "trainable_artifact_is_candidate_subset": observations[
            "trainable_artifact_unknown_spec_ids"
        ]
        == 0,
        "trainable_artifact_has_no_unexpected_rows": observations[
            "trainable_artifact_unexpected_spec_ids"
        ]
        == 0,
        "trainable_artifact_is_complete": observations[
            "trainable_artifact_missing_spec_ids"
        ]
        == 0,
    }
    return {"passed": all(checks.values()), "checks": checks, "observations": observations}


def build_verified_dataset(
    source_paths: Sequence[Path],
    output_dir: Path,
    *,
    holdout_fraction: float = 0.2,
    split_seed: str = "phychip-rlvr-v1",
) -> dict[str, Any]:
    if not source_paths:
        raise ValueError("at least one source JSONL is required")
    missing = [path.as_posix() for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source JSONL not found: {', '.join(missing)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(source_paths)
    deduplicated = deduplicate_prompts(records)
    candidates, holdout = assign_group_disjoint_split(
        deduplicated, holdout_fraction=holdout_fraction, split_seed=split_seed
    )
    quarantined = sorted(
        (record for record in records if record.disposition == "quarantine"),
        key=lambda record: (record.row_hash, record.record_id),
    )

    candidate_path = output_dir / "verified_candidates.jsonl"
    holdout_path = output_dir / "holdout.jsonl"
    trainable_path = output_dir / "trainable_candidates.jsonl"
    quarantine_path = output_dir / "quarantine.jsonl"
    manifest_path = output_dir / "manifest.jsonl"
    report_path = output_dir / "report.json"

    _write_jsonl(candidate_path, (record.enriched_row() for record in candidates))
    _write_jsonl(holdout_path, (record.enriched_row() for record in holdout))
    _write_jsonl(
        trainable_path,
        (
            record.enriched_row()
            for record in candidates
            if record.trainability.get("trainable") is True
        ),
    )
    _write_jsonl(
        quarantine_path,
        (
            {
                "record": record.row,
                "manifest": record.manifest(),
            }
            for record in quarantined
        ),
    )
    _write_jsonl(
        manifest_path,
        (record.manifest() for record in sorted(records, key=lambda value: value.record_id)),
    )

    prompt_overlap = {record.prompt_hash for record in candidates} & {
        record.prompt_hash for record in holdout
    }
    topology_overlap = {record.topology_hash for record in candidates} & {
        record.topology_hash for record in holdout
    }
    if prompt_overlap or topology_overlap:
        raise RuntimeError("internal error: group-disjoint split invariant violated")

    rejection_counts = Counter(reason for record in quarantined for reason in set(record.reasons))
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        family = str(record.row.get("circuit_id", "unknown")) if record.row else "invalid_json"
        family_counts[family][record.disposition] += 1
    rejection_classes = {
        "schema_rejected_rows": sum(
            any(reason.startswith("schema:") for reason in record.reasons) for record in records
        ),
        "semantic_rejected_rows": sum(
            any(reason.startswith("semantic:") for reason in record.reasons) for record in records
        ),
        "reward_contract_rejected_rows": sum(
            any(reason.startswith("reward_contract:") for reason in record.reasons)
            for record in records
        ),
        "trainability_evidence_rejected_rows": sum(
            any(reason.startswith("trainability:") for reason in record.reasons)
            for record in records
        ),
        "deduplicated_rows": sum(
            "dedup:duplicate_normalized_prompt" in record.reasons for record in records
        ),
    }
    trainability_status = Counter(
        str(record.trainability.get("status", "rejected_or_absent")) for record in records
    )
    output_trainability_status = Counter(
        str(record.trainability.get("status", "rejected_or_absent"))
        for record in (*candidates, *holdout)
    )
    quality_by_source: dict[str, dict[str, int]] = {}
    for source_path in sorted({record.source_path for record in records}):
        source_records = [record for record in records if record.source_path == source_path]
        quality_by_source[source_path] = {
            "rows": len(source_records),
            "schema_rejected_rows": sum(
                any(reason.startswith("schema:") for reason in record.reasons)
                for record in source_records
            ),
            "semantic_rejected_rows": sum(
                any(reason.startswith("semantic:") for reason in record.reasons)
                for record in source_records
            ),
            "reward_contract_rejected_rows": sum(
                any(reason.startswith("reward_contract:") for reason in record.reasons)
                for record in source_records
            ),
            "trainability_evidence_rejected_rows": sum(
                any(reason.startswith("trainability:") for reason in record.reasons)
                for record in source_records
            ),
            "deduplicated_rows_after_quality_gates": sum(
                "dedup:duplicate_normalized_prompt" in record.reasons
                for record in source_records
            ),
        }
    raw_accounting = _source_data_accounting(records)
    preflight = check_verified_artifacts(output_dir)
    if not preflight["passed"]:
        raise RuntimeError(f"generated RLVR artifacts failed static preflight: {preflight}")
    report: dict[str, Any] = {
        "gate_schema": GATE_SCHEMA_VERSION,
        "semantic_policy": SEMANTIC_POLICY_VERSION,
        "reward_contract_policy": {
            "version": REWARD_CONTRACT_POLICY_VERSION,
            "sha256": reward_contract_policy_hash(),
            "rules": REWARD_CONTRACT_RULES,
        },
        "trainability_schema": TRAINABILITY_SCHEMA,
        "raw_data_accounting": raw_accounting,
        "split": {
            "method": "sha256 component assignment after prompt dedup; union by prompt or topology",
            "seed": split_seed,
            "holdout_fraction": holdout_fraction,
            "normalized_prompt_overlap": len(prompt_overlap),
            "topology_overlap": len(topology_overlap),
        },
        "counts": {
            "source_rows": len(records),
            "verified_candidates": len(candidates),
            "holdout": len(holdout),
            "quarantine": len(quarantined),
            "trainable_candidates": sum(
                bool(record.trainability.get("trainable")) for record in candidates
            ),
            "trainable_holdout": sum(bool(record.trainability.get("trainable")) for record in holdout),
        },
        "quality_rejection_classes": rejection_classes,
        "quality_rejections_by_source": quality_by_source,
        "trainability_evidence_status": dict(sorted(trainability_status.items())),
        "verified_output_trainability_status": dict(sorted(output_trainability_status.items())),
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "family_dispositions": {
            family: dict(sorted(counts.items())) for family, counts in sorted(family_counts.items())
        },
        "source_files": [
            {"path": path.as_posix(), "sha256": sha256_file(path)}
            for path in sorted(source_paths, key=lambda value: value.as_posix())
        ],
        "static_preflight": preflight,
        "artifacts": {},
    }
    # The report does not hash itself; all data-bearing artifacts are covered.
    for path in (candidate_path, holdout_path, trainable_path, quarantine_path, manifest_path):
        report["artifacts"][path.name] = {"sha256": sha256_file(path), "path": path.as_posix()}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _row_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    quality = row.get("data_quality")
    quality = quality if isinstance(quality, Mapping) else {}
    spec_id = row.get("spec_id") or quality.get("spec_id")
    spec_hash = row.get("spec_hash") or quality.get("spec_hash")
    if not isinstance(spec_id, str) or not spec_id:
        raise ValueError("label/candidate row is missing spec_id")
    if not isinstance(spec_hash, str) or not _HEX_256_RE.fullmatch(spec_hash):
        raise ValueError(f"row {spec_id} is missing a SHA-256 spec_hash")
    return spec_id, spec_hash.casefold()


def merge_exact_regime_labels(
    artifact_dir: Path,
    labels_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Attach exact-regime-v2 label evidence without changing spec identity.

    ``labels_path`` must contain one JSON object per labeled specification with
    ``spec_id``, ``spec_hash`` (either top-level or under ``data_quality``), and
    a nested ``trainability_label``.  Full pass@k rows are accepted because the
    candidate's ``data_quality`` identity is preserved by JSON expansion.
    Labels may cover a subset of candidates; unmatched candidates remain
    explicitly unlabeled and cannot enter ``trainable_candidates.jsonl``.
    """
    if artifact_dir.resolve() == output_dir.resolve():
        raise ValueError("label merge output_dir must differ from the source artifact_dir")
    candidate_path = artifact_dir / "verified_candidates.jsonl"
    holdout_path = artifact_dir / "holdout.jsonl"
    candidates = _read_jsonl_objects(candidate_path)
    holdout = _read_jsonl_objects(holdout_path)
    labels = _read_jsonl_objects(labels_path)

    candidates_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    for candidate in candidates:
        spec_id, spec_hash = _row_identity(candidate)
        if spec_id in candidates_by_id:
            raise ValueError(f"duplicate candidate spec_id: {spec_id}")
        candidates_by_id[spec_id] = (spec_hash, candidate)

    labels_by_id: dict[str, dict[str, Any]] = {}
    for label_record in labels:
        spec_id, spec_hash = _row_identity(label_record)
        if spec_id in labels_by_id:
            raise ValueError(f"duplicate label spec_id: {spec_id}")
        if spec_id not in candidates_by_id:
            raise ValueError(f"label references unknown candidate spec_id: {spec_id}")
        candidate_hash = candidates_by_id[spec_id][0]
        if spec_hash != candidate_hash:
            raise ValueError(
                f"label spec_hash mismatch for {spec_id}: {spec_hash} != {candidate_hash}"
            )
        evidence = label_record.get("trainability_label")
        if not isinstance(evidence, Mapping):
            raise ValueError(f"label {spec_id} is missing trainability_label")
        labels_by_id[spec_id] = dict(evidence)

    merged_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        spec_id, _ = _row_identity(candidate)
        merged = dict(candidate)
        if spec_id in labels_by_id:
            merged["trainability_label"] = labels_by_id[spec_id]
            derived, reasons = validate_trainability_label(merged)
            if reasons:
                raise ValueError(f"invalid exact-regime label for {spec_id}: {reasons}")
            quality = dict(merged.get("data_quality", {}))
            quality["trainability"] = derived
            merged["data_quality"] = quality
            merged["trainable"] = bool(derived["trainable"])
            merged["trainability_status"] = derived["status"]
        merged_candidates.append(merged)

    trainable_candidates = [
        row for row in merged_candidates if row.get("trainable") is True
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_candidate_path = output_dir / "verified_candidates.jsonl"
    merged_holdout_path = output_dir / "holdout.jsonl"
    trainable_path = output_dir / "trainable_candidates.jsonl"
    _write_jsonl(merged_candidate_path, merged_candidates)
    _write_jsonl(merged_holdout_path, holdout)
    _write_jsonl(trainable_path, trainable_candidates)

    preflight = check_verified_artifacts(output_dir)
    if not preflight["passed"]:
        raise RuntimeError(f"label-merged artifacts failed static preflight: {preflight}")
    report = {
        "merge_schema": "exact-regime-label-merge-v1",
        "trainability_schema": TRAINABILITY_SCHEMA,
        "source": {
            "artifact_dir": artifact_dir.as_posix(),
            "verified_candidates_sha256": sha256_file(candidate_path),
            "holdout_sha256": sha256_file(holdout_path),
            "source_report_sha256": (
                sha256_file(artifact_dir / "report.json")
                if (artifact_dir / "report.json").is_file()
                else None
            ),
            "labels_path": labels_path.as_posix(),
            "labels_sha256": sha256_file(labels_path),
        },
        "counts": {
            "verified_candidates": len(merged_candidates),
            "holdout": len(holdout),
            "labels_supplied": len(labels),
            "labels_matched": len(labels_by_id),
            "trainable_candidates": len(trainable_candidates),
            "unlabeled_candidates": sum(
                row.get("trainability_status") == "unlabeled" for row in merged_candidates
            ),
        },
        "artifacts": {
            path.name: {"path": path.as_posix(), "sha256": sha256_file(path)}
            for path in (merged_candidate_path, merged_holdout_path, trainable_path)
        },
        "static_preflight": preflight,
    }
    (output_dir / "label_merge_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="source JSONL file(s)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", default="phychip-rlvr-v1")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_verified_dataset(
        args.sources,
        args.output_dir,
        holdout_fraction=args.holdout_fraction,
        split_seed=args.split_seed,
    )
    print(json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
