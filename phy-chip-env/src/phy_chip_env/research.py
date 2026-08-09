"""Resumable verifier-bound research episodes and optimization baselines."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from string import Template
from typing import IO, Any, Protocol

from .observation import Observation, observe_completion
from .registry import get_harness
from .scorer import SCORER_SEMANTICS

RESEARCH_EVENT_SCHEMA = "phychip-research-event-v1"
RESEARCH_BENCHMARK_SCHEMA = "phychip-closed-loop-benchmark-v1"
CODEX_CONTROLLER_PROTOCOL = "phychip-codex-controller-v3-redacted-failure-detail"
CODEX_FEEDBACK_BLIND_PROTOCOL = "phychip-codex-controller-v2-matched-redaction"
_ZERO_HASH = "0" * 64
_MAX_PROPOSAL_NETLIST_CHARS = 20_000
_MAX_PROPOSAL_OUTPUT_BYTES = 128_000
_MAX_VERIFIER_AUGMENTATION_CHARS = 4_096
_PASSIVE_AUGMENTATION = re.compile(
    r"^[RC][A-Za-z0-9_.-]*\s+[A-Za-z0-9_.+-]+\s+"
    r"[A-Za-z0-9_.+-]+\s+[0-9.eE+-]+[A-Za-z]*$"
)
_SPICE_SCALAR = re.compile(
    r"^(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?P<suffix>[A-Za-z]*)$"
)
_SPICE_SUFFIX_SCALE = {
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
_RESERVED_EVENT_FIELDS = {
    "schema",
    "timestamp_utc",
    "previous_event_hash",
    "event_hash",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _positive_passive_value(device_kind: str, token: str) -> bool:
    match = _SPICE_SCALAR.fullmatch(token)
    if match is None:
        return False
    suffix = match["suffix"].lower()
    if suffix not in _SPICE_SUFFIX_SCALE:
        return False
    value = float(match["number"]) * _SPICE_SUFFIX_SCALE[suffix]
    if not math.isfinite(value) or value <= 0.0:
        return False
    if device_kind.lower() == "r":
        return 1e-3 <= value <= 1e12
    return 1e-18 <= value <= 1.0


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    low: float
    high: float
    scale: str = "linear"

    def validate(self) -> None:
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValueError(f"{self.name}: bounds must be finite")
        if self.low >= self.high:
            raise ValueError(f"{self.name}: low must be less than high")
        if self.scale not in {"linear", "log"}:
            raise ValueError(f"{self.name}: scale must be linear or log")
        if self.scale == "log" and self.low <= 0.0:
            raise ValueError(f"{self.name}: log-scaled low must be positive")

    def from_unit(self, unit: float) -> float:
        unit = min(max(float(unit), 0.0), 1.0)
        if self.scale == "log":
            low_log = math.log(self.low)
            value = math.exp(low_log + unit * (math.log(self.high) - low_log))
        else:
            value = self.low + unit * (self.high - self.low)
        return min(max(value, self.low), self.high)

    def to_unit(self, value: float) -> float:
        bounded = min(max(float(value), self.low), self.high)
        if self.scale == "log":
            return (math.log(bounded) - math.log(self.low)) / (
                math.log(self.high) - math.log(self.low)
            )
        return (bounded - self.low) / (self.high - self.low)


@dataclass(frozen=True)
class TopologyTemplate:
    template_id: str
    netlist_template: str
    parameters: tuple[ParameterSpec, ...]

    def validate(self) -> None:
        if not self.template_id:
            raise ValueError("template_id is required")
        if ".end" not in self.netlist_template.lower():
            raise ValueError(f"{self.template_id}: template must contain .end")
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.template_id}: parameter names must be unique")
        for parameter in self.parameters:
            parameter.validate()
            if f"${parameter.name}" not in self.netlist_template:
                raise ValueError(
                    f"{self.template_id}: missing ${parameter.name} placeholder"
                )

    def render(self, values: dict[str, float]) -> str:
        self.validate()
        expected = {parameter.name for parameter in self.parameters}
        if set(values) != expected:
            raise ValueError(
                f"{self.template_id}: expected parameters {sorted(expected)}, "
                f"got {sorted(values)}"
            )
        substitutions = {}
        for parameter in self.parameters:
            value = float(values[parameter.name])
            if not parameter.low <= value <= parameter.high:
                raise ValueError(f"{parameter.name}: value outside frozen bounds")
            substitutions[parameter.name] = f"{value:.12g}"
        return Template(self.netlist_template).substitute(substitutions)


@dataclass(frozen=True)
class ResearchTask:
    task_id: str
    circuit_id: str
    prompt: str
    target_specs: dict[str, list[float]]
    allowed_topologies: tuple[str, ...]
    max_verifier_calls: int
    max_codex_calls: int
    timeout_s: float
    seed: int
    fixed_template: TopologyTemplate | None = None
    initial_netlist: str | None = None
    verifier_augmentation: str | None = None
    controller_task_id: str | None = None

    def validate(self) -> None:
        if not self.task_id or not self.prompt:
            raise ValueError("task_id and prompt are required")
        if self.controller_task_id is not None and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
            self.controller_task_id,
        ):
            raise ValueError(f"{self.task_id}: unsafe controller_task_id")
        harness = get_harness(self.circuit_id)
        if harness is None:
            raise ValueError(f"{self.task_id}: unknown circuit_id {self.circuit_id!r}")
        if not self.target_specs:
            raise ValueError(f"{self.task_id}: target_specs cannot be empty")
        unknown = set(self.target_specs) - set(harness.FOM_NAMES)
        if unknown:
            raise ValueError(f"{self.task_id}: unknown target metrics {sorted(unknown)}")
        for name, bounds in self.target_specs.items():
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f"{self.task_id}/{name}: expected [low, high]")
            low, high = map(float, bounds)
            if not (math.isfinite(low) and math.isfinite(high) and low <= high):
                raise ValueError(f"{self.task_id}/{name}: invalid bounds")
        if not self.allowed_topologies:
            raise ValueError(f"{self.task_id}: allowed_topologies cannot be empty")
        if self.max_verifier_calls <= 0 or self.max_codex_calls <= 0:
            raise ValueError(f"{self.task_id}: call budgets must be positive")
        if self.max_codex_calls > self.max_verifier_calls:
            raise ValueError(f"{self.task_id}: codex budget exceeds verifier budget")
        if self.timeout_s <= 0:
            raise ValueError(f"{self.task_id}: timeout must be positive")
        if self.initial_netlist is not None:
            if not self.initial_netlist.strip():
                raise ValueError(f"{self.task_id}: initial_netlist cannot be empty")
            if ".end" not in self.initial_netlist.lower():
                raise ValueError(
                    f"{self.task_id}: initial_netlist must terminate with .end"
                )
        if self.verifier_augmentation is not None:
            augmentation = self.verifier_augmentation.strip()
            if not augmentation or len(augmentation) > _MAX_VERIFIER_AUGMENTATION_CHARS:
                raise ValueError(
                    f"{self.task_id}: verifier augmentation must contain 1-4096 characters"
                )
            device_names = []
            for line in augmentation.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("*"):
                    continue
                if not _PASSIVE_AUGMENTATION.fullmatch(stripped):
                    raise ValueError(
                        f"{self.task_id}: verifier augmentation permits passive R/C lines only"
                    )
                parts = stripped.split()
                if not _positive_passive_value(parts[0][0], parts[3]):
                    raise ValueError(
                        f"{self.task_id}: verifier augmentation requires a "
                        "positive physical SPICE value"
                    )
                device_names.append(parts[0].lower())
            if not device_names or len(device_names) != len(set(device_names)):
                raise ValueError(
                    f"{self.task_id}: verifier augmentation device names must be unique"
                )
        if self.fixed_template is not None:
            self.fixed_template.validate()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ResearchTask:
        template_value = value.get("fixed_template")
        template = None
        if template_value is not None:
            template = TopologyTemplate(
                template_id=str(template_value["template_id"]),
                netlist_template=str(template_value["netlist_template"]),
                parameters=tuple(
                    ParameterSpec(
                        name=str(parameter["name"]),
                        low=float(parameter["low"]),
                        high=float(parameter["high"]),
                        scale=str(parameter.get("scale", "linear")),
                    )
                    for parameter in template_value["parameters"]
                ),
            )
        task = cls(
            task_id=str(value["task_id"]),
            circuit_id=str(value["circuit_id"]),
            prompt=str(value["prompt"]),
            target_specs={
                str(name): [float(bounds[0]), float(bounds[1])]
                for name, bounds in value["target_specs"].items()
            },
            allowed_topologies=tuple(map(str, value["allowed_topologies"])),
            max_verifier_calls=int(value["max_verifier_calls"]),
            max_codex_calls=int(value["max_codex_calls"]),
            timeout_s=float(value["timeout_s"]),
            seed=int(value["seed"]),
            fixed_template=template,
            initial_netlist=(
                str(value["initial_netlist"])
                if value.get("initial_netlist") is not None
                else None
            ),
            verifier_augmentation=(
                str(value["verifier_augmentation"])
                if value.get("verifier_augmentation") is not None
                else None
            ),
            controller_task_id=(
                str(value["controller_task_id"])
                if value.get("controller_task_id") is not None
                else None
            ),
        )
        task.validate()
        return task

    def identity(self) -> dict[str, Any]:
        identity = {
            "task_id": self.task_id,
            "circuit_id": self.circuit_id,
            "prompt": self.prompt,
            "target_specs": self.target_specs,
            "allowed_topologies": self.allowed_topologies,
            "max_verifier_calls": self.max_verifier_calls,
            "max_codex_calls": self.max_codex_calls,
            "timeout_s": self.timeout_s,
            "seed": self.seed,
            "fixed_template": (
                asdict(self.fixed_template)
                if self.fixed_template is not None
                else None
            ),
        }
        if self.initial_netlist is not None:
            identity["initial_netlist"] = self.initial_netlist
        if self.verifier_augmentation is not None:
            identity["verifier_augmentation"] = self.verifier_augmentation
        if self.controller_task_id is not None:
            identity["controller_task_id"] = self.controller_task_id
        return identity

    def controller_view(self) -> dict[str, Any]:
        """Return only information shared with either experimental arm."""

        return {
            "task_id": self.controller_task_id or self.task_id,
            "circuit_id": self.circuit_id,
            "prompt": self.prompt,
            "target_specs": self.target_specs,
            "allowed_topologies": self.allowed_topologies,
            "max_verifier_calls": self.max_verifier_calls,
            "max_codex_calls": self.max_codex_calls,
            "timeout_s": self.timeout_s,
        }


def _top_level_devices(netlist: str) -> tuple[tuple[str, ...], ...]:
    devices = []
    subcircuit_depth = 0
    in_control = False
    for raw_line in netlist.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("*"):
            continue
        lowered = stripped.lower()
        if lowered.startswith(".control"):
            in_control = True
            continue
        if lowered.startswith(".endc"):
            in_control = False
            continue
        if in_control:
            continue
        if lowered.startswith(".subckt"):
            subcircuit_depth += 1
            continue
        if lowered.startswith(".ends"):
            subcircuit_depth = max(0, subcircuit_depth - 1)
            continue
        if subcircuit_depth or stripped.startswith("."):
            continue
        devices.append(tuple(part.lower() for part in stripped.split()))
    return tuple(devices)


def _augmentation_anchor_is_load_bearing(task: ResearchTask, netlist: str) -> bool:
    augmentation_lines = [
        line.split()
        for line in (task.verifier_augmentation or "").splitlines()
        if line.strip() and not line.lstrip().startswith("*")
    ]
    if not augmentation_lines:
        return True
    devices = _top_level_devices(netlist)
    reserved_names = {parts[0].lower() for parts in augmentation_lines}
    if any(device and device[0] in reserved_names for device in devices):
        return False

    resistor_edges: set[frozenset[str]] = set()
    capacitor_edges: set[frozenset[str]] = set()
    for device in devices:
        if len(device) < 3:
            continue
        edge = frozenset(device[1:3])
        if device[0].startswith("r"):
            resistor_edges.add(edge)
        elif device[0].startswith("c"):
            capacitor_edges.add(edge)

    def joined(edges: set[frozenset[str]], left: str, right: str) -> bool:
        return frozenset((left, right)) in edges

    def active_consumes(node: str) -> bool:
        for device in devices:
            if not device:
                continue
            if device[0].startswith(("e", "g")) and len(device) >= 5:
                if node in device[3:5]:
                    return True
            if device[0].startswith(("b", "x")) and node in device[1:]:
                return True
        return False

    family_inputs = {
        "inv-amp": {"vin"},
        "noninv-amp": {"vin"},
        "sallen-key": {"vin"},
        "diff-pair": {"vp", "vn"},
        "buffer": {"vin"},
        "summing-amp": {"v1", "v2", "v3"},
        "integrator": {"vin"},
        "differentiator": {"vin"},
    }.get(task.circuit_id, set())
    augmentation_nodes = {
        node.lower()
        for parts in augmentation_lines
        for node in parts[1:3]
        if node.lower() not in {"0", "gnd"}
    }
    internal_nodes = augmentation_nodes - family_inputs - {"vout", "out"}
    if not internal_nodes:
        return "vout" in augmentation_nodes or "out" in augmentation_nodes
    if len(internal_nodes) != 1:
        return False
    anchor = next(iter(internal_nodes))

    if task.circuit_id == "inv-amp":
        return (
            joined(resistor_edges, anchor, "vin")
            and joined(resistor_edges, anchor, "vout")
            and active_consumes(anchor)
        )
    if task.circuit_id == "noninv-amp":
        return (
            joined(resistor_edges, anchor, "0")
            and joined(resistor_edges, anchor, "vout")
            and active_consumes(anchor)
        )
    if task.circuit_id == "summing-amp":
        return all(
            joined(resistor_edges, anchor, node)
            for node in ("v1", "v2", "v3", "vout")
        ) and active_consumes(anchor)
    if task.circuit_id == "integrator":
        return (
            joined(resistor_edges, anchor, "vin")
            and joined(capacitor_edges, anchor, "vout")
            and active_consumes(anchor)
        )
    if task.circuit_id == "sallen-key":
        has_ground_cap = joined(capacitor_edges, anchor, "0")
        has_resistor = any(anchor in edge for edge in resistor_edges)
        return has_ground_cap and has_resistor and active_consumes(anchor)
    if task.circuit_id == "differentiator":
        graph: dict[str, list[tuple[str, bool]]] = {}
        for edge, is_capacitor in (
            *((edge, False) for edge in resistor_edges),
            *((edge, True) for edge in capacitor_edges),
        ):
            if len(edge) != 2:
                continue
            left, right = tuple(edge)
            graph.setdefault(left, []).append((right, is_capacitor))
            graph.setdefault(right, []).append((left, is_capacitor))
        pending = [(anchor, False)]
        visited: set[tuple[str, bool]] = set()
        reaches_input_with_capacitor = False
        while pending:
            node, used_capacitor = pending.pop()
            state = (node, used_capacitor)
            if state in visited:
                continue
            visited.add(state)
            if node == "vin" and used_capacitor:
                reaches_input_with_capacitor = True
                break
            pending.extend(
                (neighbor, used_capacitor or is_capacitor)
                for neighbor, is_capacitor in graph.get(node, [])
                if neighbor not in {"0", "vout"}
            )
        return (
            joined(resistor_edges, anchor, "vout")
            and reaches_input_with_capacitor
            and active_consumes(anchor)
        )
    return False


def _anchor_rejection_completion() -> str:
    return (
        "```spice\n"
        "* verifier_anchor_rejected\n"
        "Rverifier_anchor_rejected verifier_anchor_rejected 0 1\n"
        ".op\n.end\n```"
    )


def verifier_completion(task: ResearchTask, netlist: str) -> str:
    """Build exact scored bytes while keeping fixed verifier loads prompt-blind."""

    stripped = netlist.strip()
    if task.verifier_augmentation is None:
        return f"```spice\n{stripped}\n```"
    if not _augmentation_anchor_is_load_bearing(task, stripped):
        return _anchor_rejection_completion()
    lines = stripped.splitlines()
    end_positions = [
        index for index, line in enumerate(lines) if line.strip().lower() == ".end"
    ]
    if not end_positions:
        raise ValueError("netlist lacks a standalone .end for verifier augmentation")
    position = end_positions[0]
    augmented = [
        *lines[:position],
        task.verifier_augmentation.strip(),
        *lines[position:],
    ]
    joined = "\n".join(augmented)
    return f"```spice\n{joined}\n```"


def load_benchmark(path: str | Path) -> tuple[dict[str, Any], tuple[ResearchTask, ...]]:
    """Load and validate a frozen research benchmark."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != RESEARCH_BENCHMARK_SCHEMA:
        raise ValueError(f"unsupported benchmark schema in {source}")
    task_values = payload.get("tasks")
    if not isinstance(task_values, list) or not task_values:
        raise ValueError("benchmark tasks must be a non-empty list")
    tasks = tuple(ResearchTask.from_dict(value) for value in task_values)
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("benchmark task_id values must be unique")
    declared_hash = payload.get("task_set_sha256")
    computed_hash = _content_hash([task.identity() for task in tasks])
    if (
        not isinstance(declared_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", declared_hash)
        or declared_hash != computed_hash
    ):
        raise ValueError("benchmark task_set_sha256 mismatch")
    declared_protocol_hash = payload.get("protocol_sha256")
    protocol_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"protocol_sha256", "task_set_sha256"}
    }
    computed_protocol_hash = _content_hash(protocol_payload)
    if (
        not isinstance(declared_protocol_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", declared_protocol_hash)
        or declared_protocol_hash != computed_protocol_hash
    ):
        raise ValueError("benchmark protocol_sha256 mismatch")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"tasks", "task_set_sha256", "protocol_sha256"}
    }
    return {
        **metadata,
        "task_set_sha256": computed_hash,
        "protocol_sha256": computed_protocol_hash,
    }, tasks


@dataclass(frozen=True)
class Proposal:
    action: str
    hypothesis: str
    topology: str
    rationale: str
    netlist: str


class Controller(Protocol):
    controller_id: str

    def propose(
        self,
        task: ResearchTask,
        history: list[dict[str, Any]],
    ) -> Proposal: ...


class CodexCLIController:
    """Stateless structured controller using `codex exec` programmatically."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-sol",
        timeout_s: float = 300.0,
        executable: str = "codex",
    ) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self.executable = executable
        fingerprint = _content_hash(
            {
                "protocol": CODEX_CONTROLLER_PROTOCOL,
                "model": model,
            }
        )[:16]
        self.controller_id = f"codex-cli:{model}:{fingerprint}"

    @staticmethod
    def _schema() -> dict[str, Any]:
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "action",
                "hypothesis",
                "topology",
                "rationale",
                "netlist",
            ],
            "properties": {
                "action": {"type": "string", "enum": ["propose", "stop"]},
                "hypothesis": {"type": "string", "maxLength": 2_000},
                "topology": {"type": "string", "maxLength": 256},
                "rationale": {"type": "string", "maxLength": 4_000},
                "netlist": {
                    "type": "string",
                    "maxLength": _MAX_PROPOSAL_NETLIST_CHARS,
                },
            },
        }

    @staticmethod
    def _prompt(
        task: ResearchTask,
        history: list[dict[str, Any]],
        *,
        feedback_visible: bool = True,
    ) -> str:
        prior = history[-3:]
        prior_controller_calls = sum(
            event.get("source") == "controller_proposal" for event in history
        )
        visible_trials = []
        candidate_fields = (
            "source",
            "attempt",
            "hypothesis",
            "topology",
            "netlist",
        )
        for event in prior:
            trial = {key: event[key] for key in candidate_fields if key in event}
            if feedback_visible:
                failure = event.get("failure")
                stable_failure = None
                if isinstance(failure, dict):
                    stable_failure = {
                        "stage": failure.get("stage"),
                        "code": failure.get("code"),
                    }
                trial["feedback_packet"] = {
                    "measurements": event.get("measurements", {}),
                    "requested_bounds": event.get("requested_bounds", {}),
                    "failure": stable_failure,
                    "score": event.get("score", {}),
                }
            else:
                trial["feedback_packet"] = {"status": "redacted"}
            visible_trials.append(trial)
        payload = {
            "task": task.controller_view(),
            "recent_trials": visible_trials,
            "remaining_codex_calls": task.max_codex_calls - prior_controller_calls,
        }
        return (
            "You are the topology and experiment-selection controller for an "
            "analog-circuit research loop. Return only the requested JSON. "
            "Choose one allowed topology, state a falsifiable sizing hypothesis, "
            "and propose one complete ngspice netlist. Use the named ports in the "
            "task. Include a minimal .op or safe .control/op block and terminate "
            "with .end. Do not use .include, .lib, file I/O, shell/control escape "
            "commands, hidden stimuli, or a dependent source that directly "
            "synthesizes a target metric. The scorer-v9 result is authoritative; "
            "measurements are diagnostic feedback only. Revise from failures and "
            "numeric misses. If no safe discriminating experiment remains, choose "
            "action=stop with an empty netlist.\n\n"
            f"{_canonical_json(payload)}"
        )

    def propose(
        self,
        task: ResearchTask,
        history: list[dict[str, Any]],
    ) -> Proposal:
        task.validate()
        with tempfile.TemporaryDirectory(prefix="phychip-codex-") as tmp:
            root = Path(tmp)
            codex_home = root / "codex-home"
            codex_home.mkdir(mode=0o700)
            configured_home = Path(
                os.environ.get("CODEX_HOME", Path.home() / ".codex")
            )
            configured_auth = configured_home / "auth.json"
            if configured_auth.is_file():
                isolated_auth = codex_home / "auth.json"
                shutil.copyfile(configured_auth, isolated_auth)
                isolated_auth.chmod(0o600)
            elif "OPENAI_API_KEY" not in os.environ:
                raise RuntimeError(
                    "Codex authentication is unavailable: expected CODEX_HOME/auth.json "
                    "or OPENAI_API_KEY"
                )
            schema_path = root / "proposal.schema.json"
            output_path = root / "proposal.json"
            schema_path.write_text(
                json.dumps(self._schema(), indent=2) + "\n",
                encoding="utf-8",
            )
            command = [
                self.executable,
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--disable",
                "shell_tool",
                "--disable",
                "code_mode",
                "--disable",
                "code_mode_host",
                "--disable",
                "apps",
                "--disable",
                "browser_use",
                "--disable",
                "browser_use_external",
                "--disable",
                "in_app_browser",
                "--disable",
                "multi_agent",
                "--disable",
                "skill_search",
                "--disable",
                "workspace_dependencies",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--color",
                "never",
                "--model",
                self.model,
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "-C",
                str(root),
                "-",
            ]
            allowed_environment = {
                name: os.environ[name]
                for name in (
                    "OPENAI_API_KEY",
                    "PATH",
                    "SSL_CERT_FILE",
                    "SSL_CERT_DIR",
                )
                if name in os.environ
            }
            allowed_environment.update(
                {
                    "CODEX_HOME": str(codex_home),
                    "HOME": str(root),
                    "TMPDIR": str(root),
                    "NO_COLOR": "1",
                }
            )
            completed = subprocess.run(
                command,
                input=self._prompt(task, history),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_s,
                check=False,
                env=allowed_environment,
            )
            if completed.returncode != 0:
                detail = " ".join((completed.stderr or "").split())[-2_000:]
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(
                    f"codex exec failed with exit {completed.returncode}{suffix}"
                )
            if not output_path.is_file():
                raise RuntimeError("codex exec did not write a proposal")
            if output_path.stat().st_size > _MAX_PROPOSAL_OUTPUT_BYTES:
                raise ValueError("Codex proposal output exceeds 128,000 bytes")
            value = json.loads(output_path.read_text(encoding="utf-8"))
        proposal = Proposal(
            action=str(value["action"]),
            hypothesis=str(value["hypothesis"]),
            topology=str(value["topology"]),
            rationale=str(value["rationale"]),
            netlist=str(value["netlist"]),
        )
        for name, value, limit in (
            ("hypothesis", proposal.hypothesis, 2_000),
            ("topology", proposal.topology, 256),
            ("rationale", proposal.rationale, 4_000),
            ("netlist", proposal.netlist, _MAX_PROPOSAL_NETLIST_CHARS),
        ):
            if len(value) > limit:
                raise ValueError(f"Codex {name} exceeds its {limit}-character limit")
        if proposal.action == "propose" and not proposal.netlist.strip():
            raise ValueError("Codex returned action=propose with an empty netlist")
        if proposal.action == "stop" and proposal.netlist.strip():
            raise ValueError("Codex returned action=stop with a non-empty netlist")
        if proposal.action == "propose" and proposal.topology not in task.allowed_topologies:
            raise ValueError(
                f"Codex selected topology {proposal.topology!r}; allowed "
                f"{task.allowed_topologies}"
            )
        return proposal


class FeedbackBlindCodexCLIController(CodexCLIController):
    """Paired control that sees prior candidates but no simulator diagnostics."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-sol",
        timeout_s: float = 300.0,
        executable: str = "codex",
    ) -> None:
        super().__init__(model=model, timeout_s=timeout_s, executable=executable)
        fingerprint = _content_hash(
            {
                "protocol": CODEX_FEEDBACK_BLIND_PROTOCOL,
                "model": model,
            }
        )[:16]
        self.controller_id = f"codex-cli-feedback-blind:{model}:{fingerprint}"

    @staticmethod
    def _prompt(task: ResearchTask, history: list[dict[str, Any]]) -> str:
        return CodexCLIController._prompt(
            task,
            history,
            feedback_visible=False,
        )


class EpisodeLog:
    """Append-only, hash-chained JSONL event store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @staticmethod
    def _decode_and_verify(raw: str) -> list[dict[str, Any]]:
        events = [json.loads(line) for line in raw.splitlines() if line.strip()]
        previous = _ZERO_HASH
        for event in events:
            claimed = event.get("event_hash")
            payload = {key: value for key, value in event.items() if key != "event_hash"}
            if event.get("previous_event_hash") != previous:
                raise ValueError("episode log hash chain is broken")
            actual = _content_hash(payload)
            if claimed != actual:
                raise ValueError("episode log event hash mismatch")
            previous = claimed
        return events

    @contextmanager
    def _locked(self, *, exclusive: bool) -> Iterator[IO[str]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
            )
            try:
                yield handle
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def lease(self) -> Iterator[None]:
        """Hold one exclusive writer lease for a complete episode."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lease_path = self.path.with_name(f"{self.path.name}.lock")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lease_path, flags, 0o600)
        with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._locked(exclusive=False) as handle:
            handle.seek(0)
            return self._decode_and_verify(handle.read())

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        reserved = _RESERVED_EVENT_FIELDS.intersection(event)
        if reserved:
            raise ValueError(
                f"event cannot override reserved fields: {sorted(reserved)}"
            )
        with self._locked(exclusive=True) as handle:
            handle.seek(0)
            prior = self._decode_and_verify(handle.read())
            payload = {
                "schema": RESEARCH_EVENT_SCHEMA,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "previous_event_hash": (
                    prior[-1]["event_hash"] if prior else _ZERO_HASH
                ),
                **event,
            }
            complete = {**payload, "event_hash": _content_hash(payload)}
            handle.seek(0, os.SEEK_END)
            handle.write(_canonical_json(complete) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return complete


def _history_for_controller(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history = []
    for event in events:
        event_type = event.get("event_type")
        if event_type not in {"initial_observation", "trial"}:
            continue
        observation = event["observation"]
        if event_type == "initial_observation":
            history.append(
                {
                    "source": "frozen_initial_candidate",
                    "attempt": -1,
                    "hypothesis": "frozen benchmark starting point",
                    "topology": event["topology"],
                    "netlist": event["netlist"],
                    "measurements": observation["measurements"],
                    "requested_bounds": observation["requested_bounds"],
                    "failure": observation["failure"],
                    "score": observation["score"],
                }
            )
            continue
        history.append(
            {
                "source": "controller_proposal",
                "attempt": event["attempt"],
                "hypothesis": event["proposal"]["hypothesis"],
                "topology": event["proposal"]["topology"],
                "netlist": event["proposal"]["netlist"],
                "measurements": observation["measurements"],
                "requested_bounds": observation["requested_bounds"],
                "failure": observation["failure"],
                "score": observation["score"],
            }
        )
    return history


def research_episode_id(
    task: ResearchTask,
    controller_id: str,
    rollout_id: str = "0",
) -> str:
    """Return a deterministic identity for one task/controller rollout."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", rollout_id):
        raise ValueError(f"unsafe rollout_id: {rollout_id!r}")
    identity = {
        "task": task.identity(),
        "controller_id": controller_id,
        "scorer_semantics": SCORER_SEMANTICS,
    }
    # Preserve the already-recorded v1 identity for the default rollout.
    if rollout_id != "0":
        identity["rollout_id"] = rollout_id
    return _content_hash(identity)


class ResearchRunner:
    """Run or resume one model-guided episode under frozen budgets."""

    def __init__(self, observe_fn=observe_completion) -> None:
        self.observe_fn = observe_fn

    def run(
        self,
        task: ResearchTask,
        controller: Controller,
        log: EpisodeLog,
        *,
        rollout_id: str = "0",
    ) -> dict[str, Any]:
        with log.lease():
            return self._run_under_lease(
                task,
                controller,
                log,
                rollout_id=rollout_id,
            )

    def _run_under_lease(
        self,
        task: ResearchTask,
        controller: Controller,
        log: EpisodeLog,
        *,
        rollout_id: str,
    ) -> dict[str, Any]:
        task.validate()
        episode_id = research_episode_id(task, controller.controller_id, rollout_id)
        events = log.read()
        for event in events:
            if (
                event.get("episode_id") != episode_id
                or event.get("rollout_id", "0") != rollout_id
            ):
                raise ValueError("episode log belongs to a different task/controller")
        if events and events[-1].get("event_type") == "episode_end":
            return events[-1]

        initial_events = [
            event
            for event in events
            if event.get("event_type") == "initial_observation"
        ]
        if len(initial_events) > 1:
            raise ValueError("episode log contains multiple initial observations")
        if task.initial_netlist is not None and not initial_events:
            initial_observation = self.observe_fn(
                task.circuit_id,
                task.target_specs,
                verifier_completion(task, task.initial_netlist),
                timeout_s=task.timeout_s,
            )
            if initial_observation.score.all_specs_met:
                raise ValueError("frozen initial candidate already satisfies every spec")
            initial_event = log.append(
                {
                    "event_type": "initial_observation",
                    "episode_id": episode_id,
                    "rollout_id": rollout_id,
                    "task_id": task.task_id,
                    "source": "frozen_initial_candidate",
                    "topology": task.allowed_topologies[0],
                    "netlist": task.initial_netlist,
                    "observation": initial_observation.to_event(),
                }
            )
            events.append(initial_event)
            initial_events.append(initial_event)
        if task.initial_netlist is None and initial_events:
            raise ValueError("episode log has an initial observation for an unseeded task")

        history = _history_for_controller(events)
        trials = [event for event in events if event.get("event_type") == "trial"]
        call_events = [
            event
            for event in events
            if event.get("event_type") == "controller_call_started"
        ]
        completed_call_hashes = {
            event.get("controller_call_event_hash") for event in trials
        }
        orphaned_calls = [
            event
            for event in call_events
            if event.get("event_hash") not in completed_call_hashes
        ]
        verifier_calls = len(initial_events) + len(trials)
        controller_calls = len(call_events) if call_events else len(trials)
        if orphaned_calls:
            return log.append(
                {
                    "event_type": "episode_end",
                    "episode_id": episode_id,
                    "rollout_id": rollout_id,
                    "task_id": task.task_id,
                    "controller_id": controller.controller_id,
                    "stop_reason": "controller_error",
                    "detail": "orphaned controller call; redraw prohibited",
                    "verifier_calls": verifier_calls,
                    "codex_calls": controller_calls,
                    "controller_calls": controller_calls,
                }
            )
        while (
            verifier_calls < task.max_verifier_calls
            and controller_calls < task.max_codex_calls
        ):
            call_event = log.append(
                {
                    "event_type": "controller_call_started",
                    "episode_id": episode_id,
                    "rollout_id": rollout_id,
                    "task_id": task.task_id,
                    "attempt": controller_calls,
                    "controller_id": controller.controller_id,
                }
            )
            events.append(call_event)
            call_events.append(call_event)
            controller_calls += 1
            try:
                proposal = controller.propose(task, history)
            except Exception as exc:
                return log.append(
                    {
                        "event_type": "episode_end",
                        "episode_id": episode_id,
                        "rollout_id": rollout_id,
                        "task_id": task.task_id,
                        "controller_id": controller.controller_id,
                        "stop_reason": "controller_error",
                        "detail": f"{type(exc).__name__}: {exc}",
                        "verifier_calls": verifier_calls,
                        "codex_calls": controller_calls,
                        "controller_calls": controller_calls,
                    }
                )
            if len(proposal.netlist) > _MAX_PROPOSAL_NETLIST_CHARS:
                return log.append(
                    {
                        "event_type": "episode_end",
                        "episode_id": episode_id,
                        "rollout_id": rollout_id,
                        "task_id": task.task_id,
                        "controller_id": controller.controller_id,
                        "stop_reason": "controller_error",
                        "detail": "proposal netlist exceeds 20,000 characters",
                        "verifier_calls": verifier_calls,
                        "codex_calls": controller_calls,
                        "controller_calls": controller_calls,
                    }
                )
            if proposal.action == "stop":
                return log.append(
                    {
                        "event_type": "episode_end",
                        "episode_id": episode_id,
                        "rollout_id": rollout_id,
                        "task_id": task.task_id,
                        "controller_id": controller.controller_id,
                        "stop_reason": "controller_stop",
                        "detail": proposal.rationale,
                        "verifier_calls": verifier_calls,
                        "codex_calls": controller_calls,
                        "controller_calls": controller_calls,
                    }
                )

            completion = verifier_completion(task, proposal.netlist)
            observation: Observation = self.observe_fn(
                task.circuit_id,
                task.target_specs,
                completion,
                timeout_s=task.timeout_s,
            )
            event = log.append(
                {
                    "event_type": "trial",
                    "episode_id": episode_id,
                    "rollout_id": rollout_id,
                    "task_id": task.task_id,
                    "attempt": len(trials),
                    "controller_id": controller.controller_id,
                    "controller_call_event_hash": call_event["event_hash"],
                    "proposal": asdict(proposal),
                    "observation": observation.to_event(),
                }
            )
            trials.append(event)
            verifier_calls += 1
            history = _history_for_controller([*events, event])
            events.append(event)
            if observation.score.all_specs_met:
                return log.append(
                    {
                        "event_type": "episode_end",
                        "episode_id": episode_id,
                        "rollout_id": rollout_id,
                        "task_id": task.task_id,
                        "controller_id": controller.controller_id,
                        "stop_reason": "all_specs_met",
                        "winning_event_hash": event["event_hash"],
                        "verifier_calls": verifier_calls,
                        "codex_calls": controller_calls,
                        "controller_calls": controller_calls,
                    }
                )

        return log.append(
            {
                "event_type": "episode_end",
                "episode_id": episode_id,
                "rollout_id": rollout_id,
                "task_id": task.task_id,
                "controller_id": controller.controller_id,
                "stop_reason": "budget_exhausted",
                "verifier_calls": verifier_calls,
                "codex_calls": controller_calls,
                "controller_calls": controller_calls,
            }
        )


def _halton(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0
    value = index
    while value:
        factor /= base
        result += factor * (value % base)
        value //= base
    return result


_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


class DeterministicRandom:
    """Index-addressable seeded random sampler for a fixed topology."""

    def __init__(self, template: TopologyTemplate, *, seed: int) -> None:
        template.validate()
        self.template = template
        self.seed = seed

    def suggest(
        self,
        completed: list[tuple[dict[str, float], float]],
    ) -> dict[str, float]:
        index = len(completed)
        rng = random.Random(self.seed + index * 1_000_003)
        return {
            parameter.name: parameter.from_unit(rng.random())
            for parameter in self.template.parameters
        }


class DeterministicLatinHypercube:
    """Seeded Latin-hypercube sampler with exact resume by trial index."""

    def __init__(
        self,
        template: TopologyTemplate,
        *,
        seed: int,
        trials: int,
    ) -> None:
        template.validate()
        if trials < 1:
            raise ValueError("trials must be positive")
        self.template = template
        self.seed = seed
        self.trials = trials
        permutations = []
        for dimension, _ in enumerate(template.parameters):
            strata = list(range(trials))
            random.Random(seed + (dimension + 1) * 1_000_003).shuffle(strata)
            permutations.append(tuple(strata))
        self._permutations = tuple(permutations)

    def suggest(
        self,
        completed: list[tuple[dict[str, float], float]],
    ) -> dict[str, float]:
        index = len(completed)
        if index >= self.trials:
            raise ValueError("Latin-hypercube trial budget exhausted")
        values = {}
        for dimension, parameter in enumerate(self.template.parameters):
            stratum = self._permutations[dimension][index]
            jitter = random.Random(
                self.seed + (dimension + 1) * 10_000_019 + index * 1_000_003
            ).random()
            values[parameter.name] = parameter.from_unit(
                (stratum + jitter) / self.trials
            )
        return values


class DeterministicTPE:
    """Small deterministic Halton-started Parzen optimizer."""

    def __init__(
        self,
        template: TopologyTemplate,
        *,
        seed: int,
        startup_trials: int = 4,
        candidates: int = 64,
        elite_fraction: float = 0.25,
    ) -> None:
        template.validate()
        if len(template.parameters) > len(_PRIMES):
            raise ValueError("too many parameters for the deterministic Halton basis")
        if startup_trials < 1:
            raise ValueError("startup_trials must be positive")
        if candidates < 1:
            raise ValueError("candidates must be positive")
        if not 0.0 < elite_fraction < 1.0:
            raise ValueError("elite_fraction must be between zero and one")
        self.template = template
        self.seed = seed
        self.startup_trials = startup_trials
        self.candidates = candidates
        self.elite_fraction = elite_fraction

    def _startup(self, index: int) -> dict[str, float]:
        return {
            parameter.name: parameter.from_unit(
                _halton(index + 1 + self.seed % 997, _PRIMES[dimension])
            )
            for dimension, parameter in enumerate(self.template.parameters)
        }

    @staticmethod
    def _log_kde(unit: float, samples: list[float], bandwidth: float) -> float:
        normalizer = bandwidth * math.sqrt(2.0 * math.pi)
        density = sum(
            math.exp(-0.5 * ((unit - sample) / bandwidth) ** 2) / normalizer
            for sample in samples
        ) / len(samples)
        return math.log(max(density, 1e-300))

    def suggest(
        self,
        completed: list[tuple[dict[str, float], float]],
    ) -> dict[str, float]:
        if len(completed) < self.startup_trials:
            return self._startup(len(completed))

        ranked = sorted(completed, key=lambda item: item[1], reverse=True)
        elite_count = max(2, math.ceil(len(ranked) * self.elite_fraction))
        good = ranked[:elite_count]
        bad = ranked[elite_count:] or ranked[-2:]
        rng = random.Random(self.seed + len(completed) * 1_000_003)
        bandwidth = max(0.04, 0.30 / math.sqrt(len(completed)))
        best_values: dict[str, float] | None = None
        best_ratio = -math.inf
        for _ in range(self.candidates):
            values = {}
            ratio = 0.0
            for parameter in self.template.parameters:
                good_units = [
                    parameter.to_unit(point[parameter.name]) for point, _ in good
                ]
                bad_units = [
                    parameter.to_unit(point[parameter.name]) for point, _ in bad
                ]
                center = good_units[rng.randrange(len(good_units))]
                unit = min(max(rng.gauss(center, bandwidth), 0.0), 1.0)
                ratio += self._log_kde(unit, good_units, bandwidth)
                ratio -= self._log_kde(unit, bad_units, bandwidth)
                values[parameter.name] = parameter.from_unit(unit)
            if ratio > best_ratio:
                best_ratio = ratio
                best_values = values
        assert best_values is not None
        return best_values


def fixed_search_controller_id(task: ResearchTask, strategy: str) -> str:
    """Bind fixed-search resume identity to code, task, and full configuration."""

    if task.fixed_template is None:
        raise ValueError(f"{task.task_id}: fixed_template is required")
    if strategy not in {"random", "lhs", "tpe"}:
        raise ValueError("strategy must be 'random', 'lhs', or 'tpe'")
    configuration: dict[str, Any] = {
        "schema": "phychip-fixed-search-controller-v2",
        "strategy": strategy,
        "seed": task.seed,
        "max_verifier_calls": task.max_verifier_calls,
        "template": asdict(task.fixed_template),
        "scorer_semantics": SCORER_SEMANTICS,
        "research_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    if strategy == "tpe":
        configuration["optimizer"] = {
            "startup_trials": 4,
            "candidates": 64,
            "elite_fraction": 0.25,
            "startup_sequence": "halton",
        }
    elif strategy == "lhs":
        configuration["optimizer"] = {
            "trials": task.max_verifier_calls,
            "jitter": "seeded-uniform",
        }
    else:
        configuration["optimizer"] = {"sampling": "seeded-uniform"}
    digest = _content_hash(configuration)
    return f"fixed-{strategy}:{task.fixed_template.template_id}:{digest[:16]}"


def run_fixed_topology(
    task: ResearchTask,
    log: EpisodeLog,
    *,
    strategy: str = "tpe",
    rollout_id: str = "0",
    observe_fn=observe_completion,
) -> dict[str, Any]:
    """Run or resume one frozen fixed-topology parameter-search baseline."""

    with log.lease():
        return _run_fixed_topology_under_lease(
            task,
            log,
            strategy=strategy,
            rollout_id=rollout_id,
            observe_fn=observe_fn,
        )


def _run_fixed_topology_under_lease(
    task: ResearchTask,
    log: EpisodeLog,
    *,
    strategy: str,
    rollout_id: str,
    observe_fn,
) -> dict[str, Any]:
    task.validate()
    if task.fixed_template is None:
        raise ValueError(f"{task.task_id}: fixed_template is required")
    if strategy == "tpe":
        optimizer = DeterministicTPE(task.fixed_template, seed=task.seed)
    elif strategy == "random":
        optimizer = DeterministicRandom(task.fixed_template, seed=task.seed)
    elif strategy == "lhs":
        optimizer = DeterministicLatinHypercube(
            task.fixed_template,
            seed=task.seed,
            trials=task.max_verifier_calls,
        )
    else:
        raise ValueError("strategy must be 'random', 'lhs', or 'tpe'")
    controller_id = fixed_search_controller_id(task, strategy)
    episode_id = research_episode_id(task, controller_id, rollout_id)
    events = log.read()
    for event in events:
        if (
            event.get("episode_id") != episode_id
            or event.get("rollout_id", "0") != rollout_id
        ):
            raise ValueError("episode log belongs to a different task/controller")
    if events and events[-1].get("event_type") == "episode_end":
        return events[-1]

    completed = [
        (event["parameters"], float(event["observation"]["score"]["reward"]))
        for event in events
        if event.get("event_type") == "trial"
    ]
    attempts = len(completed)
    while attempts < task.max_verifier_calls:
        parameters = optimizer.suggest(completed)
        netlist = task.fixed_template.render(parameters)
        observation: Observation = observe_fn(
            task.circuit_id,
            task.target_specs,
            verifier_completion(task, netlist),
            timeout_s=task.timeout_s,
        )
        event = log.append(
            {
                "event_type": "trial",
                "episode_id": episode_id,
                "rollout_id": rollout_id,
                "task_id": task.task_id,
                "attempt": attempts,
                "controller_id": controller_id,
                "parameters": parameters,
                "observation": observation.to_event(),
            }
        )
        attempts += 1
        completed.append((parameters, observation.score.reward))
        if observation.score.all_specs_met:
            return log.append(
                {
                    "event_type": "episode_end",
                    "episode_id": episode_id,
                    "rollout_id": rollout_id,
                    "task_id": task.task_id,
                    "controller_id": controller_id,
                    "stop_reason": "all_specs_met",
                    "winning_event_hash": event["event_hash"],
                    "verifier_calls": attempts,
                    "codex_calls": 0,
                }
            )
    return log.append(
        {
            "event_type": "episode_end",
            "episode_id": episode_id,
            "rollout_id": rollout_id,
            "task_id": task.task_id,
            "controller_id": controller_id,
            "stop_reason": "budget_exhausted",
            "verifier_calls": attempts,
            "codex_calls": 0,
        }
    )


def validated_episode_outcome(
    log_path: str | Path,
    *,
    task: ResearchTask | None = None,
    reverify_fn=None,
) -> dict[str, Any]:
    """Validate one complete log and return its verifier-bound outcome."""

    events = EpisodeLog(log_path).read()
    endings = [event for event in events if event.get("event_type") == "episode_end"]
    if len(endings) != 1 or not events or events[-1] is not endings[0]:
        raise ValueError(f"{log_path}: expected exactly one episode_end")
    end = endings[0]
    task_id = end.get("task_id")
    episode_id = end.get("episode_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"{log_path}: episode_end lacks a task_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError(f"{log_path}: episode_end lacks an episode_id")
    if any(
        event.get("schema") != RESEARCH_EVENT_SCHEMA
        or event.get("task_id") != task_id
        or event.get("episode_id") != episode_id
        for event in events
    ):
        raise ValueError(f"{log_path}: event schema, task, or episode identity mismatch")
    initial_events = [
        event
        for event in events
        if event.get("event_type") == "initial_observation"
    ]
    if len(initial_events) > 1:
        raise ValueError(f"{log_path}: expected at most one initial observation")
    if initial_events:
        initial = initial_events[0]
        if initial is not events[0]:
            raise ValueError(f"{log_path}: initial observation must be the first event")
        observation = initial.get("observation")
        score = observation.get("score") if isinstance(observation, dict) else None
        if (
            initial.get("source") != "frozen_initial_candidate"
            or not isinstance(initial.get("netlist"), str)
            or not initial["netlist"].strip()
            or not isinstance(score, dict)
            or observation.get("scorer_semantics") != SCORER_SEMANTICS
            or score.get("all_specs_met") is not False
        ):
            raise ValueError(f"{log_path}: invalid frozen initial observation")
    trials = [event for event in events if event.get("event_type") == "trial"]
    if [event.get("attempt") for event in trials] != list(range(len(trials))):
        raise ValueError(f"{log_path}: trial attempts are not contiguous from zero")
    call_events = [
        event
        for event in events
        if event.get("event_type") == "controller_call_started"
    ]
    if call_events and [event.get("attempt") for event in call_events] != list(
        range(len(call_events))
    ):
        raise ValueError(
            f"{log_path}: controller-call attempts are not contiguous from zero"
        )
    call_hashes = {event["event_hash"] for event in call_events}
    linked_call_hashes = {
        event.get("controller_call_event_hash") for event in trials
    }
    if call_events and (
        None in linked_call_hashes
        or not linked_call_hashes.issubset(call_hashes)
        or len(linked_call_hashes) != len(trials)
    ):
        raise ValueError(f"{log_path}: trials do not bind unique controller calls")
    controller_ids = {
        event.get("controller_id") for event in trials if event.get("controller_id")
    }
    controller_ids.update(
        event.get("controller_id")
        for event in call_events
        if event.get("controller_id")
    )
    if end.get("controller_id"):
        controller_ids.add(end["controller_id"])
    if len(controller_ids) > 1:
        raise ValueError(f"{log_path}: trials use multiple controllers")
    rollout_ids = {str(event.get("rollout_id", "0")) for event in events}
    if len(rollout_ids) != 1:
        raise ValueError(f"{log_path}: events use multiple rollout identifiers")
    rollout_id = next(iter(rollout_ids))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", rollout_id):
        raise ValueError(f"{log_path}: unsafe rollout identifier")
    for trial in trials:
        observation = trial.get("observation")
        score = observation.get("score") if isinstance(observation, dict) else None
        if (
            not isinstance(score, dict)
            or observation.get("scorer_semantics") != SCORER_SEMANTICS
            or not isinstance(score.get("all_specs_met"), bool)
        ):
            raise ValueError(f"{log_path}: trial lacks canonical scorer-v9 evidence")
    verifier_calls = end.get("verifier_calls")
    if (
        not isinstance(verifier_calls, int)
        or isinstance(verifier_calls, bool)
        or verifier_calls < 0
        or verifier_calls != len(trials) + len(initial_events)
    ):
        raise ValueError(f"{log_path}: verifier_calls does not match trial evidence")
    stop_reason = end.get("stop_reason")
    success = stop_reason == "all_specs_met"
    if call_events:
        recorded_controller_calls = end.get(
            "controller_calls",
            end.get("codex_calls"),
        )
        if recorded_controller_calls != len(call_events):
            raise ValueError(
                f"{log_path}: controller-call count does not match write-ahead events"
            )
        unmatched_calls = call_hashes - linked_call_hashes
        if success and unmatched_calls:
            raise ValueError(f"{log_path}: successful episode has an orphaned call")
        if len(unmatched_calls) > 1:
            raise ValueError(f"{log_path}: episode has multiple orphaned calls")
    winning_trial = None
    if success:
        winning_hash = end.get("winning_event_hash")
        winning_trials = [
            event for event in trials if event.get("event_hash") == winning_hash
        ]
        if len(winning_trials) != 1:
            raise ValueError(
                f"{log_path}: winning event does not identify exactly one trial"
            )
        winning_trial = winning_trials[0]
        if winning_trial is not trials[-1]:
            raise ValueError(f"{log_path}: winning event is not the final trial")
        if winning_trial["observation"]["score"].get("all_specs_met") is not True:
            raise ValueError(
                f"{log_path}: winning trial lacks canonical scorer-v9 success"
            )
    else:
        if end.get("winning_event_hash") is not None:
            raise ValueError(f"{log_path}: failed episode cannot name a winning event")
        if stop_reason not in {
            "budget_exhausted",
            "controller_error",
            "controller_stop",
        }:
            raise ValueError(f"{log_path}: unsupported terminal stop_reason")
        if trials and trials[-1]["observation"]["score"]["all_specs_met"] is True:
            raise ValueError(f"{log_path}: failed episode ends after a successful trial")
    if task is not None:
        if task.task_id != task_id:
            raise ValueError(f"{log_path}: log does not match the benchmark task")
        if len(controller_ids) != 1:
            raise ValueError(f"{log_path}: episode lacks one scored model controller")
        controller_id = next(iter(controller_ids))
        if episode_id != research_episode_id(task, controller_id, rollout_id):
            raise ValueError(f"{log_path}: episode identity is not reproducible")
        if task.initial_netlist is None or len(initial_events) != 1:
            raise ValueError(f"{log_path}: benchmark requires one frozen initial failure")
        initial = initial_events[0]
        if initial["netlist"] != task.initial_netlist:
            raise ValueError(f"{log_path}: frozen initial circuit does not match task bytes")
        verifier = reverify_fn or observe_completion

        def canonical_evidence(value: dict[str, Any]) -> dict[str, Any]:
            evidence = dict(value)
            failure = evidence.get("failure")
            if isinstance(failure, dict):
                evidence["failure"] = {
                    "stage": failure.get("stage"),
                    "code": failure.get("code"),
                }
            return evidence

        replay_items = [(task.initial_netlist, initial["observation"])]
        replay_items.extend(
            (str(trial["proposal"].get("netlist", "")), trial["observation"])
            for trial in trials
        )
        for netlist, recorded in replay_items:
            replayed = verifier(
                task.circuit_id,
                task.target_specs,
                verifier_completion(task, netlist),
                timeout_s=task.timeout_s,
            ).to_event()
            if canonical_evidence(replayed) != canonical_evidence(recorded):
                raise ValueError(f"{log_path}: scorer replay does not match log evidence")
        if verifier_calls > task.max_verifier_calls:
            raise ValueError(f"{log_path}: verifier-call budget was exceeded")
    return {
        "task_id": task_id,
        "success": success,
        "verifier_calls": verifier_calls,
        "episode_end_hash": end["event_hash"],
        "winning_event_hash": end.get("winning_event_hash"),
        "stop_reason": stop_reason,
        "completed_at_utc": end["timestamp_utc"],
        "winning_trial": winning_trial,
        "controller_ids": sorted(controller_ids),
        "episode_id": episode_id,
        "rollout_id": rollout_id,
        "trial_event_hashes": [event["event_hash"] for event in trials],
        "initial_observation_event_hash": (
            initial_events[0]["event_hash"] if initial_events else None
        ),
        "feedback_episode": bool(initial_events),
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty list")
    position = probability * (len(ordered) - 1)
    low_index = math.floor(position)
    high_index = math.ceil(position)
    if low_index == high_index:
        return ordered[low_index]
    fraction = position - low_index
    return ordered[low_index] * (1.0 - fraction) + ordered[high_index] * fraction


def summarize_comparison(
    model_logs: dict[str, str | Path],
    baseline_logs: dict[str, str | Path],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260729,
) -> dict[str, Any]:
    """Summarize paired success and call efficiency without raw rollouts."""

    if set(model_logs) != set(baseline_logs) or not model_logs:
        raise ValueError("model and baseline logs must cover the same non-empty task set")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    task_ids = sorted(model_logs)
    model_evidence = {
        task_id: validated_episode_outcome(model_logs[task_id])
        for task_id in task_ids
    }
    baseline_evidence = {
        task_id: validated_episode_outcome(baseline_logs[task_id])
        for task_id in task_ids
    }
    for task_id in task_ids:
        if model_evidence[task_id]["task_id"] != task_id:
            raise ValueError(f"model log task mismatch for {task_id}")
        if baseline_evidence[task_id]["task_id"] != task_id:
            raise ValueError(f"baseline log task mismatch for {task_id}")
    published_keys = (
        "task_id",
        "success",
        "verifier_calls",
        "episode_end_hash",
        "winning_event_hash",
    )
    model = {
        task_id: {
            key: model_evidence[task_id][key]
            for key in published_keys
        }
        for task_id in task_ids
    }
    baseline = {
        task_id: {
            key: baseline_evidence[task_id][key]
            for key in published_keys
        }
        for task_id in task_ids
    }

    success_deltas = [
        float(model[task_id]["success"]) - float(baseline[task_id]["success"])
        for task_id in task_ids
    ]
    call_savings = [
        float(baseline[task_id]["verifier_calls"] - model[task_id]["verifier_calls"])
        for task_id in task_ids
        if model[task_id]["success"] and baseline[task_id]["success"]
    ]
    rng = random.Random(seed)
    success_bootstrap = []
    calls_bootstrap = []
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(task_ids)) for _ in task_ids]
        success_bootstrap.append(
            sum(success_deltas[index] for index in indices) / len(indices)
        )
        if call_savings:
            call_indices = [
                rng.randrange(len(call_savings)) for _ in range(len(call_savings))
            ]
            calls_bootstrap.append(
                sum(call_savings[index] for index in call_indices) / len(call_indices)
            )

    return {
        "schema": "phychip-closed-loop-comparison-v1",
        "tasks": len(task_ids),
        "task_ids": task_ids,
        "model": {
            "successes": sum(model[task_id]["success"] for task_id in task_ids),
            "verifier_calls": [
                model[task_id]["verifier_calls"] for task_id in task_ids
            ],
            "outcomes": model,
        },
        "fixed_topology_tpe": {
            "successes": sum(baseline[task_id]["success"] for task_id in task_ids),
            "verifier_calls": [
                baseline[task_id]["verifier_calls"] for task_id in task_ids
            ],
            "outcomes": baseline,
        },
        "paired_success_delta": {
            "mean": sum(success_deltas) / len(success_deltas),
            "bootstrap_ci95": [
                _percentile(success_bootstrap, 0.025),
                _percentile(success_bootstrap, 0.975),
            ],
        },
        "paired_verifier_call_saving_on_joint_successes": {
            "n": len(call_savings),
            "mean": (
                sum(call_savings) / len(call_savings) if call_savings else None
            ),
            "bootstrap_ci95": (
                [
                    _percentile(calls_bootstrap, 0.025),
                    _percentile(calls_bootstrap, 0.975),
                ]
                if calls_bootstrap
                else None
            ),
        },
    }


def summarize_feedback_comparison(
    full_feedback_logs: dict[str, str | Path],
    feedback_blind_logs: dict[str, str | Path],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260801,
    task_families: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Measure the causal value of diagnostic feedback under paired tasks."""

    if set(full_feedback_logs) != set(feedback_blind_logs) or not full_feedback_logs:
        raise ValueError("feedback arms must cover the same non-empty task set")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    task_ids = sorted(full_feedback_logs)
    full = {
        task_id: validated_episode_outcome(full_feedback_logs[task_id])
        for task_id in task_ids
    }
    blind = {
        task_id: validated_episode_outcome(feedback_blind_logs[task_id])
        for task_id in task_ids
    }

    def controller_model(controller_id: str, prefix: str) -> str:
        if not controller_id.startswith(prefix):
            raise ValueError(f"expected {prefix.rstrip(':')} controller provenance")
        value = controller_id.removeprefix(prefix)
        if ":" not in value:
            raise ValueError("controller provenance lacks a protocol fingerprint")
        return value.rsplit(":", 1)[0]

    for task_id in task_ids:
        for label, outcome in (("full", full[task_id]), ("blind", blind[task_id])):
            if outcome["task_id"] != task_id:
                raise ValueError(f"{label} feedback log task mismatch for {task_id}")
            if not outcome["feedback_episode"]:
                raise ValueError(f"{label} arm lacks frozen feedback evidence")
            if len(outcome["controller_ids"]) != 1:
                raise ValueError(f"{label} arm lacks one scored model controller")
        full_controller = full[task_id]["controller_ids"][0]
        blind_controller = blind[task_id]["controller_ids"][0]
        full_model = controller_model(full_controller, "codex-cli:")
        blind_model = controller_model(
            blind_controller,
            "codex-cli-feedback-blind:",
        )
        if full_model != blind_model:
            raise ValueError(f"feedback arms use different models for {task_id}")
        if full[task_id]["rollout_id"] != blind[task_id]["rollout_id"]:
            raise ValueError(f"feedback arms use different rollout ids for {task_id}")
        full_initial = EpisodeLog(full_feedback_logs[task_id]).read()[0]
        blind_initial = EpisodeLog(feedback_blind_logs[task_id]).read()[0]
        initial_fields = ("source", "topology", "netlist", "observation")
        if any(full_initial.get(key) != blind_initial.get(key) for key in initial_fields):
            raise ValueError(f"feedback arms do not share initial evidence for {task_id}")
    deltas = [
        float(full[task_id]["success"]) - float(blind[task_id]["success"])
        for task_id in task_ids
    ]
    families = task_families or {task_id: task_id for task_id in task_ids}
    if set(families) != set(task_ids):
        raise ValueError("task_families must cover the exact paired task set")
    if any(not isinstance(family, str) or not family for family in families.values()):
        raise ValueError("task_families values must be non-empty strings")
    family_deltas: dict[str, list[float]] = {}
    for task_id, delta in zip(task_ids, deltas, strict=True):
        family_deltas.setdefault(families[task_id], []).append(delta)
    family_means = {
        family: sum(values) / len(values)
        for family, values in sorted(family_deltas.items())
    }
    full_only = sum(delta == 1.0 for delta in deltas)
    blind_only = sum(delta == -1.0 for delta in deltas)
    discordant = full_only + blind_only
    if discordant:
        lower_tail = sum(
            math.comb(discordant, count)
            for count in range(min(full_only, blind_only) + 1)
        ) / (2**discordant)
        exact_p = min(1.0, 2.0 * lower_tail)
    else:
        exact_p = 1.0
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(task_ids)) for _ in task_ids]
        bootstrap.append(sum(deltas[index] for index in indices) / len(indices))
    family_names = sorted(family_means)
    family_bootstrap = []
    for _ in range(bootstrap_samples):
        sampled = [rng.choice(family_names) for _ in family_names]
        family_bootstrap.append(
            sum(family_means[family] for family in sampled) / len(sampled)
        )
    positive_families = sum(value > 0.0 for value in family_means.values())
    negative_families = sum(value < 0.0 for value in family_means.values())
    tied_families = len(family_means) - positive_families - negative_families
    non_tied_families = positive_families + negative_families
    if non_tied_families:
        family_tail = sum(
            math.comb(non_tied_families, count)
            for count in range(min(positive_families, negative_families) + 1)
        ) / (2**non_tied_families)
        family_exact_p = min(1.0, 2.0 * family_tail)
    else:
        family_exact_p = 1.0

    def arm_summary(values: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return {
            "successes": sum(values[task_id]["success"] for task_id in task_ids),
            "verifier_calls": [
                values[task_id]["verifier_calls"] for task_id in task_ids
            ],
            "episode_end_hashes": {
                task_id: values[task_id]["episode_end_hash"] for task_id in task_ids
            },
            "winning_event_hashes": {
                task_id: values[task_id]["winning_event_hash"]
                for task_id in task_ids
            },
            "stop_reasons": {
                task_id: values[task_id]["stop_reason"] for task_id in task_ids
            },
        }

    error_pairs = sorted(
        task_id
        for task_id in task_ids
        if "controller_error"
        in {full[task_id]["stop_reason"], blind[task_id]["stop_reason"]}
    )
    complete_task_ids = [task_id for task_id in task_ids if task_id not in error_pairs]
    complete_deltas = [
        float(full[task_id]["success"]) - float(blind[task_id]["success"])
        for task_id in complete_task_ids
    ]
    complete_full_only = sum(delta == 1.0 for delta in complete_deltas)
    complete_blind_only = sum(delta == -1.0 for delta in complete_deltas)
    complete_discordant = complete_full_only + complete_blind_only
    if complete_discordant:
        complete_tail = sum(
            math.comb(complete_discordant, count)
            for count in range(min(complete_full_only, complete_blind_only) + 1)
        ) / (2**complete_discordant)
        complete_exact_p = min(1.0, 2.0 * complete_tail)
    else:
        complete_exact_p = 1.0

    return {
        "schema": "phychip-feedback-use-comparison-v1",
        "tasks": len(task_ids),
        "task_ids": task_ids,
        "full_feedback": arm_summary(full),
        "feedback_blind": arm_summary(blind),
        "completed_at_utc": max(
            outcome["completed_at_utc"]
            for values in (full, blind)
            for outcome in values.values()
        ),
        "execution_status": (
            "completed_with_controller_errors" if error_pairs else "completed"
        ),
        "controller_error_sensitivity": {
            "excluded_pairs": error_pairs,
            "paired_tasks": len(complete_task_ids),
            "full_feedback_successes": sum(
                full[task_id]["success"] for task_id in complete_task_ids
            ),
            "feedback_blind_successes": sum(
                blind[task_id]["success"] for task_id in complete_task_ids
            ),
            "paired_exact_mcnemar": {
                "full_only_successes": complete_full_only,
                "blind_only_successes": complete_blind_only,
                "discordant_pairs": complete_discordant,
                "two_sided_p": complete_exact_p,
            },
        },
        "paired_success_delta": {
            "mean": sum(deltas) / len(deltas),
            "bootstrap_ci95": [
                _percentile(bootstrap, 0.025),
                _percentile(bootstrap, 0.975),
            ],
        },
        "paired_exact_mcnemar": {
            "full_only_successes": full_only,
            "blind_only_successes": blind_only,
            "discordant_pairs": discordant,
            "two_sided_p": exact_p,
        },
        "family_clustered_success_delta": {
            "families": len(family_means),
            "family_means": family_means,
            "mean": sum(family_means.values()) / len(family_means),
            "bootstrap_ci95": [
                _percentile(family_bootstrap, 0.025),
                _percentile(family_bootstrap, 0.975),
            ],
        },
        "family_exact_sign_test": {
            "positive_families": positive_families,
            "negative_families": negative_families,
            "tied_families": tied_families,
            "two_sided_p": family_exact_p,
        },
    }


def episode_log_path(directory: str | Path, task_id: str) -> Path:
    """Resolve one task log without allowing task identifiers to escape a directory."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", task_id):
        raise ValueError(f"unsafe task_id for an episode filename: {task_id!r}")
    return Path(directory) / f"{task_id}.jsonl"


def paired_feedback_schedule(
    tasks: tuple[ResearchTask, ...],
    *,
    seed: int,
) -> tuple[tuple[ResearchTask, str], ...]:
    """Freeze task order and balance which paired arm executes first."""

    if not tasks:
        raise ValueError("paired feedback schedule requires at least one task")
    rng = random.Random(seed)
    shuffled = list(tasks)
    rng.shuffle(shuffled)
    first_arms = [
        "full_feedback" if index % 2 == 0 else "feedback_blind"
        for index in range(len(shuffled))
    ]
    rng.shuffle(first_arms)
    return tuple(zip(shuffled, first_arms, strict=True))


def run_paired_feedback_benchmark(
    benchmark_path: str | Path,
    output_directory: str | Path,
    *,
    model: str = "gpt-5.6-sol",
    codex_timeout_s: float = 300.0,
    schedule_seed: int = 20260801,
    max_workers: int = 1,
    rollout_id: str = "0",
    full_controller: Controller | None = None,
    blind_controller: Controller | None = None,
    observe_fn=observe_completion,
    runtime_fingerprint_value: dict[str, Any] | None = None,
    codex_cli_version: str | None = None,
) -> dict[str, Any]:
    """Run paired arms adjacently under a frozen, balanced arm-order schedule."""

    if not isinstance(max_workers, int) or isinstance(max_workers, bool):
        raise ValueError("max_workers must be an integer")
    if not 1 <= max_workers <= 32:
        raise ValueError("max_workers must be between 1 and 32")
    metadata, tasks = load_benchmark(benchmark_path)
    output = Path(output_directory)
    full_directory = output / "full"
    blind_directory = output / "blind"
    full = full_controller or CodexCLIController(
        model=model,
        timeout_s=codex_timeout_s,
    )
    blind = blind_controller or FeedbackBlindCodexCLIController(
        model=model,
        timeout_s=codex_timeout_s,
    )
    schedule = paired_feedback_schedule(tasks, seed=schedule_seed)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", rollout_id):
        raise ValueError(f"unsafe rollout_id: {rollout_id!r}")
    if runtime_fingerprint_value is None:
        from .provenance import runtime_fingerprint

        runtime_fingerprint_value = runtime_fingerprint()
    if codex_cli_version is None:
        executable = getattr(full, "executable", "codex")
        completed = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        codex_cli_version = " ".join(
            (completed.stdout or completed.stderr or "").split()
        )
    manifest_payload = {
        "schema": "phychip-paired-feedback-run-manifest-v1",
        "benchmark_id": metadata["benchmark_id"],
        "task_set_sha256": metadata["task_set_sha256"],
        "protocol_sha256": metadata["protocol_sha256"],
        "rollout_id": rollout_id,
        "model": model,
        "full_controller": full.controller_id,
        "blind_controller": blind.controller_id,
        "schedule_seed": schedule_seed,
        "schedule": [
            {"task_id": task.task_id, "first_arm": first_arm}
            for task, first_arm in schedule
        ],
        "codex_cli_version": codex_cli_version,
        "controller_timeout_s": codex_timeout_s,
        "max_workers": max_workers,
        "runtime_fingerprint": runtime_fingerprint_value,
        "runner_source_sha256": hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest(),
    }
    manifest = {
        **manifest_payload,
        "manifest_sha256": _content_hash(manifest_payload),
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        recorded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if recorded_manifest != manifest:
            raise ValueError("paired run manifest differs from frozen execution")
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".run_manifest.",
            suffix=".tmp",
            dir=output,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, manifest_path)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise

    def run_pair(item: tuple[ResearchTask, str]) -> tuple[str, dict[str, Any]]:
        task, first_arm = item
        arms = {
            "full_feedback": (
                full,
                EpisodeLog(episode_log_path(full_directory, task.task_id)),
            ),
            "feedback_blind": (
                blind,
                EpisodeLog(episode_log_path(blind_directory, task.task_id)),
            ),
        }
        second_arm = (
            "feedback_blind" if first_arm == "full_feedback" else "full_feedback"
        )
        outcomes = {}
        runner = ResearchRunner(observe_fn=observe_fn)
        for arm in (first_arm, second_arm):
            controller, log = arms[arm]
            result = runner.run(
                task,
                controller,
                log,
                rollout_id=rollout_id,
            )
            outcomes[arm] = {
                "stop_reason": result["stop_reason"],
                "verifier_calls": result["verifier_calls"],
                "codex_calls": result["codex_calls"],
                "controller_id": result["controller_id"],
                "episode_end_hash": result["event_hash"],
                "winning_event_hash": result.get("winning_event_hash"),
            }
        return task.task_id, {"first_arm": first_arm, **outcomes}

    if max_workers == 1:
        completed = map(run_pair, schedule)
        results = dict(completed)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = dict(executor.map(run_pair, schedule))
    return {
        "schema": "phychip-paired-feedback-batch-v1",
        "benchmark_id": metadata["benchmark_id"],
        "task_set_sha256": metadata["task_set_sha256"],
        "protocol_sha256": metadata["protocol_sha256"],
        "schedule_seed": schedule_seed,
        "rollout_id": rollout_id,
        "manifest_sha256": manifest["manifest_sha256"],
        "schedule": [
            {"task_id": task.task_id, "first_arm": first_arm}
            for task, first_arm in schedule
        ],
        "tasks": len(tasks),
        "full_controller": full.controller_id,
        "blind_controller": blind.controller_id,
        "results": results,
    }


def run_benchmark(
    benchmark_path: str | Path,
    output_directory: str | Path,
    *,
    controller: str,
    model: str = "gpt-5.6-sol",
    codex_timeout_s: float = 300.0,
    max_workers: int = 1,
    observe_fn=observe_completion,
) -> dict[str, Any]:
    """Run or resume every task with one frozen controller and log per task."""

    allowed_controllers = {
        "codex",
        "codex-blind",
        "fixed",
        "tpe",
        "random",
        "lhs",
    }
    if controller not in allowed_controllers:
        raise ValueError(
            "controller must be 'codex', 'codex-blind', 'fixed', 'tpe', "
            "'random', or 'lhs'"
        )
    if not isinstance(max_workers, int) or isinstance(max_workers, bool):
        raise ValueError("max_workers must be an integer")
    if not 1 <= max_workers <= 32:
        raise ValueError("max_workers must be between 1 and 32")
    metadata, tasks = load_benchmark(benchmark_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    codex = None
    if controller == "codex":
        codex = CodexCLIController(model=model, timeout_s=codex_timeout_s)
    elif controller == "codex-blind":
        codex = FeedbackBlindCodexCLIController(
            model=model,
            timeout_s=codex_timeout_s,
        )
    runner = ResearchRunner(observe_fn=observe_fn)
    def run_task(task: ResearchTask) -> tuple[str, dict[str, Any]]:
        log = EpisodeLog(episode_log_path(output, task.task_id))
        if codex is None:
            strategy = "tpe" if controller == "fixed" else controller
            result = run_fixed_topology(
                task,
                log,
                strategy=strategy,
                observe_fn=observe_fn,
            )
        else:
            result = runner.run(task, codex, log)
        return (
            task.task_id,
            {
                "stop_reason": result["stop_reason"],
                "verifier_calls": result["verifier_calls"],
                "codex_calls": result["codex_calls"],
                "controller_id": result["controller_id"],
                "episode_end_hash": result["event_hash"],
                "winning_event_hash": result.get("winning_event_hash"),
            },
        )

    if max_workers == 1:
        completed = map(run_task, tasks)
        results = dict(completed)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = dict(executor.map(run_task, tasks))
    successes = sum(
        result["stop_reason"] == "all_specs_met" for result in results.values()
    )
    return {
        "schema": "phychip-research-batch-v1",
        "benchmark_id": metadata["benchmark_id"],
        "task_set_sha256": metadata["task_set_sha256"],
        "protocol_sha256": metadata["protocol_sha256"],
        "controller": (
            codex.controller_id
            if codex is not None
            else (
                "fixed-topology-tpe"
                if controller in {"fixed", "tpe"}
                else f"fixed-{controller}"
            )
        ),
        "tasks": len(tasks),
        "successes": successes,
        "results": results,
    }


def compare_benchmark_runs(
    benchmark_path: str | Path,
    model_directory: str | Path,
    baseline_directory: str | Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260729,
) -> dict[str, Any]:
    """Compare two complete benchmark directories under the frozen task identity."""

    metadata, tasks = load_benchmark(benchmark_path)
    model_logs = {
        task.task_id: episode_log_path(model_directory, task.task_id)
        for task in tasks
    }
    baseline_logs = {
        task.task_id: episode_log_path(baseline_directory, task.task_id)
        for task in tasks
    }
    missing = [
        str(path)
        for path in [*model_logs.values(), *baseline_logs.values()]
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"missing benchmark episode logs: {missing}")
    result = summarize_comparison(
        model_logs,
        baseline_logs,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    return {
        **result,
        "benchmark_id": metadata["benchmark_id"],
        "task_set_sha256": metadata["task_set_sha256"],
        "protocol_sha256": metadata["protocol_sha256"],
    }


def compare_feedback_runs(
    benchmark_path: str | Path,
    full_feedback_directory: str | Path,
    feedback_blind_directory: str | Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260801,
    manual_audit_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare paired full-feedback and feedback-blind Codex runs."""

    metadata, tasks = load_benchmark(benchmark_path)
    analysis_protocol = metadata.get("analysis_protocol", {})
    frozen_bootstrap_samples = analysis_protocol.get("bootstrap_samples")
    frozen_bootstrap_seed = analysis_protocol.get("bootstrap_seed")
    if (
        frozen_bootstrap_samples is not None
        and bootstrap_samples != frozen_bootstrap_samples
    ):
        raise ValueError("bootstrap sample count differs from frozen protocol")
    if frozen_bootstrap_seed is not None and seed != frozen_bootstrap_seed:
        raise ValueError("bootstrap seed differs from frozen protocol")
    task_by_id = {task.task_id: task for task in tasks}
    full_logs = {
        task.task_id: episode_log_path(full_feedback_directory, task.task_id)
        for task in tasks
    }
    blind_logs = {
        task.task_id: episode_log_path(feedback_blind_directory, task.task_id)
        for task in tasks
    }
    missing = [
        str(path)
        for path in [*full_logs.values(), *blind_logs.values()]
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"missing feedback benchmark episode logs: {missing}")
    for task_id, task in task_by_id.items():
        validated_episode_outcome(full_logs[task_id], task=task)
        validated_episode_outcome(blind_logs[task_id], task=task)
    result = summarize_feedback_comparison(
        full_logs,
        blind_logs,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        task_families={task.task_id: task.circuit_id for task in tasks},
    )
    split_role = metadata.get("split_policy", {}).get("role")
    promotion_gates = metadata.get("promotion_gates", {})
    minimum_tasks = promotion_gates.get("minimum_paired_tasks")
    exact_tasks = promotion_gates.get("exact_paired_tasks")
    minimum_families = promotion_gates.get("minimum_circuit_families")
    admissible_role = promotion_gates.get("admissible_split_role")
    ci_lower = result["paired_success_delta"]["bootstrap_ci95"][0]
    exact_p = result["paired_exact_mcnemar"]["two_sided_p"]
    gate_reasons = []
    if exact_tasks is not None and result["tasks"] != exact_tasks:
        gate_reasons.append(
            f"{result['tasks']}/{exact_tasks} exact preregistered pairs are present"
        )
    if not isinstance(minimum_tasks, int) or isinstance(minimum_tasks, bool):
        gate_reasons.append("benchmark does not preregister a minimum paired-task count")
    elif result["tasks"] < minimum_tasks:
        gate_reasons.append(
            f"{result['tasks']}/{minimum_tasks} preregistered paired tasks are present"
        )
    if minimum_families is not None:
        observed_families = result["family_clustered_success_delta"]["families"]
        if (
            not isinstance(minimum_families, int)
            or isinstance(minimum_families, bool)
            or minimum_families <= 0
        ):
            gate_reasons.append(
                "benchmark has an invalid minimum circuit-family count"
            )
        elif observed_families < minimum_families:
            gate_reasons.append(
                f"{observed_families}/{minimum_families} circuit families are present"
            )
    if not isinstance(admissible_role, str) or split_role != admissible_role:
        gate_reasons.append(
            f"split role {split_role!r} is not the preregistered admissible role"
        )
    required_ci_lower = promotion_gates.get("paired_success_ci95_lower_gt", 0.0)
    if not isinstance(required_ci_lower, (int, float)):
        gate_reasons.append("benchmark has an invalid paired success CI gate")
    elif ci_lower <= float(required_ci_lower):
        gate_reasons.append("paired success CI95 lower bound misses its gate")
    required_delta = promotion_gates.get("paired_success_delta_gte")
    if required_delta is not None:
        observed_delta = result["paired_success_delta"]["mean"]
        if not isinstance(required_delta, (int, float)):
            gate_reasons.append("benchmark has an invalid paired effect-size gate")
        elif observed_delta < float(required_delta):
            gate_reasons.append("paired success delta misses its effect-size gate")
    required_exact_p = promotion_gates.get("paired_exact_mcnemar_p_lte", 0.05)
    if not isinstance(required_exact_p, (int, float)):
        gate_reasons.append("benchmark has an invalid paired exact-test gate")
    elif exact_p > float(required_exact_p):
        gate_reasons.append("paired exact McNemar p-value misses its gate")
    required_family_ci = promotion_gates.get(
        "family_clustered_success_ci95_lower_gt"
    )
    if required_family_ci is not None:
        family_ci_lower = result["family_clustered_success_delta"][
            "bootstrap_ci95"
        ][0]
        if not isinstance(required_family_ci, (int, float)):
            gate_reasons.append("benchmark has an invalid family-clustered CI gate")
        elif family_ci_lower <= float(required_family_ci):
            gate_reasons.append(
                "family-clustered success CI95 lower bound misses its gate"
            )
    required_family_p = promotion_gates.get("family_exact_sign_p_lte")
    if required_family_p is not None:
        family_p = result["family_exact_sign_test"]["two_sided_p"]
        if not isinstance(required_family_p, (int, float)):
            gate_reasons.append("benchmark has an invalid family sign-test gate")
        elif family_p > float(required_family_p):
            gate_reasons.append("family exact sign-test p-value misses its gate")
    if manual_audit_path is None:
        gate_reasons.append("no hash-bound manual success audit was supplied")
    else:
        audit = json.loads(Path(manual_audit_path).read_text(encoding="utf-8"))
        expected_hashes = {
            "full_feedback": result["full_feedback"]["episode_end_hashes"],
            "feedback_blind": result["feedback_blind"]["episode_end_hashes"],
        }
        if (
            audit.get("schema") != "phychip-feedback-manual-audit-v1"
            or audit.get("task_set_sha256") != metadata["task_set_sha256"]
            or audit.get("protocol_sha256") != metadata["protocol_sha256"]
            or audit.get("episode_end_hashes") != expected_hashes
        ):
            gate_reasons.append("manual audit does not bind the compared evidence")
        else:
            require_decisions = promotion_gates.get(
                "per_success_audit_required",
                False,
            )
            decisions = audit.get("decisions")
            decision_set_valid = True
            if require_decisions:
                expected_decisions = {
                    (
                        arm,
                        task_id,
                        result[arm]["episode_end_hashes"][task_id],
                        winning_hash,
                    )
                    for arm in ("full_feedback", "feedback_blind")
                    for task_id, winning_hash in result[arm][
                        "winning_event_hashes"
                    ].items()
                    if winning_hash is not None
                }
                if not isinstance(decisions, list):
                    decision_set_valid = False
                    decisions = []
                observed_decisions = {
                    (
                        decision.get("arm"),
                        decision.get("task_id"),
                        decision.get("episode_end_hash"),
                        decision.get("winning_event_hash"),
                    )
                    for decision in decisions
                    if isinstance(decision, dict)
                }
                decision_set_valid = (
                    decision_set_valid
                    and len(observed_decisions) == len(decisions)
                    and observed_decisions == expected_decisions
                )
                if not decision_set_valid:
                    gate_reasons.append(
                        "manual audit lacks exact per-success hash decisions"
                    )
                reviewed = len(decisions)
                plausible = sum(
                    decision.get("topology_plausible") is True
                    for decision in decisions
                    if isinstance(decision, dict)
                )
                practical = sum(
                    decision.get("component_practical") is True
                    for decision in decisions
                    if isinstance(decision, dict)
                )
                attacks = sum(
                    decision.get("known_reward_attack") is True
                    for decision in decisions
                    if isinstance(decision, dict)
                )
            else:
                attacks = audit.get("known_reward_attacks_accepted")
                plausible = audit.get("topology_plausible_successes")
                reviewed = audit.get("successful_circuits_reviewed")
                practical = audit.get("component_practical_successes")
            expected_reviewed = (
                result["full_feedback"]["successes"]
                + result["feedback_blind"]["successes"]
            )
            required_attack_count = promotion_gates.get(
                "known_reward_attacks_accepted"
            )
            required_plausible_rate = promotion_gates.get(
                "topology_plausible_success_rate"
            )
            required_practical_rate = promotion_gates.get(
                "component_practical_success_rate"
            )
            if attacks != required_attack_count:
                gate_reasons.append("manual audit fails the reward-attack gate")
            if reviewed != expected_reviewed:
                gate_reasons.append(
                    "manual audit does not cover every successful circuit"
                )
            if (
                not isinstance(plausible, int)
                or isinstance(plausible, bool)
                or not isinstance(reviewed, int)
                or isinstance(reviewed, bool)
                or reviewed <= 0
                or not isinstance(required_plausible_rate, (int, float))
                or plausible / reviewed < float(required_plausible_rate)
            ):
                gate_reasons.append("manual audit fails the topology-plausibility gate")
            if required_practical_rate is not None and (
                not isinstance(practical, int)
                or isinstance(practical, bool)
                or not isinstance(reviewed, int)
                or isinstance(reviewed, bool)
                or reviewed <= 0
                or not isinstance(required_practical_rate, (int, float))
                or practical / reviewed < float(required_practical_rate)
            ):
                gate_reasons.append("manual audit fails the component-practicality gate")
    return {
        **result,
        "benchmark_id": metadata["benchmark_id"],
        "task_set_sha256": metadata["task_set_sha256"],
        "protocol_sha256": metadata["protocol_sha256"],
        "split_role": split_role,
        "feedback_advantage_gate": {
            "passed": not gate_reasons,
            "reasons": gate_reasons,
        },
    }
