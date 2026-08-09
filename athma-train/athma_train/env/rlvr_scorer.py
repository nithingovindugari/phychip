"""Canonical PhyChip RLVR scorer.

Training, difficulty labeling, offline evaluation, and auditable rescoring must
all call :func:`score_completion`.  Keeping this contract in the environment
package prevents an evaluation script from silently importing reward semantics
from a trainer entry point.

The scorer deliberately returns a structured :class:`ScoreResult`; callers may
use ``reward`` for optimization, but scientific evaluation must retain the
decomposition and define success with ``all_specs_met``.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from athma_train.spice_gate import grade_netlist, normalize_netlist


SCORER_SCHEMA = "phychip-rlvr-score-v1"
SCORER_SEMANTICS = "athma_train.env.rlvr_scorer.score_completion@v9"
HARNESSES_DIR = Path(__file__).resolve().parent.parent / "harnesses"
_HARNESS_CACHE: dict[str, object] = {}


@dataclass(frozen=True)
class ScoreResult:
    """Auditable result for exactly one completion/specification pair."""

    reward: float
    simulated: bool
    anticheat_blocked: bool
    requested_specs: int
    measured_specs: int
    met_specs: int
    spec_score: float
    all_specs_met: bool
    has_spice_block: bool
    has_end: bool
    has_device: bool
    has_analysis: bool
    schema: str = SCORER_SCHEMA

    def to_event(self) -> dict:
        """Return the JSON-safe reward decomposition used by logs/evaluations."""

        return asdict(self)


def load_harness(circuit_id: str):
    """Load one active measurement harness by circuit family."""

    if circuit_id in _HARNESS_CACHE:
        return _HARNESS_CACHE[circuit_id]
    path = HARNESSES_DIR / f"{circuit_id}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"harness_{circuit_id}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"  warn: failed to load harness {circuit_id}: {exc}", file=sys.stderr)
        return None
    _HARNESS_CACHE[circuit_id] = module
    return module


def extract_netlist(text: str) -> str:
    """Extract the fenced SPICE deck, falling back to normalized raw text."""

    match = re.search(r"```spice\s*(.*?)```", text or "", re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    return normalize_netlist(text or "")


def _active_top_level_lines(netlist: str) -> list[list[str]]:
    """Return executable top-level device lines.

    Control-language commands such as ``meas`` and ``quit`` are not SPICE
    devices.  Excluding complete ``.control`` and ``.subckt`` bodies before
    topology checks prevents their first letters from impersonating M/Q cards.
    """

    active_lines: list[str] = []
    for raw_line in (netlist or "").splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if line and not line.startswith("*"):
            active_lines.append(line)

    subckt_depth = 0
    control_depth = 0
    top_level: list[list[str]] = []
    for line in active_lines:
        parts = line.lower().split()
        if not parts:
            continue
        if parts[0] == ".control":
            control_depth += 1
            continue
        if parts[0] == ".endc":
            control_depth = max(0, control_depth - 1)
            continue
        if control_depth:
            continue
        if parts[0] == ".subckt":
            subckt_depth += 1
            continue
        if parts[0] == ".ends":
            subckt_depth = max(0, subckt_depth - 1)
            continue
        if not subckt_depth and not parts[0].startswith("."):
            top_level.append(parts)
    return top_level


def anticheat_ok(netlist: str, circuit_id: str) -> bool:
    """Reject known non-physical shortcuts and family-topology violations.

    This is intentionally a conservative structural admission gate, not a
    learned quality score.  Every rule is covered by adversarial tests and must
    preserve all verified references before it can become reward-bearing.
    """

    active_lines = []
    for raw_line in (netlist or "").splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if line and not line.startswith("*"):
            active_lines.append(line)
    active_netlist = "\n".join(active_lines)
    low = active_netlist.lower()
    if "laplace" in low or "tf(" in low:
        return False

    top_level_lines = _active_top_level_lines(active_netlist)

    def count_devices(prefixes: str) -> int:
        allowed = set(prefixes.lower())
        return sum(bool(parts) and parts[0][0] in allowed for parts in top_level_lines)

    need_transistor = {
        "current-mirror",
        "cs-stage",
        "diff-pair",
        "diff-amp",
        "bandgap",
        "level-shift",
        "current-sense",
    }
    need_caps = {"sallen-key": 2, "integrator": 1, "differentiator": 1, "tia": 1}
    if circuit_id in need_transistor and count_devices("mq") < 1:
        return False
    if circuit_id in need_caps and count_devices("c") < need_caps[circuit_id]:
        return False

    # The powered-study prompt contract requires each AC stimulus source name
    # and driven node to match a declared input port. An additional AC source
    # on an internal node can otherwise drive the measurement while the named
    # port is disconnected or irrelevant. Keep DC-only supplies/biases legal.
    declared_ac_ports = {
        "buffer": {"vin"},
        "cs-stage": {"vin"},
        "diff-amp": {"vp", "vn"},
        "diff-pair": {"vp", "vn"},
        "differentiator": {"vin"},
        "gain-trim": {"vin", "vtrim"},
        "integrator": {"vin"},
        "inv-amp": {"vin"},
        "noninv-amp": {"vin"},
        "sallen-key": {"vin"},
        "schmitt": {"vin"},
        "summing-amp": {"v1", "v2", "v3"},
    }.get(circuit_id)
    if declared_ac_ports is not None:
        for parts in top_level_lines:
            if len(parts) < 4 or parts[0][0] not in {"v", "i"}:
                continue
            try:
                ac_index = parts.index("ac")
            except ValueError:
                continue
            magnitude = parts[ac_index + 1] if ac_index + 1 < len(parts) else ""
            try:
                ac_is_nonzero = float(magnitude) != 0.0
            except ValueError:
                # Parameterized or malformed AC drives are not safe undeclared
                # stimuli; fail closed instead of guessing their runtime value.
                ac_is_nonzero = True
            if not ac_is_nonzero:
                continue
            source_name = parts[0]
            source_nodes = set(parts[1:3])
            if source_name not in declared_ac_ports or source_name not in source_nodes:
                return False

    def passive_edges(prefix: str) -> set[frozenset[str]]:
        return {
            frozenset((parts[1], parts[2]))
            for parts in top_level_lines
            if len(parts) >= 4 and parts[0].startswith(prefix)
        }

    resistor_edges = passive_edges("r")
    capacitor_edges = passive_edges("c")
    reference_nodes = {"0", "gnd", "vref", "vcm", "vbias", "vss"}
    output_nodes = {"vout", "out"}
    independent_voltage_nodes = {
        node
        for parts in top_level_lines
        if len(parts) >= 3 and parts[0][0] == "v"
        for node in parts[1:3]
        if node not in {"0", "gnd"}
    }
    family_inputs = {
        "buffer": {"vin"},
        "cs-stage": {"vin"},
        "diff-amp": {"vp", "vn"},
        "diff-pair": {"vp", "vn"},
        "differentiator": {"vin"},
        "gain-trim": {"vin", "vtrim"},
        "integrator": {"vin"},
        "inv-amp": {"vin"},
        "noninv-amp": {"vin"},
        "sallen-key": {"vin"},
        "schmitt": {"vin"},
        "summing-amp": {"v1", "v2", "v3"},
    }.get(circuit_id, set())
    fixed_nodes = reference_nodes | (independent_voltage_nodes - family_inputs)

    # An independent source across the scored output can manufacture any DC or
    # AC target without implementing the requested circuit.  Source *names*
    # such as ``Vout iout 0`` remain legal; only source terminals matter.
    if any(
        len(parts) >= 3
        and parts[0].startswith("v")
        and (parts[1] in output_nodes or parts[2] in output_nodes)
        for parts in top_level_lines
    ):
        return False

    def joined(edges: set[frozenset[str]], left: str, right: str) -> bool:
        return frozenset((left, right)) in edges

    def joined_any(edges: set[frozenset[str]], left: str, candidates: set[str]) -> bool:
        return any(joined(edges, left, candidate) for candidate in candidates)

    def resistor_path_exists(
        start: str,
        targets: set[str],
        *,
        blocked: set[str],
    ) -> bool:
        """Return whether an all-resistor path reaches a target safely."""

        graph: dict[str, set[str]] = {}
        for edge in resistor_edges:
            if len(edge) != 2:
                continue
            left, right = tuple(edge)
            graph.setdefault(left, set()).add(right)
            graph.setdefault(right, set()).add(left)
        seen = {start}
        pending = [start]
        while pending:
            current = pending.pop()
            if current in targets:
                return True
            if current != start and current in blocked:
                continue
            for neighbor in graph.get(current, ()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        return False

    def path_with_capacitor_exists(
        start: str,
        targets: set[str],
        *,
        blocked: set[str],
    ) -> bool:
        """Whether an R/C path reaches a target after traversing a capacitor."""

        graph: dict[str, list[tuple[str, bool]]] = {}
        for edges, is_capacitor in ((resistor_edges, False), (capacitor_edges, True)):
            for edge in edges:
                if len(edge) != 2:
                    continue
                left, right = tuple(edge)
                graph.setdefault(left, []).append((right, is_capacitor))
                graph.setdefault(right, []).append((left, is_capacitor))
        seen = {(start, False)}
        pending = [(start, False)]
        while pending:
            current, crossed_capacitor = pending.pop()
            if current in targets and crossed_capacitor:
                return True
            if current != start and current in blocked:
                continue
            for neighbor, is_capacitor in graph.get(current, ()):
                state = (neighbor, crossed_capacitor or is_capacitor)
                if state not in seen:
                    seen.add(state)
                    pending.append(state)
        return False

    resistor_nodes = set().union(*resistor_edges) if resistor_edges else set()
    driver_path_boundaries = (
        {"v1", "v2", "v3", "vin", "vdd"} | reference_nodes
    )

    def output_driver(parts: list[str]) -> bool:
        """Whether a two-terminal source safely reaches the scored output."""

        if len(parts) < 3:
            return False
        terminals = parts[1:3]
        return any(
            node not in driver_path_boundaries
            and (
                node in output_nodes
                or resistor_path_exists(
                    node,
                    output_nodes,
                    blocked=driver_path_boundaries,
                )
            )
            for node in terminals
        )

    def controlled_nodes(parts: list[str]) -> set[str]:
        """Return voltage-control nodes used by a top-level B/E/G source."""

        if not parts or parts[0][0] not in {"b", "e", "g"} or len(parts) < 4:
            return set()
        expression = " ".join(parts[3:])
        nodes = {
            node.lower()
            for body in re.findall(r"(?i)\bv\s*\(([^)]*)\)", expression)
            for node in re.split(r"\s*,\s*|\s+", body.strip())
            if node
        }
        if not nodes and parts[0][0] in {"e", "g"} and len(parts) >= 5:
            nodes = set(parts[3:5])
        return nodes

    controlled_sources = [
        parts for parts in top_level_lines if parts[0][0] in {"b", "e", "g"}
    ]

    def node_tracks_input(node: str, input_node: str, seen: set[str] | None = None) -> bool:
        """Trace resistor aliases and controlled-source producers to an input."""

        if node == input_node:
            return True
        boundaries = fixed_nodes | output_nodes | (family_inputs - {input_node})
        if node in boundaries:
            return False
        if resistor_path_exists(node, {input_node}, blocked=boundaries):
            return True
        seen = set() if seen is None else set(seen)
        if node in seen:
            return False
        seen.add(node)
        for source in controlled_sources:
            outputs = set(source[1:3]) - fixed_nodes
            if node not in outputs and not resistor_path_exists(
                node, outputs, blocked=boundaries
            ):
                continue
            if any(
                node_tracks_input(control, input_node, seen)
                for control in controlled_nodes(source)
            ):
                return True
        return False

    def signal_reaches_output(node: str, seen: set[str] | None = None) -> bool:
        """Trace a controlled-stage signal through later stages to Vout."""

        boundaries = fixed_nodes | family_inputs
        if node in output_nodes or resistor_path_exists(
            node, output_nodes, blocked=boundaries
        ):
            return True
        seen = set() if seen is None else set(seen)
        if node in seen:
            return False
        seen.add(node)
        for source in controlled_sources:
            controls = controlled_nodes(source)
            if node not in controls and not resistor_path_exists(
                node, controls, blocked=boundaries
            ):
                continue
            if any(signal_reaches_output(output, seen) for output in source[1:3]):
                return True
        return False

    def _source_has_nonzero_value(parts: list[str]) -> bool:
        values = parts[3:]
        if values and values[0] == "dc":
            values = values[1:]
        for token in values:
            match = re.fullmatch(
                r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)"
                r"(?:meg|[tgkmunpf])?",
                token,
                re.IGNORECASE,
            )
            if match:
                return float(match.group(1)) != 0.0
        return False

    def transistor_ports(parts: list[str]) -> tuple[str, str, str] | None:
        if not parts or len(parts) < 4:
            return None
        if parts[0][0] == "m":
            return parts[1], parts[2], parts[3]
        if parts[0][0] == "q":
            return parts[1], parts[2], parts[3]
        return None

    transistors = [parts for parts in top_level_lines if transistor_ports(parts)]

    if circuit_id == "cs-stage":
        valid_stage = False
        for parts in transistors:
            output_terminal, control_terminal, return_terminal = transistor_ports(parts) or ()
            if not node_tracks_input(control_terminal, "vin"):
                continue
            output_reaches = output_terminal in output_nodes or resistor_path_exists(
                output_terminal,
                output_nodes,
                blocked=fixed_nodes | family_inputs,
            )
            return_reaches = return_terminal in fixed_nodes or resistor_path_exists(
                return_terminal,
                fixed_nodes,
                blocked=output_nodes | family_inputs,
            )
            valid_stage |= output_reaches and return_reaches
        if not valid_stage:
            return False

    if circuit_id in {"diff-pair", "diff-amp"}:
        pair_found = False
        for left in transistors:
            left_output, left_control, left_tail = transistor_ports(left) or ()
            if not node_tracks_input(left_control, "vp") or node_tracks_input(
                left_control, "vn"
            ):
                continue
            for right in transistors:
                if right is left:
                    continue
                right_output, right_control, right_tail = transistor_ports(right) or ()
                if (
                    right_tail != left_tail
                    or not node_tracks_input(right_control, "vn")
                    or node_tracks_input(right_control, "vp")
                ):
                    continue
                output_reaches = any(
                    terminal in output_nodes
                    or resistor_path_exists(
                        terminal,
                        output_nodes,
                        blocked=fixed_nodes | family_inputs | {left_tail},
                    )
                    for terminal in (left_output, right_output)
                )
                nonzero_tail_source = any(
                    parts[0][0] == "i"
                    and left_tail in parts[1:3]
                    and _source_has_nonzero_value(parts)
                    for parts in top_level_lines
                    if len(parts) >= 4
                )
                resistive_tail = resistor_path_exists(
                    left_tail,
                    fixed_nodes,
                    blocked=output_nodes | family_inputs,
                )
                active_tail = any(
                    other is not left
                    and other is not right
                    and left_tail in (transistor_ports(other) or ())
                    for other in transistors
                )
                pair_found |= output_reaches and (
                    nonzero_tail_source or resistive_tail or active_tail
                )
        if not pair_found:
            return False

    def wrong_positive_e_polarity(candidate_nodes: set[str]) -> bool:
        """Catch positive-gain VCVSs that put a feedback sum on ``+``."""

        for parts in top_level_lines:
            if not parts[0].startswith("e") or len(parts) < 6 or not output_driver(parts):
                continue
            try:
                gain = float(parts[5])
            except ValueError:
                continue
            positive_control, negative_control = parts[3], parts[4]
            if gain > 0 and positive_control in candidate_nodes and negative_control not in candidate_nodes:
                return True
        return False

    if circuit_id == "inv-amp":
        inverting_nodes = {
            node
            for node in resistor_nodes - output_nodes - reference_nodes - {"vin"}
            if
            joined(resistor_edges, node, "vin")
            and joined_any(resistor_edges, node, output_nodes)
        }
        if not inverting_nodes or wrong_positive_e_polarity(inverting_nodes):
            return False
        for parts in top_level_lines:
            if not parts[0].startswith("e") or len(parts) < 6 or not output_driver(parts):
                continue
            try:
                gain = float(parts[5])
            except ValueError:
                continue
            positive_control, negative_control = parts[3], parts[4]
            positive_tracks_input = (
                positive_control not in fixed_nodes
                and node_tracks_input(positive_control, "vin")
            )
            if gain > 0 and negative_control in inverting_nodes and positive_tracks_input:
                return False

    if circuit_id == "buffer":
        input_path_boundaries = reference_nodes | output_nodes | {"vdd"}

        def tracks_buffer_input(node: str) -> bool:
            return node == "vin" or (
                node not in input_path_boundaries
                and resistor_path_exists(
                    node,
                    {"vin"},
                    blocked=input_path_boundaries,
                )
            )

        # A positive-gain VCVS whose negative control, but not positive
        # control, tracks Vin is an inverting macro.  Magnitude-only AC FOMs
        # previously let that wrong family receive full buffer reward.
        for parts in top_level_lines:
            if not parts[0].startswith("e") or len(parts) < 6 or not output_driver(parts):
                continue
            try:
                gain = float(parts[5])
            except ValueError:
                continue
            positive_control, negative_control = parts[3], parts[4]
            if (
                gain > 0
                and tracks_buffer_input(negative_control)
                and not tracks_buffer_input(positive_control)
            ):
                return False

    if circuit_id == "noninv-amp":
        feedback_nodes = {
            node
            for node in resistor_nodes - output_nodes - fixed_nodes - {"vin"}
            if
            joined_any(resistor_edges, node, output_nodes)
            and joined_any(resistor_edges, node, reference_nodes)
            and not joined(resistor_edges, node, "vin")
        }
        if not feedback_nodes or wrong_positive_e_polarity(feedback_nodes):
            return False
        driver_seen = False
        feedback_consumed = False
        input_consumed = False
        input_path_boundaries = fixed_nodes | output_nodes
        for parts in top_level_lines:
            kind = parts[0][0]
            # Multi-stage macro models often consume the feedback node in an
            # error amplifier whose output reaches Vout through later pole and
            # buffer stages. Requiring that first element to drive Vout
            # directly would reject valid cascades; require consumption by any
            # controlled stage in the executable signal chain instead.
            if kind in {"b", "e"}:
                driver_seen = True
                expression = " ".join(parts[3:])
                controlled_nodes = {
                    node.lower()
                    for body in re.findall(r"(?i)\bv\s*\(([^)]*)\)", expression)
                    for node in re.split(r"\s*,\s*|\s+", body.strip())
                    if node
                }
                if not controlled_nodes and kind == "e" and len(parts) >= 5:
                    controlled_nodes = set(parts[3:5])
                feedback_consumed |= bool(controlled_nodes & feedback_nodes)
                input_consumed |= any(
                    node not in input_path_boundaries
                    and resistor_path_exists(
                        node,
                        {"vin"},
                        blocked=input_path_boundaries,
                    )
                    for node in controlled_nodes
                )
            elif kind == "x" and len(parts) >= 3:
                instance_nodes = set(parts[1:-1])
                driver_seen = True
                feedback_consumed |= bool(instance_nodes & feedback_nodes)
                input_consumed |= any(
                    node not in input_path_boundaries
                    and resistor_path_exists(
                        node,
                        {"vin"},
                        blocked=input_path_boundaries,
                    )
                    for node in instance_nodes
                )
        if driver_seen and (not feedback_consumed or not input_consumed):
            return False

    if circuit_id == "summing-amp":
        feedback_boundaries = (
            {"v1", "v2", "v3", "vin", "vdd"} | reference_nodes
        )
        summing_nodes = {
            node
            for node in resistor_nodes - {"v1", "v2", "v3", "vin"} - output_nodes - reference_nodes
            if
            all(joined(resistor_edges, node, input_node) for input_node in {"v1", "v2", "v3"})
            and resistor_path_exists(
                node,
                output_nodes,
                blocked=feedback_boundaries,
            )
        }
        if not summing_nodes or wrong_positive_e_polarity(summing_nodes):
            return False
        driver_seen = False
        summing_node_consumed = False
        for parts in top_level_lines:
            kind = parts[0][0]
            if kind in {"b", "e"} and output_driver(parts):
                driver_seen = True
                expression = " ".join(parts[3:])
                controlled_nodes = {
                    node.lower()
                    for body in re.findall(r"(?i)\bv\s*\(([^)]*)\)", expression)
                    for node in re.split(r"\s*,\s*|\s+", body.strip())
                    if node
                }
                if not controlled_nodes and kind == "e" and len(parts) >= 5:
                    controlled_nodes = set(parts[3:5])
                if kind == "e" and len(parts) >= 5 and parts[3] == parts[4]:
                    return False
                summing_node_consumed |= bool(controlled_nodes & summing_nodes)
            elif kind == "x" and len(parts) >= 3:
                instance_nodes = set(parts[1:-1])
                if instance_nodes & output_nodes:
                    driver_seen = True
                    summing_node_consumed |= bool(instance_nodes & summing_nodes)
        if driver_seen and not summing_node_consumed:
            return False

    if circuit_id in {"integrator", "differentiator"}:
        topology_boundaries = fixed_nodes | output_nodes
        if circuit_id == "integrator":
            summing_nodes = {
                node
                for node in resistor_nodes - fixed_nodes - output_nodes - {"vin"}
                if resistor_path_exists(
                    node,
                    {"vin"},
                    blocked=topology_boundaries,
                )
                and joined_any(capacitor_edges, node, output_nodes)
            }
        else:
            summing_nodes = {
                node
                for node in resistor_nodes | (set().union(*capacitor_edges) if capacitor_edges else set())
                if node not in fixed_nodes | output_nodes | {"vin"}
                and path_with_capacitor_exists(
                    node,
                    {"vin"},
                    blocked=topology_boundaries,
                )
                and resistor_path_exists(
                    node,
                    output_nodes,
                    blocked=fixed_nodes | family_inputs,
                )
            }
        if not summing_nodes:
            return False

        consumed = False
        for source in controlled_sources:
            controls = controlled_nodes(source)
            if not controls & summing_nodes:
                continue
            if not any(signal_reaches_output(output) for output in source[1:3]):
                continue
            parallel_input = any(
                control not in summing_nodes and node_tracks_input(control, "vin")
                for control in controls
            )
            if not parallel_input:
                consumed = True
        if not consumed:
            return False

    if circuit_id == "sallen-key":
        capacitor_nodes = set().union(*capacitor_edges) if capacitor_edges else set()
        internal_nodes = capacitor_nodes - output_nodes - fixed_nodes - {"vin"}
        sallen_states = {
            node_b
            for node_a in internal_nodes
            for node_b in internal_nodes
            if node_a != node_b
            and joined_any(capacitor_edges, node_a, output_nodes)
            and joined_any(capacitor_edges, node_b, fixed_nodes)
            and joined(resistor_edges, node_a, node_b)
            and resistor_path_exists(
                "vin",
                {node_a},
                blocked=fixed_nodes | output_nodes,
            )
        }
        if not sallen_states:
            return False
        state_consumed = False
        for source in controlled_sources:
            if not output_driver(source):
                continue
            controls = controlled_nodes(source)
            if source[0][0] in {"e", "g"} and len(source) >= 5:
                positive, negative = source[3], source[4]
                valid_controls = positive in sallen_states and (
                    negative in fixed_nodes
                    or negative in output_nodes
                    or resistor_path_exists(
                        negative,
                        output_nodes,
                        blocked=fixed_nodes | family_inputs,
                    )
                )
            else:
                other_controls = controls - sallen_states
                valid_controls = bool(controls & sallen_states) and all(
                    node in fixed_nodes
                    or node in output_nodes
                    or resistor_path_exists(
                        node,
                        output_nodes,
                        blocked=fixed_nodes | family_inputs,
                    )
                    for node in other_controls
                )
            state_consumed |= valid_controls
        state_consumed |= any(
            parts[0][0] == "x"
            and len(parts) >= 4
            and bool(set(parts[1:-1]) & sallen_states)
            and bool(set(parts[1:-1]) & output_nodes)
            for parts in top_level_lines
        )
        if not state_consumed:
            return False

    if circuit_id == "schmitt":
        feedback_nodes = {
            node
            for node in resistor_nodes - output_nodes - {"vin"}
            if
            joined_any(resistor_edges, node, output_nodes)
            and joined_any(resistor_edges, node, reference_nodes)
            # A legitimate load-bearing divider node is often named VREF.
            # Requiring two explicit edges already prevents a rail-only match,
            # so reference-like names must remain eligible candidates.
        }
        if not feedback_nodes:
            return False

        # A divider is not feedback unless the output-driving comparator
        # consumes it.  Accept behavioral/VCVS drivers and subcircuit op amps;
        # transistor-only implementations remain governed by the graph gate.
        comparator_driver_seen = False
        feedback_consumed = False
        for parts in top_level_lines:
            kind = parts[0][0]
            if kind in {"b", "e"} and output_driver(parts):
                comparator_driver_seen = True
                expression = " ".join(parts[3:])
                controlled_nodes = {
                    node.lower()
                    for body in re.findall(r"(?i)\bv\s*\(([^)]*)\)", expression)
                    for node in re.split(r"\s*,\s*|\s+", body.strip())
                    if node
                }
                if not controlled_nodes and kind == "e" and len(parts) >= 5:
                    controlled_nodes = set(parts[3:5])
                feedback_consumed |= bool(controlled_nodes & feedback_nodes)
            elif kind == "x" and len(parts) >= 3:
                instance_nodes = set(parts[1:-1])
                if instance_nodes & output_nodes:
                    comparator_driver_seen = True
                    feedback_consumed |= bool(instance_nodes & feedback_nodes)
        if comparator_driver_seen and not feedback_consumed:
            return False

    if circuit_id == "gain-trim":
        external_nodes = {"vin", "vtrim", "0", "vdd", "vss"}
        supply_nodes = {"0", "vdd", "vss"}

        passive_graph: dict[str, set[str]] = {}
        for parts in top_level_lines:
            if len(parts) >= 4 and parts[0][0] in {"r", "c", "l"}:
                left, right = parts[1], parts[2]
                passive_graph.setdefault(left, set()).add(right)
                passive_graph.setdefault(right, set()).add(left)

        def passively_reaches(node: str, targets: set[str]) -> bool:
            seen = {node}
            pending = [node]
            while pending:
                current = pending.pop()
                if current in targets:
                    return True
                # Independent input sources are graph boundaries.  A path may
                # terminate at Vin/Vtrim when testing for an input alias, but
                # it must not cross that clamped node and masquerade as a
                # load-bearing feedback path to Vout.
                if current in supply_nodes or (current in {"vin", "vtrim"} and current != node):
                    continue
                for neighbor in passive_graph.get(current, ()):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        pending.append(neighbor)
            return False

        sources = [
            (parts, set(parts[1:3]), controlled_nodes(parts))
            for parts in top_level_lines
            if parts[0][0] in {"b", "e"} and len(parts) >= 4
        ]
        has_subcircuit_output_stage = any(
            parts[0][0] == "x"
            and len(parts) >= 3
            and bool(set(parts[1:-1]) & output_nodes)
            for parts in top_level_lines
        )
        drivers = [
            source
            for source in sources
            if bool(source[1] & output_nodes)
            or (not has_subcircuit_output_stage and output_driver(source[0]))
        ]

        for parts, _source_outputs, source_controls in drivers:
            has_feedback_control = any(
                node in output_nodes
                or (
                    node not in external_nodes
                    and passively_reaches(node, output_nodes)
                )
                for node in source_controls
            )
            direct_external = (
                "vin" in source_controls
                and "vtrim" in source_controls
                and not has_feedback_control
            )
            internal_controls = [node for node in source_controls if node not in external_nodes]
            passive_input_alias = (
                bool(internal_controls)
                and all(passively_reaches(node, {"vin", "vtrim"}) for node in internal_controls)
                and not any(passively_reaches(node, output_nodes) for node in internal_controls)
            )
            if direct_external or passive_input_alias:
                return False

            # Follow controlled-source producers upstream from the actual
            # output driver.  A source that combines Vin and Vtrim directly,
            # without consuming output feedback, is an algebraic target
            # synthesizer even when a later E source wraps it in a decorative
            # feedback loop.
            frontier = set(source_controls)
            seen_source_ids = {id(parts)}
            while frontier:
                next_frontier: set[str] = set()
                for upstream_parts, upstream_outputs, upstream_controls in sources:
                    if id(upstream_parts) in seen_source_ids:
                        continue
                    signal_outputs = upstream_outputs - supply_nodes
                    feeds_frontier = any(
                        output in frontier or passively_reaches(output, frontier)
                        for output in signal_outputs
                    )
                    if not feeds_frontier:
                        continue
                    seen_source_ids.add(id(upstream_parts))
                    has_vin = any(
                        node == "vin" or passively_reaches(node, {"vin"})
                        for node in upstream_controls
                    )
                    has_vtrim = any(
                        node == "vtrim" or passively_reaches(node, {"vtrim"})
                        for node in upstream_controls
                    )
                    has_feedback = any(
                        node in output_nodes
                        or (
                            node not in {"vin", "vtrim"}
                            and passively_reaches(node, output_nodes)
                        )
                        for node in upstream_controls
                    )
                    if has_vin and has_vtrim and not has_feedback:
                        return False
                    next_frontier.update(upstream_controls)
                frontier = next_frontier
    return True


def _result(
    reward: float,
    *,
    simulated: bool,
    anticheat_blocked: bool = False,
    requested_specs: int = 0,
    measured_specs: int = 0,
    met_specs: int = 0,
    spec_score: float = 0.0,
    has_spice_block: bool = False,
    has_end: bool = False,
    has_device: bool = False,
    has_analysis: bool = False,
) -> ScoreResult:
    return ScoreResult(
        reward=float(reward),
        simulated=bool(simulated),
        anticheat_blocked=bool(anticheat_blocked),
        requested_specs=int(requested_specs),
        measured_specs=int(measured_specs),
        met_specs=int(met_specs),
        spec_score=float(spec_score),
        all_specs_met=bool(requested_specs and met_specs == requested_specs),
        has_spice_block=bool(has_spice_block),
        has_end=bool(has_end),
        has_device=bool(has_device),
        has_analysis=bool(has_analysis),
    )


def score_completion(
    circuit_id: str,
    target_specs: dict,
    completion: str,
    *,
    eval_tol: float = 0.30,
    timeout_s: float = 15,
    grade_fn: Callable = grade_netlist,
    harness_loader: Callable = load_harness,
    anticheat_fn: Callable = anticheat_ok,
) -> ScoreResult:
    """Score one completion with the frozen, spec-dominant RLVR contract."""

    target_specs = target_specs if isinstance(target_specs, dict) else {}
    requested_specs = len(target_specs)
    if not completion or not isinstance(completion, str):
        return _result(0.0, simulated=False, requested_specs=requested_specs)

    netlist = extract_netlist(completion)
    has_end = bool(re.search(r"\.end\b", netlist, re.IGNORECASE))
    has_spice_block = bool(re.search(r"```spice\s*.*?```", completion, re.DOTALL | re.IGNORECASE))
    has_device = bool(
        re.search(r"^\s*[RCLMQVIX]\w*\s+\S+\s+\S+", netlist, re.MULTILINE | re.IGNORECASE)
    )
    has_analysis = bool(re.search(r"\.(op|tran|ac|dc)\b|\.control\b", netlist, re.IGNORECASE))
    structure = {
        "requested_specs": requested_specs,
        "has_spice_block": has_spice_block,
        "has_end": has_end,
        "has_device": has_device,
        "has_analysis": has_analysis,
    }
    if not (has_spice_block and has_end):
        return _result(0.0, simulated=False, **structure)

    passed, _ = grade_fn(netlist, timeout_s=timeout_s)
    if not passed:
        return _result(0.0, simulated=False, **structure)

    if not anticheat_fn(netlist, circuit_id):
        return _result(
            0.0,
            simulated=True,
            anticheat_blocked=True,
            **structure,
        )

    harness = harness_loader(circuit_id)
    if harness is None or not target_specs:
        return _result(0.0, simulated=True, **structure)
    try:
        measured = harness.measure(netlist, spec=target_specs, timeout_s=timeout_s)
    except Exception:
        return _result(0.0, simulated=True, **structure)
    measured = measured if isinstance(measured, dict) else {}

    # A gain-trim stage must respond to its trim input even when a generated
    # single-objective prompt asks only for gain.  Otherwise a fixed-gain
    # amplifier can receive full credit for the wrong circuit family.
    if circuit_id == "gain-trim":
        try:
            gain_modulation_db = float(measured.get("gain_modulation_db"))
            functional_trim = (
                math.isfinite(gain_modulation_db)
                and gain_modulation_db >= 0.01
            )
        except (TypeError, ValueError, OverflowError):
            functional_trim = False
        if not functional_trim:
            return _result(
                0.0,
                simulated=True,
                anticheat_blocked=True,
                **structure,
            )

    # A buffer must preserve signal polarity and remain a low-gain stage even
    # when the prompt requests only bandwidth.  The signed 1 Hz transfer is a
    # simulator measurement, so the rule also covers aliases hidden inside a
    # subcircuit where the structural gate cannot inspect control polarity.
    if circuit_id == "buffer":
        try:
            signed_gain_vv = float(measured.get("signed_gain_vv"))
            functional_buffer = (
                math.isfinite(signed_gain_vv) and 0.25 <= signed_gain_vv <= 5.0
            )
        except (TypeError, ValueError, OverflowError):
            functional_buffer = False
        if not functional_buffer:
            return _result(
                0.0,
                simulated=True,
                anticheat_blocked=True,
                **structure,
            )

    # Magnitude-only AC gain can mistake a non-inverting transfer for an
    # inverting amplifier. Require the injected DC sweep to show a negative
    # signed small-signal slope before any shaped or exact reward is paid.
    if circuit_id == "inv-amp":
        try:
            signed_gain_vv = float(measured.get("signed_gain_vv"))
            functional_inversion = math.isfinite(signed_gain_vv) and signed_gain_vv < -1e-6
        except (TypeError, ValueError, OverflowError):
            functional_inversion = False
        if not functional_inversion:
            return _result(
                0.0,
                simulated=True,
                anticheat_blocked=True,
                **structure,
            )

    if circuit_id == "summing-amp":
        try:
            signed_gains = [
                float(measured.get(f"signed_gain_v{index}_vv"))
                for index in (1, 2, 3)
            ]
            functional_inversion = all(
                math.isfinite(value) and value < -1e-6 for value in signed_gains
            )
        except (TypeError, ValueError, OverflowError):
            functional_inversion = False
        if not functional_inversion:
            return _result(
                0.0,
                simulated=True,
                anticheat_blocked=True,
                **structure,
            )

    # A Schmitt trigger must have two ordered switching thresholds.  Matching
    # one requested window with an ordinary comparator is not partial success.
    if circuit_id == "schmitt":
        try:
            high = float(measured.get("vth_high_v"))
            low = float(measured.get("vth_low_v"))
            functional_hysteresis = (
                math.isfinite(high) and math.isfinite(low) and high - low > 1e-6
            )
        except (TypeError, ValueError, OverflowError):
            functional_hysteresis = False
        if not functional_hysteresis:
            return _result(
                0.0,
                simulated=True,
                anticheat_blocked=True,
                **structure,
            )

    measured_specs = 0
    met_specs = 0
    raw_spec_score = 0.0
    for key, target in target_specs.items():
        measured_value = measured.get(key)
        if measured_value is None:
            continue
        try:
            measured_value = float(measured_value)
            if isinstance(target, (list, tuple)) and len(target) == 2:
                low, high = sorted((float(target[0]), float(target[1])))
            else:
                target_value = float(target)
                low, high = sorted(
                    (target_value * (1 - eval_tol), target_value * (1 + eval_tol))
                )
        except (TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(value) for value in (measured_value, low, high)):
            continue
        measured_specs += 1
        if low <= measured_value <= high:
            raw_spec_score += 1.0
            met_specs += 1
        else:
            distance = low - measured_value if measured_value < low else measured_value - high
            scale = max(abs(high - low), abs((low + high) / 2.0), 1e-9)
            raw_spec_score += 0.25 * max(0.0, 1.0 - distance / scale)

    normalized_spec_score = raw_spec_score / requested_specs if requested_specs else 0.0
    reward = 0.70 * normalized_spec_score
    if requested_specs and met_specs == requested_specs:
        reward += 0.60
    return _result(
        reward,
        simulated=True,
        measured_specs=measured_specs,
        met_specs=met_specs,
        spec_score=normalized_spec_score,
        **structure,
    )
