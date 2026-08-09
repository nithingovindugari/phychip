"""Diagnostic observations built around the frozen scorer-v9 authority."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any

from .registry import get_harness
from .sandbox import grade_netlist
from .scorer import (
    SCORER_SEMANTICS,
    ScoreResult,
    anticheat_ok,
    extract_netlist,
    score_completion,
)

OBSERVATION_SCHEMA = "phy-chip-env-observation-v1"


@dataclass(frozen=True)
class ObservationFailure:
    """Typed failure emitted without changing reward semantics."""

    stage: str
    code: str
    detail: str


@dataclass(frozen=True)
class Observation:
    """One canonical verifier evaluation plus optimizer-facing diagnostics."""

    circuit_id: str
    netlist: str
    netlist_sha256: str
    requested_bounds: dict[str, tuple[float, float]]
    measurements: dict[str, float | None]
    score: ScoreResult
    failure: ObservationFailure | None
    verifier_calls: int = 1
    scorer_semantics: str = SCORER_SEMANTICS
    schema: str = OBSERVATION_SCHEMA

    def to_event(self) -> dict[str, Any]:
        """Return a stable JSON-safe event."""

        value = asdict(self)
        value["requested_bounds"] = {
            name: list(bounds) for name, bounds in self.requested_bounds.items()
        }
        return value


def _normalize_bounds(
    target_specs: dict,
    eval_tol: float,
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    if not isinstance(target_specs, dict):
        return bounds
    for name, target in target_specs.items():
        try:
            if isinstance(target, list | tuple) and len(target) == 2:
                low, high = sorted((float(target[0]), float(target[1])))
            else:
                center = float(target)
                low, high = sorted(
                    (center * (1.0 - eval_tol), center * (1.0 + eval_tol))
                )
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(low) and math.isfinite(high):
            bounds[str(name)] = (low, high)
    return bounds


def _format_failure(
    score: ScoreResult,
    *,
    grade_called: bool,
    grade_passed: bool | None,
    grade_reason: str,
    harness_found: bool,
    measurement_error: str,
) -> ObservationFailure | None:
    if not score.has_spice_block:
        return ObservationFailure("format", "missing_spice_block", "expected a ```spice block")
    if not score.has_end:
        return ObservationFailure("format", "missing_end", "netlist must terminate with .end")
    if not score.has_device:
        return ObservationFailure("format", "missing_device", "no supported device card found")
    if not score.has_analysis:
        return ObservationFailure(
            "format",
            "missing_analysis",
            "netlist needs .op/.tran/.ac/.dc or a .control block",
        )
    if grade_called and grade_passed is False:
        code = "unsafe_netlist" if grade_reason.startswith("unsafe netlist:") else "simulation_failed"
        return ObservationFailure("simulation", code, grade_reason)
    if score.anticheat_blocked:
        return ObservationFailure(
            "topology",
            "family_contract_rejected",
            "scorer-v9 rejected the circuit-family implementation",
        )
    if not harness_found:
        return ObservationFailure(
            "measurement",
            "unknown_harness",
            "no registered measurement harness for circuit_id",
        )
    if measurement_error:
        return ObservationFailure("measurement", "measurement_exception", measurement_error)
    if score.measured_specs < score.requested_specs:
        return ObservationFailure(
            "measurement",
            "missing_measurements",
            f"measured {score.measured_specs}/{score.requested_specs} requested metrics",
        )
    return None


def observe_completion(
    circuit_id: str,
    target_specs: dict,
    completion: str,
    *,
    eval_tol: float = 0.30,
    timeout_s: float = 15.0,
    grade_fn=grade_netlist,
    harness_loader=get_harness,
    anticheat_fn=anticheat_ok,
) -> Observation:
    """Observe one candidate through the canonical scorer exactly once.

    The wrapper captures the scorer's own grade and harness calls. It does not
    compute success or reward independently.
    """

    captured: dict[str, Any] = {
        "grade_called": False,
        "grade_passed": None,
        "grade_reason": "",
        "harness_found": False,
        "measurement_error": "",
        "measurements": {},
    }

    def capturing_grade(netlist: str, *, timeout_s: float):
        captured["grade_called"] = True
        passed, reason = grade_fn(netlist, timeout_s=timeout_s)
        captured["grade_passed"] = bool(passed)
        captured["grade_reason"] = str(reason)
        return passed, reason

    def capturing_harness_loader(requested_circuit_id: str):
        harness = harness_loader(requested_circuit_id)
        captured["harness_found"] = harness is not None
        if harness is None:
            return None

        class CapturingHarness:
            @staticmethod
            def measure(netlist: str, spec: dict | None = None, timeout_s: int = 30):
                try:
                    measured = harness.measure(netlist, spec=spec, timeout_s=timeout_s)
                except Exception as exc:
                    captured["measurement_error"] = f"{type(exc).__name__}: {exc}"
                    raise
                if isinstance(measured, dict):
                    captured["measurements"] = dict(measured)
                return measured

        return CapturingHarness

    score = score_completion(
        circuit_id,
        target_specs,
        completion,
        eval_tol=eval_tol,
        timeout_s=timeout_s,
        grade_fn=capturing_grade,
        harness_loader=capturing_harness_loader,
        anticheat_fn=anticheat_fn,
    )
    netlist = extract_netlist(completion if isinstance(completion, str) else "")
    failure = _format_failure(
        score,
        grade_called=bool(captured["grade_called"]),
        grade_passed=captured["grade_passed"],
        grade_reason=str(captured["grade_reason"]),
        harness_found=bool(captured["harness_found"]),
        measurement_error=str(captured["measurement_error"]),
    )
    return Observation(
        circuit_id=circuit_id,
        netlist=netlist,
        netlist_sha256=hashlib.sha256(netlist.encode("utf-8")).hexdigest(),
        requested_bounds=_normalize_bounds(target_specs, eval_tol),
        measurements={
            str(name): (
                float(value)
                if isinstance(value, int | float) and math.isfinite(float(value))
                else None
            )
            for name, value in captured["measurements"].items()
            if value is None or isinstance(value, int | float)
        },
        score=score,
        failure=failure,
    )
