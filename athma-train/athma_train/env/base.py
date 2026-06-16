"""phy-chip-env base interfaces.

A *physics environment* turns a model output (typically a netlist or RTL string)
into a scalar reward for RLVR. It is the analog of `gym.Env` for verifiable physics.

Design constraints driving the interface:

  1. **Sandboxed execution.** Simulators (ngspice, verilator, yosys) are external
     processes. They must run with CPU/memory/time limits and a writable scratch
     dir that is wiped between rollouts. We use `subprocess` + `resource` rlimits
     on Linux and a Docker fallback on macOS dev boxes.

  2. **Batched rollouts.** A GRPO step generates K samples per prompt across N
     prompts → K·N simulations per step. With ngspice typically 50-500ms per
     run, single-threaded execution is the bottleneck. Envs MUST expose
     `simulate_batch` that fan-outs to a process pool.

  3. **Spec-driven reward.** A reward function consumes (a) sim success/failure,
     (b) measured quantities pulled from the sim output, and (c) a dict of
     target spec values. Reward shaping is a separate concern from running the
     simulator — split for testability.

  4. **Pickle-safety for Ray.** veRL drives rollouts via Ray actors; envs must
     be constructible from a config dict and contain no live processes / open FDs.

The concrete adapters (`SpiceEnv`, `VerilatorEnv`) live in sibling modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SimResult:
    """Output of a single simulator run."""

    success: bool
    measurements: dict[str, float] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    wall_time_s: float = 0.0
    exit_code: int = 0
    artifacts: dict[str, str] = field(default_factory=dict)  # path → label


@dataclass
class RewardBreakdown:
    """Per-sample reward decomposition for diagnostics + W&B logging."""

    total: float
    components: dict[str, float] = field(default_factory=dict)
    spec_satisfied: dict[str, bool] = field(default_factory=dict)


class PhysicsEnv(ABC):
    """Common interface for all RLVR physics environments."""

    name: str = "abstract"

    @abstractmethod
    def simulate(self, design: str, spec: dict[str, Any]) -> SimResult:
        """Run the simulator on `design` (e.g. a SPICE netlist string)."""

    def simulate_batch(self, designs: list[str], specs: list[dict[str, Any]]) -> list[SimResult]:
        """Default: serial fallback. Adapters MUST override with a process pool."""
        return [self.simulate(d, s) for d, s in zip(designs, specs, strict=True)]

    @abstractmethod
    def reward(self, sim: SimResult, spec: dict[str, Any]) -> RewardBreakdown:
        """Compose a scalar reward from sim output and target specs."""

    def __call__(self, design: str, spec: dict[str, Any]) -> RewardBreakdown:
        return self.reward(self.simulate(design, spec), spec)
