"""Low-level generic ngspice environment.

This module provides the original :class:`PhysicsEnv`-compatible simulator and
generic measurement-dictionary reward. It is not the PhyChip RLVR reward used
for training or evaluation. The authoritative circuit-family scorer is
``athma_train.env.rlvr_scorer.score_completion``; it owns extraction,
family-specific harness measurement, anti-cheat checks, and reward
decomposition. Keeping the distinction explicit prevents the old scaffold
reward from being packaged as the research result.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from athma_train.env.base import PhysicsEnv, RewardBreakdown, SimResult
from athma_train.spice_gate import UnsafeNetlistError, run_ngspice_sandboxed


@dataclass
class SpiceEnvConfig:
    ngspice_bin: str = "ngspice"
    timeout_s: float = 10.0
    mem_limit_mb: int = 512
    scratch_root: str = "/tmp/athma_spice"
    pool_size: int | None = None  # None → os.cpu_count()
    base_reward: float = 1.0
    margin_bonus_weight: float = 0.5
    power_penalty_weight: float = 0.2


class SpiceEnv(PhysicsEnv):
    name = "spice"

    def __init__(self, config: SpiceEnvConfig | None = None) -> None:
        self.cfg = config or SpiceEnvConfig()

    def simulate(self, design: str, spec: dict[str, Any]) -> SimResult:
        """Run ngspice in batch mode on *design* (raw SPICE netlist string).

        Returns a SimResult with node voltages under ``measurements``
        (keys ``v(<node>)``) and branch currents (keys ``i(<source>)``).
        """
        start = time.monotonic()
        try:
            proc = run_ngspice_sandboxed(
                design,
                timeout_s=self.cfg.timeout_s,
                memory_limit_mb=self.cfg.mem_limit_mb,
                ngspice_bin=self.cfg.ngspice_bin,
                scratch_root=self.cfg.scratch_root,
            )
        except subprocess.TimeoutExpired:
            return SimResult(
                success=False,
                stdout="",
                stderr="ngspice timeout",
                wall_time_s=time.monotonic() - start,
                exit_code=-1,
            )
        except UnsafeNetlistError as exc:
            return SimResult(
                success=False,
                stdout="",
                stderr=f"unsafe netlist: {exc}",
                wall_time_s=time.monotonic() - start,
                exit_code=-3,
            )
        except OSError:
            return SimResult(
                success=False,
                stdout="",
                stderr=f"{self.cfg.ngspice_bin} not found",
                wall_time_s=time.monotonic() - start,
                exit_code=-2,
            )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        success = proc.returncode == 0 and "run simulation(s) aborted" not in stdout

        measurements: dict[str, float] = {}

        # --- Parse node voltages ---
        lines = stdout.splitlines()
        for i, line in enumerate(lines):
            if "Node" in line and "Voltage" in line:
                # Parse subsequent lines until a blank line or the next header.
                for j in range(i + 1, len(lines)):
                    row = lines[j]
                    if row.strip() == "" or ("Source" in row and "Current" in row):
                        break
                    # Exclude #branch lines that sometimes leak into the voltage table.
                    m = re.match(
                        r"^\s*([a-zA-Z0-9_]+)\s+([+-]?\d+\.?\d*[eE][+-]?\d+)\s*$", row
                    )
                    if m:
                        node, val = m.group(1), float(m.group(2))
                        if "#branch" not in node:
                            measurements[f"v({node})"] = val
                break

        # --- Parse branch currents ---
        for i, line in enumerate(lines):
            if "Source" in line and "Current" in line:
                for j in range(i + 1, len(lines)):
                    row = lines[j]
                    if row.strip() == "":
                        continue
                    # Skip dashed separator lines.
                    if set(row.strip()) <= {"-", " ", "\t"}:
                        continue
                    m = re.match(
                        r"^\s*([a-zA-Z0-9_#]+)\s+([+-]?\d+\.?\d*[eE][+-]?\d+)\s*$", row
                    )
                    if m:
                        src, val = m.group(1), float(m.group(2))
                        measurements[f"i({src})"] = val
                    else:
                        break
                break

        # A successful sim must have at least one non-zero current to confirm
        # the circuit is not floating / open.
        has_current = any(abs(v) > 1e-12 for k, v in measurements.items() if k.startswith("i("))
        if not has_current:
            success = False

        return SimResult(
            success=success,
            measurements=measurements,
            stdout=stdout,
            stderr=stderr,
            wall_time_s=time.monotonic() - start,
            exit_code=proc.returncode,
        )

    def simulate_file(self, path: str) -> SimResult:
        with open(path) as f:
            return self.simulate(f.read(), spec={})

    def reward(self, sim: SimResult, spec: dict[str, Any]) -> RewardBreakdown:
        """Return the legacy generic reward, not the canonical RLVR score."""
        if not sim.success:
            return RewardBreakdown(total=0.0, components={"sim_failed": 0.0})

        components: dict[str, float] = {"base": self.cfg.base_reward}
        spec_satisfied: dict[str, bool] = {}

        for key, target in spec.items():
            measured = sim.measurements.get(key)
            if measured is None:
                spec_satisfied[key] = False
                continue
            ok, margin = _check_spec(key, measured, target)
            spec_satisfied[key] = ok
            if ok:
                components[f"spec_{key}"] = 1.0 / max(len(spec), 1)
                components[f"margin_{key}"] = self.cfg.margin_bonus_weight * margin

        # Power penalty: spec target "power_max_mw" vs measured "power_mw"
        if "power_max_mw" in spec and "power_mw" in sim.measurements:
            excess = max(0.0, sim.measurements["power_mw"] - spec["power_max_mw"])
            components["power_penalty"] = -self.cfg.power_penalty_weight * (excess / spec["power_max_mw"])

        total = sum(components.values())
        return RewardBreakdown(total=total, components=components, spec_satisfied=spec_satisfied)


def _check_spec(key: str, measured: float, target: Any) -> tuple[bool, float]:
    """Return (satisfied, normalized_margin).

    Spec encoding convention:
        target = float                  → equality within ±5%
        target = (">=", value)          → measured ≥ value
        target = ("<=", value)          → measured ≤ value
        target = ("range", lo, hi)      → lo ≤ measured ≤ hi
    """
    if isinstance(target, (int, float)):
        rel = abs(measured - target) / max(abs(target), 1e-12)
        return rel <= 0.05, max(0.0, 0.05 - rel) / 0.05
    if isinstance(target, tuple):
        op = target[0]
        if op == ">=":
            return measured >= target[1], (measured - target[1]) / max(abs(target[1]), 1e-12)
        if op == "<=":
            return measured <= target[1], (target[1] - measured) / max(abs(target[1]), 1e-12)
        if op == "range":
            lo, hi = target[1], target[2]
            return lo <= measured <= hi, 0.0
    return False, 0.0
