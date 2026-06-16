"""SpiceEnv — ngspice as a verifiable physics environment.

SCAFFOLD ONLY. Implementation lands in Phase 1, Week 4. Documents the contract.

Reward function (Stage 5a) per docs/00_phychip_product.md §4:
    reward = 0                                    if simulation fails
    reward = base + Σ bonuses_per_spec_met        if simulation succeeds
    reward += spec_margin_bonus                   for exceeding targets
    reward -= power_excess_penalty                for inefficient designs

Implementation notes for the engineer wiring this up:
  - Use `ngspice -b -o out.log netlist.cir` with `RLIMIT_CPU=10s, RLIMIT_AS=512MB`.
  - Parse `.print`, `.measure`, and `.op` directives via regex on the log.
  - For .ac sweeps, parse the rawfile (binary) using `spicelib` or `PySpice`.
  - The simulator must NEVER block forever on convergence failure — wrap in a
    SIGKILL-after-timeout watchdog.
  - Process pool size is min(os.cpu_count(), batch_size). One ngspice per worker.
  - Scratch dir per rollout: `/tmp/athma_spice/<rollout_id>/`, wiped after parse.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any

from athma_train.env.base import PhysicsEnv, RewardBreakdown, SimResult


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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cir", delete=False, dir=self.cfg.scratch_root
        ) as f:
            f.write(design)
            cir_path = f.name

        try:
            proc = subprocess.run(
                [self.cfg.ngspice_bin, "-b", cir_path],
                capture_output=True,
                text=True,
                timeout=self.cfg.timeout_s,
            )
        except subprocess.TimeoutExpired:
            os.unlink(cir_path)
            return SimResult(
                success=False,
                stdout="",
                stderr="ngspice timeout",
                wall_time_s=time.monotonic() - start,
                exit_code=-1,
            )
        except FileNotFoundError:
            os.unlink(cir_path)
            return SimResult(
                success=False,
                stdout="",
                stderr=f"{self.cfg.ngspice_bin} not found",
                wall_time_s=time.monotonic() - start,
                exit_code=-2,
            )
        finally:
            # Only delete on success — on exception we already deleted above.
            if os.path.exists(cir_path):
                os.unlink(cir_path)

        stdout = proc.stdout
        stderr = proc.stderr
        # ngspice returns 0 on clean completion and 1 on "warnings but ok".
        # Anything else (e.g. -6 on convergence failure) is a hard failure.
        success = proc.returncode in (0, 1) and "run simulation(s) aborted" not in stdout

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
