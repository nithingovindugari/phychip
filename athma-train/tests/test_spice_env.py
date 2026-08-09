"""Unit tests for SpiceEnv reward shaping — runs WITHOUT ngspice installed.

These test the reward composition logic only (the simulator is mocked). Real
ngspice integration tests live in tests/integration/ and are skipped if `which
ngspice` returns nothing.
"""

from __future__ import annotations

import pytest

from athma_train.env.base import SimResult
from athma_train.env.spice_env import SpiceEnv, SpiceEnvConfig


def test_reward_zero_on_sim_failure() -> None:
    env = SpiceEnv()
    sim = SimResult(success=False, stderr="no convergence")
    out = env.reward(sim, spec={"gain_db": 20.0})
    assert out.total == 0.0
    assert out.components == {"sim_failed": 0.0}


def test_reward_base_plus_spec_bonus_on_success() -> None:
    env = SpiceEnv(SpiceEnvConfig(base_reward=1.0, margin_bonus_weight=0.5))
    sim = SimResult(success=True, measurements={"gain_db": 20.0})
    out = env.reward(sim, spec={"gain_db": 20.0})
    assert out.total > 1.0
    assert out.spec_satisfied["gain_db"] is True


def test_reward_power_penalty() -> None:
    env = SpiceEnv(SpiceEnvConfig(power_penalty_weight=0.2))
    sim = SimResult(success=True, measurements={"gain_db": 20.0, "power_mw": 10.0})
    out = env.reward(sim, spec={"gain_db": 20.0, "power_max_mw": 5.0})
    assert "power_penalty" in out.components
    assert out.components["power_penalty"] < 0.0


def test_spec_inequality_ge() -> None:
    env = SpiceEnv()
    sim = SimResult(success=True, measurements={"bw_mhz": 5.0})
    out = env.reward(sim, spec={"bw_mhz": (">=", 1.0)})
    assert out.spec_satisfied["bw_mhz"] is True


def test_spec_range() -> None:
    env = SpiceEnv()
    sim = SimResult(success=True, measurements={"vout": 0.6})
    out = env.reward(sim, spec={"vout": ("range", 0.5, 0.7)})
    assert out.spec_satisfied["vout"] is True
