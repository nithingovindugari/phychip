from __future__ import annotations

from types import SimpleNamespace

from phy_chip_env.observation import observe_completion

COMPLETION = """```spice
Vin vin 0 DC 0 AC 1
R1 vin vout 1k
.op
.end
```"""


def test_observation_captures_canonical_measurements_and_score():
    harness = SimpleNamespace(
        measure=lambda netlist, spec, timeout_s: {
            "gain_db": 0.0,
            "signed_gain_vv": 1.0,
            "extra": 7.0,
        }
    )

    observation = observe_completion(
        "buffer",
        {"gain_db": [-0.1, 0.1]},
        COMPLETION,
        grade_fn=lambda netlist, timeout_s: (True, "ok"),
        harness_loader=lambda circuit_id: harness,
        anticheat_fn=lambda netlist, circuit_id: True,
    )

    assert observation.score.all_specs_met
    assert observation.score.reward == 1.2999999999999998
    assert observation.measurements == {
        "gain_db": 0.0,
        "signed_gain_vv": 1.0,
        "extra": 7.0,
    }
    assert observation.failure is None
    assert observation.verifier_calls == 1


def test_observation_types_simulation_failure():
    observation = observe_completion(
        "buffer",
        {"gain_db": [-0.1, 0.1]},
        COMPLETION,
        grade_fn=lambda netlist, timeout_s: (False, "exit=1"),
        harness_loader=lambda circuit_id: None,
    )

    assert not observation.score.simulated
    assert observation.failure is not None
    assert observation.failure.stage == "simulation"
    assert observation.failure.code == "simulation_failed"


def test_observation_types_topology_rejection_after_measurement():
    harness = SimpleNamespace(
        measure=lambda netlist, spec, timeout_s: {"gain_db": 0.0}
    )
    observation = observe_completion(
        "buffer",
        {"gain_db": [-0.1, 0.1]},
        COMPLETION,
        grade_fn=lambda netlist, timeout_s: (True, "ok"),
        harness_loader=lambda circuit_id: harness,
        anticheat_fn=lambda netlist, circuit_id: False,
    )

    assert observation.score.anticheat_blocked
    assert observation.failure is not None
    assert observation.failure.stage == "topology"


def test_observation_types_measurement_exception():
    def fail_measure(netlist, spec, timeout_s):
        raise RuntimeError("bad measurement")

    observation = observe_completion(
        "buffer",
        {"gain_db": [-0.1, 0.1]},
        COMPLETION,
        grade_fn=lambda netlist, timeout_s: (True, "ok"),
        harness_loader=lambda circuit_id: SimpleNamespace(measure=fail_measure),
        anticheat_fn=lambda netlist, circuit_id: True,
    )

    assert observation.failure is not None
    assert observation.failure.code == "measurement_exception"
    assert "RuntimeError" in observation.failure.detail
