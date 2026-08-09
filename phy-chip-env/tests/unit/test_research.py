from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from phy_chip_env.campaign import (
    SKY130_CAMPAIGN_SCHEMA,
    load_sky130_campaign,
    preflight_sky130_campaign,
)
from phy_chip_env.cli import main
from phy_chip_env.observation import Observation, observe_completion
from phy_chip_env.provenance import runtime_fingerprint
from phy_chip_env.readiness import assess_training_readiness
from phy_chip_env.research import (
    CodexCLIController,
    DeterministicLatinHypercube,
    DeterministicRandom,
    DeterministicTPE,
    EpisodeLog,
    FeedbackBlindCodexCLIController,
    Proposal,
    ResearchRunner,
    compare_benchmark_runs,
    compare_feedback_runs,
    fixed_search_controller_id,
    load_benchmark,
    paired_feedback_schedule,
    research_episode_id,
    run_benchmark,
    run_fixed_topology,
    run_paired_feedback_benchmark,
    summarize_comparison,
    summarize_feedback_comparison,
    validated_episode_outcome,
    verifier_completion,
)
from phy_chip_env.scorer import ScoreResult

BENCHMARK = Path(__file__).parents[2] / "benchmarks" / "closed_loop_v1.json"
FEEDBACK_BENCHMARK = (
    Path(__file__).parents[2] / "benchmarks" / "feedback_use_v3.json"
)
SKY130_CAMPAIGN = (
    Path(__file__).parents[2] / "benchmarks" / "sky130_opamp_campaign_v1.json"
)
BENCHMARK_RESULTS = Path(__file__).parents[2] / "benchmarks"
PARAMETER_SEARCH_RESULT = BENCHMARK_RESULTS / "parameter_search_v1_result.json"
RELEASE_RUNTIME_FINGERPRINT = (
    Path(__file__).parents[3]
    / "release"
    / "v0.1"
    / "phy_chip_env_runtime_fingerprint.json"
)
TEST_CODEX_CONTROLLER_ID = CodexCLIController(model="test").controller_id


def _score(*, success: bool, reward: float) -> ScoreResult:
    return ScoreResult(
        reward=reward,
        simulated=True,
        anticheat_blocked=False,
        requested_specs=1,
        measured_specs=1,
        met_specs=int(success),
        spec_score=1.0 if success else 0.5,
        all_specs_met=success,
        has_spice_block=True,
        has_end=True,
        has_device=True,
        has_analysis=True,
    )


def _observation(*, success: bool, reward: float) -> Observation:
    return Observation(
        circuit_id="inv-amp",
        netlist="R1 vin vout 1k\n.op\n.end",
        netlist_sha256="a" * 64,
        requested_bounds={"gain_db": (19.0, 21.0)},
        measurements={"gain_db": 20.0 if success else 10.0},
        score=_score(success=success, reward=reward),
        failure=None,
    )


def test_frozen_benchmark_and_templates_validate():
    metadata, tasks = load_benchmark(BENCHMARK)

    assert metadata["benchmark_id"] == "phychip-closed-loop-v1"
    assert len(metadata["task_set_sha256"]) == 64
    assert len(tasks) == 4
    assert all(task.fixed_template is not None for task in tasks)
    for task in tasks:
        assert task.max_verifier_calls == 8
        assert task.max_codex_calls == 8


def test_random_sampler_is_deterministic_and_index_addressable():
    _, tasks = load_benchmark(BENCHMARK)
    template = tasks[0].fixed_template
    assert template is not None
    first = DeterministicRandom(template, seed=17)
    replay = DeterministicRandom(template, seed=17)

    indexed = [first.suggest([({}, 0.0)] * index) for index in range(2)]
    replayed = [replay.suggest([({}, 0.0)] * index) for index in range(2)]

    assert indexed == replayed
    assert indexed[0] != indexed[1]
    for point in indexed:
        for parameter in template.parameters:
            assert parameter.low <= point[parameter.name] <= parameter.high


def test_latin_hypercube_covers_every_stratum_and_resumes_exactly():
    _, tasks = load_benchmark(BENCHMARK)
    template = tasks[0].fixed_template
    assert template is not None
    trials = tasks[0].max_verifier_calls
    sampler = DeterministicLatinHypercube(template, seed=29, trials=trials)
    replay = DeterministicLatinHypercube(template, seed=29, trials=trials)

    points = [sampler.suggest([({}, 0.0)] * index) for index in range(trials)]
    replayed = [replay.suggest([({}, 0.0)] * index) for index in range(trials)]

    assert points == replayed
    for parameter in template.parameters:
        strata = sorted(int(parameter.to_unit(point[parameter.name]) * trials) for point in points)
        assert strata == list(range(trials))
    with pytest.raises(ValueError, match="exhausted"):
        sampler.suggest([({}, 0.0)] * trials)


def test_feedback_benchmark_freezes_one_initial_failure_per_task():
    metadata, tasks = load_benchmark(FEEDBACK_BENCHMARK)

    assert metadata["benchmark_id"] == "phychip-feedback-use-v3-eight-family-preflight"
    assert metadata["split_policy"]["role"] == "paired_feedback_evaluation"
    assert len(tasks) == 24
    assert all(task.initial_netlist for task in tasks)
    assert all(task.fixed_template is None for task in tasks)
    assert all(task.max_verifier_calls == 8 for task in tasks)
    assert all(task.max_codex_calls == 7 for task in tasks)


def test_verifier_augmentation_is_hidden_from_controller_and_restricted():
    _, tasks = load_benchmark(FEEDBACK_BENCHMARK)
    base = tasks[0]
    task = type(base)(
        **{
            **base.__dict__,
            "controller_task_id": "opaque-task-01",
            "verifier_augmentation": "Csecret vout 0 37.2p",
        }
    )

    task.validate()
    prompt = CodexCLIController._prompt(task, [])
    assert base.task_id not in prompt
    assert "opaque-task-01" in prompt
    assert "Csecret" not in prompt
    assert "37.2p" not in prompt
    assert task.identity()["verifier_augmentation"] == "Csecret vout 0 37.2p"

    unsafe = type(base)(
        **{
            **base.__dict__,
            "verifier_augmentation": "Bsecret vout 0 v={10*v(vin)}",
        }
    )
    with pytest.raises(ValueError, match="passive R/C"):
        unsafe.validate()

    for invalid_value in ("0", "-1n", "1wat", "1e"):
        invalid = type(base)(
            **{
                **base.__dict__,
                "verifier_augmentation": f"Csecret vout 0 {invalid_value}",
            }
        )
        with pytest.raises(ValueError, match="positive physical SPICE value"):
            invalid.validate()


@pytest.mark.integration
def test_feedback_benchmark_augmented_initial_candidates_are_replayable_failures():
    _, tasks = load_benchmark(FEEDBACK_BENCHMARK)

    for task in tasks:
        result = observe_completion(
            task.circuit_id,
            task.target_specs,
            verifier_completion(task, task.initial_netlist),
            timeout_s=task.timeout_s,
        )
        assert result.failure is None
        assert result.score.simulated is True
        assert result.score.all_specs_met is False
        assert result.measurements


@pytest.mark.integration
def test_hidden_parasitic_benchmark_is_frozen_prompt_blind_and_replayable():
    metadata, tasks = load_benchmark(FEEDBACK_BENCHMARK)

    assert metadata["split_policy"]["role"] == "paired_feedback_evaluation"
    assert metadata["promotion_gates"]["minimum_paired_tasks"] == 24
    assert len(tasks) == 24
    for task in tasks:
        assert task.verifier_augmentation
        prompt = CodexCLIController._prompt(task, [])
        assert task.controller_task_id
        assert task.task_id not in prompt
        assert task.verifier_augmentation not in prompt
        assert task.verifier_augmentation.split()[0] not in prompt
        result = observe_completion(
            task.circuit_id,
            task.target_specs,
            verifier_completion(task, task.initial_netlist),
            timeout_s=task.timeout_s,
        )
        assert result.failure is None
        assert result.score.simulated is True
        assert result.score.all_specs_met is False
        assert result.measurements


@pytest.mark.integration
def test_powered_feedback_benchmark_is_preregistered_diverse_and_replayable():
    metadata, tasks = load_benchmark(FEEDBACK_BENCHMARK)

    gates = metadata["promotion_gates"]
    assert metadata["status"] == "preregistered-frozen-unexecuted"
    assert metadata["split_policy"]["role"] == "paired_feedback_evaluation"
    assert gates["minimum_paired_tasks"] == 24
    assert gates["minimum_circuit_families"] == 8
    assert gates["component_practical_success_rate"] == 1.0
    assert len(tasks) == 24
    assert len({task.circuit_id for task in tasks}) == 8
    assert len({task.controller_task_id for task in tasks}) == len(tasks)
    assert len(
        {
            (
                task.prompt,
                json.dumps(task.target_specs, sort_keys=True),
                task.initial_netlist,
                task.allowed_topologies,
            )
            for task in tasks
        }
    ) == len(tasks)
    for task in tasks:
        assert task.initial_netlist
        assert task.verifier_augmentation
        assert task.controller_task_id
        prompt = CodexCLIController._prompt(task, [])
        assert task.task_id not in prompt
        assert task.verifier_augmentation not in prompt
        result = observe_completion(
            task.circuit_id,
            task.target_specs,
            verifier_completion(task, task.initial_netlist),
            timeout_s=task.timeout_s,
        )
        assert result.failure is None
        assert result.score.simulated is True
        assert result.score.all_specs_met is False
        assert result.measurements
        unaugmented = observe_completion(
            task.circuit_id,
            task.target_specs,
            f"```spice\n{task.initial_netlist}\n```",
            timeout_s=task.timeout_s,
        )
        assert any(
            result.measurements.get(metric) != unaugmented.measurements.get(metric)
            for metric in task.target_specs
        )


def test_checked_in_research_decisions_bind_current_contracts():
    metadata, tasks = load_benchmark(BENCHMARK)
    comparison = json.loads(
        (BENCHMARK_RESULTS / "closed_loop_v1_result.json").read_text(
            encoding="utf-8"
        )
    )
    readiness = json.loads(
        (BENCHMARK_RESULTS / "training_readiness_v1_result.json").read_text(
            encoding="utf-8"
        )
    )
    preflight = json.loads(
        (BENCHMARK_RESULTS / "sky130_opamp_preflight_v1_result.json").read_text(
            encoding="utf-8"
        )
    )

    assert comparison["task_set_sha256"] == metadata["task_set_sha256"]
    assert comparison["protocol_sha256"] == metadata["protocol_sha256"]
    assert comparison["model_controller"]["tasks"] == len(tasks)
    assert comparison["decision"]["autonomous_research"] == "not established"
    assert readiness["task_set_sha256"] == metadata["task_set_sha256"]
    assert readiness["protocol_sha256"] == metadata["protocol_sha256"]
    assert readiness["gates"]["trajectory_pilot"]["passed"] is False
    assert preflight == preflight_sky130_campaign(SKY130_CAMPAIGN)


def test_frozen_benchmark_rejects_empty_or_uncommitted_task_set(tmp_path):
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    payload["tasks"] = []
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        load_benchmark(empty)

    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    payload.pop("task_set_sha256")
    uncommitted = tmp_path / "uncommitted.json"
    uncommitted.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="task_set_sha256"):
        load_benchmark(uncommitted)

    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    payload["promotion_gates"]["all_specs_success_rate"] = 0.0
    changed_protocol = tmp_path / "changed-protocol.json"
    changed_protocol.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol_sha256"):
        load_benchmark(changed_protocol)


def test_tpe_is_deterministic_and_stays_inside_bounds():
    _, tasks = load_benchmark(BENCHMARK)
    template = tasks[0].fixed_template
    assert template is not None
    first = DeterministicTPE(template, seed=7)
    second = DeterministicTPE(template, seed=7)
    completed = []
    suggestions = []
    for index in range(8):
        a = first.suggest(completed)
        b = second.suggest(completed)
        assert a == b
        template.render(a)
        suggestions.append(a)
        completed.append((a, index / 10.0))
    assert len({_serialize(value) for value in suggestions}) == 8


def test_tpe_rejects_invalid_search_contract():
    _, tasks = load_benchmark(BENCHMARK)
    template = tasks[0].fixed_template
    assert template is not None

    with pytest.raises(ValueError, match="startup_trials"):
        DeterministicTPE(template, seed=1, startup_trials=0)
    with pytest.raises(ValueError, match="candidates"):
        DeterministicTPE(template, seed=1, candidates=0)
    with pytest.raises(ValueError, match="elite_fraction"):
        DeterministicTPE(template, seed=1, elite_fraction=1.0)


def _serialize(value: dict[str, float]) -> str:
    return json.dumps(value, sort_keys=True)


class TwoStepController:
    controller_id = "test-controller"

    def __init__(self) -> None:
        self.calls = 0

    def propose(self, task, history):
        self.calls += 1
        return Proposal(
            action="propose",
            hypothesis=f"trial-{self.calls}",
            topology=task.allowed_topologies[0],
            rationale="test",
            netlist="R1 vin vout 1k\n.op\n.end",
        )


def test_research_runner_resumes_without_replaying_completed_episode(tmp_path):
    _, tasks = load_benchmark(BENCHMARK)
    task = tasks[0]
    controller = TwoStepController()
    observations = iter(
        [
            _observation(success=False, reward=0.35),
            _observation(success=True, reward=1.3),
        ]
    )
    runner = ResearchRunner(observe_fn=lambda *args, **kwargs: next(observations))
    log = EpisodeLog(tmp_path / "episode.jsonl")

    result = runner.run(task, controller, log)
    replay = runner.run(task, controller, log)

    assert result["stop_reason"] == "all_specs_met"
    assert result["verifier_calls"] == 2
    assert replay == result
    assert controller.calls == 2
    assert len(log.read()) == 5


def test_research_runner_records_frozen_initial_failure_before_model_revision(
    tmp_path,
):
    _, tasks = load_benchmark(BENCHMARK)
    base = tasks[0]
    task = type(base)(
        **{
            **base.__dict__,
            "initial_netlist": "Rseed vin vout 1k\n.op\n.end",
        }
    )
    histories = []

    class FeedbackController:
        controller_id = "feedback-controller"

        def propose(self, task, history):
            histories.append(history)
            return Proposal(
                action="propose",
                hypothesis="The measured gain is low, so increase feedback ratio.",
                topology=task.allowed_topologies[0],
                rationale="Use the frozen measurement to choose one revision.",
                netlist="R1 vin vout 10k\n.op\n.end",
            )

    observations = iter(
        [
            _observation(success=False, reward=0.35),
            _observation(success=True, reward=1.3),
        ]
    )
    log = EpisodeLog(tmp_path / "feedback.jsonl")
    result = ResearchRunner(
        observe_fn=lambda *args, **kwargs: next(observations)
    ).run(task, FeedbackController(), log)

    events = log.read()
    assert [event["event_type"] for event in events] == [
        "initial_observation",
        "controller_call_started",
        "trial",
        "episode_end",
    ]
    assert histories[0][0]["source"] == "frozen_initial_candidate"
    assert histories[0][0]["measurements"]["gain_db"] == 10.0
    assert events[1]["attempt"] == 0
    assert events[2]["controller_call_event_hash"] == events[1]["event_hash"]
    assert result["verifier_calls"] == 2
    assert result["controller_calls"] == 1


def test_research_runner_applies_hidden_verifier_augmentation(tmp_path):
    _, tasks = load_benchmark(FEEDBACK_BENCHMARK)
    base = tasks[0]
    task = type(base)(
        **{
            **base.__dict__,
            "verifier_augmentation": "Csecret vout 0 37.2p",
        }
    )
    completions = []

    class StopAfterOne:
        controller_id = "test-augmented-controller"

        def propose(self, task, history):
            if history:
                return Proposal("stop", "done", "", "done", "")
            raise AssertionError("seed observation must be present")

    def observe(*args, **kwargs):
        completions.append(args[2])
        return _observation(success=False, reward=0.35)

    ResearchRunner(observe_fn=observe).run(
        task,
        StopAfterOne(),
        EpisodeLog(tmp_path / "augmented.jsonl"),
    )

    assert len(completions) == 1
    assert "Csecret vout 0 37.2p\n.end" in completions[0]


def test_verifier_augmentation_precedes_first_end_to_prevent_truncation_bypass():
    _, tasks = load_benchmark(FEEDBACK_BENCHMARK)
    task = tasks[0]
    completion = verifier_completion(
        task,
        task.initial_netlist.rstrip() + "\nRignored vout 0 1k\n.end",
    )

    lines = completion.splitlines()
    assert next(i for i, line in enumerate(lines) if line.startswith("Csecret")) < next(
        i for i, line in enumerate(lines) if line == ".end"
    )


def test_verifier_augmentation_rejects_renamed_or_dummy_internal_anchor():
    _, tasks = load_benchmark(FEEDBACK_BENCHMARK)
    task = next(task for task in tasks if task.circuit_id == "inv-amp")
    bypass = (
        task.initial_netlist.replace("nminus", "fb")
        .replace(".op\n", "Rdummy nminus 0 1\n.op\n")
    )

    completion = verifier_completion(task, bypass)

    assert "verifier_anchor_rejected" in completion
    assert task.verifier_augmentation not in completion


def test_verifier_augmentation_rejects_reserved_device_name_collision():
    _, tasks = load_benchmark(FEEDBACK_BENCHMARK)
    task = tasks[0]
    collision = task.initial_netlist.replace(
        ".op\n",
        "Csecret vout 0 1p\n.op\n",
    )

    completion = verifier_completion(task, collision)

    assert "verifier_anchor_rejected" in completion


def test_feedback_comparison_measures_full_feedback_against_blind_control(tmp_path):
    task_ids = ("a", "b", "c", "d")
    full_logs = {}
    blind_logs = {}
    for task_id, full_success, blind_success in zip(
        task_ids,
        (True, True, True, False),
        (True, False, False, False),
        strict=True,
    ):
        full_path = tmp_path / f"full-{task_id}.jsonl"
        blind_path = tmp_path / f"blind-{task_id}.jsonl"
        _completed_log(
            full_path,
            task_id=task_id,
            success=full_success,
            verifier_calls=2,
            initial_failure=True,
            controller_id="codex-cli:test-model:full-protocol",
        )
        _completed_log(
            blind_path,
            task_id=task_id,
            success=blind_success,
            verifier_calls=2,
            initial_failure=True,
            controller_id="codex-cli-feedback-blind:test-model:blind-protocol",
        )
        full_logs[task_id] = full_path
        blind_logs[task_id] = blind_path

    result = summarize_feedback_comparison(
        full_logs,
        blind_logs,
        bootstrap_samples=1_000,
        task_families={
            "a": "family-1",
            "b": "family-1",
            "c": "family-2",
            "d": "family-2",
        },
    )

    assert result["schema"] == "phychip-feedback-use-comparison-v1"
    assert result["full_feedback"]["successes"] == 3
    assert result["feedback_blind"]["successes"] == 1
    assert result["paired_success_delta"]["mean"] == 0.5
    assert result["paired_exact_mcnemar"] == {
        "full_only_successes": 2,
        "blind_only_successes": 0,
        "discordant_pairs": 2,
        "two_sided_p": 0.5,
    }
    assert result["family_clustered_success_delta"]["families"] == 2
    assert result["family_clustered_success_delta"]["mean"] == 0.5
    assert result["family_exact_sign_test"] == {
        "positive_families": 2,
        "negative_families": 0,
        "tied_families": 0,
        "two_sided_p": 0.5,
    }
    assert result["controller_error_sensitivity"]["excluded_pairs"] == []
    assert result["controller_error_sensitivity"]["paired_tasks"] == 4


def test_feedback_comparison_reports_complete_case_controller_error_sensitivity(tmp_path):
    full_logs = {}
    blind_logs = {}
    for task_id, full_success, blind_success, blind_stop in (
        ("a", True, False, "controller_error"),
        ("b", True, False, "budget_exhausted"),
        ("c", True, True, "all_specs_met"),
    ):
        full_path = tmp_path / f"full-{task_id}.jsonl"
        blind_path = tmp_path / f"blind-{task_id}.jsonl"
        _completed_log(
            full_path,
            task_id=task_id,
            success=full_success,
            verifier_calls=2,
            initial_failure=True,
            controller_id="codex-cli:test-model:full-protocol",
        )
        _completed_log(
            blind_path,
            task_id=task_id,
            success=blind_success,
            verifier_calls=2,
            initial_failure=True,
            controller_id="codex-cli-feedback-blind:test-model:blind-protocol",
            stop_reason=blind_stop,
        )
        full_logs[task_id] = full_path
        blind_logs[task_id] = blind_path

    result = summarize_feedback_comparison(full_logs, blind_logs)

    assert result["feedback_blind"]["stop_reasons"]["a"] == "controller_error"
    assert result["controller_error_sensitivity"] == {
        "excluded_pairs": ["a"],
        "paired_tasks": 2,
        "full_feedback_successes": 2,
        "feedback_blind_successes": 1,
        "paired_exact_mcnemar": {
            "full_only_successes": 1,
            "blind_only_successes": 0,
            "discordant_pairs": 1,
            "two_sided_p": 1.0,
        },
    }


def test_feedback_comparison_rejects_unpaired_controller_provenance(tmp_path):
    full_path = tmp_path / "full.jsonl"
    blind_path = tmp_path / "blind.jsonl"
    for path in (full_path, blind_path):
        _completed_log(
            path,
            task_id="a",
            success=False,
            verifier_calls=2,
            initial_failure=True,
            controller_id="codex-cli:test-model:full-protocol",
        )

    with pytest.raises(ValueError, match="feedback-blind controller"):
        summarize_feedback_comparison({"a": full_path}, {"a": blind_path})


def test_research_runner_holds_an_episode_lifetime_lease(tmp_path):
    _, tasks = load_benchmark(BENCHMARK)
    task = tasks[0]
    entered = threading.Event()
    release = threading.Event()

    class BlockingController:
        controller_id = "blocking-controller"

        def __init__(self):
            self.calls = 0
            self.lock = threading.Lock()

        def propose(self, task, history):
            with self.lock:
                self.calls += 1
            entered.set()
            assert release.wait(timeout=2)
            return Proposal(
                action="propose",
                hypothesis="one writer",
                topology=task.allowed_topologies[0],
                rationale="test",
                netlist="R1 vin vout 1k\n.op\n.end",
            )

    controller = BlockingController()
    runner = ResearchRunner(
        observe_fn=lambda *args, **kwargs: _observation(
            success=True,
            reward=1.3,
        )
    )
    log = EpisodeLog(tmp_path / "episode.jsonl")
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(runner.run, task, controller, log)
        assert entered.wait(timeout=2)
        second = executor.submit(runner.run, task, controller, log)
        release.set()
        first_result = first.result(timeout=2)
        second_result = second.result(timeout=2)

    assert controller.calls == 1
    assert first_result == second_result
    assert len(log.read()) == 3


def test_episode_log_rejects_tampering(tmp_path):
    log = EpisodeLog(tmp_path / "episode.jsonl")
    log.append({"event_type": "probe", "episode_id": "x"})
    path = tmp_path / "episode.jsonl"
    event = json.loads(path.read_text(encoding="utf-8"))
    event["episode_id"] = "tampered"
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        log.read()


def test_episode_log_serializes_concurrent_writers(tmp_path):
    log = EpisodeLog(tmp_path / "episode.jsonl")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda index: log.append(
                    {
                        "event_type": "probe",
                        "episode_id": "x",
                        "index": index,
                    }
                ),
                range(32),
            )
        )

    events = log.read()
    assert len(events) == 32
    assert {event["index"] for event in events} == set(range(32))


def test_episode_log_rejects_reserved_fields_and_symlinks(tmp_path):
    log = EpisodeLog(tmp_path / "episode.jsonl")
    with pytest.raises(ValueError, match="reserved"):
        log.append({"event_type": "probe", "event_hash": "x"})

    target = tmp_path / "target.jsonl"
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    with pytest.raises(OSError):
        EpisodeLog(link).append({"event_type": "probe", "episode_id": "x"})
    assert not target.exists()


def test_codex_schema_requires_structured_safe_action():
    schema = CodexCLIController._schema()

    assert schema["additionalProperties"] is False
    assert schema["properties"]["action"]["enum"] == ["propose", "stop"]
    assert "netlist" in schema["required"]
    assert schema["properties"]["netlist"]["maxLength"] == 20_000


def test_codex_prompt_cannot_see_fixed_baseline_template():
    _, tasks = load_benchmark(BENCHMARK)
    task = tasks[0]
    assert task.fixed_template is not None

    prompt = CodexCLIController._prompt(task, [])

    assert task.fixed_template.netlist_template not in prompt
    assert '"fixed_template"' not in prompt
    assert '"parameters"' not in prompt
    assert '"seed"' not in prompt


def test_feedback_blind_controller_removes_diagnostic_results_only():
    _, tasks = load_benchmark(BENCHMARK)
    task = tasks[0]
    history = [
        {
            "source": "frozen_initial_candidate",
            "attempt": -1,
            "hypothesis": "frozen benchmark starting point",
            "topology": task.allowed_topologies[0],
            "netlist": "R1 vin vout 1k\n.op\n.end",
            "measurements": {"gain_db": 7.0},
            "requested_bounds": {"gain_db": [19.0, 21.0]},
            "failure": {
                "stage": "simulation",
                "code": "simulation_failed",
                "detail": "secret device Csecret=47n failed",
            },
            "score": {"reward": 0.35, "all_specs_met": False},
        }
    ]

    prompt = FeedbackBlindCodexCLIController._prompt(task, history)

    assert "R1 vin vout 1k" in prompt
    assert '"feedback_packet":{"status":"redacted"}' in prompt
    assert '"measurements"' not in prompt
    assert '"requested_bounds"' not in prompt
    assert '"failure"' not in prompt
    assert '"score"' not in prompt
    assert "Csecret" not in prompt
    assert (
        FeedbackBlindCodexCLIController(model="test").controller_id
        != CodexCLIController(model="test").controller_id
    )
    assert '"remaining_codex_calls":8' in prompt


def test_full_feedback_prompt_redacts_untrusted_failure_detail():
    _, tasks = load_benchmark(BENCHMARK)
    history = [
        {
            "source": "controller_proposal",
            "attempt": 0,
            "hypothesis": "measure",
            "topology": tasks[0].allowed_topologies[0],
            "netlist": "R1 vin vout 1k\n.op\n.end",
            "measurements": {},
            "requested_bounds": {},
            "failure": {
                "stage": "simulation",
                "code": "simulation_failed",
                "detail": "secret Csecret value 47n",
            },
            "score": {"reward": 0.0, "all_specs_met": False},
        }
    ]

    prompt = CodexCLIController._prompt(tasks[0], history)

    assert '"stage":"simulation"' in prompt
    assert '"code":"simulation_failed"' in prompt
    assert '"detail"' not in prompt
    assert "Csecret" not in prompt


def test_paired_feedback_schedule_is_seeded_balanced_and_adjacent():
    _, tasks = load_benchmark(FEEDBACK_BENCHMARK)

    first = paired_feedback_schedule(tasks, seed=20260801)
    second = paired_feedback_schedule(tasks, seed=20260801)

    assert first == second
    assert {task.task_id for task, _ in first} == {task.task_id for task in tasks}
    assert [arm for _, arm in first].count("full_feedback") == len(tasks) // 2
    assert [arm for _, arm in first].count("feedback_blind") == len(tasks) // 2


def test_paired_feedback_batch_runs_each_task_in_both_arms(tmp_path):
    class OneShotController:
        def __init__(self, controller_id):
            self.controller_id = controller_id

        def propose(self, task, history):
            return Proposal(
                action="propose",
                hypothesis="paired test",
                topology=task.allowed_topologies[0],
                rationale="paired test",
                netlist="Rwinner vin vout 1k\n.op\n.end",
            )

    def observe(*args, **kwargs):
        return _observation(
            success="Rwinner" in args[2],
            reward=1.3 if "Rwinner" in args[2] else 0.35,
        )

    result = run_paired_feedback_benchmark(
        FEEDBACK_BENCHMARK,
        tmp_path,
        schedule_seed=17,
        full_controller=OneShotController("codex-cli:test-model:full"),
        blind_controller=OneShotController(
            "codex-cli-feedback-blind:test-model:blind"
        ),
        observe_fn=observe,
        runtime_fingerprint_value={"verifier_hash": "a" * 64},
        codex_cli_version="codex-test",
    )

    assert result["schema"] == "phychip-paired-feedback-batch-v1"
    assert result["tasks"] == 24
    assert len(result["schedule"]) == 24
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert len(manifest["runner_source_sha256"]) == 64
    assert manifest["controller_timeout_s"] == 300.0
    assert manifest["max_workers"] == 1
    assert all((tmp_path / "full" / f"{task_id}.jsonl").is_file() for task_id in result["results"])
    assert all((tmp_path / "blind" / f"{task_id}.jsonl").is_file() for task_id in result["results"])


def test_seed_observation_does_not_consume_controller_call_budget():
    _, tasks = load_benchmark(FEEDBACK_BENCHMARK)
    history = [
        {
            "source": "frozen_initial_candidate",
            "attempt": -1,
            "netlist": tasks[0].initial_netlist,
        }
    ]

    prompt = CodexCLIController._prompt(tasks[0], history)

    assert '"remaining_codex_calls":7' in prompt


def test_orphaned_started_controller_call_is_not_redrawn_on_resume(tmp_path):
    _, tasks = load_benchmark(FEEDBACK_BENCHMARK)
    task = tasks[0]
    log = EpisodeLog(tmp_path / "orphaned.jsonl")

    class CrashAfterStart:
        controller_id = "codex-cli:test-model:write-ahead"

        def propose(self, task, history):
            raise KeyboardInterrupt("simulated hard interruption")

    with pytest.raises(KeyboardInterrupt):
        ResearchRunner(observe_fn=lambda *args, **kwargs: _observation(
            success=False,
            reward=0.35,
        )).run(task, CrashAfterStart(), log)

    class MustNotRedraw:
        controller_id = CrashAfterStart.controller_id

        def propose(self, task, history):
            raise AssertionError("orphaned draw must not be repeated")

    result = ResearchRunner(
        observe_fn=lambda *args, **kwargs: _observation(
            success=False,
            reward=0.35,
        )
    ).run(task, MustNotRedraw(), log)

    assert result["stop_reason"] == "controller_error"
    assert result["detail"] == "orphaned controller call; redraw prohibited"
    assert result["codex_calls"] == 1


def test_codex_controller_isolated_command_and_structured_output(
    tmp_path,
    monkeypatch,
):
    _, tasks = load_benchmark(BENCHMARK)
    task = tasks[0]
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex-home"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-be-inherited")

    def fake_run(command, **kwargs):
        assert "--ignore-user-config" in command
        assert command.count("--disable") >= 8
        assert "shell_tool" in command
        assert kwargs["stdout"] is not None
        assert kwargs["stderr"] is not None
        assert "UNRELATED_SECRET" not in kwargs["env"]
        assert kwargs["env"]["HOME"] != str(Path.home())
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text(
            json.dumps(
                {
                    "action": "propose",
                    "hypothesis": "Raising gm should increase bandwidth.",
                    "topology": task.allowed_topologies[0],
                    "rationale": "One controlled sizing change.",
                    "netlist": "R1 vin vout 1k\n.op\n.end",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("phy_chip_env.research.subprocess.run", fake_run)
    proposal = CodexCLIController().propose(task, [])

    assert proposal.action == "propose"
    assert proposal.topology == task.allowed_topologies[0]


@pytest.mark.parametrize(
    ("returncode", "payload", "message"),
    [
        (7, None, "exit 7"),
        (
            0,
            {
                "action": "propose",
                "hypothesis": "x",
                "topology": "not-allowed",
                "rationale": "x",
                "netlist": "R1 vin vout 1k\n.op\n.end",
            },
            "allowed",
        ),
        (
            0,
            {
                "action": "stop",
                "hypothesis": "x",
                "topology": "",
                "rationale": "x",
                "netlist": "not empty",
            },
            "non-empty",
        ),
    ],
)
def test_codex_controller_fails_closed(
    tmp_path,
    monkeypatch,
    returncode,
    payload,
    message,
):
    _, tasks = load_benchmark(BENCHMARK)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex-home"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")

    def fake_run(command, **kwargs):
        if payload is not None:
            output = Path(command[command.index("--output-last-message") + 1])
            output.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(
            returncode=returncode,
            stderr="provider unavailable" if returncode else "",
        )

    monkeypatch.setattr("phy_chip_env.research.subprocess.run", fake_run)
    with pytest.raises((RuntimeError, ValueError), match=message):
        CodexCLIController().propose(tasks[0], [])

    if returncode:
        with pytest.raises(RuntimeError, match="provider unavailable"):
            CodexCLIController().propose(tasks[0], [])


def _completed_log(
    path: Path,
    *,
    task_id: str,
    success: bool,
    verifier_calls: int,
    initial_failure: bool = False,
    controller_id: str = "test-controller",
    stop_reason: str | None = None,
) -> None:
    log = EpisodeLog(path)
    episode_id = f"{task_id}-episode"
    if initial_failure:
        log.append(
            {
                "event_type": "initial_observation",
                "episode_id": episode_id,
                "task_id": task_id,
                "source": "frozen_initial_candidate",
                "topology": "test",
                "netlist": "Rseed vin vout 1k\n.op\n.end",
                "observation": _observation(
                    success=False,
                    reward=0.35,
                ).to_event(),
            }
        )
    winning_hash = None
    model_calls = verifier_calls - int(initial_failure)
    for attempt in range(model_calls):
        trial_success = success and attempt == model_calls - 1
        trial = log.append(
            {
                "event_type": "trial",
                "episode_id": episode_id,
                "task_id": task_id,
                "attempt": attempt,
                "controller_id": controller_id,
                "proposal": {
                    "action": "propose",
                    "hypothesis": "test",
                    "topology": "test",
                    "rationale": "test",
                    "netlist": "R1 vin vout 1k\n.op\n.end",
                },
                "observation": _observation(
                    success=trial_success,
                    reward=1.3 if trial_success else 0.35,
                ).to_event(),
            }
        )
        if trial_success:
            winning_hash = trial["event_hash"]
    outcome: dict[str, object] = {
        "event_type": "episode_end",
        "episode_id": episode_id,
        "task_id": task_id,
        "stop_reason": stop_reason or ("all_specs_met" if success else "budget_exhausted"),
        "verifier_calls": verifier_calls,
        "codex_calls": verifier_calls,
    }
    if success:
        outcome["winning_event_hash"] = winning_hash
    log.append(outcome)


def _completed_model_log(
    path: Path,
    *,
    task_id: str,
    rollout_id: str = "0",
) -> None:
    _, tasks = load_benchmark(BENCHMARK)
    task = next(task for task in tasks if task.task_id == task_id)
    log = EpisodeLog(path)
    controller_id = TEST_CODEX_CONTROLLER_ID
    episode_id = research_episode_id(task, controller_id, rollout_id)
    trial = log.append(
        {
            "event_type": "trial",
            "episode_id": episode_id,
            "rollout_id": rollout_id,
            "task_id": task_id,
            "attempt": 0,
            "controller_id": controller_id,
            "proposal": {
                "action": "propose",
                "hypothesis": "test",
                "topology": "test",
                "rationale": "test",
                "netlist": "R1 vin vout 1k\n.op\n.end",
            },
            "observation": _observation(success=True, reward=1.3).to_event(),
        }
    )
    log.append(
        {
            "event_type": "episode_end",
            "episode_id": episode_id,
            "rollout_id": rollout_id,
            "task_id": task_id,
            "stop_reason": "all_specs_met",
            "winning_event_hash": trial["event_hash"],
            "verifier_calls": 1,
            "codex_calls": 1,
        }
    )


def test_comparison_is_paired_and_reports_call_savings(tmp_path):
    task_ids = ("a", "b", "c", "d")
    model_logs = {}
    baseline_logs = {}
    for task_id, model_calls, baseline_calls in zip(
        task_ids,
        (1, 1, 1, 1),
        (8, 4, 3, 6),
        strict=True,
    ):
        model_path = tmp_path / f"model-{task_id}.jsonl"
        baseline_path = tmp_path / f"baseline-{task_id}.jsonl"
        _completed_log(
            model_path,
            task_id=task_id,
            success=True,
            verifier_calls=model_calls,
        )
        _completed_log(
            baseline_path,
            task_id=task_id,
            success=True,
            verifier_calls=baseline_calls,
        )
        model_logs[task_id] = model_path
        baseline_logs[task_id] = baseline_path

    result = summarize_comparison(
        model_logs,
        baseline_logs,
        bootstrap_samples=1_000,
    )

    assert result["model"]["successes"] == 4
    assert result["fixed_topology_tpe"]["successes"] == 4
    assert result["paired_success_delta"]["mean"] == 0.0
    savings = result["paired_verifier_call_saving_on_joint_successes"]
    assert savings["n"] == 4
    assert savings["mean"] == 4.25
    assert savings["bootstrap_ci95"] == [2.5, 6.0]


def test_training_readiness_fails_closed_without_scale_or_approvals(tmp_path):
    log_path = tmp_path / "model.jsonl"
    _completed_model_log(log_path, task_id="clv1-inv-amp-gain-bw")

    result = assess_training_readiness(
        BENCHMARK,
        [log_path],
        reverify_fn=lambda *args, **kwargs: _observation(
            success=True,
            reward=1.3,
        ),
    )

    assert result["counts"]["episodes"] == 1
    assert result["counts"]["approved_successes"] == 0
    assert result["gates"]["trajectory_pilot"]["passed"] is False
    assert result["gates"]["multi_turn_rlvr"]["passed"] is False
    assert result["gates"]["on_policy_distillation"]["passed"] is False
    audit_reasons = " ".join(result["gates"]["trajectory_audit_pilot"]["reasons"])
    assert "not training-eligible" in audit_reasons
    assert "no failed revision episodes" in audit_reasons


def test_benchmark_bound_validation_rejects_forged_episode_identity(tmp_path):
    _, tasks = load_benchmark(FEEDBACK_BENCHMARK)
    path = tmp_path / "forged.jsonl"
    _completed_log(
        path,
        task_id=tasks[0].task_id,
        success=False,
        verifier_calls=2,
        initial_failure=True,
        controller_id="codex-cli:test-model:protocol",
    )

    with pytest.raises(ValueError, match="episode identity"):
        validated_episode_outcome(
            path,
            task=tasks[0],
            reverify_fn=lambda *args, **kwargs: _observation(
                success=False,
                reward=0.35,
            ),
        )


def test_feedback_directory_comparison_replays_and_keeps_development_gate_closed(
    tmp_path,
    monkeypatch,
):
    _, tasks = load_benchmark(FEEDBACK_BENCHMARK)

    def fake_observe(*args, **kwargs):
        return _observation(
            success="Rwinner" in args[2],
            reward=1.3 if "Rwinner" in args[2] else 0.35,
        )

    class OneShotController:
        def __init__(self, controller_id):
            self.controller_id = controller_id

        def propose(self, task, history):
            return Proposal(
                action="propose",
                hypothesis="test",
                topology=task.allowed_topologies[0],
                rationale="test",
                netlist="Rwinner vin vout 1k\n.op\n.end",
            )

    full = tmp_path / "full"
    blind = tmp_path / "blind"
    for task in tasks:
        ResearchRunner(observe_fn=fake_observe).run(
            task,
            OneShotController("codex-cli:test-model:full-protocol"),
            EpisodeLog(full / f"{task.task_id}.jsonl"),
        )
        ResearchRunner(observe_fn=fake_observe).run(
            task,
            OneShotController(
                "codex-cli-feedback-blind:test-model:blind-protocol"
            ),
            EpisodeLog(blind / f"{task.task_id}.jsonl"),
        )

    monkeypatch.setattr("phy_chip_env.research.observe_completion", fake_observe)
    result = compare_feedback_runs(FEEDBACK_BENCHMARK, full, blind)

    assert result["feedback_advantage_gate"]["passed"] is False
    assert "no hash-bound manual success audit" in " ".join(
        result["feedback_advantage_gate"]["reasons"]
    )

    audit = tmp_path / "incomplete-audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema": "phychip-feedback-manual-audit-v1",
                "task_set_sha256": result["task_set_sha256"],
                "protocol_sha256": result["protocol_sha256"],
                "episode_end_hashes": {
                    "full_feedback": result["full_feedback"][
                        "episode_end_hashes"
                    ],
                    "feedback_blind": result["feedback_blind"][
                        "episode_end_hashes"
                    ],
                },
                "successful_circuits_reviewed": 1,
                "topology_plausible_successes": 1,
                "known_reward_attacks_accepted": 0,
            }
        ),
        encoding="utf-8",
    )
    audited = compare_feedback_runs(
        FEEDBACK_BENCHMARK,
        full,
        blind,
        manual_audit_path=audit,
    )
    assert "does not cover every successful circuit" in " ".join(
        audited["feedback_advantage_gate"]["reasons"]
    )

    monkeypatch.setattr(
        "phy_chip_env.research.observe_completion",
        lambda *args, **kwargs: _observation(success=False, reward=0.35),
    )
    with pytest.raises(ValueError, match="scorer replay"):
        compare_feedback_runs(FEEDBACK_BENCHMARK, full, blind)


def test_seeded_zero_trial_stop_preserves_controller_provenance(tmp_path):
    _, tasks = load_benchmark(FEEDBACK_BENCHMARK)

    class StopController:
        controller_id = "codex-cli:test-model:stop-protocol"

        def propose(self, task, history):
            return Proposal("stop", "done", "", "no safe revision", "")

    def observe(*args, **kwargs):
        return _observation(success=False, reward=0.35)

    path = tmp_path / "stop.jsonl"
    ResearchRunner(observe_fn=observe).run(
        tasks[0],
        StopController(),
        EpisodeLog(path),
    )

    outcome = validated_episode_outcome(path, task=tasks[0], reverify_fn=observe)
    assert outcome["success"] is False
    assert outcome["controller_ids"] == [StopController.controller_id]
    assert outcome["verifier_calls"] == 1


def test_training_readiness_rejects_threshold_downgrades(tmp_path):
    log_path = tmp_path / "model.jsonl"
    _completed_model_log(log_path, task_id="clv1-inv-amp-gain-bw")

    with pytest.raises(ValueError, match="cannot be below 100"):
        assess_training_readiness(
            BENCHMARK,
            [log_path],
            required_episodes=1,
            reverify_fn=lambda *args, **kwargs: _observation(
                success=True,
                reward=1.3,
            ),
        )


def test_training_readiness_rejects_end_only_or_replayed_evidence(tmp_path):
    end_only = tmp_path / "end-only.jsonl"
    EpisodeLog(end_only).append(
        {
            "event_type": "episode_end",
            "episode_id": "fabricated",
            "task_id": "clv1-inv-amp-gain-bw",
            "stop_reason": "all_specs_met",
            "winning_event_hash": "f" * 64,
            "verifier_calls": 0,
            "codex_calls": 0,
        }
    )
    with pytest.raises(ValueError, match="winning event"):
        assess_training_readiness(BENCHMARK, [end_only])

    valid = tmp_path / "valid.jsonl"
    _completed_model_log(valid, task_id="clv1-inv-amp-gain-bw")
    with pytest.raises(
        ValueError,
        match="duplicate episode_end|duplicate episode identity|replayed event",
    ):
        assess_training_readiness(
            BENCHMARK,
            [valid, valid],
            reverify_fn=lambda *args, **kwargs: _observation(
                success=True,
                reward=1.3,
            ),
        )


def test_training_readiness_replays_scorer_and_deduplicates_content(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _completed_model_log(
        first,
        task_id="clv1-inv-amp-gain-bw",
        rollout_id="rollout-a",
    )
    _completed_model_log(
        second,
        task_id="clv1-inv-amp-gain-bw",
        rollout_id="rollout-b",
    )

    with pytest.raises(ValueError, match="scorer replay"):
        assess_training_readiness(
            BENCHMARK,
            [first],
            reverify_fn=lambda *args, **kwargs: _observation(
                success=False,
                reward=0.35,
            ),
        )
    with pytest.raises(ValueError, match="duplicate trajectory content"):
        assess_training_readiness(
            BENCHMARK,
            [first, second],
            reverify_fn=lambda *args, **kwargs: _observation(
                success=True,
                reward=1.3,
            ),
        )


def test_unverified_token_distribution_digest_never_admits_opd(tmp_path):
    log_path = tmp_path / "model.jsonl"
    _completed_model_log(log_path, task_id="clv1-inv-amp-gain-bw")
    outcome = EpisodeLog(log_path).read()[-1]
    approvals = tmp_path / "approvals.json"
    approvals.write_text(
        json.dumps(
            {
                "schema": "phychip-trajectory-approvals-v1",
                "benchmark_task_set_sha256": load_benchmark(BENCHMARK)[0][
                    "task_set_sha256"
                ],
                "benchmark_protocol_sha256": load_benchmark(BENCHMARK)[0][
                    "protocol_sha256"
                ],
                "teacher_token_distributions_preserved": True,
                "teacher_token_distributions_sha256": "0" * 64,
                "approvals": [
                    {
                        "episode_end_hash": outcome["event_hash"],
                        "winning_event_hash": outcome["winning_event_hash"],
                        "task_id": outcome["task_id"],
                        "controller_id": TEST_CODEX_CONTROLLER_ID,
                        "outcome": "success",
                        "outcome_verified": True,
                        "topology_plausible": True,
                        "decision": "approve",
                        "reviewer": "human",
                        "reviewed_at": "2026-07-29T12:00:00-07:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = assess_training_readiness(
        BENCHMARK,
        [log_path],
        approval_manifest=approvals,
        reverify_fn=lambda *args, **kwargs: _observation(
            success=True,
            reward=1.3,
        ),
    )

    assert result["counts"]["approved_episodes"] == 1
    assert result["gates"]["on_policy_distillation"]["passed"] is False
    assert "no token-distribution manifest" in " ".join(
        result["gates"]["on_policy_distillation"]["reasons"]
    )

    shard = tmp_path / "tokens.jsonl"
    shard.write_text("arbitrary bytes\n", encoding="utf-8")
    token_manifest = tmp_path / "token-manifest.json"
    token_manifest.write_text(
        json.dumps(
            {
                "schema": "phychip-token-distributions-v1",
                "benchmark_task_set_sha256": load_benchmark(BENCHMARK)[0][
                    "task_set_sha256"
                ],
                "benchmark_protocol_sha256": load_benchmark(BENCHMARK)[0][
                    "protocol_sha256"
                ],
                "controller_id": TEST_CODEX_CONTROLLER_ID,
                "vocabulary_sha256": "1" * 64,
                "capture_revision_sha256": "2" * 64,
                "episode_end_hashes": [outcome["event_hash"]],
                "shards": [
                    {
                        "path": shard.name,
                        "sha256": _sha256(shard),
                        "examples": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    approval_payload = json.loads(approvals.read_text(encoding="utf-8"))
    approval_payload["teacher_token_distributions_sha256"] = _sha256(token_manifest)
    approvals.write_text(json.dumps(approval_payload), encoding="utf-8")

    malformed = assess_training_readiness(
        BENCHMARK,
        [log_path],
        approval_manifest=approvals,
        teacher_token_distributions_artifact=token_manifest,
        reverify_fn=lambda *args, **kwargs: _observation(
            success=True,
            reward=1.3,
        ),
    )
    assert "not valid JSONL" in " ".join(
        malformed["gates"]["on_policy_distillation"]["reasons"]
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ready_campaign(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    pdk_root = tmp_path / "pdk"
    model = pdk_root / "models" / "sky130.lib.spice"
    model.parent.mkdir(parents=True)
    model.write_text("* pinned model\n", encoding="utf-8")
    sealed = tmp_path / "sealed.json"
    sealed_tasks = [
        {
            "task_id": "held-out-tt",
            "split": "held_out",
            "target_specs": {"open_loop_gain_db": [60.0, 70.0]},
            "pvt": {"corner": "tt", "vdd_v": 1.8, "temperature_c": 25},
        }
    ]
    sealed.write_text(
        json.dumps(
            {
                "schema": "phychip-sealed-sky130-eval-v1",
                "campaign_id": "test-campaign",
                "pdk_source_revision": "a" * 40,
                "task_set_sha256": hashlib.sha256(
                    json.dumps(
                        sealed_tasks,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
                "tasks": sealed_tasks,
            }
        ),
        encoding="utf-8",
    )
    release = tmp_path / "release-evidence.json"
    release.write_text(
        json.dumps(
            {
                "schema": "phychip-v0.1-evidence-manifest-v2",
                "release_status": "built_from_clean_source_commit",
                "source_git": {"working_tree_dirty": False},
            }
        ),
        encoding="utf-8",
    )
    m1 = tmp_path / "m1-evidence.json"
    m1.write_text(
        json.dumps(
            {
                "schema": "phychip-design-agent-m1-v1",
                "campaign_id": "test-campaign",
                "decision": "pass",
                "pdk_source_revision": "a" * 40,
                "sealed_eval_manifest_sha256": _sha256(sealed),
            }
        ),
        encoding="utf-8",
    )
    topologies = []
    for index, topology_id in enumerate(("telescopic", "folded", "two-stage")):
        template = tmp_path / f"{topology_id}.cir"
        template.write_text(
            f"""* template {index}
.param width=$w length=$l
VDD vdd vss 1.8
M1 vout vinp vss vss nmos w={{width}} l={{length}}
M2 vout vinn vss vss nmos w={{width}} l={{length}}
M3 vout vinp vdd vdd pmos w={{width}} l={{length}}
M4 vout vinn vdd vdd pmos w={{width}} l={{length}}
.op
.end
""",
            encoding="utf-8",
        )
        topologies.append(
            {
                "topology_id": topology_id,
                "template_status": "verified",
                "template_path": template.name,
                "template_sha256": _sha256(template),
                "continuous_parameters": ["w", "l"],
            }
        )
    campaign = {
        "schema": SKY130_CAMPAIGN_SCHEMA,
        "campaign_id": "test-campaign",
        "prerequisites": {
            "v0_1_release_locally_validated": True,
            "design_agent_clears_m1": True,
            "sealed_eval_manifest_frozen": True,
        },
        "prerequisite_evidence_sha256": {
            "v0_1_release": _sha256(release),
            "design_agent_m1": _sha256(m1),
        },
        "sealed_eval_manifest_sha256": _sha256(sealed),
        "pdk": {
            "source_revision": "a" * 40,
            "required_models": [
                {
                    "path": "models/sky130.lib.spice",
                    "sha256": _sha256(model),
                }
            ],
        },
        "constraints": {
            "vdd_nominal_v": 1.8,
            "quiescent_current_max_a": 0.0001,
        },
        "metrics": ["gain_db"],
        "discovery_grid": {
            "load_capacitance_f": [1e-12],
            "minimum_open_loop_gain_db": [60.0],
            "minimum_unity_gain_bandwidth_hz": [1e6],
            "minimum_phase_margin_deg": 60.0,
        },
        "pvt_replay_grid": {
            "corners": ["tt", "ss", "ff", "sf", "fs"],
            "vdd_v": [1.62, 1.8, 1.98],
            "temperature_c": [-40, 25, 125],
        },
        "topologies": topologies,
        "budgets": {
            "fixed_optimizer_calls_per_region": 4,
            "model_guided_calls_per_region": 4,
            "pvt_replays_per_finalist": 1,
        },
        "arms": [
            "fixed_topology_deterministic_sweep_tpe",
            "model_guided_topology_and_experiment_selection",
        ],
        "promotion_gates": {
            "all_designs_replay_under_pinned_environment_and_pdk": True,
            "known_reward_attacks_accepted": 0,
            "paired_success_or_hypervolume_ci95_lower_bound_gt": 0.0,
            "held_out_topology_rule_beats_majority_baseline": True,
            "manual_topology_plausibility_required": True,
        },
    }
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    return campaign_path, pdk_root, sealed, release, m1


def test_sky130_campaign_positive_preflight_and_symlink_escape(tmp_path):
    campaign, pdk_root, sealed, release, m1 = _ready_campaign(tmp_path)

    result = preflight_sky130_campaign(
        campaign,
        pdk_root=pdk_root,
        sealed_eval_manifest=sealed,
        release_evidence=release,
        design_m1_evidence=m1,
    )
    assert result["ready"] is True
    assert len(result["resolved_models"]) == 1
    assert len(result["resolved_templates"]) == 3
    assert (
        main(
            [
                "sky130-preflight",
                "--campaign",
                str(campaign),
                "--pdk-root",
                str(pdk_root),
                "--sealed-eval-manifest",
                str(sealed),
                "--release-evidence",
                str(release),
                "--design-m1-evidence",
                str(m1),
            ]
        )
        == 0
    )

    outside = tmp_path / "outside.spice"
    outside.write_text("* outside\n", encoding="utf-8")
    model = pdk_root / "models" / "sky130.lib.spice"
    model.unlink()
    model.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        preflight_sky130_campaign(
            campaign,
            pdk_root=pdk_root,
            sealed_eval_manifest=sealed,
            release_evidence=release,
            design_m1_evidence=m1,
        )


def test_sky130_campaign_rejects_unfrozen_values_and_false_end_card(tmp_path):
    campaign, pdk_root, sealed, release, m1 = _ready_campaign(tmp_path / "protocol")
    payload = json.loads(campaign.read_text(encoding="utf-8"))
    payload["promotion_gates"]["known_reward_attacks_accepted"] = 1
    campaign.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen protocol"):
        load_sky130_campaign(campaign)

    campaign, pdk_root, sealed, release, m1 = _ready_campaign(tmp_path / "template")
    payload = json.loads(campaign.read_text(encoding="utf-8"))
    template = campaign.parent / payload["topologies"][0]["template_path"]
    template.write_text(
        template.read_text(encoding="utf-8").replace(".end\n", ".ends\n"),
        encoding="utf-8",
    )
    payload["topologies"][0]["template_sha256"] = _sha256(template)
    campaign.write_text(json.dumps(payload), encoding="utf-8")
    result = preflight_sky130_campaign(
        campaign,
        pdk_root=pdk_root,
        sealed_eval_manifest=sealed,
        release_evidence=release,
        design_m1_evidence=m1,
    )
    assert result["ready"] is False
    assert "unverified topology templates: telescopic" in result["blockers"]

    campaign, pdk_root, sealed, release, m1 = _ready_campaign(tmp_path / "sealed")
    sealed_payload = json.loads(sealed.read_text(encoding="utf-8"))
    sealed_payload["tasks"][0]["pvt"]["vdd_v"] = float("nan")
    sealed_payload["task_set_sha256"] = "0" * 64
    sealed.write_text(json.dumps(sealed_payload), encoding="utf-8")
    payload = json.loads(campaign.read_text(encoding="utf-8"))
    payload["sealed_eval_manifest_sha256"] = _sha256(sealed)
    payload["prerequisite_evidence_sha256"]["design_agent_m1"] = _sha256(m1)
    campaign.write_text(json.dumps(payload), encoding="utf-8")
    result = preflight_sky130_campaign(
        campaign,
        pdk_root=pdk_root,
        sealed_eval_manifest=sealed,
        release_evidence=release,
        design_m1_evidence=m1,
    )
    assert result["ready"] is False
    assert "sealed evaluation manifest is empty or contract-invalid" in result["blockers"]


def test_sky130_campaign_refuses_missing_pdk_and_unmet_prerequisites():
    result = preflight_sky130_campaign(SKY130_CAMPAIGN)

    assert result["ready"] is False
    assert result["topology_ids"] == [
        "sky130-telescopic-cascode",
        "sky130-folded-cascode",
        "sky130-two-stage-miller",
    ]
    assert "no local PDK root was supplied" in result["blockers"]
    assert any("source_revision" in blocker for blocker in result["blockers"])


def test_cli_gate_and_empty_batch_exit_codes(tmp_path, monkeypatch, capsys):
    log_path = tmp_path / "model.jsonl"
    _completed_model_log(log_path, task_id="clv1-inv-amp-gain-bw")
    monkeypatch.setattr(
        "phy_chip_env.readiness.observe_completion",
        lambda *args, **kwargs: _observation(success=True, reward=1.3),
    )
    readiness_exit = main(
        [
            "training-readiness",
            "--benchmark",
            str(BENCHMARK),
            "--episode-log",
            str(log_path),
            "--require-gate",
            "trajectory_pilot",
        ]
    )
    assert readiness_exit == 1
    assert '"passed": false' in capsys.readouterr().out

    monkeypatch.setattr(
        "phy_chip_env.cli.run_benchmark",
        lambda *args, **kwargs: {"tasks": 0, "successes": 0},
    )
    empty_exit = main(
        [
            "research-batch",
            "--benchmark",
            str(BENCHMARK),
            "--controller",
            "fixed",
            "--output-directory",
            str(tmp_path / "out"),
        ]
    )
    assert empty_exit == 1

    monkeypatch.setattr(
        "phy_chip_env.cli.run_benchmark",
        lambda *args, **kwargs: {
            "tasks": 2,
            "successes": 1,
            "results": {
                "success": {"stop_reason": "all_specs_met"},
                "scientific_failure": {"stop_reason": "budget_exhausted"},
            },
        },
    )
    partial_exit = main(
        [
            "research-batch",
            "--benchmark",
            str(BENCHMARK),
            "--controller",
            "fixed",
            "--output-directory",
            str(tmp_path / "partial"),
        ]
    )
    assert partial_exit == 0


def test_batch_runner_and_directory_comparison_are_resumable(tmp_path):
    baseline = tmp_path / "baseline"
    result = run_benchmark(
        BENCHMARK,
        baseline,
        controller="fixed",
        observe_fn=lambda *args, **kwargs: _observation(
            success=True,
            reward=1.3,
        ),
    )
    replay = run_benchmark(
        BENCHMARK,
        baseline,
        controller="fixed",
        observe_fn=lambda *args, **kwargs: pytest.fail("resume replayed a verifier call"),
    )
    comparison = compare_benchmark_runs(BENCHMARK, baseline, baseline)

    assert result == replay
    assert result["successes"] == result["tasks"] == 4
    assert comparison["paired_success_delta"]["mean"] == 0.0
    assert (
        comparison["paired_verifier_call_saving_on_joint_successes"]["mean"]
        == 0.0
    )


@pytest.mark.parametrize(
    ("controller", "expected_id"),
    (
        ("random", "fixed-random"),
        ("lhs", "fixed-lhs"),
        ("tpe", "fixed-topology-tpe"),
        ("fixed", "fixed-topology-tpe"),
    ),
)
def test_batch_runner_exposes_matched_parameter_search_arms(
    tmp_path,
    controller,
    expected_id,
):
    result = run_benchmark(
        BENCHMARK,
        tmp_path / controller,
        controller=controller,
        observe_fn=lambda *args, **kwargs: _observation(
            success=True,
            reward=1.3,
        ),
    )

    assert result["controller"] == expected_id
    assert result["successes"] == result["tasks"] == 4
    assert all(
        outcome["controller_id"].startswith(
            f"fixed-{'tpe' if controller == 'fixed' else controller}:"
        )
        for outcome in result["results"].values()
    )


@pytest.mark.parametrize("controller", ("random", "lhs", "tpe"))
def test_cli_accepts_parameter_search_controller_names(
    monkeypatch,
    tmp_path,
    controller,
):
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return {"tasks": 1, "successes": 1, "results": {}}

    monkeypatch.setattr("phy_chip_env.cli.run_benchmark", fake_run)

    result = main(
        [
            "research-batch",
            "--benchmark",
            str(BENCHMARK),
            "--controller",
            controller,
            "--output-directory",
            str(tmp_path / controller),
        ]
    )

    assert result == 0
    assert captured["controller"] == controller


@pytest.mark.parametrize("strategy", ("random", "lhs", "tpe"))
def test_parameter_search_exhaustion_and_resume_preserve_exact_sequence(
    tmp_path,
    strategy,
):
    _, tasks = load_benchmark(BENCHMARK)
    task = tasks[0]
    path = tmp_path / f"{strategy}.jsonl"
    calls = 0

    def always_fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _observation(success=False, reward=0.2)

    result = run_fixed_topology(
        task,
        EpisodeLog(path),
        strategy=strategy,
        observe_fn=always_fail,
    )
    events = EpisodeLog(path).read()
    parameters = [event["parameters"] for event in events if event["event_type"] == "trial"]
    replay = run_fixed_topology(
        task,
        EpisodeLog(path),
        strategy=strategy,
        observe_fn=lambda *args, **kwargs: pytest.fail("resume spent another call"),
    )

    assert result == replay
    assert result["stop_reason"] == "budget_exhausted"
    assert result["verifier_calls"] == task.max_verifier_calls == calls
    assert len(parameters) == task.max_verifier_calls
    assert result["controller_id"].startswith(f"fixed-{strategy}:")


@pytest.mark.parametrize("strategy", ("random", "lhs", "tpe"))
def test_parameter_search_protocol_identity_binds_seed_and_template(strategy):
    _, tasks = load_benchmark(BENCHMARK)
    first = tasks[0]
    changed_seed = replace(first, seed=first.seed + 1)
    other_template = tasks[1]

    assert fixed_search_controller_id(first, strategy) != fixed_search_controller_id(
        changed_seed,
        strategy,
    )
    assert fixed_search_controller_id(first, strategy) != fixed_search_controller_id(
        other_template,
        strategy,
    )


def test_parameter_search_development_result_binds_source_runtime_and_protocol():
    result = json.loads(PARAMETER_SEARCH_RESULT.read_text(encoding="utf-8"))
    recorded_runtime = json.loads(
        RELEASE_RUNTIME_FINGERPRINT.read_text(encoding="utf-8")
    )
    metadata, tasks = load_benchmark(BENCHMARK)
    task_by_id = {task.task_id: task for task in tasks}
    source_root = Path(__file__).parents[2] / "src" / "phy_chip_env"
    current_runtime = runtime_fingerprint()

    assert result["status"] == "development-only-not-sealed-not-a-generalization-result"
    assert result["task_set_sha256"] == metadata["task_set_sha256"]
    assert result["protocol_sha256"] == metadata["protocol_sha256"]
    assert result["research_source_sha256"] == hashlib.sha256(
        (source_root / "research.py").read_bytes()
    ).hexdigest()
    assert result["cli_source_sha256"] == hashlib.sha256(
        (source_root / "cli.py").read_bytes()
    ).hexdigest()
    # Bind the frozen result to the exact runtime that produced it. Separately
    # require current source parity without requiring every public checkout to
    # have the same package-install mode, ngspice path, or binary build.
    assert result["verifier_hash"] == recorded_runtime["verifier_hash"]
    assert (
        result["ngspice"]["binary_sha256"]
        == recorded_runtime["ngspice"]["sha256"]
    )
    assert (
        current_runtime["scorer_semantics"]
        == recorded_runtime["scorer_semantics"]
    )
    assert current_runtime["sources"] == recorded_runtime["sources"]
    assert "ngspice-46" in current_runtime["ngspice"]["version_output"]
    assert result["raw_episode_logs_distributed"] is False

    for arm_name, strategy in (
        ("random", "random"),
        ("latin_hypercube", "lhs"),
        ("tpe", "tpe"),
    ):
        arm = result["arms"][arm_name]
        assert arm["successes"] == sum(
            outcome["success"] for outcome in arm["outcomes"].values()
        )
        assert arm["simulator_calls"] == sum(
            outcome["calls"] for outcome in arm["outcomes"].values()
        )
        for task_id, outcome in arm["outcomes"].items():
            assert outcome["controller_id"] == fixed_search_controller_id(
                task_by_id[task_id],
                strategy,
            )
