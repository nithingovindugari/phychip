from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from athma_train.grpo_contract import (
    SYS_PROMPT,
    configure_chat_termination,
    render_grpo_prompt,
)
from athma_train.experiment_provenance import (
    build_experiment_provenance,
    canonical_ngspice_runtime_identity,
)
from athma_train.grpo_stats import (
    load_policy_results,
    paired_cluster_verdict,
    spec_identity,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "athma-train" / "scripts"


def _load_script(name: str):
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / f"{name}.py"
    if not path.is_file():
        pytest.skip(f"{name}.py is not distributed in the public release tree")
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def __init__(self):
        self._eos_token = "<|end_of_text|>"
        self.eos_token_id = 9
        self.pad_token_id = None
        self.calls = []

    @property
    def eos_token(self):
        return self._eos_token

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value
        self.eos_token_id = self.convert_tokens_to_ids(value)

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 7, "<|end_of_text|>": 9}.get(token, -1)

    def convert_ids_to_tokens(self, token_id):
        return {7: "<|im_end|>", 9: "<|end_of_text|>"}.get(token_id, "<unk>")

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return json.dumps(
            {"messages": messages, "kwargs": kwargs},
            sort_keys=True,
            separators=(",", ":"),
        )


class FakeModel:
    class GenerationConfig:
        eos_token_id = None
        pad_token_id = None

    generation_config = GenerationConfig()


class CharacterTokenizer:
    """Minimal reversible tokenizer for policy-loss-mask unit tests."""

    @staticmethod
    def decode(token_ids, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in token_ids)

    @staticmethod
    def __call__(text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": [ord(character) for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


class CharacterTokenizerWithTermination(CharacterTokenizer):
    eos_token_id = 999_999

    @classmethod
    def decode(cls, token_ids, skip_special_tokens=False):
        return "".join(
            "<|im_end|>" if token_id == cls.eos_token_id else chr(token_id)
            for token_id in token_ids
        )

    @classmethod
    def __call__(cls, text, add_special_tokens=False, return_offsets_mapping=False):
        suffix = "<|im_end|>"
        body = text[: -len(suffix)] if text.endswith(suffix) else text
        input_ids = [ord(character) for character in body]
        offsets = [(index, index + 1) for index in range(len(body))]
        if text.endswith(suffix):
            input_ids.append(cls.eos_token_id)
            offsets.append((len(body), len(text)))
        result = {"input_ids": input_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    @classmethod
    def convert_tokens_to_ids(cls, token):
        return cls.eos_token_id if token == "<|im_end|>" else -1


class NonCanonicalMergeTokenizer(CharacterTokenizer):
    """A generated BPE split can decode to text that re-encodes as one token."""

    merges = {"Vin": 900_001, "stale": 900_002}
    reverse_merges = {value: key for key, value in merges.items()}
    eos_token_id = 999_999

    @classmethod
    def decode(cls, token_ids, skip_special_tokens=False):
        return "".join(
            cls.reverse_merges[token_id]
            if token_id in cls.reverse_merges
            else chr(token_id)
            for token_id in token_ids
        )

    @classmethod
    def __call__(cls, text, add_special_tokens=False, return_offsets_mapping=False):
        input_ids = []
        offsets = []
        cursor = 0
        while cursor < len(text):
            merged = next(
                (value for value in cls.merges if text.startswith(value, cursor)),
                None,
            )
            if merged is None:
                input_ids.append(ord(text[cursor]))
                offsets.append((cursor, cursor + 1))
                cursor += 1
            else:
                input_ids.append(cls.merges[merged])
                offsets.append((cursor, cursor + len(merged)))
                cursor += len(merged)
        result = {"input_ids": input_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    @classmethod
    def convert_tokens_to_ids(cls, token):
        return cls.eos_token_id if token == "<|im_end|>" else -1


@pytest.fixture(scope="module")
def trainer_module():
    return _load_script("train_lora_grpo")


@pytest.fixture(scope="module")
def eval_module():
    return _load_script("passk_eval")


def _spec(index: int = 0) -> dict:
    return {
        "circuit_id": "noninv-amp",
        "spec_prompt": f"Design amplifier {index}.",
        "target_measurements": {"gain_db": [5.0, 7.0]},
        "input_ports": ["Vin"],
        "output_ports": ["Vout"],
    }


def test_train_and_eval_render_identical_chat_prompt(trainer_module, eval_module):
    tokenizer = FakeTokenizer()
    spec = _spec()

    training_prompt = trainer_module.build_training_rows([spec], tokenizer)[0]["prompt"]
    eval_prompt = eval_module.build_eval_prompt(tokenizer, spec)

    assert training_prompt == eval_prompt == render_grpo_prompt(tokenizer, spec)
    decoded = json.loads(training_prompt)
    assert decoded["messages"][0] == {"role": "system", "content": SYS_PROMPT}
    assert "GRADER INTERFACE CONTRACT" in decoded["messages"][1]["content"]
    assert decoded["kwargs"] == {
        "add_generation_prompt": True,
        "enable_thinking": False,
        "tokenize": False,
    }


def test_executable_policy_mask_neutralizes_comments_and_outer_prose(trainer_module):
    text = (
        "unrequested reasoning\n"
        "```spice\n"
        "* stale claim: gain is 99\n"
        ".title valid title\n"
        "R1 Vin Vout 1k ; stale inline claim\n"
        ".end\n"
        "```\n"
        "trailing prose"
    )
    token_ids = [ord(character) for character in text]

    mask = trainer_module.executable_policy_token_mask(CharacterTokenizer(), token_ids)
    retained = "".join(character for character, keep in zip(text, mask, strict=True) if keep)

    assert "stale claim" not in retained
    assert "unrequested reasoning" not in retained
    assert "trailing prose" not in retained
    assert "```spice" in retained
    assert ".title valid title" not in retained
    assert "R1 Vin Vout 1k" in retained
    assert "stale inline claim" not in retained
    assert ".end" in retained


def test_executable_policy_mask_keeps_malformed_completion_for_negative_gradient(trainer_module):
    text = "I forgot the SPICE fence and never produced a deck"
    token_ids = [ord(character) for character in text]

    mask = trainer_module.executable_policy_token_mask(CharacterTokenizer(), token_ids)

    assert mask == [1] * len(token_ids)


def test_executable_policy_mask_retains_assistant_termination(trainer_module):
    tokenizer = CharacterTokenizerWithTermination()
    text = "```spice\n* stale\nVin Vin 0 AC 1\n.op\n.end\n```<|im_end|>"
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    mask = trainer_module.executable_policy_token_mask(tokenizer, token_ids)

    assert token_ids[-1] == tokenizer.eos_token_id
    assert mask[-1] == 1


def test_executable_policy_mask_aligns_noncanonical_bpe_merges(trainer_module):
    tokenizer = NonCanonicalMergeTokenizer()
    text = "```spice\n* stale claim\nVin Vin 0 AC 1\n.op\n.end\n```"
    # Simulate generation choosing smaller adjacent tokens even though the BPE
    # encoder would canonically merge `stale` and `Vin`.
    generated_ids = [ord(character) for character in text]

    mask = trainer_module.executable_policy_token_mask(tokenizer, generated_ids)
    retained = "".join(
        character for character, keep in zip(text, mask, strict=True) if keep
    )

    assert len(mask) == len(generated_ids)
    assert "stale claim" not in retained
    assert "Vin Vin 0 AC 1" in retained


def test_executable_policy_mask_with_cached_smol_tokenizer(trainer_module):
    transformers = pytest.importorskip("transformers")
    checkpoint = ROOT / "checkpoints" / "sft_clean_v1"
    if not checkpoint.is_dir():
        pytest.skip("clean-SFT tokenizer artifact is distributed separately")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    text = (
        "```spice\n"
        "* stale numeric claim\n"
        ".title stale title\n"
        "Vin Vin 0 AC 1 ; stale inline claim\n"
        ".op\n.end\n```<|im_end|>"
    )
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    mask = trainer_module.executable_policy_token_mask(tokenizer, token_ids)
    retained_ids = [token_id for token_id, keep in zip(token_ids, mask, strict=True) if keep]
    retained = tokenizer.decode(retained_ids, skip_special_tokens=False)

    assert mask[-1] == 1
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") in retained_ids
    assert "stale numeric claim" not in retained
    assert "stale title" not in retained
    assert "stale inline claim" not in retained
    assert "Vin Vin 0 AC 1" in retained


def test_executable_policy_mask_with_real_noncanonical_smol_tokens(trainer_module):
    transformers = pytest.importorskip("transformers")
    checkpoint = ROOT / "checkpoints" / "sft_clean_v1"
    if not checkpoint.is_dir():
        pytest.skip("clean-SFT tokenizer artifact is distributed separately")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    text = "```spice\n* stale claim\nVin Vin 0 AC 1\n.op\n.end\n```"
    canonical_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    generated_ids = None
    for split in range(1, len(text)):
        candidate = tokenizer(text[:split], add_special_tokens=False)["input_ids"]
        candidate += tokenizer(text[split:], add_special_tokens=False)["input_ids"]
        if (
            candidate != canonical_ids
            and tokenizer.decode(candidate, skip_special_tokens=False) == text
        ):
            generated_ids = candidate
            break
    assert generated_ids is not None, "fixture must exercise a noncanonical BPE path"

    mask = trainer_module.executable_policy_token_mask(tokenizer, generated_ids)
    retained_ids = [
        token_id
        for token_id, keep in zip(generated_ids, mask, strict=True)
        if keep
    ]
    retained = tokenizer.decode(retained_ids, skip_special_tokens=False)

    assert len(mask) == len(generated_ids)
    assert "stale claim" not in retained
    assert "Vin Vin 0 AC 1" in retained


def test_chat_termination_uses_im_end_and_distinct_padding():
    tokenizer = FakeTokenizer()
    model = FakeModel()

    result = configure_chat_termination(tokenizer, model)

    assert tokenizer.eos_token == "<|im_end|>"
    assert tokenizer.eos_token_id == 7
    assert tokenizer.pad_token_id == 9
    assert model.generation_config.eos_token_id == 7
    assert model.generation_config.pad_token_id == 9
    assert result["assistant_eos_token_id"] == 7


def test_gradient_checkpointing_is_explicitly_off_unless_safely_enabled(trainer_module):
    assert trainer_module.gradient_checkpointing_config({}) == {
        "gradient_checkpointing": False
    }


def test_trl_accepts_k8_generation_with_microbatch_one():
    """Generation groups and optimizer microbatches are independent in TRL 1.6."""
    GRPOConfig = pytest.importorskip("trl").GRPOConfig

    config = GRPOConfig(
        output_dir="/tmp/phychip-grpo-config-test",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        generation_batch_size=8,
        num_generations=8,
        num_iterations=2,
        report_to="none",
        bf16=False,
        use_cpu=True,
    )

    assert config.per_device_train_batch_size == 1
    assert config.gradient_accumulation_steps == 8
    assert config.generation_batch_size == config.num_generations == 8


def test_completion_text_handles_plain_and_conversational_rollouts(trainer_module):
    assert trainer_module.completion_text("plain") == "plain"
    assert trainer_module.completion_text([]) == ""
    assert trainer_module.completion_text(
        [{"role": "assistant", "content": "netlist"}]
    ) == "netlist"
    assert trainer_module.gradient_checkpointing_config(
        {"PHYCHIP_GRAD_CKPT": "1"}
    ) == {
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }


def test_exact_duplicate_training_rollouts_share_one_verifier_call(trainer_module):
    items = [
        (
            [{"role": "assistant", "content": "same"}],
            "noninv-amp",
            '{"gain_db":[5,7]}',
        ),
        ("same", "noninv-amp", '{"gain_db":[5,7]}'),
        ("same", "noninv-amp", '{"gain_db":[8,10]}'),
        ("different", "noninv-amp", '{"gain_db":[5,7]}'),
    ]

    unique, inverse = trainer_module.deduplicate_reward_items(items)

    assert len(unique) == 3
    assert inverse == [0, 0, 1, 2]
    assert unique[0][0] == "same"


def test_trainable_parameter_fingerprint_changes_only_with_trainable_state(trainer_module):
    torch = pytest.importorskip("torch")

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    for parameter in model[1].parameters():
        parameter.requires_grad = False
    initial = trainer_module.trainable_parameter_sha256(model)
    with torch.no_grad():
        model[1].weight.add_(1.0)
    assert trainer_module.trainable_parameter_sha256(model) == initial
    with torch.no_grad():
        model[0].weight.add_(1.0)
    assert trainer_module.trainable_parameter_sha256(model) != initial


def test_exact_regime_label_preserves_spec_id_and_drops_stale_bucket(eval_module):
    spec = {
        **_spec(),
        "spec_id": "spec_real_identity",
        "bucket": "train",
        "trainable": True,
        "trainability_label": {"stale": True},
    }
    provenance = {
        "policy_hash": "a" * 64,
        "verifier_hash": "b" * 64,
        "prompt_hash": "c" * 64,
    }

    row = eval_module.build_result_row(
        spec,
        [False, True, False],
        [0.3, 1.3, 0.4],
        [0.0, 1.0, 0.1],
        label_provenance=provenance,
    )

    assert row["spec_id"] == "spec_real_identity"
    assert "bucket" not in row
    assert "trainable" not in row
    assert row["trainability_label"] == {
        "label_schema": "exact-regime-v2",
        "K": 3,
        "success_vector": [False, True, False],
        "reward_vector": [0.3, 1.3, 0.4],
        "reward_semantics": eval_module.SCORER_SEMANTICS,
        "policy_hash": "a" * 64,
        "verifier_hash": "b" * 64,
        "prompt_contract_hash": "c" * 64,
    }

    with pytest.raises(ValueError, match="SHA-256"):
        eval_module.build_trainability_label(
            [False, True],
            [0.0, 1.3],
            policy_hash="not-a-hash",
            verifier_hash="b" * 64,
            prompt_hash="c" * 64,
        )


def test_grpo_launch_rejects_stale_or_wrong_k_trainability_labels(trainer_module):
    row = {
        "trainable": True,
        "trainability_label": {
            "label_schema": "exact-regime-v2",
            "K": 3,
            "reward_vector": [0.3, 1.3, 0.4],
            "success_vector": [False, True, False],
            "policy_hash": "a" * 64,
            "verifier_hash": "b" * 64,
            "prompt_contract_hash": "c" * 64,
        },
    }
    expected = {
        "policy_hash": "a" * 64,
        "verifier_hash": "b" * 64,
        "prompt_contract_hash": "c" * 64,
    }

    evidence = trainer_module.require_matching_trainability_evidence(
        row, expected_k=3, expected_hashes=expected
    )
    assert evidence["status"] == "trainable_mixed_success"
    with pytest.raises(ValueError, match="launch K"):
        trainer_module.require_matching_trainability_evidence(
            row, expected_k=8, expected_hashes=expected
        )
    with pytest.raises(ValueError, match="policy_hash"):
        trainer_module.require_matching_trainability_evidence(
            row,
            expected_k=3,
            expected_hashes={**expected, "policy_hash": "d" * 64},
        )


def test_experiment_provenance_hashes_real_policy_verifier_and_prompt_bytes(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"real-adapter-weights")
    specs = tmp_path / "specs.jsonl"
    specs.write_text(json.dumps(_spec()) + "\n", encoding="utf-8")
    template = tmp_path / "chat_template.jinja"
    template.write_text("{{ messages }}", encoding="utf-8")
    train_root = tmp_path / "athma-train"
    (train_root / "scripts").mkdir(parents=True)
    (train_root / "athma_train" / "env").mkdir(parents=True)
    (train_root / "athma_train" / "harnesses").mkdir(parents=True)
    (train_root / "athma_train" / "env" / "rlvr_scorer.py").write_text(
        "# canonical reward implementation\n", encoding="utf-8"
    )
    (train_root / "athma_train" / "spice_gate.py").write_text(
        "# sandbox gate\n", encoding="utf-8"
    )
    (train_root / "athma_train" / "harnesses" / "noninv-amp.py").write_text(
        "# exact harness\n", encoding="utf-8"
    )
    shared_measure_utils = (
        train_root / "athma_train" / "harnesses" / "_measure_utils.py"
    )
    shared_measure_utils.write_text("# shared measurement parser\n", encoding="utf-8")

    first = build_experiment_provenance(
        base_repo="HuggingFaceTB/SmolLM3-3B-Base",
        base_revision="d78a42f79198603e614095753484a04c10c2b940",
        adapter_paths=[adapter],
        specs_path=specs,
        train_root=train_root,
        chat_template_path=template,
        version="** ngspice-46\n** Compiled with KLU\n** Creation Date: host A",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"changed-adapter-weights")
    second = build_experiment_provenance(
        base_repo="HuggingFaceTB/SmolLM3-3B-Base",
        base_revision="d78a42f79198603e614095753484a04c10c2b940",
        adapter_paths=[adapter],
        specs_path=specs,
        train_root=train_root,
        chat_template_path=template,
        version="** ngspice-46\n** Compiled with KLU\n** Creation Date: host B",
    )
    shared_measure_utils.write_text("# changed shared measurement parser\n", encoding="utf-8")
    third = build_experiment_provenance(
        base_repo="HuggingFaceTB/SmolLM3-3B-Base",
        base_revision="d78a42f79198603e614095753484a04c10c2b940",
        adapter_paths=[adapter],
        specs_path=specs,
        train_root=train_root,
        chat_template_path=template,
        version="** ngspice-46\n** Compiled with KLU",
    )

    assert first["policy_hash"] != second["policy_hash"]
    assert first["verifier_hash"] == second["verifier_hash"]
    assert second["policy_hash"] == third["policy_hash"]
    assert second["verifier_hash"] != third["verifier_hash"]
    assert first["prompt_contract_hash"] == second["prompt_contract_hash"]
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", first[name])
        for name in ("policy_hash", "verifier_hash", "prompt_contract_hash")
    )


def test_ngspice_runtime_identity_ignores_only_creation_date():
    mac = "** ngspice-46\n** Compiled with KLU Direct Linear Solver"
    pod = mac + "\n** Creation Date: Tue Jul 21 20:55:46 UTC 2026"
    changed_version = "** ngspice-47\n** Compiled with KLU Direct Linear Solver"
    changed_features = "** ngspice-46\n** Compiled without KLU"

    assert canonical_ngspice_runtime_identity(mac) == canonical_ngspice_runtime_identity(pod)
    assert canonical_ngspice_runtime_identity(mac) != canonical_ngspice_runtime_identity(
        changed_version
    )
    assert canonical_ngspice_runtime_identity(mac) != canonical_ngspice_runtime_identity(
        changed_features
    )


def test_rescore_does_not_invent_trainability_evidence_for_eval_rows():
    module = _load_script("rescore_auditable_eval")
    provenance = {"policy_hash": "a" * 64, "prompt_hash": "b" * 64}

    assert (
        module.trainability_provenance_for_materialization(
            {"success_vector": [True]}, provenance, "c" * 64
        )
        is None
    )
    assert module.trainability_provenance_for_materialization(
        {"trainability_label": {"K": 8}}, provenance, "c" * 64
    ) == {
        "policy_hash": "a" * 64,
        "verifier_hash": "c" * 64,
        "prompt_hash": "b" * 64,
    }


def test_rescore_retargets_only_to_byte_identical_specs(tmp_path: Path):
    module = _load_script("rescore_auditable_eval")
    original = tmp_path / "original.jsonl"
    identical = tmp_path / "identical.jsonl"
    changed = tmp_path / "changed.jsonl"
    original.write_text('{"spec_id":"a"}\n', encoding="utf-8")
    identical.write_bytes(original.read_bytes())
    changed.write_text('{"spec_id":"b"}\n', encoding="utf-8")
    provenance = {
        "specs": original.as_posix(),
        "specs_sha256": module.sha256_file(original),
    }

    resolved, stored_path, digest = module.resolve_rescore_specs(
        provenance, identical
    )

    assert resolved == identical
    assert stored_path == identical.as_posix()
    assert digest == provenance["specs_sha256"]
    with pytest.raises(ValueError, match="not byte-identical"):
        module.resolve_rescore_specs(provenance, changed)

    assert module.generation_provenance_value(
        {"verifier_hash": "new", "generation_verifier_hash": "original"},
        "verifier_hash",
    ) == "original"


def test_missing_requested_measurement_cannot_count_as_all_specs_met(
    trainer_module, monkeypatch
):
    class Harness:
        @staticmethod
        def measure(_netlist, spec, timeout_s):
            assert timeout_s == 15
            return {"gain_db": 6.0}  # bandwidth_hz is missing

    monkeypatch.setattr(trainer_module, "grade_netlist", lambda *_args, **_kwargs: (True, "ok"))
    monkeypatch.setattr(trainer_module, "_anticheat_ok", lambda *_args: True)
    monkeypatch.setattr(trainer_module, "load_harness", lambda _cid: Harness())
    sink = []
    completion = """```spice
V1 Vin 0 1
R1 Vin Vout 1k
.op
.end
```"""

    reward = trainer_module.reward_fn_v2(
        "noninv-amp",
        {"gain_db": [5.0, 7.0], "bandwidth_hz": [1e3, 2e3]},
        completion,
        _sink=sink,
    )

    assert reward == pytest.approx(0.35)
    assert len(sink) == 1
    assert sink[0]["reward"] == pytest.approx(0.35)
    assert sink[0]["simulated"] is True
    assert sink[0]["anticheat_blocked"] is False
    assert sink[0]["requested_specs"] == 2
    assert sink[0]["measured_specs"] == 1
    assert sink[0]["met_specs"] == 1
    assert sink[0]["spec_score"] == pytest.approx(0.5)
    assert sink[0]["all_specs_met"] is False
    assert sink[0]["schema"] == "phychip-rlvr-score-v1"


def test_training_defaults_to_canonical_reward_and_rejects_legacy(trainer_module):
    assert trainer_module.select_reward_function({}) is trainer_module.reward_fn_v2
    assert (
        trainer_module.select_reward_function({"PHYCHIP_REWARD": "v2"})
        is trainer_module.reward_fn_v2
    )
    with pytest.raises(RuntimeError, match="legacy PHYCHIP_REWARD"):
        trainer_module.select_reward_function({"PHYCHIP_REWARD": "v1"})


def test_anticheat_ignores_comments_but_rejects_real_transfer_blocks(trainer_module):
    comment_only = """* This comment discusses a laplace transfer function.
Vin Vin 0 AC 1
R1 Vin n1 1k
R2 n1 n2 1k
C1 n1 Vout 1n
C2 n2 0 1n
Ebuf Vout 0 n2 Vout 1e6
.ac dec 10 1 1Meg
.end
"""
    actual_transfer = """Vin Vin 0 AC 1
B1 Vout 0 V=laplace(V(Vin), {1}, {1 1})
C1 Vout 0 1n
C2 Vin 0 1n
.ac dec 10 1 1Meg
.end
"""

    assert trainer_module._anticheat_ok(comment_only, "sallen-key") is True
    assert trainer_module._anticheat_ok(actual_transfer, "sallen-key") is False


def test_gain_trim_rejects_direct_input_synthesis_but_allows_feedback_opamp(
    trainer_module,
):
    direct_synthesis = """Vin Vin 0 AC 1
Vtrim Vtrim 0 AC 1
Bdrv ndrv 0 V={10.75*V(Vin)+1.95*V(Vtrim)}
Rout ndrv Vout 10
Rf Vout nfb 100k
Rtrim Vtrim nfb 1Meg
.op
.end
"""
    input_alias_synthesis = """Vin Vin 0 AC 1
Vtrim Vtrim 0 AC 1
Rin Vin nvar 10k
Rhub nvar 0 100Meg
Bdrv ndrv 0 V={4.6*V(nvar)}
Rout ndrv Vout 10
Rload Vout 0 10k
.op
.end
"""
    feedback_opamp = """Vin Vin 0 AC 1
Vtrim Vtrim 0 AC 1
Eop ndrv 0 Vin nfb 1e6
Rout ndrv Vout 10
Rf Vout nfb 100k
Rg nfb 0 10k
Rtrim Vtrim nfb 1Meg
.op
.end
"""

    assert trainer_module._anticheat_ok(direct_synthesis, "gain-trim") is False
    assert trainer_module._anticheat_ok(input_alias_synthesis, "gain-trim") is False
    assert trainer_module._anticheat_ok(feedback_opamp, "gain-trim") is True


def test_gain_trim_anticheat_keeps_all_verified_references(trainer_module):
    path = ROOT / "data" / "dataset_v3_verified_v1" / "verified_candidates.jsonl"
    if not path.is_file():
        pytest.skip("private verified-reference corpus is not distributed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    references = [
        row["reference_netlist"] for row in rows if row.get("circuit_id") == "gain-trim"
    ]

    assert len(references) == 268
    assert all(trainer_module._anticheat_ok(netlist, "gain-trim") for netlist in references)


@pytest.mark.parametrize(
    ("circuit_id", "netlist"),
    [
        (
            "inv-amp",
            """Vin Vin 0 AC 1
Rg nminus 0 10k
Rf Vout nminus 100k
Eop Vout 0 Vin nminus 1e6
.op
.end
""",
        ),
        (
            "noninv-amp",
            """Vin Vin 0 AC 1
Rin Vin nminus 10k
Rf Vout nminus 100k
Eop Vout 0 0 nminus 1e6
.op
.end
""",
        ),
        (
            "summing-amp",
            """V1 V1 0 AC 1
V2 V2 0 AC 1
V3 V3 0 AC 1
R1 V1 nsum 10k
R2 V2 0 10k
R3 V3 0 10k
Rf Vout nsum 10k
Eop Vout 0 0 nsum 1e6
.op
.end
""",
        ),
        (
            "sallen-key",
            """Vin Vin 0 AC 1
R1 Vin n1 10k
R2 n1 Vout 10k
C1 n1 0 10n
C2 Vout 0 10n
Ebuf Vout 0 n1 0 1
.ac dec 20 1 1Meg
.end
""",
        ),
        (
            "schmitt",
            """Vin Vin 0 DC 0
Rin Vin nminus 1k
Rload Vout 0 10k
Ecmp Vout 0 0 nminus 1e6
.dc Vin -2 2 0.01
.end
""",
        ),
    ],
)
def test_family_topology_anticheat_rejects_missing_load_bearing_structure(
    trainer_module, circuit_id, netlist
):
    assert trainer_module._anticheat_ok(netlist, circuit_id) is False


@pytest.mark.parametrize(
    ("circuit_id", "expected_count", "expected_rejected"),
    [
        ("buffer", 125, 0),
        ("inv-amp", 194, 0),
        ("noninv-amp", 129, 0),
        ("summing-amp", 188, 0),
        ("sallen-key", 178, 2),
        ("schmitt", 16, 0),
    ],
)
def test_current_anticheat_rechecks_historical_verified_references(
    trainer_module, circuit_id, expected_count, expected_rejected
):
    path = ROOT / "data" / "dataset_v3_verified_v1" / "verified_candidates.jsonl"
    if not path.is_file():
        pytest.skip("private verified-reference corpus is not distributed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    references = [
        row["reference_netlist"]
        for row in rows
        if row.get("circuit_id") == circuit_id
    ]

    assert len(references) == expected_count
    rejected = [
        netlist
        for netlist in references
        if not trainer_module._anticheat_ok(netlist, circuit_id)
    ]
    assert len(rejected) == expected_rejected
    if expected_rejected:
        trainable_path = (
            ROOT
            / "data"
            / "dataset_v3_verified_v1"
            / "trainable_candidates.jsonl"
        )
        assert trainable_path.read_bytes() == b""


def test_reward_harnesses_do_not_turn_censored_bandwidth_into_exact_value():
    harness_root = ROOT / "athma-train" / "athma_train" / "harnesses"
    offenders = [
        path.name
        for path in harness_root.glob("*.py")
        if path.name != "_measure_utils.py"
        and "fallback_hz=" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_reward_decomp_reports_group_variance_not_just_pass_fraction(
    trainer_module, capsys
):
    events = [
        {
            "reward": 0.30,
            "simulated": True,
            "anticheat_blocked": False,
            "requested_specs": 1,
            "measured_specs": 1,
            "met_specs": 0,
            "spec_score": 0.0,
            "all_specs_met": False,
        },
        {
            "reward": 0.475,
            "simulated": True,
            "anticheat_blocked": False,
            "requested_specs": 1,
            "measured_specs": 1,
            "met_specs": 0,
            "spec_score": 0.25,
            "all_specs_met": False,
        },
    ]

    trainer_module._log_reward_decomp(events, group_size=2)
    output = capsys.readouterr().err

    assert "frac_spec_met=0.00" in output
    assert "frac_group_reward_var=1.00" in output
    assert "frac_group_spec_var=1.00" in output
    assert "frac_group_objective_var=0.00" in output


def _result_row(index: int, pass_count: int, k: int = 8) -> dict:
    return {**_spec(index), "pass_count": pass_count, "k": k}


def test_paired_verdict_clusters_by_spec_instead_of_pooling_seed_rows():
    baseline_rows = [_result_row(i, value) for i, value in enumerate([0, 1, 0, 1])]
    seed_values = ([1, 2, 1, 2], [0, 2, 1, 1], [1, 1, 2, 2])
    random_values = [0, 1, 0, 1]

    def keyed(rows):
        return {spec_identity(row): row for row in rows}

    verdict = paired_cluster_verdict(
        keyed(baseline_rows),
        [keyed([_result_row(i, value) for i, value in enumerate(seed)]) for seed in seed_values],
        keyed([_result_row(i, value) for i, value in enumerate(random_values)]),
        bootstrap_samples=500,
        seed=12,
    )

    metric = verdict["pass_at_1"]
    assert metric["n_specs"] == 4
    assert metric["n_treatment_seeds"] == 3
    assert metric["baseline_mean"] == pytest.approx(2 / 32)
    assert metric["treatment_mean_across_seeds"] == pytest.approx(16 / 96)
    assert metric["treatment_seed_means"] == pytest.approx([6 / 32, 4 / 32, 6 / 32])
    assert metric["treatment_seed_mean_range"] == pytest.approx([4 / 32, 6 / 32])
    assert metric["paired_delta"] == pytest.approx(10 / 96)
    assert "specification-cluster" in verdict["method"]


def test_negative_random_control_does_not_veto_positive_treatment():
    def keyed(rows):
        return {spec_identity(row): row for row in rows}

    baseline = keyed([_result_row(index, 1) for index in range(20)])
    treatments = [
        keyed([_result_row(index, pass_count) for index in range(20)])
        for pass_count in (3, 4, 5)
    ]
    negative_control = keyed([_result_row(index, 0) for index in range(20)])

    metric = paired_cluster_verdict(
        baseline,
        treatments,
        negative_control,
        bootstrap_samples=200,
        seed=9,
    )["pass_at_1"]

    assert metric["paired_delta_ci"][0] > 0.0
    assert metric["random_paired_delta_ci"][1] < 0.0
    assert metric["random_control_not_positive"] is True
    assert metric["grpo_lift"] is True


def test_result_loader_rejects_duplicate_spec_identity(tmp_path: Path):
    row = _result_row(0, 1)
    path = tmp_path / "duplicates.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate spec identity"):
        load_policy_results(path)
