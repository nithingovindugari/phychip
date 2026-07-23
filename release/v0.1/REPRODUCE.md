# Reproducing the v0.1 release candidate

> **Local scorer-v9 candidate (2026-07-22):** 384 frozen current-policy
> rollouts were completely reviewed, defect classes were converted into
> mirrored regressions, and the preserved set was deterministically rescored.
> Source, parity, attack, and clean-wheel gates must all pass before tagging.

Run commands from the repository root. Do not compare results unless ngspice 46,
the prompt mode, chat template, termination IDs, completion budget, model
revision, adapter order, benchmark bytes, and scorer semantics all match.

## 1. Verify the standalone environment

```bash
bash phy-chip-env/scripts/install_ngspice46.sh
PHYCHIP_REQUIRE_NGSPICE=46 .venv/bin/python -m pytest \
  phy-chip-env/tests -q -p no:cacheprovider
```

Expected on the scorer-v9 release-candidate tree: `175 passed`.

## 2. Build and test the wheel cleanly

```bash
UV_CACHE_DIR=/tmp/phychip-uv-cache uv build --wheel phy-chip-env
clean_dir=$(mktemp -d /tmp/phychip-clean-install.XXXXXX)
python3 -m venv "$clean_dir/venv"
"$clean_dir/venv/bin/python" -m pip install --no-deps \
  phy-chip-env/dist/phy_chip_env-0.1.0-py3-none-any.whl
cd /tmp
PHYCHIP_REQUIRE_NGSPICE=46 "$clean_dir/venv/bin/python" \
  /path/to/phy-chip/phy-chip-env/scripts/verify_clean_install.py
```

The clean smoke must report 23 harnesses, gold reward 1.3, attack rejection,
timeout, and a 64-character verifier hash. The scorer-v9 clean install reported
`7d3a215983533d44ed54fc677fcd36ea0479f6a86cdb0e266bdef35d84f0b3aa`;
the wheel SHA-256 was
`d668f0470cd0a1849b8e87be9bbfcf1dec16086771370446ed2d66b1674b454e`.
This verifies installation and core runtime behavior; the parent-repository
suite supplies the corrected-reference/106-output parity gate.

## 3. Reproduce the clean-SFT primary matrix

The exact GPU runner is `plan/sft_clean_eval_pod_run.sh`. Before launching,
verify:

```bash
shasum -a 256 checkpoints/sft_clean_v1/adapter_model.safetensors
shasum -a 256 checkpoints/sft_clean_v1/chat_template.jinja
shasum -a 256 data/final_data/sft_v0_verified_v1/sft_training_ready.jsonl
```

Expected hashes appear in `MODEL_CARD.md`. Then run the repository's pinned pod
workflow in a CUDA environment matching `eval-environment.lock.json`:

```bash
bash plan/sft_clean_eval_pod_run.sh
```

The primary outputs are:

- `eval_results/sft_clean_v1/analogcoder_sft_chat/`
- `eval_results/sft_clean_v1/bench_v1_sft_chat/`
- `eval_results/sft_clean_v1/bench_v2_sft_chat/`
- `eval_results/sft_clean_v1/physics_mmlu_comparison.json`

The primary contract is native SFT chat, greedy decoding,
`max_new_tokens=2560`, and ngspice 46. Do not replace it with legacy raw
continuation or with the separate 16,384-token control.

## 4. Verify content addresses

```bash
python3 release/v0.1/build_manifest.py
shasum -a 256 -c release/v0.1/SHA256SUMS
```

The generator deliberately records `prepared_uncommitted` while the working
tree is dirty. After human approval, run it once more from the intended clean
release commit, review the diff, then tag. No script here commits, pushes, or
publishes anything.
