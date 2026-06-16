# PhyChip

**A physics simulator as a verifiable reward for reinforcement learning — training a language model to design analog circuits.**

[![License](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Models](https://img.shields.io/badge/models-🤗%20Hugging%20Face-yellow.svg)](https://huggingface.co/NithinReddyG)

---

## The idea

Most generative tasks are hard to grade automatically — you need a human, or a model judging another model. Analog circuit design is different: a circuit is **correct or not**, and a physics simulator can tell you which. Run the design in [ngspice](https://ngspice.sourceforge.io/), measure what it actually does, and compare it to the target spec. That objective check is exactly what reinforcement learning needs — a reward you can trust.

PhyChip turns that check into the reward signal. A language model reads a natural-language spec, writes a SPICE netlist, and the simulator decides the reward:

```
   spec (natural language)
            │
            ▼
   ┌──────────────────┐     SPICE netlist
   │  language model  │ ───────────────────►  ┌──────────┐
   └──────────────────┘                       │  ngspice │  simulate
            ▲                                  └────┬─────┘
            │                                       ▼
            │                                measure the circuit
            │      reward                    (gain, bandwidth, …)
            └──────────────────────────────  meets spec within ±30%?
                                                   pass / fail
```

The model only improves by producing circuits that **actually simulate and meet spec**. There is no learned reward model to game — the reward *is* the physics.

## What's here

- **The environment** — ngspice wrapped as a deterministic reward oracle, with **23 per-circuit measurement harnesses** (102 tests) that measure real figures-of-merit and reject designs that fake them.
- **Training** — SFT on ~15K simulator-verified pairs → GRPO (group-relative, no critic; 1,282-spec pool) on a small base model with LoRA.
- **Two benchmarks** — an in-distribution set and a **contamination-free** set of novel circuit types, with automatic ngspice scoring.
- **A reward-robustness audit** — adversarial "reward hacks" and the topology guard that defeats them.

## Results

| Benchmark | What it measures | Best model |
|---|---|---|
| **AnalogCoder** (24, external) | textbook circuits | **22/24 (91.7%)** — ahead of much larger open models (gpt-oss-20B: 19/24) |
| **phy-chip-bench-v1** (40) | in-distribution capability | **19/23** topology |
| **phy-chip-bench-v2** (50) | generalization to *unseen circuit types* | **only RL generalizes** — base/SFT 0/50, RL ~10/50 |

- **Reward robustness:** an adversarial audit took reward hacks from **4/12 → 0/12** with no regression on legitimate circuits.
- **Base vs instruct:** the same recipe **lifts a base model (16/40)** but **collapses an instruction-tuned one (0/40)** — fine-tune the base.

Full write-up: **[`final_report/PhyChip_Technical_Report.md`](final_report/PhyChip_Technical_Report.md)**.

## Models

LoRA adapters on Hugging Face — [`NithinReddyG/PhyChip-SmolLM3-3B-*`](https://huggingface.co/NithinReddyG):
`base-SFT`, `base-GRPO-v3`, `base-L2L3-GRPO`, `instruct-SFT`, `instruct-GRPO-v1`, `instruct-GRPO-v2`.

## Quickstart

```bash
pip install -e athma-train
ngspice --version            # ngspice must be on PATH (e.g. via conda-forge)

# evaluate an adapter on the contamination-free benchmark
python athma-train/scripts/eval_on_bench_v1.py \
  --base HuggingFaceTB/SmolLM3-3B-Base \
  --adapter NithinReddyG/PhyChip-SmolLM3-3B-base-GRPO-v3 \
  --bench eval_sets/phychip_bench_v2/bench_v2.jsonl --output-dir /tmp/eval
```

## How scoring works

Every benchmark task is graded the same deterministic way:

1. The model emits a SPICE netlist (greedy, pass@1).
2. Extract it → check it has real devices and an analysis directive → **run it in ngspice**.
3. The circuit's harness measures the actual figure-of-merit and checks it is **within ±30% of the target**.
4. **Pass = it simulates *and* meets spec.** (pass@k with bootstrap confidence intervals for the headline claims.)

## Layout

```
athma-train/athma_train/   the environment: spice_gate + 23 measurement harnesses
athma-train/scripts/       trainers (SFT, GRPO), evaluation, reward-hack grader
athma-train/tests/         harness tests
eval_sets/                 the two benchmarks
final_report/              technical report + figures
```

## License

Code is **Apache-2.0**. Model weights are released **CC-BY-NC-SA-4.0** (research/educational). Training-data sources are license-tagged in `NOTICE`.

## Author

Nithin Reddy Govindugari — nithingovindugari@gmail.com

Evaluated with the [AnalogCoder](https://arxiv.org/abs/2405.14918) benchmark (eval-only) on [SmolLM3-3B-Base](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base).
