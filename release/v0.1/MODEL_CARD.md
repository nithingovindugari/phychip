---
base_model: HuggingFaceTB/SmolLM3-3B-Base
library_name: peft
pipeline_tag: text-generation
license: cc-by-nc-sa-4.0
tags:
  - analog-circuit-design
  - ngspice
  - research-preview
  - lora
  - sft
---

# PhyChip-3B clean SFT v0.1

PhyChip-3B clean SFT v0.1 is a **limited research preview** of a 3B model that
generates ngspice decks from natural-language circuit requests. Its main role is
to demonstrate the separately released `phy-chip-env` verifiable physics
environment. It is not a production circuit designer, a tape-out tool, or an
RL-generalization result.

## Model

- Base: `HuggingFaceTB/SmolLM3-3B-Base`
- Base revision: `d78a42f79198603e614095753484a04c10c2b940`
- Adapter: LoRA rank 16, alpha 32, dropout 0.05, all attention and MLP projections
- Adapter weights SHA-256: `2307ebd794046f3b0be8f8a4de07469f0e6045d91869d3e7058b1897b3936dff`
- Adapter policy hash recorded by evaluation: `0f89352b0fb5e3e7432785416a949a5ea98683710fb5e2f5b9cfe80f30feed0d`
- Chat template SHA-256: `3e0e275a527f249670056e0078e474eadbdd007c134b4fdfae5a7b3089702814`

This adapter was trained with supervised fine-tuning, not RL. Do not label the
released clean-SFT weights as an RLVR or GRPO model.

## Training data and procedure

The exact training input contains 15,660 three-message examples. All rows
passed static structure/safety checks, the saved SmolLM3 assistant-mask and
4,096-token gate, sandbox validation, and an ngspice-46 simulation.

That means **simulator-passing**, not that every response met every requested
numeric specification or used the intended topology. Of the final rows, 48.1%
share a normalized netlist template; the row count therefore overstates
structural diversity.

- Training corpus SHA-256: `02af3e6c58b668acec130850ea31d468fb8fd49f938272854d4d857b0fd39b9b`
- Epochs: 2
- Optimizer steps: 1,958
- Peak learning rate: `2e-4`
- Per-device batch size: 4
- LoRA rank/alpha: 16/32
- Framework record: PyTorch 2.13.0, Transformers 5.8.1, PEFT 0.19.1,
  TRL 1.6.0, Datasets 4.8.5, Tokenizers 0.22.2

The training loss/accuracy confirm that optimization ran; they do not establish
circuit correctness or generalization.

## Primary evaluation

The primary release contract uses native SFT chat, greedy generation,
`max_new_tokens=2560`, the pinned base revision above, and ngspice 46.

| Evaluation | Pass@1 | Scope |
|---|---:|---|
| AnalogCoder-24 raw-SPICE proxy | 21/24 (87.5%) | Simulatability proxy, not the official benchmark |
| phy-chip-bench v1 | 23/40 (57.5%) | 23/23 Level 1; 0/17 Levels 2/3 |
| phy-chip-bench v2 | 25/46 (54.35%) | 24/28 Level 1; 1/13 Level 2; 0/5 Level 3 |

These are automatic proxy scores, not upper or lower bounds on topology
correctness. In a separate, longer 16,384-token matched control, circuit-level
review found only 38/110 complete topology-plausible Smol outputs and 47 proxy
passes with clearly wrong topology. That control used a different completion
budget and must not be substituted for the primary matrix.

## General-capability retention

Physics-MMLU non-inferiority is **not established**. The base scored 365/785
(46.50%) and clean SFT scored 356/785 (45.35%), a paired difference of
-1.15 percentage points with 95% CI `[-3.31,+0.89]pp`. The predeclared
non-inferiority margin was 2 percentage points, and the interval crosses it.

## RL evidence

The historical GRPO-v3 candidate was rejected under the corrected contract:
16/24 AnalogCoder proxy, 19/40 bench-v1, and 10/46 bench-v2. It also used a
different SFT parent, so it is not a clean causal comparison with this model.

A later matched rank-64 smoke run on 53 policy-relative holdout prompts moved
from 3/53 to 6/53. The paired 95% CI is `[-1.89,+13.21]pp` and exact McNemar
`p=0.375`. The point estimate is positive but inconclusive and concentrated in
two circuit families. It does not establish broad RL generalization.

A subsequent scorer-v6 seed completed 140 optimizer steps but failed its frozen
entropy health gate. Manual review also found one invalid positive-feedback
group winner that escaped the then-current graph check. Seed 7 and paired
evaluation were not run. That adapter is not a model candidate or evidence
that RL worked or failed.

A later verifier/data pilot generated 384 frozen current-policy rollouts. Full
review found 46 invalid positive-reward rows under scorer v8. Scorer v9 replay
retains 11 safe mixed prompts across 7 families and 18 exact successes, with no
known-invalid circuit among the 33 group winners. No optimizer step used these
labels. This validates a pilot data contract, not RL improvement.

## Intended use

Suitable uses:

- research on verifiable rewards and physics-backed evaluation;
- educational demonstrations of netlist generation;
- analysis of verifier design, reward hacking, and train/eval contracts.

Out-of-scope uses:

- production, safety-critical, medical, automotive, aerospace, or power-system design;
- tape-out or component selection without independent expert review;
- treating simulator success as proof of physical correctness;
- claiming that RL improved generalization from these weights.

Always run generated decks through `phy-chip-env`, inspect its component-level
diagnostics, and manually review topology and sizing.

## Contamination and licensing

The executable decontamination gate checked 1,786 registered evaluation prompts
and 877 reference netlists and found 0 exact/registered leaks. This is evidence
against the implemented checks, not proof of semantic non-contamination.

Weights and released data use **CC-BY-NC-SA-4.0** for educational and research
use. The standalone `phy-chip-env` framework is Apache-2.0. See
`DATA_AND_LICENSE_NOTICE.md` and `BENCHMARK_CONTAMINATION.md` in the release
bundle.
