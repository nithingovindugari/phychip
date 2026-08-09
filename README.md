# PhyChip

**A verifier-first RLVR environment for analog circuit design.** PhyChip turns
ngspice-46 into a sandboxed, versioned reward oracle for language models that
generate SPICE netlists.

[![Framework: Apache-2.0](https://img.shields.io/badge/framework-Apache--2.0-blue.svg)](phy-chip-env/LICENSE)
[![phy-chip-env CI](https://github.com/nithingovindugari/phychip/actions/workflows/phy-chip-env-ci.yml/badge.svg)](https://github.com/nithingovindugari/phychip/actions/workflows/phy-chip-env-ci.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-3776AB.svg)](phy-chip-env/pyproject.toml)
[![ngspice 46](https://img.shields.io/badge/ngspice-46-555.svg)](phy-chip-env/docs/REPRODUCIBILITY.md)
[![Tests: 257 pass + 1 private skip](https://img.shields.io/badge/tests-257%20pass%20%2B%201%20private%20skip-brightgreen.svg)](release/v0.1/EVIDENCE.md)

The main artifact is [`phy-chip-env`](phy-chip-env/), not a claim that a 3B
model has solved analog design. The model is the proof case for a broader
research question: can on-policy reinforcement learning use deterministic
physics feedback to improve an engineering policy beyond static imitation?

## What is ready

| Artifact | Evidence-backed status |
|---|---|
| `phy-chip-env` | 23 circuit harnesses, scorer v9, 257 public-checkout tests passing plus 1 private-corpus skip, 0/22 frozen attacks successful |
| Clean-SFT preview | 21/24 AnalogCoder raw-SPICE proxy, 23/40 bench-v1, 25/46 bench-v2 |
| GRPO stack | Dr.GRPO plus selected DAPO components, exact prompt/EOS parity, executable-token loss, provenance gates |
| RL result | Not established; historical candidates are rejected, confounded, or statistically inconclusive |
| Fresh RLVR pilot | 48 references, 384 K=8 rollouts, 11 safe mixed prompts across 7 families, 18 exact successes, 0 optimizer steps |
| Closed-loop development probe | Random 3/4 in 18 calls; Latin hypercube 3/4 in 17; fixed topology + TPE 4/4 in 21; n=4 development tasks, no superiority or generalization claim |
| Eight-family feedback preflight | Raw full 19/24 vs blind 10/24; excluding 7 error-touched pairs gives 17/17 vs 10/17 (`p=.015625`); family sign `p=.0625`, audit/practicality gates failed; gate closed |

Physics-MMLU non-inferiority is not established. The clean SFT adapter is a
limited research preview; scorer-v9 pilot evidence validates the data contract,
not RL improvement.

## System

```text
natural-language specification
            │
            ▼
     policy model rollout
            │  SPICE netlist
            ▼
 parser + structural gates ─────► structured failure event
            │ accepted
            ▼
 sandboxed ngspice-46 execution
            │ measurements
            ▼
 canonical ScoreResult + reward
            │
            ├──► current-policy data labeling
            ├──► GRPO / RLVR training
            └──► paired evaluation and audits
```

The same scorer semantics serve labeling, training, evaluation, rescoring, and
the standalone environment. Repository/package parity tests prevent silent
drift between the training implementation and distributable package.

## Why the verifier is the research contribution

A simulator is deterministic, but that does not automatically make its reward
correct. A policy can satisfy a measured proxy while violating the requested
circuit family: disconnected inputs, decorative feedback, hidden stimuli,
wrong reactive paths, or output offset mistaken for gain modulation.

PhyChip treats policy rollouts as adversarial verifier tests. Complete review of
384 scorer-v8 rollouts found 46 semantically invalid positive-reward circuits,
including 38 that would have received positive advantage. The defect classes
became fail-closed scorer-v9 regressions before any optimizer update.

Under deterministic v9 replay:

- 18 rollouts meet every requested specification;
- 11/48 prompts have mixed K=8 success across 7 circuit families;
- 37/48 have zero observed successes and 0/48 are easy 8/8 groups;
- 181/384 are blocked by anti-cheat admission;
- 382/384 terminate with the assistant EOS and 2 reach the cap;
- all 33 group winners and all exact rows were reviewed;
- no known-invalid v8 circuit remains on the v9 reward frontier.

See the [machine-readable aggregate](poc_experiments/results/rlvr_scorer_v9_pilot_summary.json),
[full diagnosis](poc_experiments/results/rlvr_diagnosis_2026-07-22.md), and
[candidate audit](poc_experiments/results/rlvr_candidate_v8_audit_2026-07-22.md).
Raw prompts, references, completions, and review queues are intentionally not
published.

## Reward contract

For normalized measured specification score `s ∈ [0,1]`:

```text
malformed, unsafe, non-simulating, or family-invalid  -> 0
valid measured specification progress                -> 0.70 * s
all requested specifications met                     -> 1.30
```

Formatting and successful simulation remain observable diagnostics, but scorer
v9 gives them no policy reward. Scientific success is always
`ScoreResult.all_specs_met`, never a scalar threshold.

The scorer combines:

- fenced-netlist extraction and static safety validation;
- parsed top-level device checks outside `.control` and `.subckt` bodies;
- declared-input and load-bearing family topology checks;
- resource-bounded ngspice-46 execution;
- family-specific injected measurement harnesses;
- missing-measurement and non-finite-value fail-closed behavior;
- decomposed diagnostics and content-backed runtime fingerprints.

The exact contract and its limits are documented in
[`REWARD_CONTRACT.md`](phy-chip-env/docs/REWARD_CONTRACT.md) and the
23-family [`CAPABILITY_MATRIX.md`](phy-chip-env/docs/CAPABILITY_MATRIX.md).

## Quick start

```bash
cd phy-chip-env
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
./scripts/install_ngspice46.sh
pytest -q

phy-chip-score \
  --circuit-id noninv-amp \
  --targets '{"gain_db":[19.5,20.5]}' \
  --completion completion.txt \
  --require-success
```

Programmatic use:

```python
from phy_chip_env import score

result = score(
    "noninv-amp",
    {"gain_db": [19.5, 20.5]},
    completion,
)
print(result.to_event())
```

Generated SPICE is untrusted input. The runner uses a new scratch directory,
a minimal environment, pre-execution command/file-I/O checks, CPU/address-space/
file/process/output limits, a wall timeout, and privilege dropping when invoked
as root. See [`SECURITY.md`](phy-chip-env/SECURITY.md).

## Post-training design

The original plan had five stages: physics continued pretraining, semiconductor
continued pretraining, SFT, DPO, and RLVR. Controlled experiments removed the
non-load-bearing stages:

- small-scale physics CPT hurt a healthy SFT reference by about 5 percentage
  points;
- DPO did not establish incremental value;
- clean SFT remains the strongest release candidate;
- the active research path is clean SFT → verifier-audited GRPO.

The RL implementation is accurately described as **Dr.GRPO plus selected DAPO
components**, not full DAPO:

- K=8 on-policy groups and no learned critic;
- `scale_rewards=False`;
- token-level DAPO loss aggregation;
- asymmetric clipping and two policy iterations per group;
- truncation masking and real chat EOS;
- no KL penalty in the bounded verifier study;
- policy loss only on executable SPICE tokens;
- current-policy mixed-success curriculum;
- fail-closed scorer/prompt/policy/data fingerprints.

Native DAPO dynamic resampling, asynchronous rollout infrastructure, and an
overlong soft-penalty schedule are not implemented. The complete design and
claim ladder are in [`posttraining.md`](posttraining.md).

## Model evidence and claim boundary

| Evaluation | Clean SFT | Interpretation |
|---|---:|---|
| AnalogCoder-24 | 21/24 | Adapted raw-SPICE/ngspice proxy, not the official task |
| bench-v1 | 23/40 | 23/23 Level 1, 0/17 Levels 2–3 |
| bench-v2 | 25/46 | Project-authored repaired benchmark; one Level-2 pass |
| Physics-MMLU | 356/785 vs base 365/785 | Non-inferiority not established |

The corrected historical GRPO-v3 candidate scores 16/24, 19/40, and 10/46 and
is rejected. A matched rank-64 smoke moves 3/53 to 6/53, but its paired 95%
confidence interval crosses zero and exact McNemar `p=.375`. It is an
inconclusive positive point estimate, not a generalization result.

The benchmark identity, grading, independence, and contamination boundaries are
documented in the
[`benchmark authenticity audit`](docs/BENCHMARK_AUTHENTICITY_AUDIT_2026-07-22.md).

## What comes next

The closed-loop scaffold, fixed baseline, Codex controller, and machine-readable
gates are implemented. A first 4/4 Codex result was discarded after review found
that its prompt leaked the fixed baseline template. On the corrected four-task
probe, Codex solved 2/4 and used `[8,8,4,3]` verifier calls; fixed TPE solved
4/4 with `[8,4,3,6]`. Paired success delta was `-0.50` with bootstrap 95% CI
`[-1.0,0.0]`. On the two joint successes, call saving was `+1.0` with CI
`[-1.0,3.0]`.

The first readiness gate requires 100 manually approved, replay-valid episodes
across at least 8 families. The separate scale gate requires 500 approved
episodes across all 23 families before multi-turn RLVR is considered. RFT,
rejection SFT, and sequence distillation remain blocked on the audit gate;
on-policy distillation additionally requires authentic teacher token
distributions, and multi-turn RLVR requires authenticated current-policy
provenance and mixed K=8 support.

The exact dependency sequence, exit criteria, commands, and stop rules are in
[`plan.md`](plan.md). More optimizer steps, LoRA rank, or repeated easy
trajectories would not answer the research question.

Two earlier four- and eight-pair development probes informed the v3 protocol,
including opaque controller IDs after one exploratory run leaked hidden values.
Their machine-readable artifacts are not part of this public release, so their
point estimates are not public reproducibility claims.

The preregistered v3 preflight expanded this to 24 paired tasks across 8
families with matched prompts, opaque IDs, adjacent arm execution, immutable
run provenance, write-ahead model-call accounting, semantic hidden-passive
anchors, family-clustered inference, and exact per-success audit. Full feedback
solved 19/24 and blind solved 10/24. The task-level delta was `+0.375`, bootstrap
CI `[0.167,0.583]`, with nine full-only successes and exact McNemar
`p=.003906`. The complete-case sensitivity excludes every pair touched by a
controller error and still gives full 17/17 versus blind 10/17, with 7/0
discordances and `p=.015625`. This is positive task-level evidence, but not a
promoted result: only 5/8 families were positive (`p=.0625` exact sign test),
manual review accepted 28/29 successes as topology-plausible and 27/29 as
component-practical, and 10/48 arms ended in `TimeoutExpired` controller
errors. Audit-adjusted topology-valid success is 18/24 versus 10/24
(`p=.0078125`); component-practical success is 18/24 versus 9/24
(`p=.00390625`). The formal gate failed family-sign, topology, and practicality
criteria; controller reliability is a separate post-hoc execution blocker. The
next valid experiment is a freshly powered passive confirmation after an
infrastructure smoke, with task- and family-level power handled separately—not
a selective v3 rerun. PVT remains untested. See the
[machine-readable comparison](poc_experiments/results/feedback_use_v3_summary_2026-08-03.json)
and [hash-bound audit](poc_experiments/results/feedback_use_v3_manual_audit_2026-08-03.json).
Publication reveals the v3 verifier augmentations, so v3 is retrospective
evidence and a development benchmark only. It cannot serve as a sealed future
confirmation set or as training data.

## Repository map

```text
phy-chip-env/                 standalone Apache-2.0 verifier package
athma-train/                  training scorer, harness mirror, SFT/GRPO code
release/v0.1/                frozen claims, model card, hashes, reproduction
poc_experiments/results/      public aggregate audits and negative results
docs/                         benchmark and data-quality audits
posttraining.md               full post-training architecture and decisions
plan.md                       gated path from feedback use to OOD research
```

## Licenses

- `phy-chip-env` and training-pipeline code: Apache-2.0.
- Released model weights and data: CC-BY-NC-SA-4.0,
  educational/research/non-commercial.
- ngspice and the base model remain under their upstream terms.

See [`DATA_AND_LICENSE_NOTICE.md`](release/v0.1/DATA_AND_LICENSE_NOTICE.md) for
the exact v0.1 boundary.

## Author

Nithin Reddy Govindugari
