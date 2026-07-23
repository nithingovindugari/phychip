# PhyChip Post-Training System

**Status:** living technical record, 2026-07-22
**Positioning:** a modern, verifier-first post-training stack for a physics
environment—not a claim that the current 3B policy is state of the art at analog
circuit design.

## Technical summary

PhyChip's production path is **base model → clean SFT → RLVR with GRPO**. The
original five-stage plan also contained physics CPT, semiconductor CPT, and
DPO. Controlled experiments removed those stages from the load-bearing path:
small-scale CPT hurt the healthy SFT reference by about 5 percentage points,
and DPO never established incremental value. The best release candidate remains
the clean SmolLM3-3B SFT adapter. The RL study is deliberately separate from
the v0.1 release. One scorer-v6 seed completed 140 updates but failed its frozen
final health gate, and manual rollout review found an invalid positive-feedback
group winner. The second seed and paired evaluation were not run. This is a
verifier/health diagnostic, not a positive, null, or negative RL result.

The RL implementation is more accurately called **Dr.GRPO plus selected DAPO
components** than “full DAPO.” It uses K=8 on-policy groups, no critic, no KL
penalty, no reward standard-deviation scaling, DAPO token-level loss
aggregation, asymmetric clipping, two policy iterations per rollout group,
truncation masking, a real chat EOS, executable-token policy-loss masking, and
an externally measured mixed-success curriculum. It does **not** implement
DAPO's native dynamic resampling loop, an overlong soft-penalty schedule, or an
industrial asynchronous rollout architecture such as verl.

The strongest engineering result is the verifier contract. Labeling, training,
evaluation, and auditable rescoring call one canonical training scorer. The
standalone `phy-chip-env` package carries a release-oriented mirror of those
semantics, with repository/package parity tests preventing silent divergence.
ngspice-46 runs inside a resource-limited sandbox. Policy, prompt, simulator,
verifier, data, and adapter identities are fingerprinted. A SHA-bound manual
gate blocks optimization when the scorer or labels change. The current
1,152-rollout audit found real reward hacks before training—wrong loop polarity,
decorative feedback, hidden output sources, non-hysteretic comparators, and an
undeclared internal AC stimulus—and turned each into a regression test.

This is a **SOTA-inspired and unusually audited small-scale setup**, but the
scientific outcome is not yet SOTA. Existing evidence supports SFT learning
executable circuit conventions. It does not prove that RL improves
out-of-distribution circuit generalization, and the completed scorer-v6 seed
cannot answer even the narrower sharpening question because reward validity
failed before comparative evaluation.

Two current-policy pilots then hardened the data contract before any new
training. A 12-prompt scorer-v7 pilot failed its frozen quality gate after
manual review found two full-reward shortcuts. A fresh 48-prompt/K=8 pilot
generated 384 rollouts under scorer v8; complete review found 46 invalid
positive-reward rows, including 38 with positive advantage. Those defects were
converted into scorer-v9 regressions before deterministic replay. Under v9 the
same rollouts contain **11 safe mixed prompts across 7 families**, 18 exact
successes, 37 zero-success prompts, and no easy 8/8 prompt. All 33 group winners
and 18 exact rows were reviewed; none intersected the known-invalid set. The
pilot gate passed, but 11 prompts remain too small for the decisive GRPO study.

## The thesis and the claim ladder

The research thesis is that on-policy RL with a deterministic physics verifier
can improve generalization beyond static imitation. That thesis is split into
three claims so a positive result cannot silently grow into a stronger one:

1. **Format and basic topology acquisition:** SFT teaches the policy to emit
   runnable netlists under the deployment interface.
2. **On-policy sharpening:** RL increases the probability of an already
   reachable successful behavior on held-out specifications from the same
   family and target distribution.
3. **Out-of-distribution generalization:** RL transfers to disjoint numeric
   bands, unseen executable templates, or topology shifts without eroding
   unrelated capability.

The clean SFT result supports claim 1. The attempted two-seed study was designed
for claim 2, but stopped after its first seed failed health/reward-validity
review. It cannot establish claim 3 because target windows overlap and many
executable templates repeat. A later thesis-grade experiment needs a
semantically disjoint target-band or topology holdout, at least three training
seeds, and a matched random-reward control.

## The pipeline we actually use

```text
pinned SmolLM3-3B-Base
        │
        ├── clean, assistant-only SFT
        │     └── 15,660 sandboxed simulator-passing rows
        │
        ├── exact K=8 current-policy labeling
        │     ├── canonical scorer + ngspice-46
        │     ├── policy/prompt/verifier hashes
        │     └── keep only 0 < pass_count < 8 prompts
        │
        ├── Dr.GRPO + selected DAPO components
        │     ├── executable-token policy loss
        │     ├── live reward/entropy/gradient health gate
        │     └── planned two seeds; seed 42 rejected before seed 7
        │
        └── matched evaluation on 193 held-out specifications
              ├── sampled K=8 pass@1 estimate
              ├── greedy pass@1
              ├── paired specification-cluster bootstrap
              └── manual winner/discordance audit
```

RLVR uses a deterministic programmatic verifier rather than a learned reward
model. GRPO samples several responses for one prompt and uses their relative
rewards as a baseline, avoiding PPO's learned value model. These are the core
reasons the task is a natural RLVR proof case. See `[[wiki: RLVR]]` and
`[[wiki: GRPO]]`.

## Decisions by stage

| Stage | Decision | Evidence and consequence |
|---|---|---|
| Base | Keep `HuggingFaceTB/SmolLM3-3B-Base`, pinned to revision `d78a42f…` | It stays inside the project's 1–3B scope and is the only base with the full causal chain. Qwen3.5-4B is an architecture control, not a replacement. |
| Physics CPT | Dropped from Phase 1 | At the tested LoRA/token scale it acted as a distribution regularizer, not a knowledge injector, and hurt a healthy SFT model by about 5 pp. This does not rule out large-scale full-parameter CPT. |
| Semiconductor CPT | Dropped from Phase 1 | No evidence justified the cost after the CPT ablation. |
| SFT | Required cold start | It teaches chat format, interface discipline, executable SPICE, and basic topology. SFT loss is applied only to assistant response tokens; prompts are masked. This matches the role described in `[[wiki: SFT]]`. |
| DPO | Not load-bearing | DPO is a stable offline preference method, but this project never produced evidence that it improved the verified physics objective. Static preference data also cannot replace fresh verifier interaction. See `[[wiki: DPO]]`. |
| RLVR/GRPO | Bounded seed rejected; fresh data pilot stopped before training | Seed 42 failed the frozen entropy and rollout-validity gates. The later scorer-v7 K=8 pilot shrank from 5 machine-mixed to 3 manually safe prompts after two new full-reward exploits. No result is accepted without semantic reward review and paired held-out evaluation. |
| On-policy distillation | Deferred comparison, not silently substituted | OPD is promising for lightweight models and may be more compute-efficient, but true logit distillation requires teacher distributions. Answer-only synthetic trajectories are sequence-level teacher data, not OPD. See `[[wiki: On-Policy Distillation]]`. |
| verl / asynchronous RL | Deferred | A one-A100, 3B, bounded experiment does not yet justify a framework migration. TRL 1.6 supplies the needed algorithm fields and keeps the current scorer/eval path auditable. verl becomes relevant when rollout throughput, multi-GPU scheduling, or long-horizon environments dominate. |

## Clean SFT is the baseline RL must beat

The source corpus contained 23,026 rows. Static tokenizer/template checks kept
16,177. The production sandbox admitted 15,660 simulator-passing rows and
quarantined another 517 safety failures after 6,849 rows failed the static
gate. The exact training file has zero detected
benchmark leaks under the recorded exact, 8-gram, and normalized-reference
checks, but it is not 15,660 independent circuit ideas: 7,537 rows belong to a
repeated normalized-template cluster. There are 9,442 unique normalized
templates; the duplicate excess is 6,218 rows.

The pinned clean SFT run used two epochs and 1,958 optimizer steps, processed
20.42M tokens, and ended at loss 0.4677 and assistant-token accuracy 0.9394.
Those are optimization diagnostics, not quality metrics. Held-out native-chat
scores are:

| Evaluation | Clean SFT | Interpretation |
|---|---:|---|
| AnalogCoder raw-SPICE proxy | 21/24 | Strong Level-1 floor, one below the historical 22/24 target |
| bench-v1 | 23/40 | 23/23 Level 1, 0/17 Levels 2/3 |
| bench-v2 | 25/46 | 24/28 Level 1, 1/13 Level 2, 0/5 Level 3 |
| physics-MMLU | not established non-inferior | No general-capability preservation claim |

The Qwen3.5-4B same-data SFT control scored 20/24, 21/40, and 24/46, versus a
matched Smol run at 23/24, 23/40, and 23/46. The automatic total favors Smol
69/110 to 65/110; manual topology plausibility favored Qwen 53/110 to 38/110.
This disagreement is itself useful: automatic simulator success and engineering
quality are not interchangeable.

## Exact RL data selection: difficulty is measured, not guessed

GRPO needs within-prompt reward variance. If all K completions fail or all pass,
the group has no useful relative objective signal. PhyChip therefore labels
candidate prompts with the **current policy, exact prompt contract, exact
scorer, and K=8 sampling regime**, then admits only mixed groups:

```text
trainable(prompt) := 0 < pass_count(prompt, current_policy, K=8) < 8
```

This is an external curriculum and a data gate. It resembles DAPO dynamic
sampling in purpose, but it is not full online drop-and-resample dynamic
sampling. Every admitted row carries:

- policy hash;
- prompt-contract hash;
- verifier hash;
- scorer-semantics version;
- K and exact success/reward vectors;
- stable specification identity.

The trainer runs with `--require-trainable-label` and fails closed on a missing
or mismatched field. Raw label files intentionally remain `unlabeled`; the
merge step derives and binds the `trainable_mixed_success` status. This prevents
an accidental launch on all candidates or on labels made by an older verifier.

The historical scorer-v6 replay contains 144 prompts × 8 completions
= 1,152 rollouts, 135 exact successes, 70 mixed-success prompts, 74 zero-success
prompts, and 12 capped completions. Its materialized SHA-256 is
`11ea5f0d04e412a117aa7eedcfc680efc8170ad936a0139b9a4e999374f452a5`.
The final pre-training decision-delta audit approved this exact hash before the
first optimizer update. Post-training review subsequently found a new
multi-hop output-driver escape, so the artifact remains historical evidence
under scorer-v6 and is invalid for any scorer-v7+ run.

The 70 rows are not family-balanced:

| Circuit family | Label candidates | Mixed-success rows | Unique reference topologies in mixed rows | Exact successes across K=8 labels |
|---|---:|---:|---:|---:|
| Inverting amplifier | 32 | 20 | 15 | 26 |
| Non-inverting amplifier | 32 | 28 | 21 | 78 |
| Summing amplifier | 32 | 21 | 15 | 28 |
| Sallen-Key filter | 32 | 1 | 1 | 3 |
| Schmitt trigger | 16 | 0 | 0 | 0 |
| **Total** | **144** | **70** | **52** | **135** |

Thirty-eight of the 70 mixed rows had only one success in eight samples; 15
had two. This is useful relative-reward signal, but it is concentrated in
three amplifier families. All 70 specification hashes and normalized prompts
are unique, but the references contain only 52 unique topology fingerprints;
the most frequent topology occurs five times. Thus the pool is not suffering
from exact prompt duplication, but its effective design diversity is smaller
than its row count. The 193-spec holdout has 69 unique reference-topology
fingerprints and zero overlap with training in specification, normalized-
prompt, row, or topology hashes. This is meaningful structural separation
inside familiar families, but target bands still overlap and Schmitt has only
three holdout rows. A zero-success K=8 observation does not prove a task is
unreachable; it means only that the frozen sampling attempt observed no
successful trajectory. Labeling used temperature 0.8 while training uses 0.9,
so the live 20-group variance gate—not the offline label alone—must confirm
that the actual training regime still has usable signal.

## The canonical reward

One canonical scorer is shared by training, labeling, evaluation, and auditable
rescoring. `phy-chip-env` mirrors the same versioned semantics and is checked by
parity tests. For a requested specification set with normalized partial score
`s ∈ [0,1]`:

```text
malformed, unsafe, nonsimulating, or family-invalid:  reward = 0
valid measured specification progress:               reward = 0.70 s
all requested specifications met:                    reward = 1.30
```

Correctness is therefore gated. Formatting and successful simulation receive
no policy reward and cannot compensate for a topology exploit. Partial shaping exists only after the
sandbox, structural checks, family semantics, and measurement harness pass.
This follows the same general lesson as gated reward aggregation: auxiliary
quality should not pay for being wrong.

The `ScoreResult` decomposition records simulation status, anti-cheat status,
requested/measured/met objective counts, normalized spec score, exact success,
and structure fields. Training logs both rollout-level fractions and the
fraction of complete K-sized groups with reward, spec-score, and exact-objective
variance. This distinguishes “the reward moved” from “the useful objective
moved.”

## Reward hacking: attacks found and the defenses they created

Reward hacking is a policy finding a way to increase the measured reward
without satisfying the intended task. In RLVR it appears as verifier gaming,
format exploitation, repetition, or length exploitation. See
`[[wiki: Reward Hacking]]`.

The table below records 11 discovered failure modes and defenses. It should not
be summarized as “11 reward exploits”: it also includes one verifier false
negative, a prompt-contract mismatch, and a sandbox-concurrency defect.

| Failure found in PhyChip | Why it was invalid | Current defense |
|---|---|---|
| Independent voltage source across `Vout` | Clamps the measured answer instead of designing a circuit | Reject independent sources directly driving output |
| Fixed-gain “gain-trim” stage | Meets gain while ignoring the trim control | Require measured nonzero trim response |
| Positive-feedback inverting/summing stages | AC magnitude looked correct while the transfer sign was wrong or unstable | Structural loop-polarity checks plus dynamic signed-gain measurements |
| Decorative Schmitt divider | A physical-looking divider existed but the comparator ignored it | Require the output-driving stage to consume the feedback node |
| Ordinary comparator labeled Schmitt | High and low thresholds were equal | Require finite `vth_high > vth_low` before any partial reward |
| Hidden internal `AC 1` source | Internal stimulus made the named input port irrelevant | Reject nonzero AC stimuli whose source name and driven node do not match declared ports |
| Single-edge feedback assumption | Valid two-resistor series feedback was hard-blocked | Permit an all-resistor feedback path that cannot traverse input, reference, ground, or supply nodes |
| False design comments | Correct executable deck carried stale numeric claims that sequence reward would reinforce | Mask full-line comments, `.title` prose, inline comments, and outside-fence prose from policy loss |
| Endless repeated comments/titles | Every rollout hit the length cap in an earlier experiment | Correct chat EOS, retain a finite 3,072-token cap, and mask truncated completions |
| Train/eval prompt mismatch | Training used completion format while evaluation used chat format | One byte-identical shared chat renderer and prompt hash |
| Shared-UID sandbox concurrency failure | `RLIMIT_NPROC=0` applied per real UID caused concurrent ngspice children to fail | Serialize sandboxed ngspice execution while retaining the no-fork boundary |

The lesson is not that a verifier can be made “perfect.” The lesson is that
verifier development is empirical red teaming: preserve generations, inspect
the high-reward frontier, convert every discovered exploit into a test, version
the scorer, and invalidate all old labels when semantics move.

## GRPO, Dr.GRPO, and the DAPO subset

The attempted bounded recipe is frozen in
[`plan/grpo_powered_pod.sh`](plan/grpo_powered_pod.sh) and
[`train_lora_grpo.py`](athma-train/scripts/train_lora_grpo.py).

| Component | Exact implementation | Status |
|---|---|---|
| On-policy group sampling | K=8 completions per prompt, temperature 0.9 during training | Implemented |
| Critic-free baseline | Group-relative advantages; no learned value model | Implemented |
| Dr.GRPO reward scaling | `scale_rewards=False`, so group reward standard deviation does not rescale the update | Implemented |
| DAPO token loss | `loss_type="dapo"` | Implemented |
| Clip-higher | `epsilon_high=0.28` | Implemented; meaningful because `num_iterations=2` can move ratios away from 1 |
| KL penalty | `beta=0` | Deliberately removed for this verifier-grounded bounded run |
| Multiple updates per rollout | `num_iterations=2` | Implemented |
| Truncated-output handling | `mask_truncated_completions=True` | Implemented |
| Completion ceiling | 3,072 tokens with `<|im_end|>` as assistant EOS | Implemented; intentionally not unlimited |
| Dynamic sampling | Exact mixed-success prompts selected before launch | Partial/external approximation; no online DAPO resampling loop |
| DAPO soft overlong penalty | None | Not implemented; current defense is EOS + hard cap + truncation mask |
| Async inference/training | None; synchronous TRL path on one A100 | Not implemented |
| External-engine importance correction | None beyond TRL's policy-ratio/clipping path | Not required for this synchronous trainer; measure mismatch before adding it |

The wiki records DAPO's overlong penalty as an important mitigation for
length-hacking reasoning models. PhyChip does not copy it blindly: all 12
observed caps were pathological repetition, so the current choice is to give
truncated trajectories no policy loss and retain a hard safety ceiling. A soft
cache-region penalty remains a future ablation, not a claimed feature.

## Executable-token policy loss

SFT's “response-only loss” means the prompt is masked but every assistant token
is normally trained. That is not sufficient for RL here: a scalar sequence
reward would assign the same advantage to correct circuit lines and incorrect
comments in the response.

PhyChip keeps the full on-policy completion in attention/context but applies the
policy-gradient loss only to:

- the SPICE fence;
- executable non-comment lines;
- line separators needed by the deck;
- the closing fence;
- the actual assistant `<|im_end|>` token.

It masks outside-fence prose, full-line `*` comments, `.title` payloads, and
semicolon comments. Malformed or unfenced completions retain an all-token mask
so negative advantage can still suppress the failure. Decoded text must
round-trip exactly. Generated token IDs need not: adjacent BPE IDs can decode
to text that retokenizes as one merged ID, so the trainer aligns actual and
canonical token sequences and fails closed on ambiguous mixed-mask spans. The
mask is attached through TRL's `tool_mask`, and DAPO's token denominator is
recomputed from active tokens across processes. W&B logs
`policy/executable_token_fraction`; raw completions remain in a private local
rollout-audit file and are never sent to W&B.

This is a targeted credit-assignment choice, not a claim that comments vanish
immediately. Masked comments still appear in the conditioning history and may
continue to be generated indirectly. The smoke gate must therefore inspect
comment rate, EOS rate, cap rate, and manual output quality.

## LoRA and optimization choices

The GRPO adapter uses LoRA rank 64 over attention and MLP projection modules.
The powered launcher currently uses alpha 64, learning rate `1e-5`, cosine
decay, 5% warmup, bfloat16, non-reentrant gradient checkpointing, per-device
batch 1, generation batch 8, and gradient accumulation 8. Rank 64 is retained
because the prior matched rank-64 result moved 3/53 to 6/53 but remained
statistically inconclusive; there is no evidence that rank 128 is the missing
lever. More rank cannot repair sparse rewards, invalid labels, or a capability
ceiling.

The powered schedule was derived from the final mixed-success row count rather
than a hard-coded historical step count. With 70 scorer-v6 mixed groups, TRL
generates 70 fresh K=8 groups (560 rollouts) per seed and reuses each group for
two policy iterations. That produces 140 optimizer steps; it does **not** mean
140 unique prompts or 1,120 fresh rollouts. Seed 42 completed that schedule;
seed 7 did not start after the final gate/audit rejection. This was
intentionally bounded, not a hyperparameter sweep, and more epochs over the
same 70 prompts would increase memorization risk rather than repair the
missing-family coverage.

## Safety, reproducibility, and provenance

ngspice is executable code. Every call is therefore treated as untrusted:

- ngspice version pinned to 46;
- temporary isolated working directory;
- syntax and directive validation;
- subprocess timeout;
- CPU, memory, file-size, and process limits;
- privilege drop when available;
- serialized execution where `RLIMIT_NPROC` is UID-scoped;
- gold-reference, adversarial, timeout, and package/repository parity tests.

Every result is accompanied by the base revision, adapter chain and hashes,
prompt-contract hash, verifier hash, scorer version, ngspice banner, generation
seed and temperature, K, length limit, and artifact SHA-256. Cached evaluation
is reused only if those identities match. The environment and trainer maintain
separate tests but share semantics.

## W&B: only metrics that can change a decision

W&B receives scalar diagnostics and provenance, not prompts, completions,
netlists, private rows, code, logs, or model artifacts. Relevant panels are:

- exact reward mean/std and `frac_reward_zero_std`;
- reward/spec/exact-objective variance by K-sized group;
- exact, partial, gate-only, nonsimulating, and blocked advantage-winner mix;
- anti-cheat block rate and measurement coverage;
- entropy, gradient norm, policy-ratio clip region, KL/drift diagnostics exposed
  by TRL;
- completion length and clipped ratio;
- executable-token fraction;
- learning rate, step time, token count, and GPU utilization;
- prompt/policy/verifier hashes and resolved algorithm configuration.

The live gate waits for 20 fresh generation groups and fails the run if entropy
falls below 0.15, gradients are absent/non-finite/explosive, clip-region mean
exceeds 0.25, completion clipping exceeds 5%, too many groups have zero reward
variance, invalid rollouts win positive advantage, spec signal does not
dominate winners, or anti-cheat blocking suggests the policy is fighting the
verifier.

## Evaluation and statistical decision rule

The 193-spec holdout is disjoint from the training rows and evaluated under the
same chat prompt, termination, simulator, and scorer. Each policy gets:

1. K=8 stochastic sampling at the same temperature/top-p/seed;
2. greedy decoding;
3. paired per-spec comparison against the frozen clean-SFT baseline;
4. a family-stratified specification bootstrap 95% confidence interval for the
   primary K=8 mean-success-rate comparison;
5. a paired sign-flip p-value;
6. a paired greedy comparison as a secondary view (its current bootstrap is
   not family-stratified);
7. a manual audit of all winners, treatment/baseline discordances, caps, and
   anti-cheat events.

The primary sampled metric is the mean per-spec success rate, which estimates
pass@1 from eight trials while preserving specification as the statistical
unit. Greedy pass@1 is a secondary deployment view. Two seeds must point in the
same direction before the effect is treated as stable. Even then, the bounded
study reports familiar-distribution sharpening only.

The sample size is not a universal “powered” guarantee. For 193 specifications,
the normal-approximation 80%-power sensitivity of the primary paired estimand is
`2.802 × SD(D_spec) / sqrt(193)`: plausible paired standard deviations of
0.10–0.30 imply a minimum detectable effect of roughly 2–6 percentage points.
A provisional baseline-only calculation suggests about 3.4 pp, but the actual
paired variance is observable only after valid trained outputs exist. The
interval is also conditional on the two observed training seeds; two seeds do
not estimate the population distribution over training runs. Greedy and pass@8
binary comparisons are less sensitive than the K=8 mean-rate primary.

The strongest previous rank-64 matched result was 3/53 → 6/53, a +5.66 pp point
estimate whose paired 95% CI was [-1.89,+13.21] pp and whose McNemar p-value was
0.375. That is not evidence of a win. The July 4 study was also not decisive:
baseline pass@1 0.075 versus pooled GRPO 0.057 on only 40 holdout prompts, with
a train/eval prompt mismatch and 100% capped rollouts. The correct conclusion
is that those runs did not execute a design capable of answering the thesis.

## Powered seed-42 outcome

The scorer-v6 seed completed all 70 fresh K=8 groups and 140 optimizer updates.
Its optimization diagnostics were mostly healthy: reward variance 70/70,
exact-objective variance 54/70, finite positive gradients 100%, cap rate 1/560,
and final entropy 0.1989. The final health gate still rejected it because the
minimum entropy reached 0.1486 below the frozen 0.15 floor at step 100.

Manual inspection was load-bearing. Across all 560 rollouts, one summing-
amplifier completion was an invalid positive-feedback group winner. The VCVS
sensed the summing node on its positive terminal and reached `Vout` through a
three-resistor chain, bypassing scorer-v6's local output-driver check. A second
non-inverting completion ignored `Vin` and earned the 0.30 simulation floor,
although it had negative advantage. Independent audits found no other invalid
positive-advantage rows; stale numeric/topology comments were widespread even
when executable circuits were valid.

The reward diagnostic was written before scorer changes. Scorer-v7 follows
safe transitive resistor paths, checks non-inverting input/feedback consumption
and polarity, and carries both failures as regressions. Scorer-v8 additionally
requires a finite positive measured buffer transfer and traces gain-trim
controlled-source producers transitively. Scorer-v9 adds parsed top-level
device admission, declared-port checks, load-bearing family topology, measured
gain modulation, and no positive simulation floor. The package passes 175/175
tests. Seed 7 and paired model evaluation
were not run, so this outcome is not evidence that RL worked, failed, or
regressed.

## What is modern here—and what is not

The strongest “state-of-the-art” aspects are methodological:

- current-policy trainability measurement instead of guessed difficulty;
- one training scorer across label/train/eval, plus a parity-tested package
  mirror;
- Dr.GRPO and the useful DAPO pieces supported by the pinned framework;
- exact prompt/EOS parity;
- executable-token credit assignment;
- semantic reward gates combining topology and dynamic physics checks;
- SHA-bound manual review before optimization;
- live failure gates and private rollout preservation;
- paired, specification-level statistics with explicit claim boundaries.

The stack is not yet an industrial SOTA system. It has one synchronous A100,
only one completed-but-rejected seed, no matched random-reward arm in the
bounded run, no native DAPO resampling, no async rollout engine, and no
demonstrated OOD RL gain. Calling
the outcome SOTA before those gaps close would weaken the project. The credible
claim is: **PhyChip implements a modern, verifier-first RLVR research pipeline
and documents the failure modes most small RL projects hide.**

## Current state and next decisions

At the time of this update:

- clean SFT is the v0.1 research-preview candidate;
- `phy-chip-env` is the primary career/research artifact;
- scorer-v9 and the aligned executable-token mask pass focused and full tests;
- all 1,152 preserved K=8 generations were manually reviewed under their
  frozen scorer-v6 labels; those labels are now historical only;
- the scorer-v6 completion-preserving replay has 135 exact successes and 70
  mixed-success groups, but it cannot seed a scorer-v7 run;
- the final manual audit of every v4→v6 changed decision approved the bound
  pre-training scorer-v6 artifact SHA;
- the frozen clean-SFT baseline is complete: K=8 has 225/1,544 successes and
  pass@8 116/193; greedy is 15/193, both under scorer-v6;
- seed 42 completed 70 K=8 groups / 560 rollouts / 140 updates, then failed the
  final entropy gate (`0.1486 < 0.15`);
- full rollout review found one invalid positive-feedback group winner and one
  lower-impact non-inverting unused-input admission gap;
- seed 7 and paired evaluation were not run; Prime has zero active pods;
- scorer-v9 passes 175 standalone tests, 45 focused training-tree tests,
  repository/package parity, 22 known attacks, 44/53 corrected historical
  references, and deterministic replay of the 384-rollout pilot;
- the scorer-v9 replay yields 11 safe mixed prompts across 7 families, 18
  exact successes, 37 zero-success prompts, and no 8/8 prompt;
- all 33 v9 group winners and 18 exact rows were manually reviewed with zero
  overlap against the known-invalid v8 set; no GRPO training occurred;
- private source commit `b0e2e7d` is pushed for review, but nothing is merged,
  tagged, made public, or published as a model/package.

The ordered next steps are:

1. Regenerate the release manifest/checksums from the intended clean commit and
   finish the public v0.1 review bundle.
2. Preserve scorer-v7/v8 outputs as diagnostic evidence; never reuse their
   labels for training under v9.
3. Scale the frozen v9 contract to 150--250 new evaluation-blind references,
   label at K=8, and manually inspect every exact and group-winning circuit.
4. Freeze a disjoint holdout and power calculation before optimization.
5. Run GRPO only after the scaled pool retains enough safe mixed prompts; use
   at least three seeds and matched controls.
6. Keep the thesis-grade OOD study separate; do not relabel familiar-family
   sharpening as generalization.

The data target is now source-grounded rather than guessed. A scorer/optimizer
smoke needs 16--32 manually safe current-policy mixed prompts. The first
defensible bounded causal study targets 150--250 across 8--12 families and at
least 100 topology fingerprints. The stronger generalization thesis targets
500+ training prompts plus a sealed 300--500-prompt topology/numeric-disjoint
holdout. Public examples range from deliberately selected one-shot prompts to
8K--40.3K-prompt campaigns; there is no universal prompts-per-parameter rule.

## Primary implementation and evidence files

- [`README.md`](README.md) — current project truth and release state
- [`progress.md`](progress.md) — timestamped experimental history
- [`learning.md`](learning.md) — failure-derived engineering lessons
- [`research.md`](research.md) — methodology findings and claim boundaries
- [`docs/DATA_QUALITY_AUDIT_2026-07-19.md`](docs/DATA_QUALITY_AUDIT_2026-07-19.md) — superseding scorer-v6 curriculum quality decision
- [`docs/BENCHMARK_AUTHENTICITY_AUDIT_2026-07-22.md`](docs/BENCHMARK_AUTHENTICITY_AUDIT_2026-07-22.md) — exact source/grader/independence audit of the three circuit matrices
- [`poc_experiments/results/rlvr_diagnosis_2026-07-21.md`](poc_experiments/results/rlvr_diagnosis_2026-07-21.md) — pre-training reward diagnostics
- [`poc_experiments/results/rlvr_diagnosis_2026-07-22.md`](poc_experiments/results/rlvr_diagnosis_2026-07-22.md) — complete scorer-v7/v8 diagnosis and scorer-v9 deterministic replay
- [`poc_experiments/results/rlvr_scorer_v9_pilot_summary.json`](poc_experiments/results/rlvr_scorer_v9_pilot_summary.json) — machine-readable public pilot aggregate
- [`poc_experiments/results/rlvr_data_size_research_2026-07-22.md`](poc_experiments/results/rlvr_data_size_research_2026-07-22.md) — primary-source prompt-scale survey and staged PhyChip targets
- [`athma-train/athma_train/env/rlvr_scorer.py`](athma-train/athma_train/env/rlvr_scorer.py) — canonical training scorer
- [`phy-chip-env/src/phy_chip_env/scorer.py`](phy-chip-env/src/phy_chip_env/scorer.py) — packaged scorer mirror
- [`athma-train/athma_train/grpo_contract.py`](athma-train/athma_train/grpo_contract.py) — shared prompt and EOS contract
- [`athma-train/scripts/train_lora_grpo.py`](athma-train/scripts/train_lora_grpo.py) — GRPO/Dr.GRPO/DAPO trainer
- [`athma-train/scripts/check_grpo_train_health.py`](athma-train/scripts/check_grpo_train_health.py) — live fail-closed health gate
- [`plan/grpo_powered_pod.sh`](plan/grpo_powered_pod.sh) — frozen powered experiment
- [`phy-chip-env/docs/CAPABILITY_MATRIX.md`](phy-chip-env/docs/CAPABILITY_MATRIX.md) — exact verifier capabilities and limits
