# PhyChip execution plan

## Objective

Test whether a small circuit-design policy can learn to use deterministic
ngspice feedback to improve analog designs beyond static imitation, then test
whether that learned revision behavior transfers to sealed circuit-design
problems.

The target is not a generic coding agent. The target is a verifier-grounded
research loop:

```text
specification + failing circuit
              -> propose revision
              -> run scorer-v9/ngspice
              -> observe measurements and failure classes
              -> revise
              -> preserve an auditable trajectory
```

Autonomous chip research is earned only when this loop produces repeatable,
OOD design improvement under fixed budgets and human-auditable evidence.

## Current truth

- Clean SFT is the strongest model candidate.
- The scorer-v9 environment is the strongest project artifact.
- Historical RL candidates do not establish improvement.
- The old four-task closed-loop probe favored fixed TPE over Codex and does not
  admit training.
- Earlier four- and eight-pair development probes established the paired
  mechanics and exposed task-ID leakage. Their machine-readable artifacts are
  not part of the public release, so they are history rather than reproducible
  public evidence.
- `feedback_use_v3.json` is a preregistered 24-pair preflight across 8 circuit
  families, not a powered confirmatory study. Full feedback solved 19/24 and
  blind solved 10/24: delta `+0.375`, task bootstrap CI
  `[0.167,0.583]`, and exact McNemar `p=.003906`. The family-clustered CI was
  `[0.167,0.625]`, but only 5/8 families were positive and the exact family
  sign test missed its gate at `p=.0625`. Exact review of all 29 successes
  also rejected one full-feedback buffer topology and two circuits on
  component practicality. Ten of 48 arms ended in controller errors. Excluding
  all 7 error-touched pairs leaves full 17/17 versus blind 10/17, with 7/0
  discordances and exact `p=.015625`. This is positive task-level preflight
  evidence with a closed promotion gate, not a training or autonomy result.
- There are zero manually approved feedback-use trajectories.

## Dependency sequence

```text
v0.1 environment release
  -> feedback-use development experiment
  -> sealed benchmark expansion
  -> 100 approved trajectories across >=8 families
  -> supervised post-training comparison
  -> 500 approved trajectories across all 23 families
  -> conditional multi-turn RLVR
  -> sealed OOD transfer evaluation
  -> autonomous-research candidate
```

Do not skip a gate by adding optimizer steps, repeating easy tasks, or changing
the benchmark after seeing results.

## M0 — Freeze v0.1

Exit criteria:

- Release evidence and hashes correspond to one clean source commit.
- Public documentation states the negative and inconclusive results honestly.
- The environment remains Apache-2.0; weights and released data retain the
  documented CC-BY-NC-SA-4.0 posture.

Current status: source work exists locally on
`feature/autonomous-chip-research`; release metadata must be regenerated only
after a future authorized commit. Do not publish yet.

## M1 — Prove feedback use

Run the same model, frozen task, rollout identifier, and verifier budget in two
arms:

- full feedback: receives measurements, requested bounds, failure classes, and
  score after every simulation;
- feedback blind: receives the prior circuit but none of those diagnostics.

Commands:

```bash
cd phy-chip-env
phy-chip-env research-batch \
  --benchmark benchmarks/feedback_use_v3.json \
  --controller codex \
  --output-directory results/feedback-use/full \
  --max-workers 1

phy-chip-env research-batch \
  --benchmark benchmarks/feedback_use_v3.json \
  --controller codex-blind \
  --output-directory results/feedback-use/blind \
  --max-workers 1

phy-chip-env feedback-compare \
  --benchmark benchmarks/feedback_use_v3.json \
  --full-feedback-directory results/feedback-use/full \
  --feedback-blind-directory results/feedback-use/blind
```

Every episode begins with a frozen, scorer-replayed failing observation. That
observation consumes one verifier call but no controller call. Episode logs are
hash-chained and revalidated from exact task and circuit bytes.

Exit criteria:

- zero accepted known reward attacks;
- manually reviewed successful circuits are topology-plausible;
- paired full-minus-blind success has a bootstrap 95% CI lower bound above 0.

Current result: failed. Full feedback and blind both solved 4/4, and all eight
successful circuits were topology-plausible on manual review. The frozen
failures exposed their corrective values too directly, so the model did not
need simulator diagnostics.

The eight-task hidden-parasitic follow-up produced three full-only successes
and zero blind-only successes. It also cut verifier calls from 38 to 21. The effect is
promising, but three discordant pairs give exact two-sided McNemar `p=.25`.
The gate therefore remains closed even though the bootstrap interval clears
zero. Do not collect training trajectories from either evaluation split.

The 24-pair, eight-family v3 preflight produced nine full-only successes and
zero blind-only successes, but its family sign test was `p=.0625`. Manual
review accepted 28/29 successful circuits as topology-plausible and 27/29 as
component-practical. Counting only topology-valid successes gives 18/24 versus
10/24 (`p=.0078125`); counting only component-practical successes gives 18/24
versus 9/24 (`p=.00390625`). The formal gate fails family-sign, topology, and
practicality criteria. Separately, 10 `TimeoutExpired` controller errors are a
post-hoc execution blocker: the run used four concurrent pairs and a 300-second
controller timeout, but this run does not identify concurrency as the cause.
These evaluation trajectories remain permanently ineligible for training.

The four-task set is development-only and cannot satisfy the promotion gate,
even if its bootstrap interval looks positive. A negative result means improve
the observation or controller contract before training; it does not mean
collect more trajectories from a broken loop.

## M2 — Expand and seal the benchmark

Build independent task variants along four axes: numeric target band, initial
failure mode, topology variant, and circuit family. Freeze generation seeds,
task bytes, scorer fingerprint, split assignments, and protocol hashes before
running the model.

Required split roles:

- development: prompt and observation-contract debugging;
- trajectory collection: training-eligible only after human approval;
- sealed OOD: never shown to the teacher or training policy.

Exit criteria: enough paired tasks for a meaningful confidence interval, at
least 8 circuit families represented in the audit pool, and signed split
manifests with no AnalogCoder content.

Current status: the 24-pair v3 preflight covers 8 families and validates the
paired machinery, family-aware statistics, exact per-success audit, semantic
hidden-passive anchors, immutable run manifest, and write-ahead call ledger.
It is not powered confirmation. The frozen 48/96 figures are planning floors,
not a complete McNemar power calculation: the next design must preregister
full-only and blind-only discordance assumptions, alpha, target power, and
separate family-level power. Retaining only 8 families cannot strengthen the
family sign test by adding more tasks inside each family. PVT perturbations
remain a separate later benchmark and must not be claimed from passive-only
tasks.

## M3 — Audit 100 trajectories

Collect exact feedback-use episodes, then manually approve or reject each one.
Approval must bind the task hash, initial observation, all proposed netlists,
all scorer events, controller protocol/model identity, scorer/runtime
fingerprint, and final outcome. Codex CLI does not expose an authenticated
backend model revision; these remain teacher trajectories and cannot satisfy
current-policy RLVR provenance.

The implemented readiness gate remains closed until there are at least 100
approved episodes across 8 families. Do not create a training export before
the approved schema and audit are real; exporting zero approved trajectories
would only freeze a speculative format.

Exit criteria:

- >=100 approved, replay-valid episodes;
- >=8 circuit families;
- no benchmark/train contamination;
- empirical success and failure trajectories, not repeated copies;
- reviewer agreement and rejection reasons reported.

## M4 — Compare supervised uses of trajectories

Use the same base checkpoint, data budget, split, and evaluation harness for:

1. clean SFT control;
2. rejection SFT on successful final circuits;
3. trajectory SFT / agentic RFT on observation-action traces;
4. sequence-level distillation.

Do not label sequence distillation as on-policy distillation. True on-policy
distillation additionally requires authenticated teacher token distributions
for student-sampled prefixes; Codex CLI text trajectories do not provide them.

Exit criteria: a preregistered method beats clean SFT on paired sealed
evaluation without regressing syntax, topology validity, or simulator safety.

## M4.5 — Conditional environment-aware OPD/MOPD research track

Treat this as a post-M4 controlled comparison, not as an interview-critical
feature or a substitute for the feedback-use gate.

The research question is whether dense teacher supervision on states visited
by the student, combined with action-conditioned prediction of ngspice
observations, improves sample efficiency and sealed-task performance beyond
trajectory SFT and GRPO.

Keep the terminology exact:

- trajectory or sequence distillation trains on teacher text and does not
  become on-policy merely because the teacher interacted with ngspice;
- true on-policy distillation (OPD) samples from the current student and needs
  authenticated teacher probabilities on those student-sampled tokens;
- multi-rollout OPD uses successful and failed sibling rollouts for the same
  prompt to construct a peer-conditioned teacher signal;
- multi-teacher OPD is a later capability-integration method and is not
  justified until independently useful circuit-family or skill specialists
  exist.

Codex CLI trajectories do not expose authenticated teacher token
probabilities. They may support trajectory SFT, rejection SFT, and
sequence-level distillation, but they cannot support a true OPD claim. An OPD
experiment therefore requires a self-hosted compatible teacher, a teacher API
that exposes the required sampled-token probabilities, or a separately
validated cross-tokenizer method.

The candidate environment-aware objective is:

```text
student action a_t ~ policy(history h_t)
ngspice observation o_{t+1} = environment(state s_t, action a_t)

loss = action OPD/MOPD loss
     + lambda_env * next-observation prediction loss
     + verified outcome objective
```

Train only on policy action tokens for OPD/MOPD. Predict structured failure
classes with cross-entropy and normalized numeric measurements with a robust
regression loss; do not train on unfiltered raw stdout. The real scorer result
remains the correctness gate and cannot be replaced by teacher confidence or
world-model likelihood.

Run a matched six-arm study only after M3/M4 data gates are real:

1. clean SFT;
2. trajectory SFT;
3. GRPO;
4. single-teacher OPD;
5. multi-rollout OPD using K=8 verified sibling attempts;
6. multi-rollout OPD plus action-conditioned observation prediction.

Hold the student checkpoint, task splits, simulator-call budget, generated
tokens, prompt/tool contract, and scorer fixed. Measure sealed success,
simulator calls to success, normalized specification error, topology validity,
next-observation error, failure-class accuracy, calibration, reward-attack
success, and held-out-family transfer.

Promotion requires the environment-aware arm to beat the strongest simpler
control on paired sealed evaluation without increasing reward attacks or
regressing topology validity. If trajectory SFT matches it, keep trajectory
SFT. Do not introduce a new transformer, MoE, graph encoder, or multi-teacher
system until this objective-level ablation demonstrates that additional model
complexity is necessary.

## M5 — Scale evidence, then consider RLVR

The strict scale gate requires >=500 approved episodes across all 23 supported
families. The legacy `trajectory_pilot` name remains a compatibility alias for
this strict gate; `trajectory_audit_pilot` is the 100/8 gate.

Only after that gate passes, run multi-turn RLVR with:

- K=8 current-policy groups;
- authenticated policy/checkpoint/prompt/scorer provenance;
- mixed-success support, not all-zero or repeated-easy groups;
- clean-SFT and trajectory-SFT controls;
- fixed compute and verifier-call budgets;
- held-out OOD evaluation untouched by training.

Exit criteria: statistically supported improvement over the strongest
supervised control. If trajectory SFT matches RLVR, prefer the simpler method.

## M6 — Autonomous-research candidate

The candidate may search values, revise topology within allowed families, run
the simulator, diagnose failures, preserve provenance, and stop under a fixed
budget. It must not change its own scorer, targets, benchmark, or approval
policy.

The claim becomes credible only if it improves sealed tasks along at least one
held-out axis and a blind human review confirms physically plausible designs.
Until then, call it a closed-loop circuit-design research harness, not an
autonomous chip researcher.

## Immediate next actions

1. Finish v0.1 source and evidence consistency without publishing.
2. Run a controller-infrastructure smoke outside the evaluation split, then
   freeze concurrency and timeout values that avoid the v3 timeout mode.
3. Power task-level McNemar and family-level sign/cluster gates separately,
   including both discordance probabilities, alpha, and target power. Then
   preregister the resulting passive-confirmation pair count and more than 8
   balanced families. Keep opaque IDs, paired adjacency, family-clustered
   inference, exact circuit audit, and the existing fail-closed gates.
4. Add a frozen controller-error ceiling and pair invalidation rule. Bind
   concurrency and controller timeout in the run manifest; the implementation
   now does this for future runs, but v3's original manifest did not.
5. Add automated topology constraints for the rejected gain-of-two buffer
   pre-stage and engineering ranges that reject the 10-ohm summer before
   scoring. Do not retrofit v3 or rerun selected arms.
6. Build PVT corners as a separate sealed benchmark only after passive-feedback
   confirmation; v3 contains no PVT evidence.
7. Proceed to 100 approved trajectory-collection episodes only if the powered
   feedback-use gate passes. Evaluation episodes are never training data.
8. Compare supervised post-training methods before spending compute on RLVR.
