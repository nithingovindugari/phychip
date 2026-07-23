# PhyChip v0.1: what the verifier taught us

## Abstract

PhyChip began as a five-stage model-training plan and ended with a more useful
artifact: a standalone, physics-backed reward environment. The v0.1 release
contains 23 analog-circuit verifiers, a pinned ngspice-46 runtime contract, a
resource-bounded sandbox, structured reward diagnostics, and parity/adversarial
tests. A clean SFT adapter is included as a limited proof case. The historical
GRPO result does not survive corrected evaluation, while a later matched
rank-64 smoke has a positive but statistically inconclusive point estimate.

## Why the environment is the result

A simulator makes circuit behavior observable, but “it simulated” is not the
same as “it satisfied the request.” Early reward versions blurred structure,
simulation, measurement coverage, and spec success into one scalar. That made
format compliance and simulator admission look like physical design progress.
The v0.1 API therefore returns every component: whether a fenced deck ended,
contained devices and analyses, simulated, was blocked by anti-cheat rules, how
many requested metrics were measured/met, normalized spec score, and the final
`all_specs_met` decision.

## Five verifier lessons

1. **Pin the physics runtime.** The same model produced materially different
   scores under ngspice 36 and 46. The environment now requires major version
   46 and fingerprints the executable and verifier sources.
2. **Missing measurements are failures, not absent objectives.** Dropping an
   unmeasured FoM from the denominator creates a direct reward-hacking route.
   v0.1 divides by every requested spec.
3. **Topology is not implied by numeric agreement.** Behavioral sources can
   hard-code gain, feedback can be decorative, and requested ports can be
   disconnected. Structural admission gates help, but the capability matrix
   states their limits and manual topology review remains necessary for model
   claims.
4. **Train/eval contracts are part of the experiment.** Prompt templates,
   assistant EOS, generation caps, extraction, scorer bytes, simulator version,
   and benchmark bytes must be frozen together. Earlier GRPO conclusions were
   invalidated by mismatches across those boundaries.
5. **Graph checks must follow the executable signal path.** A scorer-v6
   polarity rule caught direct and one-resistor positive feedback but missed a
   VCVS that reached the scored output through a three-resistor chain. That
   invalid circuit became a positive-advantage group winner. Scorer-v7 uses a
   bounded transitive resistor path, requires non-inverting stages to consume
   their input and feedback, and keeps the exact escapes as regressions.
6. **The policy is part of the red team.** Complete review of 384 later
   rollouts found 46 invalid positive-reward rows after the fixed attack suite
   was green. Scorer v9 parses top-level devices, binds ports to load-bearing
   family topology, measures gain modulation, and pays no reward for mere
   simulation. Frozen replay makes the effect of that semantic change auditable.

## Model evidence

The clean-SFT adapter reaches 21/24 on the AnalogCoder raw-SPICE proxy, 23/40 on
bench-v1, and 25/46 on bench-v2 under the primary contract. It is strong on
Level 1 and nearly absent on Levels 2/3. Physics-MMLU non-inferiority is not
established. A separate long-generation manual audit further shows that
automatic proxy passes can contain wrong topologies.

The corrected historical GRPO-v3 candidate scores 16/24, 19/40, and 10/46 and
is rejected. A newer matched rank-64 run moves 3/53 to 6/53, but its paired 95%
CI crosses zero and McNemar p is .375. Its four treatment-only wins cluster in
inverse amplifiers and Schmitt triggers, while three families remain 0/10.
That pattern is consistent with local template sharpening, not demonstrated
broad generalization.

A later scorer-v6 seed completed 140 optimizer steps, but its final health gate
failed and manual review found one invalid positive-feedback group winner. The
second seed and paired evaluation were not run. It is preserved as verifier
diagnostic evidence, not an RL result.

The scorer-v9 data-contract pilot contains 11 safe mixed prompts across 7
families after replay of 384 frozen rollouts. That clears the verifier-feasibility
gate but is too small for the planned GRPO study; 150--250 new references and a
separate powered holdout remain necessary.

## What v0.1 does and does not claim

v0.1 claims that a reusable analog-circuit reward environment can be packaged,
audited, sandboxed, and tested against gold and adversarial circuits. It also
shows that SFT can teach a small model executable Level-1 circuit conventions.

It does not claim that PhyChip solves precise analog design, that simulator
success proves topology correctness, that GRPO beats SFT, or that general
physics capability is preserved. Those are future experimental questions, not
requirements for shipping the environment.

## Next scientific experiment

After v0.1, a causal RL study should freeze one SFT parent, scorer, prompt pool,
and held-out topology/spec split, then compare no update, matched-compute
rejection-SFT, real-reward GRPO, and random-reward GRPO over at least three
seeds. The data must be policy-relative and mixed-success under the final
verifier. Until that exists, more optimizer tuning would add activity rather
than evidence.
