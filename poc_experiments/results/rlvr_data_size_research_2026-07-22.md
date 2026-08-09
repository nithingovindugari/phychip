# How much RLVR data does PhyChip need?

**Date:** 2026-07-22
**Decision:** the next defensible GRPO study targets **150--250 unique,
manually safe, current-policy mixed-success prompts**. This is an engineering
target for a bounded PhyChip causal study, not a universal RLVR scaling law.

## Short answer

There is no published minimum that transfers cleanly across reward functions,
models, and domains. Published results range from one deliberately selected
prompt to tens of thousands of unique prompts. The unit that matters for
PhyChip is not a generated row. It is a prompt that:

1. has an authentic circuit-family identity;
2. is clean against the evaluation sets;
3. is accepted by the exact current scorer and ngspice-46;
4. produces both successes and failures under the exact launch policy at K=8;
5. survives manual review of every high-reward rollout; and
6. adds a new family, target regime, or topology fingerprint rather than a
   duplicate template.

The 2026-07-22 scorer-v7 pilot created 12 accepted references and 96 rollouts,
but manual review leaves only **3 safe mixed prompts**. That is enough to debug
the pipeline and discover reward attacks. It is not enough to train or support
a generalization claim.

## What primary sources actually show

| Source | Model/task scale | Training prompts | Relevant lesson |
|---|---|---:|---|
| [DAPO](https://arxiv.org/abs/2503.14476) | Qwen2.5-32B, math | **17K** | Dynamic sampling discards 0%- and 100%-accuracy groups because they have zero group-relative gradient. The paper explicitly says its transformed dataset has 17K prompts. |
| [SimpleRL-Zoo](https://github.com/hkust-nlp/simpleRL-reason) | 0.5B--32B, math | **8K** | A shared 8K-example recipe improved 10 base models by roughly 10--20+ absolute points. This is useful evidence that small models do not require hundreds of thousands of RL prompts. |
| [DeepScaleR](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) | 1.5B, math | **40.3K** | A strong small-model result still used about 40K unique problem-answer pairs. |
| [Tulu 3 RLVR mix](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints) | 8B/70B, mixed math + constraints | **29.9K** | Broad behavior coverage is built from tens of thousands of prompts, not repeated copies of a few templates. |
| [One-Shot RLVR](https://arxiv.org/abs/2504.20571) | 1.5B/3B/7B, math | **1--1,209** in ablations | A carefully selected prompt can move a policy and sometimes match a 1.2K subset. The paper also shows large prompt-to-prompt variation and selects by historical reward variance. It proves possibility, not that one prompt is a safe general recipe. |
| [Prompt Replay](https://arxiv.org/abs/2603.21177) | Llama-3.2-3B and Qwen3-8B, math | online reuse | Medium-difficulty prompts near pass rate 0.5 improve early sample efficiency. Aggressive reuse plateaued and converged with the baseline, so repetition does not replace fresh diversity. |

These results are not directly comparable: model scale, initial checkpoint,
reward, completion length, batch construction, number of rollouts, steps, and
task entropy all differ. They bound the plausible range; they do not define a
formula such as "N prompts per billion parameters."

## PhyChip recommendation

| Purpose | Safe mixed prompts | Coverage requirement | What it can establish |
|---|---:|---|---|
| Scorer/pipeline smoke | 16--32 | at least 6 families | The reward and optimizer receive nonzero, non-exploited signal. |
| First defensible GRPO study | **150--250** | 8--12 families; at least 100 topology fingerprints; no family >15%; no fingerprint >2 rows | A bounded causal claim about in-distribution sharpening, with 3 seeds and controls. |
| Generalization thesis | **500+** plus a sealed 300--500-spec holdout | topology-disjoint and numeric-regime-disjoint splits; all supported families represented | Evidence about transfer beyond memorized templates. This still does not prove broad analog-design generalization. |
| Broad model-building campaign | thousands | diverse verifier-safe families and continuously refreshed frontier | Comparable in spirit, not scale, to public 8K--40K RLVR recipes. This is post-v0.1 work. |

The **150--250** target is the smallest campaign worth buying for the current
thesis test. It is intentionally larger than the old 70-prompt pool but far
smaller than broad math corpora because PhyChip has 23 narrow harnessed
families and each label requires a sandboxed simulator.

## How many candidates must be authored?

The pilot's manually safe mixed yield was `3 / 12 = 25%`. At that point
estimate:

- 150 promoted prompts require about **600** reference candidates;
- 250 promoted prompts require about **1,000** reference candidates;
- K=8 labeling costs 4,800--8,000 policy rollouts before boundary relabels.

The yield estimate is extremely uncertain at n=12 and was measured with a
scorer now known to be incomplete. Therefore the campaign must proceed in
waves:

1. implement and parity-test scorer v8;
2. author a fresh 48-reference pilot with new IDs;
3. manually audit every high-reward rollout;
4. estimate safe-mixed yield and family coverage again;
5. scale only if at least 8 safe mixed prompts span 6 families and there are
   zero new high-reward exploits;
6. stop and fix the policy, target bands, or verifier if safe-mixed yield is
   below 15% rather than brute-forcing thousands of weak candidates.

## Difficulty distribution

Do not manufacture a 30/40/30 success-count mix. Keep every authentic K=8
success vector. Prefer prompts nearer 4/8 because binary-reward variance
`p(1-p)` is largest near `p=0.5`, but admit the whole honest 1--7/8 band.
Quarantine 0/8 and 8/8 rows; retargeting creates a new spec ID and requires a
fresh label. Re-label against the current policy because difficulty moves as
the policy changes.

## Separate training data from evaluation power

More training prompts do not repair an underpowered holdout. The earlier
53-spec comparison had only about 8% power for a +5.7-point effect. A thesis
claim needs a sealed evaluation set and paired sampled evaluation independent
of the 150--250 training prompts. Training reward is a health signal, not the
headline result: DAPO reports that final training reward can correlate poorly
with validation accuracy.

## Go/no-go

The 48-reference pilot now passes under scorer v9 with 11 safe mixed prompts
across 7 families and zero known-invalid reward-frontier rows. That validates
the contract but is still too small for the decisive GRPO run. Scale the same
frozen contract to 150--250 new evaluation-blind references, then freeze the
holdout and power calculation before training. Older scorer labels remain
quarantined and must never be rebound to v9.
