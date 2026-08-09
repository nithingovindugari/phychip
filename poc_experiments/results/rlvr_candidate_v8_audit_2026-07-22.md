# Scorer-v8 Candidate Pool and Scorer-v9 Replay Audit

**Date:** 2026-07-22
**Status:** frozen K=8 pilot fully rescored with a recorded circuit-review
attestation under scorer v9
**Scope:** fresh, evaluation-blind synthetic candidates for a bounded in-distribution RLVR study

## Decision

The 48-reference pool completed current-policy labeling, but the rollout audit
found scorer-v8 shortcuts. After test-first scorer-v9 remediation, all 384
frozen rollouts were rescored and reviewed. The final pilot contains **11 safe
mixed-success prompts across 7 families**, which passes the frozen pilot gate.
It is a validated seed pool, not enough data for the decisive GRPO study.

## Frozen reference pool

| Check | Result |
|---|---:|
| References retained | 48 |
| Circuit families | 12 |
| References per family | 4 |
| Distinct topology fingerprints | 48 |
| Reference circuits meeting their frozen windows | 48/48 |
| Reference anti-cheat blocks | 0/48 |
| Machine-rejected or manually rejected attempts | 65 |
| Reference-admission scorer | `phy_chip_env.score@v8` |
| Verifier hash | `8c8a1006dc0c9bc3a67a9d0be35533f54963e2806f26e0847af7f7b28a0802a9` |
| Frozen private reference SHA-256 | `2aef168e3d04eebc799eb48f9fb3b3b08320ec81adaa9d961091da4f80504a5e` |

The pool contains four references for each of: buffer, common-source stage,
differential amplifier, differential pair, differentiator, gain-trim,
integrator, inverting amplifier, non-inverting amplifier, Sallen-Key filter,
Schmitt trigger, and summing amplifier.

## Admission and decontamination

The audit replayed every reference through the packaged scorer and scanned
124,425 training rows plus 125 sealed evaluation rows.

| Check | Result |
|---|---:|
| Stale scorer or verifier fingerprints | 0 |
| References failing replay | 0 |
| Exact training-reference matches | 0 |
| Exact evaluation-reference matches | 0 |
| Prompt 8-gram overlaps at or above 0.40 | 0 |
| Final audit failures | 0 |

Quarantined candidate-authoring directories are deliberately excluded from the
training corpus scan. They are diagnostics, not model inputs. A regression test
guards that boundary after an earlier audit incorrectly compared the new pool
against an older quarantined pilot.

## Manual circuit review

All 48 retained netlists were read at the circuit level. The review checked
that named ports were actually consumed, feedback was load-bearing, target
behavior came from executable topology rather than comments, and controlled
sources were internal amplifier blocks rather than direct target-value sources.

Five initially admitted references were rejected during this review: one
inverting amplifier, one non-inverting amplifier, one summing amplifier, and two
gain-trim stages. Their transfer magnitudes were roughly -22 to -52 dB: they met
self-derived windows but were near-dead circuits for an amplifier family. The
authoring-only plausibility gate now admits amplifier references only from -12
to 80 dB. Five replacements passed the same scorer, decontamination, topology,
and manual-review gates. The deployed reward function was not changed.

## Promotion gate after K=8 labeling

The pilot continues only when all conditions hold:

1. At least 8 of the 48 prompts have mixed success (`0 < pass_count < 8`).
2. Mixed prompts cover at least 6 circuit families.
3. No successful or high-reward rollout contains a reward exploit.
4. Cap rate, anti-cheat rate, reward decomposition, and termination behavior are
   reported alongside the yield.

Raw references, prompts, completions, and rollout text remain private. The
public repository contains the scorer, tests, audit method, aggregate evidence,
and reproducibility contracts—not the private candidate or rollout corpus.

## Final scorer-v9 replay

The first full rollout review read all 384 completions and found 46 invalid
positive-reward rows under scorer v8, including 38 that would have received
positive advantage. Four of the 22 nominal exact successes were invalid. The
defects included `.control` commands impersonating transistor cards,
disconnected inputs, decorative feedback, wrong reactive topology, and
gain-trim circuits that moved output offset without changing gain.

Each defect class became a regression before the preserved rollout set was
rescored. Scorer v9 also removed the positive simulation floor: a runnable
circuit earns zero unless a valid family implementation makes measured
specification progress.

| Final v9 quantity | Result |
|---|---:|
| Frozen prompts / rollouts | 48 / 384 |
| K | 8 |
| Exact successes | 18 |
| Mixed / zero / easy prompts | 11 / 37 / 0 |
| Mixed families | 7 |
| Anti-cheat blocks | 181 |
| Assistant EOS / capped | 382 / 2 |
| Group winners reviewed | 33 |
| Known invalid v8 rows among v9 winners/exact rows | 0 |
| Audit failures | 0 |

Mixed prompts cover differential pair, differentiator, integrator, inverting
amplifier, non-inverting amplifier, Sallen-Key filter, and summing amplifier.
All 33 v9 group winners and all 18 exact successes were read as executable
circuits. None intersected the invalid set from the complete v8 audit.

The public summary records this as a content-addressed project attestation.
Per-row verdicts and rollout text are private, so the review conclusion is not
independently reproducible from the public checkout.

The historical preservation set now passes **44/53** under v9. The nine
blocked rows are all gain-trim references whose trim input changed output
offset rather than gain. This is a correction to stale gold labels, not a
regression in valid circuits. The package additionally rejects 22 frozen
reward attacks and passes repository/package parity.

Content addresses for the private, access-controlled evidence are recorded
without publishing prompts or completions:

| Artifact | SHA-256 |
|---|---|
| v9 aggregate audit | `7372a282b17f299a289d76bbed4161e897903071019ef19c74f01c04bff861f5` |
| v9 manual-review queue | `70ed05ad6dc8b966547714b528f10fd3214c56dbe6921f956283621762246206` |
| v9 rescored rollouts | `d44caf212ea05b20fd55b225d00e00808f859c29cf6ad7b789b0554ddada112e` |

## Decision

The pilot gate passed, so the data contract is credible enough to scale. It is
not credible enough to train on 11 prompts and call the result a generalization
study. The next data stage is 150--250 new evaluation-blind references under
the same frozen scorer-v9 contract, labeled at K=8 and filtered only by the
predeclared `0 < pass_count < 8` rule. Training begins only after the scaled
pool retains enough manually safe mixed prompts for a powered, held-out study.
