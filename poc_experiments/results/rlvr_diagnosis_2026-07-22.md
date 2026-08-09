# Scorer-v7 RLVR pilot diagnosis

**Date:** 2026-07-22
**Decision:** **STOP before training.** The current-policy label pass found useful mixed-success signal, but manual review found two new full-reward scorer escapes. Scorer-v7 labels are diagnostic evidence, not a trainable artifact.

## Reproduction contract

- Policy: clean-SFT adapter at `checkpoints/sft_clean_v1`
- K: 8 per prompt
- Sampling: temperature 0.9, top-p 0.95, seed 20260722
- Maximum completion: 3,072 tokens
- Prompt renderer: byte-identical train/eval chat contract, thinking disabled
- Termination: assistant `<|im_end|>` recognized as EOS
- Runtime: one Prime A100 80 GB PCIe, ngspice-46 sandbox
- Scorer: `athma_train.env.rlvr_scorer.score_completion@v7`
- Evidence: `data/rlvr_candidates_v7_20260722/labels/`

The standalone package and training tree have different content fingerprints
because they hash different source paths. Both implementations were rerun on
the exact 96 pilot completions; all 96 full `ScoreResult` events were identical.
The two hashes are retained rather than treated as interchangeable.

## What the machine labels said

| Quantity | Result |
|---|---:|
| Candidate specs | 12 |
| Rollouts | 96 |
| EOS / capped | 96 / 0 |
| Simulated / not simulated | 70 / 26 |
| Anti-cheat blocked | 29 |
| Exact successes | 6 |
| Machine-labeled mixed prompts | 5 |
| 0/8 prompts | 7 |
| 8/8 prompts | 0 |

Reward buckets were 55 at 0, 21 at the 0.30 simulation floor, 14 partially shaped, and 6 exact at 1.30. Mean completion length was 547.5 tokens (range 130–2,083). This rules out the old 100%-cap/EOS failure: the new problem is reward validity and current-policy reachability, not termination.

## Manual review changed the answer

Every one of the 96 completions was inspected in a compact executable-line ledger, and all six exact-reward completions were read in full.

| Family | Machine pass | Manual verdict | Finding |
|---|---:|---|---|
| buffer | 1/8 | reject | The winner is an inverting/difference-amplifier macro around `Vref`; a bandwidth-only target does not prove buffer identity. |
| cs-stage | 1/8 | retain for pilot | A MOS common-source stage produces the measured gain. |
| sallen-key | 1/8 | retain for pilot | The winner contains the expected active two-R/two-C filter network. |
| summing-amp | 2/8 | retain for pilot | Both winners use three input resistors and real output feedback into one summing node. |
| gain-trim | 1/8 | reject | `Epos` directly combines `Vin` and `Vtrim` one hop before the output amplifier, bypassing scorer-v7's direct-output-only alias check. |

The safe result is therefore **3 mixed prompts and 4 authentic successful rollouts**, not 5 prompts and 6 successes. Seven other prompts remain honest 0/8 quarantines. There are no 8/8 rows to retire.

## Why this blocks training

The scope required at least 8 mixed prompts across 6 families and zero manually observed full-reward exploits. The pilot has 3 safe mixed prompts across 3 families and 2 distinct full-reward exploit classes. Training would optimize known reward holes and would be too small and concentrated to answer the RL generalization question.

This is also why “balance” cannot be created by resampling. The observed current-policy distribution is the scientific result. We may author new specifications and measure them, but we may not alter K=8 outcomes, duplicate the three survivors until a desired ratio appears, or loosen a target after reading its rollouts while retaining the same spec identity.

## Required promotion boundary

No production scorer change is made from this PoC automatically. A deliberate **Promote-to-athma-train** step must:

1. add a buffer identity gate that requires unity-like low-frequency transfer and the intended non-inverting signal path in addition to bandwidth;
2. make gain-trim controlled-source alias detection transitive through internal one-hop/multi-hop drivers, while retaining genuine feedback macros;
3. add both exact decks above as red regression tests, plus gold circuits that must remain green;
4. mirror the change into the packaged scorer and prove repository/package parity;
5. bump scorer semantics to v8 and invalidate every v7 policy label;
6. rerun the 48-reference pilot from fresh spec IDs, then scale only if all frozen gates pass.

The methodology follows `[[wiki: Reward Hacking]]`: a simulator is verifiable only relative to the executable property it actually checks. A full-reward counterexample is a verifier defect, not an RL success.

---

## Scorer-v8 48-reference follow-up

**Decision:** **STOP before training again.** The larger pilot passed the raw
mixed-group count, but complete review of the reward frontier found three
full-reward construct failures in two additional classes. Scorer-v8 labels are
also diagnostic evidence, not a trainable artifact.

### Frozen run

| Quantity | Result |
|---|---:|
| References | 48 across 12 families and 48 topology fingerprints |
| Rollouts | 384 (K=8) |
| Machine-mixed / 0-of-8 / 8-of-8 | 14 / 34 / 0 |
| Exact successes | 22 |
| Anti-cheat blocked | 122 |
| Assistant EOS / capped | 382 / 2 |
| Machine-mixed families | 8 |
| Raw-rollout SHA-256 | `b32d609e...ce0995` |

Sampling, policy, renderer, seed, and token cap were unchanged from the frozen
contract above. Four-spec batching changed throughput only; it did not change
the per-prompt K=8 sampling contract. The pod was terminated after all evidence
was recovered, and the provider reported zero active pods.

### Manual reward-frontier findings

All 22 exact successes and all 42 per-group reward winners were read as
executable circuits. Three exact-reward outputs are not safe training winners:

1. **Integrator follower-pole shortcut.** One deck had no resistor path from
   `Vin` into the capacitor-feedback summing node. A non-inverting follower plus
   finite amplifier pole produced the requested crossing.
2. **Integrator dual-feed cancellation.** A second deck connected `Vin` to the
   summing node through a resistor, but also copied `Vin` onto the amplifier's
   non-inverting path. In the ideal loop, the intended integrating error is
   cancelled and the remaining pole can mimic the requested slope.
3. **Gain-trim direct formula with offset camouflage.** A behavioral source
   directly computed a weighted sum of `Vin` and `Vtrim`, then added an
   unrelated offset-control node. The v8 subset check treated that extra node
   as evidence that the source was internal and failed to require load-bearing
   output feedback.

The first invalid integrator was the only success for its prompt. The second
shared a prompt with one authentic inverting integrator. The invalid gain-trim
was the only success for its prompt. Removing those two unsafe prompt groups
would leave 12 machine-mixed groups before a repaired-scorer rescore; family
coverage remains potentially sufficient, but **zero-exploit did not pass**.

One non-success buffer row scored 0 remotely and the 0.30 simulation floor on
local replay. Both scorer source trees agree locally; the difference is a
15-second ngspice timeout boundary, not an exact-success or anti-cheat
disagreement. The audit tool now records mismatch coordinates and uses the
same timeout as labeling. This row does not change any pass count, but exact
event parity is reported as 383/384 for this environment pair rather than
silently rounded to 384/384.

### Required v9 boundary

Before any GRPO step, a test-first promotion must:

1. require an integrator summing node that is reached from `Vin` through a
   passive input resistor and from `Vout` through a feedback capacitor;
2. require the active amplifier path to consume that summing node without a
   parallel `Vin`-tracking non-inverting control;
3. reject a gain-trim output driver that directly depends on both `Vin` and
   `Vtrim` without output-derived feedback, even when unrelated internal
   controls are also present;
4. run the complete verified-reference preservation audit and manually
   quarantine any old row that encodes the same invalid construct;
5. mirror tests and implementation in both scorer trees, bump semantics to v9,
   re-admit the 48 references, and rescore all 384 preserved completions;
6. sample no training updates unless the repaired labels still contain at
   least 8 safe mixed prompts across 6 families with zero reviewed exploits.

This is the intended verifier-first workflow: policy samples are an adversarial
test generator for the reward, and a discovered shortcut becomes a regression
before it can become an optimizer target. [[wiki: Reward Hacking]]

### Complete 384-rollout audit supersedes the frontier-only count

The initial v8 review concentrated on exact successes and group maxima. Two
independent read-only passes then inspected **all 384 executable rollouts**.
This broader audit found that v8's exact-success problem was larger than the
first three examples and, more importantly, that partially rewarded outputs
would also teach wrong circuit families.

| Slice | Positive reward | Invalid positive | Positive advantage | Invalid positive advantage | Nominal exact | Valid exact |
|---|---:|---:|---:|---:|---:|---:|
| References 0--23 | 97 | 27 | 64 | 21 | 11 | 9 |
| References 24--47 | 81 | 19 | 66 | 17 | 11 | 9 |
| **Total** | **178** | **46** | **130** | **38** | **22** | **18** |

Thus **29.2% of positive-advantage rollouts were semantically invalid**. The
failure is not confined to a few decorative comments: those rows would receive
the sign of policy-gradient update that increases their probability.

The complete audit found four invalid exact successes:

- two integrator decks already described above;
- a gain-trim direct formula with an unrelated offset control; and
- a closed-loop gain-trim deck where `Vtrim` is only an additive offset input,
  so the gain from `Vin` does not change with trim.

It also exposed reward-bearing partial failures in common-source, differential
pair/amplifier, differentiator, integrator, Sallen-Key, non-inverting-amplifier,
and gain-trim families. Representative causes were disconnected inputs,
floating transistor branches, missing shared-tail topology, wrong capacitor
placement, decorative feedback, and output drivers that did not consume the
filter state.

#### Lexical device-count defect

The transistor-family gate counted lines with a regular expression over the
entire active deck. Commands inside `.control` were not excluded, so `meas`
could satisfy the MOS-device requirement and `quit` could satisfy the
bipolar-device requirement. Twelve rollouts with no top-level `M` or `Q` card
were admitted; five received reward and positive advantage. Device admission
must count parsed top-level device cards only, outside `.control` and
`.subckt` bodies.

#### Additional contract defects

- A common-source group winner fixed the transistor gate with an independent
  source while `Vin` was disconnected from the active stage.
- Several gain-trim winners changed output offset rather than gain. A trim
  response alone is therefore not a gain-modulation measurement.
- One non-inverting-amplifier row drove the output directly from `Vin` while
  the visible feedback divider terminated at a fixed offset source.
- One Sallen-Key winner buffered an intermediate RC node without consuming the
  intended two-pole filter state.
- A buffer replay contained an internal `AC 1` source. Buffer was missing from
  declared-AC-port enforcement, and the row moved from reward 0 to the 0.30
  simulation floor across repeated timeout-boundary runs.
- One legitimate inverting-amplifier topology was over-blocked because a
  ground-connected input resistor made the polarity heuristic treat ground as
  an input-tracking node. v9 must fix false positives as well as false
  negatives.

Comments remain masked from policy loss, but five nominal exact outputs in the
first slice and several in the second contained materially false topology or
numeric explanations. They are not safe teacher prose and must remain outside
any response-supervised artifact.

**Revised decision:** v8 is not admission-safe for GRPO, even after simply
dropping the four invalid exact outputs. v9 must repair device parsing and the
load-bearing family contracts, then rescore the frozen 384 rollouts. No new
rollouts or optimizer steps are authorized until that deterministic replay and
another complete reward-frontier review pass.

#### Preservation audit corrected stale "gold" labels

The first v9 preservation run rechecked the 53-row rank-64 holdout rather than
assuming every historical reference was semantically gold. **44/53** remain
exact successes. The nine blocked rows are all labeled `gain-trim`; seven had
previously been counted as passing. Inspection showed that the trim input adds
an output offset while the small-signal gain from `Vin` remains fixed. They are
valid summing/calibration circuits, but they do not implement gain modulation.
They are quarantined from gain-trim reward preservation instead of weakening
the new functional gate.

Ten Sallen-Key references were initially over-blocked because their authentic
active stage was instantiated through a top-level `X` card rather than a
top-level controlled source. v9 now accepts an instantiated stage only when it
consumes the filter-state node and reaches the scored output; all ten return to
the passing set. This is the required preservation behavior: fix a genuine
false positive, quarantine mislabeled references, and never redefine the
family merely to recover an old count.

## Scorer-v9 deterministic replay result

Scorer v9 converted every defect class above into a mirrored regression in the
training and standalone scorer trees, removed the positive simulation floor,
and added a measured gain-modulation check. The frozen 384 completions were
then rescored without generating new policy output.

| Quantity | v9 result |
|---|---:|
| Exact successes | 18 |
| Mixed / zero / easy prompts | 11 / 37 / 0 |
| Mixed families | 7 |
| Anti-cheat blocks | 181 |
| Assistant EOS / capped | 382 / 2 |
| Reviewed group winners | 33 |
| Known invalid v8 rows retained as winners/exact | 0 |

All 33 group winners and 18 exact rows were read again as executable circuits.
The manually invalid v8 set has zero intersection with the v9 reward frontier.
The pilot therefore passes its predeclared 8-mixed/6-family/zero-known-exploit
gate.

This is a verifier and data-contract success, not an RL result. Eleven mixed
prompts cannot support the planned generalization claim. The next authorized
research step is to scale the same frozen v9 contract to 150--250 new
evaluation-blind references, label each at K=8, and manually audit every exact
or group-winning completion before any optimizer update.
