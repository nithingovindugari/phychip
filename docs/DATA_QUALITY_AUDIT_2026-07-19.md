# PhyChip data-quality audit — 2026-07-19

## 2026-07-21 superseding status

The 2026-07-19 decision below is preserved as the pre-training audit. Its
statement that the exact-trainable set was empty is no longer current. Clean
SFT was retrained and evaluated, exact K=8 labels were generated and manually
audited, and a bounded scorer-v6 pilot was approved before any optimizer
update.

### Dataset and grain

| Artifact | Intended grain | Current evidence |
|---|---|---:|
| Candidate replay | one specification with eight preserved current-policy completions | 144 specs / 1,152 completions |
| Trainable curriculum | one specification with `0 < exact successes < 8` | 70 specs / 560 completions per fresh pass |
| Powered holdout | one held-out specification, paired across policies | 193 specs / 1,544 completions at K=8 |

The approved replay was provenance-rematerialized as
`data/rlvr_grpo_powered_v1/exact_labels_v3.jsonl`, SHA-256
`5362370d620b665b0148037c439780fc36b81c11e8692ad2c3bec08450cc02b9`.
Its source v2 SHA is `11ea5f0d...452a5`; all completion, success, reward,
decomposition, and trainability vectors are identical. The change removes a
nondeterministic ngspice `Creation Date` from runtime identity without changing
scorer semantics.
It contains 135 exact successes, 70 mixed-success specifications, 74 observed
0/8 specifications, and 12 capped completions. Every completion and every
v4-to-v6 scorer decision change was manually reviewed before approval.

### Checks and findings

| Finding | Evidence | Risk | Severity / confidence |
|---|---|---|---|
| Exact identity and prompt uniqueness pass | 70/70 unique `spec_id`, `spec_hash`, normalized-prompt hash, and row hash | low exact-duplication risk | Low / high |
| Structural diversity is lower than row count | 52 unique numeric-insensitive reference-topology fingerprints among 70 rows; maximum repeat 5 | nominal row count overstates independent design behavior | High / high |
| Family coverage is strongly skewed | 28 non-inverting + 21 summing + 20 inverting + 1 Sallen-Key + 0 Schmitt; 69/70 in three amplifier families | cannot teach or support a balanced circuit-generalization claim | Critical for broad claim; acceptable for narrow pilot / high |
| Objective coverage is narrow by design | all 70 admitted rows have one objective | good reward density, but no evidence of multi-objective composition | High for broad claim / high |
| Reward contrast exists but is often sparse | mixed pass-count histogram: 1:38, 2:15, 3:6, 4:7, 5:3, 6:1 | high-variance advantages on many rows | Medium / high |
| Completion termination is healthy | 7/560 caps in admitted rows (1.25%); 12/1,152 overall (1.04%) | earlier 100%-cap failure is not present | Low / high |
| Holdout is hash-disjoint | 193 rows, 69 topology fingerprints, zero overlap with train in spec, prompt, row, or topology hash | supports transfer to unseen reference structures within familiar families | Low leakage risk under recorded fingerprint / high |
| Holdout is not broad OOD | same five families, overlapping target regime; only 3 Schmitt rows | cannot establish unseen-family or broad numeric-regime generalization | High / high |
| Offline difficulty regime differs slightly | labels use temperature 0.8; training uses 0.9 | mixed status can change under the live sampler | Medium / high |

### Current decision

The data is trustworthy for a **bounded, amplifier-heavy, within-family
structural-transfer/sharpening pilot**. It is not sufficient for the headline
claim that RL improves general analog-circuit-design generalization. The active
run therefore keeps a fail-closed 20-fresh-group health gate under the actual
temperature-0.9 sampler. Passing the gate validates usable live contrast; it
does not upgrade the claim.

Per seed, 70 unique prompts create 70 fresh K=8 groups = 560 rollouts. TRL's
`num_iterations=2` reuses each group for a second policy update, producing 140
optimizer steps without doubling the data horizon. More epochs or a higher
LoRA rank would not repair missing family coverage.

### Smallest useful remediation after the pilot

1. Preserve the current run as the narrow diagnostic; do not reinterpret it as
   broad OOD evidence.
2. Generate fresh, verifier-valid candidate specifications with explicit
   per-family and topology quotas, especially Sallen-Key and Schmitt.
3. Relabel with the exact training sampler and admit only current-policy mixed
   groups; do not call 0/8 rows unreachable.
4. Cap repeated topology fingerprints and report prompt, target, topology,
   family, objective, and source diversity separately.
5. Add non-overlapping numeric bands or held-out families for the stronger
   generalization estimand, then use at least three training seeds and matched
   no-update/rejection-SFT/random-reward controls.

Current test evidence: **426 passed / 3 skipped** in `athma-train` and
**158/158 passed** in `phy-chip-env`. [[wiki: Data Curation]] [[wiki: GRPO]]

### Powered execution status

The frozen 193-spec baseline completed at 225/1,544 K=8 successes
(pass@8 116/193) and 15/193 greedy. Seed 42 then failed closed before its first
optimizer update because the old runtime fingerprint treated ngspice build
time as verifier semantics. No trained treatment exists yet. Both baselines
now validate against the byte-identical v3 holdout and stable verifier hash;
the retry can reuse them after remote tests pass.

---

**The remaining sections preserve the historical 2026-07-19 pre-training
audit. They are superseded where the status block above says so.**

## Decision

The data is suitable for the next **gated** Phase-1 steps, but not for an
immediate GRPO run.

- SFT has a materialized 15,660-row training subset that passed both static
  tokenizer/template checks and the production ngspice sandbox. Retrain SFT on
  this file before labeling RLVR data.
- RLVR has 2,235 clean candidates and a 641-row disjoint holdout whose
  reference netlists all pass the current verifier. The candidate set still
  lacks exact current-policy rollout labels, so its trainable subset is
  intentionally empty.

## Sources and lineage

| Corpus | Source | Membership evidence |
|---|---|---|
| SFT | 281 JSONL files selected by the historical trainer glob | 23,026 rows reconstructed from the actual loader contract; every gated row carries source file/line hashes |
| RLVR | `data/dataset_v3/train_pool.jsonl` + `holdout_pool.jsonl` | 4,999 rows; every output carries stable spec identity, source hashes, and a disposition manifest |

The historical SFT run did not save an exact input manifest. That limits
perfect reconstruction, but the trainer's glob, row counts, and step evidence
agree on the 23,026-row inventory. The new trainer writes exact data-file
hashes, row counts, base/template revisions, and chat-template hash.

## SFT audit

### Static gate

The static gate used the exact local `checkpoints/sft_masked` tokenizer/chat
template at max length 4,096 and checked schema, assistant structure, true
assistant terminator, truncation, duplicated content, reasoning-mode contract,
and unsafe SPICE directives.

| Metric | Rows |
|---|---:|
| Input | 23,026 |
| Static clean | 16,177 |
| Static quarantine | 6,849 |

Reasons overlap. The largest signals were 4,512 excessive-comment rows, 3,654
repeated-ngram rows, 3,121 repeated-line rows, and 2,165 rendered sequences
over 4,096 tokens. These are not all independent counts.

### Dynamic gate

All 16,177 static-pass rows were graded through the centralized ngspice-46
sandbox.

| Metric | Rows |
|---|---:|
| Dynamic pass | 15,660 |
| Dynamic quarantine | 517 |
| Pass rate | 96.804% |

All 517 failures were rejected by the safety gate before simulation. The most
affected families were level-shift (126), current-mirror (111), and cs-stage
(94). Across the four generation partitions, failures were 337/3,225,
173/5,022, 7/571, and 0/7,359. Public artifacts retain provider-neutral source
partitions; operational provenance remains in the private audit record. This
pattern supports quarantining the rows, not relaxing the central allowlist.

Canonical training file:

`data/final_data/sft_v0_verified_v1/sft_training_ready.jsonl`

SHA-256: `02af3e6c58b668acec130850ea31d468fb8fd49f938272854d4d857b0fd39b9b`

## RLVR audit

The gate applies schema checks, semantic target checks, the current reward
contract, prompt deduplication, and topology-aware split isolation.

| Disposition | Rows |
|---|---:|
| Source | 4,999 |
| Verified candidates | 2,235 |
| Holdout | 641 |
| Quarantine | 2,123 |
| Exact trainable now | 0 |

The new candidate/holdout split has zero prompt and topology overlap under the
recorded fingerprints. Full dynamic validation found:

| Check | Candidates | Holdout |
|---|---:|---:|
| Reference simulates | 2,235/2,235 | 641/641 |
| Reference meets every requested metric | 2,235/2,235 | 641/641 |
| Anti-cheat blocked | 0 | 0 |
| Scoring exception | 0 | 0 |

The old split looked clean only at exact-row grain. It had zero exact-row
overlap, yet 162/495 holdout rows shared prompts with training and 182/495
shared topology fingerprints. This is why row-level deduplication alone was
not adequate.

## Trainability contract

RLVR trainability is not inferred from a hand label or a sampled extrapolation.
An exact-regime-v2 label must include:

- `K` matching the training group size;
- finite `reward_vector` and boolean `success_vector`;
- immutable `spec_id` and `spec_hash`;
- exact `policy_hash`, `verifier_hash`, and `prompt_contract_hash`.

A candidate is admitted only when `0 < success_count < K`. This rejects dead
all-fail groups and already-saturated all-pass groups. Reward variance remains
a diagnostic rather than a substitute for exact success evidence. This is the
data-quality-first sequence recommended by [[wiki: Data Curation]] and is a
necessary GRPO gate, not evidence that GRPO will improve the model
[[wiki: GRPO]].

## Verification and limitations

- 189 tests passed; 3 legacy multi-turn-generator tests are explicitly skipped
  because that unimplemented path is outside Phase 1.
- 102/102 active harness tests passed.
- The static RLVR preflight passed all 14 current checks.
- The sandbox has process/resource isolation and UID drop on Linux, but no
  dedicated filesystem/network namespace.
- The topology fingerprint is deterministic and numeric-insensitive, but no
  heuristic fingerprint can prove semantic independence. The manifest makes
  its exact contract auditable.
- Statistical power and total cloud cost remain unknown until clean-pipeline
  pilot throughput and variance are observed.

## Required next evidence

1. Retrain the pinned clean-SFT policy on exactly 15,660 rows and evaluate it.
2. Generate K=8 labels for all 2,235 candidates with that policy.
3. Merge labels and require a non-empty, hash-consistent trainable artifact.
4. Run a bounded GRPO smoke before a multi-seed experiment.

No Claude distillation, CPT/DPO reopening, or further optimizer work is needed
to clear these evidence gates.
