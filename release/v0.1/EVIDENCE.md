# Frozen v0.1 evidence

Status: **local scorer-v9 candidate validated on 2026-07-29**. A complete audit
of 384 frozen current-policy rollouts found 46 invalid positive-reward rows
under scorer v8, including 38 with positive advantage. Every defect class was
converted into a regression before deterministic replay. Scorer v9 retains 11
safe mixed prompts across 7 families and 18 exact successes. The project’s
content-addressed circuit-review attestation records zero overlap between the
known-invalid set and the 33 positive-reward group winners or 18 exact rows.
Per-row verdicts and rollout text remain private, so this attestation is not
independently reproducible from the public checkout. The quantitative model
results remain frozen historical evidence.

Package scorer semantics advanced to v9 with parsed top-level device
admission, declared-port checks, load-bearing family topology, measured
gain-modulation semantics, and removal of the positive simulation floor. The
public-checkout standalone suite passes 257 tests, intentionally skips the
private-corpus parity node, and rejects all 22 frozen attacks. The historical
preservation set is now 44/53: nine gain-trim rows were correctly quarantined
because trim changed output offset rather than gain. The exact clean-wheel
verifier hash and wheel SHA-256 are frozen in the runtime fingerprint and
evidence manifest generated from the intended clean source commit.
Model result files below remain immutable historical evidence under their own
recorded scorer identities.

The public `athma-train` tree separately passes 200 tests and skips 13 nodes
whose operational scripts, private reference corpus, or optional `torch`/`trl`
dependencies are absent. The public paths that still execute ngspice now route
through `athma_train.spice_gate`; the public test suite rejects any raw
`ngspice -b` subprocess outside that canonical boundary. These public skips do
not substitute for the hydrated maintainer replay.

The newest regressions cover control-language device impersonation,
disconnected inputs, wrong reactive topology, decorative feedback, hidden AC
stimuli, and gain-trim offset movement without gain modulation. See
`poc_experiments/results/rlvr_diagnosis_2026-07-22.md`.

## Authoritative claims

| Claim | Result | Interpretation |
|---|---:|---|
| Clean SFT / AnalogCoder-24 raw-SPICE proxy | 21/24 | Primary 2,560-token native-chat result; not official AnalogCoder |
| Clean SFT / bench-v1 | 23/40 | All 23 Level-1 rows; no Level-2/3 passes |
| Clean SFT / bench-v2 | 25/46 | One Level-2 pass; no Level-3 passes |
| Physics-MMLU retention | 356/785 vs base 365/785 | Non-inferiority not established |
| Corrected historical GRPO-v3 | 16/24, 19/40, 10/46 | Rejected release candidate; different SFT parent |
| Matched rank-64 diagnostic | 3/53 -> 6/53 | +5.66pp; CI crosses zero; McNemar p=.375 |
| Closed-loop development probe | Uncontaminated Codex 2/4; fixed topology + TPE 4/4 | Paired success delta -0.50, 95% CI [-1.0, 0.0]; no model-guided advantage |
| Matched parameter-search smoke | Random 3/4 (18 calls); LHS 3/4 (17 calls); TPE 4/4 (21 calls) | Four development tasks under an eight-call/task ceiling; protocol-bound smoke only, not superiority or generalization evidence |
| Eight-family feedback preflight | Full feedback 19/24; blind 10/24 | Task-level McNemar p=.003906; family sign p=.0625; topology/practicality gates failed |

## Regimes must remain separate

1. **Primary model matrix:** clean SFT at `max_new_tokens=2560` on three
   evaluation sets. bench-v1/v2 detail files retain only 600-character output
   heads, so their aggregate records are content-addressed but cannot be fully
   rescored from saved completions.
2. **Corrected GRPO-v3 candidate:** same current prompt/evaluator contract, but
   the chain starts from `wrap_v2_B_dprime_sft_lora`, not the clean-SFT adapter.
   It ranks candidates; it does not estimate a causal clean-SFT-to-GRPO effect.
3. **Matched rank-64 smoke:** exact clean-SFT parent, same 53 prompts, same
   generation contract, full completions retained, manual review completed.
   It is powered only as a smoke test: 25 fresh training groups / 200 rollouts
   and one evaluation seed.
4. **Powered pre-training audit:** 1,152 current-policy K=8 completions were
   preserved and read before any new optimizer update. Scorer v3 changes two
   false successes to failures (147 -> 145). This is verifier-red-team evidence,
   not a trained-model result.
5. **Completed scorer-v6 seed-42 diagnostic:** 70 fresh K=8 groups produced
   560 rollouts and 140 optimizer steps. The final health gate rejected the
   adapter because minimum entropy reached 0.1486 below the frozen 0.15 floor.
   Manual review then found one invalid positive-feedback group winner whose
   source reached `Vout` through three resistors and escaped scorer-v6. Seed 7
   and comparative evaluation were not run. This adapter is verifier-debugging
   evidence, not evidence for or against RL generalization.
6. **Scorer-v9 data-contract pilot:** 48 new references produced 384 K=8
   current-policy rollouts. The v8 audit found 46 invalid positive-reward rows;
   v9 deterministic replay retained 11 safe mixed prompts across 7 families,
   18 exact successes, 37 zero-success prompts, and no 8/8 prompt. This is
   verifier/data evidence only. No optimizer step used these labels.
7. **Eight-family feedback preflight:** 24 paired repair tasks compare complete
   simulator diagnostics with a feedback-blind control. The task-level effect
   is positive, but only 5/8 families favor feedback, 10/48 controller arms
   timed out, and manual review rejected one topology and two component sets.
   The promotion gate failed. Publication exposes the verifier augmentations,
   so this benchmark is retrospective evidence and cannot be reused as a
   sealed confirmation set or as training data.

## Primary artifact hashes

| Artifact | SHA-256 |
|---|---|
| Clean SFT adapter weights | `2307ebd794046f3b0be8f8a4de07469f0e6045d91869d3e7058b1897b3936dff` |
| Clean SFT corpus | `02af3e6c58b668acec130850ea31d468fb8fd49f938272854d4d857b0fd39b9b` |
| AnalogCoder primary summary | `e6725e69eb321b5af0d017ef415f6d9cf579ca207f136309d181f75474b7e672` |
| AnalogCoder primary details | `5e24b1e4301f303aafc5a324936fce82874a8ba823c74a7eafa5f2e1be33bf72` |
| bench-v1 primary summary | `a18116b21acdb6cf3138ce65ca437615434745daea007a4839899876e4cdca49` |
| bench-v1 primary details | `79325144e191543b370d91cad7ac69aa68a39d92f9fa31f6436f98179e48d647` |
| bench-v2 primary summary | `53870ae5644b6b7d82f9d94b2e2d8c9df45de401579e2d6317a99248709463d7` |
| bench-v2 primary details | `18b6807dbe42af067b48473619fdadd9101732f216142a6b0cca4b09fe58ffe3` |
| Physics-MMLU comparison | `fc6ba7f3fbe297145dedcafdeae72c4253c68dc539722ed241c7e14b8abe3e8d` |
| Matched rank-64 comparison | `618e2b25ed6352797b0e5cb1997de189dc86865ae329967f7c4deadf54087f1f` |

The machine-readable manifest contains every supporting path, distribution
class, and hash. `repository` artifacts are verified in a public clone;
`external-release` artifacts are distributed separately; and
`maintainer-evidence` artifacts remain access-controlled but are required by
the full pre-release validation gate.

## Verifier and environment fingerprints

- All primary generation summaries record the exact base revision, policy
  hash, benchmark hash, chat-template hash, prompt-render hash, ngspice-46
  banner, and core Python package versions.
- The matched rank-64 artifacts additionally retain all 106 completions,
  per-arm policy hashes, prompt hash, termination token IDs, scorer semantics,
  and verifier hash.
- The extracted package records a content hash over the canonical scorer,
  sandbox, measurement utility, and all 23 harnesses, plus the local ngspice
  binary hash.

Limitation: the July primary GPU summaries store an ngspice version banner but
not that pod's executable hash or a source-tree hash for every evaluator file.
The release does not pretend otherwise.

## Superseded evidence

Historical `22/24` GRPO headlines, July 4's `0.075` versus `0.057` train-pool
comparison, and pre-`rescore_v2` rank-64 totals are preserved for audit history
but are not v0.1 headline results. Their contracts were affected by prompt/EOS,
reward-denominator, contamination, topology-gate, or parser defects documented
in `progress.md` and `research.md`.
