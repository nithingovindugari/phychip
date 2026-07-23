# Authenticity audit of the three PhyChip circuit evaluations

**Date:** 2026-07-22
**Scope:** source identity, row identity, grading contract, reference validity, contamination controls, licensing/provenance, and the claims the artifacts support.
**Conclusion:** the recorded clean-SFT scores are materially authentic for their implemented harnesses, but the three evaluations must not be presented as three independent external benchmarks.

## Executive verdict

| Evaluation | Artifact identity | Grading authenticity | Independence | Safe claim |
|---|---|---|---|---|
| AnalogCoder | Exact current official 24-row source hash | Reproducible local raw-SPICE/ngspice adaptation | External source, adapted locally | **21/24 on an AnalogCoder raw-SPICE proxy**; not the official PySpice/planning benchmark |
| phy-chip-bench v1 | Evaluated IDs equal all 40 `verify_passed` rows and the published v1 split | L1 is compile/simulator-gate only; L2/L3 use project harnesses | Project-authored; teacher-generated references | **23/40 on project-authored bench-v1**, including 23/23 L1 and 0/17 L2–L3 |
| phy-chip-bench v2 | Exact 46-row local, release, evaluation, and Hub identity | Repaired, simulated, decontaminated, and well-posedness-gated; some topology checks remain incomplete | Project-authored, model-assisted companion | **25/46 on project-authored bench-v2**, not third-party validation |

The three results provide useful triangulation across prompt/grader constructions. They do not provide three independent replications of model quality.

## 1. AnalogCoder raw-SPICE proxy

### What was verified

- Official source repository: `laiyao1/AnalogCoder`.
- Official `problem_set.tsv`: 24 data rows, IDs 1–24.
- Current official/local cache SHA-256: `b37bf4613f2238232713dd1dff0e5aba167bba41c096d1278d4ad17870d41859`.
- Clean-SFT artifact: 24 problems, 21 passes, pass@1 0.875, the same source hash, the same 24 IDs, ngspice-46 output, package versions, policy hash, and prompt hash.

### Material caveat

The official task asks for a design plan and PySpice code. PhyChip renders the problem as a raw-SPICE generation task and grades it through the local strict ngspice gate. This changes both output format and evaluator. The result is therefore a **proxy**, not an official AnalogCoder reproduction and not directly comparable with the paper's reported metric.

The upstream GitHub repository does not expose an explicit license in its root/API metadata. PhyChip should keep the content evaluation-only, cite/link the official source, avoid republishing the prompt file, and seek license clarification before bundling source text. AnalogCoder content must never enter training.

**Verdict:** authentic adapted result; naming must always include “raw-SPICE proxy.”

## 2. phy-chip-bench v1

### What was verified

- Source file: 55 unique rows, SHA-256 `676ce6a6bd909f7643a5e9e9239a123e4cd47a354d6e82b6f28670570f416e1b`.
- Exactly 40 rows have `verify_passed=true`: 23 L1, 12 L2, and 5 L3.
- The 40 evaluation detail IDs exactly equal those 40 verified IDs.
- The published Hub v1 split contains the same 40 IDs and row contents.
- Clean-SFT artifact: 23/40, comprising 23/23 L1 and 0/17 L2–L3.

### Material caveats

1. **The benchmark is internal.** Its references were created through a project-controlled, model-assisted synthesis path. “Programmatically generated from parametric templates” alone is incomplete provenance.
2. **L1 is a floor, not functional design proof.** The build contract sets `spec_check_python` to unconditional true for L1, and the evaluator accepts L1 after the strict simulator gate. L1 establishes structured, runnable output under the gate; it does not establish measured circuit performance.
3. **Fingerprint mismatch.** The result summary hashes the 55-row source file even though the evaluator selects its 40 `verify_passed` rows. The detail IDs reconcile exactly, so the score is not corrupted, but the provenance should hash a canonical 40-row evaluation artifact (and preferably its ordered ID list and rendered prompts) in future runs.
4. **Shared synthesis lineage weakens independence.** The SFT corpus also contains model-assisted rows. Even with no exact prompt match, this benchmark is not fully independent of the training pipeline. Report it as project-authored and do not describe v1 as third-party validation.

**Verdict:** real simulator-backed project benchmark with a valid 40-row evaluation subset; not independent, and L1 should be described as compile/simulator-gate success.

## 3. phy-chip-bench v2

### What was verified

- Pre-well-posedness staging set: 50 rows.
- Canonical set: 46 rows; 28 L1, 13 L2, 5 L3.
- Canonical SHA-256: `4d7b9e7eb5fb718bdfa6530024895304ac55ceb6b31572896e2f07e275895796`.
- The local canonical file, release copy, clean-SFT result fingerprint, evaluation IDs, and published Hub v2 split agree exactly.
- The recorded well-posedness gate dropped four rows for objective reasons: near-zero amplifier transfer, missing headline unity-gain frequency, non-physical boost efficiency, and near-zero current-sense gain.
- Clean-SFT artifact: 25/46, including 24/28 L1, 1/13 L2, and 0/5 L3.

### Material caveats

1. **The benchmark is internal.** It is a project-authored, model-assisted companion, not an external benchmark or independent replication.
2. **Repair history matters.** The canonical 46 rows result from repair and removal of invalid references. Release claims should point to the canonical hash and retain the validation/well-posedness reports.
3. **Verifier scope is narrower than analog correctness.** The capability matrix explicitly states that several families have only global shortcut checks. Passing means measured requested ranges passed under the harness; it does not prove the named topology, device realism, corners, stability, layout, or manufacturability.
4. **L1 remains a structural/simulation floor.** It should not be mixed rhetorically with L2/L3 measured-spec success.
5. **Exclusion timing must be auditable.** The four well-posedness drops are technically objective, but this audit has not yet established that their rule was frozen without reference to model outcomes. Preserve timestamps/commits proving model-blind exclusion before treating the 46-row set as preregistered.
6. **Synthesis lineage is shared with new data work.** Separate prompts, seeds, and workspaces; decontaminate every candidate pool; and disclose that both artifacts are project-authored. This mitigates leakage but does not create benchmark independence.

**Verdict:** the strongest-provenanced of the two internal benchmarks and suitable for an honest research-preview evaluation, but still project-authored and verifier-limited.

## 4. Cross-benchmark contamination and claim rules

- All three sets remain evaluation-only and are excluded from candidate authoring, prompt tuning, target-band tuning, rejection sampling, RL rewards, and checkpoint selection beyond a predeclared evaluation schedule.
- v2 has an explicit local decontamination gate against training and existing evaluation corpora. The same exact/near-duplicate report should be regenerated against every new scorer-v7 candidate campaign.
- v1's historical creation path predates the strongest decontamination workflow. Before a paper claim, run the current exact, 8-gram, topology-fingerprint, and semantic-neighbor audit against the final SFT/RL corpora and publish aggregate results.
- Also publish cross-benchmark overlap among AnalogCoder, v1, and v2; checking only train-versus-eval misses repeated evaluation items.
- AnalogCoder's external provenance does not make the local result official: prompt and evaluator adaptation remain the dominant caveat.
- Report each benchmark separately. Do not average the three scores: their levels, graders, origins, and difficulty distributions are not commensurate.

## 5. Required remediation before public paper-style claims

1. Materialize and hash the canonical ordered 40-row bench-v1 subset; make future result files fingerprint that artifact rather than the 55-row source.
2. Update the Hub/release provenance language for v1 to disclose model-assisted project authorship and the compile-only L1 contract.
3. Add a benchmark manifest with source URL, source/release hash, row IDs, license/terms posture, generator provenance, grader version, prompt renderer, and validation reports.
4. Regenerate the v1 decontamination audit against the exact final training corpus.
5. Keep AnalogCoder phrased as “raw-SPICE proxy” everywhere and link the official repository/paper rather than redistributing its content.
6. Use manual topology review as a reported secondary measure for L2/L3 winners and discordant model comparisons.
7. Build a sealed final evaluation set for any OOD/generalization paper claim. Do not use it during candidate authoring, scorer tuning, checkpoint selection, or repeated methodology debugging.

## Sources

- AnalogCoder official repository: https://github.com/laiyao1/AnalogCoder
- AnalogCoder paper: https://arxiv.org/abs/2405.14918
- PhyChip benchmark Hub repository: https://huggingface.co/datasets/NithinReddyG/phy-chip-bench
