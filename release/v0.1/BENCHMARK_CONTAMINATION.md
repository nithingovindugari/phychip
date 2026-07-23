# Benchmark contamination statement

The exact 15,660-row clean-SFT corpus was checked against a registered
evaluation set containing 1,786 prompts and 877 reference netlists:

| Source | Registered prompts |
|---|---:|
| AnalogCoder proxy | 24 |
| phy-chip-bench v1 | 40 |
| phy-chip-bench v2 | 46 |
| capability benchmark | 150 |
| Masala-CHAI evaluation slice | 100 |
| physics-MMLU | 785 |
| RLVR holdout | 641 |

The executable gate reported 0 leaks and no missing registered assets.

- Corpus SHA-256: `02af3e6c58b668acec130850ea31d468fb8fd49f938272854d4d857b0fd39b9b`
- Report SHA-256: `3f3ad5ef03df278217ab6d4e6344335203b457ba22888e263f1a0de6b6d112d9`
- Report: `data/final_data/sft_v0_verified_v1/decontam_report.json`

This supports a narrow claim: no leak was found by the implemented registered
prompt/reference checks. It does not prove absence of paraphrases, shared public
source material, memorized base-model knowledge, or semantic/topological
near-duplicates. AnalogCoder content remains evaluation-only and was not
intentionally added to training.
