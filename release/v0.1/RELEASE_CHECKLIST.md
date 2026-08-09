# v0.1 pre-release checklist

## Scorer-v9 local gate complete on 2026-07-22

Complete review of 384 frozen current-policy rollouts found 46 invalid
positive-reward rows under scorer v8. Scorer v9 converts the defect classes
into fail-closed regressions, removes reward for mere simulation, and passes a
deterministic replay whose 33 group winners and 18 exact rows contain no known
invalid circuit. Earlier scorer wheels and labels remain superseded.

- [x] Received explicit bounded `Promote-to-athma-train` approval.
- [x] Added 22 attack regressions across the training and package suites.
- [x] Corrected the historical preservation set to 44/53; quarantined nine
      gain-trim rows that changed offset rather than gain.
- [x] Bumped both scorer semantics to v9 and proved source-tree parity.
- [x] Rescored all 384 frozen rollouts and reviewed all 33 winners/18 exact rows.
- [x] Reran source, parity, expanded-attack, timeout, and clean-install gates.
- [x] Replaced the dirty-tree-only release validator with separate public and
      maintainer-full gates; regenerate once more from the clean source commit.

## Passed locally on 2026-07-21

- [x] Standalone `score(...) -> ScoreResult` API
- [x] 23 explicitly registered measurement harnesses
- [x] ngspice 46 required and source archive SHA-256 pinned
- [x] Static allowlist plus CPU, memory, file, process, output, and wall-time bounds
- [x] Root execution drops to `nobody`; security limitations documented
- [x] Reward-component diagnostics and source/runtime fingerprint
- [x] 44/53 corrected historical references meet their own specs
- [x] 0/22 known attacks reach `all_specs_met`
- [x] Real timeout fails closed
- [x] Package/repository scorer parity on 44 preserved references and 106 retained completions
- [x] Public checkout: 257 standalone tests pass and the private-corpus parity
      node skips; hydrated maintainer checkout: all 189 nodes must pass
- [x] Public training tree: 200 tests pass and 13 explicitly bounded
      private/optional-evidence nodes skip
- [ ] Hydrated maintainer training-tree replay passes with private operational
      scripts, reference data, and required model-framework dependencies
- [x] Multi-hop positive-feedback and non-inverting unused-input regressions
      fail closed while the valid split-feedback/polarity controls pass
- [x] Wheel includes package sources, LICENSE, and NOTICE
- [x] Clean virtual-environment install passes gold/attack/timeout/runtime smoke
- [x] Model card, evidence note, contamination statement, license notice,
      reproduction guide, environment lock, and technical note prepared
- [x] Rebuilt/reinstalled the wheel under scorer-v9 and recorded its verifier
      and wheel hashes in content-addressed release metadata
- [x] Regenerated the machine-readable manifest/checksums and reran release
      validation on the current review tree
- [x] Added a regression requiring ngspice exit code zero even when failed
      output contains an analysis marker

## Must happen after user review

- [x] Review every release file and current diff
- [x] Pin the Python base image to its immutable multi-architecture OCI digest
- [x] Re-run manifest from clean source commit `bc1c470`
- [x] Commit only the approved release scope locally; no push or publication
- [ ] Tag v0.1
- [ ] Push/publish model, package, benchmark references, and note

No commit, tag, push, upload, or publication is performed by this checklist.
