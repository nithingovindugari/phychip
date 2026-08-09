# athma-train

`athma-train` is the Apache-2.0 reference training tree used by PhyChip. It
contains the canonical training-side mirror of scorer v9, experiment
provenance and statistics contracts, and the SFT/GRPO scripts needed to
reproduce the documented research controls.

The release artifact is verifier-first. The clean-SFT adapter is the limited
model preview; no historical GRPO adapter is a promoted result.

## Install and test

```bash
python -m pip install -e ".[dev]"
pytest -q -p no:cacheprovider
```

Some tests intentionally skip in a public checkout:

- model-framework contract tests when optional `torch` or `trl` dependencies
  are not installed;
- deterministic rechecks that require the private retained-completion or
  verified-reference corpora;
- tests for operational scripts omitted from the public release boundary.

Those skips are evidence-access boundaries, not successful validations. The
maintainer release checklist separately requires the hydrated private parity
and reference-replay gates.

## Authority boundaries

- `athma_train.spice_gate` is the only raw ngspice subprocess boundary.
- `athma_train.env.rlvr_scorer.score_completion` is the canonical
  circuit-family reward authority.
- `athma_train.env.spice_env.SpiceEnv.reward` is a legacy generic measurement
  reward and is not used for the published RLVR result.
- Generated netlists are untrusted and must pass static validation before the
  resource-bounded sandbox starts.

Training admission is provenance-bound. A filename such as `promoted` or
`trainable` is not evidence; policy, prompt, tokenizer, simulator, scorer,
manual review, and exact content hashes must match the frozen experiment.

## License

Training-pipeline code is Apache-2.0. Datasets and model weights have the
separate educational/research license posture documented in the repository
release notice.
