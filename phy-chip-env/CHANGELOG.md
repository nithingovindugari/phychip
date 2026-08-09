# Changelog

## 0.1.0 — release candidate

- Extracted the canonical simulator-backed scorer from the training repository.
- Added an explicit registry for all 23 active circuit harnesses.
- Added structured `ScoreResult` diagnostics and ordered batch scoring.
- Made ngspice 46 the default mandatory runtime.
- Hardened version-check caching so a caught mismatch cannot disable later checks.
- Preserved the line-anchored control-block parser repair.
- Added gold, adversarial, timeout, security, harness, wheel, and athma-train
  parity gates.
- Added content-backed runtime/verifier fingerprints.
- Advanced scorer semantics through v9 before release (frozen constant
  `SCORER_SEMANTICS = "phy_chip_env.score@v9"`). Through v7: reject direct
  output clamps, wrong-polarity positive-gain VCVS feedback in inverse/summing
  amplifiers even when the driver reaches the output through a resistor chain,
  non-inverting stages that ignore `Vin`, decorative Schmitt dividers, and
  non-hysteretic Schmitt partial reward. v8-v9 add control-language device
  impersonation, disconnected inputs, wrong reactive topology, decorative
  feedback, hidden AC stimuli, and gain-trim offset movement without gain
  modulation.
- Require ngspice exit code zero even when its output contains an analysis
  marker.
