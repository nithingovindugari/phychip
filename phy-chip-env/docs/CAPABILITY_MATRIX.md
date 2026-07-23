# Capability matrix

Every row names the figures of merit returned by the v0.1 harness. “Structural
admission” is a conservative anti-cheat check before measurements; “global”
means only the cross-family shortcut rules plus simulator/harness behavior.
Passing a row means the requested measured ranges passed, not that layout,
device-model fidelity, stability across corners, or manufacturability was proven.

| Circuit ID | Tier | Injected analyses / measured figures of merit | Structural admission |
|---|---:|---|---|
| `inv-amp` | 0 | DC/AC: gain, bandwidth, output-swing proxy, input offset | Requires a Vin/output resistor feedback node; a positive-gain output-driving VCVS may not sense that node on its positive input, including through a resistor-chain output path |
| `noninv-amp` | 0 | DC/AC: gain, bandwidth, output-swing proxy, input offset | Requires an output/reference feedback divider isolated from Vin, consumption of both Vin and feedback by the controlled/subcircuit stages, and non-positive feedback polarity for an output-driving positive-gain VCVS |
| `buffer` | 0 | DC/AC plus AC current injection: gain, signed low-frequency transfer, bandwidth, output impedance, input offset | Requires the output driver to consume the non-inverting Vin path and rejects a tracked negative-control path; measured signed gain must be finite, positive, and 0.25--5 V/V. This is a low-gain non-inverting family check, not proof of exact unity feedback |
| `summing-amp` | 0 | Separate input excitation + AC: gain from V1/V2/V3, bandwidth | Requires a V1/V2/V3/output resistor summing node; a positive-gain VCVS whose output reaches Vout directly or through a safe resistor path may not sense that node on its positive input |
| `diff-amp` | 0 | Differential/common-mode DC/AC: differential gain, CMRR, bandwidth, offset | Requires distinct Vp/Vn-controlled input devices, a shared biased tail, and an output branch reached from the pair |
| `integrator` | 0 | AC response: unity-gain frequency, DC gain, output-swing proxy | Requires resistor input and capacitor feedback at a consumed summing node; rejects a parallel Vin-tracking cancellation path |
| `differentiator` | 0 | AC response: unity-gain frequency, high-frequency gain, output-swing proxy | Requires capacitor input and resistor feedback at a consumed summing node |
| `current-mirror` | 0 | Reference-current injection + output sweep: ratio, output resistance, compliance | Requires at least one transistor |
| `diff-pair` | 0 | Differential/common-mode AC + operating point: gain, CMRR, tail current, offset | Requires distinct Vp/Vn-controlled input devices, a shared biased tail, and an output branch reached from the pair |
| `cs-stage` | 0 | Operating point + AC: gain, bandwidth, output-swing proxy, quiescent current | Requires Vin to control a load-bearing transistor whose signal path reaches Vout; full small-signal mechanism is not formally proven |
| `buck` | 1 | Transient/steady-state: output voltage, ripple, efficiency, settling time | Global only; load/source branches must be measurable |
| `boost` | 1 | Transient/steady-state: output voltage, ripple, efficiency, average inductor current | Global only; inductor/load/source branches must be measurable |
| `ldo` | 1 | DC supply/load sweeps + AC: output, dropout, line/load regulation, 1-kHz PSRR | Global only; regulator topology not proven |
| `bandgap` | 1 | Temperature/supply DC sweeps + 1-kHz AC: Vref, tempco, line regulation, PSRR | Requires at least one transistor; bandgap mechanism not proven |
| `sallen-key` | 1 | AC sweep: cutoff, DC gain, Q estimate, stopband attenuation | Requires two capacitors, a two-node filter skeleton, and an active output stage that consumes the shunt-capacitor state |
| `schmitt` | 1 | DC input sweep + transient: high/low thresholds, hysteresis, delay | Requires output/reference feedback consumed by the behavioral/VCVS/subcircuit output driver, plus finite ordered high > low thresholds before any reward |
| `current-sense` | 1 | Differential/common-mode DC/AC: gain, bandwidth, common-mode range, offset | Requires at least one transistor; sense topology not proven |
| `level-shift` | 1 | Transient logic excitation: VOL, VOH, rise/fall time, propagation delay | Requires at least one transistor; rail translation mechanism not proven |
| `ina` | 1 | Differential/common-mode AC: gain, CMRR, bandwidth, input offset | Global only; three-op-amp INA topology not proven |
| `gain-trim` | 1 | AC + DC trim sweep: gain, offset, trim range, bandwidth, gain modulation | Rejects direct and transitive controlled-source Vin/Vtrim output aliases lacking feedback and requires at least 0.01 dB measured gain change across the trim sweep; additive offset alone does not qualify |
| `tia` | 2 | AC/noise-oriented probes: transimpedance, bandwidth, input-noise proxy, phase-margin proxy | Requires at least one capacitor; exact photodiode/TIA topology not proven |
| `bridge-ina` | 2 | Bridge-imbalance/common-mode DC + AC: sensitivity, CMRR, output offset, bandwidth | Global only; bridge/INA topology not fully proven |
| `h-bridge` | 2 | PWM transient: motor swing, on-resistance proxy, switching-loss proxy, deadtime | Global only; four-switch direction/safety topology not fully proven |

## Known limitations

- Several “output swing” and stability values are testbench proxies, not device
  corner sign-off measurements.
- Independent voltage sources directly across `Vout`/`out` are rejected
  globally. The scorer does not otherwise prove rail-limited large-signal
  behavior for ideal controlled-source amplifier macros.
- The harness may use requested values to choose a sweep range; it must never
  copy a target into the returned measurement.
- Harness failures currently fail closed as missing measurements rather than
  returning a typed failure reason.
- Formatting, successful simulation, and family admission are diagnostics only;
  scorer v9 pays reward solely for measured specification progress.
- The scorer may run the original deck and an injected measurement deck, so one
  score can launch ngspice more than once.
- Manual review remains necessary for benchmark claims about topology quality.
