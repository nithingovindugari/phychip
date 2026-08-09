# Reward contract

## Public function

```python
score(
    circuit_id: str,
    target_specs: dict,
    completion: str,
    *,
    eval_tol: float = 0.30,
    timeout_s: float = 15.0,
) -> ScoreResult
```

`target_specs` maps a harness figure-of-merit name to either an inclusive
`[low, high]` range or a scalar interpreted with `eval_tol`. The scientific
success field is `all_specs_met`, never a reward threshold.

## Ordered evaluation

1. Extract a closed `spice` Markdown fence. Raw-text normalization exists for
   compatibility, but an unfenced completion receives no score.
2. Record structural diagnostics: exact `.end`, at least one device, and an
   analysis directive.
3. Run the deck through the static safety gate and sandboxed ngspice 46.
4. Apply conservative family/topology anti-cheat checks.
5. Load the named family harness and run its injected measurement testbench.
6. Compare finite measured values with requested ranges.

## Scalar shaping

The current contract is `phy_chip_env.score@v9`:

- malformed, unsafe, non-simulating, or family-invalid: `0.00`;
- measured specification credit: `0.70 * normalized_spec_score`;
- all specifications met: `+0.60`.

Formatting, device presence, analysis directives, and simulator success remain
observable diagnostics in `ScoreResult`; they are not reward. This prevents a
runnable but physically irrelevant deck from receiving positive advantage.

For a requested metric outside its interval, partial credit decreases linearly
from 0.25 using a scale based on interval width and midpoint magnitude. A
missing, invalid, or non-finite measurement contributes zero but remains in the
denominator. The maximum reward remains 1.30.

## Anti-cheat admission

Every family rejects `laplace` and `tf(` shortcuts and an independent voltage
source directly across the evaluated `Vout`/`out` node. Transistor families
require parsed top-level MOSFET/BJT cards outside `.control` and `.subckt`
bodies. Frequency-dependent families require their minimum capacitor count.
Scorer v9 also checks declared AC ports and load-bearing topology for the
families below:

- inverting amplifier: Vin/output resistor feedback node; reject wrong
  positive-input polarity on a positive-gain output-driving VCVS;
- non-inverting amplifier: output/reference feedback divider not tied to Vin;
- summing amplifier: V1, V2, V3 and output meet at one resistor node; reject
  wrong positive-input polarity on a positive-gain output-driving VCVS;
- common-source stage: Vin controls a load-bearing transistor whose signal
  path reaches the measured output;
- differential pair/amplifier: distinct Vp/Vn input devices share a biased
  tail and feed an output branch;
- integrator: a resistor input and capacitor feedback meet at the consumed
  summing node, with no parallel Vin-tracking cancellation path;
- differentiator: a capacitor input and resistor feedback meet at the
  consumed summing node;
- Sallen-Key: the active output stage consumes the two-R/two-C filter state;
- Schmitt trigger: output-to-reference feedback divider consumed by the
  output-driving comparator, followed by finite `vth_high_v > vth_low_v`
  before any partial or exact reward;
- gain trim: reject direct behavioral/VCVS output transfer from Vin/Vtrim,
  including passive aliases without output feedback, and require measured
  gain modulation rather than output-offset movement.

These rules reduce known reward hacking; they do not prove that every accepted
circuit is semantically the topology requested. The capability matrix marks
families without a topology-specific admission rule.

## Versioning

Changing extraction, static safety, topology admission, measurement semantics,
partial-credit math, or success criteria requires a new scorer semantics ID and
new verifier hash. Historical evidence keeps the historical verifier identity;
it is not relabeled after package extraction.

`score@v9` was introduced after complete manual review of 384 current-policy
rollouts found control-language device impersonation, disconnected inputs,
decorative feedback, wrong reactive topology, and trim-offset circuits that
did not modulate gain. Historical scores retain their original identities.
