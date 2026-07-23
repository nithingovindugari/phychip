# phy-chip-env

`phy-chip-env` is the standalone verifiable-reward environment from PhyChip.
It turns a natural-language model completion containing an ngspice netlist and
a structured circuit specification into an auditable `ScoreResult`.

The package is the v0.1 research artifact. It is not a circuit-design product,
an official AnalogCoder evaluator, or proof that the released model generalizes
through RL. The current matched rank-64 GRPO result is positive but statistically
inconclusive.

## What is included

- one canonical `score(...) -> ScoreResult` API;
- 23 circuit-family measurement harnesses;
- an ngspice static safety gate and resource-bounded subprocess runner;
- structured reward components rather than a scalar-only result;
- ordered concurrent batch scoring;
- content-backed verifier/runtime fingerprints;
- authored gold, adversarial, timeout, harness, security, and parity tests.

Runtime Python dependencies are intentionally empty. ngspice is an external
system dependency and **version 46 is required** because observed benchmark
scores changed materially across simulator versions.

## Install

Supported production runtime: Linux, Python 3.10–3.12, ngspice 46.
macOS is supported for development with a native ngspice 46 install. Windows is
not supported by the POSIX resource sandbox.

```bash
bash scripts/install_ngspice46.sh
python -m pip install .
phy-chip-env doctor
```

The installer verifies the official source archive against the locked SHA-256
before building. A reproducible container recipe is in
[`docker/Dockerfile.ngspice46`](docker/Dockerfile.ngspice46).

## Score one completion

````python
from phy_chip_env import score

completion = """```spice
Vin Vin 0 DC 0 AC 1
Rg Vfb 0 10k
Rf Vout Vfb 90k
X1 Vin Vfb Vout ideal_opamp
.subckt ideal_opamp inp inn out
EGAIN out 0 inp inn 1e6
.ends ideal_opamp
.control
op
.endc
.end
```"""

result = score(
    circuit_id="noninv-amp",
    target_specs={"gain_db": [19.5, 20.5]},
    completion=completion,
)

print(result.reward)          # scalar for optimization
print(result.all_specs_met)   # scientific success criterion
print(result.to_event())      # complete reward decomposition
````

Command line:

```bash
phy-chip-score \
  --circuit-id noninv-amp \
  --targets '{"gain_db":[19.5,20.5]}' \
  --completion completion.txt \
  --require-success
```

Do not report `reward >= x` as physical success. The release contract defines
success as `ScoreResult.all_specs_met`, after simulation, anti-cheat admission,
measurement coverage, and every requested range check.

## Reward contract

The v0.1 scorer is intentionally spec-dominant and inspectable:

| Condition | Reward contribution |
|---|---:|
| malformed, unsafe, non-simulating, or family-invalid deck | `0.00` |
| valid deck with measured specification progress | `0.70 * spec_score` |
| every requested specification met | `+0.60` |

Mere formatting and simulation receive no policy reward. Known structural
shortcuts fail closed at zero. Missing measurements count against the
requested-spec denominator, and exact success remains 1.30. See
[`docs/REWARD_CONTRACT.md`](docs/REWARD_CONTRACT.md) and
[`docs/CAPABILITY_MATRIX.md`](docs/CAPABILITY_MATRIX.md).

## Security boundary

Model-generated SPICE is untrusted. The runner rejects host-command and file-I/O
features before process creation, uses a new scratch directory, a minimal
environment, CPU/address-space/file/process limits, bounded output files, a
wall-clock timeout, and drops to `nobody` when invoked as root.

This is not a complete kernel sandbox: it does not create network, mount, or
seccomp namespaces, and a non-root caller runs ngspice as that caller. Use the
container for hostile multi-tenant workloads. See [`SECURITY.md`](SECURITY.md).

## Tests

```bash
uv sync --extra dev
PHYCHIP_REQUIRE_NGSPICE=46 uv run pytest -q -p no:cacheprovider
```

The parent PhyChip repository adds a mandatory parity test over the 44
historical references that remain valid under scorer v9 and all 106 retained
clean-SFT/rank-64 completions. Private model
outputs are deliberately not distributed in the wheel.

## License

Framework source and authored test fixtures are Apache-2.0. ngspice is a
separate external program under its own license; it is not bundled in the wheel.
PhyChip model weights and released datasets have a separate
CC-BY-NC-SA-4.0 educational/research posture.
