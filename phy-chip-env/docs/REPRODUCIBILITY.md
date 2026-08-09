# Reproducibility

## Supported release environment

- Python 3.10–3.12;
- Linux for production, macOS for development;
- ngspice 46 built from the official source archive;
- archive SHA-256
  `a0d1699af1940b06649276dcd6ff5a566c8c0cad01b2f7b5e99dedbb4d64c19b`;
- configure flags `--disable-debug --with-readline=yes --enable-openmp`;
- serial `make -j1 CCLD=g++` to avoid the observed final-link race.

## Clean wheel check

`scripts/install_ngspice46.sh` is an Ubuntu/Debian installer: it uses
`apt-get`, `sha256sum`, and `ldconfig`. Run it on Linux or in the pinned
Dockerfile; do not run it on macOS.

Linux:

```bash
python3.12 -m venv /tmp/phy-chip-env-venv
. /tmp/phy-chip-env-venv/bin/activate
python -m pip install -U pip uv
bash scripts/install_ngspice46.sh
uv build --wheel .
python -m pip install --force-reinstall dist/phy_chip_env-0.1.0-py3-none-any.whl
cd /tmp
python -I -c 'from phy_chip_env import score, SCORER_SEMANTICS; print(SCORER_SEMANTICS)'
phy-chip-env doctor
```

macOS development requires an independently installed ngspice 46. Install the
build prerequisites with Homebrew, build the official ngspice 46 archive using
the pinned SHA-256 and configure flags above, put that binary on `PATH`, and
verify `ngspice --version` reports `ngspice-46` before installing the wheel.
The Linux installer is the release recipe; a macOS build is a development
prerequisite and does not replace the recorded release runtime.

Run tests against the installed wheel from the source checkout:

```bash
cd /path/to/phy-chip-env
python -m pip install '.[dev]'
PHYCHIP_REQUIRE_NGSPICE=46 pytest -q -p no:cacheprovider
```

The maintainer release gate additionally requires:

- 44/53 corrected retained references pass and the nine invalid historical
  gain-trim labels remain quarantined;
- all 106 retained clean-SFT/rank-64 completions match the athma-train scorer;
- all 22 authored reward-hacking cases stay below success;
- an actual sub-millisecond wall-clock timeout fails closed.

These counts are preservation and attack-regression checks, not a model-quality
result. A scored completion is reported as four separate outcomes: structure
accepted, simulation completed, topology accepted, and every requested numeric
constraint met. Merely compiling or simulating is never called success.

The public checkout does not distribute the retained reference/completion
corpus, so the private-corpus parity test is an intentional skip. Maintainers
run it before release and record the hash-bound result in the release evidence;
the public suite's exact pass count is intentionally not hard-coded here.

`PhyChipEnv` is the configuration authority for both `score()` and
`score_batch()`: its `eval_tol` and `timeout_s` override values carried by an
individual `ScoreRequest`. The functional `score_batch()` API instead uses each
request's values. Batch order, errors, and timeout behavior match repeated
functional scoring.

## Provenance

`phy-chip-env doctor` returns:

- package/scorer version;
- complete ngspice version banner;
- resolved ngspice binary path and SHA-256;
- hashes of the scorer, sandbox, shared measurement utilities, and 23 harnesses;
- one canonical verifier hash over that payload.

Preserve the complete JSON beside every train, label, or evaluation result.
