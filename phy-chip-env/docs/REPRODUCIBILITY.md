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

Run tests against the installed wheel from the source checkout:

```bash
cd /path/to/phy-chip-env
python -m pip install '.[dev]'
PHYCHIP_REQUIRE_NGSPICE=46 pytest -q -p no:cacheprovider
```

The parent repository parity gate additionally requires:

- all 53 retained gold references pass;
- all 106 retained clean-SFT/rank-64 completions match the athma-train scorer;
- all 12 authored reward-hacking cases stay below success;
- an actual sub-millisecond wall-clock timeout fails closed.

## Provenance

`phy-chip-env doctor` returns:

- package/scorer version;
- complete ngspice version banner;
- resolved ngspice binary path and SHA-256;
- hashes of the scorer, sandbox, shared measurement utilities, and 23 harnesses;
- one canonical verifier hash over that payload.

Preserve the complete JSON beside every train, label, or evaluation result.
