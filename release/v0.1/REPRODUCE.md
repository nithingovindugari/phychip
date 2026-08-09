# Reproducing the v0.1 release candidate

> **Local scorer-v9 candidate (2026-07-22):** 384 frozen current-policy
> rollouts were completely reviewed, defect classes were converted into
> mirrored regressions, and the preserved set was deterministically rescored.
> Source, parity, attack, and clean-wheel gates must all pass before tagging.

Run commands from the repository root. Do not compare results unless ngspice 46,
the prompt mode, chat template, termination IDs, completion budget, model
revision, adapter order, benchmark bytes, and scorer semantics all match.

## 1. Verify the standalone environment

```bash
bash phy-chip-env/scripts/install_ngspice46.sh
python3 -m venv /tmp/phychip-source-venv
/tmp/phychip-source-venv/bin/python -m pip install -e 'phy-chip-env[dev]'
PHYCHIP_REQUIRE_NGSPICE=46 /tmp/phychip-source-venv/bin/python -m pytest \
  phy-chip-env/tests -q -p no:cacheprovider
```

Expected in a public checkout: `257 passed, 1 skipped`. The single skip is the
private-corpus parity test. With the retained corpus present, that node must
pass instead of skip.

## 2. Build and test the wheel cleanly

```bash
UV_CACHE_DIR=/tmp/phychip-uv-cache uv build --wheel phy-chip-env
clean_dir=$(mktemp -d /tmp/phychip-clean-install.XXXXXX)
python3 -m venv "$clean_dir/venv"
"$clean_dir/venv/bin/python" -m pip install --no-deps \
  phy-chip-env/dist/phy_chip_env-0.1.0-py3-none-any.whl
cd /tmp
PHYCHIP_REQUIRE_NGSPICE=46 "$clean_dir/venv/bin/python" \
  /path/to/phy-chip/phy-chip-env/scripts/verify_clean_install.py
```

The clean smoke must report 23 harnesses, gold reward 1.3, attack rejection,
timeout, and a 64-character verifier hash. Compare that hash with
`phy_chip_env_runtime_fingerprint.json` and the wheel SHA-256 with the
`phy_chip_env_wheel` entry in `evidence_manifest.json`. This verifies
installation and core runtime behavior; the maintainer-only full gate supplies
the corrected-reference/106-output parity evidence.

## 3. Verify the clean-SFT primary-matrix record

The public source checkout does not contain the original GPU runner, adapter,
training corpus, or per-completion evaluation tree. It therefore cannot
independently rerun the published clean-SFT matrix. Those items are external or
maintainer-only evidence whose paths and SHA-256 values are recorded in
`evidence_manifest.json`; the evaluated contract is recorded in
`eval-environment.lock.json` and `MODEL_CARD.md`.

Public users can verify the repository artifacts and internal claim
consistency with the default validator in section 4. Maintainers hydrate the
hash-bound evidence tree and run the full validator. The original primary
contract was native SFT chat, greedy decoding, `max_new_tokens=2560`, and
ngspice 46; changing any of those settings is a new evaluation, not a
reproduction of the recorded matrix.

## 4. Verify content addresses

From a clean intended source commit, a maintainer with the private evidence
tree runs:

```bash
python3 release/v0.1/build_manifest.py \
  --evidence-root /path/to/private/phy-chip
python3 release/v0.1/validate_release.py
python3 release/v0.1/validate_release.py --full \
  --evidence-root /path/to/private/phy-chip
shasum -a 256 -c release/v0.1/SHA256SUMS
```

The generator refuses a dirty source tree. The default validator works in a
public checkout and verifies repository evidence plus internal claim
consistency. `--full` additionally requires every external and
maintainer-evidence artifact. The manifest records which of the three
distribution classes applies to each hash. No script commits, tags, pushes, or
publishes anything.

The binding is deliberately two-step: the manifest names a clean source
commit, and the eventual tag points to a metadata-only child. Validation rejects
any post-source change outside the manifest, runtime fingerprint, checksums,
and operational checklist. Before commit approval, maintainers may exercise
the same code paths with explicit `--draft`; draft status is never taggable.
