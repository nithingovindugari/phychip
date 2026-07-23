"""Content-backed provenance fingerprints for exact-regime RLVR labeling."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

from athma_train.grpo_contract import prompt_contract_hash_from_template


_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_MODEL_FILE_NAMES = {"adapter_config.json", "config.json"}
_MODEL_FILE_SUFFIXES = {".safetensors", ".bin"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _local_model_identity(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise ValueError(f"policy adapter must be a local directory: {path}")
    files = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file()
        and (
            candidate.name in _MODEL_FILE_NAMES
            or candidate.suffix in _MODEL_FILE_SUFFIXES
        )
    )
    if not files:
        raise ValueError(f"no adapter/model identity files found in {path}")
    if not any(candidate.name == "adapter_config.json" for candidate in files):
        raise ValueError(f"adapter_config.json is missing from {path}")
    if not any(candidate.suffix in _MODEL_FILE_SUFFIXES for candidate in files):
        raise ValueError(f"adapter weights are missing from {path}")
    return {
        "files": [
            {"name": candidate.name, "sha256": sha256_file(candidate)}
            for candidate in files
        ]
    }


def policy_hash(
    base_repo: str,
    base_revision: str,
    adapter_paths: Sequence[Path],
) -> str:
    """Hash the pinned base identity and exact local adapter bytes in merge order."""

    revision = base_revision.casefold()
    if not _COMMIT_RE.fullmatch(revision):
        raise ValueError("base revision must be a 40-64 character commit hash")
    if not base_repo.strip():
        raise ValueError("base repo must be non-empty")
    return canonical_hash(
        {
            "schema": "phychip-policy-provenance-v1",
            "base": {"repo_id": base_repo, "revision": revision},
            "adapters_in_merge_order": [
                _local_model_identity(Path(path)) for path in adapter_paths
            ],
        }
    )


def _circuit_ids(specs_path: Path) -> set[str]:
    circuit_ids: set[str] = set()
    with specs_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{specs_path}:{line_number}: invalid JSON") from exc
            circuit_id = row.get("circuit_id") if isinstance(row, dict) else None
            if not isinstance(circuit_id, str) or not circuit_id:
                raise ValueError(f"{specs_path}:{line_number}: missing circuit_id")
            circuit_ids.add(circuit_id)
    if not circuit_ids:
        raise ValueError(f"no specifications found in {specs_path}")
    return circuit_ids


def ngspice_version() -> str:
    try:
        completed = subprocess.run(
            ["ngspice", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"cannot fingerprint ngspice runtime: {exc}") from exc
    version = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if not version:
        raise ValueError("cannot fingerprint ngspice runtime: empty version output")
    return version


def canonical_ngspice_runtime_identity(version_output: str) -> str:
    """Remove nondeterministic build timestamps from an ngspice banner.

    Source builds embed a ``Creation Date`` even when the ngspice version,
    compiled features, scorer, sandbox, and harness sources are identical.
    Keeping that timestamp in the verifier hash made a Mac rescore and an A100
    launch disagree for no semantic reason.  All other non-empty banner lines
    remain part of the identity, so version/build-feature changes still fail
    closed.
    """

    stable_lines: list[str] = []
    for raw_line in version_output.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        marker = line.lstrip("*").strip().casefold()
        if marker.startswith("creation date:"):
            continue
        stable_lines.append(line)
    identity = "\n".join(stable_lines)
    if not identity:
        raise ValueError("cannot fingerprint ngspice runtime: empty stable identity")
    return identity


def verifier_hash(specs_path: Path, train_root: Path, version: str) -> str:
    """Hash the canonical scorer, sandbox gate, harnesses, and ngspice version."""

    package = train_root / "athma_train"
    requested_circuit_ids = _circuit_ids(specs_path)
    harness_dir = package / "harnesses"
    active_harnesses = sorted(
        path
        for path in harness_dir.glob("*.py")
        if path.name != "__init__.py" and not path.name.startswith("_")
    )
    available_circuit_ids = {path.stem for path in active_harnesses}
    missing_circuit_ids = sorted(requested_circuit_ids - available_circuit_ids)
    if missing_circuit_ids:
        raise ValueError(
            "cannot fingerprint missing verifier harnesses: "
            + ", ".join(missing_circuit_ids)
        )
    files = [
        package / "env" / "rlvr_scorer.py",
        package / "spice_gate.py",
        harness_dir / "_measure_utils.py",
    ]
    # Hash every active harness so labeling the full candidate pool and training
    # a derived family subset produce the same verifier identity.
    files.extend(active_harnesses)
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise ValueError(f"cannot fingerprint missing verifier files: {missing}")
    return canonical_hash(
        {
            "schema": "phychip-rlvr-scorer-provenance-v3",
            "ngspice_runtime_identity": canonical_ngspice_runtime_identity(version),
            "files": [
                {
                    "path": path.relative_to(train_root).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in files
            ],
        }
    )


def build_experiment_provenance(
    *,
    base_repo: str,
    base_revision: str,
    adapter_paths: Sequence[Path],
    specs_path: Path,
    train_root: Path,
    chat_template_path: Path,
    version: str | None = None,
) -> dict[str, Any]:
    """Build the three hashes required by exact-regime-v2 labels."""

    if not chat_template_path.is_file():
        raise ValueError(f"chat template is missing: {chat_template_path}")
    runtime_version = version if version is not None else ngspice_version()
    return {
        "schema": "phychip-experiment-provenance-v1",
        "base_repo": base_repo,
        "base_revision": base_revision.casefold(),
        "adapter_paths": [str(path) for path in adapter_paths],
        "specs_path": str(specs_path),
        "ngspice_version_output": runtime_version,
        "policy_hash": policy_hash(base_repo, base_revision, adapter_paths),
        "verifier_hash": verifier_hash(specs_path, train_root, runtime_version),
        "prompt_contract_hash": prompt_contract_hash_from_template(
            chat_template_path.read_text(encoding="utf-8")
        ),
    }
