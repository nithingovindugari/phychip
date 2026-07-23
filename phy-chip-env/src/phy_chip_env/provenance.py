"""Content-backed provenance for the installed verifier and ngspice runtime."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from . import harnesses, sandbox, scorer
from .registry import list_harnesses


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _package_version() -> str:
    try:
        return version("phy-chip-env")
    except PackageNotFoundError:
        return "0.1.0+source"


def runtime_fingerprint(ngspice_bin: str = "ngspice") -> dict[str, Any]:
    """Fingerprint scorer sources, all harnesses, and the simulator binary."""

    executable_text = shutil.which(ngspice_bin) or ngspice_bin
    executable = Path(executable_text).resolve()
    completed = subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    version_output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    sandbox.require_ngspice_version(str(executable))

    harness_root = Path(harnesses.__file__).resolve().parent
    source_files = [
        Path(scorer.__file__).resolve(),
        Path(sandbox.__file__).resolve(),
        harness_root / "_measure_utils.py",
    ]
    source_files.extend(harness_root / f"{info.module_name}.py" for info in list_harnesses())
    source_manifest = [{"name": path.name, "sha256": _sha256_file(path)} for path in source_files]
    payload = {
        "schema": "phy-chip-env-runtime-fingerprint-v1",
        "package_version": _package_version(),
        "scorer_semantics": scorer.SCORER_SEMANTICS,
        "ngspice": {
            "path": str(executable),
            "sha256": _sha256_file(executable),
            "version_output": version_output,
        },
        "sources": source_manifest,
    }
    return {**payload, "verifier_hash": _canonical_hash(payload)}
