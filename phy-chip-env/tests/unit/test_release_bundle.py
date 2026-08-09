from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
RELEASE = REPO / "release" / "v0.1"


def _load_release_validator():
    spec = importlib.util.spec_from_file_location(
        "phychip_release_validator",
        RELEASE / "validate_release.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not RELEASE.is_dir(), reason="release bundle is not in this checkout")
def test_release_bundle_validator_passes_for_recorded_source_state():
    manifest = json.loads(
        (RELEASE / "evidence_manifest.json").read_text(encoding="utf-8")
    )
    command = [sys.executable, str(RELEASE / "validate_release.py")]
    if manifest["release_status"] == "draft_from_dirty_feature_branch":
        command.append("--draft")

    completed = subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr

    optimized = subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONOPTIMIZE": "1"},
    )
    assert optimized.returncode == 0, optimized.stderr

    validator_tree = ast.parse(
        (RELEASE / "validate_release.py").read_text(encoding="utf-8")
    )
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(validator_tree))


def test_public_boundary_rejects_private_paths_credentials_and_stale_artifacts():
    validator = _load_release_validator()
    entries = {
        "plan/private_runner.sh": b"echo hidden",
        "slides/stale-paper.pdf": b"%PDF-1.7",
        "config/ssh_config": b"Host *\n  HostName 203.0.113.8\n",
        "notes.txt": (
            b"artifact at /"
            + b"Users/example/private and PI_API"
            + b"_KEY=secret"
        ),
        "ops.txt": (
            b"plan/sft_clean_eval_pod_run"
            + b".sh\n-----BEGIN OPENSSH "
            + b"PRIVATE KEY-----"
        ),
    }

    violations = validator.public_boundary_violations(entries)

    assert any("forbidden public path" in item for item in violations)
    assert any("forbidden public artifact type" in item for item in violations)
    assert any("SSH wildcard host" in item for item in violations)
    assert any("SSH real-IP host" in item for item in violations)
    assert any("absolute user-home path" in item for item in violations)
    assert any("credential assignment" in item for item in violations)
    assert any("private key" in item for item in violations)
    assert any("absent private runner" in item for item in violations)
