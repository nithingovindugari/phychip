from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGED_SCORER = REPO_ROOT / "phy-chip-env" / "src" / "phy_chip_env" / "scorer.py"
PACKAGED_SANDBOX = REPO_ROOT / "phy-chip-env" / "src" / "phy_chip_env" / "sandbox.py"
TRAIN_SCORER = REPO_ROOT / "athma-train" / "athma_train" / "env" / "rlvr_scorer.py"
TRAIN_SANDBOX = REPO_ROOT / "athma-train" / "athma_train" / "spice_gate.py"
PACKAGED_HARNESSES = (
    REPO_ROOT / "phy-chip-env" / "src" / "phy_chip_env" / "harnesses"
)
TRAIN_HARNESSES = REPO_ROOT / "athma-train" / "athma_train" / "harnesses"

SCORER_SHARED_DEFINITIONS = {
    "ScoreResult",
    "_active_top_level_lines",
    "_result",
    "anticheat_ok",
    "extract_netlist",
    "score_completion",
}
PACKAGED_ONLY_SCORER_DEFINITIONS = {"ScoreRequest", "score"}
TRAINING_ONLY_SCORER_DEFINITIONS = {"load_harness"}
SCORER_SHARED_ASSIGNMENTS = {"SCORER_SCHEMA"}
PACKAGED_ONLY_SCORER_ASSIGNMENTS = {"SCORER_SEMANTICS"}
TRAINING_ONLY_SCORER_ASSIGNMENTS = {
    "HARNESSES_DIR",
    "SCORER_SEMANTICS",
    "_HARNESS_CACHE",
}
SANDBOX_SHARED_DEFINITIONS = {
    "UnsafeNetlistError",
    "_nobody_ids",
    "_read_limited",
    "_sandbox_environment",
    "grade_netlist",
    "has_analysis_directive",
    "has_device_line",
    "normalize_netlist",
    "run_ngspice_sandboxed",
    "validate_netlist_safety",
}
PACKAGED_ONLY_SANDBOX_DEFINITIONS = {
    "ngspice_version_output",
    "require_ngspice_version",
}
TRAINING_ONLY_SANDBOX_DEFINITIONS = {"_check_ngspice_version"}
SANDBOX_SHARED_ASSIGNMENTS = {
    "ANALYSIS_DIRECTIVES",
    "ANALYSIS_RAN_MARKERS",
    "DANGEROUS_DOTCARDS",
    "DEVICE_PREFIXES",
    "HARD_FAILURES",
    "SAFE_CONTROL_COMMANDS",
    "SAFE_DOTCARDS",
    "_CONTROL_COMMAND_RE",
    "_CONTROL_META_RE",
    "_DEFAULT_MEMORY_LIMIT_MB",
    "_DOTCARD_RE",
    "_EXTERNAL_FILE_RE",
    "_NGSPICE_EXECUTION_LOCK",
    "_OPEN_FILE_LIMIT",
    "_OUTPUT_FILE_LIMIT_BYTES",
    "_SANDBOX_LAUNCHER",
}
PACKAGED_ONLY_SANDBOX_ASSIGNMENTS = {
    "REQUIRED_NGSPICE_MAJOR",
    "_NGSPICE_VERSION_CACHE",
    "_NGSPICE_VERSION_LOCK",
    "_check_ngspice_version",
}
TRAINING_ONLY_SANDBOX_ASSIGNMENTS = {"_NGSPICE_VERSION_CHECKED"}


def _definitions(path: Path) -> dict[str, ast.AST]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }


def _assignments(path: Path) -> dict[str, ast.AST]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = node.value
    return assignments


class _ScorerCompatibilityNormalizer(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in {"get_harness", "load_harness"}:
            return ast.copy_location(ast.Name(id="_harness_loader", ctx=node.ctx), node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and len(node.args) == 2
            and isinstance(node.args[1], ast.BinOp)
            and isinstance(node.args[1].op, ast.BitOr)
        ):
            node.args[1] = ast.Tuple(
                elts=[node.args[1].left, node.args[1].right],
                ctx=ast.Load(),
            )
        return node


def _normalized_dump(node: ast.AST, *, scorer: bool = False) -> str:
    if scorer:
        node = _ScorerCompatibilityNormalizer().visit(ast.fix_missing_locations(node))
    return ast.dump(node, include_attributes=False)


class _HarnessCompatibilityNormalizer(ast.NodeTransformer):
    """Normalize package imports and supported-Python syntax differences."""

    def visit_Import(self, node: ast.Import) -> None:
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        return None

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and len(node.args) == 2
            and isinstance(node.args[1], ast.BinOp)
            and isinstance(node.args[1].op, ast.BitOr)
        ):
            node.args[1] = ast.Tuple(
                elts=[node.args[1].left, node.args[1].right],
                ctx=ast.Load(),
            )
        return node


def _normalized_harness_imports(tree: ast.AST) -> tuple[tuple[str, ...], ...]:
    module_aliases = {
        "athma_train.harnesses._measure_utils": "harnesses._measure_utils",
        "athma_train.spice_gate": "sandbox",
        "phy_chip_env.harnesses._measure_utils": "harnesses._measure_utils",
        "phy_chip_env.sandbox": "sandbox",
    }
    imports: list[tuple[str, ...]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                ("import", alias.name, alias.asname or "") for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            module = module_aliases.get(node.module or "", node.module or "")
            imports.extend(
                ("from", module, alias.name, alias.asname or "")
                for alias in node.names
            )
    return tuple(sorted(imports))


def _normalized_harness(path: Path) -> tuple[str, tuple[tuple[str, ...], ...]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = _normalized_harness_imports(tree)
    normalized = _HarnessCompatibilityNormalizer().visit(tree)
    source = ast.dump(ast.fix_missing_locations(normalized), include_attributes=False)
    return source, imports


def test_all_23_training_harness_modules_are_source_identical() -> None:
    packaged = sorted(
        path
        for path in PACKAGED_HARNESSES.glob("*.py")
        if path.name not in {"__init__.py", "_measure_utils.py"}
    )
    training = sorted(
        path
        for path in TRAIN_HARNESSES.glob("*.py")
        if path.name not in {"__init__.py", "_measure_utils.py"}
    )

    assert len(packaged) == 23
    assert {path.name.replace("_", "-") for path in packaged} == {
        path.name for path in training
    }
    for packaged_path in packaged:
        training_path = TRAIN_HARNESSES / packaged_path.name.replace("_", "-")
        assert training_path.is_file(), training_path.name
        assert _normalized_harness(packaged_path) == _normalized_harness(
            training_path
        ), packaged_path.name


def test_training_measurement_utilities_are_source_identical() -> None:
    packaged_path = PACKAGED_HARNESSES / "_measure_utils.py"
    training_path = TRAIN_HARNESSES / "_measure_utils.py"

    assert _normalized_harness(packaged_path) == _normalized_harness(training_path)


def test_training_sandbox_shared_functions_are_source_identical() -> None:
    packaged = _definitions(PACKAGED_SANDBOX)
    training = _definitions(TRAIN_SANDBOX)

    assert set(packaged) == (
        SANDBOX_SHARED_DEFINITIONS | PACKAGED_ONLY_SANDBOX_DEFINITIONS
    )
    assert set(training) == (
        SANDBOX_SHARED_DEFINITIONS | TRAINING_ONLY_SANDBOX_DEFINITIONS
    )
    for name in sorted(SANDBOX_SHARED_DEFINITIONS):
        assert _normalized_dump(packaged[name]) == _normalized_dump(training[name]), name


def test_training_sandbox_constant_inventory_and_values_are_source_identical() -> None:
    packaged = _assignments(PACKAGED_SANDBOX)
    training = _assignments(TRAIN_SANDBOX)

    assert set(packaged) == (
        SANDBOX_SHARED_ASSIGNMENTS | PACKAGED_ONLY_SANDBOX_ASSIGNMENTS
    )
    assert set(training) == (
        SANDBOX_SHARED_ASSIGNMENTS | TRAINING_ONLY_SANDBOX_ASSIGNMENTS
    )
    for name in sorted(SANDBOX_SHARED_ASSIGNMENTS):
        assert _normalized_dump(packaged[name]) == _normalized_dump(training[name]), name


def test_training_scorer_is_source_identical_except_loader_compatibility() -> None:
    packaged = _definitions(PACKAGED_SCORER)
    training = _definitions(TRAIN_SCORER)

    assert set(packaged) == (
        SCORER_SHARED_DEFINITIONS | PACKAGED_ONLY_SCORER_DEFINITIONS
    )
    assert set(training) == (
        SCORER_SHARED_DEFINITIONS | TRAINING_ONLY_SCORER_DEFINITIONS
    )
    for name in sorted(SCORER_SHARED_DEFINITIONS):
        assert _normalized_dump(packaged[name], scorer=True) == _normalized_dump(
            training[name], scorer=True
        ), name


def test_training_scorer_constant_inventory_and_values_are_source_identical() -> None:
    packaged = _assignments(PACKAGED_SCORER)
    training = _assignments(TRAIN_SCORER)

    assert set(packaged) == (
        SCORER_SHARED_ASSIGNMENTS | PACKAGED_ONLY_SCORER_ASSIGNMENTS
    )
    assert set(training) == (
        SCORER_SHARED_ASSIGNMENTS | TRAINING_ONLY_SCORER_ASSIGNMENTS
    )
    for name in sorted(SCORER_SHARED_ASSIGNMENTS):
        assert _normalized_dump(packaged[name]) == _normalized_dump(training[name]), name


def test_both_runtime_boundaries_require_ngspice_46(monkeypatch) -> None:
    def load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    packaged = load("packaged_sandbox_parity", PACKAGED_SANDBOX)
    training = load("training_sandbox_parity", TRAIN_SANDBOX)
    monkeypatch.setenv("PHYCHIP_REQUIRE_NGSPICE", "46")
    monkeypatch.setattr(packaged.shutil, "which", lambda _name: "/usr/bin/ngspice")
    monkeypatch.setattr(training.shutil, "which", lambda _name: "/usr/bin/ngspice")
    version = "ngspice-45"
    monkeypatch.setattr(packaged, "ngspice_version_output", lambda _path: version)
    monkeypatch.setattr(
        training.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Completed", (), {"stdout": version})(),
    )

    for check in (packaged.require_ngspice_version, training._check_ngspice_version):
        try:
            check("ngspice")
        except RuntimeError as exc:
            assert "46" in str(exc)
        else:
            raise AssertionError("ngspice-45 did not fail the ngspice-46 contract")
