from __future__ import annotations

import concurrent.futures
import importlib.util
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from athma_train import spice_gate
from athma_train.env.spice_env import SpiceEnv, SpiceEnvConfig

BASE_DECK = """\
* ordinary circuit
V1 in 0 DC 1
R1 in out 1k
R2 out 0 1k
.op
.end
"""


@pytest.mark.parametrize(
    "command",
    [
        "shell touch /tmp/pwned",
        "source /etc/passwd",
        "cd /",
        "write /tmp/x",
        "wrdata /tmp/x all",
    ],
)
def test_dangerous_control_commands_are_rejected_before_subprocess(monkeypatch, command):
    calls = []

    def must_not_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("subprocess must not run for an unsafe netlist")

    monkeypatch.setattr(spice_gate.subprocess, "run", must_not_run)
    deck = f"V1 in 0 1\nR1 in 0 1k\n.control\nop\n{command}\n.endc\n.end\n"

    passed, reason = spice_gate.grade_netlist(deck)

    assert not passed
    assert "unsafe netlist" in reason
    assert not calls


@pytest.mark.parametrize(
    "dotcard",
    [
        ".include /etc/passwd",
        ".lib /tmp/models.lib",
        ".load /tmp/plugin.so",
        ".osdi /tmp/model.osdi",
    ],
)
def test_file_loading_dotcards_are_rejected_before_subprocess(monkeypatch, dotcard):
    monkeypatch.setattr(
        spice_gate.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("unsafe deck reached subprocess"),
    )
    deck = f"V1 in 0 1\nR1 in 0 1k\n{dotcard}\n.op\n.end\n"

    passed, reason = spice_gate.grade_netlist(deck)

    assert not passed
    assert "dangerous dotcard" in reason


@pytest.mark.parametrize(
    ("body", "reason_fragment"),
    [
        (".control\nop\n.control\n.endc\n.endc", "nested .control"),
        (".endc", ".endc without .control"),
        (".control\nop", "unterminated .control"),
        (".control\nop; shell true\n.endc", "chaining/redirection"),
        (".control extra\nop\n.endc", "malformed .control"),
    ],
)
def test_malformed_or_chained_control_blocks_fail_closed(body, reason_fragment):
    safe, reason = spice_gate.validate_netlist_safety(
        f"V1 in 0 1\nR1 in 0 1k\n{body}\n.end\n"
    )

    assert not safe
    assert reason_fragment in reason


def test_canonical_harness_control_commands_are_allowlisted():
    deck = """\
V1 in 0 DC 0 AC 1
R1 in out 1k
C1 out 0 1u
.control
set noaskquit
alter @V1[acmag]=1
ac dec 50 1 1e8
let gain = vm(out)
meas ac peak max vm(out) from=1 to=1e8
echo __measure__
print frequency gain
save all
setplot ac1
reset
quit
.endc
.end
"""

    assert spice_gate.validate_netlist_safety(deck) == (True, "ok")
    assert spice_gate.has_analysis_directive(deck.lower())


def test_ordinary_deck_runs_in_isolated_sanitized_sandbox(monkeypatch):
    captured = {"returncode": 0}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        assert Path(kwargs["cwd"]).name.startswith("phychip-ngspice-")
        assert Path(argv[-1]).parent == Path(kwargs["cwd"])
        assert Path(argv[-1]).read_text(encoding="utf-8") == BASE_DECK.strip()
        return subprocess.CompletedProcess(
            argv,
            captured["returncode"],
            "Doing analysis at TEMP\n",
            "",
        )

    monkeypatch.setattr(spice_gate, "_NGSPICE_VERSION_CHECKED", True)
    monkeypatch.setattr(spice_gate.shutil, "which", lambda _: "/usr/bin/ngspice")
    monkeypatch.setattr(spice_gate.subprocess, "run", fake_run)

    passed, _ = spice_gate.grade_netlist(BASE_DECK, timeout_s=7, memory_limit_mb=256)

    assert passed
    assert captured["start_new_session"] is True
    assert "preexec_fn" not in captured
    assert captured["argv"][:4] == [sys.executable, "-I", "-S", "-c"]
    assert captured["env"]["HOME"] == str(captured["cwd"])
    assert captured["env"]["TMPDIR"] == str(captured["cwd"])
    assert captured["env"]["LC_ALL"] == "C"
    assert set(captured["env"]) == {"HOME", "LANG", "LC_ALL", "PATH", "TMPDIR"}

    captured["returncode"] = 1
    passed, reason = spice_gate.grade_netlist(BASE_DECK)
    assert not passed
    assert "exit=1" in reason


def test_direct_runner_rejects_unsafe_deck_before_subprocess(monkeypatch):
    monkeypatch.setattr(
        spice_gate.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("unsafe deck reached subprocess"),
    )
    deck = "V1 in 0 1\nR1 in 0 1k\n.control\nop\nshell true\n.endc\n.end\n"

    with pytest.raises(spice_gate.UnsafeNetlistError):
        spice_gate.run_ngspice_sandboxed(deck)


def test_threaded_sandbox_calls_are_serialized_for_shared_nobody_uid(monkeypatch):
    state = {"active": 0, "maximum": 0}
    state_lock = threading.Lock()

    def fake_run(argv, **kwargs):
        with state_lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        time.sleep(0.01)
        with state_lock:
            state["active"] -= 1
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(spice_gate, "_NGSPICE_VERSION_CHECKED", True)
    monkeypatch.setattr(spice_gate.shutil, "which", lambda _: "/usr/bin/ngspice")
    monkeypatch.setattr(spice_gate.subprocess, "run", fake_run)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: spice_gate.run_ngspice_sandboxed(BASE_DECK), range(16)))

    assert state["maximum"] == 1


def test_spice_env_fails_closed_before_subprocess(monkeypatch, tmp_path):
    monkeypatch.setattr(
        spice_gate.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("unsafe design reached subprocess"),
    )
    env = SpiceEnv(SpiceEnvConfig(scratch_root=str(tmp_path)))
    result = env.simulate(
        "V1 in 0 1\nR1 in 0 1k\n.control\nop\nsource /etc/passwd\n.endc\n.end\n",
        {},
    )

    assert not result.success
    assert result.exit_code == -3
    assert "unsafe netlist" in result.stderr


def test_active_harnesses_have_no_raw_ngspice_subprocess():
    harness_root = Path(__file__).parents[1] / "athma_train" / "harnesses"
    offenders = []
    for path in harness_root.glob("*.py"):
        if "subprocess.run(" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)

    assert offenders == []


def test_no_raw_ngspice_batch_subprocess_outside_canonical_runner():
    package_root = Path(__file__).parents[1]
    direct_batch = re.compile(
        r"subprocess\.run\(\s*\[\s*(?:['\"]ngspice['\"]|NGSPICE_BIN|"
        r"self\.cfg\.ngspice_bin)[^\]]*['\"]-b['\"]",
        re.DOTALL,
    )
    offenders = []
    for source_root in (package_root / "athma_train", package_root / "scripts"):
        for path in source_root.rglob("*.py"):
            if path.name == "spice_gate.py":
                continue
            if direct_batch.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(package_root)))

    assert offenders == []


def test_public_eval_never_executes_benchmark_supplied_python():
    script = Path(__file__).parents[1] / "scripts" / "eval_on_bench_v1.py"
    source = script.read_text(encoding="utf-8")

    assert "exec(" not in source
    assert "spec_check_python" not in source
    assert "PHYCHIP_L2L3_GRADER" not in source


def test_closed_loop_rejects_unsafe_generation_before_harness(monkeypatch):
    script = Path(__file__).parents[1] / "scripts" / "closed_loop_probe.py"
    module_spec = importlib.util.spec_from_file_location("closed_loop_probe_security", script)
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)

    class MustNotMeasure:
        def measure(self, *args, **kwargs):
            pytest.fail("unsafe generation reached harness.measure")

    unsafe = """```spice
V1 in 0 1
R1 in 0 1k
.control
op
shell true
.endc
.end
```"""
    monkeypatch.setitem(module.GENERATORS, "codex", lambda prompt: unsafe)
    monkeypatch.setattr(module, "load_harness", lambda circuit_id: MustNotMeasure())
    spec = {
        "circuit_id": "buffer",
        "difficulty": "test",
        "tier": 0,
        "spec_prompt": "Design a buffer",
        "target_measurements": {"gain_db": [0.0, 1.0]},
    }

    result = module.run_spec(spec, max_rounds=1, log=lambda message: None)

    assert result["solved_round"] is None
    assert result["history"][0]["measured"]["gain_db"] is None
