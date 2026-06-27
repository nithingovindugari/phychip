#!/usr/bin/env python3
"""Closed-loop simulator-in-the-loop probe.

Tests the hypothesis from the RL negative result: one-shot generation cannot
tune to tight numeric FOM windows, but a model that SEES the measured FOMs and
revises should converge.

Controlled variable = the feedback loop ONLY.  Each turn is a fresh, stateless
`codex exec` (gpt-5.5).  The only state carried between turns is what THIS script
feeds back (previous netlist + measured FOMs vs target).  So:

  round 0  = pure one-shot (what RL / greedy decode does)
  round R  = one-shot + R rounds of ngspice feedback (the closed loop)

If round-R in-spec rate >> round-0, the L2/L3 wall is task-structure, not model
capability, and is crackable with simulator-in-the-loop sizing.

Generation: codex only (authorized SFT/teacher).  No GPU, no pod.
Measurement: the same per-circuit harnesses used to build the RLVR specs and to
compute the GRPO reward, so the numbers are identical to training-time grading.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HARNESSES_DIR = ROOT / "athma-train" / "athma_train" / "harnesses"
_HCACHE: dict[str, object] = {}


def load_harness(cid: str):
    if cid in _HCACHE:
        return _HCACHE[cid]
    path = HARNESSES_DIR / f"{cid}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"h_{cid.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _HCACHE[cid] = mod
    return mod


FENCE_RE = re.compile(r"```[ \t]*\w*[ \t]*\n(.*?)```", re.DOTALL)
_DEV_RE = re.compile(r"(?im)^\s*[vrcleibgfh]\w*\s", re.MULTILINE)  # spice device lines


def _looks_like_netlist(block: str) -> bool:
    b = block.lower()
    return (".end" in b or ".op" in b or ".tran" in b or ".ac" in b
            or ".dc" in b or ".control" in b) and bool(_DEV_RE.search(block))


def extract_netlist(text: str) -> str | None:
    """Pick the LAST fenced block that actually looks like a SPICE netlist.

    Guards against codex echoing the prompt (which mentions a 'spice' fence) by
    requiring the block to contain real device lines + an analysis/.end marker.
    """
    cands = [b.strip() for b in FENCE_RE.findall(text)]
    real = [b for b in cands if _looks_like_netlist(b)]
    if real:
        nl = real[-1].strip()
        return nl if ".end" in nl.lower() else nl + "\n.end"
    # fence-less fallback: codex sometimes returns the netlist unfenced.
    # Grab the contiguous tail ending at the last `.end` that has device lines.
    low = text.lower()
    if ".end" in low and _DEV_RE.search(text):
        end_idx = low.rfind(".end") + len(".end")
        head = text[:end_idx]
        lines = head.splitlines()
        # walk back to the first SPICE-ish line (comment/device/.param/.subckt)
        start = 0
        for i, ln in enumerate(lines):
            s = ln.strip()
            if s.startswith("*") or s.startswith(".") or _DEV_RE.match(ln):
                start = i
                break
        nl = "\n".join(lines[start:]).strip()
        if _looks_like_netlist(nl):
            return nl
    if cands:                       # last resort: any fenced block
        nl = cands[-1].strip()
        return nl if ".end" in nl.lower() else nl + "\n.end"
    return None


import tempfile, os


def codex(prompt: str, timeout_s: int = 300) -> str:
    """Stateless single-shot codex call.

    Uses --output-last-message to capture ONLY the agent's final message
    (no SessionStart/Stop hook noise, no prompt echo), which makes netlist
    extraction robust.
    """
    tf = tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False)
    tf.close()
    try:
        subprocess.run(
            ["codex", "exec", "--sandbox", "read-only",
             "--output-last-message", tf.name, prompt],
            capture_output=True, text=True, timeout=timeout_s,
        )
        with open(tf.name) as f:
            msg = f.read()
        return msg
    except subprocess.TimeoutExpired:
        return "__CODEX_TIMEOUT__"
    finally:
        try:
            os.unlink(tf.name)
        except OSError:
            pass


def claude_p(prompt: str, timeout_s: int = 300) -> str:
    """Stateless single-shot `claude -p` call (tools disabled => pure generator).

    Prompt is fed on stdin.  --allowedTools "" keeps it from going agentic in the
    repo, so it behaves like a plain completion expert, comparable to codex().
    """
    try:
        p = subprocess.run(
            ["claude", "-p", "--allowedTools", ""],
            input=prompt, capture_output=True, text=True, timeout=timeout_s,
        )
        return (p.stdout or "") + "\n" + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return "__CLAUDE_TIMEOUT__"


GENERATORS = {"codex": codex, "claude": claude_p}


def in_window(v, lo, hi) -> bool:
    if v is None:
        return False
    if lo == hi:                      # degenerate target (e.g. 0)
        scale = abs(lo) if lo else 1.0
        return abs(v - lo) <= 0.05 * scale
    return lo <= v <= hi


def pct_off(v, lo, hi) -> float:
    """Signed % distance outside the window (0 if inside)."""
    if v is None:
        return float("inf")
    if lo <= v <= hi:
        return 0.0
    center = (lo + hi) / 2.0 or 1.0
    edge = hi if v > hi else lo
    return 100.0 * (v - edge) / (abs(center) or 1.0)


def build_feedback(prev_netlist: str, measured: dict, targets: dict,
                   contract_txt: str = "") -> str:
    lines = []
    all_none = all(measured.get(k) is None for k in targets)
    for k, (lo, hi) in targets.items():
        v = measured.get(k)
        if in_window(v, lo, hi):
            lines.append(f"  - {k}: measured {v:.4g}  -> IN window [{lo:.4g}, {hi:.4g}]  OK")
        else:
            off = pct_off(v, lo, hi)
            vs = "None (not measurable)" if v is None else f"{v:.4g}"
            direction = ""
            if v is not None:
                direction = "TOO HIGH (decrease it)" if v > hi else "TOO LOW (increase it)"
            lines.append(
                f"  - {k}: measured {vs}  -> OUT of [{lo:.4g}, {hi:.4g}]  ({off:+.1f}% off)  {direction}"
            )
    none_hint = ""
    if all_none:
        none_hint = (
            "\n\nALL measurements came back unmeasurable (None). This almost always means the "
            "grader could not drive your circuit: the input-port voltage sources are not named "
            "exactly as the ports, the output node name is wrong, or you included your own "
            "analysis/.meas block. Re-read the GRADER INTERFACE CONTRACT and fix the interface "
            "FIRST."
        )
    return (
        "Your previous ngspice netlist measured the following (simulated by the SAME "
        "verifier that grades you):\n"
        + "\n".join(lines)
        + none_hint
        + "\n\nRevise the COMPONENT VALUES (and only values / behavioral gains; keep the "
        "topology and the exact named ports) so EVERY measurement lands inside its window. "
        "Reason briefly about which values to change and by how much, then output the "
        "corrected netlist wrapped in a Markdown code fence tagged spice, ending with .end. "
        "Put nothing after the closing fence."
        + contract_txt
        + "\n\nPrevious netlist:\n[BEGIN NETLIST]\n" + prev_netlist + "\n[END NETLIST]"
    )


def interface_contract(spec: dict) -> str:
    """The grader's I/O contract.  Harnesses inject their OWN stimulus by
    `alter`-ing a source whose name == the input-port name and read `v(<out>)`,
    so a free-form netlist that names its sources differently (VPLUS/VMINUS) or
    adds its own .ac/.meas measures as None.  This is the same contract the
    reference netlists follow; stating it is the environment spec, not a hint at
    the answer (the model still has to design + size the circuit)."""
    ins = spec.get("input_ports") or []
    outs = spec.get("output_ports") or []
    in_lines = "; ".join(f"`{p} {p} 0 DC 0 AC 1`" for p in ins) or "(none)"
    return (
        "\n\nGRADER INTERFACE CONTRACT (follow exactly or the circuit cannot be measured):\n"
        f"  - For each input port, include an independent voltage source whose NAME equals the "
        f"port name, driving a node of that name, with an AC magnitude: {in_lines}. The grader "
        "injects/overrides the stimulus by altering these sources, so a DC bias + `AC 1` is enough.\n"
        f"  - Drive nothing else; the output(s) MUST be node(s) named exactly: {', '.join(outs) or '(see prompt)'}.\n"
        "  - End the netlist with a MINIMAL self-test block so it simulates standalone: a "
        "`.control` / `op` / `.endc` block (or a bare `.op`) then `.end`. The grader STRIPS this "
        "and injects its own AC/DC/transient measurement testbench — so do NOT add any "
        "`.ac/.tran/.dc/.meas/.print/.plot` analysis of your own; just the `op`.\n"
        "  - Otherwise provide only: a title comment, the circuit devices, and any behavioral "
        "op-amp `.subckt` you need.\n"
        "  - Do NOT use an ideal dependent source that directly synthesizes a measured FOM."
    )


def run_spec(spec: dict, max_rounds: int, log, contract: bool = False,
             generator: str = "codex") -> dict:
    gen = GENERATORS[generator]
    cid = spec["circuit_id"]
    targets = {k: tuple(v) for k, v in spec["target_measurements"].items()}
    harness = load_harness(cid)
    contract_txt = interface_contract(spec) if contract else ""
    base_prompt = (
        spec["spec_prompt"]
        + contract_txt
        + "\n\nReason briefly about component values, then output the netlist wrapped in a "
        "Markdown code fence tagged spice, ending with .end. Put nothing after the closing fence."
    )
    history = []
    prompt = base_prompt
    solved_round = None
    for r in range(max_rounds):
        raw = gen(prompt)
        netlist = extract_netlist(raw)
        if not netlist:
            measured, ok_keys = {}, []
            n_in = 0
        else:
            try:
                measured = harness.measure(netlist, spec=targets, timeout_s=15)
            except Exception as e:  # noqa: BLE001
                measured = {"__error__": str(e)[:120]}
            ok_keys = [k for k, (lo, hi) in targets.items()
                       if in_window(measured.get(k), lo, hi)]
            n_in = len(ok_keys)
        n_tot = len(targets)
        all_in = n_in == n_tot and netlist is not None
        history.append({
            "round": r, "n_in": n_in, "n_tot": n_tot,
            "measured": {k: measured.get(k) for k in targets},
            "netlist": netlist,
        })
        log(f"    round {r}: {n_in}/{n_tot} FOMs in-window"
            + ("  <-- ONE-SHOT" if r == 0 else "")
            + ("  *** ALL IN SPEC ***" if all_in else ""))
        if all_in:
            solved_round = r
            break
        if netlist is None:
            log("      (no netlist parsed from codex output; retrying with same prompt)")
            continue
        prompt = build_feedback(netlist, measured, targets, contract_txt)
    return {
        "circuit_id": cid,
        "difficulty": spec["difficulty"],
        "tier": spec["tier"],
        "n_tot": len(targets),
        "round0_in": history[0]["n_in"],
        "best_in": max(h["n_in"] for h in history),
        "solved_round": solved_round,
        "solved_oneshot": history[0]["n_in"] == len(targets),
        "history": history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="final_artifacts/evals/closed_loop/selected_specs.jsonl")
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--out", default="final_artifacts/evals/closed_loop/results.json")
    ap.add_argument("--contract", action="store_true",
                    help="give codex the grader I/O contract (source/node naming) so free-form "
                         "netlists are measurable; the legitimate environment spec")
    ap.add_argument("--generator", choices=list(GENERATORS), default="codex",
                    help="which model acts as the closed-loop expert")
    args = ap.parse_args()

    specs = [json.loads(l) for l in open(args.specs)]
    print(f"closed-loop probe: {len(specs)} specs, max_rounds={args.max_rounds}, "
          f"contract={args.contract}, generator={args.generator}+local-ngspice\n")

    def log(m):
        print(m, flush=True)

    results = []
    for i, s in enumerate(specs):
        log(f"[{i+1}/{len(specs)}] {s['difficulty']} {s['circuit_id']} (tier{s['tier']})")
        results.append(run_spec(s, args.max_rounds, log, contract=args.contract,
                                generator=args.generator))
        log("")

    # ---- summary ----
    oneshot = sum(r["solved_oneshot"] for r in results)
    closed = sum(r["solved_round"] is not None for r in results)
    log("=" * 64)
    log(f"ONE-SHOT  fully in-spec: {oneshot}/{len(results)}")
    log(f"CLOSED-LOOP fully in-spec: {closed}/{len(results)}  (within {args.max_rounds} rounds)")
    log("per-spec (round0_in/tot -> best_in/tot, solved@round):")
    for r in results:
        sr = r["solved_round"]
        log(f"  {r['difficulty']:7} {r['circuit_id']:13} "
            f"{r['round0_in']}/{r['n_tot']} -> {r['best_in']}/{r['n_tot']}  "
            + (f"solved@{sr}" if sr is not None else "unsolved"))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "oneshot_solved": oneshot, "closedloop_solved": closed,
        "n": len(results), "max_rounds": args.max_rounds, "results": results,
    }, open(args.out, "w"), indent=2)
    log(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
