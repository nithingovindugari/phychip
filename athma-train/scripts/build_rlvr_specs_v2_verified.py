#!/usr/bin/env python3
"""Build a VERIFIED, achievable-by-construction RLVR spec pool (v2).

Motivation
----------
The v1 pool (`data/rlvr_specs_v1/*.jsonl`) is 100% `unverified:True` with NO
reference netlist. GRPO over it suffers rollout sparsity: targets may be
physically unreachable, so almost no rollout lands inside the FOM windows and
the reward signal is flat. This script fixes that by *measuring first, then
naming the target*: every emitted spec's window is provably hit by a concrete,
simulable reference netlist.

Method (per `circuit_id`)
-------------------------
1. Collect topology seeds for the family from
     - data/phychip_bench_v2/l2l3_templates.json   (circuit_type -> circuit_id)
     - eval_sets/bench_v1/bench_v1.jsonl           (id tier*-<cid>-l*)
     - eval_sets/phychip_bench_v2/bench_v2.jsonl   (id tier*-<cid>-l*)
   A seed is kept only if its harness `measure()` returns NON-DEGENERATE FOMs
   (same rule as wellposed_gate_bench_v2.py).
2. Sweep component values: scale every R/C/L magnitude and MOS W/L by random
   per-device multipliers over a sensible grid (~n_per_family variants).
   Netlists stay self-contained (the harness injects its own analysis deck and
   strips the seed's .control/.end, so we only perturb device values).
3. Re-measure each variant. Keep it ONLY if the harness returns finite,
   physically-plausible FOMs (well-posedness gate).
4. Build the spec from the MEASURED FOMs (not invented):
     target_measurements[metric] = [measured*(1-tol), measured*(1+tol)]
   Emit three difficulty tiers spanning the band:
     tier 1 / loose  : +/-30%   (difficulty="loose")
     tier 2 / medium : +/-15%   (difficulty="medium")
     tier 3 / tight  : +/- 5%   (difficulty="tight")
5. Decontaminate against ALL bench prompts/netlists: drop any spec with an
   exact prompt match, >=0.40 8-gram prompt overlap, or a verbatim bench
   reference netlist. (Reuses the ngram/overlap idea from validate_bench_v2.py.)

This is pure template-sweep + local ngspice simulation. No network, no teacher
model. ngspice-46 must be on PATH.

Usage
-----
  python athma-train/scripts/build_rlvr_specs_v2_verified.py \
    --n-per-family 60 --seed 0 \
    --out    data/rlvr_specs_v2_verified/specs.jsonl \
    --report data/rlvr_specs_v2_verified/report.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ATHMA = ROOT / "athma-train"
sys.path.insert(0, str(ATHMA))

from athma_train.spice_gate import grade_netlist, normalize_netlist  # noqa: E402

HARNESSES_DIR = ATHMA / "athma_train" / "harnesses"

TRAIN_FAMILIES = [
    "inv-amp", "noninv-amp", "diff-amp", "buffer", "integrator", "summing-amp",
    "ina", "current-mirror", "schmitt", "h-bridge", "tia", "cs-stage",
    "gain-trim", "level-shift", "diff-pair", "differentiator", "buck",
    "bandgap", "boost", "current-sense", "bridge-ina", "ldo", "sallen-key",
]

# l2l3_templates.json circuit_type string -> training circuit_id
CTYPE_TO_CID = {
    "Inverting amplifier": "inv-amp",
    "Non-inverting amplifier": "noninv-amp",
    "Difference amplifier": "diff-amp",
    "Voltage follower / unity-gain buffer": "buffer",
    "Op-amp integrator": "integrator",
    "Summing amplifier (3-input)": "summing-amp",
    "3-op-amp instrumentation amplifier": "ina",
    "MOSFET current mirror (2-transistor)": "current-mirror",
    "Schmitt trigger comparator with hysteresis": "schmitt",
    "H-bridge motor driver (4 power MOSFETs)": "h-bridge",
    "Photodiode transimpedance amplifier": "tia",
    "Common-source MOSFET amplifier with resistive load": "cs-stage",
    "Op-amp gain stage with offset trim": "gain-trim",
    "Voltage level shifter (1.8V to 3.3V)": "level-shift",
    "Differential pair with active load": "diff-pair",
    "Op-amp differentiator": "differentiator",
    "Buck converter (DC-DC step-down)": "buck",
    "Bandgap voltage reference": "bandgap",
    "Boost converter (DC-DC step-up)": "boost",
    "High-side current sense amplifier": "current-sense",
    "Wheatstone bridge with instrumentation amp readout": "bridge-ina",
    "LDO regulator": "ldo",
    "Sallen-Key low-pass filter (2nd-order active)": "sallen-key",
}

DIFFICULTIES = [
    ("loose", 0.30, 1),
    ("medium", 0.15, 2),
    ("tight", 0.05, 3),
]

# ---------------------------------------------------------------- harness ----
_HCACHE: dict[str, object] = {}


def load_harness(cid: str):
    """Import athma_train/harnesses/<cid>.py exactly as train_lora_grpo does."""
    if cid in _HCACHE:
        return _HCACHE[cid]
    path = HARNESSES_DIR / f"{cid}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"harness_{cid.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001
        print(f"  warn: failed to load harness {cid}: {e}", file=sys.stderr)
        return None
    _HCACHE[cid] = mod
    return mod


# ---------------------------------------------------- well-posedness gate ----
def degenerate(fom: dict | None) -> str | None:
    """Reason string if FOMs are degenerate/non-physical, else None.

    Mirrors wellposed_gate_bench_v2.py with extra finite/runaway checks.
    """
    if not fom:
        return "no FOMs"
    items = list(fom.items())
    # headline FOM (first key) must be present + finite
    h_name, h_val = items[0]
    if h_val is None:
        return f"headline {h_name} is None"
    # all values that are present must be finite
    for k, v in fom.items():
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return f"{k} not numeric ({v!r})"
        if not math.isfinite(fv):
            return f"{k} non-finite ({fv})"
        # runaway-current guard: any current-named FOM beyond 1 kA is unphysical
        if ("current" in k or k.endswith("_a")) and abs(fv) > 1e3:
            return f"{k}={fv:g} runaway current"
    # dB-gain floor: a named amplifier with every dB gain < -60 dB is dead
    gdb = {k: float(v) for k, v in fom.items()
           if "gain" in k and k.endswith("_db") and v is not None}
    if gdb and all(v < -60 for v in gdb.values()):
        return f"all dB gains <-60 ({gdb})"
    # linear-gain floor: |gain| < 1e-3 V/V == no transfer
    for k, v in fom.items():
        if "gain" in k and k.endswith("_v_per_v") and v is not None and abs(float(v)) < 1e-3:
            return f"{k}={float(v):g} (<1e-3 V/V, no transfer)"
    # efficiency must be a real percentage
    eff = fom.get("efficiency_pct")
    if eff is not None and not (1.0 <= float(eff) <= 100.0):
        return f"efficiency_pct={eff} non-physical"
    return None


def finite_metrics(fom: dict) -> dict:
    """Metrics that are finite + non-zero enough to build a +/- band around."""
    out = {}
    for k, v in fom.items():
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            continue
        # a band around 0 collapses to [0,0]; skip near-zero metrics
        if abs(fv) < 1e-12:
            continue
        out[k] = fv
    return out


# ------------------------------------------------------------ value sweep ----
# Component value token, e.g. trailing "1k", "10meg", "2.265u", "47u", "100u"
SI = {
    "t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "m": 1e-3, "u": 1e-6,
    "n": 1e-9, "p": 1e-12, "f": 1e-15, "mil": 25.4e-6,
}
# order matters: 'meg'/'mil' before 'm'
_SUFFIXES = ["meg", "mil", "t", "g", "k", "m", "u", "n", "p", "f"]


def parse_si(tok: str) -> float | None:
    tok = tok.strip().lower().rstrip("ohmfhsv")  # drop unit letters like 1kohm, 47uf
    m = re.match(r"^([-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?)([a-z]*)$", tok)
    if not m:
        return None
    base = float(m.group(1))
    suf = m.group(2)
    if not suf:
        return base
    for s in _SUFFIXES:
        if suf.startswith(s):
            return base * SI[s]
    return None


def fmt_si(val: float) -> str:
    """Format a value back into a compact SI-suffixed token."""
    if val == 0:
        return "0"
    a = abs(val)
    for suf, mul in (("meg", 1e6), ("k", 1e3), ("", 1.0), ("m", 1e-3),
                     ("u", 1e-6), ("n", 1e-9), ("p", 1e-12), ("f", 1e-15)):
        if a >= mul * 0.9999:
            v = val / mul
            return f"{v:.4g}{suf}"
    return f"{val:.4g}"


def make_gradeable(nl: str) -> str:
    """Return a self-contained, independently-simulable copy of a reference.

    The reward in train_lora_grpo.py runs the *raw* reference through
    grade_netlist (ngspice -b -n) and only awards harness FOM credit if it
    passes. Many bench/template references embed self-referential .measure
    expressions (e.g. AT='t_fall_cross') or a bare top-level .tran with no
    .print, which fail in batch mode even though the harness (which strips and
    injects its own deck) accepts them. This rewrites the reference so it:
      - keeps topology, .model, .subckt, .param-definitions, sources, loads
      - drops .control blocks, standalone .measure/.print/.plot/.four lines
      - re-emits the ORIGINAL analysis directives inside one .control..endc
        block (which always "runs simulations" in batch mode, no .print needed)
    Result: the reference is gradeable on its own AND still measures via the
    harness, so every emitted spec scores >=1.0 (gate pass) in the real reward.
    """
    nl = normalize_netlist(nl)
    analyses: list[str] = []
    for line in nl.splitlines():
        m = re.match(r"\.(tran|ac|dc|op|noise)\b(.*)", line.strip(), re.I)
        if m:
            analyses.append((m.group(1).lower() + " " + m.group(2).strip()).strip())
    ctl = re.search(r"\.control\b(.*?)\.endc\b", nl, re.DOTALL | re.I)
    if ctl:
        for line in ctl.group(1).splitlines():
            if re.match(r"(tran|ac|dc|op|noise)\b", line.strip(), re.I):
                analyses.append(line.strip())
    nl = re.sub(r"\.control\b.*?\.endc\b", "", nl, flags=re.DOTALL | re.I)
    body = []
    for line in nl.splitlines():
        s = line.strip().lower()
        if s.startswith((".meas", ".measure", ".print", ".plot", ".four",
                         ".probe", ".tran", ".ac", ".dc", ".op", ".noise")):
            continue
        if s.startswith(".end") and not s.startswith(".ends"):
            continue
        body.append(line)
    out = "\n".join(body).rstrip()
    if not analyses:
        analyses = ["op"]
    seen: set[str] = set()
    uniq = [a for a in analyses if not (a in seen or seen.add(a))]
    return (out + "\n.control\nset noaskquit\n" + "\n".join(uniq)
            + "\nquit\n.endc\n.end\n")


def sweep_netlist(netlist: str, rng: random.Random) -> str:
    """Return a value-perturbed copy of the seed netlist.

    Scales R/C/L device values and MOS W=/L= by independent random multipliers.
    Only device VALUES change; topology, ports, models, sources are preserved.
    """
    lines = netlist.splitlines()
    out = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("*") or s.startswith("."):
            out.append(line)
            continue
        head = s[0].upper()
        new = line
        # --- passive R/C/L: last bare value token on the device line ---
        if head in ("R", "C", "L"):
            toks = line.split()
            # device value is the 4th token (name n+ n- value [params])
            if len(toks) >= 4:
                val = parse_si(toks[3])
                if val is not None and val != 0:
                    mul = rng.uniform(0.4, 2.5)
                    toks[3] = fmt_si(val * mul)
                    # preserve original leading whitespace
                    lead = line[: len(line) - len(line.lstrip())]
                    new = lead + " ".join(toks)
        # --- MOS W= / L= params (any device line) ---
        if "w=" in line.lower() or "l=" in line.lower():
            def _scale(m, lo=0.5, hi=2.0):
                key = m.group(1)
                v = parse_si(m.group(2))
                if v is None or v == 0:
                    return m.group(0)
                return f"{key}={fmt_si(v * rng.uniform(lo, hi))}"
            new = re.sub(r"\b([WwLl])\s*=\s*([0-9.eE+\-]+[a-zA-Z]*)",
                         _scale, new)
        out.append(new)
    return "\n".join(out)


# ------------------------------------------------------------- seed loading --
def cid_from_bench_id(bid: str) -> str:
    m = re.match(r"tier\d+-(.+?)-l[123]", bid or "")
    return m.group(1) if m else ""


def collect_seeds() -> dict[str, list[dict]]:
    """circuit_id -> [ {src, ref} ] raw topology seeds (unmeasured)."""
    seeds: dict[str, list[dict]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)

    def add(cid, ref, src):
        if cid not in TRAIN_FAMILIES or not ref:
            return
        key = normalize_netlist(ref).strip()
        if key in seen[cid]:
            return
        seen[cid].add(key)
        seeds[cid].append({"src": src, "ref": ref})

    tmpl = json.loads((ROOT / "data/phychip_bench_v2/l2l3_templates.json").read_text())
    for x in tmpl:
        add(CTYPE_TO_CID.get(x.get("circuit_type", ""), ""),
            x.get("reference_netlist"), "l2l3_template")

    for bf in ("eval_sets/bench_v1/bench_v1.jsonl",
               "eval_sets/phychip_bench_v2/bench_v2.jsonl"):
        fp = ROOT / bf
        if not fp.exists():
            continue
        for ln in fp.read_text().splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            add(cid_from_bench_id(r.get("id", "")),
                r.get("reference_netlist"), Path(bf).name)
    return seeds


# ---------------------------------------------------------- worker (sweep) ----
def _measure_variant(args):
    """Worker: build + measure a single value-swept variant.

    Returns (ok, ref_netlist, fom_dict, reason).
    """
    cid, ref, vseed = args
    rng = random.Random(vseed)
    try:
        harness = load_harness(cid)
        if harness is None:
            return (False, None, None, "no harness")
        variant = make_gradeable(sweep_netlist(ref, rng))
        # 1. harness must return non-degenerate FOMs
        fom = harness.measure(variant, spec=None, timeout_s=15)
        why = degenerate(fom)
        if why:
            return (False, None, None, why)
        clean = {k: float(v) for k, v in finite_metrics(fom).items()}
        if not clean:
            return (False, None, None, "no finite metrics")
        # 2. reference must ALSO pass the reward's stricter ngspice gate, so the
        #    spec scores >=1.0 in train_lora_grpo.reward_fn
        passed, _ = grade_netlist(variant, timeout_s=15)
        if not passed:
            return (False, None, None, "grade_netlist fail")
        return (True, variant, clean, "ok")
    except Exception as e:  # noqa: BLE001
        return (False, None, None, f"err:{str(e)[:60]}")


# ---------------------------------------------------------------- decontam ----
NGRAM = 8
OVERLAP_BLOCK = 0.40


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _ngrams(s: str, n: int = NGRAM) -> set[str]:
    w = _norm(s).split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def _overlap(cand: set[str], corpus: set[str]) -> float:
    if not cand:
        return 0.0
    return len(cand & corpus) / len(cand)


def load_bench_decontam():
    """Return (bench 8-grams, bench prompt-norms, bench netlist-norms)."""
    grams: set[str] = set()
    prompts: set[str] = set()
    netlists: set[str] = set()
    for bf in sorted((ROOT / "eval_sets").rglob("bench_*.jsonl")):
        for ln in bf.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            p = r.get("prompt") or ""
            if p:
                grams |= _ngrams(p)
                prompts.add(_norm(p))
            nl = r.get("reference_netlist") or ""
            if nl:
                netlists.add(_norm(nl))
    print(f"  decontam corpus: {len(prompts)} bench prompts, "
          f"{len(grams)} 8-grams, {len(netlists)} netlists", file=sys.stderr)
    return grams, prompts, netlists


# ------------------------------------------------------------- prompt build --
PORTS_BY_CID = {}  # filled from templates/bench at runtime


def build_prompt(cid: str, metrics: dict, ports: dict, difficulty: str,
                 tol: float) -> str:
    inp = ", ".join(ports.get("input_ports", [])) or "Vin"
    out = ", ".join(ports.get("output_ports", [])) or "Vout"
    pct = int(round(tol * 100))
    lines = [
        f"Design a {cid} circuit that meets the following measured-FOM targets "
        f"(each target is a +/-{pct}% achievable window centered on a "
        f"simulator-verified value).",
        "",
        f"Input port(s): {inp}",
        f"Output port(s): {out}",
        "",
        "Target measurements:",
    ]
    for k, v in metrics.items():
        lo, hi = v
        lines.append(f"  - {k}: [{lo:.4g}, {hi:.4g}]")
    lines += [
        "",
        "Return a self-contained ngspice netlist (devices + a behavioral op-amp "
        "model where appropriate) inside a ```spice``` block ending in .end. "
        "Use the named ports exactly. Do not insert an ideal dependent source "
        "that directly synthesizes a measured FOM.",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------- main -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-family", type=int, default=60,
                    help="value-swept variants to attempt per family (per seed pool)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/rlvr_specs_v2_verified/specs.jsonl")
    ap.add_argument("--report", default="data/rlvr_specs_v2_verified/report.json")
    a = ap.parse_args()

    out_path = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    rep_path = (ROOT / a.report) if not Path(a.report).is_absolute() else Path(a.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(a.seed)

    # ports per family from templates (fallback to amp default)
    tmpl = json.loads((ROOT / "data/phychip_bench_v2/l2l3_templates.json").read_text())
    for x in tmpl:
        cid = CTYPE_TO_CID.get(x.get("circuit_type", ""), "")
        if cid and cid not in PORTS_BY_CID:
            PORTS_BY_CID[cid] = {"input_ports": x.get("input_ports", []),
                                 "output_ports": x.get("output_ports", [])}

    print("collecting topology seeds...", file=sys.stderr)
    seeds = collect_seeds()

    # 1. filter seeds: keep only those that measure non-degenerate
    good_seeds: dict[str, list[dict]] = {}
    missing_topology = []
    seed_report = {}
    for cid in TRAIN_FAMILIES:
        cand = seeds.get(cid, [])
        keep = []
        for s in cand:
            h = load_harness(cid)
            if h is None:
                continue
            graded_ref = make_gradeable(s["ref"])
            try:
                fom = h.measure(graded_ref, spec=None, timeout_s=15)
            except Exception:  # noqa: BLE001
                fom = None
            why = degenerate(fom)
            if why is not None:
                continue
            passed, _ = grade_netlist(graded_ref, timeout_s=15)
            if not passed:
                continue
            s["base_fom"] = finite_metrics(fom)
            keep.append(s)
        seed_report[cid] = {"candidates": len(cand), "wellposed_seeds": len(keep),
                            "seed_srcs": [s["src"] for s in keep]}
        if keep:
            good_seeds[cid] = keep
        else:
            missing_topology.append(cid)
        print(f"  {cid:16s} seeds {len(keep)}/{len(cand)} wellposed", file=sys.stderr)

    # 2. sweep + measure variants per family (parallel)
    tasks = []
    for cid, slist in good_seeds.items():
        # distribute n_per_family across available wellposed seeds
        n = a.n_per_family
        per_seed = max(1, math.ceil(n / len(slist)))
        for s in slist:
            for _ in range(per_seed):
                tasks.append((cid, s["ref"], rng.randrange(2**31)))
    print(f"sweeping {len(tasks)} variants across {len(good_seeds)} families "
          f"with {a.workers} workers...", file=sys.stderr)

    # collect: cid -> list of (variant_netlist, metrics)
    verified: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    sim_counts = Counter()
    kept_counts = Counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_measure_variant, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            cid = futs[fut][0]
            sim_counts[cid] += 1
            ok, variant, fom, _why = fut.result()
            if ok:
                verified[cid].append((variant, fom))
                kept_counts[cid] += 1
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{len(tasks)} simulated", file=sys.stderr)

    # 3. dedupe variants per family (by metric-tuple, to avoid near-identical specs)
    for cid in list(verified):
        seen = set()
        uniq = []
        for nl, fom in verified[cid]:
            sig = tuple(round(v, 6) for _, v in sorted(fom.items()))
            if sig in seen:
                continue
            seen.add(sig)
            uniq.append((nl, fom))
        verified[cid] = uniq

    # 4. build specs (3 difficulty tiers per verified variant) + decontam
    print("building decontam corpus...", file=sys.stderr)
    bench_grams, bench_prompts, bench_netlists = load_bench_decontam()

    specs = []
    decontam_drops = {"exact_prompt": 0, "ngram_overlap": 0, "verbatim_netlist": 0}
    for cid in TRAIN_FAMILIES:
        ports = PORTS_BY_CID.get(cid, {"input_ports": ["Vin"], "output_ports": ["Vout"]})
        for vi, (nl, fom) in enumerate(verified.get(cid, [])):
            nl_norm = _norm(nl)
            for diff, tol, tier in DIFFICULTIES:
                metrics = {k: [round(v * (1 - tol), 6), round(v * (1 + tol), 6)]
                           for k, v in fom.items()}
                # normalize windows so lo<=hi (negative-valued FOMs)
                for k in metrics:
                    lo, hi = metrics[k]
                    if lo > hi:
                        metrics[k] = [hi, lo]
                prompt = build_prompt(cid, metrics, ports, diff, tol)
                # --- decontam ---
                if _norm(prompt) in bench_prompts:
                    decontam_drops["exact_prompt"] += 1
                    continue
                if _overlap(_ngrams(prompt), bench_grams) >= OVERLAP_BLOCK:
                    decontam_drops["ngram_overlap"] += 1
                    continue
                if nl_norm in bench_netlists:
                    decontam_drops["verbatim_netlist"] += 1
                    continue
                specs.append({
                    "circuit_id": cid,
                    "tier": tier,
                    "difficulty": diff,
                    "spec_prompt": prompt,
                    "target_measurements": metrics,
                    "gen_measured": {k: round(v, 6) for k, v in fom.items()},
                    "reference_netlist": nl,
                    "input_ports": ports.get("input_ports", []),
                    "output_ports": ports.get("output_ports", []),
                    "verified": True,
                    "source_teacher": "template-sweep+ngspice-46",
                    "format_version": "rlvr_spec_v2_verified",
                })

    rng.shuffle(specs)
    with out_path.open("w") as f:
        for s in specs:
            f.write(json.dumps(s) + "\n")

    # ---- report ----
    per_family = Counter(s["circuit_id"] for s in specs)
    diff_hist = Counter(s["difficulty"] for s in specs)
    tier_hist = Counter(s["tier"] for s in specs)
    yield_tbl = {}
    for cid in TRAIN_FAMILIES:
        sim = sim_counts.get(cid, 0)
        kept = kept_counts.get(cid, 0)
        yield_tbl[cid] = {
            "wellposed_seeds": seed_report[cid]["wellposed_seeds"],
            "seed_srcs": seed_report[cid]["seed_srcs"],
            "variants_simulated": sim,
            "variants_kept": kept,
            "yield": round(kept / sim, 3) if sim else 0.0,
            "unique_variants": len(verified.get(cid, [])),
            "specs_emitted": per_family.get(cid, 0),
        }
    report = {
        "total_specs": len(specs),
        "families_with_specs": len([c for c in per_family if per_family[c]]),
        "missing_topology": missing_topology,
        "difficulty_histogram": dict(diff_hist),
        "tier_histogram": {str(k): v for k, v in sorted(tier_hist.items())},
        "decontam_drops": decontam_drops,
        "per_family": yield_tbl,
        "params": {"n_per_family": a.n_per_family, "seed": a.seed},
        "out": str(out_path), "report": str(rep_path),
    }
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report, indent=2))

    print(f"\nWROTE {len(specs)} verified specs -> {out_path}")
    print(f"  families with specs: {report['families_with_specs']}/23")
    print(f"  difficulty: {dict(diff_hist)}")
    print(f"  tier:       {report['tier_histogram']}")
    print(f"  decontam drops: {decontam_drops}")
    if missing_topology:
        print(f"  MISSING TOPOLOGY: {missing_topology}")
    print(f"  report -> {rep_path}")


if __name__ == "__main__":
    main()
