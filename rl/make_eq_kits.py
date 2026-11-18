"""Write per-family kits asking a frontier model for equation-carrying netlist templates."""
import collections
import json
import os
import sys

sys.path.insert(0, "./phy-chip/poc_experiments")
sys.path.insert(0, "./work")
import generate_rlvr_candidates_v7 as gen  # noqa: E402
from common import interface_contract  # noqa: E402

OUT = "./cli/kits_eq"
os.makedirs(OUT, exist_ok=True)
roots = collections.defaultdict(list)
for line in open("./work/hl_roots.jsonl"):
    r = json.loads(line)
    if r["stage"] == "ok":
        roots[r["circuit_id"]].append(r)

for fam, (ins, outs, keys) in gen.FAMILY_SPECS.items():
    seen = sorted(r["measured"][keys[0]] for r in roots[fam])
    lo, hi = seen[0], seen[-1]
    text = f"""# Equation-carrying template kit: {fam}

Write 6 DIFFERENT parameterised ngspice netlists for a genuine {fam} circuit. In each one the design
law is written into the netlist itself, so that changing ONE target parameter re-sizes the circuit and
the simulated figure follows the target. The simulator does the arithmetic; you supply the physics.

Ports: input(s) {', '.join(ins)}; output(s) {', '.join(outs)}.
Figure(s) the grader measures and the template must track: {', '.join(keys)}.
Verified circuits of this family measured {keys[0]} between {lo:.4g} and {hi:.4g}; cover at least that range.

Rules for a template
- Line 1: `* {fam}`.
- One `.param` line per tracked figure, named exactly `T_<figure>`, with a sensible default
  (here: {', '.join('`.param T_' + k + '=<default>`' for k in keys)}).
- Every component value that sets a tracked figure must be a brace expression of the target
  parameter(s) and of free design choices fixed by their own `.param` lines, for example
  `.param Rin=10k` then `Rf n1 out {{Rin*pow(10, T_gain_db/20)}}`. Use only functions ngspice-46
  supports in `.param` expressions (pow, sqrt, exp, ln, log10, abs, min, max, pi via `.param pi=3.14159265`).
- Account for non-idealities you introduce (finite op-amp gain, loading, device drops) so the simulated
  figure lands within about 10% (or 1 dB) of the target across the range, not only at the default.
- Vary the six templates: different topologies or sub-variants, not six copies with other defaults.
- No comments other than line 1, no `.meas/.print/.ac/.tran/.dc` of your own.
{interface_contract(ins, outs)}

Global rejections: `laplace`, `tf(`, an independent voltage source across the output, a behavioural or
controlled source that directly synthesises the measured figure, `.include` or file access. An ideal
op-amp macro (high-gain VCVS, top-level or in a `.subckt`) is fine when real feedback components set
the behaviour.

Two circuits the grader accepted for this family (fixed values; shown for the conventions only):
"""
    for e in roots[fam][:2]:
        text += f"\nmeasured: { {k: e['measured'][k] for k in keys} }\n```spice\n{e['netlist']}```\n"
    text += (f"\nOutput format: reply with the six templates, each as a line `### {fam}_T<NN>` (NN = 01..06) "
             "followed by one ```spice fenced netlist ending with `.control` / `op` / `.endc` / `.end`. Nothing else.\n")
    open(os.path.join(OUT, f"{fam}.md"), "w").write(text)
print("kits", len(gen.FAMILY_SPECS))
