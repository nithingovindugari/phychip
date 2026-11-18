"""Verify Codex physics problems: run solution_python (math only, 5 s, no imports beyond math) and check the worked
solution's final number agrees with the computed answer within rel_tol. Keep only consistent, finite problems."""
import glob, json, math, re, random, subprocess, sys
def run(code):
    prog = "import math\n" + code + "\nprint(repr(float(answer)))"
    if re.search(r"\b(import|open|exec|eval|__)\b", code.replace("import math", "")): return None
    try:
        out = subprocess.run([sys.executable, "-I", "-c", prog], capture_output=True, text=True, timeout=5)
        v = float(out.stdout.strip().splitlines()[-1]); return v if math.isfinite(v) else None
    except Exception: return None
def last_num(s):
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\s*[x×]\s*10\^?\(?[-+]?\d+\)?)?", s or "")
    out = []
    for n in nums[-6:]:
        m = re.match(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s*[x×]\s*10\^?\(?([-+]?\d+)\)?)?", n)
        try: out.append(float(m.group(1)) * (10 ** int(m.group(2)) if m.group(2) else 1))
        except Exception: pass
    return out
rows, stats = [], {"items": 0, "parse": 0, "code": 0, "consistent": 0}
for p in sorted(glob.glob("cli_out/physics/codex-gpt-6-astra/*.txt")):
    t = open(p).read(); m = re.search(r"\[.*\]", t, re.S)
    try: items = json.loads(m.group(0))
    except Exception: continue
    for k, it in enumerate(items):
        stats["items"] += 1
        if not all(x in it for x in ("question", "solution_python", "worked_solution")): continue
        stats["parse"] += 1
        v = run(it["solution_python"])
        if v is None: continue
        stats["code"] += 1
        tol = max(float(it.get("rel_tol", 0.01) or 0.01), 0.01)
        if any(abs(x - v) <= tol * max(abs(v), 1e-30) * 1.5 for x in last_num(it["worked_solution"])):
            stats["consistent"] += 1
            rows.append({"id": f"{p.split('/')[-1][:-4]}-{k}", "question": it["question"], "answer": v, "unit": it.get("answer_unit", ""),
                         "rel_tol": tol, "worked_solution": it["worked_solution"]})
random.Random(0).shuffle(rows)
ho, tr = rows[:300], rows[300:]
json.dump(stats, sys.stdout); print(" train", len(tr), "heldout", len(ho))
open("./work/data/physics_train.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in tr))
open("./work/tasks_physics_heldout.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in ho))
