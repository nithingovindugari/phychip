"""On-policy post-training for single-turn SPICE generation with the ngspice verifier in the loop.

Rollouts come from a colocated vLLM server that hot-loads the current LoRA adapter each step
(the trainer and the sampler share one GPU). Scoring is scorer v11 behind sanitize-v12.

Dynamic sampling (DAPO): each step over-samples prompts and keeps only groups whose K rewards
differ, so every optimizer step is spent on prompts that carry signal.

Policy-gradient objectives (--algo), with --iters updates per rollout batch and overlong filtering:
  drgrpo : A = r - mean(group). No std scaling, no per-sequence length normalisation. PPO clip 0.2/0.2.
  dapo   : A = (r - mean) / std. Token-level loss, asymmetric "clip-higher" range 0.2/0.28.
  gspo   : A = (r - mean) / std. One ratio per sequence, s = exp(mean_t(dlogp)), clip (1-3e-4, 1+4e-4).

Reward shapes (--reward):
  sparse  : 1 if every requested spec is met, else 0
  shaped  : 0.70 * spec_score + 0.60 * pass            (repository contract: partial credit capped near the window)
  logdist : 0.70 * mean_k 1/(1+e_k) + 0.60 * pass      (e_k = distance to the window in window widths; long tail, never flat)
  margin  : pass * (1 + 0.3 * centring)                (only successes are paid; centred designs are paid more)
  ladder  : 0.1 netlist + 0.2 simulates + 0.2 admitted and measured + 0.5 * spec_score, 1.3 on pass (pays for validity)
  random  : shaped rewards shuffled across the batch   (control)

Extras
  --her-coef  : hindsight relabelling inside RL. A valid circuit that misses its target is still a correct
                answer to the request "meet windows around what it measured"; add that pair as a supervised term.
  --teacher   : on-policy distillation. Per-token KL(student || teacher) on the student's own samples.
                --pg-coef 0 --kl-coef 1 is pure OPD; both non-zero is teacher-anchored RL.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request

NEED_MEASURE = False


def _score_physics(targets, text):
    """Physics RLVR task: reward 1 if the last 'Answer: <number>' is within max(rel_tol, 2%) of the verified answer."""
    found = re.findall(r"Answer:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text or "")
    v = float(found[-1]) if found else None
    a, tol = float(targets["answer"]), max(float(targets.get("rel_tol", 0.01)), 0.02)
    ok = v is not None and abs(v - a) <= tol * max(abs(a), 1e-30)
    return {"all_specs_met": ok, "reward": 1.0 if ok else 0.0, "spec_score": 1.0 if ok else 0.0, "simulated": v is not None,
            "anticheat_blocked": False, "physics": True}


def _score(job):
    from phychip_v12 import score_v12
    family, targets, text = job
    if family == "physics":
        return _score_physics(targets, re.sub(r"<think>.*?</think>", "", text or "", flags=re.S))
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    try:
        event = score_v12(family, targets, text, timeout_s=25).to_event()
    except Exception as error:  # noqa: BLE001
        return {"all_specs_met": False, "reward": 0.0, "spec_score": 0.0, "simulated": False,
                "anticheat_blocked": False, "error": repr(error)[:120]}
    if NEED_MEASURE and event.get("simulated") and not event.get("anticheat_blocked"):
        try:
            from phy_chip_env import get_harness
            from phychip_v12 import sanitize_netlist
            match = re.search(r"```spice\s*(.*?)```", text, re.S | re.I)
            raw = get_harness(family).measure(sanitize_netlist(match.group(1), family), spec=targets, timeout_s=25) or {}
            measured = {}
            for key, value in raw.items():
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    measured[key] = value
            event["measured"] = measured
        except Exception:  # noqa: BLE001
            event["measured"] = {}
    return event


def window_error(key, value, lo, hi):
    """Distance from the window in units of the window width (0 inside)."""
    if lo <= value <= hi:
        return 0.0
    width = max(hi - lo, 1e-12)
    return (lo - value if value < lo else value - hi) / width


def shape_reward(mode, event, targets):
    passed = bool(event.get("all_specs_met"))
    if mode == "sparse":
        return 1.0 if passed else 0.0
    if mode in ("shaped", "random"):
        return float(event.get("reward", 0.0))
    if mode == "ladder":
        if passed:
            return 1.3
        r = 0.1 if event.get("has_spice_block") and event.get("has_end") else 0.0
        if event.get("simulated"):
            r += 0.2
            if not event.get("anticheat_blocked") and event.get("measured_specs", 0) > 0:
                r += 0.2 + 0.5 * float(event.get("spec_score", 0.0))
        return r
    measured = event.get("measured") or {}
    if mode == "logdist":
        credit = 0.0
        for key, (lo, hi) in targets.items():
            if key in measured:
                credit += 1.0 / (1.0 + window_error(key, measured[key], min(lo, hi), max(lo, hi)))
        return 0.70 * credit / max(len(targets), 1) + (0.60 if passed else 0.0)
    if mode == "margin":
        if not passed:
            return 0.0
        centring = []
        for key, (lo, hi) in targets.items():
            lo, hi = min(lo, hi), max(lo, hi)
            if key in measured and hi > lo:
                centring.append(1.0 - 2.0 * abs(measured[key] - (lo + hi) / 2) / (hi - lo))
        return 1.0 + 0.3 * (sum(centring) / len(centring) if centring else 0.0)
    raise ValueError(mode)


def http(url, payload=None, timeout=1800):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


class Sampler:
    """vLLM OpenAI server with runtime LoRA loading."""

    def __init__(self, base, port, gpu_frac, log_path, max_rank):
        self.url = f"http://127.0.0.1:{port}"
        env = dict(os.environ, VLLM_ALLOW_RUNTIME_LORA_UPDATING="True")
        cmd = ["/w/vllm-env/bin/vllm", "serve", base, "--port", str(port), "--dtype", "bfloat16",
               "--max-model-len", "4096", "--gpu-memory-utilization", str(gpu_frac), "--enable-lora",
               "--max-lora-rank", str(max_rank), "--max-loras", "1", "--served-model-name", "base"]
        self.proc = subprocess.Popen(cmd, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
        for _ in range(240):
            try:
                http(self.url + "/v1/models", timeout=5)
                return
            except Exception:  # noqa: BLE001
                if self.proc.poll() is not None:
                    raise RuntimeError("vLLM server exited; see " + log_path)
                time.sleep(5)
        raise RuntimeError("vLLM server did not come up")

    def load(self, name, path):
        http(self.url + "/v1/load_lora_adapter", {"lora_name": name, "lora_path": path})

    def unload(self, name):
        try:
            http(self.url + "/v1/unload_lora_adapter", {"lora_name": name})
        except Exception:  # noqa: BLE001
            pass

    def sample(self, name, prompts, n, temperature, max_tokens, seed):
        out = http(self.url + "/v1/completions", {"model": name, "prompt": prompts, "n": n, "temperature": temperature,
                                                    "top_p": 1.0, "max_tokens": max_tokens, "seed": seed})
        texts = [None] * (len(prompts) * n)
        done = [False] * (len(prompts) * n)
        for choice in out["choices"]:
            texts[choice["index"]] = choice["text"]
            done[choice["index"]] = choice["finish_reason"] == "stop"
        return texts, done

    def close(self):
        self.proc.terminate()


def main():
    global NEED_MEASURE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--algo", choices=["drgrpo", "dapo", "gspo"], default="drgrpo")
    ap.add_argument("--reward", choices=["sparse", "shaped", "logdist", "margin", "ladder", "random"], default="sparse")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prompts", type=int, default=16, help="live groups wanted per step")
    ap.add_argument("--oversample", type=float, default=1.5, help="prompts sampled per step = prompts * oversample")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--mb", type=int, default=8, help="max sequences per backward pass")
    ap.add_argument("--tok-budget", type=int, default=3200, help="max padded tokens per backward pass (activation memory is ~4 MB per token)")
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu-frac", type=float, default=0.30)
    ap.add_argument("--eval-tasks", default=None)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--pg-coef", type=float, default=1.0)
    ap.add_argument("--her-coef", type=float, default=0.0)
    ap.add_argument("--her-max", type=int, default=32)
    ap.add_argument("--teacher", default=None, help="merged teacher model dir (same tokenizer as the student)")
    ap.add_argument("--kl-coef", type=float, default=0.0)
    ap.add_argument("--adv", choices=["mean", "passk"], default="mean",
                    help="passk: analytic pass@k group advantage (binary reward), rewards exploration")
    ap.add_argument("--passk-k", type=int, default=4)
    ap.add_argument("--retire-after", type=int, default=0,
                    help=">0: stop sampling a task after it was solved K/K this many times in a row (keeps groups live)")
    ap.add_argument("--novelty-bonus", type=float, default=0.0,
                    help=">0: extra reward for a PASSING answer whose device-type signature is absent from --novelty-ref "
                         "(decays as bonus/sqrt(1+times seen in training)); pushes RL toward new structures")
    ap.add_argument("--novelty-ref", default="", help="jsonl of SFT rows (messages) defining known signatures per family")
    ap.add_argument("--anneal", default="",
                    help="window-annealing curriculum, e.g. '1.0,0.5,0.25': a task solved K/K moves to a narrower spec window "
                         "(same centre), an all-fail group moves back; keeps groups live instead of retiring tasks")
    ap.add_argument("--turns", type=int, default=1, help=">1: multi-turn RL with simulator feedback between attempts")
    ap.add_argument("--turn-penalty", type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    NEED_MEASURE = args.reward in ("logdist", "margin") or args.her_coef > 0 or args.turns > 1
    pool = mp.get_context("fork").Pool(args.procs)  # forked before CUDA is touched (inherits NEED_MEASURE)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, os.path.join(os.environ.get("PHYCHIP_REPO", "/w/phy-chip"), "poc_experiments"))
    from common import messages as make_messages, spec_prompt
    from feedback import feedback_text, repair_messages

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    eot = eot if isinstance(eot, int) and eot >= 0 and eot != tok.unk_token_id else tok.eos_token_id
    rank = json.load(open(os.path.join(args.adapter, "adapter_config.json")))["r"]
    sampler = Sampler(args.base, args.port, args.gpu_frac, os.path.join(args.out, "vllm.log"), rank)
    from transformers import AutoConfig, AutoModelForImageTextToText
    is_vl = getattr(AutoConfig.from_pretrained(args.base), "vision_config", None) is not None  # Qwen3.5 ships as a VL checkpoint
    loader = AutoModelForImageTextToText if is_vl else AutoModelForCausalLM
    model = loader.from_pretrained(args.base, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    teacher = None
    if args.teacher and args.kl_coef > 0:
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16, device_map={"": 0}).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    def render(msgs):
        try:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def prep(path):
        rows = [json.loads(line) for line in open(path)]
        for t in rows:
            t["prompt_text"] = render(t["messages"][:2])
            t["prompt_ids"] = tok(t["prompt_text"], add_special_tokens=False)["input_ids"]
        return rows

    family_ports = None
    if args.her_coef > 0:
        import generate_rlvr_candidates_v7 as gen
        family_ports = gen.FAMILY_SPECS

    def hindsight_prompt(family, measured):
        inputs, outputs, keys = family_ports[family]
        if any(k not in measured for k in keys):
            return None
        primary = {k: measured[k] for k in keys}
        ok, _ = gen.plausible(family, primary)
        if not ok:
            return None
        if family == "ldo" and abs(measured.get("line_reg_v_per_v", 0.0)) >= 0.5:
            return None
        if family == "h-bridge" and measured.get("vmotor_pp_v", 0.0) < 1.0:
            return None
        targets = {k: gen.interval(k, v, 0.30) for k, v in primary.items()}
        text = render(make_messages(spec_prompt(family, inputs, outputs, targets), inputs, outputs, "grpo"))
        return tok(text, add_special_tokens=False)["input_ids"]

    def dev_sig(text):
        m = re.search(r"```spice\s*(.*?)```", text or "", re.S | re.I)
        net = m.group(1) if m else (text or "")
        c = {}
        for ln in net.splitlines():
            s0 = ln.strip()
            if not s0 or s0[0] in "*.+":
                continue
            t0 = s0[0].upper()
            if t0 in "VI":
                continue
            if t0 in "MQ":
                t0 += "p" if re.search(r"\b(pmos|pnp|p\w*mod|pch)\b", s0.lower()) else "n"
            c[t0] = c.get(t0, 0) + 1
        return tuple(sorted(c.items()))

    known_sig, novel_seen = {}, {}
    if args.novelty_bonus > 0 and args.novelty_ref:
        for line in open(args.novelty_ref):
            r = json.loads(line)
            m = r["messages"] if isinstance(r["messages"], list) else eval(r["messages"])  # noqa: S307 - our own data files
            known_sig.setdefault(r["circuit_id"], set()).add(dev_sig(m[-1]["content"]))
    anneal = [float(x) for x in args.anneal.split(",")] if args.anneal else [1.0]
    task_level, hard_cache = {}, {}

    def at_level(t, level):
        """Task t with every spec window shrunk to anneal[level] of its width around the same centre."""
        if level == 0:
            return t
        key = (t["id"], level)
        if key not in hard_cache:
            f = anneal[level]
            targets = {k: [(lo + hi) / 2 - (hi - lo) / 2 * f, (lo + hi) / 2 + (hi - lo) / 2 * f]
                       for k, (lo, hi) in t["target_measurements"].items()}
            msgs = [dict(m) for m in t["messages"][:2]]
            for k, (lo, hi) in targets.items():  # rewrite the numbers in place; the prompt format stays identical
                msgs[1]["content"] = re.sub(rf"{re.escape(k)} in \[[^\]]*\]", f"{k} in [{lo:.8g}, {hi:.8g}]", msgs[1]["content"])
            text = render(msgs)
            hard_cache[key] = dict(t, id=f"{t['id']}@L{level}", base_id=t["id"], target_measurements=targets,
                                   messages=msgs, prompt_text=text, prompt_ids=tok(text, add_special_tokens=False)["input_ids"])
        return hard_cache[key]

    tasks = prep(args.tasks)
    eval_tasks = prep(args.eval_tasks) if args.eval_tasks else []
    log = open(os.path.join(args.out, "metrics.jsonl"), "a")
    evlog = open(os.path.join(args.out, "eval.jsonl"), "a")
    rollouts = open(os.path.join(args.out, "rollouts.jsonl"), "a")
    live_dir = os.path.join(args.out, "live_adapter")

    def publish(name):
        shutil.rmtree(live_dir, ignore_errors=True)
        model.save_pretrained(live_dir)
        sampler.load(name, live_dir)

    def evaluate(step, name):
        if not eval_tasks:
            return
        prompts = [t["prompt_text"] for t in eval_tasks]
        g_text, _ = sampler.sample(name, prompts, 1, 0.0, args.max_new, 0)
        k_text, _ = sampler.sample(name, prompts, 8, 0.8, args.max_new, 0)
        g = score_many([(t["circuit_id"], t["target_measurements"], x) for t, x in zip(eval_tasks, g_text)])
        k = score_many([(eval_tasks[i // 8]["circuit_id"], eval_tasks[i // 8]["target_measurements"], x) for i, x in enumerate(k_text)])
        rec = {"step": step, "greedy_pass": sum(s["all_specs_met"] for s in g), "tasks": len(eval_tasks),
               "k8_pass_rate": sum(s["all_specs_met"] for s in k) / len(k),
               "k8_any": sum(any(s["all_specs_met"] for s in k[i * 8:(i + 1) * 8]) for i in range(len(eval_tasks))),
               "k8_blocked": sum(bool(s.get("anticheat_blocked")) for s in k) / len(k),
               "greedy_ids": [t["id"] for t, s in zip(eval_tasks, g) if s["all_specs_met"]]}
        evlog.write(json.dumps(rec) + "\n"); evlog.flush()
        print("EVAL", json.dumps({k_: v for k_, v in rec.items() if k_ != "greedy_ids"}), flush=True)

    def forward_sel(net, chunk):
        """Logits at completion positions for a list of (prompt_ids, completion_ids)."""
        w = max(len(p) + len(c) for p, c in chunk)
        x = torch.full((len(chunk), w), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(chunk), w), dtype=torch.long)
        tm = torch.zeros((len(chunk), w), dtype=torch.bool)
        for r, (p, c) in enumerate(chunk):
            n = len(p) + len(c)
            x[r, :n] = torch.tensor(p + c)
            am[r, :n] = 1
            tm[r, len(p):n] = True
        x, am, tm = x.to(0), am.to(0), tm.to(0)
        logits = net(input_ids=x, attention_mask=am).logits[:, :-1]
        sel = tm[:, 1:]
        return logits[sel].float(), x[:, 1:][sel]

    def chunks(seqs, length):
        """Greedy length-sorted batches under a padded-token budget."""
        seqs = sorted(seqs, key=length)
        batch, out = [], []
        for item in seqs:
            widest = max([length(x) for x in batch] + [length(item)])
            if batch and (len(batch) >= args.mb or widest * (len(batch) + 1) > args.tok_budget):
                out.append(batch)
                batch = []
            batch.append(item)
        if batch:
            out.append(batch)
        return out

    score_cache = {}

    def score_many(jobs):
        """Score (family, targets, text) jobs; identical answers to the same task are simulated once."""
        keys = [(f, json.dumps(t, sort_keys=True), re.sub(r"<think>.*?</think>", "", x or "", flags=re.S).strip()) for f, t, x in jobs]
        todo = list({k: j for k, j in zip(keys, jobs) if k not in score_cache}.items())
        for (k, _), ev in zip(todo, pool.map(_score, [j for _, j in todo], chunksize=2)):
            score_cache[k] = ev
        if len(score_cache) > 200000:
            score_cache.clear()
        return [dict(score_cache[k]) if k in score_cache else _score(j) for k, j in zip(keys, jobs)]

    solved_streak = {}
    name = "pol0"
    publish(name)
    evaluate(0, name)
    for step in range(1, args.steps + 1):
        t0 = time.time()
        pool_now = [t for t in tasks if not args.retire_after or solved_streak.get(t["id"], 0) < args.retire_after] or tasks
        cand = rng.sample(pool_now, min(len(pool_now), int(round(args.prompts * args.oversample))))
        cand = [at_level(t, task_level.get(t["id"], 0)) for t in cand]
        texts, done = sampler.sample(name, [t["prompt_text"] for t in cand], args.k, 1.0, args.max_new, args.seed * 100000 + step)
        t_gen = time.time() - t0
        scores = score_many([(cand[i // args.k]["circuit_id"], cand[i // args.k]["target_measurements"], texts[i])
                             for i in range(len(texts))])
        # trajectory per sample: list of turns (prompt_ids, text, done, event)
        traj = [[(cand[i // args.k]["prompt_ids"], texts[i], done[i], scores[i])] for i in range(len(texts))]
        first_pass = [bool(s["all_specs_met"]) for s in scores]
        for turn in range(2, args.turns + 1):
            active = [i for i in range(len(traj)) if not traj[i][-1][3]["all_specs_met"] and traj[i][-1][2]]
            if not active:
                break
            hist_texts = []
            for i in active:
                t = cand[i // args.k]
                history = [(x[1], feedback_text(x[3], t["target_measurements"])) for x in traj[i]]
                hist_texts.append(render(repair_messages(t["messages"][:2], history)))
            r_texts, r_done = sampler.sample(name, hist_texts, 1, 1.0, args.max_new, args.seed * 100000 + step * 10 + turn)
            r_scores = score_many([(cand[i // args.k]["circuit_id"], cand[i // args.k]["target_measurements"], x)
                                   for i, x in zip(active, r_texts)])
            for i, ht, x, d, ev in zip(active, hist_texts, r_texts, r_done, r_scores):
                traj[i].append((tok(ht, add_special_tokens=False)["input_ids"], x, d, ev))
        scores = [tr[-1][3] for tr in traj]
        texts = [tr[-1][1] for tr in traj]
        t_score = time.time() - t0 - t_gen
        rewards = [shape_reward(args.reward, s, cand[i // args.k]["target_measurements"]) for i, s in enumerate(scores)]
        if args.turns > 1:
            rewards = [r - args.turn_penalty * (len(traj[i]) - 1) * (1.0 if scores[i]["all_specs_met"] else 0.0)
                       for i, r in enumerate(rewards)]
        n_novel = 0
        if args.novelty_bonus > 0:
            for i, sc in enumerate(scores):
                fam = cand[i // args.k]["circuit_id"]
                if sc.get("all_specs_met") and fam != "physics":
                    k = (fam, dev_sig(texts[i]))
                    if k[1] not in known_sig.get(fam, set()):
                        rewards[i] += args.novelty_bonus / math.sqrt(1 + novel_seen.get(k, 0))
                        novel_seen[k] = novel_seen.get(k, 0) + 1
                        n_novel += 1
        if args.reward == "random":
            rng.shuffle(rewards)
        for g, t in enumerate(cand):
            all_pass = all(scores[i]["all_specs_met"] for i in range(g * args.k, (g + 1) * args.k))
            all_fail = not any(scores[i]["all_specs_met"] for i in range(g * args.k, (g + 1) * args.k))
            base = t.get("base_id", t["id"])
            if len(anneal) > 1:
                lvl = task_level.get(base, 0)
                task_level[base] = min(lvl + 1, len(anneal) - 1) if all_pass else max(lvl - 1, 0) if all_fail else lvl
            solved_streak[base] = solved_streak.get(base, 0) + 1 if all_pass else 0
        turn_items = [[(p, tok(x, add_special_tokens=False)["input_ids"] + [eot]) for (p, x, d, _) in tr if d] for tr in traj]
        comp_ids = [tr[-1] and (tok(tr[-1][1], add_special_tokens=False)["input_ids"] + [eot] if tr[-1][2] else None) for tr in traj]
        # ---- dynamic sampling: keep groups with reward variance, up to the target count
        items, adv_all, live = [], [0.0] * len(texts), 0
        for g in range(len(cand)):
            sl = slice(g * args.k, (g + 1) * args.k)
            rs = rewards[sl]
            mean = sum(rs) / len(rs)
            std = (sum((r - mean) ** 2 for r in rs) / len(rs)) ** 0.5
            is_live = max(rs) - min(rs) > 1e-9
            live += is_live
            use_pg = args.pg_coef > 0 and is_live and live <= args.prompts
            use_kl = teacher is not None and g < args.prompts
            if not (use_pg or use_kl):
                continue
            if args.adv == "passk":
                n_all, n_pos = len(rs), sum(1 for r in rs if r > 0.5)
                kk = min(args.passk_k, n_all)
                p_k = 1.0 - math.comb(n_all - n_pos, kk) / math.comb(n_all, kk)
                sig = max(math.sqrt(p_k * (1.0 - p_k)), 1e-6)
                a_pos = (1.0 - p_k) / sig
                a_neg = ((1.0 - math.comb(n_all - n_pos - 1, kk - 1) / math.comb(n_all - 1, kk - 1)) - p_k) / sig if n_all - n_pos >= 1 else 0.0
            for i in range(sl.start, sl.stop):
                a = 0.0
                if use_pg:
                    if args.adv == "passk":
                        a = a_pos if rs[i - sl.start] > 0.5 else a_neg
                    else:
                        a = rs[i - sl.start] - mean
                        if args.algo in ("dapo", "gspo"):
                            a /= std + 1e-6
                    adv_all[i] = a
                for p_ids, c_ids in turn_items[i]:  # overlong filtering: unfinished turns carry no loss
                    items.append({"p": p_ids, "c": c_ids, "a": a, "pg": use_pg, "kl": use_kl})
        # ---- hindsight relabelling of valid misses
        her = []
        if args.her_coef > 0:
            order = list(range(len(texts)))
            rng.shuffle(order)
            for i in order:
                s = scores[i]
                if len(her) >= args.her_max:
                    break
                if comp_ids[i] is None or s.get("all_specs_met") or not s.get("measured"):
                    continue
                prompt = hindsight_prompt(cand[i // args.k]["circuit_id"], s["measured"])
                if prompt is not None:
                    her.append((prompt, comp_ids[i]))
        # ---- updates
        stats = {"loss": 0.0, "grad_norm": 0.0, "clip_frac": 0.0, "approx_kl": 0.0, "teacher_kl": 0.0, "her_nll": 0.0}
        if items or her:
            model.train()
            items.sort(key=lambda it: len(it["p"]) + len(it["c"]))
            pg_items = [it for it in items if it["pg"]]
            n_tok = max(sum(len(it["c"]) for it in pg_items), 1)
            n_kl_tok = max(sum(len(it["c"]) for it in items if it["kl"]), 1)
            n_her_tok = max(sum(len(c) for _, c in her), 1)
            for it_idx in range(args.iters):
                opt.zero_grad(set_to_none=True)
                clipped, kl_sum, loss_sum, tkl_sum, her_sum = 0, 0.0, 0.0, 0.0, 0.0
                for chunk in chunks(items, lambda it: len(it["p"]) + len(it["c"])):
                    pairs = [(it["p"], it["c"]) for it in chunk]
                    s_logits, tgt = forward_sel(model, pairs)
                    lps = (-torch.nn.functional.cross_entropy(s_logits, tgt, reduction="none")).split([len(it["c"]) for it in chunk])
                    loss = s_logits.sum() * 0.0
                    for it, lp in zip(chunk, lps):
                        if not it["pg"]:
                            continue
                        if it_idx == 0:
                            it["old"] = lp.detach()
                        delta = lp - it["old"]
                        kl_sum += float(-delta.sum())
                        if args.algo == "gspo":
                            s = torch.exp(delta.mean())
                            obj = torch.minimum(s * it["a"], torch.clamp(s, 1 - 3e-4, 1 + 4e-4) * it["a"])
                            clipped += int(float(s) < 1 - 3e-4 or float(s) > 1 + 4e-4) * len(it["c"])
                            loss = loss - args.pg_coef * obj / len(pg_items)
                        else:
                            ratio = torch.exp(delta)
                            hi = 0.28 if args.algo == "dapo" else 0.2
                            obj = torch.minimum(ratio * it["a"], torch.clamp(ratio, 1 - 0.2, 1 + hi) * it["a"])
                            clipped += int(((ratio < 1 - 0.2) | (ratio > 1 + hi)).sum())
                            loss = loss - args.pg_coef * obj.sum() / n_tok
                    if teacher is not None:
                        with torch.no_grad():
                            t_logits, _ = forward_sel(teacher, pairs)
                            t_logp = torch.log_softmax(t_logits, dim=-1)
                        s_logp = torch.log_softmax(s_logits, dim=-1)
                        kl_tok = (s_logp.exp() * (s_logp - t_logp)).sum(-1)
                        keep = torch.cat([torch.full((len(it["c"]),), bool(it["kl"])) for it in chunk]).to(kl_tok.device)
                        kl = (kl_tok * keep).sum()
                        tkl_sum += float(kl)
                        loss = loss + args.kl_coef * kl / n_kl_tok
                        del t_logits, t_logp, s_logp
                    loss.backward()
                    loss_sum += float(loss)
                    del s_logits
                for chunk in chunks(her, lambda pc: len(pc[0]) + len(pc[1])):
                    h_logits, tgt = forward_sel(model, chunk)
                    nll = torch.nn.functional.cross_entropy(h_logits, tgt, reduction="sum")
                    (args.her_coef * nll / n_her_tok).backward()
                    her_sum += float(nll)
                    del h_logits
                stats["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                opt.step()
                stats.update(loss=loss_sum, clip_frac=clipped / n_tok, approx_kl=kl_sum / n_tok,
                             teacher_kl=tkl_sum / n_kl_tok, her_nll=her_sum / n_her_tok)
            for it in items:
                it.pop("old", None)
        n = len(scores)
        rec = {"step": step, "algo": args.algo, "reward_mode": args.reward,
               "pass_rate": sum(s["all_specs_met"] for s in scores) / n,
               "pass_rate_first_turn": sum(first_pass) / n, "retired": sum(v >= args.retire_after for v in solved_streak.values()) if args.retire_after else 0,
               "score_cache": len(score_cache), "novel_passes": n_novel, "novel_signatures": len(novel_seen),
               "mean_level": sum(task_level.values()) / max(len(task_level), 1),
               "batch_level": sum(int(t.get("id", "").rsplit("@L", 1)[1]) if "@L" in t["id"] else 0 for t in cand) / len(cand), "mean_turns": sum(len(tr) for tr in traj) / n,
               "mean_reward": sum(rewards) / n,
               "mean_shaped_reward": sum(float(s.get("reward", 0)) for s in scores) / n,
               "mean_spec_score": sum(float(s.get("spec_score", 0)) for s in scores) / n,
               "blocked": sum(bool(s.get("anticheat_blocked")) for s in scores) / n,
               "simulated": sum(bool(s.get("simulated")) for s in scores) / n,
               "unfinished": 1 - sum(done) / n, "live_group_frac": live / len(cand), "groups_trained": min(live, args.prompts),
               "trained_sequences": len(items), "her_pairs": len(her), **stats,
               "sec_gen": round(t_gen, 1), "sec_score": round(t_score, 1), "sec_total": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        for i, s in enumerate(scores):
            t = cand[i // args.k]
            rollouts.write(json.dumps({"step": step, "task": t["id"], "circuit_id": t["circuit_id"], "completion": texts[i],
                                       "all_specs_met": s["all_specs_met"], "reward": rewards[i], "canonical_reward": s.get("reward"),
                                       "blocked": s.get("anticheat_blocked"), "measured": s.get("measured"), "adv": adv_all[i]}) + "\n")
        rollouts.flush()
        print(json.dumps(rec), flush=True)
        sampler.unload(name)
        name = f"pol{step}"
        publish(name)
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step, name)
            ck = os.path.join(args.out, f"step_{step}")
            shutil.rmtree(ck, ignore_errors=True)
            shutil.copytree(live_dir, ck)
    json.dump(vars(args), open(os.path.join(args.out, "grpo_args.json"), "w"), indent=1)
    sampler.close()
    print("GRPO_DONE", flush=True)


if __name__ == "__main__":
    main()
