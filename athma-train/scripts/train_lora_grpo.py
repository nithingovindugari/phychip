#!/usr/bin/env python3
"""LoRA GRPO trainer for ablation condition D (and E's chained GRPO step).

100-step canary RLVR over LoRA on top of SFT+DPO. Reward = hierarchical
ngspice gate + per-spec margin from the appropriate harness module.

Usage:
    ABLATION_CONDITION=D ABLATION_STAGE=grpo \\
    .venv/bin/python athma-train/scripts/train_lora_grpo.py \\
        --base HuggingFaceTB/SmolLM3-3B-Base \\
        --sft-dpo-adapter checkpoints/ablation_C_sft_dpo_lora \\
        --specs data/rlvr_specs_v1/codex_session_1.jsonl \\
        --specs-limit 50 \\
        --out checkpoints/ablation_D_grpo_lora
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from athma_train.spice_gate import grade_netlist, normalize_netlist  # noqa: E402

HARNESSES_DIR = Path(__file__).resolve().parent.parent / "athma_train" / "harnesses"
_HARNESS_CACHE: dict[str, object] = {}


def load_harness(circuit_id: str):
    """Dynamically import athma_train.harnesses.<circuit_id>."""
    if circuit_id in _HARNESS_CACHE:
        return _HARNESS_CACHE[circuit_id]
    path = HARNESSES_DIR / f"{circuit_id}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"harness_{circuit_id}", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"  warn: failed to load harness {circuit_id}: {e}", file=sys.stderr)
        return None
    _HARNESS_CACHE[circuit_id] = mod
    return mod


def extract_netlist(text: str) -> str:
    """Pull the spice block from a completion."""
    m = re.search(r"```spice\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    # fallback: whole text
    return normalize_netlist(text)


def reward_fn(circuit_id: str, target_specs: dict, completion: str) -> float:
    """Reward-shaped hierarchical reward.

    Layers (each provides gradient signal so rare-pass rollouts aren't the only
    way to learn):

        0.00 → empty / no spice block / no .end
        0.10 → has ```spice``` block with .end
        0.20 → has at least one device line (R/C/L/M/Q/V/I/X)
        0.30 → has analysis directive (.op/.tran/.ac/.dc or .control block)
        0.50 → has both device lines AND analysis directive
        1.00 → ngspice simulation passes the gate
        1.0 + mean(per-spec margin) → harness-measured spec satisfaction (cap 2.0)

    This breaks the 0-reward floor that killed 20-step GRPO at sparse-pass
    rates. Now every rollout has variance even when none pass.
    """
    import re

    if not completion or not isinstance(completion, str):
        return 0.0

    # --- Layer 1: syntactic structure (cheap regex, no ngspice) ---
    has_spice_block = bool(re.search(r"```spice\s*.*?```", completion, re.DOTALL))
    netlist = extract_netlist(completion)
    has_end = bool(re.search(r"\.end\b", netlist, re.IGNORECASE))
    has_device = bool(re.search(r"^\s*[RCLMQVIX]\w*\s+\S+\s+\S+", netlist, re.MULTILINE | re.IGNORECASE))
    has_analysis = bool(re.search(r"\.(op|tran|ac|dc)\b|\.control\b", netlist, re.IGNORECASE))

    if not (has_spice_block and has_end):
        return 0.0

    base = 0.1
    if has_device: base = 0.2
    if has_analysis: base = 0.3
    if has_device and has_analysis: base = 0.5

    # --- Layer 2: ngspice simulation gate ---
    passed, _ = grade_netlist(netlist, timeout_s=15)
    if not passed:
        return base
    base = 1.0

    # --- Layer 3: harness-measured spec satisfaction ---
    harness = load_harness(circuit_id)
    if harness is None or not target_specs:
        return base
    try:
        measured = harness.measure(netlist, spec=target_specs, timeout_s=15)
    except Exception:
        return base

    n_total = 0
    margin_sum = 0.0
    for k, v_target in target_specs.items():
        v_meas = measured.get(k)
        if v_meas is None:
            continue
        n_total += 1
        try:
            v_meas = float(v_meas)
            # Window target [lo, hi] (codex spec format): full credit inside,
            # graded penalty by distance outside relative to window scale.
            if isinstance(v_target, (list, tuple)) and len(v_target) == 2:
                lo, hi = float(v_target[0]), float(v_target[1])
                if lo > hi:
                    lo, hi = hi, lo
                if lo <= v_meas <= hi:
                    margin_sum += 1.0
                else:
                    d = (lo - v_meas) if v_meas < lo else (v_meas - hi)
                    scale = max(abs(hi - lo), abs((lo + hi) / 2.0), 1e-9)
                    margin_sum += max(0.0, 1.0 - d / scale)
            else:
                # Scalar target: relative-error credit.
                tgt = float(v_target)
                rel_err = abs(v_meas - tgt) / max(abs(tgt), 1e-9)
                margin_sum += 1.0 if rel_err < 0.1 else max(0.0, 1.0 - rel_err)
        except (TypeError, ValueError):
            pass
    if n_total == 0:
        return base
    return base + (margin_sum / n_total)


def _anticheat_ok(netlist: str, circuit_id: str) -> bool:
    """Anti-reward-hacking topology guard (Kimi-k1.5 style; found via the
    2026-06-14 adversarial audit). A circuit must contain the real devices that
    *compute* its function — not a behavioral source that synthesizes the measured
    FOM directly. Closes the holes the audit found:
      - current-mirror gamed by a CCCS / fixed I-source (no transistor);
      - sallen-key gamed by a `laplace` B/E transfer (no RC network).
    Legit bench references pass (current-mirror has M×2; sallen-key has C×2).
    """
    import re as _re
    nl = (netlist or "")
    low = nl.lower()
    # No legitimate reference uses a Laplace/transfer-function block; it directly
    # synthesizes any frequency response -> always a cheat.
    if "laplace" in low or "tf(" in low:
        return False
    def count(prefixes):
        return len(_re.findall(rf"(?m)^\s*[{prefixes}]\w*\s", nl))
    need_transistor = {"current-mirror", "cs-stage", "diff-pair", "diff-amp",
                       "bandgap", "level-shift", "current-sense"}
    need_caps = {"sallen-key": 2, "integrator": 1, "differentiator": 1, "tia": 1}
    if circuit_id in need_transistor and count("MQmq") < 1:
        return False
    if circuit_id in need_caps and count("Cc") < need_caps[circuit_id]:
        return False
    return True


def reward_fn_v2(circuit_id: str, target_specs: dict, completion: str,
                 eval_tol: float = 0.30) -> float:
    """Spec-DOMINANT, tolerance-ALIGNED reward (the L2/L3-wall lever).

    Diagnosis of why reward_fn (v1) did not convert L2 pass@8=11/12 into greedy
    pass@1:
      (a) the ngspice gate contributes 1.0 — it DOMINATES, so GRPO's strongest
          gradient is "simulate", not "meet spec";
      (b) v1 gives smooth partial credit for being *close*, but the eval is a HARD
          ±30% window — so the policy learns "be close" and lands just outside.

    v2 fixes both:
      - structure+gate are capped at 0.30 of the reward (necessary, not sufficient);
      - spec score is 0.70 of the reward and gives FULL credit only INSIDE the eval
        tolerance (±30%), tiny shaped credit (×0.25) outside to keep a gradient;
      - +0.30 BONUS when ALL measured specs are within tolerance (the L2/L3 win
        condition) — sharpens the gradient toward complete satisfaction.
    Range: 0..1.30. A circuit that only simulates ≈0.30; meets-all ≈1.30.
    """
    import re
    if not completion or not isinstance(completion, str):
        return 0.0
    netlist = extract_netlist(completion)
    has_end = bool(re.search(r"\.end\b", netlist, re.IGNORECASE))
    has_spice_block = bool(re.search(r"```spice\s*.*?```", completion, re.DOTALL))
    has_device = bool(re.search(r"^\s*[RCLMQVIX]\w*\s+\S+\s+\S+", netlist, re.MULTILINE | re.IGNORECASE))
    has_analysis = bool(re.search(r"\.(op|tran|ac|dc)\b|\.control\b", netlist, re.IGNORECASE))
    if not (has_spice_block and has_end):
        return 0.0
    # structure sub-reward (max 0.15 before the gate)
    struct = 0.05
    if has_device: struct = 0.08
    if has_device and has_analysis: struct = 0.15
    passed, _ = grade_netlist(netlist, timeout_s=15)
    if not passed:
        return struct
    base = 0.30  # structure + gate, capped
    # Anti-cheat: a netlist that fakes the FOM with a behavioral source instead of
    # the real device network gets gate-only credit, never the spec bonus.
    if not _anticheat_ok(netlist, circuit_id):
        return base
    harness = load_harness(circuit_id)
    if harness is None or not target_specs:
        return base
    try:
        measured = harness.measure(netlist, spec=target_specs, timeout_s=15)
    except Exception:
        return base
    n_total, met, spec_score = 0, 0, 0.0
    for k, v_target in target_specs.items():
        v_meas = measured.get(k)
        if v_meas is None:
            continue
        try:
            v_meas = float(v_meas)
        except (TypeError, ValueError):
            continue
        n_total += 1
        # derive [lo,hi] from window or scalar±eval_tol
        if isinstance(v_target, (list, tuple)) and len(v_target) == 2:
            lo, hi = sorted((float(v_target[0]), float(v_target[1])))
        else:
            tgt = float(v_target); lo, hi = sorted((tgt * (1 - eval_tol), tgt * (1 + eval_tol)))
        if lo <= v_meas <= hi:
            spec_score += 1.0; met += 1
        else:
            d = (lo - v_meas) if v_meas < lo else (v_meas - hi)
            scale = max(abs(hi - lo), abs((lo + hi) / 2.0), 1e-9)
            spec_score += 0.25 * max(0.0, 1.0 - d / scale)  # small, capped
    if n_total == 0:
        return base
    r = base + 0.70 * (spec_score / n_total)
    if met == n_total:
        r += 0.30  # all-specs-met bonus (the eval win condition)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM3-3B-Base")
    ap.add_argument("--sft-dpo-adapter", required=True,
                    help="Stacked SFT+DPO LoRA adapter — will be merged before GRPO LoRA")
    ap.add_argument("--specs", type=Path,
                    default=Path("data/rlvr_specs_v1/codex_session_1.jsonl"))
    ap.add_argument("--specs-limit", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume-from", default=None,
                    help="path to a checkpoint-N dir to resume GRPO from (restores optimizer/scheduler/step)")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--steps", type=int, default=100,
                    help="Number of GRPO update steps (canary = 100)")
    ap.add_argument("--rollouts-per-step", type=int, default=4,
                    help="K in GRPO (group size per prompt)")
    # generation_batch_size = bs * world_size must be divisible by num_generations (K).
    # With 2 GPUs and K=4, bs=2 -> gen_batch=4 ✓ (bs=1 -> gen_batch=2 ✗).
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--max-prompt-len", type=int, default=1024)
    ap.add_argument("--max-completion-len", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    condition = os.environ.get("ABLATION_CONDITION", "?")
    stage = os.environ.get("ABLATION_STAGE", "grpo")
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.no_wandb:
        from athma_train.tracking import (
            configure_wandb_env, wandb_init_kwargs, assert_wandb_token_present,
        )
        assert_wandb_token_present()
        configure_wandb_env("ablation")
        import wandb
        wandb.init(**wandb_init_kwargs(
            stage="ablation",
            base=args.base,
            descriptor=f"{condition}-grpo-loraR{args.lora_rank}-K{args.rollouts_per_step}",
            framework="trl",
            extra_tags=[
                f"condition:{condition}", f"stage:{stage}",
                f"lora_rank:{args.lora_rank}",
                "adapter_chain:" + ("sft→dpo→grpo" if condition == "D" else "cpt→sft→dpo→grpo"),
            ],
            config={**vars(args), "condition": condition},
        ))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    from trl import GRPOConfig, GRPOTrainer

    print(f"Loading base: {args.base}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Base tokenizer has no chat_template; pull from Instruct variant so that
    # GRPO's conversational-prompt path generates chat-formatted text matching
    # the SFT adapter's training distribution. Without this, raw prompts hit a
    # chat-trained model and it emits a single EOS token (mean_length=1, reward=0).
    if tok.chat_template is None:
        try:
            from transformers import AutoTokenizer as _ATK
            inst = _ATK.from_pretrained("HuggingFaceTB/SmolLM3-3B")
            tok.chat_template = inst.chat_template
            print(f"  loaded chat_template from SmolLM3-3B (instruct)", file=sys.stderr)
        except Exception as e:
            print(f"  warn: couldn't load instruct chat_template: {e}", file=sys.stderr)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = {"": local_rank} if local_rank >= 0 else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map=device_map,
        attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
    )
    print(f"Loading SFT+DPO adapter: {args.sft_dpo_adapter}", file=sys.stderr)
    model = PeftModel.from_pretrained(model, args.sft_dpo_adapter, is_trainable=False)
    model = model.merge_and_unload()

    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()  # required for gradient_checkpointing on a PEFT model
    model.print_trainable_parameters()

    # --- Stop-token fix (critical): the SFT model ends assistant turns with
    # <|im_end|>, but the tokenizer's eos is <|end_of_text|>. Without this,
    # GRPO generation never stops -> every rollout hits max_completion_length,
    # the netlist never parses, reward=0, advantage variance=0, grad_norm=0
    # (a silent no-op run). Make both TRL's eos logic and HF generate stop on
    # <|im_end|> (and keep end_of_text as a fallback eos).
    _im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(_im_end, int) and _im_end >= 0:
        _eot = tok.eos_token_id
        tok.eos_token = "<|im_end|>"          # TRL uses processing_class.eos_token_id
        if tok.pad_token is None or tok.pad_token_id == _im_end:
            tok.pad_token_id = _eot           # keep pad != eos
        _eos_ids = sorted({_im_end, _eot})
        try:
            model.generation_config.eos_token_id = _eos_ids
            model.generation_config.pad_token_id = tok.pad_token_id
        except Exception as e:
            print(f"  warn: gen eos set failed: {e}", file=sys.stderr)
        print(f"  stop-token fix: eos -> <|im_end|>({_im_end}); gen eos_ids={_eos_ids}, "
              f"pad={tok.pad_token_id}", file=sys.stderr)
    else:
        print("  warn: <|im_end|> not in vocab; generation may not terminate", file=sys.stderr)

    # Load RLVR specs — expect schema {circuit_id, spec_prompt, target_measurements}
    from datasets import Dataset
    print(f"Loading RLVR specs: {args.specs}", file=sys.stderr)
    specs = []
    with args.specs.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = row.get("circuit_id")
            prompt = row.get("spec_prompt") or row.get("prompt")
            targets = row.get("target_measurements") or row.get("targets") or {}
            if cid and prompt:
                specs.append({"circuit_id": cid, "prompt": prompt, "targets": targets})
    specs = specs[:args.specs_limit]
    print(f"  using {len(specs)} specs", file=sys.stderr)
    # Prompt format. Default = conversational (chat_template). But the SFT model
    # terminates cleanly + emits parseable netlists in COMPLETION format (the
    # format its eval scored 26/46 in); in chat format at temp>0 it rambles to
    # max_completion_length (clipped_ratio=1, 0 terminated, reward=0, grad_norm=0
    # -> no-op). PHYCHIP_GRPO_COMPLETION_FMT=1 uses the completion prompt that
    # primes a direct netlist and stops at the netlist's stop token.
    import os as _os0
    _COMPL = _os0.environ.get("PHYCHIP_GRPO_COMPLETION_FMT") == "1"
    _CTMPL = ("{spec}\n\nOutput your complete ngspice netlist between ```spice and "
              "``` fences. Include device lines, an analysis directive (.op/.tran/.ac "
              "inside a .control/.endc block), and end with .end.\n\n```spice\n")
    if _COMPL:
        ds = Dataset.from_list([{
            "prompt": _CTMPL.format(spec=s["prompt"]),
            "circuit_id": s["circuit_id"],
            "targets": json.dumps(s["targets"]),
        } for s in specs])
        # NOTE: tried generation_config.stop_strings=['.end','```'] to terminate
        # rollouts — but TRL GRPO doesn't pass a tokenizer to generate, so it raises
        # ("could not locate a tokenizer"). Instead we rely on mask_truncated_completions
        # =False (below): completions clip at max_len but, unmasked, still contribute
        # their (valid-netlist) reward to the gradient.
        print("  prompt format: COMPLETION (primes direct netlist; no stop-string, mask off)", file=sys.stderr)
    else:
        ds = Dataset.from_list([{
            "prompt": [{"role": "user", "content": s["prompt"]}],
            "circuit_id": s["circuit_id"],
            "targets": json.dumps(s["targets"]),
        } for s in specs])

    # Reward function (TRL signature). PHYCHIP_REWARD=v2 -> spec-dominant,
    # tolerance-aligned reward (the L2/L3-wall lever; ~1.9x sharper spec contrast).
    import os as _os
    _reward = reward_fn_v2 if _os.environ.get("PHYCHIP_REWARD") == "v2" else reward_fn
    # MFU lever: the reward is bs*K independent ngspice sims/step (CPU-bound). Serial
    # scoring strands the GPU at ~28% util (lesson #152/#154). PHYCHIP_REWARD_WORKERS>1
    # scores the group on a thread pool — grade_netlist uses per-call NamedTemporaryFile
    # + subprocess, so it is thread-safe and the GIL releases during the ngspice wait.
    # Default 1 = serial (production-unchanged). Set ~= min(vCPUs, bs*K) on the pod.
    _RW = max(1, int(_os.environ.get("PHYCHIP_REWARD_WORKERS", "1")))
    _POOL = None
    if _RW > 1:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max_workers=_RW)
    print(f"  reward fn: {_reward.__name__} | reward workers: {_RW}", file=sys.stderr)

    def _score_one(item):
        comp, cid, tgt_json = item
        if isinstance(comp, list):  # list of {"role":"assistant","content":...}
            comp = comp[-1].get("content", "") if comp else ""
        # COMPLETION-FORMAT FIX: the prompt primed the opening ```spice fence, so the
        # model's completion starts at the netlist and lacks it. reward_fn_v2 requires
        # a ```spice...``` block (has_spice_block) -> would score every valid netlist 0.
        # Re-add the opening fence so scoring matches (verified: 0.0 -> 0.3-0.55).
        if _COMPL and isinstance(comp, str) and "```spice" not in comp:
            comp = "```spice\n" + comp.rstrip() + "\n```"  # add BOTH fences (clean-stopped netlist has neither)
        try:
            tgt = json.loads(tgt_json) if isinstance(tgt_json, str) else (tgt_json or {})
        except json.JSONDecodeError:
            tgt = {}
        return _reward(cid, tgt, comp)

    def reward_func(completions, **kwargs):
        # kwargs gives full row passthrough via dataset columns
        circuit_ids = kwargs.get("circuit_id", [None] * len(completions))
        targets_list = kwargs.get("targets", ["{}"] * len(completions))
        items = list(zip(completions, circuit_ids, targets_list))
        if _POOL is not None:
            return list(_POOL.map(_score_one, items))  # order-preserving
        return [_score_one(it) for it in items]

    # TRL 1.5.1: GRPOConfig has max_completion_length but not max_prompt_length.
    cfg_kwargs = dict(
        output_dir=str(args.out),
        per_device_train_batch_size=args.bs,
        num_generations=args.rollouts_per_step,
        max_steps=args.steps,
        max_completion_length=args.max_completion_len,
        temperature=args.temperature,
        learning_rate=args.lr,
        beta=args.kl_coef,
        save_steps=args.save_steps,
        logging_steps=1,  # every step for canary
        bf16=True,
        report_to="wandb" if not args.no_wandb else "none",
        seed=args.seed,
    )
    # gradient_checkpointing is OFF by default: with this PEFT+merge_and_unload
    # setup it zeroed the gradient (grad_norm=0, silent no-op). Re-enable only via
    # PHYCHIP_GRAD_CKPT=1 if memory forces it (then prefer smaller K/bs first).
    if _os.environ.get("PHYCHIP_GRAD_CKPT") == "1":
        cfg_kwargs["gradient_checkpointing"] = True
        cfg_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    # --- 2026 RLVR recipe, env-gated so default behavior is unchanged ---
    # PHYCHIP_GRPO_RECIPE=modern enables: Dr.GRPO (no std-norm) + DAPO loss
    # (token-level + asymmetric clip) + dynamic sampling (drop all-same groups).
    # Each is also individually overridable. Levers only applied if the installed
    # TRL's GRPOConfig accepts them (introspected) so we never crash on version skew.
    import inspect as _inspect
    _gc_params = set(_inspect.signature(GRPOConfig.__init__).parameters)
    if _os.environ.get("PHYCHIP_GRPO_RECIPE") == "modern":
        _wanted = dict(
            scale_rewards=False,          # Dr.GRPO: drop std-normalization bias
            loss_type="dapo",             # token-level loss aggregation
            epsilon_high=0.28,            # DAPO asymmetric clip (preserve exploration)
            # In completion format the model emits a complete netlist then rambles
            # (no <|im_end|>), so completions "clip" at max_len -> masking them ALL
            # zeroes the gradient despite real reward variance. Don't mask in COMPL.
            mask_truncated_completions=(not _COMPL),
        )
        for _k, _v in _wanted.items():
            if _k in _gc_params:
                cfg_kwargs[_k] = _v
            else:
                print(f"  (TRL lacks GRPOConfig.{_k}; skipped)", file=sys.stderr)
    # individual overrides (e.g. PHYCHIP_GRPO_SCALE_REWARDS=0)
    if "PHYCHIP_GRPO_SCALE_REWARDS" in _os.environ and "scale_rewards" in _gc_params:
        cfg_kwargs["scale_rewards"] = _os.environ["PHYCHIP_GRPO_SCALE_REWARDS"] not in ("0", "false", "False")
    cfg = GRPOConfig(**cfg_kwargs)
    print(f"  GRPOConfig recipe: scale_rewards={cfg_kwargs.get('scale_rewards', True)} "
          f"loss_type={cfg_kwargs.get('loss_type','grpo')} beta(KL)={args.kl_coef} "
          f"K={args.rollouts_per_step}", file=sys.stderr)
    # Optional: Microsoft post-training-toolkit diagnostics (GRPO group
    # rewards / advantages / KL, crash postmortem). Never breaks the run.
    callbacks = []
    try:
        from post_training_toolkit import DiagnosticsCallback
        callbacks.append(DiagnosticsCallback())
        print("  + post-training-toolkit DiagnosticsCallback enabled", file=sys.stderr)
    except Exception as e:
        print(f"  (post-training-toolkit not active: {e})", file=sys.stderr)

    trainer = GRPOTrainer(
        model=model, args=cfg, train_dataset=ds,
        reward_funcs=[reward_func],
        processing_class=tok,
        callbacks=callbacks or None,
    )
    if args.resume_from:
        print(f"  RESUMING GRPO from {args.resume_from}", file=sys.stderr)
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))

    (args.out / "ablation_manifest.json").write_text(json.dumps({
        "condition": condition, "stage": stage,
        "base": args.base, "sft_dpo_adapter": args.sft_dpo_adapter,
        "specs": str(args.specs), "specs_used": len(specs),
        "lora_rank": args.lora_rank, "steps": args.steps,
        "rollouts_per_step": args.rollouts_per_step, "lr": args.lr, "kl_coef": args.kl_coef,
    }, indent=2))
    print(f"DONE: GRPO adapter saved to {args.out}", file=sys.stderr)
    if not args.no_wandb:
        import wandb; wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
