#!/bin/bash
# Runs ON a fresh GPU pod. Phase 1-4 of plan/rl_ood_proof_plan.md:
# build ngspice-46 -> eval masked-SFT baseline -> best GRPO (modern recipe,
# verified data, 3 seeds) -> eval every checkpoint. Idempotent-ish; nohup-safe.
# Prereq: ~/phy-chip already rsynced (code, data/rlvr_specs_v2_verified, benches,
# checkpoints/sft_masked) by plan/resume_grpo.sh.
set -uo pipefail
cd ~/phy-chip || exit 1
LOG(){ echo "[$(date -u +%H:%M:%S)] $*"; }
# non-root massedcompute pods need sudo for apt/make install (learning #183)
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
export PATH=/usr/local/bin:$PATH   # make install lands ngspice here; non-login ssh drops it

BASE=HuggingFaceTB/SmolLM3-3B-Base
SFT=checkpoints/sft_masked            # the corrected masked-SFT adapter
SPECS=data/rlvr_specs_v2_verified/specs.jsonl
EVAL=athma-train/scripts/eval_on_bench_v1.py
V1=eval_sets/bench_v1/bench_v1.jsonl
V2=eval_sets/phychip_bench_v2/bench_v2.jsonl
export PHYCHIP_REQUIRE_NGSPICE=46
export PHYCHIP_REWARD=v2
export PHYCHIP_GRPO_RECIPE=modern     # Dr.GRPO + DAPO + dynamic sampling
export HF_HUB_OFFLINE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation OOM

# W&B: PHYCHIP_WANDB=1 enables online logging (needs WANDB_API_KEY in env).
# Each seed becomes its own run (name = output_dir grpo_ood_s<seed>); set
# WANDB_NAME externally per launch to be explicit.
export WANDB_PROJECT="${WANDB_PROJECT:-phy-chip-grpo-ood}"
if [ "${PHYCHIP_WANDB:-0}" = "1" ]; then WANDB_ARG=""; else WANDB_ARG="--no-wandb"; fi

# --- multi-GPU + long-context knobs (env-overridable) -----------------------
# 1 GPU default; PHYCHIP_NPROC=2 → DDP via torchrun (each rank holds bs prompts,
# so use smaller per-device BS to fit longer COMPLETION). Constraint: BS*NPROC
# must be divisible by K (rollouts-per-step). 8k context needs NPROC=2 + BS=4 +
# grad-ckpt (single 80GB OOMs at bs8/4096 — measured 77.5/80GB).
NPROC="${PHYCHIP_NPROC:-1}"
BS="${PHYCHIP_BS:-8}"
COMPLETION="${PHYCHIP_COMPLETION:-4096}"
K="${PHYCHIP_K:-8}"
# reward threads: respect an explicit override (for parallel single-GPU runs that
# share the vCPUs); else split across ranks so DDP doesn't oversubscribe.
export PHYCHIP_REWARD_WORKERS="${PHYCHIP_REWARD_WORKERS:-$(( $(nproc) / NPROC ))}"; [ "$PHYCHIP_REWARD_WORKERS" -lt 1 ] && export PHYCHIP_REWARD_WORKERS=1
export PYTHONUNBUFFERED=1   # stream trainer output to the log (torch.distributed.run buffers otherwise)
if [ "$NPROC" -gt 1 ]; then
  # python3 -u -m torch.distributed.run avoids the ~/.local/bin PATH issue (#147)
  LAUNCH="python3 -u -m torch.distributed.run --nproc_per_node=$NPROC athma-train/scripts/train_lora_grpo.py"
else
  LAUNCH="python3 -u athma-train/scripts/train_lora_grpo.py"
fi

LOG "=== STEP 0: deps (ngspice-46 + python) ==="
if ! ngspice --version 2>/dev/null | grep -q "ngspice-46"; then
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq build-essential bison flex libxaw7-dev libreadline-dev wget gfortran python3-pip 2>&1 | tail -1
  cd /tmp && wget -q https://downloads.sourceforge.net/project/ngspice/ng-spice-rework/46/ngspice-46.tar.gz -O n.tgz
  tar xzf n.tgz && cd ngspice-46 && ./configure --disable-debug --with-readline=yes >/tmp/ngc.log 2>&1
  # CCLD=g++ (NOT -j, NOT --enable-cider): the known-good link recipe (#183)
  make CCLD=g++ >/tmp/ngm.log 2>&1 && $SUDO make install CCLD=g++ >/tmp/ngi.log 2>&1 && $SUDO ldconfig
  cd ~/phy-chip
fi
ngspice --version 2>&1 | tr -d '\000' | grep -i "ngspice-46" | head -1 || { LOG "FATAL: ngspice-46 missing"; exit 1; }
# pip + torch (fresh ubuntu image has neither)
command -v pip >/dev/null 2>&1 || { $SUDO apt-get install -y -qq python3-pip 2>&1 | tail -1; }
$SUDO python3 -m pip install -q -U pip 2>&1 | tail -1
python3 -c "import torch" 2>/dev/null || $SUDO python3 -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
$SUDO python3 -m pip install -q -U "trl>=1.6" transformers peft datasets accelerate safetensors "jinja2>=3.1.0" wandb 2>&1 | tail -1
python3 -c "import torch;assert torch.cuda.is_available();print('cuda ok',torch.cuda.get_device_name(0))"

LOG "=== STEP 0b: \$0 reward-spread gate (#131) — abort if data not learnable ==="
PYTHONPATH=athma-train python3 - <<'PY' || { echo "REWARD GATE FAILED"; exit 1; }
import json,sys,importlib.util,random
spec=importlib.util.spec_from_file_location('g','athma-train/scripts/train_lora_grpo.py')
g=importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(g)
except SystemExit: pass
rows=[json.loads(l) for l in open('data/rlvr_specs_v2_verified/specs.jsonl')]
random.seed(0); random.shuffle(rows)
good=[]
for r in rows[:20]:
    v=g.reward_fn(r['circuit_id'],r['target_measurements'],'```spice\n'+r['reference_netlist']+'\n```')
    if isinstance(v,(int,float)): good.append(v)
m=sum(good)/len(good); print(f"reward gate: mean={m:.2f} >1.0={sum(x>1 for x in good)}/{len(good)}")
sys.exit(0 if m>1.3 else 1)
PY

LOG "=== STEP 1: masked-SFT baseline on bench v1 + v2 (corrected baseline) ==="
for B in "v1 $V1" "v2 $V2"; do set -- $B
  [ -f "eval_results/masked_sft_$1/summary.json" ] && { LOG "baseline $1 already done, skip"; continue; }
  PYTHONPATH=athma-train python3 "$EVAL" --base "$BASE" --adapter "$SFT" \
    --bench "$2" --output-dir "eval_results/masked_sft_$1" --temperature 0.0 \
    --max-new-tokens 600 2>&1 | grep -viE "warning|deprecat" | tail -3
done

LOG "=== STEP 2: best GRPO from masked SFT (modern recipe) ==="
# K=8 rollouts, KL low (0.01), reward v2 + workers, 300 steps, save every 50.
# max-completion-len=2048: 1024 caused EVERY rollout to clip -> reward 0 ->
# dead gradient (feedback_long_completion_budget). 2048 gives live reward/std.
# Seeds configurable so each pod runs one seed in parallel (PHYCHIP_SEEDS="42").
LOG "    launch: NPROC=$NPROC BS=$BS K=$K COMPLETION=$COMPLETION GRAD_CKPT=${PHYCHIP_GRAD_CKPT:-0} REWARD_WORKERS=$PHYCHIP_REWARD_WORKERS"
for SEED in ${PHYCHIP_SEEDS:-42 1 7}; do
  LOG "--- GRPO seed $SEED ---"
  PYTHONPATH=athma-train $LAUNCH \
    --base "$BASE" --sft-dpo-adapter "$SFT" \
    --specs "$SPECS" --specs-limit 3669 \
    --out "checkpoints/grpo_ood_s${SEED}" \
    --steps "${PHYCHIP_STEPS:-300}" --rollouts-per-step "$K" --bs "$BS" --lr 1e-5 \
    --kl-coef "${PHYCHIP_KL:-0.01}" --save-steps 50 --seed "$SEED" $WANDB_ARG \
    --max-completion-len "$COMPLETION" \
    2>&1 | stdbuf -oL grep -viE "warning|deprecat"   # stream live; NO `tail` (it buffers to EOF — #168)
  pkill -9 -f train_lora_grpo 2>/dev/null; pkill -9 -f torchrun 2>/dev/null; sleep 8   # free GPU + zombie fix (#165)
  LOG "--- GRPO seed $SEED DONE; eval checkpoints ---"
  for CK in checkpoints/grpo_ood_s${SEED}/checkpoint-*; do
    [ -d "$CK" ] || continue
    for B in "v1 $V1" "v2 $V2"; do set -- $B
      PYTHONPATH=athma-train python3 "$EVAL" --base "$BASE" \
        --pre-adapters "$SFT" --adapter "$CK" --bench "$2" \
        --output-dir "eval_results/grpo_ood_s${SEED}_$(basename $CK)_$1" \
        --temperature 0.0 --max-new-tokens 600 2>&1 | grep -viE "warning|deprecat" | tail -2
    done
  done
done

LOG "=== ALL_GRPO_DONE ==="
grep -rhoE '"n_passed": [0-9]+|"pass@1": [0-9.]+' eval_results/grpo_ood_* eval_results/masked_sft_* 2>/dev/null | head -40
