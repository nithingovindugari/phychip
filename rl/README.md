# PhyChip RL: training small models against a circuit simulator

Code for the experiments in "What Breaks When Physics Becomes the Reward?". Code only. Training data, model weights and the held-out test sets are not released.

## What is here

| File | Purpose |
|---|---|
| `grpo.py` | RL trainer. vLLM generation with per-step LoRA reload, Dr.GRPO / DAPO / GSPO, online dynamic sampling, rewards (sparse, log-distance, shaped, margin, ladder, random), pass@k advantage, novelty bonus, window-tightening curriculum (`--anneal`), multi-turn RL with simulator feedback (`--turns`), on-policy distillation (`--teacher`, `--kl-coef`), physics answer reward, score caching |
| `train_sft.py`, `merge.py` | LoRA SFT (assistant-only loss) and adapter merge |
| `hindsight.py`, `amplify.py`, `build_sft.py`, `build_repair.py`, `mix_eq.py`, `make_eq_kits.py`, `validate_templates.py` | Data engine: hindsight relabelling, component-value amplification, equation-carrying netlists, repair data |
| `label_pool.py` | Offline dynamic sampling: keep tasks the current model solves 1-7 of 8 times |
| `phychip_v12.py` | Scorer wrapper. `PHYCHIP_SCORER=v13/v14/v15` (v14 adds the precharge and ideal-source checks) |
| (not included) | Patches to the canonical scorer (v13, v15) stay private; `phychip_v12.py` has the v14 checks |
| `score.py`, `vgen.py`, `gen.py`, `loop_eval.py`, `forward_model.py`, `yield_eval.py`, `phys_eval.py` | Evaluation: greedy and k=8 sampling, closed-loop feedback vs blind resampling, forward-model self-check, tolerance yield, physics answers |
| `scripts/` | Replay saved generations under a new scorer (`rescore14.py`, `rescore15.py`), structural diversity, physics-problem verification, SFT/eval helpers |

## Setup

- ngspice 46. Results depend on the ngspice version.
- The `phy_chip_env` scorer package (not public). Set `PHYCHIP_REPO` to its checkout.
- Python 3.12, torch, transformers, peft, trl and vLLM. Paths in the scripts are relative to a working directory that holds `base_models/`, `data/`, `runs/` and `merged/`.

## Typical run

```bash
python train_sft.py --base base_models/Qwen3-4B --data data/<sft_set>.jsonl --out runs/ckpt/sft --epochs 2 --lr 2e-4
PHYCHIP_SCORER=v15 python grpo.py --base base_models/Qwen3-4B --adapter runs/ckpt/sft \
  --tasks data/<rl_pool>.jsonl --eval-tasks <dev_tasks>.jsonl --out runs/rl/run0 \
  --algo drgrpo --reward sparse --k 8 --prompts 12 --oversample 2.0 --steps 100
```

Lessons from the paper that matter when reusing this:
- The verifier is the bottleneck. Audit passing answers after every RL run.
- Keep every generation, so a scorer fix is a replay, not a retrain.
- Re-filter the task pool with each new starting model.
- Simulation, not the GPU, sets the step time. Overlap scoring with generation.
