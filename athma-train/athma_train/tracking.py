"""Weights & Biases naming conventions for PhyChip training runs.

Single source of truth for W&B project, run name, and tag conventions.
Every training script (phase0, ablation1, stage1_cpt, stage2_cpt,
stage3_sft, stage4_dpo, stage5_rlvr) imports from here so naming never
drifts between scripts.

NAMING CONVENTION :

    Project:   phychip-<stage>          → e.g. phychip-stage1-cpt
    Run name:  <base>-<descriptor>-<YYYYMMDD>
                                        → e.g. SmolLM3-3B-Base-loraR16-20260514
    Tags:      [<stage>, <base>, framework:<fw>, <extras...>]
                                        → e.g. ["ablation1", "SmolLM3-3B-Base",
                                                 "framework:unsloth", "lora-r16"]
    Group:     <stage>                  → so cross-base runs cluster

Why one project per stage (not one project per run, not one project total):
- Within-stage hyperparameter sweeps cluster naturally in the same project
- Cross-stage comparison is rare (different metrics anyway)
- W&B project naming is free — no cost to having 7

Stage keys here are different from athma_train.hub.STAGE_REPOS keys.
The hub keys are per-base (ablation1-smollm3, ablation1-qwen3.5);
the W&B keys are per-stage (ablation1) so both bases share a project
and you can compare them side-by-side in W&B Reports.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Literal

StageKey = Literal[
    "phase0",
    "ablation1",
    "ablation",      # 2026-06: 5-condition CPT-vs-no-CPT ablation
    "stage1-cpt",
    "stage2-cpt",
    "stage3-sft",
    "stage4-dpo",
    "stage5-rlvr",
]

STAGE_WANDB_PROJECTS: dict[StageKey, str] = {
    "phase0": "phychip-phase0",
    "ablation1": "phychip-ablation1",
    "ablation": "phy-chip-ablation",  # all 5 conditions share one project
    "stage1-cpt": "phychip-stage1-cpt",
    "stage2-cpt": "phychip-stage2-cpt",
    "stage3-sft": "phychip-stage3-sft",
    "stage4-dpo": "phychip-stage4-dpo",
    "stage5-rlvr": "phychip-stage5-rlvr",
}

WANDB_ENTITY = os.environ.get("WANDB_ENTITY")  # only set if you have an org


def _normalize(name: str) -> str:
    """Strip HF org prefix + replace any path separators."""
    return name.replace("/", "-").replace(" ", "-")


def wandb_run_name(
    stage: StageKey,
    base: str,
    descriptor: str | None = None,
    date: str | None = None,
) -> str:
    """Convention: <base>-<descriptor>-<YYYYMMDD>.

    `base` can be either a short name (SmolLM3-3B-Base) or full HF id
    (HuggingFaceTB/SmolLM3-3B-Base) — both normalize to the same string.
    `descriptor` is a short hp-sweep marker (loraR16, fullft, lr5e-5).
    """
    parts = [_normalize(base)]
    if descriptor:
        parts.append(descriptor)
    parts.append(date or datetime.now().strftime("%Y%m%d"))
    return "-".join(parts)


def wandb_tags(
    stage: StageKey,
    base: str,
    framework: str | None = None,
    extras: list[str] | None = None,
) -> list[str]:
    """Standardized tag list — used for filtering + grouping in W&B UI."""
    tags = [stage, _normalize(base)]
    if framework:
        tags.append(f"framework:{framework}")
    if extras:
        tags.extend(extras)
    return tags


def wandb_init_kwargs(
    stage: StageKey,
    base: str,
    descriptor: str | None = None,
    framework: str | None = None,
    extra_tags: list[str] | None = None,
    config: dict | None = None,
) -> dict:
    """Returns kwargs ready for either `wandb.init()` or for setting on
    `TrainingArguments` fields (project via WANDB_PROJECT env var,
    run_name via TrainingArguments.run_name, tags via wandb.init).

    For TrainingArguments / SFTConfig / GRPOConfig usage, set:
        os.environ["WANDB_PROJECT"] = kwargs["project"]
        TrainingArguments(..., run_name=kwargs["name"], report_to="wandb")
        # then in a TrainerCallback.setup() or before train(), set tags:
        wandb.run.tags = kwargs["tags"]
    """
    kwargs = {
        "project": STAGE_WANDB_PROJECTS[stage],
        "name": wandb_run_name(stage, base, descriptor),
        "tags": wandb_tags(stage, base, framework, extra_tags),
        "group": stage,
    }
    if WANDB_ENTITY:
        kwargs["entity"] = WANDB_ENTITY
    if config:
        kwargs["config"] = config
    return kwargs


def configure_wandb_env(stage: StageKey) -> None:
    """Set WANDB_PROJECT in the environment so HF Trainer's
    `report_to='wandb'` picks up the right project automatically.

    Call this before constructing TrainingArguments.
    """
    os.environ["WANDB_PROJECT"] = STAGE_WANDB_PROJECTS[stage]
    if WANDB_ENTITY:
        os.environ["WANDB_ENTITY"] = WANDB_ENTITY


def assert_wandb_token_present() -> None:
    """Hard-fail early if WANDB_API_KEY is missing — better than 5 hours
    into a CPT run discovering tracking is silently dropping data.

    Note: same `source .env` subprocess gotcha as HF_TOKEN — must be
    exported, not just sourced. See learning.md #45.
    """
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY env var is not set. Use `set -a; source .env; "
            "set +a` (the bare `source .env` does NOT export to subprocesses), "
            "or pass --no-wandb to skip tracking."
        )
