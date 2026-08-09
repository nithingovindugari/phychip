"""HuggingFace Hub push helpers for PhyChip training scripts.

Single source of truth for stage → repo mapping. Every training script
(ablation1, stage1_cpt, stage2_cpt, stage3_sft, stage4_dpo, stage5_rlvr)
imports `push_stage_checkpoint` from here so the repo names live in one
place and we don't lose checkpoints to ephemeral GPU pods again.

Usage from a training script:
    from athma_train.hub import push_stage_checkpoint
    push_stage_checkpoint(
        local_dir=Path("eval_results/ablation1_SmolLM3-3B-Base/final"),
        stage="ablation1-smollm3",
        step=886,
        extra_metrics={"ngspice_pass@1": 0.75},
    )

Storage architecture: see helper_tools/storage_setup.md
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

StageKey = Literal[
    "ablation1-smollm3",
    "ablation1-qwen3.5",
    "stage1-cpt",
    "stage2-cpt",
    "stage3-sft",
    "stage4-dpo",
    "stage5-rlvr",
    "v0-final",
]

STAGE_REPOS: dict[StageKey, str] = {
    "ablation1-smollm3": "NithinReddyG/PhyChip-Ablation1-SmolLM3-LoRA",
    "ablation1-qwen3.5": "NithinReddyG/PhyChip-Ablation1-Qwen3.5-LoRA",
    "stage1-cpt": "NithinReddyG/PhyChip-3B-Stage1-CPT",
    "stage2-cpt": "NithinReddyG/PhyChip-3B-Stage2-CPT",
    "stage3-sft": "NithinReddyG/PhyChip-3B-SFT",
    "stage4-dpo": "NithinReddyG/PhyChip-3B-DPO",
    "stage5-rlvr": "NithinReddyG/PhyChip-3B-v0",
    "v0-final": "NithinReddyG/PhyChip-3B-v0",
}

EVAL_DATASET_REPO = "NithinReddyG/phy-chip-bench-v0"
SHARDS_BUCKET = "hf://buckets/NithinReddyG/phy-chip-shards"
ROLLOUTS_BUCKET = "hf://buckets/NithinReddyG/phy-chip-rollouts"


def push_checkpoint(
    local_dir: Path,
    repo_id: str,
    *,
    private: bool = True,
    commit_message: str | None = None,
    tag: str | None = None,
    create_if_missing: bool = True,
    repo_type: str = "model",
) -> str:
    """Push a local checkpoint folder to a HF Hub repo.

    Returns the commit URL. Idempotent: if `create_if_missing=True` and the
    repo already exists, it is reused. Tagging is best-effort (logs warning
    on failure rather than raising).
    """
    from huggingface_hub import HfApi

    if not local_dir.exists():
        raise FileNotFoundError(f"local_dir does not exist: {local_dir}")
    if not any(local_dir.iterdir()):
        raise ValueError(f"local_dir is empty: {local_dir}")

    api = HfApi(token=os.environ.get("HF_TOKEN"))

    if create_if_missing:
        api.create_repo(
            repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )

    commit_message = commit_message or f"Push checkpoint from {local_dir.name}"
    commit_info = api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
    )

    if tag:
        try:
            api.create_tag(
                repo_id=repo_id,
                tag=tag,
                repo_type=repo_type,
                tag_message=f"Tagged {tag} at {commit_info.oid[:8]}",
            )
        except Exception as exc:
            print(f"  [hub] tag {tag!r} failed (non-fatal): {exc}")

    return commit_info.commit_url


def push_stage_checkpoint(
    local_dir: Path,
    stage: StageKey,
    *,
    step: int | None = None,
    is_final: bool = False,
    extra_metrics: dict | None = None,
) -> str:
    """Push a stage checkpoint with conventional naming + auto-tagging.

    `step` adds a step-stamped tag (e.g. "step-886"). `is_final` adds a
    "final" tag (use for the locked checkpoint at end of stage). Both can
    coexist.

    `extra_metrics` is written into a `phychip_metrics.json` file in the
    upload folder so the model card has structured metric data attached
    to the same commit.
    """
    repo_id = STAGE_REPOS[stage]

    if extra_metrics:
        metrics_path = local_dir / "phychip_metrics.json"
        payload = dict(extra_metrics)
        payload["stage"] = stage
        if step is not None:
            payload["step"] = step
        if is_final:
            payload["is_final"] = True
        metrics_path.write_text(json.dumps(payload, indent=2))

    parts = []
    if step is not None:
        parts.append(f"step-{step}")
    if is_final:
        parts.append("final")
    suffix = " (" + ", ".join(parts) + ")" if parts else ""
    commit_message = f"{stage}{suffix}"

    primary_tag = "final" if is_final else (f"step-{step}" if step is not None else None)

    return push_checkpoint(
        local_dir=local_dir,
        repo_id=repo_id,
        private=True,
        commit_message=commit_message,
        tag=primary_tag,
        create_if_missing=True,
        repo_type="model",
    )


def assert_hf_token_present() -> None:
    """Hard-fail early if HF_TOKEN is missing — better than 5 hours into
    a CPT run discovering you can't push the checkpoint.
    """
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN env var is not set. Source .env or export it before "
            "starting a training run, or pass --no-push to skip the upload."
        )
