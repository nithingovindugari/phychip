#!/bin/bash
# LOCAL orchestrator: one trigger to resume the GRPO-OOD plan when usage is back.
# Usage:  bash plan/resume_grpo.sh <POD_USER@POD_IP>
#   e.g.  bash plan/resume_grpo.sh ubuntu@1.2.3.4
# Provision a fresh A100 first (prefer US massedcompute, lesson #147):
#   prime availability list --gpu-type A100_80GB --gpu-count 1
#   prime pods create --id <ID> --name phychip-grpo --disk-size 120 --image ubuntu_22_cuda_12
# Then run this with the pod's ssh target. It rsyncs everything and launches
# plan/grpo_pod_run.sh under nohup (survives your usage limit / disconnect).
set -uo pipefail
TARGET="${1:?usage: resume_grpo.sh <user@ip>}"
KEY=~/.ssh/prime_intellect_rsa
SSH="ssh -i $KEY -o StrictHostKeyChecking=no"
RS="rsync -az -e \"$SSH\""
cd "$(dirname "$0")/.." || exit 1

echo "=== make tree on pod ==="
$SSH "$TARGET" 'mkdir -p ~/phy-chip/athma-train ~/phy-chip/eval_sets/bench_v1 ~/phy-chip/eval_sets/phychip_bench_v2 ~/phy-chip/data/rlvr_specs_v2_verified ~/phy-chip/checkpoints ~/phy-chip/eval_results'

echo "=== sync code + data + adapter + benches ==="
eval $RS --exclude __pycache__ --exclude '*.pyc' athma-train/athma_train athma-train/scripts "$TARGET":'~/phy-chip/athma-train/'
eval $RS data/rlvr_specs_v2_verified/specs.jsonl "$TARGET":'~/phy-chip/data/rlvr_specs_v2_verified/'
eval $RS eval_sets/bench_v1/bench_v1.jsonl "$TARGET":'~/phy-chip/eval_sets/bench_v1/'
eval $RS eval_sets/phychip_bench_v2/bench_v2.jsonl "$TARGET":'~/phy-chip/eval_sets/phychip_bench_v2/'
eval $RS checkpoints/sft_masked "$TARGET":'~/phy-chip/checkpoints/'
eval $RS plan/grpo_pod_run.sh "$TARGET":'~/phy-chip/'

echo "=== launch GRPO plan under nohup (survives disconnect) ==="
$SSH "$TARGET" 'cd ~/phy-chip && chmod +x grpo_pod_run.sh && nohup bash grpo_pod_run.sh > ~/grpo_run.log 2>&1 & echo launched pid $!'
echo "Done. Tail with:  $SSH $TARGET 'tail -f ~/grpo_run.log'"
echo "Pull results with: rsync -az -e \"$SSH\" $TARGET:'~/phy-chip/eval_results/grpo_ood_*' eval_results/"
echo "REMINDER: terminate the pod when ALL_GRPO_DONE appears (no-idle rule)."
