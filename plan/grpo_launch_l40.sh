#!/bin/bash
# Background orchestrator: wait for pod SSH -> rsync -> launch bounded GRPO.
# Provisioned pod id passed as $1. Bounded: 1 seed, PHYCHIP_STEPS steps.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
POD="${1:?usage: grpo_launch_l40.sh <pod_id>}"
STEPS="${PHYCHIP_STEPS:-250}"
KEY=~/.ssh/prime_intellect_rsa
set -a; source .env 2>/dev/null; set +a
export PRIME_API_KEY="${PI_API_KEY:-$PRIME_API_KEY}" PRIME_DISABLE_VERSION_CHECK=1
PRIME=~/.local/bin/prime
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"

echo "[$(date -u +%H:%M:%S)] waiting for pod $POD SSH..."
TARGET=""
for i in $(seq 1 60); do
  line=$($PRIME pods status "$POD" --plain 2>/dev/null | grep -iE '^SSH' )
  conn=$(echo "$line" | sed -E 's/^SSH[[:space:]]+//; s/[[:space:]]*$//')
  if echo "$conn" | grep -qE '@'; then
    # conn like "ssh user@ip -p PORT" or "user@ip"
    TARGET="$conn"; echo "[$(date -u +%H:%M:%S)] SSH ready: $conn"; break
  fi
  sleep 20
done
[ -z "$TARGET" ] && { echo "FATAL: SSH never came up"; exit 1; }

# Normalize: extract user@ip and optional port
USERIP=$(echo "$TARGET" | grep -oE '[a-z0-9_]+@[0-9.]+' | head -1)
PORT=$(echo "$TARGET" | grep -oE '\-p[[:space:]]*[0-9]+' | grep -oE '[0-9]+' | head -1)
PORT="${PORT:-22}"
SSHP="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -p $PORT"
RS="rsync -az -e \"$SSHP\""
echo "[$(date -u +%H:%M:%S)] target=$USERIP port=$PORT"

# wait for sshd to actually accept
for i in $(seq 1 30); do
  $SSHP "$USERIP" 'echo up' 2>/dev/null | grep -q up && break; sleep 10
done

echo "[$(date -u +%H:%M:%S)] mk tree + rsync"
$SSHP "$USERIP" 'mkdir -p ~/phy-chip/athma-train ~/phy-chip/eval_sets/bench_v1 ~/phy-chip/eval_sets/phychip_bench_v2 ~/phy-chip/data/rlvr_specs_v2_verified ~/phy-chip/checkpoints ~/phy-chip/eval_results'
eval $RS --exclude __pycache__ --exclude '*.pyc' athma-train/athma_train athma-train/scripts "$USERIP":'~/phy-chip/athma-train/'
eval $RS data/rlvr_specs_v2_verified/specs.jsonl "$USERIP":'~/phy-chip/data/rlvr_specs_v2_verified/'
eval $RS eval_sets/bench_v1/bench_v1.jsonl "$USERIP":'~/phy-chip/eval_sets/bench_v1/'
eval $RS eval_sets/phychip_bench_v2/bench_v2.jsonl "$USERIP":'~/phy-chip/eval_sets/phychip_bench_v2/'
eval $RS checkpoints/sft_masked "$USERIP":'~/phy-chip/checkpoints/'
eval $RS plan/grpo_pod_run.sh "$USERIP":'~/phy-chip/'

COMPLETION="${PHYCHIP_COMPLETION:-2048}"
echo "[$(date -u +%H:%M:%S)] launch GRPO: 1 seed, STEPS=$STEPS COMPLETION=$COMPLETION"
$SSHP "$USERIP" "cd ~/phy-chip && chmod +x grpo_pod_run.sh && PHYCHIP_SEEDS=42 PHYCHIP_STEPS=$STEPS PHYCHIP_BS=8 PHYCHIP_K=8 PHYCHIP_COMPLETION=$COMPLETION setsid bash grpo_pod_run.sh >grpo_run.log 2>&1 </dev/null & disown; sleep 2; echo launched"
echo "$USERIP|$PORT" > /tmp/grpo_target.txt
echo "[$(date -u +%H:%M:%S)] LAUNCHED. target saved to /tmp/grpo_target.txt"
