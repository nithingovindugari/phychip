#!/bin/bash
# Watchdog: poll pod GRPO log; on ALL_GRPO_DONE -> pull results + terminate pod.
# Hard deadline guards the $30 budget cap. Runs locally in background.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
POD="${1:-5de394c87df242ff9ebb13d33847022e}"
IP="${2:-ubuntu@216.81.245.39}"
KEY=$HOME/.ssh/prime_intellect_rsa
DEADLINE_HRS="${DEADLINE_HRS:-8}"   # hard budget guard; fires on ALL_GRPO_DONE first. L40 @ $0.86/hr
set -a; source .env 2>/dev/null; set +a
export PRIME_API_KEY="${PI_API_KEY:-$PRIME_API_KEY}" PRIME_DISABLE_VERSION_CHECK=1
PRIME=~/.local/bin/prime
sshc(){ ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 "$IP" "$@"; }

START=$(date +%s)
while true; do
  sleep 180
  now=$(date +%s); elapsed=$(( (now-START)/60 ))
  log=$(sshc 'tail -3 ~/phy-chip/grpo_run.log 2>/dev/null' 2>/dev/null)
  done=$(echo "$log" | grep -c ALL_GRPO_DONE)
  fatal=$(echo "$log" | grep -ciE 'FATAL|REWARD GATE FAILED|Traceback')
  echo "[$(date -u +%H:%M:%S)] +${elapsed}m done=$done fatal=$fatal"
  if [ "$done" -ge 1 ] || [ "$elapsed" -ge $((DEADLINE_HRS*60)) ]; then
    echo "=== pulling results ==="
    mkdir -p eval_results checkpoints
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" "$IP":'~/phy-chip/eval_results/grpo_ood_*' eval_results/ 2>/dev/null || true
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" "$IP":'~/phy-chip/eval_results/masked_sft_*' eval_results/ 2>/dev/null || true
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --exclude 'checkpoint-*/optimizer.pt' "$IP":'~/phy-chip/checkpoints/grpo_ood_*' checkpoints/ 2>/dev/null || true
    sshc 'tail -60 ~/phy-chip/grpo_run.log' 2>/dev/null > eval_results/GRPO_RUN_TAIL.txt
    echo "=== terminating pod $POD ==="
    yes | $PRIME pods terminate "$POD" 2>&1 | tail -2
    echo "WATCHDOG_DONE elapsed=${elapsed}m reason=$([ "$done" -ge 1 ] && echo done || echo deadline)"
    break
  fi
done
