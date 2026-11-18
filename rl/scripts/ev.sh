#!/usr/bin/env bash
# ev.sh NAME MODEL_DIR TASKS [NOTE]: vLLM sample K=8 (T=0.8) + greedy, score both, append to the ledger
set -uo pipefail
NAME=$1; MODEL=$2; TASKS=$3; NOTE=${4:-}
OUT=./runs/eval/$NAME; mkdir -p $OUT; cd .
if ! python vgen.py --model $MODEL --tasks $TASKS --out $OUT/k8.jsonl --k 8 --also-greedy > $OUT/vgen.log 2>&1; then
  echo "vLLM failed for $NAME; falling back to transformers generate"
  python gen.py --base $MODEL --tasks $TASKS --out $OUT/k8.greedy.jsonl --greedy --bs 46 > $OUT/hfgen_greedy.log 2>&1 || { echo "FAIL gen $NAME"; exit 1; }
  python gen.py --base $MODEL --tasks $TASKS --out $OUT/k8.jsonl --k 8 --temperature 0.8 --bs 92 > $OUT/hfgen_k8.log 2>&1 || { echo "FAIL gen $NAME"; exit 1; }
fi
python score.py --tasks $TASKS --gen $OUT/k8.greedy.jsonl --out $OUT/greedy_scored.jsonl --procs ${PROCS:-8} --scorer v12 --name "$NAME/greedy" --ledger ./runs/ledger.jsonl --note "$NOTE" > $OUT/score_greedy.log 2>&1
python score.py --tasks $TASKS --gen $OUT/k8.jsonl --out $OUT/k8_scored.jsonl --procs ${PROCS:-8} --scorer v12 --name "$NAME/k8_T0.8" --ledger ./runs/ledger.jsonl --note "$NOTE" > $OUT/score_k8.log 2>&1
python - <<P
import json
g=json.load(open("$OUT/greedy_scored.summary.json")); k=json.load(open("$OUT/k8_scored.summary.json"))
print("RESULT $NAME greedy %d/%d | k8 rate %.3f any %d/%d | sim %.2f blocked %.2f unfinished %.2f" % (g["pass_first"],g["tasks"],k["mean_pass_rate"],k["pass_any"],k["tasks"],k["simulated"],k["anticheat_blocked"],k["unfinished"]))
P
