#!/usr/bin/env bash
# arm.sh GPU NAME BASE DATA EPOCHS LR NOTE [extra train args]: train -> merge -> eval (v12) on one GPU
GPU=$1; NAME=$2; BASE=$3; DATA=$4; EP=$5; LR=$6; NOTE=$7; shift 7
export CUDA_VISIBLE_DEVICES=$GPU
cd .; mkdir -p ./runs/ckpt ./runs/logs ./merged
TPL=(); [[ "$BASE" == *SmolLM3* ]] && TPL=(--chat-template /w/hf/models/PhyChip-SmolLM3-3B-base-SFT/chat_template.jinja)
python train_sft.py --base $BASE --data $DATA --out ./runs/ckpt/$NAME --epochs $EP --lr $LR --bs 4 --grad-accum 2 "${TPL[@]}" "$@" > ./runs/logs/train_$NAME.log 2>&1 || { echo "FAIL train $NAME"; exit 1; }
python merge.py $BASE ./runs/ckpt/$NAME ./merged/$NAME > ./runs/logs/merge_$NAME.log 2>&1 || { echo "FAIL merge $NAME"; exit 1; }
bash ev.sh $NAME ./merged/$NAME tasks_dev46_grpo.jsonl "$NOTE"
