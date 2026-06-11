#!/bin/bash

# Usage: bash run_stage1.sh [CUDA_ID] [BATCH_SIZE]
# Example: bash run_stage1.sh 1 2

CUDA_ID=${1:-1}
BATCH_SIZE=${2:-2}

export CUDA_VISIBLE_DEVICES=$CUDA_ID
export TRAIN_BATCH_SIZE=$BATCH_SIZE

echo "CUDA_VISIBLE_DEVICES=$CUDA_ID"
echo "TRAIN_BATCH_SIZE=$BATCH_SIZE"

cd "./qwen"
/data/chl343/miniconda3/envs/vlmped/bin/python fine_tuning_COT-joint_stage_1.py
