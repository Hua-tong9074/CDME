#!/bin/bash
# Usage: sh train_target.sh <dataset> <source_idx> <target_idx>
# Example:
#   sh train_target.sh VisDA 0 1
#   sh train_target.sh OfficeHome 0 1
#   sh train_target.sh Office 0 1

DATASET=${1:-VisDA}
S_IDX=${2:-0}
T_IDX=${3:-1}

# 自动设置骨干网络
if [ "$DATASET" == "VisDA" ]; then
    BACKBONE="resnet101"
else
    BACKBONE="resnet50"
fi

# 源模型路径
CHECKPOINT=./checkpoints_sfda/${DATASET}/source_${S_IDX}/source_checkpoint_smooth_source_seed_2021/${DATASET}_latest_source_checkpoint.pth

echo "Start target model adaptation on the ${DATASET} Dataset (source ${S_IDX} → target ${T_IDX})"

python main_target.py \
  --dataset ${DATASET} \
  --backbone_arch ${BACKBONE} \
  --lr 0.001 \
  --without_wandb \
  --checkpoint ${CHECKPOINT} \
  --note smooth_source \
  --num_workers 8 \
  --s_idx ${S_IDX} \
  --t_idx ${T_IDX} \
  --seed 2021
