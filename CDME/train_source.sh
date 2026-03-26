#!/bin/bash
# Usage: sh train_source.sh <dataset> <source_idx>

DATASET=${1:-VisDA}        # 默认为 VisDA，可手动输入 OfficeHome 或 Office
S_IDX=${2:-0}

# 自动选择骨干网络
if [ "$DATASET" == "VisDA" ]; then
    BACKBONE="resnet101"
else
    BACKBONE="resnet50"
fi

echo "Start source model preparing on the ${DATASET} Dataset (source ${S_IDX})"

python main_source.py \
  --dataset ${DATASET} \
  --backbone_arch ${BACKBONE} \
  --lr 0.001 \
  --without_wandb \
  --note smooth_source \
  --s_idx ${S_IDX} \
  --num_workers 8 \
  --seed 2021 \
  --epochs 10
