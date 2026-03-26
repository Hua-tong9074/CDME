#!/usr/bin/env bash
# AutoDL one-click runner for BMD (SOURCE training stage)
# Usage:
#   bash run_source_autodl.sh [GPU_ID] [DATASET] [SOURCE_IDX]
# Example:
#   bash run_source_autodl.sh 0 VisDA 0
#   bash run_source_autodl.sh 0 OfficeHome 0
#   bash run_source_autodl.sh 0 Office 0

set -euo pipefail

GPU_ID="${1:-0}"
DATASET="${2:-VisDA}"
S_IDX="${3:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WANDB_MODE=offline

REPO_DIR="$(pwd)"

# 选择数据路径
if [ "$DATASET" == "VisDA" ]; then
    DATA_DIR="${REPO_DIR}/data/VisDA"
    BACKBONE="resnet101"
elif [ "$DATASET" == "OfficeHome" ]; then
    DATA_DIR="${REPO_DIR}/data/OfficeHome"
    BACKBONE="resnet50"
elif [ "$DATASET" == "Office" ]; then
    DATA_DIR="${REPO_DIR}/data/Office"
    BACKBONE="resnet50"
else
    echo "[ERR] Unknown dataset: ${DATASET}"
    exit 1
fi

OUT_DIR="${REPO_DIR}/checkpoints_sfda/${DATASET}_source_${BACKBONE}_s${S_IDX}"
mkdir -p "${OUT_DIR}"

echo "[INFO] Dataset=${DATASET}"
echo "[INFO] Using DATA_DIR=${DATA_DIR}"
echo "[INFO] Using OUT_DIR=${OUT_DIR}"

python -u main_source.py \
  --dataset ${DATASET} \
  --backbone_arch ${BACKBONE} \
  --s_idx ${S_IDX} \
  --lr 0.001 \
  --epochs 10 \
  --num_workers 8 \
  --without_wandb \
  --note autodl_source \
  --seed 2021

echo "[OK] ${DATASET} source training finished. Checkpoints saved to ${OUT_DIR}"
