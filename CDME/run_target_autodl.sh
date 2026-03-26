#!/usr/bin/env bash
# One-click TARGET SFDA runner (works with main_target.py)
# Usage:
#   # 优先使用已有 source ckpt（可选）
#   SRC_CKPT=/path/to/source_xxx.pth bash run_target_autodl.sh [GPU] [DATASET] [S_IDX] [T_IDX]
#   # 若不提供 SRC_CKPT，则按常用保存规则自动推断

set -euo pipefail

GPU_ID="${1:-0}"
DATASET="${2:-VisDA}"       # VisDA | OfficeHome | Office
S_IDX="${3:-0}"
T_IDX="${4:-1}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WANDB_MODE=offline

REPO_DIR="$(pwd)"

# ---- 选择数据根与backbone ----
if [ "$DATASET" = "VisDA" ]; then
  DATA_DIR="${REPO_DIR}/data/VisDA"
  BACKBONE="resnet101"
elif [ "$DATASET" = "OfficeHome" ]; then
  DATA_DIR="${REPO_DIR}/data/OfficeHome"
  BACKBONE="resnet50"
elif [ "$DATASET" = "Office" ]; then
  DATA_DIR="${REPO_DIR}/data/Office"
  BACKBONE="resnet50"
else
  echo "[ERR] Unknown dataset: ${DATASET}"; exit 1
fi

# ---- 推断/使用 source ckpt ----
if [ -z "${SRC_CKPT:-}" ]; then
  # 按你目前的 source 保存规则推断常见路径（必要时改成你的实际路径）
  SRC_CKPT="${REPO_DIR}/checkpoints_sfda/${DATASET}/source_${S_IDX}/source_checkpoint_smooth_source_seed_2021/${DATASET}_latest_source_checkpoint.pth"
fi
[ -f "${SRC_CKPT}" ] || { echo "[ERR] Source checkpoint not found: ${SRC_CKPT}"; exit 2; }

# ---- 输出目录（可选）----
OUT_DIR="${REPO_DIR}/checkpoints_sfda/${DATASET}/s_${S_IDX}_t_${T_IDX}"
mkdir -p "${OUT_DIR}"

echo "[INFO] Dataset=${DATASET}  S->T=${S_IDX}->${T_IDX}"
echo "[INFO] Backbone=${BACKBONE}"
echo "[INFO] DATA_DIR=${DATA_DIR}"
echo "[INFO] SRC_CKPT=${SRC_CKPT}"
echo "[INFO] OUT_DIR=${OUT_DIR}"

# ---- 直接调用与你代码一致的参数 ----
python -u main_target.py \
  --dataset "${DATASET}" \
  --backbone_arch "${BACKBONE}" \
  --s_idx "${S_IDX}" \
  --t_idx "${T_IDX}" \
  --checkpoint "${SRC_CKPT}" \
  --lr 1e-3 \
  --epochs 20 \
  --num_workers 8 \
  --without_wandb \
  --note autodl_target

echo "[OK] Target SFDA finished. Check ${OUT_DIR}"
