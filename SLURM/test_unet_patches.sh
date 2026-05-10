#!/bin/bash
#SBATCH --job-name=test_unet_patches
#SBATCH --output=logs/test_unet_patches-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=04:00:00

set -euo pipefail

# Per-patch UNet evaluation on materialized trees from SLURM/unet_patch_masks_from_yolo.sh
# ($SCRATCH/GrainSeg/dataset/test/unet_from_yolo/<VARIANT>/images|masks).
#
# Pattern: copy patch images/masks + model to $TMPDIR, run evaluate.py there, then copy
# metrics.json and preds/ back to $SCRATCH.
#
# True per-patch tiles: use stride == patch size (default 1024) so each patch file is one
# non-overlapping sliding window (matches typical 1024 YOLO crops).
#
# Override examples:
#   sbatch --export=ALL,VARIANT=PPLPPXblend SLURM/test_unet_patches.sh
#   sbatch --export=ALL,VARIANT=PPL,MODEL_PATH=/path/to/model.keras,PATCH_SIZE=1024 SLURM/test_unet_patches.sh

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

VARIANT="${VARIANT:-PPL}"
TF_WHEEL_NAME="tensorflow-2.17.0+nv25.2-cp312-cp312-linux_x86_64.whl"
JOB_TAG="${SLURM_JOB_ID:-local}"

PATCH_SIZE="${PATCH_SIZE:-1024}"
# One full window per patch file; keep STRIDE equal to PATCH_SIZE unless you intentionally overlap.
STRIDE="${STRIDE:-$PATCH_SIZE}"
BATCH_SIZE="${BATCH_SIZE:-1}"

TEST_ROOT="$SCRATCH/GrainSeg/dataset/test"
UNET_SRC_ROOT="$TEST_ROOT/unet_from_yolo/$VARIANT"
UNET_SRC_IMAGES="$UNET_SRC_ROOT/images"
UNET_SRC_MASKS="$UNET_SRC_ROOT/masks"

OUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/GrainSeg/eval/unet_patches/$VARIANT/$JOB_TAG}"
OUTPUT_JSON="$OUT_ROOT/metrics.json"
PRED_DIR="$OUT_ROOT/preds"

function require_dir {
    local path="$1"
    local message="$2"
    if [ ! -d "$path" ]; then
        echo "$message: $path" >&2
        exit 1
    fi
}

function require_file {
    local path="$1"
    local message="$2"
    if [ ! -f "$path" ]; then
        echo "$message: $path" >&2
        exit 1
    fi
}

case "$VARIANT" in
    PPL)
        IMAGE_SUFFIXES=("_PPL")
        DEFAULT_MODEL_BASENAME="unet_PPL.keras"
        ;;
    PPLPPXblend)
        IMAGE_SUFFIXES=("_PPLPPXblend")
        DEFAULT_MODEL_BASENAME="unet_PPLPPXblend.keras"
        ;;
    PPL+PPXblend|PPL+AllPPX)
        echo "VARIANT=$VARIANT needs multi-input UNet; use PPL or PPLPPXblend, or extend this script." >&2
        exit 1
        ;;
    *)
        echo "Unknown VARIANT for UNet patch eval: $VARIANT" >&2
        exit 1
        ;;
esac

MODEL_PATH="${MODEL_PATH:-$SCRATCH/GrainSeg/models/$DEFAULT_MODEL_BASENAME}"

require_dir "$UNET_SRC_IMAGES" "Patch image directory not found (run unet_patch_masks_from_yolo.sh?)"
require_dir "$UNET_SRC_MASKS" "Patch mask directory not found (run unet_patch_masks_from_yolo.sh?)"
require_file "$MODEL_PATH" "Model not found"

source SLURM/prepare_env.sh
export TF_CPP_MIN_LOG_LEVEL=2

WORK_ROOT="$TMPDIR/unet_patch_eval_${VARIANT}_$JOB_TAG"
LOCAL_IMAGES="$WORK_ROOT/images"
LOCAL_MASKS="$WORK_ROOT/masks"
LOCAL_MODEL_DIR="$WORK_ROOT/model"
TMP_OUTPUT_JSON="$WORK_ROOT/metrics.json"
TMP_PRED_DIR="$WORK_ROOT/preds"

rm -rf "$WORK_ROOT"
mkdir -p "$LOCAL_IMAGES" "$LOCAL_MASKS" "$LOCAL_MODEL_DIR" "$TMP_PRED_DIR"

echo "Staging UNet patch images, masks, and model to TMPDIR ($WORK_ROOT)..."
cp -r "$UNET_SRC_IMAGES"/. "$LOCAL_IMAGES"/
cp -r "$UNET_SRC_MASKS"/. "$LOCAL_MASKS"/
LOCAL_MODEL_PATH="$LOCAL_MODEL_DIR/$(basename "$MODEL_PATH")"
cp -f "$MODEL_PATH" "$LOCAL_MODEL_PATH"

cd "$REPO_ROOT/src/training"
echo "Syncing evaluation environment..."
uv sync

WHEEL_PATH="$SCRATCH/GrainSeg/wheels/$TF_WHEEL_NAME"
require_file "$WHEEL_PATH" "TensorFlow wheel not found"
echo "Installing TensorFlow wheel..."
uv pip install nvidia-cudnn-cu12~=9.0 nvidia-nccl-cu12 nvidia-cuda-runtime-cu12~=12.8.0 nvidia-cusparse-cu12 nvidia-cufft-cu12 nvidia-cusolver-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 "$WHEEL_PATH"

echo "Running evaluate.py on patch directories (TMPDIR)..."
eval_cmd=(
    uv run --no-sync python -u ../evaluation/evaluate.py
    --model-type unet
    --variant "$VARIANT"
    --model-path "$LOCAL_MODEL_PATH"
    --image-dir "$LOCAL_IMAGES"
    --mask-dir "$LOCAL_MASKS"
    --output-json "$TMP_OUTPUT_JSON"
    --save-predictions-dir "$TMP_PRED_DIR"
    --num-inputs "1"
    --image-suffixes
    "${IMAGE_SUFFIXES[@]}"
    --patch-size "$PATCH_SIZE"
    --stride "$STRIDE"
    --batch-size "$BATCH_SIZE"
    --mask-ext ".tif"
    --mask-stem-suffix "_labels"
)

"${eval_cmd[@]}"

echo "Copying metrics and predictions to $OUT_ROOT..."
mkdir -p "$OUT_ROOT" "$PRED_DIR"
cp -f "$TMP_OUTPUT_JSON" "$OUTPUT_JSON"
cp -r "$TMP_PRED_DIR"/. "$PRED_DIR"/

echo "Wrote $OUTPUT_JSON"
