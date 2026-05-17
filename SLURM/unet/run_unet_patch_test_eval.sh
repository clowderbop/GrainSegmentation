#!/bin/bash
#SBATCH --job-name=test_unet_patches
#SBATCH --output=logs/test_unet_patches-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=04:00:00

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_ROOT="$(cd "$THIS_DIR/.." && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SLURM_ROOT/.." && pwd)}"
cd "$REPO_ROOT"

VARIANT="${VARIANT:-PPL}"
TF_WHEEL_NAME="tensorflow-2.17.0+nv25.2-cp312-cp312-linux_x86_64.whl"
JOB_TAG="${SLURM_JOB_ID:-local}"

PATCH_SIZE="${PATCH_SIZE:-1024}"

STRIDE="${STRIDE:-$PATCH_SIZE}"
BATCH_SIZE="${BATCH_SIZE:-1}"

TEST_ROOT="$SCRATCH/GrainSeg/dataset/test"
UNET_SRC_ROOT="$TEST_ROOT/unet_from_yolo/$VARIANT"
UNET_SRC_IMAGES="$UNET_SRC_ROOT/images"
GT_GPKG="${GT_GPKG:-$TEST_ROOT/test_labels.gpkg}"

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

require_dir "$UNET_SRC_IMAGES" "Patch image directory not found (run 08_create_unet_test_patches_from_yolo_patches.sh?)"
require_file "$MODEL_PATH" "Model not found"
require_file "$GT_GPKG" "Ground-truth GeoPackage not found"

source "$SLURM_ROOT/prepare_env.sh"
export TF_CPP_MIN_LOG_LEVEL=2

WORK_ROOT="$TMPDIR/unet_patch_eval_${VARIANT}_$JOB_TAG"
LOCAL_IMAGES="$WORK_ROOT/images"
LOCAL_MODEL_DIR="$WORK_ROOT/model"
LOCAL_GT_GPKG="$WORK_ROOT/$(basename "$GT_GPKG")"
TMP_OUTPUT_JSON="$WORK_ROOT/metrics.json"
TMP_PRED_DIR="$WORK_ROOT/preds"

rm -rf "$WORK_ROOT"
mkdir -p "$LOCAL_IMAGES" "$LOCAL_MODEL_DIR" "$TMP_PRED_DIR"

echo "Staging UNet patch images and model to TMPDIR ($WORK_ROOT)..."
cp -r "$UNET_SRC_IMAGES"/. "$LOCAL_IMAGES"/
cp -f "$GT_GPKG" "$LOCAL_GT_GPKG"
LOCAL_MODEL_PATH="$LOCAL_MODEL_DIR/$(basename "$MODEL_PATH")"
cp -f "$MODEL_PATH" "$LOCAL_MODEL_PATH"

cd "$REPO_ROOT/src/evaluation"
echo "Syncing evaluation environment..."
uv sync --extra unet

WHEEL_PATH="$SCRATCH/GrainSeg/wheels/$TF_WHEEL_NAME"
require_file "$WHEEL_PATH" "TensorFlow wheel not found"
echo "Installing TensorFlow wheel..."
uv pip install nvidia-cudnn-cu12~=9.0 nvidia-nccl-cu12 nvidia-cuda-runtime-cu12~=12.8.0 nvidia-cusparse-cu12 nvidia-cufft-cu12 nvidia-cusolver-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 "$WHEEL_PATH"

echo "Running evaluate.py on patch directories (TMPDIR)..."
eval_cmd=(
    uv run --no-sync python -u -m evaluation.evaluate
    --model-type unet
    --variant "$VARIANT"
    --model-path "$LOCAL_MODEL_PATH"
    --image-dir "$LOCAL_IMAGES"
    --gt-gpkg "$LOCAL_GT_GPKG"
    --output-json "$TMP_OUTPUT_JSON"
    --save-predictions-dir "$TMP_PRED_DIR"
    --num-inputs "1"
    --image-suffixes
    "${IMAGE_SUFFIXES[@]}"
    --patch-size "$PATCH_SIZE"
    --stride "$STRIDE"
    --batch-size "$BATCH_SIZE"
)

WATERSHED_JSON_HELPER="$REPO_ROOT/src/evaluation/watershed_json_to_eval_args.py"
if [[ -n "${WATERSHED_JSON:-}" ]]; then
    require_file "$WATERSHED_JSON" "WATERSHED_JSON not found"
    mapfile -t _watershed_eval_args < <(python3 "$WATERSHED_JSON_HELPER" "$WATERSHED_JSON")
    eval_cmd+=("${_watershed_eval_args[@]}")
else
    eval_cmd+=(--instance-method watershed)
fi

"${eval_cmd[@]}"

echo "Copying metrics and predictions to $OUT_ROOT..."
mkdir -p "$OUT_ROOT" "$PRED_DIR"
cp -f "$TMP_OUTPUT_JSON" "$OUTPUT_JSON"
cp -r "$TMP_PRED_DIR"/. "$PRED_DIR"/

echo "Wrote $OUTPUT_JSON"
