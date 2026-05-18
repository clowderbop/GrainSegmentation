#!/bin/bash
#SBATCH --job-name=test_unet_patches
#SBATCH --output=logs/test_unet_patches-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=04:00:00

# Patch-wise U-Net test evaluation for one input variant (VARIANT).
# Optional env: MODEL_PATH, WATERSHED_JSON, WATERSHED_TUNE_ROOT, OUTPUT_ROOT, GT_GPKG,
# PATCH_SIZE, STRIDE, BATCH_SIZE, TEST_ROOT.

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

TEST_ROOT="${TEST_ROOT:-$SCRATCH/GrainSeg/dataset/test}"
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

function infer_watershed_tune_subdir_from_stem {
    local model_stem="$1"

    if [[ "$model_stem" == *"PPL+AllPPX"* ]]; then
        printf '%s\n' "PPL_AllPPX"
        return 0
    fi
    if [[ "$model_stem" == *"PPL+PPXblend"* ]]; then
        printf '%s\n' "PPL_PlusPPXblend"
        return 0
    fi
    if [[ "$model_stem" == *"PPLPPXblend"* ]]; then
        printf '%s\n' "PPLPPXblend"
        return 0
    fi
    if [[ "$model_stem" == *"PPL"* ]]; then
        printf '%s\n' "PPL"
        return 0
    fi

    return 1
}

function pick_latest_watershed_best_json {
    local dir="$1"
    shopt -s nullglob
    local matches=("$dir"/watershed_best_*.json)
    shopt -u nullglob

    if [ "${#matches[@]}" -eq 0 ]; then
        echo "No watershed_best_*.json files in: $dir" >&2
        return 1
    fi

    local newest=""
    local newest_mtime=0
    for f in "${matches[@]}"; do
        local m
        m="$(stat -c '%Y' "$f" 2>/dev/null || stat -f '%m' "$f")"
        if [ "$m" -gt "$newest_mtime" ]; then
            newest_mtime="$m"
            newest="$f"
        fi
    done
    printf '%s\n' "$newest"
}

NUM_INPUTS=0
IMAGE_SUFFIXES=()
DEFAULT_MODEL_BASENAME=""

case "$VARIANT" in
    PPL)
        NUM_INPUTS=1
        IMAGE_SUFFIXES=("_PPL")
        DEFAULT_MODEL_BASENAME="unet_finetuned_PPL.keras"
        ;;
    PPLPPXblend)
        NUM_INPUTS=1
        IMAGE_SUFFIXES=("_PPLPPXblend")
        DEFAULT_MODEL_BASENAME="unet_finetuned_PPLPPXblend.keras"
        ;;
    PPL+PPXblend)
        NUM_INPUTS=2
        IMAGE_SUFFIXES=("_PPL" "_PPXblend")
        DEFAULT_MODEL_BASENAME="unet_finetuned_PPL+PPXblend.keras"
        ;;
    PPL+AllPPX)
        NUM_INPUTS=7
        IMAGE_SUFFIXES=("_PPL" "_PPX1" "_PPX2" "_PPX3" "_PPX4" "_PPX5" "_PPX6")
        DEFAULT_MODEL_BASENAME="unet_finetuned_PPL+AllPPX.keras"
        ;;
    *)
        echo "Unknown VARIANT for UNet patch eval: $VARIANT" >&2
        echo "Expected one of: PPL, PPLPPXblend, PPL+PPXblend, PPL+AllPPX" >&2
        exit 1
        ;;
esac

MODEL_PATH="${MODEL_PATH:-$SCRATCH/GrainSeg/models/$DEFAULT_MODEL_BASENAME}"
MODEL_DIR="$(dirname "$MODEL_PATH")"

require_dir "$UNET_SRC_IMAGES" "Patch image directory not found (prepare test patches under $UNET_SRC_ROOT/images)"
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

WATERSHED_TUNE_ROOT="${WATERSHED_TUNE_ROOT:-$SCRATCH/GrainSeg/runs/watershed_tune}"
RESOLVED_WATERSHED_JSON=""
if [[ -n "${WATERSHED_JSON:-}" ]]; then
    if [[ "$WATERSHED_JSON" = /* ]]; then
        RESOLVED_WATERSHED_JSON="$WATERSHED_JSON"
    else
        RESOLVED_WATERSHED_JSON="$MODEL_DIR/$WATERSHED_JSON"
    fi
    require_file "$RESOLVED_WATERSHED_JSON" "WATERSHED_JSON not found"
elif [[ -n "$WATERSHED_TUNE_ROOT" ]]; then
    model_stem="$(basename "$LOCAL_MODEL_PATH" .keras)"
    if subdir="$(infer_watershed_tune_subdir_from_stem "$model_stem")"; then
        variant_tune_dir="$WATERSHED_TUNE_ROOT/$subdir"
        if [[ ! -d "$variant_tune_dir" ]]; then
            echo "Note: watershed tune directory not found: $variant_tune_dir; using default watershed args." >&2
        elif picked="$(pick_latest_watershed_best_json "$variant_tune_dir" 2>/dev/null)"; then
            RESOLVED_WATERSHED_JSON="$picked"
        else
            echo "Note: no watershed_best_*.json under $variant_tune_dir; using default watershed args." >&2
        fi
    else
        echo "Note: cannot map model stem '$model_stem' to a watershed tune subdir; using default watershed args." >&2
    fi
fi

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
    --num-inputs "$NUM_INPUTS"
    --image-suffixes
    "${IMAGE_SUFFIXES[@]}"
    --patch-size "$PATCH_SIZE"
    --stride "$STRIDE"
    --batch-size "$BATCH_SIZE"
)

WATERSHED_JSON_HELPER="$REPO_ROOT/src/evaluation/watershed_json_to_eval_args.py"
if [[ -n "$RESOLVED_WATERSHED_JSON" ]]; then
    mapfile -t _watershed_eval_args < <(python3 "$WATERSHED_JSON_HELPER" "$RESOLVED_WATERSHED_JSON")
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
