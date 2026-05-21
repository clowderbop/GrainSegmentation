#!/bin/bash
#SBATCH --job-name=test_unet_patches
#SBATCH --output=logs/test_unet_patches-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=04:00:00

# Patch-wise U-Net test evaluation for one input variant (VARIANT).
# Staged pipeline: predict -> extract_instances -> evaluate_instances.
# Semantic metrics (evaluate_semantic) are skipped when test patches have no raster masks.
# Optional env: MODEL_PATH, WATERSHED_JSON, WATERSHED_TUNE_ROOT, OUTPUT_ROOT, GT_GPKG,
# PATCH_SIZE, STRIDE, BATCH_SIZE, TEST_ROOT, MASK_DIR (enables semantic metrics).

set -euo pipefail
# shellcheck source=SLURM/utils/source_job.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/source_job.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$SLURM_ROOT/utils/variants.sh"
# shellcheck source=SLURM/utils/watershed.sh
source "$SLURM_ROOT/utils/watershed.sh"
# shellcheck source=SLURM/utils/tensorflow.sh
source "$SLURM_ROOT/utils/tensorflow.sh"

GRAINSEG_ROOT="$(grainseg_root)"

VARIANT="${VARIANT:-PPL}"
JOB_TAG="${SLURM_JOB_ID:-local}"

PATCH_SIZE="${PATCH_SIZE:-1024}"
STRIDE="${STRIDE:-$PATCH_SIZE}"
BATCH_SIZE="${BATCH_SIZE:-1}"

TEST_ROOT="${TEST_ROOT:-$GRAINSEG_ROOT/dataset/test}"
UNET_SRC_ROOT="$TEST_ROOT/unet_from_yolo/$VARIANT"
UNET_SRC_IMAGES="$UNET_SRC_ROOT/images"
GT_GPKG="${GT_GPKG:-$TEST_ROOT/test_labels.gpkg}"
MASK_DIR="${MASK_DIR:-}"

OUT_ROOT="${OUTPUT_ROOT:-$GRAINSEG_ROOT/eval/unet_patches/$VARIANT/$JOB_TAG}"
INSTANCE_METRICS_JSON="$OUT_ROOT/instance_metrics.json"

unet_patch_config_for_variant "$VARIANT"

MODEL_PATH="${MODEL_PATH:-$GRAINSEG_ROOT/models/unet/$DEFAULT_MODEL_BASENAME}"
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
TMP_OUT="$WORK_ROOT/out"

rm -rf "$WORK_ROOT"
mkdir -p "$LOCAL_IMAGES" "$LOCAL_MODEL_DIR" "$TMP_OUT"

echo "Staging UNet patch images and model to TMPDIR ($WORK_ROOT)..."
cp -r "$UNET_SRC_IMAGES"/. "$LOCAL_IMAGES"/
cp -f "$GT_GPKG" "$LOCAL_GT_GPKG"
LOCAL_MODEL_PATH="$LOCAL_MODEL_DIR/$(basename "$MODEL_PATH")"
cp -f "$MODEL_PATH" "$LOCAL_MODEL_PATH"

cd "$REPO_ROOT/src/unet"
echo "Syncing U-Net environment..."
uv sync

install_unet_tensorflow_wheel

WATERSHED_TUNE_ROOT="${WATERSHED_TUNE_ROOT:-$GRAINSEG_ROOT/runs/watershed_tune}"
RESOLVED_WATERSHED_JSON="$(resolve_watershed_json_lenient "$MODEL_DIR" "${WATERSHED_JSON:-}" "$LOCAL_MODEL_PATH" "$WATERSHED_TUNE_ROOT")"
if [[ -n "$RESOLVED_WATERSHED_JSON" ]]; then
    require_file "$RESOLVED_WATERSHED_JSON" "WATERSHED_JSON not found"
fi
build_watershed_extract_args "$RESOLVED_WATERSHED_JSON"

echo "1/3 unet.predict (semantic TIFFs)..."
predict_cmd=(
    uv run --no-sync python -u -m unet.predict
    --variant "$VARIANT"
    --unit patch
    --model-path "$LOCAL_MODEL_PATH"
    --image-dir "$LOCAL_IMAGES"
    --output-dir "$TMP_OUT"
    --num-inputs "$NUM_INPUTS"
    --image-suffixes
    "${IMAGE_SUFFIXES[@]}"
    --patch-size "$PATCH_SIZE"
    --stride "$STRIDE"
    --batch-size "$BATCH_SIZE"
)
"${predict_cmd[@]}"

echo "2/3 unet.extract_instances (instance label-map TIFFs)..."
extract_cmd=(
    uv run --no-sync python -u -m unet.extract_instances
    --semantic-dir "$TMP_OUT/semantic"
    --output-dir "$TMP_OUT"
    "${extract_args[@]}"
)
"${extract_cmd[@]}"

if [[ -n "$MASK_DIR" && -d "$MASK_DIR" ]]; then
    echo "Optional: unet.evaluate_semantic (raster GT masks provided)..."
    semantic_cmd=(
        uv run --no-sync python -u -m unet.evaluate_semantic
        --semantic-dir "$TMP_OUT/semantic"
        --mask-dir "$MASK_DIR"
        --output-json "$TMP_OUT/semantic_metrics.json"
        --variant "$VARIANT"
        --unit patch
    )
    "${semantic_cmd[@]}"
else
    echo "Skipping unet.evaluate_semantic (no MASK_DIR; patch test set often lacks raster masks)."
fi

echo "3/3 common.evaluate_instances..."
instance_cmd=(
    uv run --no-sync python -u -m common.evaluate_instances
    --variant "$VARIANT"
    --unit patch
    --model-type unet
    --image-dir "$LOCAL_IMAGES"
    --pred-instances-dir "$TMP_OUT/instances"
    --gt-gpkg "$LOCAL_GT_GPKG"
    --gt-origin patch_stem
    --image-stem-suffix "${IMAGE_SUFFIXES[0]}"
    --output-json "$TMP_OUT/instance_metrics.json"
)
"${instance_cmd[@]}"

echo "Copying artifacts to $OUT_ROOT..."
mkdir -p "$OUT_ROOT"
cp -r "$TMP_OUT"/. "$OUT_ROOT"/
echo "Wrote $INSTANCE_METRICS_JSON"
