#!/bin/bash
#SBATCH --job-name=TuneWatershed
#SBATCH --output=logs/%x-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=04:00:00

set -euo pipefail
# shellcheck source=SLURM/utils/source_job.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/source_job.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$SLURM_ROOT/utils/variants.sh"
# shellcheck source=SLURM/utils/tensorflow.sh
source "$SLURM_ROOT/utils/tensorflow.sh"
mkdir -p "$REPO_ROOT/logs"

GRAINSEG_ROOT="$(grainseg_root)"
DATASET_DIR="${DATASET_DIR:-$GRAINSEG_ROOT/dataset/train}"
GT_GPKG="${GT_GPKG:-$DATASET_DIR/train_labels.gpkg}"
MODEL_PATH="$GRAINSEG_ROOT/models/unet/unet_finetuned_PPL+AllPPX.keras"
OUTPUT_DIR="$GRAINSEG_ROOT/runs/watershed_tune"

PREDS_DIR=""

PATCH_SIZE=1024
STRIDE=512
BATCH_SIZE=1

MIN_DISTANCE=(1 3 5)
BOUNDARY_DILATE_ITER=(0 1)
WATERSHED_CONNECTIVITY=(1 2)
MIN_AREA_PX=(0)
EXCLUDE_BORDER=(0 1)

function usage {
    local status="${1:-1}"
    cat <<EOF >&2
Usage: run_unet_watershed_tuning.sh [options]

Tune watershed postprocessing on U-Net semantic predictions for the train
section. Requires dataset/train/manifests/{variant}.whole.json.

Options:
  --variant NAME         registry variant (default: VARIANT env)
  --model-path PATH
  --dataset-dir PATH
  --gt-gpkg PATH
  --output-dir PATH
  --help
EOF
    exit "$status"
}

VARIANT="${VARIANT:-PPL+AllPPX}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset-dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --gt-gpkg)
            GT_GPKG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            usage 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

unet_patch_config_for_variant "$VARIANT"

CANONICAL_MANIFEST="$GRAINSEG_ROOT/dataset/train/manifests/${VARIANT}.whole.json"
require_file "$CANONICAL_MANIFEST" \
    "Train whole manifest missing for $VARIANT; run write_whole_manifests.py"

if [ -z "$PREDS_DIR" ]; then
    require_file "$MODEL_PATH" "Model not found"
else
    require_dir "$PREDS_DIR" "PREDS_DIR is not a directory"
fi

require_dir "$DATASET_DIR" "Dataset dir not found"
require_file "$GT_GPKG" "Ground-truth GeoPackage not found"

if [ ! -f "$SLURM_ROOT/prepare_env.sh" ]; then
    echo "prepare_env.sh not found at: $SLURM_ROOT/prepare_env.sh" >&2
    exit 1
fi
source "$SLURM_ROOT/prepare_env.sh"
export TF_CPP_MIN_LOG_LEVEL=2

WATERSHED_SUBDIR="$(watershed_tune_subdir_for_variant "$VARIANT")"
VARIANT_OUTPUT_DIR="$OUTPUT_DIR/$WATERSHED_SUBDIR"

WORK_DIR="${TMPDIR:-/tmp}/tune_watershed_${SLURM_JOB_ID:-$$}"
mkdir -p "$WORK_DIR"
cp "$GT_GPKG" "$WORK_DIR/gt.gpkg"
LOCAL_IMAGE_DIR="$WORK_DIR/dataset"
LOCAL_GT_GPKG="$WORK_DIR/gt.gpkg"

echo "Staging train whole manifest to $LOCAL_IMAGE_DIR ..."
uv run --directory "$REPO_ROOT" python -m common.stage_manifest run \
    "$CANONICAL_MANIFEST" "$LOCAL_IMAGE_DIR"
STAGED_MANIFEST="$LOCAL_IMAGE_DIR/manifest.json"
require_file "$STAGED_MANIFEST" "Staged train manifest missing"

mkdir -p "$VARIANT_OUTPUT_DIR"
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUT_CSV="$VARIANT_OUTPUT_DIR/watershed_grid_${JOB_TAG}.csv"
OUT_JSON="$VARIANT_OUTPUT_DIR/watershed_best_${JOB_TAG}.json"

cd "$REPO_ROOT/src/unet"
echo "Syncing U-Net environment..."
uv sync

install_unet_tensorflow_wheel

TUNE_CMD=(
    uv run --no-sync python -u -m unet.tune_watershed
    --manifest "$STAGED_MANIFEST"
    --gt-gpkg "$LOCAL_GT_GPKG"
    --output-csv "$OUT_CSV"
    --output-json "$OUT_JSON"
    --num-inputs "$NUM_INPUTS"
    --patch-size "$PATCH_SIZE"
    --stride "$STRIDE"
    --batch-size "$BATCH_SIZE"
    --min-distance "${MIN_DISTANCE[@]}"
    --boundary-dilate-iter "${BOUNDARY_DILATE_ITER[@]}"
    --watershed-connectivity "${WATERSHED_CONNECTIVITY[@]}"
    --min-area-px "${MIN_AREA_PX[@]}"
    --exclude-border "${EXCLUDE_BORDER[@]}"
)

if [ -n "$PREDS_DIR" ]; then
    TUNE_CMD+=(--preds-dir "$PREDS_DIR")
else
    LOCAL_MODEL="$WORK_DIR/model.keras"
    cp "$MODEL_PATH" "$LOCAL_MODEL"
    TUNE_CMD+=(--model-path "$LOCAL_MODEL")
fi

echo "Running watershed tuning (variant=$VARIANT, subdir=$WATERSHED_SUBDIR)..."
echo "  dataset: $DATASET_DIR"
echo "  model:   $MODEL_PATH"
echo "  CSV:     $OUT_CSV"
echo "  JSON:    $OUT_JSON"
"${TUNE_CMD[@]}"

echo "Done."
