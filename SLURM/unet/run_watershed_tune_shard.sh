#!/bin/bash
#SBATCH --job-name=TuneWatershedShard
#SBATCH --output=logs/%x-%A_%a.log
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00

set -euo pipefail
# shellcheck source=SLURM/utils/enter_job.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/SLURM/utils/enter_job.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$SLURM_ROOT/utils/variants.sh"
# shellcheck source=SLURM/utils/manifest_shell.sh
source "$SLURM_ROOT/utils/manifest_shell.sh"
# shellcheck source=SLURM/utils/assertions.sh
source "$SLURM_ROOT/utils/assertions.sh"
mkdir -p "$REPO_ROOT/logs"

: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set (watershed tune shard array)}"
: "${RUN_TAG:?RUN_TAG must be set by submit script}"

GRAINSEG_ROOT="$(grainseg_root)"
DATASET_DIR="${DATASET_DIR:-$GRAINSEG_ROOT/dataset/train}"
GT_GPKG="${GT_GPKG:-$DATASET_DIR/train_labels.gpkg}"
OUTPUT_DIR="$GRAINSEG_ROOT/runs/watershed_tune"

PREDS_DIR="${PREDS_DIR:-}"
GRID_CONFIG="${GRID_CONFIG:-$REPO_ROOT/config/watershed_tune_grid.yaml}"

function usage {
    local status="${1:-1}"
    cat <<EOF >&2
Usage: run_watershed_tune_shard.sh [options]

Score one axis-aligned shard of the watershed tune grid from cached semantic
predictions. Requires SLURM_ARRAY_TASK_ID and RUN_TAG from submit script.

Options:
  --variant NAME         registry variant (default: VARIANT env)
  --preds-dir PATH       directory with {sample_id}_pred.tif files
  --dataset-dir PATH
  --gt-gpkg PATH
  --output-dir PATH
  --run-tag TAG          shared run tag (default: RUN_TAG env)
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
        --preds-dir)
            PREDS_DIR="$2"
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
        --run-tag)
            RUN_TAG="$2"
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

WATERSHED_SUBDIR="$(watershed_tune_subdir_for_variant "$VARIANT")"
if [ -z "$PREDS_DIR" ]; then
    PREDS_DIR="$GRAINSEG_ROOT/runs/watershed_tune_preds/$WATERSHED_SUBDIR/semantic"
fi
require_dir "$PREDS_DIR" "PREDS_DIR is not a directory: $PREDS_DIR"

require_dir "$DATASET_DIR" "Dataset dir not found"
require_file "$GT_GPKG" "Ground-truth GeoPackage not found"
require_file "$GRID_CONFIG" "Watershed tune grid config not found: $GRID_CONFIG"

if [ ! -f "$SLURM_ROOT/prepare_env.sh" ]; then
    echo "prepare_env.sh not found at: $SLURM_ROOT/prepare_env.sh" >&2
    exit 1
fi
source "$SLURM_ROOT/prepare_env.sh"

read -r shard_min_distance shard_boundary_dilate_iter <<< "$(
    uv run --directory "$REPO_ROOT/src/unet" python -m unet.watershed_tune_shard_for_index \
        --grid-config "$GRID_CONFIG" \
        --shard-index "$SLURM_ARRAY_TASK_ID"
)"

WORK_DIR="${TMPDIR:-/tmp}/tune_watershed_shard_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WORK_DIR"
cp "$GT_GPKG" "$WORK_DIR/gt.gpkg"
LOCAL_IMAGE_DIR="$WORK_DIR/dataset"
LOCAL_GT_GPKG="$WORK_DIR/gt.gpkg"

cd "$REPO_ROOT/src/unet"
echo "Syncing U-Net environment..."
uv sync

echo "Staging train whole manifest metadata to $LOCAL_IMAGE_DIR (no channel copies) ..."
stage_manifest_metadata_in_unet_env "$CANONICAL_MANIFEST" "$LOCAL_IMAGE_DIR"
STAGED_MANIFEST="$LOCAL_IMAGE_DIR/manifest.json"
require_file "$STAGED_MANIFEST" "Staged train manifest missing"

VARIANT_OUTPUT_DIR="$OUTPUT_DIR/$WATERSHED_SUBDIR"
mkdir -p "$VARIANT_OUTPUT_DIR"
OUT_CSV="$VARIANT_OUTPUT_DIR/watershed_grid_${RUN_TAG}_shard_${SLURM_ARRAY_TASK_ID}.csv"

TUNE_CMD=(
    uv run --no-sync python -u -m unet.tune_watershed
    --manifest "$STAGED_MANIFEST"
    --gt-gpkg "$LOCAL_GT_GPKG"
    --output-csv "$OUT_CSV"
    --preds-dir "$PREDS_DIR"
    --num-inputs "$NUM_INPUTS"
    --grid-config "$GRID_CONFIG"
    --shard-index "$SLURM_ARRAY_TASK_ID"
    --shard-min-distance "$shard_min_distance"
    --shard-boundary-dilate-iter "$shard_boundary_dilate_iter"
)
if [ "${LOG_EXTRACTION_CACHE:-0}" = "1" ]; then
    TUNE_CMD+=(--log-extraction-cache)
fi

echo "Running watershed tune shard (variant=$VARIANT, subdir=$WATERSHED_SUBDIR)..."
echo "  shard:    ${SLURM_ARRAY_TASK_ID} (min_distance=$shard_min_distance, boundary_dilate_iter=$shard_boundary_dilate_iter)"
echo "  dataset:  $DATASET_DIR"
echo "  preds:    $PREDS_DIR"
echo "  grid:     $GRID_CONFIG"
echo "  run tag:  $RUN_TAG"
echo "  CSV:      $OUT_CSV"
"${TUNE_CMD[@]}"

echo "Done."
