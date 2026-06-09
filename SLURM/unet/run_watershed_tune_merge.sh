#!/bin/bash
#SBATCH --job-name=TuneWatershedMerge
#SBATCH --output=logs/%x-%j.log
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

set -euo pipefail
# shellcheck source=SLURM/utils/enter_job.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/SLURM/utils/enter_job.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$SLURM_ROOT/utils/variants.sh"
# shellcheck source=SLURM/utils/assertions.sh
source "$SLURM_ROOT/utils/assertions.sh"
mkdir -p "$REPO_ROOT/logs"

GRAINSEG_ROOT="$(grainseg_root)"
DATASET_DIR="${DATASET_DIR:-$GRAINSEG_ROOT/dataset/train}"
OUTPUT_DIR="$GRAINSEG_ROOT/runs/watershed_tune"
GRID_CONFIG="${GRID_CONFIG:-$REPO_ROOT/config/watershed_tune_grid.yaml}"

function usage {
    local status="${1:-1}"
    cat <<EOF >&2
Usage: run_watershed_tune_merge.sh [options]

Merge shard watershed tune grid CSVs into canonical merged grid CSV and best JSON.

Options:
  --variant NAME         registry variant (default: VARIANT env)
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

: "${RUN_TAG:?RUN_TAG required (--run-tag or RUN_TAG env)}"

CANONICAL_MANIFEST="$GRAINSEG_ROOT/dataset/train/manifests/${VARIANT}.whole.json"
require_file "$CANONICAL_MANIFEST" "Train whole manifest missing for $VARIANT"
require_file "$GRID_CONFIG" "Watershed tune grid config not found: $GRID_CONFIG"

# Resolve variant subdir before prepare_env: variants.sh uses the repo workspace
# venv when UV_PROJECT_ENVIRONMENT is unset; after prepare_env it points at an
# empty job venv that has not been uv sync'd yet.
WATERSHED_SUBDIR="$(watershed_tune_subdir_for_variant "$VARIANT")"
VARIANT_OUTPUT_DIR="$OUTPUT_DIR/$WATERSHED_SUBDIR"
require_dir "$VARIANT_OUTPUT_DIR" "Watershed tune output dir not found: $VARIANT_OUTPUT_DIR"

MERGE_JOB_TAG="${SLURM_JOB_ID:-manual}"
OUT_CSV="$VARIANT_OUTPUT_DIR/watershed_grid_${RUN_TAG}.csv"
OUT_JSON="$VARIANT_OUTPUT_DIR/watershed_best_${MERGE_JOB_TAG}.json"
SHARD_GLOB="$VARIANT_OUTPUT_DIR/watershed_grid_${RUN_TAG}_shard_*.csv"

if [ ! -f "$SLURM_ROOT/prepare_env.sh" ]; then
    echo "prepare_env.sh not found at: $SLURM_ROOT/prepare_env.sh" >&2
    exit 1
fi
source "$SLURM_ROOT/prepare_env.sh"

cd "$REPO_ROOT/src/unet"
echo "Syncing U-Net environment..."
uv sync

echo "Merging watershed tune shards (variant=$VARIANT, subdir=$WATERSHED_SUBDIR)..."
echo "  shard glob: $SHARD_GLOB"
echo "  grid:       $GRID_CONFIG"
echo "  run tag:    $RUN_TAG"
echo "  CSV:        $OUT_CSV"
echo "  JSON:       $OUT_JSON"

uv run --no-sync python -u -m unet.watershed_tune_merge_cli \
    --shard-csv-glob "$SHARD_GLOB" \
    --grid-config "$GRID_CONFIG" \
    --manifest "$CANONICAL_MANIFEST" \
    --output-csv "$OUT_CSV" \
    --output-json "$OUT_JSON"

echo "Done."
