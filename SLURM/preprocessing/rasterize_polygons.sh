#!/bin/bash
#SBATCH --job-name=GPKG2Raster
#SBATCH --output=logs/gpkg-to-raster-%j.log
#SBATCH --mem=5G
#SBATCH --time=00:05:00

set -euo pipefail

# shellcheck source=SLURM/utils/enter_job.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/SLURM/utils/enter_job.sh"
source "$SLURM_ROOT/prepare_env.sh"

GRAINSEG_ROOT="$(grainseg_root)"

INPUT_GPKG="${INPUT_GPKG:-$GRAINSEG_ROOT/dataset/test/test_labels.gpkg}"
REFERENCE_TIFF="${REFERENCE_TIFF:-$GRAINSEG_ROOT/dataset/test/test_PPL.tif}"
OUTPUT_RASTER="${OUTPUT_RASTER:-$GRAINSEG_ROOT/dataset/test/test_labels.tif}"
BOUNDARY_WIDTH="${BOUNDARY_WIDTH:-3.0}"

if [[ -z "$INPUT_GPKG" || -z "$REFERENCE_TIFF" || -z "$OUTPUT_RASTER" ]]; then
    echo "Error: INPUT_GPKG, REFERENCE_TIFF, and OUTPUT_RASTER must be set."
    exit 1
fi

echo "Copying input files to fast local storage ($TMPDIR)..."
WORK_DIR="$TMPDIR/gpkg_to_raster_$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"

INPUT_GPKG_NAME="$(basename "$INPUT_GPKG")"
REFERENCE_TIFF_NAME="$(basename "$REFERENCE_TIFF")"
OUTPUT_RASTER_NAME="$(basename "$OUTPUT_RASTER")"

cp "$INPUT_GPKG" "$WORK_DIR/"
cp "$REFERENCE_TIFF" "$WORK_DIR/"

echo "Syncing data prep environment..."
cd "$REPO_ROOT/src/data_prep"
uv sync

CMD=(uv run --no-sync python -u gpkg_to_raster.py
    --input "$WORK_DIR/$INPUT_GPKG_NAME"
    --reference "$WORK_DIR/$REFERENCE_TIFF_NAME"
    --output "$WORK_DIR/$OUTPUT_RASTER_NAME"
    --boundary-width "$BOUNDARY_WIDTH"
)

if [[ "${NO_FLIP_Y:-}" == "1" || "${NO_FLIP_Y:-}" == "true" || "${NO_FLIP_Y:-}" == "True" ]]; then
    CMD+=(--no-flip-y)
fi

echo "Running GPKG to Raster conversion on local storage..."
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Copying results back to persistent storage..."

mkdir -p "$(dirname "$OUTPUT_RASTER")"
cp "$WORK_DIR/$OUTPUT_RASTER_NAME" "$OUTPUT_RASTER"

echo "Done!"
