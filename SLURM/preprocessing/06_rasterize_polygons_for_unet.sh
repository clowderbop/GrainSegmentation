#!/bin/bash
#SBATCH --job-name=GPKG2Raster
#SBATCH --output=logs/gpkg-to-raster-%j.log
#SBATCH --mem=5G
#SBATCH --time=00:05:00

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_ROOT="$(cd "$THIS_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SLURM_ROOT/.." && pwd)"
cd "$REPO_ROOT"
source "$SLURM_ROOT/prepare_env.sh"

INPUT_GPKG="${INPUT_GPKG:-$SCRATCH/GrainSeg/dataset/MWD-1#121/labels_cropped.gpkg}"
REFERENCE_TIFF="${REFERENCE_TIFF:-$SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/PPL.tif}"
OUTPUT_RASTER="${OUTPUT_RASTER:-$SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/labels_raster.tif}"
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
