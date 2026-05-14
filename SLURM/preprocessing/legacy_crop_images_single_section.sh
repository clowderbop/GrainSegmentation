#!/bin/bash
#SBATCH --job-name=legacy_crop_images
#SBATCH --output=logs/legacy_crop_images-%j.log
#SBATCH --time=00:10:00
#SBATCH --mem=20GB

# Legacy single-section crop for dataset/MWD-1#121 (superseded by
# 02_split_overlaps_and_crop_train_test.sh for train/test pipeline).

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_ROOT="$(cd "$THIS_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SLURM_ROOT/.." && pwd)"
cd "$REPO_ROOT"

# === Configuration ===
BBOX="5000, -10000, 57000, 0"
IN_LABELS="labels_no_overlap.gpkg"
OUT_LABELS="labels_cropped.gpkg"
DATA_DIR="$SCRATCH/GrainSeg/dataset/MWD-1#121"
SUFFIX=""
# =====================

source "$SLURM_ROOT/prepare_env.sh"

echo "Copying input files to fast local storage ($TMPDIR)..."
WORK_DIR="$TMPDIR/crop_images_$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"

cp "$DATA_DIR/$IN_LABELS" "$WORK_DIR/"
cp "$DATA_DIR/"*.tif "$WORK_DIR/"

echo "Syncing data prep environment..."
cd "$REPO_ROOT/src/data_prep"
uv sync

echo "Running cropping script on local storage..."
uv run --no-sync python -u crop_images.py \
    --vector "$WORK_DIR/$IN_LABELS" \
    --out-vector "$WORK_DIR/$OUT_LABELS" \
    --image-dir "$WORK_DIR/" \
    --bbox "$BBOX" \
    --suffix "$SUFFIX"

echo "Copying result back to persistent storage..."
cp "$WORK_DIR/$OUT_LABELS" "$DATA_DIR/$OUT_LABELS"
mkdir -p "$DATA_DIR/cropped"
cp -r "$WORK_DIR/cropped/"* "$DATA_DIR/cropped/"

echo "Done!"
