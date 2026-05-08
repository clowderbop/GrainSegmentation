#!/bin/bash
#SBATCH --job-name=blend_PPX
#SBATCH --output=logs/blend_PPX-%j.log
#SBATCH --mem=50GB
#SBATCH --time=00:30:00
set -euo pipefail

source SLURM/prepare_env.sh

WORK_DIR="$TMPDIR/blend_PPX_$SLURM_JOB_ID"
TRAIN_DIR="$WORK_DIR/train"
TEST_DIR="$WORK_DIR/test"
RESULT_DIR="$WORK_DIR/result"
mkdir -p "$TRAIN_DIR"
mkdir -p "$TEST_DIR"
mkdir -p "$RESULT_DIR"

TRAIN_DEST="$SCRATCH/GrainSeg/dataset/train"
TEST_DEST="$SCRATCH/GrainSeg/dataset/test"

echo "Syncing data prep environment..."
cd src/data_prep
uv sync

echo "Blending PPX images..."

# Get PPX images
for i in {1..6}; do
    cp "$TRAIN_DEST/train_PPX${i}.tif" "$TRAIN_DIR/"
done

for i in {1..6}; do
    cp "$TEST_DEST/test_PPX${i}.tif" "$TEST_DIR/"
done

uv run --no-sync python -u blend_tiffs.py \
    "$TRAIN_DIR/" \
    "$RESULT_DIR/train_PPXblend.tif"

mv "$RESULT_DIR/train_PPXblend.tif" $TRAIN_DEST/train_PPXblend.tif


uv run --no-sync python -u blend_tiffs.py \
    "$TEST_DIR/" \
    "$RESULT_DIR/test_PPXblend.tif"

mv "$RESULT_DIR/test_PPXblend.tif" $TEST_DEST/test_PPXblend.tif


echo "Blending PPL and PPX images..."

# train_PPL + train_PPX1..6: lexicographic order puts PPL first (base for blend)
cp "$TRAIN_DEST/train_PPL.tif" "$TRAIN_DIR/train_PPL.tif"
cp "$TEST_DEST/test_PPL.tif" "$TEST_DIR/test_PPL.tif"

uv run --no-sync python -u blend_tiffs.py \
    "$TRAIN_DIR" \
    "$RESULT_DIR/train_PPLPPXblend.tif"

mv "$RESULT_DIR/train_PPLPPXblend.tif" $TRAIN_DEST/train_PPLPPXblend.tif

uv run --no-sync python -u blend_tiffs.py \
    "$TEST_DIR/" \
    "$RESULT_DIR/test_PPLPPXblend.tif"

mv "$RESULT_DIR/test_PPLPPXblend.tif" $TEST_DEST/test_PPLPPXblend.tif

echo "Done!"
