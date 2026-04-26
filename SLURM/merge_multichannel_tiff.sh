#!/bin/bash
#SBATCH --job-name=merge_multichannel_tiff
#SBATCH --output=logs/merge_multichannel_tiff-%j.log
#SBATCH --mem=50GB
#SBATCH --time=00:30:00
set -euo pipefail

source SLURM/prepare_env.sh

WORK_DIR="$TMPDIR/merge_multichannel_tiff_$SLURM_JOB_ID"
TRAIN_DEST="$SCRATCH/GrainSeg/dataset/train"
TEST_DEST="$SCRATCH/GrainSeg/dataset/test"
TRAIN_WORK="$WORK_DIR/train"
TEST_WORK="$WORK_DIR/test"
mkdir -p "$TRAIN_WORK"
mkdir -p "$TEST_WORK"

echo "Syncing data prep environment..."
cd src/data_prep
uv sync

echo "Merging PPL and PPX blend into multichannel TIFFs (train)..."
cp "$TRAIN_DEST/PPL.tif" "$TRAIN_WORK/"
cp "$TRAIN_DEST/PPXblend.tif" "$TRAIN_WORK/"
uv run --no-sync python -u stack_tiff_channels.py \
    "$TRAIN_WORK" \
    "$TRAIN_WORK/PPL+PPXblend.tif"
mv "$TRAIN_WORK/PPL+PPXblend.tif" "$TRAIN_DEST/PPL+PPXblend.tif"
rm -f "$TRAIN_WORK/PPXblend.tif" # Keep PPL.tif for later

echo "Merging PPL and PPX blend into multichannel TIFFs (test)..."
cp "$TEST_DEST/PPL.tif" "$TEST_WORK/"
cp "$TEST_DEST/PPXblend.tif" "$TEST_WORK/"
uv run --no-sync python -u stack_tiff_channels.py \
    "$TEST_WORK" \
    "$TEST_WORK/PPL+PPXblend.tif"
mv "$TEST_WORK/PPL+PPXblend.tif" "$TEST_DEST/PPL+PPXblend.tif"
rm -f "$TEST_WORK/PPXblend.tif" # Keep PPL.tif for later

echo "Merging PPL and all PPX channels (train)..."
for i in {1..6}; do
    cp "$TRAIN_DEST/PPX${i}.tif" "$TRAIN_WORK/"
done
uv run --no-sync python -u stack_tiff_channels.py \
    "$TRAIN_WORK" \
    "$TRAIN_WORK/PPL+AllPPX.tif"
mv "$TRAIN_WORK/PPL+AllPPX.tif" "$TRAIN_DEST/PPL+AllPPX.tif"

echo "Merging PPL and all PPX channels (test)..."
for i in {1..6}; do
    cp "$TEST_DEST/PPX${i}.tif" "$TEST_WORK/"
done
uv run --no-sync python -u stack_tiff_channels.py \
    "$TEST_WORK" \
    "$TEST_WORK/PPL+AllPPX.tif"
mv "$TEST_WORK/PPL+AllPPX.tif" "$TEST_DEST/PPL+AllPPX.tif"

echo "Done!"
