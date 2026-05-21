#!/bin/bash
#SBATCH --job-name=merge_multichannel_tiff
#SBATCH --output=logs/merge_multichannel_tiff-%j.log
#SBATCH --mem=50GB
#SBATCH --time=00:30:00
set -euo pipefail

# shellcheck source=SLURM/utils/source_job.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/source_job.sh"
source "$SLURM_ROOT/prepare_env.sh"

GRAINSEG_ROOT="$(grainseg_root)"

WORK_DIR="$TMPDIR/merge_multichannel_tiff_$SLURM_JOB_ID"
TRAIN_DEST="$GRAINSEG_ROOT/dataset/train"
TEST_DEST="$GRAINSEG_ROOT/dataset/test"
TRAIN_WORK="$WORK_DIR/train"
TEST_WORK="$WORK_DIR/test"
mkdir -p "$TRAIN_WORK"
mkdir -p "$TEST_WORK"

echo "Syncing data prep environment..."
cd "$REPO_ROOT/src/data_prep"
uv sync

echo "Merging PPL and PPX blend into multichannel TIFFs (train)..."

cp "$TRAIN_DEST/train_PPL.tif" "$TRAIN_WORK/"
cp "$TRAIN_DEST/train_PPXblend.tif" "$TRAIN_WORK/"
uv run --no-sync python -u stack_tiff_channels.py \
    "$TRAIN_WORK" \
    "$TRAIN_WORK/PPL+PPXblend.tif"
mv "$TRAIN_WORK/PPL+PPXblend.tif" "$TRAIN_DEST/train_PPL+PPXblend.tif"
rm -f "$TRAIN_WORK/train_PPXblend.tif" # Keep train_PPL for the All-PPX merge

echo "Merging PPL and PPX blend into multichannel TIFFs (test)..."
cp "$TEST_DEST/test_PPL.tif" "$TEST_WORK/"
cp "$TEST_DEST/test_PPXblend.tif" "$TEST_WORK/"
uv run --no-sync python -u stack_tiff_channels.py \
    "$TEST_WORK" \
    "$TEST_WORK/PPL+PPXblend.tif"
mv "$TEST_WORK/PPL+PPXblend.tif" "$TEST_DEST/test_PPL+PPXblend.tif"
rm -f "$TEST_WORK/test_PPXblend.tif" # Keep test_PPL for the All-PPX merge

echo "Merging PPL and all PPX channels (train)..."
for i in {1..6}; do
    cp "$TRAIN_DEST/train_PPX${i}.tif" "$TRAIN_WORK/"
done
uv run --no-sync python -u stack_tiff_channels.py \
    "$TRAIN_WORK" \
    "$TRAIN_WORK/PPL+AllPPX.tif"
mv "$TRAIN_WORK/PPL+AllPPX.tif" "$TRAIN_DEST/train_PPL+AllPPX.tif"

echo "Merging PPL and all PPX channels (test)..."
for i in {1..6}; do
    cp "$TEST_DEST/test_PPX${i}.tif" "$TEST_WORK/"
done
uv run --no-sync python -u stack_tiff_channels.py \
    "$TEST_WORK" \
    "$TEST_WORK/PPL+AllPPX.tif"
mv "$TEST_WORK/PPL+AllPPX.tif" "$TEST_DEST/test_PPL+AllPPX.tif"

echo "Done!"
