#!/bin/bash
#SBATCH --job-name=patchify
#SBATCH --output=logs/patchify-%j.log
#SBATCH --mem=100GB
#SBATCH --time=02:00:00
set -euo pipefail

source SLURM/prepare_env.sh

WORK_DIR="$TMPDIR/patchify_$SLURM_JOB_ID"
TRAIN_DEST="$SCRATCH/GrainSeg/dataset/train"
TEST_DEST="$SCRATCH/GrainSeg/dataset/test"
TRAIN_WORK="$WORK_DIR/train"
TEST_WORK="$WORK_DIR/test"
mkdir -p "$TRAIN_WORK"
mkdir -p "$TEST_WORK"

# YOLO data YAMLs (path relative to this file; multichannel only where the TIFF has extra bands).
# held_out=1: all patches live under images/val/ only (--no-split test mosaics); point train/val/test
# there so Ultralytics model.val(..., split="test") resolves paths without an empty train/ split.
write_yolo_dataset_yamls() {
    local yolo_root=$1
    local held_out="${2:-0}"
    local train_p val_p test_p
    if [[ "$held_out" == "1" ]]; then
        train_p="images/val"
        val_p="images/val"
        test_p="images/val"
    else
        train_p="images/train"
        val_p="images/val"
        test_p=""
    fi

    cat > "$yolo_root/PPL/PPL.yaml" <<EOF
path: .
train: $train_p
val: $val_p
test: $test_p

# Classes
names:
  0: grain
EOF
    cat > "$yolo_root/PPL+AllPPX/PPL+AllPPX.yaml" <<EOF
path: .
train: $train_p
val: $val_p
test: $test_p

channels: 21

# Classes
names:
  0: grain
EOF
    cat > "$yolo_root/PPL+PPXblend/PPL_PPXblend.yaml" <<EOF
path: .
train: $train_p
val: $val_p
test: $test_p

channels: 6

# Classes
names:
  0: grain
EOF
    cat > "$yolo_root/PPLPPXblend/PPLPPXblend.yaml" <<EOF
path: .
train: $train_p
val: $val_p
test: $test_p

# Classes
names:
  0: grain
EOF
}

cd src/data_prep

echo "Syncing data prep environment..."
uv sync

echo "Copying train inputs to fast local storage ($TMPDIR)..."
cp "$TRAIN_DEST/train_PPL+PPXblend.tif" "$TRAIN_WORK/"
cp "$TRAIN_DEST/train_PPL+AllPPX.tif" "$TRAIN_WORK/"
cp "$TRAIN_DEST/train_PPLPPXblend.tif" "$TRAIN_WORK/"
cp "$TRAIN_DEST/train_PPL.tif" "$TRAIN_WORK/"
cp "$TRAIN_DEST/train_labels.gpkg" "$TRAIN_WORK/labels.gpkg"

echo "Running split_tiff_gpkg_to_yolo for all variants (train)..."
uv run --no-sync python -u split_tiff_gpkg_to_yolo.py \
    --image "$TRAIN_WORK/train_PPL.tif" \
    --polygons "$TRAIN_WORK/labels.gpkg" \
    --output-dir "$TRAIN_WORK/PPL" \
    --patch-size 1024 \
    --patch-overlap 0.5 \
    --tile-size 4096 \
    --validation-fraction 0.2 \
    --random-state 42

uv run --no-sync python -u split_tiff_gpkg_to_yolo.py \
    --image "$TRAIN_WORK/train_PPLPPXblend.tif" \
    --polygons "$TRAIN_WORK/labels.gpkg" \
    --output-dir "$TRAIN_WORK/PPLPPXblend" \
    --patch-size 1024 \
    --patch-overlap 0.5 \
    --tile-size 4096 \
    --validation-fraction 0.2 \
    --random-state 42

uv run --no-sync python -u split_tiff_gpkg_to_yolo.py \
    --image "$TRAIN_WORK/train_PPL+PPXblend.tif" \
    --polygons "$TRAIN_WORK/labels.gpkg" \
    --output-dir "$TRAIN_WORK/PPL+PPXblend" \
    --patch-size 1024 \
    --patch-overlap 0.5 \
    --tile-size 4096 \
    --validation-fraction 0.2 \
    --random-state 42

uv run --no-sync python -u split_tiff_gpkg_to_yolo.py \
    --image "$TRAIN_WORK/train_PPL+AllPPX.tif" \
    --polygons "$TRAIN_WORK/labels.gpkg" \
    --output-dir "$TRAIN_WORK/PPL+AllPPX" \
    --patch-size 1024 \
    --patch-overlap 0.5 \
    --tile-size 4096 \
    --validation-fraction 0.2 \
    --random-state 42

echo "Copying YOLO train variants to persistent storage..."
mkdir -p "$TRAIN_DEST/yolo"
mv "$TRAIN_WORK/PPL" "$TRAIN_DEST/yolo/PPL"
mv "$TRAIN_WORK/PPLPPXblend" "$TRAIN_DEST/yolo/PPLPPXblend"
mv "$TRAIN_WORK/PPL+PPXblend" "$TRAIN_DEST/yolo/PPL+PPXblend"
mv "$TRAIN_WORK/PPL+AllPPX" "$TRAIN_DEST/yolo/PPL+AllPPX"

write_yolo_dataset_yamls "$TRAIN_DEST/yolo"

echo "Copying test inputs to fast local storage ($TMPDIR)..."
cp "$TEST_DEST/test_PPL+PPXblend.tif" "$TEST_WORK/"
cp "$TEST_DEST/test_PPL+AllPPX.tif" "$TEST_WORK/"
cp "$TEST_DEST/test_PPLPPXblend.tif" "$TEST_WORK/"
cp "$TEST_DEST/test_PPL.tif" "$TEST_WORK/"
cp "$TEST_DEST/test_labels.gpkg" "$TEST_WORK/labels.gpkg"

echo "Running split_tiff_gpkg_to_yolo for all variants (test, full mosaic -> val/ only)..."
uv run --no-sync python -u split_tiff_gpkg_to_yolo.py \
    --image "$TEST_WORK/test_PPL.tif" \
    --polygons "$TEST_WORK/labels.gpkg" \
    --output-dir "$TEST_WORK/PPL" \
    --patch-size 1024 \
    --patch-overlap 0 \
    --tile-size 4096 \
    --validation-fraction 0.2 \
    --random-state 42 \
    --no-split

uv run --no-sync python -u split_tiff_gpkg_to_yolo.py \
    --image "$TEST_WORK/test_PPLPPXblend.tif" \
    --polygons "$TEST_WORK/labels.gpkg" \
    --output-dir "$TEST_WORK/PPLPPXblend" \
    --patch-size 1024 \
    --patch-overlap 0 \
    --tile-size 4096 \
    --validation-fraction 0.2 \
    --random-state 42 \
    --no-split

uv run --no-sync python -u split_tiff_gpkg_to_yolo.py \
    --image "$TEST_WORK/test_PPL+PPXblend.tif" \
    --polygons "$TEST_WORK/labels.gpkg" \
    --output-dir "$TEST_WORK/PPL+PPXblend" \
    --patch-size 1024 \
    --patch-overlap 0 \
    --tile-size 4096 \
    --validation-fraction 0.2 \
    --random-state 42 \
    --no-split

uv run --no-sync python -u split_tiff_gpkg_to_yolo.py \
    --image "$TEST_WORK/test_PPL+AllPPX.tif" \
    --polygons "$TEST_WORK/labels.gpkg" \
    --output-dir "$TEST_WORK/PPL+AllPPX" \
    --patch-size 1024 \
    --patch-overlap 0 \
    --tile-size 4096 \
    --validation-fraction 0.2 \
    --random-state 42 \
    --no-split

echo "Copying YOLO test variants to persistent storage..."
mkdir -p "$TEST_DEST/yolo"
mv "$TEST_WORK/PPL" "$TEST_DEST/yolo/PPL"
mv "$TEST_WORK/PPLPPXblend" "$TEST_DEST/yolo/PPLPPXblend"
mv "$TEST_WORK/PPL+PPXblend" "$TEST_DEST/yolo/PPL+PPXblend"
mv "$TEST_WORK/PPL+AllPPX" "$TEST_DEST/yolo/PPL+AllPPX"

write_yolo_dataset_yamls "$TEST_DEST/yolo" 1

echo "Done!"
