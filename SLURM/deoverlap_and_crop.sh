#!/bin/bash
#SBATCH --job-name=split_overlaps
#SBATCH --output=logs/split_overlaps-%j.log
#SBATCH --mem=100GB
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

source SLURM/prepare_env.sh

echo "Copying input files to fast local storage ($TMPDIR)..."
WORK_DIR="$TMPDIR/split_overlaps_$SLURM_JOB_ID"

TEST_DIR="$WORK_DIR/test"
TRAIN_DIR="$WORK_DIR/train"
mkdir -p "$TRAIN_DIR"
mkdir -p "$TEST_DIR"

cp $SCRATCH/GrainSeg/dataset/train/train_raw.gpkg "$TRAIN_DIR/"
cp $SCRATCH/GrainSeg/dataset/test/test_raw.gpkg "$TEST_DIR/"

cp $SCRATCH/GrainSeg/dataset/uncropped/PPL.tif "$WORK_DIR/"
cp $SCRATCH/GrainSeg/dataset/uncropped/PPX*.tif "$WORK_DIR/"

cd src/data_prep
echo "Running split overlaps script on train..."
uv run split_overlaps -u split_overlaps.py \
    --input "$TRAIN_DIR/train_raw.gpkg" \
    --output "$TRAIN_DIR/train_split.gpkg" \
    --min-area 300 \
    --workers 10

echo "Running split overlaps script on test..."
uv run python -u split_overlaps.py \
    --input "$TEST_DIR/test_raw.gpkg" \
    --output "$TEST_DIR/test_split.gpkg" \
    --min-area 300 \
    --workers 10


echo "Running cropping script on train..."
uv run python -u crop_images.py \
    --vector "$TRAIN_DIR/train_split.gpkg" \
    --out-vector "$TRAIN_DIR/train_cropped.gpkg" \
    --image-dir "$WORK_DIR/" \
    --out-image-dir "$TRAIN_DIR/cropped" \
    --suffix "" \
    --bbox "5000, -10000, 57000, 0"

echo "Running cropping script on test..."
uv run python -u crop_images.py \
    --vector "$TEST_DIR/test_split.gpkg" \
    --out-vector "$TEST_DIR/test_cropped.gpkg" \
    --image-dir "$WORK_DIR/" \
    --out-image-dir "$TEST_DIR/cropped" \
    --suffix "" \
    --bbox "0, -30000, 10000, -40000"

echo "Copying results back to persistent storage..."
cp "$TRAIN_DIR/train_cropped.gpkg" $SCRATCH/GrainSeg/dataset/train/train_labels.gpkg
cp "$TEST_DIR/test_cropped.gpkg" $SCRATCH/GrainSeg/dataset/test/test_labels.gpkg

cp -r "$TRAIN_DIR/cropped/"* $SCRATCH/GrainSeg/dataset/train/
cp -r "$TEST_DIR/cropped/"* $SCRATCH/GrainSeg/dataset/test/

echo "Done!"