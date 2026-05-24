#!/bin/bash
#SBATCH --job-name=split_overlaps
#SBATCH --output=logs/split_overlaps-%j.log
#SBATCH --mem=100GB
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

# shellcheck source=SLURM/utils/enter_job.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/SLURM/utils/enter_job.sh"
source "$SLURM_ROOT/prepare_env.sh"

GRAINSEG_ROOT="$(grainseg_root)"

echo "Copying input files to fast local storage ($TMPDIR)..."
WORK_DIR="$TMPDIR/split_overlaps_$SLURM_JOB_ID"

TEST_DIR="$WORK_DIR/test"
TRAIN_DIR="$WORK_DIR/train"
mkdir -p "$TRAIN_DIR"
mkdir -p "$TEST_DIR"

cp $GRAINSEG_ROOT/dataset/uncropped/train_raw.gpkg "$TRAIN_DIR/"
cp $GRAINSEG_ROOT/dataset/uncropped/test_raw.gpkg "$TEST_DIR/"

cp $GRAINSEG_ROOT/dataset/uncropped/PPL.tif "$WORK_DIR/"
cp $GRAINSEG_ROOT/dataset/uncropped/PPX*.tif "$WORK_DIR/"

cd "$REPO_ROOT/src/data_prep"
uv sync
echo "Running split overlaps script on train..."
uv run python -u split_overlaps.py \
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
cp "$TRAIN_DIR/train_cropped.gpkg" $GRAINSEG_ROOT/dataset/train/train_labels.gpkg
cp "$TEST_DIR/test_cropped.gpkg" $GRAINSEG_ROOT/dataset/test/test_labels.gpkg

cp -r "$TRAIN_DIR/cropped/"* $GRAINSEG_ROOT/dataset/train/
cp -r "$TEST_DIR/cropped/"* $GRAINSEG_ROOT/dataset/test/

for base in PPL PPX1 PPX2 PPX3 PPX4 PPX5 PPX6; do
  if [[ -f $GRAINSEG_ROOT/dataset/train/${base}.tif ]]; then
    mv -f $GRAINSEG_ROOT/dataset/train/${base}.tif \
      $GRAINSEG_ROOT/dataset/train/train_${base}.tif
  fi
  if [[ -f $GRAINSEG_ROOT/dataset/test/${base}.tif ]]; then
    mv -f $GRAINSEG_ROOT/dataset/test/${base}.tif \
      $GRAINSEG_ROOT/dataset/test/test_${base}.tif
  fi
done

echo "Done!"