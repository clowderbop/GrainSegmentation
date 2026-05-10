#!/bin/bash
#SBATCH --job-name=unet_yolo_masks
#SBATCH --output=logs/unet_yolo_masks-%j.log
#SBATCH --mem=64GB
#SBATCH --time=01:00:00

set -euo pipefail

# Crop full test_labels.tif to match YOLO val patches from patchify.
# Copy inputs to $TMPDIR, run crop_unet_masks_from_yolo_patches.py there, then copy
# results to $SCRATCH (.../unet_from_yolo/<VARIANT>/images|masks).
#
# Supported VARIANT values are single-input layouts only (one GeoTIFF per patch);
# PPL+PPXblend and PPL+AllPPX need multiple files per stem for UNet—extend this
# script or split channels before using list_samples.
#
# Override: sbatch --export=ALL,VARIANT=PPLPPXblend SLURM/unet_patch_masks_from_yolo.sh

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

VARIANT="${VARIANT:-PPL}"
PATCH_SIZE=1024
TILE_SIZE=4096
IMAGE_SUFFIX="_PPL"
JOB_TAG="${SLURM_JOB_ID:-local}"

source SLURM/prepare_env.sh

TEST_ROOT="$SCRATCH/GrainSeg/dataset/test"
YOLO_ROOT="$TEST_ROOT/yolo"
OUTPUT_ROOT="$TEST_ROOT/unet_from_yolo"

case "$VARIANT" in
    PPL)
        REF_TIFF="$TEST_ROOT/test_PPL.tif"
        IMAGE_SUFFIX="_PPL"
        ;;
    PPLPPXblend)
        REF_TIFF="$TEST_ROOT/test_PPLPPXblend.tif"
        IMAGE_SUFFIX="_PPLPPXblend"
        ;;
    PPL+PPXblend|PPL+AllPPX)
        echo "VARIANT=$VARIANT is multi-input for UNet; this job only copies one TIFF per patch." >&2
        echo "Use PPL or PPLPPXblend, or add per-suffix band splitting for this variant." >&2
        exit 1
        ;;
    *)
        echo "Unknown variant: $VARIANT" >&2
        exit 1
        ;;
esac

REF_MASK="$TEST_ROOT/test_labels.tif"
YOLO_IMAGES="$YOLO_ROOT/$VARIANT/images/val"
OUT_IMAGES="$OUTPUT_ROOT/$VARIANT/images"
OUT_MASKS="$OUTPUT_ROOT/$VARIANT/masks"

if [[ ! -d "$YOLO_IMAGES" ]]; then
    echo "YOLO val images not found (run patchify first?): $YOLO_IMAGES" >&2
    exit 1
fi
if [[ ! -f "$REF_TIFF" ]]; then
    echo "Reference TIFF missing: $REF_TIFF" >&2
    exit 1
fi
if [[ ! -f "$REF_MASK" ]]; then
    echo "Reference mask missing: $REF_MASK (see SLURM/test_unet_prepare_labels.sh)" >&2
    exit 1
fi

WORK_ROOT="$TMPDIR/unet_yolo_masks_${VARIANT}_$JOB_TAG"
LOCAL_YOLO_IMAGES="$WORK_ROOT/yolo_val_images"
LOCAL_REF_TIFF="$WORK_ROOT/$(basename "$REF_TIFF")"
LOCAL_REF_MASK="$WORK_ROOT/$(basename "$REF_MASK")"
LOCAL_OUT_IMAGES="$WORK_ROOT/out_images"
LOCAL_OUT_MASKS="$WORK_ROOT/out_masks"

rm -rf "$WORK_ROOT"
mkdir -p "$LOCAL_YOLO_IMAGES" "$LOCAL_OUT_IMAGES" "$LOCAL_OUT_MASKS"

echo "Staging YOLO val patches + reference raster/mask to TMPDIR ($WORK_ROOT)..."
cp -r "$YOLO_IMAGES"/. "$LOCAL_YOLO_IMAGES"/
cp -f "$REF_TIFF" "$LOCAL_REF_TIFF"
cp -f "$REF_MASK" "$LOCAL_REF_MASK"

cd src/data_prep
echo "Syncing data prep environment..."
uv sync

uv run python -u crop_unet_masks_from_yolo_patches.py \
    --reference-tiff "$LOCAL_REF_TIFF" \
    --reference-mask "$LOCAL_REF_MASK" \
    --yolo-images-dir "$LOCAL_YOLO_IMAGES" \
    --output-images-dir "$LOCAL_OUT_IMAGES" \
    --output-masks-dir "$LOCAL_OUT_MASKS" \
    --patch-size "$PATCH_SIZE" \
    --tile-size "$TILE_SIZE" \
    --image-suffix "$IMAGE_SUFFIX"

echo "Copying cropped patch outputs to $SCRATCH..."
mkdir -p "$OUT_IMAGES" "$OUT_MASKS"
cp -r "$LOCAL_OUT_IMAGES"/. "$OUT_IMAGES"/
cp -r "$LOCAL_OUT_MASKS"/. "$OUT_MASKS"/

echo "Done. UNet image-dir: $OUT_IMAGES"
echo "             mask-dir: $OUT_MASKS"
