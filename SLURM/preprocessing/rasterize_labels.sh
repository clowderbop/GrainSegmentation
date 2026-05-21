#!/bin/bash
set -euo pipefail

# shellcheck source=SLURM/utils/repo_root.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/repo_root.sh"
cd "$REPO_ROOT"

GRAINSEG_ROOT="$(grainseg_root)"

echo "Submitting job to rasterize train labels for UNet"

INPUT_GPKG="$GRAINSEG_ROOT/dataset/train/train_labels.gpkg"
REFERENCE_TIFF="$GRAINSEG_ROOT/dataset/train/train_PPL.tif"
OUTPUT_RASTER="$GRAINSEG_ROOT/dataset/train/train_labels.tif"

sbatch --export=ALL,INPUT_GPKG="$INPUT_GPKG",REFERENCE_TIFF="$REFERENCE_TIFF",OUTPUT_RASTER="$OUTPUT_RASTER" SLURM/preprocessing/rasterize_polygons.sh

echo "Submitting job to rasterize test labels for UNet"

INPUT_GPKG="$GRAINSEG_ROOT/dataset/test/test_labels.gpkg"
REFERENCE_TIFF="$GRAINSEG_ROOT/dataset/test/test_PPL.tif"
OUTPUT_RASTER="$GRAINSEG_ROOT/dataset/test/test_labels.tif"

sbatch --export=ALL,INPUT_GPKG="$INPUT_GPKG",REFERENCE_TIFF="$REFERENCE_TIFF",OUTPUT_RASTER="$OUTPUT_RASTER" SLURM/preprocessing/rasterize_polygons.sh
