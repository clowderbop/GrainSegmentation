#!/bin/bash
#SBATCH --job-name=TuneWatershed
#SBATCH --output=logs/%x-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=04:00:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    # shellcheck source=SLURM/bootstrap_paths.sh
    source "$SLURM_SUBMIT_DIR/SLURM/bootstrap_paths.sh"
else
    # shellcheck source=SLURM/bootstrap_paths.sh
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/SLURM/bootstrap_paths.sh"
fi
mkdir -p "$REPO_ROOT/logs"

GRAINSEG_ROOT="${SCRATCH:-/scratch/${USER}}/GrainSeg"
DATASET_DIR="${DATASET_DIR:-$GRAINSEG_ROOT/dataset/train}"
GT_GPKG="${GT_GPKG:-$DATASET_DIR/train_labels.gpkg}"
MODEL_PATH="$GRAINSEG_ROOT/models/unet/unet_finetuned_PPL+AllPPX.keras"
OUTPUT_DIR="$GRAINSEG_ROOT/runs/watershed_tune"

PREDS_DIR=""

NUM_INPUTS=7
PATCH_SIZE=1024
STRIDE=512
BATCH_SIZE=1
IMAGE_SUFFIXES=(_PPL _PPX1 _PPX2 _PPX3 _PPX4 _PPX5 _PPX6)
IMAGE_SUFFIXES_CLI=""

MIN_DISTANCE=(1 3 5)
BOUNDARY_DILATE_ITER=(0 1)
WATERSHED_CONNECTIVITY=(1 2)
MIN_AREA_PX=(0)
EXCLUDE_BORDER=(0 1)

TF_WHEEL_NAME="tensorflow-2.17.0+nv25.2-cp312-cp312-linux_x86_64.whl"

function usage {
    local status="${1:-1}"
    cat <<'EOF' >&2
Usage: run_unet_watershed_tuning.sh [options]

Tune watershed postprocessing on U-Net semantic predictions for the train
section. Runs sliding-window inference on train_*.tif mosaics and grid-searches
watershed parameters against train_labels.gpkg (AJI).

Expects:
  models/unet/unet_finetuned_<variant>.keras
  dataset/train/train_labels.gpkg (instance GT for AJI)
  dataset/train/train_<suffix>.tif inputs per variant

Options:
  --model-path PATH
  --dataset-dir PATH      (default: $SCRATCH/GrainSeg/dataset/train)
  --gt-gpkg PATH          (default: <dataset-dir>/train_labels.gpkg)
  --num-inputs N
  --image-suffixes "_PPL ..."
  --output-dir PATH
  --help
EOF
    exit "$status"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --num-inputs)
            NUM_INPUTS="$2"
            shift 2
            ;;
        --image-suffixes)
            IMAGE_SUFFIXES_CLI="$2"
            shift 2
            ;;
        --dataset-dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --gt-gpkg)
            GT_GPKG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            usage 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

if [ -n "$IMAGE_SUFFIXES_CLI" ]; then
    read -r -a IMAGE_SUFFIXES <<< "$IMAGE_SUFFIXES_CLI"
fi

function require_file {
    local path="$1"
    local message="$2"
    if [ ! -f "$path" ]; then
        echo "$message: $path"
        exit 1
    fi
}

function require_dir {
    local path="$1"
    local message="$2"
    if [ ! -d "$path" ]; then
        echo "$message: $path"
        exit 1
    fi
}

function stage_train_inputs {
    local src_dir="$1"
    local dst_dir="$2"
    mkdir -p "$dst_dir"
    for suffix in "${IMAGE_SUFFIXES[@]}"; do
        local found=false
        for ext in tif tiff; do
            local candidate="$src_dir/train${suffix}.${ext}"
            if [ -f "$candidate" ]; then
                cp "$candidate" "$dst_dir/"
                found=true
                break
            fi
        done
        if [ "$found" = false ]; then
            echo "Missing train input for suffix ${suffix}: expected train${suffix}.tif under $src_dir"
            exit 1
        fi
    done
}

if [ -z "$PREDS_DIR" ]; then
    require_file "$MODEL_PATH" "Model not found"
else
    require_dir "$PREDS_DIR" "PREDS_DIR is not a directory"
fi

require_dir "$DATASET_DIR" "Dataset dir not found"
require_file "$GT_GPKG" "Ground-truth GeoPackage not found"

if [ ! -f "$SLURM_ROOT/prepare_env.sh" ]; then
    echo "prepare_env.sh not found at: $SLURM_ROOT/prepare_env.sh" >&2
    echo "Submit from the repo root, e.g.: cd $REPO_ROOT && sbatch SLURM/unet/run_unet_watershed_tuning.sh ..." >&2
    exit 1
fi
source "$SLURM_ROOT/prepare_env.sh"
export TF_CPP_MIN_LOG_LEVEL=2

WORK_DIR="${TMPDIR:-/tmp}/tune_watershed_${SLURM_JOB_ID:-$$}"
mkdir -p "$WORK_DIR"
echo "Staging train-section inputs to $WORK_DIR ..."
cp "$GT_GPKG" "$WORK_DIR/gt.gpkg"
LOCAL_IMAGE_DIR="$WORK_DIR/dataset"
LOCAL_GT_GPKG="$WORK_DIR/gt.gpkg"
stage_train_inputs "$DATASET_DIR" "$LOCAL_IMAGE_DIR"

mkdir -p "$OUTPUT_DIR"
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUT_CSV="$OUTPUT_DIR/watershed_grid_${JOB_TAG}.csv"
OUT_JSON="$OUTPUT_DIR/watershed_best_${JOB_TAG}.json"

cd "$REPO_ROOT/src/unet"
echo "Syncing U-Net environment..."
uv sync

WHEEL_PATH="$SCRATCH/GrainSeg/wheels/$TF_WHEEL_NAME"
require_file "$WHEEL_PATH" "TensorFlow wheel not found"
echo "Installing TensorFlow wheel..."
uv pip install nvidia-cudnn-cu12~=9.0 nvidia-nccl-cu12 nvidia-cuda-runtime-cu12~=12.8.0 nvidia-cusparse-cu12 nvidia-cufft-cu12 nvidia-cusolver-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 "$WHEEL_PATH"

TUNE_CMD=(
    uv run --no-sync python -u -m unet.tune_watershed
    --image-dir "$LOCAL_IMAGE_DIR"
    --gt-gpkg "$LOCAL_GT_GPKG"
    --output-csv "$OUT_CSV"
    --output-json "$OUT_JSON"
    --num-inputs "$NUM_INPUTS"
    --image-suffixes
    "${IMAGE_SUFFIXES[@]}"
    --patch-size "$PATCH_SIZE"
    --stride "$STRIDE"
    --batch-size "$BATCH_SIZE"
    --min-distance "${MIN_DISTANCE[@]}"
    --boundary-dilate-iter "${BOUNDARY_DILATE_ITER[@]}"
    --watershed-connectivity "${WATERSHED_CONNECTIVITY[@]}"
    --min-area-px "${MIN_AREA_PX[@]}"
    --exclude-border "${EXCLUDE_BORDER[@]}"
)

if [ -n "$PREDS_DIR" ]; then
    TUNE_CMD+=(--preds-dir "$PREDS_DIR")
else
    LOCAL_MODEL="$WORK_DIR/model.keras"
    cp "$MODEL_PATH" "$LOCAL_MODEL"
    TUNE_CMD+=(--model-path "$LOCAL_MODEL")
fi

echo "Running watershed tuning..."
echo "  dataset: $DATASET_DIR"
echo "  model:   $MODEL_PATH"
echo "  CSV:     $OUT_CSV"
echo "  JSON:    $OUT_JSON"
"${TUNE_CMD[@]}"

echo "Done."
