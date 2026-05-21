#!/bin/bash
#SBATCH --job-name=Train
#SBATCH --output=logs/%x-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=rtx_pro_6000:2
#SBATCH --time=12:00:00

set -euo pipefail
# shellcheck source=SLURM/utils/source_job.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/source_job.sh"
# shellcheck source=SLURM/utils/assertions.sh
source "$SLURM_ROOT/utils/assertions.sh"
# shellcheck source=SLURM/utils/cancel_duplicate_jobs.sh
source "$SLURM_ROOT/utils/cancel_duplicate_jobs.sh"
# shellcheck source=SLURM/utils/tensorflow.sh
source "$SLURM_ROOT/utils/tensorflow.sh"

GRAINSEG_ROOT="$(grainseg_root)"

TF_STDERR_FILTER="$REPO_ROOT/SLURM/filter_tensorflow_stderr.py"
cancel_duplicate_slurm_jobs

function usage {
    local status="${1:-1}"
    cat <<EOF >&2
Usage: run_unet_tune_and_train_variant.sh [options]

Train one U-Net input variant on the preprocessed train section. Expects
dataset/train/train_labels.tif (from SLURM/preprocessing/rasterize_labels.sh)
and per-channel train_*.tif mosaics from the preprocessing pipeline.

Options:
  --num-inputs N
  --image-suffixes "_PPL _PPX1 ..."
  --run-name NAME
  --dataset-dir PATH   (default: $GRAINSEG_ROOT/dataset/train)
  --output-model PATH
  --resume [CHECKPOINT]
  --skip-tuning
  --verbose
  --help
EOF
    exit "$status"
}

NUM_INPUTS=7
IMAGE_SUFFIXES="_PPL _PPX1 _PPX2 _PPX3 _PPX4 _PPX5 _PPX6"
RUN_NAME="7in_PPL_AllPPX"
DATASET_DIR=""
OUTPUT_MODEL=""
RESUME_MODEL=""
SKIP_TUNING_FLAG=""
VALIDATION_FRACTION="0.2"
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-inputs)
            NUM_INPUTS="$2"
            shift 2
            ;;
        --image-suffixes)
            IMAGE_SUFFIXES="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --dataset-dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --output-model)
            OUTPUT_MODEL="$2"
            shift 2
            ;;
        --resume)
            if [[ $# -gt 1 && "${2:-}" != -* ]]; then
                RESUME_MODEL="$2"
                shift 2
            else
                RESUME_MODEL="__LATEST__"
                shift
            fi
            ;;
        --skip-tuning)
            SKIP_TUNING_FLAG="--skip-tuning"
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
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

source "$SLURM_ROOT/prepare_env.sh"

DATASET_DIR="${DATASET_DIR:-$GRAINSEG_ROOT/dataset/train}"
LABELS_RASTER="$DATASET_DIR/train_labels.tif"

if [ -z "$OUTPUT_MODEL" ]; then
    OUTPUT_MODEL="$GRAINSEG_ROOT/models/unet/unet_finetuned_${RUN_NAME}.keras"
fi

read -r -a IMAGE_SUFFIX_ARGS <<< "$IMAGE_SUFFIXES"

require_file "$LABELS_RASTER" \
    "Semantic label raster not found (run SLURM/preprocessing/rasterize_labels.sh)"

echo "Staging train mosaics to TMPDIR..."
LOCAL_DIR="$TMPDIR/unet_train_${RUN_NAME}_${SLURM_JOB_ID:-local}"
rm -rf "$LOCAL_DIR"
mkdir -p "$LOCAL_DIR"
cp "$LABELS_RASTER" "$LOCAL_DIR/"

for suffix in "${IMAGE_SUFFIX_ARGS[@]}"; do
    src="$DATASET_DIR/train${suffix}.tif"
    require_file "$src" "Missing training image for suffix ${suffix}"
    cp "$src" "$LOCAL_DIR/"
done

export TF_CPP_MIN_LOG_LEVEL=2

echo "Syncing training environment..."
cd "$REPO_ROOT/src/unet"
uv sync

install_unet_tensorflow_wheel

LATEST_MODEL="${OUTPUT_MODEL%.keras}_latest.keras"

if [ "$RESUME_MODEL" = "__LATEST__" ]; then
    RESUME_MODEL="$LATEST_MODEL"
fi

if [ -n "$RESUME_MODEL" ]; then
    require_file "$RESUME_MODEL" "Resume checkpoint not found"
    CHECKPOINT_ARGS=("--resume" "$RESUME_MODEL")
    echo "Resuming final training from: $RESUME_MODEL"
else
    PRETRAINED_CHECKPOINT="$GRAINSEG_ROOT/models/unet/pretrained/starting_point.keras"
    if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
        PRETRAINED_CHECKPOINT="$REPO_ROOT/models/pretrained/starting_point.keras"
    fi
    require_file "$PRETRAINED_CHECKPOINT" "Pretrained starting checkpoint not found"
    CHECKPOINT_ARGS=("--checkpoint" "$PRETRAINED_CHECKPOINT")
fi

echo "Running training (run_name=$RUN_NAME, inputs=$NUM_INPUTS)..."
TRAIN_CMD=(uv run --no-sync python -u train_unet_multi_input.py)

if [ -n "$SKIP_TUNING_FLAG" ]; then
    TRAIN_CMD+=("$SKIP_TUNING_FLAG")
fi

TRAIN_CMD+=(
    --run-name "$RUN_NAME"
    --tuning-dir "$GRAINSEG_ROOT/tuning_logs"
    --image-dir "$LOCAL_DIR"
    --mask-dir "$LOCAL_DIR"
    --validation-fraction "$VALIDATION_FRACTION"
    "${CHECKPOINT_ARGS[@]}"
    --output-model "$OUTPUT_MODEL"
    --patch-size 1024
    --split-tile-size 4096
    --patch-overlap 0.5
    --tune-epochs 50
    --num-inputs "$NUM_INPUTS"
    --image-suffixes
    "${IMAGE_SUFFIX_ARGS[@]}"
    --mask-ext .tif
    --mask-stem-suffix _labels
)

if [ "$VERBOSE" = true ]; then
    echo "Verbose mode enabled. Raw TensorFlow/XLA stderr will be logged."
    "${TRAIN_CMD[@]}"
else
    "${TRAIN_CMD[@]}" 2> >(python -u "$TF_STDERR_FILTER" >&2)
fi

echo "Done. Wrote model to $OUTPUT_MODEL"
