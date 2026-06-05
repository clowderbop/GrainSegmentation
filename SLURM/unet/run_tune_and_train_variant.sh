#!/bin/bash
#SBATCH --job-name=Train
#SBATCH --output=logs/%x-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=rtx_pro_6000:2
#SBATCH --time=12:00:00

set -euo pipefail
# shellcheck source=SLURM/utils/enter_job.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/SLURM/utils/enter_job.sh"
# shellcheck source=SLURM/utils/assertions.sh
source "$SLURM_ROOT/utils/assertions.sh"
# shellcheck source=SLURM/utils/cancel_duplicate_jobs.sh
source "$SLURM_ROOT/utils/cancel_duplicate_jobs.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$SLURM_ROOT/utils/variants.sh"
# shellcheck source=SLURM/utils/manifest_shell.sh
source "$SLURM_ROOT/utils/manifest_shell.sh"
# shellcheck source=SLURM/utils/tensorflow.sh
source "$SLURM_ROOT/utils/tensorflow.sh"

GRAINSEG_ROOT="$(grainseg_root)"

TF_STDERR_FILTER="$REPO_ROOT/SLURM/filter_tensorflow_stderr.py"
cancel_duplicate_slurm_jobs

function usage {
    local status="${1:-1}"
    cat <<EOF >&2
Usage: run_tune_and_train_variant.sh [options]

Train one U-Net input variant on the preprocessed train section. Requires
dataset/train/manifests/{variant}.whole.json (write_whole_manifests.py).

Options:
  --variant NAME         registry variant (default: VARIANT env or RUN_NAME)
  --run-name NAME        training run / checkpoint stem
  --dataset-dir PATH     (default: $GRAINSEG_ROOT/dataset/train)
  --output-model PATH
  --resume [CHECKPOINT]
  --skip-tuning
  --verbose
  --help
EOF
    exit "$status"
}

VARIANT="${VARIANT:-}"
RUN_NAME=""
DATASET_DIR=""
OUTPUT_MODEL=""
RESUME_MODEL=""
SKIP_TUNING_FLAG=""
VALIDATION_FRACTION="0.2"
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            VARIANT="$2"
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

if [ -z "$VARIANT" ]; then
    echo "VARIANT is required (--variant or VARIANT env)" >&2
    usage
fi

unet_patch_config_for_variant "$VARIANT"

RUN_NAME="${RUN_NAME:-$VARIANT}"
DATASET_DIR="${DATASET_DIR:-$GRAINSEG_ROOT/dataset/train}"

if [ -z "$OUTPUT_MODEL" ]; then
    OUTPUT_MODEL="$GRAINSEG_ROOT/models/unet/$DEFAULT_MODEL_BASENAME"
fi

source "$SLURM_ROOT/prepare_env.sh"

CANONICAL_MANIFEST="$GRAINSEG_ROOT/dataset/train/manifests/${VARIANT}.whole.json"
require_file "$CANONICAL_MANIFEST" \
    "Train whole manifest missing for $VARIANT; run write_whole_manifests.py"

LOCAL_DIR="$TMPDIR/unet_train_${RUN_NAME}_${SLURM_JOB_ID:-local}"
rm -rf "$LOCAL_DIR"
mkdir -p "$LOCAL_DIR"

export TF_CPP_MIN_LOG_LEVEL=2

echo "Syncing training environment..."
cd "$REPO_ROOT/src/unet"
uv sync

install_unet_tensorflow_wheel

echo "Staging train whole manifest to TMPDIR..."
stage_manifest_run_in_unet_env "$CANONICAL_MANIFEST" "$LOCAL_DIR"
STAGED_MANIFEST="$LOCAL_DIR/manifest.json"
require_file "$STAGED_MANIFEST" "Staged train manifest missing"

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

echo "Running training (variant=$VARIANT, run_name=$RUN_NAME, inputs=$NUM_INPUTS)..."
TRAIN_CMD=(
    uv run --no-sync python -u train_unet_multi_input.py
    --run-name "$RUN_NAME"
    --tuning-dir "$GRAINSEG_ROOT/tuning_logs"
    --manifest "$STAGED_MANIFEST"
    --validation-fraction "$VALIDATION_FRACTION"
    "${CHECKPOINT_ARGS[@]}"
    --output-model "$OUTPUT_MODEL"
    --patch-size 1024
    --split-tile-size 4096
    --patch-overlap 0.5
    --tune-epochs 50
    --num-inputs "$NUM_INPUTS"
    --mask-ext .tif
    --mask-stem-suffix _labels
)

if [ -n "$SKIP_TUNING_FLAG" ]; then
    TRAIN_CMD+=("$SKIP_TUNING_FLAG")
fi

if [ "$VERBOSE" = true ]; then
    echo "Verbose mode enabled. Raw TensorFlow/XLA stderr will be logged."
    "${TRAIN_CMD[@]}"
else
    "${TRAIN_CMD[@]}" 2> >(python -u "$TF_STDERR_FILTER" >&2)
fi

echo "Done. Wrote model to $OUTPUT_MODEL"
