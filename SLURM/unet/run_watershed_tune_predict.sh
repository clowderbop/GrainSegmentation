#!/bin/bash
#SBATCH --job-name=PredWatershed
#SBATCH --output=logs/%x-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=04:00:00

set -euo pipefail
# shellcheck source=SLURM/utils/enter_job.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/SLURM/utils/enter_job.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$SLURM_ROOT/utils/variants.sh"
# shellcheck source=SLURM/utils/manifest_shell.sh
source "$SLURM_ROOT/utils/manifest_shell.sh"
# shellcheck source=SLURM/utils/tensorflow.sh
source "$SLURM_ROOT/utils/tensorflow.sh"
mkdir -p "$REPO_ROOT/logs"

GRAINSEG_ROOT="$(grainseg_root)"
DATASET_DIR="${DATASET_DIR:-$GRAINSEG_ROOT/dataset/train}"
MODEL_PATH="$GRAINSEG_ROOT/models/unet/unet_finetuned_PPL+AllPPX.keras"
PREDS_ROOT="$GRAINSEG_ROOT/runs/watershed_tune_preds"

PATCH_SIZE=1024
STRIDE=512
BATCH_SIZE=1

function usage {
    local status="${1:-1}"
    cat <<EOF >&2
Usage: run_watershed_tune_predict.sh [options]

Write durable train whole-section semantic predictions for watershed tuning.
Requires dataset/train/manifests/{variant}.whole.json and a finetuned model.

Options:
  --variant NAME         registry variant (default: VARIANT env)
  --model-path PATH
  --dataset-dir PATH
  --preds-root PATH      scratch root for cached preds (default: runs/watershed_tune_preds)
  --help
EOF
    exit "$status"
}

VARIANT="${VARIANT:-PPL+AllPPX}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset-dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --preds-root)
            PREDS_ROOT="$2"
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

unet_patch_config_for_variant "$VARIANT"

CANONICAL_MANIFEST="$GRAINSEG_ROOT/dataset/train/manifests/${VARIANT}.whole.json"
require_file "$CANONICAL_MANIFEST" \
    "Train whole manifest missing for $VARIANT; run write_whole_manifests.py"
require_file "$MODEL_PATH" "Model not found: $MODEL_PATH"
require_dir "$DATASET_DIR" "Dataset dir not found: $DATASET_DIR"

if [ ! -f "$SLURM_ROOT/prepare_env.sh" ]; then
    echo "prepare_env.sh not found at: $SLURM_ROOT/prepare_env.sh" >&2
    exit 1
fi
source "$SLURM_ROOT/prepare_env.sh"
export TF_CPP_MIN_LOG_LEVEL=2

WORK_DIR="${TMPDIR:-/tmp}/pred_watershed_${SLURM_JOB_ID:-$$}"
mkdir -p "$WORK_DIR"
LOCAL_IMAGE_DIR="$WORK_DIR/dataset"

cd "$REPO_ROOT/src/unet"
echo "Syncing U-Net environment..."
uv sync

install_unet_tensorflow_wheel

echo "Staging train whole manifest to $LOCAL_IMAGE_DIR ..."
stage_manifest_run_in_unet_env "$CANONICAL_MANIFEST" "$LOCAL_IMAGE_DIR"
STAGED_MANIFEST="$LOCAL_IMAGE_DIR/manifest.json"
require_file "$STAGED_MANIFEST" "Staged train manifest missing"

WATERSHED_SUBDIR="$(watershed_tune_subdir_for_variant "$VARIANT")"
VARIANT_PREDS_DIR="$PREDS_ROOT/$WATERSHED_SUBDIR"
mkdir -p "$VARIANT_PREDS_DIR"
LOCAL_MODEL="$WORK_DIR/model.keras"
cp "$MODEL_PATH" "$LOCAL_MODEL"

echo "Predicting train whole-section semantics (variant=$VARIANT, subdir=$WATERSHED_SUBDIR)..."
echo "  model:   $MODEL_PATH"
echo "  output:  $VARIANT_PREDS_DIR/semantic/"
uv run --no-sync python -u -m unet.predict \
    --model-path "$LOCAL_MODEL" \
    --manifest "$STAGED_MANIFEST" \
    --output-dir "$VARIANT_PREDS_DIR" \
    --num-inputs "$NUM_INPUTS" \
    --patch-size "$PATCH_SIZE" \
    --stride "$STRIDE" \
    --batch-size "$BATCH_SIZE" \
    --unit whole \
    --variant "$VARIANT"

echo "Done."
