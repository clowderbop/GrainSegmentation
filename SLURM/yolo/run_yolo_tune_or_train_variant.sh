#!/bin/bash
#SBATCH --job-name=Train_YOLO
#SBATCH --output=logs/%x-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=rtx_pro_6000:2
#SBATCH --time=03:00:00

set -euo pipefail
# shellcheck source=SLURM/utils/source_job.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/source_job.sh"
# shellcheck source=SLURM/utils/yolo_dataset.sh
source "$SLURM_ROOT/utils/yolo_dataset.sh"

function usage {
    exit 1
}

VARIANT="PPL"
VARIANT_EXPLICIT=false
DATA_YAML=""
DATA_OVERRIDE=false
RUN_NAME=""
PROJECT_DIR=""
RESUME_MODE=""
EPOCHS=""
TUNE=false
TUNE_EPOCHS=""
TUNE_ITERATIONS=""
DEVICE="0,1"
VERBOSE=false
BATCH=32
LR=""
DROPOUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            VARIANT="$2"
            VARIANT_EXPLICIT=true
            shift 2
            ;;
        --data-yaml)
            DATA_YAML="$2"
            DATA_OVERRIDE=true
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --project)
            PROJECT_DIR="$2"
            shift 2
            ;;
        --resume)
            if [[ $# -gt 1 && "${2:-}" != -* ]]; then
                RESUME_MODE="$2"
                shift 2
            else
                RESUME_MODE="__LATEST__"
                shift
            fi
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --tune)
            TUNE=true
            shift
            ;;
        --tune-epochs)
            TUNE_EPOCHS="$2"
            shift 2
            ;;
        --tune-iterations)
            TUNE_ITERATIONS="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

source "$SLURM_ROOT/prepare_env.sh"

GRAINSEG_ROOT="$(grainseg_root)"

if [[ -z "$RUN_NAME" ]]; then
    if [[ "$DATA_OVERRIDE" == true && "$VARIANT_EXPLICIT" == false ]]; then
        RUN_NAME="$(basename "${DATA_YAML%.*}")"
    else
        RUN_NAME="$VARIANT"
    fi
fi

if [[ -z "$PROJECT_DIR" ]]; then
    PROJECT_DIR="$GRAINSEG_ROOT/runs/yolo26-seg/$VARIANT"
fi

if [[ -z "$DATA_YAML" ]]; then
    stage_yolo_patch_dataset "$VARIANT" train
fi

echo "Syncing YOLO environment..."
cd "$REPO_ROOT/src/yolo"
uv sync

export YOLO_DISABLE_TQDM=True
TRAIN_CMD=(
    uv run python -u train.py
    --data "$DATA_YAML"
    --name "$RUN_NAME"
    --project "$PROJECT_DIR"
    --device "$DEVICE"
    --batch "$BATCH"
    --weights "$GRAINSEG_ROOT/pretrained/yolo26l-seg.pt"
)

if [[ -n "$EPOCHS" ]]; then
    TRAIN_CMD+=(--epochs "$EPOCHS")
fi

if [[ "$TUNE" == true ]]; then
    TRAIN_CMD+=(--tune)
fi

if [[ -n "$TUNE_EPOCHS" ]]; then
    TRAIN_CMD+=(--tune-epochs "$TUNE_EPOCHS")
fi

if [[ -n "$TUNE_ITERATIONS" ]]; then
    TRAIN_CMD+=(--tune-iterations "$TUNE_ITERATIONS")
fi

if [[ -n "$LR" ]]; then
    TRAIN_CMD+=(--lr "$LR")
fi

if [[ -n "$DROPOUT" ]]; then
    TRAIN_CMD+=(--dropout "$DROPOUT")
fi

if [[ "$DATA_OVERRIDE" == false || "$VARIANT_EXPLICIT" == true ]]; then
    TRAIN_CMD+=(--variant "$VARIANT")
fi

if [[ "$RESUME_MODE" == "__LATEST__" ]]; then
    TRAIN_CMD+=(--resume)
elif [[ -n "$RESUME_MODE" ]]; then
    TRAIN_CMD+=(--resume-checkpoint "$RESUME_MODE")
fi

if [[ "$VERBOSE" == true ]]; then
    echo "Verbose mode enabled."
fi

"${TRAIN_CMD[@]}"
