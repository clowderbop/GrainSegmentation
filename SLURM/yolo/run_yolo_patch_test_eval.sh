#!/bin/bash
#SBATCH --job-name=test_yolo_patches
#SBATCH --output=logs/test_yolo_patches-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=04:00:00

set -euo pipefail
# shellcheck source=SLURM/utils/source_job.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/source_job.sh"
# shellcheck source=SLURM/utils/yolo_dataset.sh
source "$SLURM_ROOT/utils/yolo_dataset.sh"

GRAINSEG_ROOT="$(grainseg_root)"

VARIANT="${VARIANT:-PPL}"
DEVICE="0"
IMGSZ=1024
BATCH=16
RUN_NAME="test"
PROJECT_DIR="$GRAINSEG_ROOT/runs/yolo26-seg-val/$VARIANT"
JOB_TAG="${SLURM_JOB_ID:-local}"
OUT_ROOT="${OUTPUT_ROOT:-$GRAINSEG_ROOT/eval/yolo_patches/$VARIANT/$JOB_TAG}"
INSTANCE_METRICS_JSON="$OUT_ROOT/instance_metrics.json"
RUN_ULTRALYTICS_VAL="${RUN_ULTRALYTICS_VAL:-0}"

source "$SLURM_ROOT/prepare_env.sh"

WEIGHTS="$GRAINSEG_ROOT/runs/yolo26-seg/$VARIANT/weights/best.pt"
stage_yolo_patch_dataset "$VARIANT" test

echo "Syncing YOLO environment..."
cd "$REPO_ROOT/src/yolo"
uv sync

export YOLO_DISABLE_TQDM=True

mkdir -p "$OUT_ROOT"

echo "1/2 yolo.predict (instance TIFFs and mask NPZs)..."
uv run python -u -m yolo.predict \
    --unit patch \
    --weights "$WEIGHTS" \
    --variant "$VARIANT" \
    --data "$DATA_YAML" \
    --device "$DEVICE" \
    --imgsz "$IMGSZ" \
    --conf "${CONF:-0.25}" \
    --output-dir "$OUT_ROOT"

DATASET_ROOT="$(dirname "$DATA_YAML")"
IMAGE_DIR="$DATASET_ROOT/images/test"
LABEL_DIR="$DATASET_ROOT/labels/test"

echo "2/2 common.evaluate_instances..."
uv run python -u -m common.evaluate_instances \
    --unit patch \
    --model-type yolo \
    --variant "$VARIANT" \
    --image-dir "$IMAGE_DIR" \
    --pred-instances-dir "$OUT_ROOT/instances" \
    --gt-labels-dir "$LABEL_DIR" \
    --output-json "$INSTANCE_METRICS_JSON"

if [[ "$RUN_ULTRALYTICS_VAL" == "1" ]]; then
    echo "Optional: yolo.val (Ultralytics test split metrics)..."
    uv run python -u -m yolo.val \
        --weights "$WEIGHTS" \
        --variant "$VARIANT" \
        --data "$DATA_YAML" \
        --device "$DEVICE" \
        --imgsz "$IMGSZ" \
        --batch "$BATCH" \
        --name "$RUN_NAME" \
        --project "$PROJECT_DIR"
fi

echo "Wrote $INSTANCE_METRICS_JSON"
