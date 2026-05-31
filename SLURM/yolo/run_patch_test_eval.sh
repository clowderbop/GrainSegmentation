#!/bin/bash
#SBATCH --job-name=test_yolo_patches
#SBATCH --output=logs/test_yolo_patches-%j.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=00:08:00

set -euo pipefail
# shellcheck source=SLURM/utils/enter_job.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/SLURM/utils/enter_job.sh"
# shellcheck source=SLURM/utils/assertions.sh
source "$SLURM_ROOT/utils/assertions.sh"
# shellcheck source=SLURM/utils/yolo_dataset.sh
source "$SLURM_ROOT/utils/yolo_dataset.sh"
# shellcheck source=SLURM/utils/test_inference.sh
source "$SLURM_ROOT/utils/test_inference.sh"

GRAINSEG_ROOT="$(grainseg_root)"

VARIANT="${VARIANT:-PPL}"
DEVICE="0"
load_test_inference_exports
IMGSZ="$TEST_PATCH_IMGSZ"
BATCH="$YOLO_PATCH_BATCH"
RUN_NAME="test"
PROJECT_DIR="$GRAINSEG_ROOT/runs/yolo26-seg-val/$VARIANT"
JOB_TAG="${SLURM_JOB_ID:-local}"
OUT_ROOT="${OUTPUT_ROOT:-$GRAINSEG_ROOT/eval/yolo_patches/$VARIANT/$JOB_TAG}"
INSTANCE_METRICS_JSON="$OUT_ROOT/instance_metrics.json"

PATCH_MANIFEST="$GRAINSEG_ROOT/dataset/test/patches/$VARIANT/manifest.json"
require_file "$PATCH_MANIFEST" \
    "YOLO patch manifest not found at $PATCH_MANIFEST (run write_patch_manifests.py)"

source "$SLURM_ROOT/prepare_env.sh"

WEIGHTS="$GRAINSEG_ROOT/runs/yolo26-seg/$VARIANT/weights/best.pt"
require_file "$WEIGHTS" "YOLO weights not found"

stage_yolo_patch_dataset "$VARIANT" test

echo "Syncing YOLO environment..."
cd "$REPO_ROOT/src/yolo"
uv sync

export YOLO_DISABLE_TQDM=True

WORK_ROOT="$TMPDIR/yolo_patch_eval_${VARIANT}_$JOB_TAG"
STAGED_PATCH="$WORK_ROOT/patch_manifest"
rm -rf "$WORK_ROOT"
mkdir -p "$OUT_ROOT" "$WORK_ROOT"

echo "Staging YOLO patch manifest to TMPDIR..."
uv run --directory "$REPO_ROOT" python -m common.stage_manifest run \
    "$PATCH_MANIFEST" "$STAGED_PATCH"
STAGED_MANIFEST="$STAGED_PATCH/manifest.json"
require_file "$STAGED_MANIFEST" "Staged patch manifest missing"

echo "1/3 yolo.predict (instance prediction sets)..."
uv run python -u -m yolo.predict \
    --unit patch \
    --weights "$WEIGHTS" \
    --manifest "$STAGED_MANIFEST" \
    --device "$DEVICE" \
    --imgsz "$IMGSZ" \
    --conf "${CONF:-$YOLO_CONF}" \
    --output-dir "$OUT_ROOT"

EVAL_MANIFEST="$OUT_ROOT/eval_manifest.json"
echo "Building eval manifest..."
uv run --directory "$REPO_ROOT" python -m common.stage_manifest write-eval \
    --source "$STAGED_MANIFEST" \
    --prediction-set-dir "$OUT_ROOT" \
    --output "$EVAL_MANIFEST"

echo "2/3 common.evaluate_instances..."
uv run python -u -m common.evaluate_instances \
    --unit patch \
    --model-type yolo \
    --variant "$VARIANT" \
    --manifest "$EVAL_MANIFEST" \
    --output-json "$INSTANCE_METRICS_JSON"

echo "3/3 yolo.val (Ultralytics test split metrics)..."
uv run python -u -m yolo.val \
    --weights "$WEIGHTS" \
    --variant "$VARIANT" \
    --data "$DATA_YAML" \
    --device "$DEVICE" \
    --imgsz "$YOLO_VAL_IMGSZ" \
    --batch "$YOLO_VAL_BATCH" \
    --name "$RUN_NAME" \
    --project "$PROJECT_DIR"

echo "Wrote $INSTANCE_METRICS_JSON"
