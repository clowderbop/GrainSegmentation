#!/bin/bash
#SBATCH --job-name=yolo_prof_det
#SBATCH --output=logs/yolo_prof_det-%a-%j.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=00:10:00

set -euo pipefail
# shellcheck source=SLURM/utils/enter_job.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/SLURM/utils/enter_job.sh"
# shellcheck source=SLURM/utils/assertions.sh
source "$SLURM_ROOT/utils/assertions.sh"

: "${OUTPUT_DIR:?OUTPUT_DIR must be set by submit script}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set (profile tune detector array)}"

GRAINSEG_ROOT="$(grainseg_root)"
RUN_ROOT="$GRAINSEG_ROOT/runs/yolo26-seg"
GRID_CONFIG="${GRID_CONFIG:-$REPO_ROOT/configs/yolo_inference_profile_tune.yaml}"
DEVICE="${DEVICE:-0}"

source "$SLURM_ROOT/prepare_env.sh"

cd "$REPO_ROOT/src/yolo"
uv sync
export YOLO_DISABLE_TQDM=True

echo "Detector array task ${SLURM_ARRAY_TASK_ID} → $OUTPUT_DIR"
uv run python -u -m yolo.profile_tune_detector \
    --output-dir "$OUTPUT_DIR" \
    --grid-config "$GRID_CONFIG" \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    --grainseg-root "$GRAINSEG_ROOT" \
    --run-root "$RUN_ROOT" \
    --device "$DEVICE"
