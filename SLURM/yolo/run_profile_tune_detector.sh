#!/bin/bash
#SBATCH --job-name=yolo_prof_det
#SBATCH --output=logs/yolo_prof_det-%j.log
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
: "${VARIANT:?VARIANT must be set}"
: "${CONF:?CONF must be set}"
: "${MASK_THRESHOLD:?MASK_THRESHOLD must be set}"

GRAINSEG_ROOT="$(grainseg_root)"
RUN_ROOT="$GRAINSEG_ROOT/runs/yolo26-seg"
DEVICE="${DEVICE:-0}"

weights="$RUN_ROOT/$VARIANT/weights/best.pt"
require_file "$weights" "YOLO weights required: $weights"

source "$SLURM_ROOT/prepare_env.sh"

cd "$REPO_ROOT/src/yolo"
uv sync
export YOLO_DISABLE_TQDM=True

echo "Detector cache: variant=$VARIANT conf=$CONF mask_threshold=$MASK_THRESHOLD → $OUTPUT_DIR"
uv run python -u -m yolo.profile_tune_detector \
    --output-dir "$OUTPUT_DIR" \
    --variant "$VARIANT" \
    --conf "$CONF" \
    --mask-threshold "$MASK_THRESHOLD" \
    --grainseg-root "$GRAINSEG_ROOT" \
    --run-root "$RUN_ROOT" \
    --device "$DEVICE"
