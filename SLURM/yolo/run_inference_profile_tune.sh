#!/bin/bash
#SBATCH --job-name=yolo_prof_tune
#SBATCH --output=logs/yolo_prof_tune-%j.log
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=24:00:00

set -euo pipefail
# shellcheck source=SLURM/utils/enter_job.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/SLURM/utils/enter_job.sh"
# shellcheck source=SLURM/utils/assertions.sh
source "$SLURM_ROOT/utils/assertions.sh"

GRAINSEG_ROOT="$(grainseg_root)"
RUN_ROOT="$GRAINSEG_ROOT/runs/yolo26-seg"
OUTPUT_DIR="${OUTPUT_DIR:-$GRAINSEG_ROOT/runs/yolo_inference_profile_tune/${SLURM_JOB_ID:-local}}"
STAGE="${STAGE:-all}"
DEVICE="${DEVICE:-0}"

source "$SLURM_ROOT/prepare_env.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$SLURM_ROOT/utils/variants.sh"

for variant in "${MICROSCOPY_VARIANTS[@]}"; do
    weights="$RUN_ROOT/$variant/weights/best.pt"
    require_file "$weights" "YOLO weights required before profile tune: $weights"
done

train_gpkg="$GRAINSEG_ROOT/dataset/train/train_labels.gpkg"
require_file "$train_gpkg" "Train labels GeoPackage not found"

cd "$REPO_ROOT/src/yolo"
uv sync

export YOLO_DISABLE_TQDM=True

tune_args=(
    --output-dir "$OUTPUT_DIR"
    --grainseg-root "$GRAINSEG_ROOT"
    --run-root "$RUN_ROOT"
    --stage "$STAGE"
    --device "$DEVICE"
)

if [ -n "${STAGE1_WINNER:-}" ]; then
    tune_args+=(--stage1-winner "$STAGE1_WINNER")
fi

echo "YOLO inference profile tune → $OUTPUT_DIR (stage=$STAGE)"
uv run python -u -m yolo.tune_inference_profile "${tune_args[@]}"

echo "Audit artifacts: $OUTPUT_DIR/stage1/results.csv, stage2/ (if --stage all)"
echo "Promote winner: uv run --directory $REPO_ROOT/src/yolo python -m yolo.promote_inference_profile --winner-json $OUTPUT_DIR/stage2/winner.json"
