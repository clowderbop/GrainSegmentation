#!/bin/bash
#SBATCH --job-name=yolo_prof_cand
#SBATCH --output=logs/yolo_prof_cand-%a-%j.log
#SBATCH --mem=50G
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00

set -euo pipefail
# shellcheck source=SLURM/utils/enter_job.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/SLURM/utils/enter_job.sh"
# shellcheck source=SLURM/utils/assertions.sh
source "$SLURM_ROOT/utils/assertions.sh"

: "${OUTPUT_DIR:?OUTPUT_DIR must be set by submit script}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set (profile tune candidate array)}"

GRAINSEG_ROOT="$(grainseg_root)"
RUN_ROOT="$GRAINSEG_ROOT/runs/yolo26-seg"
GRID_CONFIG="${GRID_CONFIG:-$REPO_ROOT/configs/yolo_inference_profile_tune.yaml}"

source "$SLURM_ROOT/prepare_env.sh"
# shellcheck source=SLURM/utils/yolo_venv.sh
source "$SLURM_ROOT/utils/yolo_venv.sh"
yolo_venv_stage_local

cd "$REPO_ROOT/src/yolo"
export YOLO_DISABLE_TQDM=True

WORK_ROOT="$TMPDIR/profile_tune_cand_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID}"
rm -rf "$WORK_ROOT"
mkdir -p "$WORK_ROOT"

echo "Staging candidate caches from ${OUTPUT_DIR}/.cache to TMPDIR..."
stage_args=(
    candidate
    --output-dir "$OUTPUT_DIR"
    --tmp-work-root "$WORK_ROOT"
    --grid-config "$GRID_CONFIG"
    --array-index "$SLURM_ARRAY_TASK_ID"
)
uv run --no-sync python -u -m yolo.profile_tune_cache_stage "${stage_args[@]}"

candidate_args=(
    --output-dir "$OUTPUT_DIR"
    --grainseg-root "$GRAINSEG_ROOT"
    --run-root "$RUN_ROOT"
    --grid-config "$GRID_CONFIG"
    --array-index "$SLURM_ARRAY_TASK_ID"
    --work-root "$WORK_ROOT"
)

if [ "${NO_RESUME:-0}" = "1" ]; then
    candidate_args+=(--no-resume)
fi

echo "Candidate array task ${SLURM_ARRAY_TASK_ID} → $OUTPUT_DIR (work_root=$WORK_ROOT)"
uv run --no-sync python -u -m yolo.profile_tune_candidate "${candidate_args[@]}"
