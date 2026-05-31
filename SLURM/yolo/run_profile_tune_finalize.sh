#!/bin/bash
#SBATCH --job-name=yolo_prof_fin
#SBATCH --output=logs/yolo_prof_fin-%j.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

set -euo pipefail
# shellcheck source=SLURM/utils/enter_job.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/SLURM/utils/enter_job.sh"

: "${OUTPUT_DIR:?OUTPUT_DIR must be set by submit script}"
GRID_CONFIG="${GRID_CONFIG:-$REPO_ROOT/configs/yolo_inference_profile_tune.yaml}"

source "$SLURM_ROOT/prepare_env.sh"

cd "$REPO_ROOT/src/yolo"
uv sync

echo "Finalize profile selection → $OUTPUT_DIR/grid/"
uv run python -u -m yolo.profile_tune_finalize \
    --output-dir "$OUTPUT_DIR" \
    --grid-config "$GRID_CONFIG"

echo "Audit: $OUTPUT_DIR/grid/results.csv, $OUTPUT_DIR/grid/winner.json"
echo "Promote: uv run --directory $REPO_ROOT/src/yolo python -m yolo.promote_inference_profile --winner-json $OUTPUT_DIR/grid/winner.json"
