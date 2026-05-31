#!/bin/bash
#SBATCH --job-name=yolo_prof_grid
#SBATCH --output=logs/yolo_prof_grid-%j.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00

set -euo pipefail
# shellcheck source=SLURM/utils/enter_job.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/SLURM/utils/enter_job.sh"
# shellcheck source=SLURM/utils/assertions.sh
source "$SLURM_ROOT/utils/assertions.sh"

: "${OUTPUT_DIR:?OUTPUT_DIR must be set by submit script}"

GRAINSEG_ROOT="$(grainseg_root)"
RUN_ROOT="$GRAINSEG_ROOT/runs/yolo26-seg"

for variant in "${MICROSCOPY_VARIANTS[@]}"; do
    weights="$RUN_ROOT/$variant/weights/best.pt"
    require_file "$weights" "YOLO weights required before profile tune: $weights"
done

train_gpkg="$GRAINSEG_ROOT/dataset/train/train_labels.gpkg"
require_file "$train_gpkg" "Train labels GeoPackage not found"

source "$SLURM_ROOT/prepare_env.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$SLURM_ROOT/utils/variants.sh"

cd "$REPO_ROOT/src/yolo"
uv sync
export YOLO_DISABLE_TQDM=True

grid_args=(
    --output-dir "$OUTPUT_DIR"
    --grainseg-root "$GRAINSEG_ROOT"
    --run-root "$RUN_ROOT"
)

if [ "${NO_RESUME:-0}" = "1" ]; then
    grid_args+=(--no-resume)
fi

echo "Grid coordinator → $OUTPUT_DIR"
uv run python -u -m yolo.profile_tune_grid "${grid_args[@]}"

echo "Audit: $OUTPUT_DIR/grid/results.csv, $OUTPUT_DIR/grid/winner.json"
echo "Promote: uv run --directory $REPO_ROOT/src/yolo python -m yolo.promote_inference_profile --winner-json $OUTPUT_DIR/grid/winner.json"
