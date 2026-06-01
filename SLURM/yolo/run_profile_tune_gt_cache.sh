#!/bin/bash
# Profile selection ground truth cache (ADR 0006): OpenCV rasterize train_labels.gpkg
# → OUTPUT_DIR/_work/gt_cache/train/; sync src/common only (GPKG copy + timings in CLI).
#SBATCH --job-name=yolo_prof_gt
#SBATCH --output=logs/yolo_prof_gt-%j.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -euo pipefail
# shellcheck source=SLURM/utils/enter_job.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/SLURM/utils/enter_job.sh"
# shellcheck source=SLURM/utils/assertions.sh
source "$SLURM_ROOT/utils/assertions.sh"

: "${OUTPUT_DIR:?OUTPUT_DIR must be set by submit script}"

GRAINSEG_ROOT="$(grainseg_root)"

train_gpkg="$GRAINSEG_ROOT/dataset/train/train_labels.gpkg"
require_file "$train_gpkg" "Train labels GeoPackage not found"

source "$SLURM_ROOT/prepare_env.sh"

cd "$REPO_ROOT/src/common"
echo "[$(date -Is)] uv sync (common only) …"
uv sync
echo "[$(date -Is)] uv sync done"

echo "GT cache → $OUTPUT_DIR/_work/gt_cache/train/"
echo "  train_labels: $train_gpkg"
echo "  grainseg_root: $GRAINSEG_ROOT"
uv run python -u -m common.profile_tune_gt_cache \
    --output-dir "$OUTPUT_DIR" \
    --grainseg-root "$GRAINSEG_ROOT"
