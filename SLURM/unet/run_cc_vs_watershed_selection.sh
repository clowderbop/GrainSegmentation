#!/bin/bash
#SBATCH --job-name=cc_ws_select
#SBATCH --output=logs/%x-%j.log
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
# Select CC vs tuned watershed on train using whole-section PQ (ADR 0003).
# Ops: docs/runbooks/unet.md#cc-vs-watershed-train-section

set -euo pipefail
# shellcheck source=SLURM/utils/enter_job.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/SLURM/utils/enter_job.sh"
mkdir -p "$REPO_ROOT/logs"

GRAINSEG_ROOT="$(grainseg_root)"
EVAL_ROOT="$GRAINSEG_ROOT/eval"
CC_EVAL_DIR="$EVAL_ROOT/instance_val_cc"
WATERSHED_EVAL_DIR="$EVAL_ROOT/instance_val_watershed"
OUTPUT_JSON="$EVAL_ROOT/extraction_method_selection.json"

usage() {
    cat <<'EOF' >&2
Usage: run_cc_vs_watershed_selection.sh [options]

Options:
  --cc-eval-dir DIR          (default: $GRAINSEG_ROOT/eval/instance_val_cc)
  --watershed-eval-dir DIR   (default: $GRAINSEG_ROOT/eval/instance_val_watershed)
  --output-json PATH         (default: $GRAINSEG_ROOT/eval/extraction_method_selection.json)
EOF
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --cc-eval-dir)
            CC_EVAL_DIR="$2"
            shift 2
            ;;
        --watershed-eval-dir)
            WATERSHED_EVAL_DIR="$2"
            shift 2
            ;;
        --output-json)
            OUTPUT_JSON="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            ;;
    esac
done

if [ ! -d "$CC_EVAL_DIR" ]; then
    echo "CC eval dir not found: $CC_EVAL_DIR" >&2
    exit 1
fi
if [ ! -d "$WATERSHED_EVAL_DIR" ]; then
    echo "Watershed eval dir not found: $WATERSHED_EVAL_DIR" >&2
    exit 1
fi

cd "$REPO_ROOT/src/unet"
uv sync

echo "Selecting extraction method from train whole-section PQ..."
uv run --no-sync python -u -m unet.select_extraction_method \
    --cc-eval-dir "$CC_EVAL_DIR" \
    --watershed-eval-dir "$WATERSHED_EVAL_DIR" \
    --output-json "$OUTPUT_JSON"

echo "Wrote train extraction method selection to $OUTPUT_JSON"
