#!/bin/bash

set -euo pipefail

# shellcheck source=SLURM/utils/repo_root.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/repo_root.sh"

GRAINSEG_ROOT="$(grainseg_root)"
OUTPUT_DIR="${OUTPUT_DIR:-$GRAINSEG_ROOT/runs/yolo_inference_profile_tune}"
STAGE="${STAGE:-all}"
DRY_RUN=false

function usage {
    cat <<'EOF' >&2
Usage: submit_inference_profile_tune.sh [--dry-run] [--stage 1|2|all]

Submit staged YOLO inference profile selection on the train whole section.
Requires all four variant weights under runs/yolo26-seg/{variant}/weights/best.pt.

Run after score merge at predict is deployed; re-run when train labels or YOLO
weights change materially (not after every single-variant training job).

Environment:
  OUTPUT_DIR     scratch audit root (default: $GRAINSEG_ROOT/runs/yolo_inference_profile_tune)
  STAGE          1, 2, or all (default: all)
  STAGE1_WINNER  path to stage1/winner.json when STAGE=2
EOF
    exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            usage 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

export_vars="ALL,OUTPUT_DIR=${OUTPUT_DIR},STAGE=${STAGE}"
if [ -n "${STAGE1_WINNER:-}" ]; then
    export_vars="${export_vars},STAGE1_WINNER=${STAGE1_WINNER}"
fi

cmd=(
    sbatch
    "--export=${export_vars}"
    "$REPO_ROOT/SLURM/yolo/run_inference_profile_tune.sh"
)

if [ "$DRY_RUN" = true ]; then
    printf '%q ' "${cmd[@]}"
    echo
else
    "${cmd[@]}"
    echo "Submitted YOLO inference profile tune (stage=$STAGE) → $OUTPUT_DIR/<job_id>/"
fi
