#!/bin/bash

set -euo pipefail

# shellcheck source=SLURM/utils/repo_root.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/repo_root.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/variants.sh"

GRAINSEG_ROOT="$(grainseg_root)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$GRAINSEG_ROOT/runs/yolo_inference_profile_tune/$RUN_ID}"
DRY_RUN=false
GRID_CONFIG="${GRID_CONFIG:-$REPO_ROOT/configs/yolo_inference_profile_tune.yaml}"

function usage {
    cat <<'EOF' >&2
Usage: submit_inference_profile_tune.sh [--dry-run] [--output-dir PATH] [--run-id ID]

Submit parallel YOLO inference profile tune (ADR 0005):
  one GPU detector job per (variant, conf, mask_threshold), then one CPU grid job.

Requires all registry variant weights under runs/yolo26-seg/{variant}/weights/best.pt.

Environment:
  OUTPUT_DIR   full run directory (default: .../yolo_inference_profile_tune/<run_id>)
  RUN_ID       run folder name when OUTPUT_DIR unset
  GRID_CONFIG  search grid YAML (default: configs/yolo_inference_profile_tune.yaml)
  NO_RESUME    set to 1 to pass --no-resume to the grid coordinator
EOF
    exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
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

if [ ! -f "$GRID_CONFIG" ]; then
    echo "Grid config not found: $GRID_CONFIG" >&2
    exit 1
fi

detector_job_ids=()
while IFS=$'\t' read -r variant conf mask; do
    export_vars="ALL,OUTPUT_DIR=${OUTPUT_DIR},VARIANT=${variant},CONF=${conf},MASK_THRESHOLD=${mask}"
    cmd=(
        sbatch
        "--export=${export_vars}"
        "$REPO_ROOT/SLURM/yolo/run_profile_tune_detector.sh"
    )
    if [ "$DRY_RUN" = true ]; then
        printf '%q ' "${cmd[@]}"
        echo
    else
        job_id="$("${cmd[@]}" | awk '{print $NF}')"
        detector_job_ids+=("$job_id")
    fi
done < <(
    uv run --directory "$REPO_ROOT/src/yolo" python -m yolo.profile_tune_list_detector_jobs \
        --grid-config "$GRID_CONFIG"
)

dep=""
if [ "$DRY_RUN" = false ] && [ "${#detector_job_ids[@]}" -gt 0 ]; then
    dep=$(IFS=:; echo "${detector_job_ids[*]}")
fi

grid_export="ALL,OUTPUT_DIR=${OUTPUT_DIR}"
if [ -n "${NO_RESUME:-}" ]; then
    grid_export="${grid_export},NO_RESUME=${NO_RESUME}"
fi

grid_cmd=(
    sbatch
    "--export=${grid_export}"
)
if [ -n "$dep" ]; then
    grid_cmd+=("--dependency=afterok:${dep}")
fi
grid_cmd+=("$REPO_ROOT/SLURM/yolo/run_profile_tune_grid.sh")

if [ "$DRY_RUN" = true ]; then
    printf '%q ' "${grid_cmd[@]}"
    echo
else
    "${grid_cmd[@]}"
    echo "Submitted profile tune run → $OUTPUT_DIR"
    echo "  ${#detector_job_ids[@]} detector GPU jobs + 1 grid coordinator (afterok)"
    echo "Promote: uv run --directory $REPO_ROOT/src/yolo python -m yolo.promote_inference_profile --winner-json $OUTPUT_DIR/grid/winner.json"
fi
