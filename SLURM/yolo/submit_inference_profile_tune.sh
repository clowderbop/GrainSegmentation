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
SKIP_DETECTORS="${SKIP_DETECTORS:-0}"
GRID_CONFIG="${GRID_CONFIG:-$REPO_ROOT/configs/yolo_inference_profile_tune.yaml}"

function usage {
    cat <<'EOF' >&2
Usage: submit_inference_profile_tune.sh [--dry-run] [--output-dir PATH] [--run-id ID] [--skip-detectors]

Submit parallel YOLO inference profile selection on the train whole section.
Cluster workflow, resources, and promotion: docs/runbooks/yolo.md#profile-selection

Requires all registry variant weights under runs/yolo26-seg/{variant}/weights/best.pt
when running detectors.

--skip-detectors  Skip the detector array when this OUTPUT_DIR already has valid v2 _work/
                  (same run). GT cache, venv prep, candidates, and finalize still run.

Environment:
  OUTPUT_DIR       full run directory (default: .../yolo_inference_profile_tune/<run_id>)
  RUN_ID           run folder name when OUTPUT_DIR unset
  GRID_CONFIG      search grid YAML (default: configs/yolo_inference_profile_tune.yaml)
  SKIP_DETECTORS   set to 1 (or use --skip-detectors) to skip detector array
  DETECTOR_MAX_PARALLEL  max concurrent detector array tasks (default: 6)
  NO_RESUME        set to 1 to clear grid/rows/*.json before candidate array and pass --no-resume
  SHARED_VENV_ROOT   set by submit to $SCRATCH/.venvs/yolo-profile-tune/<uv.lock-sha256>/
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
        --skip-detectors)
            SKIP_DETECTORS=1
            shift
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

candidate_count="$(
    uv run --directory "$REPO_ROOT/src/yolo" python -m yolo.profile_tune_list_candidates \
        --grid-config "$GRID_CONFIG" | wc -l
)"
candidate_count="${candidate_count//[[:space:]]/}"
if [ "$candidate_count" -lt 1 ]; then
    echo "No grid candidates in $GRID_CONFIG" >&2
    exit 1
fi

RUN_ROOT="$GRAINSEG_ROOT/runs/yolo26-seg"
for variant in "${MICROSCOPY_VARIANTS[@]}"; do
    weights="$RUN_ROOT/$variant/weights/best.pt"
    if [ ! -f "$weights" ]; then
        echo "Missing YOLO weights (required for profile tune): $weights" >&2
        exit 1
    fi
done

detector_job_id=""
detector_max_parallel="${DETECTOR_MAX_PARALLEL:-6}"
if [ "$SKIP_DETECTORS" != "1" ]; then
    detector_count="$(
        uv run --directory "$REPO_ROOT/src/yolo" python -m yolo.profile_tune_list_detector_jobs \
            --grid-config "$GRID_CONFIG" | wc -l
    )"
    detector_count="${detector_count//[[:space:]]/}"
    if [ "$detector_count" -lt 1 ]; then
        echo "No detector jobs in $GRID_CONFIG" >&2
        exit 1
    fi

    det_export="ALL,OUTPUT_DIR=${OUTPUT_DIR},GRID_CONFIG=${GRID_CONFIG}"
    det_cmd=(
        sbatch
        "--export=${det_export}"
        "--array=1-${detector_count}%${detector_max_parallel}"
        "$REPO_ROOT/SLURM/yolo/run_profile_tune_detector.sh"
    )
    if [ "$DRY_RUN" = true ]; then
        printf '%q ' "${det_cmd[@]}"
        echo
    else
        detector_job_id="$("${det_cmd[@]}" | awk '{print $NF}')"
        if [ -z "${detector_job_id:-}" ]; then
            echo "Detector array sbatch did not return a job id" >&2
            exit 1
        fi
    fi
else
    echo "Skipping detector jobs (SKIP_DETECTORS=1); using existing _work/ in $OUTPUT_DIR"
fi

detector_dep=""
if [ "$DRY_RUN" = false ] && [ -n "$detector_job_id" ]; then
    detector_dep="$detector_job_id"
fi

gt_export="ALL,OUTPUT_DIR=${OUTPUT_DIR},GRID_CONFIG=${GRID_CONFIG}"
gt_cmd=(sbatch "--export=${gt_export}")
if [ -n "$detector_dep" ]; then
    gt_cmd+=("--dependency=afterok:${detector_dep}")
fi
gt_cmd+=("$REPO_ROOT/SLURM/yolo/run_profile_tune_gt_cache.sh")

if [ "$DRY_RUN" = true ]; then
    printf '%q ' "${gt_cmd[@]}"
    echo
else
    gt_job_id="$("${gt_cmd[@]}" | awk '{print $NF}')"
    if [ -z "${gt_job_id:-}" ]; then
        echo "GT cache sbatch did not return a job id" >&2
        exit 1
    fi
fi

if [ "${NO_RESUME:-0}" = "1" ] && [ "$DRY_RUN" = false ]; then
    OUTPUT_DIR="$OUTPUT_DIR" uv run --directory "$REPO_ROOT/src/yolo" python - <<'PY'
from pathlib import Path
import os
from yolo.inference_profile_tune import clear_profile_selection_rows

clear_profile_selection_rows(Path(os.environ["OUTPUT_DIR"]) / "grid")
PY
fi

# shellcheck source=SLURM/utils/yolo_venv.sh
source "$REPO_ROOT/SLURM/utils/yolo_venv.sh"
SHARED_VENV_ROOT="$(yolo_venv_shared_root)"

venv_export="ALL,OUTPUT_DIR=${OUTPUT_DIR},GRID_CONFIG=${GRID_CONFIG},SHARED_VENV_ROOT=${SHARED_VENV_ROOT}"
venv_cmd=(sbatch "--export=${venv_export}")
if [ "$DRY_RUN" = false ]; then
    venv_cmd+=("--dependency=afterok:${gt_job_id}")
fi
venv_cmd+=("$REPO_ROOT/SLURM/yolo/run_profile_tune_venv_prep.sh")

if [ "$DRY_RUN" = true ]; then
    printf '%q ' "${venv_cmd[@]}"
    echo
else
    venv_prep_job_id="$("${venv_cmd[@]}" | awk '{print $NF}')"
    if [ -z "${venv_prep_job_id:-}" ]; then
        echo "Venv prep sbatch did not return a job id" >&2
        exit 1
    fi
fi

cand_export="ALL,OUTPUT_DIR=${OUTPUT_DIR},GRID_CONFIG=${GRID_CONFIG},SHARED_VENV_ROOT=${SHARED_VENV_ROOT}"
if [ -n "${NO_RESUME:-}" ]; then
    cand_export="${cand_export},NO_RESUME=${NO_RESUME}"
fi
cand_cmd=(
    sbatch
    "--export=${cand_export}"
    "--array=1-${candidate_count}"
)
# sbatch requires all options before the script path (otherwise --dependency is ignored).
if [ "$DRY_RUN" = false ]; then
    cand_cmd+=("--dependency=afterok:${venv_prep_job_id}")
fi
cand_cmd+=("$REPO_ROOT/SLURM/yolo/run_profile_tune_candidate.sh")

if [ "$DRY_RUN" = true ]; then
    printf '%q ' "${cand_cmd[@]}"
    echo
else
    cand_job_id="$("${cand_cmd[@]}" | awk '{print $NF}')"
fi

fin_export="ALL,OUTPUT_DIR=${OUTPUT_DIR},GRID_CONFIG=${GRID_CONFIG},SHARED_VENV_ROOT=${SHARED_VENV_ROOT}"
fin_cmd=(sbatch "--export=${fin_export}")
if [ "$DRY_RUN" = false ]; then
    fin_cmd+=("--dependency=afterok:${cand_job_id}")
fi
fin_cmd+=("$REPO_ROOT/SLURM/yolo/run_profile_tune_finalize.sh")

if [ "$DRY_RUN" = true ]; then
    printf '%q ' "${fin_cmd[@]}"
    echo
else
    "${fin_cmd[@]}"
    echo "Submitted profile tune run → $OUTPUT_DIR"
    if [ "$SKIP_DETECTORS" = "1" ]; then
        echo "  detectors skipped (same OUTPUT_DIR); GT cache + venv prep + ${candidate_count} candidate tasks + finalize"
    else
        echo "  detector array (${detector_count} tasks, max ${detector_max_parallel} parallel) + GT cache + venv prep + ${candidate_count} candidate tasks + finalize"
    fi
    echo "Promote: uv run --directory $REPO_ROOT/src/yolo python -m yolo.promote_inference_profile --winner-json $OUTPUT_DIR/grid/winner.json"
fi
