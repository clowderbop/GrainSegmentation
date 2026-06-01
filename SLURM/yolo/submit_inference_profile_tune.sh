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

Submit parallel YOLO inference profile selection (ADR 0005 orchestration; see ADRs below):
  (1) GPU detector array per (variant, conf, mask_threshold) — tiled detector proposals v2
  (2) CPU GT-cache job (after detectors, or immediately if detectors skipped)
  (3) CPU array: one job per grid candidate (1 CPU, 32G, 4h per task)
  (4) CPU finalize job (afterok on candidate array)

Requires all registry variant weights under runs/yolo26-seg/{variant}/weights/best.pt
when running detectors.

ADR consequences (read before salvaging a failed run):
  docs/adr/0006-gpkg-ground-truth-rasterization.md — OpenCV GT cache at _work/gt_cache/train/
  docs/adr/0007-profile-selection-proposal-cache-and-scoring.md — proposal cache schema_version 2

Salvage after ADR 0006/0007 (or any in-flight pre-fix run):
  Delete the entire runs/yolo_inference_profile_tune/<run_id>/ directory on scratch.
  Submit a new RUN_ID and run the full pipeline (detector → GT cache → candidate → finalize).
  Do not pass --skip-detectors to reuse _work/ from a pre-fix or partial run (v1 proposal caches
  and old GT layouts are rejected). Grid winners from pre-ADR runs are not comparable to reruns.

--skip-detectors is only for re-submitting candidate/finalize when _work/ was produced by the
current ADR stack in the same OUTPUT_DIR (e.g. detectors already finished successfully).

Environment:
  OUTPUT_DIR       full run directory (default: .../yolo_inference_profile_tune/<run_id>)
  RUN_ID           run folder name when OUTPUT_DIR unset
  GRID_CONFIG      search grid YAML (default: configs/yolo_inference_profile_tune.yaml)
  SKIP_DETECTORS   set to 1 (or use --skip-detectors) only when v2 caches exist in this OUTPUT_DIR
  DETECTOR_MAX_PARALLEL  max concurrent detector array tasks (default: 6)
  NO_RESUME        set to 1 to clear grid/rows/*.json before candidate array and pass --no-resume
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
    echo "Skipping detector jobs (SKIP_DETECTORS=1); using existing v2 _work/ in $OUTPUT_DIR"
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

cand_export="ALL,OUTPUT_DIR=${OUTPUT_DIR},GRID_CONFIG=${GRID_CONFIG}"
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
    cand_cmd+=("--dependency=afterok:${gt_job_id}")
fi
cand_cmd+=("$REPO_ROOT/SLURM/yolo/run_profile_tune_candidate.sh")

if [ "$DRY_RUN" = true ]; then
    printf '%q ' "${cand_cmd[@]}"
    echo
else
    cand_job_id="$("${cand_cmd[@]}" | awk '{print $NF}')"
fi

fin_export="ALL,OUTPUT_DIR=${OUTPUT_DIR},GRID_CONFIG=${GRID_CONFIG}"
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
        echo "  detectors skipped (same OUTPUT_DIR, post-ADR v2 _work/); GT cache + ${candidate_count} candidate tasks + finalize"
    else
        echo "  detector array (${detector_count} tasks, max ${detector_max_parallel} parallel) + GT cache + ${candidate_count} candidate tasks + finalize"
    fi
    echo "Promote: uv run --directory $REPO_ROOT/src/yolo python -m yolo.promote_inference_profile --winner-json $OUTPUT_DIR/grid/winner.json"
fi
