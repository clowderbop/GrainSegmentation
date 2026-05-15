#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    exit 0
fi

VARIANTS=(PPL PPLPPXblend "PPL+PPXblend" "PPL+AllPPX")

job_slug() {

    printf '%s' "$1" | tr '+#' '__'
}

for variant in "${VARIANTS[@]}"; do
    slug="$(job_slug "$variant")"
    echo "Submitting run_yolo_sahi_test_eval (SAHI) variant=$variant"
    sbatch --export=ALL,VARIANT="$variant" --job-name="test_yolo_${slug}" SLURM/yolo/run_yolo_sahi_test_eval.sh
    echo "Submitting run_yolo_patch_test_eval (val) variant=$variant"
    sbatch --export=ALL,VARIANT="$variant" --job-name="test_yp_${slug}" SLURM/yolo/run_yolo_patch_test_eval.sh
done

echo "Submitted $(( ${#VARIANTS[@]} * 2 )) job(s)."
