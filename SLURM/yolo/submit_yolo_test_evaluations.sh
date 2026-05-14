#!/bin/bash
# Submit SAHI (run_yolo_sahi_test_eval.sh) and patch val (run_yolo_patch_test_eval.sh) for each training variant.
# run_yolo_sahi_test_eval.sh reads the held-out multichannel test TIFFs (test_*.tif) in $SCRATCH/GrainSeg/dataset/test/; channel TIFFs also use the test_ prefix.
# Run from the repository root: bash SLURM/yolo/submit_yolo_test_evaluations.sh
# Optional: VARIANTS=(PPL) bash SLURM/yolo/submit_yolo_test_evaluations.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: submit_yolo_test_evaluations.sh"
    echo "  Submits run_yolo_sahi_test_eval.sh and run_yolo_patch_test_eval.sh with VARIANT set for each model type."
    exit 0
fi

VARIANTS=(PPL PPLPPXblend "PPL+PPXblend" "PPL+AllPPX")

job_slug() {
    # Slurm job names: avoid problematic characters.
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
