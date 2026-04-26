#!/bin/bash
# Submit SAHI (test_yolo.sh) and patch val (test_yolo_patches.sh) for each training variant.
# Run from the repository root: bash SLURM/test_yolo_submit.sh
# Optional: VARIANTS=(PPL) bash SLURM/test_yolo_submit.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: test_yolo_submit.sh"
    echo "  Submits test_yolo.sh and test_yolo_patches.sh with VARIANT set for each model type."
    exit 0
fi

VARIANTS=(PPL PPLPPXblend "PPL+PPXblend" "PPL+AllPPX")

job_slug() {
    # Slurm job names: avoid problematic characters.
    printf '%s' "$1" | tr '+#' '__'
}

for variant in "${VARIANTS[@]}"; do
    slug="$(job_slug "$variant")"
    echo "Submitting test_yolo (SAHI) variant=$variant"
    sbatch --export=ALL,VARIANT="$variant" --job-name="test_yolo_${slug}" SLURM/test_yolo.sh
    echo "Submitting test_yolo_patches (val) variant=$variant"
    sbatch --export=ALL,VARIANT="$variant" --job-name="test_yp_${slug}" SLURM/test_yolo_patches.sh
done

echo "Submitted $(( ${#VARIANTS[@]} * 2 )) job(s)."
