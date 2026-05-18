#!/bin/bash
# Submit patch-wise U-Net test evaluations for all microscopy input variants.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $(basename "$0")"
    echo "Submits one SLURM job per VARIANT (PPL, PPLPPXblend, PPL+PPXblend, PPL+AllPPX)."
    exit 0
fi

VARIANTS=(PPL PPLPPXblend "PPL+PPXblend" "PPL+AllPPX")

job_slug() {
    printf '%s' "$1" | tr '+#' '__'
}

for variant in "${VARIANTS[@]}"; do
    slug="$(job_slug "$variant")"
    echo "Submitting run_unet_patch_test_eval variant=$variant"
    sbatch --export=ALL,VARIANT="$variant" --job-name="test_unet_p_${slug}" \
        "$REPO_ROOT/SLURM/unet/run_unet_patch_test_eval.sh"
done

echo "Submitted ${#VARIANTS[@]} job(s)."
