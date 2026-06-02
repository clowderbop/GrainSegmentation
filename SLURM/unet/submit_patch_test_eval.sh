#!/bin/bash
# Submit patch-wise U-Net test evaluations for all microscopy input variants.

set -euo pipefail
# shellcheck source=SLURM/utils/repo_root.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/repo_root.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$SLURM_ROOT/utils/variants.sh"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $(basename "$0")"
    echo "Submits one SLURM job per VARIANT (PPL, PPLPPXblend, PPL+PPXblend, PPL+AllPPX)."
    echo "Cluster workflow: docs/runbooks/unet.md#patch-test-eval"
    exit 0
fi

for variant in "${MICROSCOPY_VARIANTS[@]}"; do
    slug="$(job_slug "$variant")"
    echo "Submitting run_patch_test_eval variant=$variant"
    sbatch --export=ALL,VARIANT="$variant" --job-name="test_unet_p_${slug}" \
        "$REPO_ROOT/SLURM/unet/run_patch_test_eval.sh"
done

echo "Submitted ${#MICROSCOPY_VARIANTS[@]} job(s)."
