#!/bin/bash

set -euo pipefail
# shellcheck source=SLURM/utils/repo_root.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/repo_root.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$SLURM_ROOT/utils/variants.sh"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    exit 0
fi

for variant in "${MICROSCOPY_VARIANTS[@]}"; do
    slug="$(job_slug "$variant")"
    echo "Submitting run_sahi_test_eval (SAHI) variant=$variant"
    sbatch --export=ALL,VARIANT="$variant" --job-name="test_yolo_${slug}" SLURM/yolo/run_sahi_test_eval.sh
    echo "Submitting run_patch_test_eval (val) variant=$variant"
    sbatch --export=ALL,VARIANT="$variant" --job-name="test_yp_${slug}" SLURM/yolo/run_patch_test_eval.sh
done

echo "Submitted $(( ${#MICROSCOPY_VARIANTS[@]} * 2 )) job(s)."
