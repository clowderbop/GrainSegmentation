#!/bin/bash

set -euo pipefail
# shellcheck source=SLURM/utils/repo_root.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/repo_root.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$SLURM_ROOT/utils/variants.sh"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Submit YOLO patch + whole-section (SAHI) test eval for every registry variant.

SAHI jobs use per-variant --mem (see SLURM/yolo/pipeline.md). Patch jobs use the
default in run_patch_test_eval.sh unless you override with sbatch --mem.

Manual one-off (sbatch --mem overrides #SBATCH in the run script):
  sbatch --mem=1000G --export=ALL,VARIANT='PPL+AllPPX' SLURM/yolo/run_sahi_test_eval.sh
EOF
    exit 0
fi

sahi_mem_for_variant() {
    case "$1" in
        PPL | PPLPPXblend) printf '%s\n' 400G ;;
        PPL+PPXblend) printf '%s\n' 500G ;;
        PPL+AllPPX) printf '%s\n' 1000G ;;
        *)
            echo "submit_test_evaluations: unknown variant $1" >&2
            return 1
            ;;
    esac
}

for variant in "${MICROSCOPY_VARIANTS[@]}"; do
    slug="$(job_slug "$variant")"
    sahi_mem="$(sahi_mem_for_variant "$variant")"
    echo "Submitting run_sahi_test_eval (SAHI) variant=$variant mem=$sahi_mem"
    sbatch \
        --mem="$sahi_mem" \
        --export=ALL,VARIANT="$variant" \
        --job-name="test_yolo_${slug}" \
        SLURM/yolo/run_sahi_test_eval.sh
    echo "Submitting run_patch_test_eval (val) variant=$variant"
    sbatch --export=ALL,VARIANT="$variant" --job-name="test_yp_${slug}" SLURM/yolo/run_patch_test_eval.sh
done

echo "Submitted $(( ${#MICROSCOPY_VARIANTS[@]} * 2 )) job(s)."
