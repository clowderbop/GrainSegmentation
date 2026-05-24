#!/bin/bash

set -euo pipefail

# shellcheck source=SLURM/utils/repo_root.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/repo_root.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$SLURM_ROOT/utils/variants.sh"

GRAINSEG_ROOT="$(grainseg_root)"
TRAIN_DIR="$GRAINSEG_ROOT/dataset/train"
MODELS_DIR="$GRAINSEG_ROOT/models/unet"
GT_GPKG="$TRAIN_DIR/train_labels.gpkg"
DRY_RUN=false

function usage {
    cat <<'EOF' >&2
Usage: submit_watershed_tuning.sh [--dry-run]

Submit watershed hyperparameter tuning for all registry U-Net variants.
Requires train whole manifests and finetuned models under models/unet/.
EOF
    exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
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

for variant in "${MICROSCOPY_VARIANTS[@]}"; do
    unet_patch_config_for_variant "$variant"
    model_path="$MODELS_DIR/$DEFAULT_MODEL_BASENAME"
    slug="$(job_slug "$variant")"

    cmd=(
        sbatch
        "--job-name=TuneWatershed_${slug}"
        "--export=ALL,VARIANT=${variant}"
        "$REPO_ROOT/SLURM/unet/run_watershed_tuning.sh"
        --variant "$variant"
        --model-path "$model_path"
        --dataset-dir "$TRAIN_DIR"
        --gt-gpkg "$GT_GPKG"
    )

    if [ "$DRY_RUN" = true ]; then
        printf '%q ' "${cmd[@]}"
        echo
    else
        "${cmd[@]}"
    fi
done

if [ "$DRY_RUN" = false ]; then
    echo "Submitted watershed tuning jobs for ${#MICROSCOPY_VARIANTS[@]} variants."
fi
