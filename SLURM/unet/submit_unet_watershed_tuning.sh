#!/bin/bash

set -euo pipefail

# shellcheck source=SLURM/utils/repo_root.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/repo_root.sh"
GRAINSEG_ROOT="$(grainseg_root)"
TRAIN_DIR="$GRAINSEG_ROOT/dataset/train"
MODELS_DIR="$GRAINSEG_ROOT/models/unet"
GT_GPKG="$TRAIN_DIR/train_labels.gpkg"
DRY_RUN=false

function usage {
    cat <<'EOF' >&2
Usage: submit_unet_watershed_tuning.sh [--dry-run]

Submit watershed hyperparameter tuning for all four U-Net input variants.

Expects finetuned models under models/unet/ and train-section data under
dataset/train/ (train_*.tif mosaics and train_labels.gpkg).
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

submit_one() {
    local job_name="$1"
    local model_basename="$2"
    local num_inputs="$3"
    local suffixes="$4"
    local out_subdir="$5"

    local model_path="$MODELS_DIR/$model_basename"
    local out_dir="$GRAINSEG_ROOT/runs/watershed_tune/$out_subdir"

    local -a cmd=(
        sbatch
        "--job-name=$job_name"
        "$REPO_ROOT/SLURM/unet/run_unet_watershed_tuning.sh"
        --model-path "$model_path"
        --dataset-dir "$TRAIN_DIR"
        --gt-gpkg "$GT_GPKG"
        --num-inputs "$num_inputs"
        --image-suffixes "$suffixes"
        --output-dir "$out_dir"
    )

    if [ "$DRY_RUN" = true ]; then
        printf '%q ' "${cmd[@]}"
        echo
    else
        "${cmd[@]}"
    fi
}

submit_one "TuneWatershed_PPL" "unet_finetuned_PPL.keras" 1 "_PPL" "PPL"
submit_one "TuneWatershed_PPLPPXblend" "unet_finetuned_PPLPPXblend.keras" 1 "_PPLPPXblend" "PPLPPXblend"
submit_one "TuneWatershed_PPL_PPXblend" "unet_finetuned_PPL+PPXblend.keras" 2 "_PPL _PPXblend" "PPL_PlusPPXblend"
submit_one "TuneWatershed_PPL_AllPPX" "unet_finetuned_PPL+AllPPX.keras" 7 "_PPL _PPX1 _PPX2 _PPX3 _PPX4 _PPX5 _PPX6" "PPL_AllPPX"

if [ "$DRY_RUN" = false ]; then
    echo "Submitted watershed tuning jobs for all U-Net variants."
fi
