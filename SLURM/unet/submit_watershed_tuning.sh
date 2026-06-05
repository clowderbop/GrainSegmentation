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
PREDS_ROOT="$GRAINSEG_ROOT/runs/watershed_tune_preds"
DRY_RUN=false

function usage {
    cat <<'EOF' >&2
Usage: submit_watershed_tuning.sh [--dry-run]

Submit the required two-phase watershed tuning workflow for all registry
U-Net variants: predict train whole-section semantic predictions, then tune
the watershed grid from cached preds (no U-Net inference in the tune job).

Cluster workflow: docs/runbooks/unet.md#watershed-tuning

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
    subdir="$(watershed_tune_subdir_for_variant "$variant")"
    preds_dir="$PREDS_ROOT/$subdir/semantic"

    predict_export="ALL,VARIANT=${variant},PREDS_ROOT=${PREDS_ROOT}"
    predict_cmd=(
        sbatch
        "--job-name=PredWatershed_${slug}"
        "--export=${predict_export}"
        "$REPO_ROOT/SLURM/unet/run_watershed_tune_predict.sh"
        --variant "$variant"
        --model-path "$model_path"
        --dataset-dir "$TRAIN_DIR"
        --preds-root "$PREDS_ROOT"
    )

    if [ "$DRY_RUN" = true ]; then
        echo "DRY-RUN predict → $preds_dir" >&2
        printf '%q ' "${predict_cmd[@]}"
        echo
        predict_job_id="DRYRUN"
    else
        predict_job_id="$("${predict_cmd[@]}" | awk '{print $NF}')"
        if [ -z "${predict_job_id:-}" ]; then
            echo "Predict sbatch did not return a job id for $variant" >&2
            exit 1
        fi
    fi

    tune_export="ALL,VARIANT=${variant},PREDS_DIR=${preds_dir}"
    tune_cmd=(
        sbatch
        "--job-name=TuneWatershed_${slug}"
        "--dependency=afterok:${predict_job_id}"
        "--export=${tune_export}"
        "$REPO_ROOT/SLURM/unet/run_watershed_tuning.sh"
        --variant "$variant"
        --preds-dir "$preds_dir"
        --dataset-dir "$TRAIN_DIR"
        --gt-gpkg "$GT_GPKG"
    )

    if [ "$DRY_RUN" = true ]; then
        echo "DRY-RUN tune (reads --preds-dir only)" >&2
        printf '%q ' "${tune_cmd[@]}"
        echo
    else
        "${tune_cmd[@]}"
    fi
done

if [ "$DRY_RUN" = false ]; then
    echo "Submitted predict-then-tune watershed jobs for ${#MICROSCOPY_VARIANTS[@]} variants."
    echo "  preds:  $PREDS_ROOT/{slug}/semantic/"
    echo "  tune:   $GRAINSEG_ROOT/runs/watershed_tune/{slug}/"
fi
