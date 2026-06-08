#!/bin/bash

set -euo pipefail

# shellcheck source=SLURM/utils/repo_root.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/repo_root.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$SLURM_ROOT/utils/variants.sh"
# shellcheck source=SLURM/utils/assertions.sh
source "$SLURM_ROOT/utils/assertions.sh"

GRAINSEG_ROOT="$(grainseg_root)"
TRAIN_DIR="$GRAINSEG_ROOT/dataset/train"
MODELS_DIR="$GRAINSEG_ROOT/models/unet"
GT_GPKG="$TRAIN_DIR/train_labels.gpkg"
PREDS_ROOT="$GRAINSEG_ROOT/runs/watershed_tune_preds"
DRY_RUN=false
USE_CACHED_PREDS=false

function usage {
    cat <<'EOF' >&2
Usage: submit_watershed_tuning.sh [--dry-run] [--use-cached-preds]

Submit watershed tuning for all registry U-Net variants.

Default: two-phase predict-then-tune — run sliding-window U-Net inference,
then tune the watershed grid from cached preds (no U-Net inference in tune).

With --use-cached-preds: submit tune jobs only, reading existing semantic
predictions under runs/watershed_tune_preds/{slug}/semantic/. Requires train
whole manifests and GT GPKG only (no finetuned models).

Cluster workflow: docs/runbooks/unet.md#watershed-tuning

Default mode additionally requires finetuned models under models/unet/.
EOF
    exit "${1:-1}"
}

function require_cached_preds_dir {
    local preds_dir="$1"
    local variant="$2"
    require_dir "$preds_dir" "Cached preds dir missing for $variant"
    if ! compgen -G "$preds_dir"/*_pred.tif > /dev/null; then
        echo "No *_pred.tif cached predictions in: $preds_dir" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --use-cached-preds)
            USE_CACHED_PREDS=true
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
    slug="$(job_slug "$variant")"
    subdir="$(watershed_tune_subdir_for_variant "$variant")"
    preds_dir="$PREDS_ROOT/$subdir/semantic"

    if [ "$USE_CACHED_PREDS" = true ]; then
        if [ "$DRY_RUN" = false ]; then
            require_cached_preds_dir "$preds_dir" "$variant"
        fi
    else
        unet_patch_config_for_variant "$variant"
        model_path="$MODELS_DIR/$DEFAULT_MODEL_BASENAME"
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
    fi

    tune_export="ALL,VARIANT=${variant},PREDS_DIR=${preds_dir}"
    tune_cmd=(
        sbatch
        "--job-name=TuneWatershed_${slug}"
        "--export=${tune_export}"
        "$REPO_ROOT/SLURM/unet/run_watershed_tuning.sh"
        --variant "$variant"
        --preds-dir "$preds_dir"
        --dataset-dir "$TRAIN_DIR"
        --gt-gpkg "$GT_GPKG"
    )
    if [ "$USE_CACHED_PREDS" = false ]; then
        tune_cmd=("${tune_cmd[@]:0:2}" "--dependency=afterok:${predict_job_id}" "${tune_cmd[@]:2}")
    fi

    if [ "$DRY_RUN" = true ]; then
        if [ "$USE_CACHED_PREDS" = true ]; then
            echo "DRY-RUN tune from cached preds → $preds_dir" >&2
        else
            echo "DRY-RUN tune (reads --preds-dir only)" >&2
        fi
        printf '%q ' "${tune_cmd[@]}"
        echo
    else
        "${tune_cmd[@]}"
    fi
done

if [ "$DRY_RUN" = false ]; then
    if [ "$USE_CACHED_PREDS" = true ]; then
        echo "Submitted tune-only watershed jobs for ${#MICROSCOPY_VARIANTS[@]} variants."
        echo "  preds:  $PREDS_ROOT/{slug}/semantic/ (existing)"
    else
        echo "Submitted predict-then-tune watershed jobs for ${#MICROSCOPY_VARIANTS[@]} variants."
        echo "  preds:  $PREDS_ROOT/{slug}/semantic/"
    fi
    echo "  tune:   $GRAINSEG_ROOT/runs/watershed_tune/{slug}/"
fi
