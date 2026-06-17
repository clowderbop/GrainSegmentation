#!/bin/bash

set -euo pipefail

# shellcheck source=SLURM/utils/repo_root.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/repo_root.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$SLURM_ROOT/utils/variants.sh"
# shellcheck source=SLURM/utils/assertions.sh
source "$SLURM_ROOT/utils/assertions.sh"
# shellcheck source=SLURM/utils/slurm_export.sh
source "$SLURM_ROOT/utils/slurm_export.sh"

GRAINSEG_ROOT="$(grainseg_root)"
TRAIN_DIR="$GRAINSEG_ROOT/dataset/train"
MODELS_DIR="$GRAINSEG_ROOT/models/unet"
GT_GPKG="$TRAIN_DIR/train_labels.gpkg"
PREDS_ROOT="$GRAINSEG_ROOT/runs/watershed_tune_preds"
GRID_CONFIG="${GRID_CONFIG:-$REPO_ROOT/config/watershed_tune_grid.yaml}"
DRY_RUN=false
USE_CACHED_PREDS=false
SINGLE_JOB=false

function usage {
    cat <<'EOF' >&2
Usage: submit_watershed_tuning.sh [--dry-run] [--use-cached-preds] [--single-job] [--grid-config PATH]

Submit watershed tuning for all registry U-Net variants.

Default: predict-then-tune with parallel grid shards — run sliding-window U-Net
inference, then a throttled shard job array scoring axis-aligned grid subsets,
then one merge job writing the canonical grid CSV and watershed_best_*.json.

With --use-cached-preds: submit shard array and merge only, reading existing
semantic predictions under runs/watershed_tune_preds/{slug}/semantic/.

With --single-job: one monolithic tune job per variant (no shard array, no merge).

Cluster workflow: docs/runbooks/unet.md#watershed-tuning

Default mode additionally requires finetuned models under models/unet/.

Environment:
  GRID_CONFIG                         watershed tune grid YAML
  WATERSHED_TUNE_SHARD_MAX_PARALLEL   max concurrent shard array tasks (default: 6)
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
        --single-job)
            SINGLE_JOB=true
            shift
            ;;
        --grid-config)
            GRID_CONFIG="$2"
            shift 2
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

if [ ! -f "$GRID_CONFIG" ]; then
    echo "Grid config not found: $GRID_CONFIG" >&2
    exit 1
fi

shard_count="$(
    uv run --directory "$REPO_ROOT/src/unet" python -m unet.watershed_tune_shard_count \
        --grid-config "$GRID_CONFIG" | tr -d '[:space:]'
)"
if [ "$shard_count" -lt 1 ]; then
    echo "Watershed tune shard count must be >= 1 (grid: $GRID_CONFIG)" >&2
    exit 1
fi
shard_walltime="$(
    uv run --directory "$REPO_ROOT/src/unet" python -m unet.watershed_tune_walltime \
        --role shard --grid-config "$GRID_CONFIG" | tr -d '[:space:]'
)"
monolithic_walltime="$(
    uv run --directory "$REPO_ROOT/src/unet" python -m unet.watershed_tune_walltime \
        --role monolithic --grid-config "$GRID_CONFIG" | tr -d '[:space:]'
)"
merge_walltime="$(
    uv run --directory "$REPO_ROOT/src/unet" python -m unet.watershed_tune_walltime \
        --role merge | tr -d '[:space:]'
)"
shard_max_parallel="${WATERSHED_TUNE_SHARD_MAX_PARALLEL:-6}"

for variant in "${MICROSCOPY_VARIANTS[@]}"; do
    slug="$(job_slug "$variant")"
    subdir="$(watershed_tune_subdir_for_variant "$variant")"
    preds_dir="$PREDS_ROOT/$subdir/semantic"
    run_tag="$(date +%Y%m%d_%H%M%S)_${slug}"
    predict_job_id=""

    if [ "$USE_CACHED_PREDS" = true ]; then
        if [ "$DRY_RUN" = false ]; then
            require_cached_preds_dir "$preds_dir" "$variant"
        fi
    else
        unet_patch_config_for_variant "$variant"
        model_path="$MODELS_DIR/$DEFAULT_MODEL_BASENAME"
        predict_export="$(slurm_export_line "$(slurm_export_assign VARIANT "$variant")" "$(slurm_export_assign PREDS_ROOT "$PREDS_ROOT")")"
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

    if [ "$SINGLE_JOB" = true ]; then
        tune_export="$(slurm_export_line \
            "$(slurm_export_assign VARIANT "$variant")" \
            "$(slurm_export_assign PREDS_DIR "$preds_dir")" \
            "$(slurm_export_assign GRID_CONFIG "$GRID_CONFIG")")"
        tune_cmd=(
            sbatch
            "--time=${monolithic_walltime}"
            "--job-name=TuneWatershed_${slug}"
            "--export=${tune_export}"
            "$REPO_ROOT/SLURM/unet/run_watershed_tuning.sh"
            --variant "$variant"
            --preds-dir "$preds_dir"
            --dataset-dir "$TRAIN_DIR"
            --gt-gpkg "$GT_GPKG"
        )
        if [ -n "$predict_job_id" ]; then
            tune_cmd=("${tune_cmd[@]:0:2}" "--dependency=afterok:${predict_job_id}" "${tune_cmd[@]:2}")
        fi

        if [ "$DRY_RUN" = true ]; then
            if [ "$USE_CACHED_PREDS" = true ]; then
                echo "DRY-RUN monolithic tune from cached preds → $preds_dir" >&2
            else
                echo "DRY-RUN monolithic tune (reads --preds-dir only)" >&2
            fi
            printf '%q ' "${tune_cmd[@]}"
            echo
        else
            "${tune_cmd[@]}"
        fi
        continue
    fi

    shard_export="$(slurm_export_line \
        "$(slurm_export_assign VARIANT "$variant")" \
        "$(slurm_export_assign PREDS_DIR "$preds_dir")" \
        "$(slurm_export_assign RUN_TAG "$run_tag")" \
        "$(slurm_export_assign GRID_CONFIG "$GRID_CONFIG")")"
    shard_cmd=(
        sbatch
        "--time=${shard_walltime}"
        "--job-name=TuneWatershedShard_${slug}"
        "--export=${shard_export}"
        "--array=1-${shard_count}%${shard_max_parallel}"
        "$REPO_ROOT/SLURM/unet/run_watershed_tune_shard.sh"
        --variant "$variant"
        --preds-dir "$preds_dir"
        --dataset-dir "$TRAIN_DIR"
        --gt-gpkg "$GT_GPKG"
        --run-tag "$run_tag"
    )
    if [ -n "$predict_job_id" ]; then
        shard_cmd=("${shard_cmd[@]:0:3}" "--dependency=afterok:${predict_job_id}" "${shard_cmd[@]:3}")
    fi

    if [ "$DRY_RUN" = true ]; then
        if [ "$USE_CACHED_PREDS" = true ]; then
            echo "DRY-RUN shard array from cached preds → $preds_dir (run_tag=$run_tag)" >&2
        else
            echo "DRY-RUN shard array (reads --preds-dir only, run_tag=$run_tag)" >&2
        fi
        printf '%q ' "${shard_cmd[@]}"
        echo
        shard_job_id="DRYRUN"
    else
        shard_job_id="$("${shard_cmd[@]}" | awk '{print $NF}')"
        if [ -z "${shard_job_id:-}" ]; then
            echo "Shard array sbatch did not return a job id for $variant" >&2
            exit 1
        fi
    fi

    merge_export="$(slurm_export_line \
        "$(slurm_export_assign VARIANT "$variant")" \
        "$(slurm_export_assign RUN_TAG "$run_tag")" \
        "$(slurm_export_assign GRID_CONFIG "$GRID_CONFIG")")"
    merge_cmd=(
        sbatch
        "--time=${merge_walltime}"
        "--job-name=TuneWatershedMerge_${slug}"
        "--export=${merge_export}"
        "--dependency=afterok:${shard_job_id}"
        "$REPO_ROOT/SLURM/unet/run_watershed_tune_merge.sh"
        --variant "$variant"
        --run-tag "$run_tag"
    )

    if [ "$DRY_RUN" = true ]; then
        echo "DRY-RUN merge → watershed_grid_${run_tag}.csv and watershed_best_<job_id>.json" >&2
        printf '%q ' "${merge_cmd[@]}"
        echo
    else
        "${merge_cmd[@]}"
    fi
done

if [ "$DRY_RUN" = false ]; then
    if [ "$SINGLE_JOB" = true ]; then
        if [ "$USE_CACHED_PREDS" = true ]; then
            echo "Submitted monolithic tune-only watershed jobs for ${#MICROSCOPY_VARIANTS[@]} variants."
        else
            echo "Submitted predict-then-monolithic-tune watershed jobs for ${#MICROSCOPY_VARIANTS[@]} variants."
        fi
    elif [ "$USE_CACHED_PREDS" = true ]; then
        echo "Submitted shard-array watershed jobs for ${#MICROSCOPY_VARIANTS[@]} variants."
        echo "  preds:  $PREDS_ROOT/{slug}/semantic/ (existing)"
    else
        echo "Submitted predict-then-shard-array watershed jobs for ${#MICROSCOPY_VARIANTS[@]} variants."
        echo "  preds:  $PREDS_ROOT/{slug}/semantic/"
    fi
    echo "  tune:   $GRAINSEG_ROOT/runs/watershed_tune/{slug}/"
    if [ "$SINGLE_JOB" = false ]; then
        echo "  shards: ${shard_count} tasks per variant (max ${shard_max_parallel} parallel)"
        echo "  walltime: shard=${shard_walltime} merge=${merge_walltime}"
    else
        echo "  walltime: monolithic=${monolithic_walltime}"
    fi
fi
