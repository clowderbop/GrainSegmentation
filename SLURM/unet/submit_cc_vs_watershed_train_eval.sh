#!/bin/bash
# Compare connected-components vs tuned watershed on the train section.
# Ops: docs/runbooks/unet.md#cc-vs-watershed-train-section

set -euo pipefail
# shellcheck source=SLURM/utils/repo_root.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/repo_root.sh"
cd "$REPO_ROOT"

DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: submit_cc_vs_watershed_train_eval.sh [--dry-run]" >&2
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: submit_cc_vs_watershed_train_eval.sh [--dry-run]" >&2
            exit 1
            ;;
    esac
done

GRAINSEG_ROOT="$(grainseg_root)"
MODELS_DIR="$GRAINSEG_ROOT/models/unet"
GT_GPKG="$GRAINSEG_ROOT/dataset/train/train_labels.gpkg"
WATERSHED_TUNE_ROOT="$GRAINSEG_ROOT/runs/watershed_tune"
EVAL_ROOT="$GRAINSEG_ROOT/eval"

cc_cmd=(
    sbatch
    --parsable
    --job-name=instance_val_cc
    "$REPO_ROOT/SLURM/unet/run_whole_test_eval.sh"
    --manifest-split train
    --config-file "$REPO_ROOT/SLURM/unet/whole_eval_models.tsv"
    --model-dir "$MODELS_DIR"
    --gt-gpkg "$GT_GPKG"
    --output-dir "$EVAL_ROOT/instance_val_cc"
    --instance-method cc
    --watershed-tune-root "$WATERSHED_TUNE_ROOT"
)

ws_cmd=(
    sbatch
    --parsable
    --job-name=instance_val_watershed
    "$REPO_ROOT/SLURM/unet/run_whole_test_eval.sh"
    --manifest-split train
    --config-file "$REPO_ROOT/SLURM/unet/whole_eval_models.tsv"
    --model-dir "$MODELS_DIR"
    --gt-gpkg "$GT_GPKG"
    --output-dir "$EVAL_ROOT/instance_val_watershed"
    --instance-method watershed
    --watershed-tune-root "$WATERSHED_TUNE_ROOT"
)

if [[ "$DRY_RUN" == true ]]; then
    printf '%q ' "${cc_cmd[@]}"
    echo
    cc_job=DRYRUN
    printf '%q ' "${ws_cmd[@]}"
    echo
    ws_job=DRYRUN
    select_cmd=(
        sbatch
        --parsable
        --job-name=cc_ws_select
        --dependency="afterok:${cc_job}:${ws_job}"
        "$REPO_ROOT/SLURM/unet/run_cc_vs_watershed_selection.sh"
    )
    printf '%q ' "${select_cmd[@]}"
    echo
    select_job=DRYRUN
else
    cc_job="$("${cc_cmd[@]}" | awk '{print $NF}')"
    ws_job="$("${ws_cmd[@]}" | awk '{print $NF}')"
    select_job="$(
        sbatch
        --parsable
        --job-name=cc_ws_select
        --dependency="afterok:${cc_job}:${ws_job}"
        "$REPO_ROOT/SLURM/unet/run_cc_vs_watershed_selection.sh" | awk '{print $NF}'
    )"
fi

echo "Submitted CC vs watershed train-section eval jobs."
echo "  CC job:         $cc_job"
echo "  Watershed job:  $ws_job"
echo "  Selection job:  $select_job (after both evals)"
echo "Selection output: $EVAL_ROOT/extraction_method_selection.json"
