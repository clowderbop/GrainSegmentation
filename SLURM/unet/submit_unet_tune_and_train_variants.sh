#!/bin/bash

set -euo pipefail
# shellcheck source=SLURM/utils/repo_root.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/repo_root.sh"
cd "$REPO_ROOT"

GRAINSEG_ROOT="$(grainseg_root)"
TRAIN_LABELS_RASTER="$GRAINSEG_ROOT/dataset/train/train_labels.tif"
RUN_SCRIPT="$REPO_ROOT/SLURM/unet/run_unet_tune_and_train_variant.sh"

function usage {
    local status="${1:-1}"
    cat <<EOF >&2
Usage: $(basename "$0") [options]

Submit U-Net tune+train jobs for one or more input variants.
Requires train whole manifests and rasterized labels:
  $TRAIN_LABELS_RASTER

Options:
  --ppl                  PPL only (1 input)
  --ppl-ppx-composite    PPLPPXblend (1 input)
  --ppl-plus-ppx-composite  PPL + PPXblend (2 inputs)
  --all-ppx              PPL + all PPX (7 inputs)
  --all                  all four variants
  --resume               pass --resume to training jobs
  --skip-tuning          pass --skip-tuning to training jobs
  --verbose              pass --verbose to training jobs
  --help
EOF
    exit "$status"
}

run_ppl=false
run_ppl_ppx_composite=false
run_ppl_plus_ppx_composite=false
run_all_ppx=false
resume_args=()
skip_tuning_args=()
verbose_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ppl)
            run_ppl=true
            shift
            ;;
        --ppl-ppx-composite)
            run_ppl_ppx_composite=true
            shift
            ;;
        --ppl-plus-ppx-composite)
            run_ppl_plus_ppx_composite=true
            shift
            ;;
        --all-ppx)
            run_all_ppx=true
            shift
            ;;
        --all)
            run_ppl=true
            run_ppl_ppx_composite=true
            run_ppl_plus_ppx_composite=true
            run_all_ppx=true
            shift
            ;;
        --resume)
            resume_args=(--resume)
            shift
            ;;
        --skip-tuning)
            skip_tuning_args=(--skip-tuning)
            shift
            ;;
        --verbose)
            verbose_args=(--verbose)
            shift
            ;;
        --help|-h)
            usage 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

if [ "$run_ppl" = false ] && [ "$run_ppl_ppx_composite" = false ] && \
   [ "$run_ppl_plus_ppx_composite" = false ] && [ "$run_all_ppx" = false ]; then
    usage
fi

if [ ! -f "$TRAIN_LABELS_RASTER" ]; then
    echo "Error: $TRAIN_LABELS_RASTER not found." >&2
    echo "Run SLURM/preprocessing/rasterize_labels.sh before submitting training." >&2
    exit 1
fi

submit_variant() {
    local mem="$1"
    local job_name="$2"
    local variant="$3"
    shift 3

    sbatch \
        --mem="$mem" \
        --job-name="$job_name" \
        --export=ALL,VARIANT="$variant",DATASET_DIR="$GRAINSEG_ROOT/dataset/train" \
        "$RUN_SCRIPT" \
        --variant "$variant" \
        --run-name "$variant" \
        "$@" \
        "${resume_args[@]}" \
        "${skip_tuning_args[@]}" \
        "${verbose_args[@]}"
}

submitted=false

if [ "$run_ppl" = true ]; then
    echo "Submitting PPL only (1 input) job..."
    submit_variant 256G Train_PPL PPL
    submitted=true
fi

if [ "$run_ppl_ppx_composite" = true ]; then
    echo "Submitting PPLPPXBlend (1 input) job..."
    submit_variant 256G Train_PPLPPXBlend PPLPPXblend
    submitted=true
fi

if [ "$run_ppl_plus_ppx_composite" = true ]; then
    echo "Submitting PPL + PPXblend (2 inputs) job..."
    submit_variant 512G Train_PPL+PPXblend "PPL+PPXblend"
    submitted=true
fi

if [ "$run_all_ppx" = true ]; then
    echo "Submitting PPL + All PPX (7 inputs) job..."
    submit_variant 950G Train_PPL+AllPPX "PPL+AllPPX"
    submitted=true
fi

if [ "$submitted" = true ]; then
    echo "Selected jobs submitted successfully!"
fi
