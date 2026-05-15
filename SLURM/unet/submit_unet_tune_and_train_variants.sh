#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

function usage {
    exit 1
}

run_ppl=false
run_ppl_ppx_composite=false
run_ppl_plus_ppx_composite=false
run_all_ppx=false
resume_args=()
skip_tuning_args=()
verbose_args=()

# Process flags
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
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

if [ "$run_ppl" = false ] && [ "$run_ppl_ppx_composite" = false ] && [ "$run_ppl_plus_ppx_composite" = false ] && [ "$run_all_ppx" = false ]; then
    usage
fi

submitted=false

if [ "$run_ppl" = true ]; then
    echo "Submitting PPL only (1 input) job..."
    sbatch \
        --mem=256G \
        --job-name=Train_PPL \
        SLURM/unet/run_unet_tune_and_train_variant.sh \
        --num-inputs 1 \
        --image-suffixes "_PPL" \
        --run-name "PPL" \
        "${resume_args[@]}" \
        "${skip_tuning_args[@]}" \
        "${verbose_args[@]}"
    submitted=true
fi

if [ "$run_ppl_ppx_composite" = true ]; then
    echo "Submitting PPLPPXBlend (1 input) job..."
    sbatch \
        --mem=256G \
        --job-name=Train_PPLPPXBlend \
        SLURM/unet/run_unet_tune_and_train_variant.sh \
        --num-inputs 1 \
        --image-suffixes "_PPLPPXblend" \
        --run-name "PPLPPXblend" \
        "${resume_args[@]}" \
        "${skip_tuning_args[@]}" \
        "${verbose_args[@]}"
    submitted=true
fi

if [ "$run_ppl_plus_ppx_composite" = true ]; then
    echo "Submitting PPL + PPXblend (2 inputs) job..."
    sbatch \
        --mem=512G \
        --job-name=Train_PPL+PPXblend \
        SLURM/unet/run_unet_tune_and_train_variant.sh \
        --num-inputs 2 \
        --image-suffixes "_PPL _PPXblend" \
        --run-name "PPL+PPXblend" \
        "${resume_args[@]}" \
        "${skip_tuning_args[@]}" \
        "${verbose_args[@]}"
    submitted=true
fi

if [ "$run_all_ppx" = true ]; then
    echo "Submitting PPL + All PPX (7 inputs) job..."
    sbatch \
        --mem=950G \
        --job-name=Train_PPL+AllPPX \
        SLURM/unet/run_unet_tune_and_train_variant.sh \
        --num-inputs 7 \
        --image-suffixes "_PPL _PPX1 _PPX2 _PPX3 _PPX4 _PPX5 _PPX6" \
        --run-name "PPL+AllPPX" \
        "${resume_args[@]}" \
        "${skip_tuning_args[@]}" \
        "${verbose_args[@]}"
    submitted=true
fi

if [ "$submitted" = true ]; then
    echo "Selected jobs submitted successfully!"
fi