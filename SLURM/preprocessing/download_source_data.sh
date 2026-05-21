#!/bin/bash
#SBATCH --job-name=GrainSeg_download_uncropped
#SBATCH --output=download-%j.log
#SBATCH --mem=8GB
#SBATCH --time=08:00:00

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    # shellcheck source=SLURM/bootstrap_paths.sh
    source "$SLURM_SUBMIT_DIR/SLURM/bootstrap_paths.sh"
else
    # shellcheck source=SLURM/bootstrap_paths.sh
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/SLURM/bootstrap_paths.sh"
fi
cd "$REPO_ROOT"
source "$SLURM_ROOT/prepare_env.sh"

module load lz4/1.9.4-GCCcore-12.3.0

cd "$REPO_ROOT/src/data_prep" && uv run python -u download_data.py \
  -o "$SCRATCH/GrainSeg/dataset/uncropped"
