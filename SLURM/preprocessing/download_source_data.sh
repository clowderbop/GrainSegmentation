#!/bin/bash
#SBATCH --job-name=GrainSeg_download_uncropped
#SBATCH --output=download-%j.log
#SBATCH --mem=8GB
#SBATCH --time=08:00:00

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_ROOT="$(cd "$THIS_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SLURM_ROOT/.." && pwd)"
cd "$REPO_ROOT"
source "$SLURM_ROOT/prepare_env.sh"

module load lz4/1.9.4-GCCcore-12.3.0

cd "$REPO_ROOT/src/data_prep" && uv run python -u download_data.py \
  -o "$SCRATCH/GrainSeg/dataset/uncropped"
