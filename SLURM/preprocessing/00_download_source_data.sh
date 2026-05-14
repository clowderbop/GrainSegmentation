#!/bin/bash
#SBATCH --job-name=GrainSegmentation_download
#SBATCH --output=download-%j.log
#SBATCH --mem=4GB
#SBATCH --time=00:30:00

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_ROOT="$(cd "$THIS_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SLURM_ROOT/.." && pwd)"
cd "$REPO_ROOT"
source "$SLURM_ROOT/prepare_env.sh"

cd "$REPO_ROOT/src/data_prep" && uv run python -u download_data.py -o $SCRATCH/GrainSeg/dataset/source