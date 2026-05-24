# shellcheck shell=bash
# Source from SLURM job scripts (preprocessing/, unet/, yolo/) after set -euo pipefail.
# Job scripts must load this via SLURM_SUBMIT_DIR when under sbatch (see enter_job usage there).
# Sets REPO_ROOT, SLURM_ROOT, SLURM_UTILS and cd's to the repo root.

# shellcheck source=SLURM/utils/source_job.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/source_job.sh"
