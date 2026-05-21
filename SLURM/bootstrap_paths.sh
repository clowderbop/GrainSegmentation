# Resolves REPO_ROOT and SLURM_ROOT for batch jobs executed from SLURM spool copies.
# Source from job scripts after set -euo pipefail. Submit sbatch jobs from the repo root.

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
    SLURM_ROOT="$REPO_ROOT/SLURM"
elif [ "${#BASH_SOURCE[@]}" -ge 2 ]; then
    _caller_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    SLURM_ROOT="$(cd "$_caller_dir/.." && pwd)"
    REPO_ROOT="$(cd "$SLURM_ROOT/.." && pwd)"
    unset _caller_dir
else
    SLURM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SLURM_ROOT/.." && pwd)"
fi
