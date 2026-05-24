# shellcheck shell=bash
# Resolves REPO_ROOT, SLURM_ROOT, and SLURM_UTILS. Source via enter_job.sh, source_job.sh, or repo_root.sh.

if [ -z "${REPO_ROOT:-}" ] || [ -z "${SLURM_ROOT:-}" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        REPO_ROOT="$SLURM_SUBMIT_DIR"
        SLURM_ROOT="$REPO_ROOT/SLURM"
    elif [ "${#BASH_SOURCE[@]}" -ge 2 ]; then
        _caller_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
        SLURM_ROOT="$(cd "$_caller_dir/.." && pwd)"
        REPO_ROOT="$(cd "$SLURM_ROOT/.." && pwd)"
        unset _caller_dir
    else
        SLURM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
        REPO_ROOT="$(cd "$SLURM_ROOT/.." && pwd)"
    fi
fi

# shellcheck disable=SC2034
SLURM_UTILS="$SLURM_ROOT/utils"

grainseg_root() {
    printf '%s\n' "${SCRATCH:-/scratch/${USER}}/GrainSeg"
}
