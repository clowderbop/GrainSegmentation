# shellcheck shell=bash
# Loaded by enter_job.sh from SLURM job scripts (unet/, yolo/, preprocessing/).
# Sets REPO_ROOT, SLURM_ROOT, SLURM_UTILS and cd's to the repo root.

_slurm_utils_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    # shellcheck source=SLURM/utils/paths.sh
    source "$SLURM_SUBMIT_DIR/SLURM/utils/paths.sh"
elif [ "${#BASH_SOURCE[@]}" -ge 2 ]; then
    _job_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    if [ -f "$_job_dir/../utils/paths.sh" ]; then
        SLURM_ROOT="$(cd "$_job_dir/.." && pwd)"
    else
        SLURM_ROOT="$(cd "$_job_dir/../.." && pwd)"
    fi
    # shellcheck source=SLURM/utils/paths.sh
    source "$SLURM_ROOT/utils/paths.sh"
    unset _job_dir
else
    # shellcheck source=SLURM/utils/paths.sh
    source "$_slurm_utils_dir/paths.sh"
fi
unset _slurm_utils_dir

cd "$REPO_ROOT" || exit
