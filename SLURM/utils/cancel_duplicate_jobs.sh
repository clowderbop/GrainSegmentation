# shellcheck shell=bash
# Cancel other SLURM jobs with the same job name as the current job.
cancel_duplicate_slurm_jobs() {
    if [ -z "${SLURM_JOB_NAME:-}" ] || [ -z "${SLURM_JOB_ID:-}" ]; then
        return 0
    fi

    local old_jobs
    old_jobs=$(squeue -u "$USER" -n "$SLURM_JOB_NAME" -h -o %i | grep -v "^$SLURM_JOB_ID$" || true)

    if [ -n "$old_jobs" ]; then
        echo "Canceling previous jobs with name $SLURM_JOB_NAME: $old_jobs"
        # shellcheck disable=SC2086
        scancel $old_jobs
        sleep 10
    fi
}
