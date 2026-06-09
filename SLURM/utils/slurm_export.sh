# shellcheck shell=bash
# Slurm --export assignments. Values containing '+' or ',' must be single-quoted
# (see Slurm sbatch --export documentation).

slurm_export_assign() {
    local key="$1"
    local value="$2"
    case "$value" in
        *+*|*,*)
            value="${value//\'/\'\\\'\'}"
            printf "%s='%s'" "$key" "$value"
            ;;
        *)
            printf '%s=%s' "$key" "$value"
            ;;
    esac
}

slurm_export_line() {
    local out="ALL"
    local part
    for part in "$@"; do
        out="${out},${part}"
    done
    printf '%s' "$out"
}
