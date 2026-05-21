# shellcheck shell=bash
require_file() {
    local path="$1"
    local message="$2"
    if [ ! -f "$path" ]; then
        echo "$message: $path" >&2
        exit 1
    fi
}

require_dir() {
    local path="$1"
    local message="$2"
    if [ ! -d "$path" ]; then
        echo "$message: $path" >&2
        exit 1
    fi
}
