#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == build ]]; then
    shift
    exec mojo build --num-threads 2 "$@"
fi
exec mojo "$@"
