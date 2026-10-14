#!/usr/bin/env bash
# `mojo` with the GPU named for `build` and `run`. The Metal CI job puts this
# first on PATH as `mojo`: GitHub's macOS runners are virtual machines whose
# Apple GPU (a paravirtual device) Metal serves but the compiler does not
# detect, so builds there target no accelerator unless one is named
# (`DATAFRAME_TARGET_ACCELERATOR`, e.g. apple-m1). Other commands pass through.
set -euo pipefail
real="${CONDA_PREFIX:?run inside a pixi environment}/bin/mojo"
case "${1:-}" in
  build | run)
    command="$1"
    shift
    exec "$real" "$command" \
      --target-accelerator="${DATAFRAME_TARGET_ACCELERATOR:?name the GPU}" "$@"
    ;;
  *)
    exec "$real" "$@"
    ;;
esac
