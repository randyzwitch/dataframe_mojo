#!/usr/bin/env bash
# Every Metal contract on this Mac's GPU, as the retired `native-metal` CI job
# meant to run them. GitHub's hosted macOS runners cannot: their virtual GPU
# gives Metal buffers no GPU address, which MAX's device context needs, so
# Metal changes are validated here, on Apple silicon, before a pull request.
#
# usage: scripts/test_metal_local.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "Metal contracts need an Apple silicon Mac." >&2
  exit 1
fi
mkdir -p build/contract-tests

echo "== CPU-only build of the Mojo GPU provider"
pixi run --locked mojo build --target-accelerator=unknown -I . \
  tests/tooling/mojo_gpu_cpu_compile.mojo -o build/contract-tests/mojo-gpu-cpu-compile
build/contract-tests/mojo-gpu-cpu-compile

echo "== Metal device"
pixi run --locked mojo build -I . tests/tooling/metal_available.mojo \
  -o build/contract-tests/metal-available
capability="$(build/contract-tests/metal-available)"
if [[ "$capability" != available ]]; then
  echo "No Metal device for Mojo (see the probe's reasons above)." >&2
  exit 1
fi

echo "== Parquet reader (the file-source contracts read Parquet)"
pixi run --locked -e native build-dfparquet

# Every contract group with a module that needs the device, found from the
# `# requires:` markers so a new group cannot be left out. (macOS ships
# bash 3.2: no mapfile, and empty arrays trip `set -u`.)
groups="$(grep -rlE '^# requires: (metal|mojo_gpu)' tests \
  | xargs grep -h '^# contract-group:' | awk '{print $3}' | sort -u | tr '\n' ' ')"
failed=""

echo "== Metal contracts on the device: $groups"
# shellcheck disable=SC2086 # one argument per group
pixi run --locked test --areas execution public --contracts $groups \
  --capabilities metal mojo_gpu || failed="$failed contracts"

echo "== Mojo GPU contracts and independent oracles"
for task in test-mojo-gpu test-mojo-gpu-floating test-mojo-gpu-decimal \
  test-mojo-gpu-integer test-mojo-gpu-grouping test-mojo-gpu-kernels \
  test-metal-suite-checks; do
  pixi run --locked -e gpu "$task" || failed="$failed $task"
done
if [[ -n "$failed" ]]; then
  echo "Failed:$failed" >&2
  exit 1
fi
echo "All Metal contracts passed."
