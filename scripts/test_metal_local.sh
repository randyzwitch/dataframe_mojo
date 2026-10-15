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

echo "== Backend contracts on the device"
pixi run --locked test --areas execution --contracts execution.backends \
  --capabilities metal

echo "== Mojo GPU contracts and independent oracles"
pixi run --locked -e gpu test-mojo-gpu
pixi run --locked -e gpu test-mojo-gpu-floating
pixi run --locked -e gpu test-mojo-gpu-decimal
pixi run --locked -e gpu test-mojo-gpu-integer
echo "All Metal contracts passed."
