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

echo "== Packed-word Boolean and validity counts"
pixi run --locked mojo build -O3 -I . scripts/check_metal_logical_words.mojo \
  -o build/contract-tests/metal-logical-words
build/contract-tests/metal-logical-words

echo "== Native numeric device conformance"
pixi run --locked mojo build -O3 -I . scripts/check_metal_numeric_device.mojo \
  -o build/contract-tests/metal-numeric-device
build/contract-tests/metal-numeric-device

echo "== Exact Fraction division oracle and actual-device divmod checks"
pixi run --locked python scripts/check_metal_numeric_accuracy.py \
  --output build/contract-tests/metal-divmod-oracle
pixi run --locked mojo build -O3 -I . scripts/check_metal_divmod_device.mojo \
  -o build/contract-tests/metal-divmod-device
build/contract-tests/metal-divmod-device \
  build/contract-tests/metal-divmod-oracle/fixtures.csv

echo "== Shared value interpreter control flow and checked faults"
pixi run --locked mojo build -O3 -I . scripts/check_metal_value_stack_device.mojo \
  -o build/contract-tests/metal-value-stack-device
build/contract-tests/metal-value-stack-device

echo "== Scalar and general expression fields in the production composed scan"
pixi run --locked mojo build -O3 -I . scripts/check_metal_scalar_fields.mojo \
  -o build/contract-tests/metal-scalar-fields
build/contract-tests/metal-scalar-fields

echo "== Production grouped value interpreter, including nullable inputs"
pixi run --locked mojo build -O3 -I . scripts/check_metal_numeric_group_kernel.mojo \
  -o build/contract-tests/metal-numeric-group-kernel
build/contract-tests/metal-numeric-group-kernel
echo "== Null/NaN device conformance"
pixi run --locked mojo build -O3 -I . scripts/check_metal_null_device.mojo \
  -o build/contract-tests/metal-null-device
build/contract-tests/metal-null-device

echo "== Boolean predicate device conformance"
pixi run --locked mojo build -O3 -I . scripts/check_metal_logical_device.mojo \
  -o build/contract-tests/metal-logical-device
build/contract-tests/metal-logical-device

echo "== Packed Boolean storage device conformance"
pixi run --locked mojo build -O3 -I . scripts/check_metal_bool_storage.mojo \
  -o build/contract-tests/metal-bool-storage
build/contract-tests/metal-bool-storage

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

echo "== Metal suite tooling checks"
for task in test-metal-suite-checks; do
  pixi run --locked -e gpu "$task" || failed="$failed $task"
done
if [[ -n "$failed" ]]; then
  echo "Failed:$failed" >&2
  exit 1
fi
echo "All Metal contracts passed."
