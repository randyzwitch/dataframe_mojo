#!/usr/bin/env bash
# Build the precompiled package and run an example from outside the source
# tree, so only dist/*.mojoc can satisfy `import dataframe`.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
mkdir -p dist
python3 scripts/build_package.py --output dist
# The routing contracts below use the Metal backend, so they need the Metal
# flavor: dataframe_metal and a provider that re-exports it.
python3 scripts/build_package.py --provider metal --output dist-metal
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cp examples/sales.mojo "$work/"
cd "$work"
output="$(mojo run -I "$root/dist" sales.mojo)"
echo "$output"
if [[ "$output" != $'east 90.0\nwest 180.0' ]]; then
    echo "unexpected package smoke output" >&2
    exit 1
fi

cd "$root"
mojo run -I . tests/tooling/package_contracts.mojo "$root/dist-metal" "$@"
