#!/usr/bin/env bash
# Build the precompiled package and run an example from outside the source
# tree, so only dist/dataframe.mojoc can satisfy `import dataframe`.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
mkdir -p dist
mojo precompile dataframe -o dist/dataframe.mojoc
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
