#!/usr/bin/env bash
# Install a candidate Git revision in a fresh Pixi consumer workspace.
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "usage: $0 <git-url> <git-rev>" >&2
    exit 2
fi

root="$(cd "$(dirname "$0")/../.." && pwd)"
case "$(uname -s):$(uname -m)" in
    Linux:x86_64) platform=linux-64 ;;
    Darwin:arm64) platform=osx-arm64 ;;
    *) echo "unsupported smoke test platform" >&2; exit 2 ;;
esac

mojo_pin="$(awk '
    /^\[/ { in_dependencies = ($0 == "[dependencies]") }
    in_dependencies && /^mojo = / { print; exit }
' "$root/pixi.toml")"
if [[ -z "$mojo_pin" ]]; then
    echo "missing development Mojo pin in pixi.toml" >&2
    exit 1
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cp "$root/examples/sales.mojo" "$work/"
cd "$work"

cat > pixi.toml <<EOF
[workspace]
name = "dataframe-git-consumer"
channels = [
    "https://conda.modular.com/max-nightly",
    "https://conda.modular.com/max",
    "conda-forge",
    "https://prefix.dev/pixi-build-backends",
]
platforms = ["$platform"]
preview = ["pixi-build"]
requires-pixi = "==0.84.2"

[dependencies]
$mojo_pin
EOF

pixi add --manifest-path pixi.toml --git "$1" --rev "$2" dataframe_mojo
output="$(pixi run --manifest-path pixi.toml --locked --clean-env --executable bash -e -c '
    packages="$CONDA_PREFIX/lib/mojo"
    for module in dataframe dataframe_provider; do
        if [[ ! -s "$packages/$module.mojoc" ]]; then
            echo "missing installed module $module.mojoc" >&2
            exit 1
        fi
    done
    if [[ "$1" == osx-arm64 ]]; then
        if [[ ! -s "$packages/dataframe_metal.mojoc" ]]; then
            echo "missing installed Metal module on Apple arm64" >&2
            exit 1
        fi
    elif [[ -e "$packages/dataframe_metal.mojoc" ]]; then
        echo "unexpected installed Metal module on Linux" >&2
        exit 1
    fi
    exec mojo run sales.mojo
' -- "$platform")"
echo "$output"
if [[ "$output" != $'east 90.0\nwest 180.0' ]]; then
    echo "unexpected Pixi Git consumer query result" >&2
    exit 1
fi
