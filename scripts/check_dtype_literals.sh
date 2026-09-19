#!/usr/bin/env bash
# Library code must use DataType constants, not dtype name strings; names are
# parsed only in dataframe/dtype.mojo.
set -euo pipefail
cd "$(dirname "$0")/.."
if grep -nE '"(int64|float64|bool|string)"' dataframe/*.mojo | grep -v '^dataframe/dtype.mojo:'; then
    echo "dtype string literals found outside dataframe/dtype.mojo" >&2
    exit 1
fi
