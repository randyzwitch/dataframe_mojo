#!/usr/bin/env bash
# Run via pixi so mojo and its runtime libraries are available.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

case "$(uname -s)" in
  Darwin)
    LIB=libdfparquet.dylib
    # Mach-O C symbols have a leading underscore; -U excludes undefineds.
    nm -gjU "build/dfparquet/$LIB" | sed 's/^_//' | LC_ALL=C sort > "$WORK/exports"
    ;;
  Linux)
    LIB=libdfparquet.so
    nm -D --defined-only --format=posix "build/dfparquet/$LIB" | \
      awk '{print $1}' | LC_ALL=C sort > "$WORK/exports"
    ;;
  *) echo "Unsupported platform: $(uname -s)" >&2; exit 1;;
esac
sed 's/^_//' native/dfparquet/dfparquet.sym | LC_ALL=C sort > "$WORK/expected"
diff -u "$WORK/expected" "$WORK/exports"

mojo build -I . tests/test_parquet.mojo -o "$WORK/test_parquet"
run_tests() {
  local log=$1
  shift
  if ! "$@" > "$log" 2>&1; then
    cat "$log"
    return 1
  fi
  cat "$log"
  if grep -q 'skipped:' "$log"; then
    echo 'Parquet tests skipped: the library must load in both locations' >&2
    return 1
  fi
}

echo 'Testing build/dfparquet lookup'
run_tests "$WORK/build.log" env -u DATAFRAME_PARQUET_LIBRARY -u CONDA_PREFIX \
  "$WORK/test_parquet"

# Keep the fixtures available, but no build/dfparquet directory: a successful
# read here proves the prefix search path works without the fallback.
mkdir -p "$WORK/prefix/lib" "$WORK/run"
cp "build/dfparquet/$LIB" "$WORK/prefix/lib/"
ln -s "$ROOT/tests" "$WORK/run/tests"
cd "$WORK/run"
echo 'Testing CONDA_PREFIX/lib lookup'
run_tests "$WORK/prefix.log" env -u DATAFRAME_PARQUET_LIBRARY \
  CONDA_PREFIX="$WORK/prefix" "$WORK/test_parquet"
