#!/bin/bash
# Build libdfparquet.so: a minimal static Arrow C++ (Parquet reader, bundled
# codecs, mimalloc) behind three C symbols, depending only on libc and libm.
#
#   ARROW_VERSION=24.0.0 bash native/dfparquet/build.sh [OUT_DIR]
#
# Needs cmake >= 3.25, ninja, a C++20 compiler and network access for the
# Arrow source tarball and its bundled dependencies. Takes about a minute on
# 48 cores; the result is ~15 MB.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ARROW_VERSION=${ARROW_VERSION:-24.0.0}
OUT=${1:-$HERE/out}
WORK=${DFPARQUET_WORK:-$OUT/work}
SRC=$WORK/arrow-apache-arrow-$ARROW_VERSION/cpp
BUILD=$WORK/arrow-build
mkdir -p "$OUT" "$WORK"

if [ ! -d "$SRC" ]; then
  curl -sSL -o "$WORK/arrow.tar.gz" \
    "https://github.com/apache/arrow/archive/refs/tags/apache-arrow-$ARROW_VERSION.tar.gz"
  tar xzf "$WORK/arrow.tar.gz" -C "$WORK"
fi

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DARROW_BUILD_SHARED=OFF -DARROW_BUILD_STATIC=ON \
  -DARROW_PARQUET=ON \
  -DARROW_WITH_SNAPPY=ON -DARROW_WITH_ZSTD=ON -DARROW_WITH_LZ4=ON -DARROW_WITH_ZLIB=ON \
  -DARROW_WITH_BROTLI=OFF -DARROW_WITH_BZ2=OFF \
  -DARROW_DEPENDENCY_SOURCE=BUNDLED -DARROW_DEPENDENCY_USE_SHARED=OFF \
  -DARROW_MIMALLOC=ON -DARROW_JEMALLOC=OFF \
  -DARROW_FILESYSTEM=OFF -DARROW_S3=OFF -DARROW_GCS=OFF -DARROW_AZURE=OFF -DARROW_HDFS=OFF \
  -DARROW_JSON=OFF -DARROW_CSV=OFF -DARROW_IPC=OFF -DARROW_DATASET=OFF -DARROW_ACERO=OFF \
  -DARROW_FLIGHT=OFF -DARROW_GANDIVA=OFF -DARROW_SUBSTRAIT=OFF -DARROW_ORC=OFF \
  -DARROW_WITH_OPENTELEMETRY=OFF -DARROW_WITH_RE2=OFF -DARROW_WITH_UTF8PROC=OFF \
  -DARROW_BUILD_TESTS=OFF -DARROW_BUILD_BENCHMARKS=OFF -DARROW_BUILD_INTEGRATION=OFF \
  -DARROW_BUILD_UTILITIES=OFF -DPARQUET_REQUIRE_ENCRYPTION=OFF -DPARQUET_BUILD_EXECUTABLES=OFF \
  -DARROW_USE_CCACHE=OFF
cmake --build "$BUILD" -j "${JOBS:-$(nproc)}"

case "$(uname -s)" in
  Darwin) LIB=$OUT/libdfparquet.dylib; LINK_FLAGS=(-Wl,-exported_symbols_list,"$HERE/dfparquet.sym");;
  *)      LIB=$OUT/libdfparquet.so
          LINK_FLAGS=(-Wl,--version-script="$HERE/dfparquet.map" -Wl,--exclude-libs,ALL -static-libstdc++ -static-libgcc);;
esac
${CXX:-c++} -O2 -std=c++20 -fPIC -shared \
  -I"$SRC/src" -I"$BUILD/src" "$HERE/dfparquet.cc" -o "$LIB" \
  -L"$BUILD/release" \
  -Wl,--start-group -lparquet -larrow -larrow_bundled_dependencies -Wl,--end-group \
  -lpthread -ldl "${LINK_FLAGS[@]}"
strip --strip-unneeded "$LIB" 2>/dev/null || true
ls -la "$LIB"
