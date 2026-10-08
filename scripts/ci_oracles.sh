#!/usr/bin/env bash
# Keep each interoperability check serial within this runner.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 scripts/ci_time.py --label oracle-compile -- pixi run -e oracle oracle --build-only
python3 scripts/ci_time.py --label oracle-cases -- pixi run -e oracle oracle --runner build/oracle_runner --cases 150 --seed "${ORACLE_SEED:-1}"
python3 scripts/ci_time.py --label oracle-mutation-check -- pixi run -e oracle oracle --runner build/oracle_runner --mutation-check --cases 20
python3 scripts/ci_time.py --label oracle-arrow -- pixi run -e oracle oracle-arrow
python3 scripts/ci_time.py --label oracle-geometry -- pixi run -e oracle oracle-geometry
python3 scripts/ci_time.py --label oracle-geoparquet -- pixi run -e oracle oracle-geoparquet
python3 scripts/ci_time.py --label oracle-shapefile -- pixi run -e oracle oracle-shapefile
python3 scripts/ci_time.py --label oracle-asof -- pixi run -e oracle oracle-asof
python3 scripts/ci_time.py --label oracle-time-windows -- pixi run -e oracle oracle-time-windows
python3 scripts/ci_time.py --label check_parquet_write -- pixi run -e oracle python3 scripts/check_parquet_write.py
python3 scripts/ci_time.py --label check_parquet_stream -- pixi run -e oracle python3 scripts/check_parquet_stream.py build/dfparquet/libdfparquet.so
