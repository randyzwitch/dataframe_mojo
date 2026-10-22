#!/usr/bin/env bash
# One Apple silicon checkpoint report: CPU and forced Metal timings.
#
#     pixi run -e oracle report-apple          # dev scale, 3 rounds
#     pixi run -e oracle report-apple-quick    # smoke scale, 1 round
#
# Steps: exact CPU/Metal answer receipt, then prepare inputs, build and run
# the timing workers, then render. Output goes to build/reports/apple/.
# Settings come from the environment:
#
#   APPLE_SCALE     smoke | dev | default | large   (default dev)
#   APPLE_SUITES    comma list                      (default h2o_groupby,h2o_join,pdsh,tpcds)
#   APPLE_QUERIES   comma list of query ids         (default all)
#   APPLE_ROUNDS    rounds per case                 (default 3)
#   APPLE_REPS      warm repetitions per worker     (default 3)
#   APPLE_WARMUPS   extra warmups per worker        (default 1)
#   APPLE_THREADS   worker threads                  (default 8)
#   APPLE_OUT       output directory                (default build/reports/apple)
#
# Run on an idle Apple silicon Mac with a clean dataframe/ tree.
set -euo pipefail
cd "$(dirname "$0")/../.."

scale=${APPLE_SCALE:-dev}
suites=${APPLE_SUITES:-h2o_groupby,h2o_join,pdsh,tpcds}
rounds=${APPLE_ROUNDS:-3}
reps=${APPLE_REPS:-3}
warmups=${APPLE_WARMUPS:-1}
threads=${APPLE_THREADS:-8}
out=${APPLE_OUT:-build/reports/apple}
queries=()
if [ -n "${APPLE_QUERIES:-}" ]; then
    queries=(--queries "$APPLE_QUERIES")
fi

if [ "$(uname -s)-$(uname -m)" != "Darwin-arm64" ]; then
    echo "report-apple needs an Apple silicon Mac" >&2
    exit 1
fi
if pgrep -l mojo >/dev/null; then
    echo "a mojo process is running; wait for it to finish" >&2
    exit 1
fi

rm -rf "$out"
mkdir -p "$out"
tools=benchmarks/tools

echo "== exact CPU/Metal receipt ($scale)" >&2
# It exits nonzero when any case gives a wrong answer or fails; the receipt
# still holds the whole grid and the report lists those cases as excluded.
python3 $tools/bench_metal_suites.py --suites "$suites" --scale "$scale" \
    ${queries[@]+"${queries[@]}"} --threads "$threads" \
    --output "$out/coverage.json" ||
    echo "some cases did not match; they are excluded from the ratios" >&2

driver=(python3 $tools/bench_apple_performance.py --engine-tree .
    --output "$out/performance.json" --suites "$suites" --scale "$scale"
    ${queries[@]+"${queries[@]}"} --rounds "$rounds" --reps "$reps"
    --warmups "$warmups" --threads "$threads")

echo "== prepare" >&2
"${driver[@]}" --prepare-only >/dev/null
"${driver[@]}" --prepare-inputs
echo "== build" >&2
"${driver[@]}" --build-only
echo "== run" >&2
"${driver[@]}" --strict-coverage "$out/coverage.json"
echo "== render" >&2
python3 $tools/apple_performance_html.py "$out/performance.json" \
    --strict-coverage "$out/coverage.json" --output "$out/results.html"
echo "report: $out/results.html" >&2
