#!/usr/bin/env bash
# Whole-pipeline #108 matrix. Do not start this until the benchmark owner has
# established a quiet measurement window.
set -euo pipefail

REPO="${REPO:-/home/randyzwitch/GitHub/dataframe_mojo}"
TOPK_BIN="${TOPK_BIN:-/tmp/df_topk_experiment/topk_pipeline_108}"
RADIX_BIN="${RADIX_BIN:-/tmp/df_radix_experiment/radix_pipeline_108}"
POLARS_SCRIPT="${POLARS_SCRIPT:-$REPO/experiments/cpu110/sort_matrix_polars.py}"
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-$REPO/experiments/cpu110/results}"

for binary in "$TOPK_BIN" "$RADIX_BIN"; do
    if [[ ! -x "$binary" ]]; then
        echo "missing executable: $binary" >&2
        exit 2
    fi
done
if [[ ! -f "$POLARS_SCRIPT" ]]; then
    echo "missing Polars oracle: $POLARS_SCRIPT" >&2
    exit 2
fi

mkdir -p "$RESULTS_DIR"
MANIFEST="$RESULTS_DIR/sort_matrix_${RUN_ID}.manifest"
{
    echo "run_id=$RUN_ID"
    echo "git_head=$(git -C "$REPO" rev-parse HEAD)"
    echo "topk_bin=$TOPK_BIN"
    echo "radix_bin=$RADIX_BIN"
    echo "polars_script=$POLARS_SCRIPT"
    echo "matrix=rows:{1000000,10000000};kind:{int64,string};threads:{1,32}"
} > "$MANIFEST"

run_case() {
    local candidate="$1"
    local binary="$2"
    local rows="$3"
    local kind="$4"
    local threads="$5"
    local log="$RESULTS_DIR/${candidate}_${kind}_${rows}_t${threads}_${RUN_ID}.log"
    printf 'running %s rows=%s kind=%s threads=%s\n' \
        "$candidate" "$rows" "$kind" "$threads" | tee "$log"
    env DATAFRAME_THREADS="$threads" "$binary" "$rows" "$kind" \
        2>&1 | tee -a "$log"
}

run_polars() {
    local workload="$1"
    local rows="$2"
    local kind="$3"
    local threads="$4"
    local log="$RESULTS_DIR/polars_${workload}_${kind}_${rows}_t${threads}_${RUN_ID}.log"
    printf 'running polars %s rows=%s kind=%s threads=%s\n' \
        "$workload" "$rows" "$kind" "$threads" | tee "$log"
    (
        cd "$REPO"
        env POLARS_MAX_THREADS="$threads" pixi run -e oracle python \
            "$POLARS_SCRIPT" "$workload" "$rows" "$kind"
    ) 2>&1 | tee -a "$log"
}

for threads in 1 32; do
    for rows in 1000000 10000000; do
        for kind in int64 string; do
            run_case topk "$TOPK_BIN" "$rows" "$kind" "$threads"
            run_polars topk "$rows" "$kind" "$threads"
            run_case radix "$RADIX_BIN" "$rows" "$kind" "$threads"
            run_polars radix "$rows" "$kind" "$threads"
        done
    done
done

echo "logs=$RESULTS_DIR run_id=$RUN_ID"
