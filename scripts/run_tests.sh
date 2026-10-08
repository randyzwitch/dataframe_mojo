#!/usr/bin/env bash
# Run test modules in parallel and report every failure.
#
# Usage: scripts/run_tests.sh [tests/test_x.mojo ...]
#
# With no arguments, every module runs, grouped into TEST_GROUPS driver
# programs (default 10; see scripts/test_drivers.py). Compiling dominates a
# test run, and each program compiles the library code it uses again, so one
# program per group instead of per module cuts the compile work several
# times over. Named modules run one program each, as a quick local check.
# TEST_JOBS sets how many programs build and run at once (default: CPU
# count). Output is buffered per program and printed in a stable order.
set -euo pipefail
cd "$(dirname "$0")/.."

logs="$(mktemp -d)"
trap 'rm -rf "$logs"' EXIT
if [ "$#" -gt 0 ]; then
    tests=("$@")
    modules=${#tests[@]}
else
    modules=$(ls tests/test_*.mojo | wc -l)
    tests=()
    while IFS= read -r program; do
        tests+=("$program")
    done < <(
        python3 scripts/test_drivers.py "$logs/drivers" "${TEST_GROUPS:-10}" \
            tests/test_*.mojo
    )
fi
jobs="${TEST_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
start=$SECONDS
printf '%s\n' "${tests[@]}" | xargs -P "$jobs" -I {} bash -c '
    test="$1"
    log="$2/$(basename "$test")"
    binary="$log.bin"
    if [ "$test" = tests/test_parquet.mojo ] && [ -n "${TEST_PARQUET_BINARY:-}" ]; then
        binary="$TEST_PARQUET_BINARY"
        if [ ! -x "$binary" ]; then
            echo "Parquet test binary is not executable: $binary" >"$log.out"
            echo fail >"$log.status"
            exit 0
        fi
    elif ! python3 scripts/ci_time.py --label "compile:$test" -- \
        mojo build -I . -I tests "$test" -o "$binary" >"$log.out" 2>&1; then
        echo fail >"$log.status"
        exit 0
    fi
    if python3 scripts/ci_time.py --label "execute:$test" -- "$binary" >>"$log.out" 2>&1; then
        echo pass >"$log.status"
    else
        echo fail >"$log.status"
    fi
' _ {} "$logs"

failed=()
for test in "${tests[@]}"; do
    name="$(basename "$test")"
    echo "== $test"
    cat "$logs/$name.out"
    if [ "$(cat "$logs/$name.status" 2>/dev/null)" != "pass" ]; then
        failed+=("$test")
    fi
done

echo
echo "$modules modules in ${#tests[@]} programs, ${#failed[@]} programs failed, $((SECONDS - start))s with $jobs jobs"
if [ "${#failed[@]}" -gt 0 ]; then
    # A driver names each failing module; a crash names none, so the
    # program is listed too.
    for test in "${failed[@]}"; do
        grep -h '^FAILED: ' "$logs/$(basename "$test").out" || true
        echo "FAILED: $test"
    done
    exit 1
fi
