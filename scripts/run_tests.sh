#!/usr/bin/env bash
# Run test modules in parallel and report every failure.
#
# Usage: scripts/run_tests.sh [tests/test_x.mojo ...]   (default: all modules)
# TEST_JOBS sets the number of concurrent modules (default: CPU count;
# TEST_JOBS=1 runs serially). Each module compiles and runs as its own
# process, and compilation dominates, so modules run side by side. Output is
# buffered per module and printed in a stable order.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -gt 0 ]; then
    tests=("$@")
else
    tests=(tests/test_*.mojo)
fi
jobs="${TEST_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
logs="$(mktemp -d)"
trap 'rm -rf "$logs"' EXIT

start=$SECONDS
printf '%s\n' "${tests[@]}" | xargs -P "$jobs" -I {} bash -c '
    test="$1"
    log="$2/$(basename "$test")"
    if mojo run -I . "$test" >"$log.out" 2>&1; then
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
echo "${#tests[@]} modules, ${#failed[@]} failed, $((SECONDS - start))s with $jobs jobs"
if [ "${#failed[@]}" -gt 0 ]; then
    printf 'FAILED: %s\n' "${failed[@]}"
    exit 1
fi
