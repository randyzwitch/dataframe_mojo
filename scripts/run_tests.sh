#!/usr/bin/env bash
# Run every test module; fail on the first failing module.
set -euo pipefail
cd "$(dirname "$0")/.."
for test in tests/test_*.mojo; do
    echo "== $test"
    mojo run -I . "$test"
done
