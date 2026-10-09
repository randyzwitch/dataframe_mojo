#!/usr/bin/env bash
# Rule 2 in docs/benchmarks.md: a PR that changes the engine
# (dataframe/)
# must not also change what measures it (benchmarks/, scripts/bench_*),
# unless a commit explains why in a "Benchmark-Change:" trailer.
#
# Usage: scripts/check_benchmark_separation.sh [BASE_REF]   (default origin/main)
set -euo pipefail
cd "$(dirname "$0")/.."
base_ref="${1:-origin/main}"
base="$(git merge-base "$base_ref" HEAD)"
changed="$(git diff --name-only "$base" HEAD)"
engine="$(grep -E '^dataframe/' <<<"$changed" || true)"
bench="$(grep -E '^(benchmarks/|scripts/bench_)' <<<"$changed" || true)"
if [ -z "$engine" ] || [ -z "$bench" ]; then
    echo "benchmark separation: ok"
    exit 0
fi
if git log --format=%B "$base..HEAD" | grep -q '^Benchmark-Change:'; then
    echo "benchmark separation: engine and benchmark changes together, explained by:"
    git log --format=%B "$base..HEAD" | grep '^Benchmark-Change:'
    exit 0
fi
echo "This PR changes both the engine and the benchmarks that measure it:" >&2
echo "$engine" | sed 's/^/  engine:    /' >&2
echo "$bench" | sed 's/^/  benchmark: /' >&2
echo "Split them into separate PRs (docs/benchmarks.md, rule 2), or explain" >&2
echo "why they must land together in a 'Benchmark-Change:' commit trailer." >&2
exit 1
