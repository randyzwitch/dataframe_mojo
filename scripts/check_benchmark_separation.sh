#!/usr/bin/env bash
# Rule 2 in docs/design/benchmarks.md: a PR that changes the engine
# (dataframe/, dataframe_metal/, dataframe_provider/, providers/)
# must not also change what measures it (benchmarks/, which holds the
# suites, the mechanism programs and their drivers in benchmarks/tools/),
# unless a commit explains why in a "Benchmark-Change:" trailer.
#
# Usage: scripts/check_benchmark_separation.sh [BASE_REF] [HEAD_REF]   (defaults origin/main HEAD)
set -euo pipefail
cd "$(dirname "$0")/.."
base_ref="${1:-origin/main}"
head_ref="${2:-HEAD}"
base="$(git merge-base "$base_ref" "$head_ref")"
changed="$(git diff --name-only "$base" "$head_ref")"
engine="$(grep -E '^(dataframe|dataframe_metal|dataframe_provider|providers)/' <<<"$changed" || true)"
bench="$(grep -E '^benchmarks/' <<<"$changed" || true)"
if [ -z "$engine" ] || [ -z "$bench" ]; then
    echo "benchmark separation: ok"
    exit 0
fi
if git log --format=%B "$base..$head_ref" | grep -q '^Benchmark-Change:'; then
    echo "benchmark separation: engine and benchmark changes together, explained by:"
    git log --format=%B "$base..$head_ref" | grep '^Benchmark-Change:'
    exit 0
fi
echo "This PR changes both the engine and the benchmarks that measure it:" >&2
echo "$engine" | sed 's/^/  engine:    /' >&2
echo "$bench" | sed 's/^/  benchmark: /' >&2
echo "Split them into separate PRs (docs/design/benchmarks.md, rule 2), or explain" >&2
echo "why they must land together in a 'Benchmark-Change:' commit trailer." >&2
exit 1
