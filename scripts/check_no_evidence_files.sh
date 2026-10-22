#!/usr/bin/env bash
# Rule 7 in docs/design/benchmarks.md: the repo holds code that produces reports,
# never the reports. Fails if tracked files include benchmark results or
# generated data/pages under docs/.
#
# Usage: scripts/check_no_evidence_files.sh
set -euo pipefail
cd "$(dirname "$0")/.."
offenders="$(git ls-files -- 'benchmarks/results/' 'docs/*.csv' 'docs/*.json' 'docs/*.log' 'docs/*.html' 'docs/*.js')"
if [ -n "$offenders" ]; then
    echo "evidence files are tracked (keep reports out of the tree, link a commit permalink instead):" >&2
    echo "$offenders" >&2
    exit 1
fi
echo "no evidence files: ok"
