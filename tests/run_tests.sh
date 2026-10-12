#!/usr/bin/env bash
# Primary suite: independent dataframe contracts.
set -euo pipefail
cd "$(dirname "$0")/.."
mojo run -I . tests/runner.mojo "$@"
