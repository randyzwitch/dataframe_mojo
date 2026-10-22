#!/usr/bin/env bash
# Check the core/backend boundary (docs/design/accelerator-contract.md).
#
# A backend is a package outside the core (dataframe_metal) chosen at build
# time by which `dataframe_provider` package comes first on the import path.
# This script fails, listing every offending import, when:
#   1. a .mojo file under dataframe/ imports dataframe_metal (absolute or
#      relative to a sibling package);
#   2. a core file other than dataframe/_accel_provider.mojo imports
#      dataframe_provider, or _accel_provider is imported by a core file that is
#      not one of the allowed dispatchers;
#   3. a file under dataframe_metal/ imports from tests;
#   4. the root dataframe_provider/ package imports a backend.
#
# Every allow-list entry below is a deliberate decision. Adding one needs a
# reason in the comment beside it.
set -euo pipefail
cd "$(dirname "$0")/.."

# Core files allowed to import _accel_provider. Both dispatch to the provider;
# nothing else in core should know a provider exists.
provider_importers=(
  dataframe/lazy.mojo               # execute, describe, select_auto, ...
  dataframe/execution_backend.mojo  # installed(), for engine="accel"
)

failures=0
fail() {
  printf '%s\n' "$1"
  failures=$((failures + 1))
}

in_list() {
  local needle=$1
  shift
  local item
  for item in "$@"; do
    [[ $item == "$needle" ]] && return 0
  done
  return 1
}

# Matches "from X import" / "import X" lines whose module path starts with X.
# Relative dots are allowed only where the caller asks for them.
absolute_import() {
  printf '^[[:space:]]*(from|import)[[:space:]]+%s([.[:space:]]|$)' "$1"
}
relative_import() {
  printf '^[[:space:]]*(from|import)[[:space:]]+(dataframe\\.|\\.+)?%s([.[:space:]]|$)' "$1"
}

while IFS= read -r file; do
  while IFS= read -r hit; do
    [[ -z $hit ]] && continue
    fail "core imports the backend: $file:${hit%%:*}: ${hit#*:}"
  done < <(grep -nE "$(absolute_import dataframe_metal)" "$file" || true)
  # The old in-package paths must not come back.
  while IFS= read -r hit; do
    [[ -z $hit ]] && continue
    fail "core imports a backend module: $file:${hit%%:*}: ${hit#*:}"
  done < <(grep -nE "$(relative_import accelerator)|$(relative_import metal)" "$file" || true)
done < <(find dataframe -name '*.mojo' | sort)

while IFS= read -r file; do
  [[ $file == dataframe/_accel_provider.mojo ]] && continue
  while IFS= read -r hit; do
    [[ -z $hit ]] && continue
    fail "core imports dataframe_provider outside _accel_provider: $file:${hit%%:*}: ${hit#*:}"
  done < <(grep -nE "$(absolute_import dataframe_provider)" "$file" || true)
  in_list "$file" "${provider_importers[@]}" && continue
  while IFS= read -r hit; do
    [[ -z $hit ]] && continue
    fail "unlisted importer of _accel_provider: $file:${hit%%:*}: ${hit#*:}"
  done < <(grep -nE "$(relative_import _accel_provider)" "$file" || true)
done < <(find dataframe -name '*.mojo' | sort)

while IFS= read -r file; do
  while IFS= read -r hit; do
    [[ -z $hit ]] && continue
    fail "backend imports tests: $file:${hit%%:*}: ${hit#*:}"
  done < <(grep -nE '^[[:space:]]*(from|import)[[:space:]]+(\.+)?tests([.[:space:]]|$)' "$file" || true)
done < <(find dataframe_metal -name '*.mojo' | sort)

while IFS= read -r file; do
  while IFS= read -r hit; do
    [[ -z $hit ]] && continue
    fail "the CPU-only provider imports a backend: $file:${hit%%:*}: ${hit#*:}"
  done < <(grep -nE "$(absolute_import dataframe_metal)" "$file" || true)
done < <(find dataframe_provider -name '*.mojo' | sort)

if ((failures > 0)); then
  echo "Accelerator boundary check failed: $failures violation(s)." >&2
  echo "See docs/design/accelerator-contract.md." >&2
  exit 1
fi
echo "Accelerator boundary holds."
