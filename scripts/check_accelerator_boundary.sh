#!/usr/bin/env bash
# Check the core/accelerator boundary (docs/design/accelerator-contract.md).
#
# Core code reaches an accelerator backend only through
# dataframe/_accel_provider.mojo. This script fails, listing every offending
# import, when:
#   1. a .mojo file under dataframe/ outside the backend imports
#      dataframe.accelerator, .accelerator, dataframe.metal or .metal;
#   2. a file under dataframe/accelerator/ imports from tests;
#   3. a core file other than the allowed importers imports _accel_provider.
#
# Every allow-list entry below is a deliberate decision. Adding one needs a
# reason in the comment beside it.
set -euo pipefail
cd "$(dirname "$0")/.."

# Files that are the backend or its only door. They may import the Metal
# backend. dataframe/accelerator/ is matched by prefix.
backend_files=(
  dataframe/_accel_provider.mojo  # the provider slot; hard-imports .metal today
  dataframe/metal.mojo            # the Metal runtime, part of the backend
)

# Core files allowed to import _accel_provider. Both dispatch to the provider;
# nothing else in core should know a provider exists.
provider_importers=(
  dataframe/lazy.mojo               # execute, describe, select_auto, ...
  dataframe/execution_backend.mojo  # installed(), for engine="accel"
)

# KNOWN VIOLATION, recorded rather than hidden. lazy.mojo imports these
# accelerator modules directly (plan-estimate and key-query value types that
# the AcceleratorBackend trait and the planner use). Each entry is
# "file module". The migration plan (step 2) moves these types out of the
# backend package; remove the entries then. Do not add to this list.
known_core_imports=(
  "dataframe/lazy.mojo accelerator.cost"
  "dataframe/lazy.mojo accelerator.window_keys"
  "dataframe/lazy.mojo accelerator.grouped_key_queries"
  "dataframe/lazy.mojo accelerator.opaque_dictionary"
  "dataframe/lazy.mojo accelerator.code_domains"
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

# Matches "from X import" / "import X" lines whose module path starts with
# the given name (relative dots or the dataframe. prefix allowed).
import_pattern() {
  printf '^[[:space:]]*(from|import)[[:space:]]+(dataframe\\.|\\.+)?%s([.[:space:]]|$)' "$1"
}

while IFS= read -r file; do
  [[ $file == dataframe/accelerator/* ]] && continue
  in_list "$file" "${backend_files[@]}" && continue
  while IFS= read -r hit; do
    [[ -z $hit ]] && continue
    line=${hit%%:*}
    text=${hit#*:}
    module=$(printf '%s' "$text" | sed -E 's/^[[:space:]]*(from|import)[[:space:]]+(dataframe\.|\.+)?([A-Za-z_.]+).*/\3/')
    if in_list "$file $module" "${known_core_imports[@]}"; then
      continue
    fi
    fail "core imports the backend: $file:$line: $text"
  done < <(grep -nE "$(import_pattern accelerator)|$(import_pattern metal)" "$file" || true)
done < <(find dataframe -name '*.mojo' | sort)

while IFS= read -r file; do
  while IFS= read -r hit; do
    [[ -z $hit ]] && continue
    fail "backend imports tests: $file:${hit%%:*}: ${hit#*:}"
  done < <(grep -nE '^[[:space:]]*(from|import)[[:space:]]+(\.+)?tests([.[:space:]]|$)' "$file" || true)
done < <(find dataframe/accelerator -name '*.mojo' | sort)

while IFS= read -r file; do
  in_list "$file" "${backend_files[@]}" "${provider_importers[@]}" && continue
  [[ $file == dataframe/accelerator/* ]] && continue
  while IFS= read -r hit; do
    [[ -z $hit ]] && continue
    fail "unlisted importer of _accel_provider: $file:${hit%%:*}: ${hit#*:}"
  done < <(grep -nE "$(import_pattern _accel_provider)" "$file" || true)
done < <(find dataframe -name '*.mojo' | sort)

if ((failures > 0)); then
  echo "Accelerator boundary check failed: $failures violation(s)." >&2
  echo "See docs/design/accelerator-contract.md." >&2
  exit 1
fi
echo "Accelerator boundary holds."
