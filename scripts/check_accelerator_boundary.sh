#!/usr/bin/env bash
# Check the core/backend boundary (docs/design/accelerator-contract.md).
#
# A backend is a package outside the core (dataframe_metal). The root
# `dataframe_provider` package makes Metal the default; a build can put another
# (providers/cpu, or a third-party backend) first on the import path.
# This script fails, listing every offending import, when:
#   1. a .mojo file under dataframe/ imports dataframe_metal (absolute or
#      relative to a sibling package);
#   2. a core file other than dataframe/_accel_provider.mojo imports
#      dataframe_provider, or _accel_provider is imported by a core file that is
#      not one of the allowed dispatchers;
#   3. a file under dataframe_metal/ imports from tests;
#   4. providers/cpu/dataframe_provider imports a backend. (The root
#      dataframe_provider is the Metal provider and may import dataframe_metal.)
#   5. a test imports dataframe_metal outside the Metal test files listed in
#      metal_tests. CPU tests check the core through the public API; a backend
#      is tested in its own files, so removing or replacing it never touches
#      them.
#
# Every allow-list entry below is a deliberate decision. Adding one needs a
# reason in the comment beside it.
set -euo pipefail
cd "$(dirname "$0")/.."

# Core files allowed to import _accel_provider. Both dispatch to the provider;
# nothing else in core should know a provider exists.
provider_importers=(
  dataframe/lazy_api.mojo           # public query dispatch and runtime port adapters
  dataframe/execution_backend.mojo  # installed(), for engine="accel"
)

# Test files allowed to import dataframe_metal: the backend's own tests, the
# provider dispatch tests, and the tooling programs that probe a device.
metal_tests=(
  tests/execution_modes/test_backends.mojo  # provider dispatch on a CPU target
  tests/execution_modes/test_gpu_typed_storage_metadata.mojo
  tests/execution_modes/test_mojo_gpu_scratch_pool.mojo
  tests/support/metal_outcomes.mojo         # Metal-vs-CPU result helpers
  tests/tooling/accelerator_package_consumer.mojo  # packaged provider hooks
  tests/tooling/metal_available.mojo        # device probe
  tests/tooling/mojo_gpu_cpu_compile.mojo   # CPU-target build of the backend
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
done < <(find providers/cpu -name '*.mojo' | sort)

while IFS= read -r file; do
  [[ $file == tests/execution_modes/test_metal_* ]] && continue
  in_list "$file" "${metal_tests[@]}" && continue
  while IFS= read -r hit; do
    [[ -z $hit ]] && continue
    fail "a CPU test imports the backend: $file:${hit%%:*}: ${hit#*:}"
  done < <(grep -nE "$(absolute_import dataframe_metal)" "$file" || true)
done < <(find tests -name '*.mojo' | sort)

if ((failures > 0)); then
  echo "Accelerator boundary check failed: $failures violation(s)." >&2
  echo "See docs/design/accelerator-contract.md." >&2
  exit 1
fi
echo "Accelerator boundary holds."
