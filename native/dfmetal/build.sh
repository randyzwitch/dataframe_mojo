#!/bin/bash
# Build the optional Apple Metal runtime using the selected Xcode SDK.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-build/dfmetal}
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "dfmetal requires Apple Silicon macOS" >&2
  exit 1
fi
mkdir -p "$OUT"

# The native source, ABI, compiler, SDK, and flags identify cost evidence.
# A different compiler or implementation must not reuse an old calibration.
IDENTITY=$(python3 - "$HERE" <<'PY'
import hashlib
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
identity = hashlib.sha256()
for name in ["build.sh", "dfmetal.h", "dfmetal.mm", "dfmetal.sym"]:
    identity.update(name.encode())
    identity.update((root / name).read_bytes())
for command in [["xcrun", "clang++", "--version"],
                ["xcrun", "--show-sdk-version"]]:
    identity.update(subprocess.check_output(command))
print(identity.hexdigest())
PY
)
xcrun clang++ -O3 -ffp-contract=off -std=c++20 -fobjc-arc \
  -mmacosx-version-min=15.0 -fvisibility=hidden \
  -DDFM_BUILD_ID="\"$IDENTITY\"" \
  -dynamiclib -Wl,-install_name,@rpath/libdfmetal.dylib \
  -Wl,-exported_symbols_list,"$HERE/dfmetal.sym" \
  "$HERE/dfmetal.mm" -framework Metal -framework Foundation \
  -o "$OUT/libdfmetal.dylib"
printf 'Built %s/libdfmetal.dylib (%s)\n' "$OUT" "$IDENTITY"
