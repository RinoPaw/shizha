#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-D:/Projects/Packages/识诈/src/anti_fraud_explorer/api.py}"
EXPECTED_MODIFIED="1ec39f51c4cc9f88b2458b47a7ec400ddcbe72dd2effe7c5d6689e14580656db"
EXPECTED_ORIGINAL="10f0d84b52c9608524d9f4734b037bbac92c6bfd06127c4a7ed0f59151cc012e"
ACTUAL="$(sha256sum "$TARGET" | awk '{print $1}')"
if [[ "$ACTUAL" != "$EXPECTED_MODIFIED" ]]; then
  echo "ROLLBACK REFUSED hash=$ACTUAL expected=$EXPECTED_MODIFIED"
  exit 2
fi
cp "$SCRIPT_DIR/api.py.original" "$TARGET"
RESTORED="$(sha256sum "$TARGET" | awk '{print $1}')"
[[ "$RESTORED" == "$EXPECTED_ORIGINAL" ]]
echo "ROLLBACK PASS restored_sha=$RESTORED fallback=false"