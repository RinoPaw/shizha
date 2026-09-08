#!/usr/bin/env bash
set -euo pipefail
ARTIFACT_POSIX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_POSIX="${1:-$(cd "$ARTIFACT_POSIX/.." && pwd)}"
to_windows_path() {
  local value="$1"
  if [[ "$value" =~ ^/mnt/([A-Za-z])/(.*)$ ]]; then
    local drive="${BASH_REMATCH[1]^}"
    local rest="${BASH_REMATCH[2]//\//\\}"
    printf '%s:\\%s' "$drive" "$rest"
  else
    printf '%s' "$value"
  fi
}
ARTIFACT_DIR="$(to_windows_path "$ARTIFACT_POSIX")"
TARGET_DIR="$(to_windows_path "$TARGET_POSIX")"
PYTHON_BIN="$ARTIFACT_POSIX/../../.venv/Scripts/python.exe"
"$PYTHON_BIN" - "$ARTIFACT_DIR" "$TARGET_DIR" <<'PY'
import json, shutil, sys
from pathlib import Path
artifact = Path(sys.argv[1])
target = Path(sys.argv[2])
manifest = json.loads((artifact / "manifest.json").read_text("utf-8"))
for item in manifest:
    rel = Path(item["target"])
    destination = target / rel
    original = artifact / "original" / rel
    if item.get("existed"):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
    elif destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
print(f"restored {len(manifest)} paths")
PY
