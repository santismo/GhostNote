#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_DIR="$ROOT_DIR/bridge"
PORT="${PORT:-8000}"

if [ ! -d "$BRIDGE_DIR/.venv" ]; then
  python3 -m venv "$BRIDGE_DIR/.venv"
fi

# shellcheck disable=SC1091
source "$BRIDGE_DIR/.venv/bin/activate"

python -m pip install -r "$BRIDGE_DIR/requirements.txt"

export GHOSTNOTE_NO_BROWSER=1
python "$BRIDGE_DIR/app.py" &
BACKEND_PID=$!

python -m http.server "$PORT" --directory "$ROOT_DIR" &
HTTP_PID=$!

cleanup() {
  kill "$BACKEND_PID" "$HTTP_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "GhostNote bridge running."
echo "Open http://localhost:$PORT/index.html"

wait
