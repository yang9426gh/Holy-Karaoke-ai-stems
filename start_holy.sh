#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

# Kill old listeners (best-effort)
# (Some minimal environments don't ship with lsof; pidfiles are the primary mechanism.)
if command -v lsof >/dev/null 2>&1; then
  if lsof -tiTCP:8001 -sTCP:LISTEN >/dev/null 2>&1; then
    lsof -tiTCP:8001 -sTCP:LISTEN | xargs -r kill || true
  fi
  if lsof -tiTCP:3000 -sTCP:LISTEN >/dev/null 2>&1; then
    lsof -tiTCP:3000 -sTCP:LISTEN | xargs -r kill || true
  fi
fi

# Start backend
(
  cd "$BACKEND"
  source venv/bin/activate
  exec python -m uvicorn main:app --host 127.0.0.1 --port 8001
) >"$LOGDIR/backend.log" 2>&1 &
BACK_PID=$!

echo $BACK_PID > "$LOGDIR/backend.pid"

# Start frontend
(
  cd "$FRONTEND"
  export NEXT_PUBLIC_BACKEND_URL="http://localhost:8001"
  exec npm run dev
) >"$LOGDIR/frontend.log" 2>&1 &
FRONT_PID=$!

echo $FRONT_PID > "$LOGDIR/frontend.pid"

echo "holy started"
echo "- frontend: http://localhost:3000"
echo "- backend:  http://localhost:8001"
