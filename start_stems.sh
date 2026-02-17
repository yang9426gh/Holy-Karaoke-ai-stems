#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

# Ports
BPORT=8011
FPORT=3011

LSOF_BIN=""
if command -v lsof >/dev/null 2>&1; then
  LSOF_BIN="lsof"
elif [ -x /usr/sbin/lsof ]; then
  LSOF_BIN="/usr/sbin/lsof"
fi

if [ -n "$LSOF_BIN" ]; then
  if $LSOF_BIN -tiTCP:${BPORT} -sTCP:LISTEN >/dev/null 2>&1; then
    $LSOF_BIN -tiTCP:${BPORT} -sTCP:LISTEN | xargs -r kill || true
  fi
  if $LSOF_BIN -tiTCP:${FPORT} -sTCP:LISTEN >/dev/null 2>&1; then
    $LSOF_BIN -tiTCP:${FPORT} -sTCP:LISTEN | xargs -r kill || true
  fi
else
  # fallback: best-effort kill by matching command line
  pkill -f "--port ${BPORT}" >/dev/null 2>&1 || true
  pkill -f "PORT=${FPORT}" >/dev/null 2>&1 || true
fi

(
  cd "$BACKEND"
  source venv/bin/activate
  export HOLY_DATA_DIR="$BACKEND/data"
  export HOLY_DEMUCS_MODEL="htdemucs_6s"
  export HOLY_YTDLP_COOKIES_FROM_BROWSER="${HOLY_YTDLP_COOKIES_FROM_BROWSER:-chrome}"
  export HOLY_YTDLP_JS_RUNTIMES="${HOLY_YTDLP_JS_RUNTIMES:-node}"
  export HOLY_YTDLP_REMOTE_COMPONENTS="${HOLY_YTDLP_REMOTE_COMPONENTS:-ejs:github}"
  exec python -m uvicorn main:app --host 127.0.0.1 --port ${BPORT}
) >"$LOGDIR/backend.log" 2>&1 &
echo $! > "$LOGDIR/backend.pid"

(
  cd "$FRONTEND"
  export PORT=${FPORT}
  export NEXT_PUBLIC_BACKEND_URL="http://localhost:${BPORT}"
  exec npm run dev
) >"$LOGDIR/frontend.log" 2>&1 &
echo $! > "$LOGDIR/frontend.pid"

echo "stems started"
echo "- frontend: http://localhost:${FPORT}"
echo "- backend:  http://localhost:${BPORT}"
