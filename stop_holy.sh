#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$ROOT/logs"

kill_pid_file () {
  local f="$1"
  if [[ -f "$f" ]]; then
    local pid
    pid="$(cat "$f" || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" || true
    fi
    rm -f "$f" || true
  fi
}

kill_pid_file "$LOGDIR/frontend.pid"
kill_pid_file "$LOGDIR/backend.pid"

# Also ensure ports are freed (best-effort)
if command -v lsof >/dev/null 2>&1; then
  if lsof -tiTCP:3000 -sTCP:LISTEN >/dev/null 2>&1; then
    lsof -tiTCP:3000 -sTCP:LISTEN | xargs -r kill || true
  fi
  if lsof -tiTCP:8001 -sTCP:LISTEN >/dev/null 2>&1; then
    lsof -tiTCP:8001 -sTCP:LISTEN | xargs -r kill || true
  fi
fi

echo "holy stopped"
