#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$ROOT/logs"

kill_pidfile() {
  local f="$1"
  if [ -f "$f" ]; then
    local pid
    pid="$(cat "$f" || true)"
    if [ -n "${pid}" ]; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
    rm -f "$f"
  fi
}

kill_pidfile "$LOGDIR/frontend.pid"
kill_pidfile "$LOGDIR/backend.pid"

echo "stems stopped"
