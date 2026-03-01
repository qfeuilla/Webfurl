#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$SCRIPT_DIR/.webfurl.pids"

if [ -f "$PIDFILE" ]; then
    while IFS= read -r pid; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[…] Stopping PID $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done < "$PIDFILE"
    rm -f "$PIDFILE"
    echo "[✓] All WebFurl processes stopped"
else
    echo "[~] No PID file found. Trying to find processes by name…"
    pkill -f "webfurl-server" 2>/dev/null && echo "[✓] Stopped webfurl-server" || echo "[~] webfurl-server not running"
    pkill -f "webfurl-agent" 2>/dev/null && echo "[✓] Stopped webfurl-agent" || echo "[~] webfurl-agent not running"
fi

# Stop MongoDB container
MONGO_CONTAINER="webfurl-mongo"
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${MONGO_CONTAINER}$"; then
    echo "[…] Stopping MongoDB container…"
    docker stop "$MONGO_CONTAINER"
    echo "[✓] MongoDB stopped (data persisted in Docker volume)"
fi
