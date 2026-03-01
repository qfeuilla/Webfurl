#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "[✓] Loaded .env"
else
    echo "[!] No .env file found. Copy .env.example → .env and fill in your keys."
    exit 1
fi

# Check required env vars
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "[!] OPENROUTER_API_KEY is not set in .env"
    exit 1
fi

# Auto-detect Chrome/Chromium if CHROME_PATH not set
if [ -z "${CHROME_PATH:-}" ]; then
    for candidate in \
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
        /snap/chromium/current/usr/lib/chromium-browser/chrome \
        /usr/bin/google-chrome-stable \
        /usr/bin/google-chrome \
        /usr/bin/chromium-browser \
        /usr/bin/chromium \
        /snap/bin/chromium; do
        if [ -x "$candidate" ]; then
            export CHROME_PATH="$candidate"
            echo "[✓] Auto-detected Chrome at $CHROME_PATH"
            break
        fi
    done
    if [ -z "${CHROME_PATH:-}" ]; then
        echo "[!] No Chrome/Chromium found. Set CHROME_PATH in .env"
        exit 1
    fi
else
    echo "[✓] Using CHROME_PATH=$CHROME_PATH"
fi

# Build everything
echo "[…] Building workspace…"
cargo build --release 2>&1 | tail -5

# PID file for cleanup
PIDFILE="$SCRIPT_DIR/.webfurl.pids"
> "$PIDFILE"

# Start MongoDB via Docker
MONGO_CONTAINER="webfurl-mongo"
if docker ps --format '{{.Names}}' | grep -q "^${MONGO_CONTAINER}$"; then
    echo "[✓] MongoDB container already running"
else
    if docker ps -a --format '{{.Names}}' | grep -q "^${MONGO_CONTAINER}$"; then
        echo "[…] Starting existing MongoDB container…"
        docker start "$MONGO_CONTAINER"
    else
        echo "[…] Creating MongoDB container…"
        docker run -d --name "$MONGO_CONTAINER" \
            -p 27017:27017 \
            -v webfurl-mongo-data:/data/db \
            mongo:7
    fi
    # Wait for MongoDB to be ready
    echo "[…] Waiting for MongoDB…"
    for i in $(seq 1 15); do
        if docker exec "$MONGO_CONTAINER" mongosh --eval "db.runCommand({ping:1})" --quiet &>/dev/null; then
            break
        fi
        sleep 1
    done
    echo "[✓] MongoDB is ready"
fi

# Kill any previous webfurl-server on :3001
if lsof -ti:3001 &>/dev/null; then
    echo "[…] Killing previous server on :3001…"
    kill $(lsof -ti:3001) 2>/dev/null
    sleep 0.5
fi

# Start the Axum API server in the background
echo "[…] Starting webfurl-server on :3001…"
RUST_LOG="${RUST_LOG:-info,webfurl_core=debug}" \
    ./target/release/webfurl-server &
SERVER_PID=$!
echo "$SERVER_PID" >> "$PIDFILE"
echo "[✓] webfurl-server started (PID $SERVER_PID)"

# Wait a moment for server to bind
sleep 1

# Start the agent (interactive, foreground)
echo ""
echo "════════════════════════════════════════"
echo "  WebFurl Agent — interactive mode"
echo "  Server running on http://localhost:3001"
echo "  Type /url <url> to browse a page"
echo "  Type /quit to exit"
echo "════════════════════════════════════════"
echo ""

RUST_LOG="${RUST_LOG:-info,webfurl_core=debug,webfurl_agent=debug}" \
    ./target/release/webfurl-agent

# When agent exits, clean up
echo ""
echo "[…] Agent exited, cleaning up…"
source "$SCRIPT_DIR/stop.sh"
