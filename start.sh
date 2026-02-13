#!/bin/bash
# ─────────────────────────────────────────────
#  Magnitu — Start server
#  Usage:  ~/magnitu/start.sh
# ─────────────────────────────────────────────

DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8000
HOST="127.0.0.1"
URL="http://$HOST:$PORT"

clear 2>/dev/null || true
echo ""
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Magnitu"
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if already running
if lsof -ti:$PORT > /dev/null 2>&1; then
    echo "  Already running at $URL"
    echo "  Opening browser..."
    open "$URL" 2>/dev/null || echo "  Open $URL in your browser."
    echo ""
    echo "  To stop: close this window or press Ctrl+C"
    read -r -p "  Press Enter to stop the server... "
    lsof -ti:$PORT | xargs kill 2>/dev/null
    exit 0
fi

# Check setup
cd "$DIR" || exit 1
if [ ! -f .venv/bin/python ]; then
    echo "  Not set up yet. Running installer..."
    echo ""
    /bin/bash "$DIR/install/bootstrap.sh"
    exit $?
fi

if [ ! -f magnitu_config.json ]; then
    echo "  No config found. Running installer..."
    echo ""
    /bin/bash "$DIR/install/bootstrap.sh"
    exit $?
fi

echo "  Starting on $URL ..."
echo "  Press Ctrl+C to stop."
echo ""

# Wait for server to be ready, then open browser (background)
(
    for i in $(seq 1 30); do
        sleep 1
        if curl -s -o /dev/null -w '' "http://$HOST:$PORT" 2>/dev/null; then
            open "http://$HOST:$PORT" 2>/dev/null
            break
        fi
    done
) &

# Run server (foreground — Ctrl+C stops it)
"$DIR/.venv/bin/python" -m uvicorn main:app --host "$HOST" --port $PORT

echo ""
echo "  Magnitu stopped."
