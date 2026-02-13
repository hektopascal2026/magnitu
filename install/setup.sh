#!/bin/bash
# ─────────────────────────────────────────────
#  Magnitu — One-click setup
#  Run this script to install and configure Magnitu.
# ─────────────────────────────────────────────

set -e

MAGNITU_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_FILE="$MAGNITU_DIR/magnitu_config.json"
DEFAULT_URL="https://www.hektopascal.org/seismo-staging/index.php"

clear
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Magnitu Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Step 1: Check Python ──
echo "  Checking Python..."
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo ""
    echo "  ERROR: Python 3 is required but not found."
    echo "  Install it from https://www.python.org/downloads/"
    exit 1
fi

PY_VERSION=$($PY --version 2>&1)
echo "  Found: $PY_VERSION"
echo ""

# ── Step 2: Create virtual environment ──
if [ ! -d "$MAGNITU_DIR/.venv" ]; then
    echo "  Creating virtual environment..."
    $PY -m venv "$MAGNITU_DIR/.venv"
    echo "  Done."
else
    echo "  Virtual environment exists."
fi

echo "  Installing dependencies..."
source "$MAGNITU_DIR/.venv/bin/activate"
pip install -q -r "$MAGNITU_DIR/requirements.txt"
echo "  Done."
echo ""

# ── Step 3: Configure connection ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Seismo URL
echo "  Seismo URL (press Enter for default):"
echo "  Default: $DEFAULT_URL"
read -r -p "  URL: " SEISMO_URL
if [ -z "$SEISMO_URL" ]; then
    SEISMO_URL="$DEFAULT_URL"
fi
echo ""

# API Key
echo "  Enter your Magnitu API key."
echo "  (Find it in Seismo → Settings → Magnitu section)"
read -r -p "  API Key: " API_KEY
while [ -z "$API_KEY" ]; do
    echo "  API key is required."
    read -r -p "  API Key: " API_KEY
done
echo ""

# ── Step 4: Write config ──
cat > "$CONFIG_FILE" << CONF
{
  "seismo_url": "$SEISMO_URL",
  "api_key": "$API_KEY",
  "min_labels_to_train": 20,
  "recipe_top_keywords": 200,
  "auto_train_after_n_labels": 10,
  "alert_threshold": 0.75
}
CONF

echo "  Config saved."
echo ""

# ── Step 4b: Fresh database ──
DB_FILE="$MAGNITU_DIR/magnitu.db"
if [ -f "$DB_FILE" ]; then
    echo "  Existing database found."
    read -r -p "  Reset database for fresh start? (y/N): " RESET_DB
    if [ "$RESET_DB" = "y" ] || [ "$RESET_DB" = "Y" ]; then
        rm -f "$DB_FILE" "$DB_FILE-shm" "$DB_FILE-wal"
        echo "  Database reset. Will be recreated on first run."
    else
        echo "  Keeping existing database."
    fi
fi
echo ""

# ── Step 5: Test connection ──
echo "  Testing connection to Seismo..."
source "$MAGNITU_DIR/.venv/bin/activate"
TEST_RESULT=$($PY -c "
import sys
sys.path.insert(0, '$MAGNITU_DIR')
import sync
ok, msg = sync.test_connection()
print(msg)
if not ok:
    sys.exit(1)
" 2>&1) || true

echo "  $TEST_RESULT"
echo ""

# ── Step 6: Create desktop launcher ──
DESKTOP="$HOME/Desktop"
if [ -d "$DESKTOP" ]; then
    LAUNCHER="$DESKTOP/Magnitu.command"
    cat > "$LAUNCHER" << LAUNCH
#!/bin/bash
DIR="$MAGNITU_DIR"
PORT=8000
URL="http://localhost:\$PORT"

clear
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Magnitu"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if lsof -ti:\$PORT > /dev/null 2>&1; then
    echo "  Already running at \$URL"
    open "\$URL"
    read -r -p "  Press Enter to quit and stop server... "
    lsof -ti:\$PORT | xargs kill 2>/dev/null
    exit 0
fi

cd "\$DIR" || { echo "  Error: \$DIR not found"; exit 1; }
source .venv/bin/activate 2>/dev/null || { echo "  Error: run setup.sh first"; exit 1; }

echo "  Starting on \$URL ..."
echo ""
(sleep 2 && open "\$URL") &
python -m uvicorn main:app --port \$PORT
echo ""
echo "  Magnitu stopped."
LAUNCH
    chmod +x "$LAUNCHER"
    echo "  Desktop launcher created: Magnitu.command"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  To start Magnitu:"
echo "    • Double-click Magnitu.command on your Desktop"
echo "    • Or run: cd $MAGNITU_DIR && source .venv/bin/activate && python -m uvicorn main:app --port 8000"
echo ""
echo "  Then open http://localhost:8000 in your browser."
echo ""
