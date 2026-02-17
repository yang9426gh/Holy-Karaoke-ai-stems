#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="$ROOT/dist"
STAGE="$OUTDIR/HolyKaraoke"
ZIP="$OUTDIR/HolyKaraoke-mac.zip"

rm -rf "$STAGE" "$ZIP"
mkdir -p "$STAGE"

# Copy project without large caches
rsync -a \
  --exclude 'backend/data/***' \
  --exclude 'backend/venv/***' \
  --exclude 'backend/__pycache__/***' \
  --exclude 'frontend/node_modules/***' \
  --exclude 'frontend/.next/***' \
  --exclude 'logs/***' \
  --exclude '.DS_Store' \
  "$ROOT/" "$STAGE/"

# Ensure empty data dir exists
mkdir -p "$STAGE/backend/data"

# Create installer launcher
cat > "$STAGE/Install Holy.command" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "Holy Karaoke install starting…"

echo "1) Checking Homebrew…"
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install it first: https://brew.sh/" >&2
  read -n 1 -s -r -p "Press any key to close…"
  exit 1
fi

echo "2) Installing system deps (ffmpeg, rubberband)…"
brew install ffmpeg rubberband >/dev/null

echo "3) Setting up Python venv…"
cd "$ROOT/backend"
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip wheel >/dev/null

echo "4) Installing Python deps… (this can take a while)"
pip install -r requirements.txt

echo "5) Installing frontend deps…"
cd "$ROOT/frontend"
if ! command -v npm >/dev/null 2>&1; then
  echo "npm not found. Install Node.js (https://nodejs.org/) then re-run." >&2
  read -n 1 -s -r -p "Press any key to close…"
  exit 1
fi
npm install

echo "Done. You can now run: Start Holy.command"
read -n 1 -s -r -p "Press any key to close…"
SH
chmod +x "$STAGE/Install Holy.command"

# Start/Stop launchers (double-click friendly)
cat > "$STAGE/Start Holy.command" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
./start_holy.sh
read -n 1 -s -r -p "Press any key to close…"
SH
chmod +x "$STAGE/Start Holy.command"

cat > "$STAGE/Stop Holy.command" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
./stop_holy.sh
read -n 1 -s -r -p "Press any key to close…"
SH
chmod +x "$STAGE/Stop Holy.command"

# Link file
cat > "$STAGE/Holy Karaoke.webloc" <<'WEB'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>URL</key>
  <string>http://localhost:3000</string>
</dict>
</plist>
WEB

# Zip it up
mkdir -p "$OUTDIR"
cd "$OUTDIR"
/usr/bin/zip -r "$(basename "$ZIP")" "$(basename "$STAGE")" >/dev/null

echo "Created: $ZIP"
