#!/bin/bash
# Build a macOS .app bundle for the desktop (tkinter) version of Ducking.
# Run from the repo root: ./make-app.sh
# Installs to /Applications/Ducking.app

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
APP="/Applications/Ducking.app"
PYTHON="$(command -v python3)"

if [ -z "$PYTHON" ]; then
    echo "Error: python3 not found in PATH. Install Python 3.10+ first." >&2
    exit 1
fi

echo "Using Python: $PYTHON"
echo "Building $APP from $REPO_DIR"

# Build .icns from icon.png
ICONSET="$(mktemp -d)/Ducking.iconset"
mkdir -p "$ICONSET"
for size in 16 32 64 128 256 512; do
    sips -z $size $size "$REPO_DIR/icon.png" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    sips -z $double $double "$REPO_DIR/icon.png" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
sips -z 1024 1024 "$REPO_DIR/icon.png" --out "$ICONSET/icon_512x512@2x.png" >/dev/null

# Build the .app bundle
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/Ducking.icns"

cat > "$APP/Contents/MacOS/Ducking" <<EOF
#!/bin/bash
exec "$PYTHON" "$REPO_DIR/ducking_app.py"
EOF
chmod +x "$APP/Contents/MacOS/Ducking"

cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Ducking</string>
    <key>CFBundleDisplayName</key>
    <string>Ducking</string>
    <key>CFBundleExecutable</key>
    <string>Ducking</string>
    <key>CFBundleIconFile</key>
    <string>Ducking</string>
    <key>CFBundleIdentifier</key>
    <string>com.mikeyjarrell.ducking</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# Refresh Launch Services so Finder/Spotlight pick up the icon
touch "$APP"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP" >/dev/null 2>&1 || true

echo "Built $APP"
echo "Launch from Spotlight (Cmd+Space, type 'Ducking') or Finder."
