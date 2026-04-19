#!/bin/bash
# Build a standalone macOS .app bundle for Ducking using py2app.
# Python + PyTorch + Tk + all deps are embedded — no system Python needed.
# Run from the repo root: ./make-app.sh
# Installs to /Applications/Ducking.app (old version moved to Trash).
#
# Requirements:
#   - python.org Python 3.12 at /Library/Frameworks/Python.framework/
#     (brew's python works too; anaconda's Python does NOT — it's not a
#     framework build and py2app chokes on it)

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

PYTHON=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
if [ ! -x "$PYTHON" ]; then
    echo "Error: $PYTHON not found. Install python.org Python 3.12 from python.org/downloads" >&2
    exit 1
fi

# 1. Create clean build venv (avoids polluting system Python)
if [ ! -d .venv-build ]; then
    echo "Creating build venv..."
    "$PYTHON" -m venv .venv-build
    .venv-build/bin/pip install --upgrade pip
fi
.venv-build/bin/pip install --quiet py2app torch numpy scipy silero-vad soundfile pyobjc-framework-Cocoa

# 2. Build the .icns icon from icon.png
echo "Building icon..."
ICONSET=$(mktemp -d)/Ducking.iconset
mkdir -p "$ICONSET"
for size in 16 32 64 128 256 512; do
    sips -z $size $size icon.png --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    sips -z $double $double icon.png --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
sips -z 1024 1024 icon.png --out "$ICONSET/icon_512x512@2x.png" >/dev/null
iconutil -c icns "$ICONSET" -o Ducking.icns

# 3. Build the .app bundle
echo "Building .app bundle (takes several minutes)..."
rm -rf build dist
.venv-build/bin/python setup.py py2app

# 4. Patch torchaudio's stale rpath (points at the Linux CI machine that
#    built the wheel) so it can find torch's dylibs inside the bundle
TA_LIB=dist/Ducking.app/Contents/Resources/lib/python3.12/torchaudio/lib
for f in "$TA_LIB/_torchaudio.abi3.so" "$TA_LIB/libtorchaudio.abi3.so"; do
    if [ -f "$f" ]; then
        install_name_tool -add_rpath @loader_path/../../torch/lib "$f" 2>/dev/null || true
    fi
done

# 5. Strip extended attributes and deep-sign — required for macOS TCC
#    (permissions) to remember grants across launches
xattr -cr dist/Ducking.app
codesign --force --deep --sign - dist/Ducking.app

# 6. Install to /Applications (old bundle moved to Trash for rollback)
if [ -d /Applications/Ducking.app ]; then
    mv /Applications/Ducking.app "$HOME/.Trash/Ducking-$(date +%s).app"
fi
cp -R dist/Ducking.app /Applications/
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f /Applications/Ducking.app >/dev/null 2>&1 || true

echo ""
echo "Built and installed /Applications/Ducking.app"
echo "Launch from Spotlight (Cmd+Space → 'Ducking')"
