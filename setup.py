"""
py2app build script for Ducking.

Build a standalone .app bundle (Python + Tk + deps all included):
    .venv-build/bin/python setup.py py2app

Quick iteration (alias mode, shares deps with source — not distributable):
    .venv-build/bin/python setup.py py2app -A
"""
import sys

from setuptools import find_packages, setup

sys.setrecursionlimit(10000)

APP = ["ducking_app.py"]
DATA_FILES = []
OPTIONS = {
    "iconfile": "Ducking.icns",
    "argv_emulation": False,
    "plist": {
        "CFBundleName": "Ducking",
        "CFBundleDisplayName": "Ducking",
        "CFBundleIdentifier": "com.mikeyjarrell.ducking",
        "CFBundleVersion": "1.0",
        "CFBundleShortVersionString": "1.0",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "10.13",
    },
    "packages": ["torch", "numpy", "scipy", "silero_vad", "soundfile"],
    "includes": ["tkinter"],
    # silero-vad imports torchaudio only for optional file I/O. Ducking uses
    # scipy instead, so excluding it avoids a broken compiled extension and
    # makes the standalone app smaller.
    "excludes": ["torchaudio"],
}

PACKAGE = {
    "name": "ducking-core",
    "version": "1.0.0",
    "description": "Reusable two-microphone podcast audio engine",
    "packages": find_packages(include=("ducking_core", "ducking_core.*")),
    "python_requires": ">=3.12",
    "install_requires": [
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "silero-vad>=5.0",
        "soundfile>=0.12.0",
        "torch>=2.0.0",
    ],
}

# The same setup file still builds the standalone macOS interface when the
# explicit py2app command is present. Normal wheel and editable installs receive
# only the reusable ducking_core package.
if "py2app" in sys.argv:
    PACKAGE.update(
        {
            "app": APP,
            "data_files": DATA_FILES,
            "options": {"py2app": OPTIONS},
            "setup_requires": ["py2app"],
        }
    )

setup(**PACKAGE)
