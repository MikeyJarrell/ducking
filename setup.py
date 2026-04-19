"""
py2app build script for Ducking.

Build a standalone .app bundle (Python + Tk + deps all included):
    .venv-build/bin/python setup.py py2app

Quick iteration (alias mode, shares deps with source — not distributable):
    .venv-build/bin/python setup.py py2app -A
"""
import sys
sys.setrecursionlimit(10000)

from setuptools import setup

APP = ['ducking_app.py']
DATA_FILES = []
OPTIONS = {
    'iconfile': 'Ducking.icns',
    'argv_emulation': False,
    'plist': {
        'CFBundleName': 'Ducking',
        'CFBundleDisplayName': 'Ducking',
        'CFBundleIdentifier': 'com.mikeyjarrell.ducking',
        'CFBundleVersion': '1.0',
        'CFBundleShortVersionString': '1.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.13',
    },
    'packages': ['torch', 'numpy', 'scipy', 'silero_vad', 'soundfile'],
    'includes': ['tkinter'],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
