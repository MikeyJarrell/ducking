---
title: Podcast Mic Ducking
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.55.0"
app_file: streamlit_app.py
pinned: false
---

# Podcast Mic Ducking

Upload two podcast mic tracks and the app will automatically duck each mic when the other speaker is talking, then apply compression and loudness normalization.

Uses [Silero VAD](https://github.com/snakers4/silero-vad) for voice activity detection and cross-track RMS comparison to determine who is speaking.

## Features

- **Cross-track ducking** — compares levels between mics to determine the active speaker
- **Smooth fades** — cosine transitions prevent clicks at gate open/close
- **Compression** — evens out volume differences
- **LUFS normalization** — targets -16 LUFS (podcast standard)
- **Peak limiting** — prevents clipping

## Run locally

```bash
pip install streamlit torch numpy scipy silero-vad soundfile
streamlit run streamlit_app.py
```

Or use the desktop version (tkinter GUI):

```bash
python ducking_app.py
```
