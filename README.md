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

**Clean up your two-mic podcast recordings automatically.**

When you record a podcast with two people in the same room, each microphone picks up both voices. The person sitting farther away sounds echoey or distant on the other mic. This app fixes that.

## What it does

You give it two audio files — one from each microphone. The app figures out who's talking at each moment and turns down the mic that isn't being used. This removes the echo and background bleed, so each speaker sounds clean and close.

It also:

- **Evens out the volume** so quiet words and loud words are closer to the same level
- **Sets the overall loudness** to the standard that podcast apps like Spotify and Apple Podcasts expect
- **Prevents distortion** by making sure nothing is too loud

## Try it online

Use the web version — no installation needed:

**[mikeyjarrell.com/ducking](https://mikeyjarrell.com/ducking)**

Upload your two files, click Process, and download the results.

## Run it on your own computer

Running locally is faster (especially for long recordings) and has no file-size limits. Here's how:

### 1. Install Python

If you don't already have Python, download it from [python.org](https://www.python.org/downloads/). Version 3.10 or newer works. During installation on Windows, check the box that says **"Add Python to PATH"**.

### 2. Download this app

Click the green **Code** button at the top of this page, then **Download ZIP**. Unzip it somewhere you'll remember (like your Desktop).

Or if you're comfortable with the terminal:
```
git clone https://github.com/MikeyJarrell/ducking.git
cd ducking
```

### 3. Install the dependencies

Open a terminal (Mac: Terminal app, Windows: Command Prompt or PowerShell) and navigate to the folder you just downloaded. Then run:

```
pip install streamlit torch numpy scipy silero-vad soundfile
```

This downloads the libraries the app needs. It may take a few minutes the first time (PyTorch is a large download).

### 4. Run the app

From the same terminal, run:

```
streamlit run streamlit_app.py
```

A browser window will open with the app. Upload your two mic files, click **Process**, and download the cleaned-up versions.

### Alternative: desktop version

There's also a simpler desktop version with a basic window interface (no browser needed):

```
pip install torch numpy scipy silero-vad
python ducking_app.py
```

## Supported file formats

- **WAV** (recommended — lossless, best quality)
- **MP3**, **FLAC**, **OGG** (the web version accepts these too)

Output is always WAV.

## How it works (for the curious)

The app uses a machine-learning model called [Silero VAD](https://github.com/snakers4/silero-vad) to detect when someone is speaking. It then compares the volume levels between the two mics to figure out *which* speaker is talking. The mic that isn't being used gets turned down (not all the way to silence — a little room noise is kept so it sounds natural). Smooth volume fades prevent any clicking or popping at the transitions.
