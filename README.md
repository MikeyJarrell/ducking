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

Clean up two-microphone podcast recordings automatically.

When you record a podcast with two people in the same room, each microphone picks up both voices. The person sitting farther away sounds echoey or distant on the other mic. This app fixes that.

## What it does

You give it two audio files — one from each microphone. The app figures out who's talking at each moment and turns down the mic that isn't being used. This removes the echo and background bleed, so each speaker sounds clean and close.

The default **Natural cleanup** preset changes only the mic-bleed level. It
does not boost, compress, or limit either voice.

The **Podcast-ready** preset removes low-frequency rumble, matches the two
speakers' levels, applies gentle 2:1 compression to each voice, and creates a
combined mono master. The final mix receives one fixed gain adjustment and a
true-peak safety limiter; it does not use master-bus compression. The master
targets -18 LUFS integrated loudness and a -1 dBFS true-peak ceiling, following
[Audio Engineering Society guidance for speech](https://aes.org/wp-content/uploads/2024/01/20210924_TD1008_v3.13.pdf).

Every automatic stage has an acceptance check. Ducking must measurably reduce
the non-speaker mic while leaving the active mic unchanged; the unmastered mix
must retain both speakers; and the master must pass its loudness, true-peak,
limiter-activity, duration, speaker-balance, and speaker-presence checks. The
final limiter may reduce more than 1 dB on no more than 1% of the program, and
the two speakers must remain within 3 dB during isolated speech. Uncertain speaker
detection gets one conservative retry. Natural cleanup then bypasses unsafe
ducking, while Podcast-ready stops without creating a misleading mastered file.

## Which settings should I use?

Start with **Natural cleanup (recommended)**. It reduces the unused mic by 12
dB and uses 150 ms transitions, which keeps words intact and leaves the source
recording's tone and dynamics alone.

Use **Podcast-ready** when you want Ducking to handle the routine audio work.
It saves the two cleaned stems and a separate mastered mix. Natural cleanup is
for workflows where another tool will handle level matching and mastering.

The advanced controls rarely need adjustment:

- **Duck level:** Use -9 dB for more room sound or -15 dB for stronger bleed removal.

- **Dominance:** Raise this if the app switches speakers too eagerly. Lower it if one speaker is rarely recognized.

- **Fade:** Use 100 to 200 ms for speech. Shorter fades can sound choppy.

- **Voice activity detection threshold:** Leave this at 0.5 unless the room is unusually noisy.

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

### Mac: install as a real app

If you're on a Mac and want to launch Ducking from Spotlight or the Applications folder like any other app, run:

```
chmod +x make-app.sh
./make-app.sh
```

This builds a standalone `Ducking.app` using py2app and installs it to `/Applications`. The bundle embeds Python, PyTorch, and all dependencies — no system Python needed at runtime — so it's ~600 MB but self-contained.

Requires **python.org Python 3.12** (from [python.org/downloads](https://www.python.org/downloads/)) — anaconda's Python is not a framework build and won't work with py2app.

## Supported file formats

- **WAV** (recommended — lossless, best quality)
- **MP3**, **FLAC**, **OGG** (the web version accepts these too)

Output is always WAV.

## Project files

- [ducking_app.py](ducking_app.py) contains the desktop application and canonical audio engine.
- [streamlit_app.py](streamlit_app.py) contains the web application and matching audio engine.
- [batch_validate.py](batch_validate.py) runs the private episode regression corpus.
- [test_ducking_app.py](test_ducking_app.py) contains the unit tests.
- [setup.py](setup.py) and [make-app.sh](make-app.sh) build the macOS application.
- [PROJECT_INDEX.md](PROJECT_INDEX.md) records project status.
- [AGENTS.md](AGENTS.md) contains project instructions; [CLAUDE.md](CLAUDE.md) imports them for Claude Code.

## How it works (for the curious)

The app uses a machine-learning model called [Silero VAD](https://github.com/snakers4/silero-vad) to detect when someone is speaking. It then compares the volume levels between the two mics to figure out *which* speaker is talking. The mic that isn't being used gets turned down (not all the way to silence — a little room noise is kept so it sounds natural). Smooth volume fades prevent any clicking or popping at the transitions.
