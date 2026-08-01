# AGENTS.md — Ducking

**What this is:** Podcast microphone-bleed cleanup app. Takes two synchronized mic recordings from a two-person podcast, detects who's speaking with Silero VAD, and ducks the inactive mic to remove echo/bleed. Personal repo, not research. Shared audio engine source for [Backstory](~/backstory) — Backstory's AGENTS.md pins the exact Ducking commit it depends on; when the mastering pipeline changes here, check whether that pin needs bumping.

## Start here

- [README.md](README.md) — what it does, presets, install/run instructions for end users
- `ducking_core/` — canonical, versioned, application-independent audio package
- `ducking_app.py` — desktop Tkinter interface
- `streamlit_app.py` — web app deployed to Hugging Face Spaces and mirrored at mikeyjarrell.com/ducking
- `batch_validate.py` — acceptance-check harness for the ducking/mastering pipeline
- `test_ducking_app.py` — unit tests (`VadLoadingTests`, `DuckingEnvelopeTests`, `ProcessingSafetyTests`)
- [PROJECT_INDEX.md](PROJECT_INDEX.md) — dashboard

## Rules that always apply

- GitHub is canonical; Hugging Face Spaces hosts the deployed web app. **The Space does NOT auto-sync from GitHub** — it is a separate `hf` git remote that only advances when someone pushes to it. Verified 2026-07-31: `hf/main` had sat at `733c263` (2026-03-14) while `origin/main` moved four-plus months ahead, so the live Space was running old code. Check `git ls-remote hf` before assuming the deployed app matches this repo. Deploying is outward-facing — ask Mikey first.
- **Deploy recipe (`scripts/deploy-hf.sh`).** A plain `git push hf main` is rejected: Hugging Face refuses binary files stored the ordinary way, and it scans the whole history of what you push, so `icon.png` in an old commit blocks it no matter what the current files look like. The script sidesteps this by building a **single fresh snapshot commit** in a temp directory with `icon.png` tracked by Git LFS (which is what HF wants), then force-pushing that to the Space. The Space's git log is therefore just deploy snapshots — that is fine, the real history is on GitHub, and this repo's own history is never touched or rewritten.
- Every automated processing stage needs an acceptance check (loudness, true-peak, limiter activity, speaker presence) — see README "How it works" and `batch_validate.py`. Don't ship a stage that can silently produce a bad file.
- Desktop app (`ducking_app.py`) and web app (`streamlit_app.py`) import all ducking and mastering behavior from `ducking_core`; do not copy processing functions back into either interface.
- macOS `.app` packaging (`make-app.sh`, `setup.py`, py2app) requires python.org framework Python 3.12, not Anaconda's build. `.venv-build/` and `dist/`/`build/` are gitignored local build artifacts, never committed.
- Run all three `test_ducking_*.py` suites before committing engine changes.
- Template deviation, deliberate: no `data/`, `paper/`, `deep-research/` folders — this is an app repo, not a research project.

## Current focus

Status, pipeline details, and test results live in [PROJECT_INDEX.md](PROJECT_INDEX.md) — read that, not this section, for the current state.
