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

- GitHub is canonical; Hugging Face Spaces hosts the deployed web app (auto-synced from GitHub, not edited directly).
- Every automated processing stage needs an acceptance check (loudness, true-peak, limiter activity, speaker presence) — see README "How it works" and `batch_validate.py`. Don't ship a stage that can silently produce a bad file.
- Desktop app (`ducking_app.py`) and web app (`streamlit_app.py`) import all ducking and mastering behavior from `ducking_core`; do not copy processing functions back into either interface.
- macOS `.app` packaging (`make-app.sh`, `setup.py`, py2app) requires python.org framework Python 3.12, not Anaconda's build. `.venv-build/` and `dist/`/`build/` are gitignored local build artifacts, never committed.
- Run all three `test_ducking_*.py` suites before committing engine changes.
- Template deviation, deliberate: no `data/`, `paper/`, `deep-research/` folders — this is an app repo, not a research project.

## Current focus

Ducking is feature-complete for its current scope. The Podcast-ready preset uses the listening-selected E pipeline: 2:1 stem compression, fixed −19 LUFS stem staging, no master-bus compression, one fixed final gain change toward −18 LUFS, and a true-peak safety limiter. The canonical implementation now lives in installable package `ducking-core` version `1.0.0`. Both interfaces and the acceptance harness consume that package. The 35-test suite and all nine Backstory regression episodes pass, with numerical changes far below the pinned Task 2 tolerances.
