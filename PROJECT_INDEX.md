# PROJECT_INDEX — Ducking

last-reviewed: 2026-07-21

| Workstream | Status | Notes |
|---|---|---|
| Desktop app | complete | [ducking_app.py](ducking_app.py); packaged via [make-app.sh](make-app.sh) / [setup.py](setup.py) (py2app) |
| Web app | complete | [streamlit_app.py](streamlit_app.py), deployed to Hugging Face Spaces and mirrored at mikeyjarrell.com/ducking |
| Tests | passing | [test_ducking_app.py](test_ducking_app.py) — 19 tests covering VAD loading, ducking, mastering, and processing safety |
| Acceptance harness | passing | [batch_validate.py](batch_validate.py) — nine private episodes plus independent FFmpeg measurements |
| Shared core contract | complete | `ducking_core.contracts` version `1.0.0`; 11 contract tests passing |
| Backstory dependency | extracting | Backstory Task 1 is complete; Task 2 will move the validated engine behind the shared contract |

## Near-term

- Move the canonical processing functions behind `ducking_core` without changing the desktop or web behavior or the pinned Backstory regression measurements.

## Change log

- 2026-07-21 — Added core contract version `1.0.0` for Backstory Task 1 without changing signal-processing behavior.
- 2026-07-21 — Podcast-ready mastering finalized: 2:1 stem compression, fixed −19 LUFS stem staging, no master-bus compression, fixed-gain −18 LUFS master, 1% final-limiter gate, and 3 dB speaker-balance gate.
- 2026-07-21 — keystones (AGENTS.md, PROJECT_INDEX.md, CLAUDE.md importer) added; none existed previously.
- 2026-07-20 — `4eb9aa4` fix: validate podcast-ready mastering (latest commit at keystone creation).
