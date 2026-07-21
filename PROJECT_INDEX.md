# PROJECT_INDEX — Ducking

last-reviewed: 2026-07-21

| Workstream | Status | Notes |
|---|---|---|
| Desktop app | working | [ducking_app.py](ducking_app.py); packaged via [make-app.sh](make-app.sh) / [setup.py](setup.py) (py2app) |
| Web app | live | [streamlit_app.py](streamlit_app.py), deployed to Hugging Face Spaces and mirrored at mikeyjarrell.com/ducking |
| Tests | passing at HEAD | [test_ducking_app.py](test_ducking_app.py) — VAD loading, ducking envelope, processing safety |
| Acceptance harness | in place | [batch_validate.py](batch_validate.py) |
| Backstory dependency | consumed | Backstory pins Ducking commit `4eb9aa4` as its tested audio-engine baseline |

## Near-term

- None tracked as of 2026-07-21.

## Change log

- 2026-07-21 — keystones (AGENTS.md, PROJECT_INDEX.md, CLAUDE.md importer) added; none existed previously.
- 2026-07-20 — `4eb9aa4` fix: validate podcast-ready mastering (latest commit at keystone creation).
