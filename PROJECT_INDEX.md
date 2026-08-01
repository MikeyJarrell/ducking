# PROJECT_INDEX — Ducking

last-reviewed: 2026-07-31

| Workstream | Status | Notes |
|---|---|---|
| Desktop app | complete | [ducking_app.py](ducking_app.py); packaged via [make-app.sh](make-app.sh) / [setup.py](setup.py) (py2app) |
| Web app | complete | [streamlit_app.py](streamlit_app.py), live at mikeyjarrell.com/ducking → https://mikeyjarrell-ducking.hf.space/. Deploy with `scripts/deploy-hf.sh` — there is no auto-deploy, by choice (2026-08-01) |
| Tests | passing | 35 tests across signal processing, contracts, synchronized edits, theme markers, media inspection, and end-to-end rendering |
| Acceptance harness | passing | [batch_validate.py](batch_validate.py) — nine private episodes plus independent FFmpeg measurements |
| Shared core package | complete | Installable `ducking-core` version `1.0.0`; both interfaces and the acceptance harness import the package |
| Backstory dependency | ready | Backstory Tasks 1 and 2 are complete; the package preserves the pinned nine-episode measurements |

## Near-term

- Consume tagged `ducking-core` version `1.0.0` when Backstory scaffolds its render worker in Task 3.

## Change log

- 2026-07-21 — Extracted the canonical engine into installable `ducking-core` version `1.0.0`; 35 tests and all nine private regression episodes pass.
- 2026-07-21 — Added core contract version `1.0.0` for Backstory Task 1 without changing signal-processing behavior.
- 2026-07-21 — Podcast-ready mastering finalized: 2:1 stem compression, fixed −19 LUFS stem staging, no master-bus compression, fixed-gain −18 LUFS master, 1% final-limiter gate, and 3 dB speaker-balance gate.
- 2026-07-21 — keystones (AGENTS.md, PROJECT_INDEX.md, CLAUDE.md importer) added; none existed previously.
- 2026-07-20 — `4eb9aa4` fix: validate podcast-ready mastering (latest commit at keystone creation).
