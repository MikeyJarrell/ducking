# Session Log

## 2026-07-21 19:21 — Ducking mastering guide and test interpretation
SESSLOG:2026-07-21 19:21

**Project(s):** Ducking

### Decisions
- Make the root README the canonical user guide for the app's mastering theory, presets, advanced controls, acceptance gates, and test output.
- Explain gain as a uniform correction and compression as a dynamics change; retain the validated E approach of gentle stem compression, fixed final gain, and no master-bus compressor.
- Treat PASS as evidence that defined mechanical gates succeeded, not as a substitute for listening to overlaps, transitions, loud passages, and the final program.

### Open Questions
- (none)

### Follow-ups
- [x] Continue the shared-core extraction behind `ducking_core` without changing the documented behavior. Done 2026-07-21: extracted into installable `ducking-core` version `1.0.0` (see PROJECT_INDEX.md change log).
- [ ] Rerun the 30 unit and contract tests and the nine-episode corpus after any engine behavior change. (Standing instruction, not a one-time task — test suite has since grown to 35; see README.md "Automated tests and the batch report".)

### Files
- [modified] `/Users/mikey/ducking/README.md` — expanded from a short installation guide into a complete operating, mastering, troubleshooting, and validation reference.
- [created] `/Users/mikey/ducking/SESSION_LOG.md` — records the documentation closeout and next engine step.

### Context
The README now explains every app control and every visible or batch-report measurement, including thresholds, failure actions, a worked example, and the limits of automated checks. Documentation checks passed, and all 30 current tests passed on the `codex/ducking-core-contract` branch (merged into `main` and deleted 2026-07-31; the repo has a single `main` branch now).
## 2026-07-21 — Shared core package extraction

- Moved the canonical signal-processing implementation into installable `ducking-core` version `1.0.0`.
- Reduced the desktop and web applications to interface-specific behavior; both now import Ducking processing from the package.
- Added synchronized source-time edits, configurable crossfades, exact theme-marker assembly, media inspection, checksummed outputs, and the concrete `DuckingCore` contract implementation.
- Passed 35 unit tests, built the Python 3.12 wheel, and passed all nine private Backstory episodes through the package-backed harness.
- Compared the post-extraction report with the pinned baseline: loudness and true peak were identical, limiter percentages were identical, and the largest other change was less than 0.000009 dB.
