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
- [ ] Continue the shared-core extraction behind `ducking_core` without changing the documented behavior.
- [ ] Rerun the 30 unit and contract tests and the nine-episode corpus after any engine behavior change.

### Files
- [modified] `/Users/mikey/ducking/README.md` — expanded from a short installation guide into a complete operating, mastering, troubleshooting, and validation reference.
- [created] `/Users/mikey/ducking/SESSION_LOG.md` — records the documentation closeout and next engine step.

### Context
The README now explains every app control and every visible or batch-report measurement, including thresholds, failure actions, a worked example, and the limits of automated checks. Documentation checks passed, and all 30 current tests passed on `codex/ducking-core-contract`.
