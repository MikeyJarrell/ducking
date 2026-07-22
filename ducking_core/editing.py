"""Non-destructive synchronized edit-plan helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .contracts import EditInterval


def normalize_edit_plan(
    duration_ms: int,
    trim: EditInterval | None,
    cuts: Sequence[EditInterval],
) -> tuple[EditInterval, tuple[EditInterval, ...]]:
    """Return a bounded trim and validated cuts in source time."""
    if duration_ms <= 0:
        raise ValueError("The source duration must be positive.")

    bounded_trim = trim or EditInterval(0, duration_ms)
    if bounded_trim.end_ms > duration_ms:
        raise ValueError("The trim interval extends beyond the source duration.")

    ordered = tuple(sorted(cuts, key=lambda interval: interval.start_ms))
    if tuple(cuts) != ordered:
        raise ValueError("Cuts must be ordered by source start time.")
    for cut in ordered:
        if cut.start_ms < bounded_trim.start_ms or cut.end_ms > bounded_trim.end_ms:
            raise ValueError("Cuts must fall within the trim interval.")
    for previous, current in zip(ordered, ordered[1:]):
        if current.start_ms <= previous.end_ms:
            raise ValueError("Cuts cannot overlap or be adjacent.")
    return bounded_trim, ordered


def kept_sample_intervals(
    sample_rate: int,
    sample_count: int,
    trim: EditInterval | None,
    cuts: Sequence[EditInterval],
) -> tuple[tuple[int, int], ...]:
    """Translate one source-time edit plan into sample-accurate kept regions."""
    duration_ms = round(sample_count * 1000 / sample_rate)
    bounded_trim, ordered = normalize_edit_plan(duration_ms, trim, cuts)

    def sample_at(milliseconds: int) -> int:
        return min(sample_count, round(milliseconds * sample_rate / 1000))

    regions: list[tuple[int, int]] = []
    cursor = bounded_trim.start_ms
    for cut in ordered:
        if cursor < cut.start_ms:
            regions.append((sample_at(cursor), sample_at(cut.start_ms)))
        cursor = cut.end_ms
    if cursor < bounded_trim.end_ms:
        regions.append((sample_at(cursor), sample_at(bounded_trim.end_ms)))
    return tuple((start, end) for start, end in regions if end > start)


def _join_with_crossfades(
    audio: np.ndarray,
    intervals: Sequence[tuple[int, int]],
    crossfade_samples: int,
) -> np.ndarray:
    """Join kept regions with short linear crossfades at cut boundaries."""
    if not intervals:
        raise ValueError("The edit plan removes the entire recording.")
    result = audio[slice(*intervals[0])].copy()
    for start, end in intervals[1:]:
        next_segment = audio[start:end]
        overlap = min(crossfade_samples, len(result), len(next_segment))
        if overlap <= 0:
            result = np.concatenate((result, next_segment), axis=0)
            continue

        fade_out = np.linspace(1.0, 0.0, overlap, endpoint=False, dtype=np.float32)
        fade_in = 1.0 - fade_out
        if result.ndim == 2:
            fade_out = fade_out[:, np.newaxis]
            fade_in = fade_in[:, np.newaxis]
        joined = result[-overlap:] * fade_out + next_segment[:overlap] * fade_in
        result = np.concatenate(
            (result[:-overlap], joined, next_segment[overlap:]), axis=0
        )
    return result.astype(np.float32, copy=False)


def apply_identical_edits(
    host: np.ndarray,
    guest: np.ndarray,
    sample_rate: int,
    trim: EditInterval | None,
    cuts: Sequence[EditInterval],
    *,
    crossfade_ms: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one edit plan and the same joins to both synchronized tracks."""
    if len(host) != len(guest):
        raise ValueError("Synchronized tracks must have identical sample counts.")
    if crossfade_ms < 0:
        raise ValueError("The edit crossfade cannot be negative.")
    intervals = kept_sample_intervals(sample_rate, len(host), trim, cuts)
    crossfade_samples = round(crossfade_ms * sample_rate / 1000)
    edited_host = _join_with_crossfades(host, intervals, crossfade_samples)
    edited_guest = _join_with_crossfades(guest, intervals, crossfade_samples)
    if len(edited_host) != len(edited_guest):
        raise RuntimeError("Identical edits produced mismatched track lengths.")
    return edited_host, edited_guest
