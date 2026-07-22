"""Media inspection, checksums, theme mixing, and file encoding."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .contracts import AudioMetadata, ThemeAssets
from .engine import apply_gain_db, load_wav, save_wav


def sha256_file(path: Path) -> str:
    """Return a file checksum without loading the entire asset into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_wav(path: Path) -> AudioMetadata:
    """Decode one supported WAV and return stable media facts."""
    sample_rate, audio, original_dtype = load_wav(path)
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    return AudioMetadata(
        sample_rate_hz=int(sample_rate),
        channels=int(channels),
        sample_count=len(audio),
        duration_ms=round(len(audio) * 1000 / sample_rate),
        sample_format=str(original_dtype),
        finite_samples=bool(np.all(np.isfinite(audio))),
    )


def decode_theme(path: Path, target_sample_rate: int) -> np.ndarray:
    """Decode a theme asset, resample it, and return stereo float audio."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != target_sample_rate:
        divisor = int(np.gcd(sample_rate, target_sample_rate))
        audio = resample_poly(
            audio,
            target_sample_rate // divisor,
            sample_rate // divisor,
            axis=0,
        ).astype(np.float32)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    return audio.astype(np.float32, copy=False)


def assemble_themed_program(
    speech_master: np.ndarray,
    sample_rate: int,
    theme: ThemeAssets,
    intro_speech_anchor_ms: int,
    outro_final_word_anchor_ms: int,
) -> np.ndarray:
    """Place stereo theme assets at their exact millisecond markers."""
    if speech_master.ndim == 1:
        speech = np.repeat(speech_master[:, np.newaxis], 2, axis=1)
    elif speech_master.shape[1] == 1:
        speech = np.repeat(speech_master, 2, axis=1)
    else:
        speech = speech_master[:, :2]

    speech_start_ms = theme.intro_speech_marker_ms - intro_speech_anchor_ms
    if speech_start_ms < 0:
        raise ValueError("The intro speech anchor would place speech before time zero.")
    final_word_program_ms = speech_start_ms + outro_final_word_anchor_ms
    outro_start_ms = final_word_program_ms - theme.outro_final_word_marker_ms
    if outro_start_ms < 0:
        raise ValueError("The outro marker would place the outro before time zero.")

    intro = apply_gain_db(
        decode_theme(theme.intro_path, sample_rate), theme.music_gain_db
    )
    outro = apply_gain_db(
        decode_theme(theme.outro_path, sample_rate), theme.music_gain_db
    )
    speech_start = round(speech_start_ms * sample_rate / 1000)
    outro_start = round(outro_start_ms * sample_rate / 1000)
    length = max(len(intro), speech_start + len(speech), outro_start + len(outro))
    program = np.zeros((length, 2), dtype=np.float32)
    program[: len(intro)] += intro
    program[speech_start : speech_start + len(speech)] += speech
    program[outro_start : outro_start + len(outro)] += outro
    return program


def write_derived_wav(
    path: Path,
    sample_rate: int,
    audio: np.ndarray,
    original_dtype: np.dtype,
) -> None:
    """Encode a derived WAV using Ducking's tested conversion behavior."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_wav(path, sample_rate, audio, original_dtype)
