#!/usr/bin/env python3
"""
Podcast Mic Ducking App

Takes two podcast microphone audio files (one per speaker) and:
1. Uses Silero VAD to detect speech and reduce bleed from the unused mic
2. Applies smooth fades to avoid clicks at transitions
3. Optionally applies gain, compression, limiting, and LUFS normalization

Run with: python ducking_app.py
The Silero VAD model is bundled with the installed silero-vad package.

Dependencies (all in conda base env):
  - torch, numpy, scipy, tkinter (built-in)
"""

import os
import math
import sys
import threading
import tkinter as tk
import types
from tkinter import ttk, filedialog, messagebox

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt
from scipy.ndimage import uniform_filter1d, minimum_filter1d, median_filter
import torch


# ============================================================
# CONSTANTS
# ============================================================

VAD_SAMPLE_RATE = 16000  # Silero VAD expects 16 kHz input
PODCAST_STEM_TARGET_LUFS = -19.0
PODCAST_MASTER_TARGET_LUFS = -18.0
MASTER_LIMITING_MAX_PERCENT = 1.0
MASTER_SPEAKER_BALANCE_MAX_DB = 3.0


# ============================================================
# SECTION 1: AUDIO I/O
# ============================================================


def load_wav(filepath):
    """
    Load a WAV file and normalize to float32 in the range [-1, 1].
    Returns (sample_rate, audio_array, original_dtype).
    """
    sr, data = wavfile.read(filepath)
    original_dtype = data.dtype

    # Convert integer formats to float32 normalized to [-1, 1]
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        audio = data.copy()
    elif data.dtype == np.float64:
        audio = data.astype(np.float32)
    else:
        raise ValueError(
            f"Unsupported WAV format: {data.dtype}. "
            "Please convert to 16-bit or 32-bit float."
        )

    return sr, audio, original_dtype


def save_wav(filepath, sr, audio, original_dtype):
    """Save audio back to WAV, converting to the original data type."""
    if original_dtype == np.int16:
        # Clip to prevent overflow, then convert
        data = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    elif original_dtype == np.int32:
        data = np.clip(audio * 2147483648.0, -2147483648, 2147483647).astype(np.int32)
    else:
        # Save as float32 for float formats
        data = audio.astype(np.float32)

    wavfile.write(filepath, sr, data)


def get_mono(audio):
    """Extract mono from audio (first channel if stereo)."""
    if audio.ndim == 2:
        return audio[:, 0]
    return audio


def mix_to_mono(audio):
    """Return mono audio, averaging channels when the source is stereo."""
    if audio.ndim == 2:
        return np.mean(audio, axis=1, dtype=np.float64).astype(np.float32)
    return audio


def resample_to_16k(audio_mono, orig_sr):
    """Resample mono audio to 16 kHz for VAD processing."""
    if orig_sr == VAD_SAMPLE_RATE:
        return audio_mono

    # Simplify the up/down ratio using GCD
    # e.g., 48000 -> 16000 becomes up=1, down=3
    g = math.gcd(VAD_SAMPLE_RATE, orig_sr)
    up = VAD_SAMPLE_RATE // g
    down = orig_sr // g

    return resample_poly(audio_mono, up, down).astype(np.float32)


# ============================================================
# SECTION 2: SILERO VAD
# ============================================================


def load_vad_model():
    """
    Load Silero VAD from the installed package without network access.

    Returns the model and the speech-timestamp helper function. The package
    ships the model file, so the desktop app does not depend on GitHub or the
    user's Torch Hub cache when processing audio.
    """
    # silero-vad imports torchaudio for optional file I/O helpers. Ducking uses
    # scipy for file I/O, and loading torchaudio's compiled extension inside a
    # py2app bundle causes macOS to kill the process. A minimal placeholder lets
    # us import only the model and timestamp helpers we actually need.
    sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))

    from silero_vad import load_silero_vad, get_speech_timestamps

    model = load_silero_vad()
    return model, get_speech_timestamps


def get_speech_regions(model, utils, audio_16k, threshold=0.5):
    """
    Run Silero VAD on 16 kHz mono audio.
    Returns a list of dicts: [{'start': sample_idx, 'end': sample_idx}, ...]
    where start/end are sample positions at 16 kHz.
    """
    # The second value returned by load_vad_model is the helper function.
    get_speech_timestamps = utils

    audio_tensor = torch.from_numpy(audio_16k).float()

    # Get speech regions with sensible defaults for podcast audio
    timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        threshold=threshold,
        sampling_rate=VAD_SAMPLE_RATE,
        min_speech_duration_ms=250,  # Ignore speech shorter than 250 ms
        min_silence_duration_ms=500,  # Hold gate open through brief pauses
    )

    # Reset model state so it's clean for the next track
    model.reset_states()

    return timestamps


# ============================================================
# SECTION 3: CROSS-TRACK DUCKING
# ============================================================


def _regions_to_frame_mask(speech_regions_16k, frame_count, frame_samples, sr):
    """Convert VAD regions to a compact mask with one value per audio frame."""
    mask = np.zeros(frame_count, dtype=bool)
    for region in speech_regions_16k:
        start_sample = region["start"] * sr / VAD_SAMPLE_RATE
        end_sample = region["end"] * sr / VAD_SAMPLE_RATE
        start = max(0, int(start_sample // frame_samples))
        end = min(frame_count, int(math.ceil(end_sample / frame_samples)))
        mask[start:end] = True
    return mask


def _frame_rms(audio, length, frame_samples):
    """Measure root mean square level in short frames without sample-sized work arrays."""
    frame_count = int(math.ceil(length / frame_samples))
    padded_length = frame_count * frame_samples
    if padded_length == length:
        framed = audio[:length]
    else:
        framed = np.pad(audio[:length], (0, padded_length - length))
    framed = framed.reshape(frame_count, frame_samples).astype(np.float64)
    power = np.mean(framed * framed, axis=1)

    # Average adjacent frames to make the level comparison stable over about
    # 60 ms while preserving short interjections.
    power = uniform_filter1d(power, size=3, mode="nearest")
    return np.sqrt(np.maximum(power, 1e-16))


def _estimate_ratio_center(ratio_db, active, dominance_db):
    """Estimate the midpoint between the two close-mic level-ratio clusters."""
    values = ratio_db[active & np.isfinite(ratio_db)]
    if len(values) < 40:
        return 0.0, None

    # Trim unusual transients, then fit two one-dimensional clusters. The
    # midpoint corrects for different microphone and preamp gains.
    low, high = np.percentile(values, [2, 98])
    values = values[(values >= low) & (values <= high)]
    center_low, center_high = np.percentile(values, [25, 75])
    for _ in range(20):
        split = (center_low + center_high) / 2.0
        group_low = values[values <= split]
        group_high = values[values > split]
        if len(group_low) == 0 or len(group_high) == 0:
            return 0.0, None
        new_low = float(np.median(group_low))
        new_high = float(np.median(group_high))
        if abs(new_low - center_low) < 0.01 and abs(new_high - center_high) < 0.01:
            center_low, center_high = new_low, new_high
            break
        center_low, center_high = new_low, new_high

    minimum_group = max(20, int(0.05 * len(values)))
    separation = center_high - center_low
    if (
        len(group_low) < minimum_group
        or len(group_high) < minimum_group
        or separation < max(8.0, 2 * dominance_db + 2.0)
    ):
        return 0.0, None

    return (center_low + center_high) / 2.0, (center_low, center_high)


def _smooth_envelope(gain, sr, fade_ms=75):
    """
    Smooth a gain envelope to prevent clicks at transitions.

    Uses a uniform (moving average) filter with width = fade duration.
    This turns sharp transitions (e.g., 1.0→0.1) into smooth linear ramps.
    Flat regions are unaffected since averaging identical values = same value.
    """
    fade_samples = int(fade_ms / 1000.0 * sr)
    if fade_samples < 2:
        return gain

    # uniform_filter1d smooths transitions into ramps while preserving flat regions
    smoothed = uniform_filter1d(
        gain.astype(np.float64), fade_samples, mode="nearest"
    ).astype(np.float32)

    return smoothed


def build_cross_ducking_envelopes(
    mono_a,
    mono_b,
    sr,
    speech_regions_a,
    speech_regions_b,
    fade_ms=150,
    duck_db=-12,
    dominance_db=3.0,
    return_diagnostics=False,
):
    """
    Build gain envelopes for both tracks using cross-track comparison.

    Instead of gating each mic by its own VAD (which fails when both mics
    pick up both speakers), this compares RMS levels between the two tracks
    to determine who is actually speaking.

    The level-ratio midpoint is calibrated from the recording, which accounts
    for different microphone gains. A short median filter prevents rapid
    speaker switching. Ambiguous speech and room tone leave both tracks open.

    Returns (envelope_a, envelope_b) — per-sample gain arrays with smooth fades.
    """
    length = min(len(mono_a), len(mono_b))
    if length == 0:
        raise ValueError("Audio files cannot be empty.")
    duck_gain = 10 ** (duck_db / 20.0)

    # Work in 20 ms frames. The previous sample-by-sample comparison could
    # switch rapidly around the threshold and audibly modulate a voice.
    frame_samples = max(int(round(0.020 * sr)), 1)
    frame_count = int(math.ceil(length / frame_samples))
    speech_a = _regions_to_frame_mask(speech_regions_a, frame_count, frame_samples, sr)
    speech_b = _regions_to_frame_mask(speech_regions_b, frame_count, frame_samples, sr)
    either_speech = speech_a | speech_b

    rms_a = _frame_rms(mono_a, length, frame_samples)
    rms_b = _frame_rms(mono_b, length, frame_samples)

    # Positive means A is louder. Calibrate the midpoint from the two ratio
    # clusters so a microphone gain mismatch does not make one speaker win.
    ratio_db = 20 * np.log10(rms_a / (rms_b + 1e-8))
    ratio_center, clusters = _estimate_ratio_center(
        ratio_db, either_speech, dominance_db
    )

    state = np.zeros(frame_count, dtype=np.int8)
    state[either_speech & (ratio_db > ratio_center + dominance_db)] = 1
    state[either_speech & (ratio_db < ratio_center - dominance_db)] = -1

    # Remove switches shorter than 100 ms. This prevents chopped consonants
    # and gain flutter while leaving brief interjections intact.
    state = median_filter(state, size=5, mode="nearest")
    a_primary = state == 1
    b_primary = state == -1

    gain_a_frames = np.ones(frame_count, dtype=np.float32)
    gain_b_frames = np.ones(frame_count, dtype=np.float32)
    gain_b_frames[a_primary] = duck_gain
    gain_a_frames[b_primary] = duck_gain

    gain_a = np.repeat(gain_a_frames, frame_samples)[:length]
    gain_b = np.repeat(gain_b_frames, frame_samples)[:length]
    gain_a = _smooth_envelope(gain_a, sr, fade_ms)
    gain_b = _smooth_envelope(gain_b, sr, fade_ms)

    diagnostics = {
        "ratio_center_db": float(ratio_center),
        "ratio_clusters_db": clusters,
        "a_primary_pct": float(np.mean(a_primary) * 100),
        "b_primary_pct": float(np.mean(b_primary) * 100),
        "ambiguous_pct": float(np.mean(state == 0) * 100),
    }
    if return_diagnostics:
        return gain_a, gain_b, diagnostics
    return gain_a, gain_b


def build_validated_ducking_envelopes(
    mono_a,
    mono_b,
    sr,
    speech_regions_a,
    speech_regions_b,
    fade_ms=150,
    duck_db=-12,
    dominance_db=3.0,
    require_two_speakers=False,
):
    """Build ducking envelopes, retrying once if speaker separation is unclear.

    A podcast-ready result must contain at least one second assigned to each
    close microphone and two distinct level-ratio clusters. For a safe cleanup,
    uncertain detection bypasses ducking instead of guessing and muting a voice.
    """
    duration_seconds = min(len(mono_a), len(mono_b)) / sr
    attempted_dominance = [float(dominance_db)]
    result = build_cross_ducking_envelopes(
        mono_a,
        mono_b,
        sr,
        speech_regions_a,
        speech_regions_b,
        fade_ms=fade_ms,
        duck_db=duck_db,
        dominance_db=dominance_db,
        return_diagnostics=True,
    )

    def detection_passes(diagnostics):
        a_seconds = diagnostics["a_primary_pct"] * duration_seconds / 100
        b_seconds = diagnostics["b_primary_pct"] * duration_seconds / 100
        return (
            diagnostics["ratio_clusters_db"] is not None
            and min(a_seconds, b_seconds) >= 1.0
        )

    if not detection_passes(result[2]):
        retry_dominance = max(1.5, float(dominance_db) / 2)
        if retry_dominance < dominance_db:
            attempted_dominance.append(retry_dominance)
            result = build_cross_ducking_envelopes(
                mono_a,
                mono_b,
                sr,
                speech_regions_a,
                speech_regions_b,
                fade_ms=fade_ms,
                duck_db=duck_db,
                dominance_db=retry_dominance,
                return_diagnostics=True,
            )

    envelope_a, envelope_b, diagnostics = result
    passed = detection_passes(diagnostics)
    diagnostics["detection_passed"] = passed
    diagnostics["dominance_attempts_db"] = attempted_dominance
    diagnostics["used_dominance_db"] = attempted_dominance[-1]
    diagnostics["ducking_bypassed"] = False

    if not passed:
        if require_two_speakers:
            raise ValueError(
                "Speaker detection could not confidently identify both close "
                "microphones after an automatic retry. No podcast-ready master "
                "was created; check that the two synchronized files contain "
                "different speakers."
            )
        envelope_a = np.ones(len(mono_a), dtype=np.float32)
        envelope_b = np.ones(len(mono_b), dtype=np.float32)
        diagnostics["ducking_bypassed"] = True

    return envelope_a, envelope_b, diagnostics


def apply_gain_envelope(audio, gain):
    """Multiply audio by the gain envelope. Works for mono and stereo."""
    if audio.ndim == 2:
        # Stereo: apply the same gain curve to both channels
        return audio * gain[:, np.newaxis]
    return audio * gain


# ============================================================
# SECTION 4: GAIN STAGE
# ============================================================


def apply_gain_db(audio, gain_db):
    """Apply a simple gain adjustment in dB (e.g., +3 dB boosts by ~1.41x)."""
    if gain_db == 0:
        return audio
    linear = 10 ** (gain_db / 20.0)
    return audio * linear


def apply_speech_highpass(audio, sr, cutoff_hz=65.0):
    """Remove direct current and low-frequency rumble below the speech band."""
    if sr <= cutoff_hz * 2.2:
        return audio
    coefficients = butter(2, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfilt(coefficients, audio, axis=0).astype(np.float32)


# ============================================================
# SECTION 5: COMPRESSOR
# ============================================================


def apply_compressor(
    audio, sr, threshold_db=-20, ratio=4.0, attack_ms=10, release_ms=100
):
    """
    Apply dynamic range compression to audio.

    Uses a block-based RMS level detector (vectorized, fast).
    Levels above threshold_db are reduced by the ratio (e.g., 4:1 means
    every 4 dB above threshold becomes 1 dB above threshold).
    """
    mono = get_mono(audio) if audio.ndim == 2 else audio

    # Step 1: Compute RMS level in short windows (attack-time length)
    window_samples = max(int(attack_ms / 1000.0 * sr), 64)
    squared = mono**2
    # uniform_filter1d computes a running average — gives us windowed mean square
    mean_sq = uniform_filter1d(
        squared.astype(np.float64), window_samples, mode="nearest"
    )
    rms = np.sqrt(np.maximum(mean_sq, 1e-16)).astype(np.float32)

    # Step 2: Convert RMS to dB
    level_db = 20 * np.log10(rms + 1e-8)

    # Step 3: Compute gain reduction for levels above threshold
    # over_db = how many dB above threshold the signal is
    over_db = np.maximum(level_db - threshold_db, 0)
    # gain_reduction = how much to pull it back (based on ratio)
    gain_reduction_db = over_db * (1.0 - 1.0 / ratio)

    # Step 4: Smooth the gain reduction with release-time window
    # This prevents the gain from pumping too quickly
    release_samples = max(int(release_ms / 1000.0 * sr), 64)
    gain_reduction_db = uniform_filter1d(
        gain_reduction_db.astype(np.float64), release_samples, mode="nearest"
    ).astype(np.float32)

    # Step 5: Convert dB reduction to linear gain and apply
    gain = 10 ** (-gain_reduction_db / 20.0)

    if audio.ndim == 2:
        return audio * gain[:, np.newaxis]
    return audio * gain


# ============================================================
# SECTION 6: LIMITER
# ============================================================


def apply_limiter(audio, sr, ceiling_db=-1.0, release_ms=80, return_gain=False):
    """
    Hard peak limiter — prevents any sample from exceeding the ceiling.

    Uses a look-ahead approach (minimum_filter1d) to start reducing gain
    slightly before the peak arrives, avoiding distortion.
    """
    ceiling_linear = 10 ** (ceiling_db / 20.0)

    # Find the peak level at each sample (max across channels if stereo)
    if audio.ndim == 2:
        peak = np.max(np.abs(audio), axis=1)
    else:
        peak = np.abs(audio)

    # Compute the gain needed to bring peaks down to the ceiling
    # For samples below the ceiling, gain = 1.0 (no change)
    required_gain = np.where(peak > ceiling_linear, ceiling_linear / (peak + 1e-8), 1.0)

    # Look-ahead: minimum_filter finds the lowest gain in a window ahead,
    # so we start reducing gain before the peak actually arrives
    release_samples = max(int(release_ms / 1000.0 * sr), 16)
    lookahead = max(release_samples // 2, 4)
    gain = minimum_filter1d(required_gain, lookahead, mode="nearest")

    # Smooth the gain curve to avoid sudden jumps
    smooth_len = max(lookahead // 4, 4)
    gain = uniform_filter1d(gain.astype(np.float64), smooth_len, mode="nearest").astype(
        np.float32
    )
    # Smoothing must never raise the envelope above the instantaneous gain
    # required at a peak. The old implementation could overshoot its own
    # ceiling after averaging the gain curve.
    gain = np.minimum(gain, required_gain).astype(np.float32)

    if audio.ndim == 2:
        result = audio * gain[:, np.newaxis]
    else:
        result = audio * gain
    if return_gain:
        return result, gain
    return result


# ============================================================
# SECTION 7: LUFS NORMALIZATION
# ============================================================


def k_weighting_coeffs(sr):
    """
    Compute K-weighting filter coefficients for a given sample rate.
    Based on ITU-R BS.1770-4 standard.

    K-weighting = two cascaded biquad filters:
      1. High-shelf filter (models acoustic effect of the human head)
      2. High-pass filter (RLB weighting, de-emphasizes low frequencies)

    Returns a (2, 6) array of second-order section (SOS) coefficients.
    """
    # Stage 1: Pre-filter (high shelf)
    f0 = 1681.974450955533  # Shelf center frequency (Hz)
    G = 3.999843853973347  # Shelf gain (dB)
    Q = 0.7071752369554196  # Quality factor

    K = np.tan(np.pi * f0 / sr)
    Vh = 10 ** (G / 20)
    Vb = Vh**0.4996667741545416

    a0 = 1 + K / Q + K**2
    b = np.array(
        [
            (Vh + Vb * K / Q + K**2) / a0,
            2 * (K**2 - Vh) / a0,
            (Vh - Vb * K / Q + K**2) / a0,
        ]
    )
    a = np.array(
        [
            1.0,
            2 * (K**2 - 1) / a0,
            (1 - K / Q + K**2) / a0,
        ]
    )
    sos1 = np.concatenate([b, a])

    # Stage 2: RLB weighting (high-pass at ~38 Hz)
    f0_hp = 38.13547087602444
    Q_hp = 0.5003270373238773
    K_hp = np.tan(np.pi * f0_hp / sr)

    a0_hp = 1 + K_hp / Q_hp + K_hp**2
    b_hp = np.array([1.0, -2.0, 1.0]) / a0_hp
    a_hp = np.array(
        [
            1.0,
            2 * (K_hp**2 - 1) / a0_hp,
            (1 - K_hp / Q_hp + K_hp**2) / a0_hp,
        ]
    )
    sos2 = np.concatenate([b_hp, a_hp])

    return np.array([sos1, sos2])


def measure_lufs(audio, sr):
    """
    Measure gated integrated loudness using the ITU-R BS.1770 method.

    Four-hundred-millisecond blocks use the standard -70 LUFS absolute gate
    and a relative gate 10 LU below the absolute-gated program loudness.
    """
    sos = k_weighting_coeffs(sr)
    channels = audio[:, np.newaxis] if audio.ndim == 1 else audio
    filtered = sosfilt(sos, channels, axis=0)
    sample_energy = np.sum(filtered.astype(np.float64) ** 2, axis=1)

    block_samples = max(int(round(0.400 * sr)), 1)
    hop_samples = max(int(round(0.100 * sr)), 1)
    if len(sample_energy) < block_samples:
        mean_energy = float(np.mean(sample_energy))
        return -70.0 if mean_energy < 1e-10 else -0.691 + 10 * np.log10(mean_energy)

    starts = np.arange(0, len(sample_energy) - block_samples + 1, hop_samples)
    cumulative = np.concatenate(([0.0], np.cumsum(sample_energy)))
    block_energy = (
        cumulative[starts + block_samples] - cumulative[starts]
    ) / block_samples
    block_loudness = -0.691 + 10 * np.log10(np.maximum(block_energy, 1e-20))

    absolute_gated = block_energy[block_loudness >= -70.0]
    if len(absolute_gated) == 0:
        return -70.0
    absolute_loudness = -0.691 + 10 * np.log10(np.mean(absolute_gated))
    relative_gate = absolute_loudness - 10.0
    final_blocks = block_energy[
        (block_loudness >= -70.0) & (block_loudness >= relative_gate)
    ]
    return -0.691 + 10 * np.log10(np.mean(final_blocks))


def measure_lufs_speech_only(audio, sr, envelope):
    """
    Measure LUFS only during speech regions (where envelope > 0.5).

    Standard measure_lufs() includes ducked silence, which drags the
    measurement way down and makes normalization overshoot. This version
    extracts only the speech portions before measuring.
    """
    mono = get_mono(audio) if audio.ndim == 2 else audio

    # Extract only the samples where the gate is open
    speech_mask = envelope > 0.5
    speech_audio = mono[speech_mask]

    if len(speech_audio) < 1024:
        return -70.0  # Not enough speech to measure

    return measure_lufs(speech_audio, sr)


def apply_lufs_normalization(
    audio, sr, target_lufs=-19.0, envelope=None, ceiling_db=None, peak_percentile=99.9
):
    """
    Normalize audio to a target loudness in LUFS.

    If an envelope is provided, LUFS is measured only during speech regions
    (where envelope > 0.5). This avoids the problem where ducked silence
    drags down the LUFS measurement and causes over-boosting.

    Podcast stems and the final master use separate targets so speaker
    balancing stays stable when the final delivery target changes.
    """
    if envelope is not None:
        current_lufs = measure_lufs_speech_only(audio, sr, envelope)
    else:
        current_lufs = measure_lufs(audio, sr)

    if current_lufs < -60:
        # Audio is too quiet to normalize meaningfully
        return audio

    # Compute how much gain to apply (in dB).
    gain_db = target_lufs - current_lufs

    # For the final loudness pass, keep routine peaks below the limiter ceiling.
    # The limiter can still catch isolated transients, but it no longer has to
    # reshape every strong syllable to repair an over-hot normalization pass.
    if ceiling_db is not None:
        mono = get_mono(audio) if audio.ndim == 2 else audio
        if envelope is None:
            active_audio = mono
        else:
            active_audio = mono[envelope > 0.5]
        if len(active_audio):
            peak = np.percentile(np.abs(active_audio), peak_percentile)
            peak_db = 20 * np.log10(peak + 1e-12)
            gain_db = min(gain_db, ceiling_db - peak_db)

    gain_linear = 10 ** (gain_db / 20.0)

    result = audio * gain_linear

    # Don't clip here — let the limiter handle peaks (it runs after this)
    return result


def measure_true_peak(audio, oversample=4, chunk_samples=480000):
    """Estimate true peak by oversampling manageable chunks of the signal."""
    peak = 0.0
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        oversampled = resample_poly(chunk, oversample, 1, axis=0)
        peak = max(peak, float(np.max(np.abs(oversampled))))
    return peak


def apply_true_peak_limiter(
    audio,
    sr,
    ceiling_db=-1.0,
    oversample=4,
    release_ms=15,
    chunk_seconds=30,
    reconstruction_margin_db=1.1,
):
    """Limit oversampled chunks with room for downsampling reconstruction."""
    chunk_samples = max(int(chunk_seconds * sr), 1)
    overlap_samples = max(int(0.250 * sr), 1)
    result = np.empty(len(audio), dtype=np.float32)
    limited_samples = 0
    measured_samples = 0

    for start in range(0, len(audio), chunk_samples):
        end = min(start + chunk_samples, len(audio))
        extended_start = max(0, start - overlap_samples)
        extended_end = min(len(audio), end + overlap_samples)
        extended = audio[extended_start:extended_end]
        oversampled = resample_poly(extended, oversample, 1).astype(np.float32)
        limited, gain = apply_limiter(
            oversampled,
            sr * oversample,
            ceiling_db=ceiling_db - reconstruction_margin_db,
            release_ms=release_ms,
            return_gain=True,
        )
        downsampled = resample_poly(limited, 1, oversample).astype(np.float32)

        core_start = start - extended_start
        core_length = end - start
        result[start:end] = downsampled[core_start : core_start + core_length]

        gain_start = core_start * oversample
        gain_end = gain_start + core_length * oversample
        core_gain = gain[gain_start:gain_end]
        limited_samples += int(np.sum(core_gain < 10 ** (-1.0 / 20.0)))
        measured_samples += len(core_gain)

    limited_pct = 100 * limited_samples / max(measured_samples, 1)
    return result, float(limited_pct)


def _masked_rms_db(audio, mask):
    """Measure RMS in a Boolean region, returning negative infinity if silent."""
    mono = mix_to_mono(audio)
    if mask is None or len(mask) != len(mono) or not np.any(mask):
        return float("-inf")
    rms = np.sqrt(np.mean(np.square(mono[mask], dtype=np.float64)))
    return float(20 * np.log10(rms + 1e-12))


def validate_mix_stage(track_a, track_b, envelope_a, envelope_b, sr):
    """Confirm that both owning microphones survive the unmastered mix."""
    a_mask = (envelope_a > 0.9) & (envelope_b < 0.5)
    b_mask = (envelope_b > 0.9) & (envelope_a < 0.5)
    premix = mix_to_mono(track_a) + mix_to_mono(track_b)
    minimum_samples = sr

    details = {}
    checks = {}
    for label, stem, mask in (("a", track_a, a_mask), ("b", track_b, b_mask)):
        stem_level = _masked_rms_db(stem, mask)
        mix_level = _masked_rms_db(premix, mask)
        retained_db = mix_level - stem_level
        details[f"speaker_{label}_seconds"] = float(np.sum(mask) / sr)
        details[f"speaker_{label}_stem_db"] = stem_level
        details[f"speaker_{label}_mix_db"] = mix_level
        details[f"speaker_{label}_retained_db"] = retained_db
        checks[f"speaker_{label}_region_present"] = np.sum(mask) >= minimum_samples
        checks[f"speaker_{label}_audible"] = (
            np.isfinite(stem_level) and stem_level > -70
        )
        checks[f"speaker_{label}_survives_mix"] = (
            np.isfinite(retained_db) and retained_db >= -6
        )

    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        **details,
        "speaker_a_mask": a_mask,
        "speaker_b_mask": b_mask,
    }


def _master_candidate(premix, sr, target_lufs, true_peak_ceiling_db):
    """Apply one fixed gain change and the final true-peak safety limiter."""
    premix_lufs = measure_lufs(premix, sr)
    fixed_gain_db = target_lufs - premix_lufs
    master = apply_gain_db(premix, fixed_gain_db)
    master, limited_over_1_pct = apply_true_peak_limiter(
        master,
        sr,
        ceiling_db=true_peak_ceiling_db,
        release_ms=15,
    )

    integrated_lufs = measure_lufs(master, sr)
    true_peak_db = 20 * np.log10(measure_true_peak(master) + 1e-12)
    plr_db = true_peak_db - integrated_lufs

    return master.astype(np.float32), {
        "premix_lufs": float(premix_lufs),
        "fixed_gain_db": float(fixed_gain_db),
        "integrated_lufs": float(integrated_lufs),
        "true_peak_db": float(true_peak_db),
        "limited_over_1_pct": float(limited_over_1_pct),
        "plr_db": float(plr_db),
        "master_compressor_ratio": None,
    }


def build_podcast_master(
    track_a,
    track_b,
    sr,
    target_lufs=PODCAST_MASTER_TARGET_LUFS,
    true_peak_ceiling_db=-1.0,
    speaker_a_mask=None,
    speaker_b_mask=None,
):
    """Build and accept only a master that passes every final quality gate."""
    premix = mix_to_mono(track_a) + mix_to_mono(track_b)
    master, diagnostics = _master_candidate(
        premix, sr, target_lufs, true_peak_ceiling_db
    )
    speaker_a_db = _masked_rms_db(master, speaker_a_mask)
    speaker_b_db = _masked_rms_db(master, speaker_b_mask)
    presence_required = speaker_a_mask is not None or speaker_b_mask is not None
    speaker_balance_db = abs(speaker_a_db - speaker_b_db)
    checks = {
        "finite_audio": bool(np.all(np.isfinite(master))),
        "duration_preserved": len(master) == len(premix),
        "no_clipped_samples": float(np.max(np.abs(master))) < 1.0,
        "loudness_on_target": abs(diagnostics["integrated_lufs"] - target_lufs) <= 1.0,
        "true_peak_safe": diagnostics["true_peak_db"] <= true_peak_ceiling_db + 0.05,
        "limiting_gentle": diagnostics["limited_over_1_pct"]
        <= MASTER_LIMITING_MAX_PERCENT,
        "both_speakers_audible": not presence_required
        or (
            np.isfinite(speaker_a_db)
            and np.isfinite(speaker_b_db)
            and min(speaker_a_db, speaker_b_db) > -50
            and speaker_balance_db <= MASTER_SPEAKER_BALANCE_MAX_DB
        ),
    }
    diagnostics.update(
        {
            "checks": checks,
            "checks_passed": bool(all(checks.values())),
            "speaker_a_db": speaker_a_db,
            "speaker_b_db": speaker_b_db,
            "speaker_balance_db": speaker_balance_db,
        }
    )
    if diagnostics["checks_passed"]:
        return master, diagnostics

    failed_checks = [name for name, passed in checks.items() if not passed]
    raise ValueError(
        "The final fixed-gain master failed its automatic quality checks "
        f"({', '.join(failed_checks)}). The cleaned stems were preserved, but "
        "no file was labeled podcast-ready."
    )


# ============================================================
# SECTION 8: QUALITY VALIDATION
# ============================================================


def validate_track(
    input_audio,
    output_audio,
    sr,
    envelope,
    speech_regions_16k,
    settings,
    limiter_gain=None,
):
    """
    Run quality checks on a single processed track.
    Returns a dict with pass/fail checks and informational metrics.
    """
    mono_in = get_mono(input_audio) if input_audio.ndim == 2 else input_audio
    mono_out = get_mono(output_audio) if output_audio.ndim == 2 else output_audio

    # --- Informational metrics ---
    # Measure LUFS on speech regions only (ducked silence would drag it down)
    input_lufs = measure_lufs(input_audio, sr)
    output_lufs = measure_lufs_speech_only(output_audio, sr, envelope)

    input_peak = np.max(np.abs(mono_in))
    output_peak = np.max(np.abs(mono_out))
    input_peak_db = 20 * np.log10(input_peak + 1e-8)
    output_peak_db = 20 * np.log10(output_peak + 1e-8)

    # Speech coverage: percentage of audio where the gate is open
    speech_pct = (envelope > 0.5).sum() / len(envelope) * 100

    # Number of speech regions and total speech time
    num_regions = len(speech_regions_16k)
    sr_ratio = sr / VAD_SAMPLE_RATE
    speech_seconds = sum(
        (r["end"] - r["start"]) * sr_ratio / sr for r in speech_regions_16k
    )

    # Fade smoothness: max sample-to-sample change in envelope
    # Values > 0.01 suggest harsh transitions that might click
    max_slope = float(np.max(np.abs(np.diff(envelope)))) if len(envelope) > 1 else 0

    if settings["limiter_enabled"] and limiter_gain is not None:
        limiter_reduction_db = -20 * np.log10(np.maximum(limiter_gain, 1e-12))
        limited_over_1_pct = float(np.mean(limiter_reduction_db > 1.0) * 100)
        max_limiter_reduction_db = float(np.max(limiter_reduction_db))
    else:
        limited_over_1_pct = 0.0
        max_limiter_reduction_db = 0.0

    # --- Pass/fail checks ---
    checks = {}

    # 1. No clipping: output peak should not exceed 1.0 (0 dBFS)
    # Limiter runs last in the chain, so peaks should be controlled
    checks["no_clipping"] = output_peak <= 1.001

    # 2. LUFS on target (only if LUFS normalization was enabled)
    # Podcast stems use a fixed staging target before the final mix.
    if settings["lufs_enabled"]:
        expected_lufs = settings["lufs_target"]
        if settings.get("master_enabled", False):
            expected_lufs = PODCAST_STEM_TARGET_LUFS
        checks["lufs_on_target"] = abs(output_lufs - expected_lufs) <= 2.0
    else:
        checks["lufs_on_target"] = True  # Skip if not enabled

    # 3. Duration preserved
    checks["duration_match"] = len(mono_in) == len(mono_out)

    # 4. Speech coverage in reasonable range
    checks["speech_coverage_ok"] = 5 <= speech_pct <= 95

    # Sustained limiter reduction sounds crushed even when the file never
    # crosses 0 dBFS. Keep it below three percent of the program.
    checks["limiting_ok"] = limited_over_1_pct <= 3.0

    return {
        "checks": checks,
        "input_lufs": input_lufs,
        "output_lufs": output_lufs,
        "input_peak_db": input_peak_db,
        "output_peak_db": output_peak_db,
        "speech_pct": speech_pct,
        "num_regions": num_regions,
        "speech_seconds": speech_seconds,
        "max_slope": max_slope,
        "limited_over_1_pct": limited_over_1_pct,
        "max_limiter_reduction_db": max_limiter_reduction_db,
    }


def validate_ducking(input_audio, output_audio, envelope, sr):
    """
    Check that audio is actually attenuated during ducked (gate-closed) regions.

    Compares input and output root mean square levels during regions where the
    envelope is below 0.5, excluding most of each fade.

    Returns the RMS level in dB of the ducked regions, or None if there
    aren't enough ducked samples to measure.
    """
    mono_in = get_mono(input_audio) if input_audio.ndim == 2 else input_audio
    mono_out = get_mono(output_audio) if output_audio.ndim == 2 else output_audio

    # Ducked regions: envelope well below full gain (not in fades)
    ducked_mask = envelope < 0.5

    if ducked_mask.sum() < sr:  # Need at least 1 second of ducked audio
        return None

    input_rms = np.sqrt(np.mean(mono_in[ducked_mask] ** 2))
    output_rms = np.sqrt(np.mean(mono_out[ducked_mask] ** 2))
    return 20 * np.log10((output_rms + 1e-10) / (input_rms + 1e-10))


def validate_ducking_stage(audio_a, audio_b, envelope_a, envelope_b, sr, duck_db):
    """Verify attenuation and preservation on the unmastered, ducked stems."""
    results = {"checks": {}}
    maximum_ducked_gain_db = min(-1.0, float(duck_db) + 3.0)

    for label, audio, envelope in (
        ("a", audio_a, envelope_a),
        ("b", audio_b, envelope_b),
    ):
        ducked = apply_gain_envelope(audio, envelope)
        attenuation_db = validate_ducking(audio, ducked, envelope, sr)
        open_mask = envelope > 0.99
        mono_in = get_mono(audio) if audio.ndim == 2 else audio
        mono_out = get_mono(ducked) if ducked.ndim == 2 else ducked
        if np.sum(open_mask) >= sr:
            input_rms = np.sqrt(
                np.mean(np.square(mono_in[open_mask], dtype=np.float64))
            )
            output_rms = np.sqrt(
                np.mean(np.square(mono_out[open_mask], dtype=np.float64))
            )
            preserved_change_db = 20 * np.log10(
                (output_rms + 1e-12) / (input_rms + 1e-12)
            )
        else:
            preserved_change_db = None

        results[f"speaker_{label}_attenuation_db"] = attenuation_db
        results[f"speaker_{label}_preserved_change_db"] = preserved_change_db
        results["checks"][f"speaker_{label}_attenuated"] = (
            attenuation_db is not None and attenuation_db <= maximum_ducked_gain_db
        )
        results["checks"][f"speaker_{label}_preserved"] = (
            preserved_change_db is not None and abs(preserved_change_db) <= 0.25
        )

    results["passed"] = bool(all(results["checks"].values()))
    return results


def format_quality_report(
    report_a, report_b, ducking_a_db, ducking_b_db, path_a, path_b
):
    """Format validation results into a human-readable report string."""
    lines = []
    lines.append("Processing Complete")
    lines.append("")
    lines.append(f"Speaker A: {os.path.basename(path_a)}")
    lines.append(f"Speaker B: {os.path.basename(path_b)}")
    lines.append(f"Location:  {os.path.dirname(path_a)}")
    lines.append("")
    lines.append("--- Quality Report ---")
    lines.append("")

    # Metrics table
    lines.append(f"{'':30s} {'Speaker A':>12s}  {'Speaker B':>12s}")
    lines.append(
        f"{'LUFS (in > out)':30s} "
        f"{report_a['input_lufs']:5.1f} > {report_a['output_lufs']:5.1f}"
        f"  {report_b['input_lufs']:5.1f} > {report_b['output_lufs']:5.1f}"
    )
    lines.append(
        f"{'Peak dBFS (in > out)':30s} "
        f"{report_a['input_peak_db']:5.1f} > {report_a['output_peak_db']:5.1f}"
        f"  {report_b['input_peak_db']:5.1f} > {report_b['output_peak_db']:5.1f}"
    )
    lines.append(
        f"{'Speech regions':30s} "
        f"{report_a['num_regions']:>12d}  {report_b['num_regions']:>12d}"
    )
    lines.append(
        f"{'Speech coverage':30s} "
        f"{report_a['speech_pct']:11.1f}%  {report_b['speech_pct']:11.1f}%"
    )
    lines.append(
        f"{'Speech time (sec)':30s} "
        f"{report_a['speech_seconds']:12.1f}  {report_b['speech_seconds']:12.1f}"
    )
    lines.append(
        f"{'Max envelope slope':30s} "
        f"{report_a['max_slope']:12.4f}  {report_b['max_slope']:12.4f}"
    )
    lines.append(
        f"{'Limiter >1 dB (% of file)':30s} "
        f"{report_a['limited_over_1_pct']:11.1f}%  "
        f"{report_b['limited_over_1_pct']:11.1f}%"
    )
    lines.append(
        f"{'Maximum limiter reduction':30s} "
        f"{report_a['max_limiter_reduction_db']:11.1f} dB  "
        f"{report_b['max_limiter_reduction_db']:11.1f} dB"
    )
    lines.append("")

    # Pass/fail checks
    lines.append("Checks:")

    # Combine checks from both tracks
    all_pass = True
    for name, label in [
        ("no_clipping", "No clipping"),
        ("lufs_on_target", "LUFS within target"),
        ("duration_match", "Duration preserved"),
        ("speech_coverage_ok", "Speech coverage normal"),
        ("limiting_ok", "Limiter activity gentle"),
    ]:
        pass_a = report_a["checks"][name]
        pass_b = report_b["checks"][name]
        passed = pass_a and pass_b
        mark = "PASS" if passed else "FAIL"
        detail = ""
        if not passed:
            all_pass = False
            fails = []
            if not pass_a:
                fails.append("A")
            if not pass_b:
                fails.append("B")
            detail = f"  (failed: Speaker {', '.join(fails)})"
        lines.append(f"  [{mark}] {label}{detail}")

    # Ducking effectiveness check
    # A reduction of at least 6 dB confirms that the envelope changed the file.
    for label, duck_level in [("A", ducking_a_db), ("B", ducking_b_db)]:
        if duck_level is None:
            lines.append(
                f"  [N/A ] Ducking Speaker {label}: "
                f"not enough ducked regions to measure"
            )
        else:
            duck_pass = duck_level <= -6
            duck_mark = "PASS" if duck_pass else "FAIL"
            if not duck_pass:
                all_pass = False
            lines.append(
                f"  [{duck_mark}] Ducking Speaker {label}: "
                f"{duck_level:.0f} dB attenuation"
            )

    lines.append("")
    if all_pass:
        lines.append("All checks passed.")
    else:
        lines.append("Some checks failed -- review settings and re-process.")

    return "\n".join(lines)


# ============================================================
# SECTION 9: PROCESSING PIPELINE
# ============================================================


def process_track_audio(
    audio,
    sr,
    original_dtype,
    envelope,
    filepath,
    output_dir,
    settings,
    progress_callback,
    status_callback,
):
    """
    Apply the processing chain to a single track (after envelope is computed).

    Pipeline: duck → gain → gentle level staging → compress →
    peak-aware loudness normalization → limit → save.
    The envelope comes from build_cross_ducking_envelopes() which uses both tracks.

    Returns (output_path, track_data) for quality validation.
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    output_path = os.path.join(output_dir, f"{basename}_processed.wav")

    # Keep a copy of the input for validation
    input_audio = audio.copy()

    if settings.get("master_enabled", False):
        status_callback("Removing low-frequency rumble...")
        audio = apply_speech_highpass(audio, sr)

    # --- Apply ducking envelope ---
    status_callback(f"Applying ducking to {os.path.basename(filepath)}...")
    result = apply_gain_envelope(audio, envelope)
    progress_callback(0.15)

    # --- Optional: Gain ---
    if settings["gain_enabled"] and settings["gain_db"] != 0:
        status_callback("Applying gain...")
        result = apply_gain_db(result, settings["gain_db"])
    progress_callback(0.25)

    track_target = settings["lufs_target"]
    if settings.get("master_enabled", False):
        track_target = PODCAST_STEM_TARGET_LUFS

    # Give the compressor a predictable input level. Three decibels of
    # headroom prevents the first pass from being mistaken for finished audio.
    if settings["lufs_enabled"] and settings["comp_enabled"]:
        status_callback("Staging loudness...")
        result = apply_lufs_normalization(
            result, sr, target_lufs=track_target - 3.0, envelope=envelope
        )
    progress_callback(0.40)

    # --- Optional: Compressor (now has normalized signal to compress) ---
    if settings["comp_enabled"]:
        status_callback("Compressing...")
        result = apply_compressor(
            result,
            sr,
            threshold_db=settings["comp_threshold"],
            ratio=settings["comp_ratio"],
            attack_ms=settings["comp_attack"],
            release_ms=settings["comp_release"],
        )
    progress_callback(0.55)

    # Apply loudness once at the end, with enough peak headroom that the
    # limiter only catches isolated transients.
    if settings["lufs_enabled"]:
        status_callback("Finishing loudness...")
        result = apply_lufs_normalization(
            result,
            sr,
            target_lufs=track_target,
            envelope=envelope,
            ceiling_db=(
                settings["limiter_ceiling"] if settings["limiter_enabled"] else -1.0
            ),
        )
    progress_callback(0.70)

    # --- Optional: Limiter (last, catches any peaks) ---
    limiter_gain = np.ones(len(envelope), dtype=np.float32)
    if settings["limiter_enabled"]:
        stem_ceiling = settings["limiter_ceiling"]
        if settings.get("master_enabled", False):
            # These stems are also editor-facing files. Keep them playable on
            # ordinary fixed-point systems, with headroom for the later mix.
            stem_ceiling = min(stem_ceiling, -3.0)
            status_callback("Protecting stem peaks...")
        else:
            status_callback("Limiting...")
        result, limiter_gain = apply_limiter(
            result, sr, ceiling_db=stem_ceiling, release_ms=15, return_gain=True
        )
    progress_callback(0.85)

    # --- Save ---
    status_callback(f"Saving {os.path.basename(output_path)}...")
    save_wav(output_path, sr, result, original_dtype)
    progress_callback(1.0)

    # Return path and data needed for validation
    track_data = {
        "input_audio": input_audio,
        "output_audio": result,
        "sr": sr,
        "envelope": envelope,
        "limiter_gain": limiter_gain,
        "speech_regions": [],  # Populated by caller
    }
    return output_path, track_data


def master_output_path(path_a, path_b, output_dir):
    """Choose a concise master filename from the two source track names."""
    name_a = os.path.splitext(os.path.basename(path_a))[0]
    name_b = os.path.splitext(os.path.basename(path_b))[0]
    common = os.path.commonprefix([name_a, name_b]).rstrip(" _-")
    if len(common) < 3:
        common = "podcast"
    return os.path.join(output_dir, f"{common}_mastered.wav")


# ============================================================
# SECTION 10: GUI
# ============================================================


class DuckingApp(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("Podcast Mic Ducking")

        # Processing state
        self.processing = False
        self.vad_model = None
        self.vad_utils = None
        self._progress_value = 0
        self._status_text = "Ready"
        self._done = False
        self._error = None
        self._result = None
        self._report = ""

        self._build_gui()

        # Force a layout pass so the window sizes itself to fit all widgets
        # before we lock the size — otherwise resizable(False, False) freezes
        # the window at its initial near-zero geometry, and clicks below the
        # visible-but-not-claimed area are lost.
        self.update_idletasks()
        self.resizable(False, False)

        # Force window to become the frontmost foreground app on first Map event.
        # macOS sometimes fails to activate shell-wrapped Python GUIs properly,
        # which breaks click-to-focus on widgets.
        self.bind("<Map>", self._on_first_map, add="+")
        self._mapped_once = False

    def _on_first_map(self, event):
        if self._mapped_once:
            return
        self._mapped_once = True
        try:
            from AppKit import NSApp, NSApplicationActivationPolicyRegular

            NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular)
            NSApp.activateIgnoringOtherApps_(True)
        except Exception:
            pass
        self.lift()
        self.focus_force()

    def _build_gui(self):
        """Build all the GUI widgets."""
        main = ttk.Frame(self, padding=15)
        main.grid(row=0, column=0, sticky="nsew")

        row = 0

        # --- File Selection ---
        ttk.Label(main, text="Audio Files", font=("", 13, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        row += 1

        # Speaker A file picker
        ttk.Label(main, text="Speaker A:").grid(row=row, column=0, sticky="w")
        self.file_a_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.file_a_var, width=45).grid(
            row=row, column=1, padx=5
        )
        ttk.Button(main, text="Browse...", command=lambda: self._browse_file("a")).grid(
            row=row, column=2
        )
        row += 1

        # Speaker B file picker
        ttk.Label(main, text="Speaker B:").grid(row=row, column=0, sticky="w")
        self.file_b_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.file_b_var, width=45).grid(
            row=row, column=1, padx=5
        )
        ttk.Button(main, text="Browse...", command=lambda: self._browse_file("b")).grid(
            row=row, column=2
        )
        row += 1

        # Output directory picker
        ttk.Label(main, text="Output dir:").grid(
            row=row, column=0, sticky="w", pady=(5, 0)
        )
        self.output_dir_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.output_dir_var, width=45).grid(
            row=row, column=1, padx=5, pady=(5, 0)
        )
        ttk.Button(main, text="Browse...", command=self._browse_output).grid(
            row=row, column=2, pady=(5, 0)
        )
        row += 1

        # --- Ducking Settings ---
        ttk.Separator(main, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=10
        )
        row += 1

        ttk.Label(main, text="Ducking", font=("", 12, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w"
        )
        row += 1

        duck_frame = ttk.Frame(main)
        duck_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=3)

        ttk.Label(duck_frame, text="VAD threshold:").grid(row=0, column=0, sticky="w")
        self.vad_thresh_var = tk.StringVar(value="0.5")
        ttk.Entry(duck_frame, textvariable=self.vad_thresh_var, width=6).grid(
            row=0, column=1, padx=(5, 15)
        )

        ttk.Label(duck_frame, text="Fade:").grid(row=0, column=2, sticky="w")
        self.fade_var = tk.StringVar(value="150")
        ttk.Entry(duck_frame, textvariable=self.fade_var, width=6).grid(
            row=0, column=3, padx=(5, 0)
        )
        ttk.Label(duck_frame, text="ms").grid(row=0, column=4, padx=(2, 15))

        ttk.Label(duck_frame, text="Duck:").grid(row=0, column=5, sticky="w")
        self.duck_db_var = tk.StringVar(value="-12")
        ttk.Entry(duck_frame, textvariable=self.duck_db_var, width=6).grid(
            row=0, column=6, padx=(5, 0)
        )
        ttk.Label(duck_frame, text="dB").grid(row=0, column=7, padx=(2, 15))

        ttk.Label(duck_frame, text="Dominance:").grid(row=0, column=8, sticky="w")
        self.dominance_db_var = tk.StringVar(value="3")
        ttk.Entry(duck_frame, textvariable=self.dominance_db_var, width=6).grid(
            row=0, column=9, padx=(5, 0)
        )
        ttk.Label(duck_frame, text="dB").grid(row=0, column=10, padx=(2, 0))
        row += 1

        # --- Processing Chain ---
        ttk.Separator(main, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=10
        )
        row += 1

        ttk.Label(main, text="Processing Chain", font=("", 12, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w"
        )
        row += 1

        preset_frame = ttk.Frame(main)
        preset_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=(3, 5))
        ttk.Label(preset_frame, text="Preset:").grid(row=0, column=0, sticky="w")
        self.preset_var = tk.StringVar(value="Natural cleanup (recommended)")
        preset = ttk.Combobox(
            preset_frame,
            textvariable=self.preset_var,
            values=("Natural cleanup (recommended)", "Podcast-ready", "Custom"),
            width=29,
            state="readonly",
        )
        preset.grid(row=0, column=1, padx=5)
        preset.bind("<<ComboboxSelected>>", self._apply_preset)
        row += 1

        self.preset_help = ttk.Label(
            main,
            text="Turns down mic bleed while preserving the recorded voice.",
            foreground="#555555",
        )
        self.preset_help.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 5))
        row += 1

        # Gain
        gain_frame = ttk.Frame(main)
        gain_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        self.gain_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(gain_frame, text="Gain:", variable=self.gain_enabled).grid(
            row=0, column=0, sticky="w"
        )
        self.gain_db_var = tk.StringVar(value="0")
        ttk.Entry(gain_frame, textvariable=self.gain_db_var, width=6).grid(
            row=0, column=1, padx=5
        )
        ttk.Label(gain_frame, text="dB").grid(row=0, column=2)
        row += 1

        # Compressor
        comp_frame = ttk.Frame(main)
        comp_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        self.comp_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            comp_frame, text="Compressor:", variable=self.comp_enabled
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(comp_frame, text="Thresh:").grid(row=0, column=1, padx=(10, 0))
        self.comp_thresh_var = tk.StringVar(value="-24")
        ttk.Entry(comp_frame, textvariable=self.comp_thresh_var, width=5).grid(
            row=0, column=2, padx=2
        )
        ttk.Label(comp_frame, text="dB").grid(row=0, column=3)

        ttk.Label(comp_frame, text="Ratio:").grid(row=0, column=4, padx=(10, 0))
        self.comp_ratio_var = tk.StringVar(value="2.0")
        ttk.Entry(comp_frame, textvariable=self.comp_ratio_var, width=4).grid(
            row=0, column=5, padx=2
        )
        ttk.Label(comp_frame, text=":1").grid(row=0, column=6)
        row += 1

        # Compressor attack/release (indented under compressor)
        comp_frame2 = ttk.Frame(main)
        comp_frame2.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 2))
        ttk.Label(comp_frame2, text="").grid(row=0, column=0, padx=55)  # indent

        ttk.Label(comp_frame2, text="Attack:").grid(row=0, column=1)
        self.comp_attack_var = tk.StringVar(value="10")
        ttk.Entry(comp_frame2, textvariable=self.comp_attack_var, width=5).grid(
            row=0, column=2, padx=2
        )
        ttk.Label(comp_frame2, text="ms").grid(row=0, column=3)

        ttk.Label(comp_frame2, text="Release:").grid(row=0, column=4, padx=(10, 0))
        self.comp_release_var = tk.StringVar(value="150")
        ttk.Entry(comp_frame2, textvariable=self.comp_release_var, width=5).grid(
            row=0, column=5, padx=2
        )
        ttk.Label(comp_frame2, text="ms").grid(row=0, column=6)
        row += 1

        # Limiter
        lim_frame = ttk.Frame(main)
        lim_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        self.limiter_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(lim_frame, text="Limiter:", variable=self.limiter_enabled).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(lim_frame, text="Ceiling:").grid(row=0, column=1, padx=(10, 0))
        self.limiter_ceil_var = tk.StringVar(value="-1.0")
        ttk.Entry(lim_frame, textvariable=self.limiter_ceil_var, width=6).grid(
            row=0, column=2, padx=2
        )
        ttk.Label(lim_frame, text="dBFS").grid(row=0, column=3)
        row += 1

        # LUFS normalization
        lufs_frame = ttk.Frame(main)
        lufs_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        self.lufs_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(lufs_frame, text="LUFS norm:", variable=self.lufs_enabled).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(lufs_frame, text="Target:").grid(row=0, column=1, padx=(10, 0))
        self.lufs_target_var = tk.StringVar(value="-18")
        ttk.Entry(lufs_frame, textvariable=self.lufs_target_var, width=6).grid(
            row=0, column=2, padx=2
        )
        ttk.Label(lufs_frame, text="LUFS").grid(row=0, column=3)
        row += 1

        # --- Process Button ---
        ttk.Separator(main, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=10
        )
        row += 1

        self.process_btn = ttk.Button(
            main, text="Process", command=self._start_processing
        )
        self.process_btn.grid(row=row, column=0, columnspan=3, sticky="ew", ipady=5)
        row += 1

        # Status label
        self.status_label = ttk.Label(main, text="Ready")
        self.status_label.grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )
        row += 1

        # Progress bar
        self.progress_bar = ttk.Progressbar(main, mode="determinate", maximum=100)
        self.progress_bar.grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(5, 0)
        )

    def _apply_preset(self, _event=None):
        """Fill the controls with safe settings for the selected workflow."""
        preset = self.preset_var.get()
        if preset == "Natural cleanup (recommended)":
            self.fade_var.set("150")
            self.duck_db_var.set("-12")
            self.gain_enabled.set(False)
            self.gain_db_var.set("0")
            self.comp_enabled.set(False)
            self.limiter_enabled.set(False)
            self.lufs_enabled.set(False)
            self.lufs_target_var.set("-18")
            self.preset_help.config(
                text="Turns down mic bleed while preserving the recorded voice."
            )
        elif preset == "Podcast-ready":
            self.fade_var.set("150")
            self.duck_db_var.set("-15")
            self.gain_enabled.set(False)
            self.gain_db_var.set("0")
            self.comp_enabled.set(True)
            self.comp_thresh_var.set("-24")
            self.comp_ratio_var.set("2.0")
            self.comp_attack_var.set("10")
            self.comp_release_var.set("150")
            self.limiter_enabled.set(True)
            self.limiter_ceil_var.set("-1.0")
            self.lufs_enabled.set(True)
            self.lufs_target_var.set("-18")
            self.preset_help.config(
                text="Creates cleaned stems and a mastered mono mix ready for editing."
            )
        else:
            self.preset_help.config(text="Uses the settings shown below.")

    # --- File browsing callbacks ---

    def _browse_file(self, which):
        """Open file picker for speaker A or B."""
        path = filedialog.askopenfilename(
            title=f"Select Speaker {'A' if which == 'a' else 'B'} audio",
            filetypes=[("WAV files", "*.wav *.WAV"), ("All files", "*.*")],
        )
        if path:
            if which == "a":
                self.file_a_var.set(path)
            else:
                self.file_b_var.set(path)

            # Auto-fill output directory from first file selected
            if not self.output_dir_var.get():
                self.output_dir_var.set(os.path.dirname(path))

    def _browse_output(self):
        """Open folder picker for output directory."""
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir_var.set(path)

    # --- Settings ---

    def _get_settings(self):
        """Read all GUI settings into a dictionary."""
        return {
            "vad_threshold": float(self.vad_thresh_var.get()),
            "fade_ms": float(self.fade_var.get()),
            "duck_db": float(self.duck_db_var.get()),
            "dominance_db": float(self.dominance_db_var.get()),
            "gain_enabled": self.gain_enabled.get(),
            "gain_db": float(self.gain_db_var.get()),
            "comp_enabled": self.comp_enabled.get(),
            "comp_threshold": float(self.comp_thresh_var.get()),
            "comp_ratio": float(self.comp_ratio_var.get()),
            "comp_attack": float(self.comp_attack_var.get()),
            "comp_release": float(self.comp_release_var.get()),
            "limiter_enabled": self.limiter_enabled.get(),
            "limiter_ceiling": float(self.limiter_ceil_var.get()),
            "lufs_enabled": self.lufs_enabled.get(),
            "lufs_target": float(self.lufs_target_var.get()),
            "master_enabled": self.preset_var.get() == "Podcast-ready",
        }

    # --- Validation ---

    def _validate(self):
        """Check that inputs are valid before processing."""
        if not self.file_a_var.get():
            messagebox.showerror("Error", "Please select Speaker A audio file.")
            return False
        if not self.file_b_var.get():
            messagebox.showerror("Error", "Please select Speaker B audio file.")
            return False
        if not os.path.isfile(self.file_a_var.get()):
            messagebox.showerror("Error", f"File not found: {self.file_a_var.get()}")
            return False
        if not os.path.isfile(self.file_b_var.get()):
            messagebox.showerror("Error", f"File not found: {self.file_b_var.get()}")
            return False

        try:
            settings = self._get_settings()
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid setting value: {e}")
            return False

        valid_ranges = [
            ("VAD threshold", settings["vad_threshold"], 0.1, 0.9),
            ("Fade", settings["fade_ms"], 10, 1000),
            ("Duck level", settings["duck_db"], -40, 0),
            ("Dominance", settings["dominance_db"], 0, 20),
            ("Compressor ratio", settings["comp_ratio"], 1, 20),
            ("Compressor attack", settings["comp_attack"], 1, 200),
            ("Compressor release", settings["comp_release"], 10, 2000),
            ("Limiter ceiling", settings["limiter_ceiling"], -12, 0),
            ("Loudness target", settings["lufs_target"], -30, -10),
        ]
        for label, value, minimum, maximum in valid_ranges:
            if not minimum <= value <= maximum:
                messagebox.showerror(
                    "Error", f"{label} must be between {minimum} and {maximum}."
                )
                return False

        # Default output dir to same folder as Speaker A
        if not self.output_dir_var.get():
            self.output_dir_var.set(os.path.dirname(self.file_a_var.get()))

        return True

    # --- Processing ---

    def _start_processing(self):
        """Validate inputs and launch processing in a background thread."""
        if not self._validate():
            return

        self.processing = True
        self._done = False
        self._error = None
        self.process_btn.config(state="disabled")
        self.progress_bar["value"] = 0

        settings = self._get_settings()

        thread = threading.Thread(
            target=self._run_processing,
            args=(settings,),
            daemon=True,  # Thread dies when app closes
        )
        thread.start()

        # Start polling the thread's progress every 100 ms
        self.after(100, self._check_progress)

    def _run_processing(self, settings):
        """Process both tracks with cross-track ducking. Runs in a background thread."""
        try:
            # Load the VAD model bundled with the installed silero-vad package.
            self._update_status("Loading VAD model...")
            if self.vad_model is None:
                self.vad_model, self.vad_utils = load_vad_model()
            self._update_progress(3)

            output_dir = self.output_dir_var.get()
            file_a = self.file_a_var.get()
            file_b = self.file_b_var.get()

            # Step 1: Load both tracks
            self._update_status(f"Loading {os.path.basename(file_a)}...")
            sr_a, audio_a, dtype_a = load_wav(file_a)
            self._update_progress(6)

            self._update_status(f"Loading {os.path.basename(file_b)}...")
            sr_b, audio_b, dtype_b = load_wav(file_b)
            self._update_progress(9)

            if sr_a != sr_b:
                raise ValueError(
                    f"The files use different sample rates ({sr_a} and {sr_b} Hz). "
                    "Export both tracks with the same sample rate and try again."
                )
            if len(audio_a) != len(audio_b):
                difference = abs(len(audio_a) - len(audio_b)) / sr_a
                raise ValueError(
                    f"The files have different durations by {difference:.2f} seconds. "
                    "Export synchronized tracks with the same start and end points."
                )

            # Step 2: Run VAD on both tracks
            self._update_status("Running VAD on Speaker A...")
            mono_a = get_mono(audio_a)
            audio_16k_a = resample_to_16k(mono_a, sr_a)
            regions_a = get_speech_regions(
                self.vad_model,
                self.vad_utils,
                audio_16k_a,
                threshold=settings["vad_threshold"],
            )
            self._update_progress(16)

            self._update_status("Running VAD on Speaker B...")
            mono_b = get_mono(audio_b)
            audio_16k_b = resample_to_16k(mono_b, sr_b)
            regions_b = get_speech_regions(
                self.vad_model,
                self.vad_utils,
                audio_16k_b,
                threshold=settings["vad_threshold"],
            )
            self._update_progress(23)

            # Step 3: Build cross-track ducking envelopes
            # This compares RMS levels between tracks to determine who's speaking
            self._update_status("Computing cross-track ducking envelopes...")
            (
                envelope_a,
                envelope_b,
                ducking_diagnostics,
            ) = build_validated_ducking_envelopes(
                mono_a,
                mono_b,
                sr_a,
                regions_a,
                regions_b,
                fade_ms=settings["fade_ms"],
                duck_db=settings["duck_db"],
                dominance_db=settings["dominance_db"],
                require_two_speakers=settings.get("master_enabled", False),
            )

            if ducking_diagnostics["ducking_bypassed"]:
                ducking_stage = {
                    "passed": False,
                    "decision": "bypassed because speaker detection was uncertain",
                }
            else:
                ducking_stage = validate_ducking_stage(
                    audio_a,
                    audio_b,
                    envelope_a,
                    envelope_b,
                    sr_a,
                    settings["duck_db"],
                )
                if not ducking_stage["passed"]:
                    if settings.get("master_enabled", False):
                        failed = [
                            name
                            for name, passed in ducking_stage["checks"].items()
                            if not passed
                        ]
                        raise ValueError(
                            "Ducking failed its pre-master checks "
                            f"({', '.join(failed)}). No podcast-ready master was "
                            "created."
                        )
                    envelope_a = np.ones(len(mono_a), dtype=np.float32)
                    envelope_b = np.ones(len(mono_b), dtype=np.float32)
                    ducking_diagnostics["ducking_bypassed"] = True
                    ducking_stage[
                        "decision"
                    ] = "bypassed after attenuation check failed"
            self._update_progress(28)

            # Step 4: Process each track through the audio chain
            self._update_status("Processing Speaker A...")

            def progress_a(frac):
                self._update_progress(28 + frac * 30)

            path_a, data_a = process_track_audio(
                audio_a,
                sr_a,
                dtype_a,
                envelope_a,
                file_a,
                output_dir,
                settings,
                progress_a,
                self._update_status,
            )
            data_a["speech_regions"] = regions_a

            self._update_status("Processing Speaker B...")

            def progress_b(frac):
                self._update_progress(58 + frac * 30)

            path_b, data_b = process_track_audio(
                audio_b,
                sr_b,
                dtype_b,
                envelope_b,
                file_b,
                output_dir,
                settings,
                progress_b,
                self._update_status,
            )
            data_b["speech_regions"] = regions_b

            report_a = validate_track(
                data_a["input_audio"],
                data_a["output_audio"],
                data_a["sr"],
                data_a["envelope"],
                data_a["speech_regions"],
                settings,
                data_a["limiter_gain"],
            )
            report_b = validate_track(
                data_b["input_audio"],
                data_b["output_audio"],
                data_b["sr"],
                data_b["envelope"],
                data_b["speech_regions"],
                settings,
                data_b["limiter_gain"],
            )

            master_path = None
            master_diagnostics = None
            if settings.get("master_enabled", False):
                failed_stem_checks = [
                    f"Speaker {speaker}: {name}"
                    for speaker, report in (("A", report_a), ("B", report_b))
                    for name, passed in report["checks"].items()
                    if not passed
                ]
                if failed_stem_checks:
                    raise ValueError(
                        "A cleaned stem failed its podcast-ready checks "
                        f"({', '.join(failed_stem_checks)}). The cleaned stems "
                        "were preserved, but no file was labeled podcast-ready."
                    )
                mix_diagnostics = validate_mix_stage(
                    data_a["output_audio"],
                    data_b["output_audio"],
                    envelope_a,
                    envelope_b,
                    sr_a,
                )
                if not mix_diagnostics["passed"]:
                    failed = [
                        name
                        for name, passed in mix_diagnostics["checks"].items()
                        if not passed
                    ]
                    raise ValueError(
                        "The unmastered mix failed its speaker-presence checks "
                        f"({', '.join(failed)}). The cleaned stems were preserved, "
                        "but no file was labeled podcast-ready."
                    )
                self._update_status("Building mastered mono mix...")
                self._update_progress(88)
                master_audio, master_diagnostics = build_podcast_master(
                    data_a["output_audio"],
                    data_b["output_audio"],
                    sr_a,
                    target_lufs=settings["lufs_target"],
                    true_peak_ceiling_db=settings["limiter_ceiling"],
                    speaker_a_mask=mix_diagnostics["speaker_a_mask"],
                    speaker_b_mask=mix_diagnostics["speaker_b_mask"],
                )
                master_path = master_output_path(file_a, file_b, output_dir)
                save_wav(master_path, sr_a, master_audio, np.dtype("float32"))

            # Run quality validation
            self._update_status("Running quality checks...")
            self._update_progress(91)

            self._update_progress(95)

            # Check ducking effectiveness per track
            ducking_a_db = validate_ducking(
                data_a["input_audio"],
                apply_gain_envelope(data_a["input_audio"], data_a["envelope"]),
                data_a["envelope"],
                data_a["sr"],
            )
            ducking_b_db = validate_ducking(
                data_b["input_audio"],
                apply_gain_envelope(data_b["input_audio"], data_b["envelope"]),
                data_b["envelope"],
                data_b["sr"],
            )
            self._update_progress(98)

            # Format the quality report
            report_text = format_quality_report(
                report_a, report_b, ducking_a_db, ducking_b_db, path_a, path_b
            )
            clusters = ducking_diagnostics["ratio_clusters_db"]
            calibration = "automatic" if clusters is not None else "neutral"
            report_text += (
                f"\n\nSpeaker detection: {calibration} calibration, "
                f"A primary {ducking_diagnostics['a_primary_pct']:.1f}%, "
                f"B primary {ducking_diagnostics['b_primary_pct']:.1f}%, "
                f"both/quiet {ducking_diagnostics['ambiguous_pct']:.1f}%."
            )
            attempts = ", ".join(
                f"{value:g}" for value in ducking_diagnostics["dominance_attempts_db"]
            )
            report_text += f"\nDetection threshold attempts: {attempts} dB."
            if ducking_diagnostics["ducking_bypassed"]:
                report_text += (
                    "\nDucking gate: BYPASSED safely; the app did not have enough "
                    "evidence to attenuate either microphone."
                )
            else:
                report_text += "\nDucking gate: PASS before mastering."
            if master_path is not None:
                report_text += (
                    f"\n\nPodcast-ready master: PASS"
                    f"\nMaster: {os.path.basename(master_path)}"
                    f"\nIntegrated loudness: "
                    f"{master_diagnostics['integrated_lufs']:.1f} LUFS"
                    f"\nTrue peak: {master_diagnostics['true_peak_db']:.1f} dBFS"
                    f"\nLimiter >1 dB: "
                    f"{master_diagnostics['limited_over_1_pct']:.1f}% of file"
                    f"\nFixed final gain: "
                    f"{master_diagnostics['fixed_gain_db']:.1f} dB"
                    f"\nMaster compressor: off"
                    f"\nPeak-to-loudness ratio: "
                    f"{master_diagnostics['plr_db']:.1f} dB"
                    f"\nSpeaker balance: "
                    f"{master_diagnostics['speaker_balance_db']:.1f} dB"
                )

            self._update_progress(100)
            self._update_status("Done!")
            self._result = (path_a, path_b, master_path)
            self._report = report_text
            self._done = True

        except Exception as e:
            self._error = str(e)
            self._update_status(f"Error: {e}")
            self._done = True

    def _update_progress(self, value):
        """Set progress value (called from background thread)."""
        self._progress_value = value

    def _update_status(self, text):
        """Set status text (called from background thread)."""
        self._status_text = text

    def _check_progress(self):
        """Poll background thread and update GUI (runs on main thread)."""
        self.progress_bar["value"] = self._progress_value
        self.status_label.config(text=self._status_text)

        if self._done:
            # Processing finished — re-enable button and show result
            self.process_btn.config(state="normal")
            self.processing = False

            if self._error:
                messagebox.showerror("Error", self._error)
            else:
                self._show_report(self._report)
            return

        # Keep polling
        self.after(100, self._check_progress)

    def _show_report(self, report_text):
        """Show quality report in a scrollable window with monospace text."""
        win = tk.Toplevel(self)
        win.title("Quality Report")
        win.resizable(True, True)

        # Monospace text widget so the table columns align
        text = tk.Text(
            win,
            wrap="none",
            font=("Courier", 12),
            width=72,
            height=28,
            padx=10,
            pady=10,
        )
        text.insert("1.0", report_text)
        text.config(state="disabled")  # Read-only
        text.grid(row=0, column=0, sticky="nsew")

        # Scrollbar
        scrollbar = ttk.Scrollbar(win, orient="vertical", command=text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.config(yscrollcommand=scrollbar.set)

        # Close button
        ttk.Button(win, text="Close", command=win.destroy).grid(
            row=1, column=0, columnspan=2, pady=10
        )

        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        # Bring to front
        win.lift()
        win.focus_force()


# ============================================================
# SECTION 11: ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Silero VAD recommends single-threaded torch for CPU inference
    torch.set_num_threads(1)

    app = DuckingApp()
    app.mainloop()
