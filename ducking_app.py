#!/usr/bin/env python3
"""
Podcast Mic Ducking App

Takes two podcast microphone audio files (one per speaker) and:
1. Uses Silero VAD to detect speech, silencing each mic when that speaker isn't talking
2. Applies smooth cosine fades to avoid clicks at transitions
3. Optionally applies gain, compression, limiting, and LUFS normalization

Run with: python ducking_app.py
On first run, the Silero VAD model (~2 MB) will be downloaded automatically.

Dependencies (all in conda base env):
  - torch, numpy, scipy, tkinter (built-in)
"""

import os
import math
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, sosfilt
from scipy.ndimage import uniform_filter1d, minimum_filter1d
import torch


# ============================================================
# CONSTANTS
# ============================================================

VAD_SAMPLE_RATE = 16000   # Silero VAD expects 16 kHz input
VAD_FRAME_SIZE = 512      # Samples per VAD frame at 16 kHz (= 32 ms)


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
    Load Silero VAD model via torch hub.
    Downloads the model on first run (~2 MB), then uses cached version.
    Returns (model, utils) where utils contains helper functions.
    """
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    return model, utils


def get_speech_regions(model, utils, audio_16k, threshold=0.5):
    """
    Run Silero VAD on 16 kHz mono audio.
    Returns a list of dicts: [{'start': sample_idx, 'end': sample_idx}, ...]
    where start/end are sample positions at 16 kHz.
    """
    # utils[0] is the get_speech_timestamps function from Silero
    get_speech_timestamps = utils[0]

    audio_tensor = torch.from_numpy(audio_16k).float()

    # Get speech regions with sensible defaults for podcast audio
    timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        threshold=threshold,
        sampling_rate=VAD_SAMPLE_RATE,
        min_speech_duration_ms=250,    # Ignore speech shorter than 250 ms
        min_silence_duration_ms=500,   # Hold gate open through brief pauses
    )

    # Reset model state so it's clean for the next track
    model.reset_states()

    return timestamps


# ============================================================
# SECTION 3: CROSS-TRACK DUCKING
# ============================================================

def _regions_to_mask(speech_regions_16k, length, sr):
    """Convert VAD speech regions (at 16 kHz) to a boolean mask at the original sample rate."""
    sr_ratio = sr / VAD_SAMPLE_RATE
    mask = np.zeros(length, dtype=bool)
    for region in speech_regions_16k:
        start = int(round(region['start'] * sr_ratio))
        end = int(round(region['end'] * sr_ratio))
        end = min(end, length)
        mask[start:end] = True
    return mask


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
        gain.astype(np.float64), fade_samples, mode='nearest'
    ).astype(np.float32)

    return smoothed


def build_cross_ducking_envelopes(mono_a, mono_b, sr,
                                   speech_regions_a, speech_regions_b,
                                   fade_ms=75, duck_db=-20, dominance_db=3.0):
    """
    Build gain envelopes for both tracks using cross-track comparison.

    Instead of gating each mic by its own VAD (which fails when both mics
    pick up both speakers), this compares RMS levels between the two tracks
    to determine who is actually speaking.

    Logic per frame (with dominance_db as the "clearly louder" threshold):
    - Either VAD active + track A louder by >dominance_db: A=1.0, B=duck_gain
    - Either VAD active + track B louder by >dominance_db: B=1.0, A=duck_gain
    - Otherwise (no VAD, or levels within dominance_db): both=1.0
      (genuine crosstalk or room tone — don't duck either)

    Returns (envelope_a, envelope_b) — per-sample gain arrays with smooth fades.
    """
    length = min(len(mono_a), len(mono_b))
    duck_gain = 10 ** (duck_db / 20.0)

    # Step 1: Build speech masks from VAD regions
    speech_a = _regions_to_mask(speech_regions_a, length, sr)
    speech_b = _regions_to_mask(speech_regions_b, length, sr)
    either_speech = speech_a | speech_b

    # Step 2: Compute running RMS in ~50 ms windows for level comparison
    window = max(int(0.050 * sr), 64)  # 50 ms at native sample rate
    sq_a = uniform_filter1d(mono_a[:length].astype(np.float64) ** 2,
                            window, mode='nearest')
    sq_b = uniform_filter1d(mono_b[:length].astype(np.float64) ** 2,
                            window, mode='nearest')
    rms_a = np.sqrt(np.maximum(sq_a, 1e-16)).astype(np.float32)
    rms_b = np.sqrt(np.maximum(sq_b, 1e-16)).astype(np.float32)

    # Step 3: Compute level ratio in dB (positive = A louder, negative = B louder)
    ratio_db = 20 * np.log10(rms_a / (rms_b + 1e-8))

    # Step 4: Default both envelopes to full gain, then duck the quieter mic
    # only when the other is clearly dominant (and at least one VAD is firing).
    gain_a = np.ones(length, dtype=np.float32)
    gain_b = np.ones(length, dtype=np.float32)

    # A is clearly louder → duck B
    a_primary = either_speech & (ratio_db > dominance_db)
    gain_b[a_primary] = duck_gain

    # B is clearly louder → duck A
    b_primary = either_speech & (ratio_db < -dominance_db)
    gain_a[b_primary] = duck_gain

    # Step 5: Apply cosine fades at transitions to prevent clicks
    gain_a = _smooth_envelope(gain_a, sr, fade_ms)
    gain_b = _smooth_envelope(gain_b, sr, fade_ms)

    return gain_a, gain_b


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


# ============================================================
# SECTION 5: COMPRESSOR
# ============================================================

def apply_compressor(audio, sr, threshold_db=-20, ratio=4.0,
                     attack_ms=10, release_ms=100):
    """
    Apply dynamic range compression to audio.

    Uses a block-based RMS level detector (vectorized, fast).
    Levels above threshold_db are reduced by the ratio (e.g., 4:1 means
    every 4 dB above threshold becomes 1 dB above threshold).
    """
    mono = get_mono(audio) if audio.ndim == 2 else audio

    # Step 1: Compute RMS level in short windows (attack-time length)
    window_samples = max(int(attack_ms / 1000.0 * sr), 64)
    squared = mono ** 2
    # uniform_filter1d computes a running average — gives us windowed mean square
    mean_sq = uniform_filter1d(
        squared.astype(np.float64), window_samples, mode='nearest'
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
        gain_reduction_db.astype(np.float64), release_samples, mode='nearest'
    ).astype(np.float32)

    # Step 5: Convert dB reduction to linear gain and apply
    gain = 10 ** (-gain_reduction_db / 20.0)

    if audio.ndim == 2:
        return audio * gain[:, np.newaxis]
    return audio * gain


# ============================================================
# SECTION 6: LIMITER
# ============================================================

def apply_limiter(audio, sr, ceiling_db=-1.0, release_ms=50):
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
    gain = np.where(peak > ceiling_linear, ceiling_linear / (peak + 1e-8), 1.0)

    # Look-ahead: minimum_filter finds the lowest gain in a window ahead,
    # so we start reducing gain before the peak actually arrives
    release_samples = max(int(release_ms / 1000.0 * sr), 16)
    lookahead = max(release_samples // 2, 4)
    gain = minimum_filter1d(gain, lookahead, mode='nearest')

    # Smooth the gain curve to avoid sudden jumps
    smooth_len = max(lookahead // 4, 4)
    gain = uniform_filter1d(
        gain.astype(np.float64), smooth_len, mode='nearest'
    ).astype(np.float32)

    if audio.ndim == 2:
        return audio * gain[:, np.newaxis]
    return audio * gain


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
    f0 = 1681.974450955533       # Shelf center frequency (Hz)
    G = 3.999843853973347        # Shelf gain (dB)
    Q = 0.7071752369554196       # Quality factor

    K = np.tan(np.pi * f0 / sr)
    Vh = 10 ** (G / 20)
    Vb = Vh ** 0.4996667741545416

    a0 = 1 + K / Q + K ** 2
    b = np.array([
        (Vh + Vb * K / Q + K ** 2) / a0,
        2 * (K ** 2 - Vh) / a0,
        (Vh - Vb * K / Q + K ** 2) / a0,
    ])
    a = np.array([
        1.0,
        2 * (K ** 2 - 1) / a0,
        (1 - K / Q + K ** 2) / a0,
    ])
    sos1 = np.concatenate([b, a])

    # Stage 2: RLB weighting (high-pass at ~38 Hz)
    f0_hp = 38.13547087602444
    Q_hp = 0.5003270373238773
    K_hp = np.tan(np.pi * f0_hp / sr)

    a0_hp = 1 + K_hp / Q_hp + K_hp ** 2
    b_hp = np.array([1.0, -2.0, 1.0]) / a0_hp
    a_hp = np.array([
        1.0,
        2 * (K_hp ** 2 - 1) / a0_hp,
        (1 - K_hp / Q_hp + K_hp ** 2) / a0_hp,
    ])
    sos2 = np.concatenate([b_hp, a_hp])

    return np.array([sos1, sos2])


def measure_lufs(audio, sr):
    """
    Measure integrated loudness in LUFS (ITU-R BS.1770-4 simplified).

    Applies K-weighting filter, then computes mean square of the result.
    This is a simplified version without gating — good enough for
    podcast audio where speech is fairly continuous.
    """
    sos = k_weighting_coeffs(sr)
    mono = get_mono(audio) if audio.ndim == 2 else audio

    # Apply K-weighting filter
    filtered = sosfilt(sos, mono)

    # Compute mean square
    mean_sq = np.mean(filtered ** 2)

    if mean_sq < 1e-10:
        return -70.0  # Effectively silent

    # Convert to LUFS
    lufs = -0.691 + 10 * np.log10(mean_sq)
    return lufs


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


def apply_lufs_normalization(audio, sr, target_lufs=-16.0, envelope=None):
    """
    Normalize audio to a target loudness in LUFS.

    If an envelope is provided, LUFS is measured only during speech regions
    (where envelope > 0.5). This avoids the problem where ducked silence
    drags down the LUFS measurement and causes over-boosting.

    Standard targets:
      - Podcast (stereo): -16 LUFS
      - Podcast (mono):   -19 LUFS (some platforms)
    """
    if envelope is not None:
        current_lufs = measure_lufs_speech_only(audio, sr, envelope)
    else:
        current_lufs = measure_lufs(audio, sr)

    if current_lufs < -60:
        # Audio is too quiet to normalize meaningfully
        return audio

    # Compute how much gain to apply (in dB)
    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20.0)

    result = audio * gain_linear

    # Don't clip here — let the limiter handle peaks (it runs after this)
    return result


# ============================================================
# SECTION 8: QUALITY VALIDATION
# ============================================================

def validate_track(input_audio, output_audio, sr, envelope,
                   speech_regions_16k, settings):
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
        (r['end'] - r['start']) * sr_ratio / sr for r in speech_regions_16k
    )

    # Fade smoothness: max sample-to-sample change in envelope
    # Values > 0.01 suggest harsh transitions that might click
    max_slope = float(np.max(np.abs(np.diff(envelope)))) if len(envelope) > 1 else 0

    # --- Pass/fail checks ---
    checks = {}

    # 1. No clipping: output peak should not exceed 1.0 (0 dBFS)
    # Limiter runs last in the chain, so peaks should be controlled
    checks['no_clipping'] = output_peak <= 1.001

    # 2. LUFS on target (only if LUFS normalization was enabled)
    # Tolerance of ±2.0 dB accounts for simplified (ungated) LUFS measurement
    if settings['lufs_enabled']:
        checks['lufs_on_target'] = abs(output_lufs - settings['lufs_target']) <= 2.0
    else:
        checks['lufs_on_target'] = True  # Skip if not enabled

    # 3. Duration preserved
    checks['duration_match'] = len(mono_in) == len(mono_out)

    # 4. Speech coverage in reasonable range
    checks['speech_coverage_ok'] = 5 <= speech_pct <= 95

    return {
        'checks': checks,
        'input_lufs': input_lufs,
        'output_lufs': output_lufs,
        'input_peak_db': input_peak_db,
        'output_peak_db': output_peak_db,
        'speech_pct': speech_pct,
        'num_regions': num_regions,
        'speech_seconds': speech_seconds,
        'max_slope': max_slope,
    }


def validate_ducking(output_audio, envelope, sr):
    """
    Check that audio is actually attenuated during ducked (gate-closed) regions.

    Measures the RMS level during regions where the envelope is below full gain
    (not in fades — uses envelope < 0.5 to capture ducked regions).

    Returns the RMS level in dB of the ducked regions, or None if there
    aren't enough ducked samples to measure.
    """
    mono = get_mono(output_audio) if output_audio.ndim == 2 else output_audio

    # Ducked regions: envelope well below full gain (not in fades)
    ducked_mask = envelope < 0.5

    if ducked_mask.sum() < sr:  # Need at least 1 second of ducked audio
        return None

    ducked_rms = np.sqrt(np.mean(mono[ducked_mask] ** 2))
    ducked_db = 20 * np.log10(ducked_rms + 1e-10)
    return ducked_db


def format_quality_report(report_a, report_b, ducking_a_db, ducking_b_db, path_a, path_b):
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
    lines.append(f"{'LUFS (in > out)':30s} "
                 f"{report_a['input_lufs']:5.1f} > {report_a['output_lufs']:5.1f}"
                 f"  {report_b['input_lufs']:5.1f} > {report_b['output_lufs']:5.1f}")
    lines.append(f"{'Peak dBFS (in > out)':30s} "
                 f"{report_a['input_peak_db']:5.1f} > {report_a['output_peak_db']:5.1f}"
                 f"  {report_b['input_peak_db']:5.1f} > {report_b['output_peak_db']:5.1f}")
    lines.append(f"{'Speech regions':30s} "
                 f"{report_a['num_regions']:>12d}  {report_b['num_regions']:>12d}")
    lines.append(f"{'Speech coverage':30s} "
                 f"{report_a['speech_pct']:11.1f}%  {report_b['speech_pct']:11.1f}%")
    lines.append(f"{'Speech time (sec)':30s} "
                 f"{report_a['speech_seconds']:12.1f}  {report_b['speech_seconds']:12.1f}")
    lines.append(f"{'Max envelope slope':30s} "
                 f"{report_a['max_slope']:12.4f}  {report_b['max_slope']:12.4f}")
    lines.append("")

    # Pass/fail checks
    lines.append("Checks:")

    # Combine checks from both tracks
    all_pass = True
    for name, label in [
        ('no_clipping', 'No clipping'),
        ('lufs_on_target', 'LUFS within target'),
        ('duration_match', 'Duration preserved'),
        ('speech_coverage_ok', 'Speech coverage normal'),
    ]:
        pass_a = report_a['checks'][name]
        pass_b = report_b['checks'][name]
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
    # During ducked regions, audio should be significantly reduced (< -10 dB)
    for label, duck_level in [("A", ducking_a_db), ("B", ducking_b_db)]:
        if duck_level is None:
            lines.append(f"  [N/A ] Ducking Speaker {label}: "
                         f"not enough ducked regions to measure")
        else:
            duck_pass = duck_level < -10
            duck_mark = "PASS" if duck_pass else "FAIL"
            if not duck_pass:
                all_pass = False
            lines.append(f"  [{duck_mark}] Ducking Speaker {label}: "
                         f"ducked regions at {duck_level:.0f} dB")

    lines.append("")
    if all_pass:
        lines.append("All checks passed.")
    else:
        lines.append("Some checks failed -- review settings and re-process.")

    return "\n".join(lines)


# ============================================================
# SECTION 9: PROCESSING PIPELINE
# ============================================================

def process_track_audio(audio, sr, original_dtype, envelope, filepath,
                        output_dir, settings, progress_callback, status_callback):
    """
    Apply the processing chain to a single track (after envelope is computed).

    Pipeline: apply envelope → gain → compress → LUFS normalize → limit → save.
    The envelope comes from build_cross_ducking_envelopes() which uses both tracks.

    Returns (output_path, track_data) for quality validation.
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    output_path = os.path.join(output_dir, f"{basename}_processed.wav")

    # Keep a copy of the input for validation
    input_audio = audio.copy()

    # --- Apply ducking envelope ---
    status_callback(f"Applying ducking to {os.path.basename(filepath)}...")
    result = apply_gain_envelope(audio, envelope)
    progress_callback(0.15)

    # --- Optional: Gain ---
    if settings['gain_enabled'] and settings['gain_db'] != 0:
        status_callback("Applying gain...")
        result = apply_gain_db(result, settings['gain_db'])
    progress_callback(0.25)

    # --- Optional: First LUFS pass (brings speech to target level) ---
    # This must happen BEFORE compression so the compressor has enough
    # signal to work with (raw recordings are often very quiet)
    if settings['lufs_enabled']:
        status_callback("Normalizing loudness...")
        result = apply_lufs_normalization(
            result, sr, target_lufs=settings['lufs_target'],
            envelope=envelope
        )
    progress_callback(0.40)

    # --- Optional: Compressor (now has normalized signal to compress) ---
    if settings['comp_enabled']:
        status_callback("Compressing...")
        result = apply_compressor(
            result, sr,
            threshold_db=settings['comp_threshold'],
            ratio=settings['comp_ratio'],
            attack_ms=settings['comp_attack'],
            release_ms=settings['comp_release']
        )
    progress_callback(0.55)

    # --- Optional: Second LUFS pass (re-normalize after compression) ---
    # Compression lowers the overall level, so re-normalize to target
    if settings['lufs_enabled'] and settings['comp_enabled']:
        status_callback("Re-normalizing loudness...")
        result = apply_lufs_normalization(
            result, sr, target_lufs=settings['lufs_target'],
            envelope=envelope
        )
    progress_callback(0.70)

    # --- Optional: Limiter (last, catches any peaks) ---
    if settings['limiter_enabled']:
        status_callback("Limiting...")
        result = apply_limiter(result, sr, ceiling_db=settings['limiter_ceiling'])
    progress_callback(0.85)

    # --- Save ---
    status_callback(f"Saving {os.path.basename(output_path)}...")
    save_wav(output_path, sr, result, original_dtype)
    progress_callback(1.0)

    # Return path and data needed for validation
    track_data = {
        'input_audio': input_audio,
        'output_audio': result,
        'sr': sr,
        'envelope': envelope,
        'speech_regions': [],  # Populated by caller
    }
    return output_path, track_data


# ============================================================
# SECTION 9: GUI
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
        self.bind('<Map>', self._on_first_map, add='+')
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
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 8))
        row += 1

        # Speaker A file picker
        ttk.Label(main, text="Speaker A:").grid(row=row, column=0, sticky="w")
        self.file_a_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.file_a_var, width=45).grid(
            row=row, column=1, padx=5)
        ttk.Button(main, text="Browse...",
                   command=lambda: self._browse_file('a')).grid(row=row, column=2)
        row += 1

        # Speaker B file picker
        ttk.Label(main, text="Speaker B:").grid(row=row, column=0, sticky="w")
        self.file_b_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.file_b_var, width=45).grid(
            row=row, column=1, padx=5)
        ttk.Button(main, text="Browse...",
                   command=lambda: self._browse_file('b')).grid(row=row, column=2)
        row += 1

        # Output directory picker
        ttk.Label(main, text="Output dir:").grid(
            row=row, column=0, sticky="w", pady=(5, 0))
        self.output_dir_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.output_dir_var, width=45).grid(
            row=row, column=1, padx=5, pady=(5, 0))
        ttk.Button(main, text="Browse...",
                   command=self._browse_output).grid(row=row, column=2, pady=(5, 0))
        row += 1

        # --- Ducking Settings ---
        ttk.Separator(main, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1

        ttk.Label(main, text="Ducking", font=("", 12, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w")
        row += 1

        duck_frame = ttk.Frame(main)
        duck_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=3)

        ttk.Label(duck_frame, text="VAD threshold:").grid(row=0, column=0, sticky="w")
        self.vad_thresh_var = tk.StringVar(value="0.5")
        ttk.Entry(duck_frame, textvariable=self.vad_thresh_var, width=6).grid(
            row=0, column=1, padx=(5, 15))

        ttk.Label(duck_frame, text="Fade:").grid(row=0, column=2, sticky="w")
        self.fade_var = tk.StringVar(value="75")
        ttk.Entry(duck_frame, textvariable=self.fade_var, width=6).grid(
            row=0, column=3, padx=(5, 0))
        ttk.Label(duck_frame, text="ms").grid(row=0, column=4, padx=(2, 15))

        ttk.Label(duck_frame, text="Duck:").grid(row=0, column=5, sticky="w")
        self.duck_db_var = tk.StringVar(value="-20")
        ttk.Entry(duck_frame, textvariable=self.duck_db_var, width=6).grid(
            row=0, column=6, padx=(5, 0))
        ttk.Label(duck_frame, text="dB").grid(row=0, column=7, padx=(2, 15))

        ttk.Label(duck_frame, text="Dominance:").grid(row=0, column=8, sticky="w")
        self.dominance_db_var = tk.StringVar(value="3")
        ttk.Entry(duck_frame, textvariable=self.dominance_db_var, width=6).grid(
            row=0, column=9, padx=(5, 0))
        ttk.Label(duck_frame, text="dB").grid(row=0, column=10, padx=(2, 0))
        row += 1

        # --- Processing Chain ---
        ttk.Separator(main, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1

        ttk.Label(main, text="Processing Chain", font=("", 12, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w")
        row += 1

        # Gain
        gain_frame = ttk.Frame(main)
        gain_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        self.gain_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(gain_frame, text="Gain:",
                        variable=self.gain_enabled).grid(row=0, column=0, sticky="w")
        self.gain_db_var = tk.StringVar(value="0")
        ttk.Entry(gain_frame, textvariable=self.gain_db_var, width=6).grid(
            row=0, column=1, padx=5)
        ttk.Label(gain_frame, text="dB").grid(row=0, column=2)
        row += 1

        # Compressor
        comp_frame = ttk.Frame(main)
        comp_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        self.comp_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(comp_frame, text="Compressor:",
                        variable=self.comp_enabled).grid(row=0, column=0, sticky="w")

        ttk.Label(comp_frame, text="Thresh:").grid(row=0, column=1, padx=(10, 0))
        self.comp_thresh_var = tk.StringVar(value="-24")
        ttk.Entry(comp_frame, textvariable=self.comp_thresh_var, width=5).grid(
            row=0, column=2, padx=2)
        ttk.Label(comp_frame, text="dB").grid(row=0, column=3)

        ttk.Label(comp_frame, text="Ratio:").grid(row=0, column=4, padx=(10, 0))
        self.comp_ratio_var = tk.StringVar(value="3")
        ttk.Entry(comp_frame, textvariable=self.comp_ratio_var, width=4).grid(
            row=0, column=5, padx=2)
        ttk.Label(comp_frame, text=":1").grid(row=0, column=6)
        row += 1

        # Compressor attack/release (indented under compressor)
        comp_frame2 = ttk.Frame(main)
        comp_frame2.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 2))
        ttk.Label(comp_frame2, text="").grid(row=0, column=0, padx=55)  # indent

        ttk.Label(comp_frame2, text="Attack:").grid(row=0, column=1)
        self.comp_attack_var = tk.StringVar(value="10")
        ttk.Entry(comp_frame2, textvariable=self.comp_attack_var, width=5).grid(
            row=0, column=2, padx=2)
        ttk.Label(comp_frame2, text="ms").grid(row=0, column=3)

        ttk.Label(comp_frame2, text="Release:").grid(row=0, column=4, padx=(10, 0))
        self.comp_release_var = tk.StringVar(value="100")
        ttk.Entry(comp_frame2, textvariable=self.comp_release_var, width=5).grid(
            row=0, column=5, padx=2)
        ttk.Label(comp_frame2, text="ms").grid(row=0, column=6)
        row += 1

        # Limiter
        lim_frame = ttk.Frame(main)
        lim_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        self.limiter_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(lim_frame, text="Limiter:",
                        variable=self.limiter_enabled).grid(row=0, column=0, sticky="w")
        ttk.Label(lim_frame, text="Ceiling:").grid(row=0, column=1, padx=(10, 0))
        self.limiter_ceil_var = tk.StringVar(value="-1.0")
        ttk.Entry(lim_frame, textvariable=self.limiter_ceil_var, width=6).grid(
            row=0, column=2, padx=2)
        ttk.Label(lim_frame, text="dBFS").grid(row=0, column=3)
        row += 1

        # LUFS normalization
        lufs_frame = ttk.Frame(main)
        lufs_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        self.lufs_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(lufs_frame, text="LUFS norm:",
                        variable=self.lufs_enabled).grid(row=0, column=0, sticky="w")
        ttk.Label(lufs_frame, text="Target:").grid(row=0, column=1, padx=(10, 0))
        self.lufs_target_var = tk.StringVar(value="-16")
        ttk.Entry(lufs_frame, textvariable=self.lufs_target_var, width=6).grid(
            row=0, column=2, padx=2)
        ttk.Label(lufs_frame, text="LUFS").grid(row=0, column=3)
        row += 1

        # --- Process Button ---
        ttk.Separator(main, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1

        self.process_btn = ttk.Button(
            main, text="Process", command=self._start_processing)
        self.process_btn.grid(row=row, column=0, columnspan=3, sticky="ew", ipady=5)
        row += 1

        # Status label
        self.status_label = ttk.Label(main, text="Ready")
        self.status_label.grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(10, 0))
        row += 1

        # Progress bar
        self.progress_bar = ttk.Progressbar(main, mode="determinate", maximum=100)
        self.progress_bar.grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(5, 0))

    # --- File browsing callbacks ---

    def _browse_file(self, which):
        """Open file picker for speaker A or B."""
        path = filedialog.askopenfilename(
            title=f"Select Speaker {'A' if which == 'a' else 'B'} audio",
            filetypes=[("WAV files", "*.wav *.WAV"), ("All files", "*.*")]
        )
        if path:
            if which == 'a':
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
            'vad_threshold': float(self.vad_thresh_var.get()),
            'fade_ms': float(self.fade_var.get()),
            'duck_db': float(self.duck_db_var.get()),
            'dominance_db': float(self.dominance_db_var.get()),
            'gain_enabled': self.gain_enabled.get(),
            'gain_db': float(self.gain_db_var.get()),
            'comp_enabled': self.comp_enabled.get(),
            'comp_threshold': float(self.comp_thresh_var.get()),
            'comp_ratio': float(self.comp_ratio_var.get()),
            'comp_attack': float(self.comp_attack_var.get()),
            'comp_release': float(self.comp_release_var.get()),
            'limiter_enabled': self.limiter_enabled.get(),
            'limiter_ceiling': float(self.limiter_ceil_var.get()),
            'lufs_enabled': self.lufs_enabled.get(),
            'lufs_target': float(self.lufs_target_var.get()),
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
            self._get_settings()
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid setting value: {e}")
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
            daemon=True   # Thread dies when app closes
        )
        thread.start()

        # Start polling the thread's progress every 100 ms
        self.after(100, self._check_progress)

    def _run_processing(self, settings):
        """Process both tracks with cross-track ducking. Runs in a background thread."""
        try:
            # Load VAD model (downloads on first run)
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

            # Step 2: Run VAD on both tracks
            self._update_status("Running VAD on Speaker A...")
            mono_a = get_mono(audio_a)
            audio_16k_a = resample_to_16k(mono_a, sr_a)
            regions_a = get_speech_regions(
                self.vad_model, self.vad_utils, audio_16k_a,
                threshold=settings['vad_threshold']
            )
            self._update_progress(16)

            self._update_status("Running VAD on Speaker B...")
            mono_b = get_mono(audio_b)
            audio_16k_b = resample_to_16k(mono_b, sr_b)
            regions_b = get_speech_regions(
                self.vad_model, self.vad_utils, audio_16k_b,
                threshold=settings['vad_threshold']
            )
            self._update_progress(23)

            # Step 3: Build cross-track ducking envelopes
            # This compares RMS levels between tracks to determine who's speaking
            self._update_status("Computing cross-track ducking envelopes...")
            envelope_a, envelope_b = build_cross_ducking_envelopes(
                mono_a, mono_b, sr_a,
                regions_a, regions_b,
                fade_ms=settings['fade_ms'],
                duck_db=settings['duck_db'],
                dominance_db=settings['dominance_db']
            )
            self._update_progress(28)

            # Step 4: Process each track through the audio chain
            self._update_status("Processing Speaker A...")

            def progress_a(frac):
                self._update_progress(28 + frac * 30)

            path_a, data_a = process_track_audio(
                audio_a, sr_a, dtype_a, envelope_a, file_a,
                output_dir, settings, progress_a, self._update_status
            )
            data_a['speech_regions'] = regions_a

            self._update_status("Processing Speaker B...")

            def progress_b(frac):
                self._update_progress(58 + frac * 30)

            path_b, data_b = process_track_audio(
                audio_b, sr_b, dtype_b, envelope_b, file_b,
                output_dir, settings, progress_b, self._update_status
            )
            data_b['speech_regions'] = regions_b

            # Run quality validation (progress: 85% → 100%)
            self._update_status("Running quality checks...")
            self._update_progress(88)

            report_a = validate_track(
                data_a['input_audio'], data_a['output_audio'],
                data_a['sr'], data_a['envelope'],
                data_a['speech_regions'], settings
            )
            report_b = validate_track(
                data_b['input_audio'], data_b['output_audio'],
                data_b['sr'], data_b['envelope'],
                data_b['speech_regions'], settings
            )
            self._update_progress(93)

            # Check ducking effectiveness per track
            ducking_a_db = validate_ducking(
                data_a['output_audio'], data_a['envelope'], data_a['sr']
            )
            ducking_b_db = validate_ducking(
                data_b['output_audio'], data_b['envelope'], data_b['sr']
            )
            self._update_progress(97)

            # Format the quality report
            report_text = format_quality_report(
                report_a, report_b, ducking_a_db, ducking_b_db, path_a, path_b
            )

            self._update_progress(100)
            self._update_status("Done!")
            self._result = (path_a, path_b)
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
        text = tk.Text(win, wrap="none", font=("Courier", 12),
                       width=72, height=28, padx=10, pady=10)
        text.insert("1.0", report_text)
        text.config(state="disabled")  # Read-only
        text.grid(row=0, column=0, sticky="nsew")

        # Scrollbar
        scrollbar = ttk.Scrollbar(win, orient="vertical", command=text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.config(yscrollcommand=scrollbar.set)

        # Close button
        ttk.Button(win, text="Close", command=win.destroy).grid(
            row=1, column=0, columnspan=2, pady=10)

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
