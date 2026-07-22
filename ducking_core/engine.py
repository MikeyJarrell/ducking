"""Canonical signal-processing engine shared by Ducking and Backstory.

Takes two podcast microphone audio files (one per speaker) and:
1. Uses Silero VAD to detect speech and reduce bleed from the unused mic
2. Applies smooth fades to avoid clicks at transitions
3. Optionally applies gain, compression, limiting, and LUFS normalization

The Silero VAD model is bundled with the installed silero-vad package.

Dependencies (all in conda base env):
  - torch, numpy, scipy, tkinter (built-in)
"""

import os
import math
import sys
import types

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


def process_track(audio, sr, envelope, settings):
    """Duck one track and optionally apply a gentle podcast finishing chain."""
    if settings.get("master_enabled", False):
        audio = apply_speech_highpass(audio, sr)
    result = apply_gain_envelope(audio, envelope)

    if settings["gain_db"] != 0:
        result = result * (10 ** (settings["gain_db"] / 20.0))

    track_target = settings["lufs_target"]
    if settings.get("master_enabled", False):
        track_target = PODCAST_STEM_TARGET_LUFS

    if settings["lufs_enabled"] and settings["comp_enabled"]:
        result = apply_lufs_normalization(
            result, sr, track_target - 3.0, envelope=envelope
        )

    # Compress
    if settings["comp_enabled"]:
        result = apply_compressor(
            result,
            sr,
            threshold_db=settings["comp_threshold"],
            ratio=settings["comp_ratio"],
            attack_ms=settings["comp_attack"],
            release_ms=settings["comp_release"],
        )

    if settings["lufs_enabled"]:
        result = apply_lufs_normalization(
            result,
            sr,
            track_target,
            envelope=envelope,
            ceiling_db=(
                settings["limiter_ceiling"] if settings["limiter_enabled"] else -1.0
            ),
        )

    limiter_gain = np.ones(len(envelope), dtype=np.float32)
    if settings["limiter_enabled"]:
        stem_ceiling = settings["limiter_ceiling"]
        if settings.get("master_enabled", False):
            stem_ceiling = min(stem_ceiling, -3.0)
        result, limiter_gain = apply_limiter(
            result,
            sr,
            ceiling_db=stem_ceiling,
            release_ms=15,
            return_gain=True,
        )

    return result, limiter_gain


def validate_podcast_stem(input_audio, output_audio, sr, envelope, limiter_gain):
    """Reject a stem that is unsafe or requires sustained limiting."""
    reduction_db = -20 * np.log10(np.maximum(limiter_gain, 1e-12))
    output_lufs = measure_lufs_speech_only(output_audio, sr, envelope)
    return {
        "finite_audio": bool(np.all(np.isfinite(output_audio))),
        "duration_preserved": len(output_audio) == len(input_audio),
        "no_clipped_samples": float(np.max(np.abs(output_audio))) < 1.0,
        "loudness_on_target": abs(output_lufs - PODCAST_STEM_TARGET_LUFS) <= 2.0,
        "limiting_gentle": float(np.mean(reduction_db > 1.0) * 100) <= 3.0,
    }


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
