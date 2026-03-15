"""
Podcast Mic Ducking — Web App

Upload two podcast mic tracks and the app will:
- Detect who's speaking using Silero VAD
- Duck (reduce volume on) each mic when the other speaker is talking
- Apply compression, limiting, and loudness normalization
- Return two processed WAV files for download

Run locally: streamlit run streamlit_app.py
"""

import io
import math

import numpy as np
import streamlit as st
from scipy.io import wavfile
from scipy.signal import resample_poly, sosfilt
from scipy.ndimage import uniform_filter1d, minimum_filter1d


# ============================================================
# CONSTANTS
# ============================================================

VAD_SAMPLE_RATE = 16000


# ============================================================
# AUDIO PROCESSING (adapted from ducking_app.py)
# ============================================================

def load_audio_bytes(uploaded_file):
    """
    Load an audio file from an uploaded BytesIO object.
    Supports WAV (via scipy) and MP3, FLAC, OGG (via soundfile/ffmpeg).
    Returns (sample_rate, audio_float32, original_dtype).
    """
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith('.wav'):
        # Use scipy for WAV files (fast, no extra dependencies)
        sr, data = wavfile.read(io.BytesIO(file_bytes))
        original_dtype = data.dtype
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.float32:
            audio = data.copy()
        elif data.dtype == np.float64:
            audio = data.astype(np.float32)
        else:
            raise ValueError(f"Unsupported WAV format: {data.dtype}")
    else:
        # Use soundfile for MP3, FLAC, OGG, etc.
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(file_bytes), dtype='float32')
        original_dtype = np.dtype('int16')

    # Squeeze single-channel stereo to mono array
    if audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio[:, 0]

    return sr, audio.astype(np.float32), original_dtype


def audio_to_wav_bytes(sr, audio, original_dtype):
    """Convert processed audio to WAV bytes for download."""
    if original_dtype == np.int16:
        data = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    elif original_dtype == np.int32:
        data = np.clip(audio * 2147483648.0, -2147483648, 2147483647).astype(np.int32)
    else:
        data = audio.astype(np.float32)

    buf = io.BytesIO()
    wavfile.write(buf, sr, data)
    buf.seek(0)
    return buf.getvalue()


def get_mono(audio):
    """Extract mono from audio (first channel if stereo)."""
    if audio.ndim == 2:
        return audio[:, 0]
    return audio


def resample_to_16k(audio_mono, orig_sr):
    """Resample mono audio to 16 kHz for VAD processing."""
    if orig_sr == VAD_SAMPLE_RATE:
        return audio_mono
    g = math.gcd(VAD_SAMPLE_RATE, orig_sr)
    up = VAD_SAMPLE_RATE // g
    down = orig_sr // g
    return resample_poly(audio_mono, up, down).astype(np.float32)


@st.cache_resource
def load_vad_model():
    """Load Silero VAD model (cached across sessions)."""
    # Use the silero_vad pip package directly instead of torch.hub
    # (torch.hub has a bug with Authorization headers on some platforms)
    from silero_vad import load_silero_vad, get_speech_timestamps
    model = load_silero_vad()
    return model, get_speech_timestamps


def get_speech_regions(model, utils, audio_16k, threshold=0.5):
    """Run VAD on 16 kHz mono audio, return speech timestamps."""
    import torch
    get_speech_timestamps = utils  # utils is the function itself
    audio_tensor = torch.from_numpy(audio_16k).float()
    timestamps = get_speech_timestamps(
        audio_tensor, model,
        threshold=threshold,
        sampling_rate=VAD_SAMPLE_RATE,
        min_speech_duration_ms=250,
        min_silence_duration_ms=500,
    )
    model.reset_states()
    return timestamps


def _regions_to_mask(speech_regions_16k, length, sr):
    """Convert VAD speech regions (at 16 kHz) to a boolean mask."""
    sr_ratio = sr / VAD_SAMPLE_RATE
    mask = np.zeros(length, dtype=bool)
    for region in speech_regions_16k:
        start = int(round(region['start'] * sr_ratio))
        end = min(int(round(region['end'] * sr_ratio)), length)
        mask[start:end] = True
    return mask


def build_cross_ducking_envelopes(mono_a, mono_b, sr,
                                   speech_regions_a, speech_regions_b,
                                   fade_ms=75, duck_db=-20):
    """Build gain envelopes using cross-track RMS comparison + VAD."""
    length = min(len(mono_a), len(mono_b))
    duck_gain = 10 ** (duck_db / 20.0)

    speech_a = _regions_to_mask(speech_regions_a, length, sr)
    speech_b = _regions_to_mask(speech_regions_b, length, sr)
    either_speech = speech_a | speech_b

    # Running RMS in 50 ms windows
    window = max(int(0.050 * sr), 64)
    sq_a = uniform_filter1d(mono_a[:length].astype(np.float64) ** 2, window, mode='nearest')
    sq_b = uniform_filter1d(mono_b[:length].astype(np.float64) ** 2, window, mode='nearest')
    rms_a = np.sqrt(np.maximum(sq_a, 1e-16)).astype(np.float32)
    rms_b = np.sqrt(np.maximum(sq_b, 1e-16)).astype(np.float32)

    ratio_db = 20 * np.log10(rms_a / (rms_b + 1e-8))

    gain_a = np.full(length, duck_gain, dtype=np.float32)
    gain_b = np.full(length, duck_gain, dtype=np.float32)

    # A is primary speaker (louder by >3 dB)
    a_primary = either_speech & (ratio_db > 3.0)
    gain_a[a_primary] = 1.0

    # B is primary speaker
    b_primary = either_speech & (ratio_db < -3.0)
    gain_b[b_primary] = 1.0

    # Both speaking (similar levels)
    both_active = either_speech & (ratio_db >= -3.0) & (ratio_db <= 3.0)
    gain_a[both_active] = 1.0
    gain_b[both_active] = 1.0

    # Smooth transitions
    fade_samples = int(fade_ms / 1000.0 * sr)
    if fade_samples >= 2:
        gain_a = uniform_filter1d(gain_a.astype(np.float64), fade_samples, mode='nearest').astype(np.float32)
        gain_b = uniform_filter1d(gain_b.astype(np.float64), fade_samples, mode='nearest').astype(np.float32)

    return gain_a, gain_b


def apply_gain_envelope(audio, gain):
    """Multiply audio by gain envelope."""
    if audio.ndim == 2:
        return audio * gain[:, np.newaxis]
    return audio * gain


def apply_compressor(audio, sr, threshold_db=-24, ratio=3.0,
                     attack_ms=10, release_ms=100):
    """Apply dynamic range compression."""
    mono = get_mono(audio) if audio.ndim == 2 else audio
    window_samples = max(int(attack_ms / 1000.0 * sr), 64)
    squared = mono ** 2
    mean_sq = uniform_filter1d(squared.astype(np.float64), window_samples, mode='nearest')
    rms = np.sqrt(np.maximum(mean_sq, 1e-16)).astype(np.float32)
    level_db = 20 * np.log10(rms + 1e-8)
    over_db = np.maximum(level_db - threshold_db, 0)
    gain_reduction_db = over_db * (1.0 - 1.0 / ratio)
    release_samples = max(int(release_ms / 1000.0 * sr), 64)
    gain_reduction_db = uniform_filter1d(
        gain_reduction_db.astype(np.float64), release_samples, mode='nearest'
    ).astype(np.float32)
    gain = 10 ** (-gain_reduction_db / 20.0)
    if audio.ndim == 2:
        return audio * gain[:, np.newaxis]
    return audio * gain


def apply_limiter(audio, sr, ceiling_db=-1.0, release_ms=50):
    """Hard peak limiter."""
    ceiling_linear = 10 ** (ceiling_db / 20.0)
    if audio.ndim == 2:
        peak = np.max(np.abs(audio), axis=1)
    else:
        peak = np.abs(audio)
    gain = np.where(peak > ceiling_linear, ceiling_linear / (peak + 1e-8), 1.0)
    release_samples = max(int(release_ms / 1000.0 * sr), 16)
    lookahead = max(release_samples // 2, 4)
    gain = minimum_filter1d(gain, lookahead, mode='nearest')
    smooth_len = max(lookahead // 4, 4)
    gain = uniform_filter1d(gain.astype(np.float64), smooth_len, mode='nearest').astype(np.float32)
    if audio.ndim == 2:
        return audio * gain[:, np.newaxis]
    return audio * gain


def k_weighting_coeffs(sr):
    """K-weighting filter coefficients (ITU-R BS.1770-4)."""
    f0 = 1681.974450955533
    G = 3.999843853973347
    Q = 0.7071752369554196
    K = np.tan(np.pi * f0 / sr)
    Vh = 10 ** (G / 20)
    Vb = Vh ** 0.4996667741545416
    a0 = 1 + K / Q + K ** 2
    b = np.array([(Vh + Vb * K / Q + K ** 2) / a0, 2 * (K ** 2 - Vh) / a0, (Vh - Vb * K / Q + K ** 2) / a0])
    a = np.array([1.0, 2 * (K ** 2 - 1) / a0, (1 - K / Q + K ** 2) / a0])
    sos1 = np.concatenate([b, a])

    f0_hp = 38.13547087602444
    Q_hp = 0.5003270373238773
    K_hp = np.tan(np.pi * f0_hp / sr)
    a0_hp = 1 + K_hp / Q_hp + K_hp ** 2
    b_hp = np.array([1.0, -2.0, 1.0]) / a0_hp
    a_hp = np.array([1.0, 2 * (K_hp ** 2 - 1) / a0_hp, (1 - K_hp / Q_hp + K_hp ** 2) / a0_hp])
    sos2 = np.concatenate([b_hp, a_hp])
    return np.array([sos1, sos2])


def measure_lufs(audio, sr):
    """Measure integrated loudness in LUFS."""
    sos = k_weighting_coeffs(sr)
    mono = get_mono(audio) if audio.ndim == 2 else audio
    filtered = sosfilt(sos, mono)
    mean_sq = np.mean(filtered ** 2)
    if mean_sq < 1e-10:
        return -70.0
    return -0.691 + 10 * np.log10(mean_sq)


def measure_lufs_speech_only(audio, sr, envelope):
    """Measure LUFS only during speech regions."""
    mono = get_mono(audio) if audio.ndim == 2 else audio
    speech_audio = mono[envelope > 0.5]
    if len(speech_audio) < 1024:
        return -70.0
    return measure_lufs(speech_audio, sr)


def apply_lufs_normalization(audio, sr, target_lufs=-16.0, envelope=None):
    """Normalize audio to target LUFS."""
    if envelope is not None:
        current_lufs = measure_lufs_speech_only(audio, sr, envelope)
    else:
        current_lufs = measure_lufs(audio, sr)
    if current_lufs < -60:
        return audio
    gain_linear = 10 ** ((target_lufs - current_lufs) / 20.0)
    return audio * gain_linear


def process_track(audio, sr, original_dtype, envelope, settings):
    """Process one track: envelope -> gain -> LUFS -> compress -> LUFS -> limit."""
    result = apply_gain_envelope(audio, envelope)

    if settings['gain_db'] != 0:
        result = result * (10 ** (settings['gain_db'] / 20.0))

    # First LUFS pass (so compressor has signal to work with)
    if settings['lufs_enabled']:
        result = apply_lufs_normalization(result, sr, settings['lufs_target'], envelope=envelope)

    # Compress
    if settings['comp_enabled']:
        result = apply_compressor(
            result, sr,
            threshold_db=settings['comp_threshold'],
            ratio=settings['comp_ratio'],
            attack_ms=settings['comp_attack'],
            release_ms=settings['comp_release']
        )

    # Second LUFS pass (re-normalize after compression)
    if settings['lufs_enabled'] and settings['comp_enabled']:
        result = apply_lufs_normalization(result, sr, settings['lufs_target'], envelope=envelope)

    # Limit
    if settings['limiter_enabled']:
        result = apply_limiter(result, sr, ceiling_db=settings['limiter_ceiling'])

    return result


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Podcast Mic Ducking", page_icon="🎙️", layout="centered")

st.title("🎙️ Podcast Mic Ducking")
st.markdown(
    "Upload two podcast mic tracks and the app will automatically "
    "duck each mic when the other speaker is talking, then apply compression "
    "and loudness normalization. Supports WAV, MP3, FLAC, M4A, and OGG."
)

# --- File Uploads ---
SUPPORTED_FORMATS = ["wav", "mp3", "flac", "m4a", "ogg", "aac"]

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Speaker A (host)", type=SUPPORTED_FORMATS, key="file_a")
with col2:
    file_b = st.file_uploader("Speaker B (guest)", type=SUPPORTED_FORMATS, key="file_b")

# --- Settings ---
with st.expander("Advanced Settings"):
    st.markdown("**Ducking**")
    c1, c2, c3 = st.columns(3)
    with c1:
        vad_threshold = st.number_input("VAD threshold", 0.1, 0.9, 0.5, 0.05,
                                         help="How confident the model needs to be that it hears speech")
    with c2:
        fade_ms = st.number_input("Fade (ms)", 10, 200, 75, 5,
                                   help="Transition duration between ducked/unducked")
    with c3:
        duck_db = st.number_input("Duck level (dB)", -40, -5, -20, 1,
                                   help="How much quieter the non-speaking mic gets")

    st.markdown("**Compression**")
    comp_enabled = st.checkbox("Enable compression", value=True)
    if comp_enabled:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            comp_threshold = st.number_input("Threshold (dB)", -40, 0, -24, 1)
        with c2:
            comp_ratio = st.number_input("Ratio", 1.0, 20.0, 3.0, 0.5)
        with c3:
            comp_attack = st.number_input("Attack (ms)", 1, 50, 10, 1)
        with c4:
            comp_release = st.number_input("Release (ms)", 10, 500, 100, 10)
    else:
        comp_threshold, comp_ratio, comp_attack, comp_release = -24, 3.0, 10, 100

    st.markdown("**Loudness**")
    c1, c2 = st.columns(2)
    with c1:
        lufs_enabled = st.checkbox("Enable LUFS normalization", value=True)
        lufs_target = st.number_input("LUFS target", -24, -10, -16, 1) if lufs_enabled else -16
    with c2:
        limiter_enabled = st.checkbox("Enable limiter", value=True)
        limiter_ceiling = st.number_input("Limiter ceiling (dBFS)", -6.0, 0.0, -1.0, 0.5) if limiter_enabled else -1.0

    gain_db = st.number_input("Manual gain (dB)", -20, 20, 0, 1,
                               help="Additional gain before processing (0 = no change)")

settings = {
    'vad_threshold': vad_threshold,
    'fade_ms': fade_ms,
    'duck_db': duck_db,
    'gain_db': gain_db,
    'comp_enabled': comp_enabled,
    'comp_threshold': comp_threshold,
    'comp_ratio': comp_ratio,
    'comp_attack': comp_attack,
    'comp_release': comp_release,
    'lufs_enabled': lufs_enabled,
    'lufs_target': lufs_target,
    'limiter_enabled': limiter_enabled,
    'limiter_ceiling': limiter_ceiling,
}

# --- Process Button ---
if file_a and file_b:
    if st.button("Process", type="primary", use_container_width=True):
        progress = st.progress(0, text="Loading VAD model...")

        try:
            # Load VAD model (cached)
            model, utils = load_vad_model()
            import torch
            torch.set_num_threads(1)
            progress.progress(5, text="Loading audio files...")

            # Load audio
            file_a.seek(0)
            file_b.seek(0)
            sr_a, audio_a, dtype_a = load_audio_bytes(file_a)
            sr_b, audio_b, dtype_b = load_audio_bytes(file_b)
            mono_a = get_mono(audio_a)
            mono_b = get_mono(audio_b)

            # Check duration (reject >60 min)
            duration_a = len(mono_a) / sr_a
            duration_b = len(mono_b) / sr_b
            if duration_a > 3600 or duration_b > 3600:
                st.error("Files must be under 60 minutes each.")
                st.stop()

            progress.progress(10, text="Running VAD on Speaker A...")

            # Run VAD (at 16 kHz)
            audio_16k_a = resample_to_16k(mono_a, sr_a)
            regions_a = get_speech_regions(model, utils, audio_16k_a, settings['vad_threshold'])
            del audio_16k_a  # Free memory
            progress.progress(20, text="Running VAD on Speaker B...")

            audio_16k_b = resample_to_16k(mono_b, sr_b)
            regions_b = get_speech_regions(model, utils, audio_16k_b, settings['vad_threshold'])
            del audio_16k_b  # Free memory
            progress.progress(30, text="Building cross-track ducking envelopes...")

            # Build envelopes
            env_a, env_b = build_cross_ducking_envelopes(
                mono_a, mono_b, sr_a, regions_a, regions_b,
                fade_ms=settings['fade_ms'], duck_db=settings['duck_db']
            )
            progress.progress(35, text="Processing Speaker A...")

            # Process tracks one at a time to save memory
            result_a = process_track(audio_a, sr_a, dtype_a, env_a, settings)
            progress.progress(55, text="Processing Speaker B...")

            result_b = process_track(audio_b, sr_b, dtype_b, env_b, settings)
            progress.progress(75, text="Generating quality report...")

            # Quality metrics (use original audio before freeing)
            input_lufs_a = measure_lufs(audio_a, sr_a)
            input_lufs_b = measure_lufs(audio_b, sr_b)
            del audio_a, audio_b  # Free input audio after measuring
            output_lufs_a = measure_lufs_speech_only(result_a, sr_a, env_a)
            output_lufs_b = measure_lufs_speech_only(result_b, sr_b, env_b)
            peak_a = 20 * np.log10(np.max(np.abs(result_a)) + 1e-8)
            peak_b = 20 * np.log10(np.max(np.abs(result_b)) + 1e-8)
            coverage_a = (env_a > 0.5).sum() / len(env_a) * 100
            coverage_b = (env_b > 0.5).sum() / len(env_b) * 100

            progress.progress(85, text="Converting to WAV...")

            # Convert to downloadable bytes
            name_a = file_a.name.rsplit('.', 1)[0] + '_processed.wav'
            name_b = file_b.name.rsplit('.', 1)[0] + '_processed.wav'
            wav_bytes_a = audio_to_wav_bytes(sr_a, result_a, dtype_a)
            wav_bytes_b = audio_to_wav_bytes(sr_b, result_b, dtype_b)

            progress.progress(100, text="Done!")

            # Store results in session state so they persist across reruns
            # (clicking a download button triggers a rerun)
            st.session_state['results'] = {
                'wav_bytes_a': wav_bytes_a,
                'wav_bytes_b': wav_bytes_b,
                'name_a': name_a,
                'name_b': name_b,
                'report_data': {
                    "": ["LUFS (in → out)", "Peak (dBFS)", "Speech coverage", "Duration"],
                    "Speaker A": [
                        f"{input_lufs_a:.1f} → {output_lufs_a:.1f}",
                        f"{peak_a:.1f}",
                        f"{coverage_a:.1f}%",
                        f"{duration_a:.0f}s",
                    ],
                    "Speaker B": [
                        f"{input_lufs_b:.1f} → {output_lufs_b:.1f}",
                        f"{peak_b:.1f}",
                        f"{coverage_b:.1f}%",
                        f"{duration_b:.0f}s",
                    ],
                },
            }

        except Exception as e:
            st.error(f"Error: {e}")
            raise

elif not file_a or not file_b:
    st.info("Upload both audio files to get started.")

# --- Show results (persists across reruns) ---
if 'results' in st.session_state:
    r = st.session_state['results']
    st.success("Processing complete!")

    st.markdown("### Quality Report")
    st.table(r['report_data'])

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            f"Download {r['name_a']}",
            r['wav_bytes_a'],
            file_name=r['name_a'],
            mime="audio/wav",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            f"Download {r['name_b']}",
            r['wav_bytes_b'],
            file_name=r['name_b'],
            mime="audio/wav",
            use_container_width=True,
        )

# --- Footer ---
st.markdown("---")
st.markdown(
    "<small>Built by [Mikey Jarrell](https://mikeyjarrell.com). "
    "Uses [Silero VAD](https://github.com/snakers4/silero-vad) for voice activity detection.</small>",
    unsafe_allow_html=True,
)
