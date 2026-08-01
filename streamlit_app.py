"""Web interface for the shared Ducking audio engine."""

import io
import os

import numpy as np
import streamlit as st
from scipy.io import wavfile

from ducking_core.engine import (
    build_podcast_master,
    build_validated_ducking_envelopes,
    get_mono,
    get_speech_regions,
    load_vad_model as load_core_vad_model,
    measure_lufs,
    measure_lufs_speech_only,
    process_track,
    resample_to_16k,
    validate_mix_stage,
    validate_podcast_stem,
    validate_ducking_stage,
)


def load_audio_bytes(uploaded_file):
    """
    Load an audio file from an uploaded BytesIO object.
    Supports WAV (via scipy) and MP3, FLAC, OGG (via soundfile/ffmpeg).
    Returns (sample_rate, audio_float32, original_dtype).
    """
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".wav"):
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

        audio, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
        original_dtype = np.dtype("int16")

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


@st.cache_resource
def load_vad_model():
    """Load the package-owned voice detector once per web process."""
    return load_core_vad_model()


# Show the app icon in the browser tab. The same icon.png feeds the macOS .app
# build. If it is ever missing (a stripped-down deployment, say), fall back to a
# microphone emoji rather than crashing the whole page.
_ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
_PAGE_ICON = _ICON_PATH if os.path.exists(_ICON_PATH) else "🎙️"

st.set_page_config(
    page_title="Podcast Mic Ducking",
    page_icon=_PAGE_ICON,
    layout="centered",
)

st.title("Podcast Mic Ducking")
st.markdown(
    "Upload the two synchronized microphone tracks. The app turns down each "
    "mic's bleed while the other person speaks."
)

# --- File Uploads ---
SUPPORTED_FORMATS = ["wav", "mp3", "flac", "m4a", "ogg", "aac"]

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Speaker A (host)", type=SUPPORTED_FORMATS, key="file_a")
with col2:
    file_b = st.file_uploader("Speaker B (guest)", type=SUPPORTED_FORMATS, key="file_b")


def _apply_web_preset():
    """Update advanced controls when a named preset is selected."""
    selected = st.session_state.processing_preset
    if selected == "Custom":
        return
    ready = selected == "Podcast-ready"
    st.session_state.fade_ms = 150
    st.session_state.duck_db = -15 if ready else -12
    st.session_state.comp_enabled = ready
    st.session_state.comp_ratio = 2.0
    st.session_state.comp_release = 150
    st.session_state.lufs_enabled = ready
    st.session_state.lufs_target = -18
    st.session_state.limiter_enabled = ready


preset = st.selectbox(
    "Processing preset",
    ["Natural cleanup (recommended)", "Podcast-ready", "Custom"],
    help="Natural cleanup preserves the recorded voice. Podcast-ready also finishes loudness.",
    key="processing_preset",
    on_change=_apply_web_preset,
)
podcast_ready = preset == "Podcast-ready"

# --- Settings ---
with st.expander("Advanced Settings"):
    st.markdown("**Ducking**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        vad_threshold = st.number_input(
            "VAD threshold",
            0.1,
            0.9,
            0.5,
            0.05,
            help="How confident the model needs to be that it hears speech",
        )
    with c2:
        fade_ms = st.number_input(
            "Fade (ms)",
            10,
            500,
            150,
            5,
            key="fade_ms",
            help="Transition duration between ducked/unducked",
        )
    with c3:
        duck_db = st.number_input(
            "Duck level (dB)",
            -40,
            -5,
            -15 if podcast_ready else -12,
            1,
            key="duck_db",
            help="How much quieter the non-speaking mic gets",
        )
    with c4:
        dominance_db = st.number_input(
            "Dominance (dB)",
            0.0,
            12.0,
            3.0,
            0.5,
            help="Required close-mic advantage before ducking",
        )

    st.markdown("**Compression**")
    comp_enabled = st.checkbox(
        "Enable compression", value=podcast_ready, key="comp_enabled"
    )
    if comp_enabled:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            comp_threshold = st.number_input("Threshold (dB)", -40, 0, -24, 1)
        with c2:
            comp_ratio = st.number_input("Ratio", 1.0, 20.0, 2.0, 0.5, key="comp_ratio")
        with c3:
            comp_attack = st.number_input("Attack (ms)", 1, 50, 10, 1)
        with c4:
            comp_release = st.number_input(
                "Release (ms)", 10, 500, 150, 10, key="comp_release"
            )
    else:
        comp_threshold, comp_ratio, comp_attack, comp_release = -24, 3.0, 10, 100

    st.markdown("**Loudness**")
    c1, c2 = st.columns(2)
    with c1:
        lufs_enabled = st.checkbox(
            "Enable LUFS normalization", value=podcast_ready, key="lufs_enabled"
        )
        lufs_target = (
            st.number_input("LUFS target", -24, -10, -18, 1, key="lufs_target")
            if lufs_enabled
            else -18
        )
    with c2:
        limiter_enabled = st.checkbox(
            "Enable limiter", value=podcast_ready, key="limiter_enabled"
        )
        limiter_ceiling = (
            st.number_input("Limiter ceiling (dBFS)", -6.0, 0.0, -1.0, 0.5)
            if limiter_enabled
            else -1.0
        )

    gain_db = st.number_input(
        "Manual gain (dB)",
        -20,
        20,
        0,
        1,
        help="Additional gain before processing (0 = no change)",
    )

settings = {
    "vad_threshold": vad_threshold,
    "fade_ms": fade_ms,
    "duck_db": duck_db,
    "dominance_db": dominance_db,
    "gain_db": gain_db,
    "comp_enabled": comp_enabled,
    "comp_threshold": comp_threshold,
    "comp_ratio": comp_ratio,
    "comp_attack": comp_attack,
    "comp_release": comp_release,
    "lufs_enabled": lufs_enabled,
    "lufs_target": lufs_target,
    "limiter_enabled": limiter_enabled,
    "limiter_ceiling": limiter_ceiling,
    "master_enabled": preset == "Podcast-ready",
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

            if sr_a != sr_b:
                raise ValueError(
                    f"The files use different sample rates ({sr_a} and {sr_b} Hz). "
                    "Export both tracks with the same sample rate."
                )
            if len(mono_a) != len(mono_b):
                difference = abs(len(mono_a) - len(mono_b)) / sr_a
                raise ValueError(
                    f"The files differ in duration by {difference:.2f} seconds. "
                    "Export synchronized tracks with the same start and end points."
                )

            # Check duration (reject >60 min)
            duration_a = len(mono_a) / sr_a
            duration_b = len(mono_b) / sr_b
            if duration_a > 3600 or duration_b > 3600:
                st.error("Files must be under 60 minutes each.")
                st.stop()

            progress.progress(10, text="Running VAD on Speaker A...")

            # Run VAD (at 16 kHz)
            audio_16k_a = resample_to_16k(mono_a, sr_a)
            regions_a = get_speech_regions(
                model, utils, audio_16k_a, settings["vad_threshold"]
            )
            del audio_16k_a  # Free memory
            progress.progress(20, text="Running VAD on Speaker B...")

            audio_16k_b = resample_to_16k(mono_b, sr_b)
            regions_b = get_speech_regions(
                model, utils, audio_16k_b, settings["vad_threshold"]
            )
            del audio_16k_b  # Free memory
            progress.progress(30, text="Building cross-track ducking envelopes...")

            # Build and validate envelopes before mastering changes the levels.
            env_a, env_b, detection_metrics = build_validated_ducking_envelopes(
                mono_a,
                mono_b,
                sr_a,
                regions_a,
                regions_b,
                fade_ms=settings["fade_ms"],
                duck_db=settings["duck_db"],
                dominance_db=settings["dominance_db"],
                require_two_speakers=settings["master_enabled"],
            )
            if detection_metrics["ducking_bypassed"]:
                ducking_metrics = {"passed": False}
            else:
                ducking_metrics = validate_ducking_stage(
                    audio_a,
                    audio_b,
                    env_a,
                    env_b,
                    sr_a,
                    settings["duck_db"],
                )
                if not ducking_metrics["passed"]:
                    if settings["master_enabled"]:
                        failed = [
                            name
                            for name, passed in ducking_metrics["checks"].items()
                            if not passed
                        ]
                        raise ValueError(
                            "Ducking failed its pre-master checks "
                            f"({', '.join(failed)}). No podcast-ready master was "
                            "created."
                        )
                    env_a = np.ones(len(mono_a), dtype=np.float32)
                    env_b = np.ones(len(mono_b), dtype=np.float32)
                    detection_metrics["ducking_bypassed"] = True
            progress.progress(35, text="Processing Speaker A...")

            # Process tracks one at a time to save memory
            result_a, limiter_gain_a = process_track(audio_a, sr_a, env_a, settings)
            progress.progress(55, text="Processing Speaker B...")

            result_b, limiter_gain_b = process_track(audio_b, sr_b, env_b, settings)
            progress.progress(70, text="Generating quality report...")

            # Quality metrics (use original audio before freeing)
            input_lufs_a = measure_lufs(audio_a, sr_a)
            input_lufs_b = measure_lufs(audio_b, sr_b)

            master = None
            master_metrics = None
            if settings["master_enabled"]:
                stem_checks = {
                    "Speaker A": validate_podcast_stem(
                        audio_a, result_a, sr_a, env_a, limiter_gain_a
                    ),
                    "Speaker B": validate_podcast_stem(
                        audio_b, result_b, sr_b, env_b, limiter_gain_b
                    ),
                }
                failed_stem_checks = [
                    f"{speaker}: {name}"
                    for speaker, checks in stem_checks.items()
                    for name, passed in checks.items()
                    if not passed
                ]
                if failed_stem_checks:
                    raise ValueError(
                        "A cleaned stem failed its podcast-ready checks "
                        f"({', '.join(failed_stem_checks)}). No file was labeled "
                        "podcast-ready."
                    )
                mix_metrics = validate_mix_stage(result_a, result_b, env_a, env_b, sr_a)
                if not mix_metrics["passed"]:
                    failed = [
                        name
                        for name, passed in mix_metrics["checks"].items()
                        if not passed
                    ]
                    raise ValueError(
                        "The unmastered mix failed its speaker-presence checks "
                        f"({', '.join(failed)}). No file was labeled podcast-ready."
                    )
                progress.progress(75, text="Building mastered mono mix...")
                master, master_metrics = build_podcast_master(
                    result_a,
                    result_b,
                    sr_a,
                    target_lufs=settings["lufs_target"],
                    true_peak_ceiling_db=settings["limiter_ceiling"],
                    speaker_a_mask=mix_metrics["speaker_a_mask"],
                    speaker_b_mask=mix_metrics["speaker_b_mask"],
                )

            del audio_a, audio_b  # Free input audio after validation.

            output_lufs_a = measure_lufs_speech_only(result_a, sr_a, env_a)
            output_lufs_b = measure_lufs_speech_only(result_b, sr_b, env_b)
            peak_a = 20 * np.log10(np.max(np.abs(result_a)) + 1e-8)
            peak_b = 20 * np.log10(np.max(np.abs(result_b)) + 1e-8)
            coverage_a = (env_a > 0.5).sum() / len(env_a) * 100
            coverage_b = (env_b > 0.5).sum() / len(env_b) * 100

            progress.progress(85, text="Converting to WAV...")

            # Convert to downloadable bytes
            name_a = file_a.name.rsplit(".", 1)[0] + "_processed.wav"
            name_b = file_b.name.rsplit(".", 1)[0] + "_processed.wav"
            wav_bytes_a = audio_to_wav_bytes(sr_a, result_a, dtype_a)
            wav_bytes_b = audio_to_wav_bytes(sr_b, result_b, dtype_b)
            wav_bytes_master = None
            name_master = None
            if master is not None:
                common_name = os.path.commonprefix(
                    [file_a.name.rsplit(".", 1)[0], file_b.name.rsplit(".", 1)[0]]
                ).rstrip(" _-")
                if len(common_name) < 3:
                    common_name = "podcast"
                name_master = common_name + "_mastered.wav"
                wav_bytes_master = audio_to_wav_bytes(sr_a, master, np.dtype("float32"))

            progress.progress(100, text="Done!")

            # Store results in session state so they persist across reruns
            # (clicking a download button triggers a rerun)
            st.session_state["results"] = {
                "wav_bytes_a": wav_bytes_a,
                "wav_bytes_b": wav_bytes_b,
                "name_a": name_a,
                "name_b": name_b,
                "wav_bytes_master": wav_bytes_master,
                "name_master": name_master,
                "validation_summary": (
                    "Ducking safely bypassed after a failed confidence check."
                    if detection_metrics["ducking_bypassed"]
                    else "Speaker detection and ducking checks passed."
                ),
                "report_data": {
                    "": [
                        "LUFS (in → out)",
                        "Peak (dBFS)",
                        "Speech coverage",
                        "Duration",
                    ],
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
            if master_metrics is not None:
                st.session_state["results"]["report_data"]["Master"] = [
                    f"PASS · {master_metrics['integrated_lufs']:.1f}",
                    f"{master_metrics['true_peak_db']:.1f} true peak",
                    f"Limiter >1 dB: {master_metrics['limited_over_1_pct']:.1f}%",
                    f"Fixed gain: {master_metrics['fixed_gain_db']:.1f} dB; "
                    "master compressor off",
                ]

        except Exception as e:
            st.error(f"Error: {e}")
            raise

elif not file_a or not file_b:
    st.info("Upload both audio files to get started.")

# --- Show results (persists across reruns) ---
if "results" in st.session_state:
    r = st.session_state["results"]
    st.success("Processing complete!")
    st.caption(r.get("validation_summary", "Quality checks completed."))

    st.markdown("### Quality Report")
    st.table(r["report_data"])

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            f"Download {r['name_a']}",
            r["wav_bytes_a"],
            file_name=r["name_a"],
            mime="audio/wav",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            f"Download {r['name_b']}",
            r["wav_bytes_b"],
            file_name=r["name_b"],
            mime="audio/wav",
            use_container_width=True,
        )

    if r.get("wav_bytes_master") is not None:
        st.download_button(
            f"Download {r['name_master']}",
            r["wav_bytes_master"],
            file_name=r["name_master"],
            mime="audio/wav",
            type="primary",
            use_container_width=True,
        )

# --- Footer ---
st.markdown("---")
st.markdown(
    "<small>Built by [Mikey Jarrell](https://mikeyjarrell.com). "
    "Uses [Silero VAD](https://github.com/snakers4/silero-vad) for voice "
    "activity detection.</small>",
    unsafe_allow_html=True,
)
