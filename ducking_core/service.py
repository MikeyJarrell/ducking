"""Concrete implementation of the versioned Ducking audio-core contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .contracts import (
    CORE_API_VERSION,
    AudioMetadata,
    CoreErrorCode,
    CoreProcessingError,
    CoreWarning,
    OutputAsset,
    OutputKind,
    ProcessingPreset,
    QualityMeasurements,
    RenderRequest,
    RenderResult,
    SourceAsset,
    TrimSuggestion,
    WarningCode,
)
from .editing import apply_identical_edits
from .engine import (
    apply_limiter,
    build_podcast_master,
    build_validated_ducking_envelopes,
    get_mono,
    get_speech_regions,
    load_vad_model,
    load_wav,
    measure_lufs,
    measure_lufs_speech_only,
    measure_true_peak,
    mix_to_mono,
    process_track,
    resample_to_16k,
    validate_ducking_stage,
    validate_mix_stage,
    validate_podcast_stem,
)
from .media import (
    assemble_themed_program,
    inspect_wav,
    sha256_file,
    write_derived_wav,
)


ENGINE_VERSION = "1.0.0"


class DuckingCore:
    """Run immutable two-track requests without application-specific concepts."""

    api_version = CORE_API_VERSION
    engine_version = ENGINE_VERSION

    def __init__(self) -> None:
        self._vad_model = None
        self._vad_helper = None

    def _voice_detector(self):
        if self._vad_model is None or self._vad_helper is None:
            self._vad_model, self._vad_helper = load_vad_model()
        return self._vad_model, self._vad_helper

    def inspect(self, source: SourceAsset) -> AudioMetadata:
        """Verify one immutable source and return decoded metadata."""
        try:
            if not source.path.is_file():
                raise CoreProcessingError(
                    CoreErrorCode.DECODE_FAILED,
                    "The source file could not be found.",
                )
            if source.path.suffix.lower() != ".wav":
                raise CoreProcessingError(
                    CoreErrorCode.UNSUPPORTED_FORMAT,
                    "The first core release accepts WAV sources only.",
                )
            if sha256_file(source.path) != source.sha256:
                raise CoreProcessingError(
                    CoreErrorCode.DECODE_FAILED,
                    "The source checksum does not match the immutable asset record.",
                )
            metadata = inspect_wav(source.path)
            if metadata.sample_count == 0:
                raise CoreProcessingError(
                    CoreErrorCode.EMPTY_SOURCE,
                    "The source recording is empty.",
                )
            if metadata.sample_rate_hz not in (44_100, 48_000):
                raise CoreProcessingError(
                    CoreErrorCode.UNSUPPORTED_FORMAT,
                    "The source sample rate must be 44.1 or 48 kHz.",
                )
            if metadata.channels != 1:
                raise CoreProcessingError(
                    CoreErrorCode.UNSUPPORTED_FORMAT,
                    "Each source must contain one isolated microphone channel.",
                )
            if not metadata.finite_samples:
                raise CoreProcessingError(
                    CoreErrorCode.DECODE_FAILED,
                    "The source contains non-finite audio samples.",
                )
            return metadata
        except CoreProcessingError:
            raise
        except (OSError, ValueError) as error:
            raise CoreProcessingError(
                CoreErrorCode.DECODE_FAILED,
                "The WAV source could not be decoded.",
            ) from error

    def _load_pair(
        self, host: SourceAsset, guest: SourceAsset
    ) -> tuple[
        AudioMetadata, AudioMetadata, int, np.ndarray, np.ndarray, np.dtype, np.dtype
    ]:
        host_metadata = self.inspect(host)
        guest_metadata = self.inspect(guest)
        if (
            host_metadata.sample_rate_hz != guest_metadata.sample_rate_hz
            or host_metadata.sample_count != guest_metadata.sample_count
        ):
            raise CoreProcessingError(
                CoreErrorCode.SOURCE_MISMATCH,
                "The two microphone recordings are not synchronized.",
            )
        sample_rate, host_audio, host_dtype = load_wav(host.path)
        _, guest_audio, guest_dtype = load_wav(guest.path)
        return (
            host_metadata,
            guest_metadata,
            sample_rate,
            host_audio,
            guest_audio,
            host_dtype,
            guest_dtype,
        )

    def suggest_trim(self, host: SourceAsset, guest: SourceAsset) -> TrimSuggestion:
        """Suggest speech-bounded trim points without applying them."""
        (
            host_metadata,
            _,
            sample_rate,
            host_audio,
            guest_audio,
            _,
            _,
        ) = self._load_pair(host, guest)
        model, helper = self._voice_detector()
        regions = []
        for audio in (host_audio, guest_audio):
            mono = get_mono(audio)
            regions.extend(
                get_speech_regions(
                    model,
                    helper,
                    resample_to_16k(mono, sample_rate),
                    threshold=0.5,
                )
            )
        if not regions:
            raise CoreProcessingError(
                CoreErrorCode.SPEAKER_DETECTION_FAILED,
                "No speech was detected, so trim points could not be suggested.",
            )
        start_ms = max(0, round(min(item["start"] for item in regions) / 16) - 500)
        end_ms = min(
            host_metadata.duration_ms,
            round(max(item["end"] for item in regions) / 16) + 500,
        )
        return TrimSuggestion(
            start_ms=start_ms,
            end_ms=end_ms,
            confidence=0.8,
            reason="First and last detected speech across both microphone tracks, with padding.",
        )

    @staticmethod
    def _engine_settings(
        request: RenderRequest, *, gain_db: float
    ) -> dict[str, object]:
        settings = request.settings
        podcast_ready = settings.preset is ProcessingPreset.PODCAST_READY_E
        return {
            "gain_enabled": gain_db != 0,
            "gain_db": gain_db,
            "comp_enabled": podcast_ready,
            "comp_threshold": settings.compressor_threshold_db,
            "comp_ratio": settings.compressor_ratio,
            "comp_attack": settings.compressor_attack_ms,
            "comp_release": settings.compressor_release_ms,
            "lufs_enabled": podcast_ready,
            "lufs_target": settings.master_target_lufs,
            "limiter_enabled": podcast_ready,
            "limiter_ceiling": settings.true_peak_ceiling_db,
            "master_enabled": podcast_ready,
        }

    @staticmethod
    def _output_asset(kind: OutputKind, path: Path) -> OutputAsset:
        return OutputAsset(
            kind=kind,
            path=path,
            sha256=sha256_file(path),
            metadata=inspect_wav(path),
        )

    def render(self, request: RenderRequest) -> RenderResult:
        """Apply edits and the tested Ducking chain, then write derived WAVs."""
        try:
            (
                host_metadata,
                guest_metadata,
                sample_rate,
                host_audio,
                guest_audio,
                host_dtype,
                guest_dtype,
            ) = self._load_pair(request.host, request.guest)
            host_audio, guest_audio = apply_identical_edits(
                host_audio,
                guest_audio,
                sample_rate,
                request.trim,
                request.cuts,
                crossfade_ms=request.settings.edit_crossfade_ms,
            )

            model, helper = self._voice_detector()
            host_regions = get_speech_regions(
                model,
                helper,
                resample_to_16k(get_mono(host_audio), sample_rate),
                threshold=request.settings.vad_threshold,
            )
            guest_regions = get_speech_regions(
                model,
                helper,
                resample_to_16k(get_mono(guest_audio), sample_rate),
                threshold=request.settings.vad_threshold,
            )
            (
                host_envelope,
                guest_envelope,
                detection,
            ) = build_validated_ducking_envelopes(
                get_mono(host_audio),
                get_mono(guest_audio),
                sample_rate,
                host_regions,
                guest_regions,
                fade_ms=request.settings.fade_ms,
                duck_db=request.settings.duck_db,
                dominance_db=request.settings.dominance_db,
                require_two_speakers=(
                    request.settings.preset is ProcessingPreset.PODCAST_READY_E
                ),
            )
            ducking_gate = validate_ducking_stage(
                host_audio,
                guest_audio,
                host_envelope,
                guest_envelope,
                sample_rate,
                request.settings.duck_db,
            )
            if not ducking_gate["passed"] and not detection["ducking_bypassed"]:
                raise CoreProcessingError(
                    CoreErrorCode.DUCKING_GATE_FAILED,
                    "The inactive-microphone reduction failed its quality gate.",
                )

            host_result, host_limiter = process_track(
                host_audio,
                sample_rate,
                host_envelope,
                self._engine_settings(request, gain_db=request.host_gain_db),
            )
            guest_result, guest_limiter = process_track(
                guest_audio,
                sample_rate,
                guest_envelope,
                self._engine_settings(request, gain_db=request.guest_gain_db),
            )

            stem_checks = {
                "host": validate_podcast_stem(
                    host_audio, host_result, sample_rate, host_envelope, host_limiter
                ),
                "guest": validate_podcast_stem(
                    guest_audio,
                    guest_result,
                    sample_rate,
                    guest_envelope,
                    guest_limiter,
                ),
            }
            if request.settings.preset is ProcessingPreset.PODCAST_READY_E and not all(
                all(checks.values()) for checks in stem_checks.values()
            ):
                raise CoreProcessingError(
                    CoreErrorCode.STEM_GATE_FAILED,
                    "A cleaned microphone stem failed its quality gate.",
                )

            mix_gate = validate_mix_stage(
                host_result,
                guest_result,
                host_envelope,
                guest_envelope,
                sample_rate,
            )
            if (
                request.settings.preset is ProcessingPreset.PODCAST_READY_E
                and not mix_gate["passed"]
            ):
                raise CoreProcessingError(
                    CoreErrorCode.MIX_GATE_FAILED,
                    "Both speakers did not survive the unmastered sum.",
                )

            if request.settings.preset is ProcessingPreset.PODCAST_READY_E:
                master, master_metrics = build_podcast_master(
                    host_result,
                    guest_result,
                    sample_rate,
                    target_lufs=request.settings.master_target_lufs,
                    true_peak_ceiling_db=request.settings.true_peak_ceiling_db,
                    speaker_a_mask=mix_gate["speaker_a_mask"],
                    speaker_b_mask=mix_gate["speaker_b_mask"],
                )
            else:
                master = mix_to_mono(host_result) + mix_to_mono(guest_result)
                master, _ = apply_limiter(
                    master,
                    sample_rate,
                    ceiling_db=request.settings.true_peak_ceiling_db,
                    return_gain=True,
                )
                master_metrics = {
                    "integrated_lufs": measure_lufs(master, sample_rate),
                    "true_peak_db": 20 * np.log10(measure_true_peak(master) + 1e-12),
                    "limited_over_1_pct": 0.0,
                    "speaker_balance_db": 0.0,
                    "checks": {"finite_audio": bool(np.all(np.isfinite(master)))},
                }

            program = None
            if request.output.write_program_master:
                if request.theme is None:
                    raise CoreProcessingError(
                        CoreErrorCode.INVALID_EDIT_PLAN,
                        "A program master requires versioned theme assets.",
                    )
                if (
                    request.intro_speech_anchor_ms is None
                    or request.outro_final_word_anchor_ms is None
                ):
                    raise CoreProcessingError(
                        CoreErrorCode.INVALID_EDIT_PLAN,
                        "A themed program requires both confirmed speech anchors.",
                    )
                program = assemble_themed_program(
                    master,
                    sample_rate,
                    request.theme,
                    request.intro_speech_anchor_ms,
                    request.outro_final_word_anchor_ms,
                )

            request.output.directory.mkdir(parents=True, exist_ok=True)
            outputs: list[OutputAsset] = []
            if request.output.write_cleaned_stems:
                host_path = request.output.directory / "host_cleaned.wav"
                guest_path = request.output.directory / "guest_cleaned.wav"
                write_derived_wav(host_path, sample_rate, host_result, host_dtype)
                write_derived_wav(guest_path, sample_rate, guest_result, guest_dtype)
                outputs.extend(
                    (
                        self._output_asset(OutputKind.CLEANED_HOST, host_path),
                        self._output_asset(OutputKind.CLEANED_GUEST, guest_path),
                    )
                )
            if request.output.write_speech_master:
                master_path = request.output.directory / "speech_master.wav"
                write_derived_wav(master_path, sample_rate, master, np.dtype("float32"))
                outputs.append(
                    self._output_asset(OutputKind.SPEECH_MASTER, master_path)
                )
            final_audio = master
            if program is not None:
                program_path = request.output.directory / "program_master.wav"
                write_derived_wav(
                    program_path, sample_rate, program, np.dtype("float32")
                )
                outputs.append(
                    self._output_asset(OutputKind.PROGRAM_MASTER, program_path)
                )
                final_audio = program

            warnings = ()
            if detection["ducking_bypassed"]:
                warnings = (
                    CoreWarning(
                        WarningCode.DUCKING_BYPASSED,
                        "Speaker separation was uncertain, so ducking was bypassed.",
                    ),
                )
            checks = {
                f"ducking_{name}": bool(value)
                for name, value in ducking_gate["checks"].items()
            }
            checks.update(
                {
                    f"mix_{name}": bool(value)
                    for name, value in mix_gate["checks"].items()
                }
            )
            checks.update(
                {
                    f"master_{name}": bool(value)
                    for name, value in master_metrics["checks"].items()
                }
            )
            quality = QualityMeasurements(
                integrated_lufs=float(measure_lufs(final_audio, sample_rate)),
                true_peak_db=float(
                    20 * np.log10(measure_true_peak(final_audio) + 1e-12)
                ),
                clipped_sample_count=int(np.sum(np.abs(final_audio) >= 1.0)),
                host_speech_lufs=float(
                    measure_lufs_speech_only(host_result, sample_rate, host_envelope)
                ),
                guest_speech_lufs=float(
                    measure_lufs_speech_only(guest_result, sample_rate, guest_envelope)
                ),
                host_speech_coverage_pct=float(np.mean(host_envelope > 0.5) * 100),
                guest_speech_coverage_pct=float(np.mean(guest_envelope > 0.5) * 100),
                limiter_over_1_db_pct=float(master_metrics["limited_over_1_pct"]),
                speaker_balance_db=float(master_metrics["speaker_balance_db"]),
                duration_ms=round(len(final_audio) * 1000 / sample_rate),
                checks=checks,
            )
            return RenderResult(
                api_version=self.api_version,
                engine_version=self.engine_version,
                idempotency_key=request.idempotency_key,
                sources=(host_metadata, guest_metadata),
                outputs=tuple(outputs),
                quality=quality,
                warnings=warnings,
            )
        except CoreProcessingError:
            raise
        except ValueError as error:
            raise CoreProcessingError(
                CoreErrorCode.OUTPUT_FAILED,
                "The audio render could not be completed safely.",
            ) from error
        except OSError as error:
            raise CoreProcessingError(
                CoreErrorCode.OUTPUT_FAILED,
                "A derived audio file could not be written.",
            ) from error
