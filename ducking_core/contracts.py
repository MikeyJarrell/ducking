"""Stable request, result, metadata, warning, and error contracts.

The types in this module deliberately contain no web-application, database, or
user concepts. Ducking's desktop app, web app, and Backstory can therefore use
the same audio engine without sharing interface or persistence code.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


CORE_API_VERSION = "1.0.0"
ENGINE_BASELINE_COMMIT = "ad883ce550b110ce3f414fc5e89fbf673e8a8e8a"


class SourceRole(StrEnum):
    """The two synchronized microphone roles accepted by the engine."""

    HOST = "host"
    GUEST = "guest"


class ProcessingPreset(StrEnum):
    """Named processing behavior whose settings are recorded on every render."""

    NATURAL_CLEANUP = "natural_cleanup"
    PODCAST_READY_E = "podcast_ready_e"


class OutputKind(StrEnum):
    """Files that the shared engine may produce."""

    CLEANED_HOST = "cleaned_host"
    CLEANED_GUEST = "cleaned_guest"
    SPEECH_MASTER = "speech_master"


class WarningCode(StrEnum):
    """Non-fatal conditions a caller must show or record."""

    DUCKING_BYPASSED = "ducking_bypassed"
    INPUT_RESAMPLED = "input_resampled"
    OUTPUT_TARGET_CONSTRAINED = "output_target_constrained"


class CoreErrorCode(StrEnum):
    """Stable failure categories safe to persist outside restricted logs."""

    EMPTY_SOURCE = "empty_source"
    DECODE_FAILED = "decode_failed"
    UNSUPPORTED_FORMAT = "unsupported_format"
    SOURCE_MISMATCH = "source_mismatch"
    INVALID_EDIT_PLAN = "invalid_edit_plan"
    SPEAKER_DETECTION_FAILED = "speaker_detection_failed"
    DUCKING_GATE_FAILED = "ducking_gate_failed"
    STEM_GATE_FAILED = "stem_gate_failed"
    MIX_GATE_FAILED = "mix_gate_failed"
    MASTER_GATE_FAILED = "master_gate_failed"
    OUTPUT_FAILED = "output_failed"


@dataclass(frozen=True, slots=True)
class EditInterval:
    """One half-open source-time interval: start included, end excluded."""

    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.start_ms < 0:
            raise ValueError("Edit intervals cannot start before zero.")
        if self.end_ms <= self.start_ms:
            raise ValueError("Edit intervals must end after they start.")


@dataclass(frozen=True, slots=True)
class SourceAsset:
    """An immutable local source supplied to the audio worker."""

    path: Path
    role: SourceRole
    sha256: str

    def __post_init__(self) -> None:
        if not self.sha256:
            raise ValueError("A source checksum is required.")


@dataclass(frozen=True, slots=True)
class ThemeAssets:
    """Optional versioned music assets and their fixed musical markers."""

    version: str
    intro_path: Path
    outro_path: Path
    intro_speech_marker_ms: int = 8_197
    outro_final_word_marker_ms: int = 16_196
    music_gain_db: float = 0.0

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("A theme version is required.")
        if self.intro_speech_marker_ms < 0 or self.outro_final_word_marker_ms < 0:
            raise ValueError("Theme markers cannot be negative.")


@dataclass(frozen=True, slots=True)
class ProcessingSettings:
    """Complete settings for the listening-selected Podcast-ready E pipeline."""

    preset: ProcessingPreset = ProcessingPreset.PODCAST_READY_E
    vad_threshold: float = 0.5
    duck_db: float = -15.0
    fade_ms: int = 150
    dominance_db: float = 3.0
    highpass_hz: float = 65.0
    stem_precompression_lufs: float = -22.0
    compressor_threshold_db: float = -24.0
    compressor_ratio: float = 2.0
    compressor_attack_ms: int = 10
    compressor_release_ms: int = 150
    stem_target_lufs: float = -19.0
    stem_peak_ceiling_db: float = -3.0
    stem_loudness_tolerance_lu: float = 2.0
    stem_limiter_over_1_db_max_pct: float = 3.0
    master_target_lufs: float = -18.0
    true_peak_ceiling_db: float = -1.0
    master_loudness_tolerance_lu: float = 1.0
    master_limiter_over_1_db_max_pct: float = 1.0
    speaker_balance_max_db: float = 3.0

    def __post_init__(self) -> None:
        if not 0.0 < self.vad_threshold < 1.0:
            raise ValueError("The voice-detection threshold must be between 0 and 1.")
        if self.fade_ms < 0:
            raise ValueError("The fade duration cannot be negative.")
        if self.compressor_ratio < 1.0:
            raise ValueError("The compressor ratio cannot be below 1:1.")
        if self.compressor_attack_ms <= 0 or self.compressor_release_ms <= 0:
            raise ValueError("Compressor timing must be positive.")


@dataclass(frozen=True, slots=True)
class OutputTargets:
    """Output location and file behavior requested from the worker."""

    directory: Path
    write_cleaned_stems: bool = True
    write_speech_master: bool = True

    def __post_init__(self) -> None:
        if not self.write_cleaned_stems and not self.write_speech_master:
            raise ValueError("At least one output must be requested.")


@dataclass(frozen=True, slots=True)
class RenderRequest:
    """A complete, replayable request for one synchronized two-track render."""

    idempotency_key: str
    host: SourceAsset
    guest: SourceAsset
    output: OutputTargets
    settings: ProcessingSettings = ProcessingSettings()
    trim: EditInterval | None = None
    cuts: tuple[EditInterval, ...] = ()
    host_gain_db: float = 0.0
    guest_gain_db: float = 0.0
    theme: ThemeAssets | None = None
    intro_speech_anchor_ms: int | None = None
    outro_final_word_anchor_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.idempotency_key:
            raise ValueError("An idempotency key is required.")
        if self.host.role is not SourceRole.HOST:
            raise ValueError("The host source must have the host role.")
        if self.guest.role is not SourceRole.GUEST:
            raise ValueError("The guest source must have the guest role.")
        if self.host.path == self.guest.path:
            raise ValueError("The host and guest sources must be different files.")
        ordered_cuts = tuple(sorted(self.cuts, key=lambda item: item.start_ms))
        if ordered_cuts != self.cuts:
            raise ValueError("Cuts must be ordered by source start time.")
        for previous, current in zip(self.cuts, self.cuts[1:]):
            if current.start_ms <= previous.end_ms:
                raise ValueError("Cuts cannot overlap or be adjacent.")
        if self.trim is not None:
            for cut in self.cuts:
                if cut.start_ms < self.trim.start_ms or cut.end_ms > self.trim.end_ms:
                    raise ValueError("Cuts must fall within the trim interval.")
        if self.theme is None and (
            self.intro_speech_anchor_ms is not None
            or self.outro_final_word_anchor_ms is not None
        ):
            raise ValueError("Speech anchors require versioned theme assets.")


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    """Media facts measured from one decoded audio asset."""

    sample_rate_hz: int
    channels: int
    sample_count: int
    duration_ms: int
    sample_format: str
    finite_samples: bool


@dataclass(frozen=True, slots=True)
class TrimSuggestion:
    """Suggested source-time boundaries that still require user confirmation."""

    start_ms: int
    end_ms: int
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        EditInterval(self.start_ms, self.end_ms)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Trim confidence must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class CoreWarning:
    """A structured warning with a safe message for the calling application."""

    code: WarningCode
    message: str
    details: dict[str, str | int | float | bool] | None = None


@dataclass(frozen=True, slots=True)
class OutputAsset:
    """One derived file returned by the engine."""

    kind: OutputKind
    path: Path
    sha256: str
    metadata: AudioMetadata


@dataclass(frozen=True, slots=True)
class QualityMeasurements:
    """Measurements and acceptance-gate results for a completed render."""

    integrated_lufs: float
    true_peak_db: float
    clipped_sample_count: int
    host_speech_lufs: float
    guest_speech_lufs: float
    host_speech_coverage_pct: float
    guest_speech_coverage_pct: float
    limiter_over_1_db_pct: float
    speaker_balance_db: float
    duration_ms: int
    checks: dict[str, bool]

    @property
    def checks_passed(self) -> bool:
        """Return true only when every blocking gate passed."""

        return bool(self.checks) and all(self.checks.values())


@dataclass(frozen=True, slots=True)
class RenderResult:
    """Successful engine output with enough provenance to reproduce it."""

    api_version: str
    engine_version: str
    idempotency_key: str
    sources: tuple[AudioMetadata, AudioMetadata]
    outputs: tuple[OutputAsset, ...]
    quality: QualityMeasurements
    warnings: tuple[CoreWarning, ...] = ()

    def __post_init__(self) -> None:
        if self.api_version != CORE_API_VERSION:
            raise ValueError(
                f"Unsupported core API version {self.api_version}; "
                f"expected {CORE_API_VERSION}."
            )
        if not self.outputs:
            raise ValueError("A successful render must contain at least one output.")


class CoreProcessingError(RuntimeError):
    """A stable error code and safe message for a failed core operation."""

    def __init__(
        self,
        code: CoreErrorCode,
        message: str,
        *,
        details: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation without a traceback or local cause."""

        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }


class AudioCore(Protocol):
    """Behavior implemented by the reusable Ducking package."""

    api_version: str
    engine_version: str

    def inspect(self, source: SourceAsset) -> AudioMetadata:
        ...

    def suggest_trim(self, host: SourceAsset, guest: SourceAsset) -> TrimSuggestion:
        ...

    def render(self, request: RenderRequest) -> RenderResult:
        ...


def contract_to_dict(value: Any) -> Any:
    """Convert nested contract objects into JSON-safe Python primitives."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: contract_to_dict(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): contract_to_dict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [contract_to_dict(item) for item in value]
    return value
