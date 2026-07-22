"""Tests for the versioned, application-independent Ducking core boundary."""

import json
import unittest
from pathlib import Path

from ducking_core import (
    CORE_API_VERSION,
    AudioMetadata,
    CoreErrorCode,
    CoreProcessingError,
    EditInterval,
    OutputAsset,
    OutputKind,
    OutputTargets,
    ProcessingSettings,
    QualityMeasurements,
    RenderRequest,
    RenderResult,
    SourceAsset,
    SourceRole,
    contract_to_dict,
)


class CoreContractTests(unittest.TestCase):
    """Keep the shared boundary deterministic, validated, and JSON-safe."""

    def setUp(self):
        self.host = SourceAsset(Path("host.wav"), SourceRole.HOST, "host-checksum")
        self.guest = SourceAsset(Path("guest.wav"), SourceRole.GUEST, "guest-checksum")

    def test_podcast_ready_defaults_match_validated_e_pipeline(self):
        settings = ProcessingSettings()

        self.assertEqual(settings.duck_db, -15.0)
        self.assertEqual(settings.fade_ms, 150)
        self.assertEqual(settings.dominance_db, 3.0)
        self.assertEqual(settings.stem_precompression_lufs, -22.0)
        self.assertEqual(settings.compressor_ratio, 2.0)
        self.assertEqual(settings.stem_target_lufs, -19.0)
        self.assertEqual(settings.stem_limiter_over_1_db_max_pct, 3.0)
        self.assertEqual(settings.master_target_lufs, -18.0)
        self.assertEqual(settings.true_peak_ceiling_db, -1.0)
        self.assertEqual(settings.master_limiter_over_1_db_max_pct, 1.0)
        self.assertEqual(settings.speaker_balance_max_db, 3.0)

    def test_edit_intervals_are_half_open_and_nonempty(self):
        interval = EditInterval(1_000, 2_000)

        self.assertEqual(interval.start_ms, 1_000)
        self.assertEqual(interval.end_ms, 2_000)
        with self.assertRaises(ValueError):
            EditInterval(2_000, 2_000)

    def test_render_request_rejects_reversed_roles(self):
        with self.assertRaisesRegex(ValueError, "host role"):
            RenderRequest(
                "request-1",
                SourceAsset(Path("a.wav"), SourceRole.GUEST, "a"),
                self.guest,
                OutputTargets(Path("output")),
            )

    def test_render_request_rejects_overlapping_cuts(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            RenderRequest(
                "request-1",
                self.host,
                self.guest,
                OutputTargets(Path("output")),
                cuts=(EditInterval(100, 500), EditInterval(400, 700)),
            )

    def test_render_request_rejects_adjacent_cuts(self):
        with self.assertRaisesRegex(ValueError, "adjacent"):
            RenderRequest(
                "request-1",
                self.host,
                self.guest,
                OutputTargets(Path("output")),
                cuts=(EditInterval(100, 500), EditInterval(500, 700)),
            )

    def test_render_request_keeps_cuts_inside_trim(self):
        with self.assertRaisesRegex(ValueError, "within the trim"):
            RenderRequest(
                "request-1",
                self.host,
                self.guest,
                OutputTargets(Path("output")),
                trim=EditInterval(1_000, 10_000),
                cuts=(EditInterval(500, 700),),
            )

    def test_render_request_requires_theme_for_speech_anchors(self):
        with self.assertRaisesRegex(ValueError, "require versioned theme"):
            RenderRequest(
                "request-1",
                self.host,
                self.guest,
                OutputTargets(Path("output")),
                intro_speech_anchor_ms=8_197,
            )

    def test_contracts_serialize_to_json_primitives(self):
        request = RenderRequest(
            "request-1",
            self.host,
            self.guest,
            OutputTargets(Path("output")),
            trim=EditInterval(1_000, 20_000),
            cuts=(EditInterval(5_000, 6_000),),
        )

        serialized = contract_to_dict(request)
        encoded = json.dumps(serialized, sort_keys=True)

        self.assertIn('"preset": "podcast_ready_e"', encoded)
        self.assertEqual(serialized["host"]["path"], "host.wav")
        self.assertEqual(serialized["cuts"][0], {"start_ms": 5_000, "end_ms": 6_000})

    def test_quality_passes_only_when_every_gate_passes(self):
        quality = self._quality({"duration_preserved": True, "true_peak_safe": True})
        failed = self._quality({"duration_preserved": True, "true_peak_safe": False})

        self.assertTrue(quality.checks_passed)
        self.assertFalse(failed.checks_passed)

    def test_result_rejects_a_different_api_version(self):
        with self.assertRaisesRegex(ValueError, "Unsupported core API version"):
            self._result(api_version="2.0.0")

    def test_result_and_error_are_json_safe(self):
        result = self._result()
        error = CoreProcessingError(
            CoreErrorCode.SOURCE_MISMATCH,
            "The synchronized sources have different durations.",
            details={"difference_ms": 25},
        )

        json.dumps(contract_to_dict(result))
        self.assertEqual(
            error.to_dict(),
            {
                "code": "source_mismatch",
                "message": "The synchronized sources have different durations.",
                "details": {"difference_ms": 25},
            },
        )

    def _quality(self, checks):
        return QualityMeasurements(
            integrated_lufs=-18.05,
            true_peak_db=-1.2,
            clipped_sample_count=0,
            host_speech_lufs=-17.0,
            guest_speech_lufs=-17.4,
            host_speech_coverage_pct=30.0,
            guest_speech_coverage_pct=55.0,
            limiter_over_1_db_pct=0.2,
            speaker_balance_db=0.4,
            duration_ms=1_500_000,
            checks=checks,
        )

    def _result(self, api_version=CORE_API_VERSION):
        metadata = AudioMetadata(48_000, 1, 72_000_000, 1_500_000, "float32", True)
        output = OutputAsset(
            OutputKind.SPEECH_MASTER,
            Path("output/master.wav"),
            "master-checksum",
            metadata,
        )
        return RenderResult(
            api_version,
            "ad883ce",
            "request-1",
            (metadata, metadata),
            (output,),
            self._quality({"all_gates": True}),
        )


if __name__ == "__main__":
    unittest.main()
