"""Regression tests for the Ducking desktop app."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import ducking_app


class VadLoadingTests(unittest.TestCase):
    """Ensure voice detection works without downloading from Torch Hub."""

    @patch("torch.hub.load")
    def test_load_vad_model_uses_installed_package(self, torch_hub_load):
        model = Mock()
        package_load = Mock(return_value=model)
        timestamp_helper = Mock()

        # Supply a stand-in package so this regression test also runs in Python
        # environments where the optional desktop dependency is not installed.
        silero_package = types.ModuleType("silero_vad")
        silero_package.load_silero_vad = package_load
        silero_package.get_speech_timestamps = timestamp_helper

        with patch.dict(sys.modules, {"silero_vad": silero_package}):
            loaded_model, loaded_helper = ducking_app.load_vad_model()

        self.assertIs(loaded_model, model)
        self.assertIs(loaded_helper, timestamp_helper)
        package_load.assert_called_once_with()
        torch_hub_load.assert_not_called()

    def test_get_speech_regions_uses_returned_helper(self):
        model = Mock()
        timestamp_helper = Mock(return_value=[{"start": 10, "end": 20}])
        audio = np.zeros(512, dtype=np.float32)

        regions = ducking_app.get_speech_regions(
            model, timestamp_helper, audio, threshold=0.6
        )

        self.assertEqual(regions, [{"start": 10, "end": 20}])
        timestamp_helper.assert_called_once()
        call_args, call_kwargs = timestamp_helper.call_args
        self.assertIs(call_args[1], model)
        self.assertEqual(call_kwargs["threshold"], 0.6)
        self.assertEqual(call_kwargs["sampling_rate"], 16000)
        model.reset_states.assert_called_once_with()


class DuckingEnvelopeTests(unittest.TestCase):
    """Protect speaker detection from gain mismatch and rapid switching."""

    def test_automatic_calibration_preserves_both_speakers(self):
        sr = 16000
        seconds = 4
        samples = sr * seconds
        time = np.arange(samples) / sr
        tone = np.sin(2 * np.pi * 220 * time).astype(np.float32)

        # Track B was recorded about 10 dB hotter. A fixed zero-centered
        # comparison would fail to recognize A as the close microphone.
        midpoint = samples // 2
        audio_a = np.empty(samples, dtype=np.float32)
        audio_b = np.empty(samples, dtype=np.float32)
        audio_a[:midpoint] = 0.20 * tone[:midpoint]
        audio_b[:midpoint] = 0.05 * 3.16 * tone[:midpoint]
        audio_a[midpoint:] = 0.02 * tone[midpoint:]
        audio_b[midpoint:] = 0.20 * 3.16 * tone[midpoint:]
        speech = [{"start": 0, "end": seconds * ducking_app.VAD_SAMPLE_RATE}]

        gain_a, gain_b, diagnostics = ducking_app.build_cross_ducking_envelopes(
            audio_a,
            audio_b,
            sr,
            speech,
            speech,
            fade_ms=20,
            duck_db=-12,
            dominance_db=3,
            return_diagnostics=True,
        )

        margin = int(0.2 * sr)
        self.assertGreater(np.median(gain_a[margin : midpoint - margin]), 0.95)
        self.assertLess(np.median(gain_b[margin : midpoint - margin]), 0.35)
        self.assertLess(np.median(gain_a[midpoint + margin : -margin]), 0.35)
        self.assertGreater(np.median(gain_b[midpoint + margin : -margin]), 0.95)
        self.assertIsNotNone(diagnostics["ratio_clusters_db"])
        self.assertLess(diagnostics["ratio_center_db"], -5)

    def test_no_detected_speech_leaves_both_tracks_open(self):
        audio = np.ones(3200, dtype=np.float32) * 0.1
        gain_a, gain_b = ducking_app.build_cross_ducking_envelopes(
            audio, audio * 0.5, 16000, [], [], fade_ms=20
        )
        np.testing.assert_allclose(gain_a, 1.0)
        np.testing.assert_allclose(gain_b, 1.0)

    def test_uncertain_detection_bypasses_natural_cleanup(self):
        sr = 16000
        audio = np.ones(sr * 2, dtype=np.float32) * 0.1

        gain_a, gain_b, diagnostics = ducking_app.build_validated_ducking_envelopes(
            audio,
            audio,
            sr,
            [],
            [],
            dominance_db=3,
            require_two_speakers=False,
        )

        self.assertTrue(diagnostics["ducking_bypassed"])
        self.assertEqual(diagnostics["dominance_attempts_db"], [3.0, 1.5])
        np.testing.assert_allclose(gain_a, 1.0)
        np.testing.assert_allclose(gain_b, 1.0)

    def test_uncertain_detection_blocks_podcast_master(self):
        sr = 16000
        audio = np.ones(sr * 2, dtype=np.float32) * 0.1

        with self.assertRaisesRegex(ValueError, "No podcast-ready master"):
            ducking_app.build_validated_ducking_envelopes(
                audio,
                audio,
                sr,
                [],
                [],
                require_two_speakers=True,
            )


class ProcessingSafetyTests(unittest.TestCase):
    """Ensure the default cleanup cannot make an input louder or clip it."""

    def test_natural_cleanup_only_attenuates(self):
        sr = 16000
        audio = np.linspace(-0.5, 0.5, sr, dtype=np.float32)
        envelope = np.ones(sr, dtype=np.float32)
        envelope[sr // 2 :] = 10 ** (-12 / 20)
        settings = {
            "gain_enabled": False,
            "gain_db": 0,
            "lufs_enabled": False,
            "lufs_target": -19,
            "comp_enabled": False,
            "comp_threshold": -24,
            "comp_ratio": 2.5,
            "comp_attack": 10,
            "comp_release": 150,
            "limiter_enabled": False,
            "limiter_ceiling": -1,
        }

        with tempfile.TemporaryDirectory() as output_dir:
            _, data = ducking_app.process_track_audio(
                audio,
                sr,
                np.dtype("float32"),
                envelope,
                "test.wav",
                output_dir,
                settings,
                lambda _value: None,
                lambda _text: None,
            )
            self.assertTrue(Path(output_dir, "test_processed.wav").exists())

        self.assertLessEqual(
            np.max(np.abs(data["output_audio"])), np.max(np.abs(audio))
        )
        np.testing.assert_allclose(data["output_audio"][: sr // 2], audio[: sr // 2])

    def test_peak_aware_normalization_keeps_routine_peaks_below_ceiling(self):
        sr = 16000
        time = np.arange(sr * 2) / sr
        audio = (0.01 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
        envelope = np.ones(len(audio), dtype=np.float32)

        result = ducking_app.apply_lufs_normalization(
            audio,
            sr,
            target_lufs=-10,
            envelope=envelope,
            ceiling_db=-1,
            peak_percentile=99.9,
        )

        routine_peak = np.percentile(np.abs(result), 99.9)
        self.assertLessEqual(20 * np.log10(routine_peak + 1e-12), -1 + 1e-5)

    def test_limiter_never_smooths_a_peak_above_its_ceiling(self):
        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)
        audio[sr // 2] = 1.0

        limited = ducking_app.apply_limiter(audio, sr, ceiling_db=-3, release_ms=15)

        peak_db = 20 * np.log10(np.max(np.abs(limited)) + 1e-12)
        self.assertLessEqual(peak_db, -3 + 1e-5)

    def test_podcast_stem_is_safe_for_ordinary_playback(self):
        sr = 16000
        time = np.arange(sr * 2) / sr
        audio = (0.003 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
        audio[sr] = 0.5
        envelope = np.ones(len(audio), dtype=np.float32)
        settings = {
            "gain_enabled": False,
            "gain_db": 0,
            "lufs_enabled": True,
            "lufs_target": -18,
            "comp_enabled": True,
            "comp_threshold": -24,
            "comp_ratio": 2.0,
            "comp_attack": 10,
            "comp_release": 150,
            "limiter_enabled": True,
            "limiter_ceiling": -1,
            "master_enabled": True,
        }

        with tempfile.TemporaryDirectory() as output_dir:
            _, data = ducking_app.process_track_audio(
                audio,
                sr,
                np.dtype("float32"),
                envelope,
                "test.wav",
                output_dir,
                settings,
                lambda _value: None,
                lambda _text: None,
            )

        peak_db = 20 * np.log10(np.max(np.abs(data["output_audio"])) + 1e-12)
        self.assertLessEqual(peak_db, -3 + 1e-5)

    def test_ducking_gate_detects_attenuation(self):
        sr = 16000
        audio_a = np.ones(sr * 4, dtype=np.float32) * 0.1
        audio_b = np.ones(sr * 4, dtype=np.float32) * 0.08
        envelope_a = np.ones(sr * 4, dtype=np.float32)
        envelope_b = np.ones(sr * 4, dtype=np.float32)
        envelope_a[: sr * 2] = 10 ** (-12 / 20)
        envelope_b[sr * 2 :] = 10 ** (-12 / 20)

        result = ducking_app.validate_ducking_stage(
            audio_a, audio_b, envelope_a, envelope_b, sr, duck_db=-12
        )

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["speaker_a_attenuation_db"], -12, delta=0.1)
        self.assertAlmostEqual(result["speaker_b_attenuation_db"], -12, delta=0.1)

    def test_mix_gate_detects_a_missing_speaker(self):
        sr = 16000
        track_a = np.ones(sr * 4, dtype=np.float32) * 0.05
        track_b = np.zeros(sr * 4, dtype=np.float32)
        envelope_a = np.ones(sr * 4, dtype=np.float32)
        envelope_b = np.ones(sr * 4, dtype=np.float32)
        envelope_b[: sr * 2] = 0.1
        envelope_a[sr * 2 :] = 0.1

        result = ducking_app.validate_mix_stage(
            track_a, track_b, envelope_a, envelope_b, sr
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["speaker_b_audible"])

    def test_gated_loudness_ignores_appended_silence(self):
        sr = 16000
        time = np.arange(sr * 2) / sr
        speech = (0.05 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
        with_silence = np.concatenate([speech, np.zeros(sr * 2, dtype=np.float32)])

        speech_lufs = ducking_app.measure_lufs(speech, sr)
        silence_lufs = ducking_app.measure_lufs(with_silence, sr)

        self.assertAlmostEqual(speech_lufs, silence_lufs, delta=0.5)

    def test_speech_highpass_removes_rumble(self):
        sr = 16000
        time = np.arange(sr * 2) / sr
        rumble = np.sin(2 * np.pi * 20 * time)
        voice = np.sin(2 * np.pi * 220 * time)

        filtered_rumble = ducking_app.apply_speech_highpass(rumble, sr)
        filtered_voice = ducking_app.apply_speech_highpass(voice, sr)

        rumble_ratio = np.sqrt(np.mean(filtered_rumble**2)) / np.sqrt(
            np.mean(rumble**2)
        )
        voice_ratio = np.sqrt(np.mean(filtered_voice**2)) / np.sqrt(
            np.mean(voice**2)
        )
        self.assertLess(rumble_ratio, 0.15)
        self.assertGreater(voice_ratio, 0.9)

    def test_master_hits_program_loudness_and_true_peak_targets(self):
        sr = 16000
        time = np.arange(sr * 4) / sr
        track_a = np.zeros(len(time), dtype=np.float32)
        track_b = np.zeros(len(time), dtype=np.float32)
        track_a[: sr * 2] = 0.03 * np.sin(2 * np.pi * 180 * time[: sr * 2])
        track_b[sr * 2 :] = 0.03 * np.sin(2 * np.pi * 240 * time[sr * 2 :])

        a_mask = np.zeros(len(time), dtype=bool)
        b_mask = np.zeros(len(time), dtype=bool)
        a_mask[: sr * 2] = True
        b_mask[sr * 2 :] = True
        master, diagnostics = ducking_app.build_podcast_master(
            track_a,
            track_b,
            sr,
            target_lufs=-18,
            true_peak_ceiling_db=-1,
            speaker_a_mask=a_mask,
            speaker_b_mask=b_mask,
        )

        self.assertLessEqual(diagnostics["true_peak_db"], -1 + 1e-5)
        self.assertAlmostEqual(diagnostics["integrated_lufs"], -18, delta=1.0)
        self.assertTrue(diagnostics["checks_passed"])
        self.assertTrue(all(diagnostics["checks"].values()))
        self.assertIsNone(diagnostics["master_compressor_ratio"])
        self.assertEqual(master.dtype, np.float32)

    def test_master_rejects_unprotected_high_crest_peaks(self):
        sr = 16000
        time = np.arange(sr * 6) / sr
        track_a = np.zeros(len(time), dtype=np.float32)
        track_b = np.zeros(len(time), dtype=np.float32)
        track_a[: sr * 3] = 0.003 * np.sin(2 * np.pi * 180 * time[: sr * 3])
        track_b[sr * 3 :] = 0.003 * np.sin(2 * np.pi * 240 * time[sr * 3 :])
        track_a[sr] = 0.5
        track_b[sr * 4] = -0.5
        a_mask = np.arange(len(time)) < sr * 3
        b_mask = ~a_mask

        with self.assertRaisesRegex(ValueError, "loudness_on_target"):
            ducking_app.build_podcast_master(
                track_a,
                track_b,
                sr,
                target_lufs=-18,
                true_peak_ceiling_db=-1,
                speaker_a_mask=a_mask,
                speaker_b_mask=b_mask,
            )

    @patch("ducking_app.apply_compressor")
    def test_master_never_calls_bus_compressor(self, compressor):
        sr = 16000
        time = np.arange(sr * 4) / sr
        track_a = (0.02 * np.sin(2 * np.pi * 180 * time)).astype(np.float32)
        track_b = (0.01 * np.sin(2 * np.pi * 240 * time)).astype(np.float32)

        _, diagnostics = ducking_app.build_podcast_master(
            track_a,
            track_b,
            sr,
            target_lufs=-18,
            true_peak_ceiling_db=-1,
        )

        compressor.assert_not_called()
        self.assertIsNone(diagnostics["master_compressor_ratio"])
        self.assertEqual(diagnostics["fixed_gain_db"], -18 - diagnostics["premix_lufs"])

    def test_master_rejects_unbalanced_speakers(self):
        sr = 16000
        time = np.arange(sr * 4) / sr
        track_a = np.zeros(len(time), dtype=np.float32)
        track_b = np.zeros(len(time), dtype=np.float32)
        track_a[: sr * 2] = 0.04 * np.sin(2 * np.pi * 180 * time[: sr * 2])
        track_b[sr * 2 :] = 0.01 * np.sin(2 * np.pi * 240 * time[sr * 2 :])
        a_mask = np.arange(len(time)) < sr * 2
        b_mask = ~a_mask

        with self.assertRaisesRegex(ValueError, "both_speakers_audible"):
            ducking_app.build_podcast_master(
                track_a,
                track_b,
                sr,
                target_lufs=-18,
                true_peak_ceiling_db=-1,
                speaker_a_mask=a_mask,
                speaker_b_mask=b_mask,
            )

    @patch("ducking_app._master_candidate")
    def test_master_rejects_sustained_final_limiting(self, candidate):
        audio = np.ones(16000, dtype=np.float32) * 0.01
        candidate.return_value = (
            audio,
            {
                "premix_lufs": -19.0,
                "fixed_gain_db": 1.0,
                "integrated_lufs": -18.0,
                "true_peak_db": -2.0,
                "limited_over_1_pct": 1.01,
                "plr_db": 16.0,
                "master_compressor_ratio": None,
            },
        )

        with self.assertRaisesRegex(ValueError, "limiting_gentle"):
            ducking_app.build_podcast_master(
                audio,
                audio,
                16000,
                target_lufs=-18,
                true_peak_ceiling_db=-1,
            )


if __name__ == "__main__":
    unittest.main()
