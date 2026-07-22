"""Tests for reusable editing, media, and concrete core behavior."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.io import wavfile

from ducking_core import DuckingCore, SourceAsset, SourceRole, ThemeAssets
from ducking_core.editing import apply_identical_edits, kept_sample_intervals
from ducking_core.media import assemble_themed_program, sha256_file


class SynchronizedEditingTests(unittest.TestCase):
    """Protect the shared source-time edit behavior."""

    def test_same_trim_and_cut_are_applied_to_both_tracks(self):
        from ducking_core import EditInterval

        host = np.arange(1_000, dtype=np.float32)
        guest = host + 10_000
        trim = EditInterval(100, 900)
        cuts = (EditInterval(400, 600),)

        edited_host, edited_guest = apply_identical_edits(
            host,
            guest,
            1_000,
            trim,
            cuts,
            crossfade_ms=10,
        )

        self.assertEqual(len(edited_host), 590)
        self.assertEqual(len(edited_host), len(edited_guest))
        np.testing.assert_allclose(edited_guest - edited_host, 10_000)

    def test_source_time_boundaries_round_once_for_both_tracks(self):
        from ducking_core import EditInterval

        intervals = kept_sample_intervals(
            44_100,
            44_100,
            EditInterval(100, 900),
            (EditInterval(333, 667),),
        )

        self.assertEqual(intervals, ((4_410, 14_685), (29_415, 39_690)))


class ThemeAssemblyTests(unittest.TestCase):
    """Keep the measured theme markers exact to the millisecond."""

    def test_theme_markers_align_with_confirmed_speech_anchors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intro = np.zeros((10_000, 2), dtype=np.float32)
            outro = np.zeros((20_000, 2), dtype=np.float32)
            intro[8_197] = 0.25
            outro[16_196] = 0.5
            sf.write(root / "intro.wav", intro, 1_000, subtype="FLOAT")
            sf.write(root / "outro.wav", outro, 1_000, subtype="FLOAT")
            theme = ThemeAssets(
                version="test",
                intro_path=root / "intro.wav",
                outro_path=root / "outro.wav",
            )
            speech = np.zeros(30_000, dtype=np.float32)
            speech[1_000] = 0.75
            speech[25_000] = 0.75

            program = assemble_themed_program(
                speech,
                1_000,
                theme,
                intro_speech_anchor_ms=1_000,
                outro_final_word_anchor_ms=25_000,
            )

            # The intro beat and opening speech sample meet at exactly 8,197 ms.
            self.assertAlmostEqual(float(program[8_197, 0]), 1.0, places=5)
            # The outro marker and confirmed final word meet at 32,197 ms.
            self.assertAlmostEqual(float(program[32_197, 0]), 1.25, places=5)


class ConcreteCoreTests(unittest.TestCase):
    """Exercise the public media-inspection boundary."""

    def test_inspect_verifies_checksum_and_media_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "host.wav"
            wavfile.write(path, 48_000, np.zeros(48_000, dtype=np.float32))
            source = SourceAsset(path, SourceRole.HOST, sha256_file(path))

            metadata = DuckingCore().inspect(source)

            self.assertEqual(metadata.sample_rate_hz, 48_000)
            self.assertEqual(metadata.channels, 1)
            self.assertEqual(metadata.duration_ms, 1_000)
            self.assertTrue(metadata.finite_samples)


if __name__ == "__main__":
    unittest.main()
