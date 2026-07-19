"""Regression tests for the Ducking desktop app."""

import sys
import types
import unittest
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


if __name__ == "__main__":
    unittest.main()
