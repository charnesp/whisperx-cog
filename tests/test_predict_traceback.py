"""A failed prediction must leave its real cause in the server log.

Measured need: a KeyError raised deep in the model path reached the client as
"Prediction failed: KeyError: 'qwen3-asr-1.7b'", with no traceback anywhere in
the container log. The failure then has to be diagnosed blind, which cost an
afternoon. The client message stays sanitised, the server log keeps the trace.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _predict_stub import install  # noqa: E402

install()

import predict as predict_module  # noqa: E402


class TracebackIsLogged(unittest.TestCase):
    def setUp(self):
        self.predictor = predict_module.Predictor.__new__(predict_module.Predictor)

    def test_the_original_traceback_reaches_the_log(self):
        def boom(self, **kwargs):
            raise KeyError("qwen3-asr-1.7b")

        with mock.patch.object(predict_module.Predictor, "_run_predict", boom):
            with self.assertLogs("predict", level="ERROR") as captured:
                with self.assertRaises(RuntimeError):
                    self.predictor.predict(audio_file="audio.ogg")

        joined = "\n".join(captured.output)
        self.assertIn("KeyError", joined, "the exception type must be in the log")
        self.assertIn("Traceback", joined, "the log must carry the traceback")

    def test_the_client_message_stays_sanitised(self):
        def boom(self, **kwargs):
            raise KeyError("qwen3-asr-1.7b")

        with mock.patch.object(predict_module.Predictor, "_run_predict", boom):
            with self.assertLogs("predict", level="ERROR"):
                with self.assertRaises(RuntimeError) as raised:
                    self.predictor.predict(audio_file="audio.ogg")

        message = str(raised.exception)
        self.assertTrue(message.startswith("Prediction failed: KeyError:"), message)
        self.assertNotIn("Traceback", message, "no internals in the client message")
        self.assertNotIn("site-packages", message, "no internals in the client message")


if __name__ == "__main__":
    unittest.main()
