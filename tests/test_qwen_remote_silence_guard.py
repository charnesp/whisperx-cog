"""F1 RED/GREEN: a silence window never reaches the engine.

Adversarial review finding F1 (2026-10-20): ``qwen_split`` covers the WHOLE
signal, so a window with no speech at all was posted to the engine. The engine
answers with an empty ``content``, which raises ``RemoteResponseParserError``,
triggers the recovery path (retry + splits) and finally fails the whole
prediction — measured on 61 s of pure silence: 3 windows, total failure. The
former VAD path never sent silence.

Contract pinned here: when the peak absolute amplitude of a window is strictly
below ``SILENCE_PEAK_THRESHOLD`` (a named constant, ``0.003`` on the [-1, 1]
scale), the window resolves to ``("", None)`` with NO HTTP request and NO split
attempt. A window carrying speech still sends exactly ONE request.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import qwen_remote_client  # noqa: E402
import qwen_split  # noqa: E402

SAMPLE_RATE = 16000
CONFIG = {
    "backend": "remote",
    "base_url": "https://vllm.internal.invalid:9000/v1",
    "chat_url": "https://vllm.internal.invalid:9000/v1/chat/completions",
    "model": "qwen-vllm-test-model",
    "timeout_s": 30,
}
HOTWORDS = "Nespoulous, Chouette"


def _tone(seconds, amplitude=0.5, freq=220.0):
    n = int(round(seconds * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds):
    return np.zeros(int(round(seconds * SAMPLE_RATE)), dtype=np.float32)


def _envelope(text):
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": CONFIG["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "language French<asr_text>" + text},
                    "finish_reason": "stop",
                }
            ],
        }
    )


class _CallRecorder:
    """HTTP seam double: records every request, answers from ``text``."""

    def __init__(self, text="ok"):
        self.text = text
        self.calls = []

    def __call__(self, url, payload, headers, timeout=None):
        self.calls.append(json.loads(payload.decode("utf-8")))
        return 200, _envelope(self.text)

    def __len__(self):
        return len(self.calls)


class TestSilenceThresholdIsExplicit(unittest.TestCase):
    def test_named_constant_is_pinned(self):
        self.assertTrue(
            hasattr(qwen_remote_client, "SILENCE_PEAK_THRESHOLD"),
            "the silence guard needs an explicit named threshold, not a literal",
        )
        self.assertEqual(qwen_remote_client.SILENCE_PEAK_THRESHOLD, 0.003)
        # the threshold is expressed on the [-1, 1] float sample scale
        self.assertGreater(qwen_remote_client.SILENCE_PEAK_THRESHOLD, 0.0)
        self.assertLess(qwen_remote_client.SILENCE_PEAK_THRESHOLD, 1.0)


class TestSilentWindowSendsNoRequest(unittest.TestCase):
    def test_pure_silence_window_makes_zero_requests(self):
        recorder = _CallRecorder()
        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=recorder):
            texts, languages = qwen_remote_client.transcribe_windows_meta(
                [_silence(29.0)], CONFIG, batch_size=1, language="fr", context=HOTWORDS
            )
        self.assertEqual(texts, [""], "a silent window resolves to an empty text")
        self.assertEqual(languages, [None], "no language can be detected on silence")
        self.assertEqual(len(recorder), 0, "no HTTP request, no retry, no split")

    def test_silent_window_object_from_split_audio_makes_zero_requests(self):
        windows = qwen_split.split_audio(_silence(61.0))
        self.assertGreaterEqual(len(windows), 3)
        recorder = _CallRecorder()
        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=recorder):
            texts, languages = qwen_remote_client.transcribe_windows_meta(
                windows, CONFIG, batch_size=2, context=HOTWORDS
            )
        self.assertEqual(texts, [""] * len(windows))
        self.assertEqual(languages, [None] * len(windows))
        self.assertEqual(len(recorder), 0, "pure silence must never reach the engine")

    def test_below_threshold_noise_is_not_sent(self):
        recorder = _CallRecorder()
        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=recorder):
            texts, languages = qwen_remote_client.transcribe_windows_meta(
                [_tone(5.0, amplitude=0.002)], CONFIG, batch_size=1
            )
        self.assertEqual(texts, [""])
        self.assertEqual(languages, [None])
        self.assertEqual(len(recorder), 0)


class TestSpeechStillTravels(unittest.TestCase):
    def test_speech_window_sends_exactly_one_request(self):
        recorder = _CallRecorder(text="bonjour")
        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=recorder):
            texts, languages = qwen_remote_client.transcribe_windows_meta(
                [_tone(29.0, amplitude=0.5)], CONFIG, batch_size=1, language="fr", context=HOTWORDS
            )
        self.assertEqual(len(recorder), 1, "exactly one request for one healthy window")
        self.assertEqual(texts, ["bonjour"])
        self.assertEqual(languages, ["fr"])
        body = recorder.calls[0]
        self.assertNotIn("max_tokens", body)
        self.assertIn(HOTWORDS, body["messages"][0]["content"])

    def test_quiet_but_above_threshold_audio_is_still_sent(self):
        recorder = _CallRecorder(text=" murmure ")
        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=recorder):
            texts, _languages = qwen_remote_client.transcribe_windows_meta(
                [_tone(5.0, amplitude=0.01)], CONFIG, batch_size=1
            )
        self.assertEqual(len(recorder), 1, "quiet speech is still speech")
        self.assertEqual(texts, ["murmure"])

    def test_mixed_silence_and_speech_only_posts_the_speech(self):
        audio = np.concatenate([_silence(10.0), _tone(10.0, amplitude=0.4)])
        windows = qwen_split.split_audio(audio)
        recorder = _CallRecorder(text="parole")
        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=recorder):
            texts, languages = qwen_remote_client.transcribe_windows_meta(
                windows, CONFIG, batch_size=1
            )
        self.assertEqual(len(recorder), 1, "only the window carrying speech was posted")
        non_empty = [text for text in texts if text]
        self.assertEqual(non_empty, ["parole"])
        self.assertEqual(len(languages), len(windows))


if __name__ == "__main__":
    unittest.main()
