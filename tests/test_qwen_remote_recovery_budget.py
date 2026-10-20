"""F3 RED/GREEN: bounded per-window recovery (2026-10-20 adversarial review).

Findings under test:

- a split cannot repair a dead network or a server that does not answer: a
  ``RemoteTimeoutError`` / ``RemoteConnectionError`` gets ONE retry then the
  typed error, never a quiet-point split (splitting only multiplies timeouts);
- only CONTENT failures may be repaired by cutting the window shorter
  (parse / finish_reason=length / content-related HTTP 4xx-5xx);
- the total number of requests spent on ONE INPUT window is capped by an
  explicit named constant, whatever the tree shape.

``REMOTE_MAX_ATTEMPTS_PER_WINDOW`` is measured here, not assumed: the sweep test
counts the real seam calls over every failing-prefix shape and asserts the
observed maximum is at or below the constant.
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

import qwen_remote_client  # noqa: E402
from qwen_remote import QwenRemoteError  # noqa: E402

SAMPLE_RATE = 16000
CONFIG = {
    "backend": "remote",
    "base_url": "https://vllm.internal.invalid:9000/v1",
    "chat_url": "https://vllm.internal.invalid:9000/v1/chat/completions",
    "model": "qwen-vllm-test-model",
    "timeout_s": 30,
}
HOTWORDS = "Marbourg, Fibre Excellence"


def _tone(seconds, amplitude=0.5, freq=220.0):
    n = int(round(seconds * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


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


class _Recorder:
    """HTTP seam double driven by a callable ``behaviour(count) -> envelope|exc``."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = 0

    def __call__(self, url, payload, headers, timeout=None):
        self.calls += 1
        outcome = self.behaviour(self.calls)
        if isinstance(outcome, BaseException):
            raise outcome
        return 200, _envelope(outcome)


class TestNamedBudgetConstants(unittest.TestCase):
    def test_recovery_budget_is_an_explicit_constant(self):
        self.assertTrue(hasattr(qwen_remote_client, "REMOTE_MAX_ATTEMPTS_PER_WINDOW"))
        self.assertGreaterEqual(qwen_remote_client.REMOTE_MAX_ATTEMPTS_PER_WINDOW, 1)
        self.assertEqual(qwen_remote_client.REMOTE_WINDOW_RETRIES, 1)
        self.assertEqual(qwen_remote_client.REMOTE_MAX_SPLIT_DEPTH, 2)


class TestTransportFailuresAreNeverSplit(unittest.TestCase):
    def test_timeout_is_retried_once_then_typed_without_split(self):
        recorder = _Recorder(
            lambda _n: qwen_remote_client.RemoteTimeoutError("no answer in 30 s")
        )
        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=recorder):
            with self.assertRaises(qwen_remote_client.RemoteTimeoutError):
                qwen_remote_client.transcribe_windows_meta(
                    [_tone(6.0)], CONFIG, batch_size=1, context=HOTWORDS
                )
        self.assertEqual(
            recorder.calls,
            2,
            "one attempt + one retry: a shorter window cannot repair a timeout",
        )

    def test_connection_error_is_retried_once_then_typed_without_split(self):
        recorder = _Recorder(
            lambda _n: qwen_remote_client.RemoteConnectionError("connection refused")
        )
        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=recorder):
            with self.assertRaises(qwen_remote_client.RemoteConnectionError):
                qwen_remote_client.transcribe_windows_meta(
                    [_tone(6.0)], CONFIG, batch_size=1
                )
        self.assertEqual(recorder.calls, 2, "no split on a transport failure")

    def test_mixed_timeout_after_content_error_is_still_not_split(self):
        """The LAST error decides: a timeout never splits, even after a 503."""
        recorder = _Recorder(
            lambda n: (
                qwen_remote_client.RemoteHTTPStatusError(503, "busy")
                if n == 1
                else qwen_remote_client.RemoteTimeoutError("no answer")
            )
        )
        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=recorder):
            with self.assertRaises(qwen_remote_client.RemoteTimeoutError):
                qwen_remote_client.transcribe_windows_meta([_tone(6.0)], CONFIG, batch_size=1)
        self.assertEqual(recorder.calls, 2)


class TestContentFailuresStillRecover(unittest.TestCase):
    def test_http_status_splits_at_the_quiet_point(self):
        recorder = _Recorder(
            lambda n: (
                qwen_remote_client.RemoteHTTPStatusError(503, "window rejected")
                if n <= 2
                else "moitie"
            )
        )
        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=recorder):
            texts = qwen_remote_client.transcribe_windows(
                [_tone(4.0)], CONFIG, batch_size=1, context=HOTWORDS
            )
        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0].count("moitie"), 2, "both halves transcribed and fused")
        self.assertEqual(recorder.calls, 4, "parent attempt + retry, then two halves")

    def test_parse_error_splits_at_the_quiet_point(self):
        recorder = _Recorder(
            lambda n: (
                qwen_remote_client.RemoteResponseParserError("empty content")
                if n <= 2
                else "moitie"
            )
        )
        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=recorder):
            texts = qwen_remote_client.transcribe_windows([_tone(4.0)], CONFIG, batch_size=1)
        self.assertEqual(texts[0].count("moitie"), 2)
        self.assertEqual(recorder.calls, 4)

    def test_auth_failure_is_a_content_error_but_not_fixable_by_a_split(self):
        """401/403 are never repaired by a shorter window: one retry, typed error."""
        recorder = _Recorder(
            lambda _n: qwen_remote_client.RemoteHTTPStatusError(401, "bad token")
        )
        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=recorder):
            with self.assertRaises(qwen_remote_client.RemoteHTTPStatusError):
                qwen_remote_client.transcribe_windows_meta([_tone(6.0)], CONFIG, batch_size=1)
        self.assertEqual(recorder.calls, 2)


class TestBudgetIsEnforced(unittest.TestCase):
    def test_budget_stops_the_tree_even_if_the_shape_would_allow_more(self):
        recorder = _Recorder(
            lambda _n: qwen_remote_client.RemoteHTTPStatusError(503, "engine down")
        )
        with mock.patch.object(qwen_remote_client, "REMOTE_MAX_ATTEMPTS_PER_WINDOW", 3):
            with mock.patch.object(
                qwen_remote_client, "perform_http_post", side_effect=recorder
            ):
                with self.assertRaises(QwenRemoteError):
                    qwen_remote_client.transcribe_windows_meta(
                        [_tone(6.0)], CONFIG, batch_size=1
                    )
        self.assertEqual(recorder.calls, 3, "the budget is a hard bound, not a hint")

    def test_measured_worst_case_never_exceeds_the_cap(self):
        """Sweep the failing-prefix shapes and measure the real maximum."""
        cap = qwen_remote_client.REMOTE_MAX_ATTEMPTS_PER_WINDOW
        observed = {}
        for fail_first in range(0, 8):
            recorder = _Recorder(
                lambda n, fail_first=fail_first: (
                    qwen_remote_client.RemoteHTTPStatusError(503, "engine down")
                    if n <= fail_first
                    else "ok"
                )
            )
            with mock.patch.object(
                qwen_remote_client, "perform_http_post", side_effect=recorder
            ):
                try:
                    qwen_remote_client.transcribe_windows_meta(
                        [_tone(6.0)], CONFIG, batch_size=1
                    )
                except QwenRemoteError:
                    pass
            observed[fail_first] = recorder.calls
        for shape, calls in observed.items():
            self.assertLessEqual(
                calls, cap, f"shape fail_first={shape} spent {calls} (cap={cap}): {observed}"
            )

    def test_a_complete_depth_two_recovery_costs_exactly_the_cap(self):
        """The real worst case of a SUCCESSFUL recovery: measured, not assumed.

        2 (parent attempt + retry) + 4 (both halves attempt + retry) + 4 (four
        quarters accepted) = REMOTE_MAX_ATTEMPTS_PER_WINDOW. The cap therefore
        never truncates a recovery the policy promises.
        """
        audio = _tone(4.0)
        for centre in (1.0, 2.0, 3.0):  # quiet points at halves and quarters
            audio[int((centre - 0.1) * SAMPLE_RATE):int((centre + 0.1) * SAMPLE_RATE)] = 0.0
        budget_left = []

        def refuse_long_windows(url, payload, headers, timeout=None):
            body = json.loads(payload.decode("utf-8"))
            audio_url = body["messages"][1]["content"][0]["audio_url"]["url"]
            count = (len(__import__("base64").b64decode(audio_url.split("base64,", 1)[1])) - 44) // 2
            budget_left.append(count)
            if count > 20000:
                raise qwen_remote_client.RemoteHTTPStatusError(503, "window too long")
            return 200, _envelope("ok")

        with mock.patch.object(
            qwen_remote_client, "perform_http_post", side_effect=refuse_long_windows
        ):
            texts = qwen_remote_client.transcribe_windows([audio], CONFIG, batch_size=1)

        cap = qwen_remote_client.REMOTE_MAX_ATTEMPTS_PER_WINDOW
        self.assertEqual(texts[0].count("ok"), 4, "four quarters fused in time order")
        self.assertEqual(
            len(budget_left),
            cap,
            f"measured worst-case recovery = {len(budget_left)} requests, cap={cap}",
        )


if __name__ == "__main__":
    unittest.main()
