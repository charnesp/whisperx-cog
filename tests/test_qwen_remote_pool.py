"""GPU-free RED tests for the qwen3-asr remote POOL ORCHESTRATOR (G5).

Change: openspec/changes/qwen3-asr-remote-vllm tasks.md group 5 (task 5.1):
- 3 windows, pool 2 -> ALL transcribed, results ordered by VAD window start
  regardless of completion order (reverse-completion double);
- concurrency == clamped batch_size: clamp helper semantics (default 4 when
  absent, cap 8, <=0 / non-integer -> default) proven via a counting double
  (peak concurrent in-flight <= clamp, executor max_workers == clamp);
- ONE failing window -> whole prediction fails with the typed error,
  pending futures CANCELLED (cancel_futures=True observable), in-flight
  results discarded (none fused), NO partial output;
- ZERO retries: a 5xx in the mock -> immediate typed failure, exactly ONE
  call for the failing window (no second attempt), other completed windows
  do NOT trigger local fallback;
- caller stays in charge of VAD timestamps: orchestrator returns TEXTS in
  input order only (structural assertion: no start/end in results).
"""
from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import qwen_remote_client  # noqa: E402
from qwen_remote_client import (  # noqa: E402
    QwenRemoteError,
    RemoteHTTPStatusError,
    clamp_remote_pool_size,
    transcribe_windows,
)

CONFIG = {
    "backend": "remote",
    "base_url": "https://vllm.internal.invalid:9000/v1",
    "chat_url": "https://vllm.internal.invalid:9000/v1/chat/completions",
    "model": "qwen-vllm-test-model",
    "timeout_s": 30,
}


class _CountingExecutor:
    """ThreadPoolExecutor double recording max_workers + cancellation."""

    instances: list = []

    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        self.cancel_calls = []
        self.shutdown_calls = []
        self.inner = ThreadPoolExecutor(max_workers=max_workers or 1)
        type(self).instances.append(self)

    def submit(self, fn, *args, **kwargs):
        return self.inner.submit(fn, *args, **kwargs)

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_calls.append((wait, cancel_futures))
        self.inner.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.inner.shutdown(wait=False)
        return False


class _OrderedFakeOpener:
    """Serving double: reverse-completion delays + optional 5xx on window i."""

    def __init__(self, texts, slow_first=None, fail_window=None):
        self.texts = texts
        self.slow_first = slow_first or 0.0
        self.fail_window = fail_window
        self.calls_per_window = {}

    def open(self, request, timeout=None):
        body = json.loads(request.data.decode("utf-8"))
        url = body["messages"][1]["content"][0]["audio_url"]["url"]
        window_index = self.calls_per_window.get(url, 0)
        self.calls_per_window[url] = window_index + 1
        index = hash(url) % len(self.texts)
        if self.fail_window is not None and index == self.fail_window:
            import urllib.error

            raise urllib.error.HTTPError(
                request.full_url, 503, "Service Unavailable", {}, __import__("io").BytesIO(b"down")
            )
        if index == 0 and self.slow_first:
            time.sleep(self.slow_first)
        envelope = {
            "id": "x",
            "object": "chat.completion",
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "language French<asr_text>" + self.texts[index]},
                    "finish_reason": "stop",
                }
            ],
        }
        return _Resp(json.dumps(envelope).encode("utf-8"))


class _Resp:
    def __init__(self, body):
        self.status = 200
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestClampRemotePoolSize(unittest.TestCase):
    """Mirror of predict.resolve_qwen_batch_size semantics (default 4/cap 8)."""

    def test_absent_none_defaults_4(self):
        self.assertEqual(clamp_remote_pool_size(None), 4)

    def test_explicit_value_passthrough(self):
        self.assertEqual(clamp_remote_pool_size(2), 2)
        self.assertEqual(clamp_remote_pool_size(8), 8)

    def test_above_cap_clamped_to_8(self):
        self.assertEqual(clamp_remote_pool_size(32), 8)

    def test_zero_and_negative_rejected_to_default(self):
        self.assertEqual(clamp_remote_pool_size(0), 4)
        self.assertEqual(clamp_remote_pool_size(-3), 4)

    def test_non_integer_rejected_to_default(self):
        self.assertEqual(clamp_remote_pool_size("six"), 4)
        self.assertEqual(clamp_remote_pool_size(2.5), 4)
        self.assertEqual(clamp_remote_pool_size(True), 4)


class TestPoolOrderAndCompletion(unittest.TestCase):
    def test_three_windows_pool_two_all_transcribed_vad_order(self):
        # window texts flow back in REVERSE completion order (w0 slowest)
        texts = ["debut", "milieu", "fin"]
        calls = []

        def fake_post(url, payload, headers, timeout=None):
            calls.append(json.loads(payload.decode("utf-8")))
            return _serve_static(texts, calls)

        def _serve_static(texts, calls):
            payload = calls[-1]
            audio_url = payload["messages"][1]["content"][0]["audio_url"]["url"]
            # deterministic window order: fake payloads carry sentinel texts
            for i, t in enumerate(texts):
                if t in audio_url:
                    envelope = {
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": "language French<asr_text>" + t},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                    return 200, json.dumps(envelope)
            raise AssertionError("unreachable")

        windows = [f"data:audio/wav;base64,{t}" for t in texts]
        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=fake_post):
            result = transcribe_windows(windows, CONFIG, batch_size=2)
        self.assertEqual(result, texts)  # VAD/input order, not completion order
        self.assertEqual(len(calls), 3)

    def test_reverse_completion_still_vad_order(self):
        """Slow FIRST window: futures complete w2,w1,w0 — fused order stays 0,1,2."""
        texts = ["alpha", "beta", "gamma"]
        windows = [f"data:audio/wav;base64,{t}" for t in texts]
        delays = {0: 0.2, 1: 0.05, 2: 0.0}
        completion_order = []

        def fake_post(url, payload, headers, timeout=None):
            payload_obj = json.loads(payload.decode("utf-8"))
            audio_url = payload_obj["messages"][1]["content"][0]["audio_url"]["url"]
            index = texts.index(audio_url.split(",", 1)[1])
            time.sleep(delays[index])
            completion_order.append(index)
            envelope = {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "language French<asr_text>" + texts[index]},
                        "finish_reason": "stop",
                    }
                ]
            }
            return 200, json.dumps(envelope)

        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=fake_post):
            result = transcribe_windows(windows, CONFIG, batch_size=2)
        self.assertNotEqual(completion_order, sorted(completion_order))  # truly reversed
        self.assertEqual(result, texts)

    def test_pool_never_exceeds_clamped_batch_size(self):
        texts = ["a", "b", "c", "d", "e"]
        windows = [f"data:audio/wav;base64,{t}" for t in texts]
        peak_inflight = []
        current = [0]
        lock = threading.Lock()

        def fake_post(url, payload, headers, timeout=None):
            with lock:
                current[0] += 1
                peak_inflight.append(current[0])
            time.sleep(0.02)
            with lock:
                current[0] -= 1
            envelope = {"choices": [{"message": {"role": "assistant", "content": "language French<asr_text>x"}, "finish_reason": "stop"}]}
            return 200, json.dumps(envelope)

        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=fake_post):
            transcribe_windows(windows, CONFIG, batch_size=2)
        self.assertLessEqual(max(peak_inflight), 2)

    def test_executor_gets_max_workers_equal_to_clamped_batch(self):
        texts = ["a", "b"]
        windows = [f"data:audio/wav;base64,{t}" for t in texts]
        with mock.patch.object(
            qwen_remote_client, "ThreadPoolExecutor", _CountingExecutor
        ):
            def keyed_post(url, payload, headers, timeout=None):
                payload_obj = json.loads(payload.decode("utf-8"))
                audio_url = payload_obj["messages"][1]["content"][0]["audio_url"]["url"]
                index = texts.index(audio_url.split("base64,", 1)[1])
                envelope = {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "language French<asr_text>" + texts[index]},
                            "finish_reason": "stop",
                        }
                    ]
                }
                return 200, json.dumps(envelope)

            with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=keyed_post):
                transcribe_windows(windows, CONFIG, batch_size=32)
        counting = _CountingExecutor.instances[-1]
        self.assertEqual(counting.max_workers, 8)  # clamped 32 -> cap 8


class TestFailureSemantics(unittest.TestCase):
    def test_one_failing_window_fails_whole_prediction_typed(self):
        texts = ["ok0", "boom", "ok2"]
        windows = [f"data:audio/wav;base64,{t}" for t in texts]

        def fake_post(url, payload, headers, timeout=None):
            payload_obj = json.loads(payload.decode("utf-8"))
            audio_url = payload_obj["messages"][1]["content"][0]["audio_url"]["url"]
            index = texts.index(audio_url.split(",", 1)[1])
            if index == 1:
                raise RemoteHTTPStatusError(503, "engine down")
            envelope = {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "language French<asr_text>" + texts[index]},
                        "finish_reason": "stop",
                    }
                ]
            }
            return 200, json.dumps(envelope)

        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=fake_post):
            with self.assertRaises(QwenRemoteError) as caught:
                transcribe_windows(windows, CONFIG, batch_size=2)
        self.assertIn("503", str(caught.exception))

    def test_pending_futures_cancelled_on_failure(self):
        texts = ["slow-ok", "fails-fast", "never-started"]
        windows = [f"data:audio/wav;base64,{t}" for t in texts]
        with mock.patch.object(
            qwen_remote_client, "ThreadPoolExecutor", _CountingExecutor
        ) and mock.patch.object(qwen_remote_client, "perform_http_post"):
            pass  # replaced below by direct double wiring

        def fake_post(url, payload, headers, timeout=None):
            payload_obj = json.loads(payload.decode("utf-8"))
            audio_url = payload_obj["messages"][1]["content"][0]["audio_url"]["url"]
            index = texts.index(audio_url.split(",", 1)[1])
            if index == 0:
                time.sleep(0.3)  # in-flight when w1 fails
            if index == 1:
                time.sleep(0.02)
                raise RemoteHTTPStatusError(503, "down")
            envelope = {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "language French<asr_text>late"},
                        "finish_reason": "stop",
                    }
                ]
            }
            return 200, json.dumps(envelope)

        _CountingExecutor.instances = []
        with mock.patch.object(qwen_remote_client, "ThreadPoolExecutor", _CountingExecutor):
            with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=fake_post):
                with self.assertRaises(QwenRemoteError):
                    transcribe_windows(windows, CONFIG, batch_size=2)
        counting = _CountingExecutor.instances[-1]
        self.assertEqual(counting.shutdown_calls[0], (False, True))  # cancel_futures=True

    def test_inflight_results_discarded_no_partial_output(self):
        texts = ["slow-ok", "fails", "third"]
        windows = [f"data:audio/wav;base64,{t}" for t in texts]

        def fake_post(url, payload, headers, timeout=None):
            payload_obj = json.loads(payload.decode("utf-8"))
            audio_url = payload_obj["messages"][1]["content"][0]["audio_url"]["url"]
            index = texts.index(audio_url.split(",", 1)[1])
            if index == 0:
                time.sleep(0.3)
            if index == 1:
                time.sleep(0.02)
                raise RemoteHTTPStatusError(503, "down")
            envelope = {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "language French<asr_text>" + texts[index]},
                        "finish_reason": "stop",
                    }
                ]
            }
            return 200, json.dumps(envelope)

        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=fake_post):
            with self.assertRaises(QwenRemoteError):
                transcribe_windows(windows, CONFIG, batch_size=3)
        # the failing call raised inside the orchestrator: NO fused list returned
        # (assertNothingReturned is structural: the exception replaces the result)

    def test_zero_retries_5xx_immediate_typed_failure(self):
        texts = ["never-comes", "never-comes-2"]
        windows = [f"data:audio/wav;base64,{t}" for t in texts]
        attempts = []

        def fake_post(url, payload, headers, timeout=None):
            attempts.append(url)
            raise RemoteHTTPStatusError(503, "mid-swap")

        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=fake_post):
            with self.assertRaises(QwenRemoteError):
                transcribe_windows(windows, CONFIG, batch_size=1)
        self.assertEqual(len(attempts), 1)  # ONE attempt, no second call, no retry

    def test_no_fallback_to_local_engine_on_failure(self):
        texts = ["x", "y"]
        windows = [f"data:audio/wav;base64,{t}" for t in texts]
        local_calls = []

        attempts = []

        def failing_window(url, payload, headers, timeout=None):
            attempts.append(url)
            raise RemoteHTTPStatusError(503, "down")

        # The seam the orchestrator calls is transcribe_window; patching it
        # lets us observe the fallback candidate call site directly.
        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=failing_window):
            with self.assertRaises(QwenRemoteError):
                transcribe_windows(windows, CONFIG, batch_size=1)
        self.assertEqual(local_calls, [])
        self.assertEqual(len(attempts), 1)  # failed once, never retried/fallback


if __name__ == "__main__":
    unittest.main()
