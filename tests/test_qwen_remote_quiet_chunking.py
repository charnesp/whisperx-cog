"""GPU-free RED/GREEN tests: quiet-point chunking, no token ceiling, per-window recovery.

Operator decisions (2026-10-20) superseding the v1 remote posture:

1. the chat/completions payload carries NO ``max_tokens`` — a 30 s window
   yields 109-137 tokens, so the former 120-token ceiling truncated replies
   (``finish_reason=length``) and failed whole predictions. No replacement
   value, no env variable;
2. the remote path cuts windows at QUIET POINTS (vLLM "Chunking Long Audio
   for Transcription": 30 s max, 1 s search window, 1600-sample (~100 ms at
   16 kHz) RMS energy frames, full coverage, absolute offsets) instead of
   pyannote VAD ``merge_chunks``;
3. a failing window is retried ONCE, then split in two halves at its quietest
   point (max depth 2) instead of failing the prediction while a split can
   still save it; only an unsplittable failure raises the typed error;
4. hotwords stay in the SYSTEM message of the chat request — the only
   conditioning channel that works with qwen3-asr (``/v1/audio/transcriptions``
   neither honours ``hotwords`` nor ``prompt``).

Module under test: ``qwen_split`` (pure numpy), ``qwen_remote_client``
(payload + recovery), ``predict.qwen_remote_windows`` (remote windowing).
"""
from __future__ import annotations

import base64
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
from qwen_remote import QwenRemoteError  # noqa: E402

SAMPLE_RATE = 16000
CONFIG = {
    "backend": "remote",
    "base_url": "https://vllm.internal.invalid:9000/v1",
    "chat_url": "https://vllm.internal.invalid:9000/v1/chat/completions",
    "model": "qwen-vllm-test-model",
    "timeout_s": 30,
}
HOTWORDS = "TEST-HOTWORD-SIGNAL"


def _tone(seconds, amplitude=0.5, freq=220.0):
    n = int(round(seconds * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds):
    return np.zeros(int(round(seconds * SAMPLE_RATE)), dtype=np.float32)


def _wav_sample_count(audio_url):
    """Sample count of a data URI payload (44-byte PCM16 RIFF header)."""
    prefix = "data:audio/wav;base64,"
    assert audio_url.startswith(prefix), audio_url[:32]
    return (len(base64.b64decode(audio_url[len(prefix):])) - 44) // 2


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


class TestPayloadHasNoTokenCeiling(unittest.TestCase):
    """Decision 1: the 120-token ceiling is GONE, hotwords stay in the system message."""

    def _payload(self, **kwargs):
        return qwen_remote_client.build_chat_payload(
            "data:audio/wav;base64,QUJD", CONFIG["model"], **kwargs
        )

    def test_constant_is_removed_from_the_module(self):
        self.assertFalse(
            hasattr(qwen_remote_client, "REMOTE_META_MAX_TOKENS"),
            "REMOTE_META_MAX_TOKENS must be deleted, not renamed",
        )

    def test_payload_carries_no_max_tokens_field(self):
        payload = self._payload(language="fr", context=HOTWORDS)
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("max_completion_tokens", payload)
        self.assertNotIn("max_tokens", json.dumps(payload))

    def test_payload_still_pins_model_temperature_and_stream(self):
        payload = self._payload()
        self.assertEqual(payload["model"], CONFIG["model"])
        self.assertEqual(payload["temperature"], 0)
        self.assertIs(payload["stream"], False)

    def test_hotwords_ride_the_system_message_of_a_ceiling_free_payload(self):
        payload = self._payload(language="fr", context=HOTWORDS)
        system = payload["messages"][0]
        self.assertEqual(system["role"], "system")
        self.assertIn(HOTWORDS, system["content"])
        # the audio still travels in the user turn: the system message is the
        # ONLY hotword channel (no hotwords/prompt field anywhere else)
        user = payload["messages"][1]
        self.assertEqual(user["role"], "user")
        self.assertEqual(user["content"][0]["type"], "audio_url")
        self.assertNotIn("hotwords", payload)
        self.assertNotIn("prompt", payload)

    def test_no_hotwords_means_no_context_sentence(self):
        payload = self._payload()
        self.assertEqual(
            payload["messages"][0]["content"], qwen_remote_client.DEFAULT_SYSTEM_PROMPT
        )


class TestQuietPointSplitter(unittest.TestCase):
    """Decision 2: pure numpy quiet-point chunking (no GPU, no pyannote)."""

    def test_documented_constants(self):
        self.assertEqual(qwen_split.MAX_WINDOW_S, 30.0)
        self.assertEqual(qwen_split.QUIET_SEARCH_S, 1.0)
        self.assertEqual(qwen_split.ENERGY_FRAME_SAMPLES, 1600)
        self.assertEqual(qwen_split.SAMPLE_RATE, 16000)

    def test_audio_shorter_than_max_is_one_window(self):
        audio = _tone(12.5)
        windows = qwen_split.split_audio(audio)
        self.assertEqual(len(windows), 1)
        start, end, samples = windows[0]
        self.assertEqual((start, end), (0.0, 12.5))
        self.assertEqual(len(samples), len(audio))
        np.testing.assert_array_equal(samples, audio)

    def test_every_window_is_at_most_30s_and_coverage_is_complete(self):
        audio = _tone(75.0)
        windows = qwen_split.split_audio(audio)
        self.assertGreater(len(windows), 2)
        for start, end, samples in windows:
            self.assertLessEqual(end - start, qwen_split.MAX_WINDOW_S + 1e-9)
            self.assertEqual(len(samples), int(round(end * SAMPLE_RATE)) - int(round(start * SAMPLE_RATE)))
        # absolute offsets: first window starts at 0, last ends on the last sample
        self.assertEqual(windows[0][0], 0.0)
        self.assertAlmostEqual(windows[-1][1], 75.0, places=6)
        # contiguous, no overlap, nothing lost
        for previous, following in zip(windows, windows[1:]):
            self.assertEqual(previous[1], following[0])
        rebuilt = np.concatenate([samples for _start, _end, samples in windows])
        np.testing.assert_array_equal(rebuilt, audio)

    def test_cut_lands_on_the_quietest_point_of_the_search_window(self):
        """A 0.2 s dip at 29.2-29.4 s (inside the 29-30 s search window) is the cut."""
        audio = _tone(45.0)
        dip_start = int(29.2 * SAMPLE_RATE)
        dip_end = int(29.4 * SAMPLE_RATE)
        audio[dip_start:dip_end] = 0.0
        windows = qwen_split.split_audio(audio)
        first_end = windows[0][1]
        self.assertGreaterEqual(first_end, 29.2)
        self.assertLessEqual(first_end, 29.4)
        # coverage still exact across the dip
        rebuilt = np.concatenate([samples for _s, _e, samples in windows])
        np.testing.assert_array_equal(rebuilt, audio)

    def test_offsets_are_absolute_seconds_derived_from_sample_indices(self):
        audio = _tone(64.0)
        windows = qwen_split.split_audio(audio)
        self.assertGreaterEqual(len(windows), 3)
        for start, end, samples in windows:
            np.testing.assert_array_equal(
                samples,
                audio[int(round(start * SAMPLE_RATE)):int(round(end * SAMPLE_RATE))],
            )
        # windows tile the timeline: no gap, no overlap
        self.assertEqual(windows[0][0], 0.0)
        for previous, following in zip(windows, windows[1:]):
            self.assertEqual(previous[1], following[0])

    def test_total_silence_still_yields_full_coverage(self):
        audio = _silence(61.0)
        windows = qwen_split.split_audio(audio)
        self.assertGreaterEqual(len(windows), 3)
        for start, end, samples in windows:
            self.assertLessEqual(end - start, qwen_split.MAX_WINDOW_S + 1e-9)
            self.assertGreater(end, start)
        rebuilt = np.concatenate([samples for _s, _e, samples in windows])
        np.testing.assert_array_equal(rebuilt, audio)

    def test_deterministic_same_input_same_plan(self):
        audio = _tone(70.0)
        audio[int(31.0 * SAMPLE_RATE)] = 0.0
        first = [(s, e) for s, e, _ in qwen_split.split_audio(audio)]
        second = [(s, e) for s, e, _ in qwen_split.split_audio(audio)]
        self.assertEqual(first, second)

    def test_empty_audio_yields_no_window(self):
        self.assertEqual(qwen_split.split_audio(np.zeros(0, dtype=np.float32)), [])

    def test_split_window_halves_preserve_every_sample(self):
        audio = _tone(8.0)
        left, right = qwen_split.split_window(audio)
        self.assertEqual(len(left) + len(right), len(audio))
        np.testing.assert_array_equal(np.concatenate([left, right]), audio)

    def test_split_window_refuses_audio_too_short_to_cut(self):
        self.assertIsNone(qwen_split.split_window(_silence(0.01)))


class TestPredictRemoteWindowsUsesQuietPoints(unittest.TestCase):
    """Decision 2: predict.qwen_remote_windows delegates to qwen_split, no pyannote."""

    def test_remote_windows_use_the_quiet_point_plan_without_pyannote(self):
        from _predict_stub import install

        predict = install()
        sys.modules.pop("whisperx.vads", None)
        audio = _tone(65.0)
        windows = predict.qwen_remote_windows(audio)
        self.assertNotIn("whisperx.vads", sys.modules, "remote path must not load pyannote VAD")
        plan = [(s, e) for s, e, _ in qwen_split.split_audio(audio)]
        self.assertEqual([(w[0], w[1]) for w in windows], plan)
        for start, end, samples in windows:
            self.assertLessEqual(end - start, 30.0)
            self.assertEqual(len(samples), int(round(end * SAMPLE_RATE)) - int(round(start * SAMPLE_RATE)))
        rebuilt = np.concatenate([w[2] for w in windows])
        np.testing.assert_array_equal(rebuilt, audio)


class _RecoveryHarness:
    """perform_http_post double failing on payloads longer than ``fail_samples``."""

    def __init__(self, fail_samples=None, fail_first_calls=0, text="ok"):
        self.fail_samples = fail_samples
        self.fail_first_calls = fail_first_calls
        self.text = text
        self.calls = []  # (sample_count, payload dict)
        self._lock_calls = 0

    def __call__(self, url, payload, headers, timeout=None):
        body = json.loads(payload.decode("utf-8"))
        audio_url = body["messages"][1]["content"][0]["audio_url"]["url"]
        samples = _wav_sample_count(audio_url)
        self.calls.append((samples, body))
        self._lock_calls += 1
        if self._lock_calls <= self.fail_first_calls:
            raise qwen_remote_client.RemoteHTTPStatusError(503, "transient")
        if self.fail_samples is not None and samples == self.fail_samples:
            raise qwen_remote_client.RemoteHTTPStatusError(503, "window too long")
        return 200, _envelope(f"{self.text}")

    def sample_counts(self):
        return [count for count, _body in self.calls]


class TestPerWindowRecovery(unittest.TestCase):
    """Decision 3: retry once, then split at a quiet point (max depth 2)."""

    def test_failed_window_is_retried_then_saved_by_a_split(self):
        w0, w1, w2 = _tone(1.0), _tone(3.0), _tone(1.25)
        windows = [w0, w1, w2]
        harness = _RecoveryHarness(fail_samples=len(w1))
        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=harness):
            texts = qwen_remote_client.transcribe_windows(
                windows, CONFIG, batch_size=2, context=HOTWORDS
            )
        self.assertEqual(len(texts), 3, "one result per input window, VAD order kept")
        self.assertEqual(texts[0], "ok")
        self.assertEqual(texts[2], "ok")
        self.assertIn("ok", texts[1])  # joined halves of the recovered window
        self.assertEqual(texts[1].count("ok"), 2)
        counts = harness.sample_counts()
        self.assertEqual(counts.count(len(w1)), 2, "exactly one retry of the full window")
        halves = [c for c in counts if c != len(w1) and c != len(w0) and c != len(w2)]
        self.assertEqual(len(halves), 2)
        self.assertEqual(sum(halves), len(w1), "split loses no sample")
        self.assertLess(max(halves), len(w1))

    def test_flaky_window_needs_only_one_retry_no_split(self):
        w0 = _tone(2.0)
        harness = _RecoveryHarness(fail_first_calls=1)
        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=harness):
            texts = qwen_remote_client.transcribe_windows([w0], CONFIG, batch_size=1)
        self.assertEqual(texts, ["ok"])
        self.assertEqual(harness.sample_counts(), [len(w0), len(w0)])

    def test_split_recurses_at_most_two_levels_then_raises(self):
        """Every attempt fails: 6 s -> two halves (depth 1) -> a quarter (depth 2).

        At depth 2 the window is retried once and can no longer be split, so the
        typed error surfaces instead of recursing forever.
        """
        lengths = []

        def always_fails(url, payload, headers, timeout=None):
            body = json.loads(payload.decode("utf-8"))
            audio_url = body["messages"][1]["content"][0]["audio_url"]["url"]
            lengths.append(_wav_sample_count(audio_url))
            raise qwen_remote_client.RemoteHTTPStatusError(503, "engine down")

        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=always_fails):
            with self.assertRaises(QwenRemoteError):
                qwen_remote_client.transcribe_windows([_tone(6.0)], CONFIG, batch_size=1)

        self.assertEqual(lengths[0:2], [96000, 96000])  # level 0: attempt + retry
        half = lengths[2]
        self.assertEqual(lengths[2:4], [half, half])  # level 1: attempt + retry
        self.assertLess(half, 96000, "the failed window was split")
        quarter = lengths[4]
        self.assertEqual(lengths[4:6], [quarter, quarter])  # level 2: retried, never split
        self.assertLess(quarter, half, "the half was split again")
        self.assertEqual(len(lengths), 6, "depth 2 is the floor: no third split")

    def test_depth_two_quarters_save_the_window(self):
        """Quiet points at 1/2/3 s make clean halves then quarters, all accepted."""
        audio = _tone(4.0)
        for centre in (1.0, 2.0, 3.0):
            audio[int((centre - 0.1) * SAMPLE_RATE):int((centre + 0.1) * SAMPLE_RATE)] = 0.0
        lengths = []

        def refuse_long_windows(url, payload, headers, timeout=None):
            body = json.loads(payload.decode("utf-8"))
            audio_url = body["messages"][1]["content"][0]["audio_url"]["url"]
            samples = _wav_sample_count(audio_url)
            lengths.append(samples)
            if samples > 20000:
                raise qwen_remote_client.RemoteHTTPStatusError(503, "window too long")
            return 200, _envelope("ok")

        with mock.patch.object(
            qwen_remote_client, "perform_http_post", side_effect=refuse_long_windows
        ):
            texts = qwen_remote_client.transcribe_windows([audio], CONFIG, batch_size=1)
        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0].count("ok"), 4, "four quarters fused in time order")
        self.assertEqual(lengths.count(64000), 2)  # full window: attempt + retry
        halves = [n for n in lengths if 20000 < n < 64000]
        quarters = [n for n in lengths if n <= 20000]
        self.assertEqual(len(halves), 4)  # two halves, retried once each
        self.assertEqual(len(quarters), 4)  # four quarters accepted
        self.assertEqual(sum(quarters), 64000, "quarters tile the whole window")

    def test_unsplittable_window_raises_typed_after_one_retry(self):
        """Non-sample payloads (pre-encoded data URIs) cannot be split: typed failure."""
        windows = ["data:audio/wav;base64,QUJD"]
        harness = _RecoveryHarness(fail_samples=None)
        harness.fail_first_calls = 10**6  # every attempt fails
        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=harness):
            with self.assertRaises(qwen_remote_client.RemoteHTTPStatusError) as caught:
                qwen_remote_client.transcribe_windows(windows, CONFIG, batch_size=1)
        self.assertIn("503", str(caught.exception))
        self.assertEqual(len(harness.calls), 2, "one attempt + one retry, then the typed error")

    def test_hotwords_and_no_max_tokens_on_every_retry_and_split_request(self):
        w0 = _tone(3.0)
        harness = _RecoveryHarness(fail_samples=len(w0))
        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=harness):
            qwen_remote_client.transcribe_windows(
                [w0], CONFIG, batch_size=1, language="fr", context=HOTWORDS
            )
        self.assertGreaterEqual(len(harness.calls), 3)
        for _samples, body in harness.calls:
            self.assertNotIn("max_tokens", body)
            self.assertIn(HOTWORDS, body["messages"][0]["content"])
            self.assertEqual(body["messages"][0]["role"], "system")


class TestRealWindowsFromSplitAudio(unittest.TestCase):
    """Regression: the client receives ``qwen_split.Window`` objects, not arrays.

    Production wiring is ``qwen_split.split_audio(audio)`` -> ``transcribe_windows_meta``.
    A ``Window`` is a NamedTuple ``(start, end, samples)``: iterating it as if it
    were a sample sequence encodes the floats ``start``/``end`` and the array
    itself, which raised ``TypeError: only 0-dimensional arrays can be converted
    to Python scalars``. These tests pin the REAL objects, not bare arrays.
    """

    @staticmethod
    def _signal(seconds=75.0, dips=(29.3, 58.4)):
        """Tone with silent dips inside the quiet-search window of each limit."""
        audio = _tone(seconds)
        for centre in dips:
            lo = int((centre - 0.1) * SAMPLE_RATE)
            hi = int((centre + 0.1) * SAMPLE_RATE)
            audio[lo:hi] = 0.0
        return audio

    def test_split_audio_yields_window_objects_under_test(self):
        windows = qwen_split.split_audio(self._signal())
        self.assertGreaterEqual(len(windows), 3)
        for window in windows:
            self.assertIsInstance(window, qwen_split.Window)
            self.assertEqual(len(window), 3, "Window stays (start, end, samples)")

    def test_window_objects_are_transcribed_in_order_with_their_own_samples(self):
        """(a) no exception, (b) texts in window order, (c) body carries that window's samples."""
        windows = qwen_split.split_audio(self._signal())
        expected = {
            qwen_remote_client.encode_wav_data_uri(w.samples): "texte-w%d" % index
            for index, w in enumerate(windows)
        }
        counts = {
            qwen_remote_client.encode_wav_data_uri(w.samples): len(w.samples)
            for w in windows
        }
        self.assertEqual(len(expected), len(windows), "distinct windows must differ in audio")
        sent = []
        violations = []

        def seam(url, payload, headers, timeout=None):
            body = json.loads(payload.decode("utf-8"))
            audio_url = body["messages"][1]["content"][0]["audio_url"]["url"]
            sent.append((audio_url, body))
            if audio_url not in expected:
                violations.append("payload audio is not a window's own samples")
                return 200, _envelope("inconnu")
            return 200, _envelope(expected[audio_url])

        with mock.patch.object(qwen_remote_client, "perform_http_post", side_effect=seam):
            texts, languages = qwen_remote_client.transcribe_windows_meta(
                windows, CONFIG, batch_size=2, language="fr", context=HOTWORDS
            )

        self.assertEqual(violations, [])
        self.assertEqual(len(texts), len(windows))
        self.assertEqual(
            texts,
            ["texte-w%d" % index for index in range(len(windows))],
            "one text per window, rendered in INPUT order",
        )
        self.assertEqual(languages, ["fr"] * len(windows), "detected ISO language per window")
        self.assertEqual(len(sent), len(windows), "exactly one request per healthy window")
        for audio_url, body in sent:
            self.assertEqual(
                _wav_sample_count(audio_url),
                counts[audio_url],
                "every sample of that window travels, and nothing else",
            )
            self.assertNotIn("max_tokens", body)
            self.assertEqual(body["messages"][0]["role"], "system")
            self.assertIn(HOTWORDS, body["messages"][0]["content"])

    def test_failed_window_object_is_really_split_not_silently_unsplittable(self):
        """(d) recovery on a Window: the halves are actually sent, not a silent None."""
        windows = qwen_split.split_audio(self._signal(seconds=6.0))
        self.assertEqual(len(windows), 1)
        window = windows[0]
        parent_count = len(window.samples)

        halves = qwen_remote_client._split_window_payload(window)
        if halves is None:
            self.fail("a Window must be splittable from its samples")
        left, right = halves
        self.assertEqual(len(left) + len(right), parent_count, "split loses no sample")
        np.testing.assert_array_equal(np.concatenate([left, right]), window.samples)

        sent = []

        def refuse_parent(url, payload, headers, timeout=None):
            body = json.loads(payload.decode("utf-8"))
            audio_url = body["messages"][1]["content"][0]["audio_url"]["url"]
            count = _wav_sample_count(audio_url)
            sent.append(count)
            if count == parent_count:
                raise qwen_remote_client.RemoteHTTPStatusError(503, "window rejected")
            return 200, _envelope("moitie")

        with mock.patch.object(
            qwen_remote_client, "perform_http_post", side_effect=refuse_parent
        ):
            texts = qwen_remote_client.transcribe_windows([window], CONFIG, batch_size=1)

        self.assertEqual(len(texts), 1, "bounds stay the parent window's: one input, one text")
        self.assertEqual(texts[0].count("moitie"), 2, "both halves transcribed and fused")
        self.assertEqual(sent.count(parent_count), 2, "attempt + one retry of the parent")
        halves_sent = [count for count in sent if count != parent_count]
        self.assertEqual(len(halves_sent), 2, "recovery really requested the two halves")
        self.assertEqual(sum(halves_sent), parent_count)


class TestParserNonRegression(unittest.TestCase):
    """Decision 1 must not weaken the reply parser guard rails."""

    def test_length_finish_reason_still_raises(self):
        with self.assertRaises(qwen_remote_client.RemoteResponseParserError):
            qwen_remote_client.parse_asr_content(
                "truncated text", finish_reason="length"
            )

    def test_language_prefix_is_stripped(self):
        self.assertEqual(
            qwen_remote_client.parse_asr_content("language French<asr_text>bonjour"),
            "bonjour",
        )

    def test_structured_output_without_prefix_is_kept(self):
        self.assertEqual(
            qwen_remote_client.parse_asr_content("  plain text  "), "plain text"
        )

    def test_empty_transcription_is_an_empty_segment(self):
        self.assertEqual(
            qwen_remote_client.parse_asr_content("language None<asr_text>"), ""
        )


if __name__ == "__main__":
    unittest.main()
