"""GPU-free RED tests for the remote qwen3-asr HTTP client (G3).

Change: openspec/changes/qwen3-asr-remote-vllm tasks.md group 3.
Module under test: qwen_remote_client.py — injectable stdlib-urllib client
posting ONE multimodal chat/completions per VAD window (tasks 3.1-3.3):
- HTTP double = exact mirror of the vLLM 0.30.0 contract (envelope
  choices[0].message.content; connection-refused / 5xx / socket-timeout
  error shapes; never invented);
- explicit ProxyHandler({}): urllib must NOT honor HTTP(S)_PROXY for the
  internal call (decision test, pinned via the constructor call);
- request body pins: model from env config, temperature=0, bounded
  max_tokens constant; NEGATIVE assertions: never response_format
  verbose_json, never timestamp_granularities;
- audio travels as a data URI of a deterministic mono 16 kHz PCM16 wav
  (deterministic encoder test: RIFF header fields, sample fidelity table,
  byte-exact round trip against an independently built wav);
- context/hotwords system message in EVERY request; language carried in
  the system message; per-request connect+read timeout passed through
  untouched; thread-safe: no shared mutable state, payload built fresh
  per call.

resolve_remote_config is NOT here (sibling module qwen_remote.py, G2):
the module-contract test asserts its absence.
"""
from __future__ import annotations

import base64
import io
import json
import struct
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import qwen_remote  # noqa: E402,F401  (sibling module, group 2 — must exist)
from qwen_remote import QwenRemoteError  # noqa: E402

CONFIG = {
    "backend": "remote",
    "base_url": "https://vllm.internal.invalid:9000/v1",
    "chat_url": "https://vllm.internal.invalid:9000/v1/chat/completions",
    "model": "qwen-vllm-test-model",
    "timeout_s": 30,
}
HOTWORD_CONTEXT = "TEST-HOTWORD-SIGNAL"
BASE_SYSTEM = (
    "Transcribe the input audio exactly as spoken; reply with the "
    "transcription text only."
)
SYSTEM_WITH_CONTEXT = BASE_SYSTEM + " Terms and names expected: " + HOTWORD_CONTEXT + "."
SYSTEM_WITH_LANG_FR = BASE_SYSTEM + " Reply in French (ISO code 'fr')."
SYSTEM_WITH_BOTH = (
    BASE_SYSTEM + " Reply in French (ISO code 'fr')."
    " Terms and names expected: " + HOTWORD_CONTEXT + "."
)
EXPECTED_HEADERS = (
    ("Content-Type", "application/json"),
)
CHAT_COMPLETION_ENVELOPE = {
    "id": "chatcmpl-mirror-0.30.0",
    "object": "chat.completion",
    "created": 0,
    "model": "qwen-vllm-test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "language French<asr_text>Salut ça marche"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _wav_bytes(samples, sample_rate=16000):
    """Test-side trusted wav builder (independently written from the caller)."""
    converted = []
    for value in samples:
        iv = int(round(32767.0 * float(value)))
        if iv > 32767:
            iv = 32767
        elif iv < -32767:
            iv = -32767
        converted.append(iv)
    pcm = b"".join(struct.pack("<h", s) for s in converted)
    data_size = len(pcm)
    body = b"".join(
        (
            b"fmt ",
            struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16),
            b"data",
            struct.pack("<I", data_size),
            pcm,
        )
    )
    riff_size = len(body) + 4
    return b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + body


def _decode_data_uri(uri):
    prefix = "data:audio/wav;base64,"
    assert uri.startswith(prefix), uri[:64]
    return base64.b64decode(uri[len(prefix):])


class _FakeResponse:
    def __init__(self, status=200, body=b"", headers=None):
        self.status = status
        self._body = body
        self.headers = headers or dict(EXPECTED_HEADERS)
        self._read = False
        # NOTE: the HTTPError(fp=bytes) doubles carry a raw bytes fp; CPython
        # wraps it into a temp file lazily and its GC __del__ prints a harmless
        # "'bytes' object has no attribute 'close'" noise line. Unused here.

    def read(self):
        if self._read:
            return b""
        self._read = True
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeOpener:
    def __init__(self, queue):
        self.queue = queue

    def open(self, request, timeout=None):
        self.queue.append(
            {
                "full_url": request.full_url,
                "headers": dict(request.header_items()),
                "method": request.get_method(),
                "data": request.data,
                "timeout": timeout,
            }
        )
        return _FakeResponse(status=200, body=json.dumps(CHAT_COMPLETION_ENVELOPE).encode("utf-8"))


class TestModuleContract(unittest.TestCase):
    """qwen_remote_client exposes the client only; config stays in qwen_remote."""

    def test_module_exports_client_contract(self):
        import qwen_remote_client as module

        for attr in (
            "encode_wav_data_uri",
            "perform_http_post",
            "transcribe_window",
            "transcribe_windows",
            "clamp_remote_pool_size",
        ):
            self.assertTrue(hasattr(module, attr), attr)

    def test_module_does_NOT_own_resolve_remote_config(self):
        import qwen_remote_client as module

        self.assertFalse(hasattr(module, "resolve_remote_config"))

    def test_typed_errors_subclass_QwenRemoteError(self):
        import qwen_remote_client as module

        self.assertTrue(issubclass(module.QwenRemoteError, QwenRemoteError))
        for cls in (
            module.RemoteConfigError,
            module.RemoteConnectionError,
            module.RemoteTimeoutError,
            module.RemoteHTTPStatusError,
            module.RemoteResponseParserError,
        ):
            self.assertTrue(issubclass(cls, QwenRemoteError), cls)

    def test_error_messages_carry_stable_category_prefix(self):
        import qwen_remote_client as module

        for cls, message in (
            (module.RemoteConnectionError, "boom"),
            (module.RemoteTimeoutError, "no answer"),
            (module.RemoteConfigError, "bad config"),
        ):
            instance = cls(message)
            text = str(instance)
            self.assertTrue(text.startswith("QwenRemoteError:"), text)
            self.assertIn(cls.category, text)
        status = module.RemoteHTTPStatusError(502, "upstream exploded")
        self.assertTrue(str(status).startswith("QwenRemoteError: http_status"), str(status))


class TestEncodeWavPcm16Mono16k(unittest.TestCase):
    """Deterministic encoder: header fields + byte-exact fidelity."""

    def setUp(self):
        import qwen_remote_client as module

        self.module = module

    def test_riff_header_fields(self):
        wav = _decode_data_uri(self.module.encode_wav_data_uri([0, 100, -100]))
        tag, riff_size, wave, fmt_tag, fmt_size, audio_format, channels, rate, byte_rate, block_align, bits, data_tag, data_size = struct.unpack(
            "<4sI4s4sIHHIIHH4sI", wav[:44]
        )
        self.assertEqual(tag, b"RIFF")
        self.assertEqual(wave, b"WAVE")
        self.assertEqual(fmt_tag, b"fmt ")
        self.assertEqual(fmt_size, 16)
        self.assertEqual(audio_format, 1)  # PCM
        self.assertEqual(channels, 1)  # mono
        self.assertEqual(rate, 16000)  # 16 kHz
        self.assertEqual(byte_rate, 32000)
        self.assertEqual(block_align, 2)
        self.assertEqual(bits, 16)
        self.assertEqual(data_tag, b"data")

    def test_byte_exact_round_trip_against_independent_builder(self):
        samples = [0.0, 0.25, -0.25, 0.5, -1.0, 1.0, 4.0, -4.0, 0.125]
        wav = _decode_data_uri(self.module.encode_wav_data_uri(samples))
        self.assertEqual(wav, _wav_bytes(samples))

    def test_sample_fidelity_table(self):
        samples = [0.0, 0.25, -0.25, 0.5, -1.0, 1.0, 4.0, -4.0]
        wav = _decode_data_uri(self.module.encode_wav_data_uri(samples))
        self.assertEqual(len(wav), 44 + 2 * len(samples))
        blob = wav[44:]
        values = struct.unpack("<%dh" % len(samples), blob)
        self.assertEqual(
            values,
            (0, 8192, -8192, 16384, -32767, 32767, 32767, -32767),
        )

    def test_deterministic_same_input_same_bytes(self):
        uri_a = self.module.encode_wav_data_uri([0.1, 0.2, 0.3])
        uri_b = self.module.encode_wav_data_uri([0.1, 0.2, 0.3])
        self.assertEqual(uri_a, uri_b)


class TestChatCompletionPayload(unittest.TestCase):
    """Request contract (mirror of vLLM 0.30.0 chat/completions, specs B1/B6)."""

    def setUp(self):
        import qwen_remote_client as module

        self.module = module

    def _payload(self, **kwargs):
        window = kwargs.pop("window", "data:audio/wav;base64,QUJD")
        return self.module.build_chat_payload(
            window, CONFIG["model"], **kwargs
        )

    def test_pins_model_temperature_and_bounded_max_tokens(self):
        payload = self._payload()
        self.assertEqual(payload["model"], "qwen-vllm-test-model")
        self.assertEqual(payload["temperature"], 0)
        self.assertTrue(isinstance(payload["max_tokens"], int))
        self.assertLessEqual(payload["max_tokens"], 128)
        self.assertGreater(payload["max_tokens"], 0)
        self.assertEqual(
            payload["max_tokens"], self.module.REMOTE_META_MAX_TOKENS
        )

    def test_never_requests_server_side_timestamps(self):
        payload = self._payload()
        self.assertNotIn("verbose_json", json.dumps(payload))
        self.assertNotIn("timestamp_granularities", json.dumps(payload))
        self.assertNotIn("response_format", payload)

    def test_user_content_is_audio_url_data_uri(self):
        payload = self._payload(window="data:audio/wav;base64,WUla")
        user = payload["messages"][1]
        self.assertEqual(user["role"], "user")
        part = user["content"][0]
        self.assertEqual(part["type"], "audio_url")
        self.assertEqual(part["audio_url"]["url"], "data:audio/wav;base64,WUla")

    def test_system_message_present_without_hotwords(self):
        payload = self._payload()
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], BASE_SYSTEM)

    def test_hotword_context_travels_with_every_request(self):
        payload_a = self._payload(context=HOTWORD_CONTEXT)
        payload_b = self._payload(context=HOTWORD_CONTEXT)
        self.assertEqual(
            payload_a["messages"][0]["content"], SYSTEM_WITH_CONTEXT
        )
        self.assertEqual(
            payload_a["messages"][0]["content"],
            payload_b["messages"][0]["content"],
        )

    def test_language_instruction_carried_in_system_message(self):
        payload = self._payload(language="fr")
        self.assertEqual(payload["messages"][0]["content"], SYSTEM_WITH_LANG_FR)

    def test_language_and_context_together(self):
        payload = self._payload(language="fr", context=HOTWORD_CONTEXT)
        self.assertEqual(payload["messages"][0]["content"], SYSTEM_WITH_BOTH)

    def test_payload_built_fresh_per_call_no_shared_mutable_state(self):
        a = self._payload(context=HOTWORD_CONTEXT)
        b = self._payload()
        self.assertNotEqual(a["messages"][0]["content"], b["messages"][0]["content"])
        a["messages"].append({"role": "user", "content": "sentinel"})
        self.assertEqual(len(self._payload()["messages"]), 2)


class TestPerformHttpPostEntrypoint(unittest.TestCase):
    """stdlib-urllib seam: proxy decision, headers, timeout, error shapes."""

    def setUp(self):
        import qwen_remote_client as module

        self.module = module

    def test_proxy_is_explicitly_disabled_for_the_internal_call(self):
        """B310 decision test: ProxyHandler({}) — urllib must NOT honor proxies."""
        sentinel_opener = _FakeOpener([])
        with mock.patch("urllib.request.ProxyHandler", wraps=urllib.request.ProxyHandler) as pd:
            with mock.patch("urllib.request.build_opener", return_value=sentinel_opener):
                self.module.perform_http_post(
                    CONFIG["chat_url"], payload=b"{}", headers={"Content-Type": "application/json"}, timeout=3
                )
        pd.assert_called_once_with({})
        self.assertEqual(sentinel_opener.queue[0]["full_url"], CONFIG["chat_url"])

    def test_builds_request_with_method_and_json_headers(self):
        opener = _FakeOpener([])
        with mock.patch("urllib.request.build_opener", return_value=opener):
            status, body = self.module.perform_http_post(
                CONFIG["chat_url"],
                payload=json.dumps({"messages": []}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                timeout=7,
            )
        self.assertEqual(status, 200)
        self.assertEqual(body, json.dumps(CHAT_COMPLETION_ENVELOPE))
        sent = opener.queue[0]
        self.assertEqual(sent["method"], "POST")
        header_names = {key.lower() for key in sent["headers"]}
        self.assertIn("content-type", header_names)
        self.assertEqual(sent["timeout"], 7)

    def test_http_error_maps_to_typed_status_error_with_truncated_body(self):
        class BoomOpener:
            def open(self, request, timeout=None):
                body_io = io.BytesIO(b"vllm busy" * 100)
                raise urllib.error.HTTPError(
                    request.full_url, 503, "Service Unavailable", {}, body_io
                )

        with mock.patch("urllib.request.build_opener", return_value=BoomOpener()):
            with self.assertRaises(self.module.RemoteHTTPStatusError) as caught:
                self.module.perform_http_post(
                    CONFIG["chat_url"], payload=b"{}", headers={"Content-Type": "application/json"}, timeout=None
                )
        text = str(caught.exception)
        self.assertIn("503", text)
        self.assertLessEqual(len(text), 400)  # body truncated, never whole

    def test_connection_refused_maps_to_connection_error(self):
        refused = urllib.error.URLError(ConnectionRefusedError("refused"))

        class BoomOpener:
            def open(self, request, timeout=None):
                raise refused

        with mock.patch("urllib.request.build_opener", return_value=BoomOpener()):
            with self.assertRaises(self.module.RemoteConnectionError):
                self.module.perform_http_post(
                    CONFIG["chat_url"], payload=b"{}", headers={"Content-Type": "application/json"}, timeout=None
                )

    def test_socket_timeout_maps_to_timeout_error(self):
        timed_out = urllib.error.URLError(TimeoutError("connect timed out"))

        class BoomOpener:
            def open(self, request, timeout=None):
                raise timed_out

        with mock.patch("urllib.request.build_opener", return_value=BoomOpener()):
            with self.assertRaises(self.module.RemoteTimeoutError) as caught:
                self.module.perform_http_post(
                    CONFIG["chat_url"], payload=b"{}", headers={"Content-Type": "application/json"}, timeout=None
                )
        self.assertIn("timeout", str(caught.exception))


class TestTranscribeWindowWorkflow(unittest.TestCase):
    """One window -> one chat completion -> parsed text (client green path)."""

    def setUp(self):
        import qwen_remote_client as module

        self.module = module

    def test_returns_asr_text_on_mirrored_vllm_envelope(self):
        opener = _FakeOpener([])
        with mock.patch("urllib.request.build_opener", return_value=opener):
            text = self.module.transcribe_window(
                "data:audio/wav;base64,WUla", CONFIG, context=HOTWORD_CONTEXT
            )
        self.assertEqual(text, "Salut ça marche")
        sent = opener.queue[0]
        self.assertEqual(sent["full_url"], CONFIG["chat_url"])
        self.assertEqual(sent["timeout"], 30)
        body = json.loads(sent["data"])
        self.assertEqual(body["model"], "qwen-vllm-test-model")
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(
            body["messages"][0]["content"], SYSTEM_WITH_CONTEXT
        )

    def test_non_200_status_raises_typed_status_error(self):
        class BoomOpener:
            def open(self, request, timeout=None):
                raise urllib.error.HTTPError(
                    request.full_url, 502, "Bad Gateway", {}, io.BytesIO(b"mid-swap")
                )

        with mock.patch("urllib.request.build_opener", return_value=BoomOpener()):
            with self.assertRaises(self.module.RemoteHTTPStatusError) as caught:
                self.module.transcribe_window("data:audio/wav;base64,QQ", CONFIG)
        self.assertIn("502", str(caught.exception))
        self.assertIn("mid-swap", str(caught.exception))

    def test_empty_choices_raises_typed_parse_error(self):
        envelope = dict(CHAT_COMPLETION_ENVELOPE)
        envelope["choices"] = []
        opener = _FakeOpener([])
        opener.open = lambda request, timeout=None: _FakeResponse(
            status=200, body=json.dumps(envelope).encode("utf-8")
        )
        with mock.patch("urllib.request.build_opener", return_value=opener):
            with self.assertRaises(self.module.RemoteResponseParserError):
                self.module.transcribe_window("data:audio/wav;base64,QQ", CONFIG)


if __name__ == "__main__":
    unittest.main()
