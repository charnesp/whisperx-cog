"""Tests for OpenAI-compatible STT endpoint (bridge/openai_compat.py)."""

from __future__ import annotations

import io
import json
import sys
import threading
import unittest
import urllib.error
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bridge"))

import openai_compat
from openai_compat import (
    MOCK_COG_OUTPUT,
    MODEL_MAP,
    OPENAI_AUDIO_EXTENSIONS,
    auth_error_response,
    build_audio_data_uri,
    build_cog_input,
    call_cog_sync,
    convert_cog_output,
    convert_to_diarized_json,
    convert_to_json,
    convert_to_srt,
    convert_to_text,
    convert_to_verbose_json,
    convert_to_vtt,
    handle_transcription_request,
    is_diarize_request,
    validate_transcription_request,
)


def _encode_multipart(fields, files):
    boundary = "----testboundary"
    body = io.BytesIO()
    for name, value in fields:
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.write(f"{value}\r\n".encode())
    for name, filename, content, content_type in files:
        body.write(f"--{boundary}\r\n".encode())
        body.write(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.write(content)
        body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())
    return boundary, body.getvalue()


def _parse_multipart(body_bytes, content_type):
    msg = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\r\n\r\n".encode() + body_bytes
    )
    fs, err = openai_compat.parse_multipart_form(
        {"Content-Type": content_type},
        io.BytesIO(body_bytes),
        len(body_bytes),
    )
    if err:
        raise AssertionError(err)
    return fs


class MockCogResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def mock_cog_urlopen_factory(delay=0, fail_status=None, output=None):
    captured = {"requests": []}

    def urlopen(req, timeout=None):
        captured["requests"].append(
            {
                "url": req.full_url,
                "headers": dict(req.header_items()),
                "body": req.data,
                "timeout": timeout,
            }
        )
        if delay:
            import time

            time.sleep(delay)
        if fail_status == "timeout":
            raise TimeoutError("timed out")
        if fail_status == "failed":
            return MockCogResponse({"status": "failed", "error": "gpu oom"})
        payload = {
            "status": "succeeded",
            "output": output or MOCK_COG_OUTPUT,
        }
        return MockCogResponse(payload)

    urlopen.captured = captured
    return urlopen


class TestOpenAiCompatUnit(unittest.TestCase):
    def test_model_map_whisper1(self):
        self.assertEqual(MODEL_MAP["whisper-1"], "large-v3-turbo")

    def test_model_map_diarize_model(self):
        self.assertEqual(MODEL_MAP["gpt-4o-transcribe-diarize"], "large-v3-turbo")

    def test_is_diarize_request_by_model(self):
        self.assertTrue(is_diarize_request("gpt-4o-transcribe-diarize", "json"))

    def test_is_diarize_request_by_format(self):
        self.assertTrue(is_diarize_request("whisper-1", "diarized_json"))

    def test_is_diarize_request_false_for_whisper1(self):
        self.assertFalse(is_diarize_request("whisper-1", "json"))

    def test_convert_to_diarized_json_speaker_letters(self):
        cog_output = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello world",
                    "speaker": "SPEAKER_00",
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                        {"word": "world", "start": 1.0, "end": 2.5, "speaker": "SPEAKER_00"},
                    ],
                },
                {
                    "start": 2.5,
                    "end": 5.0,
                    "text": "Hi there",
                    "speaker": "SPEAKER_01",
                    "words": [
                        {"word": "Hi", "start": 2.5, "end": 3.2, "speaker": "SPEAKER_01"},
                        {"word": "there", "start": 3.2, "end": 5.0, "speaker": "SPEAKER_01"},
                    ],
                },
            ],
            "detected_language": "en",
        }
        payload = convert_to_diarized_json(cog_output)
        self.assertEqual(payload["task"], "transcribe")
        self.assertEqual(len(payload["segments"]), 2)
        self.assertEqual(payload["segments"][0]["speaker"], "A")
        self.assertEqual(payload["segments"][1]["speaker"], "B")
        self.assertEqual(payload["segments"][0]["type"], "transcript.text.segment")
        self.assertEqual(payload["segments"][0]["id"], "0")

    def test_convert_to_diarized_json_known_speaker_names(self):
        cog_output = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello",
                    "speaker": "SPEAKER_00",
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"},
                    ],
                },
            ],
            "detected_language": "en",
        }
        payload = convert_to_diarized_json(cog_output, known_speaker_names=["alice"])
        self.assertEqual(payload["segments"][0]["speaker"], "alice")

    def test_diarize_validation_rejects_prompt(self):
        boundary, body = _encode_multipart(
            [("model", "gpt-4o-transcribe-diarize"), ("prompt", "hello")],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
        )
        fs = _parse_multipart(body, f"multipart/form-data; boundary={boundary}")
        _, err = validate_transcription_request(fs)
        self.assertIsNotNone(err)
        status, payload = err
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")

    def test_diarize_validation_rejects_timestamp_granularities(self):
        boundary, body = _encode_multipart(
            [
                ("model", "gpt-4o-transcribe-diarize"),
                ("timestamp_granularities[]", "word"),
            ],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
        )
        fs = _parse_multipart(body, f"multipart/form-data; boundary={boundary}")
        _, err = validate_transcription_request(fs)
        self.assertIsNotNone(err)
        self.assertEqual(err[0], 400)

    def test_diarize_validation_rejects_inconsistent_model(self):
        boundary, body = _encode_multipart(
            [("model", "whisper-1"), ("response_format", "diarized_json")],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
        )
        fs = _parse_multipart(body, f"multipart/form-data; boundary={boundary}")
        _, err = validate_transcription_request(fs)
        self.assertIsNotNone(err)
        self.assertEqual(err[0], 400)

    def test_diarize_validation_rejects_known_speaker_references(self):
        boundary, body = _encode_multipart(
            [
                ("model", "gpt-4o-transcribe-diarize"),
                ("known_speaker_references[]", "data:audio/wav;base64,abc"),
            ],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
        )
        fs = _parse_multipart(body, f"multipart/form-data; boundary={boundary}")
        _, err = validate_transcription_request(fs)
        self.assertIsNotNone(err)
        self.assertEqual(err[0], 400)
        self.assertIn("known_speaker_references", err[1]["error"]["message"])

    def test_diarize_validation_accepts_chunking_strategy(self):
        boundary, body = _encode_multipart(
            [
                ("model", "gpt-4o-transcribe-diarize"),
                ("response_format", "diarized_json"),
                ("chunking_strategy", "auto"),
            ],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
        )
        fs = _parse_multipart(body, f"multipart/form-data; boundary={boundary}")
        parsed, err = validate_transcription_request(fs)
        self.assertIsNone(err)
        self.assertTrue(parsed["is_diarize"])

    def test_build_cog_input_diarize_path(self):
        parsed = {
            "file_bytes": b"abc",
            "extension": "ogg",
            "whisper_model": "large-v3-turbo",
            "language": None,
            "prompt": None,
            "temperature": 0.0,
            "is_diarize": True,
            "known_speaker_names": [],
        }
        cog_input = build_cog_input(parsed)
        self.assertTrue(cog_input["diarization"])
        self.assertIsNone(cog_input["huggingface_access_token"])
        self.assertNotIn("initial_prompt", cog_input)

    def test_build_cog_input_non_diarize_unchanged(self):
        parsed = {
            "file_bytes": b"abc",
            "extension": "ogg",
            "whisper_model": "large-v3-turbo",
            "language": "fr",
            "prompt": "hello",
            "temperature": 0.0,
            "is_diarize": False,
            "known_speaker_names": [],
        }
        cog_input = build_cog_input(parsed)
        self.assertFalse(cog_input["diarization"])
        self.assertEqual(cog_input["initial_prompt"], "hello")

    def test_extension_allowlist_rejects_aac(self):
        boundary, body = _encode_multipart(
            [("model", "whisper-1")],
            [("file", "audio.aac", b"abc", "audio/aac")],
        )
        fs = _parse_multipart(body, f"multipart/form-data; boundary={boundary}")
        _, err = validate_transcription_request(fs)
        self.assertIsNotNone(err)
        status, payload = err
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["message"], "unsupported audio format")

    def test_extension_allowlist_accepts_ogg(self):
        boundary, body = _encode_multipart(
            [("model", "whisper-1")],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
        )
        fs = _parse_multipart(body, f"multipart/form-data; boundary={boundary}")
        parsed, err = validate_transcription_request(fs)
        self.assertIsNone(err)
        self.assertEqual(parsed["extension"], "ogg")

    def test_build_audio_data_uri(self):
        uri = build_audio_data_uri(b"hello", "ogg")
        self.assertTrue(uri.startswith("data:audio/ogg;base64,"))
        self.assertIn("aGVsbG8=", uri)

    def test_convert_json(self):
        payload = convert_to_json(MOCK_COG_OUTPUT)
        self.assertEqual(payload["text"], "Hello world How are you")

    def test_convert_text(self):
        self.assertEqual(convert_to_text(MOCK_COG_OUTPUT), "Hello world How are you")

    def test_convert_verbose_json_words_top_level(self):
        payload = convert_to_verbose_json(
            MOCK_COG_OUTPUT,
            timestamp_granularities=["word"],
        )
        self.assertIn("words", payload)
        self.assertNotIn("segments", payload)
        self.assertEqual(payload["words"][0]["word"], "Hello")
        self.assertNotIn("score", payload["words"][0])

    def test_convert_verbose_json_segments(self):
        payload = convert_to_verbose_json(
            MOCK_COG_OUTPUT,
            timestamp_granularities=["segment"],
            temperature=0.5,
        )
        self.assertIn("segments", payload)
        self.assertNotIn("words", payload)
        self.assertEqual(payload["segments"][0]["id"], 0)
        self.assertEqual(payload["segments"][0]["temperature"], 0.5)

    def test_convert_srt_and_vtt(self):
        srt = convert_to_srt(MOCK_COG_OUTPUT)
        self.assertIn("00:00:00,000 --> 00:00:02,500", srt)
        vtt = convert_to_vtt(MOCK_COG_OUTPUT)
        self.assertTrue(vtt.startswith("WEBVTT"))
        self.assertIn("00:00:00.000 --> 00:00:02.500", vtt)

    def test_convert_cog_output_formats(self):
        headers, body = convert_cog_output(MOCK_COG_OUTPUT, "json")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(json.loads(body.decode())["text"], "Hello world How are you")

        headers, body = convert_cog_output(MOCK_COG_OUTPUT, "text")
        self.assertEqual(headers["Content-Type"], "text/plain; charset=utf-8")
        self.assertEqual(body.decode(), "Hello world How are you")


class TestOpenAiCompatHttp(unittest.TestCase):
    def _request(self, fields, files, urlopen_fn=None, auth_token="test-token"):
        boundary, body = _encode_multipart(fields, files)
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "Authorization": f"Bearer {auth_token}",
        }
        return handle_transcription_request(
            headers,
            io.BytesIO(body),
            len(body),
            urlopen_fn=urlopen_fn or mock_cog_urlopen_factory(),
        )

    def test_json_response(self):
        status, resp_headers, body, _meta = self._request(
            [("model", "whisper-1"), ("response_format", "json")],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(resp_headers["Content-Type"], "application/json")
        self.assertEqual(json.loads(body.decode())["text"], "Hello world How are you")

    def test_text_response(self):
        status, resp_headers, body, _meta = self._request(
            [("model", "whisper-1"), ("response_format", "text")],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(resp_headers["Content-Type"], "text/plain; charset=utf-8")
        self.assertEqual(body.decode(), "Hello world How are you")

    def test_verbose_json_word_and_segment(self):
        status, _headers, body, _meta = self._request(
            [
                ("model", "whisper-1"),
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "word"),
            ],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
        )
        payload = json.loads(body.decode())
        self.assertEqual(status, 200)
        self.assertIn("words", payload)
        self.assertNotIn("segments", payload)

        status, _headers, body, _meta = self._request(
            [
                ("model", "whisper-1"),
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "segment"),
            ],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
        )
        payload = json.loads(body.decode())
        self.assertEqual(status, 200)
        self.assertIn("segments", payload)

    def test_missing_file_400(self):
        status, _headers, body, _meta = self._request(
            [("model", "whisper-1")],
            [],
        )
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body.decode())["error"]["message"], "file is required")

    def test_invalid_model_400(self):
        status, _headers, body, _meta = self._request(
            [("model", "invalid-model")],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
        )
        self.assertEqual(status, 400)
        self.assertIn("invalid-model", json.loads(body.decode())["error"]["message"])

    def test_auth_error_shape(self):
        status, payload = auth_error_response()
        self.assertEqual(status, 401)
        self.assertEqual(payload["error"]["type"], "authentication_error")
        self.assertEqual(payload["error"]["message"], "invalid api key")

    def test_oversized_file_413(self):
        with patch.object(openai_compat, "OPENAI_STT_MAX_FILE_SIZE_BYTES", 4):
            status, _headers, body, _meta = self._request(
                [("model", "whisper-1")],
                [("file", "audio.ogg", b"12345", "audio/ogg")],
            )
        self.assertEqual(status, 413)

    def test_unsupported_extension_422(self):
        status, _headers, body, _meta = self._request(
            [("model", "whisper-1")],
            [("file", "audio.aac", b"abc", "audio/aac")],
        )
        self.assertEqual(status, 422)

    def test_whisper1_cog_payload(self):
        urlopen_fn = mock_cog_urlopen_factory()
        _status, _headers, _body, meta = self._request(
            [("model", "whisper-1")],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
            urlopen_fn=urlopen_fn,
        )
        req = urlopen_fn.captured["requests"][0]
        payload = json.loads(req["body"].decode())
        self.assertEqual(payload["input"]["whisper_model"], "large-v3-turbo")
        self.assertTrue(payload["input"]["audio_file"].startswith("data:audio/ogg;base64,"))

    def test_language_forwarded(self):
        urlopen_fn = mock_cog_urlopen_factory()
        self._request(
            [("model", "whisper-1"), ("language", "fr")],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
            urlopen_fn=urlopen_fn,
        )
        payload = json.loads(urlopen_fn.captured["requests"][0]["body"].decode())
        self.assertEqual(payload["input"]["language"], "fr")

    def test_timeout_504(self):
        urlopen_fn = mock_cog_urlopen_factory(fail_status="timeout")
        status, _headers, body, _meta = self._request(
            [("model", "whisper-1")],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
            urlopen_fn=urlopen_fn,
        )
        self.assertEqual(status, 504)
        self.assertEqual(json.loads(body.decode())["error"]["message"], "transcription timed out")

    def test_sync_cog_post_no_async_header(self):
        urlopen_fn = mock_cog_urlopen_factory()
        self._request(
            [("model", "whisper-1")],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
            urlopen_fn=urlopen_fn,
        )
        req = urlopen_fn.captured["requests"][0]
        self.assertIn("/predictions", req["url"])
        headers = {k.lower(): v for k, v in req["headers"].items()}
        self.assertNotIn("prefer", headers)

    def test_diarized_json_response(self):
        cog_output = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello world",
                    "speaker": "SPEAKER_00",
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                        {"word": "world", "start": 1.0, "end": 2.5, "speaker": "SPEAKER_00"},
                    ],
                },
            ],
            "detected_language": "en",
        }
        urlopen_fn = mock_cog_urlopen_factory(output=cog_output)
        status, resp_headers, body, _meta = self._request(
            [
                ("model", "gpt-4o-transcribe-diarize"),
                ("response_format", "diarized_json"),
            ],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
            urlopen_fn=urlopen_fn,
        )
        self.assertEqual(status, 200)
        self.assertEqual(resp_headers["Content-Type"], "application/json")
        payload = json.loads(body.decode())
        self.assertEqual(payload["task"], "transcribe")
        self.assertIn("segments", payload)
        self.assertEqual(payload["segments"][0]["speaker"], "A")

    def test_diarize_cog_payload(self):
        urlopen_fn = mock_cog_urlopen_factory()
        self._request(
            [
                ("model", "gpt-4o-transcribe-diarize"),
                ("response_format", "diarized_json"),
            ],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
            urlopen_fn=urlopen_fn,
        )
        payload = json.loads(urlopen_fn.captured["requests"][0]["body"].decode())
        self.assertTrue(payload["input"]["diarization"])
        self.assertIsNone(payload["input"]["huggingface_access_token"])

    def test_diarize_model_json_response_no_segments(self):
        urlopen_fn = mock_cog_urlopen_factory()
        status, resp_headers, body, _meta = self._request(
            [
                ("model", "gpt-4o-transcribe-diarize"),
                ("response_format", "json"),
            ],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
            urlopen_fn=urlopen_fn,
        )
        self.assertEqual(status, 200)
        payload = json.loads(body.decode())
        self.assertIn("text", payload)
        self.assertNotIn("segments", payload)

    def test_cog_hf_token_failure_500(self):
        urlopen_fn = mock_cog_urlopen_factory(
            fail_status="failed",
        )
        # Override to return HF token error
        captured = {"requests": []}

        def urlopen(req, timeout=None):
            captured["requests"].append(req)
            return MockCogResponse(
                {
                    "status": "failed",
                    "error": "Diarization requested but no HuggingFace token available",
                }
            )

        urlopen.captured = captured
        status, _headers, body, _meta = self._request(
            [
                ("model", "gpt-4o-transcribe-diarize"),
                ("response_format", "diarized_json"),
            ],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
            urlopen_fn=urlopen,
        )
        self.assertEqual(status, 500)
        self.assertIn("transcription failed", json.loads(body.decode())["error"]["message"])


class BridgeHandlerStub(BaseHTTPRequestHandler):
    auth_token = "test-token"

    def log_message(self, format, *args):
        return

    def do_POST(self):
        if self.path != "/v1/audio/transcriptions":
            self.send_error(404)
            return
        auth_header = self.headers.get("Authorization")
        if auth_header != f"Bearer {self.auth_token}":
            status, payload = auth_error_response()
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        cl = int(self.headers.get("Content-Length", "0"))
        status, resp_headers, body, _meta = handle_transcription_request(
            self.headers,
            self.rfile,
            cl,
            urlopen_fn=mock_cog_urlopen_factory(),
        )
        self.send_response(status)
        for key, value in resp_headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TestBridgeOpenAiAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), BridgeHandlerStub)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=2)

    def _post(self, fields, files, token=None):
        import urllib.request

        boundary, body = _encode_multipart(fields, files)
        req = urllib.request.Request(
            f"{self.base_url}/v1/audio/transcriptions",
            data=body,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        if token is not None:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=5) as res:
                return res.status, res.read(), dict(res.headers)
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), dict(exc.headers)

    def test_no_auth_401(self):
        status, body, _headers = self._post(
            [("model", "whisper-1")],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
            token=None,
        )
        payload = json.loads(body.decode())
        self.assertEqual(status, 401)
        self.assertEqual(payload["error"]["type"], "authentication_error")

    def test_wrong_auth_401(self):
        status, body, _headers = self._post(
            [("model", "whisper-1")],
            [("file", "audio.ogg", b"abc", "audio/ogg")],
            token="wrong",
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body.decode())["error"]["message"], "invalid api key")


if __name__ == "__main__":
    unittest.main()
