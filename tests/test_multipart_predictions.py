"""Tests for multipart/form-data support on POST /predictions (bridge)."""
from __future__ import annotations

import io
import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bridge"))
import bridge  # noqa: E402
from openai_compat import (  # noqa: E402
    _parse_multipart_body,
    build_audio_data_uri,
)


def _encode_multipart(fields, files):
    boundary = "----multipartpredboundary"
    body = io.BytesIO()
    for name, value in fields:
        body.write(f"--{boundary}\r\n".encode())
        body.write(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        )
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


class TestMultipartPredHelpers(unittest.TestCase):
    def test_build_audio_data_uri_wav(self):
        uri = build_audio_data_uri(b"RIFF....", "wav")
        self.assertTrue(uri.startswith("data:audio/wav;base64,"))

    def test_build_audio_data_uri_unknown_ext_falls_back_octet_stream(self):
        uri = build_audio_data_uri(b"xx", "weird")
        self.assertTrue(uri.startswith("data:application/octet-stream;base64,"))

    def test_parse_multipart_body_file_and_fields(self):
        boundary, body = _encode_multipart(
            fields=[("language", "fr")],
            files=[("file", "audio.wav", b"RIFF-data", "audio/wav")],
        )
        fs = _parse_multipart_body(
            f"multipart/form-data; boundary={boundary}", body
        )
        self.assertIn("file", fs)
        self.assertIn("language", fs)
        self.assertEqual(fs["file"].filename, "audio.wav")
        self.assertEqual(fs["file"].file.read(), b"RIFF-data")
        self.assertEqual(fs["language"].value, "fr")


class _RecordingCogHandler(BaseHTTPRequestHandler):
    """Minimal Cog stand-in: captures the POST body, answers 200."""

    captured: dict = {}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _RecordingCogHandler.captured = {
            "path": self.path,
            "body": body,
            "content_type": self.headers.get("Content-Type"),
        }
        resp = json.dumps({"id": "pred123", "status": "starting"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, format, *args):
        return


class TestMultipartPredictionsEndToEnd(unittest.TestCase):
    """Full HTTP round trip through a real bridge instance on an ephemeral port."""

    def setUp(self):
        _RecordingCogHandler.captured = {}
        self.cog_port = _free_port()
        self.cog = HTTPServer(("127.0.0.1", self.cog_port), _RecordingCogHandler)
        self.cog_thread = threading.Thread(target=self.cog.serve_forever, daemon=True)
        self.cog_thread.start()

        self._old_cog_url = bridge.COG_URL
        bridge.COG_URL = f"http://127.0.0.1:{self.cog_port}"

        self.bridge_port = _free_port()
        self.httpd = bridge.ThreadedHTTPServer(
            ("127.0.0.1", self.bridge_port), bridge.ReplicateCompatibleBridge
        )
        self.httpd_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.httpd_thread.start()

    def tearDown(self):
        bridge.COG_URL = self._old_cog_url
        self.httpd.shutdown()
        self.cog.shutdown()
        self.httpd.server_close()
        self.cog.server_close()

    def _post(self, body: bytes, content_type: str, headers_extra: dict | None = None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.bridge_port}/predictions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {bridge.AUTH_TOKEN}",
                "Content-Type": content_type,
            },
        )
        if headers_extra:
            for k, v in headers_extra.items():
                req.add_header(k, v)
        return urllib.request.urlopen(req, timeout=10)

    def test_multipart_predictions_converts_file_to_data_uri(self):
        boundary, body = _encode_multipart(
            fields=[("language", "fr")],
            files=[("file", "audio.wav", b"RIFF-data", "audio/wav")],
        )
        resp = self._post(body, f"multipart/form-data; boundary={boundary}")
        self.assertEqual(resp.status, 200)
        payload = json.loads(resp.read())
        self.assertEqual(payload["id"], "pred123")
        self.assertEqual(payload["status"], "starting")

        captured = _RecordingCogHandler.captured
        self.assertEqual(captured["path"], "/predictions")
        sent = json.loads(captured["body"])
        self.assertIn("input", sent)
        self.assertTrue(
            sent["input"]["audio_file"].startswith("data:audio/wav;base64,")
        )
        self.assertEqual(sent["input"]["language"], "fr")
        self.assertIn("webhook", sent)
        self.assertIn("id", sent)
        # Cog received JSON, not multipart
        self.assertEqual(captured["content_type"], "application/json")

    def test_multipart_missing_file_returns_400(self):
        boundary, body = _encode_multipart(fields=[("language", "fr")], files=[])
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post(body, f"multipart/form-data; boundary={boundary}")
        self.assertEqual(ctx.exception.code, 400)
        payload = json.loads(ctx.exception.read())
        self.assertEqual(payload["error"], "missing_audio_file")

    def test_multipart_empty_file_returns_400(self):
        boundary, body = _encode_multipart(
            fields=[], files=[("file", "audio.wav", b"", "audio/wav")]
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post(body, f"multipart/form-data; boundary={boundary}")
        self.assertEqual(ctx.exception.code, 400)
        payload = json.loads(ctx.exception.read())
        self.assertEqual(payload["error"], "missing_audio_file")

    def test_multipart_unsupported_extension_returns_422(self):
        boundary, body = _encode_multipart(
            fields=[], files=[("file", "notes.txt", b"hello", "text/plain")]
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post(body, f"multipart/form-data; boundary={boundary}")
        self.assertEqual(ctx.exception.code, 422)
        payload = json.loads(ctx.exception.read())
        self.assertEqual(payload["error"], "unsupported_audio_format")

    def test_multipart_oversized_file_returns_413(self):
        big = b"x" * (26 * 1024 * 1024)  # above the 25MB default
        boundary, body = _encode_multipart(
            fields=[], files=[("file", "audio.wav", big, "audio/wav")]
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post(body, f"multipart/form-data; boundary={boundary}")
        self.assertEqual(ctx.exception.code, 413)
        payload = json.loads(ctx.exception.read())
        self.assertEqual(payload["error"], "payload_too_large")

    def test_multipart_invalid_form_returns_flat_error(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post(b"not a multipart body", "multipart/form-data; boundary=")
        self.assertEqual(ctx.exception.code, 400)
        payload = json.loads(ctx.exception.read())
        self.assertEqual(payload["error"], "invalid_multipart_form")
        self.assertIn("detail", payload)

    def test_multipart_json_values_are_coerced(self):
        boundary, body = _encode_multipart(
            fields=[("min_speakers", "2"), ("language", "fr")],
            files=[("file", "audio.wav", b"RIFF-data", "audio/wav")],
        )
        resp = self._post(body, f"multipart/form-data; boundary={boundary}")
        self.assertEqual(resp.status, 200)
        sent = json.loads(_RecordingCogHandler.captured["body"])
        self.assertEqual(sent["input"]["min_speakers"], 2)
        self.assertEqual(sent["input"]["language"], "fr")

    def test_multipart_duplicated_file_parts_prefers_data_part(self):
        boundary, body = _encode_multipart(
            fields=[],
            files=[
                ("file", "", b"", "text/plain"),
                ("file", "audio.wav", b"RIFF-data", "audio/wav"),
            ],
        )
        resp = self._post(body, f"multipart/form-data; boundary={boundary}")
        self.assertEqual(resp.status, 200)
        sent = json.loads(_RecordingCogHandler.captured["body"])
        self.assertTrue(sent["input"]["audio_file"].startswith("data:audio/wav;base64,"))


def _free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    unittest.main()