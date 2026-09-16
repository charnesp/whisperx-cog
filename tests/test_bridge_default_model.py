"""E6-BRIDGE-DEFAULT-MODEL: "whisper-1" routes to BRIDGE_DEFAULT_MODEL.

Charles decision (option 2): the bridge default model is an ENV variable,
not hard-coded code. Contract:
(a) BRIDGE_DEFAULT_MODEL = os.environ.get("BRIDGE_DEFAULT_MODEL", "large-v3-turbo")
(b) the "whisper-1" entry is removed from the hard-coded MODEL_MAP dict and
    routed dynamically to BRIDGE_DEFAULT_MODEL
(c) other MODEL_MAP entries unchanged
(d) the model guard accepts "whisper-1" and BRIDGE_DEFAULT_MODEL
(e) hotwords routing (QWEN_MODELS) applies to the RESOLVED model
(f) is_diarize_request("whisper-1", "diarized_json") keeps working
(g) docs/BRIDGE.md documents BRIDGE_DEFAULT_MODEL
"""

from __future__ import annotations

import io
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bridge"))

import openai_compat
from openai_compat import (
    MODEL_MAP,
    build_cog_input,
    is_diarize_request,
    validate_transcription_request,
)


def _encode_multipart(fields, files):
    """Encode a multipart/form-data body (same helper as test_openai_stt)."""
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
    fs, err = openai_compat.parse_multipart_form(
        {"Content-Type": content_type},
        io.BytesIO(body_bytes),
        len(body_bytes),
    )
    if err:
        raise AssertionError(err)
    return fs


class TestBridgeDefaultModel(unittest.TestCase):
    """E6: whisper-1 alias routes to the BRIDGE_DEFAULT_MODEL env variable."""

    def _fs(self, model="whisper-1", **fields):
        names = [("model", model)]
        names.extend(fields.items())
        boundary, body = _encode_multipart(
            names,
            [("file", "audio.ogg", b"abc", "audio/ogg")],
        )
        return _parse_multipart(body, f"multipart/form-data; boundary={boundary}")

    def test_model_map_does_not_hardcode_whisper1(self):
        """(b) anti-regression: MODEL_MAP no longer contains the whisper-1 alias."""
        self.assertNotIn("whisper-1", MODEL_MAP)

    def test_whisper1_routes_to_bridge_default_model_env(self):
        """(b) BRIDGE_DEFAULT_MODEL=qwen3-asr -> whisper-1 resolves to qwen3-asr."""
        with mock.patch.dict(os.environ, {"BRIDGE_DEFAULT_MODEL": "qwen3-asr"}):
            parsed, err = validate_transcription_request(self._fs())
        self.assertIsNone(err)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["whisper_model"], "qwen3-asr")

    def test_whisper1_unset_env_defaults_to_large_v3_turbo(self):
        """(a) BRIDGE_DEFAULT_MODEL unset -> original large-v3-turbo default."""
        env = {k: v for k, v in os.environ.items() if k != "BRIDGE_DEFAULT_MODEL"}
        with mock.patch.dict(os.environ, env, clear=True):
            parsed, err = validate_transcription_request(self._fs())
        self.assertIsNone(err)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["whisper_model"], "large-v3-turbo")

    def test_whisper1_hotwords_follow_resolved_model(self):
        """(e) hotwords routing applies to the RESOLVED model: whisper-1 with
        BRIDGE_DEFAULT_MODEL=qwen3-asr passes hotwords as qwen context."""
        with mock.patch.dict(os.environ, {"BRIDGE_DEFAULT_MODEL": "qwen3-asr"}):
            parsed, err = validate_transcription_request(
                self._fs(hotwords="Backblaze, Supabase")
            )
            self.assertIsNone(err)
            cog_input = build_cog_input(parsed)
        self.assertEqual(parsed["whisper_model"], "qwen3-asr")
        self.assertEqual(cog_input["hotwords"], "Backblaze, Supabase")
        self.assertEqual(cog_input["whisper_model"], "qwen3-asr")

    def test_whisper1_default_whisper_path_hotwords_stay_none(self):
        """Whisper-path invariance: BRIDGE_DEFAULT_MODEL unset (turbo) keeps
        hotwords: None on the whisper path."""
        env = {k: v for k, v in os.environ.items() if k != "BRIDGE_DEFAULT_MODEL"}
        with mock.patch.dict(os.environ, env, clear=True):
            parsed, err = validate_transcription_request(
                self._fs(hotwords="Backblaze")
            )
            self.assertIsNone(err)
            cog_input = build_cog_input(parsed)
        self.assertEqual(parsed["whisper_model"], "large-v3-turbo")
        self.assertIsNone(cog_input["hotwords"])

    def test_enable_qwen_kill_switch_blocks_resolved_qwen_alias(self):
        """Defense in depth: whisper-1 resolved to qwen3-asr is gated by
        ENABLE_QWEN=0 at the bridge (400) instead of a Cog-side 500."""
        with mock.patch.dict(
            os.environ,
            {"BRIDGE_DEFAULT_MODEL": "qwen3-asr", "ENABLE_QWEN": "0"},
        ):
            parsed, err = validate_transcription_request(self._fs())
        self.assertIsNone(parsed)
        self.assertIsNotNone(err)
        status, payload = err
        self.assertEqual(status, 400)

    def test_guard_accepts_whisper1_alias(self):
        """(d) the model guard accepts whisper-1 even though it left MODEL_MAP."""
        env = {k: v for k, v in os.environ.items() if k != "BRIDGE_DEFAULT_MODEL"}
        with mock.patch.dict(os.environ, env, clear=True):
            parsed, err = validate_transcription_request(self._fs())
        self.assertIsNone(err)
        self.assertIsNotNone(parsed)

    def test_guard_accepts_bridge_default_model_value(self):
        """(d) the guard accepts a client model equal to BRIDGE_DEFAULT_MODEL."""
        with mock.patch.dict(os.environ, {"BRIDGE_DEFAULT_MODEL": "qwen3-asr"}):
            parsed, err = validate_transcription_request(self._fs(model="qwen3-asr"))
        self.assertIsNone(err)
        self.assertEqual(parsed["whisper_model"], "qwen3-asr")

    def test_is_diarize_request_whisper1_diarized_json(self):
        """(f) diarize routing by response_format keeps working for whisper-1."""
        self.assertTrue(is_diarize_request("whisper-1", "diarized_json"))
        self.assertFalse(is_diarize_request("whisper-1", "json"))

    def test_language_passthrough_unchanged(self):
        """Client language is forwarded to the cog unchanged on the alias path."""
        with mock.patch.dict(os.environ, {"BRIDGE_DEFAULT_MODEL": "qwen3-asr"}):
            parsed, err = validate_transcription_request(self._fs(language="fr"))
            self.assertIsNone(err)
            cog_input = build_cog_input(parsed)
        self.assertEqual(cog_input["language"], "fr")

    def test_other_model_map_entries_unchanged(self):
        """(c) non-whisper-1 entries keep their values."""
        self.assertEqual(MODEL_MAP["gpt-4o-transcribe-diarize"], "large-v3-turbo")
        self.assertEqual(MODEL_MAP["large-v3"], "large-v3")
        self.assertEqual(MODEL_MAP["large-v3-turbo"], "large-v3-turbo")
        self.assertEqual(MODEL_MAP["tiny"], "tiny")
        self.assertEqual(MODEL_MAP["qwen3-asr"], "qwen3-asr")

    def test_module_default_constant(self):
        """(a) module constant defaults to large-v3-turbo when env unset."""
        env = {k: v for k, v in os.environ.items() if k != "BRIDGE_DEFAULT_MODEL"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                openai_compat.os.environ.get(
                    "BRIDGE_DEFAULT_MODEL",
                    openai_compat.BRIDGE_DEFAULT_MODEL,
                ),
                "large-v3-turbo",
            )


if __name__ == "__main__":
    unittest.main()
