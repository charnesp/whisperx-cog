"""Unit tests for HuggingFace token resolution in predict.py (no GPU)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hf_token import require_diarization_token, resolve_huggingface_token


class TestResolveHuggingfaceToken(unittest.TestCase):
    def test_input_token_takes_precedence(self):
        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "env-token"}):
            self.assertEqual(resolve_huggingface_token("input-token"), "input-token")

    def test_env_fallback_when_input_empty(self):
        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "env-token"}):
            self.assertEqual(resolve_huggingface_token(None), "env-token")
            self.assertEqual(resolve_huggingface_token(""), "env-token")
            self.assertEqual(resolve_huggingface_token("  "), "env-token")

    def test_returns_none_when_no_token(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_huggingface_token(None))
        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": ""}):
            self.assertIsNone(resolve_huggingface_token(None))


class TestRequireDiarizationToken(unittest.TestCase):
    def test_no_diarization_returns_none(self):
        self.assertIsNone(require_diarization_token(False, None))

    def test_diarization_uses_env_fallback(self):
        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "env-token"}):
            self.assertEqual(require_diarization_token(True, None), "env-token")

    def test_diarization_fails_hard_without_token(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                require_diarization_token(True, None)
            self.assertIn("HuggingFace token", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
