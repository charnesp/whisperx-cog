"""Unit tests for Whisper model path resolution (no GPU)."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from model_paths import (
    BAKED_MODELS_ROOT,
    WHISPER_MODEL_HF_IDS,
    resolve_whisper_model_path,
    resolve_vad_source_path,
)


class TestResolveWhisperModelPath(unittest.TestCase):
    def test_prefers_baked_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            baked = Path(tmp) / "faster-whisper-large-v3-turbo"
            baked.mkdir()
            (baked / "model.bin").write_bytes(b"x")
            with mock.patch.dict(
                "model_paths.WHISPER_MODEL_LOCAL_PATHS",
                {"large-v3-turbo": [str(baked), "./models/faster-whisper-large-v3-turbo"]},
            ):
                self.assertEqual(
                    resolve_whisper_model_path("large-v3-turbo"),
                    str(baked),
                )

    def test_falls_back_to_relative_local_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            rel = Path(tmp) / "local-turbo"
            rel.mkdir()
            (rel / "model.bin").write_bytes(b"x")
            missing = Path(tmp) / "missing"
            with mock.patch.dict(
                "model_paths.WHISPER_MODEL_LOCAL_PATHS",
                {"large-v3-turbo": [str(missing), str(rel)]},
            ):
                self.assertEqual(
                    resolve_whisper_model_path("large-v3-turbo"),
                    str(rel),
                )

    def test_falls_back_to_hf_repo_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_a = Path(tmp) / "a"
            missing_b = Path(tmp) / "b"
            with mock.patch.dict(
                "model_paths.WHISPER_MODEL_LOCAL_PATHS",
                {"large-v3-turbo": [str(missing_a), str(missing_b)]},
            ):
                self.assertEqual(
                    resolve_whisper_model_path("large-v3-turbo"),
                    WHISPER_MODEL_HF_IDS["large-v3-turbo"],
                )

    def test_baked_root_constant(self):
        self.assertEqual(BAKED_MODELS_ROOT, "/models")


class TestResolveVadSourcePath(unittest.TestCase):
    def test_prefers_baked_vad(self):
        with tempfile.TemporaryDirectory() as tmp:
            baked = Path(tmp) / "vad" / "whisperx-vad-segmentation.bin"
            baked.parent.mkdir(parents=True)
            baked.write_bytes(b"vad")
            with mock.patch(
                "model_paths.VAD_LOCAL_CANDIDATES",
                [str(baked), "./models/vad/whisperx-vad-segmentation.bin"],
            ):
                self.assertEqual(resolve_vad_source_path(), str(baked))

    def test_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.bin"
            with mock.patch(
                "model_paths.VAD_LOCAL_CANDIDATES",
                [str(missing)],
            ):
                self.assertIsNone(resolve_vad_source_path())


if __name__ == "__main__":
    unittest.main()
