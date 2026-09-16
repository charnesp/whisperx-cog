"""Unit tests for Whisper model path resolution (no GPU)."""

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
    def test_online_mode_returns_hf_repo_id(self):
        """MODELS_MODE=online is the explicit opt-in for the HF repo id."""
        with mock.patch.dict(
            "os.environ", {"MODELS_MODE": "online"}
        ):
            self.assertEqual(
                resolve_whisper_model_path("large-v3-turbo"),
                WHISPER_MODEL_HF_IDS["large-v3-turbo"],
            )

    def test_default_mode_has_no_hf_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                "os.environ", {"MODELS_DIR": str(tmp), "MODELS_LOCK_PATH": str(Path(tmp) / "absent.lock")}
            ):
                from model_paths import ModelNotProvisioned

                with self.assertRaises(ModelNotProvisioned):
                    resolve_whisper_model_path("large-v3-turbo")

    def test_env_override_wins(self):
        import tempfile as _t

        with _t.TemporaryDirectory() as tmp:
            baked = Path(tmp) / "faster-whisper-large-v3-turbo"
            baked.mkdir()
            (baked / "model.bin").write_bytes(b"x")
            with mock.patch.dict(
                "os.environ", {"MODELS_MODE": "online", "LARGE_V3_TURBO_PATH": str(baked)}
            ):
                self.assertEqual(
                    resolve_whisper_model_path("large-v3-turbo"),
                    str(baked),
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
