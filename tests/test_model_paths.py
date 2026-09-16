"""Unit tests for model resolution (no GPU) — E5-LEGACY-HF contract.

New contract (final Charles decision): the resolution prefers the
provisioned revisioned dir (/models/<key>/<sha40>/ + .complete, per
models.lock), and falls back to the HF repo id when the model is absent —
runtime download like any HF model, exactly the pre-E5 legacy behavior.
No ModelNotProvisioned, no MODELS_MODE, no remediation prose.
"""

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
    def test_falls_back_to_hf_repo_id_when_unprovisioned(self):
        """RED (E5-LEGACY-HF): no provisioned dir => HF repo id (legacy
        runtime-download behavior), never a raise."""
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                "os.environ",
                {"MODELS_DIR": str(tmp), "MODELS_LOCK_PATH": str(Path(tmp) / "absent.lock")},
            ):
                self.assertEqual(
                    resolve_whisper_model_path("large-v3-turbo"),
                    WHISPER_MODEL_HF_IDS["large-v3-turbo"],
                )

    def test_provisioned_dir_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            baked = Path(tmp) / "faster-whisper-large-v3-turbo"
            baked.mkdir()
            (baked / "model.bin").write_bytes(b"x")
            with mock.patch.dict(
                "os.environ", {"LARGE_V3_TURBO_PATH": str(baked)}
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
