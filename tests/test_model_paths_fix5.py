"""FIX 5 (🟡, E5-CODE review): cleanups — dead constants + error message.

(a) models_registry._WHISPER_KEYS/_QWEN_MODEL_KEY/_QWEN_ALIGNER_KEY and
    model_paths._WHISPER_KEYS are unused or shadowed: model_paths must
    derive its key lists from the registry's MODELS (no second literal).
(b) model_paths.resolve_whisper_model_path's error claims "legacy flat
    dirs also unavailable: <candidates>" — but those dirs were NOT
    individually checked (only ./models/<dirname> under MODELS_MODE=dev
    was). The message must only allege the paths actually tried.

RED du cycle : les tests échouent tant que les constantes mortes
existent et tant que le message allègue des chemins non testés.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from models_registry import MODELS  # noqa: E402

from test_model_paths_failhard import mock_env  # noqa: E402

WHISPER_KEYS = tuple(
    k for k in sorted(MODELS) if k != "qwen3-forced-aligner-0.6b"
)


class TestNoDeadConstants(unittest.TestCase):
    def test_models_registry_has_no_unused_private_keys(self):
        import models_registry

        src = (REPO_ROOT / "models_registry.py").read_text()
        self.assertNotIn("_WHISPER_KEYS", src)
        self.assertNotIn("_QWEN_MODEL_KEY", src)
        self.assertNotIn("_QWEN_ALIGNER_KEY", src)

    def test_model_paths_has_no_private_key_literal(self):
        src = (REPO_ROOT / "model_paths.py").read_text()
        self.assertNotIn(
            "_WHISPER_KEYS = (",
            src,
            "model_paths must derive its keys from the registry, not a "
            "second literal",
        )

    def test_model_paths_whisper_lists_cover_whisper_keys(self):
        import model_paths as mp

        self.assertEqual(sorted(mp.WHISPER_MODEL_HF_IDS), sorted(WHISPER_KEYS))
        self.assertEqual(
            sorted(mp.WHISPER_MODEL_LOCAL_PATHS), sorted(WHISPER_KEYS)
        )
        self.assertEqual(sorted(mp.ENV_OVERRIDES), sorted(WHISPER_KEYS))


class TestErrorMessageAllegesOnlyTestedPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "models"
        self.root.mkdir()

    def test_no_legacy_clause_when_baked_dir_absent(self):
        import model_paths as mp

        with mock_env(**{"MODELS_DIR": str(self.root), "MODELS_MODE": ""}):
            with self.assertRaises(mp.ModelNotProvisioned) as ctx:
                mp.resolve_whisper_model_path("tiny")
        text = str(ctx.exception)
        self.assertNotIn(
            "legacy flat dirs also unavailable",
            text,
            "the error must not allege legacy dirs it never checked",
        )

    def test_error_lists_the_tried_lock_path(self):
        import model_paths as mp
        import models_lock

        with mock_env(**{"MODELS_DIR": str(self.root), "MODELS_MODE": ""}):
            with self.assertRaises(mp.ModelNotProvisioned) as ctx:
                mp.resolve_whisper_model_path("tiny")
        lock = models_lock.parse_lock(models_lock.DEFAULT_LOCK_PATH)
        tried = f"{self.root}/tiny/{lock['tiny']['revision']}"
        self.assertIn(tried, str(ctx.exception))
        self.assertIn("docker run --rm", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()