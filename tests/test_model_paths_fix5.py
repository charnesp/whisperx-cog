"""FIX 5 (🟡, E5-CODE review): no dead constants.

models_registry._WHISPER_KEYS/_QWEN_MODEL_KEY/_QWEN_ALIGNER_KEY and
model_paths._WHISPER_KEYS are unused or shadowed: model_paths must derive
its key lists from the registry's MODELS (no second literal).

(E5-LEGACY-HF): the error-message prose tests are gone with the fail-hard
raise itself — resolution falls back to the HF repo id, nothing to allege.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))


WHISPER_KEYS = ("large-v3", "large-v3-turbo", "tiny")  # non-qwen registry keys


class TestNoDeadConstants(unittest.TestCase):
    def test_models_registry_has_no_unused_private_keys(self):

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


if __name__ == "__main__":
    unittest.main()
