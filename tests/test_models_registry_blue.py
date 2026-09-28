"""GPU-free registry tests for BLUE-factorized constants (E5-CODE-1 T1 BLUE).

After the refactor predict.py keeps only thin back-compat aliases; the
tests already assert (test_models_registry.py) that these aliases equal
the registry values. This file adds the anti-regression guard: predict.py
must not re-declare duplicated literal lists (the duplication the registry
removes), i.e. the constants must be DERIVED, single-assignment.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from _predict_stub import install  # noqa: E402


class TestPredictConstantsAreDerived(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predict = install()

    def test_predict_declares_each_model_constant_once(self):
        """GREEN left module-level derived constants; BLUE must keep a single
        assignment per constant (no duplicated literal fallback blocks)."""
        src = (REPO_ROOT / "predict.py").read_text()
        for name in (
            "QWEN_ASR_WEIGHT_FILES",
            "QWEN_ALIGNER_WEIGHT_FILES",
            "QWEN_ASR_HF_REPO",
            "QWEN_ALIGNER_HF_REPO",
        ):
            count = sum(
                1
                for line in src.splitlines()
                if line.startswith(name) and "=" in line
            )
            self.assertEqual(count, 1, f"{name} assigned {count}x in predict.py")

    def test_predict_constants_match_registry_specs(self):
        from models_registry import MODELS

        predict = self.predict
        self.assertEqual(
            predict.QWEN_ASR_HF_REPO, MODELS["qwen3-asr-1.7b"].hf_repo
        )
        self.assertEqual(
            predict.QWEN_ALIGNER_HF_REPO,
            MODELS["qwen3-forced-aligner-0.6b"].hf_repo,
        )
        self.assertEqual(
            predict.QWEN_ASR_WEIGHT_FILES,
            list(MODELS["qwen3-asr-1.7b"].weight_files),
        )
        self.assertEqual(
            predict.QWEN_ALIGNER_WEIGHT_FILES,
            list(MODELS["qwen3-forced-aligner-0.6b"].weight_files),
        )


if __name__ == "__main__":
    unittest.main()
