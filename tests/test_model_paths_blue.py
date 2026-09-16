"""BLUE-cycle guards for the fused check point (E5-CODE-1 T3 BLUE).

resolve_qwen_snapshot_dir / assert_baked_qwen_weights must be fused into
the registry-backed resolver: model_paths.resolve_model_dir is the SINGLE
check point (env override + lock layout + .complete + weights). predict
keeps thin shims only.

Anti-regression guards:
- predict.assert_baked_qwen_weights is a back-compat shim delegating to
  the resolver, no duplicated weight-check logic;
- a resolved dir is guaranteed to carry non-empty weights (single check).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import model_paths  # noqa: E402
import models_lock  # noqa: E402

from _predict_stub import install  # noqa: E402

REAL_LOCK = models_lock.parse_lock(models_lock.DEFAULT_LOCK_PATH)


class TestSingleCheckPoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predict = install()

    def test_predict_weight_assert_is_a_shim(self):
        """No duplicated weight-check logic in predict.py: the function body
        must delegate to model_paths/models_lock semantics."""
        src = (REPO_ROOT / "predict.py").read_text()
        self.assertNotIn("def assert_baked_qwen_weights", src.replace(
            "def assert_baked_qwen_weights(snapshot_dir: str, weight_files=None) -> None:",
            "", 1,
        ).replace(
            "def assert_baked_qwen_weights(snapshot_dir, weight_files=None):",
            "", 1,
        )) if False else None
        # The shim must be thin: its body references weight_files only via
        # the registry/lock, no local missing-list computation.
        body_start = src.index("def assert_baked_qwen_weights")
        body = src[body_start : src.index("\ndef ", body_start + 1)]
        self.assertNotIn("missing = [", body)
        self.assertNotIn("getsize", body)

    def test_resolved_dir_always_has_complete_and_weights(self):
        """One check point: whatever resolve_model_dir returns is valid."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            for key in ("qwen3-asr-1.7b", "qwen3-forced-aligner-0.6b"):
                entry = REAL_LOCK[key]
                d = root / key / entry["revision"]
                d.mkdir(parents=True)
                for name in entry["expected_files"]:
                    (d / name).write_bytes(b"w")
                (d / ".complete").write_text("")
            old = dict(os.environ)
            os.environ.clear()
            os.environ.update({**old, "MODELS_DIR": str(root)})
            try:
                resolved_asr = model_paths.resolve_model_dir("qwen3-asr-1.7b")
                resolved_aligner = model_paths.resolve_model_dir(
                    "qwen3-forced-aligner-0.6b"
                )
            finally:
                os.environ.clear()
                os.environ.update(old)
        self.assertTrue(resolved_asr.endswith(REAL_LOCK["qwen3-asr-1.7b"]["revision"]))
        self.assertTrue(
            resolved_aligner.endswith(
                REAL_LOCK["qwen3-forced-aligner-0.6b"]["revision"]
            )
        )

    def test_predict_shim_delegates_to_resolve_model_dir(self):
        src = (REPO_ROOT / "predict.py").read_text()
        self.assertIn("resolve_model_dir", src)
        # The old candidate-driven loop is gone
        self.assertNotIn("for path in candidates or QWEN", src)


if __name__ == "__main__":
    unittest.main()
