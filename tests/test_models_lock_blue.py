"""BLUE-cycle guards for models_lock (E5-CODE-1 T2 BLUE).

Refactor already green; these tests lock the invariants that must survive
any factorization:
- fast_validate covers EVERY model required by the lock (all 5 keys), not
  just the first one that fails: a partially provisioned volume must
  report the exact first missing model, and a fully provisioned volume
  passes only when all 5 are present;
- the error text never embeds any environment secret value (HF_TOKEN).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import models_lock  # noqa: E402

REAL_LOCK = models_lock.parse_lock(models_lock.DEFAULT_LOCK_PATH)


def _build(root: Path, key: str) -> None:
    entry = REAL_LOCK[key]
    d = root / key / entry["revision"]
    d.mkdir(parents=True)
    # v2 lock carries per-file sizes: provisioning means the real byte
    # count (the boot validator enforces size since E5-LOCK-SYNC).
    sizes = entry.get("sizes", {})
    for name in entry["expected_files"]:
        # Sparse write (os.truncate): the v2 lock declares the REAL byte
        # sizes (model.bin up to 3GB) — allocating b"w"*size in RAM would
        # OOM the runner; fast_validate only os.stats each file.
        with open(d / name, "wb") as fh:
            fh.truncate(sizes.get(name, 1))
    (d / ".complete").write_text("")


class TestValidatorCoversAllRequiredModels(unittest.TestCase):
    def test_all_5_models_required_and_validated(self):
        """A volume provisioned for 4/5 models must still fail (5th missing)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            keys = list(REAL_LOCK)
            for key in keys[:-1]:
                _build(root, key)
            with self.assertRaises(models_lock.ModelsNotProvisioned) as ctx:
                models_lock.fast_validate(root, REAL_LOCK)
            self.assertIn(f"model={keys[-1]}", str(ctx.exception))

    def test_fully_provisioned_volume_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            for key in REAL_LOCK:
                _build(root, key)
            models_lock.fast_validate(root, REAL_LOCK)  # must not raise

    def test_empty_volume_reports_first_lock_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            with self.assertRaises(models_lock.ModelsNotProvisioned) as ctx:
                models_lock.fast_validate(root, REAL_LOCK)
            self.assertIn(f"model={list(REAL_LOCK)[0]}", str(ctx.exception))

    def test_default_lock_covers_exactly_the_5_registry_keys(self):
        from models_registry import MODELS

        self.assertEqual(set(REAL_LOCK), set(MODELS))


class TestNoSecretInBootError(unittest.TestCase):
    def test_hf_token_value_never_in_validation_error(self):
        secret = "hf_BLUEleakcheck_value_98765"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            old = dict(os.environ)
            os.environ.clear()
            os.environ.update({**old, "HF_TOKEN": secret})
            try:
                with self.assertRaises(models_lock.ModelsNotProvisioned) as ctx:
                    models_lock.fast_validate(root, REAL_LOCK)
            finally:
                os.environ.clear()
                os.environ.update(old)
        self.assertNotIn(secret, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
