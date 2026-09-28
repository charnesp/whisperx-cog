"""Unit tests for scripts/lock_audit.py (models.lock v2 — sha256 per file).

GPU-free: the HF tree API responses are fixtures (recorded 2026-09-16, see
tmp/stt/hf_trees_e5.json for the live raw data). No network in unit tests.
"""

import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml

from scripts.lock_audit import (
    LockDrift,
    build_lock_from_tree,
    check_against_lock,
    load_lock_v2,
    tree_entry_to_file_spec,
)


# ---------------------------------------------------------------------------
# Fixtures


def fake_tree() -> dict:
    """A trimmed HF tree API response (one LFS + one plain file)."""
    return {
        "repo": "Systran/faster-whisper-tiny",
        "rev": "d90ca5fe260221311c53c58e660288d3deb8d356",
        "files": [
            {
                "path": ".gitattributes",
                "size": 1477,
                "lfs": False,
                "sha256": None,
            },
            {
                "path": "README.md",
                "size": 1991,
                "lfs": False,
                "sha256": None,
            },
            {
                "path": "config.json",
                "size": 2249,
                "lfs": False,
                "sha256": None,
            },
            {
                "path": "model.bin",
                "size": 75538270,
                "lfs": True,
                "sha256": "dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1",
            },
            {
                "path": "tokenizer.json",
                "size": 2203239,
                "lfs": False,
                "sha256": None,
            },
            {
                "path": "vocabulary.txt",
                "size": 459861,
                "lfs": False,
                "sha256": None,
            },
        ],
    }


def fake_lock_v2() -> str:
    return textwrap.dedent(
        """\
        version: 2
        models:
          - repo: Systran/faster-whisper-tiny
            revision: d90ca5fe260221311c53c58e660288d3deb8d356
            required: true
            files:
              - path: model.bin
                size: 75538270
                sha256: dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1
                lfs: true
              - path: config.json
                size: 2249
                sha256: "1111111111111111111111111111111111111111111111111111111111111111"
                lfs: false
        """
    )


class TestTreeToLock(unittest.TestCase):
    def test_tree_entry_to_file_spec_lfs(self):
        spec = tree_entry_to_file_spec(
            {"path": "model.bin", "size": 10, "lfs": True, "sha256": "ab" * 32}
        )
        self.assertEqual(spec, {"path": "model.bin", "size": 10, "sha256": "ab" * 32, "lfs": True})

    def test_tree_entry_plain_needs_download(self):
        # Non-LFS files expose no sha256 in the tree API: the audit tool
        # downloads once and hashes (schema stays uniform sha256).
        spec = tree_entry_to_file_spec({"path": "config.json", "size": 2249, "lfs": False})
        self.assertEqual(spec["path"], "config.json")
        self.assertEqual(spec["size"], 2249)
        self.assertEqual(spec["lfs"], False)

    def test_build_lock_from_tree(self):
        m = build_lock_from_tree(
            fake_tree(),
            required=True,
            excluded=lambda p: p in {"README.md", ".gitattributes"},
        )
        lock = {"version": 2, "models": [m]}
        self.assertEqual(lock["version"], 2)
        self.assertEqual(m["repo"], "Systran/faster-whisper-tiny")
        self.assertEqual(m["revision"], "d90ca5fe260221311c53c58e660288d3deb8d356")
        self.assertTrue(m["required"])
        paths = [f["path"] for f in m["files"]]
        self.assertIn("model.bin", paths)
        self.assertIn("config.json", paths)
        self.assertIn("vocabulary.txt", paths, "CT2 tiny vocabulary.txt must NOT be normalized")
        self.assertNotIn("README.md", paths)
        model_bin = next(f for f in m["files"] if f["path"] == "model.bin")
        self.assertEqual(
            model_bin["sha256"],
            "dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1",
        )


class TestCheckAgainstLock(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.lock_path = self.tmp / "models.lock"
        self.lock_path.write_text(fake_lock_v2())

    def tearDown(self):
        self._tmp.cleanup()

    def test_load_lock_v2(self):
        lock = load_lock_v2(self.lock_path)
        self.assertEqual(lock["version"], 2)
        self.assertEqual(len(lock["models"]), 1)

    def test_check_passes_on_matching_local_tree(self):
        # Local tree with correct sha256 (LFS) and matching sizes.
        local = self.tmp / "tiny"
        local.mkdir()
        # Stub the local tree listing instead of hashing real 75MB files.
        local_tree = [
            {"path": "model.bin", "size": 75538270, "sha256": "dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1", "lfs": True},
            {"path": "config.json", "size": 2249, "sha256": "1111111111111111111111111111111111111111111111111111111111111111", "lfs": False},
        ]
        check_against_lock(
            self.lock_path,
            local_trees={"Systran/faster-whisper-tiny": local_tree},
        )  # must not raise

    def test_check_fails_on_drift(self):
        drifted = [
            {"path": "model.bin", "size": 75538270, "sha256": "de" * 32, "lfs": True},
            {"path": "config.json", "size": 2249, "sha256": "1111111111111111111111111111111111111111111111111111111111111111", "lfs": False},
        ]
        with self.assertRaises(LockDrift):
            check_against_lock(
                self.lock_path,
                local_trees={"Systran/faster-whisper-tiny": drifted},
            )

    def test_check_fails_on_size_drift(self):
        drifted = [
            {"path": "model.bin", "size": 1, "sha256": "dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1", "lfs": True},
            {"path": "config.json", "size": 2249, "sha256": "1111111111111111111111111111111111111111111111111111111111111111", "lfs": False},
        ]
        with self.assertRaises(LockDrift):
            check_against_lock(
                self.lock_path,
                local_trees={"Systran/faster-whisper-tiny": drifted},
            )

    def test_check_fails_on_missing_file(self):
        partial = [
            {"path": "config.json", "size": 2249, "sha256": "1111111111111111111111111111111111111111111111111111111111111111", "lfs": False},
        ]
        with self.assertRaises(LockDrift):
            check_against_lock(
                self.lock_path,
                local_trees={"Systran/faster-whisper-tiny": partial},
            )


class TestRevisionPin(unittest.TestCase):
    def test_revision_is_40hex(self):
        lock = yaml.safe_load(fake_lock_v2())
        rev = lock["models"][0]["revision"]
        self.assertEqual(len(rev), 40)
        int(rev, 16)


if __name__ == "__main__":
    unittest.main()
