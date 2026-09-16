"""parse_lock v2 adapter (E5-LOCK-SYNC — T2 × T5 reconciliation).

RED cycle first: T5 rewrote models.lock to schema v2
(version: 2 + files: [{path, size, sha256, lfs}]) while T2's parse_lock()
only understands schema v1 (expected_files: list + sizes: map). The adapter
must derive expected_files/sizes from files[].path/size so expected_paths()
and fast_validate() keep working UNCHANGED (boot check = file presence +
exact size + .complete marker, NEVER sha256).

v1 locks stay supported for backward compatibility (documented choice:
existing pinned v1 locks and the legacy fixture suite keep parsing).
An unknown schema version raises an explicit ValueError with a migration
message.
"""

from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import models_lock  # noqa: E402

LOCK_SRC = REPO_ROOT / "models.lock"

FAKE_LOCK_V2 = textwrap.dedent(
    """\
    # models.lock v2 — sha256 per file
    version: 2
    models:
    - repo: Systran/faster-whisper-tiny
      revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      required: true
      files:
      - path: config.json
        size: 10
        sha256: 1111111111111111111111111111111111111111111111111111111111111111
        lfs: false
      - path: model.bin
        size: 100
        sha256: 2222222222222222222222222222222222222222222222222222222222222222
        lfs: true
    - repo: Qwen/Qwen3-ASR-1.7B
      revision: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
      required: true
      files:
      - path: config.json
        size: 10
        sha256: 3333333333333333333333333333333333333333333333333333333333333333
        lfs: false
      - path: model-00001-of-00002.safetensors
        size: 100
        sha256: 4444444444444444444444444444444444444444444444444444444444444444
        lfs: true
      - path: model-00002-of-00002.safetensors
        size: 200
        sha256: 5555555555555555555555555555555555555555555555555555555555555555
        lfs: true
    """
)

FAKE_LOCK_V1 = textwrap.dedent(
    """\
    models:
      - repo: Systran/faster-whisper-tiny
        revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        expected_files: [config.json, model.bin]
        sizes:
          config.json: 10
          model.bin: 100
    """
)


def _fake_lock_path(tmp: Path, text: str = FAKE_LOCK_V2) -> Path:
    p = tmp / "models.lock"
    p.write_text(text)
    return p


def _build_baked(root: Path, key: str, rev: str, files: dict[str, int]) -> None:
    d = root / key / rev
    d.mkdir(parents=True)
    for name, size in files.items():
        (d / name).write_bytes(b"w" * size)
    (d / ".complete").write_text("")


class TestParseLockV2(unittest.TestCase):
    def test_parse_fake_v2_lock_derives_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = models_lock.parse_lock(_fake_lock_path(Path(tmp)))
        self.assertEqual(set(lock), {"tiny", "qwen3-asr-1.7b"})
        self.assertEqual(lock["tiny"]["repo"], "Systran/faster-whisper-tiny")
        self.assertEqual(lock["tiny"]["revision"], "a" * 40)
        self.assertEqual(
            lock["tiny"]["expected_files"], ["config.json", "model.bin"]
        )
        self.assertEqual(lock["tiny"]["sizes"], {"config.json": 10, "model.bin": 100})
        self.assertEqual(
            lock["qwen3-asr-1.7b"]["revision"], "b" * 40
        )
        self.assertEqual(
            lock["qwen3-asr-1.7b"]["expected_files"],
            [
                "config.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
        )

    def test_expected_paths_work_with_v2_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = models_lock.parse_lock(_fake_lock_path(Path(tmp)))
            paths = models_lock.expected_paths(Path(tmp) / "models", lock)
        self.assertEqual(
            paths["tiny"], str(Path(tmp) / "models" / "tiny" / ("a" * 40))
        )
        self.assertEqual(
            paths["qwen3-asr-1.7b"],
            str(Path(tmp) / "models" / "qwen3-asr-1.7b" / ("b" * 40)),
        )

    def test_expected_sizes_derived_from_v2_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = models_lock.parse_lock(_fake_lock_path(Path(tmp)))
            sizes = models_lock.expected_sizes(lock)
        self.assertEqual(sizes[("tiny", "model.bin")], 100)
        self.assertEqual(
            sizes[("qwen3-asr-1.7b", "model-00002-of-00002.safetensors")], 200
        )

    def test_fast_validate_passes_on_provisioned_v2_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            lock = models_lock.parse_lock(_fake_lock_path(Path(tmp)))
            _build_baked(root, "tiny", "a" * 40, {"config.json": 10, "model.bin": 100})
            _build_baked(
                root,
                "qwen3-asr-1.7b",
                "b" * 40,
                {
                    "config.json": 10,
                    "model-00001-of-00002.safetensors": 100,
                    "model-00002-of-00002.safetensors": 200,
                },
            )
            models_lock.fast_validate(root, lock)  # must not raise

    def test_fast_validate_size_mismatch_on_v2_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            lock = models_lock.parse_lock(_fake_lock_path(Path(tmp)))
            d = root / "tiny" / ("a" * 40)
            d.mkdir(parents=True)
            (d / "config.json").write_bytes(b"w" * 10)
            (d / "model.bin").write_bytes(b"w" * 999)  # != 100
            (d / ".complete").write_text("")
            with self.assertRaises(models_lock.ModelsNotProvisioned) as ctx:
                models_lock.fast_validate(root, lock)
        self.assertIn("model.bin", str(ctx.exception))

    def test_real_repo_lock_v2_parses_5_models(self):
        lock = models_lock.parse_lock(LOCK_SRC)
        self.assertEqual(
            set(lock),
            {
                "tiny",
                "large-v3",
                "large-v3-turbo",
                "qwen3-asr-1.7b",
                "qwen3-forced-aligner-0.6b",
            },
        )
        for key, entry in lock.items():
            self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")
            self.assertTrue(entry["expected_files"])
            self.assertTrue(entry["sizes"], f"sizes must be derived for {key}")

    def test_real_lock_v2_expected_files_match_registry_weights(self):
        from models_registry import MODELS

        lock = models_lock.parse_lock(LOCK_SRC)
        for key in ("tiny", "large-v3", "large-v3-turbo"):
            self.assertIn("model.bin", lock[key]["expected_files"])
        for name in MODELS["qwen3-asr-1.7b"].weight_files:
            self.assertIn(name, lock["qwen3-asr-1.7b"]["expected_files"])
        self.assertIn(
            "model.safetensors",
            lock["qwen3-forced-aligner-0.6b"]["expected_files"],
        )

    def test_real_lock_v2_sizes_match_declared_file_sizes(self):
        lock = models_lock.parse_lock(LOCK_SRC)
        tiny = lock["tiny"]
        self.assertEqual(tiny["sizes"]["model.bin"], 75538270)
        self.assertEqual(tiny["sizes"]["config.json"], 2249)


class TestBackwardCompatV1(unittest.TestCase):
    def test_v1_lock_still_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = models_lock.parse_lock(_fake_lock_path(Path(tmp), FAKE_LOCK_V1))
        self.assertEqual(lock["tiny"]["expected_files"], ["config.json", "model.bin"])
        self.assertEqual(lock["tiny"]["sizes"]["model.bin"], 100)

    def test_unsupported_version_raises_migration_message(self):
        bad = "version: 3\nmodels: []\n"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                models_lock.parse_lock(_fake_lock_path(Path(tmp), bad))
        self.assertIn("version", str(ctx.exception))

    def test_v2_files_must_not_be_empty(self):
        bad = (
            "version: 2\n"
            "models:\n"
            "- repo: Systran/faster-whisper-tiny\n"
            "  revision: " + "a" * 40 + "\n"
            "  files: []\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                models_lock.parse_lock(_fake_lock_path(Path(tmp), bad))


class TestParseAgreementWithLockAudit(unittest.TestCase):
    """BLUE: parse_lock and lock_audit.load_lock_v2 must agree on the real lock.

    Both tools read the same models.lock (boot validator vs provisioner/
    audit): the derived expected_files + sizes must match the PyYAML view
    of the v2 schema, else one of the two would enforce a phantom pin.
    """

    def test_parse_lock_matches_yaml_view_on_real_lock(self):
        import yaml

        from scripts.lock_audit import load_lock_v2

        raw = yaml.safe_load(LOCK_SRC.read_text())
        v2 = load_lock_v2(LOCK_SRC)
        self.assertEqual(v2, raw)
        lock = models_lock.parse_lock(LOCK_SRC)
        for model in v2["models"]:
            key = models_lock._lock_key(model["repo"])
            entry = lock[key]
            self.assertEqual(entry["revision"], model["revision"])
            paths = [f["path"] for f in model["files"]]
            self.assertEqual(entry["expected_files"], paths)
            declared_sizes = {
                f["path"]: f["size"] for f in model["files"] if f.get("size")
            }
            self.assertEqual(entry.get("sizes", {}), declared_sizes)

    def test_expected_sizes_fast_validate_path_uses_derived_sizes(self):
        """expected_sizes (T2) must reflect files[].size (T5) on the real lock."""
        lock = models_lock.parse_lock(LOCK_SRC)
        sizes = models_lock.expected_sizes(lock)
        self.assertEqual(sizes[("tiny", "model.bin")], 75538270)
        self.assertEqual(
            sizes[("qwen3-forced-aligner-0.6b", "model.safetensors")], 1835544544
        )


if __name__ == "__main__":
    unittest.main()
