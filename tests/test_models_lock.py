"""GPU-free unit tests for models_lock.py + the boot validator (E5-CODE-1 T2).

RED cycle: written FIRST — must fail with ModuleNotFoundError because
models_lock.py does not exist yet.

Contract (plan-e5-deploiement-canary.md §3 T2):
- parse_lock(): reads models.lock (repo, revision, expected_files per model)
- expected_paths(root): /models/<key>/<sha40>/ per registry key
- fast_validate(root): os.stat + exact size + .complete marker, NEVER
  sha256 at boot; fail-fast with exit code 78 (EX_CONFIG)
- structured error E_MODEL_NOT_PROVISIONED model=... rev=... path=...
  with the exact provisioner remediation command + -e HF_TOKEN mention
- anti-leak: error text never contains a token VALUE
"""

from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

LOCK_SRC = REPO_ROOT / "models.lock"

FAKE_LOCK = """\
models:
  - repo: Systran/faster-whisper-tiny
    revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    expected_files: [config.json, model.bin]
  - repo: Qwen/Qwen3-ASR-1.7B
    revision: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    expected_files: [config.json, model-00001-of-00002.safetensors, model-00002-of-00002.safetensors]
"""

SECRET_VALUE = "hf_SUPERSECRETtokenValue123456"


def _fake_lock_path(tmp: Path, text: str = FAKE_LOCK) -> Path:
    p = tmp / "models.lock"
    p.write_text(text)
    return p


def _build_baked(root: Path, key: str, rev: str, files: dict[str, int]) -> Path:
    d = root / key / rev
    d.mkdir(parents=True)
    for name, size in files.items():
        (d / name).write_bytes(b"w" * size)
    (d / ".complete").write_text("")
    return d


def _capture(fn, *a, **kw):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            fn(*a, **kw)
            raised = None
        except SystemExit as e:
            raised = e
        except BaseException as e:  # noqa: BLE001
            raised = e
    return raised, out.getvalue() + err.getvalue()


class TestParseLock(unittest.TestCase):
    def test_module_imports(self):
        import models_lock  # noqa: F401

    def test_parse_fake_lock_entries(self):
        import tempfile

        import models_lock

        with tempfile.TemporaryDirectory() as tmp:
            lock = models_lock.parse_lock(_fake_lock_path(Path(tmp)))
        self.assertEqual(len(lock), 2)
        self.assertEqual(lock["tiny"]["repo"], "Systran/faster-whisper-tiny")
        self.assertEqual(lock["tiny"]["revision"], "a" * 40)
        self.assertEqual(lock["tiny"]["expected_files"], ["config.json", "model.bin"])
        self.assertEqual(
            lock["qwen3-asr-1.7b"]["revision"], "b" * 40
        )

    def test_parse_real_repo_lock_has_5_models(self):
        import models_lock

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

    def test_real_lock_revisions_match_expected_pins(self):
        import models_lock

        lock = models_lock.parse_lock(LOCK_SRC)
        self.assertEqual(
            lock["qwen3-asr-1.7b"]["revision"],
            "7278e1e70fe206f11671096ffdd38061171dd6e5",
        )
        self.assertEqual(
            lock["qwen3-forced-aligner-0.6b"]["revision"],
            "c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
        )
        self.assertEqual(
            lock["large-v3-turbo"]["revision"],
            "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
        )

    def test_real_lock_expected_files_match_registry_weight_files(self):
        import models_lock
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

    def test_parse_rejects_missing_revision(self):
        import tempfile

        import models_lock

        bad = "models:\n  - repo: Systran/faster-whisper-tiny\n    expected_files: [model.bin]\n"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Exception):
                models_lock.parse_lock(_fake_lock_path(Path(tmp), bad))


class TestExpectedPaths(unittest.TestCase):
    def test_expected_paths_use_revisioned_layout(self):
        import tempfile

        import models_lock

        with tempfile.TemporaryDirectory() as tmp:
            lock = models_lock.parse_lock(_fake_lock_path(Path(tmp)))
            paths = models_lock.expected_paths(Path(tmp) / "models", lock)
        self.assertEqual(
            paths["tiny"],
            str(Path(tmp) / "models" / "tiny" / "a" * 40),
        )
        self.assertEqual(
            paths["qwen3-asr-1.7b"],
            str(Path(tmp) / "models" / "qwen3-asr-1.7b" / "b" * 40),
        )

    def test_expected_paths_covers_5_models_of_real_lock(self):
        import models_lock

        lock = models_lock.parse_lock(LOCK_SRC)
        paths = models_lock.expected_paths("/models", lock)
        self.assertEqual(len(paths), 5)
        for key, path in paths.items():
            self.assertTrue(path.startswith(f"/models/{key}/"))


class TestFastValidate(unittest.TestCase):
    def _validate(self, root, lock=None, lock_text=FAKE_LOCK):
        import models_lock

        if lock is None:
            lock = models_lock.parse_lock(lock_text)
        return _capture(models_lock.fast_validate, root, lock)

    def test_passes_when_all_provisioned(self):
        import tempfile

        import models_lock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            lock = models_lock.parse_lock(_fake_lock_path(Path(tmp)))
            _build_baked(
                root,
                "tiny",
                "a" * 40,
                {"config.json": 10, "model.bin": 100},
            )
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
            self._validate(root, lock)

    def test_fails_when_model_missing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            raised, out = self._validate(root)
        self.assertIsNotNone(raised, "fast_validate must fail on empty volume")

    def test_fail_message_is_structured(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            raised, out = self._validate(root)
        text = f"{raised}\n{out}"
        self.assertIn("E_MODEL_NOT_PROVISIONED", text)
        self.assertIn("model=tiny", text)
        self.assertIn("rev=" + "a" * 40, text)
        self.assertIn("path=", text)

    def test_fail_message_has_provisioner_remediation(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            raised, out = self._validate(root)
        text = f"{raised}\n{out}"
        self.assertIn(
            "docker run --rm -v /files/data/whisperx-cog/models:/models "
            "ghcr.io/charnesp/whisperx-provisioner:latest provision --model tiny",
            text,
        )

    def test_fail_message_mentions_hf_token_var_without_value(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            raised, out = self._validate(root)
        text = f"{raised}\n{out}"
        self.assertIn("-e HF_TOKEN", text)
        self.assertNotIn("=", text.split("-e HF_TOKEN", 1)[1].split("\n", 1)[0])

    def test_fail_when_complete_marker_missing(self):
        import tempfile

        import models_lock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            d = root / "tiny" / "a" * 40
            d.mkdir(parents=True)
            (d / "model.bin").write_bytes(b"w" * 100)
            lock = models_lock.parse_lock(_fake_lock_path(Path(tmp)))
            raised, out = self._validate(root, lock)
        text = f"{raised}\n{out}"
        self.assertIn("E_MODEL_NOT_PROVISIONED", text)

    def test_fail_when_file_size_mismatches(self):
        import tempfile

        import models_lock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            lock = models_lock.parse_lock(_fake_lock_path(Path(tmp)))
            d = root / "tiny" / "a" * 40
            d.mkdir(parents=True)
            (d / "config.json").write_bytes(b"w" * 10)
            (d / "model.bin").write_bytes(b"w" * 999)  # != 100
            (d / ".complete").write_text("")
            raised, out = self._validate(root, lock)
        text = f"{raised}\n{out}"
        self.assertIn("E_MODEL_NOT_PROVISIONED", text)
        self.assertIn("model.bin", text)

    def test_expected_sizes_come_from_lock(self):
        import tempfile

        import models_lock

        with tempfile.TemporaryDirectory() as tmp:
            lock = models_lock.parse_lock(_fake_lock_path(Path(tmp)))
            sizes = models_lock.expected_sizes(lock)
        self.assertEqual(sizes[("tiny", "model.bin")], 100)

    def test_lock_without_sizes_means_stat_only_complete_marker(self):
        """Lock v1 (no size field): files checked by presence + .complete only."""
        import tempfile

        import models_lock

        text = (
            "models:\n"
            "  - repo: Systran/faster-whisper-tiny\n"
            "    revision: " + "a" * 40 + "\n"
            "    expected_files: [config.json, model.bin]\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            lock = models_lock.parse_lock(_fake_lock_path(Path(tmp), text))
            d = root / "tiny" / "a" * 40
            d.mkdir(parents=True)
            (d / "config.json").write_bytes(b"w")
            (d / "model.bin").write_bytes(b"w")
            (d / ".complete").write_text("")
            models_lock.fast_validate(root, lock)  # must not raise

    def test_exit_code_78(self):
        import tempfile

        import models_lock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            lock = models_lock.parse_lock(_fake_lock_path(Path(tmp)))
            with self.assertRaises(models_lock.ModelsNotProvisioned) as ctx:
                models_lock.fast_validate(root, lock)
            self.assertEqual(ctx.exception.exit_code, 78)

    def test_boot_validate_stdout_smoke(self):
        import models_lock

        raised, out = _capture(models_lock.boot_validate, "/models", LOCK_SRC)
        self.assertIn("E_MODEL_NOT_PROVISIONED", out)


class TestSecretLeakGuard(unittest.TestCase):
    def test_no_token_value_in_error_output(self):
        import tempfile

        import models_lock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            lock = models_lock.parse_lock(_fake_lock_path(Path(tmp)))
            env = {k: v for k, v in os.environ.items()}
            env["HF_TOKEN"] = SECRET_VALUE
            old = dict(os.environ)
            os.environ.clear()
            os.environ.update(env)
            try:
                raised, out = _capture(models_lock.fast_validate, root, lock)
            finally:
                os.environ.clear()
                os.environ.update(old)
        text = f"{raised}\n{out}"
        self.assertNotIn(SECRET_VALUE, text)


class TestSetupIntegration(unittest.TestCase):
    """The boot validator must run FIRST inside Predictor.setup()."""

    @classmethod
    def setUpClass(cls):
        from _predict_stub import install

        cls.predict = install()

    def test_setup_calls_validator_before_anything_else(self):
        import tempfile

        import models_lock

        calls = []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            lock = models_lock.parse_lock(_fake_lock_path(Path(tmp)))
            _build_baked(
                root,
                "tiny",
                "a" * 40,
                {"config.json": 10, "model.bin": 100},
            )
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

            predictor = self.predict.Predictor()

            real_vad = self.predict.resolve_vad_source_path
            real_validate = self.predict.validate_boot_models

            def fake_validate(models_root=None):
                calls.append("validate")

            def fake_vad():
                calls.append("vad")
                return None

            with self.mock_env(models_root=str(root), lock_path=_fake_lock_path(Path(tmp))):
                self.predict.resolve_vad_source_path = fake_vad
                self.predict.validate_boot_models = fake_validate
                try:
                    predictor.setup()
                finally:
                    self.predict.resolve_vad_source_path = real_vad
                    self.predict.validate_boot_models = real_validate

        self.assertEqual(calls, ["validate", "vad"])

    def _mock_env(self, models_root, lock_path):
        import contextlib

        @contextlib.contextmanager
        def ctx():
            import unittest.mock as mock

            with mock.patch.dict(
                os.environ,
                {"MODELS_DIR": models_root, "MODELS_LOCK_PATH": str(lock_path)},
            ):
                yield

        return ctx()

    def test_setup_fails_fast_when_not_provisioned(self):
        import tempfile

        import models_lock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            predictor = self.predict.Predictor()
            with self.mock_env(models_root=str(root), lock_path=_fake_lock_path(Path(tmp))):
                with self.assertRaises(Exception) as ctx:
                    predictor.setup()
            self.assertIn("E_MODEL_NOT_PROVISIONED", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()