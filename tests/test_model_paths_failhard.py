"""GPU-free unit tests for model_paths.py fail-HARD resolution (E5-CODE-1 T3).

RED cycle: written FIRST — the current resolve_whisper_model_path falls
back to the HuggingFace repo ID; the new contract raises
ModelNotProvisioned instead (the deep HF_HUB_OFFLINE error drowned in the
stack trace is the failure mode this change eliminates).

Resolution order (plan §3 T3):
1. explicit env override (registry env_override, e.g. QWEN_MODEL_PATH)
2. lock /models/<key>/<sha40>/ (+ .complete)
3. ./models/<dirname> only when MODELS_MODE=dev
4. else ModelNotProvisioned — NEVER the HF repo id

MODELS_MODE=online is an explicit opt-in (laptop): the function may then
return the HF repo id; it is NEVER the default.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import model_paths  # noqa: E402


class _Base(unittest.TestCase):
    def _clean_env(self, **extra):
        return mock_env(**{"MODELS_DIR": "", "MODELS_MODE": "", **extra})

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "models"
        self.root.mkdir()
        import models_lock

        self.lock = models_lock.parse_lock(models_lock.DEFAULT_LOCK_PATH)

    def _bake(self, key: str, complete: bool = True) -> Path:
        import models_lock

        entry = self.lock[key]
        d = self.root / key / entry["revision"]
        d.mkdir(parents=True)
        for name in entry["expected_files"]:
            (d / name).write_bytes(b"w")
        if complete:
            (d / ".complete").write_text("")
        return d


def mock_env(**values):
    import contextlib

    @contextlib.contextmanager
    def ctx():
        clean = {k: v for k, v in os.environ.items() if k not in values}
        clean.update({k: v for k, v in values.items() if v})
        for k in values:
            if not values[k]:
                clean.pop(k, None)
        old = dict(os.environ)
        os.environ.clear()
        os.environ.update(clean)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(old)

    return ctx()


class TestFailHardWhisperPath(_Base):
    def test_no_local_dir_no_dev_mode_raises_model_not_provisioned(self):
        import model_paths as mp

        with self._clean_env():
            with self.assertRaises(mp.ModelNotProvisioned) as ctx:
                mp.resolve_whisper_model_path("large-v3-turbo")
        text = str(ctx.exception)
        self.assertIn("/models", text)
        self.assertIn("./models", text)
        self.assertIn("docker run --rm -v /files/data/whisperx-cog/models:/models", text)

    def test_no_fallback_to_hf_repo_id(self):
        import model_paths as mp
        from models_registry import MODELS

        with self._clean_env():
            with self.assertRaises(mp.ModelNotProvisioned):
                mp.resolve_whisper_model_path("tiny")
        # and explicitly: the HF id is never returned as a path
        self.assertNotIn(
            MODELS["tiny"].hf_repo,
            _error_text(lambda: mp.resolve_whisper_model_path("tiny"), self),
        )


class _ErrorCtx:
    pass


def _error_text(fn, test):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        return str(exc)
    return ""


class TestResolutionOrder(_Base):
    def test_env_override_explicit_wins(self):
        import model_paths as mp

        override = Path(self.tmp.name) / "custom" / "turbo"
        override.mkdir(parents=True)
        (override / "model.bin").write_bytes(b"w")
        with self._clean_env():
            with mock_env(**{"LARGE_V3_TURBO_PATH": str(override)}):
                with mock.patch.dict(
                    "model_paths.ENV_OVERRIDES",
                    {"large-v3-turbo": "LARGE_V3_TURBO_PATH"},
                ):
                    self.assertEqual(
                        mp.resolve_whisper_model_path("large-v3-turbo"),
                        str(override),
                    )

    def test_lock_revisioned_path_when_provisioned(self):
        import model_paths as mp

        self._bake("large-v3-turbo")
        with self._clean_env(**{"MODELS_DIR": str(self.root)}):
            self.assertEqual(
                mp.resolve_whisper_model_path("large-v3-turbo"),
                str(self.root / "large-v3-turbo" / self.lock["large-v3-turbo"]["revision"]),
            )

    def test_lock_path_without_complete_marker_is_rejected(self):
        import model_paths as mp

        self._bake("large-v3-turbo", complete=False)
        with self._clean_env(**{"MODELS_DIR": str(self.root)}):
            with self.assertRaises(mp.ModelNotProvisioned):
                mp.resolve_whisper_model_path("large-v3-turbo")

    def test_dev_mode_uses_local_models_dir(self):
        import model_paths as mp

        dev = Path(self.tmp.name) / "workdir" / "models" / "faster-whisper-large-v3-turbo"
        dev.mkdir(parents=True)
        (dev / "model.bin").write_bytes(b"w")
        old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        try:
            with self._clean_env(**{"MODELS_MODE": "dev", "MODELS_DIR": ""}):
                self.assertEqual(
                    mp.resolve_whisper_model_path("large-v3-turbo"),
                    str(dev),
                )
        finally:
            os.chdir(old_cwd)

    def test_dev_mode_without_local_raises(self):
        import model_paths as mp

        old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        try:
            with self._clean_env(**{"MODELS_MODE": "dev", "MODELS_DIR": str(self.root)}):
                with self.assertRaises(mp.ModelNotProvisioned):
                    mp.resolve_whisper_model_path("large-v3-turbo")
        finally:
            os.chdir(old_cwd)

    def test_models_mode_online_explicit_opt_in_returns_hf_id(self):
        import model_paths as mp
        from models_registry import MODELS

        with self._clean_env(**{"MODELS_MODE": "online"}):
            self.assertEqual(
                mp.resolve_whisper_model_path("tiny"),
                MODELS["tiny"].hf_repo,
            )

    def test_models_mode_online_never_default(self):
        import model_paths as mp

        with self._clean_env():
            with self.assertRaises(mp.ModelNotProvisioned):
                mp.resolve_whisper_model_path("tiny")


class TestQwenResolutionViaRegistry(_Base):
    def test_resolve_model_dir_qwen_uses_registry_key_and_lock(self):
        import model_paths as mp

        self._bake("qwen3-asr-1.7b")
        with self._clean_env(**{"MODELS_DIR": str(self.root)}):
            resolved = mp.resolve_model_dir("qwen3-asr-1.7b")
        self.assertEqual(
            resolved,
            str(self.root / "qwen3-asr-1.7b" / self.lock["qwen3-asr-1.7b"]["revision"]),
        )

    def test_resolve_model_dir_qwen_env_override_first(self):
        import model_paths as mp

        override = Path(self.tmp.name) / "override-qwen"
        override.mkdir()
        with self._clean_env(**{"QWEN_MODEL_PATH": str(override)}):
            self.assertEqual(mp.resolve_model_dir("qwen3-asr-1.7b"), str(override))

    def test_resolve_model_dir_qwen_missing_raises_with_remediation(self):
        import model_paths as mp

        with self._clean_env(**{"MODELS_DIR": str(self.root)}):
            with self.assertRaises(mp.ModelNotProvisioned) as ctx:
                mp.resolve_model_dir("qwen3-asr-1.7b")
        text = str(ctx.exception)
        self.assertIn("model=qwen3-asr-1.7b", text)
        self.assertIn("E_MODEL_NOT_PROVISIONED", text)
        self.assertIn("provision --model qwen3-asr-1.7b", text)

    def test_resolve_model_dir_aligner_env_override(self):
        import model_paths as mp

        override = Path(self.tmp.name) / "override-aligner"
        override.mkdir()
        with self._clean_env(**{"QWEN_ALIGNER_PATH": str(override)}):
            self.assertEqual(
                mp.resolve_model_dir("qwen3-forced-aligner-0.6b"), str(override)
            )

    def test_resolve_model_dir_dev_mode(self):
        import model_paths as mp

        dev = Path(self.tmp.name) / "workdir" / "models" / "qwen3-asr-1.7b"
        dev.mkdir(parents=True)
        old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        try:
            with self._clean_env(**{"MODELS_MODE": "dev", "MODELS_DIR": ""}):
                self.assertEqual(mp.resolve_model_dir("qwen3-asr-1.7b"), str(dev))
        finally:
            os.chdir(old_cwd)

    def test_resolve_model_dir_alias_api_name(self):
        import model_paths as mp

        self._bake("qwen3-asr-1.7b")
        with self._clean_env(**{"MODELS_DIR": str(self.root)}):
            resolved = mp.resolve_model_dir("qwen3-asr")
        self.assertEqual(resolved, str(self.root / "qwen3-asr-1.7b" / self.lock["qwen3-asr-1.7b"]["revision"]))


class TestWeightCheckIntegrated(_Base):
    def test_resolve_model_dir_rejects_empty_weight_files(self):
        import model_paths as mp

        entry = self.lock["qwen3-asr-1.7b"]
        d = self.root / "qwen3-asr-1.7b" / entry["revision"]
        d.mkdir(parents=True)
        for name in entry["expected_files"]:
            (d / name).write_bytes(b"w")
        (d / ".complete").write_text("")
        # sabotage: empty a weight file
        (d / entry["expected_files"][-1]).write_bytes(b"")
        with self._clean_env(**{"MODELS_DIR": str(self.root)}):
            with self.assertRaises(mp.ModelNotProvisioned):
                mp.resolve_model_dir("qwen3-asr-1.7b")


class TestPredictCallSitesUseRegistry(_Base):
    """_run_predict + aligner must go through the registry (no direct lists)."""

    @classmethod
    def setUpClass(cls):
        from _predict_stub import install

        cls.predict = install()

    def test_predict_source_has_no_direct_snapshot_lists(self):
        src = (REPO_ROOT / "predict.py").read_text()
        self.assertNotIn("QWEN_MODEL_LOCAL_PATHS =", src)
        self.assertNotIn("QWEN_ALIGNER_LOCAL_PATHS =", src)

    def test_single_check_point_in_registry(self):
        """resolve_qwen_snapshot_dir/assert_baked_qwen_weights fused into the
        registry: one check point (resolve_model_dir + weight validation)."""
        import model_paths as mp
        from models_registry import MODELS

        self._bake("qwen3-asr-1.7b")
        with self._clean_env(**{"MODELS_DIR": str(self.root)}):
            resolved = mp.resolve_model_dir("qwen3-asr-1.7b")
        # resolved dir is exactly the lock revision dir and carries weights
        entry = self.lock["qwen3-asr-1.7b"]
        self.assertTrue(resolved.endswith(entry["revision"]))
        for name in MODELS["qwen3-asr-1.7b"].weight_files:
            self.assertTrue((Path(resolved) / name).stat().st_size > 0)

    def test_vad_resolution_unchanged(self):
        import model_paths as mp

        vad = self.root / "vad" / "whisperx-vad-segmentation.bin"
        vad.parent.mkdir(parents=True)
        vad.write_bytes(b"v")
        with mock.patch("model_paths.VAD_LOCAL_CANDIDATES", [str(vad)]):
            self.assertEqual(mp.resolve_vad_source_path(), str(vad))


if __name__ == "__main__":
    unittest.main()