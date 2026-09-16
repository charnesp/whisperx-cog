"""GPU-free unit tests for the unified model registry (E5-CODE-1 T1).

RED cycle: written FIRST — must fail with ModuleNotFoundError because
models_registry.py does not exist yet.

Registry contract (plan-e5-deploiement-canary.md §3 T1):
- declarative mapping key -> {hf_repo, lock_key, dirname, env_override,
  weight_files} covering the 5 provisioned models:
  tiny, large-v3, large-v3-turbo, qwen3-asr-1.7b, qwen3-forced-aligner-0.6b
- VAD stays OUT of the registry (bundled whisperx asset, no stable sha)
- predict.py / model_paths.py consume the registry: no hardcoded model
  paths or HF repo ids left in those modules
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

EXPECTED_KEYS = {
    "tiny",
    "large-v3",
    "large-v3-turbo",
    "qwen3-asr-1.7b",
    "qwen3-forced-aligner-0.6b",
}

EXPECTED_HF_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-forced-aligner-0.6b": "Qwen/Qwen3-ForcedAligner-0.6B",
}


class TestRegistryModels(unittest.TestCase):
    def test_module_imports(self):
        import models_registry  # noqa: F401

    def test_registry_covers_exactly_the_5_models(self):
        from models_registry import MODELS

        self.assertEqual(set(MODELS), EXPECTED_KEYS)

    def test_each_spec_has_hf_repo_matching_models_lock(self):
        from models_registry import MODELS

        for key, hf_repo in EXPECTED_HF_REPOS.items():
            self.assertEqual(MODELS[key].hf_repo, hf_repo)

    def test_lock_key_equals_registry_key(self):
        from models_registry import MODELS

        for key in EXPECTED_KEYS:
            self.assertEqual(MODELS[key].lock_key, key)

    def test_dirnames_match_provisioned_and_dev_layouts(self):
        from models_registry import MODELS

        # Legacy/dev dirnames (./models + flat /models bake).
        self.assertEqual(MODELS["tiny"].dirname, "faster-whisper-tiny")
        self.assertEqual(MODELS["large-v3"].dirname, "faster-whisper-large-v3")
        self.assertEqual(
            MODELS["large-v3-turbo"].dirname, "faster-whisper-large-v3-turbo"
        )
        self.assertEqual(MODELS["qwen3-asr-1.7b"].dirname, "qwen3-asr-1.7b")
        self.assertEqual(
            MODELS["qwen3-forced-aligner-0.6b"].dirname,
            "qwen3-forced-aligner-0.6b",
        )

    def test_weight_files_match_models_lock_expected_weights(self):
        from models_registry import MODELS

        self.assertEqual(MODELS["tiny"].weight_files, ("model.bin",))
        self.assertEqual(MODELS["large-v3"].weight_files, ("model.bin",))
        self.assertEqual(MODELS["large-v3-turbo"].weight_files, ("model.bin",))
        self.assertEqual(
            MODELS["qwen3-asr-1.7b"].weight_files,
            (
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ),
        )
        self.assertEqual(
            MODELS["qwen3-forced-aligner-0.6b"].weight_files,
            ("model.safetensors",),
        )

    def test_env_overrides_qwen_only(self):
        from models_registry import MODELS

        self.assertEqual(MODELS["qwen3-asr-1.7b"].env_override, "QWEN_MODEL_PATH")
        self.assertEqual(
            MODELS["qwen3-forced-aligner-0.6b"].env_override, "QWEN_ALIGNER_PATH"
        )
        for key in ("tiny", "large-v3", "large-v3-turbo"):
            self.assertIsNone(MODELS[key].env_override)

    def test_specs_are_immutable(self):
        from models_registry import MODELS

        with self.assertRaises(Exception):
            MODELS["tiny"].hf_repo = "x"


class TestRegistryLocalCandidates(unittest.TestCase):
    def test_baked_first_then_dev_local(self):
        from models_registry import local_candidates

        self.assertEqual(
            local_candidates("tiny"),
            ["/models/faster-whisper-tiny", "./models/faster-whisper-tiny"],
        )
        self.assertEqual(
            local_candidates("qwen3-asr-1.7b"),
            ["/models/qwen3-asr-1.7b", "./models/qwen3-asr-1.7b"],
        )

    def test_unknown_key_raises(self):
        from models_registry import local_candidates

        with self.assertRaises(KeyError):
            local_candidates("nope")

    def test_baked_root_constant(self):
        from models_registry import BAKED_MODELS_ROOT

        self.assertEqual(BAKED_MODELS_ROOT, "/models")


class TestRegistryAliases(unittest.TestCase):
    def test_qwen_api_alias_resolves_to_lock_key(self):
        from models_registry import resolve_key

        self.assertEqual(resolve_key("qwen3-asr"), "qwen3-asr-1.7b")

    def test_registry_keys_resolve_to_themselves(self):
        from models_registry import resolve_key

        for key in EXPECTED_KEYS:
            self.assertEqual(resolve_key(key), key)

    def test_unknown_alias_raises(self):
        from models_registry import resolve_key

        with self.assertRaises(KeyError):
            resolve_key("gpt-4o-transcribe")


class TestVadStaysOutOfRegistry(unittest.TestCase):
    def test_vad_not_in_registry(self):
        from models_registry import MODELS

        self.assertNotIn("vad", MODELS)
        self.assertFalse(any("vad" in key for key in MODELS))

    def test_vad_constants_remain_in_model_paths(self):
        import model_paths

        self.assertEqual(
            model_paths.VAD_FILENAME, "whisperx-vad-segmentation.bin"
        )
        self.assertEqual(len(model_paths.VAD_LOCAL_CANDIDATES), 2)


class TestPredictConsumesRegistry(unittest.TestCase):
    """predict.py must import its model constants from the registry."""

    def _predict(self):
        from _predict_stub import install

        return install()

    def test_predict_qwen_paths_come_from_registry(self):
        from models_registry import local_candidates

        predict = self._predict()
        self.assertEqual(
            predict.QWEN_MODEL_LOCAL_PATHS, local_candidates("qwen3-asr-1.7b")
        )
        self.assertEqual(
            predict.QWEN_ALIGNER_LOCAL_PATHS,
            local_candidates("qwen3-forced-aligner-0.6b"),
        )

    def test_predict_weight_files_come_from_registry(self):
        from models_registry import MODELS

        predict = self._predict()
        self.assertEqual(
            predict.QWEN_ASR_WEIGHT_FILES,
            list(MODELS["qwen3-asr-1.7b"].weight_files),
        )
        self.assertEqual(
            predict.QWEN_ALIGNER_WEIGHT_FILES,
            list(MODELS["qwen3-forced-aligner-0.6b"].weight_files),
        )

    def test_predict_hf_repos_come_from_registry(self):
        predict = self._predict()
        self.assertEqual(predict.QWEN_ASR_HF_REPO, EXPECTED_HF_REPOS["qwen3-asr-1.7b"])
        self.assertEqual(
            predict.QWEN_ALIGNER_HF_REPO,
            EXPECTED_HF_REPOS["qwen3-forced-aligner-0.6b"],
        )

    def test_predict_source_has_no_hardcoded_model_paths_or_repos(self):
        src = (REPO_ROOT / "predict.py").read_text()
        self.assertNotIn("/models/qwen3", src)
        self.assertNotIn("./models/qwen3", src)
        self.assertNotIn("Qwen/", src)

    def test_model_paths_source_has_no_hardcoded_hf_repos(self):
        src = (REPO_ROOT / "model_paths.py").read_text()
        self.assertNotIn("Systran/", src)
        self.assertNotIn("mobiuslabsgmbh/", src)
        self.assertNotIn("Qwen/", src)


class TestModelPathsConsumeRegistry(unittest.TestCase):
    def test_whisper_hf_ids_come_from_registry(self):
        import model_paths
        from models_registry import MODELS

        self.assertEqual(
            model_paths.WHISPER_MODEL_HF_IDS,
            {key: MODELS[key].hf_repo for key in EXPECTED_KEYS
             if key in ("tiny", "large-v3", "large-v3-turbo")},
        )

    def test_whisper_local_paths_come_from_registry(self):
        import model_paths
        from models_registry import local_candidates

        self.assertEqual(
            model_paths.WHISPER_MODEL_LOCAL_PATHS,
            {
                key: local_candidates(key)
                for key in ("tiny", "large-v3", "large-v3-turbo")
            },
        )


if __name__ == "__main__":
    unittest.main()
