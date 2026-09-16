"""E5 T6 — cog.yaml dé-bake + env offline (RED/GREEN).

The current cog.yaml bakes ~11 GB of weights into the image via wget in
build.run. The canary decision (plan E5 §1) forbids weight baking: weights
live on a host bind mount provisioned by scripts/provision.py. These tests
parse cog.yaml and fail while any bake remains.
"""

import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
COG_YAML = REPO_ROOT / "cog.yaml"


class TestCogYamlDeBaked(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(COG_YAML.read_text())

    def test_no_wget_in_run_block(self):
        run = self.cfg.get("build", {}).get("run", [])
        wget_cmds = [c for c in run if "wget" in c and "/models" in c]
        self.assertEqual(
            wget_cmds,
            [],
            "build.run must not bake /models weights (de-bake: bind mount + provision.py)",
        )

    def test_no_mkdir_models_in_run_block(self):
        run = self.cfg.get("build", {}).get("run", [])
        mkdir_models = [c for c in run if c.strip().startswith("mkdir -p /models")]
        self.assertEqual(mkdir_models, [], "no /models staging dirs in build.run")

    def test_transformers_offline_env(self):
        env = self.cfg.get("environment", [])
        self.assertIn(
            "TRANSFORMERS_OFFLINE=1",
            env,
            "environment must pin TRANSFORMERS_OFFLINE=1",
        )

    def test_hf_hub_offline_env(self):
        env = self.cfg.get("environment", [])
        self.assertIn("HF_HUB_OFFLINE=1", env, "environment must keep HF_HUB_OFFLINE=1")

    def test_cudnn_compat_env_untouched(self):
        env = self.cfg.get("environment", [])
        self.assertIn(
            "PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK=1",
            env,
            "existing runtime env must not be broken by the de-bake",
        )


if __name__ == "__main__":
    unittest.main()