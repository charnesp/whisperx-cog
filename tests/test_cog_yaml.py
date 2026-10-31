"""Tests E5-LEGACY-HF — cog.yaml WITHOUT offline env (RED/GREEN).

History: the offline flags (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1) were
added as defense-in-depth during the /models de-bake. They are now REMOVED
from cog.yaml (final Charles decision): the offline env was redundant for
ASR/aligner and harmful to the only legitimate network call of the diarize
path (gated pyannote model, first-run download with the container token).

New contract: the environment carries NO offline flag at all. The
anti-download question for the main models is a provisioning concern
(models must exist under /models per models.lock), not an env-var concern.
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

    def test_no_offline_env_at_all(self):
        """RED (E5-LEGACY-HF): cog.yaml must NOT pin any offline env var.

        Rationale (final Charles decision, replaces the previous T6 tests
        that REQUIRED the offline pins): the offline env was global defense
        from the de-bake, redundant for ASR/aligner and harmful to diarize
        — HF_HUB_OFFLINE=1 ignored the HF token and broke the gated
        pyannote download on the first run (LocalEntryNotFoundError).
        """
        env = self.cfg.get("environment", [])
        offenders = [
            e
            for e in env
            if e.startswith(("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"))
        ]
        self.assertEqual(
            offenders,
            [],
            "environment must carry no offline flag (offline env removed: it "
            "blocked the only legitimate network call — gated pyannote on "
            "diarize; provisioning lives in model resolution, not in a "
            "global env)",
        )

    def test_cudnn_compat_env_untouched(self):
        env = self.cfg.get("environment", [])
        self.assertIn(
            "PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK=1",
            env,
            "existing runtime env must not be broken by the de-bake",
        )


if __name__ == "__main__":
    unittest.main()
