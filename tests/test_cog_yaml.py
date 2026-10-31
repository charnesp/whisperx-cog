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


class TestImageDiet(unittest.TestCase):
    """One CUDA in the image: 12.8 base + torch cu128, duplicate wheels removed.

    Measured 2026-10-30: 9.40 GB, of which 3.06 GB were nvidia-* wheels that
    duplicated the base image (4.5 GB) because the base was 12.5 while the torch
    wheel is built for 12.8.
    """

    def setUp(self):
        self.cfg = yaml.safe_load(COG_YAML.read_text())

    def test_base_cuda_matches_the_torch_wheel(self):
        self.assertEqual(
            self.cfg["build"]["cuda"],
            "12.8",
            "the base CUDA must be the one the torch wheel is built for, "
            "otherwise torch ships its own CUDA as nvidia-* wheels (double CUDA)",
        )

    def test_diet_step_is_applied(self):
        run = self.cfg["build"]["run"]
        diet = [c for c in run if "image_diet.py" in c and "--apply" in c]
        self.assertEqual(len(diet), 1, "the post-install diet must run once, with --apply")

    def test_toolchain_purge_never_autoremoves(self):
        run = "\n".join(self.cfg["build"]["run"])
        self.assertIn("apt-get purge", run, "the build toolchain must be purged")
        self.assertNotIn(
            "autoremove",
            run,
            "autoremove would be free to remove the CUDA packages the base image "
            "installed as dependencies: purge explicit packages only",
        )

    def test_loader_finds_the_base_cuda(self):
        env = self.cfg["environment"]
        self.assertTrue(
            any(e.startswith("LD_LIBRARY_PATH=") and "/usr/local/cuda/lib64" in e for e in env),
            "with the nvidia-* wheels removed, the loader must be pointed at the base CUDA",
        )

    def test_requirements_still_pin_torch_and_forbid_the_ui_stack(self):
        lines = [
            ln.split("#")[0].strip()
            for ln in (REPO_ROOT / "requirements.txt").read_text().splitlines()
        ]
        req = "\n".join(line for line in lines if line)
        self.assertIn("torch==2.8.0", req)
        for forbidden in ("gradio", "flask", "vllm"):
            self.assertNotIn(
                forbidden, req,
                "the forbidden UI/serving stack must stay out of the installed "
                "requirements (comments about them are fine)",
            )

    def test_build_fails_if_the_import_chain_breaks(self):
        run = "\n".join(self.cfg["build"]["run"])
        self.assertIn(
            "import torch",
            run,
            "the build must prove that torch still imports after the diet removed "
            "the CUDA wheels; without it a missing library ships silently and only "
            "fails at the first prediction",
        )
        self.assertIn("torch.version.cuda", run, "the build log must show the CUDA the engine loads against")


if __name__ == "__main__":
    unittest.main()
