"""Tests for scripts/image_diet.py (post-install slimming, pure filesystem).

RED first: this module did not exist before the change.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "image_diet.py"


def build_tree(root: Path, nvidia=(), packages=()):
    site = root / "site-packages"
    site.mkdir(parents=True)
    for name in nvidia:
        d = site / "nvidia" / name / "lib"
        d.mkdir(parents=True)
        (d / f"lib{name}.so.12").write_bytes(b"x")
        dist = site / f"nvidia_{name}_cu12-1.0.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text("Name: nvidia-" + name)
    for name in packages:
        d = site / name
        d.mkdir()
        (d / "__init__.py").write_text("")
        dist = site / f"{name}-1.0.dist-info"
        dist.mkdir()
    torch = site / "torch"
    (torch / "include").mkdir(parents=True)
    (torch / "test").mkdir()
    (torch / "lib").mkdir()
    (torch / "lib" / "libtorch_cuda.so").write_bytes(b"x")
    (site / "pyyaml").mkdir(exist_ok=True)
    return site


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.site = build_tree(
            self.tmp,
            nvidia=("cublas", "cudnn", "cusparselt", "cufile", "nccl"),
            packages=("triton", "pyyaml"),
        )

    def run_script(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--site-packages", str(self.site), *args],
            capture_output=True, text=True,
        )

    def exists(self, *parts):
        return (self.site.joinpath(*parts)).exists()

    def test_duplicated_cuda_is_removed(self):
        r = self.run_script("--apply")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(self.exists("nvidia", "cublas"))
        self.assertFalse(self.exists("nvidia", "cudnn"))
        self.assertFalse(self.exists("nvidia", "nccl"))

    def test_wheels_the_base_image_lacks_are_kept(self):
        r = self.run_script("--apply")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(self.exists("nvidia", "cusparselt", "lib"),
                        "libcusparseLt.so.0 is needed by libtorch_python.so")
        self.assertTrue(self.exists("nvidia", "cufile", "lib"),
                        "libcufile.so.0 is needed by libtorch_cuda.so")

    def test_triton_is_removed_and_its_neighbours_are_not(self):
        r = self.run_script("--apply")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(self.exists("triton"))
        self.assertTrue(self.exists("pyyaml"), "unrelated packages must survive")

    def test_torch_headers_and_tests_are_removed_but_libs_stay(self):
        self.run_script("--apply")
        self.assertFalse(self.exists("torch", "include"))
        self.assertFalse(self.exists("torch", "test"))
        self.assertTrue(self.exists("torch", "lib", "libtorch_cuda.so"))

    def test_dist_info_of_removed_packages_is_removed(self):
        self.run_script("--apply")
        leftovers = sorted(p.name for p in self.site.glob("*.dist-info"))
        self.assertNotIn("nvidia_cublas_cu12-1.0.dist-info", leftovers)
        self.assertNotIn("triton-1.0.dist-info", leftovers)
        self.assertIn("nvidia_cusparselt_cu12-1.0.dist-info", leftovers)

    def test_dry_run_changes_nothing(self):
        r = self.run_script()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(self.exists("nvidia", "cublas"))
        self.assertTrue(self.exists("triton"))
        self.assertIn("cublas", r.stdout)

    def test_missing_kept_wheel_fails_loudly(self):
        shutil.rmtree(self.site / "nvidia" / "cusparselt")
        r = self.run_script("--apply")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("cusparselt", r.stderr)

    def test_json_report_counts_what_it_did(self):
        r = self.run_script("--apply", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads(r.stdout)
        self.assertTrue(report["applied"])
        self.assertGreaterEqual(report["removed_dirs"], 6)  # 3 nvidia + 3 dist-info + triton + dist-info + 2 torch
        self.assertIn("cusparselt", report["kept_nvidia"])
        self.assertGreater(report["removed_bytes"], 0)

    def test_wrong_site_packages_path_fails_loudly(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--site-packages", str(self.tmp / "nope")],
            capture_output=True, text=True,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("introuvable", r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main()
