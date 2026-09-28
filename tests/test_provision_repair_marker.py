"""FIX 3 (🟡, E5-CODE review): verify --repair invalide .complete AVANT.

Contrat : le marker .complete ne doit jamais être visible pendant la
réparation — un lecteur (resolve_model_dir, boot validator) qui regarde
le dossier pendant la réparation doit voir un dossier INCOMPLET, pas un
dossier « complet » avec un fichier fautif. verify --repair unlink le
marker AVANT de toucher aux fichiers et le réécrit APRÈS fsync.

RED du cycle : la seconde exécution du mock capture un état où
config.json est corrompu ET .complete toujours présent (marker non
invalidé pendant la réparation).
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from scripts.provision import (  # noqa: E402
    CommandError,
    complete_marker_path,
    import_command,
    verify_command,
)

SHA = "a" * 40


class TestRepairInvalidatesCompleteMarker(unittest.TestCase):
    def setUp(self):
        import tempfile

        from test_provision import make_source, write_lock

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.models_root = self.tmp / "models"
        self.models_root.mkdir()
        self.staging = self.tmp / "staging"
        self.staging.mkdir()
        write_lock(self.models_root)
        self.src = make_source(self.staging)
        import_command(
            ["--model", "qwen3-asr-1.7b", "--from", str(self.src)],
            models_root=self.models_root,
            staging_root=self.staging,
        )
        self.final_dir = self.models_root / "qwen3-asr-1.7b" / SHA
        self.marker = complete_marker_path(self.models_root, "qwen3-asr-1.7b", SHA)

    def tearDown(self):
        self._tmp.cleanup()

    def _run_repair(self):
        # Corrupt AFTER .complete exists, then repair from the good source.
        (self.final_dir / "model.safetensors").write_text("corrupted!!")
        verify_command(
            ["--model", "qwen3-asr-1.7b", "--repair", "--from", str(self.src)],
            models_root=self.models_root,
            staging_root=self.staging,
        )

    def test_marker_absent_before_repair_writes(self):
        """Snapshot the marker's existence BEFORE each repair file copy: the
        marker must be unlinked before any faulty file is touched."""
        import scripts.provision as prov

        states: list[bool] = []
        orig_copyfile = prov.shutil.copyfile

        def spy_copy(src, dst, *a, **k):
            # dst == the faulty file being repaired in place
            if str(dst).endswith("model.safetensors"):
                states.append(self.marker.exists())
            return orig_copyfile(src, dst, *a, **k)

        import unittest.mock as mock

        with mock.patch.object(prov.shutil, "copyfile", side_effect=spy_copy):
            self._run_repair()
        self.assertTrue(states, "repair must copy back the faulty file(s)")
        self.assertFalse(
            any(states),
            ".complete must be UNLINKED before repair copies any file",
        )
        # and restored afterwards
        self.assertTrue(self.marker.exists())

    def test_marker_restored_after_successful_repair(self):
        self._run_repair()
        self.assertTrue(self.marker.exists(), "repair must rewrite .complete")
        from scripts.provision import verify_command as vc

        vc(["--model", "qwen3-asr-1.7b"], models_root=self.models_root)  # clean

    def test_failed_repair_leaves_marker_absent(self):
        # Source is ALSO corrupted: repair fails, marker must stay absent
        # (never a .complete over a broken snapshot).
        from test_provision import make_source

        bad_src = make_source(self.staging, tamper="bad")
        (self.final_dir / "model.safetensors").write_text("corrupted!!")
        with self.assertRaises(CommandError):
            verify_command(
                ["--model", "qwen3-asr-1.7b", "--repair", "--from", str(bad_src)],
                models_root=self.models_root,
                staging_root=self.staging,
            )
        self.assertFalse(
            self.marker.exists(),
            "a failed repair must NOT restore .complete on a broken snapshot",
        )


if __name__ == "__main__":
    unittest.main()
