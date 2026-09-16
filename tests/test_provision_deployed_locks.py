"""FIX 2 (🟡, E5-CODE review): GC protège les shas des locks déployés.

Race démontrée : le protected-set du GC ignore le sha référencé par le
models.lock EMBARQUÉ d'une image en cours d'exécution (canary). Si le
lock hôte est bumpé et qu'une révision devient vieille (> N jours) et
absente de history.jsonl, `gc --apply` la supprime alors qu'une image
déployée pointe encore dessus (crash-loop au reboot).

Contrat :
1. Chaque `verify` journalise le lock (sha par modèle) dans
   .state/deployed_locks.jsonl (append-only) ;
2. `gc --apply` ajoute ces shas au protected-set : une révision présente
   dans deployed_locks est PRÉSERVÉE même si elle est vieille et absente
   de history.jsonl.
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from scripts.provision import (
    DEPLOYED_LOCKS_FILE,
    gc_command,
    import_command,
    record_deployed_lock,
    deployed_lock_shas,
)

SHA = "a" * 40


class _TmpTree(unittest.TestCase):
    def setUp(self):

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.models_root = self.tmp / "models"
        self.models_root.mkdir()
        self.staging = self.tmp / "staging"
        self.staging.mkdir()
        # Reuse the shared fake-lock helpers from test_provision.
        tests_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(tests_dir))
        from test_provision import make_source, write_lock

        write_lock(self.models_root)
        self.src = make_source(self.staging)

    def tearDown(self):
        self._tmp.cleanup()

    def _import_rev(self, rev):
        from test_provision import make_source

        src = make_source(self.staging, rev)
        import_command(
            ["--model", "qwen3-asr-1.7b", "--from", str(src), "--revision", rev],
            models_root=self.models_root,
            staging_root=self.staging,
        )


class TestVerifyJournalsDeployedLock(_TmpTree):
    def test_verify_appends_lock_shas_to_deployed_locks(self):
        self._import_rev(SHA)
        from scripts.provision import verify_command

        verify_command(
            ["--model", "qwen3-asr-1.7b"], models_root=self.models_root
        )
        journal = self.models_root / ".state" / DEPLOYED_LOCKS_FILE
        self.assertTrue(journal.is_file(), "verify must journal the lock")
        lines = [
            json.loads(line)
            for line in journal.read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(lines[-1]["models"]["qwen3-asr-1.7b"], SHA)

    def test_record_deployed_lock_appends_only(self):
        record_deployed_lock(
            self.models_root, {"qwen3-asr-1.7b": SHA}, ts=1234
        )
        record_deployed_lock(
            self.models_root, {"qwen3-asr-1.7b": "b" * 40}, ts=5678
        )
        journal = self.models_root / ".state" / DEPLOYED_LOCKS_FILE
        lines = [json.loads(x) for x in journal.read_text().splitlines() if x]
        self.assertEqual(len(lines), 2, "append-only, never rewrite")
        self.assertEqual(lines[0]["ts"], 1234)
        self.assertEqual(lines[1]["models"]["qwen3-asr-1.7b"], "b" * 40)

    def test_deployed_lock_shas_reader(self):
        record_deployed_lock(self.models_root, {"qwen3-asr-1.7b": SHA})
        record_deployed_lock(self.models_root, {"qwen3-asr-1.7b": "c" * 40})
        self.assertEqual(
            deployed_lock_shas(self.models_root, "qwen3-asr-1.7b"),
            {SHA, "c" * 40},
        )


class TestGcProtectsDeployedLocks(_TmpTree):
    def _hand_provision(self, rev: str, stale: float) -> None:
        """Create a revision dir WITHOUT touching history.jsonl (as if the
        models tree predates the journal or history was rotated)."""

        d = self.models_root / "qwen3-asr-1.7b" / rev
        d.mkdir(parents=True)
        (d / "config.json").write_text("hello-world")
        (d / "model.safetensors").write_text("weights")
        (d / ".complete").write_text("ok\n")
        os.utime(d, (stale, stale))

    def test_gc_apply_preserves_revision_present_in_deployed_locks(self):
        # Deployed sha: old (> recent window), NOT in history.jsonl, but an
        # image currently running references it via its embedded models.lock
        # (journaled by verify). GC must PRESERVE it.
        deployed = "d" * 40
        self._import_rev(SHA)  # active; history holds only SHA
        stale = time.time() - 30 * 86400
        self._hand_provision("1" * 40, stale)  # extra old rev (GC-eligible)
        self._hand_provision(deployed, stale)
        record_deployed_lock(self.models_root, {"qwen3-asr-1.7b": deployed})
        gc_command(
            ["--model", "qwen3-asr-1.7b", "--apply"],
            models_root=self.models_root,
        )
        remaining = sorted(
            p.name for p in (self.models_root / "qwen3-asr-1.7b").iterdir()
        )
        self.assertIn(
            deployed, remaining, "deployed lock sha must be GC-protected"
        )
        self.assertIn(SHA, remaining)  # current

    def test_gc_without_journal_would_delete_the_deployed_sha(self):
        # Causality: the exact same tree WITHOUT the journal — the deployed
        # sha is collected (old + absent from history + not protected).
        deployed = "d" * 40
        self._import_rev(SHA)
        stale = time.time() - 30 * 86400
        self._hand_provision("1" * 40, stale)
        self._hand_provision(deployed, stale)
        gc_command(
            ["--model", "qwen3-asr-1.7b", "--apply"],
            models_root=self.models_root,
        )
        remaining = sorted(
            p.name for p in (self.models_root / "qwen3-asr-1.7b").iterdir()
        )
        self.assertNotIn(
            deployed, remaining, "without the journal the sha is GC-eligible"
        )

    def test_gc_still_deletes_when_not_in_deployed_locks(self):
        orphan = "e" * 40  # never referenced by any deployed lock
        self._import_rev("1" * 40)
        self._import_rev(orphan)
        self._import_rev(SHA)
        self._import_rev("2" * 40)  # becomes previous
        record_deployed_lock(self.models_root, {"qwen3-asr-1.7b": SHA})
        stale = time.time() - 30 * 86400
        os.utime(self.models_root / "qwen3-asr-1.7b" / orphan, (stale, stale))
        os.utime(self.models_root / "qwen3-asr-1.7b" / ("1" * 40), (stale, stale))
        gc_command(
            ["--model", "qwen3-asr-1.7b", "--apply"],
            models_root=self.models_root,
        )
        remaining = sorted(
            p.name for p in (self.models_root / "qwen3-asr-1.7b").iterdir()
        )
        self.assertNotIn(orphan, remaining)
        self.assertIn(SHA, remaining)


if __name__ == "__main__":
    unittest.main()
