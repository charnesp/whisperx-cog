"""Unit tests for the atomic model provisioner (no GPU, no network).

All subcommands are exercised against a tmp tree: a fake models.lock (v2),
a staging source dir, and a fake /models root. Download-based provision
shares import's checksum/rename/.complete code path, so it is exercised
through import (network stays out of unit CI).

Structure per plan-e5-deploiement-canary.md §2.1/§2.2:
    <models>/<model>/<sha40>/... + .complete (written LAST, after fsync)
    <models>/.locks/<model>.lock (flock, OUTSIDE versioned dirs)
    <models>/.state/active.json + history.jsonl
"""

import fcntl
import hashlib
import json
import os
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

from scripts.provision import (
    CommandError,
    acquire_lock,
    complete_marker_path,
    gc_command,
    import_command,
    is_locked_elsewhere,
    load_lock,
    model_revisions_on_disk,
    provision_one,
    resolve_models_root,
    verify_command,
)

SHA = "a" * 40
CFG_SHA = hashlib.sha256(b"hello-world").hexdigest()
W_SHA = hashlib.sha256(b"weights").hexdigest()


def lock_yaml(sha: str) -> str:
    return textwrap.dedent(
        f"""\
        version: 2
        models:
          - repo: Qwen/Qwen3-ASR-1.7B
            revision: {sha}
            required: true
            files:
              - path: config.json
                size: 11
                sha256: {CFG_SHA}
                lfs: false
              - path: model.safetensors
                size: 7
                sha256: {W_SHA}
                lfs: true
        """
    )


def write_lock(models_root: Path, sha: str = SHA) -> Path:
    lock_path = models_root / "models.lock"
    lock_path.write_text(lock_yaml(sha))
    return lock_path


def make_source(staging: Path, sha: str = SHA, tamper: str | None = None) -> Path:
    """Build a staging dir whose file hashes match the fake lock."""
    src = staging / f"src-{sha[:8]}"
    src.mkdir(parents=True, exist_ok=True)
    (src / "config.json").write_text("hello-world")
    weights = src / "model.safetensors"
    weights.write_text(tamper if tamper else "weights")
    return src


class ProvisionTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.models_root = self.tmp / "models"
        self.models_root.mkdir()
        self.staging = self.tmp / "staging"
        self.staging.mkdir()
        write_lock(self.models_root)
        self.src = make_source(self.staging)

    def tearDown(self):
        self._tmp.cleanup()

    # -- helpers -----------------------------------------------------------

    def run_import(self, extra=None, model="qwen3-asr-1.7b", revision=None, source=None):
        args = ["--model", model, "--from", str(source or self.src)]
        if revision:
            args += ["--revision", revision]
        if extra:
            args += extra
        return import_command(args, models_root=self.models_root, staging_root=self.staging)

    def provision_rev(self, rev):
        src = make_source(self.staging, rev)
        self.run_import(revision=rev, source=src)

    def _backdate(self, rev, days=30):
        stale = time.time() - days * 86400
        os.utime(self.models_root / "qwen3-asr-1.7b" / rev, (stale, stale))

    def final_dir(self, revision=None):
        return self.models_root / "qwen3-asr-1.7b" / (revision or SHA)

    # -- lock parsing --------------------------------------------------------

    def test_load_lock_v2(self):
        lock = load_lock(self.models_root / "models.lock")
        self.assertEqual(lock["version"], 2)
        m = lock["models"][0]
        self.assertEqual(m["revision"], SHA)
        self.assertEqual(len(m["files"]), 2)
        self.assertEqual(m["files"][0]["path"], "config.json")

    def test_load_lock_rejects_v1(self):
        bad = self.tmp / "bad.lock"
        bad.write_text("models:\n  - repo: x\n")
        with self.assertRaises(CommandError):
            load_lock(bad)

    # -- atomicity -----------------------------------------------------------

    def test_import_creates_complete_last(self):
        self.run_import()
        d = self.final_dir()
        self.assertTrue((d / "config.json").exists())
        self.assertTrue((d / "model.safetensors").exists())
        self.assertTrue((d / ".complete").exists())
        newest = max(d.rglob("*"), key=lambda p: p.stat().st_mtime)
        self.assertEqual(newest.name, ".complete")

    def test_no_complete_marker_on_checksum_failure(self):
        src = make_source(self.staging, tamper="tampered")
        with self.assertRaises(CommandError):
            self.run_import(source=src)
        self.assertFalse(self.final_dir().exists(), "dest must not appear on failed import")

    def test_failed_import_leaves_no_staging_leftovers(self):
        src = make_source(self.tmp, tamper="tampered")
        with self.assertRaises(CommandError):
            self.run_import(source=src)
        leftovers = [p.name for p in self.staging.iterdir() if p.is_dir() and p.name.startswith("qwen3-asr-1.7b")]
        self.assertEqual(leftovers, [], "failed staging dirs must be cleaned up")

    def test_import_is_idempotent(self):
        self.run_import()
        first_mtime = (self.final_dir() / "config.json").stat().st_mtime
        self.run_import()  # re-run must be a no-op
        self.assertEqual((self.final_dir() / "config.json").stat().st_mtime, first_mtime)

    def test_import_accepts_previous_revision_for_rollback(self):
        # import (local source) may provision a non-pinned revision: that is
        # the rollback / history-keeper path (plan §6.5 keep-2).
        self.run_import(revision="b" * 40)
        self.assertTrue(self.final_dir("b" * 40, ).exists())

    def test_provision_download_guard_rejects_unpinned_revision(self):
        # The HF-download subcommand refuses an unpinned sha (lock is truth).
        entry = {"repo": "Qwen/Qwen3-ASR-1.7B", "revision": SHA}
        from scripts.provision import _resolve_revision
        with self.assertRaises(CommandError):
            _resolve_revision(entry, "b" * 40)

    def test_import_missing_source_file_fails(self):
        src = make_source(self.staging)
        (src / "config.json").unlink()
        with self.assertRaises(CommandError):
            self.run_import(source=src)
        self.assertFalse(self.final_dir().exists())

    # -- verify / repair -----------------------------------------------------

    def test_verify_ok(self):
        self.run_import()
        verify_command(["--model", "qwen3-asr-1.7b"], models_root=self.models_root)

    def test_verify_fails_on_corrupt_file(self):
        self.run_import()
        (self.final_dir() / "model.safetensors").write_text("corrupted!!")
        with self.assertRaises(CommandError):
            verify_command(["--model", "qwen3-asr-1.7b"], models_root=self.models_root)

    def test_verify_repair_reprovisions(self):
        self.run_import()
        (self.final_dir() / "model.safetensors").write_text("corrupted!!")
        verify_command(
            ["--model", "qwen3-asr-1.7b", "--repair", "--from", str(self.src)],
            models_root=self.models_root,
            staging_root=self.staging,
        )
        verify_command(["--model", "qwen3-asr-1.7b"], models_root=self.models_root)

    def test_verify_unprovisioned_model_raises(self):
        with self.assertRaises(CommandError):
            verify_command(["--model", "qwen3-asr-1.7b"], models_root=self.models_root)

    # -- GC ------------------------------------------------------------------

    def test_gc_dry_run_by_default_deletes_nothing(self):
        for rev in ("1" * 40, "2" * 40, SHA):
            self.provision_rev(rev)
        before = sorted(str(p) for p in self.models_root.rglob("*"))
        gc_command(["--model", "qwen3-asr-1.7b"], models_root=self.models_root)
        after = sorted(str(p) for p in self.models_root.rglob("*"))
        self.assertEqual(before, after, "gc without --apply must not delete anything")
        self.assertEqual(len(list((self.models_root / "qwen3-asr-1.7b").iterdir())), 3)

    def test_gc_apply_keeps_two(self):
        for rev in ("1" * 40, "2" * 40, "3" * 40, SHA):
            self.provision_rev(rev)
        self._backdate("1" * 40)
        self._backdate("2" * 40)
        gc_command(["--model", "qwen3-asr-1.7b", "--apply"], models_root=self.models_root)
        remaining = sorted(p.name for p in (self.models_root / "qwen3-asr-1.7b").iterdir())
        self.assertEqual(len(remaining), 2)
        self.assertIn(SHA, remaining, "active sha always kept")

    def test_gc_protected_set_includes_history(self):
        self.provision_rev("1" * 40)
        prev = "9" * 40
        self.provision_rev(prev)
        self.provision_rev(SHA)
        self._backdate("1" * 40)
        self._backdate(prev)
        history = self.models_root / ".state" / "history.jsonl"
        with history.open("a") as fh:
            fh.write(json.dumps({"model": "qwen3-asr-1.7b", "sha": prev}) + "\n")
        gc_command(["--model", "qwen3-asr-1.7b", "--apply"], models_root=self.models_root)
        remaining = sorted(p.name for p in (self.models_root / "qwen3-asr-1.7b").iterdir())
        self.assertIn(SHA, remaining)  # current
        self.assertIn(prev, remaining)  # previous (history.jsonl)

    def test_gc_skips_locked_model(self):
        for rev in ("1" * 40, "2" * 40, SHA):
            self.provision_rev(rev)
        self._backdate("1" * 40)
        self._backdate("2" * 40)
        lock_dir = self.models_root / ".locks"
        lock_path = lock_dir / "qwen3-asr-1.7b.lock"
        with open(lock_path, "w") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                gc_command(
                    ["--model", "qwen3-asr-1.7b", "--apply"],
                    models_root=self.models_root,
                )
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        self.assertEqual(len(list((self.models_root / "qwen3-asr-1.7b").iterdir())), 3)

    def test_gc_protects_recent_revisions(self):
        old_rev, recent_rev = "1" * 40, "2" * 40
        self.provision_rev(old_rev)
        self.provision_rev(recent_rev)
        self.provision_rev(SHA)
        old_dir = self.models_root / "qwen3-asr-1.7b" / old_rev
        stale = time.time() - 30 * 86400
        os.utime(old_dir, (stale, stale))
        gc_command(["--model", "qwen3-asr-1.7b", "--apply"], models_root=self.models_root)
        remaining = sorted(p.name for p in (self.models_root / "qwen3-asr-1.7b").iterdir())
        self.assertIn(recent_rev, remaining, "recent revision (< N days) is protected")
        self.assertIn(SHA, remaining)

    # -- locks ---------------------------------------------------------------

    def test_acquire_lock_creates_lock_outside_versioned_dirs(self):
        with acquire_lock(self.models_root, "qwen3-asr-1.7b") as fh:
            self.assertTrue(fh)
            self.assertTrue((self.models_root / ".locks" / "qwen3-asr-1.7b.lock").exists())
            self.assertNotIn(".locks", self.final_dir().parts)

    def test_is_locked_elsewhere_true_when_held(self):
        lock_dir = self.models_root / ".locks"
        lock_dir.mkdir(exist_ok=True)
        lock_path = lock_dir / "qwen3-asr-1.7b.lock"
        with open(lock_path, "w") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertTrue(is_locked_elsewhere(lock_path))
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        self.assertFalse(is_locked_elsewhere(lock_path))

    # -- state ---------------------------------------------------------------

    def test_import_updates_state_files(self):
        self.run_import()
        state = self.models_root / ".state"
        active = json.loads((state / "active.json").read_text())
        self.assertEqual(active["qwen3-asr-1.7b"], SHA)
        history = (state / "history.jsonl").read_text().strip().splitlines()
        self.assertEqual(len(history), 1)
        self.assertEqual(json.loads(history[0])["sha"], SHA)

    def test_history_grows_across_provisions(self):
        self.run_import()
        self.provision_rev("2" * 40)
        history = (self.models_root / ".state" / "history.jsonl").read_text().strip().splitlines()
        shas = [json.loads(line)["sha"] for line in history]
        self.assertIn(SHA, shas)
        self.assertIn("2" * 40, shas)

    # -- structure -----------------------------------------------------------

    def test_models_root_resolution_default(self):
        self.assertEqual(resolve_models_root(None), Path("/models"))

    def test_complete_marker_path(self):
        self.assertEqual(
            complete_marker_path(self.models_root, "qwen3-asr-1.7b", SHA),
            self.models_root / "qwen3-asr-1.7b" / SHA / ".complete",
        )

    def test_model_revisions_on_disk(self):
        self.provision_rev("1" * 40)
        self.provision_rev("2" * 40)
        revs = model_revisions_on_disk(self.models_root, "qwen3-asr-1.7b")
        self.assertEqual(sorted(revs), ["1" * 40, "2" * 40])

    def test_provision_one_rename_is_atomic_move(self):
        dest_parent = self.models_root / "qwen3-asr-1.7b"
        staging_dir = self.staging / "stage-x"
        staging_dir.mkdir()
        (staging_dir / "config.json").write_text("hello-world")
        with mock.patch("scripts.provision.fsync_dir") as fsync_mock:
            provision_one(
                staging_dir,
                dest_parent / SHA,
                [{"path": "config.json", "size": 11, "sha256": CFG_SHA, "lfs": False}],
                complete_last=True,
            )
            fsync_mock.assert_called()
        self.assertTrue((dest_parent / SHA / ".complete").exists())


if __name__ == "__main__":
    unittest.main()
