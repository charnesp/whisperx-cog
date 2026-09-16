"""FIX 1 (🔴, E5-CODE review): lock_entry() resolves the 5 registry keys.

The provisioner's lock_entry() matched lock entries by HF repo basename
only, so the registry keys 'tiny', 'large-v3', 'large-v3-turbo' and
'qwen3-forced-aligner-0.6b' raised CommandError — 4/5 keys failed (only
'qwen3-asr-1.7b' matched its repo basename). The remediation line printed
by the validators (`provision --model tiny`) failed when followed.

These tests exercise the REAL repo models.lock (not a fixture) so any
key/repo mismatch shows up. Contract: every registry key resolves to its
own lock entry, the alias resolves too, and unknown keys still fail.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models_registry import ALIASES, MODELS, resolve_key  # noqa: E402
from scripts.provision import (  # noqa: E402
    CommandError,
    load_lock,
    lock_entry,
)

REAL_LOCK = REPO_ROOT / "models.lock"


class TestLockEntryResolvesRegistryKeysOnRealLock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not REAL_LOCK.is_file():
            raise unittest.SkipTest("repo models.lock missing")
        cls.lock = load_lock(REAL_LOCK)

    def test_real_lock_has_five_entries(self):
        self.assertEqual(len(self.lock["models"]), 5)

    def test_all_five_registry_keys_resolve(self):
        for key in sorted(MODELS):
            entry = lock_entry(self.lock, key)
            self.assertEqual(entry["repo"], MODELS[key].hf_repo)
            self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")
            self.assertTrue(entry["files"], f"{key}: empty files list")

    def test_registry_hf_repos_are_lock_superset(self):
        # every registry key's HF repo must exist in the real lock
        repos = {e.get("repo") for e in self.lock["models"]}
        for key in sorted(MODELS):
            self.assertIn(MODELS[key].hf_repo, repos)

    def test_qwen_alias_resolves(self):
        for alias in sorted(ALIASES):
            entry = lock_entry(self.lock, alias)
            self.assertEqual(entry["repo"], MODELS[resolve_key(alias)].hf_repo)

    def test_full_removal_key_never_resolves(self):
        with self.assertRaises(CommandError):
            lock_entry(self.lock, "9" * 40)


if __name__ == "__main__":
    unittest.main()