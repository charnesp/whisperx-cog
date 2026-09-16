"""T7 (plan E5 §3/§4) — CI docker-publish.yml buildable depuis feat/qwen3-asr-backend.

Strict TDD: ces tests échouent (RED) tant que le workflow ne déclenche pas
sur les branches feat/**, ne pousse pas les tags canary/feat/sha-short et
risque de pousser :latest depuis une branche (jamais hors main).
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "docker-publish.yml"


def _load_workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text())


def _on_section(wf: dict) -> dict:
    # PyYAML charge la clé `on` comme True (booléen).
    if True in wf:
        return wf[True]
    return wf["on"]


class TestDockerPublishTriggers(unittest.TestCase):
    def test_workflow_file_exists(self):
        self.assertTrue(WORKFLOW_PATH.is_file(), f"missing {WORKFLOW_PATH}")

    def test_trigger_push_branches_include_feat_glob(self):
        push = _on_section(_load_workflow())["push"]
        branches = push.get("branches") or []
        self.assertTrue(
            any(b == "feat/**" for b in branches),
            f"push.branches must include 'feat/**', got {branches}",
        )

    def test_trigger_workflow_dispatch_present(self):
        on = _on_section(_load_workflow())
        self.assertIn("workflow_dispatch", on)

    def test_push_branches_still_cover_main(self):
        branches = _on_section(_load_workflow())["push"].get("branches") or []
        self.assertTrue(any(b in ("main", "master") for b in branches))


class TestDockerPublishTags(unittest.TestCase):
    def test_push_step_tags_canary_and_branch(self):
        """Le workflow calcule et pousse sha-<short8>, canary et feat-qwen3-asr-backend."""
        raw = WORKFLOW_PATH.read_text()
        self.assertIn("sha-", raw, "sha-<short8> tag prefix missing")
        self.assertTrue(
            any(s in raw for s in ("GITHUB_SHA::8", "GITHUB_SHA:0:8")),
            "short8 computation from GITHUB_SHA missing",
        )
        self.assertIn("canary", raw, "canary tag missing")
        self.assertIn("feat-qwen3-asr-backend", raw, "branch tag missing")
        self.assertIn("docker push", raw, "no docker push step")


class TestDockerPublishLatestGuard(unittest.TestCase):
    def test_never_latest_outside_main(self):
        """Aucun :latest pushé hors main: logique explicite (latest=false hors main)."""
        raw = WORKFLOW_PATH.read_text()
        wf = _load_workflow()
        branches = _on_section(wf)["push"].get("branches") or []
        main_only = set(branches) <= {"main"}
        if main_only:
            self.skipTest("workflow main-only: pas de push branche, garde non requise")
        self.assertTrue(
            "latest=false" in raw or "IS_MAIN" in raw or "IS_DEFAULT_BRANCH" in raw,
            "no explicit latest=false / main-branch guard found in workflow",
        )
        self.assertNotIn(
            "Tag image as latest",
            raw,
            "unconditional 'Tag image as latest' step must be gone (never :latest off main)",
        )


if __name__ == "__main__":
    unittest.main()
