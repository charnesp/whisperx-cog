"""T7b — workflow bridge-docker-publish.yml: :latest gardé sur main uniquement.

Contrat (revue E5-BRIDGE-FIX):
- le workflow ne construit que depuis main/master (pas de feat/** en trigger),
- la garde :latest est déléguée à scripts/compute_image_tags.sh (même contrat
  que docker-publish.yml: égalité EXACTE refs/heads/main | refs/heads/master,
  pas de prefix-match),
- les tags poussés depuis feat sont sha-<short8>, canary et le nom de branche,
- IMAGE = ghcr.io/charnesp/whisperx-cog-bridge.

RED attendu contre le workflow actuel: push :latest inconditionnel
(step "Tag image as latest" + push :latest sans garde).
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = (
    REPO_ROOT / ".github" / "workflows" / "bridge-docker-publish.yml"
)
SCRIPT_PATH = REPO_ROOT / "scripts" / "compute_image_tags.sh"

EXPECTED_IMAGE = "ghcr.io/charnesp/whisperx-cog-bridge"


def _load_workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text())


def _on_section(wf: dict) -> dict:
    # PyYAML charge la clé `on` comme True (booléen).
    if True in wf:
        return wf[True]
    return wf["on"]


class TestBridgeWorkflowTriggers(unittest.TestCase):
    def test_workflow_file_exists(self):
        self.assertTrue(WORKFLOW_PATH.is_file(), f"missing {WORKFLOW_PATH}")

    def test_trigger_push_branches_main_only(self):
        push = _on_section(_load_workflow())["push"]
        branches = push.get("branches") or []
        self.assertGreater(len(branches), 0, "push.branches must not be empty")
        for branch in branches:
            self.assertIn(
                branch,
                ("main", "master"),
                f"bridge workflow must only build from main/master, got {branch!r}",
            )

    def test_trigger_workflow_dispatch_present(self):
        on = _on_section(_load_workflow())
        self.assertIn("workflow_dispatch", on)


class TestBridgeWorkflowTags(unittest.TestCase):
    def test_workflow_delegates_to_compute_image_tags_script(self):
        """Le workflow réutilise scripts/compute_image_tags.sh (pas de nouvelle logique)."""
        raw = WORKFLOW_PATH.read_text()
        self.assertIn(
            "compute_image_tags.sh",
            raw,
            "workflow must delegate tag computation to scripts/compute_image_tags.sh",
        )
        self.assertIn(
            "eval",
            raw,
            "workflow must eval the script output (KEY=VALUE lines)",
        )

    def test_workflow_image_is_bridge_repo(self):
        raw = WORKFLOW_PATH.read_text()
        self.assertIn(
            EXPECTED_IMAGE,
            raw,
            f"IMAGE must be {EXPECTED_IMAGE}",
        )

    def test_workflow_pushes_computed_tags_and_never_unconditional_latest(self):
        raw = WORKFLOW_PATH.read_text()

        # La garde :latest n'est requise que si le workflow peut tourner sur
        # une branche hors main (trigger workflow_dispatch). Contrat: la
        # poussée :latest passe par le script (égalité exacte), jamais par un
        # step docker tag/push :latest inconditionnel.
        self.assertNotIn(
            "Tag image as latest",
            raw,
            "unconditional 'Tag image as latest' step must be gone",
        )
        self.assertNotIn(
            "${{ env.IMAGE_NAME }}:latest",
            raw,
            "hardcoded :latest push must be gone — tags computed by compute_image_tags.sh",
        )
        self.assertIn(
            "docker push",
            raw,
            "no docker push step",
        )
        # Le calcul des tags (sha-<short8>, canary, branche) vit dans le script.
        script = SCRIPT_PATH.read_text()
        combined = raw + "\n" + script
        self.assertIn("sha-", combined, "sha-<short8> tag prefix missing")
        self.assertTrue(
            any(s in combined for s in ("GITHUB_SHA::8", "GITHUB_SHA:0:8")),
            "short8 computation from GITHUB_SHA missing",
        )
        self.assertIn("canary", combined, "canary tag missing")
        # Garde :latest: égalité exacte, dans le script (source unique de vérité).
        self.assertIn('[ "$GITHUB_REF" = "refs/heads/main" ]', script)
        self.assertIn('[ "$GITHUB_REF" = "refs/heads/master" ]', script)
        self.assertNotIn(
            "GITHUB_REF::12",
            raw + "\n" + script,
            "prefix-match guard is forbidden",
        )


if __name__ == "__main__":
    unittest.main()
