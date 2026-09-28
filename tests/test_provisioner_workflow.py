"""T4b (plan E5 §2.2/§3-T4) — CI provisioner-docker-publish.yml + Dockerfile provisioner.

Strict TDD: échoue (RED) tant que la workflow de build/push de
ghcr.io/charnesp/whisperx-provisioner et son Dockerfile minimal n'existent pas.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "provisioner-docker-publish.yml"
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile.provisioner"


def _load_workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text())


def _on_section(wf: dict) -> dict:
    if True in wf:
        return wf[True]
    return wf["on"]


class TestProvisionerWorkflowTriggers(unittest.TestCase):
    def test_workflow_file_exists(self):
        self.assertTrue(WORKFLOW_PATH.is_file(), f"missing {WORKFLOW_PATH}")

    def test_trigger_workflow_dispatch_present(self):
        self.assertIn("workflow_dispatch", _on_section(_load_workflow()))

    def test_push_paths_include_provision_script(self):
        push = _on_section(_load_workflow())["push"]
        paths = push.get("paths") or []
        self.assertTrue(
            any(p.lstrip("./") == "scripts/provision.py" for p in paths),
            f"push.paths must include scripts/provision.py, got {paths}",
        )

    def test_push_branches_include_feat_glob(self):
        branches = _on_section(_load_workflow())["push"].get("branches") or []
        self.assertTrue(
            any(b == "feat/**" for b in branches),
            f"push.branches must include 'feat/**', got {branches}",
        )


class TestProvisionerWorkflowContent(unittest.TestCase):
    def test_targets_whisperx_provisioner_image(self):
        raw = WORKFLOW_PATH.read_text()
        self.assertIn("ghcr.io/charnesp/whisperx-provisioner", raw)

    def test_docker_build_uses_provisioner_dockerfile(self):
        raw = WORKFLOW_PATH.read_text()
        self.assertIn("Dockerfile.provisioner", raw, "build must use Dockerfile.provisioner")
        self.assertTrue(
            "docker build" in raw or "docker/build-push-action" in raw,
            "no docker build step found",
        )

    def test_pushes_immutable_short_sha_tag(self):
        raw = WORKFLOW_PATH.read_text()
        # Le calcul sha-<short8> vit dans scripts/compute_image_tags.sh.
        script = (REPO_ROOT / "scripts" / "compute_image_tags.sh").read_text()
        combined = raw + "\n" + script
        self.assertIn("sha-", combined)
        self.assertTrue(any(s in combined for s in ("GITHUB_SHA::8", "GITHUB_SHA:0:8")))
        self.assertTrue(
            "docker push" in raw or "push: true" in raw,
            "no docker push step found",
        )

    def test_never_latest_outside_main(self):
        """HIGH 2 (revue): garde :latest mordante — délégation + égalité exacte."""
        raw = WORKFLOW_PATH.read_text()
        branches = _on_section(_load_workflow())["push"].get("branches") or []
        main_only = set(branches) <= {"main"}
        if main_only:
            self.skipTest("workflow main-only: garde non requise")
        # Délégation au script testé unitairement (sémantique réelle).
        self.assertIn(
            "compute_image_tags.sh",
            raw,
            "workflow must delegate tag computation to scripts/compute_image_tags.sh",
        )
        # Le buggy prefix-match est interdit dans le workflow.
        self.assertNotIn(
            "GITHUB_REF::12",
            raw,
            "prefix-match guard (${GITHUB_REF::12}) must be gone",
        )
        # L'égalité exacte vit dans le script (source unique de vérité).
        script = (REPO_ROOT / "scripts" / "compute_image_tags.sh").read_text()
        self.assertIn('[ "$GITHUB_REF" = "refs/heads/main" ]', script)
        self.assertIn('[ "$GITHUB_REF" = "refs/heads/master" ]', script)

    def test_ghcr_login_uses_github_token(self):
        raw = WORKFLOW_PATH.read_text()
        self.assertIn("secrets.GITHUB_TOKEN", raw)
        self.assertIn("packages: write", raw)

    def test_tags_persisted_newline_separated(self):
        """FIX (revue E5): build-push-action splitte `tags:` sur newlines.

        Un TAGS espace-séparé persisté via `printf 'TAGS=%s'` arrive côté
        buildx comme UN SEUL tag invalide. Le step Compute tags doit donc:
        - écrire TAGS en heredoc multi-ligne (TAGS<<EOF),
        - convertir les espaces en newlines (${TAGS// /$'\\n'} ou équivalent),
        - ne plus persister TAGS avec un printf à espace direct.
        docker-publish.yml, lui, garde sa boucle `for t in $TAGS` (shell
        word-splitting) — aucune conversion newline requise là-bas.
        """
        provisioner = WORKFLOW_PATH.read_text()
        compute_step = provisioner.split("Compute tags", 1)[1].split("- name:", 1)[0]
        self.assertIn("TAGS<<EOF", compute_step, "TAGS must be persisted as GITHUB_ENV heredoc")
        self.assertTrue(
            "${TAGS// /$'\\n'}" in compute_step
            or "${TAGS// /\\n}" in compute_step
            or "${TAGS// /" in compute_step,
            "spaces in TAGS must be converted to newlines before persistence",
        )
        self.assertNotRegex(
            compute_step,
            r"printf\s+'TAGS=%s\\n'[^|]*\"\$\{?TAGS\}?\"",
            "direct space-separated TAGS persistence is forbidden",
        )
        # Pas de régression: docker-publish.yml garde sa boucle for (shell).
        docker_publish = (REPO_ROOT / ".github" / "workflows" / "docker-publish.yml").read_text()
        self.assertIn("for t in $TAGS", docker_publish, "docker-publish.yml must keep its for loop over $TAGS")


class TestProvisionerDockerfile(unittest.TestCase):
    def test_dockerfile_exists(self):
        self.assertTrue(DOCKERFILE_PATH.is_file(), f"missing {DOCKERFILE_PATH}")

    def test_minimal_python_slim_base(self):
        raw = DOCKERFILE_PATH.read_text()
        self.assertIn("FROM python:3.12-slim", raw)

    def test_copies_provision_script_registry_and_lock(self):
        raw = DOCKERFILE_PATH.read_text()
        self.assertIn("scripts/provision.py", raw)
        self.assertIn("models_registry.py", raw)
        self.assertIn("models.lock", raw)

    def test_entrypoint_is_provision(self):
        raw = DOCKERFILE_PATH.read_text()
        self.assertIn("ENTRYPOINT", raw)
        self.assertIn("provision.py", raw.split("ENTRYPOINT", 1)[1])


if __name__ == "__main__":
    unittest.main()
