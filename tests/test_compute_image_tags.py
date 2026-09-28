"""T7+T4b FIX (revue) — garde :latest en égalité exacte via scripts/compute_image_tags.sh.

Rejet revue T7+T4b:
  HIGH 1: garde main en prefix-match (${GITHUB_REF::12} == 'refs/heads/m')
          matche refs/heads/mainx, masterx, main2, maint → :latest poussé hors main.
  HIGH 2: test_never_latest_outside_main ne mord pas (grep de présence).

Approche: le calcul des tags est extrait dans scripts/compute_image_tags.sh,
exécutable unitairement; les workflows l'appellent; ces tests exécutent le
script avec GITHUB_REF variés et simulent le sabotage (garde retirée) pour
vérifier que la détection mord.

Strict TDD: RED tant que le script n'existe pas et que les workflows gardent
le calcul inline préfixe.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "compute_image_tags.sh"
DOCKER_WF = REPO_ROOT / ".github" / "workflows" / "docker-publish.yml"
PROVISIONER_WF = REPO_ROOT / ".github" / "workflows" / "provisioner-docker-publish.yml"

IMAGE = "ghcr.io/test/repo"
SHA = "deadbeefdeadbeef"


def _run_script(script_path: Path, ref: str, ref_name: str = "") -> dict:
    """Exécute le script et dés-quote ses sorties (%q) comme le font les workflows.

    Le script sort des lignes KEY=%q (quoté bash pour eval). On évalue sa
    sortie dans un bash puis on imprime les valeurs brutes, une par ligne.
    """
    env = dict(os.environ)
    env.update(
        IMAGE=IMAGE,
        GITHUB_REF=ref,
        GITHUB_REF_NAME=ref_name,
        GITHUB_SHA=SHA,
    )
    env.pop("GITHUB_ENV", None)
    inner = (
        f'eval "$(IMAGE="$IMAGE" GITHUB_SHA="$GITHUB_SHA" GITHUB_REF="$GITHUB_REF" '
        f'GITHUB_REF_NAME="$GITHUB_REF_NAME" bash {script_path})"\n'
        'printf "IMAGE=%s\\nTAGS=%s\\nIS_DEFAULT_BRANCH=%s\\nSHORT_SHA=%s\\n" '
        '"$IMAGE" "$TAGS" "$IS_DEFAULT_BRANCH" "$SHORT_SHA"\n'
    )
    proc = subprocess.run(
        ["bash", "-c", inner],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"compute_image_tags.sh failed rc={proc.returncode}\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
    out = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def _tags(out: dict) -> set:
    return set(out.get("TAGS", "").split())


def _script_contract_violations(script_path: Path) -> list:
    """Contrat fonctionnel du script. Retourne la liste des violations (vide = OK)."""
    violations = []

    def _run(ref, ref_name=""):
        try:
            return _run_script(script_path, ref, ref_name)
        except AssertionError as exc:
            violations.append(f"execution failed for {ref}: {exc}")
            return {"TAGS": ""}

    # main/master (égalité exacte) → sha- + canary + latest
    for ref in ("refs/heads/main", "refs/heads/master"):
        tags = _tags(_run(ref))
        expected_base = {f"{IMAGE}:sha-{SHA[:8]}", f"{IMAGE}:canary"}
        if not expected_base <= tags:
            violations.append(f"{ref}: missing sha-/canary tags, got {sorted(tags)}")
        if f"{IMAGE}:latest" not in tags:
            violations.append(f"{ref}: :latest missing on exact default branch")
        if any(t for t in tags if "feat" in t):
            violations.append(f"{ref}: unexpected feat tag in {sorted(tags)}")

    # Variantes préfixe de main/master → JAMAIS :latest
    for ref in (
        "refs/heads/mainx",
        "refs/heads/masterx",
        "refs/heads/main2",
        "refs/heads/maint",
    ):
        tags = _tags(_run(ref, ref.split("/")[-1]))
        if f"{IMAGE}:latest" in tags:
            violations.append(f"{ref}: :latest pushed outside main (prefix-match bug)")
        if not {f"{IMAGE}:sha-{SHA[:8]}", f"{IMAGE}:canary"} <= tags:
            violations.append(f"{ref}: missing sha-/canary tags, got {sorted(tags)}")

    # Branche feat → sha- + canary + tag de branche, sans :latest
    tags = _tags(_run("refs/heads/feat/qwen3-asr-backend", "feat/qwen3-asr-backend"))
    if f"{IMAGE}:feat-qwen3-asr-backend" not in tags:
        violations.append(f"feat branch: branch tag missing, got {sorted(tags)}")
    if f"{IMAGE}:latest" in tags:
        violations.append("feat branch: :latest pushed outside main")
    if not {f"{IMAGE}:sha-{SHA[:8]}", f"{IMAGE}:canary"} <= tags:
        violations.append(f"feat branch: missing sha-/canary, got {sorted(tags)}")

    return violations


def _on_section(wf: dict) -> dict:
    if True in wf:
        return wf[True]
    return wf["on"]


class TestComputeImageTagsScript(unittest.TestCase):
    def test_script_exists_and_is_bash(self):
        self.assertTrue(SCRIPT_PATH.is_file(), f"missing {SCRIPT_PATH}")
        self.assertTrue(os.access(SCRIPT_PATH, os.X_OK), "script must be executable")
        head = SCRIPT_PATH.read_text().splitlines()[0]
        self.assertIn("bash", head, "script must have a bash shebang")

    def test_main_exact_gets_latest_canary_sha(self):
        out = _run_script(SCRIPT_PATH, "refs/heads/main")
        tags = _tags(out)
        self.assertIn(f"{IMAGE}:latest", tags)
        self.assertIn(f"{IMAGE}:canary", tags)
        self.assertIn(f"{IMAGE}:sha-{SHA[:8]}", tags)

    def test_master_exact_gets_latest(self):
        tags = _tags(_run_script(SCRIPT_PATH, "refs/heads/master"))
        self.assertIn(f"{IMAGE}:latest", tags)

    def test_prefix_variants_never_get_latest(self):
        """Le bug initial: prefix-match 'refs/heads/m' matchait mainx/masterx/maint/main2."""
        for ref in ("refs/heads/mainx", "refs/heads/masterx", "refs/heads/maint", "refs/heads/main2"):
            tags = _tags(_run_script(SCRIPT_PATH, ref, ref.split("/")[-1]))
            self.assertNotIn(
                f"{IMAGE}:latest", tags, f":latest must never be pushed for {ref}"
            )
            self.assertIn(f"{IMAGE}:sha-{SHA[:8]}", tags, f"sha tag missing for {ref}")
            self.assertIn(f"{IMAGE}:canary", tags, f"canary missing for {ref}")

    def test_feat_branch_gets_branch_tag_without_latest(self):
        tags = _tags(_run_script(SCRIPT_PATH, "refs/heads/feat/qwen3-asr-backend", "feat/qwen3-asr-backend"))
        self.assertIn(f"{IMAGE}:sha-{SHA[:8]}", tags)
        self.assertIn(f"{IMAGE}:canary", tags)
        self.assertIn(f"{IMAGE}:feat-qwen3-asr-backend", tags)
        self.assertNotIn(f"{IMAGE}:latest", tags)

    def test_full_contract(self):
        """Contrat complet: vide = aucune violation."""
        violations = _script_contract_violations(SCRIPT_PATH)
        self.assertEqual(violations, [], f"contract violations: {violations}")


class TestEvalSplit(unittest.TestCase):
    """E5-BUILD-FIX — eval "$(compute_image_tags.sh)" préserve TAGS multi-mots.

    Bug constaté en CI (exit 127): le script sortait `TAGS=a b c` non-quoté;
    l'eval du workflow assignait TAGS au 1er mot puis EXÉCUTAIT les mots
    suivants comme des commandes ('ghcr.io/...:canary: No such file or
    directory'). Le contrat: rc=0, aucun mot exécuté, TAGS complet.
    """

    def _eval_script(self, script_path: Path, ref: str, ref_name: str = "") -> subprocess.CompletedProcess:
        """Évalue la sortie du script dans un bash, comme le font les workflows."""
        env = dict(os.environ)
        env.update(
            IMAGE=IMAGE,
            GITHUB_REF=ref,
            GITHUB_REF_NAME=ref_name,
            GITHUB_SHA=SHA,
        )
        env.pop("GITHUB_ENV", None)
        inner = (
            f'eval "$(IMAGE="$IMAGE" GITHUB_SHA="$GITHUB_SHA" GITHUB_REF="$GITHUB_REF" '
            f'GITHUB_REF_NAME="$GITHUB_REF_NAME" bash {script_path})"\n'
            'rc=$?\n'
            'echo "EVAL_RC=$rc"\n'
            'echo "TAGS_VALUE=$TAGS"\n'
        )
        return subprocess.run(
            ["bash", "-c", inner],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

    @staticmethod
    def _parsed(proc: subprocess.CompletedProcess) -> tuple:
        """Extrait (eval_rc, tags_value) de la sortie de la couche eval."""
        eval_rc = None
        tags_value = None
        for line in proc.stdout.splitlines():
            if line.startswith("EVAL_RC="):
                eval_rc = int(line.partition("=")[2])
            elif line.startswith("TAGS_VALUE="):
                tags_value = line.partition("=")[2]
        return eval_rc, tags_value

    def test_eval_preserves_full_tags_and_executes_nothing(self):
        """rc=0, aucun stderr d'exécution, TAGS contient toutes les valeurs."""
        cases = [
            (
                "refs/heads/feat/qwen3-asr-backend",
                "feat/qwen3-asr-backend",
                {f"{IMAGE}:sha-{SHA[:8]}", f"{IMAGE}:canary", f"{IMAGE}:feat-qwen3-asr-backend"},
            ),
            (
                "refs/heads/main",
                "main",
                {f"{IMAGE}:sha-{SHA[:8]}", f"{IMAGE}:canary", f"{IMAGE}:latest"},
            ),
        ]
        for ref, ref_name, expected in cases:
            with self.subTest(ref=ref):
                proc = self._eval_script(SCRIPT_PATH, ref, ref_name)
                self.assertEqual(
                    proc.returncode,
                    0,
                    f"eval layer rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}",
                )
                self.assertNotIn("No such file or directory", proc.stderr, "a tag word was EXECUTED as a command")
                self.assertNotIn("command not found", proc.stderr, "a tag word was EXECUTED as a command")
                eval_rc, tags_value = self._parsed(proc)
                self.assertEqual(eval_rc, 0, f"script rc inside eval={eval_rc}\nstderr={proc.stderr}")
                self.assertIsNotNone(tags_value, "TAGS not set after eval")
                got = set(tags_value.split())
                self.assertEqual(
                    got,
                    expected,
                    f"eval lost tag values: got {sorted(got)}, expected {sorted(expected)}",
                )


class TestSabotageBites(unittest.TestCase):
    """Mordant: si la garde est retirée, le contrat doit échouer (détection réelle)."""

    def _sabotized_copy(self, tmpdir: Path) -> Path:
        """Sabotage = retirer la garde (latest inconditionnel)."""
        text = SCRIPT_PATH.read_text()
        # Sabotage = retirer la garde d'égalité exacte (latest inconditionnel),
        # en conservant une syntaxe valide: condition remplacée par `true`.
        sabotaged, n = re.subn(
            r'if\s+\[\s+"\$GITHUB_REF"\s+=\s+"refs/heads/main"\s+\]\s*'
            r'\|\|\s*\[\s+"\$GITHUB_REF"\s+=\s+"refs/heads/master"\s+\]\s*;\s*then',
            "if true; then # SABOTAGE: garde d'égalité exacte retirée",
            text,
        )
        if n == 0:
            # variante: garde mono-condition
            sabotaged, n = re.subn(
                r'if\s+\[\s+"\$GITHUB_REF"\s+=\s+"refs/heads/(?:main|master)"\s+\]\s*;\s*then',
                "if true; then # SABOTAGE: garde d'égalité exacte retirée",
                text,
            )
        self.assertGreater(n, 0, "sabotage pattern not found — guard not where expected")
        copy = tmpdir / "compute_image_tags_sabotaged.sh"
        copy.write_text(sabotaged)
        copy.chmod(0o755)
        return copy

    def test_removing_guard_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            copy = self._sabotized_copy(Path(td))
            out = _run_script(copy, "refs/heads/mainx", "mainx")
            self.assertIn(
                f"{IMAGE}:latest",
                _tags(out),
                "sabotage did not change behavior — test setup invalid",
            )
            violations = _script_contract_violations(copy)
            self.assertTrue(violations, "sabotaged script must violate the contract")

    def test_pristine_script_has_no_violations(self):
        self.assertEqual(_script_contract_violations(SCRIPT_PATH), [])


class TestWorkflowsDelegateToScript(unittest.TestCase):
    """Les 2 workflows appellent le script; plus de calcul de tags inline préfixe."""

    def test_docker_publish_calls_script(self):
        raw = DOCKER_WF.read_text()
        self.assertIn("compute_image_tags.sh", raw, "workflow must call the shared script")

    def test_provisioner_calls_script(self):
        raw = PROVISIONER_WF.read_text()
        self.assertIn("compute_image_tags.sh", raw, "workflow must call the shared script")

    def test_no_prefix_match_github_ref_left_in_workflows(self):
        """HIGH 1: ${GITHUB_REF::12} == 'refs/heads/m' (prefix-match) interdit."""
        for wf_path in (DOCKER_WF, PROVISIONER_WF):
            raw = wf_path.read_text()
            self.assertNotIn("GITHUB_REF::12", raw, f"prefix-match left in {wf_path.name}")
            self.assertNotIn('refs/heads/m"', raw, f"buggy prefix literal left in {wf_path.name}")

    def test_exact_equality_guards_in_both_workflows_or_script(self):
        """Égalité exacte refs/heads/main et refs/heads/master exigée (workflow ou script)."""
        wf_raw = DOCKER_WF.read_text() + "\n" + PROVISIONER_WF.read_text()
        script_raw = SCRIPT_PATH.read_text() if SCRIPT_PATH.is_file() else ""
        combined = wf_raw + "\n" + script_raw
        self.assertIn('[ "$GITHUB_REF" = "refs/heads/main" ]', combined)
        self.assertIn('[ "$GITHUB_REF" = "refs/heads/master" ]', combined)

    def test_provisioner_paths_include_models_registry(self):
        """LOW: paths du provisioner workflow doivent inclure models_registry.py."""
        wf = yaml.safe_load(PROVISIONER_WF.read_text())
        paths = _on_section(wf)["push"].get("paths") or []
        self.assertIn("models_registry.py", paths, f"models_registry.py missing from paths: {paths}")


if __name__ == "__main__":
    unittest.main()
