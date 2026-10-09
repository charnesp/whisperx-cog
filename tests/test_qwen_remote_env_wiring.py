"""GPU-free RED tests for compose/k8s env wiring (qwen3-asr-remote-vllm, group 8).

Config-shape test, written FIRST (same RED pattern as the other groups: it
asserts on manifests that do not carry the remote entries yet).

Spec: openspec/changes/qwen3-asr-remote-vllm specs/openai-stt-api/spec.md
- The four remote-backend variables MUST be wired into BOTH deployment
  manifests: `QWEN_BACKEND`, `QWEN_REMOTE_BASE_URL`, `QWEN_REMOTE_MODEL`,
  `QWEN_REMOTE_TIMEOUT_S`.
- The address and the model name MUST stay empty/neutral in the files: the
  concrete host, port and model are injected at deploy time, never committed.
  Compose uses `${VAR}` indirection; k8s carries an empty value.
- They belong to the whisperx (Cog) container, which reads them at runtime.
- The bridge dual-copy sync check stays green (group 1 invariant).
"""
from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE = REPO_ROOT / "docker-compose.yml"
K8S = REPO_ROOT / "k8s" / "whisperx-stack.yaml"

QWEN_VARS = (
    "QWEN_BACKEND",
    "QWEN_REMOTE_BASE_URL",
    "QWEN_REMOTE_MODEL",
    "QWEN_REMOTE_TIMEOUT_S",
)
# Variables whose concrete value is injected at deploy time: the repo must
# never carry a host, port or model name for them.
INJECTED_VARS = ("QWEN_REMOTE_BASE_URL", "QWEN_REMOTE_MODEL")

# A committed value that names an address would carry a URL scheme.
_CONCRETE_URL = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://")


def _compose_env() -> dict:
    data = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    return data["services"]["whisperx"]["environment"]


def _k8s_whisperx_env() -> list:
    for doc in yaml.safe_load_all(K8S.read_text(encoding="utf-8")):
        if not isinstance(doc, dict) or doc.get("kind") != "Deployment":
            continue
        for container in doc["spec"]["template"]["spec"]["containers"]:
            if container.get("name") == "whisperx":
                return container.get("env") or []
    return []


class TestComposeWiring(unittest.TestCase):
    def test_compose_carries_the_four_vars(self):
        env = _compose_env()
        for name in QWEN_VARS:
            self.assertIn(name, env, f"docker-compose.yml whisperx env missing {name}")

    def test_compose_injects_address_and_model(self):
        env = _compose_env()
        for name in INJECTED_VARS:
            value = str(env.get(name, ""))
            # Indirection only: the value references the variable itself.
            self.assertIn("${" + name, value)
            # No committed address: no URL scheme in the value.
            self.assertIsNone(
                _CONCRETE_URL.search(value),
                f"docker-compose.yml {name} must not carry a concrete address",
            )

    def test_compose_backend_and_timeout_have_safe_defaults(self):
        env = _compose_env()
        self.assertIn("local", str(env.get("QWEN_BACKEND", "")))
        self.assertIn("300", str(env.get("QWEN_REMOTE_TIMEOUT_S", "")))


class TestK8sWiring(unittest.TestCase):
    def test_k8s_carries_the_four_vars(self):
        names = [entry.get("name") for entry in _k8s_whisperx_env()]
        for name in QWEN_VARS:
            self.assertIn(name, names, f"k8s whisperx env missing {name}")

    def test_k8s_address_and_model_are_empty_placeholders(self):
        env = {entry.get("name"): entry.get("value") for entry in _k8s_whisperx_env()}
        for name in INJECTED_VARS:
            value = env.get(name) or ""
            self.assertIsNone(
                _CONCRETE_URL.search(value),
                f"k8s {name} must stay an empty placeholder (deploy-time injection)",
            )

    def test_k8s_backend_and_timeout_have_safe_defaults(self):
        env = {entry.get("name"): entry.get("value") for entry in _k8s_whisperx_env()}
        self.assertEqual(env.get("QWEN_BACKEND"), "local")
        self.assertEqual(env.get("QWEN_REMOTE_TIMEOUT_S"), "300")


class TestBridgeSyncStillGreen(unittest.TestCase):
    def test_bridge_sync_check_passes(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "check-bridge-sync.py")],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
