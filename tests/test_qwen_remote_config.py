"""GPU-free RED tests for resolve_remote_config (qwen3-asr-remote-vllm, group 2).

Written FIRST — must fail with ModuleNotFoundError because qwen_remote.py
does not exist yet (same RED pattern as tests/test_models_registry.py).

Spec: openspec/changes/qwen3-asr-remote-vllm specs/openai-stt-api/spec.md
- `QWEN_BACKEND` read at CALL time: unset -> `local`; `remote` -> remote;
  invalid value -> `local` + captured warning (spec scenario "Invalid
  selector resolves local with warning").
- Remote mode fails fast with typed `QwenRemoteError` naming the missing
  variable: `QWEN_REMOTE_BASE_URL` or `QWEN_REMOTE_MODEL` mandatory (no
  default address, no service-name guess — design decisions 1/6/9).
- `QWEN_REMOTE_TIMEOUT_S` default 300; non-integer -> typed config error.
- URL scheme limited to http/https; trailing slash normalized; the
  chat-completions endpoint is base_url + `/chat/completions`.
- NO code constant holds any host/port/model name (hygiene decision 5).
Env resolution reads `os.environ` per call (patched via mock.patch.dict).
"""
from __future__ import annotations
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from qwen_remote import QwenRemoteError, resolve_remote_config

BASE_URL = "http://vllm.internal" + ".invalid:9000/v1"


def _patch_env(**overrides):
    """Clean QWEN_* env + overrides, cleared (deterministic per call)."""
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("QWEN_"):
            del env[key]
    env.update(overrides)
    return mock.patch.dict(os.environ, env, clear=True)


class TestBackendSelector(unittest.TestCase):
    def test_unset_defaults_local(self):
        with _patch_env():
            cfg = resolve_remote_config()
        self.assertEqual(cfg["backend"], "local")
        self.assertIsNone(cfg["base_url"])
        self.assertIsNone(cfg["model"])

    def test_remote_accepted(self):
        with _patch_env(
            QWEN_BACKEND="remote",
            QWEN_REMOTE_BASE_URL=BASE_URL,
            QWEN_REMOTE_MODEL="qwen3-asr-demo-model",
        ):
            cfg = resolve_remote_config()
        self.assertEqual(cfg["backend"], "remote")
        self.assertEqual(cfg["chat_url"], BASE_URL + "/chat/completions")
        self.assertEqual(cfg["model"], "qwen3-asr-demo-model")

    def test_invalid_value_resolves_local_with_warning(self):
        with _patch_env(QWEN_BACKEND="rem0te"):
            with self.assertLogs("qwen_remote", level="WARNING") as captured:
                cfg = resolve_remote_config()
        self.assertEqual(cfg["backend"], "local")
        joined = "\n".join(captured.output)
        self.assertIn("rem0te", joined)


class TestRemoteFailFast(unittest.TestCase):
    def test_missing_base_url_named(self):
        with _patch_env(QWEN_BACKEND="remote"):
            with self.assertRaises(QwenRemoteError) as ctx:
                resolve_remote_config()
        self.assertIn("QWEN_REMOTE_BASE_URL", str(ctx.exception))

    def test_missing_model_named(self):
        with _patch_env(
            QWEN_BACKEND="remote",
            QWEN_REMOTE_BASE_URL=BASE_URL,
        ):
            with self.assertRaises(QwenRemoteError) as ctx:
                resolve_remote_config()
        self.assertIn("QWEN_REMOTE_MODEL", str(ctx.exception))

    def test_non_integer_timeout_rejected(self):
        with _patch_env(
            QWEN_BACKEND="remote",
            QWEN_REMOTE_BASE_URL=BASE_URL,
            QWEN_REMOTE_MODEL="qwen3-asr-demo-model",
            QWEN_REMOTE_TIMEOUT_S="three-hundred",
        ):
            with self.assertRaises(QwenRemoteError) as ctx:
                resolve_remote_config()
        self.assertIn("QWEN_REMOTE_TIMEOUT_S", str(ctx.exception))

    def test_bad_scheme_rejected(self):
        with _patch_env(
            QWEN_BACKEND="remote",
            QWEN_REMOTE_BASE_URL="ftp://vllm.internal.invalid:9000/v1",
            QWEN_REMOTE_MODEL="qwen3-asr-demo-model",
        ):
            with self.assertRaises(QwenRemoteError) as ctx:
                resolve_remote_config()
        self.assertIn("http", str(ctx.exception))


class TestUrlNormalization(unittest.TestCase):
    def test_trailing_slash_normalized(self):
        with _patch_env(
            QWEN_BACKEND="remote",
            QWEN_REMOTE_BASE_URL=BASE_URL + "/",
            QWEN_REMOTE_MODEL="qwen3-asr-demo-model",
        ):
            cfg = resolve_remote_config()
        self.assertNotIn("//chat", cfg["chat_url"])
        self.assertEqual(cfg["chat_url"], BASE_URL + "/chat/completions")

    def test_default_timeout(self):
        with _patch_env(
            QWEN_BACKEND="remote",
            QWEN_REMOTE_BASE_URL=BASE_URL,
            QWEN_REMOTE_MODEL="qwen3-asr-demo-model",
        ):
            cfg = resolve_remote_config()
        self.assertEqual(cfg["timeout_s"], 300)

    def test_integer_timeout_passthrough(self):
        with _patch_env(
            QWEN_BACKEND="remote",
            QWEN_REMOTE_BASE_URL=BASE_URL,
            QWEN_REMOTE_MODEL="qwen3-asr-demo-model",
            QWEN_REMOTE_TIMEOUT_S="45",
        ):
            cfg = resolve_remote_config()
        self.assertEqual(cfg["timeout_s"], 45)


class TestModuleHygiene(unittest.TestCase):
    def test_no_host_or_model_constant_in_module(self):
        src = (REPO_ROOT / "qwen_remote.py").read_text(encoding="utf-8")
        self.assertNotIn("192.168", src)
        self.assertNotIn("localhost:", src)


if __name__ == "__main__":
    unittest.main()
