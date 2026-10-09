"""qwen_remote — env resolution + remote config (qwen3-asr-remote-vllm, group 2).

Minimal GREEN module: `resolve_remote_config()` reads the environment at
CALL time (resolved once in predict.setup(), fail-fast per design
decision 9). No host/port/model constant lives here (hygiene decision 5):
every address comes from env.

Categories (design decision 10): config/connection/timeout/http_status/
parse. `QwenRemoteError` is the typed root; only config is used here.
Messages never carry the full URL (userinfo possible), so the offending
variable NAME is reported, not the value.
"""
from __future__ import annotations
import logging
import os
from urllib.parse import urlsplit

logger = logging.getLogger("qwen_remote")

BACKEND_LOCAL = "local"
BACKEND_REMOTE = "remote"

CHAT_COMPLETIONS_SUFFIX = "chat/completions"

DEFAULT_REMOTE_TIMEOUT_S = 300


class QwenRemoteError(Exception):
    """Typed remote-engine failure (categories per design decision 10)."""


def _warn_invalid_selector(value):
    logger.warning(
        "Invalid QWEN_BACKEND value %r: defaulting to the local backend "
        "(valid values: local, remote)",
        value,
    )


def resolve_batch_size(batch_size, default: int, cap: int) -> int:
    """Single source of truth for a requested concurrency/batch value.

    Semantics (mirrors predict.resolve_qwen_batch_size exactly): `None` or a
    non-integer value -> `default`; `<= 0` -> `default`; otherwise clamped to
    `[1, cap]`. `int()` is applied, so `2.5 -> 2` and `True -> 1`, exactly as
    the local engine path. Pure (no logging, no GPU import): shared by
    `predict.resolve_qwen_batch_size` and `qwen_remote_client.clamp_remote_pool_size`
    so the local clamp and the remote pool bound cannot drift.
    """
    if batch_size is None:
        return default
    try:
        value = int(batch_size)
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return default
    return max(1, min(cap, value))


def resolve_remote_config():
    """Resolve the backend + remote settings from env, at call time.

    Returns a dict for the LOCAL backend:
      {backend: 'local', base_url: None, model: None, timeout_s: None,
       chat_url: None}
    for the REMOTE backend:
      {backend: 'remote', base_url: <normalized>, model: <env>,
       timeout_s: <int>, chat_url: base_url + '/chat/completions'}
    Raises QwenRemoteError (config category) on remote-mode
    misconfiguration: missing base URL / model, non-integer timeout,
    non-http(s) scheme.
    """
    backend = (os.environ.get("QWEN_BACKEND") or BACKEND_LOCAL).strip().lower()
    if backend == BACKEND_REMOTE:
        base_url = _resolve_remote_base_url()
        model = _resolve_remote_model()
        timeout_s = _resolve_remote_timeout()
        return {
            "backend": BACKEND_REMOTE,
            "base_url": base_url,
            "model": model,
            "timeout_s": timeout_s,
            "chat_url": base_url.rstrip("/") + "/" + CHAT_COMPLETIONS_SUFFIX,
        }
    if backend != BACKEND_LOCAL:
        _warn_invalid_selector(os.environ.get("QWEN_BACKEND"))
    return {
        "backend": BACKEND_LOCAL,
        "base_url": None,
        "model": None,
        "timeout_s": None,
        "chat_url": None,
    }


def _resolve_remote_base_url():
    raw = (os.environ.get("QWEN_REMOTE_BASE_URL") or "").strip()
    if not raw:
        raise QwenRemoteError(
            "QwenRemoteError: config — QWEN_REMOTE_BASE_URL is required when "
            "QWEN_BACKEND=remote (set the scheme http:// or https://, host "
            "and base path, no default is provided)"
        )
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https"):
        raise QwenRemoteError(
            "QwenRemoteError: config — QWEN_REMOTE_BASE_URL must use the "
            "http or https scheme"
        )
    if not parts.netloc:
        raise QwenRemoteError(
            "QwenRemoteError: config — QWEN_REMOTE_BASE_URL is missing a "
            "host (http://<host>[:<port>]<base-path>)"
        )
    return raw.rstrip("/")


def _resolve_remote_model():
    raw = (os.environ.get("QWEN_REMOTE_MODEL") or "").strip()
    if not raw:
        raise QwenRemoteError(
            "QwenRemoteError: config — QWEN_REMOTE_MODEL is required when "
            "QWEN_BACKEND=remote (the model name is sent verbatim in every "
            "request, no default is provided)"
        )
    return raw


def _resolve_remote_timeout():
    raw = (os.environ.get("QWEN_REMOTE_TIMEOUT_S") or "").strip()
    if not raw:
        return DEFAULT_REMOTE_TIMEOUT_S
    try:
        value = int(raw)
    except ValueError as exc:
        raise QwenRemoteError(
            "QwenRemoteError: config — QWEN_REMOTE_TIMEOUT_S must be an "
            "integer number of seconds per request (got %r)" % (raw,)
        ) from exc
    if value <= 0:
        raise QwenRemoteError(
            "QwenRemoteError: config — QWEN_REMOTE_TIMEOUT_S must be a "
            "positive integer (got %r)" % (raw,)
        )
    return value
