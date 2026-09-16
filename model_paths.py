"""Whisper / Qwen model path resolution — fail-HARD (E5-CODE-1 T3).

Resolution order per model key:
1. explicit env override (registry env_override, e.g. QWEN_MODEL_PATH)
2. provisioned lock layout  $MODELS_DIR/<key>/<sha40>/  (+ .complete,
   non-empty weights)
3. ./models/<dirname> ONLY when MODELS_MODE=dev (bash build.sh workflow)
4. raise ModelNotProvisioned (E_MODEL_NOT_PROVISIONED + provisioner
   remediation) — NEVER the HuggingFace repo id.

MODELS_MODE=online is an explicit opt-in (laptop dev): resolve may then
return the HF repo id. It is NEVER the default.

VAD stays a special case (bundled whisperx asset, outside registry/lock).
Model constants come from the unified registry (T1); the revisioned
layout + .complete semantics come from models_lock (T2).
"""

from __future__ import annotations

import os

from models_lock import COMPLETE_MARKER, parse_lock
from models_registry import (
    MODELS,
    hf_repo,
    local_candidates,
    resolve_key,
)

BAKED_MODELS_ROOT = "/models"

_WHISPER_KEYS = ("tiny", "large-v3", "large-v3-turbo")

WHISPER_MODEL_HF_IDS: dict[str, str] = {key: hf_repo(key) for key in _WHISPER_KEYS}

# Ordered candidates: baked absolute path first, then repo-relative
# (legacy flat layouts, still exported for golden_set/build.sh tooling).
WHISPER_MODEL_LOCAL_PATHS: dict[str, list[str]] = {
    key: local_candidates(key) for key in _WHISPER_KEYS
}

VAD_FILENAME = "whisperx-vad-segmentation.bin"
VAD_LOCAL_CANDIDATES = [
    f"{BAKED_MODELS_ROOT}/vad/{VAD_FILENAME}",
    f"./models/vad/{VAD_FILENAME}",
]

PROVISIONER_HOST_DIR = "/files/data/whisperx-cog/models"
PROVISIONER_IMAGE = "ghcr.io/charnesp/whisperx-provisioner:latest"

# Env overrides for the faster-whisper keys (qwen keys carry their own
# env_override in the registry).
ENV_OVERRIDES: dict[str, str] = {
    "tiny": "TINY_PATH",
    "large-v3": "LARGE_V3_PATH",
    "large-v3-turbo": "LARGE_V3_TURBO_PATH",
}


class ModelNotProvisioned(RuntimeError):
    """Fail-hard resolution error: no local provisioned copy, no fallback."""


def _remediation(key: str) -> str:
    return (
        f"docker run --rm -v {PROVISIONER_HOST_DIR}:/models "
        f"{PROVISIONER_IMAGE} provision --model {key}"
    )


def _models_mode() -> str:
    return os.environ.get("MODELS_MODE", "").strip().lower()


def _models_dir() -> str:
    return os.environ.get("MODELS_DIR") or BAKED_MODELS_ROOT


def _dir_has_weights(path: str, weight_files) -> bool:
    if not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, COMPLETE_MARKER)):
        return False
    for name in weight_files:
        fpath = os.path.join(path, name)
        if not os.path.isfile(fpath) or os.path.getsize(fpath) == 0:
            return False
    return True


def _lock() -> dict[str, dict]:
    """Parse models.lock (cached per mtime); unreadable/invalid lock = {}."""
    global _LOCK_CACHE, _LOCK_CACHE_MTIME
    import models_lock

    path = models_lock._lock_path()
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        return {}
    if _LOCK_CACHE is not None and _LOCK_CACHE_MTIME == mtime:
        return _LOCK_CACHE
    try:
        _LOCK_CACHE = parse_lock(path)
    except (ValueError, OSError):
        _LOCK_CACHE = {}
    _LOCK_CACHE_MTIME = mtime
    return _LOCK_CACHE


_LOCK_CACHE: dict[str, dict] | None = None
_LOCK_CACHE_MTIME: float | None = None


def resolve_model_dir(key: str) -> str:
    """Resolve a provisioned snapshot dir for a registry key (or alias).

    Order: env override → lock $MODELS_DIR/<key>/<sha40>/ (+ .complete,
    non-empty weights) → ./models/<dirname> in MODELS_MODE=dev → raise
    ModelNotProvisioned. NEVER returns the HF repo id (MODELS_MODE=online
    callers use resolve_hf_repo() explicitly).
    """
    global _LOCK_CACHE
    key = resolve_key(key)
    spec = MODELS[key]
    tried: list[str] = []

    if spec.env_override:
        override = os.environ.get(spec.env_override, "").strip()
        if override:
            return override

    lock_entry = _lock().get(key)
    if lock_entry:
        lock_path = os.path.join(_models_dir(), key, lock_entry["revision"])
        tried.append(lock_path)
        # Check ALL lock-declared files (index/shard/tokenizer) — the same
        # set the T2 boot validator enforces — not only the weight files.
        if _dir_has_weights(lock_path, lock_entry["expected_files"]):
            return lock_path

    dev_path = os.path.join("./models", spec.dirname)
    tried.append(dev_path)
    if _models_mode() == "dev" and os.path.isdir(dev_path):
        return dev_path

    if lock_entry:
        head = (
            f"E_MODEL_NOT_PROVISIONED model={key} rev={lock_entry['revision']}"
        )
    else:
        head = f"E_MODEL_NOT_PROVISIONED model={key}"
    raise ModelNotProvisioned(
        f"{head} — tried: {', '.join(tried)}. "
        f"Remediation: {_remediation(key)}. "
        "MODELS_MODE=dev resolves ./models/<dirname>; MODELS_MODE=online is "
        "the explicit HF-download opt-in."
    )


def resolve_hf_repo(key: str) -> str:
    """Explicit HF repo id (only for MODELS_MODE=online opt-in callers)."""
    return hf_repo(resolve_key(key))


def resolve_whisper_model_path(whisper_model: str) -> str:
    """Fail-hard resolution for the faster-whisper backends.

    No HF fallback: an unprovisioned model raises ModelNotProvisioned with
    the paths tried + remediation. MODELS_MODE=online returns the HF repo
    id (explicit opt-in only).
    """
    key = resolve_key(whisper_model)
    override = os.environ.get(ENV_OVERRIDES[key], "").strip()
    if override:
        return override

    if _models_mode() == "online":
        return WHISPER_MODEL_HF_IDS[key]

    try:
        return resolve_model_dir(key)
    except ModelNotProvisioned as exc:
        legacy = ", ".join(local_candidates(key))
        raise ModelNotProvisioned(
            f"{exc} (legacy flat dirs also unavailable: {legacy})"
        ) from None


def resolve_vad_source_path() -> str | None:
    """Return path to baked/local VAD weights, or None if missing."""
    for path in VAD_LOCAL_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None
