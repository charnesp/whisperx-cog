"""Whisper / Qwen model path resolution — legacy HF fallback (E5-LEGACY-HF).

Final Charles decision: NO anti-download guards. Resolution order per
model key:
1. explicit env override (registry env_override, e.g. QWEN_MODEL_PATH)
2. provisioned lock layout  $MODELS_DIR/<key>/<sha40>/  (+ .complete,
   non-empty weights)
3. else the HF repo id — runtime download like any HF model (pre-E5
   legacy behavior). no raise, no env switch.

The 5 main models are provisioned on /models by the host bind mount
(scripts/provision.py, models.lock v2); the resolver just prefers that
copy when present. VAD stays a special case (bundled whisperx asset).
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

# Keys derived from the unified registry (single source of truth): the
# faster-whisper keys are the non-qwen ones; the qwen keys carry their
# own env_override in the registry.
_WHISPER_KEYS = tuple(sorted(k for k, spec in MODELS.items() if not spec.env_override))

WHISPER_MODEL_HF_IDS: dict[str, str] = {key: hf_repo(key) for key in _WHISPER_KEYS}

# Legacy flat layout candidates (build.sh dev bake tooling export).
WHISPER_MODEL_LOCAL_PATHS: dict[str, list[str]] = {
    key: local_candidates(key) for key in _WHISPER_KEYS
}

VAD_FILENAME = "whisperx-vad-segmentation.bin"
VAD_LOCAL_CANDIDATES = [
    f"{BAKED_MODELS_ROOT}/vad/{VAD_FILENAME}",
    f"./models/vad/{VAD_FILENAME}",
]

# Env overrides for the faster-whisper keys (qwen keys carry their own
# env_override in the registry).
ENV_OVERRIDES: dict[str, str] = {
    "tiny": "TINY_PATH",
    "large-v3": "LARGE_V3_PATH",
    "large-v3-turbo": "LARGE_V3_TURBO_PATH",
}


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


def _models_dir() -> str:
    return os.environ.get("MODELS_DIR") or BAKED_MODELS_ROOT


def resolve_model_dir(key: str) -> str:
    """Resolve a snapshot dir for a registry key (or alias) — legacy HF
    fallback (E5-LEGACY-HF).

    Order: env override → provisioned lock dir $MODELS_DIR/<key>/<sha40>/
    (+ .complete, non-empty weights) → the HF repo id (runtime download,
    like any HF model, pre-E5 legacy behavior). No raise.
    """
    key = resolve_key(key)
    spec = MODELS[key]

    if spec.env_override:
        override = os.environ.get(spec.env_override, "").strip()
        if override:
            return override

    lock_entry = _lock().get(key)
    if lock_entry:
        lock_path = os.path.join(_models_dir(), key, lock_entry["revision"])
        if _dir_has_weights(lock_path, lock_entry["expected_files"]):
            return lock_path

    return hf_repo(key)


def resolve_hf_repo(key: str) -> str:
    """Explicit HF repo id for a registry key."""
    return hf_repo(resolve_key(key))


def resolve_whisper_model_path(whisper_model: str) -> str:
    """Resolve the faster-whisper backend source — legacy HF fallback
    (E5-LEGACY-HF).

    Env override → provisioned lock dir → the HF repo id (runtime
    download possible). No raise.
    """
    key = resolve_key(whisper_model)
    override = os.environ.get(ENV_OVERRIDES[key], "").strip()
    if override:
        return override
    return resolve_model_dir(key)


def resolve_vad_source_path() -> str | None:
    """Return path to baked/local VAD weights, or None if missing."""
    for path in VAD_LOCAL_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None
