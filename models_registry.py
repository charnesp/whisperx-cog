"""Unified model registry (E5-CODE-1 T1).

Single declarative source of truth for every provisioned model:
key -> {hf_repo, lock_key, dirname, env_override, weight_files}.

Consumed by predict.py (Qwen snapshot paths / weight files / HF repos),
model_paths.py (faster-whisper HF ids + local candidates) and, from T2,
models_lock.py (expected /models/<key>/<sha40> paths).

VAD is deliberately NOT in the registry: it is bundled in the pinned
whisperx fork (whisperx/assets), has no stable revision of its own, and
stays a special case in model_paths.py.

Keys are the models.lock keys (provisioning layout
/models/<key>/<sha40>/). The qwen API alias "qwen3-asr" resolves to its
lock key via ALIASES + resolve_key().
"""

from __future__ import annotations

from types import MappingProxyType

BAKED_MODELS_ROOT = "/models"

# The API model alias used by clients for the Qwen path (predict.py
# QWEN_MODEL_NAME, bridge MODEL_MAP). All other registry keys are their
# own alias.
QWEN_API_ALIAS = "qwen3-asr"

ALIASES: dict[str, str] = {QWEN_API_ALIAS: "qwen3-asr-1.7b"}


class _ModelSpec:
    """Immutable per-model spec (frozen attribute surface, no dict mutation)."""

    __slots__ = (
        "key",
        "hf_repo",
        "lock_key",
        "dirname",
        "env_override",
        "weight_files",
    )

    def __init__(
        self,
        key: str,
        hf_repo: str,
        dirname: str,
        weight_files: tuple[str, ...],
        env_override: str | None = None,
    ):
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "hf_repo", hf_repo)
        object.__setattr__(self, "lock_key", key)
        object.__setattr__(self, "dirname", dirname)
        object.__setattr__(self, "env_override", env_override)
        object.__setattr__(self, "weight_files", weight_files)

    def __setattr__(self, name, value):  # immutability guard (tests assert this)
        raise AttributeError("ModelSpec is immutable")

    def __repr__(self) -> str:
        return f"ModelSpec(key={self.key!r}, hf_repo={self.hf_repo!r})"


MODELS: dict[str, _ModelSpec] = MappingProxyType(
    {
        spec.key: spec
        for spec in (
            _ModelSpec(
                key="tiny",
                hf_repo="Systran/faster-whisper-tiny",
                dirname="faster-whisper-tiny",
                weight_files=("model.bin",),
            ),
            _ModelSpec(
                key="large-v3",
                hf_repo="Systran/faster-whisper-large-v3",
                dirname="faster-whisper-large-v3",
                weight_files=("model.bin",),
            ),
            _ModelSpec(
                key="large-v3-turbo",
                hf_repo="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
                dirname="faster-whisper-large-v3-turbo",
                weight_files=("model.bin",),
            ),
            _ModelSpec(
                key="qwen3-asr-1.7b",
                hf_repo="Qwen/Qwen3-ASR-1.7B",
                dirname="qwen3-asr-1.7b",
                weight_files=(
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                ),
                env_override="QWEN_MODEL_PATH",
            ),
            _ModelSpec(
                key="qwen3-forced-aligner-0.6b",
                hf_repo="Qwen/Qwen3-ForcedAligner-0.6B",
                dirname="qwen3-forced-aligner-0.6b",
                weight_files=("model.safetensors",),
                env_override="QWEN_ALIGNER_PATH",
            ),
        )
    }
)

# Legacy/dev dirnames: /models/<dirname> (old flat bake) and
# ./models/<dirname> (bash build.sh). The provisioned revisioned layout
# /models/<key>/<sha40>/ is owned by models_lock.py (T2).


def resolve_key(key_or_alias: str) -> str:
    """Resolve an API alias (e.g. qwen3-asr) to its models.lock key."""
    if key_or_alias in ALIASES:
        return ALIASES[key_or_alias]
    if key_or_alias in MODELS:
        return key_or_alias
    raise KeyError(
        f"Unknown model key or alias: {key_or_alias!r}. "
        f"Known keys: {sorted(MODELS)}, aliases: {sorted(ALIASES)}."
    )


def local_candidates(key: str) -> list[str]:
    """Ordered local snapshot dirs: baked /models first, then ./models dev."""
    spec = MODELS[key]
    return [
        f"{BAKED_MODELS_ROOT}/{spec.dirname}",
        f"./models/{spec.dirname}",
    ]


def hf_repo(key: str) -> str:
    """HuggingFace repo id for a registry key (models.lock `repo`)."""
    return MODELS[key].hf_repo
