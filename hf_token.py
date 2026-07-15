"""HuggingFace token resolution for diarization (no torch/cog deps)."""

from __future__ import annotations

import os


def resolve_huggingface_token(huggingface_access_token: str | None) -> str | None:
    """Resolve HF token from Cog input or whisperx process env."""
    if huggingface_access_token and huggingface_access_token.strip():
        return huggingface_access_token.strip()
    env_token = os.environ.get("HUGGINGFACE_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()
    return None


def require_diarization_token(
    diarization: bool, huggingface_access_token: str | None
) -> str | None:
    """Return resolved HF token when diarization is on; raise if missing."""
    if not diarization:
        return None
    token = resolve_huggingface_token(huggingface_access_token)
    if not token:
        raise RuntimeError(
            "Diarization requested but no HuggingFace token available. "
            "Provide huggingface_access_token or set HUGGINGFACE_TOKEN on the whisperx container."
        )
    return token
