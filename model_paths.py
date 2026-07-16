"""Whisper / VAD model path resolution (no torch/cog deps).

Baked weights live under /models (outside Cog's COPY . /src) so Docker
build-time downloads survive the final source copy. Local ./models/ is
kept as a fallback for `bash build.sh` during development.
"""

from __future__ import annotations

import os

BAKED_MODELS_ROOT = "/models"

WHISPER_MODEL_HF_IDS = {
    "tiny": "Systran/faster-whisper-tiny",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

# Ordered candidates: baked absolute path first, then repo-relative.
WHISPER_MODEL_LOCAL_PATHS: dict[str, list[str]] = {
    name: [
        f"{BAKED_MODELS_ROOT}/faster-whisper-{name}",
        f"./models/faster-whisper-{name}",
    ]
    for name in WHISPER_MODEL_HF_IDS
}

VAD_FILENAME = "whisperx-vad-segmentation.bin"
VAD_LOCAL_CANDIDATES = [
    f"{BAKED_MODELS_ROOT}/vad/{VAD_FILENAME}",
    f"./models/vad/{VAD_FILENAME}",
]


def _has_model_bin(directory: str) -> bool:
    return os.path.isdir(directory) and os.path.isfile(
        os.path.join(directory, "model.bin")
    )


def resolve_whisper_model_path(whisper_model: str) -> str:
    """Use baked/local path if present, otherwise HuggingFace repo ID."""
    for local_path in WHISPER_MODEL_LOCAL_PATHS[whisper_model]:
        if _has_model_bin(local_path):
            return local_path
    return WHISPER_MODEL_HF_IDS[whisper_model]


def resolve_vad_source_path() -> str | None:
    """Return path to baked/local VAD weights, or None if missing."""
    for path in VAD_LOCAL_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None
