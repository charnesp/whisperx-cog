"""JSON-safe serialization helpers for Cog prediction output (no torch/cog imports)."""

from __future__ import annotations

import math
from typing import Any


def sanitize_error_message(msg: str, max_len: int = 500) -> str:
    """Avoid logging binary or huge payloads in exception messages."""
    if not msg:
        return msg
    if len(msg) > max_len:
        return msg[:max_len] + "... (truncated)"
    non_printable = sum(1 for c in msg if ord(c) < 32 and c not in "\n\r\t")
    if non_printable > 50 or "\\x" in repr(msg):
        return "(binary or non-printable data omitted)"
    return msg


def sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/inf and convert numpy/torch types for JSON-serializable output."""
    if obj is None:
        return None
    if isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if hasattr(obj, "item"):
        try:
            return sanitize_for_json(obj.item())
        except (ValueError, RuntimeError):
            return None
    if hasattr(obj, "tolist"):
        try:
            return sanitize_for_json(obj.tolist())
        except (ValueError, RuntimeError):
            return None
    return obj
