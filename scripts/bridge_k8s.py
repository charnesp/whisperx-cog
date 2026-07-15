#!/usr/bin/env python3
"""Shared helpers for bridge deployment checks (k8s GHCR image, source files)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRIDGE_DIR = ROOT / "bridge"
K8S = ROOT / "k8s" / "whisperx-stack.yaml"

BRIDGE_IMAGE = "ghcr.io/charnesp/whisperx-cog-bridge:latest"

BRIDGE_SOURCE_FILES = (
    "bridge.py",
    "openai_compat.py",
)


def read_k8s_text() -> str:
    return K8S.read_text()


def bridge_container_image(k8s_text: str) -> str | None:
    """Return the bridge container image from the Deployment, if declared."""
    match = re.search(
        r"- name: bridge\s*\n(?:[^\n]*\n)*?\s*image:\s*(\S+)",
        k8s_text,
    )
    return match.group(1) if match else None


def k8s_uses_bridge_configmap(k8s_text: str) -> bool:
    return "name: cog-bridge-script" in k8s_text or "configMap:" in k8s_text


def bridge_source_files_present() -> list[str]:
    missing: list[str] = []
    for name in BRIDGE_SOURCE_FILES:
        if not (BRIDGE_DIR / name).is_file():
            missing.append(name)
    return missing
