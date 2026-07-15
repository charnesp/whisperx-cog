#!/usr/bin/env python3
"""Shared helpers for bridge scripts ↔ k8s ConfigMap sync."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRIDGE_DIR = ROOT / "bridge"
K8S = ROOT / "k8s" / "whisperx-stack.yaml"

CONFIGMAP_FILES = {
    "bridge.py": BRIDGE_DIR / "bridge.py",
    "openai_compat.py": BRIDGE_DIR / "openai_compat.py",
}


def _block_start(name: str) -> str:
    return f"  {name}: |"


def read_standalone_bridge() -> str:
    text = CONFIGMAP_FILES["bridge.py"].read_text()
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def read_standalone_file(name: str) -> str:
    text = CONFIGMAP_FILES[name].read_text()
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def _extract_configmap_block(k8s_text: str, block_name: str) -> str | None:
    block_start = _block_start(block_name)
    lines = k8s_text.splitlines(keepends=True)
    try:
        start = next(i for i, line in enumerate(lines) if line.rstrip("\n") == block_start)
    except StopIteration:
        return None

    block_lines: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.rstrip("\n")
        if stripped == "---":
            break
        if stripped.startswith("    "):
            block_lines.append(stripped[4:] + "\n")
        elif stripped == "":
            block_lines.append("\n")
        else:
            break

    if not block_lines:
        return None
    text = "".join(block_lines)
    return text.rstrip("\n") + "\n"


def extract_configmap_bridge(k8s_text: str) -> str | None:
    return _extract_configmap_block(k8s_text, "bridge.py")


def extract_configmap_file(k8s_text: str, name: str) -> str | None:
    return _extract_configmap_block(k8s_text, name)


def _embed_configmap_block(bridge_text: str, k8s_text: str, block_name: str) -> str | None:
    block_start = _block_start(block_name)
    lines = k8s_text.splitlines(keepends=True)
    try:
        start = next(i for i, line in enumerate(lines) if line.rstrip("\n") == block_start)
    except StopIteration:
        return _insert_configmap_block(bridge_text, k8s_text, block_name)

    end = start + 1
    while end < len(lines):
        stripped = lines[end].rstrip("\n")
        if stripped == "---":
            break
        if stripped.startswith("    ") or stripped == "":
            end += 1
            continue
        break

    if bridge_text and not bridge_text.endswith("\n"):
        bridge_text += "\n"

    embedded: list[str] = [block_start + "\n"]
    for line in bridge_text.splitlines(keepends=True):
        if line in ("", "\n"):
            embedded.append("\n")
        else:
            embedded.append(f"    {line}")

    return "".join(lines[:start] + embedded + lines[end:])


def _insert_configmap_block(file_text: str, k8s_text: str, block_name: str) -> str | None:
    """Insert a new ConfigMap file block before the ConfigMap document separator."""
    anchor = _block_start("bridge.py")
    lines = k8s_text.splitlines(keepends=True)
    try:
        start = next(i for i, line in enumerate(lines) if line.rstrip("\n") == anchor)
    except StopIteration:
        return None

    end = start + 1
    while end < len(lines):
        stripped = lines[end].rstrip("\n")
        if stripped == "---":
            break
        if stripped.startswith("    ") or stripped == "":
            end += 1
            continue
        break

    if file_text and not file_text.endswith("\n"):
        file_text += "\n"

    embedded: list[str] = [_block_start(block_name) + "\n"]
    for line in file_text.splitlines(keepends=True):
        if line in ("", "\n"):
            embedded.append("\n")
        else:
            embedded.append(f"    {line}")

    return "".join(lines[:end] + embedded + lines[end:])


def embed_bridge_in_k8s(bridge_text: str, k8s_text: str) -> str | None:
    return _embed_configmap_block(bridge_text, k8s_text, "bridge.py")


def embed_file_in_k8s(file_text: str, k8s_text: str, block_name: str) -> str | None:
    return _embed_configmap_block(file_text, k8s_text, block_name)


def embed_all_bridge_files_in_k8s(k8s_text: str) -> str | None:
    updated = k8s_text
    for name in CONFIGMAP_FILES:
        content = read_standalone_file(name)
        result = embed_file_in_k8s(content, updated, name)
        if result is None:
            return None
        updated = result
    return updated
