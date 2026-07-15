#!/usr/bin/env python3
"""Shared helpers for bridge.py ↔ k8s ConfigMap sync."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRIDGE = ROOT / "bridge" / "bridge.py"
K8S = ROOT / "k8s" / "whisperx-stack.yaml"

BLOCK_START = "  bridge.py: |"


def read_standalone_bridge() -> str:
    text = BRIDGE.read_text()
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def extract_configmap_bridge(k8s_text: str) -> str | None:
    lines = k8s_text.splitlines(keepends=True)
    try:
        start = next(i for i, line in enumerate(lines) if line.rstrip("\n") == BLOCK_START)
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


def embed_bridge_in_k8s(bridge_text: str, k8s_text: str) -> str | None:
    lines = k8s_text.splitlines(keepends=True)
    try:
        start = next(i for i, line in enumerate(lines) if line.rstrip("\n") == BLOCK_START)
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

    if bridge_text and not bridge_text.endswith("\n"):
        bridge_text += "\n"

    embedded: list[str] = [BLOCK_START + "\n"]
    for line in bridge_text.splitlines(keepends=True):
        if line in ("", "\n"):
            embedded.append("\n")
        else:
            embedded.append(f"    {line}")

    return "".join(lines[:start] + embedded + lines[end:])
