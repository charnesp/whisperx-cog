#!/usr/bin/env python3
"""Verify k8s manifest uses the published bridge GHCR image (no ConfigMap drift)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge_k8s import (
    BRIDGE_IMAGE,
    BRIDGE_SOURCE_FILES,
    K8S,
    bridge_container_image,
    bridge_source_files_present,
    k8s_uses_bridge_configmap,
)


def main() -> int:
    if not K8S.is_file():
        print(f"error: missing {K8S}", file=sys.stderr)
        return 1

    missing = bridge_source_files_present()
    if missing:
        for name in missing:
            print(f"error: missing bridge/{name}", file=sys.stderr)
        return 1

    k8s_text = K8S.read_text()
    if k8s_uses_bridge_configmap(k8s_text):
        print(
            "error: k8s manifest still references a bridge ConfigMap; "
            f"use image {BRIDGE_IMAGE} instead",
            file=sys.stderr,
        )
        return 1

    image = bridge_container_image(k8s_text)
    if image != BRIDGE_IMAGE:
        print(
            f"error: bridge container image is {image!r}, expected {BRIDGE_IMAGE!r}",
            file=sys.stderr,
        )
        return 1

    print(f"OK: k8s bridge uses {BRIDGE_IMAGE}")
    print(f"OK: bridge source files present ({', '.join(BRIDGE_SOURCE_FILES)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
