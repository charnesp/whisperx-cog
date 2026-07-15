#!/usr/bin/env python3
"""Embed bridge scripts into the cog-bridge-script ConfigMap in k8s/whisperx-stack.yaml."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge_k8s import K8S, CONFIGMAP_FILES, embed_all_bridge_files_in_k8s


def main() -> int:
    k8s = K8S.read_text()
    updated = embed_all_bridge_files_in_k8s(k8s)
    if updated is None:
        print("error: bridge ConfigMap blocks not found in k8s manifest", file=sys.stderr)
        return 1
    K8S.write_text(updated)
    for name in CONFIGMAP_FILES:
        print(f"Updated ConfigMap {name} in {K8S.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
