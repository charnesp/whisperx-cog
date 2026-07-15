#!/usr/bin/env python3
"""Embed bridge/bridge.py into the cog-bridge-script ConfigMap in k8s/whisperx-stack.yaml."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge_k8s import K8S, read_standalone_bridge, embed_bridge_in_k8s


def main() -> int:
    bridge = read_standalone_bridge()
    k8s = K8S.read_text()
    updated = embed_bridge_in_k8s(bridge, k8s)
    if updated is None:
        print("error: bridge.py ConfigMap block not found in k8s manifest", file=sys.stderr)
        return 1
    K8S.write_text(updated)
    print(f"Updated ConfigMap bridge.py in {K8S.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
