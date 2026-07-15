#!/usr/bin/env python3
"""Verify bridge/bridge.py matches the embedded ConfigMap in k8s/whisperx-stack.yaml."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge_k8s import read_standalone_bridge, extract_configmap_bridge, K8S


def main() -> int:
    standalone = read_standalone_bridge()
    embedded = extract_configmap_bridge(K8S.read_text())
    if embedded is None:
        print("error: could not extract bridge.py from k8s manifest", file=sys.stderr)
        return 1

    if standalone == embedded:
        print("OK: bridge/bridge.py matches k8s ConfigMap")
        return 0

    print("MISMATCH: bridge/bridge.py differs from k8s ConfigMap", file=sys.stderr)
    print("Run: python3 scripts/sync-bridge-to-k8s.py", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
