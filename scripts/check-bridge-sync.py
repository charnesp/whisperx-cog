#!/usr/bin/env python3
"""Verify bridge scripts match the embedded ConfigMap in k8s/whisperx-stack.yaml."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge_k8s import CONFIGMAP_FILES, read_standalone_file, extract_configmap_file, K8S


def main() -> int:
    k8s_text = K8S.read_text()
    mismatches: list[str] = []

    for name in CONFIGMAP_FILES:
        standalone = read_standalone_file(name)
        embedded = extract_configmap_file(k8s_text, name)
        if embedded is None:
            print(f"error: could not extract {name} from k8s manifest", file=sys.stderr)
            return 1
        if standalone != embedded:
            mismatches.append(name)

    if not mismatches:
        print("OK: bridge scripts match k8s ConfigMap")
        return 0

    for name in mismatches:
        print(f"MISMATCH: bridge/{name} differs from k8s ConfigMap", file=sys.stderr)
    print("Run: python3 scripts/sync-bridge-to-k8s.py", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
