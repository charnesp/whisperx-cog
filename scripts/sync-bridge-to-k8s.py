#!/usr/bin/env python3
"""No-op: k8s loads bridge from GHCR; push bridge/ changes to rebuild the image."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge_k8s import BRIDGE_IMAGE


def main() -> int:
    print(
        "k8s uses the published bridge image; no ConfigMap sync required.\n"
        f"  image: {BRIDGE_IMAGE}\n"
        "  after editing bridge/*.py: commit and push to main (or run "
        "`.github/workflows/bridge-docker-publish.yml` via workflow_dispatch)\n"
        "  then: kubectl rollout restart deployment/whisperx-stack"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
