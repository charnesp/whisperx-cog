#!/usr/bin/env python3
"""
Simulate Cog's base image selection (config resolution + base image choice).

Cog does not expose a "dry-run" for this; this script replicates the logic
from cog.yaml + requirements.txt so you can see what gets resolved without
running a full build.

Usage (from repo root):
  python scripts/simulate_cog_base_selection.py
  python scripts/simulate_cog_base_selection.py --config path/to/cog.yaml
"""

import argparse
import re
import sys
from pathlib import Path


def load_yaml_simple(path: Path) -> dict:
    """Minimal YAML parsing for cog.yaml build section (no PyYAML required)."""
    text = path.read_text()
    build: dict = {}
    in_build = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.match(r"^build\s*:", stripped):
            in_build = True
            continue
        if in_build and re.match(r"^(predict|image)\s*:", stripped):
            break
        if in_build:
            m = re.match(r"^(\w+)\s*:\s*(.+)$", line.strip())
            if m:
                k, v = m.group(1), m.group(2).strip()
                if k == "cuda":
                    build["cuda"] = v.strip('"').strip("'")
                elif k == "python_version":
                    build["python_version"] = v.strip('"').strip("'")
                elif k == "gpu":
                    build["gpu"] = v.lower() in ("true", "yes", "1")
    return build


def get_torch_version_from_requirements(path: Path) -> str | None:
    """Parse requirements.txt for torch version (e.g. torch==2.5.1 or torch>=2.4)."""
    if not path.exists():
        return None
    text = path.read_text()
    # Match torch, torch==x.y.z, torch>=x.y, etc.
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line or not line.lower().startswith("torch"):
            continue
        # torch or torch==2.5.1 or torch>=2.4
        m = re.search(r"torch\s*([=<>~!]+)\s*([\d.]+)", line, re.I)
        if m:
            return m.group(2).strip()
        if re.match(r"^torch\s*$", line, re.I):
            return "(unpinned)"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Cog base image selection")
    parser.add_argument(
        "--config", "-f",
        default="cog.yaml",
        help="Path to cog.yaml",
    )
    parser.add_argument(
        "--requirements", "-r",
        default="requirements.txt",
        help="Path to requirements file",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    config_path = root / args.config
    requirements_path = root / args.requirements

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    build = load_yaml_simple(config_path)
    cuda = build.get("cuda") or "(from PyTorch/TF compat matrix)"
    python_version = build.get("python_version") or "3.13"
    gpu = build.get("gpu", False)

    torch_version = get_torch_version_from_requirements(requirements_path)

    print("=== Resolved config (what Cog uses for base image selection) ===")
    print(f"  Python:  {python_version}")
    print(f"  CUDA:   {cuda}")
    print(f"  Torch:  {torch_version or '(none)'}")
    print(f"  GPU:    {gpu}")
    print()

    if not gpu:
        print("Base image: python slim (no GPU)")
        print(f"  → python:{python_version}-slim")
        return

    if not torch_version or torch_version == "(none)" or torch_version == "(unpinned)":
        print("Base image selection: ⚠ Torch not pinned in requirements → Cog shows 'Torch: (none)'")
        print("  → No cog-base tag can be chosen (format is cudaX-pythonY-torchZ).")
        print("  → Cog falls back to: 'continuing without base image support' and uses NVIDIA CUDA image.")
        print()
        print("Fix: add an explicit torch version in requirements.txt, e.g.:")
        print("  torch>=2.4   or   torch==2.5.1")
        return

    # Normalize to major.minor for tag (e.g. 2.5.1 -> 2.5)
    torch_major_minor = ".".join(torch_version.split(".")[:2]) if torch_version else ""
    cuda_short = (cuda or "").replace('"', "").replace("'", "").strip()
    py_short = python_version.replace(".", "")  # 3.11 -> 311

    print("Cog base image lookup:")
    print(f"  1. Look for pre-built cog-base: r8.im/cog-base:cuda{cuda_short}-python{py_short}-torch{torch_major_minor}")
    print("     (Cog checks the registry; if the tag exists, this image is used.)")
    print()
    print("  2. If no matching cog-base exists:")
    print(f"     → Fallback: nvidia/cuda:{cuda_short}.x-cudnnX-devel-ubuntu22.04")
    print("     → Python and PyTorch are then installed in the Dockerfile (slower build).")
    print()
    print("To see Cog's actual choice and warnings, run:")
    print("  cog build -t dummy:test --progress=plain 2>&1 | head -20")
    print("  (Cancel the build after the first lines if you only want the selection output.)")


if __name__ == "__main__":
    main()
