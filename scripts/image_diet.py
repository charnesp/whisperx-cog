#!/usr/bin/env python3
"""Post-install diet: drop from the image what the runtime never loads.

Why
---
The Cog base image already installs CUDA 12.8 for the whole stack (cudart,
cublas, cudnn, cusparse, cusolver, cufft, curand, nccl, nvrtc, nvtx, cupti).
torch's PyPI wheel declares the same libraries again as `nvidia-*` pip
packages, so the image carried CUDA twice: 3.06 GB of duplicated wheels on top
of a 4.5 GB CUDA base (measured 2026-10-30 on ghcr.io/charnesp/whisperx-cog).

What is removed
---------------
- `nvidia/*` wheels the base already provides. Their absence is safe because
  the wheels are the same CUDA major.minor as the base image (12.8): torch
  resolves the libraries from the base through the loader cache.
- `triton`: only used by torch.compile, which this service never calls.
- `torch/include` and `torch/test`: headers and the test suite of torch, never
  used at runtime.
- the `.dist-info` of every removed package, so the environment does not
  advertise packages that are gone.

What is KEPT, and why it is checked
----------------------------------
`nvidia/cusparselt` and `nvidia/cufile` are NOT in the base image, but
libtorch_python.so needs libcusparseLt.so.0 and libtorch_cuda.so needs
libcufile.so.0 at import time (readelf -d, 2026-10-31). Removing them breaks
`import torch`.

`nvidia/nccl` IS in the base image, but in a version too old for torch 2.8.0:
the build failed with `libtorch_cuda.so: undefined symbol: ncclCommWindowRegister`
(measured 2026-10-31, after the diet). The base copy cannot replace the wheel.

Their absence therefore aborts the build instead of shipping a broken image.

Usage
-----
    python3 scripts/image_diet.py                      # simulation
    python3 scripts/image_diet.py --apply              # suppression
    python3 scripts/image_diet.py --site-packages /path --apply --json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sysconfig
import sys
from pathlib import Path

# Duplicated by the Cog base image (cog-base:cuda12.8): safe to drop.
REMOVE_NVIDIA = (
    "cublas",
    "cuda_cupti",
    "cuda_nvrtc",
    "cuda_runtime",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nvjitlink",
    "nvtx",
)

# Absent from the base image and needed at import: never remove.
KEEP_NVIDIA = ("nccl", "cusparselt", "cufile")

REMOVE_PACKAGES = ("triton",)
REMOVE_TORCH_SUBDIRS = ("include", "test")


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _dist_infos(site: Path, name: str) -> list[Path]:
    """dist-info directories of a pip package, whatever its name flavour."""
    patterns = (f"nvidia_{name}_cu12-*.dist-info", f"nvidia-{name}-*.dist-info",
                f"{name}-*.dist-info")
    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(site.glob(pattern)))
    return found


def plan(site: Path) -> dict:
    """Compute what to remove. Raises before touching anything if unsafe."""
    if not site.is_dir():
        raise FileNotFoundError(f"site-packages introuvable: {site}")

    nvidia = site / "nvidia"
    missing_kept = [name for name in KEEP_NVIDIA if not (nvidia / name).is_dir()]
    if missing_kept:
        raise RuntimeError(
            "roue(s) indispensable(s) absente(s) de " + str(nvidia) + ": "
            + ", ".join(missing_kept)
            + " — libtorch_python.so / libtorch_cuda.so en ont besoin a l'import ; "
            "refus de produire une image cassee"
        )

    targets: list[Path] = []
    for name in REMOVE_NVIDIA:
        d = nvidia / name
        if d.is_dir():
            targets.append(d)
            targets.extend(_dist_infos(site, name))
    for name in REMOVE_PACKAGES:
        d = site / name
        if d.is_dir():
            targets.append(d)
            targets.extend(_dist_infos(site, name))
    for sub in REMOVE_TORCH_SUBDIRS:
        d = site / "torch" / sub
        if d.is_dir():
            targets.append(d)

    kept = sorted(p.name for p in nvidia.iterdir() if p.is_dir()) if nvidia.is_dir() else []
    return {
        "site_packages": str(site),
        "targets": sorted(set(targets)),
        "kept_nvidia": kept,
        "bytes": sum(_size(t) for t in set(targets)),
    }


def apply(item: dict) -> dict:
    removed = 0
    for target in item["targets"]:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        removed += 1
    return {
        "removed_dirs": removed,
        "removed_bytes": item["bytes"],
        "removed_mb": round(item["bytes"] / 1e6, 1),
        "kept_nvidia": item["kept_nvidia"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="allege l'image apres installation")
    parser.add_argument(
        "--site-packages",
        default=None,
        help="repertoire site-packages (defaut: celui de l'interpreteur courant)",
    )
    parser.add_argument("--apply", action="store_true", help="supprime (defaut: simulation)")
    parser.add_argument("--json", action="store_true", help="rapport JSON sur stdout")
    args = parser.parse_args(argv)

    site = Path(args.site_packages) if args.site_packages else Path(
        sysconfig.get_paths()["purelib"])
    try:
        item = plan(site)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ECHEC: {exc}", file=sys.stderr)
        return 2

    if not args.apply:
        if args.json:
            print(json.dumps({"targets": [str(t) for t in item["targets"]],
                              "bytes": item["bytes"], "applied": False}, indent=2))
        else:
            for target in item["targets"]:
                print(f"a supprimer: {target}")
            print(f"total: {item['bytes'] / 1e6:.1f} Mo")
        return 0

    result = apply(item)
    result["applied"] = True
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"supprime: {result['removed_dirs']} entrees, "
              f"{result['removed_mb']} Mo liberes ; conserve: "
              + ", ".join(result["kept_nvidia"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
