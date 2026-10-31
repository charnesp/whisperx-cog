# Image size: measurements and options

Measured on 2026-10-30 from the published image (`ghcr.io/charnesp/whisperx-cog:latest`),
through the registry manifest and the image config: 28 layers, **9.40 GB
compressed**. Weights are not baked (see `cog.yaml`), so this is code and runtime
only.

## Where the bytes are

| Layer | Compressed | Content |
|-------|-----------|---------|
| 18 | 4.48 GB | `pip install -r requirements.txt` |
| 9 | 2.48 GB | base: `cuda-cudart-dev-12-5` (headers, static libs) |
| 5 | 1.37 GB | base: `cuda-libraries-12-5` (cublas, cufft, curand, cusparse...) |
| 11 | 0.58 GB | base: cuDNN runtime + dev |
| 24 | 0.14 GB | ffmpeg 6 from the PPA |
| 15 | 0.14 GB | pyenv + Python build |
| 14 | 0.10 GB | build-essential, make, libssl-dev, zlib, bz2, readline... |

The pip layer is 48 % of the image on its own, and it is almost entirely CUDA:

| Wheel | Compressed |
|-------|-----------|
| `torch==2.8.0` (cp313, manylinux_2_28) | 0.89 GB |
| `nvidia-cublas-cu12` | 0.58 GB |
| `nvidia-cudnn-cu12` | 0.77 GB |
| `nvidia-cusparse-cu12` + `nvidia-cusolver-cu12` | 0.70 GB |
| `nvidia-nccl-cu12` | 0.35 GB |
| `nvidia-cusparselt-cu12` | 0.24 GB |
| `nvidia-cufft-cu12` | 0.20 GB |
| other `nvidia-*` | 0.13 GB |
| **total torch + nvidia** | **3.94 GB** |

## The finding

CUDA is shipped twice: once by the Cog base image (4.5 GB, dev variants
included) and once by the pip `nvidia-*` wheels pulled in by `torch` (3.06 GB).
The base dev packages (`cuda-cudart-dev`, cuDNN dev) exist to build CUDA
extensions; this image builds none.

## Options

**A. Drop the duplicate CUDA wheels (about -3.0 GB, expected 9.4 -> 6.4 GB).**
Install `torch` with `--no-deps` for the `nvidia-*` packages and let torch use
the base CUDA 12.5. The PyPI wheel is built for CUDA 12.8, so it must be replaced
by the cu125 build (`--index-url https://download.pytorch.org/whl/cu125`);
CUDA minor-version compatibility does not allow running a 12.8 build on a 12.5
runtime. Requires a build and a GPU smoke (transcribe + diarize + align) before
any tag moves.

**B. Remove the build toolchain (-0.24 GB).** `build-essential`, `make`, headers
and the pyenv build dependencies are only needed while installing. A final
`apt-get purge` layer keeps them out of the shipped image. Risk: a dependency
that compiles at first import (numba, llvmlite) would break; both are built at
install time, so the risk is low but must be smoke-tested.

**C. Check `librosa` (-0.15 GB, unverified).** It pulls numba, llvmlite and
scipy. Whether whisperx, pyannote or qwen-asr import it at runtime has not been
checked yet; do not remove before an import audit.

**D. Nothing to gain on caches.** `pip` runs with `--no-cache-dir` and apt lists
are removed in the layers we control.

## Recommendation

A is the only large win, and it is the only one that touches the runtime: it
needs its own change, a canary build and a GPU smoke, not a quiet edit. B can
ride along in the same change. C stays open until the import audit is done.
