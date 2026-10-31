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

## What the loader actually requires (measured 2026-10-31)

`readelf -d` on the torch 2.8.0 cu128 wheel, every library loaded at `import
torch`:

- `libtorch_cuda.so` NEEDS `libcudart.so.12`, `libcusparse.so.12`,
  `libcufft.so.11`, `libcufile.so.0`, `libcusparseLt.so.0`, `libnccl.so.2`,
  `libcurand.so.10`, `libcublas.so.12`, `libcublasLt.so.12`, `libcudnn.so.9`
- `libtorch_cpu.so` NEEDS `libcupti.so.12`, `libcudart.so.12`
- `libtorch_python.so` NEEDS `libcudnn.so.9`, `libcusparseLt.so.0`, `libcufile.so.0`
- `libtorch_cuda_linalg.so` NEEDS `libcusolver.so.11`, `libcublas.so.12`,
  `libcusparse.so.12`

So no CUDA wheel can be dropped on a 12.5 base: every one of them is a hard
dependency of a library the import loads. The only way out is to have the base
provide the same CUDA version as the wheel, which is what the base image of a
Cog build is for.

## Decision taken (2026-10-31)

1. `build.cuda` goes from `12.5` to `12.8`, the version the torch wheel is built
   for. `r8.im/cog-base:cuda12.8-python3.13` exists; the GPU host runs driver
   `610.43.02`, which covers CUDA 13.0, so 12.8 is well inside its range.
2. `scripts/image_diet.py` removes, after installation, the `nvidia-*` wheels the
   base already provides, `triton` (only used by `torch.compile`), and
   `torch/include` + `torch/test`.
3. Two wheels the base does NOT provide are kept and checked:
   `nvidia-cusparselt-cu12` (`libcusparseLt.so.0`) and `nvidia-cufile-cu12`
   (`libcufile.so.0`). The script aborts the build if either is missing instead
   of shipping an image that cannot import torch.
4. The build toolchain (gcc, make, `-dev` headers) is purged at the end, without
   `autoremove`, which could cascade into the CUDA packages installed as
   dependencies by the base.
5. `LD_LIBRARY_PATH` points at the base CUDA (`/usr/local/cuda/lib64`) and the
   system libraries, so the loader finds the libraries torch no longer carries.

Expected: about 3.2 GB less (9.40 GB -> around 6.2 GB). The measured value is
filled in below once the image is rebuilt by the publish workflow.

## Measured after the change

Pending the first build of the change (see `openspec/changes/image-cuda128-diet/`).
