## Why

The published image is 9.40 GB compressed (28 layers, measured 2026-10-30 from
the registry manifest). CUDA is shipped twice:

- the Cog base image for CUDA 12.5 installs 4.49 GB of CUDA and cuDNN packages,
  including the `-dev` variants and a command-line toolkit nobody uses at
  runtime;
- the pip layer installs `torch==2.8.0` from PyPI, whose wheel is built for CUDA
  12.8 and declares 14 `nvidia-*` wheel dependencies, 3.94 GB with torch itself.

`readelf -d` (2026-10-31) shows why the wheels cannot simply be dropped on the
current base: every CUDA library is a hard `DT_NEEDED` of a shared object loaded
at `import torch` (`libnccl.so.2`, `libcusparseLt.so.0`, `libcufile.so.0`,
`libcupti.so.12`, `libcudnn.so.9`, `libcublas.so.12`, ...). Two CUDA runtimes of
different minor versions cannot coexist for one process: the wheel's minor
version wins and the base version is dead weight.

## What Changes

- `build.cuda` goes from `12.5` to `12.8`, the CUDA the torch wheel is built for.
- New `scripts/image_diet.py`, run after installation: it removes the `nvidia-*`
  wheels the base already provides, `triton` (only used by `torch.compile`, never
  called by this service), `torch/include` and `torch/test`, plus the
  `.dist-info` of every removed package.
- It KEEPS `nvidia-cusparselt-cu12` (`libcusparseLt.so.0`) and
  `nvidia-cufile-cu12` (`libcufile.so.0`), which the base image does not provide,
  and aborts the build if either is missing rather than shipping an image whose
  `import torch` fails.
- The build toolchain (gcc, g++, make, `-dev` headers, pyenv build dependencies)
  is purged at the end of the build, without `autoremove`.
- `LD_LIBRARY_PATH` points at `/usr/local/cuda/lib64` and the system library
  directory, so the loader finds the base CUDA libraries torch no longer carries.

## Non-goals

- No change to the models, the weights, the transcription code or its behaviour:
  the image contains the same Python packages that the runtime imports.
- No change to the API, the endpoints or the environment contract of the
  predictor.
- No base image of our own: the Cog base for CUDA 12.8 is used as published, so a
  Cog update keeps working.
- `librosa` (numba, llvmlite, scipy) is NOT removed: its runtime use has not been
  audited, and an unverified removal is how a working image becomes a broken one.

## Capabilities

### Modified Capabilities

- `packaging`: the image ships one CUDA runtime, the one the torch wheel is built
  for, and no build toolchain.
