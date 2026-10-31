## Context

Facts measured on 2026-10-30/31, all reproducible from the registry and the
wheels:

- image: 28 layers, 9.40 GB compressed; pip layer 4.48 GB; CUDA layers of the
  base 4.49 GB (of which 2.48 GB is `cuda-cudart-dev`).
- torch 2.8.0 PyPI wheel: 0.89 GB; its `nvidia-*` dependencies: 3.06 GB.
- torch 2.8.0 requires CUDA 12.8 (the PyPI manylinux wheel is cu128).
- Cog base images exist for `cuda12.5` and `cuda12.8` (`r8.im/cog-base`).
- the GPU host driver is 610.43.02 (CUDA 13.0 capable), so a 12.8 base runs.
- `readelf -d` on the wheel: `libtorch_cuda.so`, `libtorch_cpu.so`,
  `libtorch_python.so` and `libtorch_cuda_linalg.so` NEED cudart, cublas,
  cublasLt, cufft, curand, cusolver, cusparse, cusparseLt, cufile, cudnn, nccl,
  cupti.

## Decisions

### D1 - Align the base with the wheel, not the other way round

The alternative (keep 12.5, drop the wheels) is impossible: the libraries are
hard dependencies at import. The alternative (use a torch built for 12.5) is
impossible too: no cu125 build exists for 2.8.0. Downgrading torch to a build
matching 12.5 would change the engine; refused.

### D2 - The diet is a script, tested on a fake tree

`scripts/image_diet.py` computes and applies a removal plan on a
`site-packages` directory. It is pure `pathlib`/`shutil`, so the whole behaviour
is unit-tested without Docker: what is removed, what is kept, what happens when
a kept wheel is missing, what a dry run does, and what the JSON report counts.

### D3 - Fail closed on the two wheels the base lacks

`nvidia-cusparselt-cu12` and `nvidia-cufile-cu12` are not in the Cog base. If
either is missing, the script exits 2 with the reason and the build fails. A
silent removal would produce an image that builds, publishes, and dies at
`import torch` in production.

### D4 - No `autoremove` after the purge

The CUDA packages of the base are installed as dependencies. `autoremove` after
purging the toolchain would be free to remove them. Only explicitly named
packages are purged.

### D5 - `LD_LIBRARY_PATH` as a belt, `ldconfig` as the braces

The Cog base for 12.8 installs CUDA under `/usr/local/cuda/lib64` and the system
path; `ldconfig` runs after the removal, and `LD_LIBRARY_PATH` names both
directories so a base image update that stops registering the CUDA path does not
turn into a mystery at import.

## Test boundaries

- `tests/test_image_diet.py` (9 tests): removals, kept wheels, triton isolation,
  torch headers/tests vs libs, dist-info cleanup, dry run, missing kept wheel,
  JSON report, bad path.
- `tests/test_cog_yaml.py` (5 new tests): base CUDA is 12.8, the diet runs once
  with `--apply`, the purge never uses `autoremove`, the loader path is set, the
  requirements keep torch pinned and the UI stack out.
- What cannot be unit-tested is the GPU runtime. That is the canary smoke
  (`whisperx-canary-smoke`), not a test in this repository, and it is the gate
  before production.

## Risks

- A CUDA library path or version mismatch shows up only on the GPU. Mitigation:
  canary first, transcribe + align + diarize on a real clip, compare the
  transcript with the previous image.
- Removing `triton` breaks `torch.compile` if anyone calls it. Nothing in this
  service does; if that changes, the fix is to add it back in `requirements.txt`.
- Purging `-dev` packages removes the ability to compile CUDA extensions inside
  the image. Nothing builds extensions at runtime; if it ever does, the build
  must be changed, not the runtime patched.

## Loader configuration (decided 2026-10-31, after a failed build)

The first attempt declared `LD_LIBRARY_PATH=/usr/local/cuda/lib64` in
`environment`. The image build failed: `environment variable "LD_LIBRARY_PATH" is
not allowed` — cog validates the `environment` block and refuses loader variables
(it rewrites process environments itself). The base CUDA path is therefore
registered in a loader configuration file, written in `run` before `ldconfig`:

```
printf '/usr/local/cuda/lib64\n/usr/lib/x86_64-linux-gnu\n' > /etc/ld.so.conf.d/zz-cuda-runtime.conf && ldconfig
```

This survives in the image, needs no environment variable, and is asserted by two
tests: the configuration file is written, and no refused variable is declared.
