## ADDED Requirements

### Requirement: One CUDA runtime in the image

The image SHALL carry the CUDA runtime the Python engine is built for, and only
that one. The Cog base CUDA version SHALL match the CUDA version of the torch
wheel pinned in `requirements.txt`. Python packages that duplicate the CUDA
libraries provided by the base SHALL be removed after installation.

#### Scenario: Base and wheel versions differ

- **WHEN** the base image provides CUDA 12.5 and the pinned torch wheel is built for 12.8
- **THEN** the build SHALL use a base image providing CUDA 12.8, and the duplicated `nvidia-*` wheels SHALL be removed after installation

#### Scenario: A CUDA library is only available as a pip wheel

- **WHEN** a library needed at `import torch` is not provided by the base image (`libcusparseLt.so.0`, `libcufile.so.0`)
- **THEN** its wheel SHALL be kept, and its absence SHALL fail the build with the missing wheel named

### Requirement: Build-time only toolchain is not shipped

The compiler toolchain and development headers used to build Python and the
wheels SHALL be removed at the end of the build. Removal SHALL NOT use
`autoremove`, so that packages installed as dependencies of the base image are
never removed as collateral.

#### Scenario: Toolchain purge

- **WHEN** the build finishes installing the runtime dependencies
- **THEN** gcc, g++, make and the `-dev` headers SHALL be purged, and no `autoremove` SHALL run

#### Scenario: Removed CUDA wheels leave the loader a path

- **WHEN** the duplicate CUDA wheels are removed
- **THEN** the loader SHALL find the base CUDA libraries (`ldconfig`, explicit `LD_LIBRARY_PATH`)

### Requirement: The diet is verifiable without a GPU

The removal plan SHALL be computed by a script that is unit-tested on a
filesystem tree: the set of removed packages, the set of kept packages, the
dry-run behaviour and the failure mode on a missing required wheel SHALL be
covered by tests that need neither Docker nor a GPU.

#### Scenario: Dry run

- **WHEN** the diet script runs without `--apply`
- **THEN** it lists what it would remove and changes nothing

#### Scenario: Required wheel missing

- **WHEN** `nvidia-cusparselt-cu12` is absent from the environment
- **THEN** the script exits non-zero, names the wheel, and removes nothing
