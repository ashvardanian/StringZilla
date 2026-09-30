# StringZilla C Libraries

This part of the project builds the compiled library behind the `_best` dispatch points.

- `stringzilla.c` holds the cross-family exports: versions, statuses, capability queries, `sz_find_kernel_punned`, the device exports of the GPU vendors a build lacks, and the LibC overrides.
- `cpu/<capability>.c` defines that capability's kernels, once for the whole library, by including its tier headers.
- `dispatch/<family>.c` holds each family's capability lists, its `_best` dispatch points and its finder, sharing the kernel pick in `dispatch.h`.
- `nvidia/cuda.cu`, `amd/rocm.hip` and `apple/metal.c` define the engines' `cuda`, `rocm` and `metal` kernels and each vendor's device exports, joining the same library under `STRINGZILLA_BUILD_CUDA`, `STRINGZILLA_BUILD_ROCM` and `STRINGZILLA_BUILD_METAL`.
- `parallel.h` and `parallel.c` run independent tiles on each platform's own thread pool, and are compiled into the Python and Node extensions, never into the libraries.

Which `STRINGZILLA_TARGET_*` capabilities a build enables is decided by the top-level `probes/` programs, which CMake compiles for every binding: each capability's probe is compiled as the library would compile it, to learn what the toolchain can emit, and again for this machine alone, to learn what it can run.
