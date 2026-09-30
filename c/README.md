# StringZilla C Libraries

This part of the project builds the compiled library behind the `_best` dispatch points.

- `stringzilla.c` holds the cross-family exports: versions, statuses, capability queries, `sz_find_kernel_punned`, the device exports of the GPU vendors a build lacks, and the LibC overrides.
- `cpu/<capability>.c` defines that capability's kernels, once for the whole library, by including its tier headers.
- `dispatch/<family>.c` holds each family's capability lists, its `_best` dispatch points and its finder, sharing the kernel pick in `dispatch.h`.
- `nvidia/cuda.cu`, `amd/rocm.hip` and `apple/metal.c` define the engines' `cuda`, `rocm` and `metal` kernels and each vendor's device exports, joining the same library under `STRINGZILLA_BUILD_CUDA`, `STRINGZILLA_BUILD_ROCM` and `STRINGZILLA_BUILD_METAL`.
- `parallel.h` and `parallel.c` run independent tiles on each platform's own thread pool, and are compiled into the Python and Node extensions, never into the libraries.

Which `STRINGZILLA_TARGET_*` capabilities a build enables is decided by the top-level `probes/` programs, which CMake compiles for every binding: each capability's probe, `probes/<capability>.c`, calls one of its kernels, compiled header-only at the baseline flags as the library compiles it, to learn whether the toolchain builds them.
Every unit compiles at the architecture's floor, and each capability's tier headers scope its kernels with a target pragma, so the library carries every capability the toolchain builds and dispatches by runtime detection.
`cpu/loongsonasx.c` and `cpu/powervsx.c` alone also take `-mlasx` and `-mcpu=power9` file-wide, as `lasxintrin.h` and `altivec.h` hide their contents without them.
`-D STRINGZILLA_TARGET_<KIT>=0` drops a capability, and `=1` keeps one only where its probe compiles.
