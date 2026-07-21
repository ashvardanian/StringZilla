# StringZilla C Libraries

This part of the project is responsible for implementing dynamic dispatch.

`stringzilla/` holds the single-threaded core ABI.
`stringzillas/` holds the batch and parallel engines, the C lowering of the C++ and CUDA templates.

Which `SZ_USE_*` SIMD tiers a build enables is decided by the top-level `probes/` programs, shared between CMake and Cargo: each tier is try-compiled to learn what the toolchain can emit, and `probes/run_capabilities.c` is executed to learn what the build machine can run.
