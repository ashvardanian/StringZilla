# StringZilla C Libraries

This part of the project is responsible for implementing dynamic dispatch.

`stringzilla/` holds the core ABI, one translation unit per kernel family, with `dispatch.h` and `runtime.c` resolving the tier once at load.

Which `STRINGZILLA_TARGET_*` SIMD tiers a build enables is decided by the top-level `probes/` programs, shared between CMake and Cargo: each tier is try-compiled to learn what the toolchain can emit, and `probes/run_capabilities.c` is executed to learn what the build machine can run.
