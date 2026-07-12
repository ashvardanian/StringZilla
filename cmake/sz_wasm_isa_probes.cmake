# cmake/sz_wasm_isa_probes.cmake — WebAssembly SIMD compiler-capability probes
#
# Probe sources live in probes/wasm_*.c — shared with build.rs. Unlike the native tiers, wasm selects SIMD with whole-TU
# flags (`-msimd128`, `-mrelaxed-simd`) rather than per-function `target` pragmas, so the probes carry those flags;
# there is no runtime capability probe on wasm, making the compile set the only gate for both dispatch modes.

include(cmake/sz_isa_probe.cmake)

sz_isa_probe_(SZ_CAN_COMPILE_V128RELAXED SOURCE probes/wasm_v128relaxed.c GNU_FLAGS -msimd128 -mrelaxed-simd)
sz_isa_probe_(SZ_CAN_COMPILE_V128 SOURCE probes/wasm_v128.c GNU_FLAGS -msimd128)

set(SZ_ISA_TIERS "V128RELAXED;V128")
