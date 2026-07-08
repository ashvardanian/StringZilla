# cmake/sz_riscv_isa_probes.cmake — RISC-V ISA compiler-capability probes
#
# Probe sources live in probes/riscv_*.c — shared with build.rs. The RVV probe carries its ISA in a per-function
# `target("arch=+v")` pragma, like the real kernels; the toolchain file's global `-march` (e.g. `rv64gcv`) also applies
# to probes through the regular flag inheritance.

include(cmake/sz_isa_probe.cmake)

sz_isa_probes_begin_()
sz_isa_probe_(SZ_CAN_COMPILE_RVVCRYPTO "" "" "probes/riscv_rvvcrypto.c")
sz_isa_probe_(SZ_CAN_COMPILE_RVV "" "" "probes/riscv_rvv.c")
sz_isa_probes_end_()

set(SZ_ISA_TIERS "RVVCRYPTO;RVV")
