# cmake/sz_riscv_isa_probes.cmake — RISC-V ISA compiler-capability probes
#
# Probe sources live in probes/riscv_*.c — shared with build.rs. The RVV probe carries its ISA in a per-function
# `target("arch=+v")` pragma, like the real kernels; the toolchain file's global `-march` (e.g. `rv64gcv`) also applies
# to probes through the regular flag inheritance.

include(cmake/sz_isa_probe.cmake)

sz_instruction_set_probe_(RVVCRYPTO SOURCE probes/riscv_rvvcrypto.c)
sz_instruction_set_probe_(RVV SOURCE probes/riscv_rvv.c)

sz_build_instruction_set_definitions_("RISC-V" "RVVCRYPTO;RVV")
