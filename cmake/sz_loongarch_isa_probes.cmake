# cmake/sz_loongarch_isa_probes.cmake — LoongArch ISA compiler-capability probes
#
# Probe sources live in probes/loongarch_*.c — shared with build.rs. The LASX kernels carry no per-function `target`
# attributes, so the whole translation unit needs LASX enabled; the toolchain file's global `-mlasx` reaches the probe
# through the regular flag inheritance, keeping the probe honest about what the real kernels will see.

include(cmake/sz_isa_probe.cmake)

sz_isa_probe_(SZ_CAN_COMPILE_LASX SOURCE probes/loongarch_lasx.c)

set(SZ_ISA_TIERS "LASX")
