# cmake/sz_power_isa_probes.cmake — IBM POWER ISA compiler-capability probes
#
# Probe sources live in probes/power_*.c — shared with build.rs. The VSX probe carries its ISA in a per-function
# `target("power9-vector")` pragma, like the real kernels; the toolchain file's global `-mcpu=power9 -mvsx` also reaches
# the probe through the regular flag inheritance.

include(cmake/sz_isa_probe.cmake)

sz_isa_probe_(SZ_CAN_COMPILE_POWERVSX SOURCE probes/power_vsx.c)

set(SZ_ISA_TIERS "POWERVSX")
