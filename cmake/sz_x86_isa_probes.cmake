# cmake/sz_x86_isa_probes.cmake — x86-64 ISA compiler-capability probes
#
# Probe sources live in probes/x86_*.c — shared with build.rs. No flag columns: the probes carry their ISA in
# per-function `target` pragmas over baseline flags on GCC/Clang, and MSVC compiles any x86 intrinsic regardless of
# `/arch`, so every tier's availability is decided by the toolchain alone.

include(cmake/sz_isa_probe.cmake)

sz_isa_probes_begin_()
sz_isa_probe_(SZ_CAN_COMPILE_ICELAKE "" "" "probes/x86_icelake.c")
sz_isa_probe_(SZ_CAN_COMPILE_SKYLAKE "" "" "probes/x86_skylake.c")
sz_isa_probe_(SZ_CAN_COMPILE_HASWELL "" "" "probes/x86_haswell.c")
sz_isa_probe_(SZ_CAN_COMPILE_GOLDMONT "" "" "probes/x86_goldmont.c")
sz_isa_probe_(SZ_CAN_COMPILE_WESTMERE "" "" "probes/x86_westmere.c")
sz_isa_probes_end_()

set(SZ_ISA_TIERS "ICELAKE;SKYLAKE;HASWELL;GOLDMONT;WESTMERE")
