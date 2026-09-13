# cmake/sz_x86_isa_probes.cmake — x86-64 ISA compiler-capability probes
#
# Probe sources live in probes/x86_*.c — shared with build.rs. No flag columns: the probes carry their ISA in
# per-function `target` pragmas over baseline flags on GCC/Clang, and MSVC compiles any x86 intrinsic regardless of
# `/arch`, so every tier's availability is decided by the toolchain alone.

include(cmake/sz_isa_probe.cmake)

sz_instruction_set_probe_(ICELAKE SOURCE probes/x86_icelake.c)
sz_instruction_set_probe_(SKYLAKE SOURCE probes/x86_skylake.c)
sz_instruction_set_probe_(HASWELL SOURCE probes/x86_haswell.c)
sz_instruction_set_probe_(GOLDMONT SOURCE probes/x86_goldmont.c)
sz_instruction_set_probe_(WESTMERE SOURCE probes/x86_westmere.c)

sz_build_instruction_set_definitions_("x86" "ICELAKE;SKYLAKE;HASWELL;GOLDMONT;WESTMERE")
