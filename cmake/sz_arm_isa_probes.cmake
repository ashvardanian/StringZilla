# cmake/sz_arm_isa_probes.cmake — Arm ISA compiler-capability probes
#
# Probe sources live in probes/arm_*.c — shared with build.rs. No flag columns: the probes carry their ISA in
# per-function `target` pragmas over baseline flags, exactly like the real kernels, and MSVC has no Arm SVE support (the
# SVE probes `#error` there by design, so those tiers probe as unavailable).

include(cmake/sz_isa_probe.cmake)

sz_isa_probe_(SZ_CAN_COMPILE_SVE2AES SOURCE probes/arm_sve2aes.c)
sz_isa_probe_(SZ_CAN_COMPILE_SVE2 SOURCE probes/arm_sve2.c)
sz_isa_probe_(SZ_CAN_COMPILE_SVE SOURCE probes/arm_sve.c)
sz_isa_probe_(SZ_CAN_COMPILE_NEONSHA SOURCE probes/arm_neonsha.c)
sz_isa_probe_(SZ_CAN_COMPILE_NEONAES SOURCE probes/arm_neonaes.c)
sz_isa_probe_(SZ_CAN_COMPILE_NEON SOURCE probes/arm_neon.c)

set(SZ_ISA_TIERS "SVE2AES;SVE2;SVE;NEONSHA;NEONAES;NEON")
