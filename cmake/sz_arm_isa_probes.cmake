# cmake/sz_arm_isa_probes.cmake — Arm ISA compiler-capability probes
#
# Probe sources live in probes/arm_*.c — shared with build.rs. No flag columns: the probes carry their ISA in
# per-function `target` pragmas over baseline flags, exactly like the real kernels, and MSVC has no Arm SVE support (the
# SVE probes `#error` there by design, so those tiers probe as unavailable).

include(cmake/sz_isa_probe.cmake)

sz_isa_probes_begin_()
sz_isa_probe_(SZ_CAN_COMPILE_SVE2AES "" "" "probes/arm_sve2aes.c")
sz_isa_probe_(SZ_CAN_COMPILE_SVE2 "" "" "probes/arm_sve2.c")
sz_isa_probe_(SZ_CAN_COMPILE_SVE "" "" "probes/arm_sve.c")
sz_isa_probe_(SZ_CAN_COMPILE_NEONSHA "" "" "probes/arm_neonsha.c")
sz_isa_probe_(SZ_CAN_COMPILE_NEONAES "" "" "probes/arm_neonaes.c")
sz_isa_probe_(SZ_CAN_COMPILE_NEON "" "" "probes/arm_neon.c")
sz_isa_probes_end_()

set(SZ_ISA_TIERS "SVE2AES;SVE2;SVE;NEONSHA;NEONAES;NEON")
