/**
 *  @file include/stringzilla/cpu.h
 *  @author Ash Vardanian
 *  @date January 21, 2024
 *  @brief CPU detection behind the capability queries: what the compiler's flags imply, what each
 *      architecture's CPU reports through its OS, and what a dispatching thread needs.
 *
 *  The library includes it once, behind its cached @c sz_capabilities_detected_cpu, and header-only
 *  builds through the tail of <tt>capabilities.h</tt>, asking on every call. Every helper here is
 *  internal, and its platform switches are the only place the library asks the OS about the CPU.
 *
 *  @sa c/stringzilla/capabilities.h
 */
#ifndef STRINGZILLA_CPU_H_
#define STRINGZILLA_CPU_H_

#include "stringzilla/capabilities.h" // `sz_cap_haswell_k`, `sz_capability_t`

/* On Apple Silicon, @c mrs is not allowed in user-space, so we need to use the @c sysctl API */
#if STRINGZILLA_OS_APPLE_
#include <sys/sysctl.h>
#endif

/* On Linux and FreeBSD Arm, the kernel reports through @c AT_HWCAP whether it emulates @c mrs reads
 * of the ID registers, so nothing has to trap to find out. */
#if STRINGZILLA_ARCH_ARM64_ && (STRINGZILLA_OS_LINUX_ || STRINGZILLA_OS_FREEBSD_) && STRINGZILLA_WITH_LIBC
#include <sys/auxv.h> // `getauxval`, `elf_aux_info`, `AT_HWCAP`
#endif

/* On 64-bit RISC-V we probe HWCAP via the auxiliary vector and vector sub-extensions via the
 * Linux @c riscv_hwprobe syscall; FreeBSD lacks it and uses @c elf_aux_info for base RVV only. */
#if STRINGZILLA_ARCH_RISCV64_ && STRINGZILLA_WITH_LIBC
#if STRINGZILLA_OS_LINUX_
#include <sys/auxv.h>    // `getauxval`, `AT_HWCAP`
#include <sys/syscall.h> // `SYS_riscv_hwprobe`
#include <unistd.h>      // `syscall`
#elif STRINGZILLA_OS_FREEBSD_
#include <sys/auxv.h> // `elf_aux_info`, `AT_HWCAP`
#endif
#endif

/* On LoongArch and IBM POWER the SIMD extensions are likewise reported through the aux vector. */
#if (STRINGZILLA_ARCH_LOONGARCH64_ || STRINGZILLA_ARCH_PPC64_) && STRINGZILLA_WITH_LIBC
#if STRINGZILLA_OS_LINUX_
#include <sys/auxv.h> // `getauxval`, `AT_HWCAP`, `AT_HWCAP2`
#elif STRINGZILLA_OS_FREEBSD_
#include <sys/auxv.h> // `elf_aux_info`, `AT_HWCAP`, `AT_HWCAP2`
#endif
#endif

/* On Windows Arm, we use IsProcessorFeaturePresent API for capability detection */
#if STRINGZILLA_OS_WINDOWS_ && STRINGZILLA_ARCH_ARM64_
#if !defined(_WINSOCKAPI_)
#include <winsock2.h> // First, so `<windows.h>` leaves out Winsock 1.1, which a later `<winsock2.h>` collides with
#endif
#include <windows.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the capabilities the compiler's own flags guarantee, for the platforms whose OS cannot
 *  be asked. An engine validates a WebAssembly module whole, so there they are what it holds. */
STRINGZILLA_CONSTEXPR sz_capability_t sz_capabilities_implied_cpu_(void) {
    sz_capability_t capabilities = sz_cap_serial_k;
#if defined(__ARM_NEON)
    capabilities |= sz_cap_neon_k;
#endif
#if defined(__ARM_FEATURE_AES) || defined(__ARM_FEATURE_CRYPTO)
    capabilities |= sz_cap_neonaes_k;
#endif
#if defined(__ARM_FEATURE_SHA2) || defined(__ARM_FEATURE_CRYPTO)
    capabilities |= sz_cap_neonsha_k;
#endif
#if defined(__ARM_FEATURE_SVE)
    capabilities |= sz_cap_sve_k;
#endif
#if defined(__ARM_FEATURE_SVE2)
    capabilities |= sz_cap_sve2_k;
#endif
#if defined(__ARM_FEATURE_SVE2AES) || defined(__ARM_FEATURE_SVE2_AES)
    capabilities |= sz_cap_sve2aes_k;
#endif
#if defined(__riscv_vector)
    capabilities |= sz_cap_rvv_k;
#endif
#if defined(__riscv_vector) && defined(__riscv_zvkned) && defined(__riscv_zvknhb) && defined(__riscv_zvkg)
    capabilities |= sz_cap_rvvcrypto_k;
#endif
#if defined(__loongarch_asx)
    capabilities |= sz_cap_loongsonasx_k;
#endif
#if defined(__POWER9_VECTOR__)
    capabilities |= sz_cap_powervsx_k;
#endif
#if defined(__wasm_simd128__)
    capabilities |= sz_cap_v128_k;
#endif
#if defined(__wasm_relaxed_simd__)
    capabilities |= sz_cap_v128relaxed_k;
#endif
    return capabilities;
}

/*  The detectors below report the full hardware capability set, independent of which
 *  `STRINGZILLA_TARGET_*` kits this build compiled in: @c sz_capabilities_enabled_cpu ANDs their
 *  result with the compile-time mask anyway, and the instructions involved are unconditionally
 *  safe. @c cpuid is baseline x86-64 with @c xgetbv behind the OSXSAVE check, and the Arm @c mrs
 *  reads run only once the kernel says it emulates them. Keeping detection unconditional lets even
 *  a serial-only build report what this machine runs. */

#if STRINGZILLA_ARCH_ARM64_

/** The capabilities of the current 64-bit Arm CPU. */
STRINGZILLA_INLINE sz_capability_t sz_capabilities_detected_arm64_(void) {
#if STRINGZILLA_OS_APPLE_

    // On Apple Silicon, `mrs` is not allowed in user-space, so we need to use the `sysctl` API.
    uint32_t supports_neon = 0;
    uint32_t supports_neonaes = 0;
    uint32_t supports_neonsha = 0;
    size_t size = sizeof(supports_neon);
    // Every Apple arm64 core has NEON, AES and SHA2, so a denied sysctl keeps them.
    if (sysctlbyname("hw.optional.neon", &supports_neon, &size, NULL, 0) != 0) supports_neon = 1;
    if (sysctlbyname("hw.optional.arm.FEAT_AES", &supports_neonaes, &size, NULL, 0) != 0) supports_neonaes = 1;
    if (sysctlbyname("hw.optional.arm.FEAT_SHA256", &supports_neonsha, &size, NULL, 0) != 0) supports_neonsha = 1;

    // The `+crypto` in both tiers' targets enables AES and SHA2 together.
    return (sz_capability_t)(                                         //
        (sz_cap_neon_k * (supports_neon)) |                           //
        (sz_cap_neonaes_k * (supports_neonaes && supports_neonsha)) | //
        (sz_cap_neonsha_k * (supports_neonaes && supports_neonsha)) | //
        (sz_cap_serial_k));

#elif (STRINGZILLA_OS_LINUX_ || STRINGZILLA_OS_FREEBSD_) && STRINGZILLA_WITH_LIBC

    unsigned long const hwcap_cpuid = 1ul << 11; // HWCAP_CPUID: the kernel emulates EL0 reads of the ID registers
#if STRINGZILLA_OS_LINUX_
    unsigned long hwcap = getauxval(AT_HWCAP);
#else
    unsigned long hwcap = 0;
    elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap));
#endif
    if (!(hwcap & hwcap_cpuid)) return sz_capabilities_implied_cpu_();

    // Each raw `mrs` writes x0, as older assemblers reject the symbolic names of the SVE registers.
    unsigned long id_aa64isar0_el1 = 0, id_aa64pfr0_el1 = 0, id_aa64zfr0_el1 = 0;
    __asm__ __volatile__(".inst 0xD5380600\n\tmov %0, x0" : "=r"(id_aa64isar0_el1) : : "x0"); // ID_AA64ISAR0_EL1
    __asm__ __volatile__(".inst 0xD5380400\n\tmov %0, x0" : "=r"(id_aa64pfr0_el1) : : "x0");  // ID_AA64PFR0_EL1

    // AdvSIMD, bits [23:20] of ID_AA64PFR0_EL1, reads 0b1111 on the R-profile CPUs that lack NEON.
    unsigned supports_neon = ((id_aa64pfr0_el1 >> 20) & 0xF) != 0xF;
    // AES, bits [7:4], and SHA2, bits [15:12], of ID_AA64ISAR0_EL1.
    unsigned supports_neonaes = ((id_aa64isar0_el1 >> 4) & 0xF) >= 1;
    unsigned supports_neonsha = ((id_aa64isar0_el1 >> 12) & 0xF) >= 1;
    // SVE, bits [35:32] of ID_AA64PFR0_EL1, gates ID_AA64ZFR0_EL1: SVEver in [3:0], AES in [7:4].
    unsigned supports_sve = ((id_aa64pfr0_el1 >> 32) & 0xF) >= 1;
    if (supports_sve)
        __asm__ __volatile__(".inst 0xD5380480\n\tmov %0, x0" : "=r"(id_aa64zfr0_el1) : : "x0"); // ID_AA64ZFR0_EL1
    unsigned supports_sve2 = ((id_aa64zfr0_el1) & 0xF) >= 1;
    unsigned supports_sve2aes = ((id_aa64zfr0_el1 >> 4) & 0xF) >= 1;

    // The `+crypto` in the NEON and SVE2 AES tiers' targets enables AES and SHA2 together.
    unsigned supports_crypto = supports_neonaes && supports_neonsha;
    return (sz_capability_t)(                                                         //
        (sz_cap_neon_k * (supports_neon)) |                                           //
        (sz_cap_neonaes_k * (supports_crypto)) |                                      //
        (sz_cap_neonsha_k * (supports_crypto)) |                                      //
        (sz_cap_sve_k * (supports_sve)) |                                             //
        (sz_cap_sve2_k * (supports_sve2)) |                                           //
        (sz_cap_sve2aes_k * (supports_sve2 && supports_sve2aes && supports_crypto)) | //
        (sz_cap_serial_k));

#elif STRINGZILLA_OS_WINDOWS_

    // On Windows ARM, use the `IsProcessorFeaturePresent` API for capability detection.
    // https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent
    unsigned supports_neon = IsProcessorFeaturePresent(PF_ARM_V8_INSTRUCTIONS_AVAILABLE);
    unsigned supports_crypto = IsProcessorFeaturePresent(PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE);

    return (sz_capability_t)(                    //
        (sz_cap_neon_k * (supports_neon)) |      //
        (sz_cap_neonaes_k * (supports_crypto)) | //
        (sz_cap_neonsha_k * (supports_crypto)) | //
        (sz_cap_serial_k));

#else
    return sz_capabilities_implied_cpu_();
#endif
}

#endif // STRINGZILLA_ARCH_ARM64_

#if STRINGZILLA_ARCH_X8664_

/** The capabilities of the current x86-64 CPU, as far as the OS saves the registers they use. */
STRINGZILLA_INLINE sz_capability_t sz_capabilities_detected_x8664_(void) {

    /// The states of 4 registers populated for a specific "cpuid" assembly call
    union four_registers_t {
        int array[4];
        struct separate_t {
            unsigned eax, ebx, ecx, edx;
        } named;
    } info0, info1, info7, info_extended;

    // Every x86-64 CPU has leaf 0x80000001, as it reports long mode itself.
#if defined(_MSC_VER) && !defined(__clang__)
    __cpuidex(info0.array, 0, 0);
    __cpuidex(info1.array, 1, 0);
    __cpuidex(info7.array, 7, 0);
    __cpuidex(info_extended.array, 0x80000001, 0);
#else
    __asm__ __volatile__( //
        "cpuid"
        : "=a"(info0.named.eax), "=b"(info0.named.ebx), "=c"(info0.named.ecx), "=d"(info0.named.edx)
        : "a"(0), "c"(0));
    __asm__ __volatile__( //
        "cpuid"
        : "=a"(info1.named.eax), "=b"(info1.named.ebx), "=c"(info1.named.ecx), "=d"(info1.named.edx)
        : "a"(1), "c"(0));
    __asm__ __volatile__( //
        "cpuid"
        : "=a"(info7.named.eax), "=b"(info7.named.ebx), "=c"(info7.named.ecx), "=d"(info7.named.edx)
        : "a"(7), "c"(0));
    __asm__ __volatile__( //
        "cpuid"
        : "=a"(info_extended.named.eax), "=b"(info_extended.named.ebx), "=c"(info_extended.named.ecx),
          "=d"(info_extended.named.edx)
        : "a"(0x80000001), "c"(0));
#endif

    // Querying a leaf above the highest supported one returns the highest leaf's data, not zeros,
    // so on early x86-64 parts (max basic leaf below 7) the "leaf 7" registers would hold unrelated
    // bits. The AVX family is already masked by the XGETBV/OSXSAVE chain below, but SHA-NI and BMI
    // are read from leaf 7 unmasked, so the whole leaf is zeroed when it does not exist.
    if (info0.named.eax < 7) info7.named.eax = info7.named.ebx = info7.named.ecx = info7.named.edx = 0;

    // Gate AVX/AVX-512 on OS-enabled extended state (XGETBV)
    unsigned has_osxsave = (info1.named.ecx & (1u << 27)) != 0; // OSXSAVE
    unsigned has_avx = (info1.named.ecx & (1u << 28)) != 0;     // AVX

    unsigned long long xcr0 = 0;
    if (has_osxsave) {
#if defined(_MSC_VER) && !defined(__clang__)
        xcr0 = _xgetbv(0);
#else
        unsigned eax, edx;
        __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0)); // xgetbv
        xcr0 = ((unsigned long long)edx << 32) | eax;
#endif
    }

    unsigned os_avx_enabled = has_osxsave && has_avx && ((xcr0 & 0x6u) == 0x6u); // XMM+YMM
    unsigned os_avx512_enabled = os_avx_enabled && ((xcr0 & 0xE0u) == 0xE0u);    // OPMASK+ZMM

    // Check for AVX2/AVX-512 (Function ID 7), masked by OS state
    // https://github.com/llvm/llvm-project/blob/50598f0ff44f3a4e75706f8c53f3380fe7faa896/clang/lib/Headers/cpuid.h#L148
    unsigned supports_avx2 = os_avx_enabled && ((info7.named.ebx & 0x00000020u) != 0);
    unsigned supports_fma = os_avx_enabled && ((info1.named.ecx & 0x00001000u) != 0);
    unsigned supports_avx512f = os_avx512_enabled && ((info7.named.ebx & 0x00010000u) != 0);
    unsigned supports_avx512dq = os_avx512_enabled && ((info7.named.ebx & 0x00020000u) != 0);
    unsigned supports_avx512bw = os_avx512_enabled && ((info7.named.ebx & 0x40000000u) != 0);
    unsigned supports_avx512vl = os_avx512_enabled && ((info7.named.ebx & 0x80000000u) != 0);
    unsigned supports_avx512vbmi = os_avx512_enabled && ((info7.named.ecx & 0x00000002u) != 0);
    unsigned supports_avx512vbmi2 = os_avx512_enabled && ((info7.named.ecx & 0x00000040u) != 0);
    unsigned supports_avx512vnni = os_avx512_enabled && ((info7.named.ecx & 0x00000800u) != 0);
    unsigned supports_vaes = os_avx512_enabled && ((info7.named.ecx & 0x00000200u) != 0);
    unsigned supports_vpclmulqdq = os_avx512_enabled && ((info7.named.ecx & 0x00000400u) != 0);

    // SSE, AES-NI, CLMUL, SHA-NI and the bit-manipulation sets, from leaves 1, 7 and 0x80000001
    unsigned supports_sse3 = ((info1.named.ecx & 0x00000001u) != 0);
    unsigned supports_ssse3 = ((info1.named.ecx & 0x00000200u) != 0);
    unsigned supports_sse41 = ((info1.named.ecx & 0x00080000u) != 0);
    unsigned supports_sse42 = ((info1.named.ecx & 0x00100000u) != 0);
    unsigned supports_popcnt = ((info1.named.ecx & 0x00800000u) != 0);
    unsigned supports_aesni = ((info1.named.ecx & 0x02000000u) != 0);
    unsigned supports_pclmul = ((info1.named.ecx & 0x00000002u) != 0);
    unsigned supports_shani = ((info7.named.ebx & 0x20000000u) != 0); // SHA-NI bit 29 in EBX from CPUID(7,0)
    unsigned supports_bmi1 = ((info7.named.ebx & 0x00000008u) != 0);
    unsigned supports_bmi2 = ((info7.named.ebx & 0x00000100u) != 0);
    unsigned supports_lzcnt = ((info_extended.named.ecx & 0x00000020u) != 0);

    // A tier needs every feature its target pragmas and flags enable, even beyond its namesake CPU.
    unsigned supports_westmere = supports_sse42 && supports_aesni && supports_pclmul;
    unsigned supports_goldmont = supports_sse3 && supports_ssse3 && supports_sse41 && supports_shani;
    unsigned supports_haswell = supports_avx2 && supports_fma && supports_bmi1 && supports_bmi2 && supports_popcnt &&
                                supports_lzcnt && supports_pclmul;
    unsigned supports_skylake = supports_avx2 && supports_avx512f && supports_avx512vl && supports_avx512bw &&
                                supports_avx512dq && supports_bmi1 && supports_bmi2 && supports_popcnt &&
                                supports_lzcnt && supports_aesni;
    unsigned supports_icelake = supports_skylake && supports_avx512vbmi && supports_avx512vbmi2 &&
                                supports_avx512vnni && supports_vaes && supports_pclmul && supports_vpclmulqdq &&
                                supports_shani;

    return (sz_capability_t)(                       //
        (sz_cap_westmere_k * (supports_westmere)) | //
        (sz_cap_goldmont_k * (supports_goldmont)) | //
        (sz_cap_haswell_k * (supports_haswell)) |   //
        (sz_cap_skylake_k * (supports_skylake)) |   //
        (sz_cap_icelake_k * (supports_icelake)) |   //
        (sz_cap_serial_k));
}

#endif // STRINGZILLA_ARCH_X8664_

#if STRINGZILLA_ARCH_RISCV64_

/** The capabilities of the current 64-bit RISC-V CPU. */
STRINGZILLA_INLINE sz_capability_t sz_capabilities_detected_riscv64_(void) {
#if STRINGZILLA_OS_LINUX_ && STRINGZILLA_WITH_LIBC

    // The base "V" extension is reported through the auxiliary vector, but the individual
    // vector sub-extensions (vector crypto, bf16, …) are only exposed through the
    // `riscv_hwprobe(2)` syscall, introduced in Linux 6.4. Values as in `asm/hwprobe.h`.
    unsigned long const hwcap_isa_v = 1ul << ('V' - 'A'); // COMPAT_HWCAP_ISA_V, RVV 1.0
    long const riscv_hwprobe = 258;                       // __NR_riscv_hwprobe
    long const hwprobe_extensions_key = 4;                // RISCV_HWPROBE_KEY_IMA_EXT_0
    unsigned long long const hwprobe_zvkg = 1ull << 20;   // RISCV_HWPROBE_EXT_ZVKG, vector GHASH
    unsigned long long const hwprobe_zvkned = 1ull << 21; // RISCV_HWPROBE_EXT_ZVKNED, vector AES
    unsigned long long const hwprobe_zvknhb = 1ull << 23; // RISCV_HWPROBE_EXT_ZVKNHB, vector SHA-2
    unsigned long hwcap = getauxval(AT_HWCAP);
    sz_capability_t caps = sz_cap_serial_k;
    if (hwcap & hwcap_isa_v) {
        caps = (sz_capability_t)(caps | sz_cap_rvv_k);
        struct {
            long long key;
            unsigned long long value;
        } pairs[1];
        pairs[0].key = hwprobe_extensions_key;
        pairs[0].value = 0;
        // `long syscall(SYS_riscv_hwprobe, pairs, pair_count, cpu_count, cpus, flags)`.
        long const probed = syscall(riscv_hwprobe, pairs, (unsigned long)1, (unsigned long)0, (void *)0,
                                    (unsigned long)0);
        // RVA23 sells AES and SHA-2 only inside `Zvkng`, which bundles GHASH, so one capability
        // covers all three.
        if (probed == 0 && (pairs[0].value & hwprobe_zvkned) && (pairs[0].value & hwprobe_zvknhb) &&
            (pairs[0].value & hwprobe_zvkg))
            caps = (sz_capability_t)(caps | sz_cap_rvvcrypto_k);
    }
    return caps;

#elif STRINGZILLA_OS_FREEBSD_ && STRINGZILLA_WITH_LIBC

    // FreeBSD exposes HWCAP through `elf_aux_info`, but lacks the Linux `riscv_hwprobe`
    // syscall, so the vector crypto sub-extensions stay compile-time only here.
    unsigned long const hwcap_isa_v = 1ul << ('V' - 'A'); // COMPAT_HWCAP_ISA_V, RVV 1.0
    unsigned long hwcap = 0;
    elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap));
    sz_capability_t caps = sz_cap_serial_k;
    if (hwcap & hwcap_isa_v) caps = (sz_capability_t)(caps | sz_cap_rvv_k);
    return caps;

#else
    return sz_capabilities_implied_cpu_();
#endif
}

#endif // STRINGZILLA_ARCH_RISCV64_

#if STRINGZILLA_ARCH_LOONGARCH64_

/** The capabilities of the current LoongArch CPU. */
STRINGZILLA_INLINE sz_capability_t sz_capabilities_detected_loongarch64_(void) {
#if STRINGZILLA_OS_LINUX_ && STRINGZILLA_WITH_LIBC

    unsigned long const hwcap_loongarch_lasx = 1ul << 5; // HWCAP_LOONGARCH_LASX, 256-bit SIMD implying LSX
    unsigned long hwcap = getauxval(AT_HWCAP);
    return (sz_capability_t)((sz_cap_loongsonasx_k * ((hwcap & hwcap_loongarch_lasx) != 0)) | sz_cap_serial_k);

#else
    return sz_capabilities_implied_cpu_();
#endif
}

#endif // STRINGZILLA_ARCH_LOONGARCH64_

#if STRINGZILLA_ARCH_PPC64_

/** The capabilities of the current IBM POWER CPU. */
STRINGZILLA_INLINE sz_capability_t sz_capabilities_detected_power64_(void) {
#if (STRINGZILLA_OS_LINUX_ || STRINGZILLA_OS_FREEBSD_) && STRINGZILLA_WITH_LIBC

    // The `powervsx` kernels target POWER9 (`-mcpu=power9`), so both facts are required.
    unsigned long const hwcap_ppc_vsx = 0x00000080ul;        // PPC_FEATURE_HAS_VSX, in AT_HWCAP
    unsigned long const hwcap2_ppc_arch_3_00 = 0x00800000ul; // PPC_FEATURE2_ARCH_3_00, in AT_HWCAP2: POWER9
    unsigned long hwcap = 0, hwcap2 = 0;
#if STRINGZILLA_OS_LINUX_
    hwcap = getauxval(AT_HWCAP);
    hwcap2 = getauxval(AT_HWCAP2);
#else
    elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap));
    elf_aux_info(AT_HWCAP2, &hwcap2, sizeof(hwcap2));
#endif
    unsigned const supports_powervsx = ((hwcap & hwcap_ppc_vsx) != 0) && ((hwcap2 & hwcap2_ppc_arch_3_00) != 0);
    return (sz_capability_t)((sz_cap_powervsx_k * supports_powervsx) | sz_cap_serial_k);

#else
    return sz_capabilities_implied_cpu_();
#endif
}

#endif // STRINGZILLA_ARCH_PPC64_

/** Prepares the calling thread for @p capabilities, behind @c sz_configure_thread_cpu. */
STRINGZILLA_INLINE sz_status_t sz_configure_thread_cpu_(sz_capability_t capabilities) {
    sz_unused_(capabilities);
    return sz_success_k;
}

/** The capabilities of the current CPU, whatever this binary compiled in. */
STRINGZILLA_INLINE sz_capability_t sz_capabilities_detected_cpu_(void) {
#if STRINGZILLA_ARCH_X8664_
    return sz_capabilities_detected_x8664_();
#elif STRINGZILLA_ARCH_ARM64_
    return sz_capabilities_detected_arm64_();
#elif STRINGZILLA_ARCH_RISCV64_
    return sz_capabilities_detected_riscv64_();
#elif STRINGZILLA_ARCH_LOONGARCH64_
    return sz_capabilities_detected_loongarch64_();
#elif STRINGZILLA_ARCH_PPC64_
    return sz_capabilities_detected_power64_();
#else
    return sz_capabilities_implied_cpu_();
#endif
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STRINGZILLA_CPU_H_
