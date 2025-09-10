/**
 *  @brief  StringZilla is a collection of advanced string algorithms, designed to be used in Big Data applications.
 *          It is generally faster than LibC, and has a broader & cleaner interface for safer @b length-bounded strings.
 *          On modern CPUs it uses AVX2, AVX-512, NEON, SVE, & SVE2 @b SIMD instructions & provides SWAR for older CPUs.
 *          On @b CUDA-capable GPUs it also provides C++ kernels for bulk processing.
 *
 *  @file   stringzilla.h
 *  @author Ash Vardanian
 *
 *  @see    StringZilla docs: https://github.com/ashvardanian/StringZilla/blob/main/README.md
 *  @see    LibC string docs: https://pubs.opengroup.org/onlinepubs/009695399/basedefs/string.h.html
 *
 *  @section Introduction
 *
 *  StringZilla is multi-language project designed for high-throughput string processing, differentiating
 *  the low-level "embeddable" mostly-C core implementation, containing:
 *
 *  - `compare.h` - byte-level comparison functions.
 *  - `memory.h` - copying, moving, and filling raw memory.
 *  - `hash.h` - hash functions and checksum algorithms.
 *  - `find.h` - searching for substrings and byte sets.
 *  - `sort.h` - single-threaded sorting algorithms.
 *  - `intersect.h` - intersections of unordered string sets.
 *  - `small_string.h` - "Small String Optimization" in C 99.
 *  - `stringzilla.h` - umbrella header for the core C API.
 *  - `stringzilla.hpp` - umbrella header for the core C++ API.
 *
 *  It also provides many higher-level parallel algorithms, implemented in C++ with Fork Union and CUDA, also exposed
 *  via the stable C 99 ABI, but requiring C++17 and CUDA 17 compilers to build the shared @b StringZillas libraries:
 *
 *  - `similarities.{hpp,cuh}` - similarity measures, like Levenshtein, Needleman-Wunsch, & Smith-Waterman scores.
 *  - `fingerprints.{hpp,cuh}` - feature extraction for TF-IDF and other Machine Learning algorithms.
 *  - `find_many.{hpp,cuh}` - Aho-Corasick multi-pattern search.
 *
 *  The core implementations of those algorithms are mostly structured as callable structure templates, as opposed to
 *  template functions to simplify specialized overloads and reusing the state between invocations.
 *
 *  @section Compilation Settings
 *
 *  Consider overriding the following macros to customize the library:
 *
 *  - `SZ_DEBUG=0` - whether to enable debug assertions and logging.
 *  - `SZ_AVOID_LIBC=0` - whether to avoid including the standard C library headers.
 *  - `SZ_DYNAMIC_DISPATCH=0` - whether to use runtime dispatching of the most advanced SIMD backend.
 *  - `SZ_USE_MISALIGNED_LOADS=0` - whether to use misaligned loads on platforms that support them.
 *
 *  Performance tuning:
 *
 *  - `SZ_SWAR_THRESHOLD=24` - threshold for switching to SWAR backend over serial byte-level for-loops.
 *  - `SZ_CACHE_LINE_WIDTH=64` - cache-line width that affects the execution of some algorithms.
 *  - `SZ_CACHE_SIZE=1048576` - the combined size of L1d and L2 caches in bytes, affecting temporal loads.
 *
 *  Different generations of CPUs and SIMD capabilities can be enabled or disabled with the following macros:
 *
 *  - `SZ_USE_HASWELL=?` - whether to use AVX2 instructions on x86_64.
 *  - `SZ_USE_SKYLAKE=?` - whether to use AVX-512 instructions on x86_64.
 *  - `SZ_USE_ICE=?` - whether to use AVX-512 VBMI & wider AES instructions on x86_64.
 *  - `SZ_USE_NEON=?` - whether to use NEON instructions on ARM.
 *  - `SZ_USE_SVE=?` - whether to use SVE instructions on ARM.
 *  - `SZ_USE_SVE2=?` - whether to use SVE2 instructions on ARM.
 *  - `SZ_USE_CUDA=?` - whether to use minimal CUDA capabilities on Nvidia GPUs.
 *  - `SZ_USE_KEPLER=?` - whether to use Kepler-level instructions on Nvidia GPUs.
 *  - `SZ_USE_HOPPER=?` - whether to use Hopper-level instructions on Nvidia GPUs.
 */
#ifndef STRINGZILLA_H_
#define STRINGZILLA_H_

#define STRINGZILLA_H_VERSION_MAJOR 4
#define STRINGZILLA_H_VERSION_MINOR 0
#define STRINGZILLA_H_VERSION_PATCH 7

#include "types.h"        // `sz_size_t`, `sz_bool_t`, `sz_ordering_t`
#include "compare.h"      // `sz_equal`, `sz_order`
#include "memory.h"       // `sz_copy`, `sz_move`, `sz_fill`
#include "hash.h"         // `sz_bytesum`, `sz_hash`, `sz_state_init`, `sz_state_stream`, `sz_state_fold`
#include "find.h"         // `sz_find`, `sz_find_byteset`, `sz_rfind`
#include "small_string.h" // `sz_string_t`, `sz_string_init`, `sz_string_free`
#include "sort.h"         // `sz_sequence_argsort`, `sz_pgrams_sort`
#include "intersect.h"    // `sz_sequence_intersect`

/* Inferring target OS: Windows, MacOS, or Linux */
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__) || defined(__CYGWIN__)
#define SZ_IS_WINDOWS_ 1
#elif defined(__APPLE__) && defined(__MACH__)
#define SZ_IS_APPLE_ 1
#elif defined(__linux__)
#define SZ_IS_LINUX_ 1
#endif

/* On Apple Silicon, `mrs` is not allowed in user-space, so we need to use the `sysctl` API */
#if defined(SZ_IS_APPLE_)
#include <sys/sysctl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Internal helper function to convert SIMD capabilities to an array of string pointers.
 *  @param[in] caps The capabilities bitfield
 *  @param[out] strings Output array to store string pointers (should have more than `SZ_CAPABILITIES_COUNT` slots)
 *  @param[in] max_count Maximum number of strings to output
 *  @return Number of capability strings written to the array
 *  @sa sz_capabilities_to_string_implementation_, sz_capabilities
 */
SZ_INTERNAL sz_size_t sz_capabilities_to_strings_implementation_(sz_capability_t caps, char const **strings,
                                                                 sz_size_t max_count) {
    // Mapping each flag to its string literal.
    struct {
        sz_capability_t flag;
        char const *name;
    } capability_map[] = {
        //
        {sz_cap_serial_k, "serial"},
        {sz_cap_parallel_k, "parallel"},
        //
        {sz_cap_haswell_k, "haswell"},
        {sz_cap_skylake_k, "skylake"},
        {sz_cap_ice_k, "ice"},
        //
        {sz_cap_neon_k, "neon"},
        {sz_cap_neon_aes_k, "neon+aes"},
        {sz_cap_sve_k, "sve"},
        {sz_cap_sve2_k, "sve2"},
        {sz_cap_sve2_aes_k, "sve2+aes"},
        //
        {sz_cap_cuda_k, "cuda"},
        {sz_cap_kepler_k, "kepler"},
        {sz_cap_hopper_k, "hopper"},
    };
    int const capabilities_count = sizeof(capability_map) / sizeof(capability_map[0]);

    // Iterate over each capability flag.
    sz_size_t count = 0;
    for (int i = 0; i < capabilities_count && count < max_count; i++)
        if (caps & capability_map[i].flag) strings[count++] = capability_map[i].name;

    return count;
}

SZ_INTERNAL sz_bool_t sz_equal_null_terminated_serial(char const *a, char const *b) {
    if (!a || !b) return sz_false_k;
    for (; *a && *b; a++, b++)
        if (*a != *b) return sz_false_k;
    return *b == '\0' ? sz_true_k : sz_false_k;
}

/**
 *  @brief Internal helper to map a capability name to its flag.
 *  @param[in] name Capability name, e.g. "serial", "neon", "sve2_aes".
 *  @return `sz_caps_none_k` if unknown name, or a valid capability flag.
 */
SZ_INTERNAL sz_capability_t sz_capability_from_string_implementation_(char const *name) {

    // CPU + execution model
    if (sz_equal_null_terminated_serial(name, "serial") == sz_true_k) return sz_cap_serial_k;
    if (sz_equal_null_terminated_serial(name, "parallel") == sz_true_k) return sz_cap_parallel_k;
    // x86
    if (sz_equal_null_terminated_serial(name, "haswell") == sz_true_k) return sz_cap_haswell_k;
    if (sz_equal_null_terminated_serial(name, "skylake") == sz_true_k) return sz_cap_skylake_k;
    if (sz_equal_null_terminated_serial(name, "ice") == sz_true_k) return sz_cap_ice_k;
    // Arm
    if (sz_equal_null_terminated_serial(name, "neon") == sz_true_k) return sz_cap_neon_k;
    if (sz_equal_null_terminated_serial(name, "sve") == sz_true_k) return sz_cap_sve_k;
    if (sz_equal_null_terminated_serial(name, "sve2") == sz_true_k) return sz_cap_sve2_k;
    if (sz_equal_null_terminated_serial(name, "neon_aes") == sz_true_k ||
        sz_equal_null_terminated_serial(name, "neon+aes") == sz_true_k)
        return sz_cap_neon_aes_k;
    if (sz_equal_null_terminated_serial(name, "sve2_aes") == sz_true_k ||
        sz_equal_null_terminated_serial(name, "sve2+aes") == sz_true_k)
        return sz_cap_sve2_aes_k;
    // GPU
    if (sz_equal_null_terminated_serial(name, "cuda") == sz_true_k) return sz_cap_cuda_k;
    if (sz_equal_null_terminated_serial(name, "kepler") == sz_true_k) return sz_cap_kepler_k;
    if (sz_equal_null_terminated_serial(name, "hopper") == sz_true_k) return sz_cap_hopper_k;
    // Any
    if (sz_equal_null_terminated_serial(name, "any") == sz_true_k) return sz_cap_any_k;
    return sz_caps_none_k;
}

/**
 *  @brief Internal helper function to convert SIMD capabilities to a string.
 *  @sa    sz_capabilities_to_string, sz_capabilities
 */
SZ_INTERNAL sz_cptr_t sz_capabilities_to_string_implementation_(sz_capability_t caps) {

    static char buf[256];
    char *p = buf;
    char *const end = buf + sizeof(buf);

    // Use the new function to get capability strings
    char const *cap_strings[SZ_CAPABILITIES_COUNT];
    sz_size_t cap_count = sz_capabilities_to_strings_implementation_(caps, cap_strings, SZ_CAPABILITIES_COUNT);

    // Build the comma-separated string
    for (sz_size_t i = 0; i < cap_count; i++) {
        if (i > 0) {
            // Add separator if this is not the first capability.
            char const sep[2] = {',', '\0'};
            char const *s = sep;
            while (*s && p < end - 1) *p++ = *s++;
        }
        // Append the capability name character by character.
        char const *s = cap_strings[i];
        while (*s && p < end - 1) *p++ = *s++;
    }

    // Null-terminate the string.
    *p = '\0';
    return buf;
}

#if SZ_IS_64BIT_ARM_

/*  Compiling the next section one may get: selected processor does not support system register name 'id_aa64zfr0_el1'.
 *  Suppressing assembler errors is very complicated, so when dealing with older ARM CPUs it's simpler to compile this
 *  function targeting newer ones.
 */
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.5-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.5-a+sve")
#endif

/**
 *  @brief  Function to determine the SIMD capabilities of the current 64-bit Arm machine at @b runtime.
 *  @return A bitmask of the SIMD capabilities represented as a `sz_capability_t` enum value.
 */
SZ_PUBLIC sz_capability_t sz_capabilities_implementation_arm_(void) {
    // https://github.com/ashvardanian/SimSIMD/blob/28e536083602f85ad0c59456782c8864463ffb0e/include/simsimd/simsimd.h#L434
    // for documentation on how we detect capabilities across different ARM platforms.
#if defined(SZ_IS_APPLE_)

    // On Apple Silicon, `mrs` is not allowed in user-space, so we need to use the `sysctl` API.
    uint32_t supports_neon = 0;
    uint32_t supports_neon_aes = 0;
    size_t size = sizeof(supports_neon);
    if (sysctlbyname("hw.optional.neon", &supports_neon, &size, NULL, 0) != 0) supports_neon = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_AES", &supports_neon_aes, &size, NULL, 0) != 0) supports_neon_aes = 0;

    return (sz_capability_t)(                       //
        (sz_cap_neon_k * (supports_neon)) |         //
        (sz_cap_neon_aes_k * (supports_neon_aes)) | //
        (sz_cap_serial_k));

#elif defined(SZ_IS_LINUX_)

    // Read CPUID registers directly
    unsigned long id_aa64isar0_el1 = 0, id_aa64isar1_el1 = 0, id_aa64pfr0_el1 = 0, id_aa64zfr0_el1 = 0;
    unsigned supports_neon = 0, supports_neon_aes = 0, supports_sve = 0, supports_sve2 = 0, supports_sve2_aes = 0;
    sz_unused_(id_aa64isar0_el1);
    sz_unused_(id_aa64isar1_el1);
    sz_unused_(id_aa64pfr0_el1);
    sz_unused_(id_aa64zfr0_el1);

#if SZ_USE_NEON || SZ_USE_SVE || SZ_USE_SVE2 || SZ_USE_NEON_AES || SZ_USE_SVE2_AES
    // Now let's unpack the status flags from ID_AA64ISAR0_EL1
    // https://developer.arm.com/documentation/ddi0601/2024-03/AArch64-Registers/ID-AA64ISAR0-EL1--AArch64-Instruction-Set-Attribute-Register-0?lang=en
    __asm__ __volatile__("mrs %0, ID_AA64ISAR0_EL1" : "=r"(id_aa64isar0_el1));
    // Now let's unpack the status flags from ID_AA64ISAR1_EL1
    // https://developer.arm.com/documentation/ddi0601/2024-03/AArch64-Registers/ID-AA64ISAR1-EL1--AArch64-Instruction-Set-Attribute-Register-1?lang=en
    __asm__ __volatile__("mrs %0, ID_AA64ISAR1_EL1" : "=r"(id_aa64isar1_el1));
    // Now let's unpack the status flags from ID_AA64PFR0_EL1
    // https://developer.arm.com/documentation/ddi0601/2024-03/AArch64-Registers/ID-AA64PFR0-EL1--AArch64-Processor-Feature-Register-0?lang=en
    __asm__ __volatile__("mrs %0, ID_AA64PFR0_EL1" : "=r"(id_aa64pfr0_el1));
#endif // SZ_USE_NEON || SZ_USE_SVE || SZ_USE_SVE2

    // AdvSIMD, bits [23:20] of ID_AA64PFR0_EL1 can be used to check for `fp16` support
    //  - 0b0000: integers, single, double precision arithmetic
    //  - 0b0001: includes support for half-precision floating-point arithmetic
    //  - 0b1111: NEON is not supported?!
    // That's a really weird way to encode lack of NEON support, but it's important to
    // check in case we are running on R-profile CPUs.
    supports_neon = ((id_aa64pfr0_el1 >> 20) & 0xF) != 0xF;
    supports_neon_aes = ((id_aa64isar0_el1 >> 4) & 0xF) >= 1;

#if SZ_USE_SVE || SZ_USE_SVE2 || SZ_USE_SVE2_AES
    // SVE, bits [35:32] of ID_AA64PFR0_EL1
    supports_sve = ((id_aa64pfr0_el1 >> 32) & 0xF) >= 1;
    // Now let's unpack the status flags from ID_AA64ZFR0_EL1
    // https://developer.arm.com/documentation/ddi0601/2024-03/AArch64-Registers/ID-AA64ZFR0-EL1--SVE-Feature-ID-Register-0?lang=en
    if (supports_sve) __asm__ __volatile__("mrs %0, ID_AA64ZFR0_EL1" : "=r"(id_aa64zfr0_el1));
    // SVEver, bits [3:0] can be used to check for capability levels:
    //  - 0b0000: SVE is implemented
    //  - 0b0001: SVE2 is implemented
    //  - 0b0010: SVE2.1 is implemented
    // This value must match the existing indicator obtained from ID_AA64PFR0_EL1:
    supports_sve2 = ((id_aa64zfr0_el1) & 0xF) >= 1;
    supports_sve2_aes = ((id_aa64zfr0_el1 >> 4) & 0xF) >= 1;
#endif // SZ_USE_SVE || SZ_USE_SVE2

    return (sz_capability_t)(                       //
        (sz_cap_neon_k * (supports_neon)) |         //
        (sz_cap_neon_aes_k * (supports_neon_aes)) | //
        (sz_cap_sve_k * (supports_sve)) |           //
        (sz_cap_sve2_k * (supports_sve2)) |         //
        (sz_cap_sve2_aes_k * (supports_sve2_aes)) | //
        (sz_cap_serial_k));

#else // if !defined(SZ_IS_APPLE_) && !defined(SZ_IS_LINUX_)
    return sz_cap_serial_k;
#endif
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // SZ_IS_64BIT_ARM_

#if SZ_IS_64BIT_X86_

SZ_PUBLIC sz_capability_t sz_capabilities_implementation_x86_(void) {

#if SZ_USE_HASWELL || SZ_USE_SKYLAKE || SZ_USE_ICE

    /// The states of 4 registers populated for a specific "cpuid" assembly call
    union four_registers_t {
        int array[4];
        struct separate_t {
            unsigned eax, ebx, ecx, edx;
        } named;
    } info1, info7;

#if defined(_MSC_VER)
    __cpuidex(info1.array, 1, 0);
    __cpuidex(info7.array, 7, 0);
#else
    __asm__ __volatile__( //
        "cpuid"
        : "=a"(info1.named.eax), "=b"(info1.named.ebx), "=c"(info1.named.ecx), "=d"(info1.named.edx)
        : "a"(1), "c"(0));
    __asm__ __volatile__( //
        "cpuid"
        : "=a"(info7.named.eax), "=b"(info7.named.ebx), "=c"(info7.named.ecx), "=d"(info7.named.edx)
        : "a"(7), "c"(0));
#endif

    // Gate AVX/AVX-512 on OS-enabled extended state (XGETBV)
    unsigned has_osxsave = (info1.named.ecx & (1u << 27)) != 0; // OSXSAVE
    unsigned has_avx = (info1.named.ecx & (1u << 28)) != 0;     // AVX

    unsigned long long xcr0 = 0;
    if (has_osxsave) {
#if defined(_MSC_VER)
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
    unsigned supports_avx512f = os_avx512_enabled && ((info7.named.ebx & 0x00010000u) != 0);
    unsigned supports_avx512bw = os_avx512_enabled && ((info7.named.ebx & 0x40000000u) != 0);
    unsigned supports_avx512vl = os_avx512_enabled && ((info7.named.ebx & 0x80000000u) != 0);
    unsigned supports_avx512vbmi = os_avx512_enabled && ((info7.named.ecx & 0x00000002u) != 0);
    unsigned supports_avx512vbmi2 = os_avx512_enabled && ((info7.named.ecx & 0x00000040u) != 0);
    unsigned supports_vaes = os_avx512_enabled && ((info7.named.ecx & 0x00000200u) != 0);

    return (sz_capability_t)(                                                                                //
        (sz_cap_haswell_k * supports_avx2) |                                                                 //
        (sz_cap_skylake_k * (supports_avx512f && supports_avx512vl && supports_avx512bw && supports_vaes)) | //
        (sz_cap_ice_k * (supports_avx512vbmi && supports_avx512vbmi2)) |                                     //
        (sz_cap_serial_k));
#else
    return sz_cap_serial_k;
#endif
}
#endif // SZ_IS_64BIT_X86_

/**
 *  @brief Function to determine the SIMD capabilities of the current 64-bit x86 machine at @b runtime.
 *  @return A bitmask of the SIMD capabilities represented as a `sz_capability_t` enum value.
 *  @note Excludes parallel-processing & GPGPU capabilities, which are detected separately in StringZillas.
 */
SZ_PUBLIC sz_capability_t sz_capabilities_implementation_(void) {
#if SZ_IS_64BIT_X86_
    return sz_capabilities_implementation_x86_();
#elif SZ_IS_64BIT_ARM_
    return sz_capabilities_implementation_arm_();
#else
    return sz_cap_serial_k;
#endif
}

#if SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC int sz_dynamic_dispatch(void);
SZ_DYNAMIC int sz_version_major(void);
SZ_DYNAMIC int sz_version_minor(void);
SZ_DYNAMIC int sz_version_patch(void);
SZ_DYNAMIC sz_capability_t sz_capabilities(void);
SZ_DYNAMIC sz_cptr_t sz_capabilities_to_string(sz_capability_t caps);
SZ_DYNAMIC void sz_dispatch_table_update(sz_capability_t caps);

#else

SZ_DYNAMIC int sz_dynamic_dispatch(void) { return 0; }
SZ_PUBLIC int sz_version_major(void) { return STRINGZILLA_H_VERSION_MAJOR; }
SZ_PUBLIC int sz_version_minor(void) { return STRINGZILLA_H_VERSION_MINOR; }
SZ_PUBLIC int sz_version_patch(void) { return STRINGZILLA_H_VERSION_PATCH; }
SZ_PUBLIC sz_capability_t sz_capabilities(void) { return sz_capabilities_implementation_(); }
SZ_PUBLIC sz_cptr_t sz_capabilities_to_string(sz_capability_t caps) {
    return sz_capabilities_to_string_implementation_(caps);
}
SZ_PUBLIC void sz_dispatch_table_update(sz_capability_t caps) { sz_unused_(caps); } // No-op in non-dynamic builds

#endif

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // STRINGZILLA_H_
