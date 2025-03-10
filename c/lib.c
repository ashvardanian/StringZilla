/**
 *  @file       lib.c
 *  @brief      StringZilla C library with dynamic backed dispatch for the most appropriate implementation.
 *  @author     Ash Vardanian
 *  @date       January 16, 2024
 */
#if SZ_AVOID_LIBC
// If we don't have the LibC, the `malloc` definition in `stringzilla.h` will be illformed.
#ifdef _MSC_VER
typedef sz_size_t size_t; // Reuse the type definition we've inferred from `stringzilla.h`
extern __declspec(dllimport) int rand(void);
extern __declspec(dllimport) void free(void *start);
extern __declspec(dllimport) void *malloc(size_t length);
#else
typedef __SIZE_TYPE__ size_t; // For GCC/Clang
extern int rand(void);
extern void free(void *start);
extern void *malloc(size_t length);
#endif
#endif

// When enabled, this library will override the symbols usually provided by the C standard library.
// It's handy if you want to use the `LD_PRELOAD` trick for non-intrusive profiling and replacing
// the C standard library implementation without recompiling.
#if !defined(SZ_OVERRIDE_LIBC)
#define SZ_OVERRIDE_LIBC SZ_AVOID_LIBC
#endif

// Overwrite `SZ_DYNAMIC_DISPATCH` before including StringZilla.
#ifdef SZ_DYNAMIC_DISPATCH
#undef SZ_DYNAMIC_DISPATCH
#endif
#define SZ_DYNAMIC_DISPATCH 1
#include <stringzilla/stringzilla.h>

// Inferring target OS: Windows, MacOS, or Linux
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__) || defined(__CYGWIN__)
#define _SZ_IS_WINDOWS 1
#elif defined(__APPLE__) && defined(__MACH__)
#define _SZ_IS_APPLE 1
#elif defined(__linux__)
#define _SZ_IS_LINUX 1
#endif

// On Apple Silicon, `mrs` is not allowed in user-space, so we need to use the `sysctl` API.
#if defined(_SZ_IS_APPLE)
#include <sys/sysctl.h>
#endif

#if defined(_SZ_IS_WINDOWS)
#include <windows.h> // `DllMain`
#endif

#if _SZ_IS_ARM64

/**
 *  @brief  Function to determine the SIMD capabilities of the current 64-bit Arm machine at @b runtime.
 *  @return A bitmask of the SIMD capabilities represented as a `sz_capability_t` enum value.
 */
SZ_INTERNAL sz_capability_t _sz_capabilities_arm(void) {
    // https://github.com/ashvardanian/SimSIMD/blob/28e536083602f85ad0c59456782c8864463ffb0e/include/simsimd/simsimd.h#L434
    // for documentation on how we detect capabilities across different ARM platforms.
#if defined(_SZ_IS_APPLE)

    // On Apple Silicon, `mrs` is not allowed in user-space, so we need to use the `sysctl` API.
    uint32_t supports_neon = 0;
    size_t size = sizeof(supports_neon);
    if (sysctlbyname("hw.optional.neon", &supports_neon, &size, NULL, 0) != 0) supports_neon = 0;

    return (sz_capability_t)(               //
        (sz_cap_neon_k * (supports_neon)) | //
        (sz_cap_serial_k));

#elif defined(_SZ_IS_LINUX)

    // Read CPUID registers directly
    unsigned long id_aa64isar0_el1 = 0, id_aa64isar1_el1 = 0, id_aa64pfr0_el1 = 0, id_aa64zfr0_el1 = 0;

    // Now let's unpack the status flags from ID_AA64ISAR0_EL1
    // https://developer.arm.com/documentation/ddi0601/2024-03/AArch64-Registers/ID-AA64ISAR0-EL1--AArch64-Instruction-Set-Attribute-Register-0?lang=en
    __asm__ __volatile__("mrs %0, ID_AA64ISAR0_EL1" : "=r"(id_aa64isar0_el1));
    // Now let's unpack the status flags from ID_AA64ISAR1_EL1
    // https://developer.arm.com/documentation/ddi0601/2024-03/AArch64-Registers/ID-AA64ISAR1-EL1--AArch64-Instruction-Set-Attribute-Register-1?lang=en
    __asm__ __volatile__("mrs %0, ID_AA64ISAR1_EL1" : "=r"(id_aa64isar1_el1));
    // Now let's unpack the status flags from ID_AA64PFR0_EL1
    // https://developer.arm.com/documentation/ddi0601/2024-03/AArch64-Registers/ID-AA64PFR0-EL1--AArch64-Processor-Feature-Register-0?lang=en
    __asm__ __volatile__("mrs %0, ID_AA64PFR0_EL1" : "=r"(id_aa64pfr0_el1));
    // SVE, bits [35:32] of ID_AA64PFR0_EL1
    unsigned supports_sve = ((id_aa64pfr0_el1 >> 32) & 0xF) >= 1;
    // Now let's unpack the status flags from ID_AA64ZFR0_EL1
    // https://developer.arm.com/documentation/ddi0601/2024-03/AArch64-Registers/ID-AA64ZFR0-EL1--SVE-Feature-ID-Register-0?lang=en
    if (supports_sve) __asm__ __volatile__("mrs %0, ID_AA64ZFR0_EL1" : "=r"(id_aa64zfr0_el1));
    // SVEver, bits [3:0] can be used to check for capability levels:
    //  - 0b0000: SVE is implemented
    //  - 0b0001: SVE2 is implemented
    //  - 0b0010: SVE2.1 is implemented
    // This value must match the existing indicator obtained from ID_AA64PFR0_EL1:
    unsigned supports_sve2 = ((id_aa64zfr0_el1) & 0xF) >= 1;
    unsigned supports_neon = 1; // NEON is always supported

    return (sz_capability_t)(               //
        (sz_cap_neon_k * (supports_neon)) | //
        (sz_cap_sve_k * (supports_sve)) |   //
        (sz_cap_sve2_k * (supports_sve2)) | //
        (sz_cap_serial_k));

#else // if !defined(_SZ_IS_APPLE) && !defined(_SZ_IS_LINUX)
    return sz_cap_serial_k;
#endif
}

#endif // _SZ_IS_ARM64

#if _SZ_IS_X86_64

SZ_INTERNAL sz_capability_t _sz_capabilities_x86(void) {

#if SZ_USE_HASWELL || SZ_USE_SKYLAKE || SZ_USE_ICE

    /// The states of 4 registers populated for a specific "cpuid" assembly call
    union four_registers_t {
        int array[4];
        struct separate_t {
            unsigned eax, ebx, ecx, edx;
        } named;
    } info1, info7;

#ifdef _MSC_VER
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

    // Check for AVX2 (Function ID 7, EBX register), you can take the relevant flags from the LLVM implementation:
    // https://github.com/llvm/llvm-project/blob/50598f0ff44f3a4e75706f8c53f3380fe7faa896/clang/lib/Headers/cpuid.h#L148
    unsigned supports_avx2 = (info7.named.ebx & 0x00000020) != 0;
    unsigned supports_avx512f = (info7.named.ebx & 0x00010000) != 0;
    unsigned supports_avx512bw = (info7.named.ebx & 0x40000000) != 0;
    unsigned supports_avx512vl = (info7.named.ebx & 0x80000000) != 0;
    unsigned supports_avx512vbmi = (info7.named.ecx & 0x00000002) != 0;
    unsigned supports_avx512vbmi2 = (info7.named.ecx & 0x00000040) != 0;
    unsigned supports_vaes = (info7.named.ecx & 0x00000200) != 0;

    return (sz_capability_t)(                                                                                //
        (sz_cap_haswell_k * supports_avx2) |                                                                 //
        (sz_cap_skylake_k * (supports_avx512f && supports_avx512vl && supports_avx512bw && supports_vaes)) | //
        (sz_cap_ice_k * (supports_avx512vbmi && supports_avx512vbmi2)) |                                     //
        (sz_cap_serial_k));
#else
    return sz_cap_serial_k;
#endif
}
#endif // _SZ_IS_X86_64

/**
 *  @brief  Function to determine the SIMD capabilities of the current 64-bit x86 machine at @b runtime.
 *  @return A bitmask of the SIMD capabilities represented as a `sz_capability_t` enum value.
 */
SZ_DYNAMIC sz_capability_t sz_capabilities(void) {
#if _SZ_IS_X86_64
    return _sz_capabilities_x86();
#elif _SZ_IS_ARM64
    return _sz_capabilities_arm();
#else
    return sz_cap_serial_k;
#endif
}

typedef struct sz_implementations_t {
    sz_equal_t equal;
    sz_order_t order;

    sz_move_t copy;
    sz_move_t move;
    sz_fill_t fill;
    sz_lookup_t lookup;

    sz_bytesum_t bytesum;
    sz_hash_t hash;
    sz_hash_state_init_t hash_state_init;
    sz_hash_state_stream_t hash_state_stream;
    sz_hash_state_fold_t hash_state_fold;
    sz_fill_random_t fill_random;

    sz_find_byte_t find_byte;
    sz_find_byte_t rfind_byte;
    sz_find_t find;
    sz_find_t rfind;
    sz_find_set_t find_from_set;
    sz_find_set_t rfind_from_set;

    sz_levenshtein_distance_t edit_distance;
    sz_needleman_wunsch_score_t alignment_score;

    sz_sequence_argsort_t sequence_argsort;
    sz_sequence_intersect_t sequence_intersect;
    sz_pgrams_sort_t pgrams_sort;

} sz_implementations_t;

#if defined(_MSC_VER)
__declspec(align(64)) static sz_implementations_t sz_dispatch_table;
#else
__attribute__((aligned(64))) static sz_implementations_t sz_dispatch_table;
#endif

/**
 *  @brief  Initializes a global static "virtual table" of supported backends
 *          Run it just once to avoiding unnecessary `if`-s.
 */
SZ_DYNAMIC void sz_dispatch_table_init(void) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_capability_t caps = sz_capabilities();
    sz_unused(caps); //< Unused when compiling on pre-SIMD machines.

    impl->equal = sz_equal_serial;
    impl->order = sz_order_serial;
    impl->copy = sz_copy_serial;
    impl->move = sz_move_serial;
    impl->fill = sz_fill_serial;
    impl->lookup = sz_lookup_serial;

    impl->bytesum = sz_bytesum_serial;
    impl->hash = sz_hash_serial;
    impl->hash_state_init = sz_hash_state_init_serial;
    impl->hash_state_stream = sz_hash_state_stream_serial;
    impl->hash_state_fold = sz_hash_state_fold_serial;
    impl->fill_random = sz_fill_random_serial;

    impl->find = sz_find_serial;
    impl->rfind = sz_rfind_serial;
    impl->find_byte = sz_find_byte_serial;
    impl->rfind_byte = sz_rfind_byte_serial;
    impl->find_from_set = sz_find_byteset_serial;
    impl->rfind_from_set = sz_rfind_byteset_serial;

    impl->edit_distance = sz_levenshtein_distance_serial;
    impl->alignment_score = sz_needleman_wunsch_score_serial;

    impl->sequence_argsort = sz_sequence_argsort_serial;
    impl->sequence_intersect = sz_sequence_intersect_serial;
    impl->pgrams_sort = sz_pgrams_sort_serial;

#if SZ_USE_HASWELL
    if (caps & sz_cap_haswell_k) {
        impl->equal = sz_equal_haswell;
        impl->order = sz_order_haswell;

        impl->copy = sz_copy_haswell;
        impl->move = sz_move_haswell;
        impl->fill = sz_fill_haswell;
        impl->lookup = sz_lookup_haswell;

        impl->bytesum = sz_bytesum_haswell;
        impl->hash = sz_hash_haswell;
        impl->hash_state_init = sz_hash_state_init_haswell;
        impl->hash_state_stream = sz_hash_state_stream_haswell;
        impl->hash_state_fold = sz_hash_state_fold_haswell;
        impl->fill_random = sz_fill_random_haswell;

        impl->find_byte = sz_find_byte_haswell;
        impl->rfind_byte = sz_rfind_byte_haswell;
        impl->find = sz_find_haswell;
        impl->rfind = sz_rfind_haswell;
        impl->find_from_set = sz_find_byteset_haswell;
        impl->rfind_from_set = sz_rfind_byteset_haswell;
    }
#endif

#if SZ_USE_SKYLAKE
    if (caps & sz_cap_skylake_k) {
        impl->equal = sz_equal_skylake;
        impl->order = sz_order_skylake;

        impl->copy = sz_copy_skylake;
        impl->move = sz_move_skylake;
        impl->fill = sz_fill_skylake;

        impl->bytesum = sz_bytesum_skylake;
        impl->hash = sz_hash_skylake;
        impl->hash_state_init = sz_hash_state_init_skylake;
        impl->hash_state_stream = sz_hash_state_stream_skylake;
        impl->hash_state_fold = sz_hash_state_fold_skylake;
        impl->fill_random = sz_fill_random_skylake;

        impl->find = sz_find_skylake;
        impl->rfind = sz_rfind_skylake;
        impl->find_byte = sz_find_byte_skylake;
        impl->rfind_byte = sz_rfind_byte_skylake;
        impl->bytesum = sz_bytesum_skylake;

        impl->sequence_argsort = sz_sequence_argsort_skylake;
        impl->pgrams_sort = sz_pgrams_sort_skylake;
    }
#endif

#if SZ_USE_ICE
    if (caps & sz_cap_ice_k) {
        impl->find_from_set = sz_find_byteset_ice;
        impl->rfind_from_set = sz_rfind_byteset_ice;

        impl->edit_distance = sz_levenshtein_distance_ice;
        impl->alignment_score = sz_needleman_wunsch_score_ice;

        impl->lookup = sz_lookup_ice;

        impl->bytesum = sz_bytesum_ice;
        impl->hash = sz_hash_ice;
        impl->hash_state_init = sz_hash_state_init_ice;
        impl->hash_state_stream = sz_hash_state_stream_ice;
        impl->hash_state_fold = sz_hash_state_fold_ice;
        impl->fill_random = sz_fill_random_ice;

        impl->sequence_intersect = sz_sequence_intersect_ice;
    }
#endif

#if SZ_USE_NEON
    if (caps & sz_cap_neon_k) {
        impl->equal = sz_equal_neon;

        impl->copy = sz_copy_neon;
        impl->move = sz_move_neon;
        impl->fill = sz_fill_neon;
        impl->lookup = sz_lookup_neon;

        impl->bytesum = sz_bytesum_neon;
        impl->hash = sz_hash_neon;
        impl->hash_state_init = sz_hash_state_init_neon;
        impl->hash_state_stream = sz_hash_state_stream_neon;
        impl->hash_state_fold = sz_hash_state_fold_neon;
        impl->fill_random = sz_fill_random_neon;

        impl->find = sz_find_neon;
        impl->rfind = sz_rfind_neon;
        impl->find_byte = sz_find_byte_neon;
        impl->rfind_byte = sz_rfind_byte_neon;
        impl->find_from_set = sz_find_byteset_neon;
        impl->rfind_from_set = sz_rfind_byteset_neon;
    }
#endif

#if SZ_USE_SVE
    if (caps & sz_cap_sve_k) {
        impl->sequence_argsort = sz_sequence_argsort_sve;
        impl->sequence_intersect = sz_sequence_intersect_sve;
        impl->pgrams_sort = sz_pgrams_sort_sve;
    }
#endif
}

#if defined(_MSC_VER)
/*
 *  Makes sure the `sz_dispatch_table_init` function is called at startup, from either an executable or when loading
 *  a DLL. The section name must be no more than 8 characters long, and must be between .CRT$XCA and .CRT$XCZ
 *  alphabetically (exclusive). The Microsoft C++ compiler puts C++ initialisation code in .CRT$XCU, so avoid that
 *  section: https://learn.microsoft.com/en-us/cpp/c-runtime-library/crt-initialization?view=msvc-170
 */
#pragma comment(linker, "/INCLUDE:_sz_dispatch_table_init")
#pragma section(".CRT$XCS", read)
__declspec(allocate(".CRT$XCS")) void (*_sz_dispatch_table_init)() = sz_dispatch_table_init;

/*  Called either from CRT code or out own `_DLLMainCRTStartup`, when a DLL is loaded. */
BOOL WINAPI DllMain(HINSTANCE hints, DWORD forward_reason, LPVOID lp) {
    switch (forward_reason) {
    case DLL_PROCESS_ATTACH:
        sz_dispatch_table_init(); // Ensure initialization
        return TRUE;
    case DLL_THREAD_ATTACH: return TRUE;
    case DLL_THREAD_DETACH: return TRUE;
    case DLL_PROCESS_DETACH: return TRUE;
    }
    return TRUE;
}

#if SZ_AVOID_LIBC
/*  Called when the DLL is loaded, and ther is no CRT code. */
BOOL WINAPI _DllMainCRTStartup(HINSTANCE hints, DWORD forward_reason, LPVOID lp) {
    DllMain(hints, forward_reason, lp);
    return TRUE;
}
#endif

#else
__attribute__((constructor)) static void sz_dispatch_table_init_on_gcc_or_clang(void) { sz_dispatch_table_init(); }
#endif

SZ_DYNAMIC int sz_dynamic_dispatch(void) { return 1; }
SZ_DYNAMIC int sz_version_major(void) { return STRINGZILLA_H_VERSION_MAJOR; }
SZ_DYNAMIC int sz_version_minor(void) { return STRINGZILLA_H_VERSION_MINOR; }
SZ_DYNAMIC int sz_version_patch(void) { return STRINGZILLA_H_VERSION_PATCH; }
SZ_DYNAMIC sz_cptr_t sz_capabilities_to_string(sz_capability_t caps) {
    return _sz_capabilities_to_string_implementation(caps);
}

SZ_DYNAMIC sz_u64_t sz_bytesum(sz_cptr_t text, sz_size_t length) { return sz_dispatch_table.bytesum(text, length); }

SZ_DYNAMIC sz_u64_t sz_hash(sz_cptr_t text, sz_size_t length, sz_u64_t seed) {
    return sz_dispatch_table.hash(text, length, seed);
}

SZ_DYNAMIC void sz_hash_state_init(sz_hash_state_t *state, sz_u64_t seed) {
    sz_dispatch_table.hash_state_init(state, seed);
}

SZ_DYNAMIC void sz_hash_state_stream(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_dispatch_table.hash_state_stream(state, text, length);
}

SZ_DYNAMIC sz_u64_t sz_hash_state_fold(sz_hash_state_t const *state) {
    return sz_dispatch_table.hash_state_fold(state);
}

SZ_DYNAMIC void sz_fill_random(sz_ptr_t result, sz_size_t result_length, sz_u64_t nonce) {
    sz_dispatch_table.fill_random(result, result_length, nonce);
}

SZ_DYNAMIC sz_bool_t sz_equal(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    return sz_dispatch_table.equal(a, b, length);
}

SZ_DYNAMIC sz_ordering_t sz_order(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    return sz_dispatch_table.order(a, a_length, b, b_length);
}

SZ_DYNAMIC void sz_copy(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    sz_dispatch_table.copy(target, source, length);
}

SZ_DYNAMIC void sz_move(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    sz_dispatch_table.move(target, source, length);
}

SZ_DYNAMIC void sz_fill(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    sz_dispatch_table.fill(target, length, value);
}

SZ_DYNAMIC void sz_lookup(sz_ptr_t target, sz_size_t length, sz_cptr_t source, sz_cptr_t lut) {
    sz_dispatch_table.lookup(target, length, source, lut);
}

SZ_DYNAMIC sz_cptr_t sz_find_byte(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle) {
    return sz_dispatch_table.find_byte(haystack, h_length, needle);
}

SZ_DYNAMIC sz_cptr_t sz_rfind_byte(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle) {
    return sz_dispatch_table.rfind_byte(haystack, h_length, needle);
}

SZ_DYNAMIC sz_cptr_t sz_find(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length) {
    return sz_dispatch_table.find(haystack, h_length, needle, n_length);
}

SZ_DYNAMIC sz_cptr_t sz_rfind(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length) {
    return sz_dispatch_table.rfind(haystack, h_length, needle, n_length);
}

SZ_DYNAMIC sz_cptr_t sz_find_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
    return sz_dispatch_table.find_from_set(text, length, set);
}

SZ_DYNAMIC sz_cptr_t sz_rfind_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
    return sz_dispatch_table.rfind_from_set(text, length, set);
}

SZ_DYNAMIC sz_status_t sz_hamming_distance( //
    sz_cptr_t a, sz_size_t a_length,        //
    sz_cptr_t b, sz_size_t b_length,        //
    sz_size_t bound, sz_size_t *result) {
    return sz_hamming_distance_serial(a, a_length, b, b_length, bound, result);
}

SZ_DYNAMIC sz_status_t sz_hamming_distance_utf8( //
    sz_cptr_t a, sz_size_t a_length,             //
    sz_cptr_t b, sz_size_t b_length,             //
    sz_size_t bound, sz_size_t *result) {
    return sz_hamming_distance_utf8_serial(a, a_length, b, b_length, bound, result);
}

SZ_DYNAMIC sz_status_t sz_levenshtein_distance( //
    sz_cptr_t a, sz_size_t a_length,            //
    sz_cptr_t b, sz_size_t b_length,            //
    sz_size_t bound, sz_memory_allocator_t *alloc, sz_size_t *result) {
    return sz_dispatch_table.edit_distance(a, a_length, b, b_length, bound, alloc, result);
}

SZ_DYNAMIC sz_status_t sz_levenshtein_distance_utf8( //
    sz_cptr_t a, sz_size_t a_length,                 //
    sz_cptr_t b, sz_size_t b_length,                 //
    sz_size_t bound, sz_memory_allocator_t *alloc, sz_size_t *result) {
    return _sz_levenshtein_distance_wagner_fisher_serial(a, a_length, b, b_length, bound, sz_true_k, alloc, result);
}

SZ_DYNAMIC sz_status_t sz_needleman_wunsch_score( //
    sz_cptr_t a, sz_size_t a_length,              //
    sz_cptr_t b, sz_size_t b_length,              //
    sz_error_cost_t const *subs, sz_error_cost_t gap, sz_memory_allocator_t *alloc, sz_ssize_t *result) {
    return sz_dispatch_table.alignment_score(a, a_length, b, b_length, subs, gap, alloc, result);
}

SZ_DYNAMIC sz_status_t sz_pgrams_sort(sz_pgram_t *array, sz_size_t count, sz_memory_allocator_t *alloc,
                                      sz_size_t *order) {
    return sz_dispatch_table.pgrams_sort(array, count, alloc, order);
}

SZ_DYNAMIC sz_status_t sz_sequence_argsort(sz_sequence_t const *array, sz_memory_allocator_t *alloc, sz_size_t *order) {
    return sz_dispatch_table.sequence_argsort(array, alloc, order);
}

SZ_DYNAMIC sz_status_t sz_sequence_intersect(sz_sequence_t const *first_array, sz_sequence_t const *second_array,
                                             sz_memory_allocator_t *alloc, sz_u64_t seed, sz_size_t *intersection_size,
                                             sz_size_t *first_positions, sz_size_t *second_positions) {
    return sz_dispatch_table.sequence_intersect(first_array, second_array, alloc, seed, intersection_size,
                                                first_positions, second_positions);
}

// Provide overrides for the libc mem* functions
#if SZ_OVERRIDE_LIBC && !defined(__CYGWIN__)

// SZ_DYNAMIC can't be use here for MSVC, because MSVC complains about different linkage (C2375), probably due
// to to the CRT headers specifying the function as `__declspec(dllimport)`, there might be a combination of
// defines that works. But for now they will be manually exported using linker flags.
// Also when building for 32-bit we must add an underscore to the exported function name, because that's
// how `__cdecl` functions are decorated in MSVC: https://stackoverflow.com/questions/62753691)

#if defined(_MSC_VER)
#if SZ_DETECT_64_BIT
#pragma comment(linker, "/export:memchr")
#else
#pragma comment(linker, "/export:_memchr")
#endif
void *__cdecl memchr(void const *s, int c_wide, size_t n) {
#else
SZ_DYNAMIC void *memchr(void const *s, int c_wide, size_t n) {
#endif
    sz_u8_t c = (sz_u8_t)c_wide;
    return (void *)sz_find_byte(s, n, (sz_cptr_t)&c);
}

#if defined(_MSC_VER)
#if SZ_DETECT_64_BIT
#pragma comment(linker, "/export:memcpy")
#else
#pragma comment(linker, "/export:_memcpy")
#endif
void *__cdecl memcpy(void *dest, void const *src, size_t n) {
#else
SZ_DYNAMIC void *memcpy(void *dest, void const *src, size_t n) {
#endif
    sz_copy(dest, src, n);
    return (void *)dest;
}

#if defined(_MSC_VER)
#if SZ_DETECT_64_BIT
#pragma comment(linker, "/export:memmove")
#else
#pragma comment(linker, "/export:_memmove")
#endif
void *__cdecl memmove(void *dest, void const *src, size_t n) {
#else
SZ_DYNAMIC void *memmove(void *dest, void const *src, size_t n) {
#endif
    sz_move(dest, src, n);
    return (void *)dest;
}

#if defined(_MSC_VER)
#if SZ_DETECT_64_BIT
#pragma comment(linker, "/export:memset")
#else
#pragma comment(linker, "/export:_memset")
#endif
void *__cdecl memset(void *s, int c, size_t n) {
#else
SZ_DYNAMIC void *memset(void *s, int c, size_t n) {
#endif
    sz_fill(s, n, c);
    return (void *)s;
}

#if !defined(_MSC_VER)
SZ_DYNAMIC void *memmem(void const *h, size_t h_len, void const *n, size_t n_len) {
    return (void *)sz_find(h, h_len, n, n_len);
}

SZ_DYNAMIC void *memrchr(void const *s, int c_wide, size_t n) {
    sz_u8_t c = (sz_u8_t)c_wide;
    return (void *)sz_rfind_byte(s, n, (sz_cptr_t)&c);
}

SZ_DYNAMIC void memfrob(void *s, size_t n) {
    static sz_u64_t nonce = 42;
    sz_fill_random(s, n, nonce++);
}

#endif
#endif // SZ_OVERRIDE_LIBC
