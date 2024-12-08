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

    return (sz_capability_t)(                   //
        (sz_cap_arm_neon_k * (supports_neon)) | //
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
    unsigned supports_sve2p1 = ((id_aa64zfr0_el1) & 0xF) >= 2;
    unsigned supports_neon = 1; // NEON is always supported

    return (sz_capability_t)(                   //
        (sz_cap_neon_k * (supports_neon)) |     //
        (sz_cap_sve_k * (supports_sve)) |       //
        (sz_cap_sve2_k * (supports_sve2)) |     //
        (sz_cap_sve2p1_k * (supports_sve2p1)) | //
        (sz_cap_serial_k));

#else // if !defined(_SZ_IS_APPLE) && !defined(_SZ_IS_LINUX)
    return sz_cap_serial_k;
#endif
}

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

/**
 *  @brief  Function to determine the SIMD capabilities of the current 64-bit x86 machine at @b runtime.
 *  @return A bitmask of the SIMD capabilities represented as a `sz_capability_t` enum value.
 */
SZ_DYNAMIC sz_capability_t sz_capabilities(void) {
#if _SZ_IS_X86
    return _sz_capabilities_x86();
#elif _SZ_IS_ARM
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
    sz_look_up_transform_t look_up_transform;
    sz_checksum_t checksum;

    sz_find_byte_t find_byte;
    sz_find_byte_t rfind_byte;
    sz_find_t find;
    sz_find_t rfind;
    sz_find_set_t find_from_set;
    sz_find_set_t rfind_from_set;

    sz_edit_distance_t edit_distance;
    sz_alignment_score_t alignment_score;
    sz_hashes_t hashes;

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
    impl->look_up_transform = sz_look_up_transform_serial;
    impl->checksum = sz_checksum_serial;

    impl->find = sz_find_serial;
    impl->rfind = sz_rfind_serial;
    impl->find_byte = sz_find_byte_serial;
    impl->rfind_byte = sz_rfind_byte_serial;
    impl->find_from_set = sz_find_charset_serial;
    impl->rfind_from_set = sz_rfind_charset_serial;

    impl->edit_distance = sz_edit_distance_serial;
    impl->alignment_score = sz_alignment_score_serial;
    impl->hashes = sz_hashes_serial;

#if SZ_USE_HASWELL
    if (caps & sz_cap_haswell_k) {
        impl->equal = sz_equal_haswell;
        impl->order = sz_order_haswell;

        impl->copy = sz_copy_haswell;
        impl->move = sz_move_haswell;
        impl->fill = sz_fill_haswell;
        impl->look_up_transform = sz_look_up_transform_haswell;
        impl->checksum = sz_checksum_haswell;

        impl->find_byte = sz_find_byte_haswell;
        impl->rfind_byte = sz_rfind_byte_haswell;
        impl->find = sz_find_haswell;
        impl->rfind = sz_rfind_haswell;
        impl->find_from_set = sz_find_charset_haswell;
        impl->rfind_from_set = sz_rfind_charset_haswell;
    }
#endif

#if SZ_USE_SKYLAKE
    if (caps & sz_cap_skylake_k) {
        impl->equal = sz_equal_skylake;
        impl->order = sz_order_skylake;

        impl->copy = sz_copy_skylake;
        impl->move = sz_move_skylake;
        impl->fill = sz_fill_skylake;

        impl->find = sz_find_skylake;
        impl->rfind = sz_rfind_skylake;
        impl->find_byte = sz_find_byte_skylake;
        impl->rfind_byte = sz_rfind_byte_skylake;
    }
#endif

#if SZ_USE_ICE
    if (caps & sz_cap_ice_k) {
        impl->find_from_set = sz_find_charset_ice;
        impl->rfind_from_set = sz_rfind_charset_ice;
        impl->edit_distance = sz_edit_distance_ice;
        impl->alignment_score = sz_alignment_score_ice;
        impl->look_up_transform = sz_look_up_transform_ice;
        impl->checksum = sz_checksum_ice;
    }
#endif

#if SZ_USE_NEON
    if (caps & sz_cap_neon_k) {
        impl->equal = sz_equal_neon;

        impl->copy = sz_copy_neon;
        impl->move = sz_move_neon;
        impl->fill = sz_fill_neon;
        impl->look_up_transform = sz_look_up_transform_neon;
        impl->checksum = sz_checksum_neon;

        impl->find = sz_find_neon;
        impl->rfind = sz_rfind_neon;
        impl->find_byte = sz_find_byte_neon;
        impl->rfind_byte = sz_rfind_byte_neon;
        impl->find_from_set = sz_find_charset_neon;
        impl->rfind_from_set = sz_rfind_charset_neon;
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

SZ_DYNAMIC sz_u64_t sz_checksum(sz_cptr_t text, sz_size_t length) { return sz_dispatch_table.checksum(text, length); }

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

SZ_DYNAMIC void sz_look_up_transform(sz_cptr_t source, sz_size_t length, sz_cptr_t lut, sz_ptr_t target) {
    sz_dispatch_table.look_up_transform(source, length, lut, target);
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

SZ_DYNAMIC sz_cptr_t sz_find_charset(sz_cptr_t text, sz_size_t length, sz_charset_t const *set) {
    return sz_dispatch_table.find_from_set(text, length, set);
}

SZ_DYNAMIC sz_cptr_t sz_rfind_charset(sz_cptr_t text, sz_size_t length, sz_charset_t const *set) {
    return sz_dispatch_table.rfind_from_set(text, length, set);
}

SZ_DYNAMIC sz_size_t sz_hamming_distance( //
    sz_cptr_t a, sz_size_t a_length,      //
    sz_cptr_t b, sz_size_t b_length,      //
    sz_size_t bound) {
    return sz_hamming_distance_serial(a, a_length, b, b_length, bound);
}

SZ_DYNAMIC sz_size_t sz_hamming_distance_utf8( //
    sz_cptr_t a, sz_size_t a_length,           //
    sz_cptr_t b, sz_size_t b_length,           //
    sz_size_t bound) {
    return sz_hamming_distance_utf8_serial(a, a_length, b, b_length, bound);
}

SZ_DYNAMIC sz_size_t sz_edit_distance( //
    sz_cptr_t a, sz_size_t a_length,   //
    sz_cptr_t b, sz_size_t b_length,   //
    sz_size_t bound, sz_memory_allocator_t *alloc) {
    return sz_dispatch_table.edit_distance(a, a_length, b, b_length, bound, alloc);
}

SZ_DYNAMIC sz_size_t sz_edit_distance_utf8( //
    sz_cptr_t a, sz_size_t a_length,        //
    sz_cptr_t b, sz_size_t b_length,        //
    sz_size_t bound, sz_memory_allocator_t *alloc) {
    return _sz_edit_distance_wagner_fisher_serial(a, a_length, b, b_length, bound, sz_true_k, alloc);
}

SZ_DYNAMIC sz_ssize_t sz_alignment_score( //
    sz_cptr_t a, sz_size_t a_length,      //
    sz_cptr_t b, sz_size_t b_length,      //
    sz_error_cost_t const *subs, sz_error_cost_t gap, sz_memory_allocator_t *alloc) {
    return sz_dispatch_table.alignment_score(a, a_length, b, b_length, subs, gap, alloc);
}

SZ_DYNAMIC void sz_hashes(                                                     //
    sz_cptr_t text, sz_size_t length, sz_size_t window_length, sz_size_t step, //
    sz_hash_callback_t callback, void *callback_handle) {
    sz_dispatch_table.hashes(text, length, window_length, step, callback, callback_handle);
}

SZ_DYNAMIC sz_cptr_t sz_find_char_from(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    sz_charset_t set;
    sz_charset_init(&set);
    for (; n_length; ++n, --n_length) sz_charset_add(&set, *n);
    return sz_find_charset(h, h_length, &set);
}

SZ_DYNAMIC sz_cptr_t sz_find_char_not_from(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    sz_charset_t set;
    sz_charset_init(&set);
    for (; n_length; ++n, --n_length) sz_charset_add(&set, *n);
    sz_charset_invert(&set);
    return sz_find_charset(h, h_length, &set);
}

SZ_DYNAMIC sz_cptr_t sz_rfind_char_from(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    sz_charset_t set;
    sz_charset_init(&set);
    for (; n_length; ++n, --n_length) sz_charset_add(&set, *n);
    return sz_rfind_charset(h, h_length, &set);
}

SZ_DYNAMIC sz_cptr_t sz_rfind_char_not_from(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    sz_charset_t set;
    sz_charset_init(&set);
    for (; n_length; ++n, --n_length) sz_charset_add(&set, *n);
    sz_charset_invert(&set);
    return sz_rfind_charset(h, h_length, &set);
}

#if !SZ_AVOID_LIBC
sz_u64_t _sz_random_generator(void *empty_state) {
    sz_unused(empty_state);
    return (sz_u64_t)rand();
}
#endif

SZ_DYNAMIC void sz_generate( //
    sz_cptr_t alphabet, sz_size_t alphabet_size, sz_ptr_t result, sz_size_t result_length,
    sz_random_generator_t generator, void *generator_user_data) {
#if !SZ_AVOID_LIBC
    if (!generator) generator = _sz_random_generator;
#endif
    sz_generate_serial(alphabet, alphabet_size, result, result_length, generator, generator_user_data);
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
    char const *base64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    sz_generate(base64, 64, s, n, SZ_NULL, SZ_NULL);
}

#endif
#endif // SZ_OVERRIDE_LIBC
