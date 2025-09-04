/**
 *  @file       stringzilla.c
 *  @brief      StringZilla C library with dynamic backed dispatch for the most appropriate implementation.
 *  @author     Ash Vardanian
 *  @date       January 16, 2024
 */

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

#if defined(SZ_IS_WINDOWS_)
#include <windows.h> // `DllMain`
#endif

typedef struct sz_implementations_t {
    sz_equal_t equal;
    sz_order_t order;

    sz_copy_t copy;
    sz_move_t move;
    sz_fill_t fill;
    sz_lookup_t lookup;

    sz_bytesum_t bytesum;
    sz_hash_t hash;
    sz_hash_state_init_t hash_state_init;
    sz_hash_state_update_t hash_state_update;
    sz_hash_state_digest_t hash_state_digest;
    sz_fill_random_t fill_random;

    sz_find_byte_t find_byte;
    sz_find_byte_t rfind_byte;
    sz_find_t find;
    sz_find_t rfind;
    sz_find_byteset_t find_byteset;
    sz_find_byteset_t rfind_byteset;

    sz_sequence_argsort_t sequence_argsort;
    sz_sequence_intersect_t sequence_intersect;
    sz_pgrams_sort_t pgrams_sort;

} sz_implementations_t;

#if defined(_MSC_VER)
__declspec(align(64)) static sz_implementations_t sz_dispatch_table;
#else
__attribute__((aligned(64))) static sz_implementations_t sz_dispatch_table;
#endif

static void sz_dispatch_table_update_implementation_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps); //< Unused when compiling on pre-SIMD machines.

    impl->equal = sz_equal_serial;
    impl->order = sz_order_serial;
    impl->copy = sz_copy_serial;
    impl->move = sz_move_serial;
    impl->fill = sz_fill_serial;
    impl->lookup = sz_lookup_serial;

    impl->bytesum = sz_bytesum_serial;
    impl->hash = sz_hash_serial;
    impl->hash_state_init = sz_hash_state_init_serial;
    impl->hash_state_update = sz_hash_state_update_serial;
    impl->hash_state_digest = sz_hash_state_digest_serial;
    impl->fill_random = sz_fill_random_serial;

    impl->find = sz_find_serial;
    impl->rfind = sz_rfind_serial;
    impl->find_byte = sz_find_byte_serial;
    impl->rfind_byte = sz_rfind_byte_serial;
    impl->find_byteset = sz_find_byteset_serial;
    impl->rfind_byteset = sz_rfind_byteset_serial;

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
        impl->hash_state_update = sz_hash_state_update_haswell;
        impl->hash_state_digest = sz_hash_state_digest_haswell;
        impl->fill_random = sz_fill_random_haswell;

        impl->find_byte = sz_find_byte_haswell;
        impl->rfind_byte = sz_rfind_byte_haswell;
        impl->find = sz_find_haswell;
        impl->rfind = sz_rfind_haswell;
        impl->find_byteset = sz_find_byteset_haswell;
        impl->rfind_byteset = sz_rfind_byteset_haswell;
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
        impl->hash_state_update = sz_hash_state_update_skylake;
        impl->hash_state_digest = sz_hash_state_digest_skylake;
        impl->fill_random = sz_fill_random_skylake;

        impl->find = sz_find_skylake;
        impl->rfind = sz_rfind_skylake;
        impl->find_byte = sz_find_byte_skylake;
        impl->rfind_byte = sz_rfind_byte_skylake;

        impl->sequence_argsort = sz_sequence_argsort_skylake;
        impl->pgrams_sort = sz_pgrams_sort_skylake;
    }
#endif

#if SZ_USE_ICE
    if (caps & sz_cap_ice_k) {
        impl->find_byteset = sz_find_byteset_ice;
        impl->rfind_byteset = sz_rfind_byteset_ice;

        impl->lookup = sz_lookup_ice;

        impl->bytesum = sz_bytesum_ice;
        impl->hash = sz_hash_ice;
        impl->hash_state_init = sz_hash_state_init_ice;
        impl->hash_state_update = sz_hash_state_update_ice;
        impl->hash_state_digest = sz_hash_state_digest_ice;
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

        impl->find = sz_find_neon;
        impl->rfind = sz_rfind_neon;
        impl->find_byte = sz_find_byte_neon;
        impl->rfind_byte = sz_rfind_byte_neon;
        impl->find_byteset = sz_find_byteset_neon;
        impl->rfind_byteset = sz_rfind_byteset_neon;
    }
#endif

#if SZ_USE_NEON_AES
    if (caps & sz_cap_neon_aes_k) {
        impl->hash = sz_hash_neon;
        impl->hash_state_init = sz_hash_state_init_neon;
        impl->hash_state_update = sz_hash_state_update_neon;
        impl->hash_state_digest = sz_hash_state_digest_neon;
        impl->fill_random = sz_fill_random_neon;
    }
#endif

#if SZ_USE_SVE
    if (caps & sz_cap_sve_k) {
        impl->equal = sz_equal_sve;
        impl->order = sz_order_sve;

        impl->copy = sz_copy_sve;
        impl->move = sz_move_sve;
        impl->fill = sz_fill_sve;

        impl->find = sz_find_sve;
        // TODO: impl->rfind = sz_rfind_sve;
        impl->find_byte = sz_find_byte_sve;
        impl->rfind_byte = sz_rfind_byte_sve;

        impl->bytesum = sz_bytesum_sve;

        impl->sequence_argsort = sz_sequence_argsort_sve;
        impl->sequence_intersect = sz_sequence_intersect_sve;
        impl->pgrams_sort = sz_pgrams_sort_sve;
    }
#endif

#if SZ_USE_SVE2
    if (caps & sz_cap_sve2_k) { impl->bytesum = sz_bytesum_sve2; }
#endif

#if SZ_USE_SVE2_AES
    if (caps & sz_cap_sve2_aes_k) {
        impl->hash = sz_hash_sve2;
        impl->hash_state_init = sz_hash_state_init_sve2;
        impl->hash_state_update = sz_hash_state_update_sve2;
        impl->hash_state_digest = sz_hash_state_digest_sve2;
        impl->fill_random = sz_fill_random_sve2;
    }
#endif
}

/**
 *  @brief  Initializes a global static "virtual table" of supported backends
 *          Run it just once to avoiding unnecessary `if`-s.
 */
SZ_DYNAMIC void sz_dispatch_table_init(void) {
    sz_capability_t caps = sz_capabilities();
    sz_dispatch_table_update_implementation_(caps);
}

SZ_DYNAMIC void sz_dispatch_table_update(sz_capability_t caps) { sz_dispatch_table_update_implementation_(caps); }

#if defined(_MSC_VER)
/*
 *  Makes sure the `sz_dispatch_table_init` function is called at startup, from either an executable or when loading
 *  a DLL. The section name must be no more than 8 characters long, and must be between .CRT$XCA and .CRT$XCZ
 *  alphabetically (exclusive). The Microsoft C++ compiler puts C++ initialisation code in .CRT$XCU, so avoid that
 *  section: https://learn.microsoft.com/en-us/cpp/c-runtime-library/crt-initialization?view=msvc-170
 */
#if defined(_WIN64)
#pragma comment(linker, "/INCLUDE:sz_dispatch_table_init_")
#else
#pragma comment(linker, "/INCLUDE:_sz_dispatch_table_init_")
#endif
#pragma section(".CRT$XCS", read)
__declspec(allocate(".CRT$XCS")) void (*sz_dispatch_table_init_)() = sz_dispatch_table_init;

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
SZ_DYNAMIC sz_capability_t sz_capabilities(void) { return sz_capabilities_implementation_(); }
SZ_DYNAMIC sz_cptr_t sz_capabilities_to_string(sz_capability_t caps) {
    return sz_capabilities_to_string_implementation_(caps);
}

SZ_DYNAMIC sz_u64_t sz_bytesum(sz_cptr_t text, sz_size_t length) { return sz_dispatch_table.bytesum(text, length); }

SZ_DYNAMIC sz_u64_t sz_hash(sz_cptr_t text, sz_size_t length, sz_u64_t seed) {
    return sz_dispatch_table.hash(text, length, seed);
}

SZ_DYNAMIC void sz_hash_state_init(sz_hash_state_t *state, sz_u64_t seed) {
    sz_dispatch_table.hash_state_init(state, seed);
}

SZ_DYNAMIC void sz_hash_state_update(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_dispatch_table.hash_state_update(state, text, length);
}

SZ_DYNAMIC sz_u64_t sz_hash_state_digest(sz_hash_state_t const *state) {
    return sz_dispatch_table.hash_state_digest(state);
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
    return sz_dispatch_table.find_byteset(text, length, set);
}

SZ_DYNAMIC sz_cptr_t sz_rfind_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
    return sz_dispatch_table.rfind_byteset(text, length, set);
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
#if defined(_WIN64)
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
#if defined(_WIN64)
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
#if defined(_WIN64)
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
#if defined(_WIN64)
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
