/**
 *  @file c/stringzilla/find.c
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *  @brief Per-domain dispatch shim for substring & byte-set search: `sz_find*` and `sz_rfind*`.
 */
#if !defined(STRINGZILLA_OVERRIDE_LIBC)
#define STRINGZILLA_OVERRIDE_LIBC (!STRINGZILLA_WITH_LIBC)
#endif
#include <stringzilla/find.h>

#include "dispatch.h"

#if !STRINGZILLA_WITH_LIBC
#ifdef _MSC_VER
typedef sz_size_t size_t; // Reuse the type definition we've inferred from `stringzilla.h`
#else
typedef __SIZE_TYPE__ size_t; // For GCC/Clang
#endif
#endif

STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_find_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_cpu_table;
    sz_unused_(caps);

    impl->find = sz_find_serial;
    impl->rfind = sz_rfind_serial;
    impl->find_byte = sz_find_byte_serial;
    impl->rfind_byte = sz_rfind_byte_serial;
    impl->find_byteset = sz_find_byteset_serial;
    impl->rfind_byteset = sz_rfind_byteset_serial;

#if STRINGZILLA_TARGET_WESTMERE
    if (caps & sz_cap_westmere_k) {
        impl->find_byte = sz_find_byte_westmere;
        impl->rfind_byte = sz_rfind_byte_westmere;
        impl->find = sz_find_westmere;
        impl->rfind = sz_rfind_westmere;
    }
#endif

#if STRINGZILLA_TARGET_HASWELL
    if (caps & sz_cap_haswell_k) {
        impl->find_byte = sz_find_byte_haswell;
        impl->rfind_byte = sz_rfind_byte_haswell;
        impl->find = sz_find_haswell;
        impl->rfind = sz_rfind_haswell;
        impl->find_byteset = sz_find_byteset_haswell;
        impl->rfind_byteset = sz_rfind_byteset_haswell;
    }
#endif

#if STRINGZILLA_TARGET_SKYLAKE
    if (caps & sz_cap_skylake_k) {
        impl->find = sz_find_skylake;
        impl->rfind = sz_rfind_skylake;
        impl->find_byte = sz_find_byte_skylake;
        impl->rfind_byte = sz_rfind_byte_skylake;
    }
#endif

#if STRINGZILLA_TARGET_ICELAKE
    if (caps & sz_cap_icelake_k) {
        impl->find_byteset = sz_find_byteset_icelake;
        impl->rfind_byteset = sz_rfind_byteset_icelake;
    }
#endif

#if STRINGZILLA_TARGET_NEON
    if (caps & sz_cap_neon_k) {
        impl->find = sz_find_neon;
        impl->rfind = sz_rfind_neon;
        impl->find_byte = sz_find_byte_neon;
        impl->rfind_byte = sz_rfind_byte_neon;
        impl->find_byteset = sz_find_byteset_neon;
        impl->rfind_byteset = sz_rfind_byteset_neon;
    }
#endif

#if STRINGZILLA_TARGET_SVE
    if (caps & sz_cap_sve_k) {
        // Wider-than-NEON registers are where the scalable substring kernels win; at the common
        // 128-bit vector length the NEON kernels stay faster, so keep them.
        if (sz_sve_wider_than_neon_()) {
            impl->find = sz_find_sve;
            impl->rfind = sz_rfind_sve;
        }

        // Byte search stays scalable at every vector length: predicated heads make short inputs
        // nearly free (Graviton 5 short words: 1.55 GiB/s vs 0.82 NEON) and long scans tie NEON.
        impl->find_byte = sz_find_byte_sve;
        impl->rfind_byte = sz_rfind_byte_sve;
    }
#endif

#if STRINGZILLA_TARGET_SVE2
    if (caps & sz_cap_sve2_k) {
        // One `MATCH` covers a whole small set per instruction; the dense-set `svtbl` path matches NEON's
        // two-table bit test with predicated tails, so SVE2 wins at every vector length.
        impl->find_byteset = sz_find_byteset_sve2;
        impl->rfind_byteset = sz_rfind_byteset_sve2;
    }
#endif

#if STRINGZILLA_TARGET_V128
    if (caps & sz_cap_v128_k) {
        impl->find = sz_find_v128;
        impl->rfind = sz_rfind_v128;
        impl->find_byte = sz_find_byte_v128;
        impl->rfind_byte = sz_rfind_byte_v128;
        impl->find_byteset = sz_find_byteset_v128;
        impl->rfind_byteset = sz_rfind_byteset_v128;
    }
#endif

#if STRINGZILLA_TARGET_V128RELAXED
    if (caps & sz_cap_v128relaxed_k) {
        impl->find = sz_find_v128relaxed;
        impl->rfind = sz_rfind_v128relaxed;
        impl->find_byte = sz_find_byte_v128relaxed;
        impl->rfind_byte = sz_rfind_byte_v128relaxed;
        impl->find_byteset = sz_find_byteset_v128relaxed;
        impl->rfind_byteset = sz_rfind_byteset_v128relaxed;
    }
#endif

#if STRINGZILLA_TARGET_RVV
    if (caps & sz_cap_rvv_k) {
        impl->find = sz_find_rvv;
        impl->rfind = sz_rfind_rvv;
        impl->find_byte = sz_find_byte_rvv;
        impl->rfind_byte = sz_rfind_byte_rvv;
        impl->find_byteset = sz_find_byteset_rvv;
        impl->rfind_byteset = sz_rfind_byteset_rvv;
    }
#endif

#if STRINGZILLA_TARGET_LASX
    if (caps & sz_cap_lasx_k) {
        impl->find = sz_find_lasx;
        impl->rfind = sz_rfind_lasx;
        impl->find_byte = sz_find_byte_lasx;
        impl->rfind_byte = sz_rfind_byte_lasx;
        impl->find_byteset = sz_find_byteset_lasx;
        impl->rfind_byteset = sz_rfind_byteset_lasx;
    }
#endif

#if STRINGZILLA_TARGET_POWERVSX
    if (caps & sz_cap_powervsx_k) {
        impl->find = sz_find_powervsx;
        impl->rfind = sz_rfind_powervsx;
        impl->find_byte = sz_find_byte_powervsx;
        impl->rfind_byte = sz_rfind_byte_powervsx;
        impl->find_byteset = sz_find_byteset_powervsx;
        impl->rfind_byteset = sz_rfind_byteset_powervsx;
    }
#endif
}

STRINGZILLA_API_RUNTIME sz_cptr_t sz_find_byte(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    return sz_dispatch_cpu_table.find_byte(haystack, haystack_length, needle);
}

STRINGZILLA_API_RUNTIME sz_cptr_t sz_rfind_byte(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    return sz_dispatch_cpu_table.rfind_byte(haystack, haystack_length, needle);
}

STRINGZILLA_API_RUNTIME sz_cptr_t sz_find(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length) {
    return sz_dispatch_cpu_table.find(haystack, haystack_length, needle, needle_length);
}

STRINGZILLA_API_RUNTIME sz_cptr_t sz_rfind(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                           sz_size_t needle_length) {
    return sz_dispatch_cpu_table.rfind(haystack, haystack_length, needle, needle_length);
}

STRINGZILLA_API_RUNTIME sz_cptr_t sz_find_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
    return sz_dispatch_cpu_table.find_byteset(text, length, set);
}

STRINGZILLA_API_RUNTIME sz_cptr_t sz_rfind_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
    return sz_dispatch_cpu_table.rfind_byteset(text, length, set);
}

/*  Overrides for the LibC `mem*` functions.
 *
 *  @c STRINGZILLA_API_RUNTIME can't be used here for MSVC, which complains about different linkage,
 *  C2375, probably because the CRT headers declare the function as `__declspec(dllimport)`; some
 *  combination of defines might work, but for now the functions are exported manually with linker
 *  flags. A 32-bit build must also prefix the exported name with an underscore, because that is how
 *  MSVC decorates @c __cdecl functions: https://stackoverflow.com/questions/62753691 */
#if STRINGZILLA_OVERRIDE_LIBC && !defined(__CYGWIN__)
#if defined(_MSC_VER)
#if defined(_WIN64)
#pragma comment(linker, "/export:memchr")
#else
#pragma comment(linker, "/export:_memchr")
#endif
void *__cdecl memchr(void const *haystack, int character_wide, size_t length) {
#else
STRINGZILLA_API_RUNTIME void *memchr(void const *haystack, int character_wide, size_t length) {
#endif
    sz_u8_t character = (sz_u8_t)character_wide;
    return (void *)sz_find_byte(haystack, length, (sz_cptr_t)&character);
}

#if !defined(_MSC_VER)
STRINGZILLA_API_RUNTIME void *memmem(void const *haystack, size_t haystack_length, void const *needle,
                                     size_t needle_length) {
    return (void *)sz_find(haystack, haystack_length, needle, needle_length);
}

STRINGZILLA_API_RUNTIME void *memrchr(void const *haystack, int character_wide, size_t length) {
    sz_u8_t character = (sz_u8_t)character_wide;
    return (void *)sz_rfind_byte(haystack, length, (sz_cptr_t)&character);
}

#endif
#endif // STRINGZILLA_OVERRIDE_LIBC
