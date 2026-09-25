/**
 *  @file c/stringzilla/memory.c
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *  @brief Per-domain dispatch shim for raw memory operations: @c sz_copy, @c sz_move, @c sz_fill,
 *      and @c sz_lookup.
 */
#if !defined(STRINGZILLA_OVERRIDE_LIBC)
#define STRINGZILLA_OVERRIDE_LIBC (!STRINGZILLA_WITH_LIBC)
#endif
#include <stringzilla/memory.h>

#include "dispatch.h"

#if !STRINGZILLA_WITH_LIBC
#ifdef _MSC_VER
typedef sz_size_t size_t; // Reuse the type definition we've inferred from `stringzilla.h`
#else
typedef __SIZE_TYPE__ size_t; // For GCC/Clang
#endif
#endif

STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_memory_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_cpu_table;
    sz_unused_(caps);

    impl->copy = sz_copy_serial;
    impl->move = sz_move_serial;
    impl->fill = sz_fill_serial;
    impl->lookup = sz_lookup_serial;

#if STRINGZILLA_TARGET_HASWELL
    if (caps & sz_cap_haswell_k) {
        impl->copy = sz_copy_haswell;
        impl->move = sz_move_haswell;
        impl->fill = sz_fill_haswell;
        impl->lookup = sz_lookup_haswell;
    }
#endif

#if STRINGZILLA_TARGET_SKYLAKE
    if (caps & sz_cap_skylake_k) {
        impl->copy = sz_copy_skylake;
        impl->move = sz_move_skylake;
        impl->fill = sz_fill_skylake;
    }
#endif

#if STRINGZILLA_TARGET_ICELAKE
    if (caps & sz_cap_icelake_k) { impl->lookup = sz_lookup_icelake; }
#endif

#if STRINGZILLA_TARGET_NEON
    if (caps & sz_cap_neon_k) {
        impl->copy = sz_copy_neon;
        impl->move = sz_move_neon;
        impl->fill = sz_fill_neon;
        impl->lookup = sz_lookup_neon;
    }
#endif

#if STRINGZILLA_TARGET_SVE
    if (caps & sz_cap_sve_k) {
        // Wider-than-NEON registers are where the scalable memory kernels win; at the common
        // 128-bit vector length the NEON kernels stay faster, so keep them.
        if (sz_sve_wider_than_neon_()) {
            impl->copy = sz_copy_sve;
            impl->move = sz_move_sve;
            impl->fill = sz_fill_sve;
            impl->lookup = sz_lookup_sve;
        }
    }
#endif

#if STRINGZILLA_TARGET_V128
    if (caps & sz_cap_v128_k) {
        impl->copy = sz_copy_v128;
        impl->move = sz_move_v128;
        impl->fill = sz_fill_v128;
        impl->lookup = sz_lookup_v128;
    }
#endif

#if STRINGZILLA_TARGET_V128RELAXED
    if (caps & sz_cap_v128relaxed_k) {
        impl->copy = sz_copy_v128relaxed;
        impl->move = sz_move_v128relaxed;
        impl->fill = sz_fill_v128relaxed;
        impl->lookup = sz_lookup_v128relaxed;
    }
#endif

#if STRINGZILLA_TARGET_RVV
    if (caps & sz_cap_rvv_k) {
        impl->copy = sz_copy_rvv;
        impl->move = sz_move_rvv;
        impl->fill = sz_fill_rvv;
        impl->lookup = sz_lookup_rvv;
    }
#endif

#if STRINGZILLA_TARGET_LASX
    if (caps & sz_cap_lasx_k) {
        impl->copy = sz_copy_lasx;
        impl->move = sz_move_lasx;
        impl->fill = sz_fill_lasx;
        impl->lookup = sz_lookup_lasx;
    }
#endif

#if STRINGZILLA_TARGET_POWERVSX
    if (caps & sz_cap_powervsx_k) {
        impl->copy = sz_copy_powervsx;
        impl->move = sz_move_powervsx;
        impl->fill = sz_fill_powervsx;
        impl->lookup = sz_lookup_powervsx;
    }
#endif
}

STRINGZILLA_API_RUNTIME void sz_copy(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    sz_dispatch_cpu_table.copy(target, source, length);
}

STRINGZILLA_API_RUNTIME void sz_move(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    sz_dispatch_cpu_table.move(target, source, length);
}

STRINGZILLA_API_RUNTIME void sz_fill(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    sz_dispatch_cpu_table.fill(target, length, value);
}

STRINGZILLA_API_RUNTIME void sz_lookup(sz_ptr_t target, sz_size_t length, sz_cptr_t source,
                                       char const lut[sz_at_least_(256)]) {
    sz_dispatch_cpu_table.lookup(target, length, source, lut);
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
#pragma comment(linker, "/export:memcpy")
#else
#pragma comment(linker, "/export:_memcpy")
#endif
void *__cdecl memcpy(void *target, void const *source, size_t length) {
#else
STRINGZILLA_API_RUNTIME void *memcpy(void *target, void const *source, size_t length) {
#endif
    sz_copy(target, source, length);
    return (void *)target;
}

#if defined(_MSC_VER)
#if defined(_WIN64)
#pragma comment(linker, "/export:memmove")
#else
#pragma comment(linker, "/export:_memmove")
#endif
void *__cdecl memmove(void *target, void const *source, size_t length) {
#else
STRINGZILLA_API_RUNTIME void *memmove(void *target, void const *source, size_t length) {
#endif
    sz_move(target, source, length);
    return (void *)target;
}

#if defined(_MSC_VER)
#if defined(_WIN64)
#pragma comment(linker, "/export:memset")
#else
#pragma comment(linker, "/export:_memset")
#endif
void *__cdecl memset(void *target, int value, size_t length) {
#else
STRINGZILLA_API_RUNTIME void *memset(void *target, int value, size_t length) {
#endif
    sz_fill(target, length, value);
    return (void *)target;
}

#endif // STRINGZILLA_OVERRIDE_LIBC
