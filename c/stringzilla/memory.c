/**
 *  @file c/stringzilla/memory.c
 *  @brief Per-domain dispatch shim for raw memory ops (`sz_copy`, `sz_move`, `sz_fill`, `sz_lookup`).
 *  @author Ash Vardanian
 *  @date January 16, 2024
 */
#if !defined(SZ_OVERRIDE_LIBC)
#define SZ_OVERRIDE_LIBC SZ_AVOID_LIBC
#endif
#include "dispatch.h"
#include <stringzilla/memory.h>

#if SZ_AVOID_LIBC
#ifdef _MSC_VER
typedef sz_size_t size_t; // Reuse the type definition we've inferred from `stringzilla.h`
#else
typedef __SIZE_TYPE__ size_t; // For GCC/Clang
#endif
#endif

SZ_DISPATCH_INTERNAL void sz_dispatch_memory_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->copy = sz_copy_serial;
    impl->move = sz_move_serial;
    impl->fill = sz_fill_serial;
    impl->lookup = sz_lookup_serial;

#if SZ_USE_HASWELL
    if (caps & sz_cap_haswell_k) {
        impl->copy = sz_copy_haswell;
        impl->move = sz_move_haswell;
        impl->fill = sz_fill_haswell;
        impl->lookup = sz_lookup_haswell;
    }
#endif

#if SZ_USE_SKYLAKE
    if (caps & sz_cap_skylake_k) {
        impl->copy = sz_copy_skylake;
        impl->move = sz_move_skylake;
        impl->fill = sz_fill_skylake;
    }
#endif

#if SZ_USE_ICELAKE
    if (caps & sz_cap_icelake_k) { impl->lookup = sz_lookup_icelake; }
#endif

#if SZ_USE_NEON
    if (caps & sz_cap_neon_k) {
        impl->copy = sz_copy_neon;
        impl->move = sz_move_neon;
        impl->fill = sz_fill_neon;
        impl->lookup = sz_lookup_neon;
    }
#endif

#if SZ_USE_SVE
    if (caps & sz_cap_sve_k) {
        if (SZ_ENFORCE_SVE_OVER_NEON) {
            impl->copy = sz_copy_sve;
            impl->move = sz_move_sve;
            impl->fill = sz_fill_sve;
            impl->lookup = sz_lookup_sve;
        }
    }
#endif

#if SZ_USE_V128
    if (caps & sz_cap_v128_k) {
        impl->copy = sz_copy_v128;
        impl->move = sz_move_v128;
        impl->fill = sz_fill_v128;
        impl->lookup = sz_lookup_v128;
    }
#endif

#if SZ_USE_V128RELAXED
    if (caps & sz_cap_v128relaxed_k) {
        impl->copy = sz_copy_v128relaxed;
        impl->move = sz_move_v128relaxed;
        impl->fill = sz_fill_v128relaxed;
        impl->lookup = sz_lookup_v128relaxed;
    }
#endif

#if SZ_USE_RVV
    if (caps & sz_cap_rvv_k) {
        impl->copy = sz_copy_rvv;
        impl->move = sz_move_rvv;
        impl->fill = sz_fill_rvv;
        impl->lookup = sz_lookup_rvv;
    }
#endif

#if SZ_USE_LASX
    if (caps & sz_cap_lasx_k) {
        impl->copy = sz_copy_lasx;
        impl->move = sz_move_lasx;
        impl->fill = sz_fill_lasx;
        impl->lookup = sz_lookup_lasx;
    }
#endif

#if SZ_USE_POWERVSX
    if (caps & sz_cap_powervsx_k) {
        impl->copy = sz_copy_powervsx;
        impl->move = sz_move_powervsx;
        impl->fill = sz_fill_powervsx;
        impl->lookup = sz_lookup_powervsx;
    }
#endif
}

SZ_API_RUNTIME void sz_copy(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    sz_dispatch_table.copy(target, source, length);
}

SZ_API_RUNTIME void sz_move(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    sz_dispatch_table.move(target, source, length);
}

SZ_API_RUNTIME void sz_fill(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    sz_dispatch_table.fill(target, length, value);
}

SZ_API_RUNTIME void sz_lookup(sz_ptr_t target, sz_size_t length, sz_cptr_t source, char const lut[sz_at_least_(256)]) {
    sz_dispatch_table.lookup(target, length, source, lut);
}

// Provide overrides for the libc mem* functions
#if SZ_OVERRIDE_LIBC && !defined(__CYGWIN__)

// SZ_API_RUNTIME can't be use here for MSVC, because MSVC complains about different linkage (C2375), probably due
// to to the CRT headers specifying the function as `__declspec(dllimport)`, there might be a combination of
// defines that works. But for now they will be manually exported using linker flags.
// Also when building for 32-bit we must add an underscore to the exported function name, because that's
// how `__cdecl` functions are decorated in MSVC: https://stackoverflow.com/questions/62753691)

#if defined(_MSC_VER)
#if defined(_WIN64)
#pragma comment(linker, "/export:memcpy")
#else
#pragma comment(linker, "/export:_memcpy")
#endif
void *__cdecl memcpy(void *target, void const *source, size_t length) {
#else
SZ_API_RUNTIME void *memcpy(void *target, void const *source, size_t length) {
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
SZ_API_RUNTIME void *memmove(void *target, void const *source, size_t length) {
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
SZ_API_RUNTIME void *memset(void *target, int value, size_t length) {
#endif
    sz_fill(target, length, value);
    return (void *)target;
}

#endif // SZ_OVERRIDE_LIBC
