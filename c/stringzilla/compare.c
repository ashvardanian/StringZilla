/**
 *  @file c/stringzilla/compare.c
 *  @brief Per-domain dispatch shim for byte-level comparison (`sz_equal`, `sz_order`).
 *  @author Ash Vardanian
 *  @date January 16, 2024
 */
#include "dispatch.h"
#include <stringzilla/compare.h>

SZ_DISPATCH_INTERNAL void sz_dispatch_compare_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->equal = sz_equal_serial;
    impl->order = sz_order_serial;

#if SZ_USE_WESTMERE
    if (caps & sz_cap_westmere_k) {
        impl->equal = sz_equal_westmere;
        impl->order = sz_order_westmere;
    }
#endif

#if SZ_USE_HASWELL
    if (caps & sz_cap_haswell_k) {
        impl->equal = sz_equal_haswell;
        impl->order = sz_order_haswell;
    }
#endif

#if SZ_USE_SKYLAKE
    if (caps & sz_cap_skylake_k) {
        impl->equal = sz_equal_skylake;
        impl->order = sz_order_skylake;
    }
#endif

#if SZ_USE_NEON
    if (caps & sz_cap_neon_k) { impl->equal = sz_equal_neon; }
#endif

#if SZ_USE_SVE
    if (caps & sz_cap_sve_k) {
        if (SZ_ENFORCE_SVE_OVER_NEON) {
            impl->equal = sz_equal_sve;
            impl->order = sz_order_sve;
        }
    }
#endif

#if SZ_USE_V128
    if (caps & sz_cap_v128_k) {
        impl->equal = sz_equal_v128;
        impl->order = sz_order_v128;
    }
#endif

#if SZ_USE_V128RELAXED
    if (caps & sz_cap_v128relaxed_k) {
        impl->equal = sz_equal_v128relaxed;
        impl->order = sz_order_v128relaxed;
    }
#endif

#if SZ_USE_RVV
    if (caps & sz_cap_rvv_k) {
        impl->equal = sz_equal_rvv;
        impl->order = sz_order_rvv;
    }
#endif

#if SZ_USE_LASX
    if (caps & sz_cap_lasx_k) {
        impl->equal = sz_equal_lasx;
        impl->order = sz_order_lasx;
    }
#endif

#if SZ_USE_POWERVSX
    if (caps & sz_cap_powervsx_k) {
        impl->equal = sz_equal_powervsx;
        impl->order = sz_order_powervsx;
    }
#endif
}

SZ_DYNAMIC sz_bool_t sz_equal(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    return sz_dispatch_table.equal(a, b, length);
}

SZ_DYNAMIC sz_ordering_t sz_order(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    return sz_dispatch_table.order(a, a_length, b, b_length);
}
