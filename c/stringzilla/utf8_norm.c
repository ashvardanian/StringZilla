/**
 *  @file c/stringzilla/utf8_norm.c
 *  @brief Per-domain dispatch shim for the single-pass Unicode normalizer (NFD / NFC / NFKD / NFKC).
 *  @author Ash Vardanian
 *
 *  Fills its slice of the shared dispatch table and defines the `SZ_DYNAMIC` public wrappers that call
 *  through it. The two entry points - the normalizer and the violation finder - share one streaming
 *  engine parameterized by a force-inlined scan primitive; the NEON backend overrides only that scan.
 */
#include "dispatch.h"
#include <stringzilla/utf8_norm.h> // `sz_utf8_norm_*`, `sz_utf8_norm_violation_*`

SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_norm_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->utf8_norm = sz_utf8_norm_serial;
    impl->utf8_norm_violation = sz_utf8_norm_violation_serial;

#if SZ_USE_HASWELL
    if (caps & sz_cap_haswell_k) {
        impl->utf8_norm = sz_utf8_norm_haswell;
        impl->utf8_norm_violation = sz_utf8_norm_violation_haswell;
    }
#endif
#if SZ_USE_SKYLAKE
    if (caps & sz_cap_skylake_k) {
        impl->utf8_norm = sz_utf8_norm_skylake;
        impl->utf8_norm_violation = sz_utf8_norm_violation_skylake;
    }
#endif
#if SZ_USE_ICELAKE
    if (caps & sz_cap_icelake_k) { // an Ice Lake CPU also reports skylake_k; this later block wins
        impl->utf8_norm = sz_utf8_norm_icelake;
        impl->utf8_norm_violation = sz_utf8_norm_violation_icelake;
    }
#endif
#if SZ_USE_NEON
    if (caps & sz_cap_neon_k) {
        impl->utf8_norm = sz_utf8_norm_neon;
        impl->utf8_norm_violation = sz_utf8_norm_violation_neon;
    }
#endif
#if SZ_USE_SVE
    if (caps & sz_cap_sve_k) {
        impl->utf8_norm = sz_utf8_norm_sve;
        impl->utf8_norm_violation = sz_utf8_norm_violation_sve;
    }
#endif
#if SZ_USE_SVE2
    if (caps & sz_cap_sve2_k) {
        impl->utf8_norm = sz_utf8_norm_sve2;
        impl->utf8_norm_violation = sz_utf8_norm_violation_sve2;
    }
#endif
#if SZ_USE_RVV
    if (caps & sz_cap_rvv_k) {
        impl->utf8_norm = sz_utf8_norm_rvv;
        impl->utf8_norm_violation = sz_utf8_norm_violation_rvv;
    }
#endif
#if SZ_USE_LASX
    if (caps & sz_cap_lasx_k) {
        impl->utf8_norm = sz_utf8_norm_lasx;
        impl->utf8_norm_violation = sz_utf8_norm_violation_lasx;
    }
#endif
#if SZ_USE_POWERVSX
    if (caps & sz_cap_powervsx_k) {
        impl->utf8_norm = sz_utf8_norm_powervsx;
        impl->utf8_norm_violation = sz_utf8_norm_violation_powervsx;
    }
#endif
#if SZ_USE_V128
    if (caps & sz_cap_v128_k) {
        impl->utf8_norm = sz_utf8_norm_v128;
        impl->utf8_norm_violation = sz_utf8_norm_violation_v128;
    }
#endif
#if SZ_USE_V128RELAXED
    if (caps & sz_cap_v128relaxed_k) {
        impl->utf8_norm = sz_utf8_norm_v128relaxed;
        impl->utf8_norm_violation = sz_utf8_norm_violation_v128relaxed;
    }
#endif
}

SZ_DYNAMIC sz_size_t sz_utf8_norm(sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form,
                                  sz_ptr_t destination) {
    return sz_dispatch_table.utf8_norm(source, source_length, form, destination);
}

SZ_DYNAMIC sz_cptr_t sz_utf8_norm_violation(sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form) {
    return sz_dispatch_table.utf8_norm_violation(source, source_length, form);
}
