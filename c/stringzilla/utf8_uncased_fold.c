/**
 *  @file c/stringzilla/utf8_uncased_fold.c
 *  @brief Per-domain dispatch shim for UTF-8 case folding.
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *
 *  Split from the uncased @b find/order shim (`utf8_uncased.c`) so the cheap
 *  folding kernels compile as their own translation unit, in parallel with the heavy AVX-512 find.
 */
#include "dispatch.h"
#include <stringzilla/utf8_uncased_fold.h> // `sz_utf8_uncased_fold_*`

SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_uncased_fold_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->utf8_uncased_fold = sz_utf8_uncased_fold_serial;

#if SZ_USE_HASWELL
    if (caps & sz_cap_haswell_k) { impl->utf8_uncased_fold = sz_utf8_uncased_fold_haswell; }
#endif

#if SZ_USE_ICELAKE
    if (caps & sz_cap_icelake_k) { impl->utf8_uncased_fold = sz_utf8_uncased_fold_icelake; }
#endif

#if SZ_USE_NEON
    if (caps & sz_cap_neon_k) { impl->utf8_uncased_fold = sz_utf8_uncased_fold_neon; }
#endif

#if SZ_USE_V128
    if (caps & sz_cap_v128_k) { impl->utf8_uncased_fold = sz_utf8_uncased_fold_v128; }
#endif

#if SZ_USE_RVV
    if (caps & sz_cap_rvv_k) { impl->utf8_uncased_fold = sz_utf8_uncased_fold_rvv; }
#endif

#if SZ_USE_LASX
    if (caps & sz_cap_lasx_k) { impl->utf8_uncased_fold = sz_utf8_uncased_fold_lasx; }
#endif

#if SZ_USE_POWERVSX
    if (caps & sz_cap_powervsx_k) { impl->utf8_uncased_fold = sz_utf8_uncased_fold_powervsx; }
#endif
}

SZ_DYNAMIC sz_size_t sz_utf8_uncased_fold(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {
    return sz_dispatch_table.utf8_uncased_fold(source, source_length, destination);
}
