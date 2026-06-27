/**
 *  @file c/stringzilla/utf8_uncased.c
 *  @brief Per-domain dispatch shim for uncased UTF-8 search & ordering.
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *
 *  Isolated from the folding shim (`utf8_uncased_fold.c`) because the AVX-512 per-script find kernels in
 *  `utf8_uncased/icelake.h` are by far the heaviest single compilation in the core; keeping
 *  them in their own translation unit lets the rest of the UTF-8 case domain build in parallel.
 */
#include "dispatch.h"
#include <stringzilla/utf8_uncased.h> // `sz_utf8_uncased_{find,order}_*`

SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_uncased_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->utf8_uncased_search = sz_utf8_uncased_search_serial;
    impl->utf8_uncased_order = sz_utf8_uncased_order_serial;

#if SZ_USE_HASWELL
    if (caps & sz_cap_haswell_k) {
        impl->utf8_uncased_search = sz_utf8_uncased_search_haswell;
        impl->utf8_uncased_order = sz_utf8_uncased_order_haswell;
    }
#endif

#if SZ_USE_ICELAKE
    if (caps & sz_cap_icelake_k) { impl->utf8_uncased_search = sz_utf8_uncased_search_icelake; }
#endif

#if SZ_USE_NEON
    if (caps & sz_cap_neon_k) { impl->utf8_uncased_search = sz_utf8_uncased_search_neon; }
#endif

#if SZ_USE_V128
    if (caps & sz_cap_v128_k) {
        impl->utf8_uncased_search = sz_utf8_uncased_search_v128;
        impl->utf8_uncased_order = sz_utf8_uncased_order_v128;
    }
#endif

#if SZ_USE_RVV
    if (caps & sz_cap_rvv_k) {
        impl->utf8_uncased_search = sz_utf8_uncased_search_rvv;
        impl->utf8_uncased_order = sz_utf8_uncased_order_rvv;
    }
#endif

#if SZ_USE_LASX
    if (caps & sz_cap_lasx_k) {
        impl->utf8_uncased_search = sz_utf8_uncased_search_lasx;
        impl->utf8_uncased_order = sz_utf8_uncased_order_lasx;
    }
#endif

#if SZ_USE_POWERVSX
    if (caps & sz_cap_powervsx_k) {
        impl->utf8_uncased_search = sz_utf8_uncased_search_powervsx;
        impl->utf8_uncased_order = sz_utf8_uncased_order_powervsx;
    }
#endif
}

SZ_DYNAMIC sz_cptr_t sz_utf8_uncased_search(       //
    sz_cptr_t haystack, sz_size_t haystack_length, //
    sz_cptr_t needle, sz_size_t needle_length,     //
    sz_utf8_uncased_needle_metadata_t *needle_metadata, sz_size_t *matched_length) {
    return sz_dispatch_table.utf8_uncased_search(haystack, haystack_length, needle, needle_length, needle_metadata,
                                                 matched_length);
}

SZ_DYNAMIC sz_ordering_t sz_utf8_uncased_order( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    return sz_dispatch_table.utf8_uncased_order(a, a_length, b, b_length);
}
