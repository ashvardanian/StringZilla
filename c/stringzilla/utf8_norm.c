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

#if SZ_USE_NEON
    if (caps & sz_cap_neon_k) {
        impl->utf8_norm = sz_utf8_norm_neon;
        impl->utf8_norm_violation = sz_utf8_norm_violation_neon;
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
