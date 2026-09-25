/**
 *  @file c/stringzilla/utf8_wordbreaks.c
 *  @author Ash Vardanian
 *  @date November 30, 2025
 *  @brief Per-domain dispatch shim for UAX-29 word boundary segmentation.
 */
#include <stringzilla/utf8_wordbreaks.h>

#include "dispatch.h"

STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_utf8_wordbreaks_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_cpu_table;
    sz_unused_(caps);

    impl->utf8_wordbreaks = sz_utf8_wordbreaks_serial;

#if STRINGZILLA_TARGET_HASWELL
    if (caps & sz_cap_haswell_k) { impl->utf8_wordbreaks = sz_utf8_wordbreaks_haswell; }
#endif
#if STRINGZILLA_TARGET_NEON
    if (caps & sz_cap_neon_k) { impl->utf8_wordbreaks = sz_utf8_wordbreaks_neon; }
#endif
#if STRINGZILLA_TARGET_ICELAKE
    if (caps & sz_cap_icelake_k) { impl->utf8_wordbreaks = sz_utf8_wordbreaks_icelake; }
#endif

#if STRINGZILLA_TARGET_V128
    if (caps & sz_cap_v128_k) { impl->utf8_wordbreaks = sz_utf8_wordbreaks_v128; }
#endif

#if STRINGZILLA_TARGET_RVV
    if (caps & sz_cap_rvv_k) { impl->utf8_wordbreaks = sz_utf8_wordbreaks_rvv; }
#endif

#if STRINGZILLA_TARGET_LASX
    if (caps & sz_cap_lasx_k) { impl->utf8_wordbreaks = sz_utf8_wordbreaks_lasx; }
#endif

#if STRINGZILLA_TARGET_POWERVSX
    if (caps & sz_cap_powervsx_k) { impl->utf8_wordbreaks = sz_utf8_wordbreaks_powervsx; }
#endif
}

STRINGZILLA_API_RUNTIME sz_size_t sz_utf8_wordbreaks(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                     sz_size_t *word_lengths, sz_size_t words_capacity,
                                                     sz_size_t *bytes_consumed) {
    return sz_dispatch_cpu_table.utf8_wordbreaks(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
}
