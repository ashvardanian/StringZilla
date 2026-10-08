/**
 *  @file include/stringzilla/utf8_uncased/powervsx.h
 *  @author Ash Vardanian
 *  @date June 7, 2026
 *  @brief IBM Power (VSX) uncased UTF-8 search, comparison & invariance backend.
 *
 *  @sa include/stringzilla/utf8_uncased.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_POWERVSX_H_
#define STRINGZILLA_UTF8_UNCASED_POWERVSX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_uncased/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  This ISA has no dedicated uncased UTF-8 kernels yet; it delegates to the serial
 *  scaffolding so the per-backend symbol set stays uniform across all targets. */
#if STRINGZILLA_ARCH_PPC64_
#if STRINGZILLA_TARGET_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

STRINGZILLA_API sz_status_t sz_utf8_uncased_search_powervsx(                               //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_utf8_uncased_needle_t const *needle, //
    sz_cptr_t *match, sz_size_t *match_length, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_utf8_uncased_search_serial_(haystack, haystack_length, needle->start, needle->length, needle,
                                            match_length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_utf8_find_cased_powervsx(sz_cptr_t text, sz_size_t length, sz_cptr_t *match,
                                                        sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_utf8_find_cased_serial_(text, length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_utf8_uncased_order_powervsx(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                           sz_size_t b_length, sz_ordering_t *ordering,
                                                           sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *ordering = sz_utf8_uncased_order_serial_(a, a_length, b, b_length);
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_POWERVSX
#endif // STRINGZILLA_ARCH_PPC64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_UNCASED_POWERVSX_H_
