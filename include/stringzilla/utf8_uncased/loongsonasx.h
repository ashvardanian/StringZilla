/**
 *  @file include/stringzilla/utf8_uncased/loongsonasx.h
 *  @author Ash Vardanian
 *  @date June 7, 2026
 *  @brief LoongArch LASX uncased UTF-8 search, comparison & invariance backend.
 *
 *  @sa include/stringzilla/utf8_uncased.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_LOONGSONASX_H_
#define STRINGZILLA_UTF8_UNCASED_LOONGSONASX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_uncased/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  This ISA has no dedicated uncased UTF-8 kernels yet; it delegates to the serial
 *  scaffolding so the per-backend symbol set stays uniform across all targets. */
#if STRINGZILLA_TARGET_LOONGSONASX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("lasx"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("lasx")
#endif

STRINGZILLA_API sz_status_t sz_utf8_uncased_search_loongsonasx(                            //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_utf8_uncased_needle_t const *needle, //
    sz_cptr_t *match, sz_size_t *match_length, void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_utf8_uncased_search_serial_(haystack, haystack_length, needle->start, needle->length, needle,
                                            match_length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_utf8_find_cased_loongsonasx(sz_cptr_t text, sz_size_t length, sz_cptr_t *match,
                                                           void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_utf8_find_cased_serial_(text, length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_utf8_uncased_order_loongsonasx(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                              sz_size_t b_length, sz_ordering_t *ordering,
                                                              void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *ordering = sz_utf8_uncased_order_serial_(a, a_length, b, b_length);
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_LOONGSONASX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_UNCASED_LOONGSONASX_H_
