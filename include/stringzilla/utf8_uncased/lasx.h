/**
 *  @brief LoongArch LASX uncased UTF-8 search, comparison & invariance backend.
 *  @file include/stringzilla/utf8_uncased/lasx.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_LASX_H_
#define STRINGZILLA_UTF8_UNCASED_LASX_H_

#include "stringzilla/utf8_uncased/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_LASX

/*  This ISA has no dedicated uncased UTF-8 kernels yet; it delegates to the serial
 *  scaffolding so the per-backend symbol set stays uniform across all targets. */

SZ_PUBLIC sz_cptr_t sz_utf8_uncased_find_lasx( //
    sz_cptr_t haystack, sz_size_t haystack_length,      //
    sz_cptr_t needle, sz_size_t needle_length,          //
    sz_utf8_uncased_needle_metadata_t *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_uncased_find_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                                matched_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_uncased_violation_lasx(sz_cptr_t str, sz_size_t length) {
    return sz_utf8_uncased_violation_serial(str, length);
}

SZ_PUBLIC sz_ordering_t sz_utf8_uncased_order_lasx(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                            sz_size_t b_length) {
    return sz_utf8_uncased_order_serial(a, a_length, b, b_length);
}

#endif // SZ_USE_LASX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_UNCASED_LASX_H_
