/**
 *  @brief IBM Power (VSX) case-insensitive UTF-8 search, comparison & invariance backend.
 *  @file include/stringzilla/utf8_case_insensitive/powervsx.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_case_insensitive.h
 */
#ifndef STRINGZILLA_UTF8_CASE_INSENSITIVE_POWERVSX_H_
#define STRINGZILLA_UTF8_CASE_INSENSITIVE_POWERVSX_H_

#include "stringzilla/utf8_case_insensitive/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

/*  This ISA has no dedicated case-insensitive UTF-8 kernels yet; it delegates to the serial
 *  scaffolding so the per-backend symbol set stays uniform across all targets. */

SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_powervsx( //
    sz_cptr_t haystack, sz_size_t haystack_length,          //
    sz_cptr_t needle, sz_size_t needle_length,              //
    sz_utf8_case_insensitive_needle_metadata_t *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_case_insensitive_find_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                                matched_length);
}

SZ_PUBLIC sz_bool_t sz_utf8_case_invariant_powervsx(sz_cptr_t str, sz_size_t length) {
    return sz_utf8_case_invariant_serial(str, length);
}

SZ_PUBLIC sz_ordering_t sz_utf8_case_insensitive_order_powervsx(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                                sz_size_t b_length) {
    return sz_utf8_case_insensitive_order_serial(a, a_length, b, b_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_POWERVSX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_CASE_INSENSITIVE_POWERVSX_H_
