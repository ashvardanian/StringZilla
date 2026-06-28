/**
 *  @brief SVE2 backend for the single-pass Unicode normalizer (NFD / NFC / NFKD / NFKC).
 *  @file include/stringzilla/utf8_norm/sve2.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_norm.h
 *
 *  The normalizer scanner is a 64-entry table lookup behind an ASCII/inert gate. SVE2's headline
 *  instruction `svmatch` matches against a 16-byte set, which does not help a 64-entry LUT scanner -
 *  there is no scalable equivalent that subsumes the four `svtbl_u8` quadrants. So the SVE2 scan
 *  primitive `sz_utf8_norm_classify_sve2_` simply reuses the proven, vector-length-agnostic SVE body
 *  `sz_utf8_norm_classify_sve_`. The two public entry points exist so SVE2-tier dispatch resolves to
 *  named `_sve2` symbols, but the hot loop is byte-for-byte the SVE kernel.
 */
#ifndef STRINGZILLA_UTF8_NORM_SVE2_H_
#define STRINGZILLA_UTF8_NORM_SVE2_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_norm/serial.h"
#include "stringzilla/utf8_norm/sve.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SVE2
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve2")
#endif

/** @brief Scan primitive (SVE2): first byte beginning a non-inert codepoint for @p form, else NULL.
 *         Reuses the SVE kernel verbatim - `svmatch` does not subsume the 64-entry LUT lookup. */
SZ_INTERNAL sz_cptr_t sz_utf8_norm_classify_sve2_(sz_cptr_t text, sz_size_t length, sz_normal_form_t form) {
    return sz_utf8_norm_classify_sve_(text, length, form);
}

SZ_PUBLIC sz_size_t sz_utf8_norm_sve2(sz_cptr_t source, sz_size_t length, sz_normal_form_t form, sz_ptr_t destination) {
    return sz_utf8_norm_engine_(source, length, form, destination, &sz_utf8_norm_classify_sve2_);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_denormalized_sve2(sz_cptr_t source, sz_size_t length, sz_normal_form_t form) {
    return sz_utf8_find_denormalized_engine_(source, length, form, &sz_utf8_norm_classify_sve2_);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE2

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_NORM_SVE2_H_
