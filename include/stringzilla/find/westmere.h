/**
 *  @brief Westmere (SSE4.2) backend for substring & byte-set search.
 *  @file include/stringzilla/find/westmere.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_WESTMERE_H_
#define STRINGZILLA_FIND_WESTMERE_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/find/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  SSE implementation of the string search algorithms for Westmere processors and newer.
 *  Very minimalistic (compared to AVX-512), but still faster than the serial implementation.
 */
#if SZ_USE_WESTMERE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sse4.2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("sse4.2")
#endif

SZ_PUBLIC sz_cptr_t sz_find_byte_westmere(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    int matches_mask;
    sz_u128_vec_t haystack_vec, needle_vec;
    needle_vec.xmm = _mm_set1_epi8(needle[0]);

    while (haystack_length >= 16) {
        haystack_vec.xmm = _mm_lddqu_si128((__m128i const *)haystack);
        matches_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(haystack_vec.xmm, needle_vec.xmm));
        if (matches_mask) return haystack + sz_u32_ctz(matches_mask);
        haystack += 16, haystack_length -= 16;
    }

    return sz_find_byte_serial(haystack, haystack_length, needle);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byte_westmere(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    int matches_mask;
    sz_u128_vec_t haystack_vec, needle_vec;
    needle_vec.xmm = _mm_set1_epi8(needle[0]);

    while (haystack_length >= 16) {
        haystack_vec.xmm = _mm_lddqu_si128((__m128i const *)(haystack + haystack_length - 16));
        matches_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(haystack_vec.xmm, needle_vec.xmm));
        if (matches_mask) return haystack + haystack_length - 1 - (sz_u32_clz(matches_mask) - 16);
        haystack_length -= 16;
    }

    return sz_rfind_byte_serial(haystack, haystack_length, needle);
}

SZ_PUBLIC sz_cptr_t sz_find_westmere(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                     sz_size_t needle_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (haystack_length < needle_length || !needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_find_byte_westmere(haystack, haystack_length, needle);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into XMM registers.
    sz_u32_vec_t matches_vec;
    sz_u128_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.xmm = _mm_set1_epi8(needle[offset_first]);
    n_mid_vec.xmm = _mm_set1_epi8(needle[offset_mid]);
    n_last_vec.xmm = _mm_set1_epi8(needle[offset_last]);

    // Scan through the string.
    for (; haystack_length >= needle_length + 16; haystack += 16, haystack_length -= 16) {
        h_first_vec.xmm = _mm_lddqu_si128((__m128i const *)(haystack + offset_first));
        h_mid_vec.xmm = _mm_lddqu_si128((__m128i const *)(haystack + offset_mid));
        h_last_vec.xmm = _mm_lddqu_si128((__m128i const *)(haystack + offset_last));
        matches_vec.i32 = //
            _mm_movemask_epi8(_mm_cmpeq_epi8(h_first_vec.xmm, n_first_vec.xmm)) &
            _mm_movemask_epi8(_mm_cmpeq_epi8(h_mid_vec.xmm, n_mid_vec.xmm)) &
            _mm_movemask_epi8(_mm_cmpeq_epi8(h_last_vec.xmm, n_last_vec.xmm));
        while (matches_vec.u32) {
            int potential_offset = sz_u32_ctz(matches_vec.u32);
            if (sz_equal_westmere(haystack + potential_offset, needle, needle_length))
                return haystack + potential_offset;
            matches_vec.u32 &= matches_vec.u32 - 1;
        }
    }

    return sz_find_serial(haystack, haystack_length, needle, needle_length);
}

SZ_PUBLIC sz_cptr_t sz_rfind_westmere(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                      sz_size_t needle_length) {
    // This almost never fires, but it's better to be safe than sorry.
    // if (haystack_length < needle_length || !needle_length) return SZ_NULL_CHAR;
    // if (needle_length == 1) return sz_rfind_byte_westmere(haystack, haystack_length, needle);
    //
    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into XMM registers.
    sz_u32_vec_t matches_vec;
    sz_u128_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.xmm = _mm_set1_epi8(needle[offset_first]);
    n_mid_vec.xmm = _mm_set1_epi8(needle[offset_mid]);
    n_last_vec.xmm = _mm_set1_epi8(needle[offset_last]);

    // Scan through the string.
    sz_cptr_t haystack_cursor;
    for (; haystack_length >= needle_length + 16; haystack_length -= 16) {
        haystack_cursor = haystack + haystack_length - needle_length - 16 + 1;
        h_first_vec.xmm = _mm_lddqu_si128((__m128i const *)(haystack_cursor + offset_first));
        h_mid_vec.xmm = _mm_lddqu_si128((__m128i const *)(haystack_cursor + offset_mid));
        h_last_vec.xmm = _mm_lddqu_si128((__m128i const *)(haystack_cursor + offset_last));
        matches_vec.i32 = //
            _mm_movemask_epi8(_mm_cmpeq_epi8(h_first_vec.xmm, n_first_vec.xmm)) &
            _mm_movemask_epi8(_mm_cmpeq_epi8(h_mid_vec.xmm, n_mid_vec.xmm)) &
            _mm_movemask_epi8(_mm_cmpeq_epi8(h_last_vec.xmm, n_last_vec.xmm));
        while (matches_vec.u32) {
            int potential_offset = sz_u32_clz(matches_vec.u32) - 16;
            if (sz_equal_westmere(haystack + haystack_length - needle_length - potential_offset, needle, needle_length))
                return haystack + haystack_length - needle_length - potential_offset;
            matches_vec.u32 &= ~(1u << (15 - potential_offset));
        }
    }

    return sz_rfind_serial(haystack, haystack_length, needle, needle_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_WESTMERE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_WESTMERE_H_
