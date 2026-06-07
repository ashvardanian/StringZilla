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

SZ_PUBLIC sz_cptr_t sz_find_byte_westmere(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    int mask;
    sz_u128_vec_t h_vec, n_vec;
    n_vec.xmm = _mm_set1_epi8(n[0]);

    while (h_length >= 16) {
        h_vec.xmm = _mm_lddqu_si128((__m128i const *)h);
        mask = _mm_movemask_epi8(_mm_cmpeq_epi8(h_vec.xmm, n_vec.xmm));
        if (mask) return h + sz_u32_ctz(mask);
        h += 16, h_length -= 16;
    }

    return sz_find_byte_serial(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byte_westmere(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    int mask;
    sz_u128_vec_t h_vec, n_vec;
    n_vec.xmm = _mm_set1_epi8(n[0]);

    while (h_length >= 16) {
        h_vec.xmm = _mm_lddqu_si128((__m128i const *)(h + h_length - 16));
        mask = _mm_movemask_epi8(_mm_cmpeq_epi8(h_vec.xmm, n_vec.xmm));
        if (mask) return h + h_length - 1 - (sz_u32_clz(mask) - 16);
        h_length -= 16;
    }

    return sz_rfind_byte_serial(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_find_westmere(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_find_byte_westmere(h, h_length, n);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into XMM registers.
    sz_u32_vec_t matches_vec;
    sz_u128_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.xmm = _mm_set1_epi8(n[offset_first]);
    n_mid_vec.xmm = _mm_set1_epi8(n[offset_mid]);
    n_last_vec.xmm = _mm_set1_epi8(n[offset_last]);

    // Scan through the string.
    for (; h_length >= n_length + 16; h += 16, h_length -= 16) {
        h_first_vec.xmm = _mm_lddqu_si128((__m128i const *)(h + offset_first));
        h_mid_vec.xmm = _mm_lddqu_si128((__m128i const *)(h + offset_mid));
        h_last_vec.xmm = _mm_lddqu_si128((__m128i const *)(h + offset_last));
        matches_vec.i32 = //
            _mm_movemask_epi8(_mm_cmpeq_epi8(h_first_vec.xmm, n_first_vec.xmm)) &
            _mm_movemask_epi8(_mm_cmpeq_epi8(h_mid_vec.xmm, n_mid_vec.xmm)) &
            _mm_movemask_epi8(_mm_cmpeq_epi8(h_last_vec.xmm, n_last_vec.xmm));
        while (matches_vec.u32) {
            int potential_offset = sz_u32_ctz(matches_vec.u32);
            if (sz_equal_westmere(h + potential_offset, n, n_length)) return h + potential_offset;
            matches_vec.u32 &= matches_vec.u32 - 1;
        }
    }

    return sz_find_serial(h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_rfind_westmere(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    // This almost never fires, but it's better to be safe than sorry.
    // if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    // if (n_length == 1) return sz_rfind_byte_westmere(h, h_length, n);
    //
    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into XMM registers.
    sz_u32_vec_t matches_vec;
    sz_u128_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.xmm = _mm_set1_epi8(n[offset_first]);
    n_mid_vec.xmm = _mm_set1_epi8(n[offset_mid]);
    n_last_vec.xmm = _mm_set1_epi8(n[offset_last]);

    // Scan through the string.
    sz_cptr_t h_reversed;
    for (; h_length >= n_length + 16; h_length -= 16) {
        h_reversed = h + h_length - n_length - 16 + 1;
        h_first_vec.xmm = _mm_lddqu_si128((__m128i const *)(h_reversed + offset_first));
        h_mid_vec.xmm = _mm_lddqu_si128((__m128i const *)(h_reversed + offset_mid));
        h_last_vec.xmm = _mm_lddqu_si128((__m128i const *)(h_reversed + offset_last));
        matches_vec.i32 = //
            _mm_movemask_epi8(_mm_cmpeq_epi8(h_first_vec.xmm, n_first_vec.xmm)) &
            _mm_movemask_epi8(_mm_cmpeq_epi8(h_mid_vec.xmm, n_mid_vec.xmm)) &
            _mm_movemask_epi8(_mm_cmpeq_epi8(h_last_vec.xmm, n_last_vec.xmm));
        while (matches_vec.u32) {
            int potential_offset = sz_u32_clz(matches_vec.u32) - 16;
            if (sz_equal_westmere(h + h_length - n_length - potential_offset, n, n_length))
                return h + h_length - n_length - potential_offset;
            matches_vec.u32 &= ~(1u << (15 - potential_offset));
        }
    }

    return sz_rfind_serial(h, h_length, n, n_length);
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
