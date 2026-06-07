/**
 *  @brief Skylake (AVX-512) backend for substring & byte-set search.
 *  @file include/stringzilla/find/skylake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_SKYLAKE_H_
#define STRINGZILLA_FIND_SKYLAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/find/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  AVX512 implementation of the string search algorithms for Skylake and newer CPUs.
 *  Includes extensions: F, CD, ER, PF, VL, DQ, BW.
 *
 *  This is the "starting level" for the advanced algorithms using K-mask registers on x86.
 */
#if SZ_USE_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2")
#endif

SZ_PUBLIC sz_cptr_t sz_find_byte_skylake(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    __mmask64 mask;
    sz_u512_vec_t h_vec, n_vec;
    n_vec.zmm = _mm512_set1_epi8(n[0]);

    while (h_length >= 64) {
        h_vec.zmm = _mm512_loadu_si512(h);
        mask = _mm512_cmpeq_epi8_mask(h_vec.zmm, n_vec.zmm);
        if (mask) return h + sz_u64_ctz(mask);
        h += 64, h_length -= 64;
    }

    if (h_length) {
        mask = sz_u64_mask_until_(h_length);
        h_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        // Reuse the same `mask` variable to find the bit that doesn't match
        mask = _mm512_mask_cmpeq_epu8_mask(mask, h_vec.zmm, n_vec.zmm);
        if (mask) return h + sz_u64_ctz(mask);
    }

    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_find_skylake(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_find_byte_skylake(h, h_length, n);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into ZMM registers.
    __mmask64 matches;
    __mmask64 mask;
    sz_u512_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.zmm = _mm512_set1_epi8(n[offset_first]);
    n_mid_vec.zmm = _mm512_set1_epi8(n[offset_mid]);
    n_last_vec.zmm = _mm512_set1_epi8(n[offset_last]);

    // Scan through the string.
    // We have several optimized versions of the algorithm for shorter strings,
    // but they all mimic the default case for unbounded length needles
    if (n_length >= 64) {
        for (; h_length >= n_length + 64; h += 64, h_length -= 64) {
            h_first_vec.zmm = _mm512_loadu_si512(h + offset_first);
            h_mid_vec.zmm = _mm512_loadu_si512(h + offset_mid);
            h_last_vec.zmm = _mm512_loadu_si512(h + offset_last);
            matches = _kand_mask64( //
                _kand_mask64(       // Intersect the masks
                    _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm),
                    _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm)),
                _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm));
            while (matches) {
                int potential_offset = sz_u64_ctz(matches);
                if (sz_equal_skylake(h + potential_offset, n, n_length)) return h + potential_offset;
                matches &= matches - 1;
            }

            // TODO: If the last character contains a bad byte, we can reposition the start of the next iteration.
            // This will be very helpful for very long needles.
        }
    }
    // If there are only 2 or 3 characters in the needle, we don't even need the nested loop.
    else if (n_length <= 3) {
        for (; h_length >= n_length + 64; h += 64, h_length -= 64) {
            h_first_vec.zmm = _mm512_loadu_si512(h + offset_first);
            h_mid_vec.zmm = _mm512_loadu_si512(h + offset_mid);
            h_last_vec.zmm = _mm512_loadu_si512(h + offset_last);
            matches = _kand_mask64( //
                _kand_mask64(       // Intersect the masks
                    _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm),
                    _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm)),
                _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm));
            if (matches) return h + sz_u64_ctz(matches);
        }
    }
    // If the needle is smaller than the size of the ZMM register, we can use masked comparisons
    // to avoid the the inner-most nested loop and compare the entire needle against a haystack
    // slice in 3 CPU cycles.
    else {
        __mmask64 n_mask = sz_u64_mask_until_(n_length);
        sz_u512_vec_t n_full_vec, h_full_vec;
        n_full_vec.zmm = _mm512_maskz_loadu_epi8(n_mask, n);
        for (; h_length >= n_length + 64; h += 64, h_length -= 64) {
            h_first_vec.zmm = _mm512_loadu_si512(h + offset_first);
            h_mid_vec.zmm = _mm512_loadu_si512(h + offset_mid);
            h_last_vec.zmm = _mm512_loadu_si512(h + offset_last);
            matches = _kand_mask64( //
                _kand_mask64(       // Intersect the masks
                    _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm),
                    _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm)),
                _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm));
            while (matches) {
                int potential_offset = sz_u64_ctz(matches);
                h_full_vec.zmm = _mm512_maskz_loadu_epi8(n_mask, h + potential_offset);
                if (_mm512_mask_cmpneq_epi8_mask(n_mask, h_full_vec.zmm, n_full_vec.zmm) == 0)
                    return h + potential_offset;
                matches &= matches - 1;
            }
        }
    }

    // The "tail" of the function uses masked loads to process the remaining bytes.
    {
        mask = sz_u64_mask_until_(h_length - n_length + 1);
        h_first_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + offset_first);
        h_mid_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + offset_mid);
        h_last_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + offset_last);
        matches = _kand_mask64( //
            _kand_mask64(       // Intersect the masks
                _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm),
                _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm)),
            _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm));
        matches &= mask;
        while (matches) {
            int potential_offset = sz_u64_ctz(matches);
            if (n_length <= 3 || sz_equal_skylake(h + potential_offset, n, n_length)) return h + potential_offset;
            matches &= matches - 1;
        }
    }
    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_rfind_byte_skylake(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    __mmask64 mask;
    sz_u512_vec_t h_vec, n_vec;
    n_vec.zmm = _mm512_set1_epi8(n[0]);

    while (h_length >= 64) {
        h_vec.zmm = _mm512_loadu_si512(h + h_length - 64);
        mask = _mm512_cmpeq_epi8_mask(h_vec.zmm, n_vec.zmm);
        if (mask) return h + h_length - 1 - sz_u64_clz(mask);
        h_length -= 64;
    }

    if (h_length) {
        mask = sz_u64_mask_until_(h_length);
        h_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        // Reuse the same `mask` variable to find the bit that doesn't match
        mask = _mm512_mask_cmpeq_epu8_mask(mask, h_vec.zmm, n_vec.zmm);
        if (mask) return h + 64 - sz_u64_clz(mask) - 1;
    }

    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_rfind_skylake(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_rfind_byte_skylake(h, h_length, n);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into ZMM registers.
    __mmask64 mask;
    __mmask64 matches;
    sz_u512_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.zmm = _mm512_set1_epi8(n[offset_first]);
    n_mid_vec.zmm = _mm512_set1_epi8(n[offset_mid]);
    n_last_vec.zmm = _mm512_set1_epi8(n[offset_last]);

    // Scan through the string.
    sz_cptr_t h_reversed;
    for (; h_length >= n_length + 64; h_length -= 64) {
        h_reversed = h + h_length - n_length - 64 + 1;
        h_first_vec.zmm = _mm512_loadu_si512(h_reversed + offset_first);
        h_mid_vec.zmm = _mm512_loadu_si512(h_reversed + offset_mid);
        h_last_vec.zmm = _mm512_loadu_si512(h_reversed + offset_last);
        matches = _kand_mask64( //
            _kand_mask64(       // Intersect the masks
                _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm),
                _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm)),
            _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm));
        while (matches) {
            int potential_offset = sz_u64_clz(matches);
            if (n_length <= 3 || sz_equal_skylake(h + h_length - n_length - potential_offset, n, n_length))
                return h + h_length - n_length - potential_offset;
            sz_assert_((matches & (1ull << (63 - potential_offset))) != 0 && "The bit must be set before we squash it");
            matches &= ~(1ull << (63 - potential_offset));
        }
    }

    // The "tail" of the function uses masked loads to process the remaining bytes.
    {
        mask = sz_u64_mask_until_(h_length - n_length + 1);
        h_first_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + offset_first);
        h_mid_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + offset_mid);
        h_last_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + offset_last);
        matches = _kand_mask64( //
            _kand_mask64(       // Intersect the masks
                _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm),
                _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm)),
            _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm));
        matches &= mask;
        while (matches) {
            int potential_offset = sz_u64_clz(matches);
            if (n_length <= 3 || sz_equal_skylake(h + 64 - potential_offset - 1, n, n_length))
                return h + 64 - potential_offset - 1;
            sz_assert_((matches & (1ull << (63 - potential_offset))) != 0 && "The bit must be set before we squash it");
            matches &= ~(1ull << (63 - potential_offset));
        }
    }

    return SZ_NULL_CHAR;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SKYLAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_SKYLAKE_H_
