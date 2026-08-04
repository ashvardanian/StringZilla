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
#if defined(__clang__) && SZ_CLANG_HAS_EVEX512_
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2,lzcnt,evex512"))), \
                             apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2,lzcnt"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2", "lzcnt")
#endif

SZ_API_COMPTIME sz_cptr_t sz_find_byte_skylake(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    __mmask64 matches_mask_m64;
    sz_u512_vec_t haystack_vec, needle_vec;
    needle_vec.zmm = _mm512_set1_epi8(needle[0]);

    while (haystack_length >= 64) {
        haystack_vec.zmm = _mm512_loadu_si512(haystack);
        matches_mask_m64 = _mm512_cmpeq_epi8_mask(haystack_vec.zmm, needle_vec.zmm);
        if (matches_mask_m64) return haystack + (int)_tzcnt_u64(matches_mask_m64);
        haystack += 64, haystack_length -= 64;
    }

    if (haystack_length) {
        __mmask64 load_mask_m64 = sz_u64_mask_until_(haystack_length);
        haystack_vec.zmm = _mm512_maskz_loadu_epi8(load_mask_m64, haystack);
        // Reuse the same `load_mask_m64` variable to find the bit that doesn't match
        matches_mask_m64 = _mm512_mask_cmpeq_epu8_mask(load_mask_m64, haystack_vec.zmm, needle_vec.zmm);
        if (matches_mask_m64) return haystack + (int)_tzcnt_u64(matches_mask_m64);
    }

    return SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_cptr_t sz_find_skylake(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length) {

    // Empty needle matches at the start, like `strstr`.
    if (!needle_length) return haystack;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_find_byte_skylake(haystack, haystack_length, needle);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into ZMM registers.
    __mmask64 matches_m64;
    __mmask64 load_mask_m64;
    sz_u512_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.zmm = _mm512_set1_epi8(needle[offset_first]);
    n_mid_vec.zmm = _mm512_set1_epi8(needle[offset_mid]);
    n_last_vec.zmm = _mm512_set1_epi8(needle[offset_last]);

    // Scan through the string.
    // We have several optimized versions of the algorithm for shorter strings,
    // but they all mimic the default case for unbounded length needles
    if (needle_length >= 64) {
        for (; haystack_length >= needle_length + 64; haystack += 64, haystack_length -= 64) {
            h_first_vec.zmm = _mm512_loadu_si512(haystack + offset_first);
            h_mid_vec.zmm = _mm512_loadu_si512(haystack + offset_mid);
            h_last_vec.zmm = _mm512_loadu_si512(haystack + offset_last);
            matches_m64 = _kand_mask64( //
                _kand_mask64(           // Intersect the masks
                    _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm),
                    _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm)),
                _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm));
            while (matches_m64) {
                int potential_offset = (int)_tzcnt_u64(matches_m64);
                if (sz_equal_skylake(haystack + potential_offset, needle, needle_length))
                    return haystack + potential_offset;
                matches_m64 &= matches_m64 - 1;
            }

            // TODO: If the last character contains a bad byte, we can reposition the start of the next iteration.
            // This will be very helpful for very long needles.
        }
    }
    // If there are only 2 or 3 characters in the needle, we don't even need the nested loop.
    else if (needle_length <= 3) {
        for (; haystack_length >= needle_length + 64; haystack += 64, haystack_length -= 64) {
            h_first_vec.zmm = _mm512_loadu_si512(haystack + offset_first);
            h_mid_vec.zmm = _mm512_loadu_si512(haystack + offset_mid);
            h_last_vec.zmm = _mm512_loadu_si512(haystack + offset_last);
            matches_m64 = _kand_mask64( //
                _kand_mask64(           // Intersect the masks
                    _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm),
                    _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm)),
                _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm));
            if (matches_m64) return haystack + (int)_tzcnt_u64(matches_m64);
        }
    }
    // If the needle is smaller than the size of the ZMM register, we can use masked comparisons
    // to avoid the the inner-most nested loop and compare the entire needle against a haystack
    // slice in 3 CPU cycles.
    else {
        __mmask64 needle_mask_m64 = sz_u64_mask_until_(needle_length);
        sz_u512_vec_t needle_full_vec, h_full_vec;
        needle_full_vec.zmm = _mm512_maskz_loadu_epi8(needle_mask_m64, needle);
        for (; haystack_length >= needle_length + 64; haystack += 64, haystack_length -= 64) {
            h_first_vec.zmm = _mm512_loadu_si512(haystack + offset_first);
            h_mid_vec.zmm = _mm512_loadu_si512(haystack + offset_mid);
            h_last_vec.zmm = _mm512_loadu_si512(haystack + offset_last);
            matches_m64 = _kand_mask64( //
                _kand_mask64(           // Intersect the masks
                    _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm),
                    _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm)),
                _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm));
            while (matches_m64) {
                int potential_offset = (int)_tzcnt_u64(matches_m64);
                h_full_vec.zmm = _mm512_maskz_loadu_epi8(needle_mask_m64, haystack + potential_offset);
                if (_mm512_mask_cmpneq_epi8_mask(needle_mask_m64, h_full_vec.zmm, needle_full_vec.zmm) == 0)
                    return haystack + potential_offset;
                matches_m64 &= matches_m64 - 1;
            }
        }
    }

    // The "tail" of the function uses masked loads to process the remaining bytes.
    {
        load_mask_m64 = sz_u64_mask_until_(haystack_length - needle_length + 1);
        h_first_vec.zmm = _mm512_maskz_loadu_epi8(load_mask_m64, haystack + offset_first);
        h_mid_vec.zmm = _mm512_maskz_loadu_epi8(load_mask_m64, haystack + offset_mid);
        h_last_vec.zmm = _mm512_maskz_loadu_epi8(load_mask_m64, haystack + offset_last);
        matches_m64 = _kand_mask64( //
            _kand_mask64(           // Intersect the masks
                _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm),
                _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm)),
            _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm));
        matches_m64 &= load_mask_m64;
        while (matches_m64) {
            int potential_offset = (int)_tzcnt_u64(matches_m64);
            if (needle_length <= 3 || sz_equal_skylake(haystack + potential_offset, needle, needle_length))
                return haystack + potential_offset;
            matches_m64 &= matches_m64 - 1;
        }
    }
    return SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_skylake(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    __mmask64 matches_mask_m64;
    sz_u512_vec_t haystack_vec, needle_vec;
    needle_vec.zmm = _mm512_set1_epi8(needle[0]);

    while (haystack_length >= 64) {
        haystack_vec.zmm = _mm512_loadu_si512(haystack + haystack_length - 64);
        matches_mask_m64 = _mm512_cmpeq_epi8_mask(haystack_vec.zmm, needle_vec.zmm);
        if (matches_mask_m64) return haystack + haystack_length - 1 - (int)_lzcnt_u64(matches_mask_m64);
        haystack_length -= 64;
    }

    if (haystack_length) {
        __mmask64 load_mask_m64 = sz_u64_mask_until_(haystack_length);
        haystack_vec.zmm = _mm512_maskz_loadu_epi8(load_mask_m64, haystack);
        // Reuse the same `load_mask_m64` variable to find the bit that doesn't match
        matches_mask_m64 = _mm512_mask_cmpeq_epu8_mask(load_mask_m64, haystack_vec.zmm, needle_vec.zmm);
        if (matches_mask_m64) return haystack + 64 - (int)_lzcnt_u64(matches_mask_m64) - 1;
    }

    return SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_skylake(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                           sz_size_t needle_length) {

    // Empty needle matches at the end.
    if (!needle_length) return haystack + haystack_length;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_rfind_byte_skylake(haystack, haystack_length, needle);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into ZMM registers.
    __mmask64 load_mask_m64;
    __mmask64 matches_m64;
    sz_u512_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.zmm = _mm512_set1_epi8(needle[offset_first]);
    n_mid_vec.zmm = _mm512_set1_epi8(needle[offset_mid]);
    n_last_vec.zmm = _mm512_set1_epi8(needle[offset_last]);

    // Scan through the string.
    sz_cptr_t haystack_cursor;
    for (; haystack_length >= needle_length + 64; haystack_length -= 64) {
        haystack_cursor = haystack + haystack_length - needle_length - 64 + 1;
        h_first_vec.zmm = _mm512_loadu_si512(haystack_cursor + offset_first);
        h_mid_vec.zmm = _mm512_loadu_si512(haystack_cursor + offset_mid);
        h_last_vec.zmm = _mm512_loadu_si512(haystack_cursor + offset_last);
        matches_m64 = _kand_mask64( //
            _kand_mask64(           // Intersect the masks
                _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm),
                _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm)),
            _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm));
        while (matches_m64) {
            int potential_offset = (int)_lzcnt_u64(matches_m64);
            if (needle_length <= 3 ||
                sz_equal_skylake(haystack + haystack_length - needle_length - potential_offset, needle, needle_length))
                return haystack + haystack_length - needle_length - potential_offset;
            sz_assert_((matches_m64 & (1ull << (63 - potential_offset))) != 0 &&
                       "The bit must be set before we squash it");
            matches_m64 &= ~(1ull << (63 - potential_offset));
        }
    }

    // The "tail" of the function uses masked loads to process the remaining bytes.
    {
        load_mask_m64 = sz_u64_mask_until_(haystack_length - needle_length + 1);
        h_first_vec.zmm = _mm512_maskz_loadu_epi8(load_mask_m64, haystack + offset_first);
        h_mid_vec.zmm = _mm512_maskz_loadu_epi8(load_mask_m64, haystack + offset_mid);
        h_last_vec.zmm = _mm512_maskz_loadu_epi8(load_mask_m64, haystack + offset_last);
        matches_m64 = _kand_mask64( //
            _kand_mask64(           // Intersect the masks
                _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm),
                _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm)),
            _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm));
        matches_m64 &= load_mask_m64;
        while (matches_m64) {
            int potential_offset = (int)_lzcnt_u64(matches_m64);
            if (needle_length <= 3 || sz_equal_skylake(haystack + 64 - potential_offset - 1, needle, needle_length))
                return haystack + 64 - potential_offset - 1;
            sz_assert_((matches_m64 & (1ull << (63 - potential_offset))) != 0 &&
                       "The bit must be set before we squash it");
            matches_m64 &= ~(1ull << (63 - potential_offset));
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
