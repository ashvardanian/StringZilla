/**
 *  @file include/stringzilla/find/skylake.h
 *  @author Ash Vardanian
 *  @date December 9, 2023
 *  @brief Skylake (AVX-512) backend for substring & byte-set search.
 *
 *  @sa c/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_SKYLAKE_H_
#define STRINGZILLA_FIND_SKYLAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/compare/skylake.h" // `sz_equal_skylake_`
#include "stringzilla/find/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  AVX512 implementation of the string search algorithms for Skylake and newer CPUs.
 *  Includes extensions: F, CD, ER, PF, VL, DQ, BW.
 *
 *  This is the "starting level" for the advanced algorithms using K-mask registers on x86. */
#if STRINGZILLA_ARCH_X8664_
#if STRINGZILLA_ARCH_X8664_SKYLAKE_
#if defined(__clang__) && STRINGZILLA_HAS_CLANG_EVEX512_
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2,lzcnt,evex512"))), \
                             apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2,lzcnt"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2", "lzcnt")
#endif

STRINGZILLA_INLINE sz_cptr_t sz_find_byte_skylake_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
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

    return STRINGZILLA_NULL_CHAR;
}

STRINGZILLA_INLINE sz_cptr_t sz_find_skylake_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                              sz_size_t needle_length) {

    // Empty needle matches at the start, like `strstr`.
    if (!needle_length) return haystack;
    if (haystack_length < needle_length) return STRINGZILLA_NULL_CHAR;
    if (needle_length == 1) return sz_find_byte_skylake_(haystack, haystack_length, needle);

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
                if (sz_equal_skylake_(haystack + potential_offset, needle, needle_length))
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
            if (needle_length <= 3 || sz_equal_skylake_(haystack + potential_offset, needle, needle_length))
                return haystack + potential_offset;
            matches_m64 &= matches_m64 - 1;
        }
    }
    return STRINGZILLA_NULL_CHAR;
}

/** Bit @c i set where @p bytes match at <tt>text + i + offsets[k]</tt> for every @c k, for the first
 *  @p starts of the 64 starts from @p text: three loads, masked only for fewer than 64 starts. */
STRINGZILLA_INLINE sz_u64_t sz_find_candidates_skylake_(sz_u8_t const *text, sz_size_t starts,
                                                        sz_size_t const offsets[3], sz_u8_t const bytes[3]) {
    __mmask64 const starts_m64 = sz_u64_mask_until_(starts);
    __m512i const first_u8x64 = starts == 64 ? _mm512_loadu_si512(text + offsets[0])
                                             : _mm512_maskz_loadu_epi8(starts_m64, text + offsets[0]);
    __m512i const second_u8x64 = starts == 64 ? _mm512_loadu_si512(text + offsets[1])
                                              : _mm512_maskz_loadu_epi8(starts_m64, text + offsets[1]);
    __m512i const third_u8x64 = starts == 64 ? _mm512_loadu_si512(text + offsets[2])
                                             : _mm512_maskz_loadu_epi8(starts_m64, text + offsets[2]);
    __mmask64 const first_m64 = _mm512_mask_cmpeq_epi8_mask(starts_m64, first_u8x64, _mm512_set1_epi8((char)bytes[0]));
    __mmask64 const second_m64 = _mm512_mask_cmpeq_epi8_mask(first_m64, second_u8x64, _mm512_set1_epi8((char)bytes[1]));
    return _cvtmask64_u64(_mm512_mask_cmpeq_epi8_mask(second_m64, third_u8x64, _mm512_set1_epi8((char)bytes[2])));
}

STRINGZILLA_INLINE sz_cptr_t sz_rfind_byte_skylake_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
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

    return STRINGZILLA_NULL_CHAR;
}

STRINGZILLA_INLINE sz_cptr_t sz_rfind_skylake_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                               sz_size_t needle_length) {

    // Empty needle matches at the end.
    if (!needle_length) return haystack + haystack_length;
    if (haystack_length < needle_length) return STRINGZILLA_NULL_CHAR;
    if (needle_length == 1) return sz_rfind_byte_skylake_(haystack, haystack_length, needle);

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
                sz_equal_skylake_(haystack + haystack_length - needle_length - potential_offset, needle, needle_length))
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
            if (needle_length <= 3 || sz_equal_skylake_(haystack + 64 - potential_offset - 1, needle, needle_length))
                return haystack + 64 - potential_offset - 1;
            sz_assert_((matches_m64 & (1ull << (63 - potential_offset))) != 0 &&
                       "The bit must be set before we squash it");
            matches_m64 &= ~(1ull << (63 - potential_offset));
        }
    }

    return STRINGZILLA_NULL_CHAR;
}

#if STRINGZILLA_TARGET_SKYLAKE

STRINGZILLA_API sz_status_t sz_find_byte_skylake(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                 sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_find_byte_skylake_(haystack, haystack_length, needle);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_find_skylake(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                            sz_size_t needle_length, sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_find_skylake_(haystack, haystack_length, needle, needle_length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_rfind_byte_skylake(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                  sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_rfind_byte_skylake_(haystack, haystack_length, needle);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_find_strings_skylake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_cptr_t needle, sz_size_t needle_length,
                                                    sz_size_t *positions, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    for (sz_size_t row = 0; !needle_length && row != count; ++row) positions[row] = 0;
    if (!needle_length) return sz_success_k;
    // Orders and accessors search string by string, their views fetched 64 at a time.
    sz_string_view_t views[64];
    sz_bool_t const by_row = (sz_bool_t)(order || strings->layout == sz_strings_accessor_k);
    for (sz_size_t row = 0; by_row && row != count; ++row) {
        if (row % 64 == 0 && order)
            sz_strings_get_(strings, order + row, sz_min_of_two(count - row, (sz_size_t)64), views);
        if (row % 64 == 0 && !order) sz_strings_get_block_(strings, row, views);
        sz_string_view_t const view = views[row % 64];
        sz_cptr_t const match = sz_find_skylake_(view.start, view.length, needle, needle_length);
        positions[row] = match ? (sz_size_t)(match - view.start) : STRINGZILLA_SIZE_MAX;
    }
    if (by_row || !count) return sz_success_k;

    sz_u8_t const *const data = (sz_u8_t const *)sz_strings_view_data_(strings);
    sz_size_t offsets[3];
    sz_locate_needle_anomalies_(needle, needle_length, &offsets[0], &offsets[1], &offsets[2]);
    sz_u8_t const bytes[3] = {(sz_u8_t)needle[offsets[0]], (sz_u8_t)needle[offsets[1]], (sz_u8_t)needle[offsets[2]]};
    sz_size_t const end = sz_strings_view_offset_(strings, count);
    sz_size_t row = 0, row_start = sz_strings_view_offset_(strings, 0), row_end = sz_strings_view_offset_(strings, 1);
    // A block of 64 starts begins at the first unsettled string, or right after the last block.
    for (sz_size_t block = row_start; row != count; block = sz_max_of_two(block + 64, row_start)) {
        sz_size_t const starts = end > block + offsets[2] ? sz_min_of_two(end - block - offsets[2], (sz_size_t)64) : 0;
        sz_u64_t candidates = sz_find_candidates_skylake_(data + block, starts, offsets, bytes);
        // Each step settles a string or drops one of its candidates, until one outruns the block.
        while (row != count && row_start < block + 64) {
            // The lanes of the string's starts in the block, none when the needle outgrows it.
            sz_size_t const first_lane = row_start > block ? row_start - block : 0;
            sz_size_t const end_lane = row_end - row_start < needle_length ? first_lane
                                                                           : row_end - needle_length + 1 - block;
            sz_u64_t const own = candidates & _cvtmask64_u64(sz_u64_clamp_mask_until_(end_lane)) &
                                 ~_cvtmask64_u64(sz_u64_mask_until_(first_lane));
            if (!own && end_lane > 64) break;
            sz_size_t const lane = (sz_size_t)_tzcnt_u64(own);
            candidates &= ~(own & (0 - own));
            if (own && needle_length > 3 && !sz_equal_skylake_((sz_cptr_t)data + block + lane, needle, needle_length))
                continue;
            positions[row] = own ? block + lane - row_start : STRINGZILLA_SIZE_MAX;
            row_start = row_end, ++row;
            row_end = sz_strings_view_offset_(strings, sz_min_of_two(row + 1, count));
        }
    }
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_rfind_skylake(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                             sz_size_t needle_length, sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_rfind_skylake_(haystack, haystack_length, needle, needle_length);
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_SKYLAKE

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_X8664_SKYLAKE_
#endif // STRINGZILLA_ARCH_X8664_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_SKYLAKE_H_
