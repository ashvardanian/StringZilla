/**
 *  @file include/stringzilla/find/haswell.h
 *  @author Ash Vardanian
 *  @date August 14, 2020
 *  @brief Haswell (AVX2) backend for substring & byte-set search.
 *
 *  @sa c/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_HASWELL_H_
#define STRINGZILLA_FIND_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/compare/haswell.h" // `sz_equal_haswell_`
#include "stringzilla/find/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  AVX2 implementation of the string search algorithms for Haswell processors and newer.
 *  Very minimalistic (compared to AVX-512), but still faster than the serial implementation. */
#if STRINGZILLA_ARCH_X8664_
#if STRINGZILLA_ARCH_X8664_HASWELL_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,lzcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "bmi", "bmi2", "lzcnt")
#endif

STRINGZILLA_INLINE sz_cptr_t sz_find_byte_haswell_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    int matches_mask;
    sz_u256_vec_t haystack_vec, needle_vec;
    needle_vec.ymm = _mm256_set1_epi8(needle[0]);

    while (haystack_length >= 32) {
        haystack_vec.ymm = _mm256_lddqu_si256((__m256i const *)haystack);
        matches_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(haystack_vec.ymm, needle_vec.ymm));
        if (matches_mask) return haystack + (int)_tzcnt_u32(matches_mask);
        haystack += 32, haystack_length -= 32;
    }

    return sz_find_byte_serial_(haystack, haystack_length, needle);
}

STRINGZILLA_INLINE sz_cptr_t sz_rfind_byte_haswell_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    int matches_mask;
    sz_u256_vec_t haystack_vec, needle_vec;
    needle_vec.ymm = _mm256_set1_epi8(needle[0]);

    while (haystack_length >= 32) {
        haystack_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack + haystack_length - 32));
        matches_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(haystack_vec.ymm, needle_vec.ymm));
        if (matches_mask) return haystack + haystack_length - 1 - (int)_lzcnt_u32(matches_mask);
        haystack_length -= 32;
    }

    return sz_rfind_byte_serial_(haystack, haystack_length, needle);
}

STRINGZILLA_INLINE sz_cptr_t sz_find_haswell_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                              sz_size_t needle_length) {

    // Empty needle matches at the start, like `strstr`.
    if (!needle_length) return haystack;
    if (haystack_length < needle_length) return STRINGZILLA_NULL_CHAR;
    if (needle_length == 1) return sz_find_byte_haswell_(haystack, haystack_length, needle);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into YMM registers.
    sz_u32_vec_t matches_vec;
    sz_u256_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.ymm = _mm256_set1_epi8(needle[offset_first]);
    n_mid_vec.ymm = _mm256_set1_epi8(needle[offset_mid]);
    n_last_vec.ymm = _mm256_set1_epi8(needle[offset_last]);

    // Scan through the string.
    for (; haystack_length >= needle_length + 32; haystack += 32, haystack_length -= 32) {
        h_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack + offset_first));
        h_mid_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack + offset_mid));
        h_last_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack + offset_last));
        matches_vec.i32 = //
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_first_vec.ymm, n_first_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_mid_vec.ymm, n_mid_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_last_vec.ymm, n_last_vec.ymm));
        while (matches_vec.u32) {
            int potential_offset = (int)_tzcnt_u32(matches_vec.u32);
            if (sz_equal_haswell_(haystack + potential_offset, needle, needle_length))
                return haystack + potential_offset;
            matches_vec.u32 &= matches_vec.u32 - 1;
        }
    }

    return sz_find_serial_(haystack, haystack_length, needle, needle_length);
}

/** Bit @c i set where @p bytes match at <tt>text + i + offsets[k]</tt> for every @c k, for the 64 starts
 *  from @p text: two steps of three loads. */
STRINGZILLA_INLINE sz_u64_t sz_find_candidates_haswell_(sz_u8_t const *text, sz_size_t const offsets[3],
                                                        sz_u8_t const bytes[3]) {
    __m256i const first_u8x32 = _mm256_set1_epi8((char)bytes[0]);
    __m256i const second_u8x32 = _mm256_set1_epi8((char)bytes[1]);
    __m256i const third_u8x32 = _mm256_set1_epi8((char)bytes[2]);
    __m256i const low_u8x32 = _mm256_and_si256(
        _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_lddqu_si256((__m256i const *)(text + offsets[0])), first_u8x32),
                         _mm256_cmpeq_epi8(_mm256_lddqu_si256((__m256i const *)(text + offsets[1])), second_u8x32)),
        _mm256_cmpeq_epi8(_mm256_lddqu_si256((__m256i const *)(text + offsets[2])), third_u8x32));
    __m256i const high_u8x32 = _mm256_and_si256(
        _mm256_and_si256(
            _mm256_cmpeq_epi8(_mm256_lddqu_si256((__m256i const *)(text + offsets[0] + 32)), first_u8x32),
            _mm256_cmpeq_epi8(_mm256_lddqu_si256((__m256i const *)(text + offsets[1] + 32)), second_u8x32)),
        _mm256_cmpeq_epi8(_mm256_lddqu_si256((__m256i const *)(text + offsets[2] + 32)), third_u8x32));
    return (sz_u64_t)(sz_u32_t)_mm256_movemask_epi8(low_u8x32) |
           ((sz_u64_t)(sz_u32_t)_mm256_movemask_epi8(high_u8x32) << 32);
}

STRINGZILLA_INLINE sz_cptr_t sz_rfind_haswell_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                               sz_size_t needle_length) {

    // Empty needle matches at the end.
    if (!needle_length) return haystack + haystack_length;
    if (haystack_length < needle_length) return STRINGZILLA_NULL_CHAR;
    if (needle_length == 1) return sz_rfind_byte_haswell_(haystack, haystack_length, needle);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into YMM registers.
    sz_u32_vec_t matches_vec;
    sz_u256_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.ymm = _mm256_set1_epi8(needle[offset_first]);
    n_mid_vec.ymm = _mm256_set1_epi8(needle[offset_mid]);
    n_last_vec.ymm = _mm256_set1_epi8(needle[offset_last]);

    // Scan through the string.
    sz_cptr_t haystack_cursor;
    for (; haystack_length >= needle_length + 32; haystack_length -= 32) {
        haystack_cursor = haystack + haystack_length - needle_length - 32 + 1;
        h_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack_cursor + offset_first));
        h_mid_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack_cursor + offset_mid));
        h_last_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack_cursor + offset_last));
        matches_vec.i32 = //
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_first_vec.ymm, n_first_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_mid_vec.ymm, n_mid_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_last_vec.ymm, n_last_vec.ymm));
        while (matches_vec.u32) {
            int potential_offset = (int)_lzcnt_u32(matches_vec.u32);
            if (sz_equal_haswell_(haystack + haystack_length - needle_length - potential_offset, needle, needle_length))
                return haystack + haystack_length - needle_length - potential_offset;
            matches_vec.u32 &= ~(1u << (31 - potential_offset));
        }
    }

    return sz_rfind_serial_(haystack, haystack_length, needle, needle_length);
}

STRINGZILLA_OUTLINED_ sz_cptr_t sz_find_byteset_haswell_(sz_cptr_t haystack, sz_size_t haystack_length,
                                                         sz_byteset_t const *filter) {

    // Let's unzip even and odd elements and replicate them into both lanes of the YMM register.
    // That way when we invoke `_mm256_shuffle_epi8` we can use the same mask for both lanes.
    // Load the 32-byte filter as two 16-byte halves, separate even/odd bytes, pack, and broadcast to YMM.
    sz_u128_vec_t byte_mask_vec;
    sz_u128_vec_t filter_low_vec, filter_high_vec;
    sz_u128_vec_t low_evens_vec, high_evens_vec;
    sz_u128_vec_t low_odds_vec, high_odds_vec;
    sz_u128_vec_t evens_xmm_vec, odds_xmm_vec;
    sz_u256_vec_t filter_even_vec, filter_odd_vec;

    sz_u256_vec_t text_vec;
    sz_u256_vec_t matches_vec;
    sz_u256_vec_t lower_nibbles_vec, higher_nibbles_vec;
    sz_u256_vec_t bitset_even_vec, bitset_odd_vec;
    sz_u256_vec_t bitmask_vec, bitmask_lookup_vec;

    byte_mask_vec.xmm = _mm_set1_epi16(0x00ff);

    filter_low_vec.xmm = _mm_lddqu_si128((__m128i const *)(filter));
    filter_high_vec.xmm = _mm_lddqu_si128((__m128i const *)(filter) + 1);
    low_evens_vec.xmm = _mm_and_si128(filter_low_vec.xmm, byte_mask_vec.xmm);
    high_evens_vec.xmm = _mm_and_si128(filter_high_vec.xmm, byte_mask_vec.xmm);
    low_odds_vec.xmm = _mm_srli_epi16(filter_low_vec.xmm, 8);
    high_odds_vec.xmm = _mm_srli_epi16(filter_high_vec.xmm, 8);

    evens_xmm_vec.xmm = _mm_packus_epi16(low_evens_vec.xmm, high_evens_vec.xmm);
    odds_xmm_vec.xmm = _mm_packus_epi16(low_odds_vec.xmm, high_odds_vec.xmm);
    filter_even_vec.ymm = _mm256_set_m128i(evens_xmm_vec.xmm, evens_xmm_vec.xmm);
    filter_odd_vec.ymm = _mm256_set_m128i(odds_xmm_vec.xmm, odds_xmm_vec.xmm);

    bitmask_lookup_vec.ymm = _mm256_set_epi8(                       //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1);

    while (haystack_length >= 32) {
        // The following algorithm is a transposed equivalent of the "SIMD-ized check which bytes are in a set"
        // solutions by Wojciech Muła. We populate the bitmask differently and target newer CPUs, so
        // StrinZilla uses a somewhat different approach.
        // http://0x80.pl/articles/simd-byte-lookup.html#alternative-implementation-new
        //
        //      sz_u8_t input = *(sz_u8_t const *)haystack;
        //      sz_u8_t lo_nibble = input & 0x0f;
        //      sz_u8_t hi_nibble = input >> 4;
        //      sz_u8_t bitset_even = filter_even_vec.u8s[hi_nibble];
        //      sz_u8_t bitset_odd = filter_odd_vec.u8s[hi_nibble];
        //      sz_u8_t bitmask = (1 << (lo_nibble & 0x7));
        //      sz_u8_t bitset = lo_nibble < 8 ? bitset_even : bitset_odd;
        //      if ((bitset & bitmask) != 0) return haystack;
        //      else { haystack_length--, haystack++; }
        //
        // The nice part about this, loading the strided data is vey easy with Arm NEON,
        // while with x86 CPUs after AVX, shuffles within 256 bits shouldn't be an issue either.
        text_vec.ymm = _mm256_lddqu_si256((__m256i const *)haystack);
        lower_nibbles_vec.ymm = _mm256_and_si256(text_vec.ymm, _mm256_set1_epi8(0x0f));
        bitmask_vec.ymm = _mm256_shuffle_epi8(bitmask_lookup_vec.ymm, lower_nibbles_vec.ymm);
        //
        // At this point we can validate the `bitmask_vec` contents like this:
        //
        //      for (sz_size_t i = 0; i != 32; ++i) {
        //          sz_u8_t input = *(sz_u8_t const *)(haystack + i);
        //          sz_u8_t lo_nibble = input & 0x0f;
        //          sz_u8_t bitmask = (1 << (lo_nibble & 0x7));
        //          sz_assert_(bitmask_vec.u8s[i] == bitmask);
        //      }
        //
        // Shift right every byte by 4 bits.
        // There is no `_mm256_srli_epi8` intrinsic, so we have to use `_mm256_srli_epi16`
        // and combine it with a mask to clear the higher bits.
        higher_nibbles_vec.ymm = _mm256_and_si256(_mm256_srli_epi16(text_vec.ymm, 4), _mm256_set1_epi8(0x0f));
        bitset_even_vec.ymm = _mm256_shuffle_epi8(filter_even_vec.ymm, higher_nibbles_vec.ymm);
        bitset_odd_vec.ymm = _mm256_shuffle_epi8(filter_odd_vec.ymm, higher_nibbles_vec.ymm);
        //
        // At this point we can validate the `bitset_even_vec` and `bitset_odd_vec` contents like this:
        //
        //      for (sz_size_t i = 0; i != 32; ++i) {
        //          sz_u8_t input = *(sz_u8_t const *)(haystack + i);
        //          sz_u8_t const *bitset_ptr = &filter->_u8s[0];
        //          sz_u8_t hi_nibble = input >> 4;
        //          sz_u8_t bitset_even = bitset_ptr[hi_nibble * 2];
        //          sz_u8_t bitset_odd = bitset_ptr[hi_nibble * 2 + 1];
        //          sz_assert_(bitset_even_vec.u8s[i] == bitset_even);
        //          sz_assert_(bitset_odd_vec.u8s[i] == bitset_odd);
        //      }
        //
        __m256i take_first_u8x32 = _mm256_cmpgt_epi8(_mm256_set1_epi8(8), lower_nibbles_vec.ymm);
        bitset_even_vec.ymm = _mm256_blendv_epi8(bitset_odd_vec.ymm, bitset_even_vec.ymm, take_first_u8x32);

        // It would have been great to have an instruction that tests the bits and then broadcasts
        // the matching bit into all bits in that byte. But we don't have that, so we have to
        // `and`, `cmpeq`, `movemask`, and then invert at the end...
        matches_vec.ymm = _mm256_and_si256(bitset_even_vec.ymm, bitmask_vec.ymm);
        matches_vec.ymm = _mm256_cmpeq_epi8(matches_vec.ymm, _mm256_setzero_si256());
        int matches_mask = ~_mm256_movemask_epi8(matches_vec.ymm);
        if (matches_mask) {
            int offset = (int)_tzcnt_u32(matches_mask);
            return haystack + offset;
        }
        else { haystack += 32, haystack_length -= 32; }
    }

    return sz_find_byteset_serial_(haystack, haystack_length, filter);
}

STRINGZILLA_OUTLINED_ sz_cptr_t sz_rfind_byteset_haswell_(sz_cptr_t haystack, sz_size_t haystack_length,
                                                          sz_byteset_t const *filter) {

    // Mirror of `sz_find_byteset_haswell` scanning from the end: the same transposed Muła nibble-bitset
    // classifier, but each iteration tests the trailing 32-byte window and resolves the highest matching
    // lane (the last occurrence) with `~movemask` + `clz`.
    sz_u128_vec_t byte_mask_vec;
    sz_u128_vec_t filter_low_vec, filter_high_vec;
    sz_u128_vec_t low_evens_vec, high_evens_vec;
    sz_u128_vec_t low_odds_vec, high_odds_vec;
    sz_u128_vec_t evens_xmm_vec, odds_xmm_vec;
    sz_u256_vec_t filter_even_vec, filter_odd_vec;

    sz_u256_vec_t text_vec;
    sz_u256_vec_t matches_vec;
    sz_u256_vec_t lower_nibbles_vec, higher_nibbles_vec;
    sz_u256_vec_t bitset_even_vec, bitset_odd_vec;
    sz_u256_vec_t bitmask_vec, bitmask_lookup_vec;

    byte_mask_vec.xmm = _mm_set1_epi16(0x00ff);

    filter_low_vec.xmm = _mm_lddqu_si128((__m128i const *)(filter));
    filter_high_vec.xmm = _mm_lddqu_si128((__m128i const *)(filter) + 1);
    low_evens_vec.xmm = _mm_and_si128(filter_low_vec.xmm, byte_mask_vec.xmm);
    high_evens_vec.xmm = _mm_and_si128(filter_high_vec.xmm, byte_mask_vec.xmm);
    low_odds_vec.xmm = _mm_srli_epi16(filter_low_vec.xmm, 8);
    high_odds_vec.xmm = _mm_srli_epi16(filter_high_vec.xmm, 8);

    evens_xmm_vec.xmm = _mm_packus_epi16(low_evens_vec.xmm, high_evens_vec.xmm);
    odds_xmm_vec.xmm = _mm_packus_epi16(low_odds_vec.xmm, high_odds_vec.xmm);
    filter_even_vec.ymm = _mm256_set_m128i(evens_xmm_vec.xmm, evens_xmm_vec.xmm);
    filter_odd_vec.ymm = _mm256_set_m128i(odds_xmm_vec.xmm, odds_xmm_vec.xmm);

    bitmask_lookup_vec.ymm = _mm256_set_epi8(                       //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1);

    while (haystack_length >= 32) {
        sz_cptr_t const window = haystack + haystack_length - 32;
        text_vec.ymm = _mm256_lddqu_si256((__m256i const *)window);
        lower_nibbles_vec.ymm = _mm256_and_si256(text_vec.ymm, _mm256_set1_epi8(0x0f));
        bitmask_vec.ymm = _mm256_shuffle_epi8(bitmask_lookup_vec.ymm, lower_nibbles_vec.ymm);
        higher_nibbles_vec.ymm = _mm256_and_si256(_mm256_srli_epi16(text_vec.ymm, 4), _mm256_set1_epi8(0x0f));
        bitset_even_vec.ymm = _mm256_shuffle_epi8(filter_even_vec.ymm, higher_nibbles_vec.ymm);
        bitset_odd_vec.ymm = _mm256_shuffle_epi8(filter_odd_vec.ymm, higher_nibbles_vec.ymm);
        __m256i take_first_u8x32 = _mm256_cmpgt_epi8(_mm256_set1_epi8(8), lower_nibbles_vec.ymm);
        bitset_even_vec.ymm = _mm256_blendv_epi8(bitset_odd_vec.ymm, bitset_even_vec.ymm, take_first_u8x32);
        matches_vec.ymm = _mm256_and_si256(bitset_even_vec.ymm, bitmask_vec.ymm);
        matches_vec.ymm = _mm256_cmpeq_epi8(matches_vec.ymm, _mm256_setzero_si256());
        int matches_mask = ~_mm256_movemask_epi8(matches_vec.ymm);
        if (matches_mask) {
            // Highest set bit = last matching byte in the window.
            int offset = 31 - (int)_lzcnt_u32((sz_u32_t)matches_mask);
            return window + offset;
        }
        else { haystack_length -= 32; }
    }

    return sz_rfind_byteset_serial_(haystack, haystack_length, filter);
}

#if STRINGZILLA_TARGET_HASWELL

STRINGZILLA_API sz_status_t sz_find_byte_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                 sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_find_byte_haswell_(haystack, haystack_length, needle);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_rfind_byte_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                  sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_rfind_byte_haswell_(haystack, haystack_length, needle);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_find_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                            sz_size_t needle_length, sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_find_haswell_(haystack, haystack_length, needle, needle_length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_find_strings_haswell(sz_strings_view_t const *strings, sz_size_t const *order,
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
        sz_cptr_t const match = sz_find_haswell_(view.start, view.length, needle, needle_length);
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
        // The last block backs off to end where the payload does, dropping the starts it repeats.
        sz_size_t const back = 64 - starts;
        sz_u64_t candidates = starts && block >= back
                                  ? sz_find_candidates_haswell_(data + block - back, offsets, bytes) >> back
                                  : sz_find_candidates_serial_(data + block, starts, offsets, bytes);
        // Each step settles a string or drops one of its candidates, until one outruns the block.
        while (row != count && row_start < block + 64) {
            // The lanes of the string's starts in the block, none when the needle outgrows it.
            sz_size_t const first_lane = row_start > block ? row_start - block : 0;
            sz_size_t const end_lane = row_end - row_start < needle_length ? first_lane
                                                                           : row_end - needle_length + 1 - block;
            sz_u64_t const own = candidates & sz_u64_mask_until_serial_(end_lane) &
                                 ~sz_u64_mask_until_serial_(first_lane);
            if (!own && end_lane > 64) break;
            sz_size_t const lane = (sz_size_t)_tzcnt_u64(own);
            candidates &= ~(own & (0 - own));
            if (own && needle_length > 3 && !sz_equal_haswell_((sz_cptr_t)data + block + lane, needle, needle_length))
                continue;
            positions[row] = own ? block + lane - row_start : STRINGZILLA_SIZE_MAX;
            row_start = row_end, ++row;
            row_end = sz_strings_view_offset_(strings, sz_min_of_two(row + 1, count));
        }
    }
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_rfind_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                             sz_size_t needle_length, sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_rfind_haswell_(haystack, haystack_length, needle, needle_length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_find_byteset_haswell(sz_cptr_t haystack, sz_size_t haystack_length,
                                                    sz_byteset_t const *filter, sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_find_byteset_haswell_(haystack, haystack_length, filter);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_rfind_byteset_haswell(sz_cptr_t haystack, sz_size_t haystack_length,
                                                     sz_byteset_t const *filter, sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_rfind_byteset_haswell_(haystack, haystack_length, filter);
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_HASWELL

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_X8664_HASWELL_
#endif // STRINGZILLA_ARCH_X8664_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_HASWELL_H_
