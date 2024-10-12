/**
 *  @brief  Experimental kernels for StringZilla.
 *  @file   drafts.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_EXPERIMENTAL_H_
#define STRINGZILLA_EXPERIMENTAL_H_

#include "stringzilla.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief  Bitap algo for exact matching of patterns up to @b 8-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_bitap_upto_8bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                        sz_size_t n_length) {
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *h_end = h_unsigned + h_length;

    // Here is our baseline:
    //
    //      sz_u8_t running_match = 0xFF;
    //      sz_u8_t character_position_masks[256];
    //      for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFF; }
    //      for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[i]] &= ~(1u << i); }
    //      for (sz_size_t i = 0; i < h_length; ++i) {
    //          running_match = (running_match << 1) | character_position_masks[h_unsigned[i]];
    //          if ((running_match & (1u << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    //      }
    //
    // On very short patterns, however, every tiny condition may have a huge affect on performance.
    // 1. Let's replace byte-level intialization of `character_position_masks` with 64-bit ops.
    // 2. Let's combine the first `n_length - 1` passes of the last loop into the previous loop.
    typedef sz_u8_t offset_mask_t;

    // Initialize the possible offset masks.
    // Even using 8-byte `wide_masks` words, this would require 64 iterations to populate 256 bytes.
    union {
        offset_mask_t masks[256];
        sz_u64_t wide_masks[sizeof(offset_mask_t) * 256 / sizeof(sz_u64_t)];
    } character_positions;
    for (sz_size_t i = 0; i != sizeof(offset_mask_t) * 256 / sizeof(sz_u64_t); ++i) {
        character_positions.wide_masks[i] = 0xFFFFFFFFFFFFFFFFull;
    }

    // Populate the mask with possible positions for each character.
    for (sz_size_t i = 0; i != n_length; ++i) { character_positions.masks[n_unsigned[i]] &= ~((offset_mask_t)1 << i); }

    // The "running match" for the serial algorithm should be at least as wide as the `offset_mask_t`.
    // But on modern systems larger integers may work better.
    offset_mask_t running_match = 0, final_match = 1;
    running_match = ~(running_match ^ running_match); //< Initialize with all-ones
    final_match <<= n_length - 1;

    for (; (h_unsigned != h_end) + ((running_match & final_match) != 0) == 2; ++h_unsigned) {
        running_match = (running_match << 1) | character_positions.masks[h_unsigned[0]];
    }
    return ((running_match & final_match) == 0) ? (sz_cptr_t)(h_unsigned - n_length) : NULL;
}

/**
 *  @brief  Bitap algorithm for exact matching of patterns up to @b 8-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_rfind_bitap_upto_8bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                         sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u8_t running_match = 0xFF;
    sz_u8_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[n_length - i - 1]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[h_length - i - 1]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algo for exact matching of patterns up to @b 16-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_bitap_upto_16bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                         sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u16_t running_match = 0xFFFF;
    sz_u16_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[i]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[i]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algorithm for exact matching of patterns up to @b 16-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_rfind_bitap_upto_16bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                          sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u16_t running_match = 0xFFFF;
    sz_u16_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[n_length - i - 1]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[h_length - i - 1]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algo for exact matching of patterns up to @b 32-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_bitap_upto_32bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                         sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u32_t running_match = 0xFFFFFFFF;
    sz_u32_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFFFFFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[i]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[i]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algorithm for exact matching of patterns up to @b 32-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_rfind_bitap_upto_32bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                          sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u32_t running_match = 0xFFFFFFFF;
    sz_u32_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFFFFFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[n_length - i - 1]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[h_length - i - 1]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algo for exact matching of patterns up to @b 64-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_bitap_upto_64bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                         sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u64_t running_match = 0xFFFFFFFFFFFFFFFFull;
    sz_u64_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFFFFFFFFFFFFFFull; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[i]] &= ~(1ull << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[i]];
        if ((running_match & (1ull << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algorithm for exact matching of patterns up to @b 64-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_rfind_bitap_upto_64bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                          sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u64_t running_match = 0xFFFFFFFFFFFFFFFFull;
    sz_u64_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFFFFFFFFFFFFFFull; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[n_length - i - 1]] &= ~(1ull << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[h_length - i - 1]];
        if ((running_match & (1ull << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algo for approximate matching of patterns up to @b 64-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_bounded_bitap_upto_64bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                                 sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u64_t running_match = 0xFFFFFFFFFFFFFFFFull;
    sz_u64_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFFFFFFFFFFFFFFull; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[i]] &= ~(1ull << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[i]];
        if ((running_match & (1ull << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algorithm for approximate matching of patterns up to @b 64-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_bounded_last_bitap_upto_64bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                                      sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u64_t running_match = 0xFFFFFFFFFFFFFFFFull;
    sz_u64_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFFFFFFFFFFFFFFull; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[n_length - i - 1]] &= ~(1ull << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[h_length - i - 1]];
        if ((running_match & (1ull << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

#if SZ_USE_AVX512

SZ_PUBLIC sz_size_t sz_edit_distance_avx512(     //
    sz_cptr_t const a, sz_size_t const a_length, //
    sz_cptr_t const b, sz_size_t const b_length, //
    sz_size_t const bound, sz_memory_allocator_t *alloc) {

    sz_u512_vec_t a_vec, b_vec, previous_vec, current_vec, permutation_vec;
    sz_u512_vec_t cost_deletion_vec, cost_insertion_vec, cost_substitution_vec;
    sz_size_t min_distance;

    b_vec.zmm = _mm512_maskz_loadu_epi8(_sz_u64_mask_until(b_length), b);
    previous_vec.zmm = _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, //
                                       47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, //
                                       31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, //
                                       15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    // Shifting bytes across the whole ZMM register is quite complicated, so let's use a permutation for that.
    permutation_vec.zmm = _mm512_set_epi8(62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, //
                                          46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, //
                                          30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, //
                                          14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 63);

    for (sz_size_t idx_a = 0; idx_a != a_length; ++idx_a) {
        min_distance = bound - 1;

        a_vec.zmm = _mm512_set1_epi8(a[idx_a]);
        // We first start by computing the cost of deletions and substitutions
        // for (sz_size_t idx_b = 0; idx_b != b_length; ++idx_b) {
        //     sz_u8_t cost_deletion = previous_vec.u8s[idx_b + 1] + 1;
        //     sz_u8_t cost_substitution = previous_vec.u8s[idx_b] + (a[idx_a] != b[idx_b]);
        //     current_vec.u8s[idx_b + 1] = sz_min_of_two(cost_deletion, cost_substitution);
        // }
        cost_deletion_vec.zmm = _mm512_add_epi8(previous_vec.zmm, _mm512_set1_epi8(1));
        cost_substitution_vec.zmm =
            _mm512_mask_set1_epi8(_mm512_setzero_si512(), _mm512_cmpneq_epi8_mask(a_vec.zmm, b_vec.zmm), 0x01);
        cost_substitution_vec.zmm = _mm512_add_epi8(previous_vec.zmm, cost_substitution_vec.zmm);
        cost_substitution_vec.zmm = _mm512_permutexvar_epi8(permutation_vec.zmm, cost_substitution_vec.zmm);
        current_vec.zmm = _mm512_min_epu8(cost_deletion_vec.zmm, cost_substitution_vec.zmm);
        current_vec.u8s[0] = idx_a + 1;

        // Now we need to compute the inclusive prefix sums using the minimum operator
        // In one line:
        //      current_vec.u8s[idx_b + 1] = sz_min_of_two(current_vec.u8s[idx_b + 1], current_vec.u8s[idx_b] + 1)
        //
        // Unrolling this:
        //      current_vec.u8s[0 + 1] = sz_min_of_two(current_vec.u8s[0 + 1], current_vec.u8s[0] + 1)
        //      current_vec.u8s[1 + 1] = sz_min_of_two(current_vec.u8s[1 + 1], current_vec.u8s[1] + 1)
        //      current_vec.u8s[2 + 1] = sz_min_of_two(current_vec.u8s[2 + 1], current_vec.u8s[2] + 1)
        //      current_vec.u8s[3 + 1] = sz_min_of_two(current_vec.u8s[3 + 1], current_vec.u8s[3] + 1)
        //
        // Alternatively, using a tree-like reduction in log2 steps:
        //      - 6 cycles of reductions shifting by 1, 2, 4, 8, 16, 32, 64 bytes;
        //      - with each cycle containing at least one shift, min, add, blend.
        //
        // Which adds meaningless complexity without any performance gains.
        for (sz_size_t idx_b = 0; idx_b != b_length; ++idx_b) {
            sz_u8_t cost_insertion = current_vec.u8s[idx_b] + 1;
            current_vec.u8s[idx_b + 1] = sz_min_of_two(current_vec.u8s[idx_b + 1], cost_insertion);
        }

        // Swap previous_distances and current_distances pointers
        sz_u512_vec_t temp_vec;
        temp_vec.zmm = previous_vec.zmm;
        previous_vec.zmm = current_vec.zmm;
        current_vec.zmm = temp_vec.zmm;
    }

    return previous_vec.u8s[b_length] < bound ? previous_vec.u8s[b_length] : bound;
}

sz_u512_vec_t sz_inclusive_min(sz_i32_t previous, sz_error_cost_t gap, sz_u512_vec_t base_vec) {

    sz_u512_vec_t gap_vec, gap_double_vec, gap_quad_vec, gap_octa_vec;
    gap_vec.zmm = _mm512_set1_epi32(gap);
    gap_double_vec.zmm = _mm512_set1_epi32(2 * gap);
    gap_quad_vec.zmm = _mm512_set1_epi32(4 * gap);
    gap_octa_vec.zmm = _mm512_set1_epi32(8 * gap);

    // __mmask16 mask_skip_one = 0xFFFF - 1;
    // __mmask16 mask_skip_two = 0xFFFF - 3;
    // __mmask16 mask_skip_four = 0xFFF0;
    // __mmask16 mask_skip_eight = 0xFF00;
    __mmask16 mask_skip_one = 0x7FFF;
    __mmask16 mask_skip_two = 0x3FFF;
    __mmask16 mask_skip_four = 0x0FFF;
    __mmask16 mask_skip_eight = 0x00FF;
    sz_u512_vec_t shift_by_one_vec, shift_by_two_vec, shift_by_four_vec, shift_by_eight_vec;
    shift_by_one_vec.zmm = _mm512_set_epi32(14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0);
    shift_by_two_vec.zmm = _mm512_set_epi32(13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0);
    shift_by_four_vec.zmm = _mm512_set_epi32(11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0);
    shift_by_eight_vec.zmm = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    sz_u512_vec_t shifted_vec;
    sz_u512_vec_t new_vec = base_vec;
    shifted_vec.zmm = _mm512_permutexvar_epi32(shift_by_one_vec.zmm, new_vec.zmm);
    shifted_vec.i32s[0] = previous;
    shifted_vec.zmm = _mm512_add_epi32(shifted_vec.zmm, gap_vec.zmm);
    new_vec.zmm = _mm512_mask_max_epi32(new_vec.zmm, mask_skip_one, new_vec.zmm, shifted_vec.zmm);
    sz_assert(new_vec.i32s[0] == max(previous + gap, base_vec.i32s[0]));

    shifted_vec.zmm = _mm512_permutexvar_epi32(shift_by_two_vec.zmm, new_vec.zmm);
    shifted_vec.zmm = _mm512_add_epi32(shifted_vec.zmm, gap_double_vec.zmm);
    new_vec.zmm = _mm512_mask_max_epi32(new_vec.zmm, mask_skip_two, new_vec.zmm, shifted_vec.zmm);
    sz_assert(new_vec.i32s[0] == max(previous + gap, base_vec.i32s[0]));

    shifted_vec.zmm = _mm512_permutexvar_epi32(shift_by_four_vec.zmm, new_vec.zmm);
    shifted_vec.zmm = _mm512_add_epi32(shifted_vec.zmm, gap_quad_vec.zmm);
    new_vec.zmm = _mm512_mask_max_epi32(new_vec.zmm, mask_skip_four, new_vec.zmm, shifted_vec.zmm);
    sz_assert(new_vec.i32s[0] == max(previous + gap, base_vec.i32s[0]));

    shifted_vec.zmm = _mm512_permutexvar_epi32(shift_by_eight_vec.zmm, new_vec.zmm);
    shifted_vec.zmm = _mm512_add_epi32(shifted_vec.zmm, gap_octa_vec.zmm);
    new_vec.zmm = _mm512_mask_max_epi32(new_vec.zmm, mask_skip_eight, new_vec.zmm, shifted_vec.zmm);

    sz_assert(new_vec.i32s[0] == max(previous + gap, base_vec.i32s[0]));
    for (sz_size_t i = 1; i < 16; i++) sz_assert(new_vec.i32s[i] == max(new_vec.i32s[i - 1] + gap, new_vec.i32s[i]));

    return new_vec;
}

SZ_PUBLIC sz_cptr_t sz_find_charset_avx512(sz_cptr_t text, sz_size_t length, sz_charset_t const *filter) {

    sz_size_t load_length;
    __mmask32 load_mask, matches_mask;
    // To store the set in the register we need just 256 bits, but the `VPERMB` instruction
    // we are going to invoke is surprisingly cheaper on ZMM registers.
    sz_u512_vec_t text_vec, filter_vec;
    filter_vec.ymms[0] = _mm256_loadu_epi64(&filter->_u64s[0]);

    // We are going to view the `filter` at 8-bit word granularity.
    sz_u512_vec_t filter_slice_offsets_vec;
    sz_u512_vec_t filter_slice_vec;
    sz_u512_vec_t offset_within_slice_vec;
    sz_u512_vec_t mask_in_filter_slice_vec;
    sz_u512_vec_t matches_vec;

    while (length) {
        // For every byte:
        // 1. Find corresponding word in a set.
        // 2. Produce a bitmask to check against that word.
        load_length = sz_min_of_two(length, 32);
        load_mask = _sz_u64_mask_until(load_length);
        text_vec.ymms[0] = _mm256_maskz_loadu_epi8(load_mask, text);

        // To shift right every byte by 3 bits we can use the GF2 affine transformations.
        // https://wunkolo.github.io/post/2020/11/gf2p8affineqb-int8-shifting/
        // After next line, all 8-bit offsets in the `filter_slice_offsets_vec` should be under 32.
        filter_slice_offsets_vec.ymms[0] =
            _mm256_gf2p8affine_epi64_epi8(text_vec.ymms[0], _mm256_set1_epi64x(0x0102040810204080ull << (3 * 8)), 0);

        // After next line, `filter_slice_vec` will contain the right word from the set,
        // needed to filter the presence of the byte in the set.
        filter_slice_vec.ymms[0] = _mm256_permutexvar_epi8(filter_slice_offsets_vec.ymms[0], filter_vec.ymms[0]);

        // After next line, all 8-bit offsets in the `filter_slice_offsets_vec` should be under 8.
        offset_within_slice_vec.ymms[0] = _mm256_and_si256(text_vec.ymms[0], _mm256_set1_epi64x(0x0707070707070707ull));

        // Instead of performing one more Galois Field operation, we can upcast to 16-bit integers,
        // and perform the fift and intersection there.
        filter_slice_vec.zmm = _mm512_cvtepi8_epi16(filter_slice_vec.ymms[0]);
        offset_within_slice_vec.zmm = _mm512_cvtepi8_epi16(offset_within_slice_vec.ymms[0]);
        mask_in_filter_slice_vec.zmm = _mm512_sllv_epi16(_mm512_set1_epi16(1), offset_within_slice_vec.zmm);
        matches_vec.zmm = _mm512_and_si512(filter_slice_vec.zmm, mask_in_filter_slice_vec.zmm);

        matches_mask = _mm512_mask_cmpneq_epi16_mask(load_mask, matches_vec.zmm, _mm512_setzero_si512());
        if (matches_mask) {
            int offset = sz_u32_ctz(matches_mask);
            return text + offset;
        }
        else { text += load_length, length -= load_length; }
    }

    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_rfind_charset_avx512(sz_cptr_t text, sz_size_t length, sz_charset_t const *filter) {

    sz_size_t load_length;
    __mmask32 load_mask, matches_mask;
    // To store the set in the register we need just 256 bits, but the `VPERMB` instruction
    // we are going to invoke is surprisingly cheaper on ZMM registers.
    sz_u512_vec_t text_vec, filter_vec;
    filter_vec.ymms[0] = _mm256_loadu_epi64(&filter->_u64s[0]);

    // We are going to view the `filter` at 8-bit word granularity.
    sz_u512_vec_t filter_slice_offsets_vec;
    sz_u512_vec_t filter_slice_vec;
    sz_u512_vec_t offset_within_slice_vec;
    sz_u512_vec_t mask_in_filter_slice_vec;
    sz_u512_vec_t matches_vec;

    while (length) {
        // For every byte:
        // 1. Find corresponding word in a set.
        // 2. Produce a bitmask to check against that word.
        load_length = sz_min_of_two(length, 32);
        load_mask = _sz_u64_mask_until(load_length);
        text_vec.ymms[0] = _mm256_maskz_loadu_epi8(load_mask, text + length - load_length);

        // To shift right every byte by 3 bits we can use the GF2 affine transformations.
        // https://wunkolo.github.io/post/2020/11/gf2p8affineqb-int8-shifting/
        // After next line, all 8-bit offsets in the `filter_slice_offsets_vec` should be under 32.
        filter_slice_offsets_vec.ymms[0] =
            _mm256_gf2p8affine_epi64_epi8(text_vec.ymms[0], _mm256_set1_epi64x(0x0102040810204080ull << (3 * 8)), 0);

        // After next line, `filter_slice_vec` will contain the right word from the set,
        // needed to filter the presence of the byte in the set.
        filter_slice_vec.ymms[0] = _mm256_permutexvar_epi8(filter_slice_offsets_vec.ymms[0], filter_vec.ymms[0]);

        // After next line, all 8-bit offsets in the `filter_slice_offsets_vec` should be under 8.
        offset_within_slice_vec.ymms[0] = _mm256_and_si256(text_vec.ymms[0], _mm256_set1_epi64x(0x0707070707070707ull));

        // Instead of performing one more Galois Field operation, we can upcast to 16-bit integers,
        // and perform the fift and intersection there.
        filter_slice_vec.zmm = _mm512_cvtepi8_epi16(filter_slice_vec.ymms[0]);
        offset_within_slice_vec.zmm = _mm512_cvtepi8_epi16(offset_within_slice_vec.ymms[0]);
        mask_in_filter_slice_vec.zmm = _mm512_sllv_epi16(_mm512_set1_epi16(1), offset_within_slice_vec.zmm);
        matches_vec.zmm = _mm512_and_si512(filter_slice_vec.zmm, mask_in_filter_slice_vec.zmm);

        matches_mask = _mm512_mask_cmpneq_epi16_mask(load_mask, matches_vec.zmm, _mm512_setzero_si512());
        if (matches_mask) {
            int offset = sz_u32_clz(matches_mask);
            return text + length - load_length + 32 - offset - 1;
        }
        else { length -= load_length; }
    }

    return SZ_NULL_CHAR;
}

#endif // SZ_USE_AVX512

#if SZ_USE_ARM_NEON

SZ_PUBLIC sz_cptr_t sz_find_neon_too_smart(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_find_byte_neon(h, h_length, n);

    // Scan through the string.
    // Assuming how tiny the Arm NEON registers are, we should avoid internal branches at all costs.
    // That's why, for smaller needles, we use different loops.
    if (n_length == 2) {
        // This is a common case. Aside from ASCII bigrams, it's also the most common case for UTF-16,
        // or any UTF8 content in Cyrillic, Greek, Armenian, Hebrew, Arabic, Coptic, Syriac, Thaana,
        // N'Ko writing system of West-African nations, and, of course, Latin scripts.
        // Dealing with 16-bit values, we can check 15 possible offsets in a single loop iteration.
        // For that we are going to keep 2 registers populated with haystack data.
        // First - bigrams at even offsets - 0, 2, 4, 6, 8, 10, 12, 14.
        // Second - bigrams at odd offsets - 1, 3, 5, 7, 9, 11, 13. One less than the first one.
        sz_u64_t matches;
        sz_u128_vec_t h_even_vec, h_odd_vec, n_vec, interleave_mask_vec, matches_vec;
        // Broadcast needle characters into SIMD registers.
        n_vec.u16x8 = vdupq_n_u16(sz_u16_load(n).u16);
        interleave_mask_vec.u16x8 = vdupq_n_u16(0x00FFu);
        for (; h_length >= 16; h += 15, h_length -= 15) {
            h_even_vec.u8x16 = vld1q_u8((sz_u8_t const *)h);
            h_odd_vec.u8x16 = vextq_u8(h_even_vec.u8x16, /* can be any noise: */ h_even_vec.u8x16, 1);
            // We can now compare both 16-bit arrays with the needle.
            // The result of each comparison will also be 16 bits long.
            // Then - we blend!
            // For odd offsets we are gonna take the bottom 8 bits, and for even - the top ones!
            matches_vec.u8x16 =
                vbslq_u8(interleave_mask_vec.u8x16, vreinterpretq_u8_u16(vceqq_u16(h_even_vec.u16x8, n_vec.u16x8)),
                         vreinterpretq_u8_u16(vceqq_u16(h_odd_vec.u16x8, n_vec.u16x8)));
            matches = vreinterpretq_u8_u4(matches_vec.u8x16);
            if (matches) return h + sz_u64_ctz(matches) / 4;
        }
    }
    else if (n_length == 3) {
        // Comparing 24-bit values is a bumer. Being lazy, I went with a simple design.
        // Instead of keeping one register per haystack offset, I keep a register per needle character.
        sz_u64_t matches;
        sz_u128_vec_t h_vec, n_first_vec, n_second_vec, n_third_vec, matches_vec;
        // Broadcast needle characters into SIMD registers.
        n_first_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[0]);
        n_second_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[1]);
        n_third_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[2]);
        for (; h_length >= 16; h += 14, h_length -= 14) {
            h_vec.u8x16 = vld1q_u8((sz_u8_t const *)h);
            // Let's compare the first character.
            matches_vec.u8x16 = vceqq_u8(h_vec.u8x16, n_first_vec.u8x16);
            // Let's compare the second one, shift the equality indicators left by 8 bits, and blend.
            matches_vec.u8x16 =
                vandq_u8(matches_vec.u8x16, vextq_u8(vceqq_u8(h_vec.u8x16, n_second_vec.u8x16), vdupq_n_u8(0), 1));
            // Let's compare the third one, shift the equality indicators left by 16 bits, and blend.
            matches_vec.u8x16 =
                vandq_u8(matches_vec.u8x16, vextq_u8(vceqq_u8(h_vec.u8x16, n_third_vec.u8x16), vdupq_n_u8(0), 2));
            // Now reduce bytes to nibbles, and check for matches.
            matches = vreinterpretq_u8_u4(matches_vec.u8x16);
            if (matches) return h + sz_u64_ctz(matches) / 4;
        }
    }
    else if (n_length == 4) {
        // This is a common case not only for ASCII 4-grams, but also UTF-32 content,
        // emojis, Chinese, and many other east-Asian languages.
        // Dealing with 32-bit values, we can analyze 13 offsets at once.
        sz_u128_vec_t h_first_vec, h_second_vec, h_third_vec, h_fourth_vec, interleave_2mask_vec, interleave_4mask_vec,
            n_vec, matches_vec;
        sz_u64_t matches;
        // Broadcast needle characters into SIMD registers.
        n_vec.u32x4 = vdupq_n_u32(sz_u32_load(n).u32);
        interleave_2mask_vec.u16x8 = vdupq_n_u16(0x00FFu);
        interleave_4mask_vec.u32x4 = vdupq_n_u32(0x0000FFFFu);
        for (; h_length >= 16; h += 13, h_length -= 13) {
            h_first_vec.u8x16 = vld1q_u8((sz_u8_t const *)h);
            h_second_vec.u8x16 = vextq_u8(h_first_vec.u8x16, /* can be any noise: */ h_first_vec.u8x16, 1);
            h_third_vec.u8x16 = vextq_u8(h_first_vec.u8x16, /* can be any noise: */ h_first_vec.u8x16, 2);
            h_fourth_vec.u8x16 = vextq_u8(h_first_vec.u8x16, /* can be any noise: */ h_first_vec.u8x16, 3);
            // We can now compare all four arrays of 32-bit values with the needle.
            // The result of each comparison will also be 32 bits long.
            // Then - we blend!
            matches_vec.u8x16 = vbslq_u8(
                interleave_4mask_vec.u8x16,
                vbslq_u8(interleave_2mask_vec.u8x16, vreinterpretq_u8_u32(vceqq_u32(h_first_vec.u32x4, n_vec.u32x4)),
                         vreinterpretq_u8_u32(vceqq_u32(h_second_vec.u32x4, n_vec.u32x4))),
                vbslq_u8(interleave_2mask_vec.u8x16, vreinterpretq_u8_u32(vceqq_u32(h_third_vec.u32x4, n_vec.u32x4)),
                         vreinterpretq_u8_u32(vceqq_u32(h_fourth_vec.u32x4, n_vec.u32x4))));
            matches = vreinterpretq_u8_u4(matches_vec.u8x16);
            if (matches) return h + sz_u64_ctz(matches) / 4;
        }
    }
    else {
        // Pick the parts of the needle that are worth comparing.
        sz_size_t offset_first, offset_mid, offset_last;
        _sz_locate_needle_anomalies(n, n_length, &offset_first, &offset_mid, &offset_last);
        // Broadcast those characters into SIMD registers.
        sz_u64_t matches;
        sz_u128_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec, matches_vec;
        n_first_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[offset_first]);
        n_mid_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[offset_mid]);
        n_last_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[offset_last]);
        // Walk through the string.
        for (; h_length >= n_length + 16; h += 16, h_length -= 16) {
            h_first_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h + offset_first));
            h_mid_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h + offset_mid));
            h_last_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h + offset_last));
            matches_vec.u8x16 = vandq_u8(                           //
                vandq_u8(                                           //
                    vceqq_u8(h_first_vec.u8x16, n_first_vec.u8x16), //
                    vceqq_u8(h_mid_vec.u8x16, n_mid_vec.u8x16)),
                vceqq_u8(h_last_vec.u8x16, n_last_vec.u8x16));
            matches = vreinterpretq_u8_u4(matches_vec.u8x16);
            while (matches) {
                int potential_offset = sz_u64_ctz(matches) / 4;
                if (sz_equal(h + potential_offset, n, n_length)) return h + potential_offset;
                matches &= matches - 1;
            }
        }
    }

    return sz_find_serial(h, h_length, n, n_length);
}

SZ_INTERNAL void interleave_uint32x4_to_uint64x2(uint32x4_t in_low, uint32x4_t in_high, uint64x2_t *out_first_second,
                                                 uint64x2_t *out_third_fourth) {
    // Interleave elements
    uint32x4x2_t interleaved = vzipq_u32(in_low, in_high);

    // The results are now in two uint32x4_t vectors, which we need to cast to uint64x2_t
    *out_first_second = vreinterpretq_u64_u32(interleaved.val[0]);
    *out_third_fourth = vreinterpretq_u64_u32(interleaved.val[1]);
}

/*  Arm NEON has several very relevant extensions for 32-bit FMA we can use for rolling hashes:
 *  * vmlaq_u32 - vector "fused-multiply-add"
 *  * vmlaq_n_u32 - vector-scalar "fused-multiply-add"
 *  * vmlsq_u32 - vector "fused-multiply-subtract"
 *  * vmlsq_n_u32 - vector-scalar "fused-multiply-subtract"
 *  Other basic intrinsics worth remembering:
 *  * vbslq_u32 - bitwise select to avoid branching
 *  * vld1q_dup_u32 - broadcast a 32-bit word into all 4 lanes of a 128-bit register
 */

SZ_PUBLIC void sz_hashes_neon_naive(sz_cptr_t start, sz_size_t length, sz_size_t window_length, sz_size_t step, //
                                    sz_hash_callback_t callback, void *callback_handle) {

    if (length < window_length || !window_length) return;
    if (length < 2 * window_length) {
        sz_hashes_serial(start, length, window_length, step, callback, callback_handle);
        return;
    }

    // Using NEON, we can perform 4 integer multiplications and additions within one register.
    // So let's slice the entire string into 4 overlapping windows, to slide over them in parallel.
    sz_u8_t const *text = (sz_u8_t const *)start;
    sz_u8_t const *text_end = text + length;

    // Prepare the `prime ^ window_length` values, that we are going to use for modulo arithmetic.
    sz_u32_t prime_power_low = 1, prime_power_high = 1;
    for (sz_size_t i = 0; i + 1 < window_length; ++i)
        prime_power_low = (prime_power_low * 31ull) % SZ_U32_MAX_PRIME,
        prime_power_high = (prime_power_high * 257ull) % SZ_U32_MAX_PRIME;

    sz_u128_vec_t hash_low_vec, hash_high_vec, hash_mix01_vec, hash_mix23_vec;
    uint8_t high_shift = 77u;
    uint32_t prime = SZ_U32_MAX_PRIME;

    sz_u128_vec_t chars_outgoing_vec, chars_incoming_vec, chars_outgoing_shifted_vec, chars_incoming_shifted_vec;
    // Let's skip the first window, as we are going to compute it in the loop.
    sz_size_t cycles = 0;
    sz_size_t step_mask = step - 1;
    sz_u32_t one = 1;

    // In every iteration we process 4 consecutive sliding windows.
    // Once each of them computes separate values, we step forward (W-1) times,
    // computing all interleaving values. That way the byte spilled from the second
    // hash, can be added to the first one. That way we minimize the number of separate loads.
    for (; text + window_length * 4 + (window_length - 1) <= text_end; text += window_length * 4) {
        hash_low_vec.u32x4 = vld1q_dup_u32(&one);
        hash_high_vec.u32x4 = vld1q_dup_u32(&one);
        for (sz_size_t i = 0; i != window_length; ++i) {
            chars_incoming_vec.u32s[0] = *(uint8_t const *)(text + window_length * 0 + i);
            chars_incoming_vec.u32s[1] = *(uint8_t const *)(text + window_length * 1 + i);
            chars_incoming_vec.u32s[2] = *(uint8_t const *)(text + window_length * 2 + i);
            chars_incoming_vec.u32s[3] = *(uint8_t const *)(text + window_length * 3 + i);
            chars_incoming_shifted_vec.u8x16 = vaddq_u8(chars_incoming_vec.u8x16, vld1q_dup_u8(&high_shift));

            // Append new data.
            hash_low_vec.u32x4 = vmlaq_n_u32(chars_incoming_vec.u32x4, hash_low_vec.u32x4, 31u);
            hash_high_vec.u32x4 = vmlaq_n_u32(chars_incoming_shifted_vec.u32x4, hash_high_vec.u32x4, 257u);
            hash_low_vec.u32x4 = vbslq_u32(hash_low_vec.u32x4, vsubq_u32(hash_low_vec.u32x4, vld1q_dup_u32(&prime)),
                                           vcgtq_u32(hash_low_vec.u32x4, vld1q_dup_u32(&prime)));
            hash_high_vec.u32x4 = vbslq_u32(hash_high_vec.u32x4, vsubq_u32(hash_high_vec.u32x4, vld1q_dup_u32(&prime)),
                                            vcgtq_u32(hash_high_vec.u32x4, vld1q_dup_u32(&prime)));
        }

        if ((cycles & step_mask) == 0) {
            interleave_uint32x4_to_uint64x2(hash_low_vec.u32x4, hash_high_vec.u32x4, &hash_mix01_vec.u64x2,
                                            &hash_mix23_vec.u64x2);
            callback((sz_cptr_t)(text + window_length * 0), window_length, hash_mix01_vec.u64s[0], callback_handle);
            callback((sz_cptr_t)(text + window_length * 1), window_length, hash_mix01_vec.u64s[1], callback_handle);
            callback((sz_cptr_t)(text + window_length * 2), window_length, hash_mix23_vec.u64s[0], callback_handle);
            callback((sz_cptr_t)(text + window_length * 3), window_length, hash_mix23_vec.u64s[1], callback_handle);
        }
        ++cycles;

        for (sz_size_t i = 0; i + 1 != window_length; ++i, ++cycles) {
            // Now, to compute 4 hashes per iteration, instead of loading 8 separate bytes (4 incoming and 4 outgoing)
            // we can limit ourselves to only 5 values, 3 of which will be reused for both append and erase operations.
            chars_outgoing_vec.u32s[0] = *(uint8_t const *)(text + window_length * 0 + i);
            chars_outgoing_vec.u32s[1] = chars_incoming_vec.u32s[0] = *(uint8_t const *)(text + window_length * 1 + i);
            chars_outgoing_vec.u32s[2] = chars_incoming_vec.u32s[1] = *(uint8_t const *)(text + window_length * 2 + i);
            chars_outgoing_vec.u32s[3] = chars_incoming_vec.u32s[2] = *(uint8_t const *)(text + window_length * 3 + i);
            chars_incoming_vec.u32s[3] = *(uint8_t const *)(text + window_length * 4 + i);
            chars_incoming_shifted_vec.u8x16 = vaddq_u8(chars_incoming_vec.u8x16, vld1q_dup_u8(&high_shift));
            chars_outgoing_shifted_vec.u8x16 = vaddq_u8(chars_outgoing_vec.u8x16, vld1q_dup_u8(&high_shift));

            // Drop old data.
            hash_low_vec.u32x4 = vmlsq_n_u32(hash_low_vec.u32x4, chars_outgoing_vec.u32x4, prime_power_low);
            hash_high_vec.u32x4 = vmlsq_n_u32(hash_high_vec.u32x4, chars_outgoing_shifted_vec.u32x4, prime_power_high);

            // Append new data.
            hash_low_vec.u32x4 = vmlaq_n_u32(chars_incoming_vec.u32x4, hash_low_vec.u32x4, 31u);
            hash_high_vec.u32x4 = vmlaq_n_u32(chars_incoming_shifted_vec.u32x4, hash_high_vec.u32x4, 257u);
            hash_low_vec.u32x4 = vbslq_u32(hash_low_vec.u32x4, vsubq_u32(hash_low_vec.u32x4, vld1q_dup_u32(&prime)),
                                           vcgtq_u32(hash_low_vec.u32x4, vld1q_dup_u32(&prime)));
            hash_high_vec.u32x4 = vbslq_u32(hash_high_vec.u32x4, vsubq_u32(hash_high_vec.u32x4, vld1q_dup_u32(&prime)),
                                            vcgtq_u32(hash_high_vec.u32x4, vld1q_dup_u32(&prime)));
            // Mix and call the user if needed
            if ((cycles & step_mask) == 0) {
                interleave_uint32x4_to_uint64x2(hash_low_vec.u32x4, hash_high_vec.u32x4, &hash_mix01_vec.u64x2,
                                                &hash_mix23_vec.u64x2);
                callback((sz_cptr_t)(text + window_length * 0), window_length, hash_mix01_vec.u64s[0], callback_handle);
                callback((sz_cptr_t)(text + window_length * 1), window_length, hash_mix01_vec.u64s[1], callback_handle);
                callback((sz_cptr_t)(text + window_length * 2), window_length, hash_mix23_vec.u64s[0], callback_handle);
                callback((sz_cptr_t)(text + window_length * 3), window_length, hash_mix23_vec.u64s[1], callback_handle);
            }
        }
    }
}

SZ_PUBLIC void sz_hashes_neon_reusing_loads(sz_cptr_t start, sz_size_t length, sz_size_t window_length, sz_size_t step,
                                            sz_hash_callback_t callback, void *callback_handle) {

    if (length < window_length || !window_length) return;
    if (length < 2 * window_length) {
        sz_hashes_serial(start, length, window_length, step, callback, callback_handle);
        return;
    }

    // Using NEON, we can perform 4 integer multiplications and additions within one register.
    // So let's slice the entire string into 4 overlapping windows, to slide over them in parallel.
    sz_u8_t const *text = (sz_u8_t const *)start;
    sz_u8_t const *text_end = text + length;

    // Prepare the `prime ^ window_length` values, that we are going to use for modulo arithmetic.
    sz_u32_t prime_power_low = 1, prime_power_high = 1;
    for (sz_size_t i = 0; i + 1 < window_length; ++i)
        prime_power_low = (prime_power_low * 31ull) % SZ_U32_MAX_PRIME,
        prime_power_high = (prime_power_high * 257ull) % SZ_U32_MAX_PRIME;

    sz_u128_vec_t hash_low_vec, hash_high_vec, hash_mix01_vec, hash_mix23_vec;
    uint8_t high_shift = 77u;
    uint32_t prime = SZ_U32_MAX_PRIME;

    sz_u128_vec_t chars_outgoing_vec, chars_incoming_vec, chars_outgoing_shifted_vec, chars_incoming_shifted_vec;
    // Let's skip the first window, as we are going to compute it in the loop.
    sz_size_t cycles = 0;
    sz_size_t const step_mask = step - 1;
    sz_u32_t const one = 1;

    // In every iteration we process 4 consecutive sliding windows.
    // Once each of them computes separate values, we step forward (W-1) times,
    // computing all interleaving values. That way the byte spilled from the second
    // hash, can be added to the first one. That way we minimize the number of separate loads.
    for (; text + window_length * 4 + (window_length - 1) <= text_end; text += window_length * 4) {
        hash_low_vec.u32x4 = vld1q_dup_u32(&one);
        hash_high_vec.u32x4 = vld1q_dup_u32(&one);
        for (sz_size_t i = 0; i != window_length; ++i) {
            chars_incoming_vec.u32s[0] = *(uint8_t const *)(text + window_length * 0 + i);
            chars_incoming_vec.u32s[1] = *(uint8_t const *)(text + window_length * 1 + i);
            chars_incoming_vec.u32s[2] = *(uint8_t const *)(text + window_length * 2 + i);
            chars_incoming_vec.u32s[3] = *(uint8_t const *)(text + window_length * 3 + i);
            chars_incoming_shifted_vec.u8x16 = vaddq_u8(chars_incoming_vec.u8x16, vld1q_dup_u8(&high_shift));

            // Append new data.
            hash_low_vec.u32x4 = vmlaq_n_u32(chars_incoming_vec.u32x4, hash_low_vec.u32x4, 31u);
            hash_high_vec.u32x4 = vmlaq_n_u32(chars_incoming_shifted_vec.u32x4, hash_high_vec.u32x4, 257u);
            hash_low_vec.u32x4 = vbslq_u32(hash_low_vec.u32x4, vsubq_u32(hash_low_vec.u32x4, vld1q_dup_u32(&prime)),
                                           vcgtq_u32(hash_low_vec.u32x4, vld1q_dup_u32(&prime)));
            hash_high_vec.u32x4 = vbslq_u32(hash_high_vec.u32x4, vsubq_u32(hash_high_vec.u32x4, vld1q_dup_u32(&prime)),
                                            vcgtq_u32(hash_high_vec.u32x4, vld1q_dup_u32(&prime)));
        }

        if ((cycles & step_mask) == 0) {
            interleave_uint32x4_to_uint64x2(hash_low_vec.u32x4, hash_high_vec.u32x4, &hash_mix01_vec.u64x2,
                                            &hash_mix23_vec.u64x2);
            callback((sz_cptr_t)(text + window_length * 0), window_length, hash_mix01_vec.u64s[0], callback_handle);
            callback((sz_cptr_t)(text + window_length * 1), window_length, hash_mix01_vec.u64s[1], callback_handle);
            callback((sz_cptr_t)(text + window_length * 2), window_length, hash_mix23_vec.u64s[0], callback_handle);
            callback((sz_cptr_t)(text + window_length * 3), window_length, hash_mix23_vec.u64s[1], callback_handle);
        }
        ++cycles;

        for (sz_size_t i = 0; i + 1 != window_length; ++i, ++cycles) {
            // Now, to compute 4 hashes per iteration, instead of loading 8 separate bytes (4 incoming and 4 outgoing)
            // we can limit ourselves to only 5 values, 3 of which will be reused for both append and erase operations.
            // Vectorizing these loads is a huge opportunity for performance optimizations, but naive prefetching
            // into the register just makes things worse.
            chars_outgoing_vec.u32s[0] = *(uint8_t const *)(text + window_length * 0 + i);
            chars_outgoing_vec.u32s[1] = chars_incoming_vec.u32s[0] = *(uint8_t const *)(text + window_length * 1 + i);
            chars_outgoing_vec.u32s[2] = chars_incoming_vec.u32s[1] = *(uint8_t const *)(text + window_length * 2 + i);
            chars_outgoing_vec.u32s[3] = chars_incoming_vec.u32s[2] = *(uint8_t const *)(text + window_length * 3 + i);
            chars_incoming_vec.u32s[3] = *(uint8_t const *)(text + window_length * 4 + i);
            chars_incoming_shifted_vec.u8x16 = vaddq_u8(chars_incoming_vec.u8x16, vld1q_dup_u8(&high_shift));
            chars_outgoing_shifted_vec.u8x16 = vaddq_u8(chars_outgoing_vec.u8x16, vld1q_dup_u8(&high_shift));

            // Drop old data.
            hash_low_vec.u32x4 = vmlsq_n_u32(hash_low_vec.u32x4, chars_outgoing_vec.u32x4, prime_power_low);
            hash_high_vec.u32x4 = vmlsq_n_u32(hash_high_vec.u32x4, chars_outgoing_shifted_vec.u32x4, prime_power_high);

            // Append new data.
            hash_low_vec.u32x4 = vmlaq_n_u32(chars_incoming_vec.u32x4, hash_low_vec.u32x4, 31u);
            hash_high_vec.u32x4 = vmlaq_n_u32(chars_incoming_shifted_vec.u32x4, hash_high_vec.u32x4, 257u);
            hash_low_vec.u32x4 = vbslq_u32(hash_low_vec.u32x4, vsubq_u32(hash_low_vec.u32x4, vld1q_dup_u32(&prime)),
                                           vcgtq_u32(hash_low_vec.u32x4, vld1q_dup_u32(&prime)));
            hash_high_vec.u32x4 = vbslq_u32(hash_high_vec.u32x4, vsubq_u32(hash_high_vec.u32x4, vld1q_dup_u32(&prime)),
                                            vcgtq_u32(hash_high_vec.u32x4, vld1q_dup_u32(&prime)));

            // Mix and call the user if needed
            if ((cycles & step_mask) == 0) {
                interleave_uint32x4_to_uint64x2(hash_low_vec.u32x4, hash_high_vec.u32x4, &hash_mix01_vec.u64x2,
                                                &hash_mix23_vec.u64x2);
                callback((sz_cptr_t)(text + window_length * 0), window_length, hash_mix01_vec.u64s[0], callback_handle);
                callback((sz_cptr_t)(text + window_length * 1), window_length, hash_mix01_vec.u64s[1], callback_handle);
                callback((sz_cptr_t)(text + window_length * 2), window_length, hash_mix23_vec.u64s[0], callback_handle);
                callback((sz_cptr_t)(text + window_length * 3), window_length, hash_mix23_vec.u64s[1], callback_handle);
            }
        }
    }
}

SZ_PUBLIC void sz_hashes_neon_readahead(sz_cptr_t start, sz_size_t length, sz_size_t window_length, sz_size_t step,
                                        sz_hash_callback_t callback, void *callback_handle) {

    if (length < window_length || !window_length) return;
    if (length < 2 * window_length) {
        sz_hashes_serial(start, length, window_length, step, callback, callback_handle);
        return;
    }

    // Using NEON, we can perform 4 integer multiplications and additions within one register.
    // So let's slice the entire string into 4 overlapping windows, to slide over them in parallel.
    sz_u8_t const *text = (sz_u8_t const *)start;
    sz_u8_t const *text_end = text + length;

    // Prepare the `prime ^ window_length` values, that we are going to use for modulo arithmetic.
    sz_u32_t prime_power_low = 1, prime_power_high = 1;
    for (sz_size_t i = 0; i + 1 < window_length; ++i)
        prime_power_low = (prime_power_low * 31ull) % SZ_U32_MAX_PRIME,
        prime_power_high = (prime_power_high * 257ull) % SZ_U32_MAX_PRIME;

    sz_u128_vec_t hash_low_vec, hash_high_vec, hash_mix01_vec, hash_mix23_vec;
    uint8_t high_shift = 77u;
    uint32_t prime = SZ_U32_MAX_PRIME;

    /// Primary buffers containing four upcasted characters as uint32_t values.
    sz_u128_vec_t chars_outgoing_low_vec, chars_incoming_low_vec;
    sz_u128_vec_t chars_outgoing_high_vec, chars_incoming_high_vec;
    // Let's skip the first window, as we are going to compute it in the loop.
    sz_size_t cycles = 0;
    sz_size_t const step_mask = step - 1;
    sz_u32_t const one = 1;

    // In every iteration we process 4 consecutive sliding windows.
    // Once each of them computes separate values, we step forward (W-1) times,
    // computing all interleaving values. That way the byte spilled from the second
    // hash, can be added to the first one. That way we minimize the number of separate loads.
    sz_size_t read_ahead_length = window_length - 1 + 16; // TODO: Instead of +16 round up to 16 multiple
    for (; text + window_length * 4 + read_ahead_length <= text_end; text += window_length * 4) {
        hash_low_vec.u32x4 = vld1q_dup_u32(&one);
        hash_high_vec.u32x4 = vld1q_dup_u32(&one);

        for (sz_size_t i = 0; i < window_length;) {
            sz_u128_vec_t chars_readahead_vec[4];
            chars_readahead_vec[0].u8x16 = vld1q_u8(text + window_length * 0 + i);
            chars_readahead_vec[1].u8x16 = vld1q_u8(text + window_length * 1 + i);
            chars_readahead_vec[2].u8x16 = vld1q_u8(text + window_length * 2 + i);
            chars_readahead_vec[3].u8x16 = vld1q_u8(text + window_length * 3 + i);

            for (; i != window_length; ++i) {
                chars_incoming_low_vec.u32s[0] = chars_readahead_vec[0].u8x16[i];
                chars_incoming_low_vec.u32s[1] = chars_readahead_vec[1].u8x16[i];
                chars_incoming_low_vec.u32s[2] = chars_readahead_vec[2].u8x16[i];
                chars_incoming_low_vec.u32s[3] = chars_readahead_vec[3].u8x16[i];
                chars_incoming_high_vec.u8x16 = vaddq_u8(chars_incoming_low_vec.u8x16, vld1q_dup_u8(&high_shift));

                // Append new data.
                hash_low_vec.u32x4 = vmlaq_n_u32(chars_incoming_low_vec.u32x4, hash_low_vec.u32x4, 31u);
                hash_high_vec.u32x4 = vmlaq_n_u32(chars_incoming_high_vec.u32x4, hash_high_vec.u32x4, 257u);
                hash_low_vec.u32x4 = vbslq_u32(hash_low_vec.u32x4, vsubq_u32(hash_low_vec.u32x4, vld1q_dup_u32(&prime)),
                                               vcgtq_u32(hash_low_vec.u32x4, vld1q_dup_u32(&prime)));
                hash_high_vec.u32x4 =
                    vbslq_u32(hash_high_vec.u32x4, vsubq_u32(hash_high_vec.u32x4, vld1q_dup_u32(&prime)),
                              vcgtq_u32(hash_high_vec.u32x4, vld1q_dup_u32(&prime)));
            }
        }

        if ((cycles & step_mask) == 0) {
            interleave_uint32x4_to_uint64x2(hash_low_vec.u32x4, hash_high_vec.u32x4, &hash_mix01_vec.u64x2,
                                            &hash_mix23_vec.u64x2);
            callback((sz_cptr_t)(text + window_length * 0), window_length, hash_mix01_vec.u64s[0], callback_handle);
            callback((sz_cptr_t)(text + window_length * 1), window_length, hash_mix01_vec.u64s[1], callback_handle);
            callback((sz_cptr_t)(text + window_length * 2), window_length, hash_mix23_vec.u64s[0], callback_handle);
            callback((sz_cptr_t)(text + window_length * 3), window_length, hash_mix23_vec.u64s[1], callback_handle);
        }
        ++cycles;

        for (sz_size_t i = 0; i + 1 < window_length; ++i, ++cycles) {
            // Now, to compute 4 hashes per iteration, instead of loading 8 separate bytes (4 incoming and 4 outgoing)
            // we can limit ourselves to only 5 values, 3 of which will be reused for both append and erase operations.
            sz_u128_vec_t chars_readahead_vec[5];
            chars_readahead_vec[0].u8x16 = vld1q_u8(text + window_length * 0 + i);
            chars_readahead_vec[1].u8x16 = vld1q_u8(text + window_length * 1 + i);
            chars_readahead_vec[2].u8x16 = vld1q_u8(text + window_length * 2 + i);
            chars_readahead_vec[3].u8x16 = vld1q_u8(text + window_length * 3 + i);
            chars_readahead_vec[4].u8x16 = vld1q_u8(text + window_length * 4 + i);

            for (; i + 1 < window_length; ++i) {
                // Transpose
                chars_outgoing_low_vec.u32s[0] = chars_readahead_vec[0].u8x16[i];
                chars_outgoing_low_vec.u32s[1] = chars_incoming_low_vec.u32s[0] = chars_readahead_vec[1].u8x16[i];
                chars_outgoing_low_vec.u32s[2] = chars_incoming_low_vec.u32s[1] = chars_readahead_vec[2].u8x16[i];
                chars_outgoing_low_vec.u32s[3] = chars_incoming_low_vec.u32s[2] = chars_readahead_vec[3].u8x16[i];
                chars_incoming_low_vec.u32s[3] = chars_readahead_vec[4].u8x16[i];

                chars_outgoing_high_vec.u8x16 = vaddq_u8(chars_outgoing_low_vec.u8x16, vld1q_dup_u8(&high_shift));
                chars_incoming_high_vec.u8x16 = vaddq_u8(chars_incoming_low_vec.u8x16, vld1q_dup_u8(&high_shift));

                // Drop old data.
                hash_low_vec.u32x4 = vmlsq_n_u32(hash_low_vec.u32x4, chars_outgoing_low_vec.u32x4, prime_power_low);
                hash_high_vec.u32x4 = vmlsq_n_u32(hash_high_vec.u32x4, chars_outgoing_high_vec.u32x4, prime_power_high);

                // Append new data.
                hash_low_vec.u32x4 = vmlaq_n_u32(chars_incoming_low_vec.u32x4, hash_low_vec.u32x4, 31u);
                hash_high_vec.u32x4 = vmlaq_n_u32(chars_incoming_high_vec.u32x4, hash_high_vec.u32x4, 257u);
                hash_low_vec.u32x4 = vbslq_u32(hash_low_vec.u32x4, vsubq_u32(hash_low_vec.u32x4, vld1q_dup_u32(&prime)),
                                               vcgtq_u32(hash_low_vec.u32x4, vld1q_dup_u32(&prime)));
                hash_high_vec.u32x4 =
                    vbslq_u32(hash_high_vec.u32x4, vsubq_u32(hash_high_vec.u32x4, vld1q_dup_u32(&prime)),
                              vcgtq_u32(hash_high_vec.u32x4, vld1q_dup_u32(&prime)));

                // Mix and call the user if needed
                if ((cycles & step_mask) == 0) {
                    interleave_uint32x4_to_uint64x2(hash_low_vec.u32x4, hash_high_vec.u32x4, &hash_mix01_vec.u64x2,
                                                    &hash_mix23_vec.u64x2);
                    callback((sz_cptr_t)(text + window_length * 0), window_length, hash_mix01_vec.u64s[0],
                             callback_handle);
                    callback((sz_cptr_t)(text + window_length * 1), window_length, hash_mix01_vec.u64s[1],
                             callback_handle);
                    callback((sz_cptr_t)(text + window_length * 2), window_length, hash_mix23_vec.u64s[0],
                             callback_handle);
                    callback((sz_cptr_t)(text + window_length * 3), window_length, hash_mix23_vec.u64s[1],
                             callback_handle);
                }
            }
        }
    }
}

#endif // SZ_USE_ARM_NEON

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STRINGZILLA_EXPERIMENTAL_H_

SZ_PUBLIC sz_ordering_t sz_order_avx2(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {

    // _bswap64;

    // while (a_length >= 8 && b_length >= 8) {
    //     sz_u64_t a_u64 = *(sz_u64_t *)a;
    //     sz_u64_t b_u64 = *(sz_u64_t *)b;
    //     if (a_u64 != b_u64) return _sz_order_scalars(a_u64, b_u64);
    //     a += 8, b += 8, a_length -= 8, b_length -= 8;
    // }

    // The rare case, when both string are very long surves as a great example to understand
    // the basic logic of the algorithm without the complexity of `("abc\0" < "abc")` corner cases.
    while ((a_length >= 64) & (b_length >= 64)) {
        a_vec.zmm = _mm512_loadu_si512(a);
        b_vec.zmm = _mm512_loadu_si512(b);
        // The AVX-512 `_mm512_mask_cmpneq_epi8_mask` intrinsics are generally handy in such environments.
        // They, however, have latency 3 on most modern CPUs. Using AVX2: `_mm256_cmpeq_epi8` would have
        // been cheaper, if we didn't have to apply `_mm256_movemask_epi8` afterwards.
        //
        //      __mmask64 mask_not_equal = _mm512_cmpneq_epi8_mask(a_vec.zmm, b_vec.zmm);
        //      if (mask_not_equal != 0) {
        //          sz_u64_t first_diff = _tzcnt_u64(mask_not_equal);
        //          char a_char = a[first_diff];
        //          char b_char = b[first_diff];
        //          return _sz_order_scalars(a_char, b_char);
        //      }
        //
        // A wiser approach to avoid serial code, is to perform 2 vector comparisons instead of quality check.
        __mmask64 less_mask = _mm512_cmplt_epu8_mask(a_vec.zmm, b_vec.zmm);
        __mmask64 greater_mask = _mm512_cmpgt_epu8_mask(a_vec.zmm, b_vec.zmm);
        // Let's assume both strings are exactly 64 bytes long, like `("abcdabcd..." < "acbdacbd...")`.
        // In that case:
        //      - if `less_mask == 0 && greater_mask == 0`, the strings are equal, and we can skip 64 bytes.
        //      - if `_tzcnt_u64(less_mask) < _tzcnt_u64(greater_mask)` than the first string is less than the second.
        // The `_tzcnt_u64` trailing zeros computation, however, also has latency of 3 cycles.
        unsigned char all_equal = _kortestz_mask8_u8(less_mask, greater_mask);
        if (all_equal) { a += 64, b += 64, a_length -= 64, b_length -= 64; }
        else { return _sz_order_scalars(_tzcnt_u64(less_mask), _tzcnt_u64(greater_mask)); }
    }

    // Assume a case like `("abc\0" < "abc")`.
    // Knowing the length masks of both strings, we can find the bytes that make up the difference
    // and enable them in the `greater_mask`, to signal the presence of null-characters in the end.
    //
    //      __mmask64 a_mask = _sz_u64_clamp_mask_until(a_length);
    //      __mmask64 b_mask = _sz_u64_clamp_mask_until(b_length);
    //      a_vec.zmm = _mm512_maskz_loadu_epi8(a_mask, a);
    //      b_vec.zmm = _mm512_maskz_loadu_epi8(b_mask, b);
    //      __mmask64 after_a_before_b_mask = _kandn_mask64(a_mask, b_mask);
    //      __mmask64 after_b_before_a_mask = _kandn_mask64(b_mask, a_mask);
    //      __mmask64 less_mask = _mm512_cmplt_epu8_mask(a_vec.zmm, b_vec.zmm);
    //      __mmask64 greater_mask = _mm512_cmpgt_epu8_mask(a_vec.zmm, b_vec.zmm);
    //      less_mask = _kor_mask64(less_mask, after_a_before_b_mask);
    //      greater_mask = _kor_mask64(greater_mask, after_b_before_a_mask);
    //      unsigned char all_equal = _kortestz_mask8_u8(less_mask, greater_mask);
    //      if (all_equal) { return sz_equal_k; }
    //      else { return earlier_in_less_mask ? sz_less_k : sz_greater_k; }
    return sz_order_serial(a, a_length, b, b_length);
}

SZ_PUBLIC sz_ordering_t sz_order_avx512(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    sz_u512_vec_t a_vec, b_vec;

    // The rare case, when both string are very long surves as a great example to understand
    // the basic logic of the algorithm without the complexity of `("abc\0" < "abc")` corner cases.
    while ((a_length >= 64) & (b_length >= 64)) {
        a_vec.zmm = _mm512_loadu_si512(a);
        b_vec.zmm = _mm512_loadu_si512(b);
        // The AVX-512 `_mm512_mask_cmpneq_epi8_mask` intrinsics are generally handy in such environments.
        // They, however, have latency 3 on most modern CPUs. Using AVX2: `_mm256_cmpeq_epi8` would have
        // been cheaper, if we didn't have to apply `_mm256_movemask_epi8` afterwards.
        //
        //      __mmask64 mask_not_equal = _mm512_cmpneq_epi8_mask(a_vec.zmm, b_vec.zmm);
        //      if (mask_not_equal != 0) {
        //          sz_u64_t first_diff = _tzcnt_u64(mask_not_equal);
        //          char a_char = a[first_diff];
        //          char b_char = b[first_diff];
        //          return _sz_order_scalars(a_char, b_char);
        //      }
        //
        // A wiser approach to avoid serial code, is to perform 2 vector comparisons instead of quality check.
        __mmask64 less_mask = _mm512_cmplt_epu8_mask(a_vec.zmm, b_vec.zmm);
        __mmask64 greater_mask = _mm512_cmpgt_epu8_mask(a_vec.zmm, b_vec.zmm);
        // Let's assume both strings are exactly 64 bytes long, like `("abcdabcd..." < "acbdacbd...")`.
        // In that case:
        //      - if `less_mask == 0 && greater_mask == 0`, the strings are equal, and we can skip 64 bytes.
        //      - if `_tzcnt_u64(less_mask) < _tzcnt_u64(greater_mask)` than the first string is less than the second.
        // The `_tzcnt_u64` trailing zeros computation, however, also has latency of 3 cycles.
        unsigned char all_equal = _kortestz_mask8_u8(less_mask, greater_mask);
        if (all_equal) { a += 64, b += 64, a_length -= 64, b_length -= 64; }
        else { return _sz_order_scalars(_tzcnt_u64(less_mask), _tzcnt_u64(greater_mask)); }
    }

    // Assume a case like `("abc\0" < "abc")`.
    // Knowing the length masks of both strings, we can find the bytes that make up the difference
    // and enable them in the `greater_mask`, to signal the presence of null-characters in the end.
    //
    //      __mmask64 a_mask = _sz_u64_clamp_mask_until(a_length);
    //      __mmask64 b_mask = _sz_u64_clamp_mask_until(b_length);
    //      a_vec.zmm = _mm512_maskz_loadu_epi8(a_mask, a);
    //      b_vec.zmm = _mm512_maskz_loadu_epi8(b_mask, b);
    //      __mmask64 after_a_before_b_mask = _kandn_mask64(a_mask, b_mask);
    //      __mmask64 after_b_before_a_mask = _kandn_mask64(b_mask, a_mask);
    //      __mmask64 less_mask = _mm512_cmplt_epu8_mask(a_vec.zmm, b_vec.zmm);
    //      __mmask64 greater_mask = _mm512_cmpgt_epu8_mask(a_vec.zmm, b_vec.zmm);
    //      less_mask = _kor_mask64(less_mask, after_a_before_b_mask);
    //      greater_mask = _kor_mask64(greater_mask, after_b_before_a_mask);
    //      unsigned char all_equal = _kortestz_mask8_u8(less_mask, greater_mask);
    //      if (all_equal) { return sz_equal_k; }
    //      else { return earlier_in_less_mask ? sz_less_k : sz_greater_k; }
    return sz_order_serial(a, a_length, b, b_length);
}

SZ_PUBLIC void sz_move_avx512(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    if (target == source) return; // Don't be silly, don't move the data if it's already there.

    // If the regions don't overlap at all, just use "copy" and save some brain cells thinking about corner cases.
    if (target + length < source || target >= source + length) {
        sz_copy_avx512(target, source, length);
        return;
    }

    // The absolute most common case of using "moves" is shifting the data within a continuous buffer
    // when adding a removing some values in it. In such cases, a typical shift is by 1, 2, 4, 8, 16,
    // or 32 bytes, rarely larger. For small shifts, under the size of the ZMM register, we can use shuffles.
    //
    // Remember: if we are shifting data left, that we are traversing to the right.
    int left_to_right_traversal = source > target;
    sz_size_t shift = left_to_right_traversal ? source - target : target - source;

    if (left_to_right_traversal) {

        // Shift until we reach the ZMM register boundary for the target to avoid unaligned loads.
        for (; (sz_size_t)target % 64 != 0 && length; ++target, ++source, --length) *target = *source;

        // Small shifts of large buffers can minimize the number of times a specific cache line will be touched
        // to guarantee one read and one write per cache line.
        if (shift < 64 && length >= 128) {

            // Now we guarantee, that the shift is from 1 to 63 bytes and the output is aligned.
            // Hopefully, we need to shift more than two ZMM registers, so we could consider `valignr` instruction.
            // Sadly, using `_mm512_alignr_epi8` doesn't make sense, as it operates at a 128-bit granularity.
            //
            //      - `_mm256_alignr_epi8` shifts entire 256-bit register, but we need many of them.
            //      - `_mm512_alignr_epi32` shifts 512-bit chunks, but only if the `shift` is a multiple of 4 bytes.
            //      - `_mm512_alignr_epi64` shifts 512-bit chunks by 8 bytes.
            //
            // All of those have a latency of 1 cycle, and the shift amount must be an immediate value!
            // For 1-byte-shift granularity, the `_mm512_permutex2var_epi8` has a latency of 6 and needs VBMI!
            // The most efficient and broadly compatible alternative would be to use a combination of align and shuffle.
            // A similar approach was outlined in "Byte-wise alignr in AVX512F" by Wojciech Muła.
            // http://0x80.pl/notesen/2016-10-16-avx512-byte-alignr.html
            //
            // That solution, is extremely mouthful, assuming we need compile time constants for the shift amount.
            sz_u512_vec_t first_vec, second_vec, combined_vec;
            // The last `64 - shift` entries of the first register should be moved to its start.
            // The first `shift` entries of the second register should be moved to its end.
            // Then we will combine:
            //      - the first `64 - shift` entries of the first register with
            //      - the first `shift` entries of the second register.
#if 1
            sz_u512_vec_t selector_vec;
            sz_size_t shifted_idx = 0;
            for (; shifted_idx != 64; ++shifted_idx) selector_vec.u8s[shifted_idx] = (sz_u8_t)(shift + shifted_idx);
            // Now that the permutations are prepared, pre-load the first cache line and start the loop.
            first_vec.zmm = _mm512_load_si512(target);
            for (; length >= 128; target += 64, source += 64, length -= 64) {
                second_vec.zmm = _mm512_load_si512(target + 64);
                combined_vec.zmm = _mm512_permutex2var_epi8(first_vec.zmm, selector_vec.zmm, second_vec.zmm);
                sz_assert(combined_vec.u8s[0] == source[0]);
                sz_assert(combined_vec.u8s[63] == source[63]);
                _mm512_store_si512(target, combined_vec.zmm);
                first_vec.zmm = second_vec.zmm;
            }
#else
            sz_u512_vec_t first_byte_permute_vec, second_byte_permute_vec;
            sz_u512_vec_t first_shuffled_vec, second_shuffled_vec;
            for (sz_size_t shifted_idx = 0; shifted_idx != (64 - shift); ++shifted_idx)
                first_byte_permute_vec.u8s[shifted_idx] = (sz_u8_t)(shift + shifted_idx), //
                    second_byte_permute_vec.u8s[shifted_idx] = (sz_u8_t)0xFF;
            for (sz_size_t shifted_idx = 0; shifted_idx != shift; ++shifted_idx)
                first_byte_permute_vec.u8s[64 - shift + shifted_idx] = (sz_u8_t)0xFF, //
                    second_byte_permute_vec.u8s[64 - shift + shifted_idx] = (sz_u8_t)shifted_idx;
            // The `_mm512_shuffle_epi8` only works within lanes, so we need to permute the lanes.
            int first_lane_permute_mask, second_lane_permute_mask;

            // Now that the permutations are prepared, pre-load the first cache line and start the loop.
            first_vec.zmm = _mm512_load_si512(target);
            for (; length >= 128; target += 64, source += 64, length -= 64) {
                second_vec.zmm = _mm512_load_si512(target + 64);
                first_shuffled_vec.zmm = _mm512_shuffle_epi8(first_vec.zmm, first_byte_permute_vec.zmm);
                second_shuffled_vec.zmm = _mm512_shuffle_epi8(second_vec.zmm, second_byte_permute_vec.zmm);
                sz_assert(first_shuffled_vec.u8s[0] == source[0]);
                sz_assert(second_shuffled_vec.u8s[63] == source[63]);
                combined_vec.zmm = _mm512_or_si512(first_shuffled_vec.zmm, second_shuffled_vec.zmm);
                _mm512_store_si512(target, combined_vec.zmm);
                first_vec.zmm = second_vec.zmm;
            }
#endif
            for (; length; ++target, ++source, --length) *target = *source;
        }
        // With really large shifts we are not going to touch the same register on the load and store.
        // Especially, if we align the stores to the ZMM register size.
        else {
            for (; length >= 64; target += 64, source += 64, length -= 64)
                _mm512_store_si512(target, _mm512_loadu_si512(source));
            // At this point the length is guaranteed to be under 64.
            __mmask64 mask = _sz_u64_mask_until(length);
            _mm512_mask_storeu_epi8(target, mask, _mm512_maskz_loadu_epi8(mask, source));
        }
    }
    else {
        // Shift until we reach the ZMM register boundary for the target to avoid unaligned loads.
        for (; (sz_size_t)(target + length) % 64 != 0 && length; --length) target[length - 1] = source[length - 1];
        // Jump to the end and walk backwards.
        for (target += length, source += length; length >= 64; length -= 64)
            _mm512_store_si512(target -= 64, _mm512_loadu_si512(source -= 64));
        // At this point the length is guaranteed to be under 64.
        __mmask64 mask = _sz_u64_mask_until(length);
        _mm512_mask_storeu_epi8(target - length, mask, _mm512_maskz_loadu_epi8(mask, source - length));
    }
}

SZ_PUBLIC void sz_move_avx512(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    if (target == source) return; // Don't be silly, don't move the data if it's already there.

    // If the regions don't overlap at all, just use "copy" and save some brain cells thinking about corner cases.
    if (target + length < source || target >= source + length) {
        sz_copy_avx512(target, source, length);
        return;
    }

    // On very short buffers, that are one cache line in width or less, we don't need any loops.
    if (length <= 64) {
        __mmask64 mask = _sz_u64_mask_until(length);
        _mm512_mask_storeu_epi8(target, mask, _mm512_maskz_loadu_epi8(mask, source));
        return;
    }

    // When the buffer is over 64 bytes, it's guaranteed to touch at least two cache lines - the head and tail,
    // and may include more cache-lines in-between. Knowing this, we can avoid expensive unaligned stores
    // by computing 2 masks - for the head and tail, using masked stores for the head and tail, and unmasked
    // for the body.
    sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64; // 63 or less.
    sz_size_t tail_length = (sz_size_t)(target + length) % 64;    // 63 or less.
    sz_size_t body_length = length - head_length - tail_length;   // Multiple of 64.
    __mmask64 head_mask = _sz_u64_mask_until(head_length);
    __mmask64 tail_mask = _sz_u64_mask_until(tail_length);

    // The absolute most common case of using "moves" is shifting the data within a continuous buffer
    // when adding a removing some values in it. In such cases, a typical shift is by 1, 2, 4, 8, 16,
    // or 32 bytes, rarely larger. For small shifts, under the size of the ZMM register, we can use shuffles.
    //
    // Remember:
    //      - if we are shifting data left, that we are traversing to the right.
    //      - if we are shifting data right, that we are traversing to the left.
    int const left_to_right_traversal = source > target;

    // If both targets are equally aligned or misaligned, the efficient implementation is trivial.
    if ((sz_size_t)target % 64 == (sz_size_t)source % 64) {
        if (left_to_right_traversal) {
            // Head, body, and tail.
            _mm512_mask_storeu_epi8(target, head_mask, _mm512_maskz_loadu_epi8(head_mask, source));
            target += head_length, source += head_length, body_length -= head_length;
            for (; body_length >= 64; target += 64, source += 64, body_length -= 64)
                _mm512_store_si512(target, _mm512_load_si512(source));
            _mm512_mask_storeu_epi8(target, tail_mask, _mm512_maskz_loadu_epi8(tail_mask, source));
        }
        else {
            // Tail, body, and head.
            _mm512_mask_storeu_epi8(target + head_length + body_length, tail_mask,
                                    _mm512_maskz_loadu_epi8(tail_mask, source + head_length + body_length));
            for (; body_length >= 64; body_length -= 64)
                _mm512_store_si512(target + head_length + body_length - 64,
                                   _mm512_load_si512(source + head_length + body_length - 64));
            _mm512_mask_storeu_epi8(target, head_mask, _mm512_maskz_loadu_epi8(head_mask, source));
        }
        return;
    }

    // Now we guarantee, that the relative shift within is from 1 to 63 bytes and the output is aligned.
    // Hopefully, we need to shift more than two ZMM registers, so we could consider `valignr` instruction.
    // Sadly, using `_mm512_alignr_epi8` doesn't make sense, as it operates at a 128-bit granularity.
    //
    //      - `_mm256_alignr_epi8` shifts entire 256-bit register, but we need many of them.
    //      - `_mm512_alignr_epi32` shifts 512-bit chunks, but only if the `shift` is a multiple of 4 bytes.
    //      - `_mm512_alignr_epi64` shifts 512-bit chunks by 8 bytes.
    //
    // All of those have a latency of 1 cycle, and the shift amount must be an immediate value!
    // For 1-byte-shift granularity, the `_mm512_permutex2var_epi8` has a latency of 6 and needs VBMI!
    // The most efficient and broadly compatible alternative could be to use a combination of align and shuffle.
    // A similar approach was outlined in "Byte-wise alignr in AVX512F" by Wojciech Muła.
    // http://0x80.pl/notesen/2016-10-16-avx512-byte-alignr.html
    //
    // That solution, is extremely mouthful, assuming we need compile time constants for the shift amount.
    // A cleaner one, with a latency of 3 cycles, is to use `_mm512_permutexvar_epi8` or `_mm512_mask_permutexvar_epi8`,
    // which can be seen as combination of a cross-register shuffle and blend, and is available with VBMI.
    sz_size_t const shift = left_to_right_traversal ? source - target : target - source;
    sz_size_t const shift_in_page = shift % 64;

    if (left_to_right_traversal) {
        // Head, body, and tail.
        _mm512_mask_storeu_epi8(target, head_mask, _mm512_maskz_loadu_epi8(head_mask, source));
        target += head_length, source += head_length;

        // Define the permutation vectors for the `permute2var` instruction for the body.
        sz_u512_vec_t first_vec, second_vec, combined_vec, selector_vec;
        selector_vec.zmm = _mm512_set_epi8(                                 //
            63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, //
            47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, //
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, //
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        selector_vec.zmm = _mm512_add_epi8(selector_vec.zmm, _mm512_set1_epi8(shift_in_page));
        selector_vec.zmm = _mm512_and_si512(selector_vec.zmm, _mm512_set1_epi8(63));

        if (body_length >= 128) {
            // Now that the permutations are prepared, pre-load the first cache line and start the loop.
            __mmask64 blend_mask = _sz_u64_mask_until(shift_in_page);
            sz_cptr_t source_page = source - (sz_size_t)source % 64;
            first_vec.zmm = _mm512_load_si512(source_page);
            for (; body_length >= 128; target += 64, source += 64, source_page += 64, body_length -= 64) {
                second_vec.zmm = _mm512_load_si512(source_page + 64);
                second_vec.zmm = _mm512_permutexvar_epi8(selector_vec.zmm, second_vec.zmm);
                combined_vec.zmm = _mm512_mask_blend_epi8(blend_mask, second_vec.zmm, first_vec.zmm);
                sz_assert(combined_vec.u8s[0] == source[0]);
                sz_assert(combined_vec.u8s[63] == source[63]);
                _mm512_store_si512(target, combined_vec.zmm);
                first_vec.zmm = second_vec.zmm;
            }
        }
        if (body_length)
            _mm512_store_si512(target, _mm512_loadu_si512(source)), target += 64, source += 64, body_length -= 64;
        _mm512_mask_storeu_epi8(target, tail_mask, _mm512_maskz_loadu_epi8(tail_mask, source));
    }
    else {
        // Tail, body, and head.
        _mm512_mask_storeu_epi8(target + head_length + body_length, head_mask,
                                _mm512_maskz_loadu_epi8(head_mask, source + head_length + body_length));

        // Define the permutation vectors for the `permute2var` instruction for the body.
        sz_u512_vec_t first_vec, second_vec, combined_vec, selector_vec;
        selector_vec.zmm = _mm512_set_epi8(                                 //
            63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, //
            47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, //
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, //
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        selector_vec.zmm = _mm512_add_epi8(selector_vec.zmm, _mm512_set1_epi8(shift_in_page));
        selector_vec.zmm = _mm512_and_si512(selector_vec.zmm, _mm512_set1_epi8(63));

        if (body_length >= 128) {
            // Now that the permutations are prepared, pre-load the first cache line and start the loop.
            __mmask64 blend_mask = _sz_u64_mask_until(shift_in_page);
            sz_cptr_t source_second_page = source + body_length - (sz_size_t)(source + body_length) % 64;
            first_vec.zmm = _mm512_load_si512(source_second_page);
            for (; body_length >= 128; source_second_page -= 64, body_length -= 64) {
                second_vec.zmm = _mm512_load_si512(source_second_page - 64);
                second_vec.zmm = _mm512_permutexvar_epi8(selector_vec.zmm, second_vec.zmm);
                combined_vec.zmm = _mm512_mask_blend_epi8(blend_mask, second_vec.zmm, first_vec.zmm);
                sz_assert(combined_vec.u8s[0] == source[0]);
                sz_assert(combined_vec.u8s[63] == source[63]);
                _mm512_store_si512(target + head_length + body_length, combined_vec.zmm);
                first_vec.zmm = second_vec.zmm;
            }
        }
        if (body_length) _mm512_store_si512(target + head_length, _mm512_loadu_si512(source + head_length));
        _mm512_mask_storeu_epi8(target, tail_mask, _mm512_maskz_loadu_epi8(tail_mask, source));
    }
}