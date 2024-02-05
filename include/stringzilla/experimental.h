/**
 *  @brief  Experimental kernels for StringZilla.
 *  @file   experimental.h
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

#endif // SZ_USE_AVX512

#if SZ_USE_ARM_NEON

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

SZ_PUBLIC void sz_hashes_neon_readhead(sz_cptr_t start, sz_size_t length, sz_size_t window_length, sz_size_t step,
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