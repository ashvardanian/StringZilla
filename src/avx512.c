/*
 *  @brief  AVX-512 implementation of the string search algorithms.
 *
 *  Different subsets of AVX-512 were introduced in different years:
 *  * 2017 SkyLake: F, CD, ER, PF, VL, DQ, BW
 *  * 2018 CannonLake: IFMA, VBMI
 *  * 2019 IceLake: VPOPCNTDQ, VNNI, VBMI2, BITALG, GFNI, VPCLMULQDQ, VAES
 *  * 2020 TigerLake: VP2INTERSECT
 */
#if SZ_USE_X86_AVX512
#include <x86intrin.h>

SZ_INTERNAL sz_size_t _sz_levenshtein_avx512_upto63bytes( //
    sz_cptr_t const a, sz_size_t const a_length,          //
    sz_cptr_t const b, sz_size_t const b_length,          //
    sz_ptr_t buffer, sz_size_t const bound) {

    sz_u512_parts_t a_vec, b_vec, previous_vec, current_vec, permutation_vec;
    sz_u512_parts_t cost_deletion_vec, cost_insertion_vec, cost_substitution_vec;
    sz_size_t min_distance;

    b_vec.zmm = _mm512_maskz_loadu_epi8(clamp_mask_up_to(b_length), b);
    previous_vec.zmm = _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, //
                                       47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, //
                                       31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, //
                                       15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    permutation_vec.zmm = _mm512_set_epi8(62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, //
                                          46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, //
                                          30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, //
                                          14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 63);

    for (sz_size_t idx_a = 0; idx_a != a_length; ++idx_a) {
        min_distance = bound;

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
        // Unrolling this:
        //      current_vec.u8s[0 + 1] = sz_min_of_two(current_vec.u8s[0 + 1], current_vec.u8s[0] + 1)
        //      current_vec.u8s[1 + 1] = sz_min_of_two(current_vec.u8s[1 + 1], current_vec.u8s[1] + 1)
        //      current_vec.u8s[2 + 1] = sz_min_of_two(current_vec.u8s[2 + 1], current_vec.u8s[2] + 1)
        //      current_vec.u8s[3 + 1] = sz_min_of_two(current_vec.u8s[3 + 1], current_vec.u8s[3] + 1)
        // Alternatively, using a tree-like reduction in log2 steps:
        //      - 6 cycles of reductions shifting by 1, 2, 4, 8, 16, 32, 64 bytes
        //      - with each cycle containing at least one shift, min, add, blend
        // Which adds meaningless complexity without any performance gains.
        for (sz_size_t idx_b = 0; idx_b != b_length; ++idx_b) {
            sz_u8_t cost_insertion = current_vec.u8s[idx_b] + 1;
            current_vec.u8s[idx_b + 1] = sz_min_of_two(current_vec.u8s[idx_b + 1], cost_insertion);
            min_distance = sz_min_of_two(min_distance, current_vec.u8s[idx_b + 1]);
        }

        // If the minimum distance in this row exceeded the bound, return early
        if (min_distance >= bound) return bound;

        // Swap previous_distances and current_distances pointers
        sz_u512_parts_t temp_vec;
        temp_vec.zmm = previous_vec.zmm;
        previous_vec.zmm = current_vec.zmm;
        current_vec.zmm = temp_vec.zmm;
    }

    return previous_vec.u8s[b_length] < bound ? previous_vec.u8s[b_length] : bound;
}

SZ_PUBLIC sz_size_t sz_levenshtein_avx512(       //
    sz_cptr_t const a, sz_size_t const a_length, //
    sz_cptr_t const b, sz_size_t const b_length, //
    sz_ptr_t buffer, sz_size_t const bound) {

    // If one of the strings is empty - the edit distance is equal to the length of the other one
    if (a_length == 0) return b_length <= bound ? b_length : bound;
    if (b_length == 0) return a_length <= bound ? a_length : bound;

    // If the difference in length is beyond the `bound`, there is no need to check at all
    if (a_length > b_length) {
        if (a_length - b_length > bound) return bound;
    }
    else {
        if (b_length - a_length > bound) return bound;
    }

    // Depending on the length, we may be able to use the optimized implementation
    if (a_length < 63 && b_length < 63)
        return _sz_levenshtein_avx512_upto63bytes(a, a_length, b, b_length, buffer, bound);
    else
        return sz_levenshtein_serial(a, a_length, b, b_length, buffer, bound);
}

/**
 *  @brief  Bitap algorithm for exact matching of patterns under @b 8-bytes long using AVX-512.
 */
sz_cptr_t sz_find_under8byte_avx512(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                    sz_size_t needle_length) {

    // Instead of evaluating one character at a time, we will keep match masks for every character in the lane
    __m512i running_match_vec = _mm512_set1_epi8(~0u);

    // We can't lookup 256 individual bytes efficiently, so we need to separate the bits into separate lookup tables.
    // The separation depends on the kinds of instructions we are allowed to use:
    // - AVX-512_BW has `_mm512_shuffle_epi8` - 1 cycle latency, 1 cycle throughput.
    // - AVX-512_VBMI has `_mm512_multishift_epi64_epi8` - 3 cycle latency, 1 cycle throughput.
    // - AVX-512_F has `_mm512_permutexvar_epi32` - 3 cycle latency, 1 cycle throughput.
    // The `_mm512_permutexvar_epi8` instrinsic is extremely easy to use.
    union {
        __m512i zmm[4];
        sz_u8_t u8[256];
    } pattern_mask;
    for (sz_size_t i = 0; i < 256; ++i) { pattern_mask.u8[i] = ~0u; }
    for (sz_size_t i = 0; i < needle_length; ++i) { pattern_mask.u8[needle[i]] &= ~(1u << i); }

    // Now during matching
    for (sz_size_t i = 0; i < haystack_length; ++i) {
        __m512i haystack_vec = _mm512_load_epi32(haystack);

        // Lookup in all tables
        __m512i pattern_matches_in_first_vec = _mm512_permutexvar_epi8(haystack_vec, pattern_mask.zmm[0]);
        __m512i pattern_matches_in_second_vec = _mm512_permutexvar_epi8(haystack_vec, pattern_mask.zmm[1]);
        __m512i pattern_matches_in_third_vec = _mm512_permutexvar_epi8(haystack_vec, pattern_mask.zmm[2]);
        __m512i pattern_matches_in_fourth_vec = _mm512_permutexvar_epi8(haystack_vec, pattern_mask.zmm[3]);

        // Depending on the value of each character, we will pick different parts
        __mmask64 use_third_or_fourth = _mm512_cmpgt_epi8_mask(haystack_vec, _mm512_set1_epi8(127));
        __mmask64 use_second = _mm512_cmpgt_epi8_mask(haystack_vec, _mm512_set1_epi8(63));
        __mmask64 use_fourth = _mm512_cmpgt_epi8_mask(haystack_vec, _mm512_set1_epi8(128 + 63));
        __m512i pattern_matches =    //
            _mm512_mask_blend_epi8(  //
                use_third_or_fourth, //
                _mm512_mask_blend_epi8(use_second, pattern_matches_in_first_vec, pattern_matches_in_second_vec),
                _mm512_mask_blend_epi8(use_fourth, pattern_matches_in_third_vec, pattern_matches_in_fourth_vec));

        // Now we need to implement the inclusive prefix-sum OR-ing of the match masks,
        // shifting the previous value left by one, similar to this code:
        //      running_match = (running_match << 1) | pattern_mask[haystack[i]];
        //      if ((running_match & (1u << (needle_length - 1))) == 0) { return haystack + i - needle_length + 1; }
        // Assuming our match is at most 8 bytes long, we need no more than 3 invocations of `_mm512_alignr_epi8`
        // and of `_mm512_or_si512`.
        pattern_matches = _mm512_or_si512(pattern_matches, _mm512_alignr_epi8(pattern_matches, running_match_vec, 1));
    }

    return NULL;
}

#endif
