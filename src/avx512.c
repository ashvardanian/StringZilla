/**
 *  @brief  AVX-512 implementation of the string search algorithms.
 *
 *  Different subsets of AVX-512 were introduced in different years:
 *  * 2017 SkyLake: F, CD, ER, PF, VL, DQ, BW
 *  * 2018 CannonLake: IFMA, VBMI
 *  * 2019 IceLake: VPOPCNTDQ, VNNI, VBMI2, BITALG, GFNI, VPCLMULQDQ, VAES
 *  * 2020 TigerLake: VP2INTERSECT
 */
#include <stringzilla/stringzilla.h>

#if SZ_USE_X86_AVX512
#include <x86intrin.h>

/**
 *  @brief  Helper structure to simpify work with 64-bit words.
 */
typedef union sz_u512_parts_t {
    __m512i zmm;
    sz_u64_t u64s[8];
    sz_u32_t u32s[16];
    sz_u16_t u16s[32];
    sz_u8_t u8s[64];
} sz_u512_parts_t;

SZ_INTERNAL __mmask64 clamp_mask_up_to(sz_size_t n) {
    // The simplest approach to compute this if we know that `n` is blow or equal 64:
    //      return (1ull << n) - 1;
    // A slighly more complex approach, if we don't know that `n` is under 64:
    return _bzhi_u64(0xFFFFFFFFFFFFFFFF, n < 64 ? n : 64);
}

SZ_INTERNAL __mmask64 mask_up_to(sz_size_t n) {
    // The simplest approach to compute this if we know that `n` is blow or equal 64:
    //      return (1ull << n) - 1;
    // A slighly more complex approach, if we don't know that `n` is under 64:
    return _bzhi_u64(0xFFFFFFFFFFFFFFFF, n);
}

SZ_PUBLIC sz_ordering_t sz_order_avx512(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    sz_ordering_t ordering_lookup[2] = {sz_greater_k, sz_less_k};
    __m512i a_vec, b_vec;

sz_order_avx512_cycle:
    // In most common scenarios at least one of the strings is under 64 bytes.
    if ((a_length < 64) + (b_length < 64)) {
        __mmask64 a_mask = clamp_mask_up_to(a_length);
        __mmask64 b_mask = clamp_mask_up_to(b_length);
        a_vec = _mm512_maskz_loadu_epi8(a_mask, a);
        b_vec = _mm512_maskz_loadu_epi8(b_mask, b);
        // The AVX-512 `_mm512_mask_cmpneq_epi8_mask` intrinsics are generally handy in such environments.
        // They, however, have latency 3 on most modern CPUs. Using AVX2: `_mm256_cmpeq_epi8` would have
        // been cheaper, if we didn't have to apply `_mm256_movemask_epi8` afterwards.
        __mmask64 mask_not_equal = _mm512_cmpneq_epi8_mask(a_vec, b_vec);
        if (mask_not_equal != 0) {
            int first_diff = _tzcnt_u64(mask_not_equal);
            char a_char = a[first_diff];
            char b_char = b[first_diff];
            return ordering_lookup[a_char < b_char];
        }
        else
            // From logic perspective, the hardest cases are "abc\0" and "abc".
            // The result must be `sz_greater_k`, as the latter is shorter.
            return a_length != b_length ? ordering_lookup[a_length < b_length] : sz_equal_k;
    }
    else {
        a_vec = _mm512_loadu_epi8(a);
        b_vec = _mm512_loadu_epi8(b);
        __mmask64 mask_not_equal = _mm512_cmpneq_epi8_mask(a_vec, b_vec);
        if (mask_not_equal != 0) {
            int first_diff = _tzcnt_u64(mask_not_equal);
            char a_char = a[first_diff];
            char b_char = b[first_diff];
            return ordering_lookup[a_char < b_char];
        }
        a += 64, b += 64, a_length -= 64, b_length -= 64;
        if ((a_length > 0) + (b_length > 0)) goto sz_order_avx512_cycle;
        return a_length != b_length ? ordering_lookup[a_length < b_length] : sz_equal_k;
    }
}

SZ_PUBLIC sz_bool_t sz_equal_avx512(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    __m512i a_vec, b_vec;
    __mmask64 mask;

sz_equal_avx512_cycle:
    if (length < 64) {
        mask = mask_up_to(length);
        a_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_vec = _mm512_maskz_loadu_epi8(mask, b);
        // Reuse the same `mask` variable to find the bit that doesn't match
        mask = _mm512_mask_cmpneq_epi8_mask(mask, a_vec, b_vec);
        return mask == 0;
    }
    else {
        a_vec = _mm512_loadu_epi8(a);
        b_vec = _mm512_loadu_epi8(b);
        mask = _mm512_cmpneq_epi8_mask(a_vec, b_vec);
        if (mask != 0) return sz_false_k;
        a += 64, b += 64, length -= 64;
        if (length) goto sz_equal_avx512_cycle;
        return sz_true_k;
    }
}

SZ_PUBLIC sz_cptr_t sz_find_byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    __m512i h_vec, n_vec = _mm512_set1_epi8(n[0]);
    __mmask64 mask;

sz_find_byte_avx512_cycle:
    if (h_length < 64) {
        mask = mask_up_to(h_length);
        h_vec = _mm512_maskz_loadu_epi8(mask, h);
        // Reuse the same `mask` variable to find the bit that doesn't match
        mask = _mm512_mask_cmpeq_epu8_mask(mask, h_vec, n_vec);
        if (mask) return h + sz_ctz64(mask);
    }
    else {
        h_vec = _mm512_loadu_epi8(h);
        mask = _mm512_cmpeq_epi8_mask(h_vec, n_vec);
        if (mask) return h + sz_ctz64(mask);
        h += 64, h_length -= 64;
        if (h_length) goto sz_find_byte_avx512_cycle;
    }
    return NULL;
}

SZ_PUBLIC sz_cptr_t sz_find_2byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    sz_u64_parts_t n_parts;
    n_parts.u64 = 0;
    n_parts.u8s[0] = n[0];
    n_parts.u8s[1] = n[1];

    // A simpler approach would ahve been to use two separate registers for
    // different characters of the needle, but that would use more registers.
    __m512i h0_vec, h1_vec, n_vec = _mm512_set1_epi16(n_parts.u16s[0]);
    __mmask64 mask;
    __mmask32 matches0, matches1;

sz_find_2byte_avx512_cycle:
    if (h_length < 2) { return NULL; }
    else if (h_length < 66) {
        mask = mask_up_to(h_length);
        h0_vec = _mm512_maskz_loadu_epi8(mask, h);
        h1_vec = _mm512_maskz_loadu_epi8(mask, h + 1);
        matches0 = _mm512_mask_cmpeq_epi16_mask(mask, h0_vec, n_vec);
        matches1 = _mm512_mask_cmpeq_epi16_mask(mask, h1_vec, n_vec);
        if (matches0 | matches1)
            return h + sz_ctz64(_pdep_u64(matches0, 0x5555555555555555) | //
                                _pdep_u64(matches1, 0xAAAAAAAAAAAAAAAA));
        return NULL;
    }
    else {
        h0_vec = _mm512_loadu_epi8(h);
        h1_vec = _mm512_loadu_epi8(h + 1);
        matches0 = _mm512_cmpeq_epi16_mask(h0_vec, n_vec);
        matches1 = _mm512_cmpeq_epi16_mask(h1_vec, n_vec);
        // https://lemire.me/blog/2018/01/08/how-fast-can-you-bit-interleave-32-bit-integers/
        if (matches0 | matches1)
            return h + sz_ctz64(_pdep_u64(matches0, 0x5555555555555555) | //
                                _pdep_u64(matches1, 0xAAAAAAAAAAAAAAAA));
        h += 64, h_length -= 64;
        goto sz_find_2byte_avx512_cycle;
    }
}

SZ_PUBLIC sz_cptr_t sz_find_4byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    sz_u64_parts_t n_parts;
    n_parts.u64 = 0;
    n_parts.u8s[0] = n[0];
    n_parts.u8s[1] = n[1];
    n_parts.u8s[2] = n[2];
    n_parts.u8s[3] = n[3];

    __m512i h0_vec, h1_vec, h2_vec, h3_vec, n_vec = _mm512_set1_epi32(n_parts.u32s[0]);
    __mmask64 mask;
    __mmask16 matches0, matches1, matches2, matches3;

sz_find_4byte_avx512_cycle:
    if (h_length < 4) { return NULL; }
    else if (h_length < 68) {
        mask = mask_up_to(h_length);
        h0_vec = _mm512_maskz_loadu_epi8(mask, h);
        h1_vec = _mm512_maskz_loadu_epi8(mask, h + 1);
        h2_vec = _mm512_maskz_loadu_epi8(mask, h + 2);
        h3_vec = _mm512_maskz_loadu_epi8(mask, h + 3);
        matches0 = _mm512_mask_cmpeq_epi32_mask(mask, h0_vec, n_vec);
        matches1 = _mm512_mask_cmpeq_epi32_mask(mask, h1_vec, n_vec);
        matches2 = _mm512_mask_cmpeq_epi32_mask(mask, h2_vec, n_vec);
        matches3 = _mm512_mask_cmpeq_epi32_mask(mask, h3_vec, n_vec);
        if (matches0 | matches1 | matches2 | matches3)
            return h + sz_ctz64(_pdep_u64(matches0, 0x1111111111111111) | //
                                _pdep_u64(matches1, 0x2222222222222222) | //
                                _pdep_u64(matches2, 0x4444444444444444) | //
                                _pdep_u64(matches3, 0x8888888888888888));
        return NULL;
    }
    else {
        h0_vec = _mm512_loadu_epi8(h);
        h1_vec = _mm512_loadu_epi8(h + 1);
        h2_vec = _mm512_loadu_epi8(h + 2);
        h3_vec = _mm512_loadu_epi8(h + 3);
        matches0 = _mm512_cmpeq_epi32_mask(h0_vec, n_vec);
        matches1 = _mm512_cmpeq_epi32_mask(h1_vec, n_vec);
        matches2 = _mm512_cmpeq_epi32_mask(h2_vec, n_vec);
        matches3 = _mm512_cmpeq_epi32_mask(h3_vec, n_vec);
        if (matches0 | matches1 | matches2 | matches3)
            return h + sz_ctz64(_pdep_u64(matches0, 0x1111111111111111) | //
                                _pdep_u64(matches1, 0x2222222222222222) | //
                                _pdep_u64(matches2, 0x4444444444444444) | //
                                _pdep_u64(matches3, 0x8888888888888888));
        h += 64, h_length -= 64;
        goto sz_find_4byte_avx512_cycle;
    }
}

SZ_PUBLIC sz_cptr_t sz_find_3byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    sz_u64_parts_t n_parts;
    n_parts.u64 = 0;
    n_parts.u8s[0] = n[0];
    n_parts.u8s[1] = n[1];
    n_parts.u8s[2] = n[2];

    // A simpler approach would ahve been to use two separate registers for
    // different characters of the needle, but that would use more registers.
    __m512i h0_vec, h1_vec, h2_vec, h3_vec, n_vec = _mm512_set1_epi32(n_parts.u32s[0]);
    __mmask64 mask;
    __mmask16 matches0, matches1, matches2, matches3;

sz_find_3byte_avx512_cycle:
    if (h_length < 3) { return NULL; }
    else if (h_length < 67) {
        mask = mask_up_to(h_length);
        // This implementation is more complex than the `sz_find_4byte_avx512`,
        // as we are going to match only 3 bytes within each 4-byte word.
        h0_vec = _mm512_maskz_loadu_epi8(mask & 0x7777777777777777, h);
        h1_vec = _mm512_maskz_loadu_epi8(mask & 0x7777777777777777, h + 1);
        h2_vec = _mm512_maskz_loadu_epi8(mask & 0x7777777777777777, h + 2);
        h3_vec = _mm512_maskz_loadu_epi8(mask & 0x7777777777777777, h + 3);
        matches0 = _mm512_mask_cmpeq_epi32_mask(mask, h0_vec, n_vec);
        matches1 = _mm512_mask_cmpeq_epi32_mask(mask, h1_vec, n_vec);
        matches2 = _mm512_mask_cmpeq_epi32_mask(mask, h2_vec, n_vec);
        matches3 = _mm512_mask_cmpeq_epi32_mask(mask, h3_vec, n_vec);
        if (matches0 | matches1 | matches2 | matches3)
            return h + sz_ctz64(_pdep_u64(matches0, 0x1111111111111111) | //
                                _pdep_u64(matches1, 0x2222222222222222) | //
                                _pdep_u64(matches2, 0x4444444444444444) | //
                                _pdep_u64(matches3, 0x8888888888888888));
        return NULL;
    }
    else {
        h0_vec = _mm512_maskz_loadu_epi8(0x7777777777777777, h);
        h1_vec = _mm512_maskz_loadu_epi8(0x7777777777777777, h + 1);
        h2_vec = _mm512_maskz_loadu_epi8(0x7777777777777777, h + 2);
        h3_vec = _mm512_maskz_loadu_epi8(0x7777777777777777, h + 3);
        matches0 = _mm512_cmpeq_epi32_mask(h0_vec, n_vec);
        matches1 = _mm512_cmpeq_epi32_mask(h1_vec, n_vec);
        matches2 = _mm512_cmpeq_epi32_mask(h2_vec, n_vec);
        matches3 = _mm512_cmpeq_epi32_mask(h3_vec, n_vec);
        if (matches0 | matches1 | matches2 | matches3)
            return h + sz_ctz64(_pdep_u64(matches0, 0x1111111111111111) | //
                                _pdep_u64(matches1, 0x2222222222222222) | //
                                _pdep_u64(matches2, 0x4444444444444444) | //
                                _pdep_u64(matches3, 0x8888888888888888));
        h += 64, h_length -= 64;
        goto sz_find_3byte_avx512_cycle;
    }
}

SZ_PUBLIC sz_cptr_t sz_find_under66byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    __mmask64 mask, n_length_body_mask = mask_up_to(n_length - 2);
    __mmask64 matches;
    sz_u512_parts_t h_first_vec, h_last_vec, h_body_vec, n_first_vec, n_last_vec, n_body_vec;
    n_first_vec.zmm = _mm512_set1_epi8(n[0]);
    n_last_vec.zmm = _mm512_set1_epi8(n[n_length - 1]);
    n_body_vec.zmm = _mm512_maskz_loadu_epi8(n_length_body_mask, n + 1);

sz_find_under66byte_avx512_cycle:
    if (h_length < n_length) { return NULL; }
    else if (h_length < n_length + 64) {
        mask = mask_up_to(h_length);
        h_first_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        h_last_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + n_length - 1);
        matches = _mm512_mask_cmpeq_epi8_mask(mask, h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_mask_cmpeq_epi8_mask(mask, h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_ctz64(matches);
            h_body_vec.zmm = _mm512_maskz_loadu_epi8(n_length_body_mask, h + potential_offset + 1);
            // Might be worth considering the `_mm256_testc_si256` intrinsic, that seems to have a lower latency
            // https://www.agner.org/optimize/blog/read.php?i=318
            if (!_mm512_cmpneq_epi8_mask(h_body_vec.zmm, n_body_vec.zmm)) return h + potential_offset;

            h += potential_offset + 1, h_length -= potential_offset + 1;
            goto sz_find_under66byte_avx512_cycle;
        }
        else
            return NULL;
    }
    else {
        h_first_vec.zmm = _mm512_loadu_epi8(h);
        h_last_vec.zmm = _mm512_loadu_epi8(h + n_length - 1);
        matches = _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_ctz64(matches);
            h_body_vec.zmm = _mm512_maskz_loadu_epi8(n_length_body_mask, h + potential_offset + 1);
            if (!_mm512_cmpneq_epi8_mask(h_body_vec.zmm, n_body_vec.zmm)) return h + potential_offset;

            h += potential_offset + 1, h_length -= potential_offset + 1;
            goto sz_find_under66byte_avx512_cycle;
        }
        else {
            h += 64, h_length -= 64;
            goto sz_find_under66byte_avx512_cycle;
        }
    }
}

SZ_PUBLIC sz_cptr_t sz_find_over66byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    __mmask64 mask;
    __mmask64 matches;
    sz_u512_parts_t h_first_vec, h_last_vec, n_first_vec, n_last_vec;
    n_first_vec.zmm = _mm512_set1_epi8(n[0]);
    n_last_vec.zmm = _mm512_set1_epi8(n[n_length - 1]);

sz_find_over66byte_avx512_cycle:
    if (h_length < n_length) { return NULL; }
    else if (h_length < n_length + 64) {
        mask = mask_up_to(h_length);
        h_first_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        h_last_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + n_length - 1);
        matches = _mm512_mask_cmpeq_epi8_mask(mask, h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_mask_cmpeq_epi8_mask(mask, h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_ctz64(matches);
            if (sz_equal_avx512(h + potential_offset + 1, n + 1, n_length - 2)) return h + potential_offset;

            h += potential_offset + 1, h_length -= potential_offset + 1;
            goto sz_find_over66byte_avx512_cycle;
        }
        else
            return NULL;
    }
    else {
        h_first_vec.zmm = _mm512_loadu_epi8(h);
        h_last_vec.zmm = _mm512_loadu_epi8(h + n_length - 1);
        matches = _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_ctz64(matches);
            if (sz_equal_avx512(h + potential_offset + 1, n + 1, n_length - 2)) return h + potential_offset;

            h += potential_offset + 1, h_length -= potential_offset + 1;
            goto sz_find_over66byte_avx512_cycle;
        }
        else {
            h += 64, h_length -= 64;
            goto sz_find_over66byte_avx512_cycle;
        }
    }
}

SZ_PUBLIC sz_cptr_t sz_find_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    switch (n_length) {
    case 0: return NULL;
    case 1: return sz_find_byte_avx512(h, h_length, n);
    case 2: return sz_find_2byte_avx512(h, h_length, n);
    case 3: return sz_find_3byte_avx512(h, h_length, n);
    case 4: return sz_find_4byte_avx512(h, h_length, n);
    default:
    }

    if (n_length <= 66) { return sz_find_under66byte_avx512(h, h_length, n, n_length); }
    else { return sz_find_over66byte_avx512(h, h_length, n, n_length); }
}

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
