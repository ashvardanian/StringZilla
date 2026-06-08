/**
 *  @brief Skylake (AVX-512) backend for string comparison utilities.
 *  @file include/stringzilla/compare/skylake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/compare.h
 */
#ifndef STRINGZILLA_COMPARE_SKYLAKE_H_
#define STRINGZILLA_COMPARE_SKYLAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/compare/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2")
#endif

SZ_PUBLIC sz_ordering_t sz_order_skylake(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    sz_u512_vec_t a_vec, b_vec;

    // Pointer arithmetic is cheap, fetching memory is not!
    // So we can use the masked loads to fetch at most one cache-line for each string,
    // compare the prefixes, and only then move forward.
    sz_size_t a_head_length = 64 - ((sz_size_t)a % 64); // 63 or less.
    sz_size_t b_head_length = 64 - ((sz_size_t)b % 64); // 63 or less.
    a_head_length = a_head_length < a_length ? a_head_length : a_length;
    b_head_length = b_head_length < b_length ? b_head_length : b_length;
    sz_size_t head_length = a_head_length < b_head_length ? a_head_length : b_head_length;
    __mmask64 head_mask = sz_u64_mask_until_(head_length);
    a_vec.zmm = _mm512_maskz_loadu_epi8(head_mask, a);
    b_vec.zmm = _mm512_maskz_loadu_epi8(head_mask, b);
    __mmask64 mask_not_equal = _mm512_cmpneq_epi8_mask(a_vec.zmm, b_vec.zmm);
    if (mask_not_equal != 0) {
        // Reload from original memory (L1 cached) to avoid ZMM-to-stack spill.
        unsigned long long first_diff = _tzcnt_u64(mask_not_equal);
        return sz_order_scalars_(a[first_diff], b[first_diff]);
    }
    else if (head_length == a_length && head_length == b_length) { return sz_equal_k; }
    else { a += head_length, b += head_length, a_length -= head_length, b_length -= head_length; }

    // The rare case, when both string are very long.
    __mmask64 a_mask, b_mask;
    while ((a_length >= 64) & (b_length >= 64)) {
        a_vec.zmm = _mm512_loadu_si512(a);
        b_vec.zmm = _mm512_loadu_si512(b);
        mask_not_equal = _mm512_cmpneq_epi8_mask(a_vec.zmm, b_vec.zmm);
        if (mask_not_equal != 0) {
            // Reload from original memory (L1 cached) to avoid ZMM-to-stack spill.
            unsigned long long first_diff = _tzcnt_u64(mask_not_equal);
            return sz_order_scalars_(a[first_diff], b[first_diff]);
        }
        a += 64, b += 64, a_length -= 64, b_length -= 64;
    }

    // In most common scenarios at least one of the strings is under 64 bytes.
    if (a_length | b_length) {
        a_mask = sz_u64_clamp_mask_until_(a_length);
        b_mask = sz_u64_clamp_mask_until_(b_length);
        a_vec.zmm = _mm512_maskz_loadu_epi8(a_mask, a);
        b_vec.zmm = _mm512_maskz_loadu_epi8(b_mask, b);
        // Restrict the comparison to bytes valid in both strings. The masked loads zero out lanes
        // past each string's end, so an unmasked compare would see spurious mismatches against
        // those zeros in the longer string's tail and read out of bounds for the shorter one
        // (e.g. `sz_order("\0baa", 4, "", 0)` would dereference `b[1]`).
        __mmask64 const common_mask = a_mask & b_mask;
        mask_not_equal = _mm512_mask_cmpneq_epi8_mask(common_mask, a_vec.zmm, b_vec.zmm);
        if (mask_not_equal != 0) {
            // Reload from original memory (L1 cached) to avoid ZMM-to-stack spill.
            unsigned long long first_diff = _tzcnt_u64(mask_not_equal);
            return sz_order_scalars_(a[first_diff], b[first_diff]);
        }
        // From logic perspective, the hardest cases are "abc\0" and "abc".
        // The result must be `sz_greater_k`, as the latter is shorter.
        else { return sz_order_scalars_(a_length, b_length); }
    }

    return sz_equal_k;
}

SZ_PUBLIC sz_bool_t sz_equal_skylake(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    __mmask64 mask;
    sz_u512_vec_t a_vec, b_vec;

    while (length >= 64) {
        a_vec.zmm = _mm512_loadu_si512(a);
        b_vec.zmm = _mm512_loadu_si512(b);
        mask = _mm512_cmpneq_epi8_mask(a_vec.zmm, b_vec.zmm);
        if (mask != 0) return sz_false_k;
        a += 64, b += 64, length -= 64;
    }

    if (length) {
        mask = sz_u64_mask_until_(length);
        a_vec.zmm = _mm512_maskz_loadu_epi8(mask, a);
        b_vec.zmm = _mm512_maskz_loadu_epi8(mask, b);
        // Reuse the same `mask` variable to find the bit that doesn't match
        mask = _mm512_mask_cmpneq_epi8_mask(mask, a_vec.zmm, b_vec.zmm);
        return (sz_bool_t)(mask == 0);
    }

    return sz_true_k;
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

#endif // STRINGZILLA_COMPARE_SKYLAKE_H_
