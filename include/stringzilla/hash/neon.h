/**
 *  @brief NEON backend for string hashing and checksums.
 *  @file include/stringzilla/hash/neon.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_NEON_H_
#define STRINGZILLA_HASH_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/hash/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

SZ_API_COMPTIME sz_u64_t sz_bytesum_neon(sz_cptr_t text, sz_size_t length) {
    uint64x2_t sum_u64x2 = vdupq_n_u64(0);

    // Process 16 bytes (128 bits) at a time
    for (; length >= 16; text += 16, length -= 16) {
        uint8x16_t block_u8x16 = vld1q_u8((sz_u8_t const *)text);        // Load 16 bytes
        uint16x8_t pairwise_sum_u16x8 = vpaddlq_u8(block_u8x16);         // Pairwise add lower and upper 8 bits
        uint32x4_t pairwise_sum_u32x4 = vpaddlq_u16(pairwise_sum_u16x8); // Pairwise add 16-bit results
        uint64x2_t pairwise_sum_u64x2 = vpaddlq_u32(pairwise_sum_u32x4); // Pairwise add 32-bit results
        sum_u64x2 = vaddq_u64(sum_u64x2, pairwise_sum_u64x2);            // Accumulate the sum
    }

    // Final reduction of `sum_u64x2` to a single scalar
    sz_u64_t sum = vaddvq_u64(sum_u64x2);
    while (length--) sum += *(sz_u8_t const *)text++; // Same as the scalar version
    return sum;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_NEON_H_
