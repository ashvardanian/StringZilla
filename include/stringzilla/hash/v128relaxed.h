/**
 *  @brief WebAssembly relaxed-SIMD backend for hash (level above SIMD128).
 *  @file include/stringzilla/hash/v128relaxed.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_V128RELAXED_H_
#define STRINGZILLA_HASH_V128RELAXED_H_

#include "stringzilla/types.h"
#include "stringzilla/hash/serial.h"
#include "stringzilla/hash/v128.h" // baseline SIMD128 fallbacks

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128RELAXED
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("relaxed-simd")
#endif

/*  `sz_bytesum_v128relaxed` uses the relaxed integer dot-product
 *  `wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a, b, c)`: it forms pairwise products of `a` (signed i8)
 *  and `b` (i7, range 0..127), accumulating them in-lane into the i32x4 `c`. With `b` = all-ones the
 *  loop folds 16 raw bytes into a 4-lane i32 accumulator using a SINGLE relaxed-dot per block — no
 *  per-block widening/extadd chain like the baseline `sz_bytesum_v128`.
 *
 *  Correctness vs serial: the dot treats each byte as a SIGNED i8, so a byte `>= 128` contributes
 *  `byte - 256` instead of `byte`. We therefore count the bytes with bit 7 set (a u8 per-lane
 *  counter, flushed before it could overflow at 255 blocks) and add `256 *` that count back at the
 *  end. The i32 lane sum stays in range: with at most 255 blocks per flush each lane folds 255*2
 *  signed bytes in `[-128, 127]`, i.e. `|lane| <= 510 * 128 < 2^31`. The result is byte-for-byte
 *  identical to `sz_bytesum_serial`.
 *
 *  Runtime caveat: some engines (node/wasmtime on x86) currently lower the i8 relaxed-dot through a
 *  multi-instruction sequence and may be SLOWER than the baseline `extadd` path; native relaxed-simd
 *  engines lower it to a single MAC and win. The level is exposed regardless; correctness is exact. */
SZ_PUBLIC sz_u64_t sz_bytesum_v128relaxed(sz_cptr_t text, sz_size_t length) {
    v128_t const ones_vec = wasm_i8x16_splat(1);
    v128_t const bit7_vec = wasm_i8x16_splat((sz_i8_t)0x80);

    sz_u128_vec_t dot32_vec, high8_vec, high64_vec;
    dot32_vec.v128 = wasm_u64x2_splat(0);  // 4x i32 signed running sum of byte values
    high64_vec.v128 = wasm_u64x2_splat(0); // 2x u64 count of bytes with bit 7 set

    sz_i64_t signed_sum = 0;

    while (length >= 16) {
        // A u8 lane can hold up to 255 high-bit hits before overflow, so flush every 255 blocks.
        sz_size_t blocks = length / 16;
        if (blocks > 255) blocks = 255;
        length -= blocks * 16;
        high8_vec.v128 = wasm_u64x2_splat(0); // 16x u8 per-lane high-bit counters
        for (sz_size_t i = 0; i < blocks; ++i, text += 16) {
            v128_t block_vec = wasm_v128_load(text);
            // Signed in-lane MAC: dot32 += sum of (i8)byte over each pair, folded into 4 i32 lanes.
            dot32_vec.v128 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(block_vec, ones_vec, dot32_vec.v128);
            // Count bytes with bit 7 set: `eq(byte & 0x80, 0x80)` yields 0xFF (-1) per hit; subtract.
            v128_t high_vec = wasm_i8x16_eq(wasm_v128_and(block_vec, bit7_vec), bit7_vec);
            high8_vec.v128 = wasm_i8x16_sub(high8_vec.v128, high_vec);
        }
        // Fold the 16x u8 high-bit counters into the 2x u64 accumulator via pairwise widenings.
        v128_t high16_vec = wasm_u16x8_extadd_pairwise_u8x16(high8_vec.v128);
        v128_t high32_vec = wasm_u32x4_extadd_pairwise_u16x8(high16_vec);
        high64_vec.v128 = wasm_i64x2_add( //
            high64_vec.v128,
            wasm_i64x2_add(wasm_u64x2_extend_low_u32x4(high32_vec), wasm_u64x2_extend_high_u32x4(high32_vec)));
    }

    // Reduce the 4x i32 signed lane sums.
    signed_sum += (sz_i64_t)dot32_vec.i32s[0] + dot32_vec.i32s[1] + dot32_vec.i32s[2] + dot32_vec.i32s[3];
    sz_u64_t high_count = high64_vec.u64s[0] + high64_vec.u64s[1];
    // Each byte >= 128 was undercounted by 256; add it back to recover the unsigned sum.
    sz_u64_t sum = (sz_u64_t)(signed_sum + (sz_i64_t)(high_count * 256));
    while (length--) sum += *(sz_u8_t const *)text++;
    return sum;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_V128RELAXED

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_V128RELAXED_H_
