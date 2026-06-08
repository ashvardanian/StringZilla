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

#pragma region Hash with relaxed-SIMD AES

/*  The vpaes tower-field AES emulation is dominated by byte-table lookups, and EVERY index it feeds to
 *  `wasm_i8x16_swizzle` is a nibble or otherwise in `[0, 15]` (the change-of-basis, affine, GF(2^4)
 *  log/antilog, ShiftRows and column-rotate tables). The strict swizzle is therefore already in range,
 *  but engines must still emit the out-of-range clamp because they can't prove it; `relaxed_swizzle`
 *  drops that clamp. Indices being in range, the result is bit-identical to the strict path (and hence
 *  to `sz_emulate_aesenc_si128_serial_`), so these relaxed kernels match the serial reference exactly. */

/** @brief `relaxed_swizzle` counterpart of `sz_aes_linear_v128_`. */
SZ_INTERNAL v128_t sz_aes_linear_v128relaxed_(v128_t lo_table, v128_t hi_table, v128_t x) {
    v128_t lo_nibbles = wasm_v128_and(x, wasm_i8x16_splat((sz_i8_t)0x0F));
    v128_t hi_nibbles = wasm_u8x16_shr(x, 4);
    return wasm_v128_xor(wasm_i8x16_relaxed_swizzle(lo_table, lo_nibbles),
                         wasm_i8x16_relaxed_swizzle(hi_table, hi_nibbles));
}

/** @brief `relaxed_swizzle` counterpart of `sz_aes_gf4_mul_v128_`. */
SZ_INTERNAL v128_t sz_aes_gf4_mul_v128relaxed_(v128_t a, v128_t b) {
    static sz_align_(16) sz_u8_t const log_table[16] = {0x00, 0x00, 0x01, 0x04, 0x02, 0x08, 0x05, 0x0a,
                                                        0x03, 0x0e, 0x09, 0x07, 0x06, 0x0d, 0x0b, 0x0c};
    static sz_align_(16) sz_u8_t const exp_lo_table[16] = {0x01, 0x02, 0x04, 0x08, 0x03, 0x06, 0x0c, 0x0b,
                                                           0x05, 0x0a, 0x07, 0x0e, 0x0f, 0x0d, 0x09, 0x01};
    static sz_align_(16) sz_u8_t const exp_hi_table[16] = {0x02, 0x04, 0x08, 0x03, 0x06, 0x0c, 0x0b, 0x05,
                                                           0x0a, 0x07, 0x0e, 0x0f, 0x0d, 0x09, 0x00, 0x00};
    v128_t log_vec = wasm_v128_load(log_table);
    v128_t exp_lo_vec = wasm_v128_load(exp_lo_table);
    v128_t exp_hi_vec = wasm_v128_load(exp_hi_table);
    v128_t zero = wasm_i8x16_splat(0);

    // `a` and `b` are GF(2^4) values in [0, 15], so the two `log` lookups are in range -> relaxed.
    v128_t sum = wasm_i8x16_add(wasm_i8x16_relaxed_swizzle(log_vec, a), wasm_i8x16_relaxed_swizzle(log_vec, b));
    // `sum` ranges over 0..28 and the antilog split DELIBERATELY relies on strict out-of-range zeroing
    // (low table for 0..15, high table fed `sum - 16` for 16..28), so these MUST stay strict swizzles.
    v128_t product = wasm_v128_xor(wasm_i8x16_swizzle(exp_lo_vec, sum),
                                   wasm_i8x16_swizzle(exp_hi_vec, wasm_i8x16_sub(sum, wasm_i8x16_splat(16))));
    v128_t any_zero = wasm_v128_or(wasm_i8x16_eq(a, zero), wasm_i8x16_eq(b, zero));
    return wasm_v128_andnot(product, any_zero);
}

/** @brief `relaxed_swizzle` counterpart of `sz_emulate_aesenc_v128_` (bit-exact with the serial round). */
SZ_INTERNAL sz_u128_vec_t sz_emulate_aesenc_v128relaxed_(sz_u128_vec_t state_vec, sz_u128_vec_t round_key_vec) {
    static sz_align_(16) sz_u8_t const fwd_lo[16] = {0x00, 0x01, 0x20, 0x21, 0x46, 0x47, 0x66, 0x67,
                                                     0x4c, 0x4d, 0x6c, 0x6d, 0x0a, 0x0b, 0x2a, 0x2b};
    static sz_align_(16) sz_u8_t const fwd_hi[16] = {0x00, 0x3c, 0xd5, 0xe9, 0x34, 0x08, 0xe1, 0xdd,
                                                     0xe5, 0xd9, 0x30, 0x0c, 0xd1, 0xed, 0x04, 0x38};
    static sz_align_(16) sz_u8_t const inv_lo[16] = {0x00, 0x01, 0x5c, 0x5d, 0xe0, 0xe1, 0xbc, 0xbd,
                                                     0x50, 0x51, 0x0c, 0x0d, 0xb0, 0xb1, 0xec, 0xed};
    static sz_align_(16) sz_u8_t const inv_hi[16] = {0x00, 0xa2, 0x02, 0xa0, 0xb8, 0x1a, 0xba, 0x18,
                                                     0xdb, 0x79, 0xd9, 0x7b, 0x63, 0xc1, 0x61, 0xc3};
    static sz_align_(16) sz_u8_t const aff_lo[16] = {0x63, 0x7c, 0x5d, 0x42, 0x1f, 0x00, 0x21, 0x3e,
                                                     0x9b, 0x84, 0xa5, 0xba, 0xe7, 0xf8, 0xd9, 0xc6};
    static sz_align_(16) sz_u8_t const aff_hi[16] = {0x00, 0xf1, 0xe3, 0x12, 0xc7, 0x36, 0x24, 0xd5,
                                                     0x8f, 0x7e, 0x6c, 0x9d, 0x48, 0xb9, 0xab, 0x5a};
    static sz_align_(16) sz_u8_t const gf4_inv[16] = {0x00, 0x01, 0x09, 0x0e, 0x0d, 0x0b, 0x07, 0x06,
                                                      0x0f, 0x02, 0x0c, 0x05, 0x0a, 0x04, 0x03, 0x08};
    static sz_align_(16) sz_u8_t const gf4_sqr[16] = {0x00, 0x01, 0x04, 0x05, 0x03, 0x02, 0x07, 0x06,
                                                      0x0c, 0x0d, 0x08, 0x09, 0x0f, 0x0e, 0x0b, 0x0a};
    static sz_align_(16) sz_u8_t const gf4_sqr_n[16] = {0x00, 0x08, 0x06, 0x0e, 0x0b, 0x03, 0x0d, 0x05,
                                                        0x0a, 0x02, 0x0c, 0x04, 0x01, 0x09, 0x07, 0x0f};
    static sz_align_(16) sz_u8_t const shift_rows[16] = {0x00, 0x05, 0x0a, 0x0f, 0x04, 0x09, 0x0e, 0x03,
                                                         0x08, 0x0d, 0x02, 0x07, 0x0c, 0x01, 0x06, 0x0b};
    static sz_align_(16) sz_u8_t const next_col[16] = {0x01, 0x02, 0x03, 0x00, 0x05, 0x06, 0x07, 0x04,
                                                       0x09, 0x0a, 0x0b, 0x08, 0x0d, 0x0e, 0x0f, 0x0c};

    v128_t state = state_vec.v128;
    v128_t low_nibble_mask = wasm_i8x16_splat((sz_i8_t)0x0F);

    v128_t mapped = sz_aes_linear_v128relaxed_(wasm_v128_load(fwd_lo), wasm_v128_load(fwd_hi), state);
    v128_t hi = wasm_u8x16_shr(mapped, 4);
    v128_t lo = wasm_v128_and(mapped, low_nibble_mask);

    v128_t hi_sqr_n = wasm_i8x16_relaxed_swizzle(wasm_v128_load(gf4_sqr_n), hi);
    v128_t hi_lo = sz_aes_gf4_mul_v128relaxed_(hi, lo);
    v128_t lo_sqr = wasm_i8x16_relaxed_swizzle(wasm_v128_load(gf4_sqr), lo);
    v128_t d = wasm_v128_xor(wasm_v128_xor(hi_sqr_n, hi_lo), lo_sqr);
    v128_t d_inv = wasm_i8x16_relaxed_swizzle(wasm_v128_load(gf4_inv), d);

    v128_t b_hi = sz_aes_gf4_mul_v128relaxed_(hi, d_inv);
    v128_t b_lo = sz_aes_gf4_mul_v128relaxed_(wasm_v128_xor(hi, lo), d_inv);
    v128_t inverted = wasm_v128_or(wasm_i8x16_shl(b_hi, 4), b_lo);

    v128_t back = sz_aes_linear_v128relaxed_(wasm_v128_load(inv_lo), wasm_v128_load(inv_hi), inverted);
    v128_t subbed = sz_aes_linear_v128relaxed_(wasm_v128_load(aff_lo), wasm_v128_load(aff_hi), back);

    v128_t shifted = wasm_i8x16_relaxed_swizzle(subbed, wasm_v128_load(shift_rows));

    v128_t col_table = wasm_v128_load(next_col);
    v128_t col1 = wasm_i8x16_relaxed_swizzle(shifted, col_table);
    v128_t col2 = wasm_i8x16_relaxed_swizzle(col1, col_table);
    v128_t col3 = wasm_i8x16_relaxed_swizzle(col2, col_table);
    v128_t row_xor = wasm_v128_xor(wasm_v128_xor(shifted, col1), wasm_v128_xor(col2, col3));

    v128_t diff = wasm_v128_xor(shifted, col1);
    v128_t need_reduce = wasm_i8x16_shr(diff, 7);
    v128_t reduce = wasm_v128_and(need_reduce, wasm_i8x16_splat((sz_i8_t)0x1B));
    v128_t xtime = wasm_v128_xor(wasm_i8x16_shl(diff, 1), reduce);

    v128_t mixed = wasm_v128_xor(wasm_v128_xor(shifted, row_xor), xtime);

    sz_u128_vec_t result;
    result.v128 = wasm_v128_xor(mixed, round_key_vec.v128);
    return result;
}

/** @brief `relaxed_swizzle` counterpart of `sz_emulate_shuffle_epi8_v128_` (order indices are in [0,15]). */
SZ_INTERNAL sz_u128_vec_t sz_emulate_shuffle_epi8_v128relaxed_(sz_u128_vec_t state_vec, v128_t order) {
    sz_u128_vec_t result;
    result.v128 = wasm_i8x16_relaxed_swizzle(state_vec.v128, order);
    return result;
}

SZ_INTERNAL void sz_hash_minimal_update_v128relaxed_(sz_hash_minimal_t_ *state, sz_u128_vec_t block) {
    v128_t shuffle = wasm_v128_load(sz_hash_u8x16x4_shuffle_());
    state->aes = sz_emulate_aesenc_v128relaxed_(state->aes, block);
    state->sum = sz_emulate_shuffle_epi8_v128relaxed_(state->sum, shuffle);
    state->sum.v128 = wasm_i64x2_add(state->sum.v128, block.v128);
}

SZ_INTERNAL sz_u64_t sz_hash_minimal_finalize_v128relaxed_(sz_hash_minimal_t_ const *state, sz_size_t length) {
    sz_u128_vec_t key_with_length = state->key;
    key_with_length.u64s[0] += length;
    sz_u128_vec_t mixed = sz_emulate_aesenc_v128relaxed_(state->sum, state->aes);
    sz_u128_vec_t mixed_in_register =
        sz_emulate_aesenc_v128relaxed_(sz_emulate_aesenc_v128relaxed_(mixed, key_with_length), mixed);
    return mixed_in_register.u64s[0];
}

SZ_INTERNAL void sz_hash_state_update_v128relaxed_(sz_hash_state_t *state) {
    v128_t shuffle = wasm_v128_load(sz_hash_u8x16x4_shuffle_());
    sz_u128_vec_t *aes_vecs = (sz_u128_vec_t *)state->aes;
    sz_u128_vec_t *sum_vecs = (sz_u128_vec_t *)state->sum;
    sz_u128_vec_t *ins_vecs = (sz_u128_vec_t *)state->ins;
    for (sz_size_t i = 0; i < 4; ++i) {
        aes_vecs[i] = sz_emulate_aesenc_v128relaxed_(aes_vecs[i], ins_vecs[i]);
        sum_vecs[i] = sz_emulate_shuffle_epi8_v128relaxed_(sum_vecs[i], shuffle);
        sum_vecs[i].v128 = wasm_i64x2_add(sum_vecs[i].v128, ins_vecs[i].v128);
    }
}

SZ_INTERNAL sz_u64_t sz_hash_state_finalize_v128relaxed_(sz_hash_state_t const *state) {
    sz_u64_t const *key_u64s = (sz_u64_t const *)state->key;
    sz_u128_vec_t key_with_length;
    key_with_length.u64s[0] = key_u64s[0] + state->ins_length;
    key_with_length.u64s[1] = key_u64s[1];

    sz_u128_vec_t const *aes_vecs = (sz_u128_vec_t const *)state->aes;
    sz_u128_vec_t const *sum_vecs = (sz_u128_vec_t const *)state->sum;

    sz_u128_vec_t mixed0 = sz_emulate_aesenc_v128relaxed_(sum_vecs[0], aes_vecs[0]);
    sz_u128_vec_t mixed1 = sz_emulate_aesenc_v128relaxed_(sum_vecs[1], aes_vecs[1]);
    sz_u128_vec_t mixed2 = sz_emulate_aesenc_v128relaxed_(sum_vecs[2], aes_vecs[2]);
    sz_u128_vec_t mixed3 = sz_emulate_aesenc_v128relaxed_(sum_vecs[3], aes_vecs[3]);

    sz_u128_vec_t mixed01 = sz_emulate_aesenc_v128relaxed_(mixed0, mixed1);
    sz_u128_vec_t mixed23 = sz_emulate_aesenc_v128relaxed_(mixed2, mixed3);
    sz_u128_vec_t mixed = sz_emulate_aesenc_v128relaxed_(mixed01, mixed23);

    sz_u128_vec_t mixed_in_register =
        sz_emulate_aesenc_v128relaxed_(sz_emulate_aesenc_v128relaxed_(mixed, key_with_length), mixed);
    return mixed_in_register.u64s[0];
}

SZ_PUBLIC SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_v128relaxed(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {
    if (length <= 16) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        sz_u128_vec_t data_vec;
        data_vec.v128 = sz_load_partial_v128_(start, length);
        sz_hash_minimal_update_v128relaxed_(&state, data_vec);
        return sz_hash_minimal_finalize_v128relaxed_(&state, length);
    }
    else if (length <= 32) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.v128 = wasm_v128_load(start);
        data1_vec.v128 = wasm_v128_load(start + length - 16);
        sz_hash_shift_in_register_serial_(&data1_vec, (int)(32 - length));
        sz_hash_minimal_update_v128relaxed_(&state, data0_vec);
        sz_hash_minimal_update_v128relaxed_(&state, data1_vec);
        return sz_hash_minimal_finalize_v128relaxed_(&state, length);
    }
    else if (length <= 48) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.v128 = wasm_v128_load(start);
        data1_vec.v128 = wasm_v128_load(start + 16);
        data2_vec.v128 = wasm_v128_load(start + length - 16);
        sz_hash_shift_in_register_serial_(&data2_vec, (int)(48 - length));
        sz_hash_minimal_update_v128relaxed_(&state, data0_vec);
        sz_hash_minimal_update_v128relaxed_(&state, data1_vec);
        sz_hash_minimal_update_v128relaxed_(&state, data2_vec);
        return sz_hash_minimal_finalize_v128relaxed_(&state, length);
    }
    else if (length <= 64) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.v128 = wasm_v128_load(start);
        data1_vec.v128 = wasm_v128_load(start + 16);
        data2_vec.v128 = wasm_v128_load(start + 32);
        data3_vec.v128 = wasm_v128_load(start + length - 16);
        sz_hash_shift_in_register_serial_(&data3_vec, (int)(64 - length));
        sz_hash_minimal_update_v128relaxed_(&state, data0_vec);
        sz_hash_minimal_update_v128relaxed_(&state, data1_vec);
        sz_hash_minimal_update_v128relaxed_(&state, data2_vec);
        sz_hash_minimal_update_v128relaxed_(&state, data3_vec);
        return sz_hash_minimal_finalize_v128relaxed_(&state, length);
    }
    else {
        sz_align_(64) sz_hash_state_t state;
        sz_hash_state_init_serial(&state, seed);
        for (; state.ins_length + 64 <= length; state.ins_length += 64) {
            for (sz_size_t k = 0; k < 4; ++k)
                wasm_v128_store(state.ins + k * 16, wasm_v128_load(start + state.ins_length + k * 16));
            sz_hash_state_update_v128relaxed_(&state);
        }
        if (state.ins_length < length) {
            v128_t zero_vec = wasm_u64x2_splat(0);
            for (sz_size_t k = 0; k < 4; ++k) wasm_v128_store(state.ins + k * 16, zero_vec);
            for (sz_size_t i = 0; state.ins_length < length; ++i, ++state.ins_length)
                state.ins[i] = start[state.ins_length];
            sz_hash_state_update_v128relaxed_(&state);
            state.ins_length = length;
        }
        return sz_hash_state_finalize_v128relaxed_(&state);
    }
}

SZ_PUBLIC void sz_hash_state_init_v128relaxed(sz_hash_state_t *state, sz_u64_t seed) {
    sz_hash_state_init_serial(state, seed);
}

SZ_PUBLIC void sz_hash_state_update_v128relaxed(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    while (length) {
        sz_size_t progress_in_block = state->ins_length % 64;
        sz_size_t to_copy = sz_min_of_two(length, 64 - progress_in_block);
        int const will_fill_block = progress_in_block + to_copy == 64;
        state->ins_length += to_copy;
        length -= to_copy;
        while (to_copy--) state->ins[progress_in_block++] = *text++;
        if (will_fill_block) {
            sz_hash_state_update_v128relaxed_(state);
            for (sz_size_t i = 0; i < 64; ++i) state->ins[i] = 0;
        }
    }
}

SZ_PUBLIC sz_u64_t sz_hash_state_digest_v128relaxed(sz_hash_state_t const *state) {
    sz_size_t length = state->ins_length;
    if (length >= 64) return sz_hash_state_finalize_v128relaxed_(state);

    sz_u64_t const *key_u64s = (sz_u64_t const *)state->key;
    sz_hash_minimal_t_ minimal_state;
    minimal_state.key.u64s[0] = key_u64s[0];
    minimal_state.key.u64s[1] = key_u64s[1];
    minimal_state.aes = *(sz_u128_vec_t const *)state->aes;
    minimal_state.sum = *(sz_u128_vec_t const *)state->sum;

    sz_u128_vec_t const *ins_vecs = (sz_u128_vec_t const *)state->ins;
    if (length <= 16) {
        sz_hash_minimal_update_v128relaxed_(&minimal_state, ins_vecs[0]);
        return sz_hash_minimal_finalize_v128relaxed_(&minimal_state, length);
    }
    else if (length <= 32) {
        sz_hash_minimal_update_v128relaxed_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_v128relaxed_(&minimal_state, ins_vecs[1]);
        return sz_hash_minimal_finalize_v128relaxed_(&minimal_state, length);
    }
    else if (length <= 48) {
        sz_hash_minimal_update_v128relaxed_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_v128relaxed_(&minimal_state, ins_vecs[1]);
        sz_hash_minimal_update_v128relaxed_(&minimal_state, ins_vecs[2]);
        return sz_hash_minimal_finalize_v128relaxed_(&minimal_state, length);
    }
    else {
        sz_hash_minimal_update_v128relaxed_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_v128relaxed_(&minimal_state, ins_vecs[1]);
        sz_hash_minimal_update_v128relaxed_(&minimal_state, ins_vecs[2]);
        sz_hash_minimal_update_v128relaxed_(&minimal_state, ins_vecs[3]);
        return sz_hash_minimal_finalize_v128relaxed_(&minimal_state, length);
    }
}

SZ_PUBLIC void sz_fill_random_v128relaxed(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_u64_t const *pi_ptr = sz_hash_pi_constants_();
    sz_u128_vec_t input_vec, pi_vec, key_vec, generated_vec;
    for (sz_size_t lane_index = 0; length; ++lane_index) {
        input_vec.v128 = wasm_u64x2_splat(nonce + lane_index);
        pi_vec = ((sz_u128_vec_t const *)pi_ptr)[lane_index % 4];
        key_vec.v128 = wasm_v128_xor(pi_vec.v128, wasm_u64x2_splat(nonce));
        generated_vec = sz_emulate_aesenc_v128relaxed_(input_vec, key_vec);
        if (length >= 16) { wasm_v128_store(text, generated_vec.v128), text += 16, length -= 16; }
        else { sz_store_partial_v128_(text, generated_vec.v128, length), length = 0; }
    }
}

#pragma endregion // Hash with relaxed-SIMD AES

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128RELAXED

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_V128RELAXED_H_
