/**
 *  @brief WebAssembly SIMD128 backend for hash.
 *  @file include/stringzilla/hash/v128.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_V128_H_
#define STRINGZILLA_HASH_V128_H_

#include "stringzilla/types.h"
#include "stringzilla/hash/serial.h"
#include "stringzilla/compare.h"     // `sz_equal`
#include "stringzilla/memory/v128.h" // `sz_load_partial_v128_`, `sz_store_partial_v128_`

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

SZ_PUBLIC sz_u64_t sz_bytesum_v128(sz_cptr_t text, sz_size_t length) {
    // WASM SIMD128 has no `psadbw`. The hot path keeps an 8-lane u16 accumulator entirely in a
    // vector register and folds raw bytes into it with a single `extadd_pairwise` per 16 bytes:
    // there are NO widening/extend/64-bit-add reductions inside the loop. Each u16 lane sums two
    // byte columns, so after `k` iterations a lane holds at most `2 * k * 255`; it stays below
    // 65535 for `k <= 128`. We therefore flush the u16 accumulator into a wide u64 accumulator
    // every 128 iterations (2 KiB), and do ONE horizontal reduction at the very end.
    sz_u128_vec_t sum16_vec, sum64_vec;
    sum16_vec.v128 = wasm_u64x2_splat(0); // 8x u16 partials
    sum64_vec.v128 = wasm_u64x2_splat(0); // 2x u64 partials

    while (length >= 16) {
        // Process up to 128 blocks before the u16 lanes could overflow.
        sz_size_t blocks = length / 16;
        if (blocks > 128) blocks = 128;
        length -= blocks * 16;
        for (sz_size_t block_index = 0; block_index < blocks; ++block_index, text += 16) {
            v128_t vec = wasm_v128_load(text);
            // 16x u8 -> 8x u16 pairwise sums, accumulated in-lane (associative, no reduction).
            sum16_vec.v128 = wasm_i16x8_add(sum16_vec.v128, wasm_u16x8_extadd_pairwise_u8x16(vec));
        }
        // Flush: widen the 8x u16 partials to 4x u32, then to 2x u64, and fold into `sum64`.
        v128_t pair32 = wasm_u32x4_extadd_pairwise_u16x8(sum16_vec.v128);
        sum64_vec.v128 = wasm_i64x2_add( //
            sum64_vec.v128, wasm_i64x2_add(wasm_u64x2_extend_low_u32x4(pair32), wasm_u64x2_extend_high_u32x4(pair32)));
        sum16_vec.v128 = wasm_u64x2_splat(0);
    }

    sz_u64_t sum = sum64_vec.u64s[0] + sum64_vec.u64s[1];
    while (length--) sum += *(sz_u8_t const *)text++;
    return sum;
}

/** @brief Evaluate a GF(2)-linear byte map `x -> low_table[x&0xF] ^ high_table[x>>4]` across all 16 lanes. */
SZ_INTERNAL v128_t sz_aes_linear_v128_(v128_t low_table, v128_t high_table, v128_t x) {
    v128_t low_nibbles = wasm_v128_and(x, wasm_i8x16_splat((sz_i8_t)0x0F));
    v128_t high_nibbles = wasm_u8x16_shr(x, 4);
    return wasm_v128_xor(wasm_i8x16_swizzle(low_table, low_nibbles), wasm_i8x16_swizzle(high_table, high_nibbles));
}

/** @brief GF(2^4) (modulus `x^4+x+1`) lane-wise multiply via log/antilog swizzles with zero masking. */
SZ_INTERNAL v128_t sz_aes_gf4_mul_v128_(v128_t a, v128_t b) {
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

    // `sum = log[a] + log[b]` ranges over 0..28; the low antilog swizzle covers 0..15 (others -> 0),
    // and the high antilog swizzle is fed `sum - 16` so it covers 16..28 (others wrap >= 16 -> 0).
    v128_t sum = wasm_i8x16_add(wasm_i8x16_swizzle(log_vec, a), wasm_i8x16_swizzle(log_vec, b));
    v128_t product = wasm_v128_xor(wasm_i8x16_swizzle(exp_lo_vec, sum),
                                   wasm_i8x16_swizzle(exp_hi_vec, wasm_i8x16_sub(sum, wasm_i8x16_splat(16))));
    // Force a zero result wherever either operand was zero (`log[0]` is otherwise indistinguishable).
    v128_t any_zero = wasm_v128_or(wasm_i8x16_eq(a, zero), wasm_i8x16_eq(b, zero));
    return wasm_v128_andnot(product, any_zero);
}

/**
 *  @brief Bit-exact `_mm_aesenc_si128` for one round, identical to `sz_emulate_aesenc_si128_serial_`.
 *  @return `MixColumns(SubBytes(ShiftRows(state))) ^ round_key`, computed with the vpaes tower field.
 */
SZ_INTERNAL sz_u128_vec_t sz_emulate_aesenc_v128_(sz_u128_vec_t state_vec, sz_u128_vec_t round_key_vec) {
    // GF(2^8) <-> GF(2^4)^2 change-of-basis (forward `M` and inverse `M^-1`), as linear nibble tables.
    static sz_align_(16) sz_u8_t const fwd_lo[16] = {0x00, 0x01, 0x20, 0x21, 0x46, 0x47, 0x66, 0x67,
                                                     0x4c, 0x4d, 0x6c, 0x6d, 0x0a, 0x0b, 0x2a, 0x2b};
    static sz_align_(16) sz_u8_t const fwd_hi[16] = {0x00, 0x3c, 0xd5, 0xe9, 0x34, 0x08, 0xe1, 0xdd,
                                                     0xe5, 0xd9, 0x30, 0x0c, 0xd1, 0xed, 0x04, 0x38};
    static sz_align_(16) sz_u8_t const inv_lo[16] = {0x00, 0x01, 0x5c, 0x5d, 0xe0, 0xe1, 0xbc, 0xbd,
                                                     0x50, 0x51, 0x0c, 0x0d, 0xb0, 0xb1, 0xec, 0xed};
    static sz_align_(16) sz_u8_t const inv_hi[16] = {0x00, 0xa2, 0x02, 0xa0, 0xb8, 0x1a, 0xba, 0x18,
                                                     0xdb, 0x79, 0xd9, 0x7b, 0x63, 0xc1, 0x61, 0xc3};
    // AES affine map (constant `0x63` folded into the low table) as linear nibble tables.
    static sz_align_(16) sz_u8_t const aff_lo[16] = {0x63, 0x7c, 0x5d, 0x42, 0x1f, 0x00, 0x21, 0x3e,
                                                     0x9b, 0x84, 0xa5, 0xba, 0xe7, 0xf8, 0xd9, 0xc6};
    static sz_align_(16) sz_u8_t const aff_hi[16] = {0x00, 0xf1, 0xe3, 0x12, 0xc7, 0x36, 0x24, 0xd5,
                                                     0x8f, 0x7e, 0x6c, 0x9d, 0x48, 0xb9, 0xab, 0x5a};
    // GF(2^4) inverse, square, and `square * N` (N = 8) nibble tables.
    static sz_align_(16) sz_u8_t const gf4_inv[16] = {0x00, 0x01, 0x09, 0x0e, 0x0d, 0x0b, 0x07, 0x06,
                                                      0x0f, 0x02, 0x0c, 0x05, 0x0a, 0x04, 0x03, 0x08};
    static sz_align_(16) sz_u8_t const gf4_sqr[16] = {0x00, 0x01, 0x04, 0x05, 0x03, 0x02, 0x07, 0x06,
                                                      0x0c, 0x0d, 0x08, 0x09, 0x0f, 0x0e, 0x0b, 0x0a};
    static sz_align_(16) sz_u8_t const gf4_sqr_n[16] = {0x00, 0x08, 0x06, 0x0e, 0x0b, 0x03, 0x0d, 0x05,
                                                        0x0a, 0x02, 0x0c, 0x04, 0x01, 0x09, 0x07, 0x0f};
    // ShiftRows (fused with the serial byte de-interleave) and within-row column rotate, as swizzles.
    static sz_align_(16) sz_u8_t const shift_rows[16] = {0x00, 0x05, 0x0a, 0x0f, 0x04, 0x09, 0x0e, 0x03,
                                                         0x08, 0x0d, 0x02, 0x07, 0x0c, 0x01, 0x06, 0x0b};
    static sz_align_(16) sz_u8_t const next_col[16] = {0x01, 0x02, 0x03, 0x00, 0x05, 0x06, 0x07, 0x04,
                                                       0x09, 0x0a, 0x0b, 0x08, 0x0d, 0x0e, 0x0f, 0x0c};

    v128_t state = state_vec.v128;
    v128_t low_nibble_mask = wasm_i8x16_splat((sz_i8_t)0x0F);

    // SubBytes: inverse in GF(2^8) via the tower field, then the AES affine map.
    v128_t mapped = sz_aes_linear_v128_(wasm_v128_load(fwd_lo), wasm_v128_load(fwd_hi), state);
    v128_t high_nibble = wasm_u8x16_shr(mapped, 4);             // high nibble = `a_hi`
    v128_t low_nibble = wasm_v128_and(mapped, low_nibble_mask); // low nibble  = `a_lo`

    // d = a_hi^2 * N  ^  a_hi * a_lo  ^  a_lo^2   (all in GF(2^4))
    v128_t high_sqr_n = wasm_i8x16_swizzle(wasm_v128_load(gf4_sqr_n), high_nibble);
    v128_t high_low_product = sz_aes_gf4_mul_v128_(high_nibble, low_nibble);
    v128_t low_sqr = wasm_i8x16_swizzle(wasm_v128_load(gf4_sqr), low_nibble);
    v128_t d = wasm_v128_xor(wasm_v128_xor(high_sqr_n, high_low_product), low_sqr);
    v128_t d_inv = wasm_i8x16_swizzle(wasm_v128_load(gf4_inv), d);

    v128_t high_inverse = sz_aes_gf4_mul_v128_(high_nibble, d_inv);
    v128_t low_inverse = sz_aes_gf4_mul_v128_(wasm_v128_xor(high_nibble, low_nibble), d_inv);
    v128_t inverted = wasm_v128_or(wasm_i8x16_shl(high_inverse, 4), low_inverse); // recombine nibbles into the byte

    v128_t back = sz_aes_linear_v128_(wasm_v128_load(inv_lo), wasm_v128_load(inv_hi), inverted);
    v128_t subbed = sz_aes_linear_v128_(wasm_v128_load(aff_lo), wasm_v128_load(aff_hi), back);

    // ShiftRows: one swizzle (also de-interleaves into the serial row-major layout).
    v128_t shifted = wasm_i8x16_swizzle(subbed, wasm_v128_load(shift_rows));

    // MixColumns: U (row XOR) plus xtime of adjacent column differences.
    v128_t col_table = wasm_v128_load(next_col);
    v128_t col1 = wasm_i8x16_swizzle(shifted, col_table);
    v128_t col2 = wasm_i8x16_swizzle(col1, col_table);
    v128_t col3 = wasm_i8x16_swizzle(col2, col_table);
    v128_t row_xor = wasm_v128_xor(wasm_v128_xor(shifted, col1), wasm_v128_xor(col2, col3));

    v128_t diff = wasm_v128_xor(shifted, col1);   // s[c] ^ s[c+1]
    v128_t need_reduce = wasm_i8x16_shr(diff, 7); // 0xFF where the top bit is set, else 0x00
    v128_t reduce = wasm_v128_and(need_reduce, wasm_i8x16_splat((sz_i8_t)0x1B));
    v128_t xtime = wasm_v128_xor(wasm_i8x16_shl(diff, 1), reduce);

    v128_t mixed = wasm_v128_xor(wasm_v128_xor(shifted, row_xor), xtime);

    // AddRoundKey.
    sz_u128_vec_t result;
    result.v128 = wasm_v128_xor(mixed, round_key_vec.v128);
    return result;
}

/** @brief Vectorized counterpart of `sz_emulate_shuffle_epi8_serial_` using one `wasm_i8x16_swizzle`. */
SZ_INTERNAL sz_u128_vec_t sz_emulate_shuffle_epi8_v128_(sz_u128_vec_t state_vec, v128_t order) {
    sz_u128_vec_t result;
    result.v128 = wasm_i8x16_swizzle(state_vec.v128, order);
    return result;
}

#pragma region Hash with SIMD128 AES

SZ_INTERNAL void sz_hash_minimal_init_v128_(sz_hash_minimal_t_ *state, sz_u64_t seed) {
    sz_hash_minimal_init_serial_(state, seed);
}

SZ_INTERNAL void sz_hash_minimal_update_v128_(sz_hash_minimal_t_ *state, sz_u128_vec_t block) {
    v128_t shuffle = wasm_v128_load(sz_hash_u8x16x4_shuffle_());
    state->aes = sz_emulate_aesenc_v128_(state->aes, block);
    state->sum = sz_emulate_shuffle_epi8_v128_(state->sum, shuffle);
    state->sum.v128 = wasm_i64x2_add(state->sum.v128, block.v128);
}

SZ_INTERNAL sz_u64_t sz_hash_minimal_finalize_v128_(sz_hash_minimal_t_ const *state, sz_size_t length) {
    sz_u128_vec_t key_with_length = state->key;
    key_with_length.u64s[0] += length;
    sz_u128_vec_t mixed = sz_emulate_aesenc_v128_(state->sum, state->aes);
    sz_u128_vec_t mixed_in_register = sz_emulate_aesenc_v128_(sz_emulate_aesenc_v128_(mixed, key_with_length), mixed);
    return mixed_in_register.u64s[0];
}

SZ_INTERNAL void sz_hash_state_update_v128_(sz_hash_state_t *state) {
    v128_t shuffle = wasm_v128_load(sz_hash_u8x16x4_shuffle_());
    sz_u128_vec_t *aes_vecs = (sz_u128_vec_t *)state->aes;
    sz_u128_vec_t *sum_vecs = (sz_u128_vec_t *)state->sum;
    sz_u128_vec_t *ins_vecs = (sz_u128_vec_t *)state->ins;
    for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) {
        aes_vecs[lane_index] = sz_emulate_aesenc_v128_(aes_vecs[lane_index], ins_vecs[lane_index]);
        sum_vecs[lane_index] = sz_emulate_shuffle_epi8_v128_(sum_vecs[lane_index], shuffle);
        sum_vecs[lane_index].v128 = wasm_i64x2_add(sum_vecs[lane_index].v128, ins_vecs[lane_index].v128);
    }
}

SZ_INTERNAL sz_u64_t sz_hash_state_finalize_v128_(sz_hash_state_t const *state) {
    v128_t shuffle = wasm_v128_load(sz_hash_u8x16x4_shuffle_());
    sz_u64_t const *key_u64s = (sz_u64_t const *)state->key;
    sz_u128_vec_t key_with_length;
    key_with_length.u64s[0] = key_u64s[0] + state->ins_length;
    key_with_length.u64s[1] = key_u64s[1];

    sz_u128_vec_t const *aes_vecs = (sz_u128_vec_t const *)state->aes;
    sz_u128_vec_t const *sum_vecs = (sz_u128_vec_t const *)state->sum;
    sz_u128_vec_t const *ins_vecs = (sz_u128_vec_t const *)state->ins;

    // Fold the deferred final block (still buffered in `ins` - a full 64 bytes or a zero-padded tail) into each
    // lane. Folding the last block here, rather than in `update`, lets both one-shot `sz_hash` and the streaming
    // digest defer it and share this single finalization with no state copy.
    sz_u128_vec_t aes0 = sz_emulate_aesenc_v128_(aes_vecs[0], ins_vecs[0]);
    sz_u128_vec_t aes1 = sz_emulate_aesenc_v128_(aes_vecs[1], ins_vecs[1]);
    sz_u128_vec_t aes2 = sz_emulate_aesenc_v128_(aes_vecs[2], ins_vecs[2]);
    sz_u128_vec_t aes3 = sz_emulate_aesenc_v128_(aes_vecs[3], ins_vecs[3]);
    sz_u128_vec_t sum0 = sz_emulate_shuffle_epi8_v128_(sum_vecs[0], shuffle);
    sz_u128_vec_t sum1 = sz_emulate_shuffle_epi8_v128_(sum_vecs[1], shuffle);
    sz_u128_vec_t sum2 = sz_emulate_shuffle_epi8_v128_(sum_vecs[2], shuffle);
    sz_u128_vec_t sum3 = sz_emulate_shuffle_epi8_v128_(sum_vecs[3], shuffle);
    sum0.v128 = wasm_i64x2_add(sum0.v128, ins_vecs[0].v128);
    sum1.v128 = wasm_i64x2_add(sum1.v128, ins_vecs[1].v128);
    sum2.v128 = wasm_i64x2_add(sum2.v128, ins_vecs[2].v128);
    sum3.v128 = wasm_i64x2_add(sum3.v128, ins_vecs[3].v128);

    sz_u128_vec_t mixed0 = sz_emulate_aesenc_v128_(sum0, aes0);
    sz_u128_vec_t mixed1 = sz_emulate_aesenc_v128_(sum1, aes1);
    sz_u128_vec_t mixed2 = sz_emulate_aesenc_v128_(sum2, aes2);
    sz_u128_vec_t mixed3 = sz_emulate_aesenc_v128_(sum3, aes3);

    sz_u128_vec_t mixed01 = sz_emulate_aesenc_v128_(mixed0, mixed1);
    sz_u128_vec_t mixed23 = sz_emulate_aesenc_v128_(mixed2, mixed3);
    sz_u128_vec_t mixed = sz_emulate_aesenc_v128_(mixed01, mixed23);

    sz_u128_vec_t mixed_in_register = sz_emulate_aesenc_v128_(sz_emulate_aesenc_v128_(mixed, key_with_length), mixed);
    return mixed_in_register.u64s[0];
}

SZ_PUBLIC SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_v128(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {
    if (length <= 16) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_v128_(&state, seed);
        sz_u128_vec_t data_vec;
        data_vec.v128 = sz_load_partial_v128_(start, length);
        sz_hash_minimal_update_v128_(&state, data_vec);
        return sz_hash_minimal_finalize_v128_(&state, length);
    }
    else if (length <= 32) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_v128_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.v128 = wasm_v128_load(start);
        data1_vec.v128 = wasm_v128_load(start + length - 16);
        sz_hash_shift_in_register_serial_(&data1_vec, (int)(32 - length));
        sz_hash_minimal_update_v128_(&state, data0_vec);
        sz_hash_minimal_update_v128_(&state, data1_vec);
        return sz_hash_minimal_finalize_v128_(&state, length);
    }
    else if (length <= 48) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_v128_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.v128 = wasm_v128_load(start);
        data1_vec.v128 = wasm_v128_load(start + 16);
        data2_vec.v128 = wasm_v128_load(start + length - 16);
        sz_hash_shift_in_register_serial_(&data2_vec, (int)(48 - length));
        sz_hash_minimal_update_v128_(&state, data0_vec);
        sz_hash_minimal_update_v128_(&state, data1_vec);
        sz_hash_minimal_update_v128_(&state, data2_vec);
        return sz_hash_minimal_finalize_v128_(&state, length);
    }
    else if (length <= 64) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_v128_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.v128 = wasm_v128_load(start);
        data1_vec.v128 = wasm_v128_load(start + 16);
        data2_vec.v128 = wasm_v128_load(start + 32);
        data3_vec.v128 = wasm_v128_load(start + length - 16);
        sz_hash_shift_in_register_serial_(&data3_vec, (int)(64 - length));
        sz_hash_minimal_update_v128_(&state, data0_vec);
        sz_hash_minimal_update_v128_(&state, data1_vec);
        sz_hash_minimal_update_v128_(&state, data2_vec);
        sz_hash_minimal_update_v128_(&state, data3_vec);
        return sz_hash_minimal_finalize_v128_(&state, length);
    }
    else {
        sz_align_(64) sz_hash_state_t state;
        sz_hash_state_init_serial(&state, seed);

        // Absorb every full 64-byte block EXCEPT the last; the final block (a full 64 or a partial tail) stays
        // buffered in `ins` for `sz_hash_state_finalize_v128_` to fold - the same deferral the streaming path uses.
        for (; state.ins_length + 64 < length; state.ins_length += 64) {
            for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index)
                wasm_v128_store(state.ins + lane_index * 16,
                                wasm_v128_load(start + state.ins_length + lane_index * 16));
            sz_hash_state_update_v128_(&state);
        }

        // Stage the final [ins_length, length) bytes (1..64) into a zero-padded buffer; finalize folds them.
        v128_t const zero_vec = wasm_u64x2_splat(0);
        for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) wasm_v128_store(state.ins + lane_index * 16, zero_vec);
        sz_size_t const tail_length = length - state.ins_length;
        for (sz_size_t byte_index = 0; byte_index < tail_length; ++byte_index)
            state.ins[byte_index] = start[state.ins_length + byte_index];
        state.ins_length = length;
        return sz_hash_state_finalize_v128_(&state);
    }
}

SZ_PUBLIC void sz_hash_state_init_v128(sz_hash_state_t *state, sz_u64_t seed) {
    sz_hash_state_init_serial(state, seed);
}

SZ_PUBLIC void sz_hash_state_update_v128(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {

    // `ins` is exactly one 64-byte block (four v128 lanes), so buffering is just: track how many bytes it holds,
    // absorb it only once it becomes interior (more bytes arrive - the deferral `digest` needs to choose
    // minimal/full by total length), and append incoming bytes with a contiguous copy. The deferred trailing block
    // reads back as `ins_length % 64 == 0 && ins_length != 0`; treat that as `buffered == 64`. The copy touches
    // only `[buffered, buffered+take)`, and we re-zero `ins` after each absorb, so the high lanes stay zero-padded
    // for `finalize` to fold.
    v128_t const zero_vec = wasm_u64x2_splat(0);
    sz_size_t buffered = state->ins_length % 64;
    if (buffered == 0 && state->ins_length) buffered = 64;
    while (length) {
        if (buffered == 64) { // the deferred block is now interior - absorb it and re-zero the buffer
            sz_hash_state_update_v128_(state);
            for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index)
                wasm_v128_store(state->ins + lane_index * 16, zero_vec);
            buffered = 0;
        }
        sz_size_t const take = sz_min_of_two(length, (sz_size_t)64 - buffered);
        for (sz_size_t byte_index = 0; byte_index < take; ++byte_index) state->ins[buffered + byte_index] = text[byte_index];
        buffered += take, text += take, length -= take, state->ins_length += take;
    }
}

SZ_PUBLIC sz_u64_t sz_hash_state_digest_v128(sz_hash_state_t const *state) {
    sz_size_t length = state->ins_length;
    // Inputs longer than one block fold through the full four-lane state. The deferred final block is still
    // buffered in `ins`, and `sz_hash_state_finalize_v128_` folds it - so this is a plain, copy-free finalize
    // that reproduces one-shot `sz_hash` exactly. A length of exactly 64 uses the minimal (<=64) path below.
    if (length > 64) return sz_hash_state_finalize_v128_(state);

    sz_u64_t const *key_u64s = (sz_u64_t const *)state->key;
    sz_hash_minimal_t_ minimal_state;
    minimal_state.key.u64s[0] = key_u64s[0];
    minimal_state.key.u64s[1] = key_u64s[1];
    minimal_state.aes = *(sz_u128_vec_t const *)state->aes;
    minimal_state.sum = *(sz_u128_vec_t const *)state->sum;

    sz_u128_vec_t const *ins_vecs = (sz_u128_vec_t const *)state->ins;
    if (length <= 16) {
        sz_hash_minimal_update_v128_(&minimal_state, ins_vecs[0]);
        return sz_hash_minimal_finalize_v128_(&minimal_state, length);
    }
    else if (length <= 32) {
        sz_hash_minimal_update_v128_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_v128_(&minimal_state, ins_vecs[1]);
        return sz_hash_minimal_finalize_v128_(&minimal_state, length);
    }
    else if (length <= 48) {
        sz_hash_minimal_update_v128_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_v128_(&minimal_state, ins_vecs[1]);
        sz_hash_minimal_update_v128_(&minimal_state, ins_vecs[2]);
        return sz_hash_minimal_finalize_v128_(&minimal_state, length);
    }
    else {
        sz_hash_minimal_update_v128_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_v128_(&minimal_state, ins_vecs[1]);
        sz_hash_minimal_update_v128_(&minimal_state, ins_vecs[2]);
        sz_hash_minimal_update_v128_(&minimal_state, ins_vecs[3]);
        return sz_hash_minimal_finalize_v128_(&minimal_state, length);
    }
}

SZ_PUBLIC void sz_fill_random_v128(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_u64_t const *pi_constants = sz_hash_pi_constants_();
    sz_u128_vec_t input_vec, pi_vec, key_vec, generated_vec;
    for (sz_size_t lane_index = 0; length; ++lane_index) {
        input_vec.v128 = wasm_u64x2_splat(nonce + lane_index); // both 64-bit lanes = nonce + block
        pi_vec = ((sz_u128_vec_t const *)pi_constants)[lane_index % 4];
        key_vec.v128 = wasm_v128_xor(pi_vec.v128, wasm_u64x2_splat(nonce)); // key = pi ^ broadcast(nonce)
        generated_vec = sz_emulate_aesenc_v128_(input_vec, key_vec);
        // Emit the whole 16-byte block with one vector store on the hot path; the ragged final block
        // (fewer than 16 bytes left) defers to the serial byte copy.
        if (length >= 16) { wasm_v128_store(text, generated_vec.v128), text += 16, length -= 16; }
        else { sz_store_partial_v128_(text, generated_vec.v128, length), length = 0; }
    }
}

#pragma region Multi-Seed Hashing

/**
 *  @brief Replays prepared text-lanes through the SIMD128 minimal AES state for a single seed.
 *  @return 64-bit hash, bit-identical to `sz_hash_v128(text, length, seed)` (hence to serial).
 *  @sa sz_hash_multiseed_replay_serial_, sz_hash_multiseed_prepare_serial_
 */
SZ_INTERNAL sz_u64_t sz_hash_multiseed_replay_v128_(sz_u512_vec_t const *text_lanes, sz_size_t text_lanes_count,
                                                    sz_size_t length, sz_u64_t seed) {
    sz_align_(16) sz_hash_minimal_t_ state;
    sz_hash_minimal_init_v128_(&state, seed);
    for (sz_size_t lane_index = 0; lane_index < text_lanes_count; ++lane_index)
        sz_hash_minimal_update_v128_(&state, text_lanes->u128s[lane_index]);
    return sz_hash_minimal_finalize_v128_(&state, length);
}

/**
 *  @brief Hashes one `text` under many `seeds`, sharing a single normalization pass for short inputs.
 *
 *  The branchy load + de-interleave of a `<= 64` byte input depends only on `(text, length)`, so it is
 *  done exactly once (`sz_hash_multiseed_prepare_serial_`) and replayed through the cheap SIMD128 AES
 *  rounds per seed — amortizing the input-load/tail cost that dominates short-string hashing. On a
 *  128-bit register each seed's minimal state is a full vector, so seeds are not lane-packed; the win is
 *  the shared normalization plus the faster v128 AES emulation. Bit-identical to `sz_hash_multiseed_serial`.
 */
SZ_PUBLIC void sz_hash_multiseed_v128(sz_cptr_t text, sz_size_t length,             //
                                      sz_u64_t const *seeds, sz_size_t seeds_count, //
                                      sz_u64_t *hashes) {
    if (seeds_count == 0) return;
    if (seeds_count == 1) {
        hashes[0] = sz_hash_v128(text, length, seeds[0]);
        return;
    }
    if (length <= 64) {
        sz_u512_vec_t text_lanes;
        sz_size_t const text_lanes_count = sz_hash_multiseed_prepare_serial_(text, length, &text_lanes);
        for (sz_size_t seed_index = 0; seed_index < seeds_count; ++seed_index)
            hashes[seed_index] = sz_hash_multiseed_replay_v128_(&text_lanes, text_lanes_count, length,
                                                                seeds[seed_index]);
    }
    else {
        for (sz_size_t seed_index = 0; seed_index < seeds_count; ++seed_index)
            hashes[seed_index] = sz_hash_v128(text, length, seeds[seed_index]);
    }
}

#pragma endregion // Multi-Seed Hashing

#pragma endregion // Hash with SIMD128 AES

/** @brief 32-bit lane-wise rotate-right (no native WASM rotate; built from two shifts and an OR). */
SZ_INTERNAL v128_t sz_sha256_rotr_v128_(v128_t x, int count) {
    return wasm_v128_or(wasm_u32x4_shr(x, count), wasm_i32x4_shl(x, 32 - count));
}
/** @brief 4-wide `sigma0` of the message schedule: ROTR(x,7) ^ ROTR(x,18) ^ SHR(x,3). */
SZ_INTERNAL v128_t sz_sha256_sigma0_lower_v128_(v128_t x) {
    return wasm_v128_xor(wasm_v128_xor(sz_sha256_rotr_v128_(x, 7), sz_sha256_rotr_v128_(x, 18)), wasm_u32x4_shr(x, 3));
}
/** @brief 4-wide `sigma1` of the message schedule: ROTR(x,17) ^ ROTR(x,19) ^ SHR(x,10). */
SZ_INTERNAL v128_t sz_sha256_sigma1_lower_v128_(v128_t x) {
    return wasm_v128_xor(wasm_v128_xor(sz_sha256_rotr_v128_(x, 17), sz_sha256_rotr_v128_(x, 19)),
                         wasm_u32x4_shr(x, 10));
}

/**
 *  @brief Process one 64-byte block with a SIMD message schedule, bit-exact with the serial reference.
 *
 *  The 64-round compression is sequential in `a..h`, so it stays scalar; the gain is the message
 *  schedule, computed four words per `i32x4` (the Gueron-Krasnov layout). `W[i]` needs `sigma1(W[i-2])`,
 *  so within a group of four the upper two lanes depend on the lower two just computed — handled with a
 *  two-phase `sigma1` and a final lane blend. The input words are loaded big-endian via one shuffle.
 */
SZ_INTERNAL void sz_sha256_process_block_v128_(sz_u32_t hash[sz_at_least_(8)], sz_u8_t const block[sz_at_least_(64)]) {
    sz_u32_t const *round_constants = sz_sha256_round_constants_();
    sz_align_(16) sz_u32_t w[64];

    // Load the first 16 words big-endian: reverse the 4 bytes of each 32-bit word. This is a
    // compile-time-CONSTANT permutation, so `wasm_i8x16_shuffle` (a fixed lane shuffle) is optimal —
    // there is no data-dependent table here, hence nothing for `relaxed_swizzle` to accelerate.
    for (sz_size_t group_index = 0; group_index < 4; ++group_index) {
        v128_t loaded = wasm_v128_load(block + group_index * 16);
        wasm_v128_store(&w[group_index * 4],
                        wasm_i8x16_shuffle(loaded, loaded, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12));
    }

    // Extend to 64 words, four at a time.
    for (sz_size_t word_index = 16; word_index < 64; word_index += 4) {
        v128_t w_m16 = wasm_v128_load(&w[word_index - 16]);
        v128_t w_m15 = wasm_v128_load(&w[word_index - 15]);
        v128_t w_m7 = wasm_v128_load(&w[word_index - 7]);
        v128_t w_m2 = wasm_v128_load(&w[word_index - 2]); // lanes 2,3 = W[i],W[i+1] are still zero (computed below)
        v128_t partial = wasm_i32x4_add(wasm_i32x4_add(w_m16, sz_sha256_sigma0_lower_v128_(w_m15)), w_m7);
        // Phase 1: sigma1 of lanes 0,1 (W[i-2],W[i-1]) gives correct W[i],W[i+1] in t0's low lanes.
        v128_t t0 = wasm_i32x4_add(partial, sz_sha256_sigma1_lower_v128_(w_m2));
        // Phase 2: feed those W[i],W[i+1] into lanes 2,3 for the remaining sigma1.
        v128_t w_hi = wasm_i32x4_shuffle(t0, t0, 0, 1, 0, 1); // lanes 2,3 = W[i],W[i+1]
        v128_t t1 = wasm_i32x4_add(partial, sz_sha256_sigma1_lower_v128_(w_hi));
        // Blend: low two lanes from phase 1, high two from phase 2.
        wasm_v128_store(&w[word_index], wasm_i32x4_shuffle(t0, t1, 0, 1, 6, 7));
    }

    sz_u32_t a = hash[0], b = hash[1], c = hash[2], d = hash[3];
    sz_u32_t e = hash[4], f = hash[5], g = hash[6], h = hash[7];
    for (sz_size_t round_index = 0; round_index < 64; ++round_index) {
        sz_u32_t temp1 = h + sz_sha256_sigma1_(e) + sz_sha256_ch_(e, f, g) + round_constants[round_index] +
                         w[round_index];
        sz_u32_t temp2 = sz_sha256_sigma0_(a) + sz_sha256_maj_(a, b, c);
        h = g, g = f, f = e;
        e = d + temp1;
        d = c, c = b, b = a;
        a = temp1 + temp2;
    }
    hash[0] += a, hash[1] += b, hash[2] += c, hash[3] += d;
    hash[4] += e, hash[5] += f, hash[6] += g, hash[7] += h;
}

SZ_PUBLIC void sz_sha256_state_init_v128(sz_sha256_state_t *state) { sz_sha256_state_init_serial(state); }

SZ_PUBLIC void sz_sha256_state_update_v128(sz_sha256_state_t *state_ptr, sz_cptr_t data, sz_size_t length) {
    // Identical driver to `sz_sha256_state_update_serial`, routed through the SIMD block processor.
    sz_u8_t const *input = (sz_u8_t const *)data;
    sz_size_t const current_block_index = state_ptr->block_length / 64;
    sz_size_t const final_block_index = (state_ptr->block_length + length) / 64;
    int const stays_in_the_block = current_block_index == final_block_index;
    int const fills_the_block = (state_ptr->block_length + length) % 64 == 0;

    state_ptr->total_length += length;
    if (stays_in_the_block && !fills_the_block) {
        for (; length; --length, ++state_ptr->block_length, ++input) state_ptr->block[state_ptr->block_length] = *input;
        return;
    }

    sz_size_t const head_length = (64 - state_ptr->block_length) % 64;
    sz_size_t const tail_length = (state_ptr->block_length + length) % 64;
    sz_size_t const body_length = length - head_length - tail_length;

    sz_align_(32) sz_u32_t hash[8];
    for (sz_size_t word_index = 0; word_index < 8; ++word_index) hash[word_index] = state_ptr->hash[word_index];

    if (head_length) {
        for (sz_size_t byte_index = 0; byte_index < head_length; ++byte_index)
            state_ptr->block[state_ptr->block_length++] = input[byte_index];
        sz_sha256_process_block_v128_(hash, state_ptr->block);
        state_ptr->block_length = 0;
        input += head_length;
    }
    for (sz_size_t processed = 0; processed < body_length; processed += 64, input += 64)
        sz_sha256_process_block_v128_(hash, input);
    for (sz_size_t byte_index = 0; byte_index < tail_length; ++byte_index)
        state_ptr->block[byte_index] = input[byte_index];
    state_ptr->block_length = tail_length;

    for (sz_size_t word_index = 0; word_index < 8; ++word_index) state_ptr->hash[word_index] = hash[word_index];
}

SZ_PUBLIC void sz_sha256_state_digest_v128(sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(32)]) {
    sz_sha256_state_digest_serial(state, digest);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_V128_H_
