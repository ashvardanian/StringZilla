/**
 *  @brief LoongArch LASX (256-bit) backend for hash.
 *  @file include/stringzilla/hash/lasx.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_LASX_H_
#define STRINGZILLA_HASH_LASX_H_

#include "stringzilla/types.h"
#include "stringzilla/hash/serial.h"
#include "stringzilla/compare.h" // `sz_equal`

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_LASX

SZ_API_COMPTIME sz_u64_t sz_bytesum_lasx(sz_cptr_t text, sz_size_t length) {
    // When the buffer is small, there isn't much to innovate.
    if (length <= 32) { return sz_bytesum_serial(text, length); }
    else {
        // LASX has no single SAD (`_mm256_sad_epu8`) instruction. Rather than collapse each block all the
        // way down to 64-bit lanes with a 3-stage widening-add chain (4 ops/block), we keep a 16-lane u16
        // accumulator across the loop and reduce ONCE at the end. Per block we widen each adjacent byte
        // pair into a u16 with `xvhaddw_hu_bu` (one op) and add it into the u16 lanes (one op). Each u16
        // lane grows by at most 2*255 = 510 per block, so 65535/510 = 128 blocks fit before overflow; we
        // flush to a u64 accumulator every 120 blocks to stay safe.
        __m256i const zero_vec = __lasx_xvreplgr2vr_b(0);
        __m256i long_sums_vec = zero_vec;  // 4x u64 long-term accumulator
        __m256i short_sums_vec = zero_vec; // 16x u16 short-term accumulator
        sz_size_t blocks_since_flush = 0;
        for (; length >= 32; text += 32, length -= 32) {
            __m256i text_vec = __lasx_xvld(text, 0);
            short_sums_vec = __lasx_xvadd_h(short_sums_vec, __lasx_xvhaddw_hu_bu(text_vec, text_vec)); // 16x u16
            if (++blocks_since_flush == 120) {
                // Drain the u16 lanes into the u64 lanes: 16x u16 -> 8x u32 -> 4x u64.
                __m256i widened_to_u32 = __lasx_xvhaddw_wu_hu(short_sums_vec, short_sums_vec);
                long_sums_vec = __lasx_xvadd_d(long_sums_vec, __lasx_xvhaddw_du_wu(widened_to_u32, widened_to_u32));
                short_sums_vec = zero_vec;
                blocks_since_flush = 0;
            }
        }
        // Final drain of any residual u16 lanes.
        {
            __m256i widened_to_u32 = __lasx_xvhaddw_wu_hu(short_sums_vec, short_sums_vec);
            long_sums_vec = __lasx_xvadd_d(long_sums_vec, __lasx_xvhaddw_du_wu(widened_to_u32, widened_to_u32));
        }

        // Reduce the 4 accumulated 64-bit lanes. `xvhaddw_q_d` folds adjacent doublewords into the two
        // 128-bit quadword lanes (lane 0 = sums[0]+sums[1], lane 1 = sums[2]+sums[3]), so a single op
        // replaces two of the four vector->GPR extracts the naive 4-way sum would need.
        __m256i pairwise_sums_vec = __lasx_xvhaddw_q_d(long_sums_vec, long_sums_vec);
        sz_u64_t result = (sz_u64_t)__lasx_xvpickve2gr_du(pairwise_sums_vec, 0) +
                          (sz_u64_t)__lasx_xvpickve2gr_du(pairwise_sums_vec, 2);
        if (length) result += sz_bytesum_serial(text, length);
        return result;
    }
}

/*  Vector-permute (tower-field) AES round for the LASX backend.
 *
 *  StringZilla's hashing (`sz_hash`) and CSPRNG (`sz_fill_random`) are built on a single round of AES
 *  (`_mm_aesenc_si128` on x86, `vaeseq`/`vaesmcq` on Arm). The base LoongArch LASX ISA exposes wide
 *  integer SIMD but @b no AES acceleration primitives. Instead of falling back to the byte-serial S-box,
 *  we implement Mike Hamburg's "vector permute" (vpaes) AES round in 128-bit `__lsx_*` lanes — the AES
 *  state is exactly 128 bits, so the 128-bit LSX path is the natural fit.
 *
 *  Pipeline of `sz_emulate_aesenc_lasx_`, matching `sz_emulate_aesenc_si128_serial_` bit-for-bit:
 *
 *   1. @b ShiftRows is a single `__lsx_vshuf_b` byte permutation applied up-front (the serial reference
 *      folds ShiftRows into its `SubBytes` indexing; we hoist it so SubBytes can be lane-parallel).
 *   2. @b SubBytes is computed as GF(2^8) inversion + affine via the tower field GF((2^4)^2):
 *        - `ipt` nibble tables map the standard AES basis into the tower basis (two `vshuf_b` + xor),
 *        - the GF(16) inversion of `e = h*Y + l` uses `d = inv4(nu*h^2 + h*l + l^2)`, `h' = h*d`,
 *          `l' = (h^l)*d`, where every GF(16) multiply is a log/antilog pair of `vshuf_b` lookups,
 *        - `sbo` nibble tables map back to the standard basis and apply the AES affine (`^ 0x63`).
 *      The tower isomorphism is a genuine GF(2)-linear field isomorphism (so the nibble decomposition is
 *      exact); its tables are derived offline and verified to reproduce the AES S-box for all 256 inputs.
 *   3. @b MixColumns is the textbook `xtime` (`__lsx_vslli_b` + a conditional `0x1b` xor selected by
 *      `__lsx_vslti_b(x, 0)`, i.e. the sign/MSB mask), combined within each 4-byte column.
 *   4. @b AddRoundKey is a single `__lsx_vxor_v`.
 *
 *  @see Mike Hamburg, "Accelerating AES with Vector Permute Instructions", CHES 2009:
 *       https://shiftleft.org/papers/vector_aes/vector_aes.pdf (origin of the tower-field nibble tables).
 *  @see Reference vpaes implementation (constant-time, S-box-equivalent), OpenSSL `vpaes-x86_64.pl`:
 *       https://github.com/openssl/openssl/blob/master/crypto/aes/asm/vpaes-x86_64.pl
 *  @see `sz_emulate_aesenc_si128_serial_` (hash/serial.h) — the byte-exact oracle every output is checked
 *       against; the `tables` here are validated to reproduce the AES S-box for all 256 inputs.
 */

/** @brief  Pre-computed vpaes-style nibble tables for the LASX AES round (see derivation above). */
SZ_HELPER_INLINE sz_u8_t const *sz_aes_lasx_tables_(void) {
    // Layout: [iptlo, ipthi, sbolo, sbohi, glog, gexp, ginv, shiftrows] x16 bytes each.
    static sz_align_(64) sz_u8_t const tables[8 * 16] = {
        // k_iptlo
        0x00,
        0x01,
        0x20,
        0x21,
        0x46,
        0x47,
        0x66,
        0x67,
        0x4C,
        0x4D,
        0x6C,
        0x6D,
        0x0A,
        0x0B,
        0x2A,
        0x2B, //
        // k_ipthi
        0x00,
        0x3C,
        0xD5,
        0xE9,
        0x34,
        0x08,
        0xE1,
        0xDD,
        0xE5,
        0xD9,
        0x30,
        0x0C,
        0xD1,
        0xED,
        0x04,
        0x38, //
        // k_sbolo
        0x00,
        0x1F,
        0xB2,
        0xAD,
        0xAB,
        0xB4,
        0x19,
        0x06,
        0x36,
        0x29,
        0x84,
        0x9B,
        0x9D,
        0x82,
        0x2F,
        0x30, //
        // k_sbohi
        0x00,
        0x52,
        0x3E,
        0x6C,
        0x65,
        0x37,
        0x5B,
        0x09,
        0x60,
        0x32,
        0x5E,
        0x0C,
        0x05,
        0x57,
        0x3B,
        0x69, //
        // k_glog  (GF(16) discrete log, base 2, poly 0x13; log[0] unused)
        0x00,
        0x00,
        0x01,
        0x04,
        0x02,
        0x08,
        0x05,
        0x0A,
        0x03,
        0x0E,
        0x09,
        0x07,
        0x06,
        0x0D,
        0x0B,
        0x0C, //
        // k_gexp  (GF(16) antilog; entry 15 wraps to entry 0)
        0x01,
        0x02,
        0x04,
        0x08,
        0x03,
        0x06,
        0x0C,
        0x0B,
        0x05,
        0x0A,
        0x07,
        0x0E,
        0x0F,
        0x0D,
        0x09,
        0x01, //
        // k_ginv  (GF(16) multiplicative inverse; inv[0] = 0)
        0x00,
        0x01,
        0x09,
        0x0E,
        0x0D,
        0x0B,
        0x07,
        0x06,
        0x0F,
        0x02,
        0x0C,
        0x05,
        0x0A,
        0x04,
        0x03,
        0x08, //
        // k_shiftrows (ShiftRows source byte indices, row-major output)
        0x00,
        0x05,
        0x0A,
        0x0F,
        0x04,
        0x09,
        0x0E,
        0x03,
        0x08,
        0x0D,
        0x02,
        0x07,
        0x0C,
        0x01,
        0x06,
        0x0B, //
    };
    return &tables[0];
}

/** @brief  `_mm_shuffle_epi8`-equivalent for indices in 0..15 (top bits clear). */
SZ_HELPER_INLINE __m128i sz_lsx_pshufb_(__m128i table, __m128i indices) {
    // `__lsx_vshuf_b(a, b, c)` masks `c` to 5 bits: 0..15 selects from `b`, 16..31 from `a`.
    // With indices in 0..15 the first operand is irrelevant; we reuse `indices` as a dummy.
    return __lsx_vshuf_b(indices, table, indices);
}

/** @brief  Lane-wise GF(16) multiply (poly 0x13) of two nibble vectors via log/antilog tables. */
SZ_HELPER_AUTO __m128i sz_lsx_gf16_mul_(__m128i factor_a, __m128i factor_b, __m128i gf16_log, __m128i gf16_exp,
                                        __m128i zero_vec) {
    __m128i log_sum = __lsx_vadd_b(sz_lsx_pshufb_(gf16_log, factor_a), sz_lsx_pshufb_(gf16_log, factor_b)); // 0..28
    __m128i fifteen_vec = __lsx_vreplgr2vr_b(15);
    __m128i wraps_mask = __lsx_vsle_b(fifteen_vec, log_sum);                           // 0xff where log_sum >= 15
    __m128i folded_log = __lsx_vsub_b(log_sum, __lsx_vand_v(wraps_mask, fifteen_vec)); // fold into 0..14
    __m128i antilog_product = sz_lsx_pshufb_(gf16_exp, folded_log);
    __m128i zero_factor_mask = __lsx_vor_v(__lsx_vseq_b(factor_a, zero_vec), __lsx_vseq_b(factor_b, zero_vec));
    return __lsx_vandn_v(zero_factor_mask, antilog_product); // (~zero_factor_mask) & antilog_product
}

/**
 *  @brief AES SubBytes on the ShiftRows-permuted state via tower-field GF((2^4)^2) inversion + affine.
 *  @return The S-box output (with the `^ 0x63` AES affine constant already applied).
 */
SZ_HELPER_AUTO __m128i sz_emulate_aes_subbytes_lasx_( //
    __m128i shifted_state, __m128i zero_vec, __m128i low_nibble_mask, __m128i input_transform_low,
    __m128i input_transform_high, __m128i sbox_output_low, __m128i sbox_output_high, __m128i gf16_log, __m128i gf16_exp,
    __m128i gf16_inverse) {
    __m128i state_low_nibbles = __lsx_vand_v(shifted_state, low_nibble_mask);
    __m128i state_high_nibbles = __lsx_vand_v(__lsx_vsrli_b(shifted_state, 4), low_nibble_mask);
    __m128i tower_basis = __lsx_vxor_v(sz_lsx_pshufb_(input_transform_low, state_low_nibbles),
                                       sz_lsx_pshufb_(input_transform_high, state_high_nibbles));
    __m128i tower_high = __lsx_vand_v(__lsx_vsrli_b(tower_basis, 4), low_nibble_mask);
    __m128i tower_low = __lsx_vand_v(tower_basis, low_nibble_mask);
    __m128i tower_low_xor_high = __lsx_vxor_v(tower_low, tower_high);
    __m128i tower_high_squared = sz_lsx_gf16_mul_(tower_high, tower_high, gf16_log, gf16_exp, zero_vec);
    __m128i tower_nu = __lsx_vreplgr2vr_b(0x08); // the irreducible-polynomial offset of the tower field
    __m128i inverse_denominator = __lsx_vxor_v(
        sz_lsx_gf16_mul_(tower_high_squared, tower_nu, gf16_log, gf16_exp, zero_vec),
        sz_lsx_gf16_mul_(tower_low, tower_low_xor_high, gf16_log, gf16_exp, zero_vec));
    __m128i denominator_inverse = sz_lsx_pshufb_(gf16_inverse, inverse_denominator);
    __m128i inverted_high = sz_lsx_gf16_mul_(tower_high, denominator_inverse, gf16_log, gf16_exp, zero_vec);
    __m128i inverted_low = sz_lsx_gf16_mul_(tower_low_xor_high, denominator_inverse, gf16_log, gf16_exp, zero_vec);
    __m128i inverted_nibbles = __lsx_vor_v(__lsx_vslli_b(inverted_high, 4), inverted_low);
    __m128i inverted_high_nibbles = __lsx_vand_v(__lsx_vsrli_b(inverted_nibbles, 4), low_nibble_mask);
    __m128i inverted_low_nibbles = __lsx_vand_v(inverted_nibbles, low_nibble_mask);
    __m128i sbox_output = __lsx_vxor_v(sz_lsx_pshufb_(sbox_output_low, inverted_low_nibbles),
                                       sz_lsx_pshufb_(sbox_output_high, inverted_high_nibbles));
    return __lsx_vxor_v(sbox_output, __lsx_vreplgr2vr_b(0x63)); // AES affine constant
}

/**
 *  @brief AES MixColumns over the four 4-byte columns of the S-box output.
 *  @return `c[j] ^ (col_base^col_rot1^col_rot2^col_rot3) ^ xtime(c[j] ^ c[j+1])` per column.
 */
SZ_HELPER_AUTO __m128i sz_emulate_aes_mixcolumns_lasx_(__m128i sbox_output) {
    // Build rotate masks on the fly: add j and (j+1 within group) shuffle indices.
    static sz_align_(16) sz_u8_t const rot1_bytes[16] = {1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12};
    static sz_align_(16) sz_u8_t const rot2_bytes[16] = {2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13};
    static sz_align_(16) sz_u8_t const rot3_bytes[16] = {3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14};
    __m128i col_base = sbox_output;
    __m128i col_rot1 = sz_lsx_pshufb_(sbox_output, __lsx_vld(rot1_bytes, 0));
    __m128i col_rot2 = sz_lsx_pshufb_(sbox_output, __lsx_vld(rot2_bytes, 0));
    __m128i col_rot3 = sz_lsx_pshufb_(sbox_output, __lsx_vld(rot3_bytes, 0));
    __m128i column_rotation_sum = __lsx_vxor_v(__lsx_vxor_v(col_base, col_rot1), __lsx_vxor_v(col_rot2, col_rot3));
    // xtime(col_base ^ col_rot1): (x << 1) ^ (0x1b where MSB set). `vslti_b(x, 0)` yields 0xff for MSB-set bytes.
    __m128i adjacent_column_xor = __lsx_vxor_v(col_base, col_rot1);
    __m128i column_xtime = __lsx_vxor_v(__lsx_vslli_b(adjacent_column_xor, 1),
                                        __lsx_vand_v(__lsx_vslti_b(adjacent_column_xor, 0), __lsx_vreplgr2vr_b(0x1b)));
    return __lsx_vxor_v(__lsx_vxor_v(col_base, column_rotation_sum), column_xtime);
}

/**
 *  @brief Emulates a single `_mm_aesenc_si128` round on LoongArch LSX (128-bit lanes).
 *  @return Result of `MixColumns(SubBytes(ShiftRows(state))) ^ round_key`, bit-identical to
 *          `sz_emulate_aesenc_si128_serial_`.
 */
SZ_HELPER_AUTO __m128i sz_emulate_aesenc_lasx_(__m128i state, __m128i round_key) {
    sz_u8_t const *tables = sz_aes_lasx_tables_();
    __m128i zero_vec = __lsx_vreplgr2vr_b(0);
    __m128i low_nibble_mask = __lsx_vreplgr2vr_b(0x0F);
    __m128i input_transform_low = __lsx_vld(tables + 0, 0), input_transform_high = __lsx_vld(tables + 16, 0);
    __m128i sbox_output_low = __lsx_vld(tables + 32, 0), sbox_output_high = __lsx_vld(tables + 48, 0);
    __m128i gf16_log = __lsx_vld(tables + 64, 0), gf16_exp = __lsx_vld(tables + 80, 0);
    __m128i gf16_inverse = __lsx_vld(tables + 96, 0), shift_rows_indices = __lsx_vld(tables + 112, 0);

    // (1) ShiftRows up-front so SubBytes is purely lane-parallel.
    __m128i shifted_state = sz_lsx_pshufb_(state, shift_rows_indices);

    // (2) SubBytes via tower-field inversion + affine.
    __m128i sbox_output = sz_emulate_aes_subbytes_lasx_(shifted_state, zero_vec, low_nibble_mask, input_transform_low,
                                                        input_transform_high, sbox_output_low, sbox_output_high,
                                                        gf16_log, gf16_exp, gf16_inverse);

    // (3) MixColumns.
    __m128i mixed = sz_emulate_aes_mixcolumns_lasx_(sbox_output);

    // (4) AddRoundKey.
    return __lsx_vxor_v(mixed, round_key);
}

/** @brief  Load 16 bytes from an `sz_u128_vec_t`-style buffer into an LSX register. */
SZ_HELPER_INLINE __m128i sz_lsx_load128_(void const *pointer) { return __lsx_vld(pointer, 0); }
/** @brief  Store an LSX register into a 16-byte buffer. */
SZ_HELPER_INLINE void sz_lsx_store128_(void *pointer, __m128i value_u8x16) { __lsx_vst(value_u8x16, pointer, 0); }

SZ_HELPER_AUTO void sz_hash_state_short_init_lasx_(sz_hash_state_aligned_for_short_t *state, sz_u64_t seed) {
    state->key.u64s[0] = seed, state->key.u64s[1] = seed;
    sz_u64_t const *pi = sz_hash_pi_constants_();
    state->aes.u64s[0] = seed ^ pi[0], state->aes.u64s[1] = seed ^ pi[1];
    state->sum.u64s[0] = seed ^ pi[8], state->sum.u64s[1] = seed ^ pi[9];
}

SZ_HELPER_AUTO void sz_hash_state_short_update_lasx_(sz_hash_state_aligned_for_short_t *state, sz_u128_vec_t block) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    __m128i aes = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&state->aes), sz_lsx_load128_(&block));
    sz_lsx_store128_(&state->aes, aes);
    state->sum = sz_emulate_shuffle_epi8_serial_(state->sum, shuffle);
    state->sum.u64s[0] += block.u64s[0], state->sum.u64s[1] += block.u64s[1];
}

SZ_HELPER_AUTO sz_u64_t sz_hash_state_short_finalize_lasx_(sz_hash_state_aligned_for_short_t const *state,
                                                           sz_size_t length) {
    sz_u128_vec_t key_with_length = state->key;
    key_with_length.u64s[0] += length;
    __m128i mixed = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&state->sum), sz_lsx_load128_(&state->aes));
    __m128i mixed_in_register = sz_emulate_aesenc_lasx_(
        sz_emulate_aesenc_lasx_(mixed, sz_lsx_load128_(&key_with_length)), mixed);
    sz_u128_vec_t result;
    sz_lsx_store128_(&result, mixed_in_register);
    return result.u64s[0];
}

SZ_API_COMPTIME void sz_hash_state_init_lasx(sz_hash_state_t *state, sz_u64_t seed);
SZ_HELPER_AUTO sz_hash_state_aligned_t sz_hash_state_load_lasx_(sz_hash_state_t const *packed);
SZ_HELPER_AUTO void sz_hash_state_store_lasx_(sz_hash_state_t *packed, sz_hash_state_aligned_t const *state);
SZ_HELPER_AUTO void sz_hash_state_update_lasx_(sz_hash_state_aligned_t *state);
SZ_HELPER_AUTO sz_u64_t sz_hash_state_finalize_lasx_(sz_hash_state_aligned_t state);

SZ_API_COMPTIME SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_lasx(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {
    if (length <= 16) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_lasx_(&state, seed);
        sz_u128_vec_t data_vec;
        data_vec.u64s[0] = data_vec.u64s[1] = 0;
        for (sz_size_t byte_index = 0; byte_index < length; ++byte_index) data_vec.u8s[byte_index] = start[byte_index];
        sz_hash_state_short_update_lasx_(&state, data_vec);
        return sz_hash_state_short_finalize_lasx_(&state, length);
    }
    else if (length <= 32) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_lasx_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.lsx = __lsx_vld(start, 0);
        data1_vec.lsx = __lsx_vld(start + length - 16, 0);
        sz_hash_shift_in_register_serial_(&data1_vec, (int)(32 - length));
        sz_hash_state_short_update_lasx_(&state, data0_vec);
        sz_hash_state_short_update_lasx_(&state, data1_vec);
        return sz_hash_state_short_finalize_lasx_(&state, length);
    }
    else if (length <= 48) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_lasx_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.lsx = __lsx_vld(start, 0);
        data1_vec.lsx = __lsx_vld(start + 16, 0);
        data2_vec.lsx = __lsx_vld(start + length - 16, 0);
        sz_hash_shift_in_register_serial_(&data2_vec, (int)(48 - length));
        sz_hash_state_short_update_lasx_(&state, data0_vec);
        sz_hash_state_short_update_lasx_(&state, data1_vec);
        sz_hash_state_short_update_lasx_(&state, data2_vec);
        return sz_hash_state_short_finalize_lasx_(&state, length);
    }
    else if (length <= 64) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_lasx_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.lsx = __lsx_vld(start, 0);
        data1_vec.lsx = __lsx_vld(start + 16, 0);
        data2_vec.lsx = __lsx_vld(start + 32, 0);
        data3_vec.lsx = __lsx_vld(start + length - 16, 0);
        sz_hash_shift_in_register_serial_(&data3_vec, (int)(64 - length));
        sz_hash_state_short_update_lasx_(&state, data0_vec);
        sz_hash_state_short_update_lasx_(&state, data1_vec);
        sz_hash_state_short_update_lasx_(&state, data2_vec);
        sz_hash_state_short_update_lasx_(&state, data3_vec);
        return sz_hash_state_short_finalize_lasx_(&state, length);
    }
    else {
        // The aligned twin lets the kernels use clean aligned lane access; one-shot never touches the packed type
        // except to reuse `init` (layout-locked by the `static_assert`s on `sz_hash_state_aligned_t`).
        sz_align_(64) sz_hash_state_aligned_t state;
        sz_hash_state_init_lasx((sz_hash_state_t *)&state, seed);

        // Absorb every full 64-byte block EXCEPT the last; the final block (a full 64 or a partial tail) stays
        // buffered in `ins` for `sz_hash_state_finalize_lasx_` to fold - the same deferral the streaming path uses.
        for (; state.ins_length + 64 < length; state.ins_length += 64) {
            __lasx_xvst(__lasx_xvld(start + state.ins_length, 0), state.ins.u8s, 0);
            __lasx_xvst(__lasx_xvld(start + state.ins_length + 32, 0), state.ins.u8s + 32, 0);
            sz_hash_state_update_lasx_(&state);
        }

        // Stage the final [ins_length, length) bytes (1..64) into a zeroed buffer; finalize folds them.
        __m256i const zero_vec = __lasx_xvreplgr2vr_b(0);
        __lasx_xvst(zero_vec, state.ins.u8s, 0);
        __lasx_xvst(zero_vec, state.ins.u8s + 32, 0);
        for (sz_size_t byte_index = 0; state.ins_length < length; ++byte_index, ++state.ins_length)
            state.ins.u8s[byte_index] = (sz_u8_t)start[state.ins_length];
        return sz_hash_state_finalize_lasx_(state);
    }
}

SZ_API_COMPTIME void sz_hash_state_init_lasx(sz_hash_state_t *state, sz_u64_t seed) {
    sz_hash_state_init_serial(state, seed);
}

/**
 *  @brief Loads the packed public state into the aligned internal twin (LASX: 2x `__lasx_xvld` per 64-byte field).
 */
SZ_HELPER_AUTO sz_hash_state_aligned_t sz_hash_state_load_lasx_(sz_hash_state_t const *packed) {
    sz_hash_state_aligned_t state;
    __lasx_xvst(__lasx_xvld(packed->aes, 0), state.aes.u8s, 0);
    __lasx_xvst(__lasx_xvld(packed->aes + 32, 0), state.aes.u8s + 32, 0);
    __lasx_xvst(__lasx_xvld(packed->sum, 0), state.sum.u8s, 0);
    __lasx_xvst(__lasx_xvld(packed->sum + 32, 0), state.sum.u8s + 32, 0);
    __lasx_xvst(__lasx_xvld(packed->ins, 0), state.ins.u8s, 0);
    __lasx_xvst(__lasx_xvld(packed->ins + 32, 0), state.ins.u8s + 32, 0);
    sz_lsx_store128_(&state.key, sz_lsx_load128_(packed->key));
    state.ins_length = packed->ins_length;
    return state;
}

/** @brief Stores the aligned internal twin back into the packed public state. */
SZ_HELPER_AUTO void sz_hash_state_store_lasx_(sz_hash_state_t *packed, sz_hash_state_aligned_t const *state) {
    __lasx_xvst(__lasx_xvld(state->aes.u8s, 0), packed->aes, 0);
    __lasx_xvst(__lasx_xvld(state->aes.u8s + 32, 0), packed->aes + 32, 0);
    __lasx_xvst(__lasx_xvld(state->sum.u8s, 0), packed->sum, 0);
    __lasx_xvst(__lasx_xvld(state->sum.u8s + 32, 0), packed->sum + 32, 0);
    __lasx_xvst(__lasx_xvld(state->ins.u8s, 0), packed->ins, 0);
    __lasx_xvst(__lasx_xvld(state->ins.u8s + 32, 0), packed->ins + 32, 0);
    sz_lsx_store128_(packed->key, state->key.lsx);
    packed->ins_length = state->ins_length;
}

/**
 *  @brief Absorbs the buffered 64-byte block into the aligned state (four 128-bit lanes), in place.
 *  @param state Pointer to the aligned hash state whose `ins` lanes are consumed.
 */
SZ_HELPER_AUTO void sz_hash_state_update_lasx_(sz_hash_state_aligned_t *state) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) {
        sz_u128_vec_t *aes = &state->aes.u128s[lane_index];
        sz_u128_vec_t *sum = &state->sum.u128s[lane_index];
        sz_u128_vec_t const *ins = &state->ins.u128s[lane_index];
        sz_lsx_store128_(aes, sz_emulate_aesenc_lasx_(sz_lsx_load128_(aes), sz_lsx_load128_(ins)));
        *sum = sz_emulate_shuffle_epi8_serial_(*sum, shuffle);
        sum->u64s[0] += ins->u64s[0], sum->u64s[1] += ins->u64s[1];
    }
}

/**
 *  @brief Finalizes the full 512-bit hash state and returns a 64-bit digest.
 *  @param state The hash state, taken by value.
 *  @return 64-bit hash value derived by folding the four AES lanes together with the key.
 */
SZ_HELPER_AUTO sz_u64_t sz_hash_state_finalize_lasx_(sz_hash_state_aligned_t state) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    sz_u128_vec_t key_with_length;
    key_with_length.u64s[0] = state.key.u64s[0] + state.ins_length;
    key_with_length.u64s[1] = state.key.u64s[1];

    // Fold the deferred final block (still buffered in `ins` - a full 64 bytes or a zero-padded tail) into each
    // lane. Folding the last block here, rather than in `update`, lets both one-shot `sz_hash` and the streaming
    // digest defer it and share this single finalization.
    __m128i aes0 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&state.aes.u128s[0]), sz_lsx_load128_(&state.ins.u128s[0]));
    __m128i aes1 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&state.aes.u128s[1]), sz_lsx_load128_(&state.ins.u128s[1]));
    __m128i aes2 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&state.aes.u128s[2]), sz_lsx_load128_(&state.ins.u128s[2]));
    __m128i aes3 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&state.aes.u128s[3]), sz_lsx_load128_(&state.ins.u128s[3]));
    sz_u128_vec_t sum0 = sz_emulate_shuffle_epi8_serial_(state.sum.u128s[0], shuffle);
    sz_u128_vec_t sum1 = sz_emulate_shuffle_epi8_serial_(state.sum.u128s[1], shuffle);
    sz_u128_vec_t sum2 = sz_emulate_shuffle_epi8_serial_(state.sum.u128s[2], shuffle);
    sz_u128_vec_t sum3 = sz_emulate_shuffle_epi8_serial_(state.sum.u128s[3], shuffle);
    sum0.u64s[0] += state.ins.u128s[0].u64s[0], sum0.u64s[1] += state.ins.u128s[0].u64s[1];
    sum1.u64s[0] += state.ins.u128s[1].u64s[0], sum1.u64s[1] += state.ins.u128s[1].u64s[1];
    sum2.u64s[0] += state.ins.u128s[2].u64s[0], sum2.u64s[1] += state.ins.u128s[2].u64s[1];
    sum3.u64s[0] += state.ins.u128s[3].u64s[0], sum3.u64s[1] += state.ins.u128s[3].u64s[1];

    __m128i mixed0 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&sum0), aes0);
    __m128i mixed1 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&sum1), aes1);
    __m128i mixed2 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&sum2), aes2);
    __m128i mixed3 = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&sum3), aes3);
    __m128i mixed01 = sz_emulate_aesenc_lasx_(mixed0, mixed1);
    __m128i mixed23 = sz_emulate_aesenc_lasx_(mixed2, mixed3);
    __m128i mixed = sz_emulate_aesenc_lasx_(mixed01, mixed23);
    __m128i mixed_in_register = sz_emulate_aesenc_lasx_(
        sz_emulate_aesenc_lasx_(mixed, sz_lsx_load128_(&key_with_length)), mixed);
    sz_u128_vec_t result;
    sz_lsx_store128_(&result, mixed_in_register);
    return result.u64s[0];
}

SZ_API_COMPTIME void sz_hash_state_update_lasx(sz_hash_state_t *packed, sz_cptr_t text, sz_size_t length) {
    // Load the packed public state (any alignment) into an aligned twin once, buffer/absorb on it, then store back.
    sz_hash_state_aligned_t state = sz_hash_state_load_lasx_(packed);
    __m256i const zero_vec = __lasx_xvreplgr2vr_b(0);
    while (length) {
        sz_size_t progress_in_block = state.ins_length % 64;
        // A full block from an earlier fill is still buffered: its absorption is DEFERRED so `digest` can choose
        // the same minimal (<=64) / full (>64) path the one-shot `sz_hash` would, keyed on the total length. Now
        // that more bytes have arrived, that block is interior - flush it and clear the buffer.
        if (progress_in_block == 0 && state.ins_length != 0) {
            sz_hash_state_update_lasx_(&state);
            __lasx_xvst(zero_vec, state.ins.u8s, 0);
            __lasx_xvst(zero_vec, state.ins.u8s + 32, 0);
        }
        sz_size_t to_copy = sz_min_of_two(length, 64 - progress_in_block);
        state.ins_length += to_copy;
        length -= to_copy;
        // Append to the internal buffer; the block that exactly fills it stays deferred (see above).
        while (to_copy--) state.ins.u8s[progress_in_block++] = (sz_u8_t)*text++;
    }
    sz_hash_state_store_lasx_(packed, &state);
}

SZ_API_COMPTIME sz_u64_t sz_hash_state_digest_lasx(sz_hash_state_t const *packed) {
    sz_hash_state_aligned_t state = sz_hash_state_load_lasx_(packed);
    sz_size_t length = state.ins_length;
    // Inputs longer than one block fold through the full four-lane state. The deferred final block is still
    // buffered in `ins`, and `sz_hash_state_finalize_lasx_` folds it - reproducing one-shot `sz_hash` exactly.
    if (length > 64) return sz_hash_state_finalize_lasx_(state);

    // Switch back to a smaller "short" state for small inputs; the aligned twin lanes are read directly.
    sz_hash_state_aligned_for_short_t minimal_state;
    minimal_state.key = state.key;
    minimal_state.aes = state.aes.u128s[0];
    minimal_state.sum = state.sum.u128s[0];
    if (length <= 16) {
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[0]);
        return sz_hash_state_short_finalize_lasx_(&minimal_state, length);
    }
    else if (length <= 32) {
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[0]);
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[1]);
        return sz_hash_state_short_finalize_lasx_(&minimal_state, length);
    }
    else if (length <= 48) {
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[0]);
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[1]);
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[2]);
        return sz_hash_state_short_finalize_lasx_(&minimal_state, length);
    }
    else {
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[0]);
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[1]);
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[2]);
        sz_hash_state_short_update_lasx_(&minimal_state, state.ins.u128s[3]);
        return sz_hash_state_short_finalize_lasx_(&minimal_state, length);
    }
}

SZ_API_COMPTIME void sz_fill_random_lasx(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_u64_t const *pi_constants = sz_hash_pi_constants_();
    sz_u128_vec_t input_vec, pi_vec, key_vec, generated_vec;
    for (sz_size_t lane_index = 0; length; ++lane_index) {
        input_vec.u64s[0] = input_vec.u64s[1] = nonce + lane_index;
        pi_vec = ((sz_u128_vec_t const *)pi_constants)[lane_index % 4];
        key_vec.u64s[0] = nonce ^ pi_vec.u64s[0];
        key_vec.u64s[1] = nonce ^ pi_vec.u64s[1];
        __m128i gen = sz_emulate_aesenc_lasx_(sz_lsx_load128_(&input_vec), sz_lsx_load128_(&key_vec));
        sz_lsx_store128_(&generated_vec, gen);
        for (sz_size_t byte_index = 0; byte_index < 16 && length; ++byte_index, --length)
            *text++ = generated_vec.u8s[byte_index];
    }
}

/*  SHA-256 with a SIMD-vectorized message schedule.
 *
 *  The compression rounds are inherently sequential (each updates `a..h` from the previous round), so for a
 *  single message they cannot be vectorized — that part stays scalar and identical to the serial reference.
 *  The @b message @b schedule, however, is a fixed recurrence we can compute four words at a time in 128-bit
 *  LSX lanes: `W[t] = sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16]`. Within a group of four, `W[t+2]`
 *  and `W[t+3]` depend on `W[t]`/`W[t+1]` produced by the same group, so we finish in two steps. There is no
 *  SHA hardware on LoongArch (unlike x86 SHA-NI / Arm `sha256h`); this is a pure-SIMD-ALU acceleration.
 *
 *  @see FIPS 180-4 (SHA-256) and J. Guilford et al., "Fast SHA-256 Implementations on Intel Architecture
 *       Processors" (the standard SIMD message-schedule decomposition this mirrors).
 *  @see `sz_sha256_process_block_serial_` (hash/serial.h) — the byte-exact oracle this is validated against. */

/** @brief  Lane-wise SHA-256 lowercase-sigma0: `ROTR(x,7) ^ ROTR(x,18) ^ SHR(x,3)` over 4x u32. */
SZ_HELPER_INLINE __m128i sz_sha256_sigma0_lower_lasx_(__m128i words_vec) {
    return __lsx_vxor_v(__lsx_vxor_v(__lsx_vrotri_w(words_vec, 7), __lsx_vrotri_w(words_vec, 18)),
                        __lsx_vsrli_w(words_vec, 3));
}

/** @brief  Lane-wise SHA-256 lowercase-sigma1: `ROTR(x,17) ^ ROTR(x,19) ^ SHR(x,10)` over 4x u32. */
SZ_HELPER_INLINE __m128i sz_sha256_sigma1_lower_lasx_(__m128i words_vec) {
    return __lsx_vxor_v(__lsx_vxor_v(__lsx_vrotri_w(words_vec, 17), __lsx_vrotri_w(words_vec, 19)),
                        __lsx_vsrli_w(words_vec, 10));
}

SZ_HELPER_AUTO void sz_sha256_process_block_lasx_(sz_u32_t hash[sz_at_least_(8)],
                                                  sz_u8_t const block[sz_at_least_(64)]) {
    sz_u32_t const *round_constants = sz_sha256_round_constants_();
    sz_align_(16) sz_u32_t message_schedule[64];

    // Load W[0..15]: byte-reverse each 4-byte group (big-endian) with one `vshuf_b` per 16-byte chunk.
    static
        sz_align_(16) sz_u8_t const byte_reverse_indices[16] = {3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12};
    __m128i byte_reverse_vec = __lsx_vld(byte_reverse_indices, 0);
    for (sz_size_t group_index = 0; group_index < 4; ++group_index) {
        __m128i raw_vec = __lsx_vld(block + group_index * 16, 0);
        __lsx_vst(__lsx_vshuf_b(raw_vec, raw_vec, byte_reverse_vec), &message_schedule[group_index * 4], 0);
    }

    // Expand W[16..63] four lanes at a time. `low_pair_mask` keeps lanes 0,1 and zeroes lanes 2,3 so that
    // `sigma1` of the not-yet-computed high pair contributes nothing in step one.
    __m128i const low_pair_mask = __lsx_vinsgr2vr_d(__lsx_vreplgr2vr_d(-1), 0, 1); // [~0, ~0, 0, 0]
    for (sz_size_t word_index = 16; word_index < 64; word_index += 4) {
        __m128i window_16_vec = __lsx_vld(&message_schedule[word_index - 16], 0); // W[t-16 .. t-13]
        __m128i window_15_vec = __lsx_vld(&message_schedule[word_index - 15], 0); // W[t-15 .. t-12]
        __m128i window_7_vec = __lsx_vld(&message_schedule[word_index - 7], 0);   // W[t-7  .. t-4]
        __m128i base_vec = __lsx_vadd_w(__lsx_vadd_w(window_16_vec, sz_sha256_sigma0_lower_lasx_(window_15_vec)),
                                        window_7_vec);
        // Step 1: sigma1 of [W[t-2], W[t-1], 0, 0] finalizes lanes 0,1 (W[t], W[t+1]).
        __m128i prev_pair_vec = __lsx_vand_v(__lsx_vld(&message_schedule[word_index - 2], 0), low_pair_mask);
        __m128i schedule_vec = __lsx_vadd_w(base_vec, sz_sha256_sigma1_lower_lasx_(prev_pair_vec));
        // Step 2: shift the fresh low pair into lanes 2,3 and add its sigma1 to finalize W[t+2], W[t+3].
        __m128i new_pair_vec = __lsx_vbsll_v(schedule_vec, 8); // [0, 0, W[t], W[t+1]]
        schedule_vec = __lsx_vadd_w(schedule_vec, sz_sha256_sigma1_lower_lasx_(new_pair_vec));
        __lsx_vst(schedule_vec, &message_schedule[word_index], 0);
    }

    // Scalar compression — sequential, identical to `sz_sha256_process_block_serial_`.
    sz_u32_t a = hash[0], b = hash[1], c = hash[2], d = hash[3];
    sz_u32_t e = hash[4], f = hash[5], g = hash[6], h = hash[7], temp1, temp2;
    for (sz_size_t round_index = 0; round_index < 64; ++round_index) {
        temp1 = h + sz_sha256_sigma1_(e) + sz_sha256_ch_(e, f, g) + round_constants[round_index] +
                message_schedule[round_index];
        temp2 = sz_sha256_sigma0_(a) + sz_sha256_maj_(a, b, c);
        h = g, g = f, f = e;
        e = d + temp1;
        d = c, c = b, b = a;
        a = temp1 + temp2;
    }
    hash[0] += a, hash[1] += b, hash[2] += c, hash[3] += d;
    hash[4] += e, hash[5] += f, hash[6] += g, hash[7] += h;
}

SZ_API_COMPTIME void sz_sha256_state_init_lasx(sz_sha256_state_t *state) { sz_sha256_state_init_serial(state); }

SZ_API_COMPTIME void sz_sha256_state_update_lasx(sz_sha256_state_t *state_ptr, sz_cptr_t data, sz_size_t length) {
    sz_u8_t const *input = (sz_u8_t const *)data;
    sz_size_t const current_block_index = state_ptr->block_length / 64;
    sz_size_t const final_block_index = (state_ptr->block_length + length) / 64;
    int const stays_in_the_block = current_block_index == final_block_index;
    int const fills_the_block = (state_ptr->block_length + length) % 64 == 0;

    state_ptr->total_length += length;

    // Fast path: stays in the same block and doesn't fill it.
    if (stays_in_the_block && !fills_the_block) {
        for (; length; --length, ++state_ptr->block_length, ++input) state_ptr->block[state_ptr->block_length] = *input;
        return;
    }

    sz_size_t const head_length = (64 - state_ptr->block_length) % 64;
    sz_size_t const tail_length = (state_ptr->block_length + length) % 64;
    sz_size_t const body_length = length - head_length - tail_length;

    sz_align_(32) sz_u32_t hash[8];
    hash[0] = state_ptr->hash[0], hash[1] = state_ptr->hash[1], hash[2] = state_ptr->hash[2],
    hash[3] = state_ptr->hash[3], hash[4] = state_ptr->hash[4], hash[5] = state_ptr->hash[5],
    hash[6] = state_ptr->hash[6], hash[7] = state_ptr->hash[7];

    // Complete the partial block with the head, then run aligned bodies, then buffer the tail.
    if (head_length) {
        for (sz_size_t byte_index = 0; byte_index < head_length; ++byte_index)
            state_ptr->block[state_ptr->block_length++] = input[byte_index];
        sz_sha256_process_block_lasx_(hash, state_ptr->block);
        state_ptr->block_length = 0;
        input += head_length;
    }
    for (sz_size_t processed = 0; processed < body_length; processed += 64, input += 64)
        sz_sha256_process_block_lasx_(hash, input);
    for (sz_size_t byte_index = 0; byte_index < tail_length; ++byte_index)
        state_ptr->block[byte_index] = input[byte_index];
    state_ptr->block_length = tail_length;

    state_ptr->hash[0] = hash[0], state_ptr->hash[1] = hash[1], state_ptr->hash[2] = hash[2],
    state_ptr->hash[3] = hash[3], state_ptr->hash[4] = hash[4], state_ptr->hash[5] = hash[5],
    state_ptr->hash[6] = hash[6], state_ptr->hash[7] = hash[7];
}

SZ_API_COMPTIME void sz_sha256_state_digest_lasx(sz_sha256_state_t const *state_ptr, sz_u8_t digest[sz_at_least_(32)]) {
    sz_sha256_state_t state = *state_ptr;

    // Append the '1' bit (0x80), then pad with zeros, processing an extra block if the length doesn't fit.
    state.block[state.block_length++] = 0x80;
    if (state.block_length > 56) {
        for (sz_size_t byte_index = state.block_length; byte_index < 64; ++byte_index) state.block[byte_index] = 0;
        sz_sha256_process_block_lasx_(state.hash, state.block);
        state.block_length = 0;
    }
    for (sz_size_t byte_index = state.block_length; byte_index < 56; ++byte_index) state.block[byte_index] = 0;
    state.block_length = 56;

    // Append the message length in bits as a 64-bit big-endian integer, then process the final block.
    sz_u64_t bit_length = state.total_length * 8;
    state.block[56] = (sz_u8_t)(bit_length >> 56);
    state.block[57] = (sz_u8_t)(bit_length >> 48);
    state.block[58] = (sz_u8_t)(bit_length >> 40);
    state.block[59] = (sz_u8_t)(bit_length >> 32);
    state.block[60] = (sz_u8_t)(bit_length >> 24);
    state.block[61] = (sz_u8_t)(bit_length >> 16);
    state.block[62] = (sz_u8_t)(bit_length >> 8);
    state.block[63] = (sz_u8_t)(bit_length >> 0);
    sz_sha256_process_block_lasx_(state.hash, state.block);

    // Emit the digest big-endian.
    for (sz_size_t word_index = 0; word_index < 8; ++word_index) {
        digest[word_index * 4 + 0] = (sz_u8_t)(state.hash[word_index] >> 24);
        digest[word_index * 4 + 1] = (sz_u8_t)(state.hash[word_index] >> 16);
        digest[word_index * 4 + 2] = (sz_u8_t)(state.hash[word_index] >> 8);
        digest[word_index * 4 + 3] = (sz_u8_t)(state.hash[word_index] >> 0);
    }
}

#endif // SZ_USE_LASX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_LASX_H_
