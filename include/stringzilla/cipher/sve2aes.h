/**
 *  @brief Arm SVE2-AES backend for AES-256 in counter and Galois/counter modes.
 *  @file include/stringzilla/cipher/sve2aes.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/cipher.h
 *
 *  The round instructions treat every 128-bit segment of a scalable vector as an independent block, so
 *  one `AESE` covers as many blocks as the vector happens to hold and counter mode only has to spread
 *  consecutive block indices across the segments. That width is a runtime property, which is why no
 *  kernel here is written against a fixed lane count: each one reads `svcntb` and derives its step from
 *  it, and the single compile-time constant is the sixteen-byte block. Round keys reach every segment
 *  through a quadword broadcast load, one instruction whatever the width.
 *
 *  The fourteen rounds themselves are written out rather than driven by a loop over the round index,
 *  which is the one thing here that is fixed at compile time. A round loop leaves the unrolling to the
 *  optimizer, which only does it reliably at `-O3` — GCC 15 emitted 85 AES round instructions across the
 *  Arm backends at `-O2` against 530 at `-O3`. Since `SZ_API_COMPTIME` kernels are `static inline` and
 *  compile into whichever translation unit consumes them, a looped kernel makes its own throughput a
 *  property of the caller's build flags. See `cipher/icelake.h`, where the same change was worth 2.1x to
 *  2.9x at `-O2` and nothing at all at `-O3`.
 */
#ifndef STRINGZILLA_CIPHER_SVE2AES_H_
#define STRINGZILLA_CIPHER_SVE2AES_H_

#include "stringzilla/types.h"
#include "stringzilla/cipher/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SVE2AES
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve+sve2+sve2-aes"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve+sve2+sve2-aes")
#endif

#pragma region Key Schedule

/**
 *  @brief Substitutes every byte of a schedule word through the round instruction's substitution box.
 *  @param word The schedule word.
 *  @return The substituted word.
 *
 *  `AESE` applies `AddRoundKey`, `SubBytes` and `ShiftRows` in that order, so a zero round key leaves a
 *  substitution followed by a row shift. Broadcasting the word into all four columns makes that row shift
 *  a permutation among identical columns, which leaves every column holding the substituted word, and
 *  every segment of the vector holding the same answer whatever the width.
 */
SZ_HELPER_INLINE sz_u32_t sz_aes256_word_substitute_sve2aes_(sz_u32_t word) {
    sz_u128_vec_t substituted_vec;
    svuint8_t const broadcast_u8x = svreinterpret_u8_u32(svdup_n_u32(word));
    svst1_u8(svptrue_pat_b8(SV_VL16), &substituted_vec.u8s[0], svaese_u8(broadcast_u8x, svdup_n_u8(0)));
    return substituted_vec.u32s[0];
}

SZ_API_COMPTIME void sz_aes256_key_init_sve2aes(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    static sz_u8_t const round_constants[7] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40};
    sz_size_t word_index = 0;
    for (; word_index != 8; ++word_index)
        key->round_keys[word_index] = sz_aes256_word_pack_serial_(secret + word_index * 4);
    for (; word_index != SZ_AES256_ROUND_KEYS; ++word_index) {
        sz_u32_t carried = key->round_keys[word_index - 1];
        if ((word_index & 7) == 0) {
            carried = sz_aes256_word_substitute_sve2aes_(sz_aes256_word_rotate_serial_(carried));
            carried ^= (sz_u32_t)round_constants[(word_index >> 3) - 1];
        }
        else if ((word_index & 7) == 4) { carried = sz_aes256_word_substitute_sve2aes_(carried); }
        key->round_keys[word_index] = key->round_keys[word_index - 8] ^ carried;
    }
}

#pragma endregion // Key Schedule

#pragma region Block Encryption

/**
 *  @brief Encrypts one block per 128-bit segment with the expanded schedule.
 *  @param key The expanded schedule.
 *  @param blocks_u8x One plaintext block per segment.
 *  @return One ciphertext block per segment.
 *
 *  Arm folds `AddRoundKey` into the front of `AESE` rather than the end of the round, so the schedule is
 *  consumed one round early and the fifteenth round key is exclusive-ored on afterwards. Thirteen rounds
 *  carry a column mix, the fourteenth does not.
 */
SZ_HELPER_INLINE svuint8_t sz_aes256_blocks_encrypt_sve2aes_(sz_aes256_key_t const *key, svuint8_t blocks_u8x) {
    svbool_t const all_b8x = svptrue_b8();
    sz_u8_t const *schedule = (sz_u8_t const *)&key->round_keys[0];
    // The thirteen fused rounds are written out rather than looped, because a loop leaves the
    // unrolling to the optimizer and the optimizer only does it at `-O3`.
    blocks_u8x = svaesmc_u8(svaese_u8(blocks_u8x, svld1rq_u8(all_b8x, schedule + 0 * SZ_AES_BLOCK_LENGTH)));
    blocks_u8x = svaesmc_u8(svaese_u8(blocks_u8x, svld1rq_u8(all_b8x, schedule + 1 * SZ_AES_BLOCK_LENGTH)));
    blocks_u8x = svaesmc_u8(svaese_u8(blocks_u8x, svld1rq_u8(all_b8x, schedule + 2 * SZ_AES_BLOCK_LENGTH)));
    blocks_u8x = svaesmc_u8(svaese_u8(blocks_u8x, svld1rq_u8(all_b8x, schedule + 3 * SZ_AES_BLOCK_LENGTH)));
    blocks_u8x = svaesmc_u8(svaese_u8(blocks_u8x, svld1rq_u8(all_b8x, schedule + 4 * SZ_AES_BLOCK_LENGTH)));
    blocks_u8x = svaesmc_u8(svaese_u8(blocks_u8x, svld1rq_u8(all_b8x, schedule + 5 * SZ_AES_BLOCK_LENGTH)));
    blocks_u8x = svaesmc_u8(svaese_u8(blocks_u8x, svld1rq_u8(all_b8x, schedule + 6 * SZ_AES_BLOCK_LENGTH)));
    blocks_u8x = svaesmc_u8(svaese_u8(blocks_u8x, svld1rq_u8(all_b8x, schedule + 7 * SZ_AES_BLOCK_LENGTH)));
    blocks_u8x = svaesmc_u8(svaese_u8(blocks_u8x, svld1rq_u8(all_b8x, schedule + 8 * SZ_AES_BLOCK_LENGTH)));
    blocks_u8x = svaesmc_u8(svaese_u8(blocks_u8x, svld1rq_u8(all_b8x, schedule + 9 * SZ_AES_BLOCK_LENGTH)));
    blocks_u8x = svaesmc_u8(svaese_u8(blocks_u8x, svld1rq_u8(all_b8x, schedule + 10 * SZ_AES_BLOCK_LENGTH)));
    blocks_u8x = svaesmc_u8(svaese_u8(blocks_u8x, svld1rq_u8(all_b8x, schedule + 11 * SZ_AES_BLOCK_LENGTH)));
    blocks_u8x = svaesmc_u8(svaese_u8(blocks_u8x, svld1rq_u8(all_b8x, schedule + 12 * SZ_AES_BLOCK_LENGTH)));
    blocks_u8x = svaese_u8(blocks_u8x, svld1rq_u8(all_b8x, schedule + 13 * SZ_AES_BLOCK_LENGTH));
    return sveor_u8_x(all_b8x, blocks_u8x, svld1rq_u8(all_b8x, schedule + 14 * SZ_AES_BLOCK_LENGTH));
}

/**
 *  @brief Moves one 128-bit segment of a vector into its leading segment.
 *  @param value_u8x The vector to read from.
 *  @param segment_index Which segment to bring forward.
 *  @return A vector whose leading segment holds that one, with the rest left undefined.
 */
SZ_HELPER_INLINE svuint8_t sz_aes256_segment_at_sve2aes_(svuint8_t value_u8x, sz_size_t segment_index) {
    svbool_t const all_b64x = svptrue_b64();
    svuint64_t const source_u64x = svadd_n_u64_x(all_b64x, svindex_u64(0, 1), (sz_u64_t)(segment_index * 2));
    return svreinterpret_u8_u64(svtbl_u64(svreinterpret_u64_u8(value_u8x), source_u64x));
}

#pragma endregion // Block Encryption

#pragma region Counter Mode

/**
 *  @brief Broadcasts one counter block, the nonce then a big-endian index word, into every segment.
 *  @param nonce The twelve nonce bytes.
 *  @param block_index The block index the whole vector starts from.
 *  @return The same counter block in every segment.
 */
SZ_HELPER_INLINE svuint8_t sz_aes256_counter_broadcast_sve2aes_(sz_u8_t const *nonce, sz_u32_t block_index) {
    sz_u128_vec_t block_vec;
    sz_size_t byte_index;
    for (byte_index = 0; byte_index != 12; ++byte_index) block_vec.u8s[byte_index] = nonce[byte_index];
    block_vec.u8s[12] = (sz_u8_t)(block_index >> 24);
    block_vec.u8s[13] = (sz_u8_t)(block_index >> 16);
    block_vec.u8s[14] = (sz_u8_t)(block_index >> 8);
    block_vec.u8s[15] = (sz_u8_t)(block_index >> 0);
    return svld1rq_u8(svptrue_b8(), &block_vec.u8s[0]);
}

/**
 *  @brief Builds the increment that adds the same step to the index word of every counter block.
 *  @param step The number of blocks to advance by.
 *  @return An increment word per lane, zero everywhere but the trailing lane of each segment.
 */
SZ_HELPER_INLINE svuint32_t sz_aes256_counter_step_sve2aes_(sz_u32_t step) {
    svbool_t const all_b32x = svptrue_b32();
    svuint32_t const lane_u32x = svindex_u32(0, 1);
    svbool_t const trailing_b32x = svcmpeq_n_u32(all_b32x, svand_n_u32_x(all_b32x, lane_u32x, 3), 3);
    return svsel_u32(trailing_b32x, svdup_n_u32(step), svdup_n_u32(0));
}

/**
 *  @brief Builds the increment that turns a broadcast counter block into consecutive block indices.
 *  @return An increment word per lane, the segment's own ordinal in its trailing lane and zero elsewhere.
 */
SZ_HELPER_INLINE svuint32_t sz_aes256_counter_spread_sve2aes_(void) {
    svbool_t const all_b32x = svptrue_b32();
    svuint32_t const lane_u32x = svindex_u32(0, 1);
    svbool_t const trailing_b32x = svcmpeq_n_u32(all_b32x, svand_n_u32_x(all_b32x, lane_u32x, 3), 3);
    return svsel_u32(trailing_b32x, svlsr_n_u32_x(all_b32x, lane_u32x, 2), svdup_n_u32(0));
}

/**
 *  @brief Adds a per-lane increment to the trailing index word of every counter block.
 *  @param counter_u8x One counter block per segment.
 *  @param increment_u32x An increment word per lane, zero outside the trailing lane of each segment.
 *  @return The advanced counter blocks.
 *
 *  The index is stored big-endian while the addition is not, so the words are byte reversed, added and
 *  reversed back. Lanes holding nonce bytes receive a zero increment, which makes the pair of reversals
 *  an identity there, and the trailing lane wraps modulo `2^32` without carrying into the nonce.
 */
SZ_HELPER_INLINE svuint8_t sz_aes256_counter_advance_sve2aes_(svuint8_t counter_u8x, svuint32_t increment_u32x) {
    svbool_t const all_b32x = svptrue_b32();
    svuint32_t const native_u32x = svrevb_u32_x(all_b32x, svreinterpret_u32_u8(counter_u8x));
    return svreinterpret_u8_u32(svrevb_u32_x(all_b32x, svadd_u32_x(all_b32x, native_u32x, increment_u32x)));
}

SZ_API_COMPTIME void sz_aes256_ctr_xor_sve2aes(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                               sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length,
                                               sz_ptr_t output) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    svbool_t const all_b8x = svptrue_b8();
    sz_size_t const vector_length = svcntb();
    sz_size_t const blocks_per_vector = vector_length / SZ_AES_BLOCK_LENGTH;
    svuint32_t const stride_u32x = sz_aes256_counter_step_sve2aes_((sz_u32_t)blocks_per_vector);
    sz_u32_t const block_index = (sz_u32_t)(byte_offset / SZ_AES_BLOCK_LENGTH);
    sz_size_t within_block = (sz_size_t)(byte_offset % SZ_AES_BLOCK_LENGTH);
    sz_size_t produced = 0;
    svuint8_t counter_u8x;

    if (length == 0) return;
    counter_u8x = sz_aes256_counter_advance_sve2aes_(sz_aes256_counter_broadcast_sve2aes_(nonce, block_index),
                                                     sz_aes256_counter_spread_sve2aes_());

    //  A start that is not block aligned generates its first block whole and discards the leading bytes,
    //  after which the buffer position and the stream's block boundaries line up again.
    if (within_block != 0) {
        sz_u128_vec_t keystream_vec;
        svst1_u8(svptrue_pat_b8(SV_VL16), &keystream_vec.u8s[0], sz_aes256_blocks_encrypt_sve2aes_(key, counter_u8x));
        for (; within_block != SZ_AES_BLOCK_LENGTH && produced != length; ++within_block, ++produced)
            output_bytes[produced] = (sz_u8_t)(input_bytes[produced] ^ keystream_vec.u8s[within_block]);
        counter_u8x = sz_aes256_counter_advance_sve2aes_(counter_u8x, sz_aes256_counter_step_sve2aes_(1));
    }

    while (produced < length) {
        svbool_t const active_b8x = svwhilelt_b8_u64((sz_u64_t)produced, (sz_u64_t)length);
        svuint8_t const keystream_u8x = sz_aes256_blocks_encrypt_sve2aes_(key, counter_u8x);
        svuint8_t const original_u8x = svld1_u8(active_b8x, input_bytes + produced);
        svst1_u8(active_b8x, output_bytes + produced, sveor_u8_x(all_b8x, original_u8x, keystream_u8x));
        counter_u8x = sz_aes256_counter_advance_sve2aes_(counter_u8x, stride_u32x);
        produced += vector_length;
    }
}

#pragma endregion // Counter Mode

#pragma region Galois Hashing

/**
 *  @brief Moves a hash block between the tag's bit order and the multiplier's.
 *  @param block_u8x The block in either order.
 *  @return The same block in the other order.
 *
 *  The tag reads a field element's lowest coefficient from the leading bit of the leading byte, while a
 *  carry-less multiply reads it from the lowest bit of the lowest byte. The bytes already ascend in both,
 *  so the whole difference is a bit reversal inside each byte, which is its own inverse and leaves the
 *  product needing no further shift.
 */
SZ_HELPER_INLINE svuint8_t sz_ghash_reflect_sve2aes_(svuint8_t block_u8x) {
    return svrbit_u8_x(svptrue_b8(), block_u8x);
}

/** @brief Loads one hash block from the tag's bit order into the multiplier's, leaving the rest zero. */
SZ_HELPER_INLINE svuint8_t sz_ghash_load_sve2aes_(sz_u8_t const *block) {
    return sz_ghash_reflect_sve2aes_(svld1_u8(svptrue_pat_b8(SV_VL16), block));
}

/** @brief Stores the leading segment of a reflected value back in the tag's bit order. */
SZ_HELPER_INLINE void sz_ghash_store_sve2aes_(svuint8_t value_u8x, sz_u8_t *block) {
    svst1_u8(svptrue_pat_b8(SV_VL16), block, sz_ghash_reflect_sve2aes_(value_u8x));
}

/** @brief Exchanges the two halves of every 128-bit segment. */
SZ_HELPER_INLINE svuint64_t sz_ghash_swap_halves_sve2aes_(svuint64_t value_u64x) {
    svbool_t const all_b64x = svptrue_b64();
    return svtbl_u64(value_u64x, sveor_n_u64_x(all_b64x, svindex_u64(0, 1), 1));
}

/** @brief The predicate selecting the low half of every 128-bit segment. */
SZ_HELPER_INLINE svbool_t sz_ghash_low_halves_sve2aes_(void) {
    svbool_t const all_b64x = svptrue_b64();
    return svcmpeq_n_u64(all_b64x, svand_n_u64_x(all_b64x, svindex_u64(0, 1), 1), 0);
}

/**
 *  @brief Sums every 128-bit segment of a vector into its leading segment.
 *  @param value_u64x The per-segment partial sums.
 *  @param lane_count The number of 64-bit lanes the vector holds.
 *  @return A vector whose leading segment holds the sum of them all.
 *
 *  A table lookup rather than an extract, because the distance halves each round and the vector length is
 *  not a compile-time constant. Indices that run off the end return zero, which is the identity here, so
 *  the last round needs no separate guard.
 */
SZ_HELPER_INLINE svuint64_t sz_ghash_fold_segments_sve2aes_(svuint64_t value_u64x, sz_size_t lane_count) {
    svbool_t const all_b64x = svptrue_b64();
    svuint64_t const lane_u64x = svindex_u64(0, 1);
    sz_size_t lane_distance;
    for (lane_distance = 2; lane_distance < lane_count; lane_distance += lane_distance)
        value_u64x = sveor_u64_x(all_b64x, value_u64x,
                                 svtbl_u64(value_u64x, svadd_n_u64_x(all_b64x, lane_u64x, (sz_u64_t)lane_distance)));
    return value_u64x;
}

/**
 *  @brief Accumulates the 256-bit carry-less product of two reflected operands into a running triple.
 *  @param first_u64x One reflected operand per segment.
 *  @param second_u64x The other reflected operand per segment.
 *  @param low_u64x Accumulates the product of the two low halves.
 *  @param middle_u64x Accumulates both cross products.
 *  @param high_u64x Accumulates the product of the two high halves.
 *
 *  The bottom and top forms of the paired multiply reach the two halves of a segment, and Karatsuba's
 *  identity recovers the cross terms from a third product of the two folded operands, so three multiplies
 *  cover a schoolbook four. Keeping the triple unreduced is what lets a whole group share one reduction:
 *  the field is linear, so partial products may be summed before folding once.
 */
SZ_HELPER_INLINE void sz_ghash_accumulate_sve2aes_(svuint64_t first_u64x, svuint64_t second_u64x, svuint64_t *low_u64x,
                                                   svuint64_t *middle_u64x, svuint64_t *high_u64x) {
    svbool_t const all_b64x = svptrue_b64();
    svuint64_t const first_folded_u64x = sveor_u64_x(all_b64x, first_u64x, sz_ghash_swap_halves_sve2aes_(first_u64x));
    svuint64_t const second_folded_u64x = sveor_u64_x(all_b64x, second_u64x,
                                                      sz_ghash_swap_halves_sve2aes_(second_u64x));
    svuint64_t const low_product_u64x = svpmullb_pair_u64(first_u64x, second_u64x);
    svuint64_t const high_product_u64x = svpmullt_pair_u64(first_u64x, second_u64x);
    svuint64_t const folded_product_u64x = svpmullb_pair_u64(first_folded_u64x, second_folded_u64x);
    svuint64_t const crossed_u64x = sveor_u64_x(all_b64x, folded_product_u64x,
                                                sveor_u64_x(all_b64x, low_product_u64x, high_product_u64x));
    *low_u64x = sveor_u64_x(all_b64x, *low_u64x, low_product_u64x);
    *middle_u64x = sveor_u64_x(all_b64x, *middle_u64x, crossed_u64x);
    *high_u64x = sveor_u64_x(all_b64x, *high_u64x, high_product_u64x);
}

/**
 *  @brief Sums an accumulated triple across the segments and folds it into one reflected block.
 *  @param low_u64x The accumulated low halves.
 *  @param middle_u64x The accumulated cross products.
 *  @param high_u64x The accumulated high halves.
 *  @param lane_count The number of 64-bit lanes the vector holds.
 *  @return The reduced product in the leading segment, with the rest left undefined.
 *
 *  The cross products straddle the halves, so they are split and merged into the two 128-bit words first.
 *  What remains is `x^128` congruent to `x^7 + x^2 + x + 1`, applied one word at a time: multiplying the
 *  top word by that constant leaves seven bits that still sit above the modulus, which are carried into
 *  the third word before the second multiply retires both. Two multiplies rather than a shift chain,
 *  because the same paired instruction the schoolbook product uses reaches each word directly.
 */
SZ_HELPER_INLINE svuint64_t sz_ghash_reduce_sve2aes_(svuint64_t low_u64x, svuint64_t middle_u64x, svuint64_t high_u64x,
                                                     sz_size_t lane_count) {
    svbool_t const all_b64x = svptrue_b64();
    svbool_t const low_halves_b64x = sz_ghash_low_halves_sve2aes_();
    svuint64_t const zeros_u64x = svdup_n_u64(0);
    svuint64_t const modulus_u64x = svdup_n_u64(0x87ull);
    svuint64_t const middle_swapped_u64x = sz_ghash_swap_halves_sve2aes_(middle_u64x);
    svuint64_t product_low_u64x = sveor_u64_x(all_b64x, low_u64x,
                                              svsel_u64(low_halves_b64x, zeros_u64x, middle_swapped_u64x));
    svuint64_t product_high_u64x = sveor_u64_x(all_b64x, high_u64x,
                                               svsel_u64(low_halves_b64x, middle_swapped_u64x, zeros_u64x));
    svuint64_t first_fold_u64x, second_fold_u64x;

    product_low_u64x = sz_ghash_fold_segments_sve2aes_(product_low_u64x, lane_count);
    product_high_u64x = sz_ghash_fold_segments_sve2aes_(product_high_u64x, lane_count);

    first_fold_u64x = sz_ghash_swap_halves_sve2aes_(svpmullt_pair_u64(product_high_u64x, modulus_u64x));
    second_fold_u64x = svpmullb_pair_u64(sveor_u64_x(all_b64x, product_high_u64x, first_fold_u64x), modulus_u64x);
    return sveor_u64_x(all_b64x, sveor_u64_x(all_b64x, product_low_u64x, second_fold_u64x),
                       svsel_u64(low_halves_b64x, zeros_u64x, first_fold_u64x));
}

/**
 *  @brief Multiplies one reflected block by one reflected subkey.
 *  @param accumulator_u8x The running hash in the leading segment, zero elsewhere.
 *  @param subkey_u8x The reflected subkey in the leading segment, zero elsewhere.
 *  @return The reduced product in the leading segment, zero elsewhere.
 */
SZ_HELPER_INLINE svuint8_t sz_ghash_multiply_sve2aes_(svuint8_t accumulator_u8x, svuint8_t subkey_u8x) {
    svuint64_t low_u64x = svdup_n_u64(0), middle_u64x = svdup_n_u64(0), high_u64x = svdup_n_u64(0);
    sz_ghash_accumulate_sve2aes_(svreinterpret_u64_u8(accumulator_u8x), svreinterpret_u64_u8(subkey_u8x), &low_u64x,
                                 &middle_u64x, &high_u64x);
    return svreinterpret_u8_u64(sz_ghash_reduce_sve2aes_(low_u64x, middle_u64x, high_u64x, 2));
}

/** @brief Absorbs one reflected block into the running hash. */
SZ_HELPER_INLINE svuint8_t sz_ghash_absorb_sve2aes_(svuint8_t accumulator_u8x, svuint8_t block_u8x,
                                                    svuint8_t subkey_u8x) {
    return sz_ghash_multiply_sve2aes_(sveor_u8_x(svptrue_b8(), accumulator_u8x, block_u8x), subkey_u8x);
}

SZ_API_COMPTIME void sz_aes256_gcm_key_init_sve2aes(sz_aes256_gcm_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    sz_u128_vec_t subkey_vec;
    svuint8_t subkey_u8x, power_u8x;
    sz_size_t power_index;

    sz_aes256_key_init_sve2aes(&key->block, secret);
    svst1_u8(svptrue_pat_b8(SV_VL16), &subkey_vec.u8s[0],
             sz_aes256_blocks_encrypt_sve2aes_(&key->block, svdup_n_u8(0)));

    subkey_u8x = sz_ghash_load_sve2aes_(&subkey_vec.u8s[0]);
    power_u8x = subkey_u8x;
    sz_ghash_store_sve2aes_(power_u8x, &key->powers[0]);
    for (power_index = 1; power_index != 8; ++power_index) {
        power_u8x = sz_ghash_multiply_sve2aes_(power_u8x, subkey_u8x);
        sz_ghash_store_sve2aes_(power_u8x, &key->powers[power_index * SZ_AES_BLOCK_LENGTH]);
    }
}

/*  A hash group spans one block per segment, so a wider vector retires more blocks between reductions.
 *  Eight is the ceiling whatever the width, because the key carries only `H^1` through `H^8` and a group
 *  of `n` blocks needs `H^n`; a hypothetical 2048-bit vector therefore leaves half its segments idle in
 *  the hash while counter mode still uses all of them. */

/** @brief Blocks one hash group covers: one per segment, capped by the eight precomputed subkey powers. */
SZ_HELPER_INLINE sz_size_t sz_ghash_group_blocks_sve2aes_(void) {
    sz_size_t const segment_count = svcntb() / SZ_AES_BLOCK_LENGTH;
    return segment_count < 8 ? segment_count : 8;
}

/**
 *  @brief Loads the leading subkey powers descending, so segment `j` pairs with block `j` of a group.
 *  @param powers The eight subkey powers in the tag's bit order, ascending.
 *  @param group_blocks Blocks in a group, which is how many powers are wanted.
 *  @return The reflected powers `H^n` down to `H^1`, one per segment, zero beyond the group.
 *
 *  The powers ascend in memory and a group wants them descending, so the segments are mirrored by a table
 *  lookup. Ordinals past the group underflow into indices no table can hold, and those lanes read back as
 *  zero, which is exactly the operand a segment carrying no block should meet.
 */
SZ_HELPER_INLINE svuint8_t sz_ghash_descending_powers_sve2aes_(sz_u8_t const *powers, sz_size_t group_blocks) {
    svbool_t const all_b64x = svptrue_b64();
    svbool_t const group_b8x = svwhilelt_b8_u64(0, (sz_u64_t)(group_blocks * SZ_AES_BLOCK_LENGTH));
    svuint64_t const lane_u64x = svindex_u64(0, 1);
    svuint64_t const mirrored_u64x = svsub_u64_x(all_b64x, svdup_n_u64((sz_u64_t)(group_blocks - 1)),
                                                 svlsr_n_u64_x(all_b64x, lane_u64x, 1));
    svuint64_t const source_u64x = svadd_u64_x(all_b64x, svlsl_n_u64_x(all_b64x, mirrored_u64x, 1),
                                               svand_n_u64_x(all_b64x, lane_u64x, 1));
    svuint64_t const ascending_u64x = svreinterpret_u64_u8(svld1_u8(group_b8x, powers));
    return sz_ghash_reflect_sve2aes_(svreinterpret_u8_u64(svtbl_u64(ascending_u64x, source_u64x)));
}

/**
 *  @brief Absorbs a whole group of reflected blocks into the running hash with one field reduction.
 *  @param accumulator_u8x The running hash in the leading segment, zero elsewhere.
 *  @param blocks_u8x One reflected block per segment, in the order they arrived, zero beyond the group.
 *  @param powers_u8x The reflected powers `H^n` down to `H^1`, one per segment.
 *  @param lane_count The number of 64-bit lanes the vector holds.
 *  @return The updated running hash in the leading segment, zero elsewhere.
 *
 *  A group of `n` absorbed blocks expands to `(Y ^ X1) H^n ^ X2 H^(n-1) ^ ... ^ Xn H`, which needs `n`
 *  multiplies either way but only one reduction instead of `n`, and every segment performs its multiply
 *  under the same instruction.
 */
SZ_HELPER_INLINE svuint8_t sz_ghash_absorb_group_sve2aes_(svuint8_t accumulator_u8x, svuint8_t blocks_u8x,
                                                          svuint8_t powers_u8x, sz_size_t lane_count) {
    svbool_t const all_b8x = svptrue_b8();
    svuint64_t low_u64x = svdup_n_u64(0), middle_u64x = svdup_n_u64(0), high_u64x = svdup_n_u64(0);
    svuint8_t reduced_u8x;
    sz_ghash_accumulate_sve2aes_(svreinterpret_u64_u8(sveor_u8_x(all_b8x, accumulator_u8x, blocks_u8x)),
                                 svreinterpret_u64_u8(powers_u8x), &low_u64x, &middle_u64x, &high_u64x);
    reduced_u8x = svreinterpret_u8_u64(sz_ghash_reduce_sve2aes_(low_u64x, middle_u64x, high_u64x, lane_count));
    return svsel_u8(svptrue_pat_b8(SV_VL16), reduced_u8x, svdup_n_u8(0));
}

#pragma endregion // Galois Hashing

#pragma region Streaming Interface

/**
 *  @brief Overwrites a finished state so the key schedule it embeds does not outlive the call.
 *
 *  The size is known at compile time but the vector length is not, so the fill runs one predicated store
 *  per vector and the trailing predicate covers a size that is a multiple of no vector width. Writes are
 *  ordinary stores followed by one barrier: routing them through a `volatile` view instead would forbid
 *  vectorization and cost 472 single-byte stores on every one-shot call.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_state_scrub_sve2aes_(sz_aes256_gcm_state_t *state) {
    sz_u8_t *const bytes = (sz_u8_t *)state;
    svuint8_t const zeros_u8x = svdup_n_u8(0);
    sz_size_t const vector_length = svcntb();
    sz_size_t byte_index;
    for (byte_index = 0; byte_index < sizeof(*state); byte_index += vector_length)
        svst1_u8(svwhilelt_b8_u64((sz_u64_t)byte_index, (sz_u64_t)sizeof(*state)), bytes + byte_index, zeros_u8x);
    sz_keep_alive_(state);
}

/** @brief Prepares the payload both directions share: counter block, tag mask and empty carries. */
SZ_HELPER_INLINE void sz_aes256_gcm_begin_sve2aes_(sz_aes256_gcm_state_t *state, sz_aes256_gcm_key_t const *key,
                                                   sz_u8_t const nonce[sz_at_least_(12)]) {
    svbool_t const first_b8x = svptrue_pat_b8(SV_VL16);
    sz_u128_vec_t initial_vec;
    sz_size_t byte_index;

    state->key = *key;

    //  With a twelve-byte nonce the initial counter block is the nonce followed by a one, and the value
    //  encrypted under it masks the finished hash. Data blocks start one past it.
    for (byte_index = 0; byte_index != 12; ++byte_index) initial_vec.u8s[byte_index] = nonce[byte_index];
    initial_vec.u8s[12] = 0, initial_vec.u8s[13] = 0, initial_vec.u8s[14] = 0, initial_vec.u8s[15] = 1;
    svst1_u8(first_b8x, state->tag_mask,
             sz_aes256_blocks_encrypt_sve2aes_(&state->key.block, svld1rq_u8(svptrue_b8(), &initial_vec.u8s[0])));
    svst1_u8(first_b8x, state->counter, svld1_u8(first_b8x, &initial_vec.u8s[0]));
    svst1_u8(first_b8x, state->accumulator, svdup_n_u8(0));
    svst1_u8(first_b8x, state->partial, svdup_n_u8(0));
    svst1_u8(first_b8x, state->keystream, svdup_n_u8(0));

    state->associated_length = 0;
    state->text_length = 0;
    state->buffered = 0;
    state->keystream_used = SZ_AES_BLOCK_LENGTH; // ? Forces the first message byte to derive a fresh block
}

/** @brief Absorbs associated data into the payload both directions share. */
SZ_HELPER_AUTO void sz_aes256_gcm_associate_sve2aes_(sz_aes256_gcm_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_size_t const lane_count = svcntb() / 8;
    sz_size_t const group_blocks = sz_ghash_group_blocks_sve2aes_();
    svuint8_t const subkey_u8x = sz_ghash_load_sve2aes_(state->key.powers);
    svuint8_t powers_u8x = sz_ghash_descending_powers_sve2aes_(state->key.powers, group_blocks);
    svbool_t pass_b8x = svwhilelt_b8_u64(0, (sz_u64_t)(group_blocks * SZ_AES_BLOCK_LENGTH));
    sz_size_t pass_blocks = group_blocks;
    svuint8_t accumulator_u8x;
    sz_size_t consumed = 0, partial_position;

    if (length == 0) return;
    accumulator_u8x = sz_ghash_load_sve2aes_(state->accumulator);
    state->associated_length += length;

    //  Associated data is hashed but never encrypted, so a partial block is completed in place. The
    //  filling loop stops at the block width rather than at a precomputed count, which keeps the write
    //  index provably inside the block for a reader and for the compiler alike.
    if (state->buffered != 0) {
        for (partial_position = state->buffered; partial_position < SZ_AES_BLOCK_LENGTH && consumed != length;
             ++partial_position, ++consumed)
            state->partial[partial_position] = input_bytes[consumed];
        state->buffered = (sz_u8_t)partial_position;
        if (state->buffered == SZ_AES_BLOCK_LENGTH) {
            accumulator_u8x = sz_ghash_absorb_sve2aes_(accumulator_u8x, sz_ghash_load_sve2aes_(state->partial),
                                                       subkey_u8x);
            state->buffered = 0;
        }
    }

    //  Every whole block goes through the group pass, which narrows once when fewer blocks are left than
    //  a group holds, so a remainder never falls back to one multiply and one reduction per block.
    while (consumed + SZ_AES_BLOCK_LENGTH <= length) {
        sz_size_t const available_blocks = (length - consumed) / SZ_AES_BLOCK_LENGTH;
        if (available_blocks < pass_blocks) {
            pass_blocks = available_blocks;
            powers_u8x = sz_ghash_descending_powers_sve2aes_(state->key.powers, pass_blocks);
            pass_b8x = svwhilelt_b8_u64(0, (sz_u64_t)(pass_blocks * SZ_AES_BLOCK_LENGTH));
        }
        accumulator_u8x = sz_ghash_absorb_group_sve2aes_(
            accumulator_u8x, sz_ghash_reflect_sve2aes_(svld1_u8(pass_b8x, input_bytes + consumed)), powers_u8x,
            lane_count);
        consumed += pass_blocks * SZ_AES_BLOCK_LENGTH;
    }
    for (; consumed != length; ++consumed) state->partial[state->buffered++] = input_bytes[consumed];

    sz_ghash_store_sve2aes_(accumulator_u8x, state->accumulator);
}

/**
 *  @brief Spends what is left of the keystream block the state carries, one byte at a time.
 *  @param state The state, whose two sixteen-byte rhythms advance together here.
 *  @param input The bytes to transform.
 *  @param output Receives the transformed bytes.
 *  @param count Bytes to consume, never more than the keystream block has left.
 *  @param accumulator_u8x The running hash, reflected.
 *  @param subkey_u8x The reflected hash subkey.
 *  @param direction Which buffer the hash absorbs.
 *  @return The updated running hash, reflected.
 *
 *  Through the message the keystream offset and the hash offset are the same number, because every byte
 *  spends one of each, so a chunk that ends mid block leaves both mid block and this resumes both.
 */
SZ_HELPER_INLINE svuint8_t sz_aes256_gcm_spend_sve2aes_(sz_aes256_gcm_state_t *state, sz_u8_t const *input,
                                                        sz_u8_t *output, sz_size_t count, svuint8_t accumulator_u8x,
                                                        svuint8_t subkey_u8x, sz_aes256_gcm_direction_t direction) {
    sz_size_t byte_index;

    //  The hash always eats ciphertext, which is the output when encrypting and the input when
    //  decrypting, and the byte is captured before the store because a caller may pass one pointer.
    for (byte_index = 0; byte_index != count; ++byte_index) {
        sz_u8_t const plaintext_byte = input[byte_index];
        sz_u8_t const transformed = (sz_u8_t)(plaintext_byte ^ state->keystream[state->keystream_used]);
        sz_u8_t const ciphertext = direction == sz_aes256_gcm_decrypting_k ? plaintext_byte : transformed;
        output[byte_index] = transformed;
        state->partial[state->buffered] = ciphertext;
        ++state->buffered;
        ++state->keystream_used;
        if (state->buffered == SZ_AES_BLOCK_LENGTH) {
            accumulator_u8x = sz_ghash_absorb_sve2aes_(accumulator_u8x, sz_ghash_load_sve2aes_(state->partial),
                                                       subkey_u8x);
            state->buffered = 0;
        }
    }
    return accumulator_u8x;
}

/**
 *  @brief Transforms a chunk and absorbs its ciphertext, whichever side of the call that is.
 *  @param state The state.
 *  @param text The chunk to transform.
 *  @param length Bytes in the chunk.
 *  @param output Receives the transformed bytes.
 *  @param direction Which buffer the hash absorbs.
 *
 *  Three passes, because two sixteen-byte rhythms run underneath a caller's arbitrary chunk sizes and
 *  neither may restart at a chunk boundary: whatever the previous chunk left of its keystream block, then
 *  whole blocks a group at a time, then a trailing block that the next chunk will resume. The group pass
 *  leaves the block it finished on in the state's keystream, so a chunk that ends on a boundary is
 *  indistinguishable from one the byte-at-a-time path produced.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_transform_sve2aes_(sz_aes256_gcm_state_t *state, sz_cptr_t text, sz_size_t length,
                                                       sz_ptr_t output, sz_aes256_gcm_direction_t direction) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    svbool_t const all_b8x = svptrue_b8();
    svbool_t const first_b8x = svptrue_pat_b8(SV_VL16);
    sz_size_t const lane_count = svcntb() / 8;
    sz_size_t const group_blocks = sz_ghash_group_blocks_sve2aes_();
    svuint8_t const subkey_u8x = sz_ghash_load_sve2aes_(state->key.powers);
    svuint32_t const stride_u32x = sz_aes256_counter_step_sve2aes_((sz_u32_t)group_blocks);
    svuint8_t powers_u8x = sz_ghash_descending_powers_sve2aes_(state->key.powers, group_blocks);
    svbool_t pass_b8x = svwhilelt_b8_u64(0, (sz_u64_t)(group_blocks * SZ_AES_BLOCK_LENGTH));
    sz_size_t pass_blocks = group_blocks;
    svuint8_t accumulator_u8x, counter_u8x, last_keystream_u8x = svdup_n_u8(0);
    sz_u32_t block_index;
    sz_size_t produced = 0, pass_count = 0, byte_index;

    if (length == 0) return;
    accumulator_u8x = sz_ghash_load_sve2aes_(state->accumulator);

    //  Associated data ends the moment the first message byte arrives, and its tail needs padding.
    if (state->text_length == 0 && state->buffered != 0) {
        for (byte_index = state->buffered; byte_index != SZ_AES_BLOCK_LENGTH; ++byte_index)
            state->partial[byte_index] = 0;
        accumulator_u8x = sz_ghash_absorb_sve2aes_(accumulator_u8x, sz_ghash_load_sve2aes_(state->partial), subkey_u8x);
        state->buffered = 0;
    }

    block_index = ((sz_u32_t)state->counter[12] << 24) | ((sz_u32_t)state->counter[13] << 16) |
                  ((sz_u32_t)state->counter[14] << 8) | (sz_u32_t)state->counter[15];

    if (state->keystream_used != SZ_AES_BLOCK_LENGTH) {
        sz_size_t const available_bytes = SZ_AES_BLOCK_LENGTH - state->keystream_used;
        sz_size_t const taken_bytes = length < available_bytes ? length : available_bytes;
        accumulator_u8x = sz_aes256_gcm_spend_sve2aes_(state, input_bytes, output_bytes, taken_bytes, accumulator_u8x,
                                                       subkey_u8x, direction);
        produced = taken_bytes;
    }

    counter_u8x = sz_aes256_counter_advance_sve2aes_(
        sz_aes256_counter_broadcast_sve2aes_(state->counter, block_index + 1), sz_aes256_counter_spread_sve2aes_());

    //  Every whole block goes through the group pass, which narrows once when fewer blocks are left than
    //  a group holds, so a remainder never falls back to one multiply and one reduction per block.
    while (produced + SZ_AES_BLOCK_LENGTH <= length) {
        sz_size_t const available_blocks = (length - produced) / SZ_AES_BLOCK_LENGTH;
        svuint8_t original_u8x, transformed_u8x, absorbed_u8x, ciphertext_u8x, keystream_u8x;
        if (available_blocks < pass_blocks) {
            pass_blocks = available_blocks;
            powers_u8x = sz_ghash_descending_powers_sve2aes_(state->key.powers, pass_blocks);
            pass_b8x = svwhilelt_b8_u64(0, (sz_u64_t)(pass_blocks * SZ_AES_BLOCK_LENGTH));
        }
        keystream_u8x = sz_aes256_blocks_encrypt_sve2aes_(&state->key.block, counter_u8x);
        original_u8x = svld1_u8(pass_b8x, input_bytes + produced);
        transformed_u8x = sveor_u8_x(all_b8x, original_u8x, keystream_u8x);
        absorbed_u8x = direction == sz_aes256_gcm_decrypting_k ? original_u8x : transformed_u8x;
        ciphertext_u8x = svsel_u8(pass_b8x, absorbed_u8x, svdup_n_u8(0));
        svst1_u8(pass_b8x, output_bytes + produced, transformed_u8x);
        accumulator_u8x = sz_ghash_absorb_group_sve2aes_(accumulator_u8x, sz_ghash_reflect_sve2aes_(ciphertext_u8x),
                                                         powers_u8x, lane_count);
        last_keystream_u8x = sz_aes256_segment_at_sve2aes_(keystream_u8x, pass_blocks - 1);
        counter_u8x = sz_aes256_counter_advance_sve2aes_(counter_u8x, stride_u32x);
        produced += pass_blocks * SZ_AES_BLOCK_LENGTH;
        block_index += (sz_u32_t)pass_blocks;
        ++pass_count;
    }

    //  A chunk that ends on a block boundary must leave behind the keystream block it finished on, which
    //  the group pass parked in the segment its last block occupied.
    if (pass_count != 0) svst1_u8(first_b8x, state->keystream, last_keystream_u8x);

    if (produced != length) {
        block_index += 1;
        svst1_u8(first_b8x, state->keystream,
                 sz_aes256_blocks_encrypt_sve2aes_(&state->key.block,
                                                   sz_aes256_counter_broadcast_sve2aes_(state->counter, block_index)));
        state->keystream_used = 0;
        accumulator_u8x = sz_aes256_gcm_spend_sve2aes_(state, input_bytes + produced, output_bytes + produced,
                                                       length - produced, accumulator_u8x, subkey_u8x, direction);
    }

    state->text_length += length;
    sz_ghash_store_sve2aes_(accumulator_u8x, state->accumulator);
    state->counter[12] = (sz_u8_t)(block_index >> 24);
    state->counter[13] = (sz_u8_t)(block_index >> 16);
    state->counter[14] = (sz_u8_t)(block_index >> 8);
    state->counter[15] = (sz_u8_t)(block_index >> 0);
}

/**
 *  @brief Pads whatever the state still carries, absorbs the length block and unmasks the running hash.
 *  @param state The state, left untouched, so a caller may digest and keep streaming.
 *  @param tag Receives the sixteen tag bytes.
 */
SZ_HELPER_AUTO void sz_aes256_gcm_digest_sve2aes_(sz_aes256_gcm_state_t const *state, sz_u8_t tag[sz_at_least_(16)]) {
    svbool_t const all_b8x = svptrue_b8();
    svbool_t const first_b8x = svptrue_pat_b8(SV_VL16);
    svuint8_t const subkey_u8x = sz_ghash_load_sve2aes_(state->key.powers);
    svuint8_t accumulator_u8x = sz_ghash_load_sve2aes_(state->accumulator);
    sz_u128_vec_t padded_vec, lengths_vec;
    sz_size_t byte_index;

    //  Whatever is still pending gets padded here: an associated-data tail when the message was empty,
    //  or the message's own trailing ciphertext bytes otherwise.
    if (state->buffered != 0) {
        for (byte_index = 0; byte_index != state->buffered; ++byte_index)
            padded_vec.u8s[byte_index] = state->partial[byte_index];
        for (; byte_index != SZ_AES_BLOCK_LENGTH; ++byte_index) padded_vec.u8s[byte_index] = 0;
        accumulator_u8x = sz_ghash_absorb_sve2aes_(accumulator_u8x, sz_ghash_load_sve2aes_(&padded_vec.u8s[0]),
                                                   subkey_u8x);
    }

    sz_aes256_gcm_lengths_serial_(&lengths_vec, state->associated_length, state->text_length);
    accumulator_u8x = sz_ghash_absorb_sve2aes_(accumulator_u8x, sz_ghash_load_sve2aes_(&lengths_vec.u8s[0]),
                                               subkey_u8x);

    svst1_u8(first_b8x, tag,
             sveor_u8_x(all_b8x, sz_ghash_reflect_sve2aes_(accumulator_u8x), svld1_u8(first_b8x, state->tag_mask)));
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_init_sve2aes(sz_aes256_gcm_encryptor_t *encryptor,
                                                          sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_aes256_gcm_begin_sve2aes_(&encryptor->state, key, nonce);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_associate_sve2aes(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                               sz_size_t length) {
    sz_aes256_gcm_associate_sve2aes_(&encryptor->state, text, length);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_update_sve2aes(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                            sz_size_t length, sz_ptr_t output) {
    sz_aes256_gcm_transform_sve2aes_(&encryptor->state, text, length, output, sz_aes256_gcm_encrypting_k);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_digest_sve2aes(sz_aes256_gcm_encryptor_t const *encryptor,
                                                            sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_digest_sve2aes_(&encryptor->state, tag);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_init_sve2aes(sz_aes256_gcm_decryptor_t *decryptor,
                                                          sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_aes256_gcm_begin_sve2aes_(&decryptor->state, key, nonce);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_associate_sve2aes(sz_aes256_gcm_decryptor_t *decryptor, sz_cptr_t text,
                                                               sz_size_t length) {
    sz_aes256_gcm_associate_sve2aes_(&decryptor->state, text, length);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_update_unverified_sve2aes(sz_aes256_gcm_decryptor_t *decryptor,
                                                                       sz_cptr_t text, sz_size_t length,
                                                                       sz_ptr_t output) {
    sz_aes256_gcm_transform_sve2aes_(&decryptor->state, text, length, output, sz_aes256_gcm_decrypting_k);
}

SZ_API_COMPTIME sz_status_t sz_aes256_gcm_decryptor_verify_sve2aes(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                   sz_u8_t const tag[sz_at_least_(16)]) {
    sz_u128_vec_t expected_vec;
    sz_aes256_gcm_digest_sve2aes_(&decryptor->state, &expected_vec.u8s[0]);
    return sz_aes256_tag_equal_serial_(&expected_vec.u8s[0], tag) == sz_true_k ? sz_success_k
                                                                               : sz_authentication_failed_k;
}

#pragma endregion // Streaming Interface

#pragma region One Shot Interface

SZ_API_COMPTIME void sz_aes256_gcm_encrypt_sve2aes(sz_aes256_gcm_key_t const *key,
                                                   sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                   sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                   sz_ptr_t output, sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_encryptor_t encryptor;
    sz_aes256_gcm_encryptor_init_sve2aes(&encryptor, key, nonce);
    if (associated_length) sz_aes256_gcm_encryptor_associate_sve2aes(&encryptor, associated, associated_length);
    sz_aes256_gcm_encryptor_update_sve2aes(&encryptor, text, length, output);
    sz_aes256_gcm_encryptor_digest_sve2aes(&encryptor, tag);
    sz_aes256_gcm_state_scrub_sve2aes_(&encryptor.state);
}

SZ_API_COMPTIME sz_status_t sz_aes256_gcm_decrypt_sve2aes(sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                          sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                          sz_ptr_t output, sz_u8_t const tag[sz_at_least_(16)]) {
    sz_aes256_gcm_decryptor_t decryptor;
    sz_status_t verdict;
    sz_aes256_gcm_decryptor_init_sve2aes(&decryptor, key, nonce);
    if (associated_length) sz_aes256_gcm_decryptor_associate_sve2aes(&decryptor, associated, associated_length);
    sz_aes256_gcm_decryptor_update_unverified_sve2aes(&decryptor, text, length, output);
    verdict = sz_aes256_gcm_decryptor_verify_sve2aes(&decryptor, tag);
    sz_aes256_gcm_state_scrub_sve2aes_(&decryptor.state);

    //  A caller who drops the status must still be unable to act on forged plaintext.
    if (verdict != sz_success_k) sz_fill(output, length, 0);
    return verdict;
}

#pragma endregion // One Shot Interface

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE2AES

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_CIPHER_SVE2AES_H_
