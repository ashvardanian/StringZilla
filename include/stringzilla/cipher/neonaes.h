/**
 *  @brief Arm NEON crypto-extension backend for AES-256 in counter and Galois/counter modes.
 *  @file include/stringzilla/cipher/neonaes.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/cipher.h
 *
 *  The fourteen rounds are written out rather than looped, for the reason `cipher/icelake.h` spells out.
 */
#ifndef STRINGZILLA_CIPHER_NEONAES_H_
#define STRINGZILLA_CIPHER_NEONAES_H_

#include "stringzilla/types.h"
#include "stringzilla/cipher/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_NEONAES
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd+crypto+aes"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd+crypto+aes")
#endif

/*  Arm splits the round function where x86 does not, and this file is written against Arm's split rather
 *  than a translation of the other one. `AESE` adds the round key first and then substitutes and shifts
 *  rows, while x86's `AESENC` substitutes, shifts, mixes columns and adds the key last. A full round is
 *  therefore `AESMC(AESE(state, round_key))` carrying the key of the round the state is entering, the
 *  thirteen mixing rounds run over keys zero through twelve, the fourteenth drops the column mixing, and
 *  the last round key is folded in afterwards with a plain exclusive-or.
 *
 *  There is also no key-generation assist. The substitution a schedule word needs comes out of `AESE`
 *  against a zero round key on a block holding four copies of that word: the row shift only ever moves a
 *  byte between columns, so four identical columns pass through it untouched and what returns is the
 *  substituted word, four times over. Rotation is then a byte rotate of the same broadcast register.
 *
 *  Both wide paths run eight blocks deep. That is what the eight subkey powers in `sz_aes256_gcm_key_t`
 *  were derived for, and it matches the round instruction's throughput: `AESE` and `AESMC` fuse into a
 *  single operation of about three cycles' latency on every core that has them, so a chain shorter than
 *  eight leaves the pipe half idle.
 */

#pragma region Key Schedule

/**
 *  @brief Broadcasts the last word of a round key across all four lanes.
 *  @param round_key_u8x16 The round key.
 *  @return Its fourth word in every lane.
 */
SZ_HELPER_INLINE uint8x16_t sz_aes256_key_broadcast_neonaes_(uint8x16_t round_key_u8x16) {
    return vreinterpretq_u8_u32(vdupq_laneq_u32(vreinterpretq_u32_u8(round_key_u8x16), 3));
}

/**
 *  @brief Rotates a broadcast word so `(a0, a1, a2, a3)` becomes `(a1, a2, a3, a0)` in every lane.
 *  @param broadcast_u8x16 The word to rotate, repeated in every lane.
 *  @return The rotated word, repeated in every lane.
 *
 *  A broadcast register repeats with a period of four bytes, so rotating the word inside each lane and
 *  rotating the whole register by one byte are the same permutation.
 */
SZ_HELPER_INLINE uint8x16_t sz_aes256_key_rotate_neonaes_(uint8x16_t broadcast_u8x16) {
    return vextq_u8(broadcast_u8x16, broadcast_u8x16, 1);
}

/**
 *  @brief Substitutes a schedule word broadcast across all four lanes.
 *  @param broadcast_u8x16 The word to substitute, repeated in every lane.
 *  @return The substituted word, repeated in every lane.
 *
 *  Against a zero round key `AESE` is exactly `ShiftRows(SubBytes(state))`, and the row shift permutes bytes
 *  between columns only.
 */
SZ_HELPER_INLINE uint8x16_t sz_aes256_key_substitute_neonaes_(uint8x16_t broadcast_u8x16) {
    return vaeseq_u8(broadcast_u8x16, vdupq_n_u8(0));
}

/**
 *  @brief Derives the word an even round key folds in: rotate, substitute, then add the round constant.
 *  @param round_key_u8x16 The round key whose last word feeds the next one.
 *  @param round_constant The round constant for this step.
 *  @return The derived word, broadcast across every lane.
 */
SZ_HELPER_INLINE uint8x16_t sz_aes256_key_rotated_word_neonaes_(uint8x16_t round_key_u8x16, sz_u32_t round_constant) {
    uint8x16_t const rotated_u8x16 = sz_aes256_key_rotate_neonaes_(sz_aes256_key_broadcast_neonaes_(round_key_u8x16));
    return veorq_u8(sz_aes256_key_substitute_neonaes_(rotated_u8x16),
                    vreinterpretq_u8_u32(vdupq_n_u32(round_constant)));
}

/**
 *  @brief Derives the word an odd round key folds in: substitute the previous key's last word.
 *  @param round_key_u8x16 The round key whose last word feeds the next one.
 *  @return The derived word, broadcast across every lane.
 */
SZ_HELPER_INLINE uint8x16_t sz_aes256_key_plain_word_neonaes_(uint8x16_t round_key_u8x16) {
    return sz_aes256_key_substitute_neonaes_(sz_aes256_key_broadcast_neonaes_(round_key_u8x16));
}

/**
 *  @brief Folds a substituted schedule word into the round key four words back, producing the next one.
 *  @param previous_u8x16 The round key four words back, all four of its words in one register.
 *  @param substituted_u8x16 The substituted word, already broadcast across every lane.
 *  @return The next round key.
 *
 *  FIPS 197 defines the schedule one word at a time, each word depending on the one before it.
 */
SZ_HELPER_INLINE uint8x16_t sz_aes256_key_fold_neonaes_(uint8x16_t previous_u8x16, uint8x16_t substituted_u8x16) {
    uint8x16_t const zeros_u8x16 = vdupq_n_u8(0);
    previous_u8x16 = veorq_u8(previous_u8x16, vextq_u8(zeros_u8x16, previous_u8x16, 12));
    previous_u8x16 = veorq_u8(previous_u8x16, vextq_u8(zeros_u8x16, previous_u8x16, 12));
    previous_u8x16 = veorq_u8(previous_u8x16, vextq_u8(zeros_u8x16, previous_u8x16, 12));
    return veorq_u8(previous_u8x16, substituted_u8x16);
}

SZ_API_COMPTIME void sz_aes256_key_init_neonaes(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    sz_u8_t *schedule = (sz_u8_t *)&key->round_keys[0];
    uint8x16_t even_round_key_u8x16 = vld1q_u8(secret);
    uint8x16_t odd_round_key_u8x16 = vld1q_u8(secret + 16);

    // An AES-256 schedule alternates two steps: a rotated substitution against a round constant and a
    // plain substitution. Seven of the first and six of the second follow the two round keys the secret
    // supplies, which is fifteen round keys in all.
    vst1q_u8(schedule + 0 * SZ_AES_BLOCK_LENGTH, even_round_key_u8x16);
    vst1q_u8(schedule + 1 * SZ_AES_BLOCK_LENGTH, odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_neonaes_(even_round_key_u8x16,
                                                       sz_aes256_key_rotated_word_neonaes_(odd_round_key_u8x16, 0x01u));
    vst1q_u8(schedule + 2 * SZ_AES_BLOCK_LENGTH, even_round_key_u8x16);
    odd_round_key_u8x16 = sz_aes256_key_fold_neonaes_(odd_round_key_u8x16,
                                                      sz_aes256_key_plain_word_neonaes_(even_round_key_u8x16));
    vst1q_u8(schedule + 3 * SZ_AES_BLOCK_LENGTH, odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_neonaes_(even_round_key_u8x16,
                                                       sz_aes256_key_rotated_word_neonaes_(odd_round_key_u8x16, 0x02u));
    vst1q_u8(schedule + 4 * SZ_AES_BLOCK_LENGTH, even_round_key_u8x16);
    odd_round_key_u8x16 = sz_aes256_key_fold_neonaes_(odd_round_key_u8x16,
                                                      sz_aes256_key_plain_word_neonaes_(even_round_key_u8x16));
    vst1q_u8(schedule + 5 * SZ_AES_BLOCK_LENGTH, odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_neonaes_(even_round_key_u8x16,
                                                       sz_aes256_key_rotated_word_neonaes_(odd_round_key_u8x16, 0x04u));
    vst1q_u8(schedule + 6 * SZ_AES_BLOCK_LENGTH, even_round_key_u8x16);
    odd_round_key_u8x16 = sz_aes256_key_fold_neonaes_(odd_round_key_u8x16,
                                                      sz_aes256_key_plain_word_neonaes_(even_round_key_u8x16));
    vst1q_u8(schedule + 7 * SZ_AES_BLOCK_LENGTH, odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_neonaes_(even_round_key_u8x16,
                                                       sz_aes256_key_rotated_word_neonaes_(odd_round_key_u8x16, 0x08u));
    vst1q_u8(schedule + 8 * SZ_AES_BLOCK_LENGTH, even_round_key_u8x16);
    odd_round_key_u8x16 = sz_aes256_key_fold_neonaes_(odd_round_key_u8x16,
                                                      sz_aes256_key_plain_word_neonaes_(even_round_key_u8x16));
    vst1q_u8(schedule + 9 * SZ_AES_BLOCK_LENGTH, odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_neonaes_(even_round_key_u8x16,
                                                       sz_aes256_key_rotated_word_neonaes_(odd_round_key_u8x16, 0x10u));
    vst1q_u8(schedule + 10 * SZ_AES_BLOCK_LENGTH, even_round_key_u8x16);
    odd_round_key_u8x16 = sz_aes256_key_fold_neonaes_(odd_round_key_u8x16,
                                                      sz_aes256_key_plain_word_neonaes_(even_round_key_u8x16));
    vst1q_u8(schedule + 11 * SZ_AES_BLOCK_LENGTH, odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_neonaes_(even_round_key_u8x16,
                                                       sz_aes256_key_rotated_word_neonaes_(odd_round_key_u8x16, 0x20u));
    vst1q_u8(schedule + 12 * SZ_AES_BLOCK_LENGTH, even_round_key_u8x16);
    odd_round_key_u8x16 = sz_aes256_key_fold_neonaes_(odd_round_key_u8x16,
                                                      sz_aes256_key_plain_word_neonaes_(even_round_key_u8x16));
    vst1q_u8(schedule + 13 * SZ_AES_BLOCK_LENGTH, odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_neonaes_(even_round_key_u8x16,
                                                       sz_aes256_key_rotated_word_neonaes_(odd_round_key_u8x16, 0x40u));
    vst1q_u8(schedule + 14 * SZ_AES_BLOCK_LENGTH, even_round_key_u8x16);
}

#pragma endregion // Key Schedule

#pragma region Block Encryption

/**
 *  @brief Reads one round key out of an expanded schedule.
 *  @param key The expanded schedule.
 *  @param round_index Which of the fifteen round keys to read, zero through fourteen.
 *  @return The round key.
 */
SZ_HELPER_INLINE uint8x16_t sz_aes256_round_key_neonaes_(sz_aes256_key_t const *key, sz_size_t round_index) {
    return vld1q_u8((sz_u8_t const *)&key->round_keys[round_index * 4]);
}

/**
 *  @brief Encrypts one block with the expanded schedule.
 *  @param key The expanded schedule.
 *  @param block_u8x16 The plaintext block.
 *  @return The ciphertext block.
 */
SZ_HELPER_INLINE uint8x16_t sz_aes256_block_encrypt_neonaes_(sz_aes256_key_t const *key, uint8x16_t block_u8x16) {
    block_u8x16 = vaesmcq_u8(vaeseq_u8(block_u8x16, sz_aes256_round_key_neonaes_(key, 0)));
    block_u8x16 = vaesmcq_u8(vaeseq_u8(block_u8x16, sz_aes256_round_key_neonaes_(key, 1)));
    block_u8x16 = vaesmcq_u8(vaeseq_u8(block_u8x16, sz_aes256_round_key_neonaes_(key, 2)));
    block_u8x16 = vaesmcq_u8(vaeseq_u8(block_u8x16, sz_aes256_round_key_neonaes_(key, 3)));
    block_u8x16 = vaesmcq_u8(vaeseq_u8(block_u8x16, sz_aes256_round_key_neonaes_(key, 4)));
    block_u8x16 = vaesmcq_u8(vaeseq_u8(block_u8x16, sz_aes256_round_key_neonaes_(key, 5)));
    block_u8x16 = vaesmcq_u8(vaeseq_u8(block_u8x16, sz_aes256_round_key_neonaes_(key, 6)));
    block_u8x16 = vaesmcq_u8(vaeseq_u8(block_u8x16, sz_aes256_round_key_neonaes_(key, 7)));
    block_u8x16 = vaesmcq_u8(vaeseq_u8(block_u8x16, sz_aes256_round_key_neonaes_(key, 8)));
    block_u8x16 = vaesmcq_u8(vaeseq_u8(block_u8x16, sz_aes256_round_key_neonaes_(key, 9)));
    block_u8x16 = vaesmcq_u8(vaeseq_u8(block_u8x16, sz_aes256_round_key_neonaes_(key, 10)));
    block_u8x16 = vaesmcq_u8(vaeseq_u8(block_u8x16, sz_aes256_round_key_neonaes_(key, 11)));
    block_u8x16 = vaesmcq_u8(vaeseq_u8(block_u8x16, sz_aes256_round_key_neonaes_(key, 12)));
    block_u8x16 = vaeseq_u8(block_u8x16, sz_aes256_round_key_neonaes_(key, 13));
    return veorq_u8(block_u8x16, sz_aes256_round_key_neonaes_(key, 14));
}

/** @brief Applies one fused round to all eight chains, so the eight issue back to back. */
SZ_HELPER_INLINE void sz_aes256_blocks_round_neonaes_(uint8x16_t *blocks_u8x16, uint8x16_t round_key_u8x16) {
    blocks_u8x16[0] = vaesmcq_u8(vaeseq_u8(blocks_u8x16[0], round_key_u8x16));
    blocks_u8x16[1] = vaesmcq_u8(vaeseq_u8(blocks_u8x16[1], round_key_u8x16));
    blocks_u8x16[2] = vaesmcq_u8(vaeseq_u8(blocks_u8x16[2], round_key_u8x16));
    blocks_u8x16[3] = vaesmcq_u8(vaeseq_u8(blocks_u8x16[3], round_key_u8x16));
    blocks_u8x16[4] = vaesmcq_u8(vaeseq_u8(blocks_u8x16[4], round_key_u8x16));
    blocks_u8x16[5] = vaesmcq_u8(vaeseq_u8(blocks_u8x16[5], round_key_u8x16));
    blocks_u8x16[6] = vaesmcq_u8(vaeseq_u8(blocks_u8x16[6], round_key_u8x16));
    blocks_u8x16[7] = vaesmcq_u8(vaeseq_u8(blocks_u8x16[7], round_key_u8x16));
}

/**
 *  @brief Encrypts eight blocks with the expanded schedule, in place.
 *  @param key The expanded schedule.
 *  @param blocks_u8x16 Eight plaintext blocks, replaced by their ciphertext.
 *
 *  The fused round instruction has several cycles of latency and issues every cycle, so a single chain of
 *  fourteen dependent rounds leaves most of that throughput idle.
 */
SZ_HELPER_INLINE void sz_aes256_blocks_encrypt_neonaes_(sz_aes256_key_t const *key, uint8x16_t *blocks_u8x16) {
    uint8x16_t round_key_u8x16;

    // The thirteen fused rounds are written out rather than looped, because a loop leaves the
    // unrolling to the optimizer and the optimizer only does it at `-O3`.
    sz_aes256_blocks_round_neonaes_(blocks_u8x16, sz_aes256_round_key_neonaes_(key, 0));
    sz_aes256_blocks_round_neonaes_(blocks_u8x16, sz_aes256_round_key_neonaes_(key, 1));
    sz_aes256_blocks_round_neonaes_(blocks_u8x16, sz_aes256_round_key_neonaes_(key, 2));
    sz_aes256_blocks_round_neonaes_(blocks_u8x16, sz_aes256_round_key_neonaes_(key, 3));
    sz_aes256_blocks_round_neonaes_(blocks_u8x16, sz_aes256_round_key_neonaes_(key, 4));
    sz_aes256_blocks_round_neonaes_(blocks_u8x16, sz_aes256_round_key_neonaes_(key, 5));
    sz_aes256_blocks_round_neonaes_(blocks_u8x16, sz_aes256_round_key_neonaes_(key, 6));
    sz_aes256_blocks_round_neonaes_(blocks_u8x16, sz_aes256_round_key_neonaes_(key, 7));
    sz_aes256_blocks_round_neonaes_(blocks_u8x16, sz_aes256_round_key_neonaes_(key, 8));
    sz_aes256_blocks_round_neonaes_(blocks_u8x16, sz_aes256_round_key_neonaes_(key, 9));
    sz_aes256_blocks_round_neonaes_(blocks_u8x16, sz_aes256_round_key_neonaes_(key, 10));
    sz_aes256_blocks_round_neonaes_(blocks_u8x16, sz_aes256_round_key_neonaes_(key, 11));
    sz_aes256_blocks_round_neonaes_(blocks_u8x16, sz_aes256_round_key_neonaes_(key, 12));

    round_key_u8x16 = sz_aes256_round_key_neonaes_(key, 13);
    blocks_u8x16[0] = vaeseq_u8(blocks_u8x16[0], round_key_u8x16);
    blocks_u8x16[1] = vaeseq_u8(blocks_u8x16[1], round_key_u8x16);
    blocks_u8x16[2] = vaeseq_u8(blocks_u8x16[2], round_key_u8x16);
    blocks_u8x16[3] = vaeseq_u8(blocks_u8x16[3], round_key_u8x16);
    blocks_u8x16[4] = vaeseq_u8(blocks_u8x16[4], round_key_u8x16);
    blocks_u8x16[5] = vaeseq_u8(blocks_u8x16[5], round_key_u8x16);
    blocks_u8x16[6] = vaeseq_u8(blocks_u8x16[6], round_key_u8x16);
    blocks_u8x16[7] = vaeseq_u8(blocks_u8x16[7], round_key_u8x16);

    round_key_u8x16 = sz_aes256_round_key_neonaes_(key, 14);
    blocks_u8x16[0] = veorq_u8(blocks_u8x16[0], round_key_u8x16);
    blocks_u8x16[1] = veorq_u8(blocks_u8x16[1], round_key_u8x16);
    blocks_u8x16[2] = veorq_u8(blocks_u8x16[2], round_key_u8x16);
    blocks_u8x16[3] = veorq_u8(blocks_u8x16[3], round_key_u8x16);
    blocks_u8x16[4] = veorq_u8(blocks_u8x16[4], round_key_u8x16);
    blocks_u8x16[5] = veorq_u8(blocks_u8x16[5], round_key_u8x16);
    blocks_u8x16[6] = veorq_u8(blocks_u8x16[6], round_key_u8x16);
    blocks_u8x16[7] = veorq_u8(blocks_u8x16[7], round_key_u8x16);
}

#pragma endregion // Block Encryption

#pragma region Counter Mode

/**
 *  @brief Places the twelve nonce bytes in a counter block whose trailing index word is zero.
 *  @param nonce The twelve nonce bytes.
 *  @return The counter block for index zero.
 */
SZ_HELPER_INLINE uint8x16_t sz_aes256_counter_base_neonaes_(sz_u8_t const *nonce) {
    // Only twelve bytes are readable and NEON has no masked load, so this reads eight then four.
    uint8x16_t const leading_u8x16 = vcombine_u8(vld1_u8(nonce), vdup_n_u8(0));
    return vreinterpretq_u8_u32(
        vld1q_lane_u32((sz_u32_t const *)(void const *)(nonce + 8), vreinterpretq_u32_u8(leading_u8x16), 2));
}

/**
 *  @brief Completes a counter block with a big-endian block index in its trailing word.
 *  @param base_u8x16 A counter block carrying the nonce, whose trailing word is overwritten.
 *  @param block_index The block index.
 *  @return The counter block for that index.
 */
SZ_HELPER_INLINE uint8x16_t sz_aes256_counter_block_neonaes_(uint8x16_t base_u8x16, sz_u32_t block_index) {
    return vreinterpretq_u8_u32(vsetq_lane_u32(sz_u32_bytes_reverse(block_index), vreinterpretq_u32_u8(base_u8x16), 3));
}

SZ_API_COMPTIME void sz_aes256_ctr_xor_neonaes(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                               sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length,
                                               sz_ptr_t output) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    uint8x16_t const counter_base_u8x16 = sz_aes256_counter_base_neonaes_(nonce);
    sz_u32_t block_index = (sz_u32_t)(byte_offset / SZ_AES_BLOCK_LENGTH);
    sz_size_t within_block = (sz_size_t)(byte_offset % SZ_AES_BLOCK_LENGTH);
    sz_size_t produced = 0, lane_index;

    // A start that is not block aligned generates its first block whole and discards the leading bytes.
    if (within_block != 0 && length != 0) {
        uint8x16_t const counter_u8x16 = sz_aes256_counter_block_neonaes_(counter_base_u8x16, block_index);
        sz_u128_vec_t keystream_vec;
        keystream_vec.u8x16 = sz_aes256_block_encrypt_neonaes_(key, counter_u8x16);
        // Stays a byte loop: NEON has no masked store, and writing the register would touch bytes the
        // caller did not hand us. Ice Lake and SVE2 do this run with predication.
        for (; within_block != SZ_AES_BLOCK_LENGTH && produced != length; ++within_block, ++produced)
            output_bytes[produced] = (sz_u8_t)(input_bytes[produced] ^ keystream_vec.u8s[within_block]);
        ++block_index;
    }

    for (; produced + 8 * SZ_AES_BLOCK_LENGTH <= length; produced += 8 * SZ_AES_BLOCK_LENGTH) {
        uint8x16_t keystream_u8x16[8];
        keystream_u8x16[0] = sz_aes256_counter_block_neonaes_(counter_base_u8x16, block_index + 0);
        keystream_u8x16[1] = sz_aes256_counter_block_neonaes_(counter_base_u8x16, block_index + 1);
        keystream_u8x16[2] = sz_aes256_counter_block_neonaes_(counter_base_u8x16, block_index + 2);
        keystream_u8x16[3] = sz_aes256_counter_block_neonaes_(counter_base_u8x16, block_index + 3);
        keystream_u8x16[4] = sz_aes256_counter_block_neonaes_(counter_base_u8x16, block_index + 4);
        keystream_u8x16[5] = sz_aes256_counter_block_neonaes_(counter_base_u8x16, block_index + 5);
        keystream_u8x16[6] = sz_aes256_counter_block_neonaes_(counter_base_u8x16, block_index + 6);
        keystream_u8x16[7] = sz_aes256_counter_block_neonaes_(counter_base_u8x16, block_index + 7);
        block_index += 8;
        sz_aes256_blocks_encrypt_neonaes_(key, keystream_u8x16);
        for (lane_index = 0; lane_index != 8; ++lane_index) {
            sz_size_t const lane_offset = produced + lane_index * SZ_AES_BLOCK_LENGTH;
            uint8x16_t const original_u8x16 = vld1q_u8(input_bytes + lane_offset);
            vst1q_u8(output_bytes + lane_offset, veorq_u8(original_u8x16, keystream_u8x16[lane_index]));
        }
    }

    for (; produced + SZ_AES_BLOCK_LENGTH <= length; produced += SZ_AES_BLOCK_LENGTH, ++block_index) {
        uint8x16_t const counter_u8x16 = sz_aes256_counter_block_neonaes_(counter_base_u8x16, block_index);
        uint8x16_t const original_u8x16 = vld1q_u8(input_bytes + produced);
        uint8x16_t const keystream_u8x16 = sz_aes256_block_encrypt_neonaes_(key, counter_u8x16);
        vst1q_u8(output_bytes + produced, veorq_u8(original_u8x16, keystream_u8x16));
    }

    if (produced != length) {
        uint8x16_t const counter_u8x16 = sz_aes256_counter_block_neonaes_(counter_base_u8x16, block_index);
        sz_u128_vec_t keystream_vec;
        keystream_vec.u8x16 = sz_aes256_block_encrypt_neonaes_(key, counter_u8x16);
        for (within_block = 0; produced != length; ++within_block, ++produced)
            output_bytes[produced] = (sz_u8_t)(input_bytes[produced] ^ keystream_vec.u8s[within_block]);
    }
}

#pragma endregion // Counter Mode

#pragma region Galois Hashing

/**
 *  @brief Reverses a hash block between the tag's byte order and the multiplier's, its own inverse.
 *  @param block_u8x16 The block in either order.
 *  @return The same block in the other order.
 *
 *  The tag is defined over blocks whose leading bit is the field element's lowest coefficient, which is the
 *  opposite of how a carry-less multiply reads its operands.
 */
SZ_HELPER_INLINE uint8x16_t sz_ghash_reflect_neonaes_(uint8x16_t block_u8x16) {
    uint8x16_t const halves_u8x16 = vrev64q_u8(block_u8x16);
    return vextq_u8(halves_u8x16, halves_u8x16, 8);
}

/**
 *  @brief Loads a hash block from the tag's byte order into the multiplier's.
 *  @param block The sixteen block bytes.
 *  @return The reflected block.
 */
SZ_HELPER_INLINE uint8x16_t sz_ghash_load_neonaes_(sz_u8_t const *block) {
    return sz_ghash_reflect_neonaes_(vld1q_u8(block));
}

/**
 *  @brief Loads a pending block with everything past @p buffered forced to zero.
 *
 *  A lane-identity compare rather than a byte loop, which the compilers lowered to branchy scalar stores and a
 *  store-forwarding stall.
 */
SZ_HELPER_INLINE uint8x16_t sz_ghash_load_padded_neonaes_(sz_u8_t const *block, sz_size_t buffered) {
    uint8x16_t const lane_ids_u8x16 = vcombine_u8(vcreate_u8(0x0706050403020100ull), //
                                                  vcreate_u8(0x0F0E0D0C0B0A0908ull));
    uint8x16_t const keep_u8x16 = vcltq_u8(lane_ids_u8x16, vdupq_n_u8((sz_u8_t)buffered));
    return sz_ghash_reflect_neonaes_(vandq_u8(vld1q_u8(block), keep_u8x16));
}

/** @brief Compares two tags in constant time; `sz_true_k` when all sixteen bytes match. */
SZ_HELPER_INLINE sz_bool_t sz_aes256_tag_equal_neonaes_(sz_u8_t const *first, sz_u8_t const *second) {
    uint8x16_t const matching_u8x16 = vceqq_u8(vld1q_u8(first), vld1q_u8(second));
    return vminvq_u8(matching_u8x16) == 0xFF ? sz_true_k : sz_false_k;
}

/**
 *  @brief Stores a hash block back in the tag's byte order.
 *  @param value_u8x16 The reflected block.
 *  @param block Receives the sixteen bytes.
 */
SZ_HELPER_INLINE void sz_ghash_store_neonaes_(uint8x16_t value_u8x16, sz_u8_t *block) {
    vst1q_u8(block, sz_ghash_reflect_neonaes_(value_u8x16));
}

/**
 *  @brief Carry-less product of two 64-bit halves already isolated in their own vectors.
 *  @param multiplicand_p64x1 One half.
 *  @param multiplier_p64x1 The other half.
 *  @return Their 128-bit product.
 *
 *  The operands arrive as vectors rather than scalars because that is the only spelling both dialects accept:
 *  MSVC lowers `vmull_p64` straight onto `neon_pmull_64`, which takes `__n64` lanes, while the ACLE prototype
 *  on GCC and Clang takes `poly64_t` scalars.
 */
SZ_HELPER_INLINE uint8x16_t sz_ghash_product_halves_neonaes_(poly64x1_t multiplicand_p64x1,
                                                             poly64x1_t multiplier_p64x1) {
#if defined(_MSC_VER) && !defined(__clang__)
    return vreinterpretq_u8_p128(vmull_p64(multiplicand_p64x1, multiplier_p64x1));
#else
    return vreinterpretq_u8_p128(vmull_p64(vget_lane_p64(multiplicand_p64x1, 0), vget_lane_p64(multiplier_p64x1, 0)));
#endif
}

/**
 *  @brief Carry-less product of the low halves of two reflected blocks.
 *  @param multiplicand_u8x16 One reflected operand.
 *  @param multiplier_u8x16 The other reflected operand.
 *  @return Their 128-bit product.
 */
SZ_HELPER_INLINE uint8x16_t sz_ghash_product_low_neonaes_(uint8x16_t multiplicand_u8x16, uint8x16_t multiplier_u8x16) {
    return sz_ghash_product_halves_neonaes_(vget_low_p64(vreinterpretq_p64_u8(multiplicand_u8x16)),
                                            vget_low_p64(vreinterpretq_p64_u8(multiplier_u8x16)));
}

/**
 *  @brief Carry-less product of the high halves of two reflected blocks.
 *  @param multiplicand_u8x16 One reflected operand.
 *  @param multiplier_u8x16 The other reflected operand.
 *  @return Their 128-bit product.
 */
SZ_HELPER_INLINE uint8x16_t sz_ghash_product_high_neonaes_(uint8x16_t multiplicand_u8x16, uint8x16_t multiplier_u8x16) {
    return vreinterpretq_u8_p128(
        vmull_high_p64(vreinterpretq_p64_u8(multiplicand_u8x16), vreinterpretq_p64_u8(multiplier_u8x16)));
}

/**
 *  @brief Carry-less product of one reflected block's low half with the other's high half.
 *  @param low_u8x16 The operand contributing its low half.
 *  @param high_u8x16 The operand contributing its high half.
 *  @return Their 128-bit product.
 */
SZ_HELPER_INLINE uint8x16_t sz_ghash_product_cross_neonaes_(uint8x16_t low_u8x16, uint8x16_t high_u8x16) {
    return sz_ghash_product_halves_neonaes_(vget_low_p64(vreinterpretq_p64_u8(low_u8x16)),
                                            vget_high_p64(vreinterpretq_p64_u8(high_u8x16)));
}

/**
 *  @brief Multiplies the low half of a value by the field's reduction polynomial.
 *  @param value_u8x16 The value whose low half is folded.
 *  @return The 128-bit product.
 *
 *  In the reflected representation the polynomial `x^128 + x^7 + x^2 + x + 1` collapses to the single 64-bit
 *  constant below, which is what lets the reduction run as two multiplies rather than a chain of shifts: the
 *  same instruction that produced the product also closes it.
 */
SZ_HELPER_INLINE uint8x16_t sz_ghash_product_polynomial_neonaes_(uint8x16_t value_u8x16) {
    poly64x1_t const polynomial_p64x1 = vcreate_p64(0xC200000000000000ull);
    return sz_ghash_product_halves_neonaes_(vget_low_p64(vreinterpretq_p64_u8(value_u8x16)), polynomial_p64x1);
}

/**
 *  @brief Accumulates the 256-bit carry-less product of two reflected blocks into a running triple.
 *  @param multiplicand_u8x16 One reflected operand.
 *  @param multiplier_u8x16 The other reflected operand.
 *  @param low_u8x16 Accumulates the product of the two low halves.
 *  @param middle_u8x16 Accumulates both cross products.
 *  @param high_u8x16 Accumulates the product of the two high halves.
 *
 *  Splitting the schoolbook product from its reduction is what lets several blocks share one reduction: the
 *  field is linear, so the partial products of a whole group may be summed before folding once.
 */
SZ_HELPER_INLINE void sz_ghash_accumulate_neonaes_(uint8x16_t multiplicand_u8x16, uint8x16_t multiplier_u8x16,
                                                   uint8x16_t *low_u8x16, uint8x16_t *middle_u8x16,
                                                   uint8x16_t *high_u8x16) {
    *low_u8x16 = veorq_u8(*low_u8x16, sz_ghash_product_low_neonaes_(multiplicand_u8x16, multiplier_u8x16));
    *middle_u8x16 = veorq_u8(*middle_u8x16, sz_ghash_product_cross_neonaes_(multiplicand_u8x16, multiplier_u8x16));
    *middle_u8x16 = veorq_u8(*middle_u8x16, sz_ghash_product_cross_neonaes_(multiplier_u8x16, multiplicand_u8x16));
    *high_u8x16 = veorq_u8(*high_u8x16, sz_ghash_product_high_neonaes_(multiplicand_u8x16, multiplier_u8x16));
}

/**
 *  @brief Folds an accumulated triple into one reflected block.
 *  @param product_low_u8x16 The accumulated low halves.
 *  @param product_middle_u8x16 The accumulated cross products.
 *  @param product_high_u8x16 The accumulated high halves.
 *  @return The reduced product, reflected.
 *
 *  The cross products straddle the halves, so they are split and merged first.
 */
SZ_HELPER_INLINE uint8x16_t sz_ghash_reduce_neonaes_(uint8x16_t product_low_u8x16, uint8x16_t product_middle_u8x16,
                                                     uint8x16_t product_high_u8x16) {
    uint8x16_t const zeros_u8x16 = vdupq_n_u8(0);
    uint8x16_t low_half_u8x16 = veorq_u8(product_low_u8x16, vextq_u8(zeros_u8x16, product_middle_u8x16, 8));
    uint8x16_t high_half_u8x16 = veorq_u8(product_high_u8x16, vextq_u8(product_middle_u8x16, zeros_u8x16, 8));
    uint8x16_t const low_carry_u8x16 = vreinterpretq_u8_u32(vshrq_n_u32(vreinterpretq_u32_u8(low_half_u8x16), 31));
    uint8x16_t const high_carry_u8x16 = vreinterpretq_u8_u32(vshrq_n_u32(vreinterpretq_u32_u8(high_half_u8x16), 31));
    uint8x16_t folded_u8x16;

    low_half_u8x16 = vorrq_u8(vreinterpretq_u8_u32(vshlq_n_u32(vreinterpretq_u32_u8(low_half_u8x16), 1)),
                              vextq_u8(zeros_u8x16, low_carry_u8x16, 12));
    high_half_u8x16 = vorrq_u8(vreinterpretq_u8_u32(vshlq_n_u32(vreinterpretq_u32_u8(high_half_u8x16), 1)),
                               vextq_u8(zeros_u8x16, high_carry_u8x16, 12));
    high_half_u8x16 = vorrq_u8(high_half_u8x16, vextq_u8(low_carry_u8x16, zeros_u8x16, 12));

    folded_u8x16 = sz_ghash_product_polynomial_neonaes_(low_half_u8x16);
    low_half_u8x16 = veorq_u8(vextq_u8(low_half_u8x16, low_half_u8x16, 8), folded_u8x16);
    folded_u8x16 = sz_ghash_product_polynomial_neonaes_(low_half_u8x16);
    low_half_u8x16 = veorq_u8(vextq_u8(low_half_u8x16, low_half_u8x16, 8), folded_u8x16);
    return veorq_u8(high_half_u8x16, low_half_u8x16);
}

/**
 *  @brief Multiplies two reflected blocks in the Galois field the tag is built over.
 *  @param multiplicand_u8x16 One reflected operand.
 *  @param multiplier_u8x16 The other reflected operand.
 *  @return The reduced product, reflected.
 */
SZ_HELPER_INLINE uint8x16_t sz_ghash_multiply_neonaes_(uint8x16_t multiplicand_u8x16, uint8x16_t multiplier_u8x16) {
    uint8x16_t product_low_u8x16 = vdupq_n_u8(0), product_middle_u8x16 = vdupq_n_u8(0),
               product_high_u8x16 = vdupq_n_u8(0);
    sz_ghash_accumulate_neonaes_(multiplicand_u8x16, multiplier_u8x16, &product_low_u8x16, &product_middle_u8x16,
                                 &product_high_u8x16);
    return sz_ghash_reduce_neonaes_(product_low_u8x16, product_middle_u8x16, product_high_u8x16);
}

/**
 *  @brief Absorbs one reflected block into the running hash.
 *  @param accumulator_u8x16 The running hash, reflected.
 *  @param block_u8x16 The reflected block to absorb.
 *  @param subkey_u8x16 The reflected hash subkey.
 *  @return The updated running hash, reflected.
 */
SZ_HELPER_INLINE uint8x16_t sz_ghash_absorb_neonaes_(uint8x16_t accumulator_u8x16, uint8x16_t block_u8x16,
                                                     uint8x16_t subkey_u8x16) {
    return sz_ghash_multiply_neonaes_(veorq_u8(accumulator_u8x16, block_u8x16), subkey_u8x16);
}

SZ_API_COMPTIME void sz_aes256_gcm_key_init_neonaes(sz_aes256_gcm_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    uint8x16_t subkey_u8x16, power_u8x16;
    sz_size_t power_index;

    sz_aes256_key_init_neonaes(&key->block, secret);
    subkey_u8x16 = sz_ghash_reflect_neonaes_(sz_aes256_block_encrypt_neonaes_(&key->block, vdupq_n_u8(0)));
    power_u8x16 = subkey_u8x16;
    sz_ghash_store_neonaes_(power_u8x16, &key->powers[0]);
    for (power_index = 1; power_index != 8; ++power_index) {
        power_u8x16 = sz_ghash_multiply_neonaes_(power_u8x16, subkey_u8x16);
        sz_ghash_store_neonaes_(power_u8x16, &key->powers[power_index * SZ_AES_BLOCK_LENGTH]);
    }
}

/**
 *  @brief Loads the eight subkey powers descending, so power `j` pairs with block `j` of a group.
 *  @param powers The eight subkey powers in the tag's byte order, ascending.
 *  @param powers_u8x16 Receives the reflected powers `H^8` down to `H^1`.
 */
SZ_HELPER_INLINE void sz_ghash_descending_powers_neonaes_(sz_u8_t const *powers, uint8x16_t *powers_u8x16) {
    powers_u8x16[0] = sz_ghash_load_neonaes_(powers + 7 * SZ_AES_BLOCK_LENGTH);
    powers_u8x16[1] = sz_ghash_load_neonaes_(powers + 6 * SZ_AES_BLOCK_LENGTH);
    powers_u8x16[2] = sz_ghash_load_neonaes_(powers + 5 * SZ_AES_BLOCK_LENGTH);
    powers_u8x16[3] = sz_ghash_load_neonaes_(powers + 4 * SZ_AES_BLOCK_LENGTH);
    powers_u8x16[4] = sz_ghash_load_neonaes_(powers + 3 * SZ_AES_BLOCK_LENGTH);
    powers_u8x16[5] = sz_ghash_load_neonaes_(powers + 2 * SZ_AES_BLOCK_LENGTH);
    powers_u8x16[6] = sz_ghash_load_neonaes_(powers + 1 * SZ_AES_BLOCK_LENGTH);
    powers_u8x16[7] = sz_ghash_load_neonaes_(powers + 0 * SZ_AES_BLOCK_LENGTH);
}

/**
 *  @brief Absorbs eight reflected blocks into the running hash with a single field reduction.
 *  @param accumulator_u8x16 The running hash, reflected.
 *  @param blocks_u8x16 The eight reflected blocks, in the order they arrived.
 *  @param powers_u8x16 The reflected powers `H^8` through `H^1`.
 *  @return The updated running hash, reflected.
 *
 *  The hash is a chain, `Y = (Y ^ X) * H`, and a chain of reductions would run at the latency of one multiply
 *  per block.
 */
SZ_HELPER_INLINE uint8x16_t sz_ghash_absorb_eight_neonaes_(uint8x16_t accumulator_u8x16, uint8x16_t const *blocks_u8x16,
                                                           uint8x16_t const *powers_u8x16) {
    uint8x16_t product_low_u8x16 = vdupq_n_u8(0), product_middle_u8x16 = vdupq_n_u8(0),
               product_high_u8x16 = vdupq_n_u8(0);
    sz_ghash_accumulate_neonaes_(veorq_u8(accumulator_u8x16, blocks_u8x16[0]), powers_u8x16[0], &product_low_u8x16,
                                 &product_middle_u8x16, &product_high_u8x16);
    sz_ghash_accumulate_neonaes_(blocks_u8x16[1], powers_u8x16[1], &product_low_u8x16, &product_middle_u8x16,
                                 &product_high_u8x16);
    sz_ghash_accumulate_neonaes_(blocks_u8x16[2], powers_u8x16[2], &product_low_u8x16, &product_middle_u8x16,
                                 &product_high_u8x16);
    sz_ghash_accumulate_neonaes_(blocks_u8x16[3], powers_u8x16[3], &product_low_u8x16, &product_middle_u8x16,
                                 &product_high_u8x16);
    sz_ghash_accumulate_neonaes_(blocks_u8x16[4], powers_u8x16[4], &product_low_u8x16, &product_middle_u8x16,
                                 &product_high_u8x16);
    sz_ghash_accumulate_neonaes_(blocks_u8x16[5], powers_u8x16[5], &product_low_u8x16, &product_middle_u8x16,
                                 &product_high_u8x16);
    sz_ghash_accumulate_neonaes_(blocks_u8x16[6], powers_u8x16[6], &product_low_u8x16, &product_middle_u8x16,
                                 &product_high_u8x16);
    sz_ghash_accumulate_neonaes_(blocks_u8x16[7], powers_u8x16[7], &product_low_u8x16, &product_middle_u8x16,
                                 &product_high_u8x16);
    return sz_ghash_reduce_neonaes_(product_low_u8x16, product_middle_u8x16, product_high_u8x16);
}

#pragma endregion // Galois Hashing

#pragma region Streaming Interface

/**
 *  @brief Overwrites a finished state so the key schedule it embeds does not outlive the call.
 *
 *  The size is known at compile time, so this is a straight-line run of whole-register stores rather than a
 *  length-driven loop.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_state_scrub_neonaes_(sz_aes256_gcm_state_t *state) {
    sz_u8_t *const bytes = (sz_u8_t *)state;
    uint8x16_t const zeros_u8x16 = vdupq_n_u8(0);
    sz_size_t vector_index;
    for (vector_index = 0; vector_index != sizeof(*state) / SZ_AES_BLOCK_LENGTH; ++vector_index)
        vst1q_u8(bytes + vector_index * SZ_AES_BLOCK_LENGTH, zeros_u8x16);
    vst1_u8(bytes + sizeof(*state) - 8, vdup_n_u8(0));
    sz_keep_alive_(state);
}

/** @brief Prepares the payload both directions share: counter block, tag mask and empty carries. */
SZ_HELPER_INLINE void sz_aes256_gcm_begin_neonaes_(sz_aes256_gcm_state_t *state, sz_aes256_gcm_key_t const *key,
                                                   sz_u8_t const nonce[sz_at_least_(12)]) {
    uint8x16_t initial_u8x16;

    state->key = *key;

    // With a twelve-byte nonce the initial counter block is the nonce followed by a one, and the value
    // encrypted under it masks the finished hash. Data blocks start one past it.
    initial_u8x16 = sz_aes256_counter_block_neonaes_(sz_aes256_counter_base_neonaes_(nonce), 1u);
    vst1q_u8(state->tag_mask, sz_aes256_block_encrypt_neonaes_(&state->key.block, initial_u8x16));
    vst1q_u8(state->counter, initial_u8x16);
    vst1q_u8(state->accumulator, vdupq_n_u8(0));
    vst1q_u8(state->partial, vdupq_n_u8(0));
    vst1q_u8(state->keystream, vdupq_n_u8(0));

    state->associated_length = 0;
    state->text_length = 0;
    state->buffered = 0;
    state->keystream_used = SZ_AES_BLOCK_LENGTH; // ? Forces the first message byte to derive a fresh block
}

/** @brief Absorbs associated data into the payload both directions share. */
SZ_HELPER_INLINE void sz_aes256_gcm_associate_neonaes_(sz_aes256_gcm_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    uint8x16_t const subkey_u8x16 = sz_ghash_load_neonaes_(state->key.powers);
    uint8x16_t powers_u8x16[8];
    uint8x16_t accumulator_u8x16;
    sz_size_t consumed = 0, byte_index;

    if (length == 0) return;
    sz_ghash_descending_powers_neonaes_(state->key.powers, powers_u8x16);
    accumulator_u8x16 = sz_ghash_load_neonaes_(state->accumulator);
    state->associated_length += length;

    // Associated data is hashed but never encrypted, so a partial block is completed in place.
    if (state->buffered != 0) {
        sz_size_t const available_bytes = SZ_AES_BLOCK_LENGTH - state->buffered;
        sz_size_t const taken_bytes = length < available_bytes ? length : available_bytes;
        for (byte_index = 0; byte_index != taken_bytes; ++byte_index)
            state->partial[state->buffered + byte_index] = input_bytes[byte_index];
        state->buffered = (sz_u8_t)(state->buffered + taken_bytes);
        consumed = taken_bytes;
        if (state->buffered == SZ_AES_BLOCK_LENGTH) {
            accumulator_u8x16 = sz_ghash_absorb_neonaes_(accumulator_u8x16, sz_ghash_load_neonaes_(state->partial),
                                                         subkey_u8x16);
            state->buffered = 0;
        }
    }

    for (; consumed + 8 * SZ_AES_BLOCK_LENGTH <= length; consumed += 8 * SZ_AES_BLOCK_LENGTH) {
        uint8x16_t blocks_u8x16[8];
        blocks_u8x16[0] = sz_ghash_load_neonaes_(input_bytes + consumed + 0 * SZ_AES_BLOCK_LENGTH);
        blocks_u8x16[1] = sz_ghash_load_neonaes_(input_bytes + consumed + 1 * SZ_AES_BLOCK_LENGTH);
        blocks_u8x16[2] = sz_ghash_load_neonaes_(input_bytes + consumed + 2 * SZ_AES_BLOCK_LENGTH);
        blocks_u8x16[3] = sz_ghash_load_neonaes_(input_bytes + consumed + 3 * SZ_AES_BLOCK_LENGTH);
        blocks_u8x16[4] = sz_ghash_load_neonaes_(input_bytes + consumed + 4 * SZ_AES_BLOCK_LENGTH);
        blocks_u8x16[5] = sz_ghash_load_neonaes_(input_bytes + consumed + 5 * SZ_AES_BLOCK_LENGTH);
        blocks_u8x16[6] = sz_ghash_load_neonaes_(input_bytes + consumed + 6 * SZ_AES_BLOCK_LENGTH);
        blocks_u8x16[7] = sz_ghash_load_neonaes_(input_bytes + consumed + 7 * SZ_AES_BLOCK_LENGTH);
        accumulator_u8x16 = sz_ghash_absorb_eight_neonaes_(accumulator_u8x16, blocks_u8x16, powers_u8x16);
    }
    for (; consumed + SZ_AES_BLOCK_LENGTH <= length; consumed += SZ_AES_BLOCK_LENGTH)
        accumulator_u8x16 = sz_ghash_absorb_neonaes_(accumulator_u8x16, sz_ghash_load_neonaes_(input_bytes + consumed),
                                                     subkey_u8x16);
    for (; consumed != length; ++consumed) state->partial[state->buffered++] = input_bytes[consumed];

    sz_ghash_store_neonaes_(accumulator_u8x16, state->accumulator);
}

/**
 *  @brief Spends what is left of the keystream block the state carries, one byte at a time.
 *  @param state The state, whose two sixteen-byte counters advance together here.
 *  @param input The bytes to transform.
 *  @param output Receives the transformed bytes.
 *  @param count Bytes to consume, never more than the keystream block has left.
 *  @param accumulator_u8x16 The running hash, reflected.
 *  @param subkey_u8x16 The reflected hash subkey.
 *  @param direction Which buffer the hash absorbs.
 *  @return The updated running hash, reflected.
 *
 *  Through the message the keystream offset and the hash offset are the same number, because every byte spends
 *  one of each, so a chunk that ends mid block leaves both mid block and this resumes both.
 */
SZ_HELPER_INLINE uint8x16_t sz_aes256_gcm_spend_neonaes_(sz_aes256_gcm_state_t *state, sz_u8_t const *input,
                                                         sz_u8_t *output, sz_size_t count, uint8x16_t accumulator_u8x16,
                                                         uint8x16_t subkey_u8x16, sz_aes256_gcm_direction_t direction) {
    sz_size_t byte_index;

    // Scalar: NEON has no masked load or store, so a partial run cannot move without a
    // round trip through memory. Ice Lake, SVE2, RVV and Power do it a run at a time.
    for (byte_index = 0; byte_index != count; ++byte_index) {
        sz_u8_t const plaintext_byte = input[byte_index];
        sz_u8_t const transformed = (sz_u8_t)(plaintext_byte ^ state->keystream[state->keystream_used]);
        sz_u8_t const ciphertext_byte = direction == sz_aes256_gcm_decrypting_k ? plaintext_byte : transformed;
        output[byte_index] = transformed;
        state->partial[state->buffered] = ciphertext_byte;
        ++state->buffered;
        ++state->keystream_used;
        if (state->buffered == SZ_AES_BLOCK_LENGTH) {
            accumulator_u8x16 = sz_ghash_absorb_neonaes_(accumulator_u8x16, sz_ghash_load_neonaes_(state->partial),
                                                         subkey_u8x16);
            state->buffered = 0;
        }
    }
    return accumulator_u8x16;
}

/**
 *  @brief Transforms a chunk and absorbs its ciphertext, whichever side of the call that is.
 *  @param state The state.
 *  @param text The chunk to transform.
 *  @param length Bytes in the chunk.
 *  @param output Receives the transformed bytes.
 *  @param direction Which buffer the hash absorbs.
 *
 *  Four passes, because two sixteen-byte rhythms run underneath a caller's arbitrary chunk sizes and neither
 *  may restart at a chunk boundary: whatever the previous chunk left of its keystream block, then whole blocks
 *  eight at a time, then any remaining whole blocks, then a trailing block the next chunk resumes.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_transform_neonaes_(sz_aes256_gcm_state_t *state, sz_cptr_t text, sz_size_t length,
                                                       sz_ptr_t output, sz_aes256_gcm_direction_t direction) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    uint8x16_t const subkey_u8x16 = sz_ghash_load_neonaes_(state->key.powers);
    uint8x16_t powers_u8x16[8];
    sz_u128_vec_t counter_vec;
    uint8x16_t accumulator_u8x16;
    sz_u32_t block_index;
    sz_size_t produced = 0, lane_index;

    if (length == 0) return;
    sz_ghash_descending_powers_neonaes_(state->key.powers, powers_u8x16);
    accumulator_u8x16 = sz_ghash_load_neonaes_(state->accumulator);

    // Associated data ends the moment the first message byte arrives, and its tail needs padding.
    if (state->text_length == 0 && state->buffered != 0) {
        accumulator_u8x16 = sz_ghash_absorb_neonaes_(
            accumulator_u8x16, sz_ghash_load_padded_neonaes_(state->partial, state->buffered), subkey_u8x16);
        state->buffered = 0;
    }

    counter_vec.u8x16 = vld1q_u8(state->counter);
    block_index = sz_u32_bytes_reverse(counter_vec.u32s[3]);

    if (state->keystream_used != SZ_AES_BLOCK_LENGTH) {
        sz_size_t const available_bytes = SZ_AES_BLOCK_LENGTH - state->keystream_used;
        sz_size_t const taken_bytes = length < available_bytes ? length : available_bytes;
        accumulator_u8x16 = sz_aes256_gcm_spend_neonaes_(state, input_bytes, output_bytes, taken_bytes,
                                                         accumulator_u8x16, subkey_u8x16, direction);
        produced = taken_bytes;
    }

    for (; produced + 8 * SZ_AES_BLOCK_LENGTH <= length; produced += 8 * SZ_AES_BLOCK_LENGTH) {
        uint8x16_t keystream_u8x16[8], ciphertext_u8x16[8];
        keystream_u8x16[0] = sz_aes256_counter_block_neonaes_(counter_vec.u8x16, block_index + 1);
        keystream_u8x16[1] = sz_aes256_counter_block_neonaes_(counter_vec.u8x16, block_index + 2);
        keystream_u8x16[2] = sz_aes256_counter_block_neonaes_(counter_vec.u8x16, block_index + 3);
        keystream_u8x16[3] = sz_aes256_counter_block_neonaes_(counter_vec.u8x16, block_index + 4);
        keystream_u8x16[4] = sz_aes256_counter_block_neonaes_(counter_vec.u8x16, block_index + 5);
        keystream_u8x16[5] = sz_aes256_counter_block_neonaes_(counter_vec.u8x16, block_index + 6);
        keystream_u8x16[6] = sz_aes256_counter_block_neonaes_(counter_vec.u8x16, block_index + 7);
        keystream_u8x16[7] = sz_aes256_counter_block_neonaes_(counter_vec.u8x16, block_index + 8);
        block_index += 8;
        sz_aes256_blocks_encrypt_neonaes_(&state->key.block, keystream_u8x16);
        for (lane_index = 0; lane_index != 8; ++lane_index) {
            sz_size_t const lane_offset = produced + lane_index * SZ_AES_BLOCK_LENGTH;
            uint8x16_t const original_u8x16 = vld1q_u8(input_bytes + lane_offset);
            uint8x16_t const transformed_u8x16 = veorq_u8(original_u8x16, keystream_u8x16[lane_index]);
            uint8x16_t const ciphered_u8x16 = direction == sz_aes256_gcm_decrypting_k ? original_u8x16
                                                                                      : transformed_u8x16;
            ciphertext_u8x16[lane_index] = sz_ghash_reflect_neonaes_(ciphered_u8x16);
            vst1q_u8(output_bytes + lane_offset, transformed_u8x16);
        }
        accumulator_u8x16 = sz_ghash_absorb_eight_neonaes_(accumulator_u8x16, ciphertext_u8x16, powers_u8x16);
    }

    for (; produced + SZ_AES_BLOCK_LENGTH <= length; produced += SZ_AES_BLOCK_LENGTH) {
        uint8x16_t const original_u8x16 = vld1q_u8(input_bytes + produced);
        uint8x16_t counter_u8x16, keystream_u8x16, transformed_u8x16, ciphered_u8x16;
        block_index += 1;
        counter_u8x16 = sz_aes256_counter_block_neonaes_(counter_vec.u8x16, block_index);
        keystream_u8x16 = sz_aes256_block_encrypt_neonaes_(&state->key.block, counter_u8x16);
        transformed_u8x16 = veorq_u8(original_u8x16, keystream_u8x16);
        ciphered_u8x16 = direction == sz_aes256_gcm_decrypting_k ? original_u8x16 : transformed_u8x16;
        vst1q_u8(output_bytes + produced, transformed_u8x16);
        accumulator_u8x16 = sz_ghash_absorb_neonaes_(accumulator_u8x16, sz_ghash_reflect_neonaes_(ciphered_u8x16),
                                                     subkey_u8x16);
    }

    if (produced != length) {
        uint8x16_t counter_u8x16, keystream_u8x16;
        block_index += 1;
        counter_u8x16 = sz_aes256_counter_block_neonaes_(counter_vec.u8x16, block_index);
        keystream_u8x16 = sz_aes256_block_encrypt_neonaes_(&state->key.block, counter_u8x16);
        vst1q_u8(state->keystream, keystream_u8x16);
        state->keystream_used = 0;
        accumulator_u8x16 = sz_aes256_gcm_spend_neonaes_(state, input_bytes + produced, output_bytes + produced,
                                                         length - produced, accumulator_u8x16, subkey_u8x16, direction);
    }

    state->text_length += length;
    sz_ghash_store_neonaes_(accumulator_u8x16, state->accumulator);
    vst1q_u8(state->counter, sz_aes256_counter_block_neonaes_(counter_vec.u8x16, block_index));
}

/**
 *  @brief Closes the running hash in registers and unmasks it, leaving the state untouched.
 *  @param state The state, read only.
 *  @param tag Receives the sixteen tag bytes.
 *
 *  The pending block and the length block are absorbed in registers rather than into the state, so a caller
 *  may take an intermediate tag and keep streaming.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_digest_neonaes_(sz_aes256_gcm_state_t const *state, sz_u8_t tag[sz_at_least_(16)]) {
    uint8x16_t const subkey_u8x16 = sz_ghash_load_neonaes_(state->key.powers);
    uint8x16_t accumulator_u8x16 = sz_ghash_load_neonaes_(state->accumulator);
    sz_u128_vec_t lengths_vec;

    // Whatever is still pending gets padded here: an associated-data tail when the message was empty,
    // or the message's own trailing ciphertext bytes otherwise.
    if (state->buffered != 0) {
        accumulator_u8x16 = sz_ghash_absorb_neonaes_(
            accumulator_u8x16, sz_ghash_load_padded_neonaes_(state->partial, state->buffered), subkey_u8x16);
    }

    sz_aes256_gcm_lengths_serial_(&lengths_vec, state->associated_length, state->text_length);
    accumulator_u8x16 = sz_ghash_absorb_neonaes_(accumulator_u8x16, sz_ghash_reflect_neonaes_(lengths_vec.u8x16),
                                                 subkey_u8x16);

    vst1q_u8(tag, veorq_u8(sz_ghash_reflect_neonaes_(accumulator_u8x16), vld1q_u8(state->tag_mask)));
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_init_neonaes(sz_aes256_gcm_encryptor_t *encryptor,
                                                          sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_aes256_gcm_begin_neonaes_(&encryptor->state, key, nonce);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_associate_neonaes(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                               sz_size_t length) {
    sz_aes256_gcm_associate_neonaes_(&encryptor->state, text, length);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_update_neonaes(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                            sz_size_t length, sz_ptr_t output) {
    sz_aes256_gcm_transform_neonaes_(&encryptor->state, text, length, output, sz_aes256_gcm_encrypting_k);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_digest_neonaes(sz_aes256_gcm_encryptor_t const *encryptor,
                                                            sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_digest_neonaes_(&encryptor->state, tag);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_init_neonaes(sz_aes256_gcm_decryptor_t *decryptor,
                                                          sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_aes256_gcm_begin_neonaes_(&decryptor->state, key, nonce);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_associate_neonaes(sz_aes256_gcm_decryptor_t *decryptor, sz_cptr_t text,
                                                               sz_size_t length) {
    sz_aes256_gcm_associate_neonaes_(&decryptor->state, text, length);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_update_unverified_neonaes(sz_aes256_gcm_decryptor_t *decryptor,
                                                                       sz_cptr_t text, sz_size_t length,
                                                                       sz_ptr_t output) {
    sz_aes256_gcm_transform_neonaes_(&decryptor->state, text, length, output, sz_aes256_gcm_decrypting_k);
}

SZ_API_COMPTIME sz_status_t sz_aes256_gcm_decryptor_verify_neonaes(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                   sz_u8_t const tag[sz_at_least_(16)]) {
    sz_u128_vec_t expected_vec;
    sz_aes256_gcm_digest_neonaes_(&decryptor->state, expected_vec.u8s);
    return sz_aes256_tag_equal_neonaes_(expected_vec.u8s, tag) == sz_true_k ? sz_success_k : sz_authentication_failed_k;
}

#pragma endregion // Streaming Interface

#pragma region One Shot Interface

SZ_API_COMPTIME void sz_aes256_gcm_encrypt_neonaes(sz_aes256_gcm_key_t const *key,
                                                   sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                   sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                   sz_ptr_t output, sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_encryptor_t encryptor;
    sz_aes256_gcm_encryptor_init_neonaes(&encryptor, key, nonce);
    if (associated_length) sz_aes256_gcm_encryptor_associate_neonaes(&encryptor, associated, associated_length);
    sz_aes256_gcm_encryptor_update_neonaes(&encryptor, text, length, output);
    sz_aes256_gcm_encryptor_digest_neonaes(&encryptor, tag);
    sz_aes256_gcm_state_scrub_neonaes_(&encryptor.state);
}

SZ_API_COMPTIME sz_status_t sz_aes256_gcm_decrypt_neonaes(sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                          sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                          sz_ptr_t output, sz_u8_t const tag[sz_at_least_(16)]) {
    sz_aes256_gcm_decryptor_t decryptor;
    sz_status_t verdict;
    sz_aes256_gcm_decryptor_init_neonaes(&decryptor, key, nonce);
    if (associated_length) sz_aes256_gcm_decryptor_associate_neonaes(&decryptor, associated, associated_length);
    sz_aes256_gcm_decryptor_update_unverified_neonaes(&decryptor, text, length, output);
    verdict = sz_aes256_gcm_decryptor_verify_neonaes(&decryptor, tag);
    sz_aes256_gcm_state_scrub_neonaes_(&decryptor.state);

    // A caller who drops the status must still be unable to act on forged plaintext.
    if (verdict != sz_success_k) sz_fill(output, length, 0);
    return verdict;
}

#pragma endregion // One Shot Interface

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEONAES

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_CIPHER_NEONAES_H_
