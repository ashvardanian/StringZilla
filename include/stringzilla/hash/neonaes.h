/**
 *  @brief NEON + AES backend for string hashing and checksums.
 *  @file include/stringzilla/hash/neonaes.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_NEONAES_H_
#define STRINGZILLA_HASH_NEONAES_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/hash/serial.h"

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

/**
 *  @brief Emulates the Intel's AES-NI `AESENC` instruction on Arm NEON.
 *  @see "Emulating x86 AES Intrinsics on ARMv8-A" by Michael Brase:
 *       https://blog.michaelbrase.com/2018/05/08/emulating-x86-aes-intrinsics-on-armv8-a/
 */
SZ_INTERNAL uint8x16_t sz_emulate_aesenc_u8x16_neon_(uint8x16_t state_u8x16, uint8x16_t round_key_u8x16) {
    return veorq_u8(vaesmcq_u8(vaeseq_u8(state_u8x16, vdupq_n_u8(0))), round_key_u8x16);
}

/**
 *  @brief Emulates the Intel's AES-NI `AESENC` instruction on Arm NEON, operating on u64x2 registers.
 *
 *  @param state_u64x2 128-bit AES state represented as two 64-bit lanes.
 *  @param round_key_u64x2 128-bit round key represented as two 64-bit lanes.
 *  @return AES-encrypted 128-bit result as two 64-bit lanes.
 */
SZ_INTERNAL uint64x2_t sz_emulate_aesenc_u64x2_neon_(uint64x2_t state_u64x2, uint64x2_t round_key_u64x2) {
    return vreinterpretq_u64_u8(               //
        sz_emulate_aesenc_u8x16_neon_(         //
            vreinterpretq_u8_u64(state_u64x2), //
            vreinterpretq_u8_u64(round_key_u64x2)));
}

/**
 *  @brief Initializes the minimal hash state using NEON with the given seed.
 *
 *  @param state Pointer to the minimal hash state to initialize.
 *  @param seed 64-bit seed value for the hash.
 */
SZ_INTERNAL void sz_hash_minimal_init_neon_(sz_hash_minimal_t_ *state, sz_u64_t seed) {

    // The key is made from the seed and half of it will be mixed with the length in the end
    uint64x2_t seed_u64x2 = vdupq_n_u64(seed);
    state->key.u64x2 = seed_u64x2;

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    uint64x2_t const pi0_u64x2 = vld1q_u64(pi);
    uint64x2_t const pi1_u64x2 = vld1q_u64(pi + 8);
    uint64x2_t aes_state_key = veorq_u64(seed_u64x2, pi0_u64x2);
    uint64x2_t sum_state_key = veorq_u64(seed_u64x2, pi1_u64x2);

    // The first 128 bits of the "sum" and "AES" blocks are the same for the "minimal" and full state
    state->aes.u64x2 = aes_state_key;
    state->sum.u64x2 = sum_state_key;
}

/**
 *  @brief Finalizes the minimal hash state and returns a 64-bit digest.
 *
 *  @param state Pointer to the minimal hash state to finalize.
 *  @param length Total number of bytes that were hashed.
 *  @return 64-bit hash digest.
 */
SZ_INTERNAL sz_u64_t sz_hash_minimal_finalize_neon_(sz_hash_minimal_t_ const *state, sz_size_t length) {
    // Mix the length into the key
    uint64x2_t key_with_length_u64x2 = vaddq_u64(state->key.u64x2, vsetq_lane_u64(length, vdupq_n_u64(0), 0));
    // Combine the "sum" and the "AES" blocks
    uint8x16_t mixed_u8x16 = sz_emulate_aesenc_u8x16_neon_(state->sum.u8x16, state->aes.u8x16);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    uint8x16_t final_mixed_u8x16 = sz_emulate_aesenc_u8x16_neon_(
        sz_emulate_aesenc_u8x16_neon_(mixed_u8x16, vreinterpretq_u8_u64(key_with_length_u64x2)), mixed_u8x16);
    // Extract the low 64 bits
    return vgetq_lane_u64(vreinterpretq_u64_u8(final_mixed_u8x16), 0);
}

/**
 *  @brief Feeds one 16-byte block into the minimal hash state.
 *
 *  @param state Pointer to the minimal hash state to update.
 *  @param block_u8x16 16-byte input block as a NEON register.
 */
SZ_INTERNAL void sz_hash_minimal_update_neon_(sz_hash_minimal_t_ *state, uint8x16_t block_u8x16) {
    uint8x16_t const order_u8x16 = vld1q_u8(sz_hash_u8x16x4_shuffle_());
    state->aes.u8x16 = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16, block_u8x16);
    uint8x16_t sum_shuffled_u8x16 = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2), order_u8x16);
    state->sum.u64x2 = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled_u8x16), vreinterpretq_u64_u8(block_u8x16));
}

SZ_PUBLIC void sz_hash_state_init_neonaes(sz_hash_state_t *state, sz_u64_t seed) {
    // The key is made from the seed and half of it will be mixed with the length in the end
    uint64x2_t seed_u64x2 = vdupq_n_u64(seed);
    vst1q_u64((sz_u64_t *)state->key, seed_u64x2);

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    for (int lane_index = 0; lane_index < 4; ++lane_index)
        vst1q_u8(state->aes + lane_index * 16,
                 vreinterpretq_u8_u64(veorq_u64(seed_u64x2, vld1q_u64(pi + lane_index * 2))));
    for (int lane_index = 0; lane_index < 4; ++lane_index)
        vst1q_u8(state->sum + lane_index * 16,
                 vreinterpretq_u8_u64(veorq_u64(seed_u64x2, vld1q_u64(pi + lane_index * 2 + 8))));

    // The inputs are zeroed out at the beginning
    uint8x16_t zeros_u8x16 = vdupq_n_u8(0);
    for (int lane_index = 0; lane_index < 4; ++lane_index) vst1q_u8(state->ins + lane_index * 16, zeros_u8x16);
    state->ins_length = 0;
}

/**
 *  @brief Mixes the current 64-byte input block into the internal hash state using NEON AES.
 *
 *  @param state Pointer to the internal hash state to update in place.
 */
SZ_INTERNAL void sz_hash_state_update_neon_(sz_hash_state_internal_t_ *state) {
    uint8x16_t const order_u8x16 = vld1q_u8(sz_hash_u8x16x4_shuffle_());
    state->aes.u8x16s[0] = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[0], state->ins.u8x16s[0]);
    uint8x16_t sum_shuffled_0_u8x16 = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[0]), order_u8x16);
    state->sum.u64x2s[0] = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled_0_u8x16), state->ins.u64x2s[0]);
    state->aes.u8x16s[1] = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[1], state->ins.u8x16s[1]);
    uint8x16_t sum_shuffled_1_u8x16 = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[1]), order_u8x16);
    state->sum.u64x2s[1] = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled_1_u8x16), state->ins.u64x2s[1]);
    state->aes.u8x16s[2] = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[2], state->ins.u8x16s[2]);
    uint8x16_t sum_shuffled_2_u8x16 = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[2]), order_u8x16);
    state->sum.u64x2s[2] = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled_2_u8x16), state->ins.u64x2s[2]);
    state->aes.u8x16s[3] = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[3], state->ins.u8x16s[3]);
    uint8x16_t sum_shuffled_3_u8x16 = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[3]), order_u8x16);
    state->sum.u64x2s[3] = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled_3_u8x16), state->ins.u64x2s[3]);
}

/**
 *  @brief Finalizes the full internal hash state and returns a 64-bit digest.
 *
 *  @param state Pointer to the internal hash state to finalize.
 *  @return 64-bit hash digest.
 */
SZ_INTERNAL sz_u64_t sz_hash_state_finalize_neon_(sz_hash_state_internal_t_ const *state) {
    // Mix the length into the key
    uint64x2_t key_with_length_u64x2 = vaddq_u64(state->key.u64x2,
                                                 vsetq_lane_u64(state->ins_length, vdupq_n_u64(0), 0));

    // Fold the deferred final block (still buffered in `ins` - a full 64 bytes or a zero-padded tail) into each
    // lane. Folding the last block here, rather than in `update`, lets both one-shot `sz_hash` and the streaming
    // digest defer it and share this single finalization with no state copy.
    uint8x16_t const order_u8x16 = vld1q_u8(sz_hash_u8x16x4_shuffle_());
    uint8x16_t aes_0_u8x16 = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[0], state->ins.u8x16s[0]);
    uint8x16_t aes_1_u8x16 = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[1], state->ins.u8x16s[1]);
    uint8x16_t aes_2_u8x16 = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[2], state->ins.u8x16s[2]);
    uint8x16_t aes_3_u8x16 = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[3], state->ins.u8x16s[3]);
    uint64x2_t sum_0_u64x2 = vaddq_u64(
        vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[0]), order_u8x16)),
        state->ins.u64x2s[0]);
    uint64x2_t sum_1_u64x2 = vaddq_u64(
        vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[1]), order_u8x16)),
        state->ins.u64x2s[1]);
    uint64x2_t sum_2_u64x2 = vaddq_u64(
        vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[2]), order_u8x16)),
        state->ins.u64x2s[2]);
    uint64x2_t sum_3_u64x2 = vaddq_u64(
        vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[3]), order_u8x16)),
        state->ins.u64x2s[3]);

    // Combine the "sum" and the "AES" blocks
    uint8x16_t mixed_0_u8x16 = sz_emulate_aesenc_u8x16_neon_(vreinterpretq_u8_u64(sum_0_u64x2), aes_0_u8x16);
    uint8x16_t mixed_1_u8x16 = sz_emulate_aesenc_u8x16_neon_(vreinterpretq_u8_u64(sum_1_u64x2), aes_1_u8x16);
    uint8x16_t mixed_2_u8x16 = sz_emulate_aesenc_u8x16_neon_(vreinterpretq_u8_u64(sum_2_u64x2), aes_2_u8x16);
    uint8x16_t mixed_3_u8x16 = sz_emulate_aesenc_u8x16_neon_(vreinterpretq_u8_u64(sum_3_u64x2), aes_3_u8x16);
    // Combine the mixed registers
    uint8x16_t mixed_01_u8x16 = sz_emulate_aesenc_u8x16_neon_(mixed_0_u8x16, mixed_1_u8x16);
    uint8x16_t mixed_23_u8x16 = sz_emulate_aesenc_u8x16_neon_(mixed_2_u8x16, mixed_3_u8x16);
    uint8x16_t mixed_u8x16 = sz_emulate_aesenc_u8x16_neon_(mixed_01_u8x16, mixed_23_u8x16);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    uint8x16_t final_mixed_u8x16 = sz_emulate_aesenc_u8x16_neon_(
        sz_emulate_aesenc_u8x16_neon_(mixed_u8x16, vreinterpretq_u8_u64(key_with_length_u64x2)), mixed_u8x16);
    // Extract the low 64 bits
    return vgetq_lane_u64(vreinterpretq_u64_u8(final_mixed_u8x16), 0);
}

SZ_PUBLIC void sz_hash_state_update_neonaes(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {

    // `ins` is exactly one 64-byte block (four NEON lanes), so buffering is just: track how many bytes it holds,
    // absorb it only once it becomes interior (more bytes arrive - the deferral `digest` needs to choose
    // minimal/full by total length), and append incoming bytes with a single contiguous copy. The deferred
    // trailing block reads back as `ins_length % 64 == 0 && ins_length != 0`; treat that as `buffered == 64`. The
    // append touches only `[buffered, buffered+take)`, and we re-zero `ins` after each absorb, so the high lanes
    // stay zero-padded for `finalize` to fold. NEON has no masked load/store, so the copy is a plain byte loop.
    uint8x16_t const order_u8x16 = vld1q_u8(sz_hash_u8x16x4_shuffle_());
    sz_size_t buffered = state->ins_length % 64;
    if (buffered == 0 && state->ins_length) buffered = 64;
    while (length) {
        if (buffered == 64) { // the deferred block is now interior - absorb it and re-zero the buffer
            uint8x16_t aes_0 = vld1q_u8(state->aes + 0);
            uint8x16_t aes_1 = vld1q_u8(state->aes + 16);
            uint8x16_t aes_2 = vld1q_u8(state->aes + 32);
            uint8x16_t aes_3 = vld1q_u8(state->aes + 48);
            uint8x16_t ins_0 = vld1q_u8(state->ins + 0);
            uint8x16_t ins_1 = vld1q_u8(state->ins + 16);
            uint8x16_t ins_2 = vld1q_u8(state->ins + 32);
            uint8x16_t ins_3 = vld1q_u8(state->ins + 48);
            vst1q_u8(state->aes + 0, sz_emulate_aesenc_u8x16_neon_(aes_0, ins_0));
            vst1q_u8(state->aes + 16, sz_emulate_aesenc_u8x16_neon_(aes_1, ins_1));
            vst1q_u8(state->aes + 32, sz_emulate_aesenc_u8x16_neon_(aes_2, ins_2));
            vst1q_u8(state->aes + 48, sz_emulate_aesenc_u8x16_neon_(aes_3, ins_3));
            uint64x2_t sum_0 = vld1q_u64((sz_u64_t const *)state->sum + 0);
            uint64x2_t sum_1 = vld1q_u64((sz_u64_t const *)state->sum + 2);
            uint64x2_t sum_2 = vld1q_u64((sz_u64_t const *)state->sum + 4);
            uint64x2_t sum_3 = vld1q_u64((sz_u64_t const *)state->sum + 6);
            vst1q_u64((sz_u64_t *)state->sum + 0,
                      vaddq_u64(vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(sum_0), order_u8x16)),
                                vreinterpretq_u64_u8(ins_0)));
            vst1q_u64((sz_u64_t *)state->sum + 2,
                      vaddq_u64(vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(sum_1), order_u8x16)),
                                vreinterpretq_u64_u8(ins_1)));
            vst1q_u64((sz_u64_t *)state->sum + 4,
                      vaddq_u64(vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(sum_2), order_u8x16)),
                                vreinterpretq_u64_u8(ins_2)));
            vst1q_u64((sz_u64_t *)state->sum + 6,
                      vaddq_u64(vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(sum_3), order_u8x16)),
                                vreinterpretq_u64_u8(ins_3)));
            uint8x16_t const zeros_u8x16 = vdupq_n_u8(0);
            vst1q_u8(state->ins + 0, zeros_u8x16);
            vst1q_u8(state->ins + 16, zeros_u8x16);
            vst1q_u8(state->ins + 32, zeros_u8x16);
            vst1q_u8(state->ins + 48, zeros_u8x16);
            buffered = 0;
        }
        sz_size_t const take = sz_min_of_two(length, (sz_size_t)64 - buffered);
        for (sz_size_t byte_index = 0; byte_index < take; ++byte_index) state->ins[buffered + byte_index] = text[byte_index];
        buffered += take, text += take, length -= take, state->ins_length += take;
    }
}

SZ_PUBLIC sz_u64_t sz_hash_state_digest_neonaes(sz_hash_state_t const *state) {
    // This whole function is identical to Haswell.
    sz_size_t length = state->ins_length;
    // Inputs longer than one block fold through the full four-lane state, where the deferred final block buffered
    // in `ins` is folded by `sz_hash_state_finalize_neon_`. A length of exactly 64 uses the minimal (<=64) path
    // below - matching one-shot `sz_hash`, whose `length <= 64` ladder also stays minimal.
    if (length > 64) {
        // Load the public state into the internal representation for finalization
        sz_hash_state_internal_t_ internal_state;
        internal_state.aes.u8x16s[0] = vld1q_u8(state->aes + 0);
        internal_state.aes.u8x16s[1] = vld1q_u8(state->aes + 16);
        internal_state.aes.u8x16s[2] = vld1q_u8(state->aes + 32);
        internal_state.aes.u8x16s[3] = vld1q_u8(state->aes + 48);
        internal_state.sum.u8x16s[0] = vld1q_u8(state->sum + 0);
        internal_state.sum.u8x16s[1] = vld1q_u8(state->sum + 16);
        internal_state.sum.u8x16s[2] = vld1q_u8(state->sum + 32);
        internal_state.sum.u8x16s[3] = vld1q_u8(state->sum + 48);
        internal_state.ins.u8x16s[0] = vld1q_u8(state->ins + 0);
        internal_state.ins.u8x16s[1] = vld1q_u8(state->ins + 16);
        internal_state.ins.u8x16s[2] = vld1q_u8(state->ins + 32);
        internal_state.ins.u8x16s[3] = vld1q_u8(state->ins + 48);
        internal_state.key.u8x16 = vld1q_u8(state->key);
        internal_state.ins_length = state->ins_length;
        return sz_hash_state_finalize_neon_(&internal_state);
    }

    // Switch back to a smaller "minimal" state for small inputs
    sz_hash_minimal_t_ minimal_state;
    minimal_state.key.u8x16 = vld1q_u8(state->key);
    minimal_state.aes.u8x16 = vld1q_u8(state->aes);
    minimal_state.sum.u8x16 = vld1q_u8(state->sum);

    // The logic is different depending on the length of the input
    uint8x16_t const *ins_blocks_u8x16 = (uint8x16_t const *)state->ins;
    if (length <= 16) {
        sz_hash_minimal_update_neon_(&minimal_state, ins_blocks_u8x16[0]);
        return sz_hash_minimal_finalize_neon_(&minimal_state, length);
    }
    else if (length <= 32) {
        sz_hash_minimal_update_neon_(&minimal_state, ins_blocks_u8x16[0]);
        sz_hash_minimal_update_neon_(&minimal_state, ins_blocks_u8x16[1]);
        return sz_hash_minimal_finalize_neon_(&minimal_state, length);
    }
    else if (length <= 48) {
        sz_hash_minimal_update_neon_(&minimal_state, ins_blocks_u8x16[0]);
        sz_hash_minimal_update_neon_(&minimal_state, ins_blocks_u8x16[1]);
        sz_hash_minimal_update_neon_(&minimal_state, ins_blocks_u8x16[2]);
        return sz_hash_minimal_finalize_neon_(&minimal_state, length);
    }
    else {
        sz_hash_minimal_update_neon_(&minimal_state, ins_blocks_u8x16[0]);
        sz_hash_minimal_update_neon_(&minimal_state, ins_blocks_u8x16[1]);
        sz_hash_minimal_update_neon_(&minimal_state, ins_blocks_u8x16[2]);
        sz_hash_minimal_update_neon_(&minimal_state, ins_blocks_u8x16[3]);
        return sz_hash_minimal_finalize_neon_(&minimal_state, length);
    }
}

SZ_PUBLIC SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_neonaes(sz_cptr_t text, sz_size_t length, sz_u64_t seed) {
    if (length <= 16) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data_vec;
        if (length == 16) { data_vec.u8x16 = vld1q_u8((sz_u8_t const *)text); } // Full in-bounds block
        else {
            data_vec.u8x16 = vdupq_n_u8(0);
            for (sz_size_t byte_index = 0; byte_index < length; ++byte_index)
                data_vec.u8s[byte_index] = text[byte_index]; // Variable partial tail
        }
        sz_hash_minimal_update_neon_(&state, data_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else if (length <= 32) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data_0_vec, data_1_vec;
        data_0_vec.u8x16 = vld1q_u8((sz_u8_t const *)(text + 0));
        data_1_vec.u8x16 = vld1q_u8((sz_u8_t const *)(text + length - 16));
        // Let's shift the data within the register to de-interleave the bytes.
        sz_hash_shift_in_register_serial_(&data_1_vec, (int)(32 - length)); //! `vextq_u8` requires immediates
        sz_hash_minimal_update_neon_(&state, data_0_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data_1_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else if (length <= 48) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data_0_vec, data_1_vec, data_2_vec;
        data_0_vec.u8x16 = vld1q_u8((sz_u8_t const *)(text + 0));
        data_1_vec.u8x16 = vld1q_u8((sz_u8_t const *)(text + 16));
        data_2_vec.u8x16 = vld1q_u8((sz_u8_t const *)(text + length - 16));
        // Let's shift the data within the register to de-interleave the bytes.
        sz_hash_shift_in_register_serial_(&data_2_vec, (int)(48 - length)); //! `vextq_u8` requires immediates
        sz_hash_minimal_update_neon_(&state, data_0_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data_1_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data_2_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else if (length <= 64) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data_0_vec, data_1_vec, data_2_vec, data_3_vec;
        data_0_vec.u8x16 = vld1q_u8((sz_u8_t const *)(text + 0));
        data_1_vec.u8x16 = vld1q_u8((sz_u8_t const *)(text + 16));
        data_2_vec.u8x16 = vld1q_u8((sz_u8_t const *)(text + 32));
        data_3_vec.u8x16 = vld1q_u8((sz_u8_t const *)(text + length - 16));
        // Let's shift the data within the register to de-interleave the bytes.
        sz_hash_shift_in_register_serial_(&data_3_vec, (int)(64 - length)); //! `vextq_u8` requires immediates
        sz_hash_minimal_update_neon_(&state, data_0_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data_1_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data_2_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data_3_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else {
        sz_align_(64) sz_hash_state_internal_t_ state;
        sz_hash_state_init_neonaes((sz_hash_state_t *)&state, seed);

        // Absorb every full 64-byte block EXCEPT the last; the final block (a full 64 or a partial tail) stays
        // buffered in `ins` for `sz_hash_state_finalize_neon_` to fold - the same deferral the streaming path uses.
        for (; state.ins_length + 64 < length; state.ins_length += 64) {
            state.ins.u8x16s[0] = vld1q_u8((sz_u8_t const *)(text + state.ins_length + 0));
            state.ins.u8x16s[1] = vld1q_u8((sz_u8_t const *)(text + state.ins_length + 16));
            state.ins.u8x16s[2] = vld1q_u8((sz_u8_t const *)(text + state.ins_length + 32));
            state.ins.u8x16s[3] = vld1q_u8((sz_u8_t const *)(text + state.ins_length + 48));
            sz_hash_state_update_neon_(&state);
        }
        // Stage the final [ins_length, length) bytes (1..64) into a zeroed buffer; finalize folds them.
        sz_size_t const tail_length = length - state.ins_length;
        state.ins.u8x16s[0] = vdupq_n_u8(0);
        state.ins.u8x16s[1] = vdupq_n_u8(0);
        state.ins.u8x16s[2] = vdupq_n_u8(0);
        state.ins.u8x16s[3] = vdupq_n_u8(0);
        for (sz_size_t byte_index = 0; byte_index < tail_length; ++byte_index)
            state.ins.u8s[byte_index] = text[state.ins_length + byte_index];
        state.ins_length = length;
        return sz_hash_state_finalize_neon_(&state);
    }
}

/**
 *  @brief Splits a short (<= 64B) input into up to four 128-bit text-lanes using NEON loads.
 *         Mirrors the loading ladder of `sz_hash_neonaes`: full `vld1q_u8` loads for complete lanes
 *         and one overlapping load + in-register shift for the partial tail; byte-by-byte only for
 *         the `< 16` case, where a 16-byte NEON load could read past the input.
 *  @return The number of populated text-lanes (1..4).
 */
SZ_INTERNAL sz_size_t sz_hash_multiseed_prepare_neon_(sz_cptr_t text, sz_size_t length, sz_u512_vec_t *text_lanes) {
    if (length <= 16) {
        sz_u128_vec_t lane;
        if (length == 16) { lane.u8x16 = vld1q_u8((sz_u8_t const *)text); }
        else {
            lane.u8x16 = vdupq_n_u8(0);
            for (sz_size_t byte_index = 0; byte_index < length; ++byte_index) lane.u8s[byte_index] = text[byte_index];
        }
        text_lanes->u128s[0] = lane;
        return 1;
    }
    sz_size_t const text_lanes_count = sz_size_divide_round_up(length, 16);
    for (sz_size_t lane_index = 0; lane_index + 1 < text_lanes_count; ++lane_index)
        text_lanes->u128s[lane_index].u8x16 = vld1q_u8((sz_u8_t const *)(text + lane_index * 16));
    // De-interleave the partial tail lane with a single table lookup: output byte i pulls input byte
    // (i + shift); indices that run past 15 make `vqtbl1q_u8` emit a zero - a clean in-register right
    // shift, without the compile-time immediate that `vextq_u8` would demand for a data-dependent shift.
    sz_align_(16) static sz_u8_t const lane_iota[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint8x16_t const tail = vld1q_u8((sz_u8_t const *)(text + length - 16));
    uint8x16_t const shift = vdupq_n_u8((sz_u8_t)(text_lanes_count * 16 - length));
    text_lanes->u128s[text_lanes_count - 1].u8x16 = vqtbl1q_u8(tail, vaddq_u8(vld1q_u8(lane_iota), shift));
    return text_lanes_count;
}

SZ_PUBLIC void sz_hash_multiseed_neonaes(sz_cptr_t text, sz_size_t length,             //
                                         sz_u64_t const *seeds, sz_size_t seeds_count, //
                                         sz_u64_t *hashes) {
    // Trivial counts don't benefit from sharing a normalization pass - go straight to the single-shot.
    if (seeds_count == 0) return;
    if (seeds_count == 1) {
        hashes[0] = sz_hash_neonaes(text, length, seeds[0]);
        return;
    }
    // NEON `AESE` is per-128-bit register, so there is no cross-lane packing; the win is the shared
    // normalization pass plus interleaving two independent AES chains to hide the AES latency.
    if (length > 64) {
        for (sz_size_t seed_index = 0; seed_index < seeds_count; ++seed_index)
            hashes[seed_index] = sz_hash_neonaes(text, length, seeds[seed_index]);
        return;
    }

    sz_u512_vec_t text_lanes;
    sz_size_t const text_lanes_count = sz_hash_multiseed_prepare_neon_(text, length, &text_lanes);

    sz_size_t seed_index = 0;
    for (; seed_index + 2 <= seeds_count; seed_index += 2) {
        sz_hash_minimal_t_ state0, state1;
        sz_hash_minimal_init_neon_(&state0, seeds[seed_index + 0]);
        sz_hash_minimal_init_neon_(&state1, seeds[seed_index + 1]);
        for (sz_size_t lane_index = 0; lane_index < text_lanes_count; ++lane_index) {
            sz_hash_minimal_update_neon_(&state0, text_lanes.u128s[lane_index].u8x16);
            sz_hash_minimal_update_neon_(&state1, text_lanes.u128s[lane_index].u8x16);
        }
        hashes[seed_index + 0] = sz_hash_minimal_finalize_neon_(&state0, length);
        hashes[seed_index + 1] = sz_hash_minimal_finalize_neon_(&state1, length);
    }
    if (seed_index < seeds_count) {
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seeds[seed_index]);
        for (sz_size_t lane_index = 0; lane_index < text_lanes_count; ++lane_index)
            sz_hash_minimal_update_neon_(&state, text_lanes.u128s[lane_index].u8x16);
        hashes[seed_index] = sz_hash_minimal_finalize_neon_(&state, length);
    }
}

SZ_PUBLIC void sz_fill_random_neonaes(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_u64_t const *pi_pointer = sz_hash_pi_constants_();
    if (length <= 16) {
        uint64x2_t input_u64x2 = vdupq_n_u64(nonce);
        uint64x2_t pi_u64x2 = vld1q_u64(pi_pointer);
        uint64x2_t key_u64x2 = veorq_u64(vdupq_n_u64(nonce), pi_u64x2);
        uint64x2_t generated_u64x2 = sz_emulate_aesenc_u64x2_neon_(input_u64x2, key_u64x2);
        // Now the tricky part is outputting this data to the user-supplied buffer
        // without masked writes, like in AVX-512.
        if (length >= 16) {
            vst1q_u8((sz_u8_t *)text, vreinterpretq_u8_u64(generated_u64x2)); // Full block
        }
        else
            for (sz_size_t byte_index = 0; byte_index < length; ++byte_index)
                text[byte_index] = ((sz_u8_t *)&generated_u64x2)[byte_index]; // Variable partial tail
    }
    // Assuming the YMM register contains two 128-bit blocks, the input to the generator
    // will be more complex, containing the sum of the nonce and the block number.
    else if (length <= 32) {
        uint64x2_t inputs_u64x2[2], pis_u64x2[2], keys_u64x2[2], generated_u64x2[2];
        inputs_u64x2[0] = vdupq_n_u64(nonce + 0);
        inputs_u64x2[1] = vdupq_n_u64(nonce + 1);
        pis_u64x2[0] = vld1q_u64(pi_pointer + 0);
        pis_u64x2[1] = vld1q_u64(pi_pointer + 2);
        keys_u64x2[0] = veorq_u64(vdupq_n_u64(nonce), pis_u64x2[0]);
        keys_u64x2[1] = veorq_u64(vdupq_n_u64(nonce), pis_u64x2[1]);
        generated_u64x2[0] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[0], keys_u64x2[0]);
        generated_u64x2[1] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[1], keys_u64x2[1]);
        // The first store can easily be vectorized, but the second can be serial for now
        vst1q_u64((sz_u64_t *)(text), generated_u64x2[0]);
        if (length >= 32) {
            vst1q_u8((sz_u8_t *)(text + 16), vreinterpretq_u8_u64(generated_u64x2[1])); // Full block
        }
        else
            for (sz_size_t byte_index = 16; byte_index < length; ++byte_index)
                text[byte_index] = ((sz_u8_t *)&generated_u64x2[1])[byte_index - 16]; // Partial tail
    }
    // The last special case we handle outside of the primary loop is for buffers up to 64 bytes long.
    else if (length <= 48) {
        uint64x2_t inputs_u64x2[3], pis_u64x2[3], keys_u64x2[3], generated_u64x2[3];
        inputs_u64x2[0] = vdupq_n_u64(nonce);
        inputs_u64x2[1] = vdupq_n_u64(nonce + 1);
        inputs_u64x2[2] = vdupq_n_u64(nonce + 2);
        pis_u64x2[0] = vld1q_u64(pi_pointer + 0);
        pis_u64x2[1] = vld1q_u64(pi_pointer + 2);
        pis_u64x2[2] = vld1q_u64(pi_pointer + 4);
        keys_u64x2[0] = veorq_u64(vdupq_n_u64(nonce), pis_u64x2[0]);
        keys_u64x2[1] = veorq_u64(vdupq_n_u64(nonce), pis_u64x2[1]);
        keys_u64x2[2] = veorq_u64(vdupq_n_u64(nonce), pis_u64x2[2]);
        generated_u64x2[0] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[0], keys_u64x2[0]);
        generated_u64x2[1] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[1], keys_u64x2[1]);
        generated_u64x2[2] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[2], keys_u64x2[2]);
        // The first store can easily be vectorized, but the second can be serial for now
        vst1q_u64((sz_u64_t *)(text + 0), generated_u64x2[0]);
        vst1q_u64((sz_u64_t *)(text + 16), generated_u64x2[1]);
        if (length >= 48) {
            vst1q_u8((sz_u8_t *)(text + 32), vreinterpretq_u8_u64(generated_u64x2[2])); // Full block
        }
        else
            for (sz_size_t byte_index = 32; byte_index < length; ++byte_index)
                text[byte_index] = ((sz_u8_t *)generated_u64x2)[byte_index]; // Partial tail
    }
    // The final part of the function is the primary loop, which processes the buffer in 64-byte chunks.
    else {
        uint64x2_t inputs_u64x2[4], pis_u64x2[4], keys_u64x2[4], generated_u64x2[4];
        inputs_u64x2[0] = vdupq_n_u64(nonce + 0);
        inputs_u64x2[1] = vdupq_n_u64(nonce + 1);
        inputs_u64x2[2] = vdupq_n_u64(nonce + 2);
        inputs_u64x2[3] = vdupq_n_u64(nonce + 3);
        // Load parts of PI into the registers
        pis_u64x2[0] = vld1q_u64(pi_pointer + 0);
        pis_u64x2[1] = vld1q_u64(pi_pointer + 2);
        pis_u64x2[2] = vld1q_u64(pi_pointer + 4);
        pis_u64x2[3] = vld1q_u64(pi_pointer + 6);
        // XOR the nonce with the PI constants
        keys_u64x2[0] = veorq_u64(vdupq_n_u64(nonce), pis_u64x2[0]);
        keys_u64x2[1] = veorq_u64(vdupq_n_u64(nonce), pis_u64x2[1]);
        keys_u64x2[2] = veorq_u64(vdupq_n_u64(nonce), pis_u64x2[2]);
        keys_u64x2[3] = veorq_u64(vdupq_n_u64(nonce), pis_u64x2[3]);

        // Produce the output, fixing the key and enumerating input chunks.
        sz_size_t byte_index = 0;
        uint64x2_t const increment_u64x2 = vdupq_n_u64(4);
        for (; byte_index + 64 <= length; byte_index += 64) {
            generated_u64x2[0] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[0], keys_u64x2[0]);
            generated_u64x2[1] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[1], keys_u64x2[1]);
            generated_u64x2[2] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[2], keys_u64x2[2]);
            generated_u64x2[3] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[3], keys_u64x2[3]);
            vst1q_u64((sz_u64_t *)(text + byte_index + 0), generated_u64x2[0]);
            vst1q_u64((sz_u64_t *)(text + byte_index + 16), generated_u64x2[1]);
            vst1q_u64((sz_u64_t *)(text + byte_index + 32), generated_u64x2[2]);
            vst1q_u64((sz_u64_t *)(text + byte_index + 48), generated_u64x2[3]);
            inputs_u64x2[0] = vaddq_u64(inputs_u64x2[0], increment_u64x2);
            inputs_u64x2[1] = vaddq_u64(inputs_u64x2[1], increment_u64x2);
            inputs_u64x2[2] = vaddq_u64(inputs_u64x2[2], increment_u64x2);
            inputs_u64x2[3] = vaddq_u64(inputs_u64x2[3], increment_u64x2);
        }

        // Handle the tail of the buffer.
        {
            generated_u64x2[0] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[0], keys_u64x2[0]);
            generated_u64x2[1] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[1], keys_u64x2[1]);
            generated_u64x2[2] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[2], keys_u64x2[2]);
            generated_u64x2[3] = sz_emulate_aesenc_u64x2_neon_(inputs_u64x2[3], keys_u64x2[3]);
            // Spill full 16-byte blocks with vector stores, leaving only the partial (<16) tail scalar
            sz_size_t generated_index = 0;
            for (; byte_index + 16 <= length; byte_index += 16, generated_index += 16)
                vst1q_u8((sz_u8_t *)(text + byte_index),
                         vld1q_u8((sz_u8_t const *)generated_u64x2 + generated_index)); // Full block
            for (; byte_index < length; ++byte_index, ++generated_index)
                text[byte_index] = ((sz_u8_t *)generated_u64x2)[generated_index]; // Variable partial tail
        }
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEONAES

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_NEONAES_H_
