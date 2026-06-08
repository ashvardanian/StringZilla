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
SZ_INTERNAL uint8x16_t sz_emulate_aesenc_u8x16_neon_(uint8x16_t state_vec, uint8x16_t round_key_vec) {
    return veorq_u8(vaesmcq_u8(vaeseq_u8(state_vec, vdupq_n_u8(0))), round_key_vec);
}

SZ_INTERNAL uint64x2_t sz_emulate_aesenc_u64x2_neon_(uint64x2_t state_vec, uint64x2_t round_key_vec) {
    return vreinterpretq_u64_u8(             //
        sz_emulate_aesenc_u8x16_neon_(       //
            vreinterpretq_u8_u64(state_vec), //
            vreinterpretq_u8_u64(round_key_vec)));
}

SZ_INTERNAL void sz_hash_minimal_init_neon_(sz_hash_minimal_t_ *state, sz_u64_t seed) {

    // The key is made from the seed and half of it will be mixed with the length in the end
    uint64x2_t seed_vec = vdupq_n_u64(seed);
    state->key.u64x2 = seed_vec;

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    uint64x2_t const pi0 = vld1q_u64(pi);
    uint64x2_t const pi1 = vld1q_u64(pi + 8);
    uint64x2_t k1 = veorq_u64(seed_vec, pi0);
    uint64x2_t k2 = veorq_u64(seed_vec, pi1);

    // The first 128 bits of the "sum" and "AES" blocks are the same for the "minimal" and full state
    state->aes.u64x2 = k1;
    state->sum.u64x2 = k2;
}

SZ_INTERNAL sz_u64_t sz_hash_minimal_finalize_neon_(sz_hash_minimal_t_ const *state, sz_size_t length) {
    // Mix the length into the key
    uint64x2_t key_with_length = vaddq_u64(state->key.u64x2, vsetq_lane_u64(length, vdupq_n_u64(0), 0));
    // Combine the "sum" and the "AES" blocks
    uint8x16_t mixed = sz_emulate_aesenc_u8x16_neon_(state->sum.u8x16, state->aes.u8x16);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    uint8x16_t mixed_in_register = sz_emulate_aesenc_u8x16_neon_(
        sz_emulate_aesenc_u8x16_neon_(mixed, vreinterpretq_u8_u64(key_with_length)), mixed);
    // Extract the low 64 bits
    return vgetq_lane_u64(vreinterpretq_u64_u8(mixed_in_register), 0);
}

SZ_INTERNAL void sz_hash_minimal_update_neon_(sz_hash_minimal_t_ *state, uint8x16_t block) {
    uint8x16_t const order = vld1q_u8(sz_hash_u8x16x4_shuffle_());
    state->aes.u8x16 = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16, block);
    uint8x16_t sum_shuffled = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2), order);
    state->sum.u64x2 = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled), vreinterpretq_u64_u8(block));
}

SZ_PUBLIC void sz_hash_state_init_neon(sz_hash_state_t *state, sz_u64_t seed) {
    // The key is made from the seed and half of it will be mixed with the length in the end
    uint64x2_t seed_vec = vdupq_n_u64(seed);
    vst1q_u64((sz_u64_t *)state->key, seed_vec);

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    for (int i = 0; i < 4; ++i)
        vst1q_u8(state->aes + i * 16, vreinterpretq_u8_u64(veorq_u64(seed_vec, vld1q_u64(pi + i * 2))));
    for (int i = 0; i < 4; ++i)
        vst1q_u8(state->sum + i * 16, vreinterpretq_u8_u64(veorq_u64(seed_vec, vld1q_u64(pi + i * 2 + 8))));

    // The inputs are zeroed out at the beginning
    uint8x16_t zeros = vdupq_n_u8(0);
    for (int i = 0; i < 4; ++i) vst1q_u8(state->ins + i * 16, zeros);
    state->ins_length = 0;
}

SZ_INTERNAL void sz_hash_state_update_neon_(sz_hash_state_internal_t_ *state) {
    uint8x16_t const order = vld1q_u8(sz_hash_u8x16x4_shuffle_());
    state->aes.u8x16s[0] = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[0], state->ins.u8x16s[0]);
    uint8x16_t sum_shuffled0 = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[0]), order);
    state->sum.u64x2s[0] = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled0), state->ins.u64x2s[0]);
    state->aes.u8x16s[1] = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[1], state->ins.u8x16s[1]);
    uint8x16_t sum_shuffled1 = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[1]), order);
    state->sum.u64x2s[1] = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled1), state->ins.u64x2s[1]);
    state->aes.u8x16s[2] = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[2], state->ins.u8x16s[2]);
    uint8x16_t sum_shuffled2 = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[2]), order);
    state->sum.u64x2s[2] = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled2), state->ins.u64x2s[2]);
    state->aes.u8x16s[3] = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[3], state->ins.u8x16s[3]);
    uint8x16_t sum_shuffled3 = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[3]), order);
    state->sum.u64x2s[3] = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled3), state->ins.u64x2s[3]);
}

SZ_INTERNAL sz_u64_t sz_hash_state_finalize_neon_(sz_hash_state_internal_t_ const *state) {
    // Mix the length into the key
    uint64x2_t key_with_length = vaddq_u64(state->key.u64x2, vsetq_lane_u64(state->ins_length, vdupq_n_u64(0), 0));
    // Combine the "sum" and the "AES" blocks
    uint8x16_t mixed0 = sz_emulate_aesenc_u8x16_neon_(state->sum.u8x16s[0], state->aes.u8x16s[0]);
    uint8x16_t mixed1 = sz_emulate_aesenc_u8x16_neon_(state->sum.u8x16s[1], state->aes.u8x16s[1]);
    uint8x16_t mixed2 = sz_emulate_aesenc_u8x16_neon_(state->sum.u8x16s[2], state->aes.u8x16s[2]);
    uint8x16_t mixed3 = sz_emulate_aesenc_u8x16_neon_(state->sum.u8x16s[3], state->aes.u8x16s[3]);
    // Combine the mixed registers
    uint8x16_t mixed01 = sz_emulate_aesenc_u8x16_neon_(mixed0, mixed1);
    uint8x16_t mixed23 = sz_emulate_aesenc_u8x16_neon_(mixed2, mixed3);
    uint8x16_t mixed = sz_emulate_aesenc_u8x16_neon_(mixed01, mixed23);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    uint8x16_t mixed_in_register = sz_emulate_aesenc_u8x16_neon_(
        sz_emulate_aesenc_u8x16_neon_(mixed, vreinterpretq_u8_u64(key_with_length)), mixed);
    // Extract the low 64 bits
    return vgetq_lane_u64(vreinterpretq_u64_u8(mixed_in_register), 0);
}

SZ_PUBLIC void sz_hash_state_update_neon(sz_hash_state_t *state_ptr, sz_cptr_t text, sz_size_t length) {

    // The worst usage pattern... that we should ironically handle first - is updating the state
    // with a very small chunk of data, potentially, one byte at a time. In such cases, we won't
    // even bother using NEON masked loads to avoid complexity.
    sz_size_t const current_block_index = state_ptr->ins_length / 64;
    sz_size_t const final_block_index = (state_ptr->ins_length + length) / 64;
    int const stays_in_the_block = current_block_index == final_block_index;
    int const fills_the_block = (state_ptr->ins_length + length) % 64 == 0;
    if (stays_in_the_block && !fills_the_block) {
        for (; length; --length, ++state_ptr->ins_length, ++text) state_ptr->ins[state_ptr->ins_length % 64] = *text;
        return;
    }

    // Now we know that our "text" parts will end up in different blocks.
    // It's a good idea to pull the state into registers, as well definitely mix them at block boundaries.
    sz_size_t const progress_in_block = state_ptr->ins_length % 64;
    sz_size_t const head_length = (64 - progress_in_block) % 64;
    sz_size_t const tail_length = (state_ptr->ins_length + length) % 64;
    sz_size_t const body_length = length - head_length - tail_length;
    sz_assert_(body_length % 64 == 0 && head_length < 64 && tail_length < 64);

    // Lets keep a local copy of the state for one or more updates.
    sz_align_(64) sz_hash_state_internal_t_ state;
    state.aes.u8x16s[0] = vld1q_u8((sz_u8_t const *)state_ptr->aes + 0);
    state.aes.u8x16s[1] = vld1q_u8((sz_u8_t const *)state_ptr->aes + 16);
    state.aes.u8x16s[2] = vld1q_u8((sz_u8_t const *)state_ptr->aes + 32);
    state.aes.u8x16s[3] = vld1q_u8((sz_u8_t const *)state_ptr->aes + 48);
    state.sum.u8x16s[0] = vld1q_u8((sz_u8_t const *)state_ptr->sum + 0);
    state.sum.u8x16s[1] = vld1q_u8((sz_u8_t const *)state_ptr->sum + 16);
    state.sum.u8x16s[2] = vld1q_u8((sz_u8_t const *)state_ptr->sum + 32);
    state.sum.u8x16s[3] = vld1q_u8((sz_u8_t const *)state_ptr->sum + 48);
    state.ins.u8x16s[0] = vld1q_u8((sz_u8_t const *)state_ptr->ins + 0);
    state.ins.u8x16s[1] = vld1q_u8((sz_u8_t const *)state_ptr->ins + 16);
    state.ins.u8x16s[2] = vld1q_u8((sz_u8_t const *)state_ptr->ins + 32);
    state.ins.u8x16s[3] = vld1q_u8((sz_u8_t const *)state_ptr->ins + 48);
    state.ins_length = state_ptr->ins_length;

    // Handle the head first, filling up the current block
    uint8x16_t const order = vld1q_u8(sz_hash_u8x16x4_shuffle_());
    if (head_length) {
        sz_ptr_t local_ptr = (sz_ptr_t)&state.ins.u8s[progress_in_block];
        sz_ptr_t const local_end = (sz_ptr_t)&state.ins.u8s[64];
        for (; local_ptr < local_end; ++local_ptr, ++text) *local_ptr = *text;
        state.aes.u8x16s[0] = sz_emulate_aesenc_u8x16_neon_(state.aes.u8x16s[0], state.ins.u8x16s[0]);
        state.aes.u8x16s[1] = sz_emulate_aesenc_u8x16_neon_(state.aes.u8x16s[1], state.ins.u8x16s[1]);
        state.aes.u8x16s[2] = sz_emulate_aesenc_u8x16_neon_(state.aes.u8x16s[2], state.ins.u8x16s[2]);
        state.aes.u8x16s[3] = sz_emulate_aesenc_u8x16_neon_(state.aes.u8x16s[3], state.ins.u8x16s[3]);
        state.sum.u64x2s[0] = vaddq_u64(
            vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(state.sum.u64x2s[0]), order)), state.ins.u64x2s[0]);
        state.sum.u64x2s[1] = vaddq_u64(
            vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(state.sum.u64x2s[1]), order)), state.ins.u64x2s[1]);
        state.sum.u64x2s[2] = vaddq_u64(
            vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(state.sum.u64x2s[2]), order)), state.ins.u64x2s[2]);
        state.sum.u64x2s[3] = vaddq_u64(
            vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(state.sum.u64x2s[3]), order)), state.ins.u64x2s[3]);
        state.ins_length += head_length;
        length -= head_length;
    }

    // Now handle the body
    for (; length >= 64; state.ins_length += 64, text += 64, length -= 64) {
        state.ins.u8x16s[0] = vld1q_u8((sz_u8_t const *)(text + 0));
        state.ins.u8x16s[1] = vld1q_u8((sz_u8_t const *)(text + 16));
        state.ins.u8x16s[2] = vld1q_u8((sz_u8_t const *)(text + 32));
        state.ins.u8x16s[3] = vld1q_u8((sz_u8_t const *)(text + 48));
        state.aes.u8x16s[0] = sz_emulate_aesenc_u8x16_neon_(state.aes.u8x16s[0], state.ins.u8x16s[0]);
        state.aes.u8x16s[1] = sz_emulate_aesenc_u8x16_neon_(state.aes.u8x16s[1], state.ins.u8x16s[1]);
        state.aes.u8x16s[2] = sz_emulate_aesenc_u8x16_neon_(state.aes.u8x16s[2], state.ins.u8x16s[2]);
        state.aes.u8x16s[3] = sz_emulate_aesenc_u8x16_neon_(state.aes.u8x16s[3], state.ins.u8x16s[3]);
        state.sum.u64x2s[0] = vaddq_u64(
            vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(state.sum.u64x2s[0]), order)), state.ins.u64x2s[0]);
        state.sum.u64x2s[1] = vaddq_u64(
            vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(state.sum.u64x2s[1]), order)), state.ins.u64x2s[1]);
        state.sum.u64x2s[2] = vaddq_u64(
            vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(state.sum.u64x2s[2]), order)), state.ins.u64x2s[2]);
        state.sum.u64x2s[3] = vaddq_u64(
            vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(state.sum.u64x2s[3]), order)), state.ins.u64x2s[3]);
    }
    state.ins.u8x16s[0] = vdupq_n_u8(0);
    state.ins.u8x16s[1] = vdupq_n_u8(0);
    state.ins.u8x16s[2] = vdupq_n_u8(0);
    state.ins.u8x16s[3] = vdupq_n_u8(0);

    // The tail is the last part we need to handle
    if (tail_length) {
        sz_ptr_t local_ptr = (sz_ptr_t)&state.ins.u8s[0];
        sz_ptr_t const local_end = (sz_ptr_t)&state.ins.u8s[tail_length];
        for (; local_ptr < local_end; ++local_ptr, ++text) *local_ptr = *text;
        state.ins_length += tail_length;
    }

    // Save the state back to the memory
    vst1q_u8((sz_u8_t *)state_ptr->aes + 0, state.aes.u8x16s[0]);
    vst1q_u8((sz_u8_t *)state_ptr->aes + 16, state.aes.u8x16s[1]);
    vst1q_u8((sz_u8_t *)state_ptr->aes + 32, state.aes.u8x16s[2]);
    vst1q_u8((sz_u8_t *)state_ptr->aes + 48, state.aes.u8x16s[3]);
    vst1q_u8((sz_u8_t *)state_ptr->sum + 0, state.sum.u8x16s[0]);
    vst1q_u8((sz_u8_t *)state_ptr->sum + 16, state.sum.u8x16s[1]);
    vst1q_u8((sz_u8_t *)state_ptr->sum + 32, state.sum.u8x16s[2]);
    vst1q_u8((sz_u8_t *)state_ptr->sum + 48, state.sum.u8x16s[3]);
    vst1q_u8((sz_u8_t *)state_ptr->ins + 0, state.ins.u8x16s[0]);
    vst1q_u8((sz_u8_t *)state_ptr->ins + 16, state.ins.u8x16s[1]);
    vst1q_u8((sz_u8_t *)state_ptr->ins + 32, state.ins.u8x16s[2]);
    vst1q_u8((sz_u8_t *)state_ptr->ins + 48, state.ins.u8x16s[3]);
    state_ptr->ins_length = state.ins_length;
}

SZ_PUBLIC sz_u64_t sz_hash_state_digest_neon(sz_hash_state_t const *state) {
    // This whole function is identical to Haswell.
    sz_size_t length = state->ins_length;
    if (length >= 64) {
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
    uint8x16_t const *ins_vecs = (uint8x16_t const *)state->ins;
    if (length <= 16) {
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[0]);
        return sz_hash_minimal_finalize_neon_(&minimal_state, length);
    }
    else if (length <= 32) {
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[1]);
        return sz_hash_minimal_finalize_neon_(&minimal_state, length);
    }
    else if (length <= 48) {
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[1]);
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[2]);
        return sz_hash_minimal_finalize_neon_(&minimal_state, length);
    }
    else {
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[1]);
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[2]);
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[3]);
        return sz_hash_minimal_finalize_neon_(&minimal_state, length);
    }
}

SZ_PUBLIC SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_neon(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {
    if (length <= 16) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data_vec;
        if (length == 16) { data_vec.u8x16 = vld1q_u8((sz_u8_t const *)start); } // Full in-bounds block
        else {
            data_vec.u8x16 = vdupq_n_u8(0);
            for (sz_size_t i = 0; i < length; ++i) data_vec.u8s[i] = start[i]; // Variable partial tail
        }
        sz_hash_minimal_update_neon_(&state, data_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else if (length <= 32) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.u8x16 = vld1q_u8((sz_u8_t const *)(start + 0));
        data1_vec.u8x16 = vld1q_u8((sz_u8_t const *)(start + length - 16));
        // Let's shift the data within the register to de-interleave the bytes.
        sz_hash_shift_in_register_serial_(&data1_vec, (int)(32 - length)); //! `vextq_u8` requires immediates
        sz_hash_minimal_update_neon_(&state, data0_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data1_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else if (length <= 48) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.u8x16 = vld1q_u8((sz_u8_t const *)(start + 0));
        data1_vec.u8x16 = vld1q_u8((sz_u8_t const *)(start + 16));
        data2_vec.u8x16 = vld1q_u8((sz_u8_t const *)(start + length - 16));
        // Let's shift the data within the register to de-interleave the bytes.
        sz_hash_shift_in_register_serial_(&data2_vec, (int)(48 - length)); //! `vextq_u8` requires immediates
        sz_hash_minimal_update_neon_(&state, data0_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data1_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data2_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else if (length <= 64) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.u8x16 = vld1q_u8((sz_u8_t const *)(start + 0));
        data1_vec.u8x16 = vld1q_u8((sz_u8_t const *)(start + 16));
        data2_vec.u8x16 = vld1q_u8((sz_u8_t const *)(start + 32));
        data3_vec.u8x16 = vld1q_u8((sz_u8_t const *)(start + length - 16));
        // Let's shift the data within the register to de-interleave the bytes.
        sz_hash_shift_in_register_serial_(&data3_vec, (int)(64 - length)); //! `vextq_u8` requires immediates
        sz_hash_minimal_update_neon_(&state, data0_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data1_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data2_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data3_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else {
        sz_align_(64) sz_hash_state_internal_t_ state;
        sz_hash_state_init_neon((sz_hash_state_t *)&state, seed);
        for (; state.ins_length + 64 <= length; state.ins_length += 64) {
            state.ins.u8x16s[0] = vld1q_u8((sz_u8_t const *)(start + state.ins_length + 0));
            state.ins.u8x16s[1] = vld1q_u8((sz_u8_t const *)(start + state.ins_length + 16));
            state.ins.u8x16s[2] = vld1q_u8((sz_u8_t const *)(start + state.ins_length + 32));
            state.ins.u8x16s[3] = vld1q_u8((sz_u8_t const *)(start + state.ins_length + 48));
            sz_hash_state_update_neon_(&state);
        }
        // Handle the tail, resetting the registers to zero first
        if (state.ins_length < length) {
            state.ins.u8x16s[0] = vdupq_n_u8(0);
            state.ins.u8x16s[1] = vdupq_n_u8(0);
            state.ins.u8x16s[2] = vdupq_n_u8(0);
            state.ins.u8x16s[3] = vdupq_n_u8(0);
            for (sz_size_t i = 0; state.ins_length < length; ++i, ++state.ins_length)
                state.ins.u8s[i] = start[state.ins_length];
            sz_hash_state_update_neon_(&state);
            state.ins_length = length;
        }
        return sz_hash_state_finalize_neon_(&state);
    }
}

SZ_PUBLIC void sz_fill_random_neon(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_u64_t const *pi_ptr = sz_hash_pi_constants_();
    if (length <= 16) {
        uint64x2_t input = vdupq_n_u64(nonce);
        uint64x2_t pi = vld1q_u64(pi_ptr);
        uint64x2_t key = veorq_u64(vdupq_n_u64(nonce), pi);
        uint64x2_t generated = sz_emulate_aesenc_u64x2_neon_(input, key);
        // Now the tricky part is outputting this data to the user-supplied buffer
        // without masked writes, like in AVX-512.
        if (length >= 16) { vst1q_u8((sz_u8_t *)text, vreinterpretq_u8_u64(generated)); } // Full block
        else
            for (sz_size_t i = 0; i < length; ++i) text[i] = ((sz_u8_t *)&generated)[i]; // Variable partial tail
    }
    // Assuming the YMM register contains two 128-bit blocks, the input to the generator
    // will be more complex, containing the sum of the nonce and the block number.
    else if (length <= 32) {
        uint64x2_t inputs[2], pis[2], keys[2], generated[2];
        inputs[0] = vdupq_n_u64(nonce + 0);
        inputs[1] = vdupq_n_u64(nonce + 1);
        pis[0] = vld1q_u64(pi_ptr + 0);
        pis[1] = vld1q_u64(pi_ptr + 2);
        keys[0] = veorq_u64(vdupq_n_u64(nonce), pis[0]);
        keys[1] = veorq_u64(vdupq_n_u64(nonce), pis[1]);
        generated[0] = sz_emulate_aesenc_u64x2_neon_(inputs[0], keys[0]);
        generated[1] = sz_emulate_aesenc_u64x2_neon_(inputs[1], keys[1]);
        // The first store can easily be vectorized, but the second can be serial for now
        vst1q_u64((sz_u64_t *)(text), generated[0]);
        if (length >= 32) { vst1q_u8((sz_u8_t *)(text + 16), vreinterpretq_u8_u64(generated[1])); } // Full block
        else
            for (sz_size_t i = 16; i < length; ++i) text[i] = ((sz_u8_t *)&generated[1])[i - 16]; // Partial tail
    }
    // The last special case we handle outside of the primary loop is for buffers up to 64 bytes long.
    else if (length <= 48) {
        uint64x2_t inputs[3], pis[3], keys[3], generated[3];
        inputs[0] = vdupq_n_u64(nonce);
        inputs[1] = vdupq_n_u64(nonce + 1);
        inputs[2] = vdupq_n_u64(nonce + 2);
        pis[0] = vld1q_u64(pi_ptr + 0);
        pis[1] = vld1q_u64(pi_ptr + 2);
        pis[2] = vld1q_u64(pi_ptr + 4);
        keys[0] = veorq_u64(vdupq_n_u64(nonce), pis[0]);
        keys[1] = veorq_u64(vdupq_n_u64(nonce), pis[1]);
        keys[2] = veorq_u64(vdupq_n_u64(nonce), pis[2]);
        generated[0] = sz_emulate_aesenc_u64x2_neon_(inputs[0], keys[0]);
        generated[1] = sz_emulate_aesenc_u64x2_neon_(inputs[1], keys[1]);
        generated[2] = sz_emulate_aesenc_u64x2_neon_(inputs[2], keys[2]);
        // The first store can easily be vectorized, but the second can be serial for now
        vst1q_u64((sz_u64_t *)(text + 0), generated[0]);
        vst1q_u64((sz_u64_t *)(text + 16), generated[1]);
        if (length >= 48) { vst1q_u8((sz_u8_t *)(text + 32), vreinterpretq_u8_u64(generated[2])); } // Full block
        else
            for (sz_size_t i = 32; i < length; ++i) text[i] = ((sz_u8_t *)generated)[i]; // Partial tail
    }
    // The final part of the function is the primary loop, which processes the buffer in 64-byte chunks.
    else {
        uint64x2_t inputs[4], pis[4], keys[4], generated[4];
        inputs[0] = vdupq_n_u64(nonce + 0);
        inputs[1] = vdupq_n_u64(nonce + 1);
        inputs[2] = vdupq_n_u64(nonce + 2);
        inputs[3] = vdupq_n_u64(nonce + 3);
        // Load parts of PI into the registers
        pis[0] = vld1q_u64(pi_ptr + 0);
        pis[1] = vld1q_u64(pi_ptr + 2);
        pis[2] = vld1q_u64(pi_ptr + 4);
        pis[3] = vld1q_u64(pi_ptr + 6);
        // XOR the nonce with the PI constants
        keys[0] = veorq_u64(vdupq_n_u64(nonce), pis[0]);
        keys[1] = veorq_u64(vdupq_n_u64(nonce), pis[1]);
        keys[2] = veorq_u64(vdupq_n_u64(nonce), pis[2]);
        keys[3] = veorq_u64(vdupq_n_u64(nonce), pis[3]);

        // Produce the output, fixing the key and enumerating input chunks.
        sz_size_t i = 0;
        uint64x2_t const increment = vdupq_n_u64(4);
        for (; i + 64 <= length; i += 64) {
            generated[0] = sz_emulate_aesenc_u64x2_neon_(inputs[0], keys[0]);
            generated[1] = sz_emulate_aesenc_u64x2_neon_(inputs[1], keys[1]);
            generated[2] = sz_emulate_aesenc_u64x2_neon_(inputs[2], keys[2]);
            generated[3] = sz_emulate_aesenc_u64x2_neon_(inputs[3], keys[3]);
            vst1q_u64((sz_u64_t *)(text + i + 0), generated[0]);
            vst1q_u64((sz_u64_t *)(text + i + 16), generated[1]);
            vst1q_u64((sz_u64_t *)(text + i + 32), generated[2]);
            vst1q_u64((sz_u64_t *)(text + i + 48), generated[3]);
            inputs[0] = vaddq_u64(inputs[0], increment);
            inputs[1] = vaddq_u64(inputs[1], increment);
            inputs[2] = vaddq_u64(inputs[2], increment);
            inputs[3] = vaddq_u64(inputs[3], increment);
        }

        // Handle the tail of the buffer.
        {
            generated[0] = sz_emulate_aesenc_u64x2_neon_(inputs[0], keys[0]);
            generated[1] = sz_emulate_aesenc_u64x2_neon_(inputs[1], keys[1]);
            generated[2] = sz_emulate_aesenc_u64x2_neon_(inputs[2], keys[2]);
            generated[3] = sz_emulate_aesenc_u64x2_neon_(inputs[3], keys[3]);
            // Spill full 16-byte blocks with vector stores, leaving only the partial (<16) tail scalar
            sz_size_t j = 0;
            for (; i + 16 <= length; i += 16, j += 16)
                vst1q_u8((sz_u8_t *)(text + i), vld1q_u8((sz_u8_t const *)generated + j)); // Full block
            for (; i < length; ++i, ++j) text[i] = ((sz_u8_t *)generated)[j];              // Variable partial tail
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
