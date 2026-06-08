/**
 *  @brief Westmere (SSE4.2 + AES-NI) backend for string hashing and checksums.
 *  @file include/stringzilla/hash/westmere.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_WESTMERE_H_
#define STRINGZILLA_HASH_WESTMERE_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/hash/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_WESTMERE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sse4.2,aes"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("sse4.2", "aes")
#endif

/**
 *  @brief Initializes the minimal single-lane AES hash state using SSE2/AES-NI intrinsics.
 *         Requires the @p state to be 16-byte aligned.
 *  @param state Pointer to the aligned minimal hash state to initialize.
 *  @param seed 64-bit seed value XOR-ed with Pi constants to form the initial state.
 */
SZ_INTERNAL void sz_hash_minimal_init_westmere_aligned_(sz_hash_minimal_t_ *state, sz_u64_t seed) {

    // The key is made from the seed and half of it will be mixed with the length in the end
    __m128i seed_vec = _mm_set1_epi64x(seed);
    state->key.xmm = seed_vec;

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    __m128i const pi0 = _mm_load_si128((__m128i const *)(pi));
    __m128i const pi1 = _mm_load_si128((__m128i const *)(pi + 8));
    __m128i aes_state_key = _mm_xor_si128(seed_vec, pi0);
    __m128i sum_state_key = _mm_xor_si128(seed_vec, pi1);

    // The first 128 bits of the "sum" and "AES" blocks are the same for the "minimal" and full state
    state->aes.xmm = aes_state_key;
    state->sum.xmm = sum_state_key;
}

/**
 *  @brief Absorbs one 128-bit block into the minimal hash state using AES-NI and SSSE3 intrinsics.
 *  @param state_ptr Pointer to the aligned minimal hash state.
 *  @param block 128-bit data block to absorb.
 *  @param order Shuffle permutation for the additive accumulator lane (loaded from `sz_hash_u8x16x4_shuffle_`).
 */
SZ_INTERNAL void sz_hash_minimal_update_westmere_aligned_(sz_hash_minimal_t_ *state_ptr, __m128i block, __m128i order) {
    state_ptr->aes.xmm = _mm_aesenc_si128(state_ptr->aes.xmm, block);
    state_ptr->sum.xmm = _mm_add_epi64(_mm_shuffle_epi8(state_ptr->sum.xmm, order), block);
}

/**
 *  @brief Finalizes the minimal AES hash state using AES-NI and returns a 64-bit digest.
 *  @param state Pointer to the (const) aligned minimal hash state.
 *  @param length Total number of bytes hashed, mixed into the key for length sensitivity.
 *  @return 64-bit hash value.
 */
SZ_INTERNAL sz_u64_t sz_hash_minimal_finalize_westmere_aligned_(sz_hash_minimal_t_ const *state, sz_size_t length) {
    // Mix the length into the key
    __m128i key_with_length = _mm_add_epi64(state->key.xmm, _mm_set_epi64x(0, length));
    // Combine the "sum" and the "AES" blocks
    __m128i mixed = _mm_aesenc_si128(state->sum.xmm, state->aes.xmm);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    __m128i mixed_in_register = _mm_aesenc_si128(_mm_aesenc_si128(mixed, key_with_length), mixed);
    // Extract the low 64 bits
    return _mm_cvtsi128_si64(mixed_in_register);
}

SZ_PUBLIC void sz_hash_state_init_westmere(sz_hash_state_t *state, sz_u64_t seed) {
    // The key is made from the seed and half of it will be mixed with the length in the end
    __m128i seed_vec = _mm_set1_epi64x(seed);

    // ! In this kernel, assuming it may be called on arbitrarily misaligned `state`,
    // ! we must use `_mm_storeu_si128` stores to update the state. Moreover, accessing `state.xmms[i]`
    // ! fools the compiler into preferring aligned operations over out misaligned ones.
    _mm_storeu_si128((__m128i *)state->key, seed_vec);

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    for (int lane_index = 0; lane_index < 4; ++lane_index)
        _mm_storeu_si128((__m128i *)&state->aes[lane_index * 16],
                         _mm_xor_si128(seed_vec, _mm_lddqu_si128((__m128i const *)(pi + lane_index * 2))));
    for (int lane_index = 0; lane_index < 4; ++lane_index)
        _mm_storeu_si128((__m128i *)&state->sum[lane_index * 16],
                         _mm_xor_si128(seed_vec, _mm_lddqu_si128((__m128i const *)(pi + lane_index * 2 + 8))));

    // The inputs are zeroed out at the beginning
    _mm_storeu_si128((__m128i *)&state->ins[0], _mm_setzero_si128());
    _mm_storeu_si128((__m128i *)&state->ins[16], _mm_setzero_si128());
    _mm_storeu_si128((__m128i *)&state->ins[32], _mm_setzero_si128());
    _mm_storeu_si128((__m128i *)&state->ins[48], _mm_setzero_si128());
    state->ins_length = 0;
}

/**
 *  @brief Finalizes the full 512-bit AES hash state using AES-NI and returns a 64-bit digest.
 *  @param state Pointer to the (const) hash state.
 *  @return 64-bit hash value derived by folding the four AES lanes together with the key.
 */
SZ_INTERNAL sz_u64_t sz_hash_state_finalize_westmere_(sz_hash_state_t const *state) {
    // Mix the length into the key
    __m128i key_with_length = _mm_add_epi64(_mm_lddqu_si128((__m128i const *)state->key),
                                            _mm_set_epi64x(0, state->ins_length));
    // Combine the "sum" and the "AES" blocks
    __m128i mixed0 = _mm_aesenc_si128(_mm_lddqu_si128((__m128i *)&state->sum[0]),
                                      _mm_lddqu_si128((__m128i *)&state->aes[0]));
    __m128i mixed1 = _mm_aesenc_si128(_mm_lddqu_si128((__m128i *)&state->sum[16]),
                                      _mm_lddqu_si128((__m128i *)&state->aes[16]));
    __m128i mixed2 = _mm_aesenc_si128(_mm_lddqu_si128((__m128i *)&state->sum[32]),
                                      _mm_lddqu_si128((__m128i *)&state->aes[32]));
    __m128i mixed3 = _mm_aesenc_si128(_mm_lddqu_si128((__m128i *)&state->sum[48]),
                                      _mm_lddqu_si128((__m128i *)&state->aes[48]));
    // Combine the mixed registers
    __m128i mixed01 = _mm_aesenc_si128(mixed0, mixed1);
    __m128i mixed23 = _mm_aesenc_si128(mixed2, mixed3);
    __m128i mixed = _mm_aesenc_si128(mixed01, mixed23);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    __m128i mixed_in_register = _mm_aesenc_si128(_mm_aesenc_si128(mixed, key_with_length), mixed);
    // Extract the low 64 bits
    return _mm_cvtsi128_si64(mixed_in_register);
}

SZ_PUBLIC SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_westmere(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {

    if (length <= 16) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_westmere_aligned_(&state, seed);

        // Load the data and update the state
        sz_u128_vec_t data_vec;
        data_vec.xmm = _mm_setzero_si128();
        for (sz_size_t byte_index = 0; byte_index < length; ++byte_index) data_vec.u8s[byte_index] = start[byte_index];

        // Shuffle with the same mask
        __m128i const order = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_minimal_update_westmere_aligned_(&state, data_vec.xmm, order);
        return sz_hash_minimal_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 32) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_westmere_aligned_(&state, seed);

        // Load the data, shifting the data within the register to de-interleave the bytes
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + length - 16));
        sz_hash_shift_in_register_serial_(&data1_vec, (int)(32 - length));

        // Shuffle with the same mask
        __m128i const order = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_minimal_update_westmere_aligned_(&state, data0_vec.xmm, order);
        sz_hash_minimal_update_westmere_aligned_(&state, data1_vec.xmm, order);
        return sz_hash_minimal_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 48) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_westmere_aligned_(&state, seed);

        // Load the data, shifting the data within the register to de-interleave the bytes
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 0));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 16));
        data2_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + length - 16));
        sz_hash_shift_in_register_serial_(&data2_vec, (int)(48 - length));

        // Shuffle with the same mask
        __m128i const order = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_minimal_update_westmere_aligned_(&state, data0_vec.xmm, order);
        sz_hash_minimal_update_westmere_aligned_(&state, data1_vec.xmm, order);
        sz_hash_minimal_update_westmere_aligned_(&state, data2_vec.xmm, order);
        return sz_hash_minimal_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 64) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_westmere_aligned_(&state, seed);

        // Load the data, shifting the data within the register to de-interleave the bytes
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 0));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 16));
        data2_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 32));
        data3_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + length - 16));
        sz_hash_shift_in_register_serial_(&data3_vec, (int)(64 - length));

        // Shuffle with the same mask
        __m128i const order = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_minimal_update_westmere_aligned_(&state, data0_vec.xmm, order);
        sz_hash_minimal_update_westmere_aligned_(&state, data1_vec.xmm, order);
        sz_hash_minimal_update_westmere_aligned_(&state, data2_vec.xmm, order);
        sz_hash_minimal_update_westmere_aligned_(&state, data3_vec.xmm, order);
        return sz_hash_minimal_finalize_westmere_aligned_(&state, length);
    }
    else {
        sz_align_(64) sz_hash_state_internal_t_ state;
        sz_hash_state_init_westmere((sz_hash_state_t *)&state, seed);

        // Shuffle with the same mask
        __m128i const order = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        for (; state.ins_length + 64 <= length; state.ins_length += 64) {
            state.ins.xmms[0] = _mm_lddqu_si128((__m128i const *)(start + state.ins_length + 0));
            state.ins.xmms[1] = _mm_lddqu_si128((__m128i const *)(start + state.ins_length + 16));
            state.ins.xmms[2] = _mm_lddqu_si128((__m128i const *)(start + state.ins_length + 32));
            state.ins.xmms[3] = _mm_lddqu_si128((__m128i const *)(start + state.ins_length + 48));
            state.aes.xmms[0] = _mm_aesenc_si128(state.aes.xmms[0], state.ins.xmms[0]);
            state.aes.xmms[1] = _mm_aesenc_si128(state.aes.xmms[1], state.ins.xmms[1]);
            state.aes.xmms[2] = _mm_aesenc_si128(state.aes.xmms[2], state.ins.xmms[2]);
            state.aes.xmms[3] = _mm_aesenc_si128(state.aes.xmms[3], state.ins.xmms[3]);
            state.sum.xmms[0] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[0], order), state.ins.xmms[0]);
            state.sum.xmms[1] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[1], order), state.ins.xmms[1]);
            state.sum.xmms[2] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[2], order), state.ins.xmms[2]);
            state.sum.xmms[3] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[3], order), state.ins.xmms[3]);
        }
        // Handle the tail, resetting the registers to zero first
        if (state.ins_length < length) {
            state.ins.xmms[0] = _mm_setzero_si128();
            state.ins.xmms[1] = _mm_setzero_si128();
            state.ins.xmms[2] = _mm_setzero_si128();
            state.ins.xmms[3] = _mm_setzero_si128();
            for (sz_size_t byte_index = 0; state.ins_length < length; ++byte_index, ++state.ins_length)
                state.ins.u8s[byte_index] = start[state.ins_length];
            state.aes.xmms[0] = _mm_aesenc_si128(state.aes.xmms[0], state.ins.xmms[0]);
            state.aes.xmms[1] = _mm_aesenc_si128(state.aes.xmms[1], state.ins.xmms[1]);
            state.aes.xmms[2] = _mm_aesenc_si128(state.aes.xmms[2], state.ins.xmms[2]);
            state.aes.xmms[3] = _mm_aesenc_si128(state.aes.xmms[3], state.ins.xmms[3]);
            state.sum.xmms[0] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[0], order), state.ins.xmms[0]);
            state.sum.xmms[1] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[1], order), state.ins.xmms[1]);
            state.sum.xmms[2] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[2], order), state.ins.xmms[2]);
            state.sum.xmms[3] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[3], order), state.ins.xmms[3]);
            state.ins_length = length;
        }
        return sz_hash_state_finalize_westmere_((sz_hash_state_t const *)&state);
    }
}

SZ_PUBLIC void sz_hash_state_update_westmere(sz_hash_state_t *state_ptr, sz_cptr_t text, sz_size_t length) {

    // The worst usage pattern... that we should ironically handle first - is updating the state
    // with a very small chunk of data, potentially, one byte at a time. In such cases, we won't
    // even bother using AVX-512 masked loads to avoid tripping the CPU state.
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
    // Keep arithmetic in u8 space to avoid alignment assumptions on __m128i pointers.
    sz_align_(64) sz_hash_state_internal_t_ state;
    state.aes.xmms[0] = _mm_lddqu_si128((__m128i const *)&state_ptr->aes[0]);
    state.aes.xmms[1] = _mm_lddqu_si128((__m128i const *)&state_ptr->aes[16]);
    state.aes.xmms[2] = _mm_lddqu_si128((__m128i const *)&state_ptr->aes[32]);
    state.aes.xmms[3] = _mm_lddqu_si128((__m128i const *)&state_ptr->aes[48]);
    state.sum.xmms[0] = _mm_lddqu_si128((__m128i const *)&state_ptr->sum[0]);
    state.sum.xmms[1] = _mm_lddqu_si128((__m128i const *)&state_ptr->sum[16]);
    state.sum.xmms[2] = _mm_lddqu_si128((__m128i const *)&state_ptr->sum[32]);
    state.sum.xmms[3] = _mm_lddqu_si128((__m128i const *)&state_ptr->sum[48]);
    state.ins.xmms[0] = _mm_lddqu_si128((__m128i const *)&state_ptr->ins[0]);
    state.ins.xmms[1] = _mm_lddqu_si128((__m128i const *)&state_ptr->ins[16]);
    state.ins.xmms[2] = _mm_lddqu_si128((__m128i const *)&state_ptr->ins[32]);
    state.ins.xmms[3] = _mm_lddqu_si128((__m128i const *)&state_ptr->ins[48]);
    state.ins_length = state_ptr->ins_length;

    // Handle the head first, filling up the current block
    __m128i const order = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
    if (head_length) {
        sz_ptr_t local_ptr = (sz_ptr_t)&state.ins.u8s[progress_in_block];
        sz_ptr_t const local_end = (sz_ptr_t)&state.ins.u8s[64];
        for (; local_ptr < local_end; ++local_ptr, ++text) *local_ptr = *text;
        state.aes.xmms[0] = _mm_aesenc_si128(state.aes.xmms[0], state.ins.xmms[0]);
        state.aes.xmms[1] = _mm_aesenc_si128(state.aes.xmms[1], state.ins.xmms[1]);
        state.aes.xmms[2] = _mm_aesenc_si128(state.aes.xmms[2], state.ins.xmms[2]);
        state.aes.xmms[3] = _mm_aesenc_si128(state.aes.xmms[3], state.ins.xmms[3]);
        state.sum.xmms[0] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[0], order), state.ins.xmms[0]);
        state.sum.xmms[1] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[1], order), state.ins.xmms[1]);
        state.sum.xmms[2] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[2], order), state.ins.xmms[2]);
        state.sum.xmms[3] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[3], order), state.ins.xmms[3]);
        state.ins_length += head_length;
        length -= head_length;
    }

    // Now handle the body
    for (; length >= 64; state.ins_length += 64, text += 64, length -= 64) {
        state.ins.xmms[0] = _mm_lddqu_si128((__m128i const *)(text + 0));
        state.ins.xmms[1] = _mm_lddqu_si128((__m128i const *)(text + 16));
        state.ins.xmms[2] = _mm_lddqu_si128((__m128i const *)(text + 32));
        state.ins.xmms[3] = _mm_lddqu_si128((__m128i const *)(text + 48));
        state.aes.xmms[0] = _mm_aesenc_si128(state.aes.xmms[0], state.ins.xmms[0]);
        state.aes.xmms[1] = _mm_aesenc_si128(state.aes.xmms[1], state.ins.xmms[1]);
        state.aes.xmms[2] = _mm_aesenc_si128(state.aes.xmms[2], state.ins.xmms[2]);
        state.aes.xmms[3] = _mm_aesenc_si128(state.aes.xmms[3], state.ins.xmms[3]);
        state.sum.xmms[0] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[0], order), state.ins.xmms[0]);
        state.sum.xmms[1] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[1], order), state.ins.xmms[1]);
        state.sum.xmms[2] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[2], order), state.ins.xmms[2]);
        state.sum.xmms[3] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[3], order), state.ins.xmms[3]);
    }
    state.ins.xmms[0] = _mm_setzero_si128();
    state.ins.xmms[1] = _mm_setzero_si128();
    state.ins.xmms[2] = _mm_setzero_si128();
    state.ins.xmms[3] = _mm_setzero_si128();

    // The tail is the last part we need to handle
    if (tail_length) {
        sz_ptr_t local_ptr = (sz_ptr_t)&state.ins.u8s[0];
        sz_ptr_t const local_end = (sz_ptr_t)&state.ins.u8s[tail_length];
        for (; local_ptr < local_end; ++local_ptr, ++text) *local_ptr = *text;
        state.ins_length += tail_length;
    }

    // Save the state back to the memory
    __m128i *aes_dst = (__m128i *)state_ptr->aes;
    __m128i *sum_dst = (__m128i *)state_ptr->sum;
    __m128i *ins_dst = (__m128i *)state_ptr->ins;
    _mm_storeu_si128(&aes_dst[0], state.aes.xmms[0]);
    _mm_storeu_si128(&aes_dst[1], state.aes.xmms[1]);
    _mm_storeu_si128(&aes_dst[2], state.aes.xmms[2]);
    _mm_storeu_si128(&aes_dst[3], state.aes.xmms[3]);
    _mm_storeu_si128(&sum_dst[0], state.sum.xmms[0]);
    _mm_storeu_si128(&sum_dst[1], state.sum.xmms[1]);
    _mm_storeu_si128(&sum_dst[2], state.sum.xmms[2]);
    _mm_storeu_si128(&sum_dst[3], state.sum.xmms[3]);
    _mm_storeu_si128(&ins_dst[0], state.ins.xmms[0]);
    _mm_storeu_si128(&ins_dst[1], state.ins.xmms[1]);
    _mm_storeu_si128(&ins_dst[2], state.ins.xmms[2]);
    _mm_storeu_si128(&ins_dst[3], state.ins.xmms[3]);
    state_ptr->ins_length = state.ins_length;
}

SZ_PUBLIC sz_u64_t sz_hash_state_digest_westmere(sz_hash_state_t const *state_ptr) {
    sz_size_t length = state_ptr->ins_length;
    if (length >= 64) return sz_hash_state_finalize_westmere_(state_ptr);

    // Switch back to a smaller "minimal" state for small inputs
    sz_align_(16) sz_hash_minimal_t_ state;
    state.key.xmm = _mm_lddqu_si128((__m128i const *)state_ptr->key);
    state.aes.xmm = _mm_lddqu_si128((__m128i const *)state_ptr->aes);
    state.sum.xmm = _mm_lddqu_si128((__m128i const *)state_ptr->sum);

    // The logic is different depending on the length of the input
    __m128i const *ins_xmms = (__m128i const *)state_ptr->ins;
    __m128i const order = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
    if (length <= 16) {
        sz_hash_minimal_update_westmere_aligned_(&state, _mm_lddqu_si128(&ins_xmms[0]), order);
        return sz_hash_minimal_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 32) {
        sz_hash_minimal_update_westmere_aligned_(&state, _mm_lddqu_si128(&ins_xmms[0]), order);
        sz_hash_minimal_update_westmere_aligned_(&state, _mm_lddqu_si128(&ins_xmms[1]), order);
        return sz_hash_minimal_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 48) {
        sz_hash_minimal_update_westmere_aligned_(&state, _mm_lddqu_si128(&ins_xmms[0]), order);
        sz_hash_minimal_update_westmere_aligned_(&state, _mm_lddqu_si128(&ins_xmms[1]), order);
        sz_hash_minimal_update_westmere_aligned_(&state, _mm_lddqu_si128(&ins_xmms[2]), order);
        return sz_hash_minimal_finalize_westmere_aligned_(&state, length);
    }
    else {
        sz_hash_minimal_update_westmere_aligned_(&state, _mm_lddqu_si128(&ins_xmms[0]), order);
        sz_hash_minimal_update_westmere_aligned_(&state, _mm_lddqu_si128(&ins_xmms[1]), order);
        sz_hash_minimal_update_westmere_aligned_(&state, _mm_lddqu_si128(&ins_xmms[2]), order);
        sz_hash_minimal_update_westmere_aligned_(&state, _mm_lddqu_si128(&ins_xmms[3]), order);
        return sz_hash_minimal_finalize_westmere_aligned_(&state, length);
    }
}

SZ_PUBLIC void sz_fill_random_westmere(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_u64_t const *pi_ptr = sz_hash_pi_constants_();
    if (length <= 16) {
        __m128i input = _mm_set1_epi64x(nonce);
        __m128i pi = _mm_load_si128((__m128i const *)pi_ptr);
        __m128i key = _mm_xor_si128(_mm_set1_epi64x(nonce), pi);
        __m128i generated = _mm_aesenc_si128(input, key);
        // Now the tricky part is outputting this data to the user-supplied buffer
        // without masked writes, like in AVX-512.
        for (sz_size_t byte_index = 0; byte_index < length; ++byte_index)
            text[byte_index] = ((sz_u8_t *)&generated)[byte_index];
    }
    // Assuming the YMM register contains two 128-bit blocks, the input to the generator
    // will be more complex, containing the sum of the nonce and the block number.
    else if (length <= 32) {
        __m128i inputs[2], pis[2], keys[2], generated[2];
        inputs[0] = _mm_set1_epi64x(nonce);
        inputs[1] = _mm_set1_epi64x(nonce + 1);
        pis[0] = _mm_load_si128((__m128i const *)(pi_ptr + 0));
        pis[1] = _mm_load_si128((__m128i const *)(pi_ptr + 2));
        keys[0] = _mm_xor_si128(_mm_set1_epi64x(nonce), pis[0]);
        keys[1] = _mm_xor_si128(_mm_set1_epi64x(nonce), pis[1]);
        generated[0] = _mm_aesenc_si128(inputs[0], keys[0]);
        generated[1] = _mm_aesenc_si128(inputs[1], keys[1]);
        // The first store can easily be vectorized, but the second can be serial for now
        _mm_storeu_si128((__m128i *)text, generated[0]);
        for (sz_size_t byte_index = 16; byte_index < length; ++byte_index)
            text[byte_index] = ((sz_u8_t *)&generated[1])[byte_index - 16];
    }
    // The last special case we handle outside of the primary loop is for buffers up to 64 bytes long.
    else if (length <= 48) {
        __m128i inputs[3], pis[3], keys[3], generated[3];
        inputs[0] = _mm_set1_epi64x(nonce);
        inputs[1] = _mm_set1_epi64x(nonce + 1);
        inputs[2] = _mm_set1_epi64x(nonce + 2);
        pis[0] = _mm_load_si128((__m128i const *)(pi_ptr + 0));
        pis[1] = _mm_load_si128((__m128i const *)(pi_ptr + 2));
        pis[2] = _mm_load_si128((__m128i const *)(pi_ptr + 4));
        keys[0] = _mm_xor_si128(_mm_set1_epi64x(nonce), pis[0]);
        keys[1] = _mm_xor_si128(_mm_set1_epi64x(nonce), pis[1]);
        keys[2] = _mm_xor_si128(_mm_set1_epi64x(nonce), pis[2]);
        generated[0] = _mm_aesenc_si128(inputs[0], keys[0]);
        generated[1] = _mm_aesenc_si128(inputs[1], keys[1]);
        generated[2] = _mm_aesenc_si128(inputs[2], keys[2]);
        // The first store can easily be vectorized, but the second can be serial for now
        _mm_storeu_si128((__m128i *)(text + 0), generated[0]);
        _mm_storeu_si128((__m128i *)(text + 16), generated[1]);
        for (sz_size_t byte_index = 32; byte_index < length; ++byte_index)
            text[byte_index] = ((sz_u8_t *)generated)[byte_index];
    }
    // The final part of the function is the primary loop, which processes the buffer in 64-byte chunks.
    else {
        __m128i inputs[4], pis[4], keys[4], generated[4];
        inputs[0] = _mm_set1_epi64x(nonce);
        inputs[1] = _mm_set1_epi64x(nonce + 1);
        inputs[2] = _mm_set1_epi64x(nonce + 2);
        inputs[3] = _mm_set1_epi64x(nonce + 3);
        // Load parts of PI into the registers
        pis[0] = _mm_load_si128((__m128i const *)(pi_ptr + 0));
        pis[1] = _mm_load_si128((__m128i const *)(pi_ptr + 2));
        pis[2] = _mm_load_si128((__m128i const *)(pi_ptr + 4));
        pis[3] = _mm_load_si128((__m128i const *)(pi_ptr + 6));
        // XOR the nonce with the PI constants
        keys[0] = _mm_xor_si128(_mm_set1_epi64x(nonce), pis[0]);
        keys[1] = _mm_xor_si128(_mm_set1_epi64x(nonce), pis[1]);
        keys[2] = _mm_xor_si128(_mm_set1_epi64x(nonce), pis[2]);
        keys[3] = _mm_xor_si128(_mm_set1_epi64x(nonce), pis[3]);

        // Produce the output, fixing the key and enumerating input chunks.
        sz_size_t byte_index = 0;
        __m128i const increment = _mm_set1_epi64x(4);
        for (; byte_index + 64 <= length; byte_index += 64) {
            generated[0] = _mm_aesenc_si128(inputs[0], keys[0]);
            generated[1] = _mm_aesenc_si128(inputs[1], keys[1]);
            generated[2] = _mm_aesenc_si128(inputs[2], keys[2]);
            generated[3] = _mm_aesenc_si128(inputs[3], keys[3]);
            _mm_storeu_si128((__m128i *)(text + byte_index + 0), generated[0]);
            _mm_storeu_si128((__m128i *)(text + byte_index + 16), generated[1]);
            _mm_storeu_si128((__m128i *)(text + byte_index + 32), generated[2]);
            _mm_storeu_si128((__m128i *)(text + byte_index + 48), generated[3]);
            inputs[0] = _mm_add_epi64(inputs[0], increment);
            inputs[1] = _mm_add_epi64(inputs[1], increment);
            inputs[2] = _mm_add_epi64(inputs[2], increment);
            inputs[3] = _mm_add_epi64(inputs[3], increment);
        }

        // Handle the tail of the buffer.
        {
            generated[0] = _mm_aesenc_si128(inputs[0], keys[0]);
            generated[1] = _mm_aesenc_si128(inputs[1], keys[1]);
            generated[2] = _mm_aesenc_si128(inputs[2], keys[2]);
            generated[3] = _mm_aesenc_si128(inputs[3], keys[3]);
            for (sz_size_t tail_index = 0; byte_index < length; ++byte_index, ++tail_index)
                text[byte_index] = ((sz_u8_t *)generated)[tail_index];
        }
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_WESTMERE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_WESTMERE_H_
