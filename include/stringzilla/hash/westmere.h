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
SZ_HELPER_INLINE void sz_hash_state_short_init_westmere_aligned_(sz_hash_state_aligned_for_short_t *state,
                                                                 sz_u64_t seed) {

    // The key is made from the seed and half of it will be mixed with the length in the end
    __m128i seed_u8x16 = _mm_set1_epi64x(seed);
    state->key.xmm = seed_u8x16;

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    __m128i const pi0_u8x16 = _mm_load_si128((__m128i const *)(pi));
    __m128i const pi1_u8x16 = _mm_load_si128((__m128i const *)(pi + 8));
    __m128i aes_state_key_u8x16 = _mm_xor_si128(seed_u8x16, pi0_u8x16);
    __m128i sum_state_key_u8x16 = _mm_xor_si128(seed_u8x16, pi1_u8x16);

    // The first 128 bits of the "sum" and "AES" blocks are the same for the "minimal" and full state
    state->aes.xmm = aes_state_key_u8x16;
    state->sum.xmm = sum_state_key_u8x16;
}

/**
 *  @brief Absorbs one 128-bit block into the minimal hash state using AES-NI and SSSE3 intrinsics.
 *  @param state_ptr Pointer to the aligned minimal hash state.
 *  @param block_u8x16 128-bit data block to absorb.
 *  @param order_u8x16 Shuffle permutation for the additive accumulator lane (loaded from `sz_hash_u8x16x4_shuffle_`).
 */
SZ_HELPER_INLINE void sz_hash_state_short_update_westmere_aligned_(sz_hash_state_aligned_for_short_t *state_ptr,
                                                                   __m128i block_u8x16, __m128i order_u8x16) {
    state_ptr->aes.xmm = _mm_aesenc_si128(state_ptr->aes.xmm, block_u8x16);
    state_ptr->sum.xmm = _mm_add_epi64(_mm_shuffle_epi8(state_ptr->sum.xmm, order_u8x16), block_u8x16);
}

/**
 *  @brief Finalizes the minimal AES hash state using AES-NI and returns a 64-bit digest.
 *  @param state Pointer to the (const) aligned minimal hash state.
 *  @param length Total number of bytes hashed, mixed into the key for length sensitivity.
 *  @return 64-bit hash value.
 */
SZ_HELPER_INLINE sz_u64_t sz_hash_state_short_finalize_westmere_aligned_(sz_hash_state_aligned_for_short_t const *state,
                                                                         sz_size_t length) {
    // Mix the length into the key
    __m128i key_with_length_u64x2 = _mm_add_epi64(state->key.xmm, _mm_set_epi64x(0, length));
    // Combine the "sum" and the "AES" blocks
    __m128i mixed_u8x16 = _mm_aesenc_si128(state->sum.xmm, state->aes.xmm);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    __m128i mixed_in_register_u8x16 = _mm_aesenc_si128(_mm_aesenc_si128(mixed_u8x16, key_with_length_u64x2),
                                                       mixed_u8x16);
    // Extract the low 64 bits
    return _mm_cvtsi128_si64(mixed_in_register_u8x16);
}

SZ_API_COMPTIME void sz_hash_state_init_westmere(sz_hash_state_t *state, sz_u64_t seed) {
    // The key is made from the seed and half of it will be mixed with the length in the end
    __m128i seed_u8x16 = _mm_set1_epi64x(seed);

    // ! In this kernel, assuming it may be called on arbitrarily misaligned `state`,
    // ! we must use `_mm_storeu_si128` stores to update the state. Moreover, accessing `state.xmms[i]`
    // ! fools the compiler into preferring aligned operations over out misaligned ones.
    _mm_storeu_si128((__m128i *)state->key, seed_u8x16);

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    for (int lane_index = 0; lane_index < 4; ++lane_index)
        _mm_storeu_si128((__m128i *)&state->aes[lane_index * 16],
                         _mm_xor_si128(seed_u8x16, _mm_lddqu_si128((__m128i const *)(pi + lane_index * 2))));
    for (int lane_index = 0; lane_index < 4; ++lane_index)
        _mm_storeu_si128((__m128i *)&state->sum[lane_index * 16],
                         _mm_xor_si128(seed_u8x16, _mm_lddqu_si128((__m128i const *)(pi + lane_index * 2 + 8))));

    // The inputs are zeroed out at the beginning
    _mm_storeu_si128((__m128i *)&state->ins[0], _mm_setzero_si128());
    _mm_storeu_si128((__m128i *)&state->ins[16], _mm_setzero_si128());
    _mm_storeu_si128((__m128i *)&state->ins[32], _mm_setzero_si128());
    _mm_storeu_si128((__m128i *)&state->ins[48], _mm_setzero_si128());
    state->ins_length = 0;
}

/**
 *  @brief Loads the packed public state into the aligned internal twin (4x `_mm_lddqu_si128` per 64-byte field).
 */
SZ_HELPER_INLINE sz_hash_state_aligned_t sz_hash_state_load_westmere_(sz_hash_state_t const *packed) {
    sz_hash_state_aligned_t state;
    for (int lane_index = 0; lane_index < 4; ++lane_index) {
        state.aes.xmms[lane_index] = _mm_lddqu_si128((__m128i const *)&packed->aes[lane_index * 16]);
        state.sum.xmms[lane_index] = _mm_lddqu_si128((__m128i const *)&packed->sum[lane_index * 16]);
        state.ins.xmms[lane_index] = _mm_lddqu_si128((__m128i const *)&packed->ins[lane_index * 16]);
    }
    state.key.xmm = _mm_lddqu_si128((__m128i const *)packed->key);
    state.ins_length = packed->ins_length;
    return state;
}

/** @brief Stores the aligned internal twin back into the packed public state (4x `_mm_storeu_si128` per field). */
SZ_HELPER_INLINE void sz_hash_state_store_westmere_(sz_hash_state_t *packed, sz_hash_state_aligned_t const *state) {
    for (int lane_index = 0; lane_index < 4; ++lane_index) {
        _mm_storeu_si128((__m128i *)&packed->aes[lane_index * 16], state->aes.xmms[lane_index]);
        _mm_storeu_si128((__m128i *)&packed->sum[lane_index * 16], state->sum.xmms[lane_index]);
        _mm_storeu_si128((__m128i *)&packed->ins[lane_index * 16], state->ins.xmms[lane_index]);
    }
    _mm_storeu_si128((__m128i *)packed->key, state->key.xmm);
    packed->ins_length = state->ins_length;
}

/**
 *  @brief Absorbs the buffered 64-byte block into the aligned state (four 128-bit lanes), in place.
 *  @param state Pointer to the aligned hash state whose `ins` lanes are consumed.
 */
SZ_HELPER_INLINE void sz_hash_state_update_westmere_(sz_hash_state_aligned_t *state) {
    __m128i const order_u8x16 = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
    state->aes.xmms[0] = _mm_aesenc_si128(state->aes.xmms[0], state->ins.xmms[0]);
    state->aes.xmms[1] = _mm_aesenc_si128(state->aes.xmms[1], state->ins.xmms[1]);
    state->aes.xmms[2] = _mm_aesenc_si128(state->aes.xmms[2], state->ins.xmms[2]);
    state->aes.xmms[3] = _mm_aesenc_si128(state->aes.xmms[3], state->ins.xmms[3]);
    state->sum.xmms[0] = _mm_add_epi64(_mm_shuffle_epi8(state->sum.xmms[0], order_u8x16), state->ins.xmms[0]);
    state->sum.xmms[1] = _mm_add_epi64(_mm_shuffle_epi8(state->sum.xmms[1], order_u8x16), state->ins.xmms[1]);
    state->sum.xmms[2] = _mm_add_epi64(_mm_shuffle_epi8(state->sum.xmms[2], order_u8x16), state->ins.xmms[2]);
    state->sum.xmms[3] = _mm_add_epi64(_mm_shuffle_epi8(state->sum.xmms[3], order_u8x16), state->ins.xmms[3]);
}

/**
 *  @brief Finalizes the full 512-bit AES hash state using AES-NI and returns a 64-bit digest.
 *  @param state Pointer to the (const) aligned hash state; lanes are read directly.
 *  @return 64-bit hash value derived by folding the four AES lanes together with the key.
 */
SZ_HELPER_INLINE sz_u64_t sz_hash_state_finalize_westmere_(sz_hash_state_aligned_t const *state) {
    // Mix the length into the key
    __m128i key_with_length_u64x2 = _mm_add_epi64(state->key.xmm, _mm_set_epi64x(0, state->ins_length));

    // Fold the deferred final block (still buffered in `ins` - a full 64 bytes or a zero-padded tail) into each
    // lane. Folding the last block here, rather than in `update`, lets both one-shot `sz_hash` and the streaming
    // digest defer it and share this single finalization with no state copy.
    __m128i const order_u8x16 = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
    __m128i ins0_u8x16 = state->ins.xmms[0];
    __m128i ins1_u8x16 = state->ins.xmms[1];
    __m128i ins2_u8x16 = state->ins.xmms[2];
    __m128i ins3_u8x16 = state->ins.xmms[3];
    __m128i aes0_u8x16 = _mm_aesenc_si128(state->aes.xmms[0], ins0_u8x16);
    __m128i aes1_u8x16 = _mm_aesenc_si128(state->aes.xmms[1], ins1_u8x16);
    __m128i aes2_u8x16 = _mm_aesenc_si128(state->aes.xmms[2], ins2_u8x16);
    __m128i aes3_u8x16 = _mm_aesenc_si128(state->aes.xmms[3], ins3_u8x16);
    __m128i sum0_u64x2 = _mm_add_epi64(_mm_shuffle_epi8(state->sum.xmms[0], order_u8x16), ins0_u8x16);
    __m128i sum1_u64x2 = _mm_add_epi64(_mm_shuffle_epi8(state->sum.xmms[1], order_u8x16), ins1_u8x16);
    __m128i sum2_u64x2 = _mm_add_epi64(_mm_shuffle_epi8(state->sum.xmms[2], order_u8x16), ins2_u8x16);
    __m128i sum3_u64x2 = _mm_add_epi64(_mm_shuffle_epi8(state->sum.xmms[3], order_u8x16), ins3_u8x16);

    // Combine the "sum" and the "AES" blocks
    __m128i mixed0_u8x16 = _mm_aesenc_si128(sum0_u64x2, aes0_u8x16);
    __m128i mixed1_u8x16 = _mm_aesenc_si128(sum1_u64x2, aes1_u8x16);
    __m128i mixed2_u8x16 = _mm_aesenc_si128(sum2_u64x2, aes2_u8x16);
    __m128i mixed3_u8x16 = _mm_aesenc_si128(sum3_u64x2, aes3_u8x16);
    // Combine the mixed registers
    __m128i mixed01_u8x16 = _mm_aesenc_si128(mixed0_u8x16, mixed1_u8x16);
    __m128i mixed23_u8x16 = _mm_aesenc_si128(mixed2_u8x16, mixed3_u8x16);
    __m128i mixed_u8x16 = _mm_aesenc_si128(mixed01_u8x16, mixed23_u8x16);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    __m128i mixed_in_register_u8x16 = _mm_aesenc_si128(_mm_aesenc_si128(mixed_u8x16, key_with_length_u64x2),
                                                       mixed_u8x16);
    // Extract the low 64 bits
    return _mm_cvtsi128_si64(mixed_in_register_u8x16);
}

SZ_API_COMPTIME SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_westmere(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {

    if (length <= 16) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_westmere_aligned_(&state, seed);

        // Load the data and update the state
        sz_u128_vec_t data_vec;
        data_vec.xmm = _mm_setzero_si128();
        for (sz_size_t byte_index = 0; byte_index < length; ++byte_index) data_vec.u8s[byte_index] = start[byte_index];

        // Shuffle with the same mask
        __m128i const order_u8x16 = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_state_short_update_westmere_aligned_(&state, data_vec.xmm, order_u8x16);
        return sz_hash_state_short_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 32) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_westmere_aligned_(&state, seed);

        // Load the data, shifting the data within the register to de-interleave the bytes
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + length - 16));
        sz_hash_shift_in_register_serial_(&data1_vec, (int)(32 - length));

        // Shuffle with the same mask
        __m128i const order_u8x16 = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_state_short_update_westmere_aligned_(&state, data0_vec.xmm, order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&state, data1_vec.xmm, order_u8x16);
        return sz_hash_state_short_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 48) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_westmere_aligned_(&state, seed);

        // Load the data, shifting the data within the register to de-interleave the bytes
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 0));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 16));
        data2_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + length - 16));
        sz_hash_shift_in_register_serial_(&data2_vec, (int)(48 - length));

        // Shuffle with the same mask
        __m128i const order_u8x16 = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_state_short_update_westmere_aligned_(&state, data0_vec.xmm, order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&state, data1_vec.xmm, order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&state, data2_vec.xmm, order_u8x16);
        return sz_hash_state_short_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 64) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_westmere_aligned_(&state, seed);

        // Load the data, shifting the data within the register to de-interleave the bytes
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 0));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 16));
        data2_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 32));
        data3_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + length - 16));
        sz_hash_shift_in_register_serial_(&data3_vec, (int)(64 - length));

        // Shuffle with the same mask
        __m128i const order_u8x16 = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_state_short_update_westmere_aligned_(&state, data0_vec.xmm, order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&state, data1_vec.xmm, order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&state, data2_vec.xmm, order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&state, data3_vec.xmm, order_u8x16);
        return sz_hash_state_short_finalize_westmere_aligned_(&state, length);
    }
    else {
        sz_align_(64) sz_hash_state_aligned_t state;
        sz_hash_state_init_westmere((sz_hash_state_t *)&state, seed);

        // Absorb every full 64-byte block EXCEPT the last; the final block (a full 64 or a partial tail) stays
        // buffered in `ins` for `sz_hash_state_finalize_westmere_` to fold - the same deferral the streaming path uses.
        for (; state.ins_length + 64 < length; state.ins_length += 64) {
            state.ins.xmms[0] = _mm_lddqu_si128((__m128i const *)(start + state.ins_length + 0));
            state.ins.xmms[1] = _mm_lddqu_si128((__m128i const *)(start + state.ins_length + 16));
            state.ins.xmms[2] = _mm_lddqu_si128((__m128i const *)(start + state.ins_length + 32));
            state.ins.xmms[3] = _mm_lddqu_si128((__m128i const *)(start + state.ins_length + 48));
            sz_hash_state_update_westmere_(&state);
        }
        // Stage the final [ins_length, length) bytes (1..64) into a zeroed buffer; finalize folds them.
        state.ins.xmms[0] = _mm_setzero_si128();
        state.ins.xmms[1] = _mm_setzero_si128();
        state.ins.xmms[2] = _mm_setzero_si128();
        state.ins.xmms[3] = _mm_setzero_si128();
        for (sz_size_t byte_index = 0; state.ins_length < length; ++byte_index, ++state.ins_length)
            state.ins.u8s[byte_index] = start[state.ins_length];
        return sz_hash_state_finalize_westmere_(&state);
    }
}

/**
 *  @brief Splits a short (<= 64B) input into up to four 128-bit text-lanes using SSE loads.
 *         Mirrors the loading ladder of `sz_hash_westmere`: full `lddqu` loads for complete lanes
 *         and one overlapping load + in-register shift for the partial tail; byte-by-byte only for
 *         the `< 16` case, where a 16-byte SSE load could read past the input.
 *  @return The number of populated text-lanes (1..4).
 */
SZ_HELPER_INLINE sz_size_t sz_hash_multiseed_prepare_westmere_(sz_cptr_t text, sz_size_t length,
                                                               sz_u512_vec_t *text_lanes_vec) {
    if (length <= 16) {
        sz_u128_vec_t lane_vec;
        if (length == 16) { lane_vec.xmm = _mm_lddqu_si128((__m128i const *)text); }
        else {
            lane_vec.xmm = _mm_setzero_si128();
            for (sz_size_t byte_index = 0; byte_index < length; ++byte_index)
                lane_vec.u8s[byte_index] = text[byte_index];
        }
        text_lanes_vec->u128s[0] = lane_vec;
        return 1;
    }
    sz_size_t const text_lanes_count = sz_size_divide_round_up(length, 16);
    for (sz_size_t lane_index = 0; lane_index + 1 < text_lanes_count; ++lane_index)
        text_lanes_vec->u128s[lane_index].xmm = _mm_lddqu_si128((__m128i const *)(text + lane_index * 16));
    sz_u128_vec_t tail_vec;
    tail_vec.xmm = _mm_lddqu_si128((__m128i const *)(text + length - 16));
    sz_hash_shift_in_register_serial_(&tail_vec, (int)(text_lanes_count * 16 - length));
    text_lanes_vec->u128s[text_lanes_count - 1] = tail_vec;
    return text_lanes_count;
}

SZ_API_COMPTIME void sz_hash_multiseed_westmere(sz_cptr_t text, sz_size_t length,             //
                                                sz_u64_t const *seeds, sz_size_t seeds_count, //
                                                sz_u64_t *hashes) {
    // Trivial counts don't benefit from sharing a normalization pass - go straight to the single-shot.
    if (seeds_count == 0) return;
    if (seeds_count == 1) {
        hashes[0] = sz_hash_westmere(text, length, seeds[0]);
        return;
    }
    // Without VAES there is no cross-lane AES, so the win here is the shared normalization pass plus
    // the instruction-level parallelism of two independent single-lane AES chains.
    if (length > 64) {
        for (sz_size_t seed_index = 0; seed_index < seeds_count; ++seed_index)
            hashes[seed_index] = sz_hash_westmere(text, length, seeds[seed_index]);
        return;
    }

    sz_u512_vec_t text_lanes_vec;
    sz_size_t const text_lanes_count = sz_hash_multiseed_prepare_westmere_(text, length, &text_lanes_vec);
    __m128i const order_u8x16 = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());

    sz_size_t seed_index = 0;
    for (; seed_index + 2 <= seeds_count; seed_index += 2) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state0, state1;
        sz_hash_state_short_init_westmere_aligned_(&state0, seeds[seed_index + 0]);
        sz_hash_state_short_init_westmere_aligned_(&state1, seeds[seed_index + 1]);
        for (sz_size_t lane_index = 0; lane_index < text_lanes_count; ++lane_index) {
            sz_hash_state_short_update_westmere_aligned_(&state0, text_lanes_vec.u128s[lane_index].xmm, order_u8x16);
            sz_hash_state_short_update_westmere_aligned_(&state1, text_lanes_vec.u128s[lane_index].xmm, order_u8x16);
        }
        hashes[seed_index + 0] = sz_hash_state_short_finalize_westmere_aligned_(&state0, length);
        hashes[seed_index + 1] = sz_hash_state_short_finalize_westmere_aligned_(&state1, length);
    }
    if (seed_index < seeds_count) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_westmere_aligned_(&state, seeds[seed_index]);
        for (sz_size_t lane_index = 0; lane_index < text_lanes_count; ++lane_index)
            sz_hash_state_short_update_westmere_aligned_(&state, text_lanes_vec.u128s[lane_index].xmm, order_u8x16);
        hashes[seed_index] = sz_hash_state_short_finalize_westmere_aligned_(&state, length);
    }
}

SZ_API_COMPTIME void sz_hash_state_update_westmere(sz_hash_state_t *state_ptr, sz_cptr_t text, sz_size_t length) {

    // Load the packed public state (any alignment) into an aligned twin once, buffer/absorb on it, then store back.
    // `ins` is one 64-byte block; track how many bytes it holds (0..64; 64 == a full block deferred by an earlier
    // call so `digest` can still choose minimal/full by total length), absorb it only once it becomes interior
    // (more bytes arrive), and append with a single contiguous copy (SSE has no masked store). Re-zeroing `ins`
    // after each absorb keeps the high lanes zero-padded for `finalize` to fold.
    sz_hash_state_aligned_t state = sz_hash_state_load_westmere_(state_ptr);
    sz_size_t buffered = state.ins_length % 64;
    if (buffered == 0 && state.ins_length) buffered = 64;
    while (length) {
        if (buffered == 64) { // the deferred block is now interior - absorb it (4x AES-NI) and re-zero the buffer
            sz_hash_state_update_westmere_(&state);
            state.ins.xmms[0] = _mm_setzero_si128();
            state.ins.xmms[1] = _mm_setzero_si128();
            state.ins.xmms[2] = _mm_setzero_si128();
            state.ins.xmms[3] = _mm_setzero_si128();
            buffered = 0;
        }
        sz_size_t const to_copy = sz_min_of_two(length, (sz_size_t)64 - buffered);
        for (sz_size_t byte_index = 0; byte_index < to_copy; ++byte_index)
            state.ins.u8s[buffered + byte_index] = (sz_u8_t)text[byte_index];
        buffered += to_copy, text += to_copy, length -= to_copy, state.ins_length += to_copy;
    }
    sz_hash_state_store_westmere_(state_ptr, &state);
}

SZ_API_COMPTIME sz_u64_t sz_hash_state_digest_westmere(sz_hash_state_t const *state_ptr) {
    sz_hash_state_aligned_t state = sz_hash_state_load_westmere_(state_ptr);
    sz_size_t length = state.ins_length;
    // Inputs longer than one block fold through the full four-lane state, where the deferred final block buffered
    // in `ins` is folded by `sz_hash_state_finalize_westmere_`. A length of exactly 64 uses the minimal (<=64)
    // path below - matching one-shot `sz_hash`, whose `length <= 64` ladder also stays minimal.
    if (length > 64) return sz_hash_state_finalize_westmere_(&state);

    // Switch back to a smaller "short" state for small inputs; the aligned twin lanes are read directly.
    sz_align_(16) sz_hash_state_aligned_for_short_t minimal_state;
    minimal_state.key = state.key;
    minimal_state.aes = state.aes.u128s[0];
    minimal_state.sum = state.sum.u128s[0];

    // The logic is different depending on the length of the input
    __m128i const order_u8x16 = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
    if (length <= 16) {
        sz_hash_state_short_update_westmere_aligned_(&minimal_state, state.ins.xmms[0], order_u8x16);
        return sz_hash_state_short_finalize_westmere_aligned_(&minimal_state, length);
    }
    else if (length <= 32) {
        sz_hash_state_short_update_westmere_aligned_(&minimal_state, state.ins.xmms[0], order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&minimal_state, state.ins.xmms[1], order_u8x16);
        return sz_hash_state_short_finalize_westmere_aligned_(&minimal_state, length);
    }
    else if (length <= 48) {
        sz_hash_state_short_update_westmere_aligned_(&minimal_state, state.ins.xmms[0], order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&minimal_state, state.ins.xmms[1], order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&minimal_state, state.ins.xmms[2], order_u8x16);
        return sz_hash_state_short_finalize_westmere_aligned_(&minimal_state, length);
    }
    else {
        sz_hash_state_short_update_westmere_aligned_(&minimal_state, state.ins.xmms[0], order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&minimal_state, state.ins.xmms[1], order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&minimal_state, state.ins.xmms[2], order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&minimal_state, state.ins.xmms[3], order_u8x16);
        return sz_hash_state_short_finalize_westmere_aligned_(&minimal_state, length);
    }
}

SZ_API_COMPTIME void sz_fill_random_westmere(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_u64_t const *pi_ptr = sz_hash_pi_constants_();
    if (length <= 16) {
        __m128i input_u8x16 = _mm_set1_epi64x(nonce);
        __m128i pi_u8x16 = _mm_load_si128((__m128i const *)pi_ptr);
        __m128i key_u8x16 = _mm_xor_si128(_mm_set1_epi64x(nonce), pi_u8x16);
        __m128i generated_u8x16 = _mm_aesenc_si128(input_u8x16, key_u8x16);
        // Now the tricky part is outputting this data to the user-supplied buffer
        // without masked writes, like in AVX-512.
        for (sz_size_t byte_index = 0; byte_index < length; ++byte_index)
            text[byte_index] = ((sz_u8_t *)&generated_u8x16)[byte_index];
    }
    // Assuming the YMM register contains two 128-bit blocks, the input to the generator
    // will be more complex, containing the sum of the nonce and the block number.
    else if (length <= 32) {
        __m128i inputs_u8x16[2], pis_u8x16[2], keys_u8x16[2], generated_u8x16[2];
        inputs_u8x16[0] = _mm_set1_epi64x(nonce);
        inputs_u8x16[1] = _mm_set1_epi64x(nonce + 1);
        pis_u8x16[0] = _mm_load_si128((__m128i const *)(pi_ptr + 0));
        pis_u8x16[1] = _mm_load_si128((__m128i const *)(pi_ptr + 2));
        keys_u8x16[0] = _mm_xor_si128(_mm_set1_epi64x(nonce), pis_u8x16[0]);
        keys_u8x16[1] = _mm_xor_si128(_mm_set1_epi64x(nonce), pis_u8x16[1]);
        generated_u8x16[0] = _mm_aesenc_si128(inputs_u8x16[0], keys_u8x16[0]);
        generated_u8x16[1] = _mm_aesenc_si128(inputs_u8x16[1], keys_u8x16[1]);
        // The first store can easily be vectorized, but the second can be serial for now
        _mm_storeu_si128((__m128i *)text, generated_u8x16[0]);
        for (sz_size_t byte_index = 16; byte_index < length; ++byte_index)
            text[byte_index] = ((sz_u8_t *)&generated_u8x16[1])[byte_index - 16];
    }
    // The last special case we handle outside of the primary loop is for buffers up to 64 bytes long.
    else if (length <= 48) {
        __m128i inputs_u8x16[3], pis_u8x16[3], keys_u8x16[3], generated_u8x16[3];
        inputs_u8x16[0] = _mm_set1_epi64x(nonce);
        inputs_u8x16[1] = _mm_set1_epi64x(nonce + 1);
        inputs_u8x16[2] = _mm_set1_epi64x(nonce + 2);
        pis_u8x16[0] = _mm_load_si128((__m128i const *)(pi_ptr + 0));
        pis_u8x16[1] = _mm_load_si128((__m128i const *)(pi_ptr + 2));
        pis_u8x16[2] = _mm_load_si128((__m128i const *)(pi_ptr + 4));
        keys_u8x16[0] = _mm_xor_si128(_mm_set1_epi64x(nonce), pis_u8x16[0]);
        keys_u8x16[1] = _mm_xor_si128(_mm_set1_epi64x(nonce), pis_u8x16[1]);
        keys_u8x16[2] = _mm_xor_si128(_mm_set1_epi64x(nonce), pis_u8x16[2]);
        generated_u8x16[0] = _mm_aesenc_si128(inputs_u8x16[0], keys_u8x16[0]);
        generated_u8x16[1] = _mm_aesenc_si128(inputs_u8x16[1], keys_u8x16[1]);
        generated_u8x16[2] = _mm_aesenc_si128(inputs_u8x16[2], keys_u8x16[2]);
        // The first store can easily be vectorized, but the second can be serial for now
        _mm_storeu_si128((__m128i *)(text + 0), generated_u8x16[0]);
        _mm_storeu_si128((__m128i *)(text + 16), generated_u8x16[1]);
        for (sz_size_t byte_index = 32; byte_index < length; ++byte_index)
            text[byte_index] = ((sz_u8_t *)generated_u8x16)[byte_index];
    }
    // The final part of the function is the primary loop, which processes the buffer in 64-byte chunks.
    else {
        __m128i inputs_u64x2[4], pis_u8x16[4], keys_u8x16[4], generated_u8x16[4];
        inputs_u64x2[0] = _mm_set1_epi64x(nonce);
        inputs_u64x2[1] = _mm_set1_epi64x(nonce + 1);
        inputs_u64x2[2] = _mm_set1_epi64x(nonce + 2);
        inputs_u64x2[3] = _mm_set1_epi64x(nonce + 3);
        // Load parts of PI into the registers
        pis_u8x16[0] = _mm_load_si128((__m128i const *)(pi_ptr + 0));
        pis_u8x16[1] = _mm_load_si128((__m128i const *)(pi_ptr + 2));
        pis_u8x16[2] = _mm_load_si128((__m128i const *)(pi_ptr + 4));
        pis_u8x16[3] = _mm_load_si128((__m128i const *)(pi_ptr + 6));
        // XOR the nonce with the PI constants
        keys_u8x16[0] = _mm_xor_si128(_mm_set1_epi64x(nonce), pis_u8x16[0]);
        keys_u8x16[1] = _mm_xor_si128(_mm_set1_epi64x(nonce), pis_u8x16[1]);
        keys_u8x16[2] = _mm_xor_si128(_mm_set1_epi64x(nonce), pis_u8x16[2]);
        keys_u8x16[3] = _mm_xor_si128(_mm_set1_epi64x(nonce), pis_u8x16[3]);

        // Produce the output, fixing the key and enumerating input chunks.
        sz_size_t byte_index = 0;
        __m128i const increment_u64x2 = _mm_set1_epi64x(4);
        for (; byte_index + 64 <= length; byte_index += 64) {
            generated_u8x16[0] = _mm_aesenc_si128(inputs_u64x2[0], keys_u8x16[0]);
            generated_u8x16[1] = _mm_aesenc_si128(inputs_u64x2[1], keys_u8x16[1]);
            generated_u8x16[2] = _mm_aesenc_si128(inputs_u64x2[2], keys_u8x16[2]);
            generated_u8x16[3] = _mm_aesenc_si128(inputs_u64x2[3], keys_u8x16[3]);
            _mm_storeu_si128((__m128i *)(text + byte_index + 0), generated_u8x16[0]);
            _mm_storeu_si128((__m128i *)(text + byte_index + 16), generated_u8x16[1]);
            _mm_storeu_si128((__m128i *)(text + byte_index + 32), generated_u8x16[2]);
            _mm_storeu_si128((__m128i *)(text + byte_index + 48), generated_u8x16[3]);
            inputs_u64x2[0] = _mm_add_epi64(inputs_u64x2[0], increment_u64x2);
            inputs_u64x2[1] = _mm_add_epi64(inputs_u64x2[1], increment_u64x2);
            inputs_u64x2[2] = _mm_add_epi64(inputs_u64x2[2], increment_u64x2);
            inputs_u64x2[3] = _mm_add_epi64(inputs_u64x2[3], increment_u64x2);
        }

        // Handle the tail of the buffer.
        {
            generated_u8x16[0] = _mm_aesenc_si128(inputs_u64x2[0], keys_u8x16[0]);
            generated_u8x16[1] = _mm_aesenc_si128(inputs_u64x2[1], keys_u8x16[1]);
            generated_u8x16[2] = _mm_aesenc_si128(inputs_u64x2[2], keys_u8x16[2]);
            generated_u8x16[3] = _mm_aesenc_si128(inputs_u64x2[3], keys_u8x16[3]);
            for (sz_size_t tail_index = 0; byte_index < length; ++byte_index, ++tail_index)
                text[byte_index] = ((sz_u8_t *)generated_u8x16)[tail_index];
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
