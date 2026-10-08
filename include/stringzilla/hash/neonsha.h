/**
 *  @file include/stringzilla/hash/neonsha.h
 *  @author Ash Vardanian
 *  @date October 3, 2025
 *  @brief NEON + SHA2 backend for string hashing and checksums.
 *
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_NEONSHA_H_
#define STRINGZILLA_HASH_NEONSHA_H_

#include "stringzilla/types.h"
#include "stringzilla/hash/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_ARM64_
#if STRINGZILLA_TARGET_NEONSHA
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd+crypto+sha2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd+crypto+sha2")
#endif

/**
 *  @brief Process a single 512-bit (64-byte) block of data using SHA256.
 *  @param[inout] hash Pointer to 8x 32-bit hash values, modified in place.
 *  @param[in] block Pointer to 64-byte message block.
 */
STRINGZILLA_INLINE void sz_sha256_process_block_neon_(
    sz_u32_t hash[sz_at_least_(8)], sz_u8_t const block[sz_at_least_(STRINGZILLA_SHA256_BLOCK_LENGTH)]) {
    sz_u32_t const *round_constants = sz_sha256_round_constants_();

    // Pre-load all round constants using multi-vector loads (4x16 = 64 bytes per load)
    uint32x4x4_t k_batch0 = vld1q_u32_x4(&round_constants[0]);  // k0-k3
    uint32x4x4_t k_batch1 = vld1q_u32_x4(&round_constants[16]); // k4-k7
    uint32x4x4_t k_batch2 = vld1q_u32_x4(&round_constants[32]); // k8-k11
    uint32x4x4_t k_batch3 = vld1q_u32_x4(&round_constants[48]); // k12-k15

    uint32x4_t k0 = k_batch0.val[0];
    uint32x4_t k1 = k_batch0.val[1];
    uint32x4_t k2 = k_batch0.val[2];
    uint32x4_t k3 = k_batch0.val[3];
    uint32x4_t k4 = k_batch1.val[0];
    uint32x4_t k5 = k_batch1.val[1];
    uint32x4_t k6 = k_batch1.val[2];
    uint32x4_t k7 = k_batch1.val[3];
    uint32x4_t k8 = k_batch2.val[0];
    uint32x4_t k9 = k_batch2.val[1];
    uint32x4_t k10 = k_batch2.val[2];
    uint32x4_t k11 = k_batch2.val[3];
    uint32x4_t k12 = k_batch3.val[0];
    uint32x4_t k13 = k_batch3.val[1];
    uint32x4_t k14 = k_batch3.val[2];
    uint32x4_t k15 = k_batch3.val[3];

    // Load current hash state
    uint32x4_t state0 = vld1q_u32(&hash[0]); // a, b, c, d
    uint32x4_t state1 = vld1q_u32(&hash[4]); // e, f, g, h
    uint32x4_t state0_saved = state0;
    uint32x4_t state1_saved = state1;

    // Load message schedule (big-endian)
    uint32x4_t msg0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(&block[0])));
    uint32x4_t msg1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(&block[16])));
    uint32x4_t msg2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(&block[32])));
    uint32x4_t msg3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(&block[48])));

    uint32x4_t scratch_0, scratch_1;

    // Rounds 0-3
    scratch_0 = vaddq_u32(msg0, k0);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);

    // Rounds 4-7
    scratch_0 = vaddq_u32(msg1, k1);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);

    // Rounds 8-11
    scratch_0 = vaddq_u32(msg2, k2);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);

    // Rounds 12-15: OpenSSL pattern - add K first, then message schedule during hash
    scratch_0 = vaddq_u32(msg3, k3);
    msg0 = vsha256su0q_u32(msg0, msg1);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg0 = vsha256su1q_u32(msg0, msg2, msg3);

    // Rounds 16-19
    scratch_0 = vaddq_u32(msg0, k4);
    msg1 = vsha256su0q_u32(msg1, msg2);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg1 = vsha256su1q_u32(msg1, msg3, msg0);

    // Rounds 20-23
    scratch_0 = vaddq_u32(msg1, k5);
    msg2 = vsha256su0q_u32(msg2, msg3);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg2 = vsha256su1q_u32(msg2, msg0, msg1);

    // Rounds 24-27
    scratch_0 = vaddq_u32(msg2, k6);
    msg3 = vsha256su0q_u32(msg3, msg0);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg3 = vsha256su1q_u32(msg3, msg1, msg2);

    // Rounds 28-31
    scratch_0 = vaddq_u32(msg3, k7);
    msg0 = vsha256su0q_u32(msg0, msg1);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg0 = vsha256su1q_u32(msg0, msg2, msg3);

    // Rounds 32-35
    scratch_0 = vaddq_u32(msg0, k8);
    msg1 = vsha256su0q_u32(msg1, msg2);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg1 = vsha256su1q_u32(msg1, msg3, msg0);

    // Rounds 36-39
    scratch_0 = vaddq_u32(msg1, k9);
    msg2 = vsha256su0q_u32(msg2, msg3);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg2 = vsha256su1q_u32(msg2, msg0, msg1);

    // Rounds 40-43
    scratch_0 = vaddq_u32(msg2, k10);
    msg3 = vsha256su0q_u32(msg3, msg0);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg3 = vsha256su1q_u32(msg3, msg1, msg2);

    // Rounds 44-47
    scratch_0 = vaddq_u32(msg3, k11);
    msg0 = vsha256su0q_u32(msg0, msg1);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg0 = vsha256su1q_u32(msg0, msg2, msg3);

    // Rounds 48-51
    scratch_0 = vaddq_u32(msg0, k12);
    msg1 = vsha256su0q_u32(msg1, msg2);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg1 = vsha256su1q_u32(msg1, msg3, msg0);

    // Rounds 52-55
    scratch_0 = vaddq_u32(msg1, k13);
    msg2 = vsha256su0q_u32(msg2, msg3);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg2 = vsha256su1q_u32(msg2, msg0, msg1);

    // Rounds 56-59
    scratch_0 = vaddq_u32(msg2, k14);
    msg3 = vsha256su0q_u32(msg3, msg0);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);
    msg3 = vsha256su1q_u32(msg3, msg1, msg2);

    // Rounds 60-63 (no next message to prepare, just process with msg3)
    scratch_0 = vaddq_u32(msg3, k15);
    scratch_1 = state0;
    state0 = vsha256hq_u32(state0, state1, scratch_0);
    state1 = vsha256h2q_u32(state1, scratch_1, scratch_0);

    // Add compressed chunk to current hash value
    state0 = vaddq_u32(state0, state0_saved);
    state1 = vaddq_u32(state1, state1_saved);

    // Store result back
    vst1q_u32(&hash[0], state0);
    vst1q_u32(&hash[4], state1);
}

STRINGZILLA_API sz_status_t sz_sha256_state_init_neonsha(sz_sha256_state_t *state, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    // Vectorize the load/store of 8x u32s using 2x 128-bit NEON loads
    sz_u32_t const *initial_hash = sz_sha256_initial_hash_();
    vst1q_u32(&state->hash[0], vld1q_u32(&initial_hash[0]));
    vst1q_u32(&state->hash[4], vld1q_u32(&initial_hash[4]));
    state->block_length = 0, state->total_length = 0;
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_sha256_state_update_neonsha(sz_sha256_state_t *state, sz_cptr_t text, sz_size_t length,
                                                           sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_u8_t const *input_cursor = (sz_u8_t const *)text;
    sz_size_t const current_block_index = state->block_length / STRINGZILLA_SHA256_BLOCK_LENGTH;
    sz_size_t const final_block_index = (state->block_length + length) / STRINGZILLA_SHA256_BLOCK_LENGTH;
    int const stays_in_the_block = current_block_index == final_block_index;
    int const fills_the_block = (state->block_length + length) % STRINGZILLA_SHA256_BLOCK_LENGTH == 0;

    state->total_length += length;

    // Fast path: stays in same block and doesn't fill it
    if (stays_in_the_block && !fills_the_block) {
        for (; length; --length, ++state->block_length, ++input_cursor)
            state->block[state->block_length] = *input_cursor;
        return sz_success_k;
    }

    // Calculate head, body, and tail lengths
    sz_size_t const head_length = (STRINGZILLA_SHA256_BLOCK_LENGTH - state->block_length) %
                                  STRINGZILLA_SHA256_BLOCK_LENGTH;
    sz_size_t const tail_length = (state->block_length + length) % STRINGZILLA_SHA256_BLOCK_LENGTH;
    sz_size_t const body_length = length - head_length - tail_length;

    // Copy hash to aligned local buffer
    sz_align_(16) sz_u32_t hash[8];
    vst1q_u32(&hash[0], vld1q_u32(&state->hash[0]));
    vst1q_u32(&hash[4], vld1q_u32(&state->hash[4]));

    // Process head to complete the current block
    if (head_length) {
        for (sz_size_t byte_index = 0; byte_index < head_length; ++byte_index)
            state->block[state->block_length++] = input_cursor[byte_index];
        sz_sha256_process_block_neon_(hash, state->block);
        state->block_length = 0;
        input_cursor += head_length;
    }

    // Process body (complete aligned blocks)
    for (sz_size_t processed = 0; processed < body_length;
         processed += STRINGZILLA_SHA256_BLOCK_LENGTH, input_cursor += STRINGZILLA_SHA256_BLOCK_LENGTH)
        sz_sha256_process_block_neon_(hash, input_cursor);

    // Process tail (remaining bytes into block buffer)
    for (sz_size_t byte_index = 0; byte_index < tail_length; ++byte_index)
        state->block[byte_index] = input_cursor[byte_index];
    state->block_length = tail_length;

    // Copy hash back
    vst1q_u32(&state->hash[0], vld1q_u32(&hash[0]));
    vst1q_u32(&state->hash[4], vld1q_u32(&hash[4]));
    return sz_success_k;
}

STRINGZILLA_INLINE void sz_sha256_state_digest_neonsha_(
    sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)]) {
    // Create a copy of the state for padding
    sz_sha256_state_t local_state = *state;

    // Append the '1' bit (0x80 byte) after the message
    local_state.block[local_state.block_length++] = 0x80;

    // If there's not enough room for the 64-bit length, pad this block and process it
    if (local_state.block_length > 56) {
        // Zero remaining bytes using vectorized writes
        sz_size_t remaining = STRINGZILLA_SHA256_BLOCK_LENGTH - local_state.block_length;
        sz_size_t vec_bytes = (remaining / 16) * 16;
        uint8x16_t zeros_u8x16 = vdupq_n_u8(0);
        for (sz_size_t byte_index = 0; byte_index < vec_bytes; byte_index += 16)
            vst1q_u8(&local_state.block[local_state.block_length + byte_index], zeros_u8x16);
        for (sz_size_t byte_index = vec_bytes; byte_index < remaining; ++byte_index)
            local_state.block[local_state.block_length + byte_index] = 0;
        sz_sha256_process_block_neon_(local_state.hash, local_state.block);
        local_state.block_length = 0;
    }

    // Pad with zeros until we have 56 bytes
    sz_size_t remaining = 56 - local_state.block_length;
    sz_size_t vec_bytes = (remaining / 16) * 16;
    uint8x16_t zeros_u8x16 = vdupq_n_u8(0);
    for (sz_size_t byte_index = 0; byte_index < vec_bytes; byte_index += 16)
        vst1q_u8(&local_state.block[local_state.block_length + byte_index], zeros_u8x16);
    for (sz_size_t byte_index = vec_bytes; byte_index < remaining; ++byte_index)
        local_state.block[local_state.block_length + byte_index] = 0;
    local_state.block_length = 56;

    // Append the message length in bits as a 64-bit big-endian integer
    sz_u64_t bit_length = local_state.total_length * 8;
    local_state.block[56] = (sz_u8_t)(bit_length >> 56);
    local_state.block[57] = (sz_u8_t)(bit_length >> 48);
    local_state.block[58] = (sz_u8_t)(bit_length >> 40);
    local_state.block[59] = (sz_u8_t)(bit_length >> 32);
    local_state.block[60] = (sz_u8_t)(bit_length >> 24);
    local_state.block[61] = (sz_u8_t)(bit_length >> 16);
    local_state.block[62] = (sz_u8_t)(bit_length >> 8);
    local_state.block[63] = (sz_u8_t)(bit_length >> 0);

    // Process the final block
    sz_sha256_process_block_neon_(local_state.hash, local_state.block);

    // Produce the final hash digest in big-endian format
    for (sz_size_t lane_index = 0; lane_index < 8; ++lane_index) {
        digest[lane_index * 4 + 0] = (sz_u8_t)(local_state.hash[lane_index] >> 24);
        digest[lane_index * 4 + 1] = (sz_u8_t)(local_state.hash[lane_index] >> 16);
        digest[lane_index * 4 + 2] = (sz_u8_t)(local_state.hash[lane_index] >> 8);
        digest[lane_index * 4 + 3] = (sz_u8_t)(local_state.hash[lane_index] >> 0);
    }
}

STRINGZILLA_API sz_status_t sz_sha256_state_digest_neonsha(
    sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)],
    sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_sha256_state_digest_neonsha_(state, digest);
    return sz_success_k;
}

/** Runs four SHA-256 rounds of one lane over the next four schedule words. */
STRINGZILLA_INLINE void sz_sha256_quad_round_neonsha_(uint32x4_t *abcd_u32x4, uint32x4_t *efgh_u32x4,
                                                      uint32x4_t schedule_u32x4, uint32x4_t constants_u32x4) {
    uint32x4_t const words_u32x4 = vaddq_u32(schedule_u32x4, constants_u32x4);
    uint32x4_t const abcd_before_u32x4 = *abcd_u32x4;
    *abcd_u32x4 = vsha256hq_u32(abcd_before_u32x4, *efgh_u32x4, words_u32x4);
    *efgh_u32x4 = vsha256h2q_u32(*efgh_u32x4, abcd_before_u32x4, words_u32x4);
}

/** Derives the next four message schedule words from the previous sixteen, oldest quarter first. */
STRINGZILLA_INLINE uint32x4_t sz_sha256_extend_neonsha_(uint32x4_t oldest_words_u32x4, uint32x4_t next_words_u32x4,
                                                        uint32x4_t ninth_words_u32x4, uint32x4_t newest_words_u32x4) {
    return vsha256su1q_u32(vsha256su0q_u32(oldest_words_u32x4, next_words_u32x4), ninth_words_u32x4,
                           newest_words_u32x4);
}

/**
 *  @brief Compresses one 64-byte block in each of four independent lanes.
 *  @param[inout] hashes_u32x4 Lane k keeps @c abcd at index 2k and @c efgh at 2k + 1.
 *  @param[in] lane_blocks One 64-byte block per lane.
 *  @param[in] active_bitmask Lanes whose result lands; the rest keep their hash bit-for-bit.
 *
 *  Every @c SHA256H waits on the one before it, so a single lane leaves the crypto pipes idle for
 *  most of each quad-round; issuing the four lanes' quad-rounds back to back fills that latency
 *  with independent work from the other lanes.
 */
STRINGZILLA_INLINE void sz_sha256_compress_x4_neonsha_(uint32x4_t hashes_u32x4[8], sz_u8_t const *const lane_blocks[4],
                                                       sz_u32_t active_bitmask) {
    sz_u32_t const *round_constants = sz_sha256_round_constants_();
    uint32x4_t abcd0_u32x4 = hashes_u32x4[0], efgh0_u32x4 = hashes_u32x4[1];
    uint32x4_t abcd1_u32x4 = hashes_u32x4[2], efgh1_u32x4 = hashes_u32x4[3];
    uint32x4_t abcd2_u32x4 = hashes_u32x4[4], efgh2_u32x4 = hashes_u32x4[5];
    uint32x4_t abcd3_u32x4 = hashes_u32x4[6], efgh3_u32x4 = hashes_u32x4[7];
    uint32x4_t schedule0_u32x4[4], schedule1_u32x4[4], schedule2_u32x4[4], schedule3_u32x4[4];
    uint32x4_t constants_u32x4;

    for (sz_size_t quarter_index = 0; quarter_index != 4; ++quarter_index) {
        schedule0_u32x4[quarter_index] = vreinterpretq_u32_u8(
            vrev32q_u8(vld1q_u8(lane_blocks[0] + quarter_index * 16)));
        schedule1_u32x4[quarter_index] = vreinterpretq_u32_u8(
            vrev32q_u8(vld1q_u8(lane_blocks[1] + quarter_index * 16)));
        schedule2_u32x4[quarter_index] = vreinterpretq_u32_u8(
            vrev32q_u8(vld1q_u8(lane_blocks[2] + quarter_index * 16)));
        schedule3_u32x4[quarter_index] = vreinterpretq_u32_u8(
            vrev32q_u8(vld1q_u8(lane_blocks[3] + quarter_index * 16)));
    }

    for (sz_size_t quarter_index = 0; quarter_index != 4; ++quarter_index) {
        constants_u32x4 = vld1q_u32(round_constants + quarter_index * 4);
        sz_sha256_quad_round_neonsha_(&abcd0_u32x4, &efgh0_u32x4, schedule0_u32x4[quarter_index], constants_u32x4);
        sz_sha256_quad_round_neonsha_(&abcd1_u32x4, &efgh1_u32x4, schedule1_u32x4[quarter_index], constants_u32x4);
        sz_sha256_quad_round_neonsha_(&abcd2_u32x4, &efgh2_u32x4, schedule2_u32x4[quarter_index], constants_u32x4);
        sz_sha256_quad_round_neonsha_(&abcd3_u32x4, &efgh3_u32x4, schedule3_u32x4[quarter_index], constants_u32x4);
    }

    for (sz_size_t turn_index = 1; turn_index != 4; ++turn_index) {
        sz_u32_t const *turn_constants = round_constants + turn_index * 16;

        constants_u32x4 = vld1q_u32(turn_constants + 0);
        schedule0_u32x4[0] = sz_sha256_extend_neonsha_(schedule0_u32x4[0], schedule0_u32x4[1], schedule0_u32x4[2],
                                                       schedule0_u32x4[3]);
        schedule1_u32x4[0] = sz_sha256_extend_neonsha_(schedule1_u32x4[0], schedule1_u32x4[1], schedule1_u32x4[2],
                                                       schedule1_u32x4[3]);
        schedule2_u32x4[0] = sz_sha256_extend_neonsha_(schedule2_u32x4[0], schedule2_u32x4[1], schedule2_u32x4[2],
                                                       schedule2_u32x4[3]);
        schedule3_u32x4[0] = sz_sha256_extend_neonsha_(schedule3_u32x4[0], schedule3_u32x4[1], schedule3_u32x4[2],
                                                       schedule3_u32x4[3]);
        sz_sha256_quad_round_neonsha_(&abcd0_u32x4, &efgh0_u32x4, schedule0_u32x4[0], constants_u32x4);
        sz_sha256_quad_round_neonsha_(&abcd1_u32x4, &efgh1_u32x4, schedule1_u32x4[0], constants_u32x4);
        sz_sha256_quad_round_neonsha_(&abcd2_u32x4, &efgh2_u32x4, schedule2_u32x4[0], constants_u32x4);
        sz_sha256_quad_round_neonsha_(&abcd3_u32x4, &efgh3_u32x4, schedule3_u32x4[0], constants_u32x4);

        constants_u32x4 = vld1q_u32(turn_constants + 4);
        schedule0_u32x4[1] = sz_sha256_extend_neonsha_(schedule0_u32x4[1], schedule0_u32x4[2], schedule0_u32x4[3],
                                                       schedule0_u32x4[0]);
        schedule1_u32x4[1] = sz_sha256_extend_neonsha_(schedule1_u32x4[1], schedule1_u32x4[2], schedule1_u32x4[3],
                                                       schedule1_u32x4[0]);
        schedule2_u32x4[1] = sz_sha256_extend_neonsha_(schedule2_u32x4[1], schedule2_u32x4[2], schedule2_u32x4[3],
                                                       schedule2_u32x4[0]);
        schedule3_u32x4[1] = sz_sha256_extend_neonsha_(schedule3_u32x4[1], schedule3_u32x4[2], schedule3_u32x4[3],
                                                       schedule3_u32x4[0]);
        sz_sha256_quad_round_neonsha_(&abcd0_u32x4, &efgh0_u32x4, schedule0_u32x4[1], constants_u32x4);
        sz_sha256_quad_round_neonsha_(&abcd1_u32x4, &efgh1_u32x4, schedule1_u32x4[1], constants_u32x4);
        sz_sha256_quad_round_neonsha_(&abcd2_u32x4, &efgh2_u32x4, schedule2_u32x4[1], constants_u32x4);
        sz_sha256_quad_round_neonsha_(&abcd3_u32x4, &efgh3_u32x4, schedule3_u32x4[1], constants_u32x4);

        constants_u32x4 = vld1q_u32(turn_constants + 8);
        schedule0_u32x4[2] = sz_sha256_extend_neonsha_(schedule0_u32x4[2], schedule0_u32x4[3], schedule0_u32x4[0],
                                                       schedule0_u32x4[1]);
        schedule1_u32x4[2] = sz_sha256_extend_neonsha_(schedule1_u32x4[2], schedule1_u32x4[3], schedule1_u32x4[0],
                                                       schedule1_u32x4[1]);
        schedule2_u32x4[2] = sz_sha256_extend_neonsha_(schedule2_u32x4[2], schedule2_u32x4[3], schedule2_u32x4[0],
                                                       schedule2_u32x4[1]);
        schedule3_u32x4[2] = sz_sha256_extend_neonsha_(schedule3_u32x4[2], schedule3_u32x4[3], schedule3_u32x4[0],
                                                       schedule3_u32x4[1]);
        sz_sha256_quad_round_neonsha_(&abcd0_u32x4, &efgh0_u32x4, schedule0_u32x4[2], constants_u32x4);
        sz_sha256_quad_round_neonsha_(&abcd1_u32x4, &efgh1_u32x4, schedule1_u32x4[2], constants_u32x4);
        sz_sha256_quad_round_neonsha_(&abcd2_u32x4, &efgh2_u32x4, schedule2_u32x4[2], constants_u32x4);
        sz_sha256_quad_round_neonsha_(&abcd3_u32x4, &efgh3_u32x4, schedule3_u32x4[2], constants_u32x4);

        constants_u32x4 = vld1q_u32(turn_constants + 12);
        schedule0_u32x4[3] = sz_sha256_extend_neonsha_(schedule0_u32x4[3], schedule0_u32x4[0], schedule0_u32x4[1],
                                                       schedule0_u32x4[2]);
        schedule1_u32x4[3] = sz_sha256_extend_neonsha_(schedule1_u32x4[3], schedule1_u32x4[0], schedule1_u32x4[1],
                                                       schedule1_u32x4[2]);
        schedule2_u32x4[3] = sz_sha256_extend_neonsha_(schedule2_u32x4[3], schedule2_u32x4[0], schedule2_u32x4[1],
                                                       schedule2_u32x4[2]);
        schedule3_u32x4[3] = sz_sha256_extend_neonsha_(schedule3_u32x4[3], schedule3_u32x4[0], schedule3_u32x4[1],
                                                       schedule3_u32x4[2]);
        sz_sha256_quad_round_neonsha_(&abcd0_u32x4, &efgh0_u32x4, schedule0_u32x4[3], constants_u32x4);
        sz_sha256_quad_round_neonsha_(&abcd1_u32x4, &efgh1_u32x4, schedule1_u32x4[3], constants_u32x4);
        sz_sha256_quad_round_neonsha_(&abcd2_u32x4, &efgh2_u32x4, schedule2_u32x4[3], constants_u32x4);
        sz_sha256_quad_round_neonsha_(&abcd3_u32x4, &efgh3_u32x4, schedule3_u32x4[3], constants_u32x4);
    }

    uint32x4_t const active0_u32x4 = vdupq_n_u32(0u - ((active_bitmask >> 0) & 1u));
    uint32x4_t const active1_u32x4 = vdupq_n_u32(0u - ((active_bitmask >> 1) & 1u));
    uint32x4_t const active2_u32x4 = vdupq_n_u32(0u - ((active_bitmask >> 2) & 1u));
    uint32x4_t const active3_u32x4 = vdupq_n_u32(0u - ((active_bitmask >> 3) & 1u));
    hashes_u32x4[0] = vbslq_u32(active0_u32x4, vaddq_u32(hashes_u32x4[0], abcd0_u32x4), hashes_u32x4[0]);
    hashes_u32x4[1] = vbslq_u32(active0_u32x4, vaddq_u32(hashes_u32x4[1], efgh0_u32x4), hashes_u32x4[1]);
    hashes_u32x4[2] = vbslq_u32(active1_u32x4, vaddq_u32(hashes_u32x4[2], abcd1_u32x4), hashes_u32x4[2]);
    hashes_u32x4[3] = vbslq_u32(active1_u32x4, vaddq_u32(hashes_u32x4[3], efgh1_u32x4), hashes_u32x4[3]);
    hashes_u32x4[4] = vbslq_u32(active2_u32x4, vaddq_u32(hashes_u32x4[4], abcd2_u32x4), hashes_u32x4[4]);
    hashes_u32x4[5] = vbslq_u32(active2_u32x4, vaddq_u32(hashes_u32x4[5], efgh2_u32x4), hashes_u32x4[5]);
    hashes_u32x4[6] = vbslq_u32(active3_u32x4, vaddq_u32(hashes_u32x4[6], abcd3_u32x4), hashes_u32x4[6]);
    hashes_u32x4[7] = vbslq_u32(active3_u32x4, vaddq_u32(hashes_u32x4[7], efgh3_u32x4), hashes_u32x4[7]);
}

/**
 *  @brief Compresses each of up to four lanes' own run of whole blocks, plus an optional buffered
 *      block ahead of them.
 *  @param[inout] states The lane states, whose hash words are loaded on entry and stored on exit.
 *  @param[in] active_lanes_count Lanes owning a state, 1 to 4; others borrow lane zero.
 *  @param[in] buffered_bitmask Lanes whose @c block the caller filled to 64 bytes, first.
 *  @param[inout] cursors Per-lane read positions, advanced past every block consumed.
 *  @param[in] blocks_per_lane Whole 64-byte blocks each lane owns; the loop runs to the largest.
 *
 *  A retiring lane parks on its last full block instead of stepping onto its short tail, so every
 *  read stays in bounds, and its result is masked off. A lane owning no blocks reads its own
 *  @c block buffer, which is always 64 valid bytes.
 */
STRINGZILLA_INLINE void sz_sha256_multistate_blocks_neonsha_(sz_sha256_state_t *states, sz_size_t active_lanes_count,
                                                             sz_u32_t buffered_bitmask, sz_u8_t const **cursors,
                                                             sz_size_t const *blocks_per_lane) {
    uint32x4_t hashes_u32x4[8];
    sz_u8_t const *sources[4];
    sz_size_t blocks_left[4];
    sz_size_t largest_blocks_count = 0;

    for (sz_size_t lane_index = 0; lane_index != 4; ++lane_index) {
        sz_size_t const source_lane = lane_index < active_lanes_count ? lane_index : 0;
        blocks_left[lane_index] = lane_index < active_lanes_count ? blocks_per_lane[lane_index] : 0;
        sources[lane_index] = states[source_lane].block;
        hashes_u32x4[lane_index * 2 + 0] = vld1q_u32(&states[source_lane].hash[0]);
        hashes_u32x4[lane_index * 2 + 1] = vld1q_u32(&states[source_lane].hash[4]);
        if (blocks_left[lane_index] > largest_blocks_count) largest_blocks_count = blocks_left[lane_index];
    }
    if (buffered_bitmask == 0 && largest_blocks_count == 0) return;

    if (buffered_bitmask) sz_sha256_compress_x4_neonsha_(hashes_u32x4, sources, buffered_bitmask);
    for (sz_size_t lane_index = 0; lane_index != 4; ++lane_index)
        if (blocks_left[lane_index] != 0) sources[lane_index] = cursors[lane_index];

    for (sz_size_t block_index = 0; block_index != largest_blocks_count; ++block_index) {
        sz_u32_t active_bitmask = 0;
        for (sz_size_t lane_index = 0; lane_index != 4; ++lane_index)
            active_bitmask |= (sz_u32_t)(blocks_left[lane_index] != 0) << lane_index;
        sz_sha256_compress_x4_neonsha_(hashes_u32x4, sources, active_bitmask);
        for (sz_size_t lane_index = 0; lane_index != 4; ++lane_index) {
            sources[lane_index] += (blocks_left[lane_index] > 1) * STRINGZILLA_SHA256_BLOCK_LENGTH;
            blocks_left[lane_index] -= blocks_left[lane_index] != 0;
        }
    }

    for (sz_size_t lane_index = 0; lane_index != active_lanes_count; ++lane_index) {
        vst1q_u32(&states[lane_index].hash[0], hashes_u32x4[lane_index * 2 + 0]);
        vst1q_u32(&states[lane_index].hash[4], hashes_u32x4[lane_index * 2 + 1]);
        cursors[lane_index] += blocks_per_lane[lane_index] * STRINGZILLA_SHA256_BLOCK_LENGTH;
    }
}

STRINGZILLA_API sz_status_t sz_sha256_multistate_update_neonsha(sz_sha256_state_t *states, sz_sequence_t const *texts,
                                                                sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_size_t const lanes_count = texts->count;

    for (sz_size_t first_lane_index = 0; first_lane_index < lanes_count; first_lane_index += 4) {
        sz_size_t const lanes_left = lanes_count - first_lane_index;
        sz_size_t const active_lanes_count = lanes_left < 4 ? lanes_left : 4;
        sz_u8_t const *cursors[4];
        sz_size_t remaining[4], blocks_per_lane[4];
        sz_u32_t buffered_bitmask = 0;

        // Top up any buffered partial block into the state's own 64-byte buffer, which then serves
        // as that lane's first block source; the whole chunk is charged to `total_length` once.
        for (sz_size_t lane_index = 0; lane_index != active_lanes_count; ++lane_index) {
            sz_sha256_state_t *const state = &states[first_lane_index + lane_index];
            cursors[lane_index] = (sz_u8_t const *)texts->get_start(texts->handle, first_lane_index + lane_index);
            remaining[lane_index] = texts->get_length(texts->handle, first_lane_index + lane_index);
            state->total_length += remaining[lane_index];
            if (state->block_length != 0) {
                sz_size_t const missing = STRINGZILLA_SHA256_BLOCK_LENGTH - state->block_length;
                if (remaining[lane_index] >= missing) {
                    for (sz_size_t byte_index = 0; byte_index != missing; ++byte_index)
                        state->block[state->block_length + byte_index] = cursors[lane_index][byte_index];
                    buffered_bitmask |= (sz_u32_t)1 << lane_index;
                    state->block_length = 0;
                    cursors[lane_index] += missing, remaining[lane_index] -= missing;
                }
            }
            blocks_per_lane[lane_index] = remaining[lane_index] / STRINGZILLA_SHA256_BLOCK_LENGTH;
        }

        sz_sha256_multistate_blocks_neonsha_(&states[first_lane_index], active_lanes_count, buffered_bitmask, cursors,
                                             blocks_per_lane);

        // Whatever is left cannot fill a block, so it only ever buffers.
        for (sz_size_t lane_index = 0; lane_index != active_lanes_count; ++lane_index) {
            sz_sha256_state_t *const state = &states[first_lane_index + lane_index];
            sz_size_t const tail_length = remaining[lane_index] % STRINGZILLA_SHA256_BLOCK_LENGTH;
            for (sz_size_t byte_index = 0; byte_index != tail_length; ++byte_index)
                state->block[state->block_length + byte_index] = cursors[lane_index][byte_index];
            state->block_length += tail_length;
        }
    }
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_sha256_multistate_digest_neonsha(sz_sha256_state_t const *states, sz_size_t states_count,
                                                                sz_u8_t *digests, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    for (sz_size_t lane_index = 0; lane_index != states_count; ++lane_index)
        sz_sha256_state_digest_neonsha_(&states[lane_index], &digests[lane_index * STRINGZILLA_SHA256_DIGEST_LENGTH]);
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_NEONSHA
#endif // STRINGZILLA_ARCH_ARM64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_NEONSHA_H_
