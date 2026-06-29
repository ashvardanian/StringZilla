/**
 *  @brief Goldmont (SHA-NI) backend for string hashing and checksums.
 *  @file include/stringzilla/hash/goldmont.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_GOLDMONT_H_
#define STRINGZILLA_HASH_GOLDMONT_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/hash/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_GOLDMONT
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sse3,ssse3,sse4.1,sha"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("sse3", "ssse3", "sse4.1", "sha")
#endif

/**
 *  @brief Process a single 512-bit (64-byte) block of data using SHA256 with SHA-NI intrinsics.
 *  @param hash Pointer to 8x 32-bit hash values, modified in place.
 *  @param block Pointer to 64-byte message block.
 */
SZ_HELPER_AUTO void sz_sha256_process_block_goldmont_(sz_u32_t hash[sz_at_least_(8)],
                                                      sz_u8_t const block[sz_at_least_(64)]) {
    sz_u32_t const *round_constants = sz_sha256_round_constants_();

    // Load and byte-swap the first 16 words (big-endian) using SSE
    __m128i const bswap_mask = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    __m128i msg0 = _mm_shuffle_epi8(_mm_lddqu_si128((__m128i const *)&block[0]), bswap_mask);
    __m128i msg1 = _mm_shuffle_epi8(_mm_lddqu_si128((__m128i const *)&block[16]), bswap_mask);
    __m128i msg2 = _mm_shuffle_epi8(_mm_lddqu_si128((__m128i const *)&block[32]), bswap_mask);
    __m128i msg3 = _mm_shuffle_epi8(_mm_lddqu_si128((__m128i const *)&block[48]), bswap_mask);

    // Load initial hash values and pack into SHA-NI state format
    // SHA-NI uses ABEF/CDGH layout instead of ABCD/EFGH
    __m128i state0 = _mm_lddqu_si128((__m128i const *)&hash[0]); // A B C D
    __m128i state1 = _mm_lddqu_si128((__m128i const *)&hash[4]); // E F G H
    __m128i perm_u32x4 = _mm_shuffle_epi32(state0, 0xB1);        // CDAB
    state1 = _mm_shuffle_epi32(state1, 0x1B);                    // HGFE
    state0 = _mm_alignr_epi8(perm_u32x4, state1, 8);             // ABEF
    state1 = _mm_blend_epi16(state1, perm_u32x4, 0xF0);          // CDGH

    __m128i state0_saved = state0;
    __m128i state1_saved = state1;

    // Rounds 0-3
    __m128i round_input = _mm_add_epi32(msg0, _mm_lddqu_si128((__m128i const *)&round_constants[0]));
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);

    // Rounds 4-7
    round_input = _mm_add_epi32(msg1, _mm_lddqu_si128((__m128i const *)&round_constants[4]));
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);

    // Rounds 8-11
    round_input = _mm_add_epi32(msg2, _mm_lddqu_si128((__m128i const *)&round_constants[8]));
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);

    // Rounds 12-15
    round_input = _mm_add_epi32(msg3, _mm_lddqu_si128((__m128i const *)&round_constants[12]));
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);

    // Rounds 16-19
    round_input = _mm_add_epi32(msg0, _mm_lddqu_si128((__m128i const *)&round_constants[16]));
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);

    // Rounds 20-23
    round_input = _mm_add_epi32(msg1, _mm_lddqu_si128((__m128i const *)&round_constants[20]));
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);

    // Rounds 24-27
    round_input = _mm_add_epi32(msg2, _mm_lddqu_si128((__m128i const *)&round_constants[24]));
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);

    // Rounds 28-31
    round_input = _mm_add_epi32(msg3, _mm_lddqu_si128((__m128i const *)&round_constants[28]));
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);

    // Rounds 32-35
    round_input = _mm_add_epi32(msg0, _mm_lddqu_si128((__m128i const *)&round_constants[32]));
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);

    // Rounds 36-39
    round_input = _mm_add_epi32(msg1, _mm_lddqu_si128((__m128i const *)&round_constants[36]));
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);

    // Rounds 40-43
    round_input = _mm_add_epi32(msg2, _mm_lddqu_si128((__m128i const *)&round_constants[40]));
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);

    // Rounds 44-47
    round_input = _mm_add_epi32(msg3, _mm_lddqu_si128((__m128i const *)&round_constants[44]));
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);

    // Rounds 48-51
    round_input = _mm_add_epi32(msg0, _mm_lddqu_si128((__m128i const *)&round_constants[48]));
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);

    // Rounds 52-55
    round_input = _mm_add_epi32(msg1, _mm_lddqu_si128((__m128i const *)&round_constants[52]));
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);

    // Rounds 56-59
    round_input = _mm_add_epi32(msg2, _mm_lddqu_si128((__m128i const *)&round_constants[56]));
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);

    // Rounds 60-63
    round_input = _mm_add_epi32(msg3, _mm_lddqu_si128((__m128i const *)&round_constants[60]));
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);

    // Add compressed chunk to current hash value
    state0 = _mm_add_epi32(state0, state0_saved);
    state1 = _mm_add_epi32(state1, state1_saved);

    // Unpack from SHA-NI state format (ABEF/CDGH) back to ABCD/EFGH
    perm_u32x4 = _mm_shuffle_epi32(state0, 0x1B);       // FEBA
    state1 = _mm_shuffle_epi32(state1, 0xB1);           // GHCD
    state0 = _mm_blend_epi16(perm_u32x4, state1, 0xF0); // ABCD
    state1 = _mm_alignr_epi8(state1, perm_u32x4, 8);    // EFGH

    // Store hash values
    _mm_storeu_si128((__m128i *)&hash[0], state0);
    _mm_storeu_si128((__m128i *)&hash[4], state1);
}

SZ_API_COMPTIME void sz_sha256_state_init_goldmont(sz_sha256_state_t *state_ptr) {
    // Vectorize the load/store of 8x u32s using 2x 128-bit SSE loads
    sz_u32_t const *initial_hash = sz_sha256_initial_hash_();
    _mm_storeu_si128((__m128i *)&state_ptr->hash[0], _mm_lddqu_si128((__m128i const *)&initial_hash[0]));
    _mm_storeu_si128((__m128i *)&state_ptr->hash[4], _mm_lddqu_si128((__m128i const *)&initial_hash[4]));
    state_ptr->block_length = 0, state_ptr->total_length = 0;
}

SZ_API_COMPTIME void sz_sha256_state_update_goldmont(sz_sha256_state_t *state_ptr, sz_cptr_t data, sz_size_t length) {
    sz_u8_t const *input = (sz_u8_t const *)data;
    sz_size_t const current_block_index = state_ptr->block_length / 64;
    sz_size_t const final_block_index = (state_ptr->block_length + length) / 64;
    int const stays_in_the_block = current_block_index == final_block_index;
    int const fills_the_block = (state_ptr->block_length + length) % 64 == 0;

    state_ptr->total_length += length;

    // Fast path: stays in same block and doesn't fill it
    if (stays_in_the_block && !fills_the_block) {
        for (; length; --length, ++state_ptr->block_length, ++input) state_ptr->block[state_ptr->block_length] = *input;
        return;
    }

    // Calculate head, body, and tail lengths
    sz_size_t const head_length = (64 - state_ptr->block_length) % 64;
    sz_size_t const tail_length = (state_ptr->block_length + length) % 64;
    sz_size_t const body_length = length - head_length - tail_length;

    // Copy hash to aligned local buffer
    sz_align_(16) sz_u32_t hash[8];
    _mm_store_si128((__m128i *)&hash[0], _mm_lddqu_si128((__m128i const *)&state_ptr->hash[0]));
    _mm_store_si128((__m128i *)&hash[4], _mm_lddqu_si128((__m128i const *)&state_ptr->hash[4]));

    // Process head to complete the current block
    if (head_length) {
        for (sz_size_t byte_index = 0; byte_index < head_length; ++byte_index)
            state_ptr->block[state_ptr->block_length++] = input[byte_index];
        sz_sha256_process_block_goldmont_(hash, state_ptr->block);
        state_ptr->block_length = 0;
        input += head_length;
    }

    // Process body (complete aligned blocks)
    for (sz_size_t processed = 0; processed < body_length; processed += 64, input += 64)
        sz_sha256_process_block_goldmont_(hash, input);

    // Process tail (remaining bytes into block buffer)
    for (sz_size_t byte_index = 0; byte_index < tail_length; ++byte_index)
        state_ptr->block[byte_index] = input[byte_index];
    state_ptr->block_length = tail_length;

    // Copy hash back
    _mm_storeu_si128((__m128i *)&state_ptr->hash[0], _mm_load_si128((__m128i const *)&hash[0]));
    _mm_storeu_si128((__m128i *)&state_ptr->hash[4], _mm_load_si128((__m128i const *)&hash[4]));
}

SZ_API_COMPTIME void sz_sha256_state_digest_goldmont(sz_sha256_state_t const *state_ptr,
                                                     sz_u8_t digest[sz_at_least_(32)]) {
    // Create a copy of the state for padding
    sz_sha256_state_t state = *state_ptr;

    // Append the '1' bit (0x80 byte) after the message
    state.block[state.block_length++] = 0x80;

    // If there's not enough room for the 64-bit length, pad this block and process it
    if (state.block_length > 56) {
        // Zero remaining bytes using 128-bit vectorized writes
        sz_size_t remaining = 64 - state.block_length;
        sz_size_t xmm_bytes = (remaining / 16) * 16;
        for (sz_size_t byte_index = 0; byte_index < xmm_bytes; byte_index += 16)
            _mm_storeu_si128((__m128i *)&state.block[state.block_length + byte_index], _mm_setzero_si128());
        for (sz_size_t byte_index = xmm_bytes; byte_index < remaining; ++byte_index)
            state.block[state.block_length + byte_index] = 0;
        sz_sha256_process_block_goldmont_(state.hash, state.block);
        state.block_length = 0;
    }

    // Pad with zeros until we have 56 bytes
    sz_size_t remaining = 56 - state.block_length;
    sz_size_t xmm_bytes = (remaining / 16) * 16;
    for (sz_size_t byte_index = 0; byte_index < xmm_bytes; byte_index += 16)
        _mm_storeu_si128((__m128i *)&state.block[state.block_length + byte_index], _mm_setzero_si128());
    for (sz_size_t byte_index = xmm_bytes; byte_index < remaining; ++byte_index)
        state.block[state.block_length + byte_index] = 0;
    state.block_length = 56;

    // Append the message length in bits as a 64-bit big-endian integer
    sz_u64_t bit_length = state.total_length * 8;
    state.block[56] = (sz_u8_t)(bit_length >> 56);
    state.block[57] = (sz_u8_t)(bit_length >> 48);
    state.block[58] = (sz_u8_t)(bit_length >> 40);
    state.block[59] = (sz_u8_t)(bit_length >> 32);
    state.block[60] = (sz_u8_t)(bit_length >> 24);
    state.block[61] = (sz_u8_t)(bit_length >> 16);
    state.block[62] = (sz_u8_t)(bit_length >> 8);
    state.block[63] = (sz_u8_t)(bit_length >> 0);

    // Process the final block
    sz_sha256_process_block_goldmont_(state.hash, state.block);

    // Produce the final hash digest in big-endian format
    for (sz_size_t lane_index = 0; lane_index < 8; ++lane_index) {
        digest[lane_index * 4 + 0] = (sz_u8_t)(state.hash[lane_index] >> 24);
        digest[lane_index * 4 + 1] = (sz_u8_t)(state.hash[lane_index] >> 16);
        digest[lane_index * 4 + 2] = (sz_u8_t)(state.hash[lane_index] >> 8);
        digest[lane_index * 4 + 3] = (sz_u8_t)(state.hash[lane_index] >> 0);
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_GOLDMONT

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_GOLDMONT_H_
