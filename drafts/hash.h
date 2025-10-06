/**
 *  @file draft/hash.h
 *  @brief Draft SVE2 hash implementations for both 128-bit and 256-bit registers
 *
 *  This file contains experimental SVE2 hash implementations that were tested
 *  but showed no performance benefit over NEON on Graviton3 (256-bit SVE).
 *  Kept for reference and potential future optimization.
 */

#pragma once

// ========================================================================
// 128-bit SVE Implementation (original approach)
// ========================================================================

/**
 *  @brief  Emulates the Intel's AES-NI `AESENC` instruction with Arm SVE2.
 *  @see    "Emulating x86 AES Intrinsics on ARMv8-A" by Michael Brase:
 *          https://blog.michaelbrase.com/2018/05/08/emulating-x86-aes-intrinsics-on-armv8-a/
 */
SZ_INTERNAL svuint8_t sz_emulate_aesenc_u8x16_sve2_(svuint8_t state_vec, svuint8_t round_key_vec) {
    return sveor_u8_x(svptrue_b8(), svaesmc_u8(svaese_u8(state_vec, svdup_n_u8(0))), round_key_vec);
}

SZ_INTERNAL svuint64_t sz_emulate_aesenc_u64x2_sve2_(svuint64_t state_vec, svuint64_t round_key_vec) {
    return svreinterpret_u64_u8(sz_emulate_aesenc_u8x16_sve2_( //
        svreinterpret_u8_u64(state_vec),                       //
        svreinterpret_u8_u64(round_key_vec)));
}

/**
 *  @brief  Hash implementation assuming 128-bit SVE registers
 *  Uses misaligned split-loads approach with 4x 16-byte loads per iteration
 */
SZ_PUBLIC sz_u64_t sz_hash_sve2_b128(sz_cptr_t text, sz_size_t length, sz_u64_t seed) {
    if (length <= 16) { return sz_hash_sve2_upto16_(text, length, seed); }
    else if (length <= 32) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);

        sz_u128_vec_t data0_vec, data1_vec;
        svbool_t ptrue = svptrue_b8();
        sz_size_t tail_len = length - 16;
        svbool_t tail_mask = svwhilelt_b8((sz_u64_t)0, (sz_u64_t)tail_len);

        data0_vec.u8x16 = svget_neonq_u8(svldnt1_u8(ptrue, (sz_u8_t const *)(text + 0)));
        data1_vec.u8x16 = svget_neonq_u8(svldnt1_u8(tail_mask, (sz_u8_t const *)(text + 16)));

        sz_hash_minimal_update_neon_(&state, data0_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data1_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else if (length <= 48) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);

        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        svbool_t ptrue = svptrue_b8();
        sz_size_t tail_len = length - 32;
        svbool_t tail_mask = svwhilelt_b8((sz_u64_t)0, (sz_u64_t)tail_len);

        data0_vec.u8x16 = svget_neonq_u8(svldnt1_u8(ptrue, (sz_u8_t const *)(text + 0)));
        data1_vec.u8x16 = svget_neonq_u8(svldnt1_u8(ptrue, (sz_u8_t const *)(text + 16)));
        data2_vec.u8x16 = svget_neonq_u8(svldnt1_u8(tail_mask, (sz_u8_t const *)(text + 32)));

        sz_hash_minimal_update_neon_(&state, data0_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data1_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data2_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else if (length <= 64) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);

        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        svbool_t ptrue = svptrue_b8();
        sz_size_t tail_len = length - 48;
        svbool_t tail_mask = svwhilelt_b8((sz_u64_t)0, (sz_u64_t)tail_len);

        data0_vec.u8x16 = svget_neonq_u8(svldnt1_u8(ptrue, (sz_u8_t const *)(text + 0)));
        data1_vec.u8x16 = svget_neonq_u8(svldnt1_u8(ptrue, (sz_u8_t const *)(text + 16)));
        data2_vec.u8x16 = svget_neonq_u8(svldnt1_u8(ptrue, (sz_u8_t const *)(text + 32)));
        data3_vec.u8x16 = svget_neonq_u8(svldnt1_u8(tail_mask, (sz_u8_t const *)(text + 48)));

        sz_hash_minimal_update_neon_(&state, data0_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data1_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data2_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data3_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else {
        // For large hashes (>64 bytes), use non-temporal loads
        sz_align_(64) sz_hash_state_t state;
        sz_hash_state_init_neon(&state, seed);

        svbool_t ptrue = svptrue_b8();
        sz_u8_t const *data_ptr = (sz_u8_t const *)text;

        for (; state.ins_length + 64 <= length; state.ins_length += 64) {
            // Non-temporal loads - 4x 16-byte loads
            svuint8_t sve_vec0 = svldnt1_u8(ptrue, data_ptr + state.ins_length + 0);
            svuint8_t sve_vec1 = svldnt1_u8(ptrue, data_ptr + state.ins_length + 16);
            svuint8_t sve_vec2 = svldnt1_u8(ptrue, data_ptr + state.ins_length + 32);
            svuint8_t sve_vec3 = svldnt1_u8(ptrue, data_ptr + state.ins_length + 48);

            // Move SVE registers to NEON for AES processing
            state.ins.u8x16s[0] = svget_neonq_u8(sve_vec0);
            state.ins.u8x16s[1] = svget_neonq_u8(sve_vec1);
            state.ins.u8x16s[2] = svget_neonq_u8(sve_vec2);
            state.ins.u8x16s[3] = svget_neonq_u8(sve_vec3);

            sz_hash_state_update_neon_(&state);
        }

        // Handle the tail, resetting the registers to zero first
        if (state.ins_length < length) {
            state.ins.u8x16s[0] = vdupq_n_u8(0);
            state.ins.u8x16s[1] = vdupq_n_u8(0);
            state.ins.u8x16s[2] = vdupq_n_u8(0);
            state.ins.u8x16s[3] = vdupq_n_u8(0);
            for (sz_size_t i = 0; state.ins_length < length; ++i, ++state.ins_length)
                state.ins.u8s[i] = text[state.ins_length];
            sz_hash_state_update_neon_(&state);
            state.ins_length = length;
        }

        return sz_hash_state_finalize_neon_(&state);
    }
}

// ========================================================================
// 256-bit SVE Implementation (store-based approach)
// ========================================================================

/**
 *  @brief  Hash implementation assuming 256-bit SVE registers
 *  Uses SVE loads (32 bytes) with stores to temporary buffers for NEON processing
 */
SZ_PUBLIC sz_u64_t sz_hash_sve2_b256(sz_cptr_t text, sz_size_t length, sz_u64_t seed) {
    if (length <= 16) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);

        sz_u8_t const *data_ptr = (sz_u8_t const *)text;
        svbool_t mask = svwhilelt_b8((sz_u64_t)0, (sz_u64_t)length);

        // Load up to 16 bytes with predicated SVE load
        svuint8_t sve_vec = svld1_u8(mask, data_ptr);

        // Store into zero-initialized aligned buffer
        sz_align_(32) sz_u8_t buffer[16] = {0};
        svst1_u8(mask, buffer, sve_vec);

        sz_u128_vec_t data0_vec;
        data0_vec.u8x16 = vld1q_u8(buffer);

        sz_hash_minimal_update_neon_(&state, data0_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else if (length <= 32) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);

        sz_u8_t const *data_ptr = (sz_u8_t const *)text;
        svbool_t ptrue = svptrue_b8();

        // Load up to 32 bytes with one 256-bit SVE vector
        svuint8_t sve_vec = svld1_u8(ptrue, data_ptr);

        // Store 32 bytes into temporary aligned buffer (one SVE store = 32 bytes)
        sz_align_(32) sz_u8_t buffer[32];
        svst1_u8(ptrue, buffer, sve_vec);

        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.u8x16 = vld1q_u8(buffer);
        data1_vec.u8x16 = vld1q_u8(buffer + 16);

        sz_hash_minimal_update_neon_(&state, data0_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data1_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else if (length <= 48) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);

        sz_u8_t const *data_ptr = (sz_u8_t const *)text;
        svbool_t ptrue = svptrue_b8();

        // Load first 32 bytes with one 256-bit SVE vector
        svuint8_t sve_vec0 = svld1_u8(ptrue, data_ptr + 0);
        // Load remaining 16 bytes with predicated load
        svbool_t tail_mask = svwhilelt_b8((sz_u64_t)0, (sz_u64_t)(length - 32));
        svuint8_t sve_vec1 = svld1_u8(tail_mask, data_ptr + 32);

        // Store into temporary aligned buffer
        sz_align_(32) sz_u8_t buffer[48];
        svst1_u8(ptrue, buffer, sve_vec0);          // Stores 32 bytes
        svst1_u8(tail_mask, buffer + 32, sve_vec1); // Stores up to 16 bytes

        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.u8x16 = vld1q_u8(buffer);
        data1_vec.u8x16 = vld1q_u8(buffer + 16);
        data2_vec.u8x16 = vld1q_u8(buffer + 32);

        sz_hash_minimal_update_neon_(&state, data0_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data1_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data2_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else if (length <= 64) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);

        sz_u8_t const *data_ptr = (sz_u8_t const *)text;
        svbool_t ptrue = svptrue_b8();

        // Load 64 bytes with two 256-bit SVE vectors
        svuint8_t sve_vec0 = svld1_u8(ptrue, data_ptr + 0);
        svbool_t tail_mask = svwhilelt_b8((sz_u64_t)0, (sz_u64_t)(length - 32));
        svuint8_t sve_vec1 = svld1_u8(tail_mask, data_ptr + 32);

        // Store into temporary aligned buffer
        sz_align_(32) sz_u8_t buffer[64];
        svst1_u8(ptrue, buffer, sve_vec0);          // Stores 32 bytes
        svst1_u8(tail_mask, buffer + 32, sve_vec1); // Stores up to 32 bytes

        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.u8x16 = vld1q_u8(buffer);
        data1_vec.u8x16 = vld1q_u8(buffer + 16);
        data2_vec.u8x16 = vld1q_u8(buffer + 32);
        data3_vec.u8x16 = vld1q_u8(buffer + 48);

        sz_hash_minimal_update_neon_(&state, data0_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data1_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data2_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data3_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else {
        // For large hashes (>64 bytes), assume 256-bit SVE registers
        // Load 32 bytes per SVE vector, store directly into state
        sz_align_(64) sz_hash_state_t state;
        sz_hash_state_init_neon(&state, seed);

        sz_u8_t const *data_ptr = (sz_u8_t const *)text;
        svbool_t ptrue = svptrue_b8();

        for (; state.ins_length + 64 <= length; state.ins_length += 64) {
            // Load 64 bytes using two 256-bit SVE loads (32 bytes each)
            svuint8_t sve_vec0 = svld1_u8(ptrue, data_ptr + state.ins_length + 0);
            svuint8_t sve_vec1 = svld1_u8(ptrue, data_ptr + state.ins_length + 32);

            // Store into state: each 256-bit SVE store writes 32 bytes
            svst1_u8(ptrue, &state.ins.u8s[0], sve_vec0);  // Bytes 0-31
            svst1_u8(ptrue, &state.ins.u8s[32], sve_vec1); // Bytes 32-63

            sz_hash_state_update_neon_(&state);
        }

        // Handle the tail, resetting the registers to zero first
        if (state.ins_length < length) {
            state.ins.u8x16s[0] = vdupq_n_u8(0);
            state.ins.u8x16s[1] = vdupq_n_u8(0);
            state.ins.u8x16s[2] = vdupq_n_u8(0);
            state.ins.u8x16s[3] = vdupq_n_u8(0);
            for (sz_size_t i = 0; state.ins_length < length; ++i, ++state.ins_length)
                state.ins.u8s[i] = text[state.ins_length];
            sz_hash_state_update_neon_(&state);
            state.ins_length = length;
        }

        return sz_hash_state_finalize_neon_(&state);
    }
}

// ========================================================================
// Performance Notes
// ========================================================================

/*
 * Benchmarks on Graviton3 (256-bit SVE, Neoverse V1):
 *
 * NEON:              21.03 GiB/s @ 220.93 ns/call
 * SVE2 (128-bit):     9.71 GiB/s @ 478.42 ns/call (with non-temporal loads)
 * SVE2 (256-bit):    Similar or slower than NEON
 *
 * Root causes of poor SVE performance:
 * 1. Non-temporal loads (svldnt1_u8) were 2x+ slower than regular loads
 * 2. Regular SVE loads (svld1_u8) provide no benefit over NEON vld1q_u8
 * 3. Extra store/load round-trips through buffers add overhead
 * 4. NEON has direct AES instructions (vaeseq_u8) - no SVE equivalent
 * 5. Data must be in NEON registers anyway for AES operations
 *
 * Conclusion: For AES-based hashing, NEON is optimal. SVE provides no benefit
 * because the crypto operations are NEON-only and memory bandwidth is not
 * the bottleneck.
 */

/**
 *  @brief  A helper function for computing 16x packed string hashes for strings up to 16 bytes long.
 *          The number 16 is derived from 2048 bits (256 bytes) being the maximum size of the SVE register
 *          and the AES block size being 128 bits (16 bytes). So in the largest SVE register, we can fit
 *          16 such individual AES blocks.
 *          It's relevant for set intersection operations and is faster than hashing each string individually.
 */
SZ_PUBLIC void sz_hash_sve2_upto16x16_(char texts[16][16], sz_size_t length[16], sz_u64_t seed, sz_u64_t hashes[16]) {
    svuint8_t state_aes, state_sum, state_key;

    // To load and store the seed, we don't even need a `svwhilelt_b64(0, 2)`.
    state_key = svreinterpret_u8_u64(svdup_n_u64(seed));

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    svuint64_t pi0 = svdupq_n_u64(pi[0], pi[1]);
    svuint64_t pi1 = svdupq_n_u64(pi[8], pi[9]);
    state_aes = sveor_u8_x(svptrue_b8(), state_key, svreinterpret_u8_u64(pi0));
    state_sum = sveor_u8_x(svptrue_b8(), state_key, svreinterpret_u8_u64(pi1));

    // We will only use the first 128 bits of the shuffle mask
    sz_u8_t const *order = sz_hash_u8x16x4_shuffle_();
    svuint8_t const order = svreinterpret_u8_u64(svdupq_n_u64( //
        *(sz_u64_t const *)(order + 0),                        //
        *(sz_u64_t const *)(order + 8)));
    svuint8_t const sum_shuffled = svtbl_u8(state_sum, order);

    // Loop throughthe input until we process all the bytes
    sz_size_t const bytes_per_register = svcntb();
    sz_size_t const texts_per_register = bytes_per_register / 16;
    for (sz_size_t progress_bytes = 0; progress_bytes < 256; progress_bytes += bytes_per_register) {
        svuint8_t blocks = svld1_u8(svwhilelt_b8((sz_u64_t)progress_bytes, (sz_u64_t)256),
                                    (sz_u8_t const *)(&texts[0][0] + progress_bytes));

        // One round of hashing logic for multiple blocks
        svuint8_t blocks_aes = sz_emulate_aesenc_u8x16_sve2_(state_aes, blocks);
        svuint8_t blocks_sum = svreinterpret_u8_u64(
            svadd_u64_x(svptrue_b64(), svreinterpret_u64_u8(sum_shuffled), svreinterpret_u64_u8(blocks)));

        // Now mix, folding the length into the key
        svuint64_t key_with_lengths =
            svadd_u64_x(svptrue_b64(), svreinterpret_u64_u8(state_key), svdupq_n_u64(length, 0));

        // Combine the "sum" and the "AES" blocks
        svuint8_t mixed = sz_emulate_aesenc_u8x16_sve2_(blocks_sum, blocks_aes);

        // Make sure the "key" mixes enough with the state,
        // as with less than 2 rounds - SMHasher fails
        svuint8_t mixed_in_register = sz_emulate_aesenc_u8x16_sve2_(
            sz_emulate_aesenc_u8x16_sve2_(mixed, svreinterpret_u8_u64(key_with_lengths)), mixed);

        // Extract the low 64 bits from each lane
        svuint64_t mixed_in_register_u64 = svreinterpret_u64_u8(mixed_in_register);
    }
}
