/**
 *  @brief IBM Power VSX backend for hash.
 *  @file include/stringzilla/hash/powervsx.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_POWERVSX_H_
#define STRINGZILLA_HASH_POWERVSX_H_

#include "stringzilla/types.h"
#include "stringzilla/hash/serial.h"
#include "stringzilla/compare.h" // `sz_equal`

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

SZ_PUBLIC sz_u64_t sz_bytesum_powervsx(sz_cptr_t text, sz_size_t length) {
    sz_u64_t sum = 0;
    __vector unsigned char const ones_vec = vec_splats((unsigned char)1);

    // `vec_msum(v, ones, acc)` is the in-lane accumulator: each step adds at most 4*255 = 1020 to any
    // 32-bit lane. A lane can therefore absorb `UINT_MAX / 1020 ≈ 4.21M` iterations before it could
    // overflow. We process the input in blocks of that many 16-byte windows so the only horizontal
    // reduction (`acc[0]+acc[1]+acc[2]+acc[3]`) happens once per block — out of the hot inner loop —
    // and the inner loop carries no per-iteration counter or branch. Results stay byte-for-byte
    // identical to `sz_bytesum_serial` on multi-gigabyte inputs.
    sz_size_t const block_windows = (sz_size_t)4000000; // 4M * 1020 < UINT_MAX.
    while (length >= 16) {
        sz_size_t windows = length / 16;
        if (windows > block_windows) windows = block_windows;
        __vector unsigned int accumulator_vec = vec_splats((unsigned int)0);
        for (sz_size_t window_index = 0; window_index < windows; ++window_index, text += 16) {
            __vector unsigned char bytes_vec = vec_xl(0, (unsigned char const *)text);
            accumulator_vec = vec_msum(bytes_vec, ones_vec,
                                       accumulator_vec); // Lane-wise sum of each group of 4 unsigned bytes.
        }
        sum += (sz_u64_t)accumulator_vec[0] + accumulator_vec[1] + accumulator_vec[2] + accumulator_vec[3];
        length -= windows * 16;
    }

    // Same as the scalar version for the misaligned tail.
    while (length--) sum += *(sz_u8_t const *)text++;
    return sum;
}

#pragma region AES-based hashing

/*
 *  StringZilla guarantees that every backend produces @b bit-identical hashes for a given input and
 *  seed. The reference is `sz_hash_serial`, built around two primitives:
 *
 *      1. `sz_emulate_aesenc_si128_serial_(state, key)` = `MixColumns(SubBytes(ShiftRows(state))) ^ key`,
 *         i.e. one full x86-style AES encryption round.
 *      2. `sz_emulate_shuffle_epi8_serial_(state, order)` = `_mm_shuffle_epi8`-style byte permutation.
 *
 *  Power8+ exposes a hardware AES round via `__builtin_crypto_vcipher`. The instruction follows the
 *  big-endian AES state mapping, which is the byte-reverse of x86's `_mm_aesenc_si128`. Empirically
 *  (validated under QEMU against the serial reference over millions of random vectors) the exact
 *  x86-compatible round is reproduced on little-endian Power by:
 *
 *      reverse the 16 state bytes -> `vcipher` with an all-zero round key -> reverse back -> XOR key.
 *
 *  `vcipher(s, 0)` performs `SubBytes(ShiftRows(MixColumns(...)))` over the BE-mapped state; framing
 *  it between two byte reversals realigns the rows/columns to x86's layout, after which we apply the
 *  round key XOR ourselves (so the key needs no reordering). The shuffle primitive maps directly onto
 *  `vec_perm(state, state, order)` because little-endian VSX byte indexing matches the in-memory
 *  byte order the serial code reads. On big-endian Power both byte-order assumptions break, so we
 *  delegate the whole hash family to the serial reference there.
 */

#if !SZ_IS_BIG_ENDIAN_

/** @brief Byte-reverse permutation selector for a 16-byte VSX register. */
SZ_INTERNAL __vector unsigned char sz_aes_byte_reverse_mask_powervsx_(void) {
    __vector unsigned char const mask = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    return mask;
}

/**
 *  @brief Bit-exact VSX equivalent of `sz_emulate_aesenc_si128_serial_` using hardware AES.
 *  @return `MixColumns(SubBytes(ShiftRows(state))) ^ round_key`, identical to the serial reference.
 */
SZ_INTERNAL sz_u128_vec_t sz_aesenc_powervsx_(sz_u128_vec_t state_vec, sz_u128_vec_t round_key_vec) {
    __vector unsigned char const rev = sz_aes_byte_reverse_mask_powervsx_();
    __vector unsigned char state_u8 = state_vec.vsx_u8;
    __vector unsigned char reversed = vec_perm(state_u8, state_u8, rev);
    __vector unsigned char zero = vec_splats((unsigned char)0);
    __vector unsigned char ciphered = (__vector unsigned char)__builtin_crypto_vcipher(
        (__vector unsigned long long)reversed, (__vector unsigned long long)zero);
    __vector unsigned char restored = vec_perm(ciphered, ciphered, rev);
    sz_u128_vec_t result;
    result.vsx_u8 = restored;
    result.u64s[0] ^= round_key_vec.u64s[0];
    result.u64s[1] ^= round_key_vec.u64s[1];
    return result;
}

/** @brief Bit-exact VSX equivalent of `sz_emulate_shuffle_epi8_serial_` via `vec_perm`. */
SZ_INTERNAL sz_u128_vec_t sz_shuffle_epi8_powervsx_(sz_u128_vec_t state_vec, sz_u8_t const order[sz_at_least_(16)]) {
    __vector unsigned char order_u8 = vec_xl(0, (unsigned char const *)order);
    sz_u128_vec_t result;
    result.vsx_u8 = vec_perm(state_vec.vsx_u8, state_vec.vsx_u8, order_u8);
    return result;
}

#pragma region Minimal state for short inputs

SZ_INTERNAL void sz_hash_minimal_update_powervsx_(sz_hash_minimal_t_ *state, sz_u128_vec_t block) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    state->aes = sz_aesenc_powervsx_(state->aes, block);
    state->sum = sz_shuffle_epi8_powervsx_(state->sum, shuffle);
    state->sum.u64s[0] += block.u64s[0], state->sum.u64s[1] += block.u64s[1];
}

SZ_INTERNAL sz_u64_t sz_hash_minimal_finalize_powervsx_(sz_hash_minimal_t_ const *state, sz_size_t length) {
    sz_u128_vec_t key_with_length = state->key;
    key_with_length.u64s[0] += length;
    sz_u128_vec_t mixed = sz_aesenc_powervsx_(state->sum, state->aes);
    sz_u128_vec_t mixed_in_register = sz_aesenc_powervsx_(sz_aesenc_powervsx_(mixed, key_with_length), mixed);
    return mixed_in_register.u64s[0];
}

#pragma endregion

#pragma region Full state for long inputs

SZ_PUBLIC void sz_hash_state_init_powervsx(sz_hash_state_t *state, sz_u64_t seed) {
    sz_u64_t *key_u64s = (sz_u64_t *)state->key;
    key_u64s[0] = seed;
    key_u64s[1] = seed;

    sz_u64_t const *pi_constants = sz_hash_pi_constants_();
    sz_u64_t *aes_u64s = (sz_u64_t *)state->aes;
    sz_u64_t *sum_u64s = (sz_u64_t *)state->sum;
    for (sz_size_t lane_index = 0; lane_index < 8; ++lane_index) aes_u64s[lane_index] = seed ^ pi_constants[lane_index];
    for (sz_size_t lane_index = 0; lane_index < 8; ++lane_index)
        sum_u64s[lane_index] = seed ^ pi_constants[lane_index + 8];

    for (sz_size_t byte_index = 0; byte_index < 64; ++byte_index) state->ins[byte_index] = 0;
    state->ins_length = 0;
}

SZ_INTERNAL void sz_hash_state_update_powervsx_(sz_hash_state_t *state) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    sz_u128_vec_t *aes_vecs = (sz_u128_vec_t *)state->aes;
    sz_u128_vec_t *sum_vecs = (sz_u128_vec_t *)state->sum;
    sz_u128_vec_t *ins_vecs = (sz_u128_vec_t *)state->ins;
    for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) {
        aes_vecs[lane_index] = sz_aesenc_powervsx_(aes_vecs[lane_index], ins_vecs[lane_index]);
        sum_vecs[lane_index] = sz_shuffle_epi8_powervsx_(sum_vecs[lane_index], shuffle);
        sum_vecs[lane_index].u64s[0] += ins_vecs[lane_index].u64s[0],
            sum_vecs[lane_index].u64s[1] += ins_vecs[lane_index].u64s[1];
    }
}

SZ_INTERNAL sz_u64_t sz_hash_state_finalize_powervsx_(sz_hash_state_t const *state) {
    sz_u64_t const *key_u64s = (sz_u64_t const *)state->key;
    sz_u128_vec_t key_with_length;
    key_with_length.u64s[0] = key_u64s[0] + state->ins_length;
    key_with_length.u64s[1] = key_u64s[1];

    sz_u128_vec_t const *aes_vecs = (sz_u128_vec_t const *)state->aes;
    sz_u128_vec_t const *sum_vecs = (sz_u128_vec_t const *)state->sum;

    sz_u128_vec_t mixed0 = sz_aesenc_powervsx_(sum_vecs[0], aes_vecs[0]);
    sz_u128_vec_t mixed1 = sz_aesenc_powervsx_(sum_vecs[1], aes_vecs[1]);
    sz_u128_vec_t mixed2 = sz_aesenc_powervsx_(sum_vecs[2], aes_vecs[2]);
    sz_u128_vec_t mixed3 = sz_aesenc_powervsx_(sum_vecs[3], aes_vecs[3]);

    sz_u128_vec_t mixed01 = sz_aesenc_powervsx_(mixed0, mixed1);
    sz_u128_vec_t mixed23 = sz_aesenc_powervsx_(mixed2, mixed3);
    sz_u128_vec_t mixed = sz_aesenc_powervsx_(mixed01, mixed23);

    sz_u128_vec_t mixed_in_register = sz_aesenc_powervsx_(sz_aesenc_powervsx_(mixed, key_with_length), mixed);
    return mixed_in_register.u64s[0];
}

#pragma endregion

SZ_PUBLIC SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_powervsx(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {
    if (length <= 16) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        sz_u128_vec_t data_vec;
        data_vec.u64s[0] = data_vec.u64s[1] = 0;
        for (sz_size_t byte_index = 0; byte_index < length; ++byte_index) data_vec.u8s[byte_index] = start[byte_index];
        sz_hash_minimal_update_powervsx_(&state, data_vec);
        return sz_hash_minimal_finalize_powervsx_(&state, length);
    }
    else if (length <= 32) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.u64s[0] = data0_vec.u64s[1] = 0;
        data1_vec.u64s[0] = data1_vec.u64s[1] = 0;
        for (sz_size_t byte_index = 0; byte_index < 16; ++byte_index) data0_vec.u8s[byte_index] = start[byte_index];
        for (sz_size_t byte_index = 0; byte_index < 16; ++byte_index)
            data1_vec.u8s[byte_index] = start[length - 16 + byte_index];
        sz_hash_shift_in_register_serial_(&data1_vec, (int)(32 - length));
        sz_hash_minimal_update_powervsx_(&state, data0_vec);
        sz_hash_minimal_update_powervsx_(&state, data1_vec);
        return sz_hash_minimal_finalize_powervsx_(&state, length);
    }
    else if (length <= 48) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.u64s[0] = data0_vec.u64s[1] = 0;
        data1_vec.u64s[0] = data1_vec.u64s[1] = 0;
        data2_vec.u64s[0] = data2_vec.u64s[1] = 0;
        for (sz_size_t byte_index = 0; byte_index < 16; ++byte_index) data0_vec.u8s[byte_index] = start[byte_index];
        for (sz_size_t byte_index = 0; byte_index < 16; ++byte_index)
            data1_vec.u8s[byte_index] = start[16 + byte_index];
        for (sz_size_t byte_index = 0; byte_index < 16; ++byte_index)
            data2_vec.u8s[byte_index] = start[length - 16 + byte_index];
        sz_hash_shift_in_register_serial_(&data2_vec, (int)(48 - length));
        sz_hash_minimal_update_powervsx_(&state, data0_vec);
        sz_hash_minimal_update_powervsx_(&state, data1_vec);
        sz_hash_minimal_update_powervsx_(&state, data2_vec);
        return sz_hash_minimal_finalize_powervsx_(&state, length);
    }
    else if (length <= 64) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.u64s[0] = data0_vec.u64s[1] = 0;
        data1_vec.u64s[0] = data1_vec.u64s[1] = 0;
        data2_vec.u64s[0] = data2_vec.u64s[1] = 0;
        data3_vec.u64s[0] = data3_vec.u64s[1] = 0;
        for (sz_size_t byte_index = 0; byte_index < 16; ++byte_index) data0_vec.u8s[byte_index] = start[byte_index];
        for (sz_size_t byte_index = 0; byte_index < 16; ++byte_index)
            data1_vec.u8s[byte_index] = start[16 + byte_index];
        for (sz_size_t byte_index = 0; byte_index < 16; ++byte_index)
            data2_vec.u8s[byte_index] = start[32 + byte_index];
        for (sz_size_t byte_index = 0; byte_index < 16; ++byte_index)
            data3_vec.u8s[byte_index] = start[length - 16 + byte_index];
        sz_hash_shift_in_register_serial_(&data3_vec, (int)(64 - length));
        sz_hash_minimal_update_powervsx_(&state, data0_vec);
        sz_hash_minimal_update_powervsx_(&state, data1_vec);
        sz_hash_minimal_update_powervsx_(&state, data2_vec);
        sz_hash_minimal_update_powervsx_(&state, data3_vec);
        return sz_hash_minimal_finalize_powervsx_(&state, length);
    }
    else {
        sz_align_(64) sz_hash_state_t state;
        sz_hash_state_init_powervsx(&state, seed);

        for (; state.ins_length + 64 <= length; state.ins_length += 64) {
            for (sz_size_t byte_index = 0; byte_index < 64; ++byte_index)
                state.ins[byte_index] = start[state.ins_length + byte_index];
            sz_hash_state_update_powervsx_(&state);
        }

        if (state.ins_length < length) {
            for (sz_size_t byte_index = 0; byte_index != 64; ++byte_index) state.ins[byte_index] = 0;
            for (sz_size_t byte_index = 0; state.ins_length < length; ++byte_index, ++state.ins_length)
                state.ins[byte_index] = start[state.ins_length];
            sz_hash_state_update_powervsx_(&state);
            state.ins_length = length;
        }
        return sz_hash_state_finalize_powervsx_(&state);
    }
}

SZ_PUBLIC void sz_hash_state_update_powervsx(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    while (length) {
        sz_size_t progress_in_block = state->ins_length % 64;
        sz_size_t to_copy = sz_min_of_two(length, 64 - progress_in_block);
        int const will_fill_block = progress_in_block + to_copy == 64;
        state->ins_length += to_copy;
        length -= to_copy;
        while (to_copy--) state->ins[progress_in_block++] = *text++;
        if (will_fill_block) {
            sz_hash_state_update_powervsx_(state);
            for (sz_size_t byte_index = 0; byte_index < 64; ++byte_index) state->ins[byte_index] = 0;
        }
    }
}

SZ_PUBLIC sz_u64_t sz_hash_state_digest_powervsx(sz_hash_state_t const *state) {
    sz_size_t length = state->ins_length;
    if (length >= 64) return sz_hash_state_finalize_powervsx_(state);

    sz_u64_t const *key_u64s = (sz_u64_t const *)state->key;
    sz_hash_minimal_t_ minimal_state;
    minimal_state.key.u64s[0] = key_u64s[0];
    minimal_state.key.u64s[1] = key_u64s[1];
    minimal_state.aes = *(sz_u128_vec_t const *)state->aes;
    minimal_state.sum = *(sz_u128_vec_t const *)state->sum;

    sz_u128_vec_t const *ins_vecs = (sz_u128_vec_t const *)state->ins;
    if (length <= 16) {
        sz_hash_minimal_update_powervsx_(&minimal_state, ins_vecs[0]);
        return sz_hash_minimal_finalize_powervsx_(&minimal_state, length);
    }
    else if (length <= 32) {
        sz_hash_minimal_update_powervsx_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_powervsx_(&minimal_state, ins_vecs[1]);
        return sz_hash_minimal_finalize_powervsx_(&minimal_state, length);
    }
    else if (length <= 48) {
        sz_hash_minimal_update_powervsx_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_powervsx_(&minimal_state, ins_vecs[1]);
        sz_hash_minimal_update_powervsx_(&minimal_state, ins_vecs[2]);
        return sz_hash_minimal_finalize_powervsx_(&minimal_state, length);
    }
    else {
        sz_hash_minimal_update_powervsx_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_powervsx_(&minimal_state, ins_vecs[1]);
        sz_hash_minimal_update_powervsx_(&minimal_state, ins_vecs[2]);
        sz_hash_minimal_update_powervsx_(&minimal_state, ins_vecs[3]);
        return sz_hash_minimal_finalize_powervsx_(&minimal_state, length);
    }
}

SZ_PUBLIC void sz_fill_random_powervsx(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_u64_t const *pi_constants = sz_hash_pi_constants_();
    sz_u128_vec_t input_vec, pi_vec, key_vec, generated_vec;
    for (sz_size_t lane_index = 0; length; ++lane_index) {
        input_vec.u64s[0] = input_vec.u64s[1] = nonce + lane_index;
        pi_vec = ((sz_u128_vec_t const *)pi_constants)[lane_index % 4];
        key_vec.u64s[0] = nonce ^ pi_vec.u64s[0];
        key_vec.u64s[1] = nonce ^ pi_vec.u64s[1];
        generated_vec = sz_aesenc_powervsx_(input_vec, key_vec);
        for (sz_size_t byte_index = 0; byte_index < 16 && length; ++byte_index, --length)
            *text++ = generated_vec.u8s[byte_index];
    }
}

#else // SZ_IS_BIG_ENDIAN_

// On big-endian Power the hardware AES byte mapping and the `vec_perm` shuffle indexing both diverge
// from the x86 layout the serial reference encodes, so we delegate to keep hashes bit-exact.

/** @brief Big-endian Power stub: delegates to `sz_hash_serial` to preserve bit-exact digests. */
SZ_PUBLIC sz_u64_t sz_hash_powervsx(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {
    return sz_hash_serial(start, length, seed);
}
/** @brief Big-endian Power stub: delegates to `sz_hash_state_init_serial`. */
SZ_PUBLIC void sz_hash_state_init_powervsx(sz_hash_state_t *state, sz_u64_t seed) {
    sz_hash_state_init_serial(state, seed);
}
/** @brief Big-endian Power stub: delegates to `sz_hash_state_update_serial`. */
SZ_PUBLIC void sz_hash_state_update_powervsx(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_hash_state_update_serial(state, text, length);
}
/** @brief Big-endian Power stub: delegates to `sz_hash_state_digest_serial`. */
SZ_PUBLIC sz_u64_t sz_hash_state_digest_powervsx(sz_hash_state_t const *state) {
    return sz_hash_state_digest_serial(state);
}
/** @brief Big-endian Power stub: delegates to `sz_fill_random_serial`. */
SZ_PUBLIC void sz_fill_random_powervsx(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_fill_random_serial(text, length, nonce);
}

#endif // SZ_IS_BIG_ENDIAN_

#pragma endregion // AES-based hashing

#pragma region SHA256

// No VSX SHA extension is targeted here, so the SHA-256 family delegates to the serial reference.

SZ_PUBLIC void sz_sha256_state_init_powervsx(sz_sha256_state_t *state) { sz_sha256_state_init_serial(state); }

SZ_PUBLIC void sz_sha256_state_update_powervsx(sz_sha256_state_t *state, sz_cptr_t data, sz_size_t length) {
    sz_sha256_state_update_serial(state, data, length);
}

SZ_PUBLIC void sz_sha256_state_digest_powervsx(sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(32)]) {
    sz_sha256_state_digest_serial(state, digest);
}

#pragma endregion // SHA256

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_POWERVSX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_POWERVSX_H_
