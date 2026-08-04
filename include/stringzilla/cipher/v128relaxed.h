/**
 *  @brief WebAssembly relaxed-SIMD backend for AES-256 in counter and Galois/counter modes.
 *  @file include/stringzilla/cipher/v128relaxed.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/cipher.h
 *
 *  The round loop stays a loop, for the reason `cipher/v128.h` spells out.
 */
#ifndef STRINGZILLA_CIPHER_V128RELAXED_H_
#define STRINGZILLA_CIPHER_V128RELAXED_H_

#include "stringzilla/types.h"
#include "stringzilla/cipher/serial.h"
#include "stringzilla/cipher/v128.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128RELAXED
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/*  Relaxed SIMD adds exactly one thing this workload can use: a swizzle that does not have to answer
 *  zero for an out-of-range index. Baseline `i8x16.swizzle` must, and on hosts whose native permute
 *  zeroes only for a different range an engine has to bracket every swizzle with a saturating clamp.
 *  The substitution box runs nineteen swizzles per block, thirteen of which carry indices this file
 *  can prove lie in zero through fifteen: two into the tower-field basis, six logarithm lookups,
 *  three four-bit tables, and two into the fused output table. Those thirteen become relaxed here.
 *
 *  The other six are the antilogarithm lookups, which are fed sums running past twenty-eight and a
 *  deliberately negative index, and whose whole correctness argument is that an out-of-range index
 *  answers zero. Those stay baseline. Relaxing them would make the substitution engine dependent,
 *  and the kernels below have to be bit-identical to the serial reference on every runtime.
 *
 *  Everything with no relaxed opportunity calls the `_v128` kernel rather than restating it: the
 *  Galois hash uses no swizzle at all, the row shift and column mixing are immediate shuffles, and
 *  the associated-data and digest paths never reach the block cipher.
 */

#pragma region Substitution Box

/**
 *  @brief Evaluates a map linear over `GF(2)` on all sixteen lanes at once.
 *  @param low_table_u8x16 The map's value on each low nibble.
 *  @param high_table_u8x16 The map's value on each high nibble.
 *  @param bytes_u8x16 The sixteen inputs.
 *  @return `low_table_u8x16[byte & 0xF] ^ high_table_u8x16[byte >> 4]` in every lane.
 *
 *  Both index vectors are nibbles by construction, so neither swizzle can go out of range and the
 *  relaxed form's implementation-defined case is unreachable.
 */
SZ_HELPER_INLINE v128_t sz_aes256_nibble_map_v128relaxed_(v128_t low_table_u8x16, v128_t high_table_u8x16,
                                                          v128_t bytes_u8x16) {
    v128_t const low_nibbles_u8x16 = wasm_v128_and(bytes_u8x16, wasm_i8x16_splat((sz_i8_t)0x0F));
    v128_t const high_nibbles_u8x16 = wasm_u8x16_shr(bytes_u8x16, 4);
    return wasm_v128_xor(wasm_i8x16_relaxed_swizzle(low_table_u8x16, low_nibbles_u8x16),
                         wasm_i8x16_relaxed_swizzle(high_table_u8x16, high_nibbles_u8x16));
}

/**
 *  @brief Multiplies sixteen pairs of `GF(2^4)` elements under `x^4 + x + 1`.
 *  @param first_u8x16 One operand per lane, each below sixteen.
 *  @param second_u8x16 The other operand per lane, each below sixteen.
 *  @return The product per lane.
 *
 *  The two logarithm lookups take nibbles and may be relaxed. The two antilogarithm lookups may not:
 *  they are handed a sum that runs to twenty-eight and that same sum less sixteen, and they rely on
 *  a baseline swizzle answering zero outside its table, which is also how a zero operand is handled.
 */
SZ_HELPER_AUTO v128_t sz_aes256_nibble_multiply_v128relaxed_(v128_t first_u8x16, v128_t second_u8x16) {
    v128_t const logarithm_table_u8x16 = wasm_v128_load(sz_aes256_nibble_logarithm_v128_());
    v128_t const exponent_low_table_u8x16 = wasm_v128_load(sz_aes256_nibble_exponent_low_v128_());
    v128_t const exponent_high_table_u8x16 = wasm_v128_load(sz_aes256_nibble_exponent_high_v128_());
    v128_t const logarithm_sum_u8x16 = wasm_i8x16_add(wasm_i8x16_relaxed_swizzle(logarithm_table_u8x16, first_u8x16),
                                                      wasm_i8x16_relaxed_swizzle(logarithm_table_u8x16, second_u8x16));
    return wasm_v128_xor(
        wasm_i8x16_swizzle(exponent_low_table_u8x16, logarithm_sum_u8x16),
        wasm_i8x16_swizzle(exponent_high_table_u8x16, wasm_i8x16_sub(logarithm_sum_u8x16, wasm_i8x16_splat(16))));
}

/**
 *  @brief Applies the substitution box of FIPS 197 to all sixteen bytes of a block.
 *  @param bytes_u8x16 The sixteen inputs.
 *  @return The substituted bytes.
 *
 *  The same tower-field construction the `_v128` kernel uses, with every provably in-range swizzle
 *  taken in its relaxed form. The norm, its inverse and both halves of the inverse are `GF(2^4)`
 *  elements, so each of them is a legal index by construction.
 */
SZ_HELPER_AUTO v128_t sz_aes256_substitute_v128relaxed_(v128_t bytes_u8x16) {
    v128_t const mapped_u8x16 = sz_aes256_nibble_map_v128relaxed_(wasm_v128_load(sz_aes256_tower_forward_low_v128_()),
                                                                  wasm_v128_load(sz_aes256_tower_forward_high_v128_()),
                                                                  bytes_u8x16);
    v128_t const high_nibbles_u8x16 = wasm_u8x16_shr(mapped_u8x16, 4);
    v128_t const low_nibbles_u8x16 = wasm_v128_and(mapped_u8x16, wasm_i8x16_splat((sz_i8_t)0x0F));

    // The tower field's norm, `high^2 * N ^ high * low ^ low^2`, is what has to be inverted in `GF(2^4)`.
    v128_t const high_scaled_u8x16 = wasm_i8x16_relaxed_swizzle(wasm_v128_load(sz_aes256_nibble_square_scaled_v128_()),
                                                                high_nibbles_u8x16);
    v128_t const crossed_u8x16 = sz_aes256_nibble_multiply_v128relaxed_(high_nibbles_u8x16, low_nibbles_u8x16);
    v128_t const low_squared_u8x16 = wasm_i8x16_relaxed_swizzle(wasm_v128_load(sz_aes256_nibble_square_v128_()),
                                                                low_nibbles_u8x16);
    v128_t const tower_norm_u8x16 = wasm_v128_xor(wasm_v128_xor(high_scaled_u8x16, crossed_u8x16), low_squared_u8x16);
    v128_t const tower_norm_inverse_u8x16 = wasm_i8x16_relaxed_swizzle(wasm_v128_load(sz_aes256_nibble_inverse_v128_()),
                                                                       tower_norm_u8x16);

    v128_t const high_inverse_u8x16 = sz_aes256_nibble_multiply_v128relaxed_(high_nibbles_u8x16,
                                                                             tower_norm_inverse_u8x16);
    v128_t const low_inverse_u8x16 = sz_aes256_nibble_multiply_v128relaxed_(
        wasm_v128_xor(high_nibbles_u8x16, low_nibbles_u8x16), tower_norm_inverse_u8x16);
    return wasm_v128_xor(
        wasm_i8x16_relaxed_swizzle(wasm_v128_load(sz_aes256_substituted_low_v128_()), low_inverse_u8x16),
        wasm_i8x16_relaxed_swizzle(wasm_v128_load(sz_aes256_substituted_high_v128_()), high_inverse_u8x16));
}

#pragma endregion // Substitution Box

#pragma region Key Schedule

/**
 *  @brief Builds the schedule's rotated and substituted word against a round constant.
 *  @param previous_u8x16 The four schedule words immediately before the new quadruple.
 *  @param round_constant The round constant for this step.
 *  @return The finished word, broadcast across all four lanes.
 */
SZ_HELPER_AUTO v128_t sz_aes256_key_turn_v128relaxed_(v128_t previous_u8x16, sz_u8_t round_constant) {
    v128_t const last_word_u8x16 = wasm_i32x4_shuffle(previous_u8x16, previous_u8x16, 3, 3, 3, 3);
    v128_t const rotated_u8x16 = wasm_i8x16_shuffle(last_word_u8x16, last_word_u8x16, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11,
                                                    8, 13, 14, 15, 12);
    return wasm_v128_xor(sz_aes256_substitute_v128relaxed_(rotated_u8x16), wasm_i32x4_splat((sz_i32_t)round_constant));
}

/**
 *  @brief Builds the schedule's plainly substituted word, the step an AES-256 schedule interleaves.
 *  @param previous_u8x16 The four schedule words immediately before the new quadruple.
 *  @return The finished word, broadcast across all four lanes.
 */
SZ_HELPER_AUTO v128_t sz_aes256_key_half_turn_v128relaxed_(v128_t previous_u8x16) {
    return sz_aes256_substitute_v128relaxed_(wasm_i32x4_shuffle(previous_u8x16, previous_u8x16, 3, 3, 3, 3));
}

SZ_API_COMPTIME void sz_aes256_key_init_v128relaxed(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    v128_t even_round_key_u8x16 = wasm_v128_load(secret);
    v128_t odd_round_key_u8x16 = wasm_v128_load(secret + 16);

    // An AES-256 schedule alternates two steps: a rotated substitution against a round constant, and
    // a plain substitution. Seven of the first and six of the second follow the two round keys the
    // secret supplies, and each fold turns one substituted word into a whole round key.
    wasm_v128_store(&key->round_keys[0], even_round_key_u8x16);
    wasm_v128_store(&key->round_keys[4], odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_v128_(even_round_key_u8x16,
                                                    sz_aes256_key_turn_v128relaxed_(odd_round_key_u8x16, 0x01));
    wasm_v128_store(&key->round_keys[8], even_round_key_u8x16);
    odd_round_key_u8x16 = sz_aes256_key_fold_v128_(odd_round_key_u8x16,
                                                   sz_aes256_key_half_turn_v128relaxed_(even_round_key_u8x16));
    wasm_v128_store(&key->round_keys[12], odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_v128_(even_round_key_u8x16,
                                                    sz_aes256_key_turn_v128relaxed_(odd_round_key_u8x16, 0x02));
    wasm_v128_store(&key->round_keys[16], even_round_key_u8x16);
    odd_round_key_u8x16 = sz_aes256_key_fold_v128_(odd_round_key_u8x16,
                                                   sz_aes256_key_half_turn_v128relaxed_(even_round_key_u8x16));
    wasm_v128_store(&key->round_keys[20], odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_v128_(even_round_key_u8x16,
                                                    sz_aes256_key_turn_v128relaxed_(odd_round_key_u8x16, 0x04));
    wasm_v128_store(&key->round_keys[24], even_round_key_u8x16);
    odd_round_key_u8x16 = sz_aes256_key_fold_v128_(odd_round_key_u8x16,
                                                   sz_aes256_key_half_turn_v128relaxed_(even_round_key_u8x16));
    wasm_v128_store(&key->round_keys[28], odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_v128_(even_round_key_u8x16,
                                                    sz_aes256_key_turn_v128relaxed_(odd_round_key_u8x16, 0x08));
    wasm_v128_store(&key->round_keys[32], even_round_key_u8x16);
    odd_round_key_u8x16 = sz_aes256_key_fold_v128_(odd_round_key_u8x16,
                                                   sz_aes256_key_half_turn_v128relaxed_(even_round_key_u8x16));
    wasm_v128_store(&key->round_keys[36], odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_v128_(even_round_key_u8x16,
                                                    sz_aes256_key_turn_v128relaxed_(odd_round_key_u8x16, 0x10));
    wasm_v128_store(&key->round_keys[40], even_round_key_u8x16);
    odd_round_key_u8x16 = sz_aes256_key_fold_v128_(odd_round_key_u8x16,
                                                   sz_aes256_key_half_turn_v128relaxed_(even_round_key_u8x16));
    wasm_v128_store(&key->round_keys[44], odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_v128_(even_round_key_u8x16,
                                                    sz_aes256_key_turn_v128relaxed_(odd_round_key_u8x16, 0x20));
    wasm_v128_store(&key->round_keys[48], even_round_key_u8x16);
    odd_round_key_u8x16 = sz_aes256_key_fold_v128_(odd_round_key_u8x16,
                                                   sz_aes256_key_half_turn_v128relaxed_(even_round_key_u8x16));
    wasm_v128_store(&key->round_keys[52], odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_v128_(even_round_key_u8x16,
                                                    sz_aes256_key_turn_v128relaxed_(odd_round_key_u8x16, 0x40));
    wasm_v128_store(&key->round_keys[56], even_round_key_u8x16);
}

#pragma endregion // Key Schedule

#pragma region Block Encryption

/**
 *  @brief Encrypts one block with the expanded schedule.
 *  @param key The expanded schedule.
 *  @param block_u8x16 The plaintext block.
 *  @return The ciphertext block.
 */
SZ_HELPER_AUTO v128_t sz_aes256_block_encrypt_v128relaxed_(sz_aes256_key_t const *key, v128_t block_u8x16) {
    sz_size_t round_index;
    block_u8x16 = wasm_v128_xor(block_u8x16, sz_aes256_round_key_v128_(key, 0));
    for (round_index = 1; round_index != 14; ++round_index)
        block_u8x16 = wasm_v128_xor(
            sz_aes256_mix_columns_v128_(sz_aes256_shift_rows_v128_(sz_aes256_substitute_v128relaxed_(block_u8x16))),
            sz_aes256_round_key_v128_(key, round_index));
    return wasm_v128_xor(sz_aes256_shift_rows_v128_(sz_aes256_substitute_v128relaxed_(block_u8x16)),
                         sz_aes256_round_key_v128_(key, 14));
}

#pragma endregion // Block Encryption

#pragma region Counter Mode

SZ_API_COMPTIME void sz_aes256_ctr_xor_v128relaxed(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                                   sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length,
                                                   sz_ptr_t output) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    v128_t const counter_base_u8x16 = sz_aes256_counter_base_v128_(nonce);
    sz_u32_t block_index = (sz_u32_t)(byte_offset / SZ_AES_BLOCK_LENGTH);
    sz_size_t within_block = (sz_size_t)(byte_offset % SZ_AES_BLOCK_LENGTH);
    sz_size_t produced = 0;

    // A start that is not block aligned generates its first block whole and discards the leading bytes.
    if (within_block != 0 && length != 0) {
        sz_u128_vec_t keystream_vec;
        keystream_vec.v128 = sz_aes256_block_encrypt_v128relaxed_(
            key, sz_aes256_counter_block_v128_(counter_base_u8x16, block_index));
        for (; within_block != SZ_AES_BLOCK_LENGTH && produced != length; ++within_block, ++produced)
            output_bytes[produced] = (sz_u8_t)(input_bytes[produced] ^ keystream_vec.u8s[within_block]);
        ++block_index;
    }

    for (; produced + SZ_AES_BLOCK_LENGTH <= length; produced += SZ_AES_BLOCK_LENGTH, ++block_index) {
        v128_t const original_u8x16 = wasm_v128_load(input_bytes + produced);
        v128_t const keystream_u8x16 = sz_aes256_block_encrypt_v128relaxed_(
            key, sz_aes256_counter_block_v128_(counter_base_u8x16, block_index));
        wasm_v128_store(output_bytes + produced, wasm_v128_xor(original_u8x16, keystream_u8x16));
    }

    if (produced != length) {
        sz_u128_vec_t keystream_vec;
        keystream_vec.v128 = sz_aes256_block_encrypt_v128relaxed_(
            key, sz_aes256_counter_block_v128_(counter_base_u8x16, block_index));
        for (within_block = 0; produced != length; ++within_block, ++produced)
            output_bytes[produced] = (sz_u8_t)(input_bytes[produced] ^ keystream_vec.u8s[within_block]);
    }
}

#pragma endregion // Counter Mode

#pragma region Galois Hashing

SZ_API_COMPTIME void sz_aes256_gcm_key_init_v128relaxed(sz_aes256_gcm_key_t *key,
                                                        sz_u8_t const secret[sz_at_least_(32)]) {
    v128_t subkey_u8x16, power_u8x16;
    sz_size_t power_index;

    sz_aes256_key_init_v128relaxed(&key->block, secret);
    subkey_u8x16 = sz_aes256_block_encrypt_v128relaxed_(&key->block, wasm_u64x2_splat(0));
    power_u8x16 = subkey_u8x16;
    wasm_v128_store(&key->powers[0], power_u8x16);
    for (power_index = 1; power_index != 8; ++power_index) {
        power_u8x16 = sz_ghash_multiply_v128_(power_u8x16, subkey_u8x16);
        wasm_v128_store(&key->powers[power_index * SZ_AES_BLOCK_LENGTH], power_u8x16);
    }
}

#pragma endregion // Galois Hashing

#pragma region Streaming Interface

/** @brief Prepares the payload both directions share: counter block, tag mask and empty carries. */
SZ_HELPER_INLINE void sz_aes256_gcm_begin_v128relaxed_(sz_aes256_gcm_state_t *state, sz_aes256_gcm_key_t const *key,
                                                       sz_u8_t const nonce[sz_at_least_(12)]) {
    v128_t initial_u8x16;

    state->key = *key;

    // With a twelve-byte nonce the initial counter block is the nonce followed by a one, and the value
    // encrypted under it masks the finished hash. Data blocks start one past it.
    initial_u8x16 = sz_aes256_counter_block_v128_(sz_aes256_counter_base_v128_(nonce), 1u);
    wasm_v128_store(state->tag_mask, sz_aes256_block_encrypt_v128relaxed_(&state->key.block, initial_u8x16));
    wasm_v128_store(state->counter, initial_u8x16);
    wasm_v128_store(state->accumulator, wasm_u64x2_splat(0));
    wasm_v128_store(state->partial, wasm_u64x2_splat(0));
    wasm_v128_store(state->keystream, wasm_u64x2_splat(0));

    state->associated_length = 0;
    state->text_length = 0;
    state->buffered = 0;
    state->keystream_used = SZ_AES_BLOCK_LENGTH; // ? Forces the first message byte to derive a fresh block
}

/**
 *  @brief Transforms a chunk and absorbs its ciphertext, whichever side of the call that is.
 *  @param state The state.
 *  @param text The chunk to transform.
 *  @param length Bytes in the chunk.
 *  @param output Receives the transformed bytes.
 *  @param direction Which buffer the hash absorbs.
 *
 *  Three passes, because two sixteen-byte rhythms run underneath a caller's arbitrary chunk sizes
 *  and neither may restart at a chunk boundary: whatever the previous chunk left of its keystream
 *  block, then whole blocks, then a trailing block that the next chunk will resume.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_transform_v128relaxed_(sz_aes256_gcm_state_t *state, sz_cptr_t text,
                                                           sz_size_t length, sz_ptr_t output,
                                                           sz_aes256_gcm_direction_t direction) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    v128_t const subkey_u8x16 = wasm_v128_load(state->key.powers);
    sz_u128_vec_t counter_vec;
    v128_t accumulator_u8x16;
    sz_u32_t block_index;
    sz_size_t produced = 0, byte_index;

    if (length == 0) return;
    accumulator_u8x16 = wasm_v128_load(state->accumulator);

    // Associated data ends the moment the first message byte arrives, and its tail needs padding.
    if (state->text_length == 0 && state->buffered != 0) {
        sz_u128_vec_t padded_vec;
        padded_vec.v128 = wasm_v128_load(state->partial);
        for (byte_index = state->buffered; byte_index != SZ_AES_BLOCK_LENGTH; ++byte_index)
            padded_vec.u8s[byte_index] = 0;
        accumulator_u8x16 = sz_ghash_absorb_v128_(accumulator_u8x16, padded_vec.v128, subkey_u8x16);
        state->buffered = 0;
    }

    counter_vec.v128 = wasm_v128_load(state->counter);
    block_index = sz_u32_bytes_reverse(counter_vec.u32s[3]);

    if (state->keystream_used != SZ_AES_BLOCK_LENGTH) {
        sz_size_t const available_bytes = SZ_AES_BLOCK_LENGTH - state->keystream_used;
        sz_size_t const taken_bytes = length < available_bytes ? length : available_bytes;
        accumulator_u8x16 = sz_aes256_gcm_spend_v128_(state, input_bytes, output_bytes, taken_bytes, accumulator_u8x16,
                                                      subkey_u8x16, direction);
        produced = taken_bytes;
    }

    for (; produced + SZ_AES_BLOCK_LENGTH <= length; produced += SZ_AES_BLOCK_LENGTH) {
        v128_t const original_u8x16 = wasm_v128_load(input_bytes + produced);
        v128_t counter_block_u8x16, keystream_u8x16, transformed_u8x16, ciphertext_u8x16;
        block_index += 1;
        counter_block_u8x16 = sz_aes256_counter_block_v128_(counter_vec.v128, block_index);
        keystream_u8x16 = sz_aes256_block_encrypt_v128relaxed_(&state->key.block, counter_block_u8x16);
        transformed_u8x16 = wasm_v128_xor(original_u8x16, keystream_u8x16);
        ciphertext_u8x16 = direction == sz_aes256_gcm_decrypting_k ? original_u8x16 : transformed_u8x16;
        wasm_v128_store(output_bytes + produced, transformed_u8x16);
        accumulator_u8x16 = sz_ghash_absorb_v128_(accumulator_u8x16, ciphertext_u8x16, subkey_u8x16);
    }

    if (produced != length) {
        v128_t counter_block_u8x16, keystream_u8x16;
        block_index += 1;
        counter_block_u8x16 = sz_aes256_counter_block_v128_(counter_vec.v128, block_index);
        keystream_u8x16 = sz_aes256_block_encrypt_v128relaxed_(&state->key.block, counter_block_u8x16);
        wasm_v128_store(state->keystream, keystream_u8x16);
        state->keystream_used = 0;
        accumulator_u8x16 = sz_aes256_gcm_spend_v128_(state, input_bytes + produced, output_bytes + produced,
                                                      length - produced, accumulator_u8x16, subkey_u8x16, direction);
    }

    state->text_length += length;
    wasm_v128_store(state->accumulator, accumulator_u8x16);
    wasm_v128_store(state->counter, sz_aes256_counter_block_v128_(counter_vec.v128, block_index));
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_init_v128relaxed(sz_aes256_gcm_encryptor_t *encryptor,
                                                              sz_aes256_gcm_key_t const *key,
                                                              sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_aes256_gcm_begin_v128relaxed_(&encryptor->state, key, nonce);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_associate_v128relaxed(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                   sz_size_t length) {
    sz_aes256_gcm_encryptor_associate_v128(encryptor, text, length);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_update_v128relaxed(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                sz_size_t length, sz_ptr_t output) {
    sz_aes256_gcm_transform_v128relaxed_(&encryptor->state, text, length, output, sz_aes256_gcm_encrypting_k);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_digest_v128relaxed(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_encryptor_digest_v128(encryptor, tag);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_init_v128relaxed(sz_aes256_gcm_decryptor_t *decryptor,
                                                              sz_aes256_gcm_key_t const *key,
                                                              sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_aes256_gcm_begin_v128relaxed_(&decryptor->state, key, nonce);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_associate_v128relaxed(sz_aes256_gcm_decryptor_t *decryptor, sz_cptr_t text,
                                                                   sz_size_t length) {
    sz_aes256_gcm_decryptor_associate_v128(decryptor, text, length);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_update_unverified_v128relaxed(sz_aes256_gcm_decryptor_t *decryptor,
                                                                           sz_cptr_t text, sz_size_t length,
                                                                           sz_ptr_t output) {
    sz_aes256_gcm_transform_v128relaxed_(&decryptor->state, text, length, output, sz_aes256_gcm_decrypting_k);
}

SZ_API_COMPTIME sz_status_t sz_aes256_gcm_decryptor_verify_v128relaxed(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                       sz_u8_t const tag[sz_at_least_(16)]) {
    return sz_aes256_gcm_decryptor_verify_v128(decryptor, tag);
}

#pragma endregion // Streaming Interface

#pragma region One Shot Interface

SZ_API_COMPTIME void sz_aes256_gcm_encrypt_v128relaxed(sz_aes256_gcm_key_t const *key,
                                                       sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                       sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                       sz_ptr_t output, sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_encryptor_t encryptor;
    sz_aes256_gcm_encryptor_init_v128relaxed(&encryptor, key, nonce);
    if (associated_length) sz_aes256_gcm_encryptor_associate_v128relaxed(&encryptor, associated, associated_length);
    sz_aes256_gcm_encryptor_update_v128relaxed(&encryptor, text, length, output);
    sz_aes256_gcm_encryptor_digest_v128relaxed(&encryptor, tag);
    sz_aes256_gcm_state_scrub_v128_(&encryptor.state);
}

SZ_API_COMPTIME sz_status_t sz_aes256_gcm_decrypt_v128relaxed(sz_aes256_gcm_key_t const *key,
                                                              sz_u8_t const nonce[sz_at_least_(12)],
                                                              sz_cptr_t associated, sz_size_t associated_length,
                                                              sz_cptr_t text, sz_size_t length, sz_ptr_t output,
                                                              sz_u8_t const tag[sz_at_least_(16)]) {
    sz_aes256_gcm_decryptor_t decryptor;
    sz_status_t verdict;
    sz_aes256_gcm_decryptor_init_v128relaxed(&decryptor, key, nonce);
    if (associated_length) sz_aes256_gcm_decryptor_associate_v128relaxed(&decryptor, associated, associated_length);
    sz_aes256_gcm_decryptor_update_unverified_v128relaxed(&decryptor, text, length, output);
    verdict = sz_aes256_gcm_decryptor_verify_v128relaxed(&decryptor, tag);
    sz_aes256_gcm_state_scrub_v128_(&decryptor.state);

    // A caller who drops the status must still be unable to act on forged plaintext.
    if (verdict != sz_success_k) sz_fill(output, length, 0);
    return verdict;
}

#pragma endregion // One Shot Interface

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128RELAXED

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_CIPHER_V128RELAXED_H_
