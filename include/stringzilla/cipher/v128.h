/**
 *  @brief WebAssembly SIMD128 backend for AES-256 in counter and Galois/counter modes.
 *  @file include/stringzilla/cipher/v128.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/cipher.h
 *
 *  The round loop stays a loop, where every other backend writes its fourteen rounds out.
 */
#ifndef STRINGZILLA_CIPHER_V128_H_
#define STRINGZILLA_CIPHER_V128_H_

#include "stringzilla/types.h"
#include "stringzilla/cipher/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

/*  WebAssembly has @b no cipher instructions: no AES round, no carry-less multiply, not even a variable
 *  byte rotate, so both halves are emulated and the bar is the serial backend rather than hardware.
 *
 *  `i8x16.swizzle` is what makes that worth doing. The substitution box becomes nibble swizzles over the
 *  tower-field decomposition of the inverse in `GF(2^8)`, substituting a whole block at once, and every
 *  table is sixteen bytes read in full, so the address stream is constant and the substitution is
 *  @b constant @b time where the serial 256-byte box is not.
 *
 *  The Galois hash stays table free for the same reason `serial.h` gives: a windowed multiplier would
 *  index a table derived from the key. Key expansion is roughly twice slower than serial, substituting a
 *  word at a time, and is kept vectorized anyway - it runs once per key on no throughput path, and it is
 *  what keeps the secret out of a memory index on the one tier with no hardware fallback.
 */

#pragma region Substitution Box

/** @brief Change of basis into the tower field `GF(2^4)^2`, as the low nibble's half of a linear map. */
SZ_HELPER_INLINE sz_u8_t const *sz_aes256_tower_forward_low_v128_(void) {
    static sz_align_(64) sz_u8_t const forward_low[16] = {0x00, 0x01, 0x20, 0x21, 0x46, 0x47, 0x66, 0x67,
                                                          0x4c, 0x4d, 0x6c, 0x6d, 0x0a, 0x0b, 0x2a, 0x2b};
    return &forward_low[0];
}

/** @brief Change of basis into the tower field `GF(2^4)^2`, as the high nibble's half of a linear map. */
SZ_HELPER_INLINE sz_u8_t const *sz_aes256_tower_forward_high_v128_(void) {
    static sz_align_(64) sz_u8_t const forward_high[16] = {0x00, 0x3c, 0xd5, 0xe9, 0x34, 0x08, 0xe1, 0xdd,
                                                           0xe5, 0xd9, 0x30, 0x0c, 0xd1, 0xed, 0x04, 0x38};
    return &forward_high[0];
}

/**
 *  @brief The way out of the tower field composed with the affine map, indexed by the inverse's low nibble.
 *
 *  Leaving the tower field and applying the map of FIPS 197 are both linear over `GF(2)`, so their composition
 *  is one linear map and needs one pair of tables rather than two.
 */
SZ_HELPER_INLINE sz_u8_t const *sz_aes256_substituted_low_v128_(void) {
    static sz_align_(64) sz_u8_t const substituted_low[16] = {0x63, 0x7c, 0xd1, 0xce, 0xc8, 0xd7, 0x7a, 0x65,
                                                              0x55, 0x4a, 0xe7, 0xf8, 0xfe, 0xe1, 0x4c, 0x53};
    return &substituted_low[0];
}

/** @brief The way out of the tower field composed with the affine map, indexed by the inverse's high nibble. */
SZ_HELPER_INLINE sz_u8_t const *sz_aes256_substituted_high_v128_(void) {
    static sz_align_(64) sz_u8_t const substituted_high[16] = {0x00, 0x52, 0x3e, 0x6c, 0x65, 0x37, 0x5b, 0x09,
                                                               0x60, 0x32, 0x5e, 0x0c, 0x05, 0x57, 0x3b, 0x69};
    return &substituted_high[0];
}

/** @brief Multiplicative inverse in `GF(2^4)` under `x^4 + x + 1`, with zero mapped to zero. */
SZ_HELPER_INLINE sz_u8_t const *sz_aes256_nibble_inverse_v128_(void) {
    static sz_align_(64) sz_u8_t const nibble_inverse[16] = {0x00, 0x01, 0x09, 0x0e, 0x0d, 0x0b, 0x07, 0x06,
                                                             0x0f, 0x02, 0x0c, 0x05, 0x0a, 0x04, 0x03, 0x08};
    return &nibble_inverse[0];
}

/** @brief Squaring in `GF(2^4)`. */
SZ_HELPER_INLINE sz_u8_t const *sz_aes256_nibble_square_v128_(void) {
    static sz_align_(64) sz_u8_t const nibble_square[16] = {0x00, 0x01, 0x04, 0x05, 0x03, 0x02, 0x07, 0x06,
                                                            0x0c, 0x0d, 0x08, 0x09, 0x0f, 0x0e, 0x0b, 0x0a};
    return &nibble_square[0];
}

/** @brief Squaring in `GF(2^4)` scaled by the tower field's norm constant. */
SZ_HELPER_INLINE sz_u8_t const *sz_aes256_nibble_square_scaled_v128_(void) {
    static sz_align_(64) sz_u8_t const nibble_square_scaled[16] = {0x00, 0x08, 0x06, 0x0e, 0x0b, 0x03, 0x0d, 0x05,
                                                                   0x0a, 0x02, 0x0c, 0x04, 0x01, 0x09, 0x07, 0x0f};
    return &nibble_square_scaled[0];
}

/**
 *  @brief Discrete logarithm in `GF(2^4)`, with zero sent far past every antilogarithm's reach.
 *
 *  Zero has no logarithm, and the usual repair is a comparison and a mask on the product.
 */
SZ_HELPER_INLINE sz_u8_t const *sz_aes256_nibble_logarithm_v128_(void) {
    static sz_align_(64) sz_u8_t const nibble_logarithm[16] = {0x40, 0x00, 0x01, 0x04, 0x02, 0x08, 0x05, 0x0a,
                                                               0x03, 0x0e, 0x09, 0x07, 0x06, 0x0d, 0x0b, 0x0c};
    return &nibble_logarithm[0];
}

/** @brief Antilogarithm in `GF(2^4)` for exponents 0 through 15. */
SZ_HELPER_INLINE sz_u8_t const *sz_aes256_nibble_exponent_low_v128_(void) {
    static sz_align_(64) sz_u8_t const nibble_exponent_low[16] = {0x01, 0x02, 0x04, 0x08, 0x03, 0x06, 0x0c, 0x0b,
                                                                  0x05, 0x0a, 0x07, 0x0e, 0x0f, 0x0d, 0x09, 0x01};
    return &nibble_exponent_low[0];
}

/** @brief Antilogarithm in `GF(2^4)` for exponents 16 through 28, indexed by the exponent less sixteen. */
SZ_HELPER_INLINE sz_u8_t const *sz_aes256_nibble_exponent_high_v128_(void) {
    static sz_align_(64) sz_u8_t const nibble_exponent_high[16] = {0x02, 0x04, 0x08, 0x03, 0x06, 0x0c, 0x0b, 0x05,
                                                                   0x0a, 0x07, 0x0e, 0x0f, 0x0d, 0x09, 0x00, 0x00};
    return &nibble_exponent_high[0];
}

/**
 *  @brief Evaluates a map linear over `GF(2)` on all sixteen lanes at once.
 *  @param low_table_u8x16 The map's value on each low nibble.
 *  @param high_table_u8x16 The map's value on each high nibble.
 *  @param bytes_u8x16 The sixteen inputs.
 *  @return `low_table_u8x16[byte & 0xF] ^ high_table_u8x16[byte >> 4]` in every lane.
 *
 *  A map linear over `GF(2)` splits across the two nibbles of its argument, so a 256-entry table that no
 *  permute could reach collapses into two sixteen-entry tables that one swizzle each can.
 */
SZ_HELPER_INLINE v128_t sz_aes256_nibble_map_v128_(v128_t low_table_u8x16, v128_t high_table_u8x16,
                                                   v128_t bytes_u8x16) {
    v128_t const low_nibbles_u8x16 = wasm_v128_and(bytes_u8x16, wasm_i8x16_splat((sz_i8_t)0x0F));
    v128_t const high_nibbles_u8x16 = wasm_u8x16_shr(bytes_u8x16, 4);
    return wasm_v128_xor(wasm_i8x16_swizzle(low_table_u8x16, low_nibbles_u8x16),
                         wasm_i8x16_swizzle(high_table_u8x16, high_nibbles_u8x16));
}

/**
 *  @brief Multiplies sixteen pairs of `GF(2^4)` elements under `x^4 + x + 1`.
 *  @param first_u8x16 One operand per lane, each below sixteen.
 *  @param second_u8x16 The other operand per lane, each below sixteen.
 *  @return The product per lane.
 *
 *  Logarithms turn the product into a sum that lands between zero and twenty-eight, which two antilogarithm
 *  swizzles cover between them: the low one answers exponents below sixteen and returns zero elsewhere, and
 *  the high one is fed the sum less sixteen so its own out-of-range indices fall away the same way.
 */
SZ_HELPER_AUTO v128_t sz_aes256_nibble_multiply_v128_(v128_t first_u8x16, v128_t second_u8x16) {
    v128_t const logarithm_table_u8x16 = wasm_v128_load(sz_aes256_nibble_logarithm_v128_());
    v128_t const exponent_low_table_u8x16 = wasm_v128_load(sz_aes256_nibble_exponent_low_v128_());
    v128_t const exponent_high_table_u8x16 = wasm_v128_load(sz_aes256_nibble_exponent_high_v128_());
    v128_t const logarithm_sum_u8x16 = wasm_i8x16_add(wasm_i8x16_swizzle(logarithm_table_u8x16, first_u8x16),
                                                      wasm_i8x16_swizzle(logarithm_table_u8x16, second_u8x16));
    return wasm_v128_xor(
        wasm_i8x16_swizzle(exponent_low_table_u8x16, logarithm_sum_u8x16),
        wasm_i8x16_swizzle(exponent_high_table_u8x16, wasm_i8x16_sub(logarithm_sum_u8x16, wasm_i8x16_splat(16))));
}

/**
 *  @brief Applies the substitution box of FIPS 197 to all sixteen bytes of a block.
 *  @param bytes_u8x16 The sixteen inputs.
 *  @return The substituted bytes.
 *
 *  The byte is carried into `GF(2^4)^2`, where its inverse costs three four-bit multiplies, two squarings and
 *  one four-bit inversion, all of them sixteen-entry swizzles.
 */
SZ_HELPER_AUTO v128_t sz_aes256_substitute_v128_(v128_t bytes_u8x16) {
    v128_t const mapped_u8x16 = sz_aes256_nibble_map_v128_(wasm_v128_load(sz_aes256_tower_forward_low_v128_()),
                                                           wasm_v128_load(sz_aes256_tower_forward_high_v128_()),
                                                           bytes_u8x16);
    v128_t const high_nibbles_u8x16 = wasm_u8x16_shr(mapped_u8x16, 4);
    v128_t const low_nibbles_u8x16 = wasm_v128_and(mapped_u8x16, wasm_i8x16_splat((sz_i8_t)0x0F));

    // The tower field's norm, `high^2 * N ^ high * low ^ low^2`, is what has to be inverted in `GF(2^4)`.
    v128_t const high_scaled_u8x16 = wasm_i8x16_swizzle(wasm_v128_load(sz_aes256_nibble_square_scaled_v128_()),
                                                        high_nibbles_u8x16);
    v128_t const crossed_u8x16 = sz_aes256_nibble_multiply_v128_(high_nibbles_u8x16, low_nibbles_u8x16);
    v128_t const low_squared_u8x16 = wasm_i8x16_swizzle(wasm_v128_load(sz_aes256_nibble_square_v128_()),
                                                        low_nibbles_u8x16);
    v128_t const tower_norm_u8x16 = wasm_v128_xor(wasm_v128_xor(high_scaled_u8x16, crossed_u8x16), low_squared_u8x16);
    v128_t const tower_norm_inverse_u8x16 = wasm_i8x16_swizzle(wasm_v128_load(sz_aes256_nibble_inverse_v128_()),
                                                               tower_norm_u8x16);

    v128_t const high_inverse_u8x16 = sz_aes256_nibble_multiply_v128_(high_nibbles_u8x16, tower_norm_inverse_u8x16);
    v128_t const low_inverse_u8x16 = sz_aes256_nibble_multiply_v128_(
        wasm_v128_xor(high_nibbles_u8x16, low_nibbles_u8x16), tower_norm_inverse_u8x16);
    return wasm_v128_xor(wasm_i8x16_swizzle(wasm_v128_load(sz_aes256_substituted_low_v128_()), low_inverse_u8x16),
                         wasm_i8x16_swizzle(wasm_v128_load(sz_aes256_substituted_high_v128_()), high_inverse_u8x16));
}

#pragma endregion // Substitution Box

#pragma region Key Schedule

/**
 *  @brief Shifts a register up by one schedule word, filling the vacated word with zeros.
 *  @param words_u8x16 The four schedule words.
 *  @return The same words moved one position later.
 */
SZ_HELPER_INLINE v128_t sz_aes256_key_carry_v128_(v128_t words_u8x16) {
    v128_t const zeros_u8x16 = wasm_u64x2_splat(0);
    return wasm_i8x16_shuffle(words_u8x16, zeros_u8x16, 16, 16, 16, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
}

/**
 *  @brief Folds a substituted schedule word into the round key eight words back, producing four more.
 *  @param previous_u8x16 The four schedule words eight positions back.
 *  @param substituted_u8x16 The substituted word, already broadcast across all four lanes.
 *  @return The next four schedule words.
 *
 *  FIPS 197 writes `w[i] = w[i - 8] ^ temp` with `temp` carried forward through the quadruple, and that carry
 *  is exactly the cumulative exclusive-or three word-wise shifts produce in one register.
 */
SZ_HELPER_INLINE v128_t sz_aes256_key_fold_v128_(v128_t previous_u8x16, v128_t substituted_u8x16) {
    previous_u8x16 = wasm_v128_xor(previous_u8x16, sz_aes256_key_carry_v128_(previous_u8x16));
    previous_u8x16 = wasm_v128_xor(previous_u8x16, sz_aes256_key_carry_v128_(previous_u8x16));
    previous_u8x16 = wasm_v128_xor(previous_u8x16, sz_aes256_key_carry_v128_(previous_u8x16));
    return wasm_v128_xor(previous_u8x16, substituted_u8x16);
}

/**
 *  @brief Builds the schedule's rotated and substituted word against a round constant.
 *  @param previous_u8x16 The four schedule words immediately before the new quadruple.
 *  @param round_constant The round constant for this step.
 *  @return The finished word, broadcast across all four lanes.
 */
SZ_HELPER_AUTO v128_t sz_aes256_key_turn_v128_(v128_t previous_u8x16, sz_u8_t round_constant) {
    v128_t const last_word_u8x16 = wasm_i32x4_shuffle(previous_u8x16, previous_u8x16, 3, 3, 3, 3);
    v128_t const rotated_u8x16 = wasm_i8x16_shuffle(last_word_u8x16, last_word_u8x16, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11,
                                                    8, 13, 14, 15, 12);
    return wasm_v128_xor(sz_aes256_substitute_v128_(rotated_u8x16), wasm_i32x4_splat((sz_i32_t)round_constant));
}

/**
 *  @brief Builds the schedule's plainly substituted word, the step an AES-256 schedule interleaves.
 *  @param previous_u8x16 The four schedule words immediately before the new quadruple.
 *  @return The finished word, broadcast across all four lanes.
 */
SZ_HELPER_AUTO v128_t sz_aes256_key_half_turn_v128_(v128_t previous_u8x16) {
    return sz_aes256_substitute_v128_(wasm_i32x4_shuffle(previous_u8x16, previous_u8x16, 3, 3, 3, 3));
}

SZ_API_COMPTIME void sz_aes256_key_init_v128(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    v128_t even_round_key_u8x16 = wasm_v128_load(secret);
    v128_t odd_round_key_u8x16 = wasm_v128_load(secret + 16);

    // An AES-256 schedule alternates two steps: a rotated substitution against a round constant, and
    // a plain substitution. Seven of the first and six of the second follow the two round keys the
    // secret supplies, and each fold turns one substituted word into a whole round key.
    wasm_v128_store(&key->round_keys[0], even_round_key_u8x16);
    wasm_v128_store(&key->round_keys[4], odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_v128_(even_round_key_u8x16,
                                                    sz_aes256_key_turn_v128_(odd_round_key_u8x16, 0x01));
    wasm_v128_store(&key->round_keys[8], even_round_key_u8x16);
    odd_round_key_u8x16 = sz_aes256_key_fold_v128_(odd_round_key_u8x16,
                                                   sz_aes256_key_half_turn_v128_(even_round_key_u8x16));
    wasm_v128_store(&key->round_keys[12], odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_v128_(even_round_key_u8x16,
                                                    sz_aes256_key_turn_v128_(odd_round_key_u8x16, 0x02));
    wasm_v128_store(&key->round_keys[16], even_round_key_u8x16);
    odd_round_key_u8x16 = sz_aes256_key_fold_v128_(odd_round_key_u8x16,
                                                   sz_aes256_key_half_turn_v128_(even_round_key_u8x16));
    wasm_v128_store(&key->round_keys[20], odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_v128_(even_round_key_u8x16,
                                                    sz_aes256_key_turn_v128_(odd_round_key_u8x16, 0x04));
    wasm_v128_store(&key->round_keys[24], even_round_key_u8x16);
    odd_round_key_u8x16 = sz_aes256_key_fold_v128_(odd_round_key_u8x16,
                                                   sz_aes256_key_half_turn_v128_(even_round_key_u8x16));
    wasm_v128_store(&key->round_keys[28], odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_v128_(even_round_key_u8x16,
                                                    sz_aes256_key_turn_v128_(odd_round_key_u8x16, 0x08));
    wasm_v128_store(&key->round_keys[32], even_round_key_u8x16);
    odd_round_key_u8x16 = sz_aes256_key_fold_v128_(odd_round_key_u8x16,
                                                   sz_aes256_key_half_turn_v128_(even_round_key_u8x16));
    wasm_v128_store(&key->round_keys[36], odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_v128_(even_round_key_u8x16,
                                                    sz_aes256_key_turn_v128_(odd_round_key_u8x16, 0x10));
    wasm_v128_store(&key->round_keys[40], even_round_key_u8x16);
    odd_round_key_u8x16 = sz_aes256_key_fold_v128_(odd_round_key_u8x16,
                                                   sz_aes256_key_half_turn_v128_(even_round_key_u8x16));
    wasm_v128_store(&key->round_keys[44], odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_v128_(even_round_key_u8x16,
                                                    sz_aes256_key_turn_v128_(odd_round_key_u8x16, 0x20));
    wasm_v128_store(&key->round_keys[48], even_round_key_u8x16);
    odd_round_key_u8x16 = sz_aes256_key_fold_v128_(odd_round_key_u8x16,
                                                   sz_aes256_key_half_turn_v128_(even_round_key_u8x16));
    wasm_v128_store(&key->round_keys[52], odd_round_key_u8x16);
    even_round_key_u8x16 = sz_aes256_key_fold_v128_(even_round_key_u8x16,
                                                    sz_aes256_key_turn_v128_(odd_round_key_u8x16, 0x40));
    wasm_v128_store(&key->round_keys[56], even_round_key_u8x16);
}

#pragma endregion // Key Schedule

#pragma region Block Encryption

/**
 *  @brief Reads one round key out of an expanded schedule.
 *  @param key The expanded schedule.
 *  @param round_index Which of the fifteen round keys to read, zero through fourteen.
 *  @return The round key.
 */
SZ_HELPER_INLINE v128_t sz_aes256_round_key_v128_(sz_aes256_key_t const *key, sz_size_t round_index) {
    return wasm_v128_load(&key->round_keys[round_index * 4]);
}

/**
 *  @brief Moves each row of the state left by its row index, as FIPS 197 defines `ShiftRows`.
 *  @param bytes_u8x16 The state.
 *  @return The row-shifted state.
 */
SZ_HELPER_INLINE v128_t sz_aes256_shift_rows_v128_(v128_t bytes_u8x16) {
    return wasm_i8x16_shuffle(bytes_u8x16, bytes_u8x16, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11);
}

/**
 *  @brief Applies `MixColumns` to all four columns at once.
 *  @param bytes_u8x16 The row-shifted state.
 *  @return The mixed state.
 *
 *  Written as `s[j] ^ parity ^ xtime(s[j] ^ s[j + 1])`, the transform needs only the column's total parity and
 *  the doubled difference of adjacent entries, so three immediate shuffles supply every neighbour a lane could
 *  want and the doubling is a shift against a mask of the bytes that overflow.
 */
SZ_HELPER_INLINE v128_t sz_aes256_mix_columns_v128_(v128_t bytes_u8x16) {
    v128_t const next_column_u8x16 = wasm_i8x16_shuffle(bytes_u8x16, bytes_u8x16, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8,
                                                        13, 14, 15, 12);
    v128_t const across_column_u8x16 = wasm_i8x16_shuffle(bytes_u8x16, bytes_u8x16, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8,
                                                          9, 14, 15, 12, 13);
    v128_t const before_column_u8x16 = wasm_i8x16_shuffle(bytes_u8x16, bytes_u8x16, 3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9,
                                                          10, 15, 12, 13, 14);
    v128_t const parity_u8x16 = wasm_v128_xor(wasm_v128_xor(bytes_u8x16, next_column_u8x16),
                                              wasm_v128_xor(across_column_u8x16, before_column_u8x16));
    v128_t const difference_u8x16 = wasm_v128_xor(bytes_u8x16, next_column_u8x16);
    v128_t const overflow_u8x16 = wasm_v128_and(wasm_i8x16_shr(difference_u8x16, 7), wasm_i8x16_splat((sz_i8_t)0x1B));
    v128_t const doubled_u8x16 = wasm_v128_xor(wasm_i8x16_shl(difference_u8x16, 1), overflow_u8x16);
    return wasm_v128_xor(wasm_v128_xor(bytes_u8x16, parity_u8x16), doubled_u8x16);
}

/**
 *  @brief Encrypts one block with the expanded schedule.
 *  @param key The expanded schedule.
 *  @param block_u8x16 The plaintext block.
 *  @return The ciphertext block.
 */
SZ_HELPER_AUTO v128_t sz_aes256_block_encrypt_v128_(sz_aes256_key_t const *key, v128_t block_u8x16) {
    sz_size_t round_index;
    block_u8x16 = wasm_v128_xor(block_u8x16, sz_aes256_round_key_v128_(key, 0));
    for (round_index = 1; round_index != 14; ++round_index)
        block_u8x16 = wasm_v128_xor(
            sz_aes256_mix_columns_v128_(sz_aes256_shift_rows_v128_(sz_aes256_substitute_v128_(block_u8x16))),
            sz_aes256_round_key_v128_(key, round_index));
    return wasm_v128_xor(sz_aes256_shift_rows_v128_(sz_aes256_substitute_v128_(block_u8x16)),
                         sz_aes256_round_key_v128_(key, 14));
}

#pragma endregion // Block Encryption

#pragma region Counter Mode

/**
 *  @brief Places the twelve nonce bytes in a counter block whose trailing index word is zero.
 *  @param nonce The twelve nonce bytes.
 *  @return The counter block for index zero.
 */
SZ_HELPER_INLINE v128_t sz_aes256_counter_base_v128_(sz_u8_t const *nonce) {
    // Only twelve bytes are readable, and SIMD128 has the eight-plus-four pair as two lane loads.
    v128_t const zeros_u8x16 = wasm_u64x2_splat(0);
    return wasm_v128_load32_lane(nonce + 8, wasm_v128_load64_lane(nonce, zeros_u8x16, 0), 2);
}

/**
 *  @brief Completes a counter block with a big-endian block index in its trailing word.
 *  @param base_u8x16 A counter block carrying the nonce, whose trailing word is overwritten.
 *  @param block_index The block index.
 *  @return The counter block for that index.
 */
SZ_HELPER_INLINE v128_t sz_aes256_counter_block_v128_(v128_t base_u8x16, sz_u32_t block_index) {
    return wasm_u32x4_replace_lane(base_u8x16, 3, sz_u32_bytes_reverse(block_index));
}

SZ_API_COMPTIME void sz_aes256_ctr_xor_v128(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                            sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length, sz_ptr_t output) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    v128_t const counter_base_u8x16 = sz_aes256_counter_base_v128_(nonce);
    sz_u32_t block_index = (sz_u32_t)(byte_offset / SZ_AES_BLOCK_LENGTH);
    sz_size_t within_block = (sz_size_t)(byte_offset % SZ_AES_BLOCK_LENGTH);
    sz_size_t produced = 0;

    // A start that is not block aligned generates its first block whole and discards the leading bytes.
    if (within_block != 0 && length != 0) {
        sz_u128_vec_t keystream_vec;
        keystream_vec.v128 = sz_aes256_block_encrypt_v128_(
            key, sz_aes256_counter_block_v128_(counter_base_u8x16, block_index));
        // Stays a byte loop: SIMD128 has no masked store, and writing the register would touch bytes
        // the caller did not hand us. Ice Lake and SVE2 do this run with predication.
        for (; within_block != SZ_AES_BLOCK_LENGTH && produced != length; ++within_block, ++produced)
            output_bytes[produced] = (sz_u8_t)(input_bytes[produced] ^ keystream_vec.u8s[within_block]);
        ++block_index;
    }

    for (; produced + SZ_AES_BLOCK_LENGTH <= length; produced += SZ_AES_BLOCK_LENGTH, ++block_index) {
        v128_t const original_u8x16 = wasm_v128_load(input_bytes + produced);
        v128_t const keystream_u8x16 = sz_aes256_block_encrypt_v128_(
            key, sz_aes256_counter_block_v128_(counter_base_u8x16, block_index));
        wasm_v128_store(output_bytes + produced, wasm_v128_xor(original_u8x16, keystream_u8x16));
    }

    if (produced != length) {
        sz_u128_vec_t keystream_vec;
        keystream_vec.v128 = sz_aes256_block_encrypt_v128_(
            key, sz_aes256_counter_block_v128_(counter_base_u8x16, block_index));
        for (within_block = 0; produced != length; ++within_block, ++produced)
            output_bytes[produced] = (sz_u8_t)(input_bytes[produced] ^ keystream_vec.u8s[within_block]);
    }
}

#pragma endregion // Counter Mode

#pragma region Galois Hashing

/**
 *  @brief Smears bit @p bit_index of every lane, counting from the most significant, across that lane.
 *  @return All ones in a lane whose bit was set, all zeros otherwise.
 *
 *  Shifting the wanted bit into the sign position and arithmetic-shifting it back down replicates it, which is
 *  how a lane selects a value without a branch or a scalar extraction.
 */
SZ_HELPER_INLINE v128_t sz_bit_smear_v128_(v128_t bytes_u8x16, sz_size_t bit_index) {
    return wasm_i8x16_shr(wasm_i8x16_shl(bytes_u8x16, (sz_u32_t)bit_index), 7);
}

/**
 *  @brief Multiplies a hash block by `x` in the Galois field the tag is built over.
 *  @param value_u8x16 The block, in the byte order the tag is defined over.
 *  @return The product.
 *
 *  The field puts a block's leading bit at the polynomial's lowest coefficient, so multiplying by `x` moves
 *  every bit one place later and the bit that leaves the block at the far end comes back as the reduction
 *  polynomial `x^7 + x^2 + x + 1`.
 */
SZ_HELPER_INLINE v128_t sz_ghash_double_v128_(v128_t value_u8x16) {
    v128_t const within_byte_u8x16 = wasm_u8x16_shr(value_u8x16, 1);
    v128_t const carried_u8x16 = wasm_i8x16_shl(value_u8x16, 7);
    v128_t const rotated_u8x16 = wasm_i8x16_shuffle(carried_u8x16, carried_u8x16, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                    11, 12, 13, 14);
    v128_t const shifted_u8x16 = wasm_v128_or(within_byte_u8x16, rotated_u8x16);
    v128_t const overflowed_u8x16 = wasm_i8x16_shr(rotated_u8x16, 7);
    v128_t const reduction_taps_u8x16 = wasm_i8x16_const(0x61, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    return wasm_v128_xor(shifted_u8x16, wasm_v128_and(overflowed_u8x16, reduction_taps_u8x16));
}

/**
 *  @brief Multiplies a hash block by `x^8` in the Galois field the tag is built over.
 *  @param value_u8x16 The block, in the byte order the tag is defined over.
 *  @return The product.
 *
 *  Eight places is a whole byte, so the shift itself is one immediate shuffle and only the byte that leaves
 *  the block needs work.
 */
SZ_HELPER_AUTO v128_t sz_ghash_double_byte_v128_(v128_t value_u8x16) {
    v128_t const zeros_u8x16 = wasm_u64x2_splat(0);
    v128_t const shifted_u8x16 = wasm_i8x16_shuffle(value_u8x16, zeros_u8x16, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                                    12, 13, 14);
    sz_u32_t const spilled_byte = wasm_u8x16_extract_lane(value_u8x16, 15);
    sz_u32_t const folded_taps = spilled_byte ^ (spilled_byte << 5) ^ (spilled_byte << 6) ^ (spilled_byte << 7);
    sz_u16_t const reduction_taps = (sz_u16_t)(((folded_taps >> 7) & 0xFFu) | ((folded_taps << 9) & 0xFF00u));
    return wasm_v128_xor(shifted_u8x16, wasm_u16x8_replace_lane(zeros_u8x16, 0, reduction_taps));
}

/**
 *  @brief Multiplies @p accumulator_u8x16 by @p subkey_u8x16 in the Galois field the tag is built over.
 *  @param accumulator_u8x16 The running hash.
 *  @param subkey_u8x16 One of the precomputed powers of the hash subkey.
 *  @return The product.
 *
 *  Deliberately branch free and table free, for the reason the serial backend states: a windowed multiplier's
 *  table is derived from the subkey, and its cache footprint would leak the key on exactly the platforms with
 *  no cipher instructions to fall back on.
 */
SZ_HELPER_AUTO v128_t sz_ghash_multiply_v128_(v128_t accumulator_u8x16, v128_t subkey_u8x16) {
    v128_t shifted_subkeys_u8x16[8];
    v128_t product_u8x16 = wasm_u64x2_splat(0);
    sz_size_t byte_index, bit_index;

    shifted_subkeys_u8x16[0] = subkey_u8x16;
    for (bit_index = 1; bit_index != 8; ++bit_index)
        shifted_subkeys_u8x16[bit_index] = sz_ghash_double_v128_(shifted_subkeys_u8x16[bit_index - 1]);

    // The accumulator never leaves the register file: a swizzle broadcasts the byte this round selects
    // on, and a shift pair smears each of its bits into a lane mask. Reading it back through a union
    // instead costs a spill, sixteen scalar loads and a splat per bit. Wasmtime 39 currently runs the
    // union form about 4% faster, which is a fact about its register allocator rather than about the
    // two algorithms; the form without a round trip through memory is the one to keep as engines improve.
    for (byte_index = 16; byte_index-- != 0;) {
        v128_t const selector_u8x16 = wasm_i8x16_swizzle(accumulator_u8x16, wasm_i8x16_splat((sz_i8_t)byte_index));
        v128_t partial_product_u8x16 = wasm_u64x2_splat(0);
        for (bit_index = 0; bit_index != 8; ++bit_index)
            partial_product_u8x16 = wasm_v128_xor(
                partial_product_u8x16,
                wasm_v128_and(shifted_subkeys_u8x16[bit_index], sz_bit_smear_v128_(selector_u8x16, bit_index)));
        product_u8x16 = wasm_v128_xor(sz_ghash_double_byte_v128_(product_u8x16), partial_product_u8x16);
    }
    return product_u8x16;
}

/**
 *  @brief Loads a pending block with everything past @p buffered forced to zero.
 *
 *  A lane-identity compare rather than a byte loop.
 */
SZ_HELPER_INLINE v128_t sz_aes256_load_padded_v128_(sz_u8_t const *block, sz_size_t buffered) {
    v128_t const lane_ids_u8x16 = wasm_i8x16_const(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    v128_t const keep_u8x16 = wasm_u8x16_lt(lane_ids_u8x16, wasm_i8x16_splat((sz_i8_t)buffered));
    return wasm_v128_and(wasm_v128_load(block), keep_u8x16);
}

/** @brief Compares two tags in constant time; `sz_true_k` when all sixteen bytes match. */
SZ_HELPER_INLINE sz_bool_t sz_aes256_tag_equal_v128_(sz_u8_t const *first, sz_u8_t const *second) {
    v128_t const matching_u8x16 = wasm_i8x16_eq(wasm_v128_load(first), wasm_v128_load(second));
    return wasm_i8x16_all_true(matching_u8x16) ? sz_true_k : sz_false_k;
}

/**
 *  @brief Absorbs one block into the running hash.
 *  @param accumulator_u8x16 The running hash.
 *  @param block_u8x16 The block to absorb.
 *  @param subkey_u8x16 The hash subkey.
 *  @return The updated running hash.
 */
SZ_HELPER_INLINE v128_t sz_ghash_absorb_v128_(v128_t accumulator_u8x16, v128_t block_u8x16, v128_t subkey_u8x16) {
    return sz_ghash_multiply_v128_(wasm_v128_xor(accumulator_u8x16, block_u8x16), subkey_u8x16);
}

SZ_API_COMPTIME void sz_aes256_gcm_key_init_v128(sz_aes256_gcm_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    v128_t subkey_u8x16, power_u8x16;
    sz_size_t power_index;

    sz_aes256_key_init_v128(&key->block, secret);
    subkey_u8x16 = sz_aes256_block_encrypt_v128_(&key->block, wasm_u64x2_splat(0));
    power_u8x16 = subkey_u8x16;
    wasm_v128_store(&key->powers[0], power_u8x16);
    for (power_index = 1; power_index != 8; ++power_index) {
        power_u8x16 = sz_ghash_multiply_v128_(power_u8x16, subkey_u8x16);
        wasm_v128_store(&key->powers[power_index * SZ_AES_BLOCK_LENGTH], power_u8x16);
    }
}

#pragma endregion // Galois Hashing

#pragma region Streaming Interface

/**
 *  @brief Overwrites a finished state so the key schedule it embeds does not outlive the call.
 *
 *  The size is known at compile time, so this is a straight-line fill through the widest store the target has
 *  rather than a length-driven loop, and both loops below fold away entirely.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_state_scrub_v128_(sz_aes256_gcm_state_t *state) {
    sz_u8_t *const bytes = (sz_u8_t *)state;
    v128_t const zeros_u8x16 = wasm_u64x2_splat(0);
    sz_size_t byte_index = 0;
    for (; byte_index + 16 <= sizeof(*state); byte_index += 16) wasm_v128_store(bytes + byte_index, zeros_u8x16);
    for (; byte_index != sizeof(*state); byte_index += 8) wasm_v128_store64_lane(bytes + byte_index, zeros_u8x16, 0);
    sz_keep_alive_(state);
}

/** @brief Prepares the payload both directions share: counter block, tag mask and empty carries. */
SZ_HELPER_INLINE void sz_aes256_gcm_begin_v128_(sz_aes256_gcm_state_t *state, sz_aes256_gcm_key_t const *key,
                                                sz_u8_t const nonce[sz_at_least_(12)]) {
    v128_t initial_u8x16;

    state->key = *key;

    // With a twelve-byte nonce the initial counter block is the nonce followed by a one, and the value
    // encrypted under it masks the finished hash. Data blocks start one past it.
    initial_u8x16 = sz_aes256_counter_block_v128_(sz_aes256_counter_base_v128_(nonce), 1u);
    wasm_v128_store(state->tag_mask, sz_aes256_block_encrypt_v128_(&state->key.block, initial_u8x16));
    wasm_v128_store(state->counter, initial_u8x16);
    wasm_v128_store(state->accumulator, wasm_u64x2_splat(0));
    wasm_v128_store(state->partial, wasm_u64x2_splat(0));
    wasm_v128_store(state->keystream, wasm_u64x2_splat(0));

    state->associated_length = 0;
    state->text_length = 0;
    state->buffered = 0;
    state->keystream_used = SZ_AES_BLOCK_LENGTH; // ? Forces the first message byte to derive a fresh block
}

/** @brief Absorbs associated data into the payload both directions share. */
SZ_HELPER_INLINE void sz_aes256_gcm_associate_v128_(sz_aes256_gcm_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    v128_t const subkey_u8x16 = wasm_v128_load(state->key.powers);
    v128_t accumulator_u8x16;
    sz_size_t consumed = 0, byte_index;

    if (length == 0) return;
    accumulator_u8x16 = wasm_v128_load(state->accumulator);
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
            accumulator_u8x16 = sz_ghash_absorb_v128_(accumulator_u8x16, wasm_v128_load(state->partial), subkey_u8x16);
            state->buffered = 0;
        }
    }

    for (; consumed + SZ_AES_BLOCK_LENGTH <= length; consumed += SZ_AES_BLOCK_LENGTH)
        accumulator_u8x16 = sz_ghash_absorb_v128_(accumulator_u8x16, wasm_v128_load(input_bytes + consumed),
                                                  subkey_u8x16);
    for (; consumed != length; ++consumed) state->partial[state->buffered++] = input_bytes[consumed];

    wasm_v128_store(state->accumulator, accumulator_u8x16);
}

/**
 *  @brief Spends what is left of the keystream block the state carries, one byte at a time.
 *  @param state The state, whose two sixteen-byte counters advance together here.
 *  @param input The bytes to transform.
 *  @param output Receives the transformed bytes.
 *  @param count Bytes to consume, never more than the keystream block has left.
 *  @param accumulator_u8x16 The running hash.
 *  @param subkey_u8x16 The hash subkey.
 *  @param direction Which buffer the hash absorbs.
 *  @return The updated running hash.
 *
 *  Through the message the keystream offset and the hash offset are the same number, because every byte spends
 *  one of each, so a chunk that ends mid block leaves both mid block and this resumes both.
 */
SZ_HELPER_INLINE v128_t sz_aes256_gcm_spend_v128_(sz_aes256_gcm_state_t *state, sz_u8_t const *input, sz_u8_t *output,
                                                  sz_size_t count, v128_t accumulator_u8x16, v128_t subkey_u8x16,
                                                  sz_aes256_gcm_direction_t direction) {
    sz_size_t byte_index;

    // Scalar: SIMD128 has no masked load or store, so a partial run cannot move without a
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
            accumulator_u8x16 = sz_ghash_absorb_v128_(accumulator_u8x16, wasm_v128_load(state->partial), subkey_u8x16);
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
 *  Three passes, because two sixteen-byte rhythms run underneath a caller's arbitrary chunk sizes and neither
 *  may restart at a chunk boundary: whatever the previous chunk left of its keystream block, then whole
 *  blocks, then a trailing block that the next chunk will resume.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_transform_v128_(sz_aes256_gcm_state_t *state, sz_cptr_t text, sz_size_t length,
                                                    sz_ptr_t output, sz_aes256_gcm_direction_t direction) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    v128_t const subkey_u8x16 = wasm_v128_load(state->key.powers);
    sz_u128_vec_t counter_vec;
    v128_t accumulator_u8x16;
    sz_u32_t block_index;
    sz_size_t produced = 0;

    if (length == 0) return;
    accumulator_u8x16 = wasm_v128_load(state->accumulator);

    // Associated data ends the moment the first message byte arrives, and its tail needs padding.
    if (state->text_length == 0 && state->buffered != 0) {
        accumulator_u8x16 = sz_ghash_absorb_v128_(
            accumulator_u8x16, sz_aes256_load_padded_v128_(state->partial, state->buffered), subkey_u8x16);
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
        keystream_u8x16 = sz_aes256_block_encrypt_v128_(&state->key.block, counter_block_u8x16);
        transformed_u8x16 = wasm_v128_xor(original_u8x16, keystream_u8x16);
        ciphertext_u8x16 = direction == sz_aes256_gcm_decrypting_k ? original_u8x16 : transformed_u8x16;
        wasm_v128_store(output_bytes + produced, transformed_u8x16);
        accumulator_u8x16 = sz_ghash_absorb_v128_(accumulator_u8x16, ciphertext_u8x16, subkey_u8x16);
    }

    if (produced != length) {
        v128_t counter_block_u8x16, keystream_u8x16;
        block_index += 1;
        counter_block_u8x16 = sz_aes256_counter_block_v128_(counter_vec.v128, block_index);
        keystream_u8x16 = sz_aes256_block_encrypt_v128_(&state->key.block, counter_block_u8x16);
        wasm_v128_store(state->keystream, keystream_u8x16);
        state->keystream_used = 0;
        accumulator_u8x16 = sz_aes256_gcm_spend_v128_(state, input_bytes + produced, output_bytes + produced,
                                                      length - produced, accumulator_u8x16, subkey_u8x16, direction);
    }

    state->text_length += length;
    wasm_v128_store(state->accumulator, accumulator_u8x16);
    wasm_v128_store(state->counter, sz_aes256_counter_block_v128_(counter_vec.v128, block_index));
}

SZ_HELPER_INLINE void sz_aes256_gcm_digest_v128_(sz_aes256_gcm_state_t const *state, sz_u8_t tag[sz_at_least_(16)]) {
    v128_t const subkey_u8x16 = wasm_v128_load(state->key.powers);
    v128_t accumulator_u8x16 = wasm_v128_load(state->accumulator);
    sz_u128_vec_t lengths_vec;

    // Whatever is still pending gets padded here: an associated-data tail when the message was empty,
    // or the message's own trailing ciphertext bytes otherwise.
    if (state->buffered != 0) {
        accumulator_u8x16 = sz_ghash_absorb_v128_(
            accumulator_u8x16, sz_aes256_load_padded_v128_(state->partial, state->buffered), subkey_u8x16);
    }

    sz_aes256_gcm_lengths_serial_(&lengths_vec, state->associated_length, state->text_length);
    accumulator_u8x16 = sz_ghash_absorb_v128_(accumulator_u8x16, lengths_vec.v128, subkey_u8x16);

    wasm_v128_store(tag, wasm_v128_xor(accumulator_u8x16, wasm_v128_load(state->tag_mask)));
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_init_v128(sz_aes256_gcm_encryptor_t *encryptor,
                                                       sz_aes256_gcm_key_t const *key,
                                                       sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_aes256_gcm_begin_v128_(&encryptor->state, key, nonce);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_associate_v128(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                            sz_size_t length) {
    sz_aes256_gcm_associate_v128_(&encryptor->state, text, length);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_update_v128(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                         sz_size_t length, sz_ptr_t output) {
    sz_aes256_gcm_transform_v128_(&encryptor->state, text, length, output, sz_aes256_gcm_encrypting_k);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_digest_v128(sz_aes256_gcm_encryptor_t const *encryptor,
                                                         sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_digest_v128_(&encryptor->state, tag);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_init_v128(sz_aes256_gcm_decryptor_t *decryptor,
                                                       sz_aes256_gcm_key_t const *key,
                                                       sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_aes256_gcm_begin_v128_(&decryptor->state, key, nonce);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_associate_v128(sz_aes256_gcm_decryptor_t *decryptor, sz_cptr_t text,
                                                            sz_size_t length) {
    sz_aes256_gcm_associate_v128_(&decryptor->state, text, length);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_update_unverified_v128(sz_aes256_gcm_decryptor_t *decryptor,
                                                                    sz_cptr_t text, sz_size_t length, sz_ptr_t output) {
    sz_aes256_gcm_transform_v128_(&decryptor->state, text, length, output, sz_aes256_gcm_decrypting_k);
}

SZ_API_COMPTIME sz_status_t sz_aes256_gcm_decryptor_verify_v128(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                sz_u8_t const tag[sz_at_least_(16)]) {
    sz_u128_vec_t expected_vec;
    sz_aes256_gcm_digest_v128_(&decryptor->state, expected_vec.u8s);
    return sz_aes256_tag_equal_v128_(expected_vec.u8s, tag) == sz_true_k ? sz_success_k : sz_authentication_failed_k;
}

#pragma endregion // Streaming Interface

#pragma region One Shot Interface

SZ_API_COMPTIME void sz_aes256_gcm_encrypt_v128(sz_aes256_gcm_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                                sz_cptr_t associated, sz_size_t associated_length, sz_cptr_t text,
                                                sz_size_t length, sz_ptr_t output, sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_encryptor_t encryptor;
    sz_aes256_gcm_encryptor_init_v128(&encryptor, key, nonce);
    if (associated_length) sz_aes256_gcm_encryptor_associate_v128(&encryptor, associated, associated_length);
    sz_aes256_gcm_encryptor_update_v128(&encryptor, text, length, output);
    sz_aes256_gcm_encryptor_digest_v128(&encryptor, tag);
    sz_aes256_gcm_state_scrub_v128_(&encryptor.state);
}

SZ_API_COMPTIME sz_status_t sz_aes256_gcm_decrypt_v128(sz_aes256_gcm_key_t const *key,
                                                       sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                       sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                       sz_ptr_t output, sz_u8_t const tag[sz_at_least_(16)]) {
    sz_aes256_gcm_decryptor_t decryptor;
    sz_status_t verdict;
    sz_aes256_gcm_decryptor_init_v128(&decryptor, key, nonce);
    if (associated_length) sz_aes256_gcm_decryptor_associate_v128(&decryptor, associated, associated_length);
    sz_aes256_gcm_decryptor_update_unverified_v128(&decryptor, text, length, output);
    verdict = sz_aes256_gcm_decryptor_verify_v128(&decryptor, tag);
    sz_aes256_gcm_state_scrub_v128_(&decryptor.state);

    // A caller who drops the status must still be unable to act on forged plaintext.
    if (verdict != sz_success_k) sz_fill(output, length, 0);
    return verdict;
}

#pragma endregion // One Shot Interface

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_CIPHER_V128_H_
