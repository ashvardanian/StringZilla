/**
 *  @brief WebAssembly SIMD128 backend for AES-256 in counter and Galois/counter modes.
 *  @file include/stringzilla/cipher/v128.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/cipher.h
 *
 *  Every other backend writes its fourteen AES rounds out rather than looping over the round index,
 *  because a loop leaves the unrolling to the optimizer and the optimizer only does it at `-O3`. This
 *  backend keeps the loop, and the difference is the round body. Elsewhere a round is one instruction on
 *  a dedicated cipher unit with several cycles of latency, so unrolling is what keeps that unit fed.
 *  Here there is no cipher instruction at all: a round is a tower-field substitution of some forty
 *  operations, long enough that the loop overhead rounds to nothing and short enough on latency that
 *  there is no stall to hide. Unrolling would multiply the module by thirteen and buy no throughput.
 *  The cost worth attacking in this file is the operation count itself, not the loop around it.
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

/*  WebAssembly has @b no cipher instructions: no AES round, no carry-less multiply, not even a
 *  variable byte rotate. Both halves of authenticated encryption are therefore emulated, and the
 *  bar this backend clears is the serial one, not a hardware one. Nothing here approaches what a
 *  machine with `aesenc` and `pclmulqdq` reaches, and the file is written on that understanding.
 *
 *  What SIMD128 does provide is `i8x16.swizzle`, a genuine sixteen-byte table permute, and that is
 *  enough to change the shape of both kernels. The substitution box becomes a chain of nibble
 *  swizzles over the tower-field decomposition of the inverse in `GF(2^8)`, so all sixteen bytes of
 *  a block are substituted at once instead of one load at a time. The row shift and the column
 *  rotates of the mixing step become shuffles with immediate indices, which cost nothing.
 *
 *  That rewrite buys a second property the serial backend openly gives up. Every table here is
 *  sixteen bytes wide and read in full by a swizzle, so the address stream is a compile-time
 *  constant: the substitution is @b constant @b time, where the serial 256-byte box is not. On a
 *  platform with no cipher instructions and no other defence, that is worth as much as the speed.
 *
 *  The Galois hash keeps the serial backend's reasoning intact. A four-bit windowed multiplier
 *  would index a table derived from the hash subkey, which is derived from the key, so its cache
 *  footprint would leak the key on exactly the platforms that reach this code. This backend
 *  therefore stays table free, and instead folds the serial loop's 128 dependent byte-shift chains
 *  into whole-register doublings: eight shifted copies of the subkey are formed once, each of the
 *  sixteen accumulator bytes selects among them under an all-ones or all-zeros mask, and a Horner
 *  pass over the byte partials multiplies by `x^8` fifteen times to place them.
 *
 *  Measured against the serial backend under Wasmtime on x86, counter mode runs about 1.7 times
 *  faster, authenticated encryption about 6 times, and the Galois multiply alone about 13. Key
 *  expansion is the one kernel that is @b slower, by roughly a factor of two: it substitutes one
 *  word at a time, so fifteen sixteenths of every vector substitution is wasted, where the serial
 *  path reads four table entries and is done. It is kept vectorized anyway, and deliberately. The
 *  cost is around a hundred nanoseconds once per key and appears on no throughput path, and what it
 *  buys is that the secret never indexes memory at all, on the one tier where a cache-timing leak
 *  of the master key has no hardware alternative to fall back on. Delegating this one function to
 *  `sz_aes256_key_init_serial` is the trade in the other direction.
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
 *  Leaving the tower field and applying the map of FIPS 197 are both linear over `GF(2)`, so their
 *  composition is one linear map and needs one pair of tables rather than two. Folding them also
 *  removes the step that recombined the inverse's two nibbles into a byte, because this table is
 *  indexed by that nibble directly. The standard's `0x63` constant rides in this half.
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
 *  Zero has no logarithm, and the usual repair is a comparison and a mask on the product. Sending it
 *  to `0x40` instead makes any sum involving it land beyond both antilogarithm tables, where a swizzle
 *  already answers zero, so the repair costs nothing at all.
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
 *  @param low_table_vec The map's value on each low nibble.
 *  @param high_table_vec The map's value on each high nibble.
 *  @param bytes_vec The sixteen inputs.
 *  @return `low_table_vec[byte & 0xF] ^ high_table_vec[byte >> 4]` in every lane.
 *
 *  A map linear over `GF(2)` splits across the two nibbles of its argument, so a 256-entry table
 *  that no permute could reach collapses into two sixteen-entry tables that one swizzle each can.
 */
SZ_HELPER_INLINE v128_t sz_aes256_nibble_map_v128_(v128_t low_table_vec, v128_t high_table_vec, v128_t bytes_vec) {
    v128_t const low_nibbles_vec = wasm_v128_and(bytes_vec, wasm_i8x16_splat((sz_i8_t)0x0F));
    v128_t const high_nibbles_vec = wasm_u8x16_shr(bytes_vec, 4);
    return wasm_v128_xor(wasm_i8x16_swizzle(low_table_vec, low_nibbles_vec),
                         wasm_i8x16_swizzle(high_table_vec, high_nibbles_vec));
}

/**
 *  @brief Multiplies sixteen pairs of `GF(2^4)` elements under `x^4 + x + 1`.
 *  @param first_vec One operand per lane, each below sixteen.
 *  @param second_vec The other operand per lane, each below sixteen.
 *  @return The product per lane.
 *
 *  Logarithms turn the product into a sum that lands between zero and twenty-eight, which two
 *  antilogarithm swizzles cover between them: the low one answers exponents below sixteen and
 *  returns zero elsewhere, and the high one is fed the sum less sixteen so its own out-of-range
 *  indices fall away the same way. A zero operand carries a logarithm large enough that both
 *  swizzles fall out of range and answer zero on their own.
 */
SZ_HELPER_AUTO v128_t sz_aes256_nibble_multiply_v128_(v128_t first_vec, v128_t second_vec) {
    v128_t const logarithm_table_vec = wasm_v128_load(sz_aes256_nibble_logarithm_v128_());
    v128_t const exponent_low_table_vec = wasm_v128_load(sz_aes256_nibble_exponent_low_v128_());
    v128_t const exponent_high_table_vec = wasm_v128_load(sz_aes256_nibble_exponent_high_v128_());
    v128_t const logarithm_sum_vec = wasm_i8x16_add(wasm_i8x16_swizzle(logarithm_table_vec, first_vec),
                                                    wasm_i8x16_swizzle(logarithm_table_vec, second_vec));
    return wasm_v128_xor(
        wasm_i8x16_swizzle(exponent_low_table_vec, logarithm_sum_vec),
        wasm_i8x16_swizzle(exponent_high_table_vec, wasm_i8x16_sub(logarithm_sum_vec, wasm_i8x16_splat(16))));
}

/**
 *  @brief Applies the substitution box of FIPS 197 to all sixteen bytes of a block.
 *  @param bytes_vec The sixteen inputs.
 *  @return The substituted bytes.
 *
 *  The byte is carried into `GF(2^4)^2`, where its inverse costs three four-bit multiplies, two
 *  squarings and one four-bit inversion, all of them sixteen-entry swizzles. The inverse's two
 *  nibbles then index the fused table that leaves the tower field and applies the standard's affine
 *  map in one step, so nothing has to be reassembled into a byte first. No step indexes memory by a
 *  data byte, which is what makes the whole substitution constant time.
 */
SZ_HELPER_AUTO v128_t sz_aes256_substitute_v128_(v128_t bytes_vec) {
    v128_t const mapped_vec = sz_aes256_nibble_map_v128_(wasm_v128_load(sz_aes256_tower_forward_low_v128_()),
                                                         wasm_v128_load(sz_aes256_tower_forward_high_v128_()),
                                                         bytes_vec);
    v128_t const high_nibbles_vec = wasm_u8x16_shr(mapped_vec, 4);
    v128_t const low_nibbles_vec = wasm_v128_and(mapped_vec, wasm_i8x16_splat((sz_i8_t)0x0F));

    // The tower field's norm, `high^2 * N ^ high * low ^ low^2`, is what has to be inverted in `GF(2^4)`.
    v128_t const high_scaled_vec = wasm_i8x16_swizzle(wasm_v128_load(sz_aes256_nibble_square_scaled_v128_()),
                                                      high_nibbles_vec);
    v128_t const crossed_vec = sz_aes256_nibble_multiply_v128_(high_nibbles_vec, low_nibbles_vec);
    v128_t const low_squared_vec = wasm_i8x16_swizzle(wasm_v128_load(sz_aes256_nibble_square_v128_()), low_nibbles_vec);
    v128_t const tower_norm_vec = wasm_v128_xor(wasm_v128_xor(high_scaled_vec, crossed_vec), low_squared_vec);
    v128_t const norm_inverse_vec = wasm_i8x16_swizzle(wasm_v128_load(sz_aes256_nibble_inverse_v128_()),
                                                       tower_norm_vec);

    v128_t const high_inverse_vec = sz_aes256_nibble_multiply_v128_(high_nibbles_vec, norm_inverse_vec);
    v128_t const low_inverse_vec = sz_aes256_nibble_multiply_v128_(wasm_v128_xor(high_nibbles_vec, low_nibbles_vec),
                                                                   norm_inverse_vec);
    return wasm_v128_xor(wasm_i8x16_swizzle(wasm_v128_load(sz_aes256_substituted_low_v128_()), low_inverse_vec),
                         wasm_i8x16_swizzle(wasm_v128_load(sz_aes256_substituted_high_v128_()), high_inverse_vec));
}

#pragma endregion // Substitution Box

#pragma region Key Schedule

/**
 *  @brief Shifts a register up by one schedule word, filling the vacated word with zeros.
 *  @param words_vec The four schedule words.
 *  @return The same words moved one position later.
 */
SZ_HELPER_INLINE v128_t sz_aes256_key_carry_v128_(v128_t words_vec) {
    v128_t const zeros_vec = wasm_u64x2_splat(0);
    return wasm_i8x16_shuffle(words_vec, zeros_vec, 16, 16, 16, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
}

/**
 *  @brief Folds a substituted schedule word into the round key eight words back, producing four more.
 *  @param previous_vec The four schedule words eight positions back.
 *  @param substituted_vec The substituted word, already broadcast across all four lanes.
 *  @return The next four schedule words.
 *
 *  FIPS 197 writes `w[i] = w[i - 8] ^ temp` with `temp` carried forward through the quadruple, and
 *  that carry is exactly the cumulative exclusive-or three word-wise shifts produce in one register.
 */
SZ_HELPER_INLINE v128_t sz_aes256_key_fold_v128_(v128_t previous_vec, v128_t substituted_vec) {
    previous_vec = wasm_v128_xor(previous_vec, sz_aes256_key_carry_v128_(previous_vec));
    previous_vec = wasm_v128_xor(previous_vec, sz_aes256_key_carry_v128_(previous_vec));
    previous_vec = wasm_v128_xor(previous_vec, sz_aes256_key_carry_v128_(previous_vec));
    return wasm_v128_xor(previous_vec, substituted_vec);
}

/**
 *  @brief Builds the schedule's rotated and substituted word against a round constant.
 *  @param previous_vec The four schedule words immediately before the new quadruple.
 *  @param round_constant The round constant for this step.
 *  @return The finished word, broadcast across all four lanes.
 */
SZ_HELPER_AUTO v128_t sz_aes256_key_turn_v128_(v128_t previous_vec, sz_u8_t round_constant) {
    v128_t const last_word_vec = wasm_i32x4_shuffle(previous_vec, previous_vec, 3, 3, 3, 3);
    v128_t const rotated_vec = wasm_i8x16_shuffle(last_word_vec, last_word_vec, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8,
                                                  13, 14, 15, 12);
    return wasm_v128_xor(sz_aes256_substitute_v128_(rotated_vec), wasm_i32x4_splat((sz_i32_t)round_constant));
}

/**
 *  @brief Builds the schedule's plainly substituted word, the step an AES-256 schedule interleaves.
 *  @param previous_vec The four schedule words immediately before the new quadruple.
 *  @return The finished word, broadcast across all four lanes.
 */
SZ_HELPER_AUTO v128_t sz_aes256_key_half_turn_v128_(v128_t previous_vec) {
    return sz_aes256_substitute_v128_(wasm_i32x4_shuffle(previous_vec, previous_vec, 3, 3, 3, 3));
}

SZ_API_COMPTIME void sz_aes256_key_init_v128(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    v128_t even_round_key_vec = wasm_v128_load(secret);
    v128_t odd_round_key_vec = wasm_v128_load(secret + 16);

    // An AES-256 schedule alternates two steps: a rotated substitution against a round constant, and
    // a plain substitution. Seven of the first and six of the second follow the two round keys the
    // secret supplies, and each fold turns one substituted word into a whole round key.
    wasm_v128_store(&key->round_keys[0], even_round_key_vec);
    wasm_v128_store(&key->round_keys[4], odd_round_key_vec);
    even_round_key_vec = sz_aes256_key_fold_v128_(even_round_key_vec,
                                                  sz_aes256_key_turn_v128_(odd_round_key_vec, 0x01));
    wasm_v128_store(&key->round_keys[8], even_round_key_vec);
    odd_round_key_vec = sz_aes256_key_fold_v128_(odd_round_key_vec, sz_aes256_key_half_turn_v128_(even_round_key_vec));
    wasm_v128_store(&key->round_keys[12], odd_round_key_vec);
    even_round_key_vec = sz_aes256_key_fold_v128_(even_round_key_vec,
                                                  sz_aes256_key_turn_v128_(odd_round_key_vec, 0x02));
    wasm_v128_store(&key->round_keys[16], even_round_key_vec);
    odd_round_key_vec = sz_aes256_key_fold_v128_(odd_round_key_vec, sz_aes256_key_half_turn_v128_(even_round_key_vec));
    wasm_v128_store(&key->round_keys[20], odd_round_key_vec);
    even_round_key_vec = sz_aes256_key_fold_v128_(even_round_key_vec,
                                                  sz_aes256_key_turn_v128_(odd_round_key_vec, 0x04));
    wasm_v128_store(&key->round_keys[24], even_round_key_vec);
    odd_round_key_vec = sz_aes256_key_fold_v128_(odd_round_key_vec, sz_aes256_key_half_turn_v128_(even_round_key_vec));
    wasm_v128_store(&key->round_keys[28], odd_round_key_vec);
    even_round_key_vec = sz_aes256_key_fold_v128_(even_round_key_vec,
                                                  sz_aes256_key_turn_v128_(odd_round_key_vec, 0x08));
    wasm_v128_store(&key->round_keys[32], even_round_key_vec);
    odd_round_key_vec = sz_aes256_key_fold_v128_(odd_round_key_vec, sz_aes256_key_half_turn_v128_(even_round_key_vec));
    wasm_v128_store(&key->round_keys[36], odd_round_key_vec);
    even_round_key_vec = sz_aes256_key_fold_v128_(even_round_key_vec,
                                                  sz_aes256_key_turn_v128_(odd_round_key_vec, 0x10));
    wasm_v128_store(&key->round_keys[40], even_round_key_vec);
    odd_round_key_vec = sz_aes256_key_fold_v128_(odd_round_key_vec, sz_aes256_key_half_turn_v128_(even_round_key_vec));
    wasm_v128_store(&key->round_keys[44], odd_round_key_vec);
    even_round_key_vec = sz_aes256_key_fold_v128_(even_round_key_vec,
                                                  sz_aes256_key_turn_v128_(odd_round_key_vec, 0x20));
    wasm_v128_store(&key->round_keys[48], even_round_key_vec);
    odd_round_key_vec = sz_aes256_key_fold_v128_(odd_round_key_vec, sz_aes256_key_half_turn_v128_(even_round_key_vec));
    wasm_v128_store(&key->round_keys[52], odd_round_key_vec);
    even_round_key_vec = sz_aes256_key_fold_v128_(even_round_key_vec,
                                                  sz_aes256_key_turn_v128_(odd_round_key_vec, 0x40));
    wasm_v128_store(&key->round_keys[56], even_round_key_vec);
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
 *  @param bytes_vec The state.
 *  @return The row-shifted state.
 */
SZ_HELPER_INLINE v128_t sz_aes256_shift_rows_v128_(v128_t bytes_vec) {
    return wasm_i8x16_shuffle(bytes_vec, bytes_vec, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11);
}

/**
 *  @brief Applies `MixColumns` to all four columns at once.
 *  @param bytes_vec The row-shifted state.
 *  @return The mixed state.
 *
 *  Written as `s[j] ^ parity ^ xtime(s[j] ^ s[j + 1])`, the transform needs only the column's total
 *  parity and the doubled difference of adjacent entries, so three immediate shuffles supply every
 *  neighbour a lane could want and the doubling is a shift against a mask of the bytes that overflow.
 */
SZ_HELPER_INLINE v128_t sz_aes256_mix_columns_v128_(v128_t bytes_vec) {
    v128_t const next_column_vec = wasm_i8x16_shuffle(bytes_vec, bytes_vec, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13,
                                                      14, 15, 12);
    v128_t const across_column_vec = wasm_i8x16_shuffle(bytes_vec, bytes_vec, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14,
                                                        15, 12, 13);
    v128_t const before_column_vec = wasm_i8x16_shuffle(bytes_vec, bytes_vec, 3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15,
                                                        12, 13, 14);
    v128_t const parity_vec = wasm_v128_xor(wasm_v128_xor(bytes_vec, next_column_vec),
                                            wasm_v128_xor(across_column_vec, before_column_vec));
    v128_t const difference_vec = wasm_v128_xor(bytes_vec, next_column_vec);
    v128_t const overflow_vec = wasm_v128_and(wasm_i8x16_shr(difference_vec, 7), wasm_i8x16_splat((sz_i8_t)0x1B));
    v128_t const doubled_vec = wasm_v128_xor(wasm_i8x16_shl(difference_vec, 1), overflow_vec);
    return wasm_v128_xor(wasm_v128_xor(bytes_vec, parity_vec), doubled_vec);
}

/**
 *  @brief Encrypts one block with the expanded schedule.
 *  @param key The expanded schedule.
 *  @param block_vec The plaintext block.
 *  @return The ciphertext block.
 */
SZ_HELPER_AUTO v128_t sz_aes256_block_encrypt_v128_(sz_aes256_key_t const *key, v128_t block_vec) {
    sz_size_t round_index;
    block_vec = wasm_v128_xor(block_vec, sz_aes256_round_key_v128_(key, 0));
    for (round_index = 1; round_index != 14; ++round_index)
        block_vec = wasm_v128_xor(
            sz_aes256_mix_columns_v128_(sz_aes256_shift_rows_v128_(sz_aes256_substitute_v128_(block_vec))),
            sz_aes256_round_key_v128_(key, round_index));
    return wasm_v128_xor(sz_aes256_shift_rows_v128_(sz_aes256_substitute_v128_(block_vec)),
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
    sz_u128_vec_t base_vec;
    sz_size_t byte_index;
    base_vec.v128 = wasm_u64x2_splat(0);
    for (byte_index = 0; byte_index != 12; ++byte_index) base_vec.u8s[byte_index] = nonce[byte_index];
    return base_vec.v128;
}

/**
 *  @brief Completes a counter block with a big-endian block index in its trailing word.
 *  @param base_vec A counter block carrying the nonce, whose trailing word is overwritten.
 *  @param block_index The block index.
 *  @return The counter block for that index.
 */
SZ_HELPER_INLINE v128_t sz_aes256_counter_block_v128_(v128_t base_vec, sz_u32_t block_index) {
    return wasm_u32x4_replace_lane(base_vec, 3, sz_u32_bytes_reverse(block_index));
}

SZ_API_COMPTIME void sz_aes256_ctr_xor_v128(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                            sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length, sz_ptr_t output) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    v128_t const counter_base_vec = sz_aes256_counter_base_v128_(nonce);
    sz_u32_t block_index = (sz_u32_t)(byte_offset / SZ_AES_BLOCK_LENGTH);
    sz_size_t within_block = (sz_size_t)(byte_offset % SZ_AES_BLOCK_LENGTH);
    sz_size_t produced = 0;

    // A start that is not block aligned generates its first block whole and discards the leading bytes.
    if (within_block != 0 && length != 0) {
        sz_u128_vec_t keystream_vec;
        keystream_vec.v128 = sz_aes256_block_encrypt_v128_(
            key, sz_aes256_counter_block_v128_(counter_base_vec, block_index));
        for (; within_block != SZ_AES_BLOCK_LENGTH && produced != length; ++within_block, ++produced)
            output_bytes[produced] = (sz_u8_t)(input_bytes[produced] ^ keystream_vec.u8s[within_block]);
        ++block_index;
    }

    for (; produced + SZ_AES_BLOCK_LENGTH <= length; produced += SZ_AES_BLOCK_LENGTH, ++block_index) {
        v128_t const original_vec = wasm_v128_load(input_bytes + produced);
        v128_t const keystream_vec = sz_aes256_block_encrypt_v128_(
            key, sz_aes256_counter_block_v128_(counter_base_vec, block_index));
        wasm_v128_store(output_bytes + produced, wasm_v128_xor(original_vec, keystream_vec));
    }

    if (produced != length) {
        sz_u128_vec_t keystream_vec;
        keystream_vec.v128 = sz_aes256_block_encrypt_v128_(
            key, sz_aes256_counter_block_v128_(counter_base_vec, block_index));
        for (within_block = 0; produced != length; ++within_block, ++produced)
            output_bytes[produced] = (sz_u8_t)(input_bytes[produced] ^ keystream_vec.u8s[within_block]);
    }
}

#pragma endregion // Counter Mode

#pragma region Galois Hashing

/**
 *  @brief Multiplies a hash block by `x` in the Galois field the tag is built over.
 *  @param value_vec The block, in the byte order the tag is defined over.
 *  @return The product.
 *
 *  The field puts a block's leading bit at the polynomial's lowest coefficient, so multiplying by
 *  `x` moves every bit one place later and the bit that leaves the block at the far end comes back
 *  as the reduction polynomial `x^7 + x^2 + x + 1`. Rotating the per-byte carries rather than
 *  shifting them delivers that bit straight into the leading byte's top position, which is the
 *  reduction's `x^0` term, so only the remaining three taps need a mask.
 */
SZ_HELPER_INLINE v128_t sz_ghash_double_v128_(v128_t value_vec) {
    v128_t const within_byte_vec = wasm_u8x16_shr(value_vec, 1);
    v128_t const carried_vec = wasm_i8x16_shl(value_vec, 7);
    v128_t const rotated_vec = wasm_i8x16_shuffle(carried_vec, carried_vec, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                                  12, 13, 14);
    v128_t const shifted_vec = wasm_v128_or(within_byte_vec, rotated_vec);
    v128_t const overflowed_vec = wasm_i8x16_shr(rotated_vec, 7);
    v128_t const reduction_taps_vec = wasm_i8x16_const(0x61, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    return wasm_v128_xor(shifted_vec, wasm_v128_and(overflowed_vec, reduction_taps_vec));
}

/**
 *  @brief Multiplies a hash block by `x^8` in the Galois field the tag is built over.
 *  @param value_vec The block, in the byte order the tag is defined over.
 *  @return The product.
 *
 *  Eight places is a whole byte, so the shift itself is one immediate shuffle and only the byte that
 *  leaves the block needs work. Its eight coefficients land on `x^128` through `x^135`, and reducing
 *  them is the carry-less product of that byte with the reduction polynomial, which occupies fifteen
 *  bits and therefore never needs reducing again. Reading the polynomial through the field's bit
 *  order turns that product into `x^7 + x^2 + x + 1` written backwards, which is the constant below.
 */
SZ_HELPER_AUTO v128_t sz_ghash_double_byte_v128_(v128_t value_vec) {
    v128_t const zeros_vec = wasm_u64x2_splat(0);
    v128_t const shifted_vec = wasm_i8x16_shuffle(value_vec, zeros_vec, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                  13, 14);
    sz_u32_t const spilled_byte = wasm_u8x16_extract_lane(value_vec, 15);
    sz_u32_t const folded_taps = spilled_byte ^ (spilled_byte << 5) ^ (spilled_byte << 6) ^ (spilled_byte << 7);
    sz_u16_t const reduction_taps = (sz_u16_t)(((folded_taps >> 7) & 0xFFu) | ((folded_taps << 9) & 0xFF00u));
    return wasm_v128_xor(shifted_vec, wasm_u16x8_replace_lane(zeros_vec, 0, reduction_taps));
}

/**
 *  @brief Multiplies @p accumulator_vec by @p subkey_vec in the Galois field the tag is built over.
 *  @param accumulator_vec The running hash.
 *  @param subkey_vec One of the precomputed powers of the hash subkey.
 *  @return The product.
 *
 *  Deliberately branch free and table free, for the reason the serial backend states: a windowed
 *  multiplier's table is derived from the subkey, and its cache footprint would leak the key on
 *  exactly the platforms with no cipher instructions to fall back on. What this backend adds is
 *  arithmetic, not indirection. The eight shifted copies `subkey * x^0` through `subkey * x^7` are
 *  formed once, each accumulator byte selects among them through an all-ones or all-zeros mask into
 *  one partial product, and a Horner pass multiplying by `x^8` places the sixteen partials. The
 *  sixteen selections are independent of one another, so only the doubling chains carry latency.
 */
SZ_HELPER_AUTO v128_t sz_ghash_multiply_v128_(v128_t accumulator_vec, v128_t subkey_vec) {
    v128_t shifted_subkeys_vec[8];
    sz_u128_vec_t selectors_vec;
    v128_t product_vec = wasm_u64x2_splat(0);
    sz_size_t byte_index, bit_index;

    shifted_subkeys_vec[0] = subkey_vec;
    for (bit_index = 1; bit_index != 8; ++bit_index)
        shifted_subkeys_vec[bit_index] = sz_ghash_double_v128_(shifted_subkeys_vec[bit_index - 1]);

    selectors_vec.v128 = accumulator_vec;
    for (byte_index = 16; byte_index-- != 0;) {
        sz_u8_t const selector = selectors_vec.u8s[byte_index];
        v128_t partial_product_vec = wasm_u64x2_splat(0);
        for (bit_index = 0; bit_index != 8; ++bit_index) {
            v128_t const selector_mask_vec = wasm_i8x16_splat(
                (sz_i8_t)(0u - (sz_u32_t)((selector >> (7 - bit_index)) & 1u)));
            partial_product_vec = wasm_v128_xor(partial_product_vec,
                                                wasm_v128_and(shifted_subkeys_vec[bit_index], selector_mask_vec));
        }
        product_vec = wasm_v128_xor(sz_ghash_double_byte_v128_(product_vec), partial_product_vec);
    }
    return product_vec;
}

/**
 *  @brief Absorbs one block into the running hash.
 *  @param accumulator_vec The running hash.
 *  @param block_vec The block to absorb.
 *  @param subkey_vec The hash subkey.
 *  @return The updated running hash.
 */
SZ_HELPER_INLINE v128_t sz_ghash_absorb_v128_(v128_t accumulator_vec, v128_t block_vec, v128_t subkey_vec) {
    return sz_ghash_multiply_v128_(wasm_v128_xor(accumulator_vec, block_vec), subkey_vec);
}

SZ_API_COMPTIME void sz_aes256_gcm_key_init_v128(sz_aes256_gcm_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    v128_t subkey_vec, power_vec;
    sz_size_t power_index;

    sz_aes256_key_init_v128(&key->block, secret);
    subkey_vec = sz_aes256_block_encrypt_v128_(&key->block, wasm_u64x2_splat(0));
    power_vec = subkey_vec;
    wasm_v128_store(&key->powers[0], power_vec);
    for (power_index = 1; power_index != 8; ++power_index) {
        power_vec = sz_ghash_multiply_v128_(power_vec, subkey_vec);
        wasm_v128_store(&key->powers[power_index * SZ_AES_BLOCK_LENGTH], power_vec);
    }
}

#pragma endregion // Galois Hashing

#pragma region Streaming Interface

/**
 *  @brief Overwrites a finished state so the key schedule it embeds does not outlive the call.
 *
 *  The size is known at compile time, so this is a straight-line fill through the widest store the
 *  target has rather than a length-driven loop, and both loops below fold away entirely. Writes are
 *  ordinary stores followed by one barrier: routing them through a `volatile` view instead would
 *  forbid vectorization and cost 472 single-byte stores on every one-shot call.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_state_scrub_v128_(sz_aes256_gcm_state_t *state) {
    sz_u8_t *const bytes = (sz_u8_t *)state;
    v128_t const zeros_vec = wasm_u64x2_splat(0);
    sz_size_t byte_index = 0;
    for (; byte_index + 16 <= sizeof(*state); byte_index += 16) wasm_v128_store(bytes + byte_index, zeros_vec);
    for (; byte_index != sizeof(*state); byte_index += 8) wasm_v128_store64_lane(bytes + byte_index, zeros_vec, 0);
    sz_keep_alive_(state);
}

/** @brief Prepares the payload both directions share: counter block, tag mask and empty carries. */
SZ_HELPER_INLINE void sz_aes256_gcm_begin_v128_(sz_aes256_gcm_state_t *state, sz_aes256_gcm_key_t const *key,
                                                sz_u8_t const nonce[sz_at_least_(12)]) {
    v128_t initial_vec;

    state->key = *key;

    // With a twelve-byte nonce the initial counter block is the nonce followed by a one, and the value
    // encrypted under it masks the finished hash. Data blocks start one past it.
    initial_vec = sz_aes256_counter_block_v128_(sz_aes256_counter_base_v128_(nonce), 1u);
    wasm_v128_store(state->tag_mask, sz_aes256_block_encrypt_v128_(&state->key.block, initial_vec));
    wasm_v128_store(state->counter, initial_vec);
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
    v128_t const subkey_vec = wasm_v128_load(state->key.powers);
    v128_t accumulator_vec;
    sz_size_t consumed = 0, byte_index;

    if (length == 0) return;
    accumulator_vec = wasm_v128_load(state->accumulator);
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
            accumulator_vec = sz_ghash_absorb_v128_(accumulator_vec, wasm_v128_load(state->partial), subkey_vec);
            state->buffered = 0;
        }
    }

    for (; consumed + SZ_AES_BLOCK_LENGTH <= length; consumed += SZ_AES_BLOCK_LENGTH)
        accumulator_vec = sz_ghash_absorb_v128_(accumulator_vec, wasm_v128_load(input_bytes + consumed), subkey_vec);
    for (; consumed != length; ++consumed) state->partial[state->buffered++] = input_bytes[consumed];

    wasm_v128_store(state->accumulator, accumulator_vec);
}

/**
 *  @brief Spends what is left of the keystream block the state carries, one byte at a time.
 *  @param state The state, whose two sixteen-byte counters advance together here.
 *  @param input The bytes to transform.
 *  @param output Receives the transformed bytes.
 *  @param count Bytes to consume, never more than the keystream block has left.
 *  @param accumulator_vec The running hash.
 *  @param subkey_vec The hash subkey.
 *  @param direction Which buffer the hash absorbs.
 *  @return The updated running hash.
 *
 *  Through the message the keystream offset and the hash offset are the same number, because every
 *  byte spends one of each, so a chunk that ends mid block leaves both mid block and this resumes
 *  both. The ciphertext byte is captured before the store because a caller may pass one pointer.
 */
SZ_HELPER_INLINE v128_t sz_aes256_gcm_spend_v128_(sz_aes256_gcm_state_t *state, sz_u8_t const *input, sz_u8_t *output,
                                                  sz_size_t count, v128_t accumulator_vec, v128_t subkey_vec,
                                                  sz_aes256_gcm_direction_t direction) {
    sz_size_t byte_index;

    for (byte_index = 0; byte_index != count; ++byte_index) {
        sz_u8_t const plaintext_byte = input[byte_index];
        sz_u8_t const transformed = (sz_u8_t)(plaintext_byte ^ state->keystream[state->keystream_used]);
        sz_u8_t const ciphertext = direction == sz_aes256_gcm_decrypting_k ? plaintext_byte : transformed;
        output[byte_index] = transformed;
        state->partial[state->buffered] = ciphertext;
        ++state->buffered;
        ++state->keystream_used;
        if (state->buffered == SZ_AES_BLOCK_LENGTH) {
            accumulator_vec = sz_ghash_absorb_v128_(accumulator_vec, wasm_v128_load(state->partial), subkey_vec);
            state->buffered = 0;
        }
    }
    return accumulator_vec;
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
SZ_HELPER_INLINE void sz_aes256_gcm_transform_v128_(sz_aes256_gcm_state_t *state, sz_cptr_t text, sz_size_t length,
                                                    sz_ptr_t output, sz_aes256_gcm_direction_t direction) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    v128_t const subkey_vec = wasm_v128_load(state->key.powers);
    sz_u128_vec_t counter_vec;
    v128_t accumulator_vec;
    sz_u32_t block_index;
    sz_size_t produced = 0, byte_index;

    if (length == 0) return;
    accumulator_vec = wasm_v128_load(state->accumulator);

    // Associated data ends the moment the first message byte arrives, and its tail needs padding.
    if (state->text_length == 0 && state->buffered != 0) {
        sz_u128_vec_t padded_vec;
        padded_vec.v128 = wasm_v128_load(state->partial);
        for (byte_index = state->buffered; byte_index != SZ_AES_BLOCK_LENGTH; ++byte_index)
            padded_vec.u8s[byte_index] = 0;
        accumulator_vec = sz_ghash_absorb_v128_(accumulator_vec, padded_vec.v128, subkey_vec);
        state->buffered = 0;
    }

    counter_vec.v128 = wasm_v128_load(state->counter);
    block_index = sz_u32_bytes_reverse(counter_vec.u32s[3]);

    if (state->keystream_used != SZ_AES_BLOCK_LENGTH) {
        sz_size_t const available_bytes = SZ_AES_BLOCK_LENGTH - state->keystream_used;
        sz_size_t const taken_bytes = length < available_bytes ? length : available_bytes;
        accumulator_vec = sz_aes256_gcm_spend_v128_(state, input_bytes, output_bytes, taken_bytes, accumulator_vec,
                                                    subkey_vec, direction);
        produced = taken_bytes;
    }

    for (; produced + SZ_AES_BLOCK_LENGTH <= length; produced += SZ_AES_BLOCK_LENGTH) {
        v128_t const original_vec = wasm_v128_load(input_bytes + produced);
        v128_t counter_block_vec, keystream_vec, transformed_vec, ciphertext_vec;
        block_index += 1;
        counter_block_vec = sz_aes256_counter_block_v128_(counter_vec.v128, block_index);
        keystream_vec = sz_aes256_block_encrypt_v128_(&state->key.block, counter_block_vec);
        transformed_vec = wasm_v128_xor(original_vec, keystream_vec);
        ciphertext_vec = direction == sz_aes256_gcm_decrypting_k ? original_vec : transformed_vec;
        wasm_v128_store(output_bytes + produced, transformed_vec);
        accumulator_vec = sz_ghash_absorb_v128_(accumulator_vec, ciphertext_vec, subkey_vec);
    }

    if (produced != length) {
        v128_t counter_block_vec, keystream_vec;
        block_index += 1;
        counter_block_vec = sz_aes256_counter_block_v128_(counter_vec.v128, block_index);
        keystream_vec = sz_aes256_block_encrypt_v128_(&state->key.block, counter_block_vec);
        wasm_v128_store(state->keystream, keystream_vec);
        state->keystream_used = 0;
        accumulator_vec = sz_aes256_gcm_spend_v128_(state, input_bytes + produced, output_bytes + produced,
                                                    length - produced, accumulator_vec, subkey_vec, direction);
    }

    state->text_length += length;
    wasm_v128_store(state->accumulator, accumulator_vec);
    wasm_v128_store(state->counter, sz_aes256_counter_block_v128_(counter_vec.v128, block_index));
}

SZ_HELPER_INLINE void sz_aes256_gcm_digest_v128_(sz_aes256_gcm_state_t const *state, sz_u8_t tag[sz_at_least_(16)]) {
    v128_t const subkey_vec = wasm_v128_load(state->key.powers);
    v128_t accumulator_vec = wasm_v128_load(state->accumulator);
    sz_u128_vec_t lengths_vec;
    sz_size_t byte_index;

    // Whatever is still pending gets padded here: an associated-data tail when the message was empty,
    // or the message's own trailing ciphertext bytes otherwise.
    if (state->buffered != 0) {
        sz_u128_vec_t padded_vec;
        padded_vec.v128 = wasm_v128_load(state->partial);
        for (byte_index = state->buffered; byte_index != SZ_AES_BLOCK_LENGTH; ++byte_index)
            padded_vec.u8s[byte_index] = 0;
        accumulator_vec = sz_ghash_absorb_v128_(accumulator_vec, padded_vec.v128, subkey_vec);
    }

    sz_aes256_gcm_lengths_serial_(&lengths_vec, state->associated_length, state->text_length);
    accumulator_vec = sz_ghash_absorb_v128_(accumulator_vec, lengths_vec.v128, subkey_vec);

    wasm_v128_store(tag, wasm_v128_xor(accumulator_vec, wasm_v128_load(state->tag_mask)));
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
    return sz_aes256_tag_equal_serial_(expected_vec.u8s, tag) == sz_true_k ? sz_success_k : sz_authentication_failed_k;
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
