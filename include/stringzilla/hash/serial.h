/**
 *  @brief Serial (scalar) backend for string hashing and checksums.
 *  @file include/stringzilla/hash/serial.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_SERIAL_H_
#define STRINGZILLA_HASH_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`

#ifdef __cplusplus
extern "C" {
#endif

SZ_PUBLIC sz_u64_t sz_bytesum_serial(sz_cptr_t text, sz_size_t length) {
    sz_u64_t bytesum = 0;
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *text_end = text_u8 + length;
    for (; text_u8 != text_end; ++text_u8) bytesum += *text_u8;
    return bytesum;
}

/**
 *  @brief Emulates the behaviour of `_mm_aesenc_si128` for a single round.
 *         This function is used as a fallback when the hardware-accelerated version is not available.
 *  @return Result of `MixColumns(SubBytes(ShiftRows(state))) ^ round_key`.
 *  @see Based on Jean-Philippe Aumasson's reference implementation: https://github.com/veorq/aesenc-noNI
 */
SZ_INTERNAL sz_u128_vec_t sz_emulate_aesenc_si128_serial_(sz_u128_vec_t state_vec, sz_u128_vec_t round_key_vec) {
    static sz_u8_t const sbox[256] = {
        // 0     1    2      3     4    5     6     7      8    9     A      B     C     D     E     F
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76, //
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, //
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, //
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, //
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, //
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, //
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, //
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, //
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, //
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, //
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, //
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, //
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, //
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, //
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf, //
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};

    // Combine `ShiftRows` and `SubBytes`
    sz_u8_t state_2d[4][4];

    state_2d[0][0] = sbox[state_vec.u8s[0]];
    state_2d[3][1] = sbox[state_vec.u8s[1]];
    state_2d[2][2] = sbox[state_vec.u8s[2]];
    state_2d[1][3] = sbox[state_vec.u8s[3]];

    state_2d[1][0] = sbox[state_vec.u8s[4]];
    state_2d[0][1] = sbox[state_vec.u8s[5]];
    state_2d[3][2] = sbox[state_vec.u8s[6]];
    state_2d[2][3] = sbox[state_vec.u8s[7]];

    state_2d[2][0] = sbox[state_vec.u8s[8]];
    state_2d[1][1] = sbox[state_vec.u8s[9]];
    state_2d[0][2] = sbox[state_vec.u8s[10]];
    state_2d[3][3] = sbox[state_vec.u8s[11]];

    state_2d[3][0] = sbox[state_vec.u8s[12]];
    state_2d[2][1] = sbox[state_vec.u8s[13]];
    state_2d[1][2] = sbox[state_vec.u8s[14]];
    state_2d[0][3] = sbox[state_vec.u8s[15]];

    // Perform `MixColumns` using GF2 multiplication by 2
#define sz_gf2_double_(x) (((x) << 1) ^ ((((x) >> 7) & 1) * 0x1b))
    // Row 0:
    sz_u8_t row0_saved = state_2d[0][0];
    sz_u8_t row0_xor = state_2d[0][0] ^ state_2d[0][1] ^ state_2d[0][2] ^ state_2d[0][3];
    state_2d[0][0] ^= row0_xor ^ sz_gf2_double_(state_2d[0][0] ^ state_2d[0][1]);
    state_2d[0][1] ^= row0_xor ^ sz_gf2_double_(state_2d[0][1] ^ state_2d[0][2]);
    state_2d[0][2] ^= row0_xor ^ sz_gf2_double_(state_2d[0][2] ^ state_2d[0][3]);
    state_2d[0][3] ^= row0_xor ^ sz_gf2_double_(state_2d[0][3] ^ row0_saved);

    // Row 1:
    sz_u8_t row1_saved = state_2d[1][0];
    sz_u8_t row1_xor = state_2d[1][0] ^ state_2d[1][1] ^ state_2d[1][2] ^ state_2d[1][3];
    state_2d[1][0] ^= row1_xor ^ sz_gf2_double_(state_2d[1][0] ^ state_2d[1][1]);
    state_2d[1][1] ^= row1_xor ^ sz_gf2_double_(state_2d[1][1] ^ state_2d[1][2]);
    state_2d[1][2] ^= row1_xor ^ sz_gf2_double_(state_2d[1][2] ^ state_2d[1][3]);
    state_2d[1][3] ^= row1_xor ^ sz_gf2_double_(state_2d[1][3] ^ row1_saved);

    // Row 2:
    sz_u8_t row2_saved = state_2d[2][0];
    sz_u8_t row2_xor = state_2d[2][0] ^ state_2d[2][1] ^ state_2d[2][2] ^ state_2d[2][3];
    state_2d[2][0] ^= row2_xor ^ sz_gf2_double_(state_2d[2][0] ^ state_2d[2][1]);
    state_2d[2][1] ^= row2_xor ^ sz_gf2_double_(state_2d[2][1] ^ state_2d[2][2]);
    state_2d[2][2] ^= row2_xor ^ sz_gf2_double_(state_2d[2][2] ^ state_2d[2][3]);
    state_2d[2][3] ^= row2_xor ^ sz_gf2_double_(state_2d[2][3] ^ row2_saved);

    // Row 3:
    sz_u8_t row3_saved = state_2d[3][0];
    sz_u8_t row3_xor = state_2d[3][0] ^ state_2d[3][1] ^ state_2d[3][2] ^ state_2d[3][3];
    state_2d[3][0] ^= row3_xor ^ sz_gf2_double_(state_2d[3][0] ^ state_2d[3][1]);
    state_2d[3][1] ^= row3_xor ^ sz_gf2_double_(state_2d[3][1] ^ state_2d[3][2]);
    state_2d[3][2] ^= row3_xor ^ sz_gf2_double_(state_2d[3][2] ^ state_2d[3][3]);
    state_2d[3][3] ^= row3_xor ^ sz_gf2_double_(state_2d[3][3] ^ row3_saved);
#undef sz_gf2_double_

    // Export `XOR`-ing with the round key
    sz_u128_vec_t result = *(sz_u128_vec_t *)state_2d;
    result.u64s[0] ^= round_key_vec.u64s[0];
    result.u64s[1] ^= round_key_vec.u64s[1];
    return result;
}

/**
 *  @brief Emulates the `_mm_shuffle_epi8` (pshufb) instruction for a single 128-bit vector.
 *         Reorders bytes of @p state_vec according to the @p order permutation table.
 *  @param state_vec Input 128-bit vector whose bytes are to be shuffled.
 *  @param order Permutation table: order[i] gives the source byte index for output byte i.
 *  @return Shuffled 128-bit vector.
 */
SZ_INTERNAL sz_u128_vec_t sz_emulate_shuffle_epi8_serial_(sz_u128_vec_t state_vec,
                                                          sz_u8_t const order[sz_at_least_(16)]) {
    sz_u128_vec_t result;
    // Unroll the loop for 16 bytes
    result.u8s[0] = state_vec.u8s[order[0]];
    result.u8s[1] = state_vec.u8s[order[1]];
    result.u8s[2] = state_vec.u8s[order[2]];
    result.u8s[3] = state_vec.u8s[order[3]];
    result.u8s[4] = state_vec.u8s[order[4]];
    result.u8s[5] = state_vec.u8s[order[5]];
    result.u8s[6] = state_vec.u8s[order[6]];
    result.u8s[7] = state_vec.u8s[order[7]];
    result.u8s[8] = state_vec.u8s[order[8]];
    result.u8s[9] = state_vec.u8s[order[9]];
    result.u8s[10] = state_vec.u8s[order[10]];
    result.u8s[11] = state_vec.u8s[order[11]];
    result.u8s[12] = state_vec.u8s[order[12]];
    result.u8s[13] = state_vec.u8s[order[13]];
    result.u8s[14] = state_vec.u8s[order[14]];
    result.u8s[15] = state_vec.u8s[order[15]];
    return result;
}

/**
 *  @brief Provides 1024 bits worth of precomputed Pi constants for the hash.
 *  @return Pointer aligned to 64 bytes on SIMD-capable platforms.
 *
 *  Bailey-Borwein-Plouffe @b (BBP) formula is used to compute the hexadecimal digits of Pi.
 *  It can be easily implemented in just 10 lines of Python and for 1024 bits requires 256 digits:
 *
 *  @code{.py}
 *      def pi(digits: int) -> str:
 *          n, d = 0, 1
 *          HEX = "0123456789ABCDEF"
 *          result = ["3."]
 *          for i in range(digits):
 *              xn = 120 * i**2 + 151 * i + 47
 *              xd = 512 * i**4 + 1024 * i**3 + 712 * i**2 + 194 * i + 15
 *              n = ((16 * n * xd) + (xn * d)) % (d * xd)
 *              d *= xd
 *              result.append(HEX[(16 * n) // d])
 *          return "".join(result)
 *  @endcode
 *
 *  For `pi(16)` the result is `3.243F6A8885A308D3` and you can find the digits after the dot in
 *  the first element of output array.
 *
 *  @see Bailey-Borwein-Plouffe @b (BBP) formula explanation by Mosè Giordano:
 *       https://giordano.github.io/blog/2017-11-21-hexadecimal-pi/
 *
 */
SZ_INTERNAL sz_u64_t const *sz_hash_pi_constants_(void) {
    static sz_align_(64) sz_u64_t const pi[16] = {
        0x243F6A8885A308D3ull, 0x13198A2E03707344ull, 0xA4093822299F31D0ull, 0x082EFA98EC4E6C89ull,
        0x452821E638D01377ull, 0xBE5466CF34E90C6Cull, 0xC0AC29B7C97C50DDull, 0x3F84D5B5B5470917ull,
        0x9216D5D98979FB1Bull, 0xD1310BA698DFB5ACull, 0x2FFD72DBD01ADFB7ull, 0xB8E1AFED6A267E96ull,
        0xBA7C9045F12C7F99ull, 0x24A19947B3916CF7ull, 0x0801F2E2858EFC16ull, 0x636920D871574E69ull,
    };
    return &pi[0];
}

/**
 *  @brief Provides a shuffle mask for the additive part, identical to "aHash" in a single lane.
 *  @return Pointer aligned to 64 bytes on SIMD-capable platforms.
 */
SZ_INTERNAL sz_u8_t const *sz_hash_u8x16x4_shuffle_(void) {
    static sz_align_(64) sz_u8_t const shuffle[64] = {
        0x04, 0x0b, 0x09, 0x06, 0x08, 0x0d, 0x0f, 0x05, //
        0x0e, 0x03, 0x01, 0x0c, 0x00, 0x07, 0x0a, 0x02, //
        0x04, 0x0b, 0x09, 0x06, 0x08, 0x0d, 0x0f, 0x05, //
        0x0e, 0x03, 0x01, 0x0c, 0x00, 0x07, 0x0a, 0x02, //
        0x04, 0x0b, 0x09, 0x06, 0x08, 0x0d, 0x0f, 0x05, //
        0x0e, 0x03, 0x01, 0x0c, 0x00, 0x07, 0x0a, 0x02, //
        0x04, 0x0b, 0x09, 0x06, 0x08, 0x0d, 0x0f, 0x05, //
        0x0e, 0x03, 0x01, 0x0c, 0x00, 0x07, 0x0a, 0x02  //
    };
    return &shuffle[0];
}

/**
 *  @brief SHA256 initial hash values: first 32 bits of fractional parts of square roots of first 8 primes.
 *  @return Pointer to 8x 32-bit constants, aligned to 64 bytes.
 *  @see FIPS 180-4 Section 5.3.3
 */
SZ_INTERNAL sz_u32_t const *sz_sha256_initial_hash_(void) {
    static sz_align_(64) sz_u32_t const h[8] = {
        0x6a09e667ul, 0xbb67ae85ul, 0x3c6ef372ul, 0xa54ff53aul, //
        0x510e527ful, 0x9b05688cul, 0x1f83d9abul, 0x5be0cd19ul, //
    };
    return &h[0];
}

/**
 *  @brief SHA256 round constants: first 32 bits of fractional parts of cube roots of first 64 primes.
 *  @return Pointer to 64x 32-bit constants, aligned to 64 bytes.
 *  @see FIPS 180-4 Section 4.2.2
 */
SZ_INTERNAL sz_u32_t const *sz_sha256_round_constants_(void) {
    static sz_align_(64) sz_u32_t const k[64] = {
        0x428a2f98ul, 0x71374491ul, 0xb5c0fbcful, 0xe9b5dba5ul, //
        0x3956c25bul, 0x59f111f1ul, 0x923f82a4ul, 0xab1c5ed5ul, //
        0xd807aa98ul, 0x12835b01ul, 0x243185beul, 0x550c7dc3ul, //
        0x72be5d74ul, 0x80deb1feul, 0x9bdc06a7ul, 0xc19bf174ul, //
        0xe49b69c1ul, 0xefbe4786ul, 0x0fc19dc6ul, 0x240ca1ccul, //
        0x2de92c6ful, 0x4a7484aaul, 0x5cb0a9dcul, 0x76f988daul, //
        0x983e5152ul, 0xa831c66dul, 0xb00327c8ul, 0xbf597fc7ul, //
        0xc6e00bf3ul, 0xd5a79147ul, 0x06ca6351ul, 0x14292967ul, //
        0x27b70a85ul, 0x2e1b2138ul, 0x4d2c6dfcul, 0x53380d13ul, //
        0x650a7354ul, 0x766a0abbul, 0x81c2c92eul, 0x92722c85ul, //
        0xa2bfe8a1ul, 0xa81a664bul, 0xc24b8b70ul, 0xc76c51a3ul, //
        0xd192e819ul, 0xd6990624ul, 0xf40e3585ul, 0x106aa070ul, //
        0x19a4c116ul, 0x1e376c08ul, 0x2748774cul, 0x34b0bcb5ul, //
        0x391c0cb3ul, 0x4ed8aa4aul, 0x5b9cca4ful, 0x682e6ff3ul, //
        0x748f82eeul, 0x78a5636ful, 0x84c87814ul, 0x8cc70208ul, //
        0x90befffaul, 0xa4506cebul, 0xbef9a3f7ul, 0xc67178f2ul, //
    };
    return &k[0];
}

/**
 *  @brief Initializes the minimal single-lane AES hash state from a 64-bit seed.
 *  @param state Pointer to the minimal hash state to initialize.
 *  @param seed 64-bit seed value mixed with Pi constants to form the initial state.
 */
SZ_INTERNAL void sz_hash_minimal_init_serial_(sz_hash_minimal_t_ *state, sz_u64_t seed) {

    // The key is made from the seed and half of it will be mixed with the length in the end
    state->key.u64s[1] = seed;
    state->key.u64s[0] = seed;

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    state->aes.u64s[0] = seed ^ pi[0];
    state->aes.u64s[1] = seed ^ pi[1];
    state->sum.u64s[0] = seed ^ pi[8];
    state->sum.u64s[1] = seed ^ pi[9];
}

/**
 *  @brief Absorbs one 128-bit block into the minimal hash state.
 *  @param state Pointer to the minimal hash state.
 *  @param block 128-bit data block to absorb.
 */
SZ_INTERNAL void sz_hash_minimal_update_serial_(sz_hash_minimal_t_ *state, sz_u128_vec_t block) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    state->aes = sz_emulate_aesenc_si128_serial_(state->aes, block);
    state->sum = sz_emulate_shuffle_epi8_serial_(state->sum, shuffle);
    state->sum.u64s[0] += block.u64s[0], state->sum.u64s[1] += block.u64s[1];
}

/**
 *  @brief Finalizes the minimal hash state, mixing in the total byte count, and returns a 64-bit digest.
 *  @param state Pointer to the (const) minimal hash state.
 *  @param length Total number of bytes hashed, mixed into the key for length sensitivity.
 *  @return 64-bit hash value.
 */
SZ_INTERNAL sz_u64_t sz_hash_minimal_finalize_serial_(sz_hash_minimal_t_ const *state, sz_size_t length) {
    // Mix the length into the key
    sz_u128_vec_t key_with_length = state->key;
    key_with_length.u64s[0] += length;
    // Combine the "sum" and the "AES" blocks
    sz_u128_vec_t mixed = sz_emulate_aesenc_si128_serial_(state->sum, state->aes);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    sz_u128_vec_t mixed_in_register = sz_emulate_aesenc_si128_serial_(
        sz_emulate_aesenc_si128_serial_(mixed, key_with_length), mixed);
    // Extract the low 64 bits
    return mixed_in_register.u64s[0];
}

/**
 *  @brief Right-shifts the 128-bit vector by @p shift_bytes bytes in software.
 *         Used to zero out trailing overlap bytes when the last block is shorter than 16 bytes.
 *  @param vec Pointer to the 128-bit vector to shift in place.
 *  @param shift_bytes Number of bytes to shift right (0–15); shifting by 0 is a no-op.
 */
SZ_INTERNAL void sz_hash_shift_in_register_serial_(sz_u128_vec_t *vec, int shift_bytes) {
    // One of the ridiculous things about x86, the `bsrli` instruction requires its operand to be an immediate.
    // On GCC and Clang, we could use the provided `__int128` type, but MSVC doesn't support it.
    // So we need to emulate it with 2x 64-bit shifts.
    if (shift_bytes >= 8) {
        vec->u64s[0] = (vec->u64s[1] >> (shift_bytes - 8) * 8);
        vec->u64s[1] = (0);
    }
    else if (shift_bytes) { //! If `shift_bytes == 0`, the shift would cause UB.
        vec->u64s[0] = (vec->u64s[0] >> shift_bytes * 8) | (vec->u64s[1] << (8 - shift_bytes) * 8);
        vec->u64s[1] = (vec->u64s[1] >> shift_bytes * 8);
    }
}

SZ_PUBLIC void sz_hash_state_init_serial(sz_hash_state_t *state, sz_u64_t seed) {

    // The key is made from the seed and half of it will be mixed with the length in the end
    sz_u64_t *key_u64s = (sz_u64_t *)state->key;
    key_u64s[0] = seed;
    key_u64s[1] = seed;

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    sz_u64_t *aes_u64s = (sz_u64_t *)state->aes;
    sz_u64_t *sum_u64s = (sz_u64_t *)state->sum;
    for (int lane_index = 0; lane_index < 8; ++lane_index) aes_u64s[lane_index] = seed ^ pi[lane_index];
    for (int lane_index = 0; lane_index < 8; ++lane_index) sum_u64s[lane_index] = seed ^ pi[lane_index + 8];

    // The inputs are zeroed out at the beginning
    for (int byte_index = 0; byte_index < 64; ++byte_index) state->ins[byte_index] = 0;
    state->ins_length = 0;
}

/**
 *  @brief Absorbs the current 64-byte input buffer into the hash state (four 128-bit lanes).
 *  @param state Pointer to the hash state whose `ins` buffer is consumed.
 */
SZ_INTERNAL void sz_hash_state_update_serial_(sz_hash_state_t *state) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();

    // To reuse the snippets above, let's cast to our familiar 128-bit vectors
    sz_u128_vec_t *aes_vecs = (sz_u128_vec_t *)state->aes;
    sz_u128_vec_t *sum_vecs = (sz_u128_vec_t *)state->sum;
    sz_u128_vec_t *ins_vecs = (sz_u128_vec_t *)state->ins;

    // First 128-bit block
    aes_vecs[0] = sz_emulate_aesenc_si128_serial_(aes_vecs[0], ins_vecs[0]);
    sum_vecs[0] = sz_emulate_shuffle_epi8_serial_(sum_vecs[0], shuffle);
    sum_vecs[0].u64s[0] += ins_vecs[0].u64s[0], sum_vecs[0].u64s[1] += ins_vecs[0].u64s[1];

    // Second 128-bit block
    aes_vecs[1] = sz_emulate_aesenc_si128_serial_(aes_vecs[1], ins_vecs[1]);
    sum_vecs[1] = sz_emulate_shuffle_epi8_serial_(sum_vecs[1], shuffle);
    sum_vecs[1].u64s[0] += ins_vecs[1].u64s[0], sum_vecs[1].u64s[1] += ins_vecs[1].u64s[1];

    // Third 128-bit block
    aes_vecs[2] = sz_emulate_aesenc_si128_serial_(aes_vecs[2], ins_vecs[2]);
    sum_vecs[2] = sz_emulate_shuffle_epi8_serial_(sum_vecs[2], shuffle);
    sum_vecs[2].u64s[0] += ins_vecs[2].u64s[0], sum_vecs[2].u64s[1] += ins_vecs[2].u64s[1];

    // Fourth 128-bit block
    aes_vecs[3] = sz_emulate_aesenc_si128_serial_(aes_vecs[3], ins_vecs[3]);
    sum_vecs[3] = sz_emulate_shuffle_epi8_serial_(sum_vecs[3], shuffle);
    sum_vecs[3].u64s[0] += ins_vecs[3].u64s[0], sum_vecs[3].u64s[1] += ins_vecs[3].u64s[1];
}

/**
 *  @brief Finalizes the full 512-bit hash state and returns a 64-bit digest.
 *  @param state Pointer to the (const) hash state.
 *  @return 64-bit hash value derived by folding the four AES lanes together with the key.
 */
SZ_INTERNAL sz_u64_t sz_hash_state_finalize_serial_(sz_hash_state_t const *state) {

    // Mix the length into the key
    sz_u64_t const *key_u64s = (sz_u64_t const *)state->key;
    sz_u128_vec_t key_with_length;
    key_with_length.u64s[0] = key_u64s[0] + state->ins_length;
    key_with_length.u64s[1] = key_u64s[1];

    // To reuse the snippets above, let's cast to our familiar 128-bit vectors
    sz_u128_vec_t const *aes_vecs = (sz_u128_vec_t const *)state->aes;
    sz_u128_vec_t const *sum_vecs = (sz_u128_vec_t const *)state->sum;

    // Combine the "sum" and the "AES" blocks
    sz_u128_vec_t mixed0 = sz_emulate_aesenc_si128_serial_(sum_vecs[0], aes_vecs[0]);
    sz_u128_vec_t mixed1 = sz_emulate_aesenc_si128_serial_(sum_vecs[1], aes_vecs[1]);
    sz_u128_vec_t mixed2 = sz_emulate_aesenc_si128_serial_(sum_vecs[2], aes_vecs[2]);
    sz_u128_vec_t mixed3 = sz_emulate_aesenc_si128_serial_(sum_vecs[3], aes_vecs[3]);

    // Combine the mixed registers
    sz_u128_vec_t mixed01 = sz_emulate_aesenc_si128_serial_(mixed0, mixed1);
    sz_u128_vec_t mixed23 = sz_emulate_aesenc_si128_serial_(mixed2, mixed3);
    sz_u128_vec_t mixed = sz_emulate_aesenc_si128_serial_(mixed01, mixed23);

    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    sz_u128_vec_t mixed_in_register = sz_emulate_aesenc_si128_serial_(
        sz_emulate_aesenc_si128_serial_(mixed, key_with_length), mixed);

    // Extract the low 64 bits
    return mixed_in_register.u64s[0];
}

SZ_PUBLIC SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_serial(sz_cptr_t text, sz_size_t length, sz_u64_t seed) {
    if (length <= 16) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data_vec;
        data_vec.u64s[0] = data_vec.u64s[1] = 0;
        for (sz_size_t byte_index = 0; byte_index < length; ++byte_index) data_vec.u8s[byte_index] = text[byte_index];
        sz_hash_minimal_update_serial_(&state, data_vec);
        return sz_hash_minimal_finalize_serial_(&state, length);
    }
    else if (length <= 32) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec;
#if SZ_USE_MISALIGNED_LOADS
        data0_vec.u64s[0] = *(sz_u64_t const *)(text);
        data0_vec.u64s[1] = *(sz_u64_t const *)(text + 8);
        data1_vec.u64s[0] = *(sz_u64_t const *)(text + length - 16);
        data1_vec.u64s[1] = *(sz_u64_t const *)(text + length - 8);
#else
        for (sz_size_t byte_index = 0; byte_index < 16; ++byte_index) data0_vec.u8s[byte_index] = text[byte_index];
        for (sz_size_t byte_index = 0; byte_index < 16; ++byte_index)
            data1_vec.u8s[byte_index] = text[length - 16 + byte_index];
#endif
        // Let's shift the data within the register to de-interleave the bytes.
        sz_hash_shift_in_register_serial_(&data1_vec, (int)(32 - length));
        sz_hash_minimal_update_serial_(&state, data0_vec);
        sz_hash_minimal_update_serial_(&state, data1_vec);
        return sz_hash_minimal_finalize_serial_(&state, length);
    }
    else if (length <= 48) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
#if SZ_USE_MISALIGNED_LOADS
        data0_vec.u64s[0] = *(sz_u64_t const *)(text);
        data0_vec.u64s[1] = *(sz_u64_t const *)(text + 8);
        data1_vec.u64s[0] = *(sz_u64_t const *)(text + 16);
        data1_vec.u64s[1] = *(sz_u64_t const *)(text + 24);
        data2_vec.u64s[0] = *(sz_u64_t const *)(text + length - 16);
        data2_vec.u64s[1] = *(sz_u64_t const *)(text + length - 8);
#else
        for (sz_size_t byte_index = 0; byte_index < 16; ++byte_index) data0_vec.u8s[byte_index] = text[byte_index];
        for (sz_size_t byte_index = 0; byte_index < 16; ++byte_index) data1_vec.u8s[byte_index] = text[16 + byte_index];
        for (sz_size_t byte_index = 0; byte_index < 16; ++byte_index)
            data2_vec.u8s[byte_index] = text[length - 16 + byte_index];
#endif
        // Let's shift the data within the register to de-interleave the bytes.
        sz_hash_shift_in_register_serial_(&data2_vec, (int)(48 - length));
        sz_hash_minimal_update_serial_(&state, data0_vec);
        sz_hash_minimal_update_serial_(&state, data1_vec);
        sz_hash_minimal_update_serial_(&state, data2_vec);
        return sz_hash_minimal_finalize_serial_(&state, length);
    }
    else if (length <= 64) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
#if SZ_USE_MISALIGNED_LOADS
        data0_vec.u64s[0] = *(sz_u64_t const *)(text);
        data0_vec.u64s[1] = *(sz_u64_t const *)(text + 8);
        data1_vec.u64s[0] = *(sz_u64_t const *)(text + 16);
        data1_vec.u64s[1] = *(sz_u64_t const *)(text + 24);
        data2_vec.u64s[0] = *(sz_u64_t const *)(text + 32);
        data2_vec.u64s[1] = *(sz_u64_t const *)(text + 40);
        data3_vec.u64s[0] = *(sz_u64_t const *)(text + length - 16);
        data3_vec.u64s[1] = *(sz_u64_t const *)(text + length - 8);
#else
        for (sz_size_t byte_index = 0; byte_index < 16; ++byte_index) data0_vec.u8s[byte_index] = text[byte_index];
        for (sz_size_t byte_index = 0; byte_index < 16; ++byte_index) data1_vec.u8s[byte_index] = text[16 + byte_index];
        for (sz_size_t byte_index = 0; byte_index < 16; ++byte_index) data2_vec.u8s[byte_index] = text[32 + byte_index];
        for (sz_size_t byte_index = 0; byte_index < 16; ++byte_index)
            data3_vec.u8s[byte_index] = text[length - 16 + byte_index];
#endif
        // Let's shift the data within the register to de-interleave the bytes.
        sz_hash_shift_in_register_serial_(&data3_vec, (int)(64 - length));
        sz_hash_minimal_update_serial_(&state, data0_vec);
        sz_hash_minimal_update_serial_(&state, data1_vec);
        sz_hash_minimal_update_serial_(&state, data2_vec);
        sz_hash_minimal_update_serial_(&state, data3_vec);
        return sz_hash_minimal_finalize_serial_(&state, length);
    }
    else {
        sz_align_(64) sz_hash_state_t state;
        sz_hash_state_init_serial(&state, seed);

#if SZ_USE_MISALIGNED_LOADS
        for (; state.ins_length + 64 <= length; state.ins_length += 64) {
            sz_u64_t *ins_u64s = (sz_u64_t *)state.ins;
            ins_u64s[0] = *(sz_u64_t const *)(text + state.ins_length);
            ins_u64s[1] = *(sz_u64_t const *)(text + state.ins_length + 8);
            ins_u64s[2] = *(sz_u64_t const *)(text + state.ins_length + 16);
            ins_u64s[3] = *(sz_u64_t const *)(text + state.ins_length + 24);
            ins_u64s[4] = *(sz_u64_t const *)(text + state.ins_length + 32);
            ins_u64s[5] = *(sz_u64_t const *)(text + state.ins_length + 40);
            ins_u64s[6] = *(sz_u64_t const *)(text + state.ins_length + 48);
            ins_u64s[7] = *(sz_u64_t const *)(text + state.ins_length + 56);
            sz_hash_state_update_serial_(&state);
        }
#else
        for (; state.ins_length + 64 <= length; state.ins_length += 64) {
            for (sz_size_t byte_index = 0; byte_index < 64; ++byte_index)
                state.ins[byte_index] = text[state.ins_length + byte_index];
            sz_hash_state_update_serial_(&state);
        }
#endif

        if (state.ins_length < length) {
            for (sz_size_t byte_index = 0; byte_index != 64; ++byte_index) state.ins[byte_index] = 0;
            for (sz_size_t byte_index = 0; state.ins_length < length; ++byte_index, ++state.ins_length)
                state.ins[byte_index] = text[state.ins_length];
            sz_hash_state_update_serial_(&state);
            state.ins_length = length;
        }
        return sz_hash_state_finalize_serial_(&state);
    }
}

SZ_PUBLIC void sz_hash_state_update_serial(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    while (length) {
        sz_size_t progress_in_block = state->ins_length % 64;
        sz_size_t to_copy = sz_min_of_two(length, 64 - progress_in_block);
        int const will_fill_block = progress_in_block + to_copy == 64;
        // Update the metadata before we modify the `to_copy` variable
        state->ins_length += to_copy;
        length -= to_copy;
        // Append to the internal buffer until it's full
        while (to_copy--) state->ins[progress_in_block++] = *text++;
        // If we've reached the end of the buffer, update the state
        if (will_fill_block) {
            sz_hash_state_update_serial_(state);
            // Reset to zeros now, so we don't have to overwrite an immutable buffer in the folding state
            for (int byte_index = 0; byte_index < 64; ++byte_index) state->ins[byte_index] = 0;
        }
    }
}

SZ_PUBLIC sz_u64_t sz_hash_state_digest_serial(sz_hash_state_t const *state) {
    sz_size_t length = state->ins_length;
    if (length >= 64) return sz_hash_state_finalize_serial_(state);

    // Switch back to a smaller "minimal" state for small inputs
    sz_u64_t const *key_u64s = (sz_u64_t const *)state->key;
    sz_hash_minimal_t_ minimal_state;
    minimal_state.key.u64s[0] = key_u64s[0];
    minimal_state.key.u64s[1] = key_u64s[1];
    minimal_state.aes = *(sz_u128_vec_t const *)state->aes;
    minimal_state.sum = *(sz_u128_vec_t const *)state->sum;

    // The logic is different depending on the length of the input
    sz_u128_vec_t const *ins_vecs = (sz_u128_vec_t const *)state->ins;
    if (length <= 16) {
        sz_hash_minimal_update_serial_(&minimal_state, ins_vecs[0]);
        return sz_hash_minimal_finalize_serial_(&minimal_state, length);
    }
    else if (length <= 32) {
        sz_hash_minimal_update_serial_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_serial_(&minimal_state, ins_vecs[1]);
        return sz_hash_minimal_finalize_serial_(&minimal_state, length);
    }
    else if (length <= 48) {
        sz_hash_minimal_update_serial_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_serial_(&minimal_state, ins_vecs[1]);
        sz_hash_minimal_update_serial_(&minimal_state, ins_vecs[2]);
        return sz_hash_minimal_finalize_serial_(&minimal_state, length);
    }
    else {
        sz_hash_minimal_update_serial_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_serial_(&minimal_state, ins_vecs[1]);
        sz_hash_minimal_update_serial_(&minimal_state, ins_vecs[2]);
        sz_hash_minimal_update_serial_(&minimal_state, ins_vecs[3]);
        return sz_hash_minimal_finalize_serial_(&minimal_state, length);
    }
}

#pragma region Serial SHA256 Implementation

/** @brief SHA256 rotate right operation. */
SZ_INTERNAL sz_u32_t sz_sha256_rotr_(sz_u32_t value, sz_u32_t count) {
    return (value >> count) | (value << (32 - count));
}

/** @brief SHA256 Ch (choose) function: (x AND y) XOR (NOT x AND z). */
SZ_INTERNAL sz_u32_t sz_sha256_ch_(sz_u32_t x, sz_u32_t y, sz_u32_t z) { return (x & y) ^ (~x & z); }

/** @brief SHA256 Maj (majority) function: (x AND y) XOR (x AND z) XOR (y AND z). */
SZ_INTERNAL sz_u32_t sz_sha256_maj_(sz_u32_t x, sz_u32_t y, sz_u32_t z) { return (x & y) ^ (x & z) ^ (y & z); }

/** @brief SHA256 Sigma0 function: ROTR(x,2) XOR ROTR(x,13) XOR ROTR(x,22). */
SZ_INTERNAL sz_u32_t sz_sha256_sigma0_(sz_u32_t x) {
    return sz_sha256_rotr_(x, 2) ^ sz_sha256_rotr_(x, 13) ^ sz_sha256_rotr_(x, 22);
}

/** @brief SHA256 Sigma1 function: ROTR(x,6) XOR ROTR(x,11) XOR ROTR(x,25). */
SZ_INTERNAL sz_u32_t sz_sha256_sigma1_(sz_u32_t x) {
    return sz_sha256_rotr_(x, 6) ^ sz_sha256_rotr_(x, 11) ^ sz_sha256_rotr_(x, 25);
}

/** @brief SHA256 sigma0 function: ROTR(x,7) XOR ROTR(x,18) XOR SHR(x,3). */
SZ_INTERNAL sz_u32_t sz_sha256_sigma0_lower_(sz_u32_t x) {
    return sz_sha256_rotr_(x, 7) ^ sz_sha256_rotr_(x, 18) ^ (x >> 3);
}

/** @brief SHA256 sigma1 function: ROTR(x,17) XOR ROTR(x,19) XOR SHR(x,10). */
SZ_INTERNAL sz_u32_t sz_sha256_sigma1_lower_(sz_u32_t x) {
    return sz_sha256_rotr_(x, 17) ^ sz_sha256_rotr_(x, 19) ^ (x >> 10);
}

/**
 *  @brief Process a single 512-bit (64-byte) block of data using SHA256.
 *  @param hash Pointer to 8x 32-bit hash values, modified in place.
 *  @param block Pointer to 64-byte message block.
 */
SZ_INTERNAL void sz_sha256_process_block_serial_(sz_u32_t hash[sz_at_least_(8)],
                                                 sz_u8_t const block[sz_at_least_(64)]) {
    sz_u32_t const *round_constants = sz_sha256_round_constants_();
    sz_u32_t message_schedule[16];
    sz_u32_t a, b, c, d, e, f, g, h, temp1, temp2;

    // Initialize working variables
    a = hash[0], b = hash[1], c = hash[2], d = hash[3];
    e = hash[4], f = hash[5], g = hash[6], h = hash[7];

    // Main compression loop - rounds until 16th
    for (sz_size_t round_index = 0; round_index < 16; ++round_index) {
        // Read big-endian 32-bit words from the block
        message_schedule[round_index] = ((sz_u32_t)block[round_index * 4 + 0] << 24) |
                                        ((sz_u32_t)block[round_index * 4 + 1] << 16) |
                                        ((sz_u32_t)block[round_index * 4 + 2] << 8) |
                                        ((sz_u32_t)block[round_index * 4 + 3] << 0);
        temp1 = h + sz_sha256_sigma1_(e) + sz_sha256_ch_(e, f, g) + round_constants[round_index] +
                message_schedule[round_index % 16];
        temp2 = sz_sha256_sigma0_(a) + sz_sha256_maj_(a, b, c);
        h = g, g = f, f = e;
        e = d + temp1;
        d = c, c = b, b = a;
        a = temp1 + temp2;
    }

    // Main compression loop - rounds from 16th to 64th
    for (sz_size_t round_index = 16; round_index < 64; ++round_index) {
        message_schedule[(round_index) % 16] = sz_sha256_sigma1_lower_(message_schedule[(round_index - 2) % 16]) +
                                               message_schedule[(round_index - 7) % 16] +
                                               sz_sha256_sigma0_lower_(message_schedule[(round_index - 15) % 16]) +
                                               message_schedule[(round_index - 16) % 16];
        temp1 = h + sz_sha256_sigma1_(e) + sz_sha256_ch_(e, f, g) + round_constants[round_index] +
                message_schedule[round_index % 16];
        temp2 = sz_sha256_sigma0_(a) + sz_sha256_maj_(a, b, c);
        h = g, g = f, f = e;
        e = d + temp1;
        d = c, c = b, b = a;
        a = temp1 + temp2;
    }

    // Add compressed chunk to current hash value
    hash[0] += a, hash[1] += b, hash[2] += c, hash[3] += d;
    hash[4] += e, hash[5] += f, hash[6] += g, hash[7] += h;
}

SZ_PUBLIC void sz_sha256_state_init_serial(sz_sha256_state_t *state_ptr) {
    // Vectorize the load/store of 8x u32s as 4x u64s
    sz_u32_t const *initial_hash = sz_sha256_initial_hash_();
    sz_u64_t const *source = (sz_u64_t const *)initial_hash;
    sz_u64_t *target = (sz_u64_t *)state_ptr->hash;
    target[0] = source[0], target[1] = source[1], target[2] = source[2], target[3] = source[3];
    state_ptr->block_length = 0, state_ptr->total_length = 0;
}

SZ_PUBLIC void sz_sha256_state_update_serial(sz_sha256_state_t *state_ptr, sz_cptr_t data, sz_size_t length) {
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
    sz_align_(32) sz_u32_t hash[8];
    hash[0] = state_ptr->hash[0], hash[1] = state_ptr->hash[1], hash[2] = state_ptr->hash[2],
    hash[3] = state_ptr->hash[3], hash[4] = state_ptr->hash[4], hash[5] = state_ptr->hash[5],
    hash[6] = state_ptr->hash[6], hash[7] = state_ptr->hash[7];

    // Process head to complete the current block
    if (head_length) {
        for (sz_size_t byte_index = 0; byte_index < head_length; ++byte_index)
            state_ptr->block[state_ptr->block_length++] = input[byte_index];
        sz_sha256_process_block_serial_(hash, state_ptr->block);
        state_ptr->block_length = 0;
        input += head_length;
    }

    // Process body (complete aligned blocks)
    for (sz_size_t processed = 0; processed < body_length; processed += 64, input += 64)
        sz_sha256_process_block_serial_(hash, input);

    // Process tail (remaining bytes into block buffer)
    for (sz_size_t byte_index = 0; byte_index < tail_length; ++byte_index)
        state_ptr->block[byte_index] = input[byte_index];
    state_ptr->block_length = tail_length;

    // Copy hash back
    state_ptr->hash[0] = hash[0], state_ptr->hash[1] = hash[1], state_ptr->hash[2] = hash[2],
    state_ptr->hash[3] = hash[3];
    state_ptr->hash[4] = hash[4], state_ptr->hash[5] = hash[5], state_ptr->hash[6] = hash[6],
    state_ptr->hash[7] = hash[7];
}

SZ_PUBLIC void sz_sha256_state_digest_serial(sz_sha256_state_t const *state_ptr, sz_u8_t digest[sz_at_least_(32)]) {
    // Create a copy of the state for padding
    sz_sha256_state_t state = *state_ptr;

    // Append the '1' bit (0x80 byte) after the message
    state.block[state.block_length++] = 0x80;

    // If there's not enough room for the 64-bit length, pad this block and process it
    if (state.block_length > 56) {
        sz_size_t remaining = 64 - state.block_length;
#if SZ_USE_MISALIGNED_LOADS
        // Use word-sized writes for better performance when misaligned stores are supported
        sz_size_t word_bytes = (remaining / 8) * 8;
        for (sz_size_t byte_index = 0; byte_index < word_bytes; byte_index += 8)
            *(sz_u64_t *)&state.block[state.block_length + byte_index] = 0;
        for (sz_size_t byte_index = word_bytes; byte_index < remaining; ++byte_index)
            state.block[state.block_length + byte_index] = 0;
#else
        // Use byte-by-byte writes to avoid alignment issues on platforms like ARMv7
        for (sz_size_t byte_index = 0; byte_index < remaining; ++byte_index)
            state.block[state.block_length + byte_index] = 0;
#endif
        sz_sha256_process_block_serial_(state.hash, state.block);
        state.block_length = 0;
    }

    // Pad with zeros until we have 56 bytes
    sz_size_t remaining = 56 - state.block_length;

#if SZ_USE_MISALIGNED_LOADS
    // Use word-sized writes for better performance when misaligned stores are supported
    sz_size_t word_bytes = (remaining / 8) * 8;
    for (sz_size_t byte_index = 0; byte_index < word_bytes; byte_index += 8)
        *(sz_u64_t *)&state.block[state.block_length + byte_index] = 0;
    for (sz_size_t byte_index = word_bytes; byte_index < remaining; ++byte_index)
        state.block[state.block_length + byte_index] = 0;
#else
    // Use byte-by-byte writes to avoid alignment issues on platforms like ARMv7
    for (sz_size_t byte_index = 0; byte_index < remaining; ++byte_index)
        state.block[state.block_length + byte_index] = 0;
#endif

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
    sz_sha256_process_block_serial_(state.hash, state.block);

    // Produce the final hash digest in big-endian format
    for (sz_size_t lane_index = 0; lane_index < 8; ++lane_index) {
        digest[lane_index * 4 + 0] = (sz_u8_t)(state.hash[lane_index] >> 24);
        digest[lane_index * 4 + 1] = (sz_u8_t)(state.hash[lane_index] >> 16);
        digest[lane_index * 4 + 2] = (sz_u8_t)(state.hash[lane_index] >> 8);
        digest[lane_index * 4 + 3] = (sz_u8_t)(state.hash[lane_index] >> 0);
    }
}

#pragma endregion // Serial SHA256 Implementation

SZ_PUBLIC void sz_fill_random_serial(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_u64_t const *pi_ptr = sz_hash_pi_constants_();
    sz_u128_vec_t input_vec, pi_vec, key_vec, generated_vec;
    for (sz_size_t lane_index = 0; length; ++lane_index) {
        // Each 128-bit block is initialized with the same nonce
        input_vec.u64s[0] = input_vec.u64s[1] = nonce + lane_index;
        // We rotate the first 512-bits of the Pi to mix with the nonce
        pi_vec = ((sz_u128_vec_t const *)pi_ptr)[lane_index % 4];
        key_vec.u64s[0] = nonce ^ pi_vec.u64s[0];
        key_vec.u64s[1] = nonce ^ pi_vec.u64s[1];
        generated_vec = sz_emulate_aesenc_si128_serial_(input_vec, key_vec);
        // Export back to the user-supplied buffer
        for (int byte_index = 0; byte_index < 16 && length; ++byte_index, --length)
            *text++ = generated_vec.u8s[byte_index];
    }
}

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_SERIAL_H_
