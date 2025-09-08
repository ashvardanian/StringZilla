/**
 *  @brief  Hardware-accelerated non-cryptographic string hashing and checksums.
 *  @file   hash.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs with hardware-specific backends:
 *
 *  - `sz_bytesum` - for byte-level 64-bit unsigned byte-level checksums.
 *  - `sz_hash` - for 64-bit single-shot hashing using AES instructions.
 *  - `sz_hash_state_init`, `sz_hash_state_update`, `sz_hash_state_digest` - for incremental hashing.
 *  - `sz_fill_random` - for populating buffers with pseudo-random noise using AES instructions.
 *
 *  Why the hell do we need a yet another hashing library?!
 *  Turns out, most existing libraries have noticeable constraints. Try finding a library that:
 *
 *  - Outputs 64-bit or 128-bit hashes and passes the @b SMHasher `--extra` tests.
 *  - Is fast for both short @b (velocity) and long strings @b (throughput).
 *  - Supports incremental @b (streaming) hashing, when the data arrives in chunks.
 *  - Supports custom @b seeds for hashes and have it affecting every bit of the output.
 *  - Provides @b dynamic-dispatch for different architectures to simplify deployment.
 *  - Uses @b SIMD, including not just AVX2 & NEON, but also masking AVX-512 & predicated SVE2.
 *  - Documents its logic and @b guarantees the same output across different platforms.
 *
 *  This includes projects like "MurmurHash", "CityHash", "SpookyHash", "FarmHash", "MetroHash", "HighwayHash", etc.
 *  There are 2 libraries that are close to meeting these requirements: "xxHash" in C++ and "aHash" in Rust:
 *
 *  - "aHash" is fast, but written in Rust, has no dynamic dispatch, and lacks AVX-512 and SVE2 support.
 *    It also does not adhere to a fixed output, and can't be used in applications like computing packet checksums
 *    in network traffic or implementing persistent data structures.
 *
 *  - "xxHash" is implemented in C, has an extremely wide set of third-party language bindings, and provides both
 *    32-, 64-, and 128-bit hashes. It is fast, but its dynamic dispatch is limited to x86 with `xxh_x86dispatch.c`.
 *
 *  StringZilla uses a scheme more similar to "aHash" and "GxHash", utilizing the AES extensions, that provide
 *  a remarkable level of "mixing per cycle" and are broadly available on modern CPUs. Similar to "aHash", they
 *  are combined with "shuffle & add" instructions to provide a high level of entropy in the output. That operation
 *  is practically free, as many modern CPUs will dispatch them on different ports. On x86, for example:
 *
 *  - `VAESENC (ZMM, ZMM, ZMM)` and `VAESDEC (ZMM, ZMM, ZMM)`:
 *    - on Intel Ice Lake: 5 cycles on port 0.
 *    - On AMD Zen4: 4 cycles on ports 0 or 1.
 *  - `VPSHUFB_Z (ZMM, K, ZMM, ZMM)`
 *    - on Intel Ice Lake: 3 cycles on port 5.
 *    - On AMD Zen4: 2 cycles on ports 1 or 2.
 *  - `VPADDQ (ZMM, ZMM, ZMM)`:
 *    - on Intel Ice Lake: 1 cycle on ports 0 or 5.
 *    - On AMD Zen4: 1 cycle on ports 0, 1, 2, 3.
 *
 *  But there several key differences:
 *
 *  - A larger state and a larger block size is used for inputs over 64 bytes longs, benefiting from wider registers
 *    on current CPUs. Like many other hash functions, the state is initialized with the seed and a set of Pi constants.
 *    Unlike others, we pull more Pi bits (1024), but only 64-bits of the seed, to keep the API sane.
 *  - The length of the input is not mixed into the AES block at the start to allow incremental construction,
 *    when the final length is not known in advance.
 *  - The vector-loads are not interleaved, meaning that each byte of input has exactly the same weight in the hash.
 *    On the implementation side it require some extra shuffling on older platforms, but on newer platforms it
 *    can be done with "masked" loads in AVX-512 and "predicated" instructions in SVE2.
 *
 *  @see Reini Urban's more active fork of SMHasher by Austin Appleby: https://github.com/rurban/smhasher
 *  @see The serial AES routines are based on Morten Jensen's "tiny-AES-c": https://github.com/kokke/tiny-AES-c
 *  @see The "xxHash" C implementation by Yann Collet: https://github.com/Cyan4973/xxHash
 *  @see The "aHash" Rust implementation by Tom Kaitchuck: https://github.com/tkaitchuck/aHash
 *
 *  Moreover, the same AES primitives are reused to implement a fast Pseudo-Random Number Generator @b (PRNG) that
 *  is consistent between different implementation backends and has reproducible output with the same "nonce".
 *  Originally, the PRNG was designed to produce random byte sequences, but combining it with @b `sz_lookup`,
 *  one can produce random strings with a given byteset.
 *
 *  Other helpers include: TODO:
 *
 *  - `sz_fill_alphabet` - combines `sz_fill_random` & `sz_lookup` to fill buffers with random ASCII characters.
 *  - `sz_fill_alphabet_utf8` - combines `sz_fill_random` & `sz_lookup` to fill buffers with random UTF-8 characters.
 */
#ifndef STRINGZILLA_HASH_H_
#define STRINGZILLA_HASH_H_

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief  Computes the 64-bit check-sum of bytes in a string.
 *          Similar to `std::ranges::accumulate`.
 *
 *  @param[in] text String to aggregate.
 *  @param[in] length Number of bytes in the text.
 *  @return 64-bit unsigned value.
 *
 *  Example usage:
 *
 *  @code{.c}
 *      #include <stringzilla/hash.h>
 *      int main() {
 *          return sz_bytesum("hi", 2) == 209 ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note   Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa     sz_bytesum_serial, sz_bytesum_haswell, sz_bytesum_skylake, sz_bytesum_ice, sz_bytesum_neon
 */
SZ_DYNAMIC sz_u64_t sz_bytesum(sz_cptr_t text, sz_size_t length);

/**
 *  @brief  Computes the 64-bit unsigned hash of a string similar to @b `std::hash` in C++.
 *          It's not cryptographically secure, but it's fast and provides a good distribution.
 *          It passes the SMHasher suite by Austin Appleby with no collisions, even with `--extra` flag.
 *  @see    HASH.md for a detailed explanation of the algorithm.
 *
 *  @param[in] text String to hash.
 *  @param[in] length Number of bytes in the text.
 *  @param[in] seed 64-bit unsigned seed for the hash.
 *  @return 64-bit hash value.
 *
 *  Example usage:
 *
 *  @code{.c}
 *      #include <stringzilla/hash.h>
 *      int main() {
 *          return sz_hash("hello", 5, 0) != sz_hash("world", 5, 0) ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note   Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa     sz_hash_serial, sz_hash_haswell, sz_hash_skylake, sz_hash_ice, sz_hash_neon, sz_hash_sve
 *
 *  @note   The algorithm must provide the same output on all platforms in both single-shot and incremental modes.
 *  @sa     sz_hash_state_init, sz_hash_state_update, sz_hash_state_digest
 */
SZ_DYNAMIC sz_u64_t sz_hash(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/**
 *  @brief  A Pseudorandom Number Generator (PRNG), inspired the AES-CTR-128 algorithm,
 *          but using only one round of AES mixing as opposed to "NIST SP 800-90A".
 *
 *  CTR_DRBG (CounTeR mode Deterministic Random Bit Generator) appears secure and indistinguishable from a
 *  true random source when AES is used as the underlying block cipher and 112 bits are taken from this PRNG.
 *  When AES is used as the underlying block cipher and 128 bits are taken from each instantiation,
 *  the required security level is delivered with the caveat that a 128-bit cipher's output in
 *  counter mode can be distinguished from a true RNG.
 *
 *  In this case, it doesn't apply, as we only use one round of AES mixing. We also don't expose a separate "key",
 *  only a "nonce", to keep the API simple, but we mix it with 512 bits of Pi constants to increase randomness.
 *
 *  @param[out] text Output string buffer to be populated.
 *  @param[in] length Number of bytes in the string.
 *  @param[in] nonce "Number used ONCE" to ensure uniqueness of produced blocks.
 *
 *  Example usage:
 *
 *  @code{.c}
 *      #include <stringzilla/hash.h>
 *      int main() {
 *          char first_buffer[5], second_buffer[5];
 *          sz_fill_random(first_buffer, 5, 0);
 *          sz_fill_random(second_buffer, 5, 0); // ? Same nonce must produce the same output
 *          return sz_bytesum(first_buffer, 5) == sz_bytesum(second_buffer, 5) ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note   Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa     sz_fill_random_serial, sz_fill_random_haswell, sz_fill_random_skylake, sz_fill_random_ice,
 *          sz_fill_random_neon, sz_fill_random_sve
 */
SZ_DYNAMIC void sz_fill_random(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/**
 *  @brief  The state for incremental construction of a hash.
 *  @see    sz_hash_state_init, sz_hash_state_update, sz_hash_state_digest.
 */
typedef struct sz_hash_state_t {
    sz_u512_vec_t aes;
    sz_u512_vec_t sum;
    sz_u512_vec_t ins;
    sz_u128_vec_t key;
    sz_size_t ins_length;
} sz_hash_state_t;

typedef struct sz_hash_minimal_t_ {
    sz_u128_vec_t aes;
    sz_u128_vec_t sum;
    sz_u128_vec_t key;
} sz_hash_minimal_t_;

/**
 *  @brief  Initializes the state for incremental construction of a hash.
 *
 *  @param[out] state The state to initialize.
 *  @param[in] seed The 64-bit unsigned seed for the hash.
 */
SZ_DYNAMIC void sz_hash_state_init(sz_hash_state_t *state, sz_u64_t seed);

/**
 *  @brief  Updates the state with new data.
 *
 *  @param[inout] state The state to stream.
 *  @param[in] text The new data to include in the hash.
 *  @param[in] length The number of bytes in the new data.
 */
SZ_DYNAMIC void sz_hash_state_update(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/**
 *  @brief  Finalizes the immutable state and returns the hash.
 *
 *  @param[in] state The state to fold.
 *  @return The 64-bit hash value.
 */
SZ_DYNAMIC sz_u64_t sz_hash_state_digest(sz_hash_state_t const *state);

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_serial(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash */
SZ_PUBLIC sz_u64_t sz_hash_serial(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/** @copydoc sz_fill_random */
SZ_PUBLIC void sz_fill_random_serial(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
SZ_PUBLIC void sz_hash_state_init_serial(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_update */
SZ_PUBLIC void sz_hash_state_update_serial(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_digest */
SZ_PUBLIC sz_u64_t sz_hash_state_digest_serial(sz_hash_state_t const *state);

#if SZ_USE_HASWELL

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_haswell(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash */
SZ_PUBLIC sz_u64_t sz_hash_haswell(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/** @copydoc sz_fill_random */
SZ_PUBLIC void sz_fill_random_haswell(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
SZ_PUBLIC void sz_hash_state_init_haswell(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_update */
SZ_PUBLIC void sz_hash_state_update_haswell(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_digest */
SZ_PUBLIC sz_u64_t sz_hash_state_digest_haswell(sz_hash_state_t const *state);

#endif

#if SZ_USE_SKYLAKE

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_skylake(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash */
SZ_PUBLIC sz_u64_t sz_hash_skylake(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/** @copydoc sz_fill_random */
SZ_PUBLIC void sz_fill_random_skylake(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
SZ_PUBLIC void sz_hash_state_init_skylake(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_update */
SZ_PUBLIC void sz_hash_state_update_skylake(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_digest */
SZ_PUBLIC sz_u64_t sz_hash_state_digest_skylake(sz_hash_state_t const *state);

#endif

#if SZ_USE_ICE

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_ice(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash */
SZ_PUBLIC sz_u64_t sz_hash_ice(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/** @copydoc sz_fill_random */
SZ_PUBLIC void sz_fill_random_ice(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
SZ_PUBLIC void sz_hash_state_init_ice(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_update */
SZ_PUBLIC void sz_hash_state_update_ice(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_digest */
SZ_PUBLIC sz_u64_t sz_hash_state_digest_ice(sz_hash_state_t const *state);

#endif

#if SZ_USE_NEON

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_neon(sz_cptr_t text, sz_size_t length);

#endif

#if SZ_USE_NEON_AES

/** @copydoc sz_hash */
SZ_PUBLIC sz_u64_t sz_hash_neon(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/** @copydoc sz_fill_random */
SZ_PUBLIC void sz_fill_random_neon(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
SZ_PUBLIC void sz_hash_state_init_neon(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_update */
SZ_PUBLIC void sz_hash_state_update_neon(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_digest */
SZ_PUBLIC sz_u64_t sz_hash_state_digest_neon(sz_hash_state_t const *state);

#endif

#if SZ_USE_SVE

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_sve(sz_cptr_t text, sz_size_t length);

#endif

#if SZ_USE_SVE2

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_sve2(sz_cptr_t text, sz_size_t length);

#endif

#if SZ_USE_SVE2_AES

/** @copydoc sz_hash */
SZ_PUBLIC sz_u64_t sz_hash_sve2(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/** @copydoc sz_fill_random */
SZ_PUBLIC void sz_fill_random_sve2(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
SZ_PUBLIC void sz_hash_state_init_sve2(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_update */
SZ_PUBLIC void sz_hash_state_update_sve2(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_digest */
SZ_PUBLIC sz_u64_t sz_hash_state_digest_sve2(sz_hash_state_t const *state);

#endif

#pragma endregion // Core API

#pragma region Helper Methods

/**
 *  @brief  Compares the state of two running hashes.
 *  @note   The current content of the `ins` buffer and its length is ignored.
 */
SZ_PUBLIC sz_bool_t sz_hash_state_equal(sz_hash_state_t const *lhs, sz_hash_state_t const *rhs) {
    int same_aes = //
        lhs->aes.u64s[0] == rhs->aes.u64s[0] && lhs->aes.u64s[1] == rhs->aes.u64s[1] &&
        lhs->aes.u64s[2] == rhs->aes.u64s[2] && lhs->aes.u64s[3] == rhs->aes.u64s[3];
    int same_sum = //
        lhs->sum.u64s[0] == rhs->sum.u64s[0] && lhs->sum.u64s[1] == rhs->sum.u64s[1] &&
        lhs->sum.u64s[2] == rhs->sum.u64s[2] && lhs->sum.u64s[3] == rhs->sum.u64s[3];
    int same_key = //
        lhs->key.u64s[0] == rhs->key.u64s[0] && lhs->key.u64s[1] == rhs->key.u64s[1];
    return same_aes && same_sum && same_key ? sz_true_k : sz_false_k;
}

#pragma endregion // Helper Methods

#pragma region Serial Implementation

SZ_PUBLIC sz_u64_t sz_bytesum_serial(sz_cptr_t text, sz_size_t length) {
    sz_u64_t bytesum = 0;
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *text_end = text_u8 + length;
    for (; text_u8 != text_end; ++text_u8) bytesum += *text_u8;
    return bytesum;
}

/**
 *  @brief  Emulates the behaviour of `_mm_aesenc_si128` for a single round.
 *          This function is used as a fallback when the hardware-accelerated version is not available.
 *  @return Result of `MixColumns(SubBytes(ShiftRows(state))) ^ round_key`.
 *  @see    Based on Jean-Philippe Aumasson's reference implementation: https://github.com/veorq/aesenc-noNI
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
    sz_u8_t t0 = state_2d[0][0];
    sz_u8_t u0 = state_2d[0][0] ^ state_2d[0][1] ^ state_2d[0][2] ^ state_2d[0][3];
    state_2d[0][0] ^= u0 ^ sz_gf2_double_(state_2d[0][0] ^ state_2d[0][1]);
    state_2d[0][1] ^= u0 ^ sz_gf2_double_(state_2d[0][1] ^ state_2d[0][2]);
    state_2d[0][2] ^= u0 ^ sz_gf2_double_(state_2d[0][2] ^ state_2d[0][3]);
    state_2d[0][3] ^= u0 ^ sz_gf2_double_(state_2d[0][3] ^ t0);

    // Row 1:
    sz_u8_t t1 = state_2d[1][0];
    sz_u8_t u1 = state_2d[1][0] ^ state_2d[1][1] ^ state_2d[1][2] ^ state_2d[1][3];
    state_2d[1][0] ^= u1 ^ sz_gf2_double_(state_2d[1][0] ^ state_2d[1][1]);
    state_2d[1][1] ^= u1 ^ sz_gf2_double_(state_2d[1][1] ^ state_2d[1][2]);
    state_2d[1][2] ^= u1 ^ sz_gf2_double_(state_2d[1][2] ^ state_2d[1][3]);
    state_2d[1][3] ^= u1 ^ sz_gf2_double_(state_2d[1][3] ^ t1);

    // Row 2:
    sz_u8_t t2 = state_2d[2][0];
    sz_u8_t u2 = state_2d[2][0] ^ state_2d[2][1] ^ state_2d[2][2] ^ state_2d[2][3];
    state_2d[2][0] ^= u2 ^ sz_gf2_double_(state_2d[2][0] ^ state_2d[2][1]);
    state_2d[2][1] ^= u2 ^ sz_gf2_double_(state_2d[2][1] ^ state_2d[2][2]);
    state_2d[2][2] ^= u2 ^ sz_gf2_double_(state_2d[2][2] ^ state_2d[2][3]);
    state_2d[2][3] ^= u2 ^ sz_gf2_double_(state_2d[2][3] ^ t2);

    // Row 3:
    sz_u8_t t3 = state_2d[3][0];
    sz_u8_t u3 = state_2d[3][0] ^ state_2d[3][1] ^ state_2d[3][2] ^ state_2d[3][3];
    state_2d[3][0] ^= u3 ^ sz_gf2_double_(state_2d[3][0] ^ state_2d[3][1]);
    state_2d[3][1] ^= u3 ^ sz_gf2_double_(state_2d[3][1] ^ state_2d[3][2]);
    state_2d[3][2] ^= u3 ^ sz_gf2_double_(state_2d[3][2] ^ state_2d[3][3]);
    state_2d[3][3] ^= u3 ^ sz_gf2_double_(state_2d[3][3] ^ t3);
#undef sz_gf2_double_

    // Export `XOR`-ing with the round key
    sz_u128_vec_t result = *(sz_u128_vec_t *)state_2d;
    result.u64s[0] ^= round_key_vec.u64s[0];
    result.u64s[1] ^= round_key_vec.u64s[1];
    return result;
}

SZ_INTERNAL sz_u128_vec_t sz_emulate_shuffle_epi8_serial_(sz_u128_vec_t state_vec, sz_u8_t const order[16]) {
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
 *  @brief  Provides 1024 bits worth of precomputed Pi constants for the hash.
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
 *  @see    Bailey-Borwein-Plouffe @b (BBP) formula explanation by Mosè Giordano:
 *          https://giordano.github.io/blog/2017-11-21-hexadecimal-pi/
 *
 */
SZ_INTERNAL sz_u64_t const *sz_hash_pi_constants_(void) {
    static SZ_ALIGN64 sz_u64_t const pi[16] = {
        0x243F6A8885A308D3ull, 0x13198A2E03707344ull, 0xA4093822299F31D0ull, 0x082EFA98EC4E6C89ull,
        0x452821E638D01377ull, 0xBE5466CF34E90C6Cull, 0xC0AC29B7C97C50DDull, 0x3F84D5B5B5470917ull,
        0x9216D5D98979FB1Bull, 0xD1310BA698DFB5ACull, 0x2FFD72DBD01ADFB7ull, 0xB8E1AFED6A267E96ull,
        0xBA7C9045F12C7F99ull, 0x24A19947B3916CF7ull, 0x0801F2E2858EFC16ull, 0x636920D871574E69ull,
    };
    return &pi[0];
}

/**
 *  @brief  Provides a shuffle mask for the additive part, identical to "aHash" in a single lane.
 *  @return Pointer aligned to 64 bytes on SIMD-capable platforms.
 */
SZ_INTERNAL sz_u8_t const *sz_hash_u8x16x4_shuffle_(void) {
    static SZ_ALIGN64 sz_u8_t const shuffle[64] = {
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

SZ_INTERNAL void sz_hash_minimal_update_serial_(sz_hash_minimal_t_ *state, sz_u128_vec_t block) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    state->aes = sz_emulate_aesenc_si128_serial_(state->aes, block);
    state->sum = sz_emulate_shuffle_epi8_serial_(state->sum, shuffle);
    state->sum.u64s[0] += block.u64s[0], state->sum.u64s[1] += block.u64s[1];
}

SZ_INTERNAL sz_u64_t sz_hash_minimal_finalize_serial_(sz_hash_minimal_t_ const *state, sz_size_t length) {
    // Mix the length into the key
    sz_u128_vec_t key_with_length = state->key;
    key_with_length.u64s[0] += length;
    // Combine the "sum" and the "AES" blocks
    sz_u128_vec_t mixed = sz_emulate_aesenc_si128_serial_(state->sum, state->aes);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    sz_u128_vec_t mixed_in_register =
        sz_emulate_aesenc_si128_serial_(sz_emulate_aesenc_si128_serial_(mixed, key_with_length), mixed);
    // Extract the low 64 bits
    return mixed_in_register.u64s[0];
}

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
    state->key.u64s[0] = seed;
    state->key.u64s[1] = seed;

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    for (int i = 0; i < 8; ++i) state->aes.u64s[i] = seed ^ pi[i];
    for (int i = 0; i < 8; ++i) state->sum.u64s[i] = seed ^ pi[i + 8];

    // The inputs are zeroed out at the beginning
    for (int i = 0; i < 8; ++i) state->ins.u64s[i] = 0;
    state->ins_length = 0;
}

SZ_INTERNAL void sz_hash_state_update_serial_(sz_hash_state_t *state) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();

    // To reuse the snippets above, let's cast to our familiar 128-bit vectors
    sz_u128_vec_t *aes_vecs = (sz_u128_vec_t *)&state->aes.u64s[0];
    sz_u128_vec_t *sum_vecs = (sz_u128_vec_t *)&state->sum.u64s[0];
    sz_u128_vec_t *ins_vecs = (sz_u128_vec_t *)&state->ins.u64s[0];

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

SZ_INTERNAL sz_u64_t sz_hash_state_finalize_serial_(sz_hash_state_t const *state) {

    // Mix the length into the key
    sz_u128_vec_t key_with_length = state->key;
    key_with_length.u64s[0] += state->ins_length;

    // To reuse the snippets above, let's cast to our familiar 128-bit vectors
    sz_u128_vec_t *aes_vecs = (sz_u128_vec_t *)&state->aes.u64s[0];
    sz_u128_vec_t *sum_vecs = (sz_u128_vec_t *)&state->sum.u64s[0];

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
    sz_u128_vec_t mixed_in_register =
        sz_emulate_aesenc_si128_serial_(sz_emulate_aesenc_si128_serial_(mixed, key_with_length), mixed);

    // Extract the low 64 bits
    return mixed_in_register.u64s[0];
}

SZ_PUBLIC sz_u64_t sz_hash_serial(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {
    if (length <= 16) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data_vec;
        data_vec.u64s[0] = data_vec.u64s[1] = 0;
        for (sz_size_t i = 0; i < length; ++i) data_vec.u8s[i] = start[i];
        sz_hash_minimal_update_serial_(&state, data_vec);
        return sz_hash_minimal_finalize_serial_(&state, length);
    }
    else if (length <= 32) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec;
#if SZ_USE_MISALIGNED_LOADS
        data0_vec.u64s[0] = *(sz_u64_t const *)(start);
        data0_vec.u64s[1] = *(sz_u64_t const *)(start + 8);
        data1_vec.u64s[0] = *(sz_u64_t const *)(start + length - 16);
        data1_vec.u64s[1] = *(sz_u64_t const *)(start + length - 8);
#else
        for (sz_size_t i = 0; i < 16; ++i) data0_vec.u8s[i] = start[i];
        for (sz_size_t i = 0; i < 16; ++i) data1_vec.u8s[i] = start[length - 16 + i];
#endif
        // Let's shift the data within the register to de-interleave the bytes.
        sz_hash_shift_in_register_serial_(&data1_vec, (int)(32 - length));
        sz_hash_minimal_update_serial_(&state, data0_vec);
        sz_hash_minimal_update_serial_(&state, data1_vec);
        return sz_hash_minimal_finalize_serial_(&state, length);
    }
    else if (length <= 48) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
#if SZ_USE_MISALIGNED_LOADS
        data0_vec.u64s[0] = *(sz_u64_t const *)(start);
        data0_vec.u64s[1] = *(sz_u64_t const *)(start + 8);
        data1_vec.u64s[0] = *(sz_u64_t const *)(start + 16);
        data1_vec.u64s[1] = *(sz_u64_t const *)(start + 24);
        data2_vec.u64s[0] = *(sz_u64_t const *)(start + length - 16);
        data2_vec.u64s[1] = *(sz_u64_t const *)(start + length - 8);
#else
        for (sz_size_t i = 0; i < 16; ++i) data0_vec.u8s[i] = start[i];
        for (sz_size_t i = 0; i < 16; ++i) data1_vec.u8s[i] = start[16 + i];
        for (sz_size_t i = 0; i < 16; ++i) data2_vec.u8s[i] = start[length - 16 + i];
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
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
#if SZ_USE_MISALIGNED_LOADS
        data0_vec.u64s[0] = *(sz_u64_t const *)(start);
        data0_vec.u64s[1] = *(sz_u64_t const *)(start + 8);
        data1_vec.u64s[0] = *(sz_u64_t const *)(start + 16);
        data1_vec.u64s[1] = *(sz_u64_t const *)(start + 24);
        data2_vec.u64s[0] = *(sz_u64_t const *)(start + 32);
        data2_vec.u64s[1] = *(sz_u64_t const *)(start + 40);
        data3_vec.u64s[0] = *(sz_u64_t const *)(start + length - 16);
        data3_vec.u64s[1] = *(sz_u64_t const *)(start + length - 8);
#else
        for (sz_size_t i = 0; i < 16; ++i) data0_vec.u8s[i] = start[i];
        for (sz_size_t i = 0; i < 16; ++i) data1_vec.u8s[i] = start[16 + i];
        for (sz_size_t i = 0; i < 16; ++i) data2_vec.u8s[i] = start[32 + i];
        for (sz_size_t i = 0; i < 16; ++i) data3_vec.u8s[i] = start[length - 16 + i];
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
        // Use a larger state to handle the main loop and add different offsets
        // to different lanes of the register
        SZ_ALIGN64 sz_hash_state_t state;
        sz_hash_state_init_serial(&state, seed);

#if SZ_USE_MISALIGNED_LOADS
        for (; state.ins_length + 64 <= length; state.ins_length += 64) {
            state.ins.u64s[0] = *(sz_u64_t const *)(start + state.ins_length);
            state.ins.u64s[1] = *(sz_u64_t const *)(start + state.ins_length + 8);
            state.ins.u64s[2] = *(sz_u64_t const *)(start + state.ins_length + 16);
            state.ins.u64s[3] = *(sz_u64_t const *)(start + state.ins_length + 24);
            state.ins.u64s[4] = *(sz_u64_t const *)(start + state.ins_length + 32);
            state.ins.u64s[5] = *(sz_u64_t const *)(start + state.ins_length + 40);
            state.ins.u64s[6] = *(sz_u64_t const *)(start + state.ins_length + 48);
            state.ins.u64s[7] = *(sz_u64_t const *)(start + state.ins_length + 56);
            sz_hash_state_update_serial_(&state);
        }
#else
        for (; state.ins_length + 64 <= length; state.ins_length += 64) {
            for (sz_size_t i = 0; i < 64; ++i) state.ins.u8s[i] = start[state.ins_length + i];
            sz_hash_state_update_serial_(&state);
        }
#endif

        if (state.ins_length < length) {
            for (sz_size_t i = 0; i != 8; ++i) state.ins.u64s[i] = 0;
            for (sz_size_t i = 0; state.ins_length < length; ++i, ++state.ins_length)
                state.ins.u8s[i] = start[state.ins_length];
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
        while (to_copy--) state->ins.u8s[progress_in_block++] = *text++;
        // If we've reached the end of the buffer, update the state
        if (will_fill_block) {
            sz_hash_state_update_serial_(state);
            // Reset to zeros now, so we don't have to overwrite an immutable buffer in the folding state
            for (int i = 0; i < 8; ++i) state->ins.u64s[i] = 0;
        }
    }
}

SZ_PUBLIC sz_u64_t sz_hash_state_digest_serial(sz_hash_state_t const *state) {
    sz_size_t length = state->ins_length;
    if (length >= 64) return sz_hash_state_finalize_serial_(state);

    // Switch back to a smaller "minimal" state for small inputs
    sz_hash_minimal_t_ minimal_state;
    minimal_state.key = state->key;
    minimal_state.aes = *(sz_u128_vec_t const *)&state->aes.u64s[0];
    minimal_state.sum = *(sz_u128_vec_t const *)&state->sum.u64s[0];

    // The logic is different depending on the length of the input
    sz_u128_vec_t const *ins_vecs = (sz_u128_vec_t const *)&state->ins.u64s[0];
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
        for (int i = 0; i < 16 && length; ++i, --length) *text++ = generated_vec.u8s[i];
    }
}

#pragma endregion // Serial Implementation

/*  AVX2 implementation of the string search algorithms for Haswell processors and newer.
 *  Very minimalistic (compared to AVX-512), but still faster than the serial implementation.
 */
#pragma region Haswell Implementation
#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,aes"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "aes")
#endif

SZ_PUBLIC sz_u64_t sz_bytesum_haswell(sz_cptr_t text, sz_size_t length) {
    // The naive implementation of this function is very simple.
    // It assumes the CPU is great at handling unaligned "loads".
    //
    // A typical AWS Skylake instance can have 32 KB x 2 blocks of L1 data cache per core,
    // 1 MB x 2 blocks of L2 cache per core, and one shared L3 cache buffer.
    // For now, let's avoid the cases beyond the L2 size.
    int is_huge = length > 1ull * 1024ull * 1024ull;

    // When the buffer is small, there isn't much to innovate.
    if (length <= 32) { return sz_bytesum_serial(text, length); }
    else if (!is_huge) {
        sz_u256_vec_t text_vec, sums_vec;
        sums_vec.ymm = _mm256_setzero_si256();
        for (; length >= 32; text += 32, length -= 32) {
            text_vec.ymm = _mm256_lddqu_si256((__m256i const *)text);
            sums_vec.ymm = _mm256_add_epi64(sums_vec.ymm, _mm256_sad_epu8(text_vec.ymm, _mm256_setzero_si256()));
        }
        // We can also avoid the final serial loop by fetching 32 bytes from end, in reverse direction,
        // and shifting the data within the register to zero-out the duplicate bytes.

        // Accumulating 256 bits is harder, as we need to extract the 128-bit sums first.
        __m128i low_xmm = _mm256_castsi256_si128(sums_vec.ymm);
        __m128i high_xmm = _mm256_extracti128_si256(sums_vec.ymm, 1);
        __m128i sums_xmm = _mm_add_epi64(low_xmm, high_xmm);
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_xmm);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_xmm, 1);
        sz_u64_t result = low + high;
        if (length) result += sz_bytesum_serial(text, length);
        return result;
    }
    // For gigantic buffers, exceeding typical L1 cache sizes, there are other tricks we can use.
    // Most notably, we can avoid populating the cache with the entire buffer, and instead traverse it in 2 directions.
    else {
        sz_size_t head_length = (32 - ((sz_size_t)text % 32)) % 32; // 31 or less.
        sz_size_t tail_length = (sz_size_t)(text + length) % 32;    // 31 or less.
        sz_size_t body_length = length - head_length - tail_length; // Multiple of 32.
        sz_u64_t result = 0;

        // Handle the tail before we start updating the `text` pointer
        while (tail_length) result += text[length - (tail_length--)];
        // Handle the head
        while (head_length--) result += *text++;

        sz_u256_vec_t text_vec, sums_vec;
        sums_vec.ymm = _mm256_setzero_si256();
        // Fill the aligned body of the buffer.
        if (!is_huge) {
            for (; body_length >= 32; text += 32, body_length -= 32) {
                text_vec.ymm = _mm256_stream_load_si256((__m256i const *)text);
                sums_vec.ymm = _mm256_add_epi64(sums_vec.ymm, _mm256_sad_epu8(text_vec.ymm, _mm256_setzero_si256()));
            }
        }
        // When the buffer is huge, we can traverse it in 2 directions.
        else {
            sz_u256_vec_t text_reversed_vec, sums_reversed_vec;
            sums_reversed_vec.ymm = _mm256_setzero_si256();
            for (; body_length >= 64; text += 32, body_length -= 64) {
                text_vec.ymm = _mm256_stream_load_si256((__m256i *)(text));
                sums_vec.ymm = _mm256_add_epi64(sums_vec.ymm, _mm256_sad_epu8(text_vec.ymm, _mm256_setzero_si256()));
                text_reversed_vec.ymm = _mm256_stream_load_si256((__m256i *)(text + body_length - 32));
                sums_reversed_vec.ymm = _mm256_add_epi64(
                    sums_reversed_vec.ymm, _mm256_sad_epu8(text_reversed_vec.ymm, _mm256_setzero_si256()));
            }
            if (body_length >= 32) {
                sz_assert_(body_length == 32);
                text_vec.ymm = _mm256_stream_load_si256((__m256i *)(text));
                sums_vec.ymm = _mm256_add_epi64(sums_vec.ymm, _mm256_sad_epu8(text_vec.ymm, _mm256_setzero_si256()));
                text += 32;
            }
            sums_vec.ymm = _mm256_add_epi64(sums_vec.ymm, sums_reversed_vec.ymm);
        }

        // Accumulating 256 bits is harder, as we need to extract the 128-bit sums first.
        __m128i low_xmm = _mm256_castsi256_si128(sums_vec.ymm);
        __m128i high_xmm = _mm256_extracti128_si256(sums_vec.ymm, 1);
        __m128i sums_xmm = _mm_add_epi64(low_xmm, high_xmm);
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_xmm);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_xmm, 1);
        result += low + high;
        return result;
    }
}

SZ_INTERNAL void sz_hash_minimal_init_haswell_(sz_hash_minimal_t_ *state, sz_u64_t seed) {

    // The key is made from the seed and half of it will be mixed with the length in the end
    __m128i seed_vec = _mm_set1_epi64x(seed);
    state->key.xmm = seed_vec;

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    __m128i const pi0 = _mm_load_si128((__m128i const *)(pi));
    __m128i const pi1 = _mm_load_si128((__m128i const *)(pi + 8));
    __m128i k1 = _mm_xor_si128(seed_vec, pi0);
    __m128i k2 = _mm_xor_si128(seed_vec, pi1);

    // The first 128 bits of the "sum" and "AES" blocks are the same for the "minimal" and full state
    state->aes.xmm = k1;
    state->sum.xmm = k2;
}

SZ_INTERNAL sz_u64_t sz_hash_minimal_finalize_haswell_(sz_hash_minimal_t_ const *state, sz_size_t length) {
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

SZ_INTERNAL void sz_hash_minimal_update_haswell_(sz_hash_minimal_t_ *state, __m128i block) {
    // ? In this kernel, assuming it's only used internally on properly aligned `state`,
    // ? we don't need `_mm_storeu_si128` stores to update the state.
    __m128i const shuffle_mask = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
    state->aes.xmm = _mm_aesenc_si128(_mm_lddqu_si128(&state->aes.xmm), block);
    state->sum.xmm = _mm_add_epi64(_mm_shuffle_epi8(_mm_lddqu_si128(&state->sum.xmm), shuffle_mask), block);
}

SZ_PUBLIC void sz_hash_state_init_haswell(sz_hash_state_t *state, sz_u64_t seed) {
    // The key is made from the seed and half of it will be mixed with the length in the end
    __m128i seed_vec = _mm_set1_epi64x(seed);

    // ! In this kernel, assuming it may be called on arbitrarily misaligned `state`,
    // ! we must use `_mm_storeu_si128` stores to update the state. Moreover, accessing `state.xmms[i]`
    // ! fools the compiler into preferring aligned operations over out misaligned ones.
    _mm_storeu_si128((__m128i *)&state->key.u8s[0], seed_vec);

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    for (int i = 0; i < 4; ++i)
        _mm_storeu_si128((__m128i *)&state->aes.u8s[i * sizeof(__m128i)],
                         _mm_xor_si128(seed_vec, _mm_lddqu_si128((__m128i const *)(pi + i * 2))));
    for (int i = 0; i < 4; ++i)
        _mm_storeu_si128((__m128i *)&state->sum.u8s[i * sizeof(__m128i)],
                         _mm_xor_si128(seed_vec, _mm_lddqu_si128((__m128i const *)(pi + i * 2 + 8))));

    // The inputs are zeroed out at the beginning
    _mm_storeu_si128((__m128i *)&state->ins.u8s[0 * sizeof(__m128i)], _mm_setzero_si128());
    _mm_storeu_si128((__m128i *)&state->ins.u8s[1 * sizeof(__m128i)], _mm_setzero_si128());
    _mm_storeu_si128((__m128i *)&state->ins.u8s[2 * sizeof(__m128i)], _mm_setzero_si128());
    _mm_storeu_si128((__m128i *)&state->ins.u8s[3 * sizeof(__m128i)], _mm_setzero_si128());
    state->ins_length = 0;
}

SZ_INTERNAL void sz_hash_state_update_haswell_(sz_hash_state_t *state) {
    __m128i const shuffle_mask = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
    _mm_storeu_si128( //
        (__m128i *)&state->aes.u8s[0 * sizeof(__m128i)],
        _mm_aesenc_si128(_mm_lddqu_si128((__m128i *)&state->aes.u8s[0 * sizeof(__m128i)]),
                         _mm_lddqu_si128((__m128i *)&state->ins.u8s[0 * sizeof(__m128i)])));
    _mm_storeu_si128( //
        (__m128i *)&state->sum.u8s[0 * sizeof(__m128i)],
        _mm_add_epi64(_mm_shuffle_epi8(_mm_lddqu_si128((__m128i *)&state->sum.u8s[0 * sizeof(__m128i)]), shuffle_mask),
                      _mm_lddqu_si128((__m128i *)&state->ins.u8s[0 * sizeof(__m128i)])));
    _mm_storeu_si128( //
        (__m128i *)&state->aes.u8s[1 * sizeof(__m128i)],
        _mm_aesenc_si128(_mm_lddqu_si128((__m128i *)&state->aes.u8s[1 * sizeof(__m128i)]),
                         _mm_lddqu_si128((__m128i *)&state->ins.u8s[1 * sizeof(__m128i)])));
    _mm_storeu_si128( //
        (__m128i *)&state->sum.u8s[1 * sizeof(__m128i)],
        _mm_add_epi64(_mm_shuffle_epi8(_mm_lddqu_si128((__m128i *)&state->sum.u8s[1 * sizeof(__m128i)]), shuffle_mask),
                      _mm_lddqu_si128((__m128i *)&state->ins.u8s[1 * sizeof(__m128i)])));
    _mm_storeu_si128( //
        (__m128i *)&state->aes.u8s[2 * sizeof(__m128i)],
        _mm_aesenc_si128(_mm_lddqu_si128((__m128i *)&state->aes.u8s[2 * sizeof(__m128i)]),
                         _mm_lddqu_si128((__m128i *)&state->ins.u8s[2 * sizeof(__m128i)])));
    _mm_storeu_si128( //
        (__m128i *)&state->sum.u8s[2 * sizeof(__m128i)],
        _mm_add_epi64(_mm_shuffle_epi8(_mm_lddqu_si128((__m128i *)&state->sum.u8s[2 * sizeof(__m128i)]), shuffle_mask),
                      _mm_lddqu_si128((__m128i *)&state->ins.u8s[2 * sizeof(__m128i)])));
    _mm_storeu_si128( //
        (__m128i *)&state->aes.u8s[3 * sizeof(__m128i)],
        _mm_aesenc_si128(_mm_lddqu_si128((__m128i *)&state->aes.u8s[3 * sizeof(__m128i)]),
                         _mm_lddqu_si128((__m128i *)&state->ins.u8s[3 * sizeof(__m128i)])));
    _mm_storeu_si128( //
        (__m128i *)&state->sum.u8s[3 * sizeof(__m128i)],
        _mm_add_epi64(_mm_shuffle_epi8(_mm_lddqu_si128((__m128i *)&state->sum.u8s[3 * sizeof(__m128i)]), shuffle_mask),
                      _mm_lddqu_si128((__m128i *)&state->ins.u8s[3 * sizeof(__m128i)])));
}

SZ_INTERNAL sz_u64_t sz_hash_state_finalize_haswell_(sz_hash_state_t const *state) {
    // Mix the length into the key
    __m128i key_with_length =
        _mm_add_epi64(_mm_lddqu_si128((__m128i *)&state->key.u8s[0]), _mm_set_epi64x(0, state->ins_length));
    // Combine the "sum" and the "AES" blocks
    __m128i mixed0 = _mm_aesenc_si128(_mm_lddqu_si128((__m128i *)&state->sum.u8s[0 * sizeof(__m128i)]),
                                      _mm_lddqu_si128((__m128i *)&state->aes.u8s[0 * sizeof(__m128i)]));
    __m128i mixed1 = _mm_aesenc_si128(_mm_lddqu_si128((__m128i *)&state->sum.u8s[1 * sizeof(__m128i)]),
                                      _mm_lddqu_si128((__m128i *)&state->aes.u8s[1 * sizeof(__m128i)]));
    __m128i mixed2 = _mm_aesenc_si128(_mm_lddqu_si128((__m128i *)&state->sum.u8s[2 * sizeof(__m128i)]),
                                      _mm_lddqu_si128((__m128i *)&state->aes.u8s[2 * sizeof(__m128i)]));
    __m128i mixed3 = _mm_aesenc_si128(_mm_lddqu_si128((__m128i *)&state->sum.u8s[3 * sizeof(__m128i)]),
                                      _mm_lddqu_si128((__m128i *)&state->aes.u8s[3 * sizeof(__m128i)]));
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

SZ_PUBLIC sz_u64_t sz_hash_haswell(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {

    if (length <= 16) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_haswell_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data_vec;
        data_vec.xmm = _mm_setzero_si128();
        for (sz_size_t i = 0; i < length; ++i) data_vec.u8s[i] = start[i];
        sz_hash_minimal_update_haswell_(&state, data_vec.xmm);
        return sz_hash_minimal_finalize_haswell_(&state, length);
    }
    else if (length <= 32) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_haswell_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + length - 16));
        // Let's shift the data within the register to de-interleave the bytes.
        sz_hash_shift_in_register_serial_(&data1_vec, (int)(32 - length));
        sz_hash_minimal_update_haswell_(&state, data0_vec.xmm);
        sz_hash_minimal_update_haswell_(&state, data1_vec.xmm);
        return sz_hash_minimal_finalize_haswell_(&state, length);
    }
    else if (length <= 48) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_haswell_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 0));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 16));
        data2_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + length - 16));
        // Let's shift the data within the register to de-interleave the bytes.
        sz_hash_shift_in_register_serial_(&data2_vec, (int)(48 - length));
        sz_hash_minimal_update_haswell_(&state, data0_vec.xmm);
        sz_hash_minimal_update_haswell_(&state, data1_vec.xmm);
        sz_hash_minimal_update_haswell_(&state, data2_vec.xmm);
        return sz_hash_minimal_finalize_haswell_(&state, length);
    }
    else if (length <= 64) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_haswell_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 0));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 16));
        data2_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 32));
        data3_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + length - 16));
        // Let's shift the data within the register to de-interleave the bytes.
        sz_hash_shift_in_register_serial_(&data3_vec, (int)(64 - length));
        sz_hash_minimal_update_haswell_(&state, data0_vec.xmm);
        sz_hash_minimal_update_haswell_(&state, data1_vec.xmm);
        sz_hash_minimal_update_haswell_(&state, data2_vec.xmm);
        sz_hash_minimal_update_haswell_(&state, data3_vec.xmm);
        return sz_hash_minimal_finalize_haswell_(&state, length);
    }
    else {
        // Use a larger state to handle the main loop and add different offsets
        // to different lanes of the register
        SZ_ALIGN64 sz_hash_state_t state;
        sz_hash_state_init_haswell(&state, seed);
        for (; state.ins_length + 64 <= length; state.ins_length += 64) {
            state.ins.xmms[0] = _mm_lddqu_si128((__m128i const *)(start + state.ins_length + 0));
            state.ins.xmms[1] = _mm_lddqu_si128((__m128i const *)(start + state.ins_length + 16));
            state.ins.xmms[2] = _mm_lddqu_si128((__m128i const *)(start + state.ins_length + 32));
            state.ins.xmms[3] = _mm_lddqu_si128((__m128i const *)(start + state.ins_length + 48));
            sz_hash_state_update_haswell_(&state);
        }
        // Handle the tail, resetting the registers to zero first
        if (state.ins_length < length) {
            state.ins.xmms[0] = _mm_setzero_si128();
            state.ins.xmms[1] = _mm_setzero_si128();
            state.ins.xmms[2] = _mm_setzero_si128();
            state.ins.xmms[3] = _mm_setzero_si128();
            for (sz_size_t i = 0; state.ins_length < length; ++i, ++state.ins_length)
                state.ins.u8s[i] = start[state.ins_length];
            sz_hash_state_update_haswell_(&state);
            state.ins_length = length;
        }
        return sz_hash_state_finalize_haswell_(&state);
    }
}

SZ_PUBLIC void sz_hash_state_update_haswell(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    while (length) {
        // Append to the internal buffer until it's full
        if (state->ins_length % 64 == 0 && length >= 64) {
            _mm_storeu_si128((__m128i *)&state->ins.u8s[0 * sizeof(__m128i)],
                             _mm_lddqu_si128((__m128i const *)(text + 0)));
            _mm_storeu_si128((__m128i *)&state->ins.u8s[1 * sizeof(__m128i)],
                             _mm_lddqu_si128((__m128i const *)(text + 16)));
            _mm_storeu_si128((__m128i *)&state->ins.u8s[2 * sizeof(__m128i)],
                             _mm_lddqu_si128((__m128i const *)(text + 32)));
            _mm_storeu_si128((__m128i *)&state->ins.u8s[3 * sizeof(__m128i)],
                             _mm_lddqu_si128((__m128i const *)(text + 48)));
            sz_hash_state_update_haswell_(state);
            state->ins_length += 64;
            text += 64;
            length -= 64;
        }
        // If vectorization isn't that trivial - fall back to the serial implementation
        else {
            sz_size_t progress_in_block = state->ins_length % 64;
            sz_size_t to_copy = sz_min_of_two(length, 64 - progress_in_block);
            int const will_fill_block = progress_in_block + to_copy == 64;
            // Update the metadata before we modify the `to_copy` variable
            state->ins_length += to_copy;
            length -= to_copy;
            // Append to the internal buffer until it's full
            while (to_copy--) state->ins.u8s[progress_in_block++] = *text++;
            // If we've reached the end of the buffer, update the state
            if (will_fill_block) {
                sz_hash_state_update_haswell_(state);
                // Reset to zeros now, so we don't have to overwrite an immutable buffer in the folding state
                for (int i = 0; i < 4; ++i)
                    _mm_storeu_si128((__m128i *)&state->ins.u8s[i * sizeof(__m128i)], _mm_setzero_si128());
            }
        }
    }
}

SZ_PUBLIC sz_u64_t sz_hash_state_digest_haswell(sz_hash_state_t const *state) {
    sz_size_t length = state->ins_length;
    if (length >= 64) return sz_hash_state_finalize_haswell_(state);

    // Switch back to a smaller "minimal" state for small inputs
    SZ_ALIGN64 sz_hash_minimal_t_ minimal_state;
    minimal_state.key.xmm = _mm_lddqu_si128(&state->key.xmm);
    minimal_state.aes.xmm = _mm_lddqu_si128(&state->aes.xmms[0]);
    minimal_state.sum.xmm = _mm_lddqu_si128(&state->sum.xmms[0]);

    // The logic is different depending on the length of the input
    __m128i const *ins_xmms = (__m128i const *)&state->ins.xmms[0];
    if (length <= 16) {
        sz_hash_minimal_update_haswell_(&minimal_state, _mm_lddqu_si128(&ins_xmms[0]));
        return sz_hash_minimal_finalize_haswell_(&minimal_state, length);
    }
    else if (length <= 32) {
        sz_hash_minimal_update_haswell_(&minimal_state, _mm_lddqu_si128(&ins_xmms[0]));
        sz_hash_minimal_update_haswell_(&minimal_state, _mm_lddqu_si128(&ins_xmms[1]));
        return sz_hash_minimal_finalize_haswell_(&minimal_state, length);
    }
    else if (length <= 48) {
        sz_hash_minimal_update_haswell_(&minimal_state, _mm_lddqu_si128(&ins_xmms[0]));
        sz_hash_minimal_update_haswell_(&minimal_state, _mm_lddqu_si128(&ins_xmms[1]));
        sz_hash_minimal_update_haswell_(&minimal_state, _mm_lddqu_si128(&ins_xmms[2]));
        return sz_hash_minimal_finalize_haswell_(&minimal_state, length);
    }
    else {
        sz_hash_minimal_update_haswell_(&minimal_state, _mm_lddqu_si128(&ins_xmms[0]));
        sz_hash_minimal_update_haswell_(&minimal_state, _mm_lddqu_si128(&ins_xmms[1]));
        sz_hash_minimal_update_haswell_(&minimal_state, _mm_lddqu_si128(&ins_xmms[2]));
        sz_hash_minimal_update_haswell_(&minimal_state, _mm_lddqu_si128(&ins_xmms[3]));
        return sz_hash_minimal_finalize_haswell_(&minimal_state, length);
    }
}

SZ_PUBLIC void sz_fill_random_haswell(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_u64_t const *pi_ptr = sz_hash_pi_constants_();
    if (length <= 16) {
        __m128i input = _mm_set1_epi64x(nonce);
        __m128i pi = _mm_load_si128((__m128i const *)pi_ptr);
        __m128i key = _mm_xor_si128(_mm_set1_epi64x(nonce), pi);
        __m128i generated = _mm_aesenc_si128(input, key);
        // Now the tricky part is outputting this data to the user-supplied buffer
        // without masked writes, like in AVX-512.
        for (sz_size_t i = 0; i < length; ++i) text[i] = ((sz_u8_t *)&generated)[i];
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
        for (sz_size_t i = 16; i < length; ++i) text[i] = ((sz_u8_t *)&generated[1])[i - 16];
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
        for (sz_size_t i = 32; i < length; ++i) text[i] = ((sz_u8_t *)generated)[i];
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
        sz_size_t i = 0;
        __m128i const increment = _mm_set1_epi64x(4);
        for (; i + 64 <= length; i += 64) {
            generated[0] = _mm_aesenc_si128(inputs[0], keys[0]);
            generated[1] = _mm_aesenc_si128(inputs[1], keys[1]);
            generated[2] = _mm_aesenc_si128(inputs[2], keys[2]);
            generated[3] = _mm_aesenc_si128(inputs[3], keys[3]);
            _mm_storeu_si128((__m128i *)(text + i + 0), generated[0]);
            _mm_storeu_si128((__m128i *)(text + i + 16), generated[1]);
            _mm_storeu_si128((__m128i *)(text + i + 32), generated[2]);
            _mm_storeu_si128((__m128i *)(text + i + 48), generated[3]);
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
            for (sz_size_t j = 0; i < length; ++i, ++j) text[i] = ((sz_u8_t *)generated)[j];
        }
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_HASWELL
#pragma endregion // Haswell Implementation

/*  AVX512 implementation of the string hashing algorithms for Skylake and newer CPUs.
 *  Includes extensions: F, CD, ER, PF, VL, DQ, BW.
 *
 *  This is the "starting level" for the advanced algorithms using K-mask registers on x86.
 */
#pragma region Skylake Implementation
#if SZ_USE_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2,aes"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2", "aes")
#endif

SZ_PUBLIC sz_u64_t sz_bytesum_skylake(sz_cptr_t text, sz_size_t length) {
    // The naive implementation of this function is very simple.
    // It assumes the CPU is great at handling unaligned "loads".
    //
    // A typical AWS Sapphire Rapids instance can have 48 KB x 2 blocks of L1 data cache per core,
    // 2 MB x 2 blocks of L2 cache per core, and one shared 60 MB buffer of L3 cache.
    // With two strings, we may consider the overall workload huge, if each exceeds 1 MB in length.
    int const is_huge = length >= 1ull * 1024ull * 1024ull;
    sz_u512_vec_t text_vec, sums_vec;

    // When the buffer is small, there isn't much to innovate.
    // Separately handling even smaller payloads doesn't increase performance even on synthetic benchmarks.
    if (length <= 16) {
        __mmask16 mask = sz_u16_mask_until_(length);
        text_vec.xmms[0] = _mm_maskz_loadu_epi8(mask, text);
        sums_vec.xmms[0] = _mm_sad_epu8(text_vec.xmms[0], _mm_setzero_si128());
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_vec.xmms[0]);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_vec.xmms[0], 1);
        return low + high;
    }
    else if (length <= 32) {
        __mmask32 mask = sz_u32_mask_until_(length);
        text_vec.ymms[0] = _mm256_maskz_loadu_epi8(mask, text);
        sums_vec.ymms[0] = _mm256_sad_epu8(text_vec.ymms[0], _mm256_setzero_si256());
        // Accumulating 256 bits is harder, as we need to extract the 128-bit sums first.
        __m128i low_xmm = _mm256_castsi256_si128(sums_vec.ymms[0]);
        __m128i high_xmm = _mm256_extracti128_si256(sums_vec.ymms[0], 1);
        __m128i sums_xmm = _mm_add_epi64(low_xmm, high_xmm);
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_xmm);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_xmm, 1);
        return low + high;
    }
    else if (length <= 64) {
        __mmask64 mask = sz_u64_mask_until_(length);
        text_vec.zmm = _mm512_maskz_loadu_epi8(mask, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512());
        return _mm512_reduce_add_epi64(sums_vec.zmm);
    }
    // For large buffers, fitting into L1 cache sizes, there are other tricks we can use.
    //
    // 1. Moving in both directions to maximize the throughput, when fetching from multiple
    //    memory pages. Also helps with cache set-associativity issues, as we won't always
    //    be fetching the same buckets in the lookup table.
    //
    // Bidirectional traversal generally adds about 10% to such algorithms.
    else if (!is_huge) {
        sz_size_t head_length = (64 - ((sz_size_t)text % 64)) % 64; // 63 or less.
        sz_size_t tail_length = (sz_size_t)(text + length) % 64;    // 63 or less.
        sz_size_t body_length = length - head_length - tail_length; // Multiple of 64.
        sz_assert_(body_length % 64 == 0 && head_length < 64 && tail_length < 64);
        __mmask64 head_mask = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask = sz_u64_mask_until_(tail_length);

        text_vec.zmm = _mm512_maskz_loadu_epi8(head_mask, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512());
        for (text += head_length; body_length >= 64; text += 64, body_length -= 64) {
            text_vec.zmm = _mm512_load_si512((__m512i const *)text);
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
        }
        text_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask, text);
        sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
        return _mm512_reduce_add_epi64(sums_vec.zmm);
    }
    // For gigantic buffers, exceeding typical L1 cache sizes, there are other tricks we can use.
    //
    // 1. Using non-temporal loads to avoid polluting the cache.
    // 2. Prefetching the next cache line, to avoid stalling the CPU. This generally useless
    //    for predictable patterns, so disregard this advice.
    //
    // Bidirectional traversal generally adds about 10% to such algorithms.
    else {
        sz_u512_vec_t text_reversed_vec, sums_reversed_vec;
        sz_size_t head_length = (64 - ((sz_size_t)text % 64)) % 64;
        sz_size_t tail_length = (sz_size_t)(text + length) % 64;
        sz_size_t body_length = length - head_length - tail_length;
        __mmask64 head_mask = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask = sz_u64_mask_until_(tail_length);

        text_vec.zmm = _mm512_maskz_loadu_epi8(head_mask, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512());
        text_reversed_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask, text + head_length + body_length);
        sums_reversed_vec.zmm = _mm512_sad_epu8(text_reversed_vec.zmm, _mm512_setzero_si512());

        // Now in the main loop, we can use non-temporal loads, performing the operation in both directions.
        for (text += head_length; body_length >= 128; text += 64, body_length -= 128) {
            text_vec.zmm = _mm512_stream_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
            text_reversed_vec.zmm = _mm512_stream_load_si512((__m512i *)(text + body_length - 64));
            sums_reversed_vec.zmm =
                _mm512_add_epi64(sums_reversed_vec.zmm, _mm512_sad_epu8(text_reversed_vec.zmm, _mm512_setzero_si512()));
        }
        if (body_length >= 64) {
            text_vec.zmm = _mm512_stream_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
        }

        return _mm512_reduce_add_epi64(_mm512_add_epi64(sums_vec.zmm, sums_reversed_vec.zmm));
    }
}

SZ_PUBLIC void sz_hash_state_init_skylake(sz_hash_state_t *state, sz_u64_t seed) {
    // The key is made from the seed and half of it will be mixed with the length in the end
    __m512i seed_vec = _mm512_set1_epi64(seed);
    // ! In this kernel, assuming it may be called on arbitrarily misaligned `state`,
    // ! we must use `_mm_storeu_si128` stores to update the state.
    _mm_storeu_si128((__m128i *)&state->key.u8s[0], _mm512_castsi512_si128(seed_vec));

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    __m512i const pi0 = _mm512_load_epi64((__m512i const *)(pi));
    __m512i const pi1 = _mm512_load_epi64((__m512i const *)(pi + 8));
    _mm512_storeu_si512(&state->aes.zmm, _mm512_xor_si512(seed_vec, pi0));
    _mm512_storeu_si512(&state->sum.zmm, _mm512_xor_si512(seed_vec, pi1));

    // The inputs are zeroed out at the beginning
    _mm512_storeu_si512(&state->ins.zmm, _mm512_setzero_si512());
    state->ins_length = 0;
}

SZ_PUBLIC sz_u64_t sz_hash_skylake(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {

    if (length <= 16) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_haswell_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data_vec;
        data_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length), start);
        sz_hash_minimal_update_haswell_(&state, data_vec.xmm);
        return sz_hash_minimal_finalize_haswell_(&state, length);
    }
    else if (length <= 32) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_haswell_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length - 16), start + 16);
        sz_hash_minimal_update_haswell_(&state, data0_vec.xmm);
        sz_hash_minimal_update_haswell_(&state, data1_vec.xmm);
        return sz_hash_minimal_finalize_haswell_(&state, length);
    }
    else if (length <= 48) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_haswell_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 16));
        data2_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length - 32), start + 32);
        sz_hash_minimal_update_haswell_(&state, data0_vec.xmm);
        sz_hash_minimal_update_haswell_(&state, data1_vec.xmm);
        sz_hash_minimal_update_haswell_(&state, data2_vec.xmm);
        return sz_hash_minimal_finalize_haswell_(&state, length);
    }
    else if (length <= 64) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_haswell_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 16));
        data2_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 32));
        data3_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length - 48), start + 48);
        sz_hash_minimal_update_haswell_(&state, data0_vec.xmm);
        sz_hash_minimal_update_haswell_(&state, data1_vec.xmm);
        sz_hash_minimal_update_haswell_(&state, data2_vec.xmm);
        sz_hash_minimal_update_haswell_(&state, data3_vec.xmm);
        return sz_hash_minimal_finalize_haswell_(&state, length);
    }
    else {
        // Use a larger state to handle the main loop and add different offsets
        // to different lanes of the register
        SZ_ALIGN64 sz_hash_state_t state;
        sz_hash_state_init_skylake(&state, seed);

        for (; state.ins_length + 64 <= length; state.ins_length += 64) {
            state.ins.zmm = _mm512_loadu_epi8(start + state.ins_length);
            sz_hash_state_update_haswell_(&state);
        }
        if (state.ins_length < length) {
            state.ins.zmm = _mm512_maskz_loadu_epi8( //
                sz_u64_mask_until_(length - state.ins_length), start + state.ins_length);
            sz_hash_state_update_haswell_(&state);
            state.ins_length = length;
        }
        return sz_hash_state_finalize_haswell_(&state);
    }
}

SZ_PUBLIC void sz_hash_state_update_skylake(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    while (length) {
        sz_size_t const progress_in_block = state->ins_length % 64;
        sz_size_t const to_copy = sz_min_of_two(length, 64 - progress_in_block);
        int const will_fill_block = progress_in_block + to_copy == 64;
        // Update the metadata before we modify the `to_copy` variable
        state->ins_length += to_copy;
        length -= to_copy;
        // Append to the internal buffer until it's full
        __mmask64 to_copy_mask = sz_u64_mask_until_(to_copy);
        _mm512_mask_storeu_epi8(&state->ins.u8s[0] + progress_in_block, to_copy_mask,
                                _mm512_maskz_loadu_epi8(to_copy_mask, text));
        text += to_copy;
        // If we've reached the end of the buffer, update the state
        if (will_fill_block) {
            sz_hash_state_update_haswell_(state);
            // Reset to zeros now, so we don't have to overwrite an immutable buffer in the folding state
            _mm512_storeu_si512(&state->ins.zmm, _mm512_setzero_si512());
        }
    }
}

SZ_PUBLIC sz_u64_t sz_hash_state_digest_skylake(sz_hash_state_t const *state) {
    // ? We don't know a better way to fold the state on Ice Lake, than to use the Haswell implementation.
    return sz_hash_state_digest_haswell(state);
}

SZ_PUBLIC void sz_fill_random_skylake(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_fill_random_haswell(text, length, nonce);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_SKYLAKE
#pragma endregion // Skylake Implementation

/*  AVX512 implementation of the string search algorithms for Ice Lake and newer CPUs.
 *  Includes extensions:
 *  - 2017 Skylake: F, CD, ER, PF, VL, DQ, BW,
 *  - 2018 CannonLake: IFMA, VBMI,
 *  - 2019 Ice Lake: VPOPCNTDQ, VNNI, VBMI2, BITALG, GFNI, VPCLMULQDQ, VAES.
 */
#pragma region Ice Lake Implementation
#if SZ_USE_ICE
#if defined(__clang__)
#pragma clang attribute push(                                                                                  \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vnni,bmi,bmi2,aes,vaes"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vnni", "bmi", "bmi2", \
                   "aes", "vaes")
#endif

SZ_PUBLIC sz_u64_t sz_bytesum_ice(sz_cptr_t text, sz_size_t length) {
    // The naive implementation of this function is very simple.
    // It assumes the CPU is great at handling unaligned "loads".
    //
    // A typical AWS Sapphire Rapids instance can have 48 KB x 2 blocks of L1 data cache per core,
    // 2 MB x 2 blocks of L2 cache per core, and one shared 60 MB buffer of L3 cache.
    // With two strings, we may consider the overall workload huge, if each exceeds 1 MB in length.
    int const is_huge = length >= 1ull * 1024ull * 1024ull;
    sz_u512_vec_t text_vec, sums_vec;

    // When the buffer is small, there isn't much to innovate.
    // Separately handling even smaller payloads doesn't increase performance even on synthetic benchmarks.
    if (length <= 16) {
        __mmask16 mask = sz_u16_mask_until_(length);
        text_vec.xmms[0] = _mm_maskz_loadu_epi8(mask, text);
        sums_vec.xmms[0] = _mm_sad_epu8(text_vec.xmms[0], _mm_setzero_si128());
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_vec.xmms[0]);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_vec.xmms[0], 1);
        return low + high;
    }
    else if (length <= 32) {
        __mmask32 mask = sz_u32_mask_until_(length);
        text_vec.ymms[0] = _mm256_maskz_loadu_epi8(mask, text);
        sums_vec.ymms[0] = _mm256_sad_epu8(text_vec.ymms[0], _mm256_setzero_si256());
        // Accumulating 256 bits is harder, as we need to extract the 128-bit sums first.
        __m128i low_xmm = _mm256_castsi256_si128(sums_vec.ymms[0]);
        __m128i high_xmm = _mm256_extracti128_si256(sums_vec.ymms[0], 1);
        __m128i sums_xmm = _mm_add_epi64(low_xmm, high_xmm);
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_xmm);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_xmm, 1);
        return low + high;
    }
    else if (length <= 64) {
        __mmask64 mask = sz_u64_mask_until_(length);
        text_vec.zmm = _mm512_maskz_loadu_epi8(mask, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512());
        return _mm512_reduce_add_epi64(sums_vec.zmm);
    }
    // For large buffers, fitting into L1 cache sizes, there are other tricks we can use.
    //
    // 1. Moving in both directions to maximize the throughput, when fetching from multiple
    //    memory pages. Also helps with cache set-associativity issues, as we won't always
    //    be fetching the same buckets in the lookup table.
    // 2. Port-level parallelism, can be used to hide the latency of expensive SIMD instructions.
    //    - `VPSADBW (ZMM, ZMM, ZMM)` combination with `VPADDQ (ZMM, ZMM, ZMM)`:
    //        - On Ice Lake, the `VPSADBW` is 3 cycles on port 5; the `VPADDQ` is 1 cycle on ports 0/5.
    //        - On Zen 4, the `VPSADBW` is 3 cycles on ports 0/1; the `VPADDQ` is 1 cycle on ports 0/1/2/3.
    //    - `VPDPBUSDS (ZMM, ZMM, ZMM)`:
    //        - On Ice Lake, the `VPDPBUSDS` is 5 cycles on port 0.
    //        - On Zen 4, the `VPDPBUSDS` is 4 cycles on ports 0/1.
    //
    // Bidirectional traversal generally adds about 10% to such algorithms.
    // Port level parallelism can yield more, but remember that one of the instructions accumulates
    // with 32-bit integers and the other one will be using 64-bit integers.
    else if (!is_huge) {
        sz_size_t head_length = (64 - ((sz_size_t)text % 64)) % 64; // 63 or less.
        sz_size_t tail_length = (sz_size_t)(text + length) % 64;    // 63 or less.
        sz_size_t body_length = length - head_length - tail_length; // Multiple of 64.
        sz_assert_(body_length % 64 == 0 && head_length < 64 && tail_length < 64);
        __mmask64 head_mask = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask = sz_u64_mask_until_(tail_length);

        sz_u512_vec_t zeros_vec, ones_vec;
        zeros_vec.zmm = _mm512_setzero_si512();
        ones_vec.zmm = _mm512_set1_epi8(1);

        // Take care of the unaligned head and tail!
        sz_u512_vec_t text_reversed_vec, sums_reversed_vec;
        text_vec.zmm = _mm512_maskz_loadu_epi8(head_mask, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, zeros_vec.zmm);
        text_reversed_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask, text + head_length + body_length);
        sums_reversed_vec.zmm = _mm512_dpbusds_epi32(zeros_vec.zmm, text_reversed_vec.zmm, ones_vec.zmm);

        // Now in the main loop, we can use aligned loads, performing the operation in both directions.
        for (text += head_length; body_length >= 128; text += 64, body_length -= 128) {
            text_reversed_vec.zmm = _mm512_load_si512((__m512i *)(text + body_length - 64));
            sums_reversed_vec.zmm = _mm512_dpbusds_epi32(sums_reversed_vec.zmm, text_reversed_vec.zmm, ones_vec.zmm);
            text_vec.zmm = _mm512_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, zeros_vec.zmm));
        }
        // There may be an aligned chunk of 64 bytes left.
        if (body_length >= 64) {
            sz_assert_(body_length == 64);
            text_vec.zmm = _mm512_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, zeros_vec.zmm));
        }

        return _mm512_reduce_add_epi64(sums_vec.zmm) + _mm512_reduce_add_epi32(sums_reversed_vec.zmm);
    }
    // For gigantic buffers, exceeding typical L1 cache sizes, there are other tricks we can use.
    //
    // 1. Using non-temporal loads to avoid polluting the cache.
    // 2. Prefetching the next cache line, to avoid stalling the CPU. This generally useless
    //    for predictable patterns, so disregard this advice.
    //
    // Bidirectional traversal generally adds about 10% to such algorithms.
    else {
        sz_u512_vec_t text_reversed_vec, sums_reversed_vec;
        sz_size_t head_length = (64 - ((sz_size_t)text % 64)) % 64;
        sz_size_t tail_length = (sz_size_t)(text + length) % 64;
        sz_size_t body_length = length - head_length - tail_length;
        __mmask64 head_mask = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask = sz_u64_mask_until_(tail_length);

        text_vec.zmm = _mm512_maskz_loadu_epi8(head_mask, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512());
        text_reversed_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask, text + head_length + body_length);
        sums_reversed_vec.zmm = _mm512_sad_epu8(text_reversed_vec.zmm, _mm512_setzero_si512());

        // Now in the main loop, we can use non-temporal loads, performing the operation in both directions.
        for (text += head_length; body_length >= 128; text += 64, body_length -= 128) {
            text_vec.zmm = _mm512_stream_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
            text_reversed_vec.zmm = _mm512_stream_load_si512((__m512i *)(text + body_length - 64));
            sums_reversed_vec.zmm =
                _mm512_add_epi64(sums_reversed_vec.zmm, _mm512_sad_epu8(text_reversed_vec.zmm, _mm512_setzero_si512()));
        }
        if (body_length >= 64) {
            text_vec.zmm = _mm512_stream_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
        }

        return _mm512_reduce_add_epi64(_mm512_add_epi64(sums_vec.zmm, sums_reversed_vec.zmm));
    }
}

SZ_INTERNAL void sz_hash_state_update_ice_(sz_hash_state_t *state) {
    __m512i const shuffle_mask = _mm512_load_si512((__m512i const *)sz_hash_u8x16x4_shuffle_());
    // ! In this kernel, assuming it may be called on arbitrarily misaligned `state`,
    // ! we must use `_mm512_storeu_si512` stores to update the state.
    _mm512_storeu_si512(&state->aes.zmm,
                        _mm512_aesenc_epi128(_mm512_loadu_si512(&state->aes.zmm), _mm512_loadu_si512(&state->ins.zmm)));
    _mm512_storeu_si512(&state->sum.zmm,
                        _mm512_add_epi64(_mm512_shuffle_epi8(_mm512_loadu_si512(&state->sum.zmm), shuffle_mask),
                                         _mm512_loadu_si512(&state->ins.zmm)));
}

SZ_PUBLIC sz_u64_t sz_hash_ice(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {

    // For short strings the "masked loads" are identical to Skylake-X and
    // the "logic" is identical to Haswell.
    if (length <= 16) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_haswell_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data_vec;
        data_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length), start);
        sz_hash_minimal_update_haswell_(&state, data_vec.xmm);
        return sz_hash_minimal_finalize_haswell_(&state, length);
    }
    else if (length <= 32) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_haswell_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length - 16), start + 16);
        sz_hash_minimal_update_haswell_(&state, data0_vec.xmm);
        sz_hash_minimal_update_haswell_(&state, data1_vec.xmm);
        return sz_hash_minimal_finalize_haswell_(&state, length);
    }
    else if (length <= 48) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_haswell_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 16));
        data2_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length - 32), start + 32);
        sz_hash_minimal_update_haswell_(&state, data0_vec.xmm);
        sz_hash_minimal_update_haswell_(&state, data1_vec.xmm);
        sz_hash_minimal_update_haswell_(&state, data2_vec.xmm);
        return sz_hash_minimal_finalize_haswell_(&state, length);
    }
    else if (length <= 64) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_haswell_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 16));
        data2_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 32));
        data3_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length - 48), start + 48);
        sz_hash_minimal_update_haswell_(&state, data0_vec.xmm);
        sz_hash_minimal_update_haswell_(&state, data1_vec.xmm);
        sz_hash_minimal_update_haswell_(&state, data2_vec.xmm);
        sz_hash_minimal_update_haswell_(&state, data3_vec.xmm);
        return sz_hash_minimal_finalize_haswell_(&state, length);
    }
    // This is where the logic differs from Skylake-X and other pre-Ice Lake CPUs:
    else {
        // Use a larger state to handle the main loop and add different offsets
        // to different lanes of the register
        SZ_ALIGN64 sz_hash_state_t state;
        sz_hash_state_init_skylake(&state, seed);

        for (; state.ins_length + 64 <= length; state.ins_length += 64) {
            state.ins.zmm = _mm512_loadu_epi8(start + state.ins_length);
            sz_hash_state_update_ice_(&state);
        }
        if (state.ins_length < length) {
            state.ins.zmm =
                _mm512_maskz_loadu_epi8(sz_u64_mask_until_(length - state.ins_length), start + state.ins_length);
            sz_hash_state_update_ice_(&state);
            state.ins_length = length;
        }
        return sz_hash_state_finalize_haswell_(&state);
    }
}

SZ_PUBLIC void sz_hash_state_init_ice(sz_hash_state_t *state, sz_u64_t seed) {
    sz_hash_state_init_skylake(state, seed);
}

SZ_PUBLIC void sz_hash_state_update_ice(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    while (length) {
        sz_size_t progress_in_block = state->ins_length % 64;
        sz_size_t to_copy = sz_min_of_two(length, 64 - progress_in_block);
        int const will_fill_block = progress_in_block + to_copy == 64;
        // Update the metadata before we modify the `to_copy` variable
        state->ins_length += to_copy;
        length -= to_copy;
        // Append to the internal buffer until it's full
        __mmask64 to_copy_mask = sz_u64_mask_until_(to_copy);
        _mm512_mask_storeu_epi8(state->ins.u8s + progress_in_block, to_copy_mask,
                                _mm512_maskz_loadu_epi8(to_copy_mask, text));
        text += to_copy;
        // If we've reached the end of the buffer, update the state
        if (will_fill_block) {
            sz_hash_state_update_ice_(state);
            // Reset to zeros now, so we don't have to overwrite an immutable buffer in the folding state
            _mm512_storeu_si512(&state->ins.zmm, _mm512_setzero_si512());
        }
    }
}

SZ_PUBLIC sz_u64_t sz_hash_state_digest_ice(sz_hash_state_t const *state) {
    // ? We don't know a better way to fold the state on Ice Lake, than to use the Haswell implementation.
    return sz_hash_state_digest_haswell(state);
}

SZ_PUBLIC void sz_fill_random_ice(sz_ptr_t output, sz_size_t length, sz_u64_t nonce) {
    if (length <= 16) {
        __m128i input = _mm_set1_epi64x(nonce);
        __m128i pi = _mm_load_si128((__m128i const *)sz_hash_pi_constants_());
        __m128i key = _mm_xor_si128(_mm_set1_epi64x(nonce), pi);
        __m128i generated = _mm_aesenc_si128(input, key);
        __mmask16 store_mask = sz_u16_mask_until_(length);
        _mm_mask_storeu_epi8((void *)output, store_mask, generated);
    }
    // Assuming the YMM register contains two 128-bit blocks, the input to the generator
    // will be more complex, containing the sum of the nonce and the block number.
    else if (length <= 32) {
        __m256i input = _mm256_set_epi64x(nonce + 1, nonce + 1, nonce, nonce);
        __m256i pi = _mm256_load_si256((__m256i const *)sz_hash_pi_constants_());
        __m256i key = _mm256_xor_si256(_mm256_set1_epi64x(nonce), pi);
        __m256i generated = _mm256_aesenc_epi128(input, key);
        __mmask32 store_mask = sz_u32_mask_until_(length);
        _mm256_mask_storeu_epi8((void *)output, store_mask, generated);
    }
    // The last special case we handle outside of the primary loop is for buffers up to 64 bytes long.
    else if (length <= 64) {
        __m512i input = _mm512_set_epi64(               //
            nonce + 3, nonce + 3, nonce + 2, nonce + 2, //
            nonce + 1, nonce + 1, nonce, nonce);
        __m512i pi = _mm512_load_si512((__m512i const *)sz_hash_pi_constants_());
        __m512i key = _mm512_xor_si512(_mm512_set1_epi64(nonce), pi);
        __m512i generated = _mm512_aesenc_epi128(input, key);
        __mmask64 store_mask = sz_u64_mask_until_(length);
        _mm512_mask_storeu_epi8((void *)output, store_mask, generated);
    }
    // The final part of the function is the primary loop, which processes the buffer in 64-byte chunks.
    else {
        __m512i const increment = _mm512_set1_epi64(4);
        __m512i input = _mm512_set_epi64(               //
            nonce + 3, nonce + 3, nonce + 2, nonce + 2, //
            nonce + 1, nonce + 1, nonce, nonce);
        __m512i const pi = _mm512_load_si512((__m512i const *)sz_hash_pi_constants_());
        __m512i const key = _mm512_xor_si512(_mm512_set1_epi64(nonce), pi);

        // Produce the output, fixing the key and enumerating input chunks.
        sz_size_t i = 0;
        for (; i + 64 <= length; i += 64) {
            __m512i generated = _mm512_aesenc_epi128(input, key);
            _mm512_storeu_epi8((void *)(output + i), generated);
            input = _mm512_add_epi64(input, increment);
        }

        // Handle the tail of the buffer.
        __m512i generated = _mm512_aesenc_epi128(input, key);
        __mmask64 store_mask = sz_u64_mask_until_(length - i);
        _mm512_mask_storeu_epi8((void *)(output + i), store_mask, generated);
    }
}

/**
 *  @brief  A wider parallel analog of `sz_hash_minimal_t_`, which is not used for computing individual hashes,
 *          but for parallel hashing of @b short 4x separate strings under 16 bytes long.
 *          Useful for higher-level Database and Machine Learning operations.
 */
typedef struct sz_hash_minimal_x4_t_ {
    sz_u512_vec_t aes;
    sz_u512_vec_t sum;
    sz_u512_vec_t key;
} sz_hash_minimal_x4_t_;

SZ_INTERNAL void sz_hash_minimal_x4_init_ice_(sz_hash_minimal_x4_t_ *state, sz_u64_t seed) {

    // The key is made from the seed and half of it will be mixed with the length in the end
    __m512i seed_vec = _mm512_set1_epi64(seed);
    state->key.zmm = seed_vec; //! This will definitely be aligned

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    __m512i pi0 = _mm512_load_si512((__m512i const *)(pi));
    __m512i pi1 = _mm512_load_si512((__m512i const *)(pi + 8));
    // We will load the entire 512-bit values, but will only use the first 128 bits,
    // replicating it 4x times across the register. The `_mm512_shuffle_i64x2` is supposed to
    // be faster than `_mm512_broadcast_i64x2` on Ice Lake.
    pi0 = _mm512_shuffle_i64x2(pi0, pi0, 0);
    pi1 = _mm512_shuffle_i64x2(pi1, pi1, 0);
    __m512i k1 = _mm512_xor_si512(seed_vec, pi0);
    __m512i k2 = _mm512_xor_si512(seed_vec, pi1);

    // The first 128 bits of the "sum" and "AES" blocks are the same for the "minimal" and full state
    state->aes.zmm = k1;
    state->sum.zmm = k2;
}

SZ_INTERNAL __m256i sz_hash_minimal_x4_finalize_ice_(sz_hash_minimal_x4_t_ const *state, //
                                                     sz_size_t length0, sz_size_t length1, sz_size_t length2,
                                                     sz_size_t length3) {
    __m512i const padded_lengths = _mm512_set_epi64(0, length3, 0, length2, 0, length1, 0, length0);
    // Mix the length into the key
    __m512i key_with_length = _mm512_add_epi64(state->key.zmm, padded_lengths);
    // Combine the "sum" and the "AES" blocks
    __m512i mixed = _mm512_aesenc_epi128(state->sum.zmm, state->aes.zmm);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    __m512i mixed_in_register = _mm512_aesenc_epi128(_mm512_aesenc_epi128(mixed, key_with_length), mixed);
    // Extract the low 64 bits from each 128-bit lane - weirdly using the `permutexvar` instruction
    // is cheaper than compressing instructions like `_mm512_maskz_compress_epi64`.
    return _mm512_castsi512_si256(
        _mm512_permutexvar_epi64(_mm512_set_epi64(0, 0, 0, 0, 6, 4, 2, 0), mixed_in_register));
}

SZ_INTERNAL void sz_hash_minimal_x4_update_ice_(sz_hash_minimal_x4_t_ *state, __m512i blocks) {
    __m512i const shuffle_mask = _mm512_load_si512((__m512i const *)sz_hash_u8x16x4_shuffle_());
    state->aes.zmm = _mm512_aesenc_epi128(state->aes.zmm, blocks);
    state->sum.zmm = _mm512_add_epi64(_mm512_shuffle_epi8(state->sum.zmm, shuffle_mask), blocks);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_ICE
#pragma endregion // Ice Lake Implementation

/*  Implementation of the string hashing algorithms using the Arm NEON instruction set, available on 64-bit
 *  Arm processors. Covers billions of mobile CPUs worldwide, including Apple's A-series, and Qualcomm's Snapdragon.
 */
#pragma region NEON Implementation
#if SZ_USE_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#endif

SZ_PUBLIC sz_u64_t sz_bytesum_neon(sz_cptr_t text, sz_size_t length) {
    uint64x2_t sum_vec = vdupq_n_u64(0);

    // Process 16 bytes (128 bits) at a time
    for (; length >= 16; text += 16, length -= 16) {
        uint8x16_t vec = vld1q_u8((sz_u8_t const *)text);      // Load 16 bytes
        uint16x8_t pairwise_sum1 = vpaddlq_u8(vec);            // Pairwise add lower and upper 8 bits
        uint32x4_t pairwise_sum2 = vpaddlq_u16(pairwise_sum1); // Pairwise add 16-bit results
        uint64x2_t pairwise_sum3 = vpaddlq_u32(pairwise_sum2); // Pairwise add 32-bit results
        sum_vec = vaddq_u64(sum_vec, pairwise_sum3);           // Accumulate the sum
    }

    // Final reduction of `sum_vec` to a single scalar
    sz_u64_t sum = vgetq_lane_u64(sum_vec, 0) + vgetq_lane_u64(sum_vec, 1);
    while (length--) sum += *(sz_u8_t const *)text++; // Same as the scalar version
    return sum;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_NEON
#pragma endregion // NEON Implementation

#pragma region NEON AES Implementation
#if SZ_USE_NEON_AES
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+crypto+aes"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+crypto+aes")
#endif

/**
 *  @brief  Emulates the Intel's AES-NI `AESENC` instruction on Arm NEON.
 *  @see    "Emulating x86 AES Intrinsics on ARMv8-A" by Michael Brase:
 *          https://blog.michaelbrase.com/2018/05/08/emulating-x86-aes-intrinsics-on-armv8-a/
 */
SZ_INTERNAL uint8x16_t sz_emulate_aesenc_u8x16_neon_(uint8x16_t state_vec, uint8x16_t round_key_vec) {
    return veorq_u8(vaesmcq_u8(vaeseq_u8(state_vec, vdupq_n_u8(0))), round_key_vec);
}

SZ_INTERNAL uint64x2_t sz_emulate_aesenc_u64x2_neon_(uint64x2_t state_vec, uint64x2_t round_key_vec) {
    return vreinterpretq_u64_u8(             //
        sz_emulate_aesenc_u8x16_neon_(       //
            vreinterpretq_u8_u64(state_vec), //
            vreinterpretq_u8_u64(round_key_vec)));
}

SZ_INTERNAL void sz_hash_minimal_init_neon_(sz_hash_minimal_t_ *state, sz_u64_t seed) {

    // The key is made from the seed and half of it will be mixed with the length in the end
    uint64x2_t seed_vec = vdupq_n_u64(seed);
    state->key.u64x2 = seed_vec;

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    uint64x2_t const pi0 = vld1q_u64(pi);
    uint64x2_t const pi1 = vld1q_u64(pi + 8);
    uint64x2_t k1 = veorq_u64(seed_vec, pi0);
    uint64x2_t k2 = veorq_u64(seed_vec, pi1);

    // The first 128 bits of the "sum" and "AES" blocks are the same for the "minimal" and full state
    state->aes.u64x2 = k1;
    state->sum.u64x2 = k2;
}

SZ_INTERNAL sz_u64_t sz_hash_minimal_finalize_neon_(sz_hash_minimal_t_ const *state, sz_size_t length) {
    // Mix the length into the key
    uint64x2_t key_with_length = vaddq_u64(state->key.u64x2, vsetq_lane_u64(length, vdupq_n_u64(0), 0));
    // Combine the "sum" and the "AES" blocks
    uint8x16_t mixed = sz_emulate_aesenc_u8x16_neon_(state->sum.u8x16, state->aes.u8x16);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    uint8x16_t mixed_in_register = sz_emulate_aesenc_u8x16_neon_(
        sz_emulate_aesenc_u8x16_neon_(mixed, vreinterpretq_u8_u64(key_with_length)), mixed);
    // Extract the low 64 bits
    return vgetq_lane_u64(vreinterpretq_u64_u8(mixed_in_register), 0);
}

SZ_INTERNAL void sz_hash_minimal_update_neon_(sz_hash_minimal_t_ *state, uint8x16_t block) {
    uint8x16_t const shuffle_mask = vld1q_u8(sz_hash_u8x16x4_shuffle_());
    state->aes.u8x16 = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16, block);
    uint8x16_t sum_shuffled = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2), shuffle_mask);
    state->sum.u64x2 = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled), vreinterpretq_u64_u8(block));
}

SZ_PUBLIC void sz_hash_state_init_neon(sz_hash_state_t *state, sz_u64_t seed) {
    // The key is made from the seed and half of it will be mixed with the length in the end
    uint64x2_t seed_vec = vdupq_n_u64(seed);
    state->key.u64x2 = seed_vec;

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    for (int i = 0; i < 4; ++i) state->aes.u64x2s[i] = veorq_u64(seed_vec, vld1q_u64(pi + i * 2));
    for (int i = 0; i < 4; ++i) state->sum.u64x2s[i] = veorq_u64(seed_vec, vld1q_u64(pi + i * 2 + 8));

    // The inputs are zeroed out at the beginning
    state->ins.u8x16s[0] = state->ins.u8x16s[1] = state->ins.u8x16s[2] = state->ins.u8x16s[3] = vdupq_n_u8(0);
    state->ins_length = 0;
}

SZ_INTERNAL void sz_hash_state_update_neon_(sz_hash_state_t *state) {
    uint8x16_t const shuffle_mask = vld1q_u8(sz_hash_u8x16x4_shuffle_());
    state->aes.u8x16s[0] = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[0], state->ins.u8x16s[0]);
    uint8x16_t sum_shuffled0 = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[0]), shuffle_mask);
    state->sum.u64x2s[0] = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled0), state->ins.u64x2s[0]);
    state->aes.u8x16s[1] = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[1], state->ins.u8x16s[1]);
    uint8x16_t sum_shuffled1 = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[1]), shuffle_mask);
    state->sum.u64x2s[1] = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled1), state->ins.u64x2s[1]);
    state->aes.u8x16s[2] = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[2], state->ins.u8x16s[2]);
    uint8x16_t sum_shuffled2 = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[2]), shuffle_mask);
    state->sum.u64x2s[2] = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled2), state->ins.u64x2s[2]);
    state->aes.u8x16s[3] = sz_emulate_aesenc_u8x16_neon_(state->aes.u8x16s[3], state->ins.u8x16s[3]);
    uint8x16_t sum_shuffled3 = vqtbl1q_u8(vreinterpretq_u8_u64(state->sum.u64x2s[3]), shuffle_mask);
    state->sum.u64x2s[3] = vaddq_u64(vreinterpretq_u64_u8(sum_shuffled3), state->ins.u64x2s[3]);
}

SZ_INTERNAL sz_u64_t sz_hash_state_finalize_neon_(sz_hash_state_t const *state) {
    // Mix the length into the key
    uint64x2_t key_with_length = vaddq_u64(state->key.u64x2, vsetq_lane_u64(state->ins_length, vdupq_n_u64(0), 0));
    // Combine the "sum" and the "AES" blocks
    uint8x16_t mixed0 = sz_emulate_aesenc_u8x16_neon_(state->sum.u8x16s[0], state->aes.u8x16s[0]);
    uint8x16_t mixed1 = sz_emulate_aesenc_u8x16_neon_(state->sum.u8x16s[1], state->aes.u8x16s[1]);
    uint8x16_t mixed2 = sz_emulate_aesenc_u8x16_neon_(state->sum.u8x16s[2], state->aes.u8x16s[2]);
    uint8x16_t mixed3 = sz_emulate_aesenc_u8x16_neon_(state->sum.u8x16s[3], state->aes.u8x16s[3]);
    // Combine the mixed registers
    uint8x16_t mixed01 = sz_emulate_aesenc_u8x16_neon_(mixed0, mixed1);
    uint8x16_t mixed23 = sz_emulate_aesenc_u8x16_neon_(mixed2, mixed3);
    uint8x16_t mixed = sz_emulate_aesenc_u8x16_neon_(mixed01, mixed23);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    uint8x16_t mixed_in_register = sz_emulate_aesenc_u8x16_neon_(
        sz_emulate_aesenc_u8x16_neon_(mixed, vreinterpretq_u8_u64(key_with_length)), mixed);
    // Extract the low 64 bits
    return vgetq_lane_u64(vreinterpretq_u64_u8(mixed_in_register), 0);
}

SZ_PUBLIC void sz_hash_state_update_neon(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    // This whole function is identical to Haswell.
    while (length) {
        // Append to the internal buffer until it's full
        if (state->ins_length % 64 == 0 && length >= 64) {
            state->ins.u8x16s[0] = vld1q_u8((sz_u8_t const *)(text + 0));
            state->ins.u8x16s[1] = vld1q_u8((sz_u8_t const *)(text + 16));
            state->ins.u8x16s[2] = vld1q_u8((sz_u8_t const *)(text + 32));
            state->ins.u8x16s[3] = vld1q_u8((sz_u8_t const *)(text + 48));
            sz_hash_state_update_neon_(state);
            state->ins_length += 64;
            text += 64;
            length -= 64;
        }
        // If vectorization isn't that trivial - fall back to the serial implementation
        else {
            sz_size_t progress_in_block = state->ins_length % 64;
            sz_size_t to_copy = sz_min_of_two(length, 64 - progress_in_block);
            int const will_fill_block = progress_in_block + to_copy == 64;
            // Update the metadata before we modify the `to_copy` variable
            state->ins_length += to_copy;
            length -= to_copy;
            // Append to the internal buffer until it's full
            while (to_copy--) state->ins.u8s[progress_in_block++] = *text++;
            // If we've reached the end of the buffer, update the state
            if (will_fill_block) {
                sz_hash_state_update_neon_(state);
                // Reset to zeros now, so we don't have to overwrite an immutable buffer in the folding state
                for (int i = 0; i < 4; ++i) vst1q_u8(state->ins.u8s + i * 16, vdupq_n_u8(0));
            }
        }
    }
}

SZ_PUBLIC sz_u64_t sz_hash_state_digest_neon(sz_hash_state_t const *state) {
    // This whole function is identical to Haswell.
    sz_size_t length = state->ins_length;
    if (length >= 64) return sz_hash_state_finalize_neon_(state);

    // Switch back to a smaller "minimal" state for small inputs
    sz_hash_minimal_t_ minimal_state;
    minimal_state.key.u8x16 = state->key.u8x16;
    minimal_state.aes.u8x16 = state->aes.u8x16s[0];
    minimal_state.sum.u8x16 = state->sum.u8x16s[0];

    // The logic is different depending on the length of the input
    uint8x16_t const *ins_vecs = (uint8x16_t const *)&state->ins.u8x16s[0];
    if (length <= 16) {
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[0]);
        return sz_hash_minimal_finalize_neon_(&minimal_state, length);
    }
    else if (length <= 32) {
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[1]);
        return sz_hash_minimal_finalize_neon_(&minimal_state, length);
    }
    else if (length <= 48) {
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[1]);
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[2]);
        return sz_hash_minimal_finalize_neon_(&minimal_state, length);
    }
    else {
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[1]);
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[2]);
        sz_hash_minimal_update_neon_(&minimal_state, ins_vecs[3]);
        return sz_hash_minimal_finalize_neon_(&minimal_state, length);
    }
}

SZ_PUBLIC sz_u64_t sz_hash_neon(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {
    if (length <= 16) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data_vec;
        data_vec.u8x16 = vdupq_n_u8(0);
        for (sz_size_t i = 0; i < length; ++i) data_vec.u8s[i] = start[i];
        sz_hash_minimal_update_neon_(&state, data_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else if (length <= 32) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.u8x16 = vld1q_u8((sz_u8_t const *)(start + 0));
        data1_vec.u8x16 = vld1q_u8((sz_u8_t const *)(start + length - 16));
        // Let's shift the data within the register to de-interleave the bytes.
        sz_hash_shift_in_register_serial_(&data1_vec, (int)(32 - length)); //! `vextq_u8` requires immediates
        sz_hash_minimal_update_neon_(&state, data0_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data1_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else if (length <= 48) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.u8x16 = vld1q_u8((sz_u8_t const *)(start + 0));
        data1_vec.u8x16 = vld1q_u8((sz_u8_t const *)(start + 16));
        data2_vec.u8x16 = vld1q_u8((sz_u8_t const *)(start + length - 16));
        // Let's shift the data within the register to de-interleave the bytes.
        sz_hash_shift_in_register_serial_(&data2_vec, (int)(48 - length)); //! `vextq_u8` requires immediates
        sz_hash_minimal_update_neon_(&state, data0_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data1_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data2_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else if (length <= 64) {
        // Initialize the AES block with a given seed
        sz_hash_minimal_t_ state;
        sz_hash_minimal_init_neon_(&state, seed);
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.u8x16 = vld1q_u8((sz_u8_t const *)(start + 0));
        data1_vec.u8x16 = vld1q_u8((sz_u8_t const *)(start + 16));
        data2_vec.u8x16 = vld1q_u8((sz_u8_t const *)(start + 32));
        data3_vec.u8x16 = vld1q_u8((sz_u8_t const *)(start + length - 16));
        // Let's shift the data within the register to de-interleave the bytes.
        sz_hash_shift_in_register_serial_(&data3_vec, (int)(64 - length)); //! `vextq_u8` requires immediates
        sz_hash_minimal_update_neon_(&state, data0_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data1_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data2_vec.u8x16);
        sz_hash_minimal_update_neon_(&state, data3_vec.u8x16);
        return sz_hash_minimal_finalize_neon_(&state, length);
    }
    else {
        // Use a larger state to handle the main loop and add different offsets
        // to different lanes of the register
        SZ_ALIGN64 sz_hash_state_t state;
        sz_hash_state_init_neon(&state, seed);
        for (; state.ins_length + 64 <= length; state.ins_length += 64) {
            state.ins.u8x16s[0] = vld1q_u8((sz_u8_t const *)(start + state.ins_length + 0));
            state.ins.u8x16s[1] = vld1q_u8((sz_u8_t const *)(start + state.ins_length + 16));
            state.ins.u8x16s[2] = vld1q_u8((sz_u8_t const *)(start + state.ins_length + 32));
            state.ins.u8x16s[3] = vld1q_u8((sz_u8_t const *)(start + state.ins_length + 48));
            sz_hash_state_update_neon_(&state);
        }
        // Handle the tail, resetting the registers to zero first
        if (state.ins_length < length) {
            state.ins.u8x16s[0] = vdupq_n_u8(0);
            state.ins.u8x16s[1] = vdupq_n_u8(0);
            state.ins.u8x16s[2] = vdupq_n_u8(0);
            state.ins.u8x16s[3] = vdupq_n_u8(0);
            for (sz_size_t i = 0; state.ins_length < length; ++i, ++state.ins_length)
                state.ins.u8s[i] = start[state.ins_length];
            sz_hash_state_update_neon_(&state);
            state.ins_length = length;
        }
        return sz_hash_state_finalize_neon_(&state);
    }
}

SZ_PUBLIC void sz_fill_random_neon(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_u64_t const *pi_ptr = sz_hash_pi_constants_();
    if (length <= 16) {
        uint64x2_t input = vdupq_n_u64(nonce);
        uint64x2_t pi = vld1q_u64(pi_ptr);
        uint64x2_t key = veorq_u64(vdupq_n_u64(nonce), pi);
        uint64x2_t generated = sz_emulate_aesenc_u64x2_neon_(input, key);
        // Now the tricky part is outputting this data to the user-supplied buffer
        // without masked writes, like in AVX-512.
        for (sz_size_t i = 0; i < length; ++i) text[i] = ((sz_u8_t *)&generated)[i];
    }
    // Assuming the YMM register contains two 128-bit blocks, the input to the generator
    // will be more complex, containing the sum of the nonce and the block number.
    else if (length <= 32) {
        uint64x2_t inputs[2], pis[2], keys[2], generated[2];
        inputs[0] = vdupq_n_u64(nonce + 0);
        inputs[1] = vdupq_n_u64(nonce + 1);
        pis[0] = vld1q_u64(pi_ptr + 0);
        pis[1] = vld1q_u64(pi_ptr + 2);
        keys[0] = veorq_u64(vdupq_n_u64(nonce), pis[0]);
        keys[1] = veorq_u64(vdupq_n_u64(nonce), pis[1]);
        generated[0] = sz_emulate_aesenc_u64x2_neon_(inputs[0], keys[0]);
        generated[1] = sz_emulate_aesenc_u64x2_neon_(inputs[1], keys[1]);
        // The first store can easily be vectorized, but the second can be serial for now
        vst1q_u64((sz_u64_t *)(text), generated[0]);
        for (sz_size_t i = 16; i < length; ++i) text[i] = ((sz_u8_t *)&generated[1])[i - 16];
    }
    // The last special case we handle outside of the primary loop is for buffers up to 64 bytes long.
    else if (length <= 48) {
        uint64x2_t inputs[3], pis[3], keys[3], generated[3];
        inputs[0] = vdupq_n_u64(nonce);
        inputs[1] = vdupq_n_u64(nonce + 1);
        inputs[2] = vdupq_n_u64(nonce + 2);
        pis[0] = vld1q_u64(pi_ptr + 0);
        pis[1] = vld1q_u64(pi_ptr + 2);
        pis[2] = vld1q_u64(pi_ptr + 4);
        keys[0] = veorq_u64(vdupq_n_u64(nonce), pis[0]);
        keys[1] = veorq_u64(vdupq_n_u64(nonce), pis[1]);
        keys[2] = veorq_u64(vdupq_n_u64(nonce), pis[2]);
        generated[0] = sz_emulate_aesenc_u64x2_neon_(inputs[0], keys[0]);
        generated[1] = sz_emulate_aesenc_u64x2_neon_(inputs[1], keys[1]);
        generated[2] = sz_emulate_aesenc_u64x2_neon_(inputs[2], keys[2]);
        // The first store can easily be vectorized, but the second can be serial for now
        vst1q_u64((sz_u64_t *)(text + 0), generated[0]);
        vst1q_u64((sz_u64_t *)(text + 16), generated[1]);
        for (sz_size_t i = 32; i < length; ++i) text[i] = ((sz_u8_t *)generated)[i];
    }
    // The final part of the function is the primary loop, which processes the buffer in 64-byte chunks.
    else {
        uint64x2_t inputs[4], pis[4], keys[4], generated[4];
        inputs[0] = vdupq_n_u64(nonce + 0);
        inputs[1] = vdupq_n_u64(nonce + 1);
        inputs[2] = vdupq_n_u64(nonce + 2);
        inputs[3] = vdupq_n_u64(nonce + 3);
        // Load parts of PI into the registers
        pis[0] = vld1q_u64(pi_ptr + 0);
        pis[1] = vld1q_u64(pi_ptr + 2);
        pis[2] = vld1q_u64(pi_ptr + 4);
        pis[3] = vld1q_u64(pi_ptr + 6);
        // XOR the nonce with the PI constants
        keys[0] = veorq_u64(vdupq_n_u64(nonce), pis[0]);
        keys[1] = veorq_u64(vdupq_n_u64(nonce), pis[1]);
        keys[2] = veorq_u64(vdupq_n_u64(nonce), pis[2]);
        keys[3] = veorq_u64(vdupq_n_u64(nonce), pis[3]);

        // Produce the output, fixing the key and enumerating input chunks.
        sz_size_t i = 0;
        uint64x2_t const increment = vdupq_n_u64(4);
        for (; i + 64 <= length; i += 64) {
            generated[0] = sz_emulate_aesenc_u64x2_neon_(inputs[0], keys[0]);
            generated[1] = sz_emulate_aesenc_u64x2_neon_(inputs[1], keys[1]);
            generated[2] = sz_emulate_aesenc_u64x2_neon_(inputs[2], keys[2]);
            generated[3] = sz_emulate_aesenc_u64x2_neon_(inputs[3], keys[3]);
            vst1q_u64((sz_u64_t *)(text + i + 0), generated[0]);
            vst1q_u64((sz_u64_t *)(text + i + 16), generated[1]);
            vst1q_u64((sz_u64_t *)(text + i + 32), generated[2]);
            vst1q_u64((sz_u64_t *)(text + i + 48), generated[3]);
            inputs[0] = vaddq_u64(inputs[0], increment);
            inputs[1] = vaddq_u64(inputs[1], increment);
            inputs[2] = vaddq_u64(inputs[2], increment);
            inputs[3] = vaddq_u64(inputs[3], increment);
        }

        // Handle the tail of the buffer.
        {
            generated[0] = sz_emulate_aesenc_u64x2_neon_(inputs[0], keys[0]);
            generated[1] = sz_emulate_aesenc_u64x2_neon_(inputs[1], keys[1]);
            generated[2] = sz_emulate_aesenc_u64x2_neon_(inputs[2], keys[2]);
            generated[3] = sz_emulate_aesenc_u64x2_neon_(inputs[3], keys[3]);
            for (sz_size_t j = 0; i < length; ++i, ++j) text[i] = ((sz_u8_t *)generated)[j];
        }
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_NEON
#pragma endregion // NEON AES Implementation

/*  Implementation of the string search algorithms using the Arm SVE variable-length registers,
 *  available in Arm v9 processors, like in Apple M4+ and Graviton 3+ CPUs.
 */
#pragma region SVE Implementation
#if SZ_USE_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#endif

SZ_PUBLIC sz_u64_t sz_bytesum_sve(sz_cptr_t text, sz_size_t length) {
    sz_u64_t sum = 0;
    sz_size_t progress = 0;
    sz_size_t const vector_length = svcntb();
    // SVE doesn't have widening accumulation, so we reduce across each loaded vector
    for (; progress < length; progress += vector_length) {
        svbool_t progress_mask = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)length);
        svuint8_t text_vec = svld1_u8(progress_mask, (sz_u8_t const *)(text + progress));
        sum += svaddv_u8(progress_mask, text_vec);
    }
    return sum;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_SVE
#pragma endregion // SVE Implementation

/*  Implementation of the string search algorithms using the Arm SVE2 variable-length registers,
 *  available in Arm v9 processors, like in AWS Graviton 4+ CPUs.
 *
 *  Our AES hashing algorithms are implemented differently depending on the size of the size of the input.
 *  Given how SVE+AES extensions are structured, we have a separate implementation for different register sizes.
 *
 *  @see https://stackoverflow.com/a/73218637/2766161
 */
#pragma region SVE2 Implementation
#if SZ_USE_SVE2
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+sve2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+sve2")
#endif

SZ_PUBLIC sz_u64_t sz_bytesum_sve2(sz_cptr_t text, sz_size_t length) {
    sz_u64_t sum = 0;
    sz_size_t progress = 0;
    sz_size_t const vector_length = svcntb();
    // In SVE2 we have an instruction, that can add 8-bit elements in one operand to 16-bit elements in another.
    // Assuming the size mismatch, there 2 such instructions - for the top and bottom elements in each 8-bit pair.
    //
    // We can use that kind of logic to accelerate the inner loop, but we still need to reduce the 64-bit results.
    while (progress < length) {
        svuint16_t sum_u16_top = svdup_n_u16(0);
        svuint16_t sum_u16_bot = svdup_n_u16(0);
        // Assuming `u16` has a 256x wider range than `u8`, we can aggregate up to 256 lanes in each value.
        for (sz_size_t loop_index = 0; progress < length && loop_index < 256; progress += vector_length, ++loop_index) {
            svbool_t progress_mask = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)length);
            svuint8_t text_vec = svld1_u8(progress_mask, (sz_u8_t const *)(text + progress));
            sum_u16_top = svaddwb_u16(sum_u16_top, text_vec);
            sum_u16_bot = svaddwt_u16(sum_u16_bot, text_vec);
        }
        sum += svaddv_u16(svptrue_b16(), sum_u16_top);
        sum += svaddv_u16(svptrue_b16(), sum_u16_bot);
    }

    return sum;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_SVE
#pragma endregion // SVE2 Implementation

#pragma region SVE2 AES Implementation
#if SZ_USE_SVE2_AES
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+sve2+sve2-aes"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+sve2+sve2-aes")
#endif

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

/** @brief A variant of `sz_hash_sve2` for strings up to 16 bytes long - smallest SVE register size. */
SZ_PUBLIC sz_u64_t sz_hash_sve2_upto16_(sz_cptr_t text, sz_size_t length, sz_u64_t seed) {
    svuint8_t state_aes, state_sum, state_key;

    // To load and store the seed, we don't even need a `svwhilelt_b64(0, 2)`.
    state_key = svreinterpret_u8_u64(svdup_n_u64(seed));

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    svuint64_t pi0 = svld1_u64(svptrue_b64(), pi);
    svuint64_t pi1 = svld1_u64(svptrue_b64(), pi + 8);
    state_aes = sveor_u8_x(svptrue_b8(), state_key, svreinterpret_u8_u64(pi0));
    state_sum = sveor_u8_x(svptrue_b8(), state_key, svreinterpret_u8_u64(pi1));

    // We will only use the first 128 bits of the shuffle mask
    svuint8_t const shuffle_mask = svld1_u8(svptrue_b8(), sz_hash_u8x16x4_shuffle_());

    // This is our best case for SVE2 dominance over NEON - we can load the data in one go with a predicate.
    svuint8_t block = svld1_u8(svwhilelt_b8((sz_u64_t)0, (sz_u64_t)length), (sz_u8_t const *)text);
    // One round of hashing logic
    state_aes = sz_emulate_aesenc_u8x16_sve2_(state_aes, block);
    svuint8_t sum_shuffled = svtbl_u8(state_sum, shuffle_mask);
    state_sum = svreinterpret_u8_u64(
        svadd_u64_x(svptrue_b64(), svreinterpret_u64_u8(sum_shuffled), svreinterpret_u64_u8(block)));

    // Now mix, folding the length into the key
    svuint64_t key_with_length = svadd_u64_x(svptrue_b64(), svreinterpret_u64_u8(state_key), svdupq_n_u64(length, 0));
    // Combine the "sum" and the "AES" blocks
    svuint8_t mixed = sz_emulate_aesenc_u8x16_sve2_(state_sum, state_aes);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    svuint8_t mixed_in_register = sz_emulate_aesenc_u8x16_sve2_(
        sz_emulate_aesenc_u8x16_sve2_(mixed, svreinterpret_u8_u64(key_with_length)), mixed);
    // Extract the low 64 bits
    svuint64_t mixed_in_register_u64 = svreinterpret_u64_u8(mixed_in_register);
    return svlasta_u64(svpfalse_b(), mixed_in_register_u64); // Extract the first element
}

SZ_PUBLIC void sz_hash_state_init_sve2(sz_hash_state_t *state, sz_u64_t seed) { sz_hash_state_init_neon(state, seed); }

SZ_PUBLIC void sz_hash_state_update_sve2(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_hash_state_update_neon(state, text, length);
}

SZ_PUBLIC sz_u64_t sz_hash_state_digest_sve2(sz_hash_state_t const *state) { //
    return sz_hash_state_digest_neon(state);
}

SZ_PUBLIC sz_u64_t sz_hash_sve2(sz_cptr_t text, sz_size_t length, sz_u64_t seed) {
    if (length <= 16) { return sz_hash_sve2_upto16_(text, length, seed); }
    else { return sz_hash_neon(text, length, seed); }
}

SZ_PUBLIC void sz_fill_random_sve2(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_fill_random_neon(text, length, nonce);
}

#if 0
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
    sz_u8_t const *shuffle_mask = sz_hash_u8x16x4_shuffle_();
    svuint8_t const shuffle_mask = svreinterpret_u8_u64(svdupq_n_u64( //
        *(sz_u64_t const *)(shuffle_mask + 0),                        //
        *(sz_u64_t const *)(shuffle_mask + 8)));
    svuint8_t const sum_shuffled = svtbl_u8(state_sum, shuffle_mask);

    // Loop throughthe input until we process all the bytes
    sz_size_t const bytes_per_register = svcntb();
    sz_size_t const texts_per_register = bytes_per_register / 16;
    for (sz_size_t progress_bytes = 0; progress_bytes < 256; progress_bytes += bytes_per_register) {
        svuint8_t blocks =
            svld1_u8(svwhilelt_b8((sz_u64_t)progress_bytes, (sz_u64_t)256), (sz_u8_t const *)(&texts[0][0] + progress_bytes));

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
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_SVE2
#pragma endregion // SVE2 Implementation

/*  Pick the right implementation for the string search algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_u64_t sz_bytesum(sz_cptr_t text, sz_size_t length) {
#if SZ_USE_ICE
    return sz_bytesum_ice(text, length);
#elif SZ_USE_SKYLAKE
    return sz_bytesum_skylake(text, length);
#elif SZ_USE_HASWELL
    return sz_bytesum_haswell(text, length);
#elif SZ_USE_SVE2
    return sz_bytesum_sve2(text, length);
#elif SZ_USE_SVE
    return sz_bytesum_sve(text, length);
#elif SZ_USE_NEON
    return sz_bytesum_neon(text, length);
#else
    return sz_bytesum_serial(text, length);
#endif
}

SZ_DYNAMIC sz_u64_t sz_hash(sz_cptr_t text, sz_size_t length, sz_u64_t seed) {
#if SZ_USE_ICE
    return sz_hash_ice(text, length, seed);
#elif SZ_USE_SKYLAKE
    return sz_hash_skylake(text, length, seed);
#elif SZ_USE_HASWELL
    return sz_hash_haswell(text, length, seed);
#elif SZ_USE_SVE2_AES
    return sz_hash_sve2(text, length, seed);
#elif SZ_USE_NEON_AES
    return sz_hash_neon(text, length, seed);
#else
    return sz_hash_serial(text, length, seed);
#endif
}

SZ_DYNAMIC void sz_fill_random(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
#if SZ_USE_ICE
    sz_fill_random_ice(text, length, nonce);
#elif SZ_USE_SKYLAKE
    sz_fill_random_skylake(text, length, nonce);
#elif SZ_USE_HASWELL
    sz_fill_random_haswell(text, length, nonce);
#elif SZ_USE_SVE2_AES
    sz_fill_random_sve2(text, length, nonce);
#elif SZ_USE_NEON_AES
    sz_fill_random_neon(text, length, nonce);
#else
    sz_fill_random_serial(text, length, nonce);
#endif
}

SZ_DYNAMIC void sz_hash_state_init(sz_hash_state_t *state, sz_u64_t seed) {
#if SZ_USE_ICE
    sz_hash_state_init_ice(state, seed);
#elif SZ_USE_SKYLAKE
    sz_hash_state_init_skylake(state, seed);
#elif SZ_USE_HASWELL
    sz_hash_state_init_haswell(state, seed);
#elif SZ_USE_SVE2_AES
    sz_hash_state_init_sve2(state, seed);
#elif SZ_USE_NEON_AES
    sz_hash_state_init_neon(state, seed);
#else
    sz_hash_state_init_serial(state, seed);
#endif
}

SZ_DYNAMIC void sz_hash_state_update(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
#if SZ_USE_ICE
    sz_hash_state_update_ice(state, text, length);
#elif SZ_USE_SKYLAKE
    sz_hash_state_update_skylake(state, text, length);
#elif SZ_USE_HASWELL
    sz_hash_state_update_haswell(state, text, length);
#elif SZ_USE_SVE2_AES
    sz_hash_state_update_sve2(state, text, length);
#elif SZ_USE_NEON_AES
    sz_hash_state_update_neon(state, text, length);
#else
    sz_hash_state_update_serial(state, text, length);
#endif
}

SZ_DYNAMIC sz_u64_t sz_hash_state_digest(sz_hash_state_t const *state) {
#if SZ_USE_ICE
    return sz_hash_state_digest_ice(state);
#elif SZ_USE_SKYLAKE
    return sz_hash_state_digest_skylake(state);
#elif SZ_USE_HASWELL
    return sz_hash_state_digest_haswell(state);
#elif SZ_USE_SVE2_AES
    return sz_hash_state_digest_sve2(state);
#elif SZ_USE_NEON_AES
    return sz_hash_state_digest_neon(state);
#else
    return sz_hash_state_digest_serial(state);
#endif
}

#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_HASH_H_
