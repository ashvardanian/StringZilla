/**
 *  @brief Hardware-accelerated non-cryptographic string hashing and checksums.
 *  @file include/stringzilla/hash.h
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

#include "stringzilla/types.h"

#include "stringzilla/compare.h" // `sz_equal`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Computes the 64-bit check-sum of bytes in a string.
 *         Similar to `std::ranges::accumulate`.
 *
 *  @param text String to aggregate.
 *  @param length Number of bytes in the text.
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
 *  @note Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa sz_bytesum_serial, sz_bytesum_haswell, sz_bytesum_skylake, sz_bytesum_icelake, sz_bytesum_neon,
 *      sz_bytesum_sve, sz_bytesum_sve2, sz_bytesum_v128, sz_bytesum_v128relaxed, sz_bytesum_rvv, sz_bytesum_lasx,
 *      sz_bytesum_powervsx
 */
SZ_DYNAMIC sz_u64_t sz_bytesum(sz_cptr_t text, sz_size_t length);

/**
 *  @brief Computes the 64-bit unsigned hash of a string similar to @b `std::hash` in C++.
 *         It's not cryptographically secure, but it's fast and provides a good distribution.
 *         It passes the SMHasher suite by Austin Appleby with no collisions, even with `--extra` flag.
 *  @see HASH.md for a detailed explanation of the algorithm.
 *
 *  @param text String to hash.
 *  @param length Number of bytes in the text.
 *  @param seed 64-bit unsigned seed for the hash.
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
 *  @note Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa sz_hash_serial, sz_hash_westmere, sz_hash_skylake, sz_hash_icelake, sz_hash_neon, sz_hash_sve2,
 *      sz_hash_v128, sz_hash_rvv, sz_hash_lasx, sz_hash_powervsx
 *
 *  @note The algorithm must provide the same output on all platforms in both single-shot and incremental modes.
 *  @sa sz_hash_state_init, sz_hash_state_update, sz_hash_state_digest
 */
SZ_DYNAMIC sz_u64_t sz_hash(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/**
 *  @brief A Pseudorandom Number Generator (PRNG), inspired the AES-CTR-128 algorithm,
 *         but using only one round of AES mixing as opposed to "NIST SP 800-90A".
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
 *  @param text Output string buffer to be populated.
 *  @param length Number of bytes in the string.
 *  @param nonce "Number used ONCE" to ensure uniqueness of produced blocks.
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
 *  @note Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa sz_fill_random_serial, sz_fill_random_westmere, sz_fill_random_skylake, sz_fill_random_icelake,
 *      sz_fill_random_neon, sz_fill_random_sve2, sz_fill_random_v128, sz_fill_random_rvv, sz_fill_random_lasx,
 *      sz_fill_random_powervsx
 */
SZ_DYNAMIC void sz_fill_random(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/**
 *  @brief The state for incremental construction of a hash.
 *  @see sz_hash_state_init, sz_hash_state_update, sz_hash_state_digest.
 *
 *  @note Uses `packed` attribute to allow placement at arbitrary addresses without UBSAN warnings.
 *        This struct uses plain byte arrays to avoid implicit alignment requirements from SIMD types.
 *        The layout matches sz_hash_state_internal_t_ for safe casting between them.
 */
#if defined(_MSC_VER)
#pragma pack(push, 1)
typedef struct sz_hash_state_t {
    sz_u8_t aes[64]; // 64 bytes, equivalent to sz_u512_vec_t
    sz_u8_t sum[64]; // 64 bytes, equivalent to sz_u512_vec_t
    sz_u8_t ins[64]; // 64 bytes, equivalent to sz_u512_vec_t
    sz_u8_t key[16]; // 16 bytes, equivalent to sz_u128_vec_t
    sz_size_t ins_length;
} sz_hash_state_t;
#pragma pack(pop)
#else
typedef struct __attribute__((packed)) sz_hash_state_t {
    sz_u8_t aes[64]; // 64 bytes, equivalent to sz_u512_vec_t
    sz_u8_t sum[64]; // 64 bytes, equivalent to sz_u512_vec_t
    sz_u8_t ins[64]; // 64 bytes, equivalent to sz_u512_vec_t
    sz_u8_t key[16]; // 16 bytes, equivalent to sz_u128_vec_t
    sz_size_t ins_length;
} sz_hash_state_t;
#endif

/**
 *  @brief Internal aligned version of sz_hash_state_t using SIMD union types.
 *         This is layout-compatible with sz_hash_state_t but provides convenient
 *         SIMD vector accessors for internal implementations.
 */
typedef struct sz_hash_state_internal_t_ {
    sz_u512_vec_t aes;
    sz_u512_vec_t sum;
    sz_u512_vec_t ins;
    sz_u128_vec_t key;
    sz_size_t ins_length;
} sz_hash_state_internal_t_;

typedef struct sz_hash_minimal_t_ {
    sz_u128_vec_t aes;
    sz_u128_vec_t sum;
    sz_u128_vec_t key;
} sz_hash_minimal_t_;

/**
 *  @brief The state for incremental construction of a SHA256 hash.
 *  @see sz_sha256_state_init, sz_sha256_state_update, sz_sha256_state_digest.
 */
typedef struct sz_sha256_state_t {
    sz_u32_t hash[8];       ///< Current hash state: 8x 32-bit values
    sz_u8_t block[64];      ///< 64-byte message block buffer
    sz_size_t block_length; ///< Current bytes in block (0-63)
    sz_u64_t total_length;  ///< Total message length in bytes
} sz_sha256_state_t;

/**
 *  @brief Initializes the state for incremental construction of a hash.
 *
 *  @param state The state to initialize.
 *  @param seed The 64-bit unsigned seed for the hash.
 */
SZ_DYNAMIC void sz_hash_state_init(sz_hash_state_t *state, sz_u64_t seed);

/**
 *  @brief Updates the state with new data.
 *
 *  @param state The state to stream.
 *  @param text The new data to include in the hash.
 *  @param length The number of bytes in the new data.
 */
SZ_DYNAMIC void sz_hash_state_update(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/**
 *  @brief Finalizes the immutable state and returns the hash.
 *
 *  @param state The state to fold.
 *  @return The 64-bit hash value.
 */
SZ_DYNAMIC sz_u64_t sz_hash_state_digest(sz_hash_state_t const *state);

/**
 *  @brief Initializes the state for incremental SHA256 hashing.
 *
 *  @param state The state to initialize.
 */
SZ_DYNAMIC void sz_sha256_state_init(sz_sha256_state_t *state);

/**
 *  @brief Updates the SHA256 state with new data.
 *
 *  @param state The state to update.
 *  @param data The new data to hash.
 *  @param length The number of bytes in the new data.
 */
SZ_DYNAMIC void sz_sha256_state_update(sz_sha256_state_t *state, sz_cptr_t data, sz_size_t length);

/**
 *  @brief Finalizes the SHA256 state and returns the hash.
 *
 *  @param state The state to finalize.
 *  @param digest Output buffer for the 32-byte (256-bit) hash.
 */
SZ_DYNAMIC void sz_sha256_state_digest(sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(32)]);

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_serial(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash */
SZ_PUBLIC SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_serial(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/** @copydoc sz_fill_random */
SZ_PUBLIC void sz_fill_random_serial(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
SZ_PUBLIC void sz_hash_state_init_serial(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_update */
SZ_PUBLIC void sz_hash_state_update_serial(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_digest */
SZ_PUBLIC sz_u64_t sz_hash_state_digest_serial(sz_hash_state_t const *state);

#if SZ_USE_WESTMERE

/** @copydoc sz_hash */
SZ_PUBLIC SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_westmere(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/** @copydoc sz_fill_random */
SZ_PUBLIC void sz_fill_random_westmere(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
SZ_PUBLIC void sz_hash_state_init_westmere(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_update */
SZ_PUBLIC void sz_hash_state_update_westmere(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_digest */
SZ_PUBLIC sz_u64_t sz_hash_state_digest_westmere(sz_hash_state_t const *state);

#endif

#if SZ_USE_GOLDMONT

/** @copydoc sz_sha256_state_init */
SZ_PUBLIC void sz_sha256_state_init_goldmont(sz_sha256_state_t *state);

/** @copydoc sz_sha256_state_update */
SZ_PUBLIC void sz_sha256_state_update_goldmont(sz_sha256_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_sha256_state_digest */
SZ_PUBLIC void sz_sha256_state_digest_goldmont(sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(32)]);

#endif

#if SZ_USE_HASWELL

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_haswell(sz_cptr_t text, sz_size_t length);

#endif

#if SZ_USE_SKYLAKE

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_skylake(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash */
SZ_PUBLIC SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_skylake(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/** @copydoc sz_fill_random */
SZ_PUBLIC void sz_fill_random_skylake(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
SZ_PUBLIC void sz_hash_state_init_skylake(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_update */
SZ_PUBLIC void sz_hash_state_update_skylake(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_digest */
SZ_PUBLIC sz_u64_t sz_hash_state_digest_skylake(sz_hash_state_t const *state);

#endif

#if SZ_USE_ICELAKE

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_icelake(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash */
SZ_PUBLIC SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_icelake(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/** @copydoc sz_fill_random */
SZ_PUBLIC void sz_fill_random_icelake(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
SZ_PUBLIC void sz_hash_state_init_icelake(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_update */
SZ_PUBLIC void sz_hash_state_update_icelake(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_digest */
SZ_PUBLIC sz_u64_t sz_hash_state_digest_icelake(sz_hash_state_t const *state);

/** @copydoc sz_sha256_state_init */
SZ_PUBLIC void sz_sha256_state_init_icelake(sz_sha256_state_t *state);

/** @copydoc sz_sha256_state_update */
SZ_PUBLIC void sz_sha256_state_update_icelake(sz_sha256_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_sha256_state_digest */
SZ_PUBLIC void sz_sha256_state_digest_icelake(sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(32)]);

#endif

#if SZ_USE_NEON

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_neon(sz_cptr_t text, sz_size_t length);

#endif

#if SZ_USE_NEONAES

/** @copydoc sz_hash */
SZ_PUBLIC SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_neon(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/** @copydoc sz_fill_random */
SZ_PUBLIC void sz_fill_random_neon(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
SZ_PUBLIC void sz_hash_state_init_neon(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_update */
SZ_PUBLIC void sz_hash_state_update_neon(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_digest */
SZ_PUBLIC sz_u64_t sz_hash_state_digest_neon(sz_hash_state_t const *state);

#endif

#if SZ_USE_NEONSHA

/** @copydoc sz_sha256_state_init */
SZ_PUBLIC void sz_sha256_state_init_neon(sz_sha256_state_t *state);

/** @copydoc sz_sha256_state_update */
SZ_PUBLIC void sz_sha256_state_update_neon(sz_sha256_state_t *state, sz_cptr_t data, sz_size_t length);

/** @copydoc sz_sha256_state_digest */
SZ_PUBLIC void sz_sha256_state_digest_neon(sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(32)]);

#endif

#if SZ_USE_SVE

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_sve(sz_cptr_t text, sz_size_t length);

#endif

#if SZ_USE_SVE2

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_sve2(sz_cptr_t text, sz_size_t length);

#endif

#if SZ_USE_SVE2AES

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
 *  @brief Compares the state of two running hashes.
 *  @note The current content of the `ins` buffer and its length is ignored.
 */
SZ_PUBLIC sz_bool_t sz_hash_state_equal(sz_hash_state_t const *lhs, sz_hash_state_t const *rhs) {
    // Compare byte-by-byte using sz_equal (safe for packed struct)
    if (!sz_equal((sz_cptr_t)lhs->aes, (sz_cptr_t)rhs->aes, 64)) return sz_false_k;
    if (!sz_equal((sz_cptr_t)lhs->sum, (sz_cptr_t)rhs->sum, 64)) return sz_false_k;
    if (!sz_equal((sz_cptr_t)lhs->key, (sz_cptr_t)rhs->key, 16)) return sz_false_k;
    return sz_true_k;
}

#pragma endregion // Helper Methods

#include "stringzilla/hash/serial.h"
#include "stringzilla/hash/westmere.h"
#include "stringzilla/hash/goldmont.h"
#include "stringzilla/hash/haswell.h"
#include "stringzilla/hash/skylake.h"
#include "stringzilla/hash/icelake.h"
#include "stringzilla/hash/neon.h"
#include "stringzilla/hash/neonaes.h"
#include "stringzilla/hash/neonsha.h"
#include "stringzilla/hash/sve.h"
#include "stringzilla/hash/sve2.h"
#include "stringzilla/hash/sve2aes.h"
#include "stringzilla/hash/v128relaxed.h"
#include "stringzilla/hash/v128.h"
#include "stringzilla/hash/rvv.h"
#include "stringzilla/hash/lasx.h"
#include "stringzilla/hash/powervsx.h"

/*  Pick the right implementation for the string search algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_u64_t sz_bytesum(sz_cptr_t text, sz_size_t length) {
#if SZ_USE_V128RELAXED
    return sz_bytesum_v128relaxed(text, length);
#elif SZ_USE_V128
    return sz_bytesum_v128(text, length);
#elif SZ_USE_RVV
    return sz_bytesum_rvv(text, length);
#elif SZ_USE_LASX
    return sz_bytesum_lasx(text, length);
#elif SZ_USE_POWERVSX
    return sz_bytesum_powervsx(text, length);
#elif SZ_USE_ICELAKE
    return sz_bytesum_icelake(text, length);
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
#if SZ_USE_V128RELAXED
    return sz_hash_v128relaxed(text, length, seed);
#elif SZ_USE_V128
    return sz_hash_v128(text, length, seed);
#elif SZ_USE_RVV
    return sz_hash_rvv(text, length, seed);
#elif SZ_USE_LASX
    return sz_hash_lasx(text, length, seed);
#elif SZ_USE_POWERVSX
    return sz_hash_powervsx(text, length, seed);
#elif SZ_USE_ICELAKE
    return sz_hash_icelake(text, length, seed);
#elif SZ_USE_SKYLAKE
    return sz_hash_skylake(text, length, seed);
#elif SZ_USE_WESTMERE
    return sz_hash_westmere(text, length, seed);
#elif SZ_USE_SVE2AES
    return sz_hash_sve2(text, length, seed);
#elif SZ_USE_NEONAES
    return sz_hash_neon(text, length, seed);
#else
    return sz_hash_serial(text, length, seed);
#endif
}

SZ_DYNAMIC void sz_fill_random(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
#if SZ_USE_V128RELAXED
    sz_fill_random_v128relaxed(text, length, nonce);
#elif SZ_USE_V128
    sz_fill_random_v128(text, length, nonce);
#elif SZ_USE_RVV
    sz_fill_random_rvv(text, length, nonce);
#elif SZ_USE_LASX
    sz_fill_random_lasx(text, length, nonce);
#elif SZ_USE_POWERVSX
    sz_fill_random_powervsx(text, length, nonce);
#elif SZ_USE_ICELAKE
    sz_fill_random_icelake(text, length, nonce);
#elif SZ_USE_SKYLAKE
    sz_fill_random_skylake(text, length, nonce);
#elif SZ_USE_WESTMERE
    sz_fill_random_westmere(text, length, nonce);
#elif SZ_USE_SVE2AES
    sz_fill_random_sve2(text, length, nonce);
#elif SZ_USE_NEONAES
    sz_fill_random_neon(text, length, nonce);
#else
    sz_fill_random_serial(text, length, nonce);
#endif
}

SZ_DYNAMIC void sz_hash_state_init(sz_hash_state_t *state, sz_u64_t seed) {
#if SZ_USE_V128RELAXED
    sz_hash_state_init_v128relaxed(state, seed);
#elif SZ_USE_V128
    sz_hash_state_init_v128(state, seed);
#elif SZ_USE_RVV
    sz_hash_state_init_rvv(state, seed);
#elif SZ_USE_LASX
    sz_hash_state_init_lasx(state, seed);
#elif SZ_USE_POWERVSX
    sz_hash_state_init_powervsx(state, seed);
#elif SZ_USE_ICELAKE
    sz_hash_state_init_icelake(state, seed);
#elif SZ_USE_SKYLAKE
    sz_hash_state_init_skylake(state, seed);
#elif SZ_USE_WESTMERE
    sz_hash_state_init_westmere(state, seed);
#elif SZ_USE_SVE2AES
    sz_hash_state_init_sve2(state, seed);
#elif SZ_USE_NEONAES
    sz_hash_state_init_neon(state, seed);
#else
    sz_hash_state_init_serial(state, seed);
#endif
}

SZ_DYNAMIC void sz_hash_state_update(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
#if SZ_USE_V128RELAXED
    sz_hash_state_update_v128relaxed(state, text, length);
#elif SZ_USE_V128
    sz_hash_state_update_v128(state, text, length);
#elif SZ_USE_RVV
    sz_hash_state_update_rvv(state, text, length);
#elif SZ_USE_LASX
    sz_hash_state_update_lasx(state, text, length);
#elif SZ_USE_POWERVSX
    sz_hash_state_update_powervsx(state, text, length);
#elif SZ_USE_ICELAKE
    sz_hash_state_update_icelake(state, text, length);
#elif SZ_USE_SKYLAKE
    sz_hash_state_update_skylake(state, text, length);
#elif SZ_USE_WESTMERE
    sz_hash_state_update_westmere(state, text, length);
#elif SZ_USE_SVE2AES
    sz_hash_state_update_sve2(state, text, length);
#elif SZ_USE_NEONAES
    sz_hash_state_update_neon(state, text, length);
#else
    sz_hash_state_update_serial(state, text, length);
#endif
}

SZ_DYNAMIC sz_u64_t sz_hash_state_digest(sz_hash_state_t const *state) {
#if SZ_USE_V128RELAXED
    return sz_hash_state_digest_v128relaxed(state);
#elif SZ_USE_V128
    return sz_hash_state_digest_v128(state);
#elif SZ_USE_RVV
    return sz_hash_state_digest_rvv(state);
#elif SZ_USE_LASX
    return sz_hash_state_digest_lasx(state);
#elif SZ_USE_POWERVSX
    return sz_hash_state_digest_powervsx(state);
#elif SZ_USE_ICELAKE
    return sz_hash_state_digest_icelake(state);
#elif SZ_USE_SKYLAKE
    return sz_hash_state_digest_skylake(state);
#elif SZ_USE_WESTMERE
    return sz_hash_state_digest_westmere(state);
#elif SZ_USE_SVE2AES
    return sz_hash_state_digest_sve2(state);
#elif SZ_USE_NEONAES
    return sz_hash_state_digest_neon(state);
#else
    return sz_hash_state_digest_serial(state);
#endif
}

SZ_DYNAMIC void sz_sha256_state_init(sz_sha256_state_t *state) {
#if SZ_USE_V128
    sz_sha256_state_init_v128(state);
#elif SZ_USE_RVV
    sz_sha256_state_init_rvv(state);
#elif SZ_USE_LASX
    sz_sha256_state_init_lasx(state);
#elif SZ_USE_POWERVSX
    sz_sha256_state_init_powervsx(state);
#elif SZ_USE_NEONSHA
    sz_sha256_state_init_neon(state);
#elif SZ_USE_ICELAKE
    sz_sha256_state_init_icelake(state);
#elif SZ_USE_GOLDMONT
    sz_sha256_state_init_goldmont(state);
#else
    sz_sha256_state_init_serial(state);
#endif
}

SZ_DYNAMIC void sz_sha256_state_update(sz_sha256_state_t *state, sz_cptr_t data, sz_size_t length) {
#if SZ_USE_V128
    sz_sha256_state_update_v128(state, data, length);
#elif SZ_USE_RVV
    sz_sha256_state_update_rvv(state, data, length);
#elif SZ_USE_LASX
    sz_sha256_state_update_lasx(state, data, length);
#elif SZ_USE_POWERVSX
    sz_sha256_state_update_powervsx(state, data, length);
#elif SZ_USE_NEONSHA
    sz_sha256_state_update_neon(state, data, length);
#elif SZ_USE_ICELAKE
    sz_sha256_state_update_icelake(state, data, length);
#elif SZ_USE_GOLDMONT
    sz_sha256_state_update_goldmont(state, data, length);
#else
    sz_sha256_state_update_serial(state, data, length);
#endif
}

SZ_DYNAMIC void sz_sha256_state_digest(sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(32)]) {
#if SZ_USE_V128
    sz_sha256_state_digest_v128(state, digest);
#elif SZ_USE_RVV
    sz_sha256_state_digest_rvv(state, digest);
#elif SZ_USE_LASX
    sz_sha256_state_digest_lasx(state, digest);
#elif SZ_USE_POWERVSX
    sz_sha256_state_digest_powervsx(state, digest);
#elif SZ_USE_NEONSHA
    sz_sha256_state_digest_neon(state, digest);
#elif SZ_USE_ICELAKE
    sz_sha256_state_digest_icelake(state, digest);
#elif SZ_USE_GOLDMONT
    sz_sha256_state_digest_goldmont(state, digest);
#else
    sz_sha256_state_digest_serial(state, digest);
#endif
}

#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_HASH_H_
