/**
 *  @brief  Hardware-accelerated non-cryptographic string hashing and checksums.
 *  @file   hash.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs:
 *
 *  - `sz_bytesum` - for byte-level 64-bit unsigned byte-level checksums.
 *  - `sz_hash` - for 64-bit single-shot hashing using AES instructions.
 *  - `sz_hash_state_init`, `sz_hash_state_stream`, `sz_hash_state_fold` - for incremental hashing.
 *  - `sz_generate` - for populating buffers with pseudo-random noise using AES instructions.
 *
 *  Why the hell do we need a yet another hashing library?!
 *  Turns out, most existing libraries have noticeable constraints. Try finding a library that:
 *
 *  - Outputs 64-bit or 128-bit hashes and passes the SMHasher test suite.
 *  - Is fast for both short and long strings.
 *  - Supports incremental @b (streaming) hashing, when the data arrives in chunks.
 *  - Supports custom seeds hashes and secret strings for security.
 *  - Provides dynamic dispatch for different architectures to simplify deployment.
 *  - Uses modern SIMD, including not just AVX2 and NEON, but also AVX-512 and SVE2.
 *  - Documents its logic and guarantees the same output across different platforms.
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
 *  StringZilla uses a scheme more similar to the "aHash" library, utilizing the AES extensions, that provide
 *  a remarkable level of "mixing per cycle" and are broadly available on modern CPUs. Similar to "aHash", they
 *  are combined with "shuffle & add" instructions to provide a high level of entropy in the output. That operation
 *  is practically free, as many modern CPUs will dispatch them on different ports. On x86, for example:
 *
 *  - `VAESDEC` (ZMM, ZMM, ZMM)`:
 *    - on Intel Ice Lake: 5 cycles on port 0.
 *    - On AMD Zen4: 4 cycles on ports 0 or 1.
 *  - `VPSHUFB_Z (ZMM, K, ZMM, ZMM)`
 *    - on Intel Ice Lake: 3 cycles on port 5.
 *    - On AMD Zen4: 2 cycles on ports 1 or 2.
 *  - `VPADDQ (ZMM, ZMM, ZMM)`:
 *    - on Intel Ice Lake: 1 cycle on ports 0 or 5.
 *    - On AMD Zen4: 1 cycle on ports 0, 1, 2, 3.
 *
 *  Unlike "aHash", on long inputs, we use a procedure that is more vector-friendly on modern servers.
 *  Unlike "aHash", we don't load interleaved memory regions, making vectorized variant more similar to sequential.
 *  On platforms like Skylake-X or newer, we also benefit from masked loads.
 *
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
 *  @sa     sz_hash_serial, sz_hash_haswell, sz_hash_skylake, sz_hash_ice, sz_hash_neon
 *
 *  @note   The algorithm must provide the same output on all platforms in both single-shot and incremental modes.
 *  @sa     sz_hash_state_init, sz_hash_state_stream, sz_hash_state_fold
 */
SZ_DYNAMIC sz_u64_t sz_hash(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/**
 *  @brief  A Pseudorandom Number Generator (PRNG), inspired the AES-CTR-128 algorithm,
 *          but using only one round of AES mixing as opposed to "NIST SP 800-90A".
 *
 *  CTR_DRBG (CounTeR mode Deterministic Random Bit Generator) appears secure and indistinguishable from a true
 *  random source when AES is used as the underlying block cipher and 112 bits are taken from this PRNG.
 *  When AES is used as the underlying block cipher and 128 bits are taken from each instantiation,
 *  the required security level is delivered with the caveat that a 128-bit cipher's output in
 *  counter mode can be distinguished from a true RNG.
 *
 *  In this case, it doesn't apply, as we only use one round of AES mixing. We also don't expose a separate "key",
 *  only a "nonce", to keep the API simple.
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
 *          sz_generate(first_buffer, 5, 0);
 *          sz_generate(second_buffer, 5, 0); //? Same nonce will produce the same output
 *          return sz_bytesum(first_buffer, 5) == sz_bytesum(second_buffer, 5) ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note   Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa     sz_generate_serial, sz_generate_haswell, sz_generate_skylake, sz_generate_ice, sz_generate_neon
 */
SZ_DYNAMIC void sz_generate(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/**
 *  @brief  The state for incremental construction of a hash.
 *  @see    sz_hash_state_init, sz_hash_state_stream, sz_hash_state_fold.
 */
typedef struct sz_hash_state_t {
    sz_u512_vec_t aes;
    sz_u512_vec_t sum;
    sz_u512_vec_t key;

    sz_u512_vec_t ins;
    sz_size_t ins_length;
} sz_hash_state_t;

typedef struct _sz_hash_minimal_t {
    sz_u128_vec_t aes;
    sz_u128_vec_t sum;
    sz_u128_vec_t key;
} _sz_hash_minimal_t;

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
SZ_DYNAMIC void sz_hash_state_stream(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/**
 *  @brief  Finalizes the immutable state and returns the hash.
 *
 *  @param[in] state The state to fold.
 *  @return The 64-bit hash value.
 */
SZ_DYNAMIC sz_u64_t sz_hash_state_fold(sz_hash_state_t const *state);

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_serial(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash */
SZ_PUBLIC sz_u64_t sz_hash_serial(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/** @copydoc sz_generate */
SZ_PUBLIC void sz_generate_serial(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
SZ_PUBLIC void sz_hash_state_init_serial(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_stream */
SZ_PUBLIC void sz_hash_state_stream_serial(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_fold */
SZ_PUBLIC sz_u64_t sz_hash_state_fold_serial(sz_hash_state_t const *state);

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_haswell(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash */
SZ_PUBLIC sz_u64_t sz_hash_haswell(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/** @copydoc sz_generate */
SZ_PUBLIC void sz_generate_haswell(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
SZ_PUBLIC void sz_hash_state_init_haswell(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_stream */
SZ_PUBLIC void sz_hash_state_stream_haswell(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_fold */
SZ_PUBLIC sz_u64_t sz_hash_state_fold_haswell(sz_hash_state_t const *state);

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_skylake(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash */
SZ_PUBLIC sz_u64_t sz_hash_skylake(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/** @copydoc sz_generate */
SZ_PUBLIC void sz_generate_skylake(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
SZ_PUBLIC void sz_hash_state_init_skylake(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_stream */
SZ_PUBLIC void sz_hash_state_stream_skylake(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_fold */
SZ_PUBLIC sz_u64_t sz_hash_state_fold_skylake(sz_hash_state_t const *state);

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_ice(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash */
SZ_PUBLIC sz_u64_t sz_hash_ice(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/** @copydoc sz_generate */
SZ_PUBLIC void sz_generate_ice(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
SZ_PUBLIC void sz_hash_state_init_ice(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_stream */
SZ_PUBLIC void sz_hash_state_stream_ice(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_fold */
SZ_PUBLIC sz_u64_t sz_hash_state_fold_ice(sz_hash_state_t const *state);

/** @copydoc sz_bytesum */
SZ_PUBLIC sz_u64_t sz_bytesum_neon(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash */
SZ_PUBLIC sz_u64_t sz_hash_neon(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/** @copydoc sz_generate */
SZ_PUBLIC void sz_generate_neon(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
SZ_PUBLIC void sz_hash_state_init_neon(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_stream */
SZ_PUBLIC void sz_hash_state_stream_neon(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_fold */
SZ_PUBLIC sz_u64_t sz_hash_state_fold_neon(sz_hash_state_t const *state);

#pragma endregion // Core API

#pragma region Serial Implementation

SZ_PUBLIC sz_u64_t sz_bytesum_serial(sz_cptr_t text, sz_size_t length) {
    sz_u64_t bytesum = 0;
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *text_end = text_u8 + length;
    for (; text_u8 != text_end; ++text_u8) bytesum += *text_u8;
    return bytesum;
}

SZ_PUBLIC sz_u64_t sz_hash_serial(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {
    sz_unused(start && length && seed);
    return 0;
}

SZ_PUBLIC void sz_generate_serial(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_unused(text && length && nonce);
}

SZ_PUBLIC void sz_hash_state_init_serial(sz_hash_state_t *state, sz_u64_t seed) { sz_unused(state && seed); }

SZ_PUBLIC void sz_hash_state_stream_serial(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_unused(state && text && length);
}

SZ_PUBLIC sz_u64_t sz_hash_state_fold_serial(sz_hash_state_t const *state) {
    sz_unused(state);
    return 0;
}

#pragma endregion // Serial Implementation

/*  AVX2 implementation of the string search algorithms for Haswell processors and newer.
 *  Very minimalistic (compared to AVX-512), but still faster than the serial implementation.
 */
#pragma region Haswell Implementation
#if SZ_USE_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2")
#pragma clang attribute push(__attribute__((target("avx3332"))), apply_to = function)

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
                _sz_assert(body_length == 32);
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

SZ_INTERNAL void _sz_hash_minimal_init_haswell(_sz_hash_minimal_t *state, sz_u64_t seed) {
    __m128i seed_vec = _mm_set1_epi64x(seed);
    __m128i pi0 = _mm_set_epi64x(0x13198a2e03707344ull, 0x243f6a8885a308d3ull);
    __m128i pi1 = _mm_set_epi64x(0x082efa98ec4e6c89ull, 0xa4093822299f31d0ull);
    // XOR the user-supplied keys with the two "pi" constants
    __m128i k1 = _mm_xor_si128(seed_vec, pi0);
    __m128i k2 = _mm_xor_si128(seed_vec, pi1);
    // Export the keys to the state
    state->aes.xmm = k1;
    state->sum.xmm = k2;
    state->key.xmm = _mm_xor_si128(pi0, pi1);
}

SZ_INTERNAL sz_u64_t _sz_hash_minimal_finalize_haswell(_sz_hash_minimal_t const *state) {
    // Combine the sum and the AES block
    __m128i mixed_registers = _mm_aesenc_si128(state->sum.xmm, state->aes.xmm);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    __m128i mixed_within_register =
        _mm_aesdec_si128(_mm_aesdec_si128(mixed_registers, state->key.xmm), mixed_registers);
    // Extract the low 64 bits
    return _mm_cvtsi128_si64(mixed_within_register);
}

SZ_INTERNAL void _sz_hash_minimal_update_haswell(_sz_hash_minimal_t *state, __m128i block) {
    // This shuffle mask is identical to "aHash":
    __m128i const shuffle_mask = _mm_set_epi8(          //
        0x04, 0x0b, 0x09, 0x06, 0x08, 0x0d, 0x0f, 0x05, //
        0x0e, 0x03, 0x01, 0x0c, 0x00, 0x07, 0x0a, 0x02);
    state->aes.xmm = _mm_aesdec_si128(state->aes.xmm, block);
    state->sum.xmm = _mm_add_epi64(_mm_shuffle_epi8(state->sum.xmm, shuffle_mask), block);
}

SZ_PUBLIC void sz_hash_state_init_haswell(sz_hash_state_t *state, sz_u64_t seed) {
    __m128i seed_vec = _mm_set1_epi64x(seed);
    __m128i pi0 = _mm_set_epi64x(0x13198a2e03707344ull, 0x243f6a8885a308d3ull);
    __m128i pi1 = _mm_set_epi64x(0x082efa98ec4e6c89ull, 0xa4093822299f31d0ull);
    // XOR the user-supplied keys with the two "pi" constants
    __m128i k1 = _mm_xor_si128(seed_vec, pi0);
    __m128i k2 = _mm_xor_si128(seed_vec, pi1);
    // Export the keys to the state
    state->aes.xmms[0] = state->aes.xmms[1] = state->aes.xmms[2] = state->aes.xmms[3] = k1;
    state->sum.xmms[0] = state->sum.xmms[1] = state->sum.xmms[2] = state->sum.xmms[3] = k2;
    state->key.xmms[0] = state->key.xmms[1] = state->key.xmms[2] = state->key.xmms[3] = _mm_xor_si128(pi0, pi1);
    state->ins_length = 0;
}

SZ_INTERNAL void _sz_hash_state_update_haswell(sz_hash_state_t *state, __m128i block0, __m128i block1, __m128i block2,
                                               __m128i block3) {
    // This shuffle mask is identical to "aHash":
    __m128i const shuffle_mask = _mm_set_epi8(          //
        0x04, 0x0b, 0x09, 0x06, 0x08, 0x0d, 0x0f, 0x05, //
        0x0e, 0x03, 0x01, 0x0c, 0x00, 0x07, 0x0a, 0x02);
    state->aes.xmms[0] = _mm_aesdec_si128(state->aes.xmms[0], block0);
    state->sum.xmms[0] = _mm_add_epi64(_mm_shuffle_epi8(state->sum.xmms[0], shuffle_mask), block0);
    state->aes.xmms[1] = _mm_aesdec_si128(state->aes.xmms[1], block1);
    state->sum.xmms[1] = _mm_add_epi64(_mm_shuffle_epi8(state->sum.xmms[1], shuffle_mask), block1);
    state->aes.xmms[2] = _mm_aesdec_si128(state->aes.xmms[2], block2);
    state->sum.xmms[2] = _mm_add_epi64(_mm_shuffle_epi8(state->sum.xmms[2], shuffle_mask), block2);
    state->aes.xmms[3] = _mm_aesdec_si128(state->aes.xmms[3], block3);
    state->sum.xmms[3] = _mm_add_epi64(_mm_shuffle_epi8(state->sum.xmms[3], shuffle_mask), block3);
}

SZ_INTERNAL sz_u64_t _sz_hash_state_finalize_haswell(sz_hash_state_t const *state) {
    // Combine the sum and the AES block
    __m128i mixed_registers0 = _mm_aesenc_si128(state->sum.xmms[0], state->aes.xmms[0]);
    __m128i mixed_registers1 = _mm_aesenc_si128(state->sum.xmms[1], state->aes.xmms[1]);
    __m128i mixed_registers2 = _mm_aesenc_si128(state->sum.xmms[2], state->aes.xmms[2]);
    __m128i mixed_registers3 = _mm_aesenc_si128(state->sum.xmms[3], state->aes.xmms[3]);
    // Combine the mixed registers
    __m128i mixed_registers01 = _mm_aesenc_si128(mixed_registers0, mixed_registers1);
    __m128i mixed_registers23 = _mm_aesenc_si128(mixed_registers2, mixed_registers3);
    __m128i mixed_registers = _mm_aesenc_si128(mixed_registers01, mixed_registers23);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    __m128i mixed_within_register = _mm_aesdec_si128( //
        _mm_aesdec_si128(mixed_registers, state->key.xmms[0]), mixed_registers);
    // Extract the low 64 bits
    return _mm_cvtsi128_si64(mixed_within_register);
}

SZ_PUBLIC sz_u64_t sz_hash_haswell(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {

    if (length <= 16) {
        // Initialize the AES block with a given seed and update with the input length
        _sz_hash_minimal_t state;
        _sz_hash_minimal_init_haswell(&state, seed);
        state.aes.xmm = _mm_add_epi64(state.aes.xmm, _mm_set_epi64x(0, length));
        // Load the data and update the state
        sz_u128_vec_t data_vec;
        data_vec.xmm = _mm_setzero_si128();
        for (sz_size_t i = 0; i < length; ++i) data_vec.u8s[i] = start[i];
        _sz_hash_minimal_update_haswell(&state, data_vec.xmm);
        return _sz_hash_minimal_finalize_haswell(&state);
    }
    else if (length <= 32) {
        // Initialize the AES block with a given seed and update with the input length
        _sz_hash_minimal_t state;
        _sz_hash_minimal_init_haswell(&state, seed);
        state.aes.xmm = _mm_add_epi64(state.aes.xmm, _mm_set_epi64x(0, length));
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.xmm = _mm_lddqu_si128(start);
        data1_vec.xmm = _mm_lddqu_si128(start + length - 16);
        // Let's shift the data within the register to de-interleave the bytes.
        data1_vec.xmm = _mm_bsrli_si128(data1_vec.xmm, 32 - length);
        _sz_hash_minimal_update_haswell(&state, data0_vec.xmm);
        _sz_hash_minimal_update_haswell(&state, data1_vec.xmm);
        return _sz_hash_minimal_finalize_haswell(&state);
    }
    else if (length <= 48) {
        // Initialize the AES block with a given seed and update with the input length
        _sz_hash_minimal_t state;
        _sz_hash_minimal_init_haswell(&state, seed);
        state.aes.xmm = _mm_add_epi64(state.aes.xmm, _mm_set_epi64x(0, length));
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.xmm = _mm_lddqu_si128(start);
        data1_vec.xmm = _mm_lddqu_si128(start + 16);
        data2_vec.xmm = _mm_lddqu_si128(start + length - 16);
        // Let's shift the data within the register to de-interleave the bytes.
        data2_vec.xmm = _mm_bsrli_si128(data2_vec.xmm, 48 - length);
        _sz_hash_minimal_update_haswell(&state, data0_vec.xmm);
        _sz_hash_minimal_update_haswell(&state, data1_vec.xmm);
        _sz_hash_minimal_update_haswell(&state, data2_vec.xmm);
        return _sz_hash_minimal_finalize_haswell(&state);
    }
    else if (length <= 64) {
        // Initialize the AES block with a given seed and update with the input length
        _sz_hash_minimal_t state;
        _sz_hash_minimal_init_haswell(&state, seed);
        state.aes.xmm = _mm_add_epi64(state.aes.xmm, _mm_set_epi64x(0, length));
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.xmm = _mm_lddqu_si128(start);
        data1_vec.xmm = _mm_lddqu_si128(start + 16);
        data2_vec.xmm = _mm_lddqu_si128(start + 32);
        data3_vec.xmm = _mm_lddqu_si128(start + length - 16);
        // Let's shift the data within the register to de-interleave the bytes.
        data3_vec.xmm = _mm_bsrli_si128(data3_vec.xmm, 64 - length);
        _sz_hash_minimal_update_haswell(&state, data0_vec.xmm);
        _sz_hash_minimal_update_haswell(&state, data1_vec.xmm);
        _sz_hash_minimal_update_haswell(&state, data2_vec.xmm);
        _sz_hash_minimal_update_haswell(&state, data3_vec.xmm);
        return _sz_hash_minimal_finalize_haswell(&state);
    }
    else {
        // Use a larger state to handle the main loop and add different offsets
        // to different lanes of the register
        sz_hash_state_t state;
        sz_hash_state_init_haswell(&state, seed);
        state.aes.xmms[0] = _mm_add_epi64(state.aes.xmms[0], _mm_set_epi64x(0, length));
        state.aes.xmms[1] = _mm_add_epi64(state.aes.xmms[1], _mm_set_epi64x(16, length));
        state.aes.xmms[2] = _mm_add_epi64(state.aes.xmms[2], _mm_set_epi64x(32, length));
        state.aes.xmms[3] = _mm_add_epi64(state.aes.xmms[3], _mm_set_epi64x(48, length));

        for (; state.ins_length + 64 <= length; state.ins_length += 64) {
            state.ins.xmms[0] = _mm_lddqu_si128(start + state.ins_length);
            state.ins.xmms[1] = _mm_lddqu_si128(start + state.ins_length + 16);
            state.ins.xmms[2] = _mm_lddqu_si128(start + state.ins_length + 32);
            state.ins.xmms[3] = _mm_lddqu_si128(start + state.ins_length + 48);
            _sz_hash_state_update_haswell(&state, state.ins.xmms[0], state.ins.xmms[1], state.ins.xmms[2],
                                          state.ins.xmms[3]);
        }
        if (state.ins_length < length) {
            state.ins.xmms[0] = _mm_setzero_si128();
            state.ins.xmms[1] = _mm_setzero_si128();
            state.ins.xmms[2] = _mm_setzero_si128();
            state.ins.xmms[3] = _mm_setzero_si128();
            for (sz_size_t i = 0; state.ins_length < length; ++i, ++state.ins_length)
                state.ins.u8s[i] = start[state.ins_length];
            _sz_hash_state_update_haswell(&state, state.ins.xmms[0], state.ins.xmms[1], state.ins.xmms[2],
                                          state.ins.xmms[3]);
        }
        return _sz_hash_state_finalize_haswell(&state);
    }
}

SZ_PUBLIC void sz_generate_haswell(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_generate_serial(text, length, nonce);
}

SZ_PUBLIC void sz_hash_state_init_haswell(sz_hash_state_t *state, sz_u64_t seed) {
    sz_hash_state_init_serial(state, seed);
}

SZ_PUBLIC void sz_hash_state_stream_haswell(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_hash_state_stream_serial(state, text, length);
}

SZ_PUBLIC sz_u64_t sz_hash_state_fold_haswell(sz_hash_state_t const *state) { return sz_hash_state_fold_serial(state); }

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_HASWELL
#pragma endregion // Haswell Implementation

/*  AVX512 implementation of the string hashing algorithms for Skylake and newer CPUs.
 *  Includes extensions: F, CD, ER, PF, VL, DQ, BW.
 *
 *  This is the "starting level" for the advanced algorithms using K-mask registers on x86.
 */
#pragma region Skylake Implementation
#if SZ_USE_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2")
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2"))), apply_to = function)

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
        __mmask16 mask = _sz_u16_mask_until(length);
        text_vec.xmms[0] = _mm_maskz_loadu_epi8(mask, text);
        sums_vec.xmms[0] = _mm_sad_epu8(text_vec.xmms[0], _mm_setzero_si128());
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_vec.xmms[0]);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_vec.xmms[0], 1);
        return low + high;
    }
    else if (length <= 32) {
        __mmask32 mask = _sz_u32_mask_until(length);
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
        __mmask64 mask = _sz_u64_mask_until(length);
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
        _sz_assert(body_length % 64 == 0 && head_length < 64 && tail_length < 64);
        __mmask64 head_mask = _sz_u64_mask_until(head_length);
        __mmask64 tail_mask = _sz_u64_mask_until(tail_length);

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
        __mmask64 head_mask = _sz_u64_mask_until(head_length);
        __mmask64 tail_mask = _sz_u64_mask_until(tail_length);

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
    __m512i seed_vec = _mm512_set1_epi64(seed);
    __m512i pi0 = _mm512_set_epi64( //
        0x13198a2e03707344ull, 0x243f6a8885a308d3ull, 0x13198a2e03707344ull, 0x243f6a8885a308d3ull,
        0x13198a2e03707344ull, 0x243f6a8885a308d3ull, 0x13198a2e03707344ull, 0x243f6a8885a308d3ull);
    __m512i pi1 = _mm512_set_epi64( //
        0x082efa98ec4e6c89ull, 0xa4093822299f31d0ull, 0x082efa98ec4e6c89ull, 0xa4093822299f31d0ull,
        0x082efa98ec4e6c89ull, 0xa4093822299f31d0ull, 0x082efa98ec4e6c89ull, 0xa4093822299f31d0ull);
    // XOR the user-supplied keys with the two "pi" constants
    __m512i k1 = _mm512_xor_si512(seed_vec, pi0);
    __m512i k2 = _mm512_xor_si512(seed_vec, pi1);
    // Export the keys to the state
    state->aes.zmm = k1;
    state->sum.zmm = k2;
    state->key.zmm = _mm512_xor_si512(pi0, pi1);
    state->ins_length = 0;
}

SZ_PUBLIC sz_u64_t sz_hash_skylake(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {

    if (length <= 16) {
        // Initialize the AES block with a given seed and update with the input length
        _sz_hash_minimal_t state;
        _sz_hash_minimal_init_haswell(&state, seed);
        state.aes.xmm = _mm_add_epi64(state.aes.xmm, _mm_set_epi64x(0, length));
        // Load the data and update the state
        sz_u128_vec_t data_vec;
        data_vec.xmm = _mm_maskz_loadu_epi8(_sz_u16_mask_until(length), start);
        _sz_hash_minimal_update_haswell(&state, data_vec.xmm);
        return _sz_hash_minimal_finalize_haswell(&state);
    }
    else if (length <= 32) {
        // Initialize the AES block with a given seed and update with the input length
        _sz_hash_minimal_t state;
        _sz_hash_minimal_init_haswell(&state, seed);
        state.aes.xmm = _mm_add_epi64(state.aes.xmm, _mm_set_epi64x(0, length));
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.xmm = _mm_lddqu_si128(start);
        data1_vec.xmm = _mm_maskz_loadu_epi8(_sz_u16_mask_until(length - 16), start + 16);
        _sz_hash_minimal_update_haswell(&state, data0_vec.xmm);
        _sz_hash_minimal_update_haswell(&state, data1_vec.xmm);
        return _sz_hash_minimal_finalize_haswell(&state);
    }
    else if (length <= 48) {
        // Initialize the AES block with a given seed and update with the input length
        _sz_hash_minimal_t state;
        _sz_hash_minimal_init_haswell(&state, seed);
        state.aes.xmm = _mm_add_epi64(state.aes.xmm, _mm_set_epi64x(0, length));
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.xmm = _mm_lddqu_si128(start);
        data1_vec.xmm = _mm_lddqu_si128(start + 16);
        data2_vec.xmm = _mm_maskz_loadu_epi8(_sz_u16_mask_until(length - 32), start + 32);
        _sz_hash_minimal_update_haswell(&state, data0_vec.xmm);
        _sz_hash_minimal_update_haswell(&state, data1_vec.xmm);
        _sz_hash_minimal_update_haswell(&state, data2_vec.xmm);
        return _sz_hash_minimal_finalize_haswell(&state);
    }
    else if (length <= 64) {
        // Initialize the AES block with a given seed and update with the input length
        _sz_hash_minimal_t state;
        _sz_hash_minimal_init_haswell(&state, seed);
        state.aes.xmm = _mm_add_epi64(state.aes.xmm, _mm_set_epi64x(0, length));
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.xmm = _mm_lddqu_si128(start);
        data1_vec.xmm = _mm_lddqu_si128(start + 16);
        data2_vec.xmm = _mm_lddqu_si128(start + 32);
        data3_vec.xmm = _mm_maskz_loadu_epi8(_sz_u16_mask_until(length - 48), start + 48);
        _sz_hash_minimal_update_haswell(&state, data0_vec.xmm);
        _sz_hash_minimal_update_haswell(&state, data1_vec.xmm);
        _sz_hash_minimal_update_haswell(&state, data2_vec.xmm);
        _sz_hash_minimal_update_haswell(&state, data3_vec.xmm);
        return _sz_hash_minimal_finalize_haswell(&state);
    }
    else {
        // Use a larger state to handle the main loop and add different offsets
        // to different lanes of the register
        sz_hash_state_t state;
        sz_hash_state_init_skylake(&state, seed);
        state.aes.zmm = _mm512_add_epi64( //
            state.aes.zmm,                //
            _mm512_set_epi64(0, length, 16, length, 32, length, 48, length));

        for (; state.ins_length + 64 <= length; state.ins_length += 64) {
            state.ins.zmm = _mm512_loadu_epi8(start + state.ins_length);
            _sz_hash_state_update_haswell(&state, state.ins.xmms[0], state.ins.xmms[1], state.ins.xmms[2],
                                          state.ins.xmms[3]);
        }
        if (state.ins_length < length) {
            state.ins.zmm = _mm512_maskz_loadu_epi8( //
                _sz_u64_mask_until(length - state.ins_length), start + state.ins_length);
            _sz_hash_state_update_skylake(&state, state.ins.zmm);
        }
        return _sz_hash_state_finalize_haswell(&state);
    }
}

SZ_PUBLIC void sz_generate_skylake(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_generate_serial(text, length, nonce);
}

SZ_PUBLIC void sz_hash_state_stream_skylake(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_hash_state_stream_serial(state, text, length);
}

SZ_PUBLIC sz_u64_t sz_hash_state_fold_skylake(sz_hash_state_t const *state) { return sz_hash_state_fold_serial(state); }

#pragma clang attribute pop
#pragma GCC pop_options
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
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vnni", "bmi", "bmi2", \
                   "aes", "vaes")
#pragma clang attribute push(                                                                                  \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vnni,bmi,bmi2,aes,vaes"))), \
    apply_to = function)

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
        __mmask16 mask = _sz_u16_mask_until(length);
        text_vec.xmms[0] = _mm_maskz_loadu_epi8(mask, text);
        sums_vec.xmms[0] = _mm_sad_epu8(text_vec.xmms[0], _mm_setzero_si128());
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_vec.xmms[0]);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_vec.xmms[0], 1);
        return low + high;
    }
    else if (length <= 32) {
        __mmask32 mask = _sz_u32_mask_until(length);
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
        __mmask64 mask = _sz_u64_mask_until(length);
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
        _sz_assert(body_length % 64 == 0 && head_length < 64 && tail_length < 64);
        __mmask64 head_mask = _sz_u64_mask_until(head_length);
        __mmask64 tail_mask = _sz_u64_mask_until(tail_length);

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
            _sz_assert(body_length == 64);
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
        __mmask64 head_mask = _sz_u64_mask_until(head_length);
        __mmask64 tail_mask = _sz_u64_mask_until(tail_length);

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

SZ_INTERNAL void _sz_hash_state_update_ice(sz_hash_state_t *state, __m512i block) {
    // This shuffle mask is identical to "aHash":
    __m512i const shuffle_mask = _mm512_set_epi8(       //
        0x04, 0x0b, 0x09, 0x06, 0x08, 0x0d, 0x0f, 0x05, //
        0x0e, 0x03, 0x01, 0x0c, 0x00, 0x07, 0x0a, 0x02, //
        0x04, 0x0b, 0x09, 0x06, 0x08, 0x0d, 0x0f, 0x05, //
        0x0e, 0x03, 0x01, 0x0c, 0x00, 0x07, 0x0a, 0x02, //
        0x04, 0x0b, 0x09, 0x06, 0x08, 0x0d, 0x0f, 0x05, //
        0x0e, 0x03, 0x01, 0x0c, 0x00, 0x07, 0x0a, 0x02, //
        0x04, 0x0b, 0x09, 0x06, 0x08, 0x0d, 0x0f, 0x05, //
        0x0e, 0x03, 0x01, 0x0c, 0x00, 0x07, 0x0a, 0x02  //
    );
    state->aes.zmm = _mm512_aesdec_epi128(state->aes.zmm, block);
    state->sum.zmm = _mm512_add_epi64(_mm512_shuffle_epi8(state->sum.zmm, shuffle_mask), block);
}

SZ_PUBLIC sz_u64_t sz_hash_ice(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {

    if (length <= 16) {
        // Initialize the AES block with a given seed and update with the input length
        _sz_hash_minimal_t state;
        _sz_hash_minimal_init_haswell(&state, seed);
        state.aes.xmm = _mm_add_epi64(state.aes.xmm, _mm_set_epi64x(0, length));
        // Load the data and update the state
        sz_u128_vec_t data_vec;
        data_vec.xmm = _mm_maskz_loadu_epi8(_sz_u16_mask_until(length), start);
        _sz_hash_minimal_update_haswell(&state, data_vec.xmm);
        return _sz_hash_minimal_finalize_haswell(&state);
    }
    else if (length <= 32) {
        // Initialize the AES block with a given seed and update with the input length
        _sz_hash_minimal_t state;
        _sz_hash_minimal_init_haswell(&state, seed);
        state.aes.xmm = _mm_add_epi64(state.aes.xmm, _mm_set_epi64x(0, length));
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.xmm = _mm_lddqu_si128(start);
        data1_vec.xmm = _mm_maskz_loadu_epi8(_sz_u16_mask_until(length - 16), start + 16);
        _sz_hash_minimal_update_haswell(&state, data0_vec.xmm);
        _sz_hash_minimal_update_haswell(&state, data1_vec.xmm);
        return _sz_hash_minimal_finalize_haswell(&state);
    }
    else if (length <= 48) {
        // Initialize the AES block with a given seed and update with the input length
        _sz_hash_minimal_t state;
        _sz_hash_minimal_init_haswell(&state, seed);
        state.aes.xmm = _mm_add_epi64(state.aes.xmm, _mm_set_epi64x(0, length));
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.xmm = _mm_lddqu_si128(start);
        data1_vec.xmm = _mm_lddqu_si128(start + 16);
        data2_vec.xmm = _mm_maskz_loadu_epi8(_sz_u16_mask_until(length - 32), start + 32);
        _sz_hash_minimal_update_haswell(&state, data0_vec.xmm);
        _sz_hash_minimal_update_haswell(&state, data1_vec.xmm);
        _sz_hash_minimal_update_haswell(&state, data2_vec.xmm);
        return _sz_hash_minimal_finalize_haswell(&state);
    }
    else if (length <= 64) {
        // Initialize the AES block with a given seed and update with the input length
        _sz_hash_minimal_t state;
        _sz_hash_minimal_init_haswell(&state, seed);
        state.aes.xmm = _mm_add_epi64(state.aes.xmm, _mm_set_epi64x(0, length));
        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.xmm = _mm_lddqu_si128(start);
        data1_vec.xmm = _mm_lddqu_si128(start + 16);
        data2_vec.xmm = _mm_lddqu_si128(start + 32);
        data3_vec.xmm = _mm_maskz_loadu_epi8(_sz_u16_mask_until(length - 48), start + 48);
        _sz_hash_minimal_update_haswell(&state, data0_vec.xmm);
        _sz_hash_minimal_update_haswell(&state, data1_vec.xmm);
        _sz_hash_minimal_update_haswell(&state, data2_vec.xmm);
        _sz_hash_minimal_update_haswell(&state, data3_vec.xmm);
        return _sz_hash_minimal_finalize_haswell(&state);
    }
    else {
        // Use a larger state to handle the main loop and add different offsets
        // to different lanes of the register
        sz_hash_state_t state;
        sz_hash_state_init_skylake(&state, seed);
        state.aes.zmm = _mm512_add_epi64( //
            state.aes.zmm,                //
            _mm512_set_epi64(0, length, 16, length, 32, length, 48, length));

        for (; state.ins_length + 64 <= length; state.ins_length += 64) {
            state.ins.zmm = _mm512_loadu_epi8(start + state.ins_length);
            _sz_hash_state_update_ice(&state, state.ins.zmm);
        }
        if (state.ins_length < length) {
            state.ins.zmm = _mm512_maskz_loadu_epi8( //
                _sz_u64_mask_until(length - state.ins_length), start + state.ins_length);
            _sz_hash_state_update_ice(&state, state.ins.zmm);
        }
        return _sz_hash_state_finalize_haswell(&state);
    }
}

SZ_PUBLIC void sz_generate_ice(sz_ptr_t output, sz_size_t length, sz_u64_t nonce) {
    // We can use `_mm512_broadcast_i32x4` and the `vbroadcasti32x4` instruction, but its latency is freaking 8 cycles.
    // The `_mm512_shuffle_i32x4` and the `vshufi32x4` instruction has a latency of 3 cycles, somewhat better.
    // The `_mm512_permutex_epi64` and the `vpermq` instruction also has a latency of 3 cycles.
    // So we want to avoid that, if possible.
    __m128i nonce_vec = _mm_set1_epi64x(nonce);
    __m128i key128 = _mm_xor_si128(nonce_vec, _mm_set_epi64x(0x13198a2e03707344ull, 0x243f6a8885a308d3ull));
    if (length <= 16) {
        __mmask16 mask = _sz_u16_mask_until(length);
        __m128i input = _mm_set1_epi64x(nonce);
        __m128i generated = _mm_aesenc_si128(input, key128);
        _mm_mask_storeu_epi8((void *)output, mask, generated);
    }
    // Assuming the YMM register contains two 128-bit blocks, the input to the generator
    // will be more complex, containing the sum of the nonce and the block number.
    else if (length <= 32) {
        __mmask32 mask = _sz_u32_mask_until(length);
        __m256i input = _mm256_set_epi64x(nonce + 1, nonce + 1, nonce, nonce);
        __m256i key256 =
            _mm256_permute2x128_si256(_mm256_castsi128_si256(key128), _mm256_castsi128_si256(key128), 0x00);
        __m256i generated = _mm256_aesenc_epi128(input, key256);
        _mm256_mask_storeu_epi8((void *)output, mask, generated);
    }
    // The last special case we handle outside of the primary loop is for buffers up to 64 bytes long.
    else if (length <= 64) {
        __mmask64 mask = _sz_u64_mask_until(length);
        __m512i input = _mm512_set_epi64(               //
            nonce + 3, nonce + 3, nonce + 2, nonce + 2, //
            nonce + 1, nonce + 1, nonce, nonce);
        __m512i key512 = _mm512_permutex_epi64(_mm512_castsi128_si512(key128), 0x00);
        __m512i generated = _mm512_aesenc_epi128(input, key512);
        _mm512_mask_storeu_epi8((void *)output, mask, generated);
    }
    // The final part of the function is the primary loop, which processes the buffer in 64-byte chunks.
    else {
        __m512i increment = _mm512_set1_epi64(4);
        __m512i input = _mm512_set_epi64(               //
            nonce + 3, nonce + 3, nonce + 2, nonce + 2, //
            nonce + 1, nonce + 1, nonce, nonce);
        __m512i key512 = _mm512_permutex_epi64(_mm512_castsi128_si512(key128), 0x00);
        sz_size_t i = 0;
        for (; i + 64 <= length; i += 64) {
            __m512i generated = _mm512_aesenc_epi128(input, key512);
            _mm512_storeu_epi8((void *)(output + i), generated);
            input = _mm512_add_epi64(input, increment);
        }
        // Handle the tail of the buffer.
        __mmask64 mask = _sz_u64_mask_until(length - i);
        __m512i generated = _mm512_aesenc_epi128(input, key512);
        _mm512_mask_storeu_epi8((void *)(output + i), mask, generated);
    }
}

SZ_PUBLIC void sz_hash_state_stream_ice(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_hash_state_stream_serial(state, text, length);
}

SZ_PUBLIC sz_u64_t sz_hash_state_fold_ice(sz_hash_state_t const *state) { return sz_hash_state_fold_serial(state); }

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_ICE
#pragma endregion // Ice Lake Implementation

/*  Implementation of the string hashing algorithms using the Arm NEON instruction set, available on 64-bit
 *  Arm processors. Covers billions of mobile CPUs worldwide, including Apple's A-series, and Qualcomm's Snapdragon.
 */
#pragma region NEON Implementation
#if SZ_USE_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)

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
    if (length) sum += sz_bytesum_serial(text, length);
    return sum;
}

SZ_PUBLIC void sz_hash_state_init_neon(sz_hash_state_t *state, sz_u64_t seed) {
    sz_hash_state_init_serial(state, seed);
}

SZ_PUBLIC void sz_hash_state_stream_neon(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_hash_state_stream_serial(state, text, length);
}

SZ_PUBLIC sz_u64_t sz_hash_state_fold_neon(sz_hash_state_t const *state) { return sz_hash_state_fold_serial(state); }

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_NEON
#pragma endregion // NEON Implementation

/*  Implementation of the string search algorithms using the Arm SVE variable-length registers,
 *  available in Arm v9 processors, like in Apple M4+ and Graviton 3+ CPUs.
 */
#pragma region SVE Implementation
#if SZ_USE_SVE
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_SVE
#pragma endregion // SVE Implementation

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
#elif SZ_USE_NEON
    return sz_hash_neon(text, length, seed);
#else
    return sz_hash_serial(text, length, seed);
#endif
}

SZ_DYNAMIC void sz_generate(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
#if SZ_USE_ICE
    sz_generate_ice(text, length, nonce);
#elif SZ_USE_SKYLAKE
    sz_generate_skylake(text, length, nonce);
#elif SZ_USE_HASWELL
    sz_generate_haswell(text, length, nonce);
#elif SZ_USE_NEON
    sz_generate_neon(text, length, nonce);
#else
    sz_generate_serial(text, length, nonce);
#endif
}

SZ_DYNAMIC void sz_hash_state_init(sz_hash_state_t *state, sz_u64_t seed) {
#if SZ_USE_ICE
    sz_hash_state_init_ice(state, seed);
#elif SZ_USE_SKYLAKE
    sz_hash_state_init_skylake(state, seed);
#elif SZ_USE_HASWELL
    sz_hash_state_init_haswell(state, seed);
#elif SZ_USE_NEON
    sz_hash_state_init_neon(state, seed);
#else
    sz_hash_state_init_serial(state, seed);
#endif
}

SZ_DYNAMIC void sz_hash_state_stream(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
#if SZ_USE_ICE
    sz_hash_state_stream_ice(state, text, length);
#elif SZ_USE_SKYLAKE
    sz_hash_state_stream_skylake(state, text, length);
#elif SZ_USE_HASWELL
    sz_hash_state_stream_haswell(state, text, length);
#elif SZ_USE_NEON
    sz_hash_state_stream_neon(state, text, length);
#else
    sz_hash_state_stream_serial(state, text, length);
#endif
}

SZ_DYNAMIC sz_u64_t sz_hash_state_fold(sz_hash_state_t const *state) {
#if SZ_USE_ICE
    return sz_hash_state_fold_ice(state);
#elif SZ_USE_SKYLAKE
    return sz_hash_state_fold_skylake(state);
#elif SZ_USE_HASWELL
    return sz_hash_state_fold_haswell(state);
#elif SZ_USE_NEON
    return sz_hash_state_fold_neon(state);
#else
    return sz_hash_state_fold_serial(state);
#endif
}

#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_HASH_H_
