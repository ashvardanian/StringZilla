/**
 *  @file include/stringzilla/hash.h
 *  @author Ash Vardanian
 *  @date December 1, 2024
 *  @brief Hardware-accelerated non-cryptographic string hashing and checksums.
 *
 *  Includes core APIs with hardware-specific backends:
 *
 *  - @c sz_bytesum - for byte-level 64-bit unsigned checksums.
 *  - @c sz_hash - for 64-bit single-shot hashing using AES instructions.
 *  - @c sz_hash_state_init, @c sz_hash_state_update, @c sz_hash_state_digest - incremental hashing.
 *  - @c sz_fill_random - for populating buffers with pseudo-random noise using AES instructions.
 *
 *  Why the hell do we need a yet another hashing library?! Turns out, most existing libraries have
 *  noticeable constraints. Try finding a library that:
 *
 *  - Outputs 64-bit or 128-bit hashes and passes the @b SMHasher `--extra` tests.
 *  - Is fast for both short @b (velocity) and long strings @b (throughput).
 *  - Supports incremental @b (streaming) hashing, when the data arrives in chunks.
 *  - Supports custom @b seeds for hashes and have it affecting every bit of the output.
 *  - Provides @b dynamic-dispatch for different architectures to simplify deployment.
 *  - Uses @b SIMD, including not just AVX2 & NEON, but also masking AVX-512 & predicated SVE2.
 *  - Documents its logic and @b guarantees the same output across different platforms.
 *
 *  This includes projects like "MurmurHash", "CityHash", "SpookyHash", "FarmHash", "MetroHash",
 *  "HighwayHash", etc. There are 2 libraries that are close to meeting these requirements: "xxHash"
 *  in C++ and "aHash" in Rust:
 *
 *  - "aHash" is fast, but written in Rust, has no dynamic dispatch, and lacks AVX-512 and SVE2
 *    support. It also does not adhere to a fixed output, and can't be used in applications like
 *    computing packet checksums in network traffic or implementing persistent data structures.
 *  - "xxHash" is implemented in C, has an extremely wide set of third-party language bindings, and
 *    provides 32-, 64-, and 128-bit hashes. It is fast, but its dynamic dispatch is limited to x86
 *    with `xxh_x86dispatch.c`.
 *
 *  StringZilla uses a scheme more similar to "aHash" and "GxHash", utilizing the AES extensions,
 *  that provide a remarkable level of "mixing per cycle" and are broadly available on modern CPUs.
 *  Similar to "aHash", they are combined with "shuffle & add" instructions to provide a high level
 *  of entropy in the output. That operation is practically free, as many modern CPUs will dispatch
 *  them on different ports. On x86, for example:
 *
 *  @verbatim
 *  Instruction                     Intel Ice Lake          AMD Zen4
 *  VAESENC (ZMM, ZMM, ZMM)         5 cycles on port 0      4 cycles on ports 0 or 1
 *  VAESDEC (ZMM, ZMM, ZMM)         5 cycles on port 0      4 cycles on ports 0 or 1
 *  VPSHUFB_Z (ZMM, K, ZMM, ZMM)    3 cycles on port 5      2 cycles on ports 1 or 2
 *  VPADDQ (ZMM, ZMM, ZMM)          1 cycle on ports 0, 5   1 cycle on ports 0, 1, 2, 3
 *  @endverbatim
 *
 *  But there are several key differences.
 *
 *  @b Wider @b state. A larger state and a larger block size is used for inputs over 64 bytes long,
 *  benefiting from wider registers on current CPUs. Like many other hash functions, the state is
 *  initialized with the seed and a set of Pi constants. Unlike others, we pull more Pi bits (1024),
 *  but only 64 bits of the seed, to keep the API sane.
 *
 *  @b Streaming. The length of the input is not mixed into the AES block at the start to allow
 *  incremental construction, when the final length is not known in advance.
 *
 *  @b Uniform @b loads. The vector-loads are not interleaved, meaning that each byte of input has
 *  exactly the same weight in the hash. On the implementation side it requires some extra shuffling
 *  on older platforms, but on newer platforms it can be done with "masked" loads in AVX-512 and
 *  "predicated" instructions in SVE2.
 *
 *  Moreover, the same AES primitives are reused to implement a fast Pseudo-Random Number Generator
 *  @b (PRNG) that is consistent between different implementation backends and has reproducible
 *  output with the same "nonce". The PRNG produces random byte sequences, and combining it with
 *  @c sz_lookup produces random strings with a given byteset.
 *
 *  Other helpers include:
 *
 *  - @c sz_fill_alphabet - fills buffers with random ASCII characters, combining @c sz_fill_random
 *    and @c sz_lookup.
 *  - @c sz_fill_alphabet_utf8 - does the same with random UTF-8 characters.
 *
 *  @see Reini Urban's more active fork of SMHasher by Austin Appleby: https://github.com/rurban/smhasher
 *  @see The serial AES routines are based on Morten Jensen's "tiny-AES-c": https://github.com/kokke/tiny-AES-c
 *  @see The "xxHash" C implementation by Yann Collet: https://github.com/Cyan4973/xxHash
 *  @see The "aHash" Rust implementation by Tom Kaitchuck: https://github.com/tkaitchuck/aHash
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
 *  @brief Computes the 64-bit check-sum of bytes in a string, similar to
 *      @c std::ranges::accumulate.
 *
 *  @param[in] text String to aggregate.
 *  @param[in] length Number of bytes in the text.
 *  @return 64-bit unsigned value.
 *
 *  Summing the bytes of "hi", 104 + 105:
 *
 *  @code{.c}
 *      #include <stringzilla/hash.h>
 *      int main() {
 *          return sz_bytesum("hi", 2) == 209 ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note Selects the fastest backend at compile- or run-time based on
 *      @c STRINGZILLA_RUNTIME_DISPATCH.
 *  @sa sz_bytesum_serial, sz_bytesum_haswell, sz_bytesum_skylake, sz_bytesum_icelake,
 *      sz_bytesum_neon, sz_bytesum_sve, sz_bytesum_sve2, sz_bytesum_v128, sz_bytesum_v128relaxed,
 *      sz_bytesum_rvv, sz_bytesum_lasx, sz_bytesum_powervsx
 */
STRINGZILLA_API_RUNTIME sz_u64_t sz_bytesum(sz_cptr_t text, sz_size_t length);

/**
 *  @brief Computes the 64-bit unsigned hash of a string, similar to @c std::hash in C++.
 *
 *  @param[in] text String to hash.
 *  @param[in] length Number of bytes in the text.
 *  @param[in] seed 64-bit unsigned seed for the hash.
 *  @return 64-bit hash value.
 *
 *  It's not cryptographically secure, but it's fast and provides a good distribution. It passes the
 *  SMHasher suite by Austin Appleby with no collisions, even with the `--extra` flag. HASH.md
 *  explains the algorithm in detail.
 *
 *  Telling two different strings apart:
 *
 *  @code{.c}
 *      #include <stringzilla/hash.h>
 *      int main() {
 *          return sz_hash("hello", 5, 0) != sz_hash("world", 5, 0) ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note Selects the fastest backend at compile- or run-time based on
 *      @c STRINGZILLA_RUNTIME_DISPATCH.
 *  @note The output is the same on all platforms, in both single-shot and incremental modes.
 *  @sa sz_hash_serial, sz_hash_westmere, sz_hash_skylake, sz_hash_icelake, sz_hash_neonaes,
 *      sz_hash_sve2aes, sz_hash_v128, sz_hash_rvv, sz_hash_lasx, sz_hash_powervsx
 *  @sa sz_hash_state_init, sz_hash_state_update, sz_hash_state_digest
 */
STRINGZILLA_API_RUNTIME sz_u64_t sz_hash(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/**
 *  @brief Hashes one string under @b many seeds at once, the "multi-seed" hash.
 *
 *  @param[in] text String to hash.
 *  @param[in] length Number of bytes in the text.
 *  @param[in] seeds Array of @p seeds_count 64-bit seeds.
 *  @param[in] seeds_count Number of seeds, and the number of hashes written to @p hashes.
 *  @param[out] hashes Caller-allocated output buffer of @p seeds_count 64-bit hashes.
 *
 *  Equivalent to calling `sz_hash(text, length, seeds[i])` for each seed, but normalizes the input
 *  into AES blocks @b once and replays the cheap per-seed rounds over that prepared form.
 *
 *  Designed for workloads that need several independent hashes of the same (often short) key:
 *  feature-hashing & Count-Min sketches for BM25/TF-IDF, Bloom filters, cuckoo hashing, MinHash /
 *  SimHash / LSH banding, and rendezvous (HRW) hashing. Two effects make it faster than a loop of
 *  @c sz_hash: the branch-heavy load & de-interleave of variable-length short strings is amortized
 *  across all seeds, and AES-capable backends advance multiple seed-lanes per instruction.
 *
 *  Three seeds, the first matching a plain @c sz_hash call:
 *
 *  @code{.c}
 *      #include <stringzilla/hash.h>
 *      int main() {
 *          sz_u64_t seeds[3] = {1, 2, 3}, hashes[3];
 *          sz_hash_multiseed("token", 5, seeds, 3, hashes);
 *          return hashes[0] == sz_hash("token", 5, 1) ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note Biggest speedups are for @p length ≤ 64; above that only the input load is shared.
 *  @note Selects the fastest backend at compile- or run-time based on
 *      @c STRINGZILLA_RUNTIME_DISPATCH.
 *  @sa sz_hash_multiseed_serial, sz_hash_multiseed_westmere, sz_hash_multiseed_icelake,
 *      sz_hash_multiseed_neonaes
 */
STRINGZILLA_API_RUNTIME void sz_hash_multiseed(   //
    sz_cptr_t text, sz_size_t length,             //
    sz_u64_t const *seeds, sz_size_t seeds_count, //
    sz_u64_t *hashes);

/**
 *  @brief A Pseudorandom Number Generator (PRNG), inspired by the AES-CTR-128 algorithm, but using
 *      only one round of AES mixing as opposed to "NIST SP 800-90A".
 *
 *  CTR_DRBG (CounTeR mode Deterministic Random Bit Generator) appears secure and indistinguishable
 *  from a true random source when AES is used as the underlying block cipher and 112 bits are taken
 *  from this PRNG. When AES is used as the underlying block cipher and 128 bits are taken from each
 *  instantiation, the required security level is delivered with the caveat that a 128-bit cipher's
 *  output in counter mode can be distinguished from a true RNG.
 *
 *  In this case, it doesn't apply, as we only use one round of AES mixing. We also don't expose a
 *  separate "key", only a "nonce", to keep the API simple, but we mix it with 512 bits of Pi
 *  constants to increase randomness.
 *
 *  @param[out] text Output string buffer to be populated.
 *  @param[in] length Number of bytes in the string.
 *  @param[in] nonce "Number used once" to ensure uniqueness of produced blocks.
 *
 *  Filling two buffers under the same nonce:
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
 *  @note Selects the fastest backend at compile- or run-time based on
 *      @c STRINGZILLA_RUNTIME_DISPATCH.
 *  @sa sz_fill_random_serial, sz_fill_random_westmere, sz_fill_random_skylake,
 *      sz_fill_random_icelake, sz_fill_random_neonaes, sz_fill_random_sve2aes, sz_fill_random_v128,
 *      sz_fill_random_rvv, sz_fill_random_lasx, sz_fill_random_powervsx
 */
STRINGZILLA_API_RUNTIME void sz_fill_random(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/**
 *  @brief The state for incremental construction of a hash.
 *  @sa sz_hash_state_init, sz_hash_state_update, sz_hash_state_digest
 *
 *  @note Uses the @c packed attribute to allow placement at arbitrary addresses without UBSAN
 *      warnings, and plain byte arrays to avoid implicit alignment requirements from SIMD types.
 *      The layout matches @c sz_hash_state_aligned_t for safe casting between them.
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

/** Bytes in a SHA256 digest, fixed by FIPS 180-4. */
#define STRINGZILLA_SHA256_DIGEST_LENGTH (32)

/**
 *  @brief Bytes in a SHA256 message block, fixed by FIPS 180-4.
 *  @note Coincides with @c STRINGZILLA_CACHE_LINE_BYTES and the ZMM width, which are unrelated
 *      reasons for the same 64.
 */
#define STRINGZILLA_SHA256_BLOCK_LENGTH (64)

/**
 *  @brief The state for incremental construction of a SHA256 hash.
 *  @sa sz_sha256_state_init, sz_sha256_state_update, sz_sha256_state_digest
 */
typedef struct sz_sha256_state_t {

    /** Message block buffer. */
    sz_u8_t block[STRINGZILLA_SHA256_BLOCK_LENGTH];

    /** Current hash state: 8x 32-bit values. */
    sz_u32_t hash[8];

    /** Total message length in bytes. */
    sz_u64_t total_length;

    /** Current bytes in block (0-63). */
    sz_u8_t block_length;

    /** Rounds the state to 128 bytes. */
    sz_u8_t padding_[23];
} sz_sha256_state_t;

/*  The batched kernels walk an array of these, so the layout is load-bearing rather than
 *  incidental. A power-of-two stride turns lane indexing into a shift, and putting @c block first
 *  gives every lane the same cache-line phase as the array itself — at 112 bytes the phase rotated
 *  per lane, splitting the 64-byte block read across two lines for most of them. @c block_length is
 *  a byte because it never exceeds 63, which also makes the struct identical on 32- and 64-bit
 *  builds. Alignment stays natural on purpose: @c malloc only promises 16 bytes, so demanding 64
 *  would under-align every heap-allocated batch. */
sz_static_assert(sizeof(sz_sha256_state_t) == 128, sha256_state_is_two_cache_lines);

/**
 *  @brief Initializes the state for incremental construction of a hash.
 *
 *  @param[out] state The state to initialize.
 *  @param[in] seed The 64-bit unsigned seed for the hash.
 */
STRINGZILLA_API_RUNTIME void sz_hash_state_init(sz_hash_state_t *state, sz_u64_t seed);

/**
 *  @brief Updates the state with new data.
 *
 *  @param[inout] state The state to stream.
 *  @param[in] text The new data to include in the hash.
 *  @param[in] length The number of bytes in the new data.
 */
STRINGZILLA_API_RUNTIME void sz_hash_state_update(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/**
 *  @brief Finalizes the immutable state and returns the hash.
 *
 *  @param[in] state The state to fold.
 *  @return The 64-bit hash value.
 */
STRINGZILLA_API_RUNTIME sz_u64_t sz_hash_state_digest(sz_hash_state_t const *state);

/**
 *  @brief Initializes the state for incremental SHA256 hashing.
 *
 *  @param[out] state The state to initialize.
 */
STRINGZILLA_API_RUNTIME void sz_sha256_state_init(sz_sha256_state_t *state);

/**
 *  @brief Updates the SHA256 state with new data.
 *
 *  @param[inout] state The state to update.
 *  @param[in] data The new data to hash.
 *  @param[in] length The number of bytes in the new data.
 */
STRINGZILLA_API_RUNTIME void sz_sha256_state_update(sz_sha256_state_t *state, sz_cptr_t data, sz_size_t length);

/**
 *  @brief Finalizes the SHA256 state and returns the digest, leaving the state open to more data.
 *
 *  @param[in] state The state to finalize.
 *  @param[out] digest Output buffer for the 32-byte (256-bit) digest.
 */
STRINGZILLA_API_RUNTIME void sz_sha256_state_digest(sz_sha256_state_t const *state,
                                                    sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)]);

/**
 *  @brief Advances many independent SHA256 states, one message per lane.
 *
 *  @param[inout] states Array of at least `texts->count` states, each initialized with
 *      @c sz_sha256_state_init.
 *  @param[in] texts Sequence supplying the next chunk of each lane's message, with @c count lanes.
 *
 *  Hashing one message is a serial dependency chain, so a wider instruction set cannot accelerate
 *  it. Independent messages compress in parallel lanes, which is what this does: sixteen at a time
 *  on AVX-512, eight on AVX2. Supply at least a few kilobytes per lane per call so the lane
 *  transposition is amortized; below one 64-byte block per lane it degrades to the single-message
 *  path. A zero-length chunk is a no-op for that lane. The states must be distinct, while the
 *  chunks may overlap.
 *
 *  Advancing two lanes by one chunk each, then taking both digests:
 *
 *  @code{.c}
 *      #include <stringzilla/hash.h>
 *      int main() {
 *          sz_cptr_t chunks[2] = {"hello", "world"};
 *          sz_sha256_state_t states[2];
 *          sz_u8_t digests[2 * STRINGZILLA_SHA256_DIGEST_LENGTH];
 *          sz_sequence_t texts;
 *          sz_sequence_from_null_terminated_strings(chunks, 2, &texts);
 *          sz_sha256_state_init(&states[0]), sz_sha256_state_init(&states[1]);
 *          sz_sha256_multistate_update(states, &texts);
 *          sz_sha256_multistate_digest(states, 2, digests);
 *          return digests[0] == 0x2c ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note Selects the fastest backend at compile- or run-time based on
 *      @c STRINGZILLA_RUNTIME_DISPATCH.
 *  @note Every lane is correct whatever its length, but throughput is best sorted by length.
 *  @sa sz_sha256_multistate_update_serial, sz_sha256_multistate_update_haswell,
 *      sz_sha256_multistate_update_skylake
 *
 *  Lanes are grouped by vector width and a group advances in lockstep, so it costs as much as its
 *  longest member. Inputs sorted by length put similar lengths in the same group, and
 *  @c sz_sequence_argsort gives that ordering.
 */
STRINGZILLA_API_RUNTIME void sz_sha256_multistate_update(sz_sha256_state_t *states, sz_sequence_t const *texts);

/**
 *  @brief Finalizes many independent SHA256 states, one digest per lane.
 *
 *  @param[in] states Array of @p states_count states.
 *  @param[in] states_count Number of states to finalize, which is the lane count.
 *  @param[out] digests Output buffer of `states_count * STRINGZILLA_SHA256_DIGEST_LENGTH` bytes,
 *      one big-endian digest per lane, in lane order.
 *
 *  Leaves every state unmodified, so a streaming caller can take an interim digest and append more.
 *
 *  @sa sz_sha256_state_digest, sz_sha256_multistate_update
 */
STRINGZILLA_API_RUNTIME void sz_sha256_multistate_digest(sz_sha256_state_t const *states, sz_size_t states_count,
                                                         sz_u8_t *digests);

/** @copydoc sz_bytesum */
STRINGZILLA_API_COMPTIME sz_u64_t sz_bytesum_serial(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash */
STRINGZILLA_API_COMPTIME STRINGZILLA_NO_STACK_PROTECTOR_ sz_u64_t sz_hash_serial(sz_cptr_t text, sz_size_t length,
                                                                                 sz_u64_t seed);

/** @copydoc sz_hash_multiseed */
STRINGZILLA_API_COMPTIME void sz_hash_multiseed_serial(sz_cptr_t text, sz_size_t length, sz_u64_t const *seeds,
                                                       sz_size_t seeds_count, sz_u64_t *hashes);

/** @copydoc sz_fill_random */
STRINGZILLA_API_COMPTIME void sz_fill_random_serial(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
STRINGZILLA_API_COMPTIME void sz_hash_state_init_serial(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_update */
STRINGZILLA_API_COMPTIME void sz_hash_state_update_serial(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_digest */
STRINGZILLA_API_COMPTIME sz_u64_t sz_hash_state_digest_serial(sz_hash_state_t const *state);

/** @copydoc sz_sha256_state_init */
STRINGZILLA_API_COMPTIME void sz_sha256_state_init_serial(sz_sha256_state_t *state);

/** @copydoc sz_sha256_state_update */
STRINGZILLA_API_COMPTIME void sz_sha256_state_update_serial(sz_sha256_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_sha256_state_digest */
STRINGZILLA_API_COMPTIME void sz_sha256_state_digest_serial(
    sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)]);

/** @copydoc sz_sha256_multistate_update */
STRINGZILLA_API_COMPTIME void sz_sha256_multistate_update_serial(sz_sha256_state_t *states, sz_sequence_t const *texts);

/** @copydoc sz_sha256_multistate_digest */
STRINGZILLA_API_COMPTIME void sz_sha256_multistate_digest_serial(sz_sha256_state_t const *states,
                                                                 sz_size_t states_count, sz_u8_t *digests);

#if STRINGZILLA_TARGET_WESTMERE

/** @copydoc sz_hash */
STRINGZILLA_API_COMPTIME STRINGZILLA_NO_STACK_PROTECTOR_ sz_u64_t sz_hash_westmere(sz_cptr_t text, sz_size_t length,
                                                                                   sz_u64_t seed);

/** @copydoc sz_hash_multiseed */
STRINGZILLA_API_COMPTIME void sz_hash_multiseed_westmere(sz_cptr_t text, sz_size_t length, sz_u64_t const *seeds,
                                                         sz_size_t seeds_count, sz_u64_t *hashes);

/** @copydoc sz_fill_random */
STRINGZILLA_API_COMPTIME void sz_fill_random_westmere(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
STRINGZILLA_API_COMPTIME void sz_hash_state_init_westmere(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_update */
STRINGZILLA_API_COMPTIME void sz_hash_state_update_westmere(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_digest */
STRINGZILLA_API_COMPTIME sz_u64_t sz_hash_state_digest_westmere(sz_hash_state_t const *state);

#endif

#if STRINGZILLA_TARGET_GOLDMONT

/** @copydoc sz_sha256_state_init */
STRINGZILLA_API_COMPTIME void sz_sha256_state_init_goldmont(sz_sha256_state_t *state);

/** @copydoc sz_sha256_state_update */
STRINGZILLA_API_COMPTIME void sz_sha256_state_update_goldmont(sz_sha256_state_t *state, sz_cptr_t text,
                                                              sz_size_t length);

/** @copydoc sz_sha256_state_digest */
STRINGZILLA_API_COMPTIME void sz_sha256_state_digest_goldmont(
    sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)]);

/** @copydoc sz_sha256_multistate_update */
STRINGZILLA_API_COMPTIME void sz_sha256_multistate_update_goldmont(sz_sha256_state_t *states,
                                                                   sz_sequence_t const *texts);

/** @copydoc sz_sha256_multistate_digest */
STRINGZILLA_API_COMPTIME void sz_sha256_multistate_digest_goldmont(sz_sha256_state_t const *states,
                                                                   sz_size_t states_count, sz_u8_t *digests);

#endif

#if STRINGZILLA_TARGET_HASWELL

/** @copydoc sz_bytesum */
STRINGZILLA_API_COMPTIME sz_u64_t sz_bytesum_haswell(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_sha256_multistate_update */
STRINGZILLA_API_COMPTIME void sz_sha256_multistate_update_haswell(sz_sha256_state_t *states,
                                                                  sz_sequence_t const *texts);

/** @copydoc sz_sha256_multistate_digest */
STRINGZILLA_API_COMPTIME void sz_sha256_multistate_digest_haswell(sz_sha256_state_t const *states,
                                                                  sz_size_t states_count, sz_u8_t *digests);

#endif

#if STRINGZILLA_TARGET_SKYLAKE

/** @copydoc sz_bytesum */
STRINGZILLA_API_COMPTIME sz_u64_t sz_bytesum_skylake(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash */
STRINGZILLA_API_COMPTIME STRINGZILLA_NO_STACK_PROTECTOR_ sz_u64_t sz_hash_skylake(sz_cptr_t text, sz_size_t length,
                                                                                  sz_u64_t seed);

/** @copydoc sz_fill_random */
STRINGZILLA_API_COMPTIME void sz_fill_random_skylake(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
STRINGZILLA_API_COMPTIME void sz_hash_state_init_skylake(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_update */
STRINGZILLA_API_COMPTIME void sz_hash_state_update_skylake(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_digest */
STRINGZILLA_API_COMPTIME sz_u64_t sz_hash_state_digest_skylake(sz_hash_state_t const *state);

/** @copydoc sz_sha256_multistate_update */
STRINGZILLA_API_COMPTIME void sz_sha256_multistate_update_skylake(sz_sha256_state_t *states,
                                                                  sz_sequence_t const *texts);

/** @copydoc sz_sha256_multistate_digest */
STRINGZILLA_API_COMPTIME void sz_sha256_multistate_digest_skylake(sz_sha256_state_t const *states,
                                                                  sz_size_t states_count, sz_u8_t *digests);

#endif

#if STRINGZILLA_TARGET_ICELAKE

/** @copydoc sz_bytesum */
STRINGZILLA_API_COMPTIME sz_u64_t sz_bytesum_icelake(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash */
STRINGZILLA_API_COMPTIME STRINGZILLA_NO_STACK_PROTECTOR_ sz_u64_t sz_hash_icelake(sz_cptr_t text, sz_size_t length,
                                                                                  sz_u64_t seed);

/** @copydoc sz_hash_multiseed */
STRINGZILLA_API_COMPTIME void sz_hash_multiseed_icelake(sz_cptr_t text, sz_size_t length, sz_u64_t const *seeds,
                                                        sz_size_t seeds_count, sz_u64_t *hashes);

/** @copydoc sz_fill_random */
STRINGZILLA_API_COMPTIME void sz_fill_random_icelake(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
STRINGZILLA_API_COMPTIME void sz_hash_state_init_icelake(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_update */
STRINGZILLA_API_COMPTIME void sz_hash_state_update_icelake(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_digest */
STRINGZILLA_API_COMPTIME sz_u64_t sz_hash_state_digest_icelake(sz_hash_state_t const *state);

#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_bytesum */
STRINGZILLA_API_COMPTIME sz_u64_t sz_bytesum_neon(sz_cptr_t text, sz_size_t length);

#endif

#if STRINGZILLA_TARGET_NEONAES

/** @copydoc sz_hash */
STRINGZILLA_API_COMPTIME STRINGZILLA_NO_STACK_PROTECTOR_ sz_u64_t sz_hash_neonaes(sz_cptr_t text, sz_size_t length,
                                                                                  sz_u64_t seed);

/** @copydoc sz_hash_multiseed */
STRINGZILLA_API_COMPTIME void sz_hash_multiseed_neonaes(sz_cptr_t text, sz_size_t length, sz_u64_t const *seeds,
                                                        sz_size_t seeds_count, sz_u64_t *hashes);

/** @copydoc sz_fill_random */
STRINGZILLA_API_COMPTIME void sz_fill_random_neonaes(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
STRINGZILLA_API_COMPTIME void sz_hash_state_init_neonaes(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_update */
STRINGZILLA_API_COMPTIME void sz_hash_state_update_neonaes(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_digest */
STRINGZILLA_API_COMPTIME sz_u64_t sz_hash_state_digest_neonaes(sz_hash_state_t const *state);

#endif

#if STRINGZILLA_TARGET_NEONSHA

/** @copydoc sz_sha256_state_init */
STRINGZILLA_API_COMPTIME void sz_sha256_state_init_neonsha(sz_sha256_state_t *state);

/** @copydoc sz_sha256_state_update */
STRINGZILLA_API_COMPTIME void sz_sha256_state_update_neonsha(sz_sha256_state_t *state, sz_cptr_t data,
                                                             sz_size_t length);

/** @copydoc sz_sha256_state_digest */
STRINGZILLA_API_COMPTIME void sz_sha256_state_digest_neonsha(
    sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)]);

#endif

#if STRINGZILLA_TARGET_SVE

/** @copydoc sz_bytesum */
STRINGZILLA_API_COMPTIME sz_u64_t sz_bytesum_sve(sz_cptr_t text, sz_size_t length);

#endif

#if STRINGZILLA_TARGET_SVE2

/** @copydoc sz_bytesum */
STRINGZILLA_API_COMPTIME sz_u64_t sz_bytesum_sve2(sz_cptr_t text, sz_size_t length);

#endif

#if STRINGZILLA_TARGET_SVE2AES

/** @copydoc sz_hash */
STRINGZILLA_API_COMPTIME sz_u64_t sz_hash_sve2aes(sz_cptr_t text, sz_size_t length, sz_u64_t seed);

/** @copydoc sz_fill_random */
STRINGZILLA_API_COMPTIME void sz_fill_random_sve2aes(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);

/** @copydoc sz_hash_state_init */
STRINGZILLA_API_COMPTIME void sz_hash_state_init_sve2aes(sz_hash_state_t *state, sz_u64_t seed);

/** @copydoc sz_hash_state_update */
STRINGZILLA_API_COMPTIME void sz_hash_state_update_sve2aes(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash_state_digest */
STRINGZILLA_API_COMPTIME sz_u64_t sz_hash_state_digest_sve2aes(sz_hash_state_t const *state);

#endif

#pragma endregion Core API

#pragma region Helper Methods

/**
 *  @brief Compares the state of two running hashes.
 *  @note The current content of the @c ins buffer and its length is ignored.
 */
STRINGZILLA_API_COMPTIME sz_bool_t sz_hash_state_equal(sz_hash_state_t const *lhs, sz_hash_state_t const *rhs) {
    // Compare byte-by-byte using sz_equal (safe for packed struct)
    if (!sz_equal((sz_cptr_t)lhs->aes, (sz_cptr_t)rhs->aes, 64)) return sz_false_k;
    if (!sz_equal((sz_cptr_t)lhs->sum, (sz_cptr_t)rhs->sum, 64)) return sz_false_k;
    if (!sz_equal((sz_cptr_t)lhs->key, (sz_cptr_t)rhs->key, 16)) return sz_false_k;
    return sz_true_k;
}

#pragma endregion Helper Methods

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
#include "stringzilla/hash/rvvcrypto.h"
#include "stringzilla/hash/lasx.h"
#include "stringzilla/hash/powervsx.h"

/*  Pick the right implementation for the hashing and checksum kernels. To override this behavior
 *  and precompile all backends - set @c STRINGZILLA_RUNTIME_DISPATCH to 1. */
#pragma region Compile Time Dispatching
#if !STRINGZILLA_RUNTIME_DISPATCH

STRINGZILLA_API_RUNTIME sz_u64_t sz_bytesum(sz_cptr_t text, sz_size_t length) {
#if STRINGZILLA_TARGET_V128RELAXED
    return sz_bytesum_v128relaxed(text, length);
#elif STRINGZILLA_TARGET_V128
    return sz_bytesum_v128(text, length);
#elif STRINGZILLA_TARGET_RVV
    return sz_bytesum_rvv(text, length);
#elif STRINGZILLA_TARGET_LASX
    return sz_bytesum_lasx(text, length);
#elif STRINGZILLA_TARGET_POWERVSX
    return sz_bytesum_powervsx(text, length);
#elif STRINGZILLA_TARGET_ICELAKE
    return sz_bytesum_icelake(text, length);
#elif STRINGZILLA_TARGET_SKYLAKE
    return sz_bytesum_skylake(text, length);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_bytesum_haswell(text, length);
#elif STRINGZILLA_TARGET_SVE2
    return sz_bytesum_sve2(text, length);
#elif STRINGZILLA_TARGET_SVE
    return sz_bytesum_sve(text, length);
#elif STRINGZILLA_TARGET_NEON
    return sz_bytesum_neon(text, length);
#else
    return sz_bytesum_serial(text, length);
#endif
}

STRINGZILLA_API_RUNTIME sz_u64_t sz_hash(sz_cptr_t text, sz_size_t length, sz_u64_t seed) {
#if STRINGZILLA_TARGET_V128RELAXED
    return sz_hash_v128relaxed(text, length, seed);
#elif STRINGZILLA_TARGET_V128
    return sz_hash_v128(text, length, seed);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    return sz_hash_rvvcrypto(text, length, seed);
#elif STRINGZILLA_TARGET_RVV
    return sz_hash_rvv(text, length, seed);
#elif STRINGZILLA_TARGET_LASX
    return sz_hash_lasx(text, length, seed);
#elif STRINGZILLA_TARGET_POWERVSX
    return sz_hash_powervsx(text, length, seed);
#elif STRINGZILLA_TARGET_ICELAKE
    return sz_hash_icelake(text, length, seed);
#elif STRINGZILLA_TARGET_SKYLAKE
    return sz_hash_skylake(text, length, seed);
#elif STRINGZILLA_TARGET_WESTMERE
    return sz_hash_westmere(text, length, seed);
#elif STRINGZILLA_TARGET_SVE2AES
    return sz_hash_sve2aes(text, length, seed);
#elif STRINGZILLA_TARGET_NEONAES
    return sz_hash_neonaes(text, length, seed);
#else
    return sz_hash_serial(text, length, seed);
#endif
}

STRINGZILLA_API_RUNTIME void sz_hash_multiseed(sz_cptr_t text, sz_size_t length,             //
                                               sz_u64_t const *seeds, sz_size_t seeds_count, //
                                               sz_u64_t *hashes) {
#if STRINGZILLA_TARGET_V128RELAXED
    sz_hash_multiseed_v128relaxed(text, length, seeds, seeds_count, hashes);
#elif STRINGZILLA_TARGET_V128
    sz_hash_multiseed_v128(text, length, seeds, seeds_count, hashes);
#elif STRINGZILLA_TARGET_ICELAKE
    sz_hash_multiseed_icelake(text, length, seeds, seeds_count, hashes);
#elif STRINGZILLA_TARGET_WESTMERE
    sz_hash_multiseed_westmere(text, length, seeds, seeds_count, hashes);
#elif STRINGZILLA_TARGET_NEONAES
    sz_hash_multiseed_neonaes(text, length, seeds, seeds_count, hashes);
#else
    sz_hash_multiseed_serial(text, length, seeds, seeds_count, hashes);
#endif
}

STRINGZILLA_API_RUNTIME void sz_fill_random(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
#if STRINGZILLA_TARGET_V128RELAXED
    sz_fill_random_v128relaxed(text, length, nonce);
#elif STRINGZILLA_TARGET_V128
    sz_fill_random_v128(text, length, nonce);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    sz_fill_random_rvvcrypto(text, length, nonce);
#elif STRINGZILLA_TARGET_RVV
    sz_fill_random_rvv(text, length, nonce);
#elif STRINGZILLA_TARGET_LASX
    sz_fill_random_lasx(text, length, nonce);
#elif STRINGZILLA_TARGET_POWERVSX
    sz_fill_random_powervsx(text, length, nonce);
#elif STRINGZILLA_TARGET_ICELAKE
    sz_fill_random_icelake(text, length, nonce);
#elif STRINGZILLA_TARGET_SKYLAKE
    sz_fill_random_skylake(text, length, nonce);
#elif STRINGZILLA_TARGET_WESTMERE
    sz_fill_random_westmere(text, length, nonce);
#elif STRINGZILLA_TARGET_SVE2AES
    sz_fill_random_sve2aes(text, length, nonce);
#elif STRINGZILLA_TARGET_NEONAES
    sz_fill_random_neonaes(text, length, nonce);
#else
    sz_fill_random_serial(text, length, nonce);
#endif
}

STRINGZILLA_API_RUNTIME void sz_hash_state_init(sz_hash_state_t *state, sz_u64_t seed) {
#if STRINGZILLA_TARGET_V128RELAXED
    sz_hash_state_init_v128relaxed(state, seed);
#elif STRINGZILLA_TARGET_V128
    sz_hash_state_init_v128(state, seed);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    sz_hash_state_init_rvvcrypto(state, seed);
#elif STRINGZILLA_TARGET_RVV
    sz_hash_state_init_rvv(state, seed);
#elif STRINGZILLA_TARGET_LASX
    sz_hash_state_init_lasx(state, seed);
#elif STRINGZILLA_TARGET_POWERVSX
    sz_hash_state_init_powervsx(state, seed);
#elif STRINGZILLA_TARGET_ICELAKE
    sz_hash_state_init_icelake(state, seed);
#elif STRINGZILLA_TARGET_SKYLAKE
    sz_hash_state_init_skylake(state, seed);
#elif STRINGZILLA_TARGET_WESTMERE
    sz_hash_state_init_westmere(state, seed);
#elif STRINGZILLA_TARGET_SVE2AES
    sz_hash_state_init_sve2aes(state, seed);
#elif STRINGZILLA_TARGET_NEONAES
    sz_hash_state_init_neonaes(state, seed);
#else
    sz_hash_state_init_serial(state, seed);
#endif
}

STRINGZILLA_API_RUNTIME void sz_hash_state_update(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
#if STRINGZILLA_TARGET_V128RELAXED
    sz_hash_state_update_v128relaxed(state, text, length);
#elif STRINGZILLA_TARGET_V128
    sz_hash_state_update_v128(state, text, length);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    sz_hash_state_update_rvvcrypto(state, text, length);
#elif STRINGZILLA_TARGET_RVV
    sz_hash_state_update_rvv(state, text, length);
#elif STRINGZILLA_TARGET_LASX
    sz_hash_state_update_lasx(state, text, length);
#elif STRINGZILLA_TARGET_POWERVSX
    sz_hash_state_update_powervsx(state, text, length);
#elif STRINGZILLA_TARGET_ICELAKE
    sz_hash_state_update_icelake(state, text, length);
#elif STRINGZILLA_TARGET_SKYLAKE
    sz_hash_state_update_skylake(state, text, length);
#elif STRINGZILLA_TARGET_WESTMERE
    sz_hash_state_update_westmere(state, text, length);
#elif STRINGZILLA_TARGET_SVE2AES
    sz_hash_state_update_sve2aes(state, text, length);
#elif STRINGZILLA_TARGET_NEONAES
    sz_hash_state_update_neonaes(state, text, length);
#else
    sz_hash_state_update_serial(state, text, length);
#endif
}

STRINGZILLA_API_RUNTIME sz_u64_t sz_hash_state_digest(sz_hash_state_t const *state) {
#if STRINGZILLA_TARGET_V128RELAXED
    return sz_hash_state_digest_v128relaxed(state);
#elif STRINGZILLA_TARGET_V128
    return sz_hash_state_digest_v128(state);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    return sz_hash_state_digest_rvvcrypto(state);
#elif STRINGZILLA_TARGET_RVV
    return sz_hash_state_digest_rvv(state);
#elif STRINGZILLA_TARGET_LASX
    return sz_hash_state_digest_lasx(state);
#elif STRINGZILLA_TARGET_POWERVSX
    return sz_hash_state_digest_powervsx(state);
#elif STRINGZILLA_TARGET_ICELAKE
    return sz_hash_state_digest_icelake(state);
#elif STRINGZILLA_TARGET_SKYLAKE
    return sz_hash_state_digest_skylake(state);
#elif STRINGZILLA_TARGET_WESTMERE
    return sz_hash_state_digest_westmere(state);
#elif STRINGZILLA_TARGET_SVE2AES
    return sz_hash_state_digest_sve2aes(state);
#elif STRINGZILLA_TARGET_NEONAES
    return sz_hash_state_digest_neonaes(state);
#else
    return sz_hash_state_digest_serial(state);
#endif
}

STRINGZILLA_API_RUNTIME void sz_sha256_state_init(sz_sha256_state_t *state) {
#if STRINGZILLA_TARGET_V128
    sz_sha256_state_init_v128(state);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    sz_sha256_state_init_rvvcrypto(state);
#elif STRINGZILLA_TARGET_RVV
    sz_sha256_state_init_rvv(state);
#elif STRINGZILLA_TARGET_LASX
    sz_sha256_state_init_lasx(state);
#elif STRINGZILLA_TARGET_POWERVSX
    sz_sha256_state_init_powervsx(state);
#elif STRINGZILLA_TARGET_NEONSHA
    sz_sha256_state_init_neonsha(state);
#elif STRINGZILLA_TARGET_GOLDMONT
    sz_sha256_state_init_goldmont(state);
#else
    sz_sha256_state_init_serial(state);
#endif
}

STRINGZILLA_API_RUNTIME void sz_sha256_state_update(sz_sha256_state_t *state, sz_cptr_t data, sz_size_t length) {
#if STRINGZILLA_TARGET_V128
    sz_sha256_state_update_v128(state, data, length);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    sz_sha256_state_update_rvvcrypto(state, data, length);
#elif STRINGZILLA_TARGET_RVV
    sz_sha256_state_update_rvv(state, data, length);
#elif STRINGZILLA_TARGET_LASX
    sz_sha256_state_update_lasx(state, data, length);
#elif STRINGZILLA_TARGET_POWERVSX
    sz_sha256_state_update_powervsx(state, data, length);
#elif STRINGZILLA_TARGET_NEONSHA
    sz_sha256_state_update_neonsha(state, data, length);
#elif STRINGZILLA_TARGET_GOLDMONT
    sz_sha256_state_update_goldmont(state, data, length);
#else
    sz_sha256_state_update_serial(state, data, length);
#endif
}

STRINGZILLA_API_RUNTIME void sz_sha256_state_digest(sz_sha256_state_t const *state,
                                                    sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)]) {
#if STRINGZILLA_TARGET_V128
    sz_sha256_state_digest_v128(state, digest);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    sz_sha256_state_digest_rvvcrypto(state, digest);
#elif STRINGZILLA_TARGET_RVV
    sz_sha256_state_digest_rvv(state, digest);
#elif STRINGZILLA_TARGET_LASX
    sz_sha256_state_digest_lasx(state, digest);
#elif STRINGZILLA_TARGET_POWERVSX
    sz_sha256_state_digest_powervsx(state, digest);
#elif STRINGZILLA_TARGET_NEONSHA
    sz_sha256_state_digest_neonsha(state, digest);
#elif STRINGZILLA_TARGET_GOLDMONT
    sz_sha256_state_digest_goldmont(state, digest);
#else
    sz_sha256_state_digest_serial(state, digest);
#endif
}

STRINGZILLA_API_RUNTIME void sz_sha256_multistate_update(sz_sha256_state_t *states, sz_sequence_t const *texts) {
#if STRINGZILLA_TARGET_SKYLAKE
    sz_sha256_multistate_update_skylake(states, texts);
#elif STRINGZILLA_TARGET_HASWELL
    sz_sha256_multistate_update_haswell(states, texts);
#elif STRINGZILLA_TARGET_GOLDMONT
    sz_sha256_multistate_update_goldmont(states, texts);
#else
    sz_sha256_multistate_update_serial(states, texts);
#endif
}

STRINGZILLA_API_RUNTIME void sz_sha256_multistate_digest(sz_sha256_state_t const *states, sz_size_t states_count,
                                                         sz_u8_t *digests) {
#if STRINGZILLA_TARGET_SKYLAKE
    sz_sha256_multistate_digest_skylake(states, states_count, digests);
#elif STRINGZILLA_TARGET_HASWELL
    sz_sha256_multistate_digest_haswell(states, states_count, digests);
#elif STRINGZILLA_TARGET_GOLDMONT
    sz_sha256_multistate_digest_goldmont(states, states_count, digests);
#else
    sz_sha256_multistate_digest_serial(states, states_count, digests);
#endif
}

#endif // !STRINGZILLA_RUNTIME_DISPATCH
#pragma endregion Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_HASH_H_
