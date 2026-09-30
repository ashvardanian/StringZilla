/**
 *  @file include/stringzilla/hash.h
 *  @author Ash Vardanian
 *  @date December 1, 2024
 *  @brief Hardware-accelerated non-cryptographic string hashing and checksums.
 *
 *  Includes core APIs with hardware-specific backends:
 *
 *  - @c sz_bytesum_best - for byte-level 64-bit unsigned checksums.
 *  - @c sz_hash_best - for 64-bit single-shot hashing using AES instructions.
 *  - @c sz_hash_state_init_best, @c sz_hash_state_update_best, @c sz_hash_state_digest_best -
 *    incremental hashing.
 *  - @c sz_fill_random_best - for populating buffers with pseudo-random noise using AES.
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
#include "stringzilla/capabilities.h" // `sz_capability_t`

#include "stringzilla/compare/serial.h" // `sz_equal_serial_`
#include "stringzilla/hash/serial.h"    // `sz_hash_state_t`, `sz_sha256_state_t`

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
 *  @param[out] checksum The 64-bit unsigned sum.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  Summing the bytes of "hi", 104 + 105:
 *
 *  @code{.c}
 *      #include <stringzilla/stringzilla.h>
 *      int main() {
 *          sz_capability_t capabilities;
 *          sz_u64_t checksum;
 *          sz_cpu_capabilities_enabled(&capabilities);
 *          sz_bytesum_best("hi", 2, &checksum, capabilities, NULL);
 *          return checksum == 209 ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @sa sz_bytesum_serial, sz_bytesum_haswell, sz_bytesum_skylake, sz_bytesum_icelake,
 *      sz_bytesum_neon, sz_bytesum_sve, sz_bytesum_sve2, sz_bytesum_v128, sz_bytesum_v128relaxed,
 *      sz_bytesum_rvv, sz_bytesum_loongsonasx, sz_bytesum_powervsx
 */
STRINGZILLA_API sz_status_t sz_bytesum_best(sz_cptr_t text, sz_size_t length, sz_u64_t *checksum,
                                            sz_capability_t capabilities, void *stream);

/**
 *  @brief Computes the 64-bit unsigned hash of a string, similar to @c std::hash in C++.
 *
 *  @param[in] text String to hash.
 *  @param[in] length Number of bytes in the text.
 *  @param[in] seed 64-bit unsigned seed for the hash.
 *  @param[out] hash The 64-bit hash value.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  It's not cryptographically secure, but it's fast and provides a good distribution. It passes the
 *  SMHasher suite by Austin Appleby with no collisions, even with the `--extra` flag. HASH.md
 *  explains the algorithm in detail.
 *
 *  Telling two different strings apart:
 *
 *  @code{.c}
 *      #include <stringzilla/stringzilla.h>
 *      int main() {
 *          sz_capability_t capabilities;
 *          sz_u64_t first_hash, second_hash;
 *          sz_cpu_capabilities_enabled(&capabilities);
 *          sz_hash_best("hello", 5, 0, &first_hash, capabilities, NULL);
 *          sz_hash_best("world", 5, 0, &second_hash, capabilities, NULL);
 *          return first_hash != second_hash ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note The output is the same on all platforms, in both single-shot and incremental modes.
 *  @sa sz_hash_serial, sz_hash_westmere, sz_hash_skylake, sz_hash_icelake, sz_hash_neonaes,
 *      sz_hash_sve2aes, sz_hash_rvv, sz_hash_rvvcrypto, sz_hash_v128, sz_hash_v128relaxed,
 *      sz_hash_loongsonasx, sz_hash_powervsx
 *  @sa sz_hash_state_init_best, sz_hash_state_update_best, sz_hash_state_digest_best
 */
STRINGZILLA_API sz_status_t sz_hash_best(sz_cptr_t text, sz_size_t length, sz_u64_t seed, sz_u64_t *hash,
                                         sz_capability_t capabilities, void *stream);

/**
 *  @brief Hashes one string under @b many seeds at once, the "multi-seed" hash.
 *
 *  @param[in] text String to hash.
 *  @param[in] length Number of bytes in the text.
 *  @param[in] seeds Array of @p seeds_count 64-bit seeds.
 *  @param[in] seeds_count Number of seeds, and the number of hashes written to @p hashes.
 *  @param[out] hashes Caller-allocated output buffer of @p seeds_count 64-bit hashes.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  Equivalent to hashing with @c sz_hash_best under each of @p seeds, but normalizes the input into
 *  AES blocks @b once and replays the cheap per-seed rounds over that prepared form.
 *
 *  Designed for workloads that need several independent hashes of the same (often short) key:
 *  feature-hashing & Count-Min sketches for BM25/TF-IDF, Bloom filters, cuckoo hashing, MinHash /
 *  SimHash / LSH banding, and rendezvous (HRW) hashing. Two effects make it faster than a loop of
 *  @c sz_hash_best: the branch-heavy load & de-interleave of variable-length short strings is
 *  amortized across all seeds, and AES-capable backends advance several seed-lanes per instruction.
 *
 *  Three seeds, the first matching a plain @c sz_hash_best call:
 *
 *  @code{.c}
 *      #include <stringzilla/stringzilla.h>
 *      int main() {
 *          sz_capability_t capabilities;
 *          sz_u64_t seeds[3] = {1, 2, 3}, hashes[3], hash;
 *          sz_cpu_capabilities_enabled(&capabilities);
 *          sz_hash_multiseed_best("token", 5, seeds, 3, hashes, capabilities, NULL);
 *          sz_hash_best("token", 5, 1, &hash, capabilities, NULL);
 *          return hashes[0] == hash ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note Biggest speedups are for @p length ≤ 64; above that only the input load is shared.
 *  @sa sz_hash_multiseed_serial, sz_hash_multiseed_westmere, sz_hash_multiseed_icelake,
 *      sz_hash_multiseed_neonaes, sz_hash_multiseed_v128, sz_hash_multiseed_v128relaxed
 */
STRINGZILLA_API sz_status_t sz_hash_multiseed_best( //
    sz_cptr_t text, sz_size_t length,               //
    sz_u64_t const *seeds, sz_size_t seeds_count,   //
    sz_u64_t *hashes, sz_capability_t capabilities, void *stream);

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
 *  @param[out] target Output string buffer to be populated.
 *  @param[in] length Number of bytes in the string.
 *  @param[in] nonce "Number used once" to ensure uniqueness of produced blocks.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  Filling two buffers under the same nonce:
 *
 *  @code{.c}
 *      #include <stringzilla/stringzilla.h>
 *      int main() {
 *          sz_capability_t capabilities;
 *          char first_buffer[5], second_buffer[5];
 *          sz_u64_t first_checksum, second_checksum;
 *          sz_cpu_capabilities_enabled(&capabilities);
 *          sz_fill_random_best(first_buffer, 5, 0, capabilities, NULL);
 *          sz_fill_random_best(second_buffer, 5, 0, capabilities, NULL); // ? Same nonce, same output
 *          sz_bytesum_best(first_buffer, 5, &first_checksum, capabilities, NULL);
 *          sz_bytesum_best(second_buffer, 5, &second_checksum, capabilities, NULL);
 *          return first_checksum == second_checksum ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @sa sz_fill_random_serial, sz_fill_random_westmere, sz_fill_random_skylake,
 *      sz_fill_random_icelake, sz_fill_random_neonaes, sz_fill_random_sve2aes, sz_fill_random_rvv,
 *      sz_fill_random_rvvcrypto, sz_fill_random_v128, sz_fill_random_v128relaxed,
 *      sz_fill_random_loongsonasx, sz_fill_random_powervsx
 */
STRINGZILLA_API sz_status_t sz_fill_random_best(sz_ptr_t target, sz_size_t length, sz_u64_t nonce,
                                                sz_capability_t capabilities, void *stream);

/**
 *  @brief Initializes the state for incremental construction of a hash.
 *
 *  @param[out] state The state to initialize.
 *  @param[in] seed The 64-bit unsigned seed for the hash.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  Every backend shares the state layout, so each call of a stream may pass other @p capabilities.
 */
STRINGZILLA_API sz_status_t sz_hash_state_init_best(sz_hash_state_t *state, sz_u64_t seed, sz_capability_t capabilities,
                                                    void *stream);

/**
 *  @brief Updates the state with new data.
 *
 *  @param[inout] state The state to stream.
 *  @param[in] text The new data to include in the hash.
 *  @param[in] length The number of bytes in the new data.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 */
STRINGZILLA_API sz_status_t sz_hash_state_update_best(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length,
                                                      sz_capability_t capabilities, void *stream);

/**
 *  @brief Finalizes the immutable state into the hash.
 *
 *  @param[in] state The state to fold.
 *  @param[out] hash The 64-bit hash value, as @c sz_hash_best gives for the same bytes and seed.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 */
STRINGZILLA_API sz_status_t sz_hash_state_digest_best(sz_hash_state_t const *state, sz_u64_t *hash,
                                                      sz_capability_t capabilities, void *stream);

/**
 *  @brief Initializes the state for incremental SHA256 hashing.
 *
 *  @param[out] state The state to initialize.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  Every backend shares the state layout, so each call of a stream may pass other @p capabilities.
 */
STRINGZILLA_API sz_status_t sz_sha256_state_init_best(sz_sha256_state_t *state, sz_capability_t capabilities,
                                                      void *stream);

/**
 *  @brief Updates the SHA256 state with new data.
 *
 *  @param[inout] state The state to update.
 *  @param[in] text The new data to hash.
 *  @param[in] length The number of bytes in the new data.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 */
STRINGZILLA_API sz_status_t sz_sha256_state_update_best(sz_sha256_state_t *state, sz_cptr_t text, sz_size_t length,
                                                        sz_capability_t capabilities, void *stream);

/**
 *  @brief Finalizes the SHA256 state into the digest, leaving the state open to more data.
 *
 *  @param[in] state The state to finalize.
 *  @param[out] digest Output buffer for the 32-byte (256-bit) digest.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 */
STRINGZILLA_API sz_status_t sz_sha256_state_digest_best(sz_sha256_state_t const *state,
                                                        sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)],
                                                        sz_capability_t capabilities, void *stream);

/**
 *  @brief Advances many independent SHA256 states, one message per lane.
 *
 *  @param[inout] states Array of at least `texts->count` states, each initialized with
 *      @c sz_sha256_state_init_best.
 *  @param[in] texts Sequence supplying the next chunk of each lane's message, with @c count lanes.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
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
 *      #include <stringzilla/stringzilla.h>
 *      int main() {
 *          sz_cptr_t chunks[2] = {"hello", "world"};
 *          sz_sha256_state_t states[2];
 *          sz_u8_t digests[2 * STRINGZILLA_SHA256_DIGEST_LENGTH];
 *          sz_sequence_t texts;
 *          sz_capability_t capabilities;
 *          sz_cpu_capabilities_enabled(&capabilities);
 *          sz_sequence_from_null_terminated_strings(chunks, 2, &texts);
 *          sz_sha256_state_init_best(&states[0], capabilities, NULL);
 *          sz_sha256_state_init_best(&states[1], capabilities, NULL);
 *          sz_sha256_multistate_update_best(states, &texts, capabilities, NULL);
 *          sz_sha256_multistate_digest_best(states, 2, digests, capabilities, NULL);
 *          return digests[0] == 0x2c ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note Every lane is correct whatever its length, but throughput is best sorted by length.
 *  @sa sz_sha256_multistate_update_serial, sz_sha256_multistate_update_goldmont,
 *      sz_sha256_multistate_update_haswell, sz_sha256_multistate_update_skylake
 *
 *  Lanes are grouped by vector width and a group advances in lockstep, so it costs as much as its
 *  longest member. Inputs sorted by length put similar lengths in the same group, and
 *  @c sz_sequence_argsort_best gives that ordering.
 */
STRINGZILLA_API sz_status_t sz_sha256_multistate_update_best(sz_sha256_state_t *states, sz_sequence_t const *texts,
                                                             sz_capability_t capabilities, void *stream);

/**
 *  @brief Finalizes many independent SHA256 states, one digest per lane.
 *
 *  @param[in] states Array of @p states_count states.
 *  @param[in] states_count Number of states to finalize, which is the lane count.
 *  @param[out] digests Output buffer of `states_count * STRINGZILLA_SHA256_DIGEST_LENGTH` bytes,
 *      one big-endian digest per lane, in lane order.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  Leaves every state unmodified, so a streaming caller can take an interim digest and append more.
 *
 *  @sa sz_sha256_state_digest_best, sz_sha256_multistate_update_best
 */
STRINGZILLA_API sz_status_t sz_sha256_multistate_digest_best(sz_sha256_state_t const *states, sz_size_t states_count,
                                                             sz_u8_t *digests, sz_capability_t capabilities,
                                                             void *stream);

/** @copydoc sz_bytesum_best */
STRINGZILLA_API sz_status_t sz_bytesum_serial(sz_cptr_t text, sz_size_t length, sz_u64_t *checksum, void *stream);

/** @copydoc sz_hash_best */
STRINGZILLA_API STRINGZILLA_NO_STACK_PROTECTOR_ sz_status_t sz_hash_serial(sz_cptr_t text, sz_size_t length,
                                                                           sz_u64_t seed, sz_u64_t *hash, void *stream);

/** @copydoc sz_hash_multiseed_best */
STRINGZILLA_API sz_status_t sz_hash_multiseed_serial(sz_cptr_t text, sz_size_t length, sz_u64_t const *seeds,
                                                     sz_size_t seeds_count, sz_u64_t *hashes, void *stream);

/** @copydoc sz_fill_random_best */
STRINGZILLA_API sz_status_t sz_fill_random_serial(sz_ptr_t target, sz_size_t length, sz_u64_t nonce, void *stream);

/** @copydoc sz_hash_state_init_best */
STRINGZILLA_API sz_status_t sz_hash_state_init_serial(sz_hash_state_t *state, sz_u64_t seed, void *stream);

/** @copydoc sz_hash_state_update_best */
STRINGZILLA_API sz_status_t sz_hash_state_update_serial(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length,
                                                        void *stream);

/** @copydoc sz_hash_state_digest_best */
STRINGZILLA_API sz_status_t sz_hash_state_digest_serial(sz_hash_state_t const *state, sz_u64_t *hash, void *stream);

/** @copydoc sz_sha256_state_init_best */
STRINGZILLA_API sz_status_t sz_sha256_state_init_serial(sz_sha256_state_t *state, void *stream);

/** @copydoc sz_sha256_state_update_best */
STRINGZILLA_API sz_status_t sz_sha256_state_update_serial(sz_sha256_state_t *state, sz_cptr_t text, sz_size_t length,
                                                          void *stream);

/** @copydoc sz_sha256_state_digest_best */
STRINGZILLA_API sz_status_t sz_sha256_state_digest_serial(
    sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)], void *stream);

/** @copydoc sz_sha256_multistate_update_best */
STRINGZILLA_API sz_status_t sz_sha256_multistate_update_serial(sz_sha256_state_t *states, sz_sequence_t const *texts,
                                                               void *stream);

/** @copydoc sz_sha256_multistate_digest_best */
STRINGZILLA_API sz_status_t sz_sha256_multistate_digest_serial(sz_sha256_state_t const *states, sz_size_t states_count,
                                                               sz_u8_t *digests, void *stream);

#if STRINGZILLA_TARGET_WESTMERE

/** @copydoc sz_hash_best */
STRINGZILLA_API STRINGZILLA_NO_STACK_PROTECTOR_ sz_status_t sz_hash_westmere(sz_cptr_t text, sz_size_t length,
                                                                             sz_u64_t seed, sz_u64_t *hash,
                                                                             void *stream);

/** @copydoc sz_hash_multiseed_best */
STRINGZILLA_API sz_status_t sz_hash_multiseed_westmere(sz_cptr_t text, sz_size_t length, sz_u64_t const *seeds,
                                                       sz_size_t seeds_count, sz_u64_t *hashes, void *stream);

/** @copydoc sz_fill_random_best */
STRINGZILLA_API sz_status_t sz_fill_random_westmere(sz_ptr_t target, sz_size_t length, sz_u64_t nonce, void *stream);

/** @copydoc sz_hash_state_init_best */
STRINGZILLA_API sz_status_t sz_hash_state_init_westmere(sz_hash_state_t *state, sz_u64_t seed, void *stream);

/** @copydoc sz_hash_state_update_best */
STRINGZILLA_API sz_status_t sz_hash_state_update_westmere(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length,
                                                          void *stream);

/** @copydoc sz_hash_state_digest_best */
STRINGZILLA_API sz_status_t sz_hash_state_digest_westmere(sz_hash_state_t const *state, sz_u64_t *hash, void *stream);

#endif

#if STRINGZILLA_TARGET_GOLDMONT

/** @copydoc sz_sha256_state_init_best */
STRINGZILLA_API sz_status_t sz_sha256_state_init_goldmont(sz_sha256_state_t *state, void *stream);

/** @copydoc sz_sha256_state_update_best */
STRINGZILLA_API sz_status_t sz_sha256_state_update_goldmont(sz_sha256_state_t *state, sz_cptr_t text, sz_size_t length,
                                                            void *stream);

/** @copydoc sz_sha256_state_digest_best */
STRINGZILLA_API sz_status_t sz_sha256_state_digest_goldmont(
    sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)], void *stream);

/** @copydoc sz_sha256_multistate_update_best */
STRINGZILLA_API sz_status_t sz_sha256_multistate_update_goldmont(sz_sha256_state_t *states, sz_sequence_t const *texts,
                                                                 void *stream);

/** @copydoc sz_sha256_multistate_digest_best */
STRINGZILLA_API sz_status_t sz_sha256_multistate_digest_goldmont(sz_sha256_state_t const *states,
                                                                 sz_size_t states_count, sz_u8_t *digests,
                                                                 void *stream);

#endif

#if STRINGZILLA_TARGET_HASWELL

/** @copydoc sz_bytesum_best */
STRINGZILLA_API sz_status_t sz_bytesum_haswell(sz_cptr_t text, sz_size_t length, sz_u64_t *checksum, void *stream);

/** @copydoc sz_sha256_multistate_update_best */
STRINGZILLA_API sz_status_t sz_sha256_multistate_update_haswell(sz_sha256_state_t *states, sz_sequence_t const *texts,
                                                                void *stream);

/** @copydoc sz_sha256_multistate_digest_best */
STRINGZILLA_API sz_status_t sz_sha256_multistate_digest_haswell(sz_sha256_state_t const *states, sz_size_t states_count,
                                                                sz_u8_t *digests, void *stream);

#endif

#if STRINGZILLA_TARGET_SKYLAKE

/** @copydoc sz_bytesum_best */
STRINGZILLA_API sz_status_t sz_bytesum_skylake(sz_cptr_t text, sz_size_t length, sz_u64_t *checksum, void *stream);

/** @copydoc sz_hash_best */
STRINGZILLA_API STRINGZILLA_NO_STACK_PROTECTOR_ sz_status_t sz_hash_skylake(sz_cptr_t text, sz_size_t length,
                                                                            sz_u64_t seed, sz_u64_t *hash,
                                                                            void *stream);

/** @copydoc sz_fill_random_best */
STRINGZILLA_API sz_status_t sz_fill_random_skylake(sz_ptr_t target, sz_size_t length, sz_u64_t nonce, void *stream);

/** @copydoc sz_hash_state_init_best */
STRINGZILLA_API sz_status_t sz_hash_state_init_skylake(sz_hash_state_t *state, sz_u64_t seed, void *stream);

/** @copydoc sz_hash_state_update_best */
STRINGZILLA_API sz_status_t sz_hash_state_update_skylake(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length,
                                                         void *stream);

/** @copydoc sz_hash_state_digest_best */
STRINGZILLA_API sz_status_t sz_hash_state_digest_skylake(sz_hash_state_t const *state, sz_u64_t *hash, void *stream);

/** @copydoc sz_sha256_multistate_update_best */
STRINGZILLA_API sz_status_t sz_sha256_multistate_update_skylake(sz_sha256_state_t *states, sz_sequence_t const *texts,
                                                                void *stream);

/** @copydoc sz_sha256_multistate_digest_best */
STRINGZILLA_API sz_status_t sz_sha256_multistate_digest_skylake(sz_sha256_state_t const *states, sz_size_t states_count,
                                                                sz_u8_t *digests, void *stream);

#endif

#if STRINGZILLA_TARGET_ICELAKE

/** @copydoc sz_bytesum_best */
STRINGZILLA_API sz_status_t sz_bytesum_icelake(sz_cptr_t text, sz_size_t length, sz_u64_t *checksum, void *stream);

/** @copydoc sz_hash_best */
STRINGZILLA_API STRINGZILLA_NO_STACK_PROTECTOR_ sz_status_t sz_hash_icelake(sz_cptr_t text, sz_size_t length,
                                                                            sz_u64_t seed, sz_u64_t *hash,
                                                                            void *stream);

/** @copydoc sz_hash_multiseed_best */
STRINGZILLA_API sz_status_t sz_hash_multiseed_icelake(sz_cptr_t text, sz_size_t length, sz_u64_t const *seeds,
                                                      sz_size_t seeds_count, sz_u64_t *hashes, void *stream);

/** @copydoc sz_fill_random_best */
STRINGZILLA_API sz_status_t sz_fill_random_icelake(sz_ptr_t target, sz_size_t length, sz_u64_t nonce, void *stream);

/** @copydoc sz_hash_state_init_best */
STRINGZILLA_API sz_status_t sz_hash_state_init_icelake(sz_hash_state_t *state, sz_u64_t seed, void *stream);

/** @copydoc sz_hash_state_update_best */
STRINGZILLA_API sz_status_t sz_hash_state_update_icelake(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length,
                                                         void *stream);

/** @copydoc sz_hash_state_digest_best */
STRINGZILLA_API sz_status_t sz_hash_state_digest_icelake(sz_hash_state_t const *state, sz_u64_t *hash, void *stream);

#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_bytesum_best */
STRINGZILLA_API sz_status_t sz_bytesum_neon(sz_cptr_t text, sz_size_t length, sz_u64_t *checksum, void *stream);

#endif

#if STRINGZILLA_TARGET_NEONAES

/** @copydoc sz_hash_best */
STRINGZILLA_API STRINGZILLA_NO_STACK_PROTECTOR_ sz_status_t sz_hash_neonaes(sz_cptr_t text, sz_size_t length,
                                                                            sz_u64_t seed, sz_u64_t *hash,
                                                                            void *stream);

/** @copydoc sz_hash_multiseed_best */
STRINGZILLA_API sz_status_t sz_hash_multiseed_neonaes(sz_cptr_t text, sz_size_t length, sz_u64_t const *seeds,
                                                      sz_size_t seeds_count, sz_u64_t *hashes, void *stream);

/** @copydoc sz_fill_random_best */
STRINGZILLA_API sz_status_t sz_fill_random_neonaes(sz_ptr_t target, sz_size_t length, sz_u64_t nonce, void *stream);

/** @copydoc sz_hash_state_init_best */
STRINGZILLA_API sz_status_t sz_hash_state_init_neonaes(sz_hash_state_t *state, sz_u64_t seed, void *stream);

/** @copydoc sz_hash_state_update_best */
STRINGZILLA_API sz_status_t sz_hash_state_update_neonaes(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length,
                                                         void *stream);

/** @copydoc sz_hash_state_digest_best */
STRINGZILLA_API sz_status_t sz_hash_state_digest_neonaes(sz_hash_state_t const *state, sz_u64_t *hash, void *stream);

#endif

#if STRINGZILLA_TARGET_NEONSHA

/** @copydoc sz_sha256_state_init_best */
STRINGZILLA_API sz_status_t sz_sha256_state_init_neonsha(sz_sha256_state_t *state, void *stream);

/** @copydoc sz_sha256_state_update_best */
STRINGZILLA_API sz_status_t sz_sha256_state_update_neonsha(sz_sha256_state_t *state, sz_cptr_t text, sz_size_t length,
                                                           void *stream);

/** @copydoc sz_sha256_state_digest_best */
STRINGZILLA_API sz_status_t sz_sha256_state_digest_neonsha(
    sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)], void *stream);

#endif

#if STRINGZILLA_TARGET_SVE

/** @copydoc sz_bytesum_best */
STRINGZILLA_API sz_status_t sz_bytesum_sve(sz_cptr_t text, sz_size_t length, sz_u64_t *checksum, void *stream);

#endif

#if STRINGZILLA_TARGET_SVE2

/** @copydoc sz_bytesum_best */
STRINGZILLA_API sz_status_t sz_bytesum_sve2(sz_cptr_t text, sz_size_t length, sz_u64_t *checksum, void *stream);

#endif

#if STRINGZILLA_TARGET_SVE2AES

/** @copydoc sz_hash_best */
STRINGZILLA_API sz_status_t sz_hash_sve2aes(sz_cptr_t text, sz_size_t length, sz_u64_t seed, sz_u64_t *hash,
                                            void *stream);

/** @copydoc sz_fill_random_best */
STRINGZILLA_API sz_status_t sz_fill_random_sve2aes(sz_ptr_t target, sz_size_t length, sz_u64_t nonce, void *stream);

/** @copydoc sz_hash_state_init_best */
STRINGZILLA_API sz_status_t sz_hash_state_init_sve2aes(sz_hash_state_t *state, sz_u64_t seed, void *stream);

/** @copydoc sz_hash_state_update_best */
STRINGZILLA_API sz_status_t sz_hash_state_update_sve2aes(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length,
                                                         void *stream);

/** @copydoc sz_hash_state_digest_best */
STRINGZILLA_API sz_status_t sz_hash_state_digest_sve2aes(sz_hash_state_t const *state, sz_u64_t *hash, void *stream);

#endif

#if STRINGZILLA_TARGET_RVV

/** @copydoc sz_bytesum_best */
STRINGZILLA_API sz_status_t sz_bytesum_rvv(sz_cptr_t text, sz_size_t length, sz_u64_t *checksum, void *stream);

/** @copydoc sz_hash_best */
STRINGZILLA_API STRINGZILLA_NO_STACK_PROTECTOR_ sz_status_t sz_hash_rvv(sz_cptr_t text, sz_size_t length, sz_u64_t seed,
                                                                        sz_u64_t *hash, void *stream);

/** @copydoc sz_fill_random_best */
STRINGZILLA_API sz_status_t sz_fill_random_rvv(sz_ptr_t target, sz_size_t length, sz_u64_t nonce, void *stream);

/** @copydoc sz_hash_state_init_best */
STRINGZILLA_API sz_status_t sz_hash_state_init_rvv(sz_hash_state_t *state, sz_u64_t seed, void *stream);

/** @copydoc sz_hash_state_update_best */
STRINGZILLA_API sz_status_t sz_hash_state_update_rvv(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length,
                                                     void *stream);

/** @copydoc sz_hash_state_digest_best */
STRINGZILLA_API sz_status_t sz_hash_state_digest_rvv(sz_hash_state_t const *state, sz_u64_t *hash, void *stream);

/** @copydoc sz_sha256_state_init_best */
STRINGZILLA_API sz_status_t sz_sha256_state_init_rvv(sz_sha256_state_t *state, void *stream);

/** @copydoc sz_sha256_state_update_best */
STRINGZILLA_API sz_status_t sz_sha256_state_update_rvv(sz_sha256_state_t *state, sz_cptr_t text, sz_size_t length,
                                                       void *stream);

/** @copydoc sz_sha256_state_digest_best */
STRINGZILLA_API sz_status_t sz_sha256_state_digest_rvv(sz_sha256_state_t const *state,
                                                       sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)],
                                                       void *stream);

#endif

#if STRINGZILLA_TARGET_RVVCRYPTO

/** @copydoc sz_hash_best */
STRINGZILLA_API STRINGZILLA_NO_STACK_PROTECTOR_ sz_status_t sz_hash_rvvcrypto(sz_cptr_t text, sz_size_t length,
                                                                              sz_u64_t seed, sz_u64_t *hash,
                                                                              void *stream);

/** @copydoc sz_fill_random_best */
STRINGZILLA_API sz_status_t sz_fill_random_rvvcrypto(sz_ptr_t target, sz_size_t length, sz_u64_t nonce, void *stream);

/** @copydoc sz_hash_state_init_best */
STRINGZILLA_API sz_status_t sz_hash_state_init_rvvcrypto(sz_hash_state_t *state, sz_u64_t seed, void *stream);

/** @copydoc sz_hash_state_update_best */
STRINGZILLA_API sz_status_t sz_hash_state_update_rvvcrypto(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length,
                                                           void *stream);

/** @copydoc sz_hash_state_digest_best */
STRINGZILLA_API sz_status_t sz_hash_state_digest_rvvcrypto(sz_hash_state_t const *state, sz_u64_t *hash, void *stream);

/** @copydoc sz_sha256_state_init_best */
STRINGZILLA_API sz_status_t sz_sha256_state_init_rvvcrypto(sz_sha256_state_t *state, void *stream);

/** @copydoc sz_sha256_state_update_best */
STRINGZILLA_API sz_status_t sz_sha256_state_update_rvvcrypto(sz_sha256_state_t *state, sz_cptr_t text, sz_size_t length,
                                                             void *stream);

/** @copydoc sz_sha256_state_digest_best */
STRINGZILLA_API sz_status_t sz_sha256_state_digest_rvvcrypto(
    sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)], void *stream);

#endif

#if STRINGZILLA_TARGET_V128

/** @copydoc sz_bytesum_best */
STRINGZILLA_API sz_status_t sz_bytesum_v128(sz_cptr_t text, sz_size_t length, sz_u64_t *checksum, void *stream);

/** @copydoc sz_hash_best */
STRINGZILLA_API STRINGZILLA_NO_STACK_PROTECTOR_ sz_status_t sz_hash_v128(sz_cptr_t text, sz_size_t length,
                                                                         sz_u64_t seed, sz_u64_t *hash, void *stream);

/** @copydoc sz_hash_multiseed_best */
STRINGZILLA_API sz_status_t sz_hash_multiseed_v128(sz_cptr_t text, sz_size_t length, sz_u64_t const *seeds,
                                                   sz_size_t seeds_count, sz_u64_t *hashes, void *stream);

/** @copydoc sz_fill_random_best */
STRINGZILLA_API sz_status_t sz_fill_random_v128(sz_ptr_t target, sz_size_t length, sz_u64_t nonce, void *stream);

/** @copydoc sz_hash_state_init_best */
STRINGZILLA_API sz_status_t sz_hash_state_init_v128(sz_hash_state_t *state, sz_u64_t seed, void *stream);

/** @copydoc sz_hash_state_update_best */
STRINGZILLA_API sz_status_t sz_hash_state_update_v128(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length,
                                                      void *stream);

/** @copydoc sz_hash_state_digest_best */
STRINGZILLA_API sz_status_t sz_hash_state_digest_v128(sz_hash_state_t const *state, sz_u64_t *hash, void *stream);

/** @copydoc sz_sha256_state_init_best */
STRINGZILLA_API sz_status_t sz_sha256_state_init_v128(sz_sha256_state_t *state, void *stream);

/** @copydoc sz_sha256_state_update_best */
STRINGZILLA_API sz_status_t sz_sha256_state_update_v128(sz_sha256_state_t *state, sz_cptr_t text, sz_size_t length,
                                                        void *stream);

/** @copydoc sz_sha256_state_digest_best */
STRINGZILLA_API sz_status_t sz_sha256_state_digest_v128(sz_sha256_state_t const *state,
                                                        sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)],
                                                        void *stream);

#endif

#if STRINGZILLA_TARGET_V128RELAXED

/** @copydoc sz_bytesum_best */
STRINGZILLA_API sz_status_t sz_bytesum_v128relaxed(sz_cptr_t text, sz_size_t length, sz_u64_t *checksum, void *stream);

/** @copydoc sz_hash_best */
STRINGZILLA_API STRINGZILLA_NO_STACK_PROTECTOR_ sz_status_t sz_hash_v128relaxed(sz_cptr_t text, sz_size_t length,
                                                                                sz_u64_t seed, sz_u64_t *hash,
                                                                                void *stream);

/** @copydoc sz_hash_multiseed_best */
STRINGZILLA_API sz_status_t sz_hash_multiseed_v128relaxed(sz_cptr_t text, sz_size_t length, sz_u64_t const *seeds,
                                                          sz_size_t seeds_count, sz_u64_t *hashes, void *stream);

/** @copydoc sz_fill_random_best */
STRINGZILLA_API sz_status_t sz_fill_random_v128relaxed(sz_ptr_t target, sz_size_t length, sz_u64_t nonce, void *stream);

/** @copydoc sz_hash_state_init_best */
STRINGZILLA_API sz_status_t sz_hash_state_init_v128relaxed(sz_hash_state_t *state, sz_u64_t seed, void *stream);

/** @copydoc sz_hash_state_update_best */
STRINGZILLA_API sz_status_t sz_hash_state_update_v128relaxed(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length,
                                                             void *stream);

/** @copydoc sz_hash_state_digest_best */
STRINGZILLA_API sz_status_t sz_hash_state_digest_v128relaxed(sz_hash_state_t const *state, sz_u64_t *hash,
                                                             void *stream);

#endif

#if STRINGZILLA_TARGET_LOONGSONASX

/** @copydoc sz_bytesum_best */
STRINGZILLA_API sz_status_t sz_bytesum_loongsonasx(sz_cptr_t text, sz_size_t length, sz_u64_t *checksum, void *stream);

/** @copydoc sz_hash_best */
STRINGZILLA_API STRINGZILLA_NO_STACK_PROTECTOR_ sz_status_t sz_hash_loongsonasx(sz_cptr_t text, sz_size_t length,
                                                                                sz_u64_t seed, sz_u64_t *hash,
                                                                                void *stream);

/** @copydoc sz_fill_random_best */
STRINGZILLA_API sz_status_t sz_fill_random_loongsonasx(sz_ptr_t target, sz_size_t length, sz_u64_t nonce, void *stream);

/** @copydoc sz_hash_state_init_best */
STRINGZILLA_API sz_status_t sz_hash_state_init_loongsonasx(sz_hash_state_t *state, sz_u64_t seed, void *stream);

/** @copydoc sz_hash_state_update_best */
STRINGZILLA_API sz_status_t sz_hash_state_update_loongsonasx(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length,
                                                             void *stream);

/** @copydoc sz_hash_state_digest_best */
STRINGZILLA_API sz_status_t sz_hash_state_digest_loongsonasx(sz_hash_state_t const *state, sz_u64_t *hash,
                                                             void *stream);

/** @copydoc sz_sha256_state_init_best */
STRINGZILLA_API sz_status_t sz_sha256_state_init_loongsonasx(sz_sha256_state_t *state, void *stream);

/** @copydoc sz_sha256_state_update_best */
STRINGZILLA_API sz_status_t sz_sha256_state_update_loongsonasx(sz_sha256_state_t *state, sz_cptr_t text,
                                                               sz_size_t length, void *stream);

/** @copydoc sz_sha256_state_digest_best */
STRINGZILLA_API sz_status_t sz_sha256_state_digest_loongsonasx(
    sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)], void *stream);

#endif

#if STRINGZILLA_TARGET_POWERVSX

/** @copydoc sz_bytesum_best */
STRINGZILLA_API sz_status_t sz_bytesum_powervsx(sz_cptr_t text, sz_size_t length, sz_u64_t *checksum, void *stream);

/** @copydoc sz_hash_best */
STRINGZILLA_API STRINGZILLA_NO_STACK_PROTECTOR_ sz_status_t sz_hash_powervsx(sz_cptr_t text, sz_size_t length,
                                                                             sz_u64_t seed, sz_u64_t *hash,
                                                                             void *stream);

/** @copydoc sz_fill_random_best */
STRINGZILLA_API sz_status_t sz_fill_random_powervsx(sz_ptr_t target, sz_size_t length, sz_u64_t nonce, void *stream);

/** @copydoc sz_hash_state_init_best */
STRINGZILLA_API sz_status_t sz_hash_state_init_powervsx(sz_hash_state_t *state, sz_u64_t seed, void *stream);

/** @copydoc sz_hash_state_update_best */
STRINGZILLA_API sz_status_t sz_hash_state_update_powervsx(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length,
                                                          void *stream);

/** @copydoc sz_hash_state_digest_best */
STRINGZILLA_API sz_status_t sz_hash_state_digest_powervsx(sz_hash_state_t const *state, sz_u64_t *hash, void *stream);

/** @copydoc sz_sha256_state_init_best */
STRINGZILLA_API sz_status_t sz_sha256_state_init_powervsx(sz_sha256_state_t *state, void *stream);

/** @copydoc sz_sha256_state_update_best */
STRINGZILLA_API sz_status_t sz_sha256_state_update_powervsx(sz_sha256_state_t *state, sz_cptr_t text, sz_size_t length,
                                                            void *stream);

/** @copydoc sz_sha256_state_digest_best */
STRINGZILLA_API sz_status_t sz_sha256_state_digest_powervsx(
    sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)], void *stream);

#endif

/**
 *  @brief Finds the hashing kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_hash_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion Core API

#pragma region Helper Methods

/**
 *  @brief Compares the state of two running hashes.
 *  @note The current content of the @c ins buffer and its length is ignored.
 */
STRINGZILLA_INLINE sz_bool_t sz_hash_state_equal(sz_hash_state_t const *lhs, sz_hash_state_t const *rhs) {
    // Compare byte-by-byte (safe for packed struct)
    if (!sz_equal_serial_((sz_cptr_t)lhs->aes, (sz_cptr_t)rhs->aes, 64)) return sz_false_k;
    if (!sz_equal_serial_((sz_cptr_t)lhs->sum, (sz_cptr_t)rhs->sum, 64)) return sz_false_k;
    if (!sz_equal_serial_((sz_cptr_t)lhs->key, (sz_cptr_t)rhs->key, 16)) return sz_false_k;
    return sz_true_k;
}

#pragma endregion Helper Methods

#if STRINGZILLA_HEADER_ONLY
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
#include "stringzilla/hash/loongsonasx.h"
#include "stringzilla/hash/powervsx.h"
#endif // STRINGZILLA_HEADER_ONLY

#pragma region Dispatch

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_bytesum_best(sz_cptr_t text, sz_size_t length, sz_u64_t *checksum,
                                            sz_capability_t capabilities, void *stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(checksum), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_hash_best(sz_cptr_t text, sz_size_t length, sz_u64_t seed, sz_u64_t *hash,
                                         sz_capability_t capabilities, void *stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(seed), sz_unused_(hash), sz_unused_(capabilities),
        sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_hash_multiseed_best( //
    sz_cptr_t text, sz_size_t length,               //
    sz_u64_t const *seeds, sz_size_t seeds_count,   //
    sz_u64_t *hashes, sz_capability_t capabilities, void *stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(seeds), sz_unused_(seeds_count), sz_unused_(hashes),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_fill_random_best(sz_ptr_t target, sz_size_t length, sz_u64_t nonce,
                                                sz_capability_t capabilities, void *stream) {
    sz_unused_(target), sz_unused_(length), sz_unused_(nonce), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_hash_state_init_best(sz_hash_state_t *state, sz_u64_t seed, sz_capability_t capabilities,
                                                    void *stream) {
    sz_unused_(state), sz_unused_(seed), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_hash_state_update_best(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length,
                                                      sz_capability_t capabilities, void *stream) {
    sz_unused_(state), sz_unused_(text), sz_unused_(length), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_hash_state_digest_best(sz_hash_state_t const *state, sz_u64_t *hash,
                                                      sz_capability_t capabilities, void *stream) {
    sz_unused_(state), sz_unused_(hash), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_sha256_state_init_best(sz_sha256_state_t *state, sz_capability_t capabilities,
                                                      void *stream) {
    sz_unused_(state), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_sha256_state_update_best(sz_sha256_state_t *state, sz_cptr_t text, sz_size_t length,
                                                        sz_capability_t capabilities, void *stream) {
    sz_unused_(state), sz_unused_(text), sz_unused_(length), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_sha256_state_digest_best(sz_sha256_state_t const *state,
                                                        sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)],
                                                        sz_capability_t capabilities, void *stream) {
    sz_unused_(state), sz_unused_(digest), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_sha256_multistate_update_best(sz_sha256_state_t *states, sz_sequence_t const *texts,
                                                             sz_capability_t capabilities, void *stream) {
    sz_unused_(states), sz_unused_(texts), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_sha256_multistate_digest_best(sz_sha256_state_t const *states, sz_size_t states_count,
                                                             sz_u8_t *digests, sz_capability_t capabilities,
                                                             void *stream) {
    sz_unused_(states), sz_unused_(states_count), sz_unused_(digests), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_hash_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_HASH_H_
