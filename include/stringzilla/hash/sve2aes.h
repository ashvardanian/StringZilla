/**
 *  @file include/stringzilla/hash/sve2aes.h
 *  @author Ash Vardanian
 *  @date March 13, 2025
 *  @brief SVE2 + AES backend for string hashing and checksums.
 *
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_SVE2AES_H_
#define STRINGZILLA_HASH_SVE2AES_H_

#include "stringzilla/types.h"
#include "stringzilla/hash/serial.h"
#include "stringzilla/hash/neonaes.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_ARM64_
#if STRINGZILLA_TARGET_SVE2AES
/* `+crypto` matches `hash/neonaes.h`, so its always-inline helpers inline into these kernels. */
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve+sve2+sve2-aes+crypto"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve+sve2+sve2-aes+crypto")
#endif

/**
 *  @brief Emulates the Intel's AES-NI @c AESENC instruction with Arm SVE2.
 *  @see Emulating x86 AES Intrinsics on ARMv8-A by Michael Brase: https://blog.michaelbrase.com/2018/05/08/emulating-x86-aes-intrinsics-on-armv8-a/
 */
STRINGZILLA_INLINE svuint8_t sz_emulate_aesenc_u8x16_sve2_(svuint8_t state_u8x, svuint8_t round_key_u8x) {
    return sveor_u8_x(svptrue_b8(), svaesmc_u8(svaese_u8(state_u8x, svdup_n_u8(0))), round_key_u8x);
}

/** A variant of @c sz_hash_sve2aes for strings up to 16 bytes long - smallest SVE register size. */
STRINGZILLA_INLINE sz_u64_t sz_hash_upto16_sve2aes_(sz_cptr_t text, sz_size_t length, sz_u64_t seed) {
    svuint8_t state_aes_u8x, state_sum_u8x, state_key_u8x;

    // To load and store the seed, we don't even need a `svwhilelt_b64(0, 2)`.
    state_key_u8x = svreinterpret_u8_u64(svdup_n_u64(seed));
    svbool_t const all_b64x = svptrue_b64();
    svbool_t const all_b8x = svptrue_b8();

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    svuint64_t pi0_u64x = svld1_u64(all_b64x, pi);
    svuint64_t pi1_u64x = svld1_u64(all_b64x, pi + 8);
    state_aes_u8x = sveor_u8_x(all_b8x, state_key_u8x, svreinterpret_u8_u64(pi0_u64x));
    state_sum_u8x = sveor_u8_x(all_b8x, state_key_u8x, svreinterpret_u8_u64(pi1_u64x));

    // We will only use the first 128 bits of the shuffle mask
    svuint8_t const order_u8x = svld1_u8(all_b8x, sz_hash_u8x16x4_shuffle_());

    // This is our best case for SVE2 dominance over NEON - we can load the data in one go with a predicate.
    svuint8_t block_u8x = svld1_u8(svwhilelt_b8((sz_u64_t)0, (sz_u64_t)length), (sz_u8_t const *)text);
    // One round of hashing logic
    state_aes_u8x = sz_emulate_aesenc_u8x16_sve2_(state_aes_u8x, block_u8x);
    svuint8_t sum_shuffled_u8x = svtbl_u8(state_sum_u8x, order_u8x);
    state_sum_u8x = svreinterpret_u8_u64(
        svadd_u64_x(all_b64x, svreinterpret_u64_u8(sum_shuffled_u8x), svreinterpret_u64_u8(block_u8x)));

    // Now mix, folding the length into the key
    svuint64_t key_with_length_u64x = svadd_u64_x(all_b64x, svreinterpret_u64_u8(state_key_u8x),
                                                  svdupq_n_u64(length, 0));
    // Combine the "sum" and the "AES" blocks
    svuint8_t mixed_u8x = sz_emulate_aesenc_u8x16_sve2_(state_sum_u8x, state_aes_u8x);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    svuint8_t final_mixed_u8x = sz_emulate_aesenc_u8x16_sve2_(
        sz_emulate_aesenc_u8x16_sve2_(mixed_u8x, svreinterpret_u8_u64(key_with_length_u64x)), mixed_u8x);
    // Extract the low 64 bits
    svuint64_t final_mixed_u64x = svreinterpret_u64_u8(final_mixed_u8x);
    return svlasta_u64(all_b64x, final_mixed_u64x); // Extract the first element
}

/*  SVE2 incremental hash functions mostly delegate to NEON for optimal performance; experimental
 *  SVE2 implementations live in `draft/hash.h`.
 *
 *  Vanilla SVE could have helped optimize loads and stores with predicated instructions, but the
 *  @c mov between Z and Q registers isn't free. Moreover, those "bridge" moving instructions are
 *  designed for the bottom 128 bits of the state, and "stores" for 256-bit registers are no faster.
 *
 *  SVE2 comes with optional AES extensions, but they yield no performance improvement at all, even
 *  for wider registers, because of the added cost and complexity of dealing with predicates. */

STRINGZILLA_API sz_status_t sz_hash_state_init_sve2aes(sz_hash_state_t *state, sz_u64_t seed, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_hash_state_init_neonaes_(state, seed);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_hash_state_update_sve2aes(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length,
                                                         sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_hash_state_update_neonaes_(state, text, length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_hash_state_digest_sve2aes(sz_hash_state_t const *state, sz_u64_t *hash,
                                                         sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *hash = sz_hash_state_digest_neonaes_(state);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_hash_sve2aes(sz_cptr_t text, sz_size_t length, sz_u64_t seed, sz_u64_t *hash,
                                            sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *hash = length <= 16 ? sz_hash_upto16_sve2aes_(text, length, seed) : sz_hash_neonaes_(text, length, seed);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_fill_random_sve2aes(sz_ptr_t target, sz_size_t length, sz_u64_t nonce,
                                                   sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_fill_random_neonaes_(target, length, nonce);
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_SVE2AES
#endif // STRINGZILLA_ARCH_ARM64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_SVE2AES_H_
