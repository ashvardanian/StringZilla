/**
 *  @brief SVE2 + AES backend for string hashing and checksums.
 *  @file include/stringzilla/hash/sve2aes.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_SVE2AES_H_
#define STRINGZILLA_HASH_SVE2AES_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/hash/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SVE2AES
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve+sve2+sve2-aes"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve+sve2+sve2-aes")
#endif

/**
 *  @brief Emulates the Intel's AES-NI `AESENC` instruction with Arm SVE2.
 *  @see "Emulating x86 AES Intrinsics on ARMv8-A" by Michael Brase:
 *       https://blog.michaelbrase.com/2018/05/08/emulating-x86-aes-intrinsics-on-armv8-a/
 */
SZ_INTERNAL svuint8_t sz_emulate_aesenc_u8x16_sve2_(svuint8_t state_u8x, svuint8_t round_key_u8x) {
    return sveor_u8_x(svptrue_b8(), svaesmc_u8(svaese_u8(state_u8x, svdup_n_u8(0))), round_key_u8x);
}

/** @brief A variant of `sz_hash_sve2aes` for strings up to 16 bytes long - smallest SVE register size. */
SZ_INTERNAL sz_u64_t sz_hash_sve2_upto16_(sz_cptr_t text, sz_size_t length, sz_u64_t seed) {
    svuint8_t state_aes_u8x, state_sum_u8x, state_key_u8x;

    // To load and store the seed, we don't even need a `svwhilelt_b64(0, 2)`.
    state_key_u8x = svreinterpret_u8_u64(svdup_n_u64(seed));
    svbool_t const all64 = svptrue_b64();
    svbool_t const all8 = svptrue_b8();

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    svuint64_t pi0_u64x = svld1_u64(all64, pi);
    svuint64_t pi1_u64x = svld1_u64(all64, pi + 8);
    state_aes_u8x = sveor_u8_x(all8, state_key_u8x, svreinterpret_u8_u64(pi0_u64x));
    state_sum_u8x = sveor_u8_x(all8, state_key_u8x, svreinterpret_u8_u64(pi1_u64x));

    // We will only use the first 128 bits of the shuffle mask
    svuint8_t const order_u8x = svld1_u8(all8, sz_hash_u8x16x4_shuffle_());

    // This is our best case for SVE2 dominance over NEON - we can load the data in one go with a predicate.
    svuint8_t block_u8x = svld1_u8(svwhilelt_b8((sz_u64_t)0, (sz_u64_t)length), (sz_u8_t const *)text);
    // One round of hashing logic
    state_aes_u8x = sz_emulate_aesenc_u8x16_sve2_(state_aes_u8x, block_u8x);
    svuint8_t sum_shuffled_u8x = svtbl_u8(state_sum_u8x, order_u8x);
    state_sum_u8x = svreinterpret_u8_u64(
        svadd_u64_x(all64, svreinterpret_u64_u8(sum_shuffled_u8x), svreinterpret_u64_u8(block_u8x)));

    // Now mix, folding the length into the key
    svuint64_t key_with_length_u64x = svadd_u64_x(all64, svreinterpret_u64_u8(state_key_u8x), svdupq_n_u64(length, 0));
    // Combine the "sum" and the "AES" blocks
    svuint8_t mixed_u8x = sz_emulate_aesenc_u8x16_sve2_(state_sum_u8x, state_aes_u8x);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    svuint8_t final_mixed_u8x = sz_emulate_aesenc_u8x16_sve2_(
        sz_emulate_aesenc_u8x16_sve2_(mixed_u8x, svreinterpret_u8_u64(key_with_length_u64x)), mixed_u8x);
    // Extract the low 64 bits
    svuint64_t final_mixed_u64x = svreinterpret_u64_u8(final_mixed_u8x);
    return svlasta_u64(all64, final_mixed_u64x); // Extract the first element
}

/*  @brief  SVE2 increment hash functions currently mostly delegate to NEON for optimal performance.
 *  @see draft/hash.h for experimental SVE2 implementations.
 *
 *  Vanilla SVE could have helped optimized loads & stores with predicated instructions,
 *  but the `mov` between Z and Q registers isn't free. Moreover, those "bridge" moving
 *  instructions are designed for the bottom 128 bits of the state. Using "stores" for
 *  larger 256-bit registers isn't fast either.
 *
 *  SVE2 comes with optional AES extensions, but they don't yield absolutely any performance
 *  improvements even for wider registers due to the added cost and complexity of dealing with
 *  predicates.
 */

SZ_PUBLIC void sz_hash_state_init_sve2aes(sz_hash_state_t *state, sz_u64_t seed) { //
    sz_hash_state_init_neonaes(state, seed);
}

SZ_PUBLIC void sz_hash_state_update_sve2aes(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_hash_state_update_neonaes(state, text, length);
}

SZ_PUBLIC sz_u64_t sz_hash_state_digest_sve2aes(sz_hash_state_t const *state) { //
    return sz_hash_state_digest_neonaes(state);
}

SZ_PUBLIC sz_u64_t sz_hash_sve2aes(sz_cptr_t text, sz_size_t length, sz_u64_t seed) {
    if (length <= 16) return sz_hash_sve2_upto16_(text, length, seed);
    return sz_hash_neonaes(text, length, seed);
}

SZ_PUBLIC void sz_fill_random_sve2aes(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_fill_random_neonaes(text, length, nonce);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE2AES

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_SVE2AES_H_
