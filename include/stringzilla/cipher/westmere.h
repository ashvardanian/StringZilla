/**
 *  @brief Westmere (SSE4.2 + AES-NI + PCLMUL) backend for AES-256 encryption in counter and Galois/counter modes.
 *  @file include/stringzilla/cipher/westmere.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/cipher.h
 *
 *  The fourteen AES rounds are written out rather than driven by a loop over the round index. A round
 *  loop leaves the unrolling to the optimizer, which only does it reliably at `-O3` — GCC 15 emitted 35
 *  AES round instructions across the x86 cipher surface at `-O2` against 390 at `-O3`, because a
 *  thirteen-iteration body exceeds the completely-peel budget at the lower level. Since `SZ_API_COMPTIME`
 *  kernels are `static inline` and compile into whichever translation unit consumes them, a looped
 *  kernel makes its own throughput a property of the caller's build flags. See `cipher/icelake.h`, where
 *  the same change was worth 2.1x to 2.9x at `-O2` and nothing at all at `-O3`.
 */
#ifndef STRINGZILLA_CIPHER_WESTMERE_H_
#define STRINGZILLA_CIPHER_WESTMERE_H_

#include "stringzilla/types.h"
#include "stringzilla/cipher/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_WESTMERE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sse4.2,aes,pclmul"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("sse4.2,aes,pclmul")
#endif

#pragma region Key Schedule

/**
 *  @brief Folds a substituted schedule word into the round key four words back, producing the next one.
 *  @param previous The round key four words back, all four of its words in one register.
 *  @param substituted The substituted word, already broadcast across every lane.
 *  @return The next round key.
 *
 *  FIPS 197 defines the schedule one word at a time, each word depending on the one before it. The three
 *  cascading byte shifts replay that dependency inside a single register, so a whole round key falls out
 *  of one exclusive-or chain rather than four.
 */
SZ_HELPER_INLINE __m128i sz_aes256_key_fold_westmere_(__m128i previous, __m128i substituted) {
    previous = _mm_xor_si128(previous, _mm_slli_si128(previous, 4));
    previous = _mm_xor_si128(previous, _mm_slli_si128(previous, 4));
    previous = _mm_xor_si128(previous, _mm_slli_si128(previous, 4));
    return _mm_xor_si128(previous, substituted);
}

SZ_API_COMPTIME void sz_aes256_key_init_westmere(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    __m128i *schedule = (__m128i *)&key->round_keys[0];
    __m128i even_round_key = _mm_lddqu_si128((__m128i const *)secret);
    __m128i odd_round_key = _mm_lddqu_si128((__m128i const *)(secret + 16));

    // An AES-256 schedule alternates two steps: a rotated substitution against a round constant, which
    // the generation assist leaves in the top lane, and a plain substitution, which it leaves in the
    // third. Seven of the first and six of the second follow the two round keys the secret supplies.
    _mm_storeu_si128(schedule + 0, even_round_key);
    _mm_storeu_si128(schedule + 1, odd_round_key);
    even_round_key = sz_aes256_key_fold_westmere_(
        even_round_key, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(odd_round_key, 0x01), 0xFF));
    _mm_storeu_si128(schedule + 2, even_round_key);
    odd_round_key = sz_aes256_key_fold_westmere_(
        odd_round_key, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(even_round_key, 0x00), 0xAA));
    _mm_storeu_si128(schedule + 3, odd_round_key);
    even_round_key = sz_aes256_key_fold_westmere_(
        even_round_key, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(odd_round_key, 0x02), 0xFF));
    _mm_storeu_si128(schedule + 4, even_round_key);
    odd_round_key = sz_aes256_key_fold_westmere_(
        odd_round_key, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(even_round_key, 0x00), 0xAA));
    _mm_storeu_si128(schedule + 5, odd_round_key);
    even_round_key = sz_aes256_key_fold_westmere_(
        even_round_key, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(odd_round_key, 0x04), 0xFF));
    _mm_storeu_si128(schedule + 6, even_round_key);
    odd_round_key = sz_aes256_key_fold_westmere_(
        odd_round_key, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(even_round_key, 0x00), 0xAA));
    _mm_storeu_si128(schedule + 7, odd_round_key);
    even_round_key = sz_aes256_key_fold_westmere_(
        even_round_key, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(odd_round_key, 0x08), 0xFF));
    _mm_storeu_si128(schedule + 8, even_round_key);
    odd_round_key = sz_aes256_key_fold_westmere_(
        odd_round_key, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(even_round_key, 0x00), 0xAA));
    _mm_storeu_si128(schedule + 9, odd_round_key);
    even_round_key = sz_aes256_key_fold_westmere_(
        even_round_key, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(odd_round_key, 0x10), 0xFF));
    _mm_storeu_si128(schedule + 10, even_round_key);
    odd_round_key = sz_aes256_key_fold_westmere_(
        odd_round_key, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(even_round_key, 0x00), 0xAA));
    _mm_storeu_si128(schedule + 11, odd_round_key);
    even_round_key = sz_aes256_key_fold_westmere_(
        even_round_key, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(odd_round_key, 0x20), 0xFF));
    _mm_storeu_si128(schedule + 12, even_round_key);
    odd_round_key = sz_aes256_key_fold_westmere_(
        odd_round_key, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(even_round_key, 0x00), 0xAA));
    _mm_storeu_si128(schedule + 13, odd_round_key);
    even_round_key = sz_aes256_key_fold_westmere_(
        even_round_key, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(odd_round_key, 0x40), 0xFF));
    _mm_storeu_si128(schedule + 14, even_round_key);
}

#pragma endregion // Key Schedule

#pragma region Block Encryption

/**
 *  @brief Reads one round key out of an expanded schedule.
 *  @param key The expanded schedule.
 *  @param round_index Which of the fifteen round keys to read, zero through fourteen.
 *  @return The round key.
 */
SZ_HELPER_INLINE __m128i sz_aes256_round_key_westmere_(sz_aes256_key_t const *key, sz_size_t round_index) {
    return _mm_lddqu_si128((__m128i const *)&key->round_keys[round_index * 4]);
}

/**
 *  @brief Encrypts one block with the expanded schedule.
 *  @param key The expanded schedule.
 *  @param block The plaintext block.
 *  @return The ciphertext block.
 */
SZ_HELPER_INLINE __m128i sz_aes256_block_encrypt_westmere_(sz_aes256_key_t const *key, __m128i block) {
    block = _mm_xor_si128(block, sz_aes256_round_key_westmere_(key, 0));
    block = _mm_aesenc_si128(block, sz_aes256_round_key_westmere_(key, 1));
    block = _mm_aesenc_si128(block, sz_aes256_round_key_westmere_(key, 2));
    block = _mm_aesenc_si128(block, sz_aes256_round_key_westmere_(key, 3));
    block = _mm_aesenc_si128(block, sz_aes256_round_key_westmere_(key, 4));
    block = _mm_aesenc_si128(block, sz_aes256_round_key_westmere_(key, 5));
    block = _mm_aesenc_si128(block, sz_aes256_round_key_westmere_(key, 6));
    block = _mm_aesenc_si128(block, sz_aes256_round_key_westmere_(key, 7));
    block = _mm_aesenc_si128(block, sz_aes256_round_key_westmere_(key, 8));
    block = _mm_aesenc_si128(block, sz_aes256_round_key_westmere_(key, 9));
    block = _mm_aesenc_si128(block, sz_aes256_round_key_westmere_(key, 10));
    block = _mm_aesenc_si128(block, sz_aes256_round_key_westmere_(key, 11));
    block = _mm_aesenc_si128(block, sz_aes256_round_key_westmere_(key, 12));
    block = _mm_aesenc_si128(block, sz_aes256_round_key_westmere_(key, 13));
    return _mm_aesenclast_si128(block, sz_aes256_round_key_westmere_(key, 14));
}

/** @brief Applies one middle round to all four chains, so the four issue back to back. */
SZ_HELPER_INLINE void sz_aes256_blocks_round_westmere_(sz_u512_vec_t *blocks_vec, __m128i round_key) {
    blocks_vec->xmms[0] = _mm_aesenc_si128(blocks_vec->xmms[0], round_key);
    blocks_vec->xmms[1] = _mm_aesenc_si128(blocks_vec->xmms[1], round_key);
    blocks_vec->xmms[2] = _mm_aesenc_si128(blocks_vec->xmms[2], round_key);
    blocks_vec->xmms[3] = _mm_aesenc_si128(blocks_vec->xmms[3], round_key);
}

/**
 *  @brief Encrypts four blocks with the expanded schedule.
 *  @param key The expanded schedule.
 *  @param blocks_vec The four plaintext blocks.
 *  @return The four ciphertext blocks.
 *
 *  One round instruction has several cycles of latency and issues every cycle, so a single chain of
 *  fourteen dependent rounds leaves most of that throughput idle. Four independent chains fill it.
 */
SZ_HELPER_INLINE sz_u512_vec_t sz_aes256_blocks_encrypt_westmere_(sz_aes256_key_t const *key,
                                                                  sz_u512_vec_t blocks_vec) {
    __m128i round_key = sz_aes256_round_key_westmere_(key, 0);

    blocks_vec.xmms[0] = _mm_xor_si128(blocks_vec.xmms[0], round_key);
    blocks_vec.xmms[1] = _mm_xor_si128(blocks_vec.xmms[1], round_key);
    blocks_vec.xmms[2] = _mm_xor_si128(blocks_vec.xmms[2], round_key);
    blocks_vec.xmms[3] = _mm_xor_si128(blocks_vec.xmms[3], round_key);
    // The thirteen middle rounds are written out rather than looped, because a loop leaves the
    // unrolling to the optimizer and the optimizer only does it at `-O3`.
    sz_aes256_blocks_round_westmere_(&blocks_vec, sz_aes256_round_key_westmere_(key, 1));
    sz_aes256_blocks_round_westmere_(&blocks_vec, sz_aes256_round_key_westmere_(key, 2));
    sz_aes256_blocks_round_westmere_(&blocks_vec, sz_aes256_round_key_westmere_(key, 3));
    sz_aes256_blocks_round_westmere_(&blocks_vec, sz_aes256_round_key_westmere_(key, 4));
    sz_aes256_blocks_round_westmere_(&blocks_vec, sz_aes256_round_key_westmere_(key, 5));
    sz_aes256_blocks_round_westmere_(&blocks_vec, sz_aes256_round_key_westmere_(key, 6));
    sz_aes256_blocks_round_westmere_(&blocks_vec, sz_aes256_round_key_westmere_(key, 7));
    sz_aes256_blocks_round_westmere_(&blocks_vec, sz_aes256_round_key_westmere_(key, 8));
    sz_aes256_blocks_round_westmere_(&blocks_vec, sz_aes256_round_key_westmere_(key, 9));
    sz_aes256_blocks_round_westmere_(&blocks_vec, sz_aes256_round_key_westmere_(key, 10));
    sz_aes256_blocks_round_westmere_(&blocks_vec, sz_aes256_round_key_westmere_(key, 11));
    sz_aes256_blocks_round_westmere_(&blocks_vec, sz_aes256_round_key_westmere_(key, 12));
    sz_aes256_blocks_round_westmere_(&blocks_vec, sz_aes256_round_key_westmere_(key, 13));
    round_key = sz_aes256_round_key_westmere_(key, 14);
    blocks_vec.xmms[0] = _mm_aesenclast_si128(blocks_vec.xmms[0], round_key);
    blocks_vec.xmms[1] = _mm_aesenclast_si128(blocks_vec.xmms[1], round_key);
    blocks_vec.xmms[2] = _mm_aesenclast_si128(blocks_vec.xmms[2], round_key);
    blocks_vec.xmms[3] = _mm_aesenclast_si128(blocks_vec.xmms[3], round_key);
    return blocks_vec;
}

#pragma endregion // Block Encryption

#pragma region Counter Mode

/**
 *  @brief Places the twelve nonce bytes in a counter block whose trailing index word is zero.
 *  @param nonce The twelve nonce bytes.
 *  @return The counter block for index zero.
 */
SZ_HELPER_INLINE __m128i sz_aes256_counter_base_westmere_(sz_u8_t const *nonce) {
    sz_u128_vec_t base_vec;
    sz_size_t byte_index;
    base_vec.xmm = _mm_setzero_si128();
    for (byte_index = 0; byte_index != 12; ++byte_index) base_vec.u8s[byte_index] = nonce[byte_index];
    return base_vec.xmm;
}

/**
 *  @brief Completes a counter block with a big-endian block index in its trailing word.
 *  @param base A counter block carrying the nonce, whose trailing word is overwritten.
 *  @param block_index The block index.
 *  @return The counter block for that index.
 */
SZ_HELPER_INLINE __m128i sz_aes256_counter_block_westmere_(__m128i base, sz_u32_t block_index) {
    return _mm_insert_epi32(base, (int)sz_u32_bytes_reverse(block_index), 3);
}

SZ_API_COMPTIME void sz_aes256_ctr_xor_westmere(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                                sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length,
                                                sz_ptr_t output) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    __m128i const counter_base = sz_aes256_counter_base_westmere_(nonce);
    sz_u32_t block_index = (sz_u32_t)(byte_offset / SZ_AES_BLOCK_LENGTH);
    sz_size_t within_block = (sz_size_t)(byte_offset % SZ_AES_BLOCK_LENGTH);
    sz_size_t produced = 0, lane_index;

    // A start that is not block aligned generates its first block whole and discards the leading bytes.
    if (within_block != 0 && length != 0) {
        __m128i const counter = sz_aes256_counter_block_westmere_(counter_base, block_index);
        sz_u128_vec_t keystream_vec;
        keystream_vec.xmm = sz_aes256_block_encrypt_westmere_(key, counter);
        for (; within_block != SZ_AES_BLOCK_LENGTH && produced != length; ++within_block, ++produced)
            output_bytes[produced] = (sz_u8_t)(input_bytes[produced] ^ keystream_vec.u8s[within_block]);
        ++block_index;
    }

    for (; produced + 4 * SZ_AES_BLOCK_LENGTH <= length; produced += 4 * SZ_AES_BLOCK_LENGTH) {
        sz_u512_vec_t keystream_vec;
        keystream_vec.xmms[0] = sz_aes256_counter_block_westmere_(counter_base, block_index + 0);
        keystream_vec.xmms[1] = sz_aes256_counter_block_westmere_(counter_base, block_index + 1);
        keystream_vec.xmms[2] = sz_aes256_counter_block_westmere_(counter_base, block_index + 2);
        keystream_vec.xmms[3] = sz_aes256_counter_block_westmere_(counter_base, block_index + 3);
        block_index += 4;
        keystream_vec = sz_aes256_blocks_encrypt_westmere_(key, keystream_vec);
        for (lane_index = 0; lane_index != 4; ++lane_index) {
            sz_size_t const lane_offset = produced + lane_index * SZ_AES_BLOCK_LENGTH;
            __m128i const original = _mm_lddqu_si128((__m128i const *)(input_bytes + lane_offset));
            _mm_storeu_si128((__m128i *)(output_bytes + lane_offset),
                             _mm_xor_si128(original, keystream_vec.xmms[lane_index]));
        }
    }

    for (; produced + SZ_AES_BLOCK_LENGTH <= length; produced += SZ_AES_BLOCK_LENGTH, ++block_index) {
        __m128i const counter = sz_aes256_counter_block_westmere_(counter_base, block_index);
        __m128i const original = _mm_lddqu_si128((__m128i const *)(input_bytes + produced));
        __m128i const keystream = sz_aes256_block_encrypt_westmere_(key, counter);
        _mm_storeu_si128((__m128i *)(output_bytes + produced), _mm_xor_si128(original, keystream));
    }

    if (produced != length) {
        __m128i const counter = sz_aes256_counter_block_westmere_(counter_base, block_index);
        sz_u128_vec_t keystream_vec;
        keystream_vec.xmm = sz_aes256_block_encrypt_westmere_(key, counter);
        for (within_block = 0; produced != length; ++within_block, ++produced)
            output_bytes[produced] = (sz_u8_t)(input_bytes[produced] ^ keystream_vec.u8s[within_block]);
    }
}

#pragma endregion // Counter Mode

#pragma region Galois Hashing

/** @brief The byte order that moves a hash block between the tag's byte order and the multiplier's. */
SZ_HELPER_INLINE sz_u8_t const *sz_ghash_byte_order_westmere_(void) {
    static sz_align_(64) sz_u8_t const order[16] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    return &order[0];
}

/**
 *  @brief Reverses a hash block between the tag's byte order and the multiplier's.
 *  @param block The block in either order.
 *  @return The same block in the other order.
 *
 *  The tag is defined over blocks whose leading bit is the field element's lowest coefficient, which is
 *  the opposite of how a carry-less multiply reads its operands. Reversing the bytes puts the polynomial
 *  the right way round, and the multiplier's leftward shift makes up the remaining single-bit offset.
 */
SZ_HELPER_INLINE __m128i sz_ghash_reflect_westmere_(__m128i block) {
    return _mm_shuffle_epi8(block, _mm_load_si128((__m128i const *)sz_ghash_byte_order_westmere_()));
}

/**
 *  @brief Loads a hash block from the tag's byte order into the multiplier's.
 *  @param block The sixteen block bytes.
 *  @return The reflected block.
 */
SZ_HELPER_INLINE __m128i sz_ghash_load_westmere_(sz_u8_t const *block) {
    return sz_ghash_reflect_westmere_(_mm_lddqu_si128((__m128i const *)block));
}

/**
 *  @brief Stores a hash block back in the tag's byte order.
 *  @param value The reflected block.
 *  @param block Receives the sixteen bytes.
 */
SZ_HELPER_INLINE void sz_ghash_store_westmere_(__m128i value, sz_u8_t *block) {
    _mm_storeu_si128((__m128i *)block, sz_ghash_reflect_westmere_(value));
}

/**
 *  @brief Accumulates the 256-bit carry-less product of two reflected blocks into a running triple.
 *  @param multiplicand One reflected operand.
 *  @param multiplier The other reflected operand.
 *  @param low Accumulates the product of the two low halves.
 *  @param middle Accumulates both cross products.
 *  @param high Accumulates the product of the two high halves.
 *
 *  Splitting the schoolbook product from its reduction is what lets several blocks share one reduction:
 *  the field is linear, so the partial products of a whole group may be summed before folding once.
 */
SZ_HELPER_INLINE void sz_ghash_accumulate_westmere_(__m128i multiplicand, __m128i multiplier, __m128i *low,
                                                    __m128i *middle, __m128i *high) {
    *low = _mm_xor_si128(*low, _mm_clmulepi64_si128(multiplicand, multiplier, 0x00));
    *middle = _mm_xor_si128(*middle, _mm_clmulepi64_si128(multiplicand, multiplier, 0x10));
    *middle = _mm_xor_si128(*middle, _mm_clmulepi64_si128(multiplicand, multiplier, 0x01));
    *high = _mm_xor_si128(*high, _mm_clmulepi64_si128(multiplicand, multiplier, 0x11));
}

/**
 *  @brief Folds an accumulated triple into one reflected block.
 *  @param product_low The accumulated low halves.
 *  @param product_middle The accumulated cross products.
 *  @param product_high The accumulated high halves.
 *  @return The reduced product, reflected.
 *
 *  The cross products straddle the halves, so they are split and merged first. The reflected field puts
 *  the whole 256-bit value one bit below where the reduction expects it, hence the leftward shift, and
 *  what remains is the standard fold modulo `x^128 + x^7 + x^2 + x + 1`, done in two passes because the
 *  three low taps are close enough together to share a shift chain.
 */
SZ_HELPER_INLINE __m128i sz_ghash_reduce_westmere_(__m128i product_low, __m128i product_middle, __m128i product_high) {
    __m128i low_carries, high_carries, crossing_carry, folded, folded_spill_bits, reduced_taps;

    product_low = _mm_xor_si128(product_low, _mm_slli_si128(product_middle, 8));
    product_high = _mm_xor_si128(product_high, _mm_srli_si128(product_middle, 8));

    low_carries = _mm_srli_epi32(product_low, 31);
    high_carries = _mm_srli_epi32(product_high, 31);
    crossing_carry = _mm_srli_si128(low_carries, 12);
    product_low = _mm_or_si128(_mm_slli_epi32(product_low, 1), _mm_slli_si128(low_carries, 4));
    product_high = _mm_or_si128(_mm_slli_epi32(product_high, 1), _mm_slli_si128(high_carries, 4));
    product_high = _mm_or_si128(product_high, crossing_carry);

    folded = _mm_slli_epi32(product_low, 31);
    folded = _mm_xor_si128(folded, _mm_slli_epi32(product_low, 30));
    folded = _mm_xor_si128(folded, _mm_slli_epi32(product_low, 25));
    folded_spill_bits = _mm_srli_si128(folded, 4);
    product_low = _mm_xor_si128(product_low, _mm_slli_si128(folded, 12));

    reduced_taps = _mm_srli_epi32(product_low, 1);
    reduced_taps = _mm_xor_si128(reduced_taps, _mm_srli_epi32(product_low, 2));
    reduced_taps = _mm_xor_si128(reduced_taps, _mm_srli_epi32(product_low, 7));
    reduced_taps = _mm_xor_si128(reduced_taps, folded_spill_bits);
    return _mm_xor_si128(product_high, _mm_xor_si128(product_low, reduced_taps));
}

/**
 *  @brief Multiplies two reflected blocks in the Galois field the tag is built over.
 *  @param accumulator The running hash, reflected.
 *  @param subkey One of the reflected subkey powers.
 *  @return The reduced product, reflected.
 */
SZ_HELPER_INLINE __m128i sz_ghash_multiply_westmere_(__m128i accumulator, __m128i subkey) {
    __m128i product_low = _mm_setzero_si128(), product_middle = _mm_setzero_si128(), product_high = _mm_setzero_si128();
    sz_ghash_accumulate_westmere_(accumulator, subkey, &product_low, &product_middle, &product_high);
    return sz_ghash_reduce_westmere_(product_low, product_middle, product_high);
}

/**
 *  @brief Absorbs one reflected block into the running hash.
 *  @param accumulator The running hash, reflected.
 *  @param block The reflected block to absorb.
 *  @param subkey The reflected hash subkey.
 *  @return The updated running hash, reflected.
 */
SZ_HELPER_INLINE __m128i sz_ghash_absorb_westmere_(__m128i accumulator, __m128i block, __m128i subkey) {
    return sz_ghash_multiply_westmere_(_mm_xor_si128(accumulator, block), subkey);
}

SZ_API_COMPTIME void sz_aes256_gcm_key_init_westmere(sz_aes256_gcm_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    __m128i subkey, power;
    sz_size_t power_index;

    sz_aes256_key_init_westmere(&key->block, secret);
    subkey = sz_ghash_reflect_westmere_(sz_aes256_block_encrypt_westmere_(&key->block, _mm_setzero_si128()));
    power = subkey;
    sz_ghash_store_westmere_(power, &key->powers[0]);
    for (power_index = 1; power_index != 8; ++power_index) {
        power = sz_ghash_multiply_westmere_(power, subkey);
        sz_ghash_store_westmere_(power, &key->powers[power_index * SZ_AES_BLOCK_LENGTH]);
    }
}

/**
 *  @brief Loads the first four subkey powers, descending, so power `j` pairs with block `j` of a group.
 *  @param powers The eight subkey powers in the tag's byte order, ascending.
 *  @return The reflected powers `H^4`, `H^3`, `H^2`, `H^1`.
 */
SZ_HELPER_INLINE sz_u512_vec_t sz_ghash_descending_powers_westmere_(sz_u8_t const *powers) {
    sz_u512_vec_t powers_vec;
    powers_vec.xmms[0] = sz_ghash_load_westmere_(powers + 3 * SZ_AES_BLOCK_LENGTH);
    powers_vec.xmms[1] = sz_ghash_load_westmere_(powers + 2 * SZ_AES_BLOCK_LENGTH);
    powers_vec.xmms[2] = sz_ghash_load_westmere_(powers + 1 * SZ_AES_BLOCK_LENGTH);
    powers_vec.xmms[3] = sz_ghash_load_westmere_(powers + 0 * SZ_AES_BLOCK_LENGTH);
    return powers_vec;
}

/**
 *  @brief Absorbs four reflected blocks into the running hash with a single field reduction.
 *  @param accumulator The running hash, reflected.
 *  @param blocks_vec The four reflected blocks, in the order they arrived.
 *  @param powers_vec The reflected powers `H^4` through `H^1`.
 *  @return The updated running hash, reflected.
 *
 *  Four absorbed blocks expand to `(Y ^ X1) H^4 ^ X2 H^3 ^ X3 H^2 ^ X4 H`, which needs four multiplies
 *  either way but only one reduction instead of four.
 */
SZ_HELPER_INLINE __m128i sz_ghash_absorb_four_westmere_(__m128i accumulator, sz_u512_vec_t blocks_vec,
                                                        sz_u512_vec_t powers_vec) {
    __m128i product_low = _mm_setzero_si128(), product_middle = _mm_setzero_si128(), product_high = _mm_setzero_si128();
    sz_ghash_accumulate_westmere_(_mm_xor_si128(accumulator, blocks_vec.xmms[0]), powers_vec.xmms[0], &product_low,
                                  &product_middle, &product_high);
    sz_ghash_accumulate_westmere_(blocks_vec.xmms[1], powers_vec.xmms[1], &product_low, &product_middle, &product_high);
    sz_ghash_accumulate_westmere_(blocks_vec.xmms[2], powers_vec.xmms[2], &product_low, &product_middle, &product_high);
    sz_ghash_accumulate_westmere_(blocks_vec.xmms[3], powers_vec.xmms[3], &product_low, &product_middle, &product_high);
    return sz_ghash_reduce_westmere_(product_low, product_middle, product_high);
}

#pragma endregion // Galois Hashing

#pragma region Streaming Interface

/**
 *  @brief Overwrites a finished state so the key schedule it embeds does not outlive the call.
 *
 *  The size is known at compile time, so both bounds are constants and the fill unrolls into twenty-nine
 *  whole-register stores and an eight-byte tail. Writes are ordinary stores followed by one barrier:
 *  routing them through a `volatile` view instead would forbid vectorization and cost 472 single-byte
 *  stores on every one-shot call.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_state_scrub_westmere_(sz_aes256_gcm_state_t *state) {
    sz_u8_t *const bytes = (sz_u8_t *)state;
    __m128i const zeros = _mm_setzero_si128();
    sz_size_t byte_index = 0;
    for (; byte_index + 16 <= sizeof(*state); byte_index += 16)
        _mm_storeu_si128((__m128i *)(bytes + byte_index), zeros);
    for (; byte_index != sizeof(*state); ++byte_index) bytes[byte_index] = 0;
    sz_keep_alive_(state);
}

/** @brief Prepares the payload both directions share: counter block, tag mask and empty carries. */
SZ_HELPER_INLINE void sz_aes256_gcm_begin_westmere_(sz_aes256_gcm_state_t *state, sz_aes256_gcm_key_t const *key,
                                                    sz_u8_t const nonce[sz_at_least_(12)]) {
    __m128i initial;

    state->key = *key;

    // With a twelve-byte nonce the initial counter block is the nonce followed by a one, and the value
    // encrypted under it masks the finished hash. Data blocks start one past it.
    initial = sz_aes256_counter_block_westmere_(sz_aes256_counter_base_westmere_(nonce), 1u);
    _mm_storeu_si128((__m128i *)state->tag_mask, sz_aes256_block_encrypt_westmere_(&state->key.block, initial));
    _mm_storeu_si128((__m128i *)state->counter, initial);
    _mm_storeu_si128((__m128i *)state->accumulator, _mm_setzero_si128());
    _mm_storeu_si128((__m128i *)state->partial, _mm_setzero_si128());
    _mm_storeu_si128((__m128i *)state->keystream, _mm_setzero_si128());

    state->associated_length = 0;
    state->text_length = 0;
    state->buffered = 0;
    state->keystream_used = SZ_AES_BLOCK_LENGTH; // ? Forces the first message byte to derive a fresh block
}

/** @brief Absorbs associated data into the payload both directions share. */
SZ_HELPER_INLINE void sz_aes256_gcm_associate_westmere_(sz_aes256_gcm_state_t *state, sz_cptr_t text,
                                                        sz_size_t length) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    __m128i const subkey = sz_ghash_load_westmere_(state->key.powers);
    sz_u512_vec_t const powers_vec = sz_ghash_descending_powers_westmere_(state->key.powers);
    __m128i accumulator;
    sz_size_t consumed = 0, byte_index;

    if (length == 0) return;
    accumulator = sz_ghash_load_westmere_(state->accumulator);
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
            accumulator = sz_ghash_absorb_westmere_(accumulator, sz_ghash_load_westmere_(state->partial), subkey);
            state->buffered = 0;
        }
    }

    for (; consumed + 4 * SZ_AES_BLOCK_LENGTH <= length; consumed += 4 * SZ_AES_BLOCK_LENGTH) {
        sz_u512_vec_t blocks_vec;
        blocks_vec.xmms[0] = sz_ghash_load_westmere_(input_bytes + consumed + 0 * SZ_AES_BLOCK_LENGTH);
        blocks_vec.xmms[1] = sz_ghash_load_westmere_(input_bytes + consumed + 1 * SZ_AES_BLOCK_LENGTH);
        blocks_vec.xmms[2] = sz_ghash_load_westmere_(input_bytes + consumed + 2 * SZ_AES_BLOCK_LENGTH);
        blocks_vec.xmms[3] = sz_ghash_load_westmere_(input_bytes + consumed + 3 * SZ_AES_BLOCK_LENGTH);
        accumulator = sz_ghash_absorb_four_westmere_(accumulator, blocks_vec, powers_vec);
    }
    for (; consumed + SZ_AES_BLOCK_LENGTH <= length; consumed += SZ_AES_BLOCK_LENGTH)
        accumulator = sz_ghash_absorb_westmere_(accumulator, sz_ghash_load_westmere_(input_bytes + consumed), subkey);
    for (; consumed != length; ++consumed) state->partial[state->buffered++] = input_bytes[consumed];

    sz_ghash_store_westmere_(accumulator, state->accumulator);
}

/**
 *  @brief Spends what is left of the keystream block the state carries, one byte at a time.
 *  @param state The state, whose two sixteen-byte counters advance together here.
 *  @param input The bytes to transform.
 *  @param output Receives the transformed bytes.
 *  @param count Bytes to consume, never more than the keystream block has left.
 *  @param accumulator The running hash, reflected.
 *  @param subkey The reflected hash subkey.
 *  @param direction Which buffer the hash absorbs.
 *  @return The updated running hash, reflected.
 *
 *  Through the message the keystream offset and the hash offset are the same number, because every byte
 *  spends one of each, so a chunk that ends mid block leaves both mid block and this resumes both.
 */
SZ_HELPER_INLINE __m128i sz_aes256_gcm_spend_westmere_(sz_aes256_gcm_state_t *state, sz_u8_t const *input,
                                                       sz_u8_t *output, sz_size_t count, __m128i accumulator,
                                                       __m128i subkey, sz_aes256_gcm_direction_t direction) {
    sz_size_t byte_index;

    // The hash always eats ciphertext, which is the output when encrypting and the input when
    // decrypting, and the byte is captured before the store because a caller may pass one pointer.
    for (byte_index = 0; byte_index != count; ++byte_index) {
        sz_u8_t const plaintext_byte = input[byte_index];
        sz_u8_t const transformed = (sz_u8_t)(plaintext_byte ^ state->keystream[state->keystream_used]);
        sz_u8_t const ciphertext = direction == sz_aes256_gcm_decrypting_k ? plaintext_byte : transformed;
        output[byte_index] = transformed;
        state->partial[state->buffered] = ciphertext;
        ++state->buffered;
        ++state->keystream_used;
        if (state->buffered == SZ_AES_BLOCK_LENGTH) {
            accumulator = sz_ghash_absorb_westmere_(accumulator, sz_ghash_load_westmere_(state->partial), subkey);
            state->buffered = 0;
        }
    }
    return accumulator;
}

/**
 *  @brief Transforms a chunk and absorbs its ciphertext, whichever side of the call that is.
 *  @param state The state.
 *  @param text The chunk to transform.
 *  @param length Bytes in the chunk.
 *  @param output Receives the transformed bytes.
 *  @param direction Which buffer the hash absorbs.
 *
 *  Three passes, because two sixteen-byte rhythms run underneath a caller's arbitrary chunk sizes and
 *  neither may restart at a chunk boundary: whatever the previous chunk left of its keystream block, then
 *  whole blocks four at a time, then a trailing block that the next chunk will resume.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_transform_westmere_(sz_aes256_gcm_state_t *state, sz_cptr_t text, sz_size_t length,
                                                        sz_ptr_t output, sz_aes256_gcm_direction_t direction) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    __m128i const subkey = sz_ghash_load_westmere_(state->key.powers);
    __m128i const cipher_mask = direction == sz_aes256_gcm_decrypting_k ? _mm_set1_epi8(-1) : _mm_setzero_si128();
    sz_u512_vec_t const powers_vec = sz_ghash_descending_powers_westmere_(state->key.powers);
    sz_u128_vec_t counter_vec;
    __m128i accumulator;
    sz_u32_t block_index;
    sz_size_t produced = 0, byte_index, lane_index;

    if (length == 0) return;
    accumulator = sz_ghash_load_westmere_(state->accumulator);

    // Associated data ends the moment the first message byte arrives, and its tail needs padding.
    if (state->text_length == 0 && state->buffered != 0) {
        sz_u128_vec_t padded_vec;
        padded_vec.xmm = _mm_lddqu_si128((__m128i const *)state->partial);
        for (byte_index = state->buffered; byte_index != SZ_AES_BLOCK_LENGTH; ++byte_index)
            padded_vec.u8s[byte_index] = 0;
        accumulator = sz_ghash_absorb_westmere_(accumulator, sz_ghash_reflect_westmere_(padded_vec.xmm), subkey);
        state->buffered = 0;
    }

    counter_vec.xmm = _mm_lddqu_si128((__m128i const *)state->counter);
    block_index = sz_u32_bytes_reverse(counter_vec.u32s[3]);

    if (state->keystream_used != SZ_AES_BLOCK_LENGTH) {
        sz_size_t const available_bytes = SZ_AES_BLOCK_LENGTH - state->keystream_used;
        sz_size_t const taken_bytes = length < available_bytes ? length : available_bytes;
        accumulator = sz_aes256_gcm_spend_westmere_(state, input_bytes, output_bytes, taken_bytes, accumulator, subkey,
                                                    direction);
        produced = taken_bytes;
    }

    for (; produced + 4 * SZ_AES_BLOCK_LENGTH <= length; produced += 4 * SZ_AES_BLOCK_LENGTH) {
        sz_u512_vec_t keystream_vec, ciphertext_vec;
        keystream_vec.xmms[0] = sz_aes256_counter_block_westmere_(counter_vec.xmm, block_index + 1);
        keystream_vec.xmms[1] = sz_aes256_counter_block_westmere_(counter_vec.xmm, block_index + 2);
        keystream_vec.xmms[2] = sz_aes256_counter_block_westmere_(counter_vec.xmm, block_index + 3);
        keystream_vec.xmms[3] = sz_aes256_counter_block_westmere_(counter_vec.xmm, block_index + 4);
        block_index += 4;
        keystream_vec = sz_aes256_blocks_encrypt_westmere_(&state->key.block, keystream_vec);
        for (lane_index = 0; lane_index != 4; ++lane_index) {
            sz_size_t const lane_offset = produced + lane_index * SZ_AES_BLOCK_LENGTH;
            __m128i const original = _mm_lddqu_si128((__m128i const *)(input_bytes + lane_offset));
            __m128i const transformed = _mm_xor_si128(original, keystream_vec.xmms[lane_index]);
            __m128i const ciphertext = _mm_blendv_epi8(transformed, original, cipher_mask);
            ciphertext_vec.xmms[lane_index] = sz_ghash_reflect_westmere_(ciphertext);
            _mm_storeu_si128((__m128i *)(output_bytes + lane_offset), transformed);
        }
        accumulator = sz_ghash_absorb_four_westmere_(accumulator, ciphertext_vec, powers_vec);
    }

    for (; produced + SZ_AES_BLOCK_LENGTH <= length; produced += SZ_AES_BLOCK_LENGTH) {
        __m128i const original = _mm_lddqu_si128((__m128i const *)(input_bytes + produced));
        __m128i counter, keystream, transformed, ciphertext;
        block_index += 1;
        counter = sz_aes256_counter_block_westmere_(counter_vec.xmm, block_index);
        keystream = sz_aes256_block_encrypt_westmere_(&state->key.block, counter);
        transformed = _mm_xor_si128(original, keystream);
        ciphertext = _mm_blendv_epi8(transformed, original, cipher_mask);
        _mm_storeu_si128((__m128i *)(output_bytes + produced), transformed);
        accumulator = sz_ghash_absorb_westmere_(accumulator, sz_ghash_reflect_westmere_(ciphertext), subkey);
    }

    if (produced != length) {
        __m128i counter, keystream;
        block_index += 1;
        counter = sz_aes256_counter_block_westmere_(counter_vec.xmm, block_index);
        keystream = sz_aes256_block_encrypt_westmere_(&state->key.block, counter);
        _mm_storeu_si128((__m128i *)state->keystream, keystream);
        state->keystream_used = 0;
        accumulator = sz_aes256_gcm_spend_westmere_(state, input_bytes + produced, output_bytes + produced,
                                                    length - produced, accumulator, subkey, direction);
    }

    state->text_length += length;
    sz_ghash_store_westmere_(accumulator, state->accumulator);
    _mm_storeu_si128((__m128i *)state->counter, sz_aes256_counter_block_westmere_(counter_vec.xmm, block_index));
}

/** @brief Folds the pending block and the length block into the hash, then unmasks it into the tag. */
SZ_HELPER_INLINE void sz_aes256_gcm_digest_westmere_(sz_aes256_gcm_state_t const *state,
                                                     sz_u8_t tag[sz_at_least_(16)]) {
    __m128i const subkey = sz_ghash_load_westmere_(state->key.powers);
    __m128i accumulator = sz_ghash_load_westmere_(state->accumulator);
    sz_u128_vec_t lengths_vec;
    sz_size_t byte_index;

    // Whatever is still pending gets padded here: an associated-data tail when the message was empty,
    // or the message's own trailing ciphertext bytes otherwise.
    if (state->buffered != 0) {
        sz_u128_vec_t padded_vec;
        padded_vec.xmm = _mm_lddqu_si128((__m128i const *)state->partial);
        for (byte_index = state->buffered; byte_index != SZ_AES_BLOCK_LENGTH; ++byte_index)
            padded_vec.u8s[byte_index] = 0;
        accumulator = sz_ghash_absorb_westmere_(accumulator, sz_ghash_reflect_westmere_(padded_vec.xmm), subkey);
    }

    sz_aes256_gcm_lengths_serial_(&lengths_vec, state->associated_length, state->text_length);
    accumulator = sz_ghash_absorb_westmere_(accumulator, sz_ghash_reflect_westmere_(lengths_vec.xmm), subkey);

    _mm_storeu_si128((__m128i *)tag, _mm_xor_si128(sz_ghash_reflect_westmere_(accumulator),
                                                   _mm_lddqu_si128((__m128i const *)state->tag_mask)));
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_init_westmere(sz_aes256_gcm_encryptor_t *encryptor,
                                                           sz_aes256_gcm_key_t const *key,
                                                           sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_aes256_gcm_begin_westmere_(&encryptor->state, key, nonce);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_associate_westmere(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                sz_size_t length) {
    sz_aes256_gcm_associate_westmere_(&encryptor->state, text, length);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_update_westmere(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                             sz_size_t length, sz_ptr_t output) {
    sz_aes256_gcm_transform_westmere_(&encryptor->state, text, length, output, sz_aes256_gcm_encrypting_k);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_digest_westmere(sz_aes256_gcm_encryptor_t const *encryptor,
                                                             sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_digest_westmere_(&encryptor->state, tag);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_init_westmere(sz_aes256_gcm_decryptor_t *decryptor,
                                                           sz_aes256_gcm_key_t const *key,
                                                           sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_aes256_gcm_begin_westmere_(&decryptor->state, key, nonce);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_associate_westmere(sz_aes256_gcm_decryptor_t *decryptor, sz_cptr_t text,
                                                                sz_size_t length) {
    sz_aes256_gcm_associate_westmere_(&decryptor->state, text, length);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_update_unverified_westmere(sz_aes256_gcm_decryptor_t *decryptor,
                                                                        sz_cptr_t text, sz_size_t length,
                                                                        sz_ptr_t output) {
    sz_aes256_gcm_transform_westmere_(&decryptor->state, text, length, output, sz_aes256_gcm_decrypting_k);
}

SZ_API_COMPTIME sz_status_t sz_aes256_gcm_decryptor_verify_westmere(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                    sz_u8_t const tag[sz_at_least_(16)]) {
    sz_u128_vec_t expected_vec;
    sz_aes256_gcm_digest_westmere_(&decryptor->state, expected_vec.u8s);
    return sz_aes256_tag_equal_serial_(expected_vec.u8s, tag) == sz_true_k ? sz_success_k : sz_authentication_failed_k;
}

#pragma endregion // Streaming Interface

#pragma region One Shot Interface

SZ_API_COMPTIME void sz_aes256_gcm_encrypt_westmere(sz_aes256_gcm_key_t const *key,
                                                    sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                    sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                    sz_ptr_t output, sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_encryptor_t encryptor;
    sz_aes256_gcm_encryptor_init_westmere(&encryptor, key, nonce);
    if (associated_length) sz_aes256_gcm_encryptor_associate_westmere(&encryptor, associated, associated_length);
    sz_aes256_gcm_encryptor_update_westmere(&encryptor, text, length, output);
    sz_aes256_gcm_encryptor_digest_westmere(&encryptor, tag);
    sz_aes256_gcm_state_scrub_westmere_(&encryptor.state);
}

SZ_API_COMPTIME sz_status_t sz_aes256_gcm_decrypt_westmere(sz_aes256_gcm_key_t const *key,
                                                           sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                           sz_size_t associated_length, sz_cptr_t text,
                                                           sz_size_t length, sz_ptr_t output,
                                                           sz_u8_t const tag[sz_at_least_(16)]) {
    sz_aes256_gcm_decryptor_t decryptor;
    sz_status_t verdict;
    sz_aes256_gcm_decryptor_init_westmere(&decryptor, key, nonce);
    if (associated_length) sz_aes256_gcm_decryptor_associate_westmere(&decryptor, associated, associated_length);
    sz_aes256_gcm_decryptor_update_unverified_westmere(&decryptor, text, length, output);
    verdict = sz_aes256_gcm_decryptor_verify_westmere(&decryptor, tag);
    sz_aes256_gcm_state_scrub_westmere_(&decryptor.state);

    // A caller who drops the status must still be unable to act on forged plaintext.
    if (verdict != sz_success_k) sz_fill(output, length, 0);
    return verdict;
}

#pragma endregion // One Shot Interface

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_WESTMERE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_CIPHER_WESTMERE_H_
