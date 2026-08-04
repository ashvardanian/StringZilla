/**
 *  @brief IBM Power VSX backend for AES-256 in counter and Galois/counter modes.
 *  @file include/stringzilla/cipher/powervsx.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/cipher.h
 *
 *  The fourteen rounds are written out rather than looped, for the same reason the eight lanes are and
 *  the one `cipher/icelake.h` spells out.
 */
#ifndef STRINGZILLA_CIPHER_POWERVSX_H_
#define STRINGZILLA_CIPHER_POWERVSX_H_

#include "stringzilla/types.h"
#include "stringzilla/cipher/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

/*  Power is the only target here with both a native round instruction and a native carry-less multiply,
 *  and both of them read a register in big-endian order no matter which endianness the translation unit
 *  is built for. That is what the `_be` in `vec_cipher_be` and `vec_pmsum_be` means: the instruction is
 *  not endian-adjusted the way `vec_perm` and `vec_splat` are, so element zero of a little-endian vector
 *  is the @b last byte the round function sees.
 *
 *  Every sixteen-byte value in this file therefore lives in one single representation, called big-endian
 *  register order below: block byte zero occupies the most significant register byte. `vec_xl_be` and
 *  `vec_xst_be` move between memory and that order on both endiannesses in one instruction each, so the
 *  choice costs nothing, and the round keys, counter blocks, keystream, hash blocks and subkey powers all
 *  share it. Two consequences are worth spelling out.
 *
 *  The Galois hash needs @b no byte reflection here. A carry-less multiplier wants the field element's
 *  lowest coefficient in the most significant bit, which is exactly where the tag's byte order already
 *  puts it once the block is in big-endian register order; the byte shuffle that the x86 backends spend
 *  on every block is simply absent. The multiplier's remaining single-bit offset is still made up by the
 *  leftward shift inside the reduction.
 *
 *  Only the round-key array is genuinely endian-dependent, because `sz_aes256_key_t` stores words as
 *  `sz_u32_t` and the byte image of a word differs between the two targets. That one conversion is the
 *  sole `SZ_IS_BIG_ENDIAN_` branch below. Everything else is expressed through `vec_sld`, which is a raw
 *  big-endian `vsldoi`, or through lane-wise operations whose meaning does not depend on element
 *  numbering, so the two branches cannot drift apart.
 */

#pragma region Byte Order

/**
 *  @brief Loads sixteen bytes into big-endian register order.
 *  @param bytes The sixteen bytes, in the order the cipher defines them.
 *  @return The block, with byte zero in the most significant register byte.
 */
SZ_HELPER_INLINE __vector unsigned char sz_aes256_block_load_powervsx_(sz_u8_t const *bytes) {
    return vec_xl_be(0, (unsigned char const *)bytes);
}

/**
 *  @brief Stores a block held in big-endian register order back to sixteen bytes.
 *  @param block_u8x16 The block.
 *  @param bytes Receives the sixteen bytes in the order the cipher defines them.
 */
SZ_HELPER_INLINE void sz_aes256_block_store_powervsx_(__vector unsigned char block_u8x16, sz_u8_t *bytes) {
    vec_xst_be(block_u8x16, 0, (unsigned char *)bytes);
}

/** @brief An all-zero block, the second operand of every `vec_sld` used as a shift. */
SZ_HELPER_INLINE __vector unsigned char sz_aes256_zero_powervsx_(void) { return vec_splats((unsigned char)0); }

#pragma endregion // Byte Order

#pragma region Key Schedule

/**
 *  @brief Reads one round key out of an expanded schedule into big-endian register order.
 *  @param key The expanded schedule.
 *  @param round_index Which of the fifteen round keys to read, zero through fourteen.
 *  @return The round key.
 *
 *  A schedule word packs its first byte into the least significant position, so on a little-endian target
 *  the word array's byte image already is the round key's byte sequence. On a big-endian target every
 *  word's four bytes are stored the other way round, and reversing inside each word repairs it.
 */
SZ_HELPER_INLINE __vector unsigned char sz_aes256_round_key_powervsx_(sz_aes256_key_t const *key,
                                                                      sz_size_t round_index) {
    __vector unsigned char const packed_u8x16 = sz_aes256_block_load_powervsx_(
        (sz_u8_t const *)&key->round_keys[round_index * 4]);
#if SZ_IS_BIG_ENDIAN_
    return (__vector unsigned char)vec_revb((__vector unsigned int)packed_u8x16);
#else
    return packed_u8x16;
#endif
}

/**
 *  @brief Writes one round key back into an expanded schedule, undoing @ref sz_aes256_round_key_powervsx_.
 *  @param round_key_u8x16 The round key in big-endian register order.
 *  @param words The four schedule words that receive it.
 */
SZ_HELPER_INLINE void sz_aes256_round_key_store_powervsx_(__vector unsigned char round_key_u8x16, sz_u32_t *words) {
#if SZ_IS_BIG_ENDIAN_
    sz_aes256_block_store_powervsx_((__vector unsigned char)vec_revb((__vector unsigned int)round_key_u8x16),
                                    (sz_u8_t *)words);
#else
    sz_aes256_block_store_powervsx_(round_key_u8x16, (sz_u8_t *)words);
#endif
}

/**
 *  @brief Broadcasts a round key's last word across all four words.
 *  @param round_key_u8x16 The round key in big-endian register order.
 *  @return The last word in every lane.
 *
 *  The schedule's recurrence reads only the word that comes last, and `vec_splat` is one of the intrinsics
 *  the compiler renumbers for little-endian, so the lane that holds the most significant word moves.
 */
SZ_HELPER_INLINE __vector unsigned char sz_aes256_key_broadcast_powervsx_(__vector unsigned char round_key_u8x16) {
    return (__vector unsigned char)vec_splat((__vector unsigned int)round_key_u8x16, SZ_IS_BIG_ENDIAN_ ? 3 : 0);
}

/**
 *  @brief Substitutes a round key's last word and broadcasts it, the schedule's plain step.
 *  @param round_key_u8x16 The round key in big-endian register order.
 *  @return The substituted word in every lane.
 */
SZ_HELPER_INLINE __vector unsigned char sz_aes256_key_substitute_powervsx_(__vector unsigned char round_key_u8x16) {
    return vec_sbox_be(sz_aes256_key_broadcast_powervsx_(round_key_u8x16));
}

/**
 *  @brief Rotates, substitutes and offsets a round key's last word, the schedule's round-constant step.
 *  @param round_key_u8x16 The round key in big-endian register order.
 *  @param round_constant The round constant for this step.
 *  @return The transformed word in every lane.
 *
 *  Broadcasting before rotating is what makes the rotation a single instruction: once all four words hold
 *  the same value, sliding the whole register one byte leaves every word holding its own bytes rotated,
 *  the last word borrowing the byte that wraps around. The constant then lands on the word's first byte,
 *  which is the most significant byte of a lane read in big-endian register order.
 */
SZ_HELPER_INLINE __vector unsigned char sz_aes256_key_rotate_substitute_powervsx_(
    __vector unsigned char round_key_u8x16, sz_u32_t round_constant) {
    __vector unsigned char const broadcast_u8x16 = sz_aes256_key_broadcast_powervsx_(round_key_u8x16);
    __vector unsigned char const rotated_u8x16 = vec_sld(broadcast_u8x16, broadcast_u8x16, 1);
    __vector unsigned int const constant_u32x4 = vec_splats((unsigned int)(round_constant << 24));
    return vec_xor(vec_sbox_be(rotated_u8x16), (__vector unsigned char)constant_u32x4);
}

/**
 *  @brief Folds a substituted schedule word into the round key four words back, producing the next one.
 *  @param previous_u8x16 The round key four words back, all four of its words in one register.
 *  @param substituted_u8x16 The substituted word, already broadcast across every lane.
 *  @return The next round key.
 *
 *  FIPS 197 defines the schedule one word at a time, each word depending on the one before it. The three
 *  cascading word shifts replay that dependency inside a single register, so a whole round key falls out
 *  of one exclusive-or chain rather than four.
 */
SZ_HELPER_INLINE __vector unsigned char sz_aes256_key_fold_powervsx_(__vector unsigned char previous_u8x16,
                                                                     __vector unsigned char substituted_u8x16) {
    __vector unsigned char const zero_u8x16 = sz_aes256_zero_powervsx_();
    previous_u8x16 = vec_xor(previous_u8x16, vec_sld(zero_u8x16, previous_u8x16, 12));
    previous_u8x16 = vec_xor(previous_u8x16, vec_sld(zero_u8x16, previous_u8x16, 12));
    previous_u8x16 = vec_xor(previous_u8x16, vec_sld(zero_u8x16, previous_u8x16, 12));
    return vec_xor(previous_u8x16, substituted_u8x16);
}

SZ_API_COMPTIME void sz_aes256_key_init_powervsx(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    sz_u32_t *schedule = &key->round_keys[0];
    __vector unsigned char even_round_key_u8x16 = sz_aes256_block_load_powervsx_(secret);
    __vector unsigned char odd_round_key_u8x16 = sz_aes256_block_load_powervsx_(secret + 16);

    //  An AES-256 schedule alternates two steps: a rotated substitution against a round constant, and a
    //  plain substitution. Seven of the first and six of the second follow the two round keys the secret
    //  supplies, which is thirteen derived round keys for fifteen in total.
    sz_aes256_round_key_store_powervsx_(even_round_key_u8x16, schedule + 0 * 4);
    sz_aes256_round_key_store_powervsx_(odd_round_key_u8x16, schedule + 1 * 4);
    even_round_key_u8x16 = sz_aes256_key_fold_powervsx_(
        even_round_key_u8x16, sz_aes256_key_rotate_substitute_powervsx_(odd_round_key_u8x16, 0x01u));
    sz_aes256_round_key_store_powervsx_(even_round_key_u8x16, schedule + 2 * 4);
    odd_round_key_u8x16 = sz_aes256_key_fold_powervsx_(odd_round_key_u8x16,
                                                       sz_aes256_key_substitute_powervsx_(even_round_key_u8x16));
    sz_aes256_round_key_store_powervsx_(odd_round_key_u8x16, schedule + 3 * 4);
    even_round_key_u8x16 = sz_aes256_key_fold_powervsx_(
        even_round_key_u8x16, sz_aes256_key_rotate_substitute_powervsx_(odd_round_key_u8x16, 0x02u));
    sz_aes256_round_key_store_powervsx_(even_round_key_u8x16, schedule + 4 * 4);
    odd_round_key_u8x16 = sz_aes256_key_fold_powervsx_(odd_round_key_u8x16,
                                                       sz_aes256_key_substitute_powervsx_(even_round_key_u8x16));
    sz_aes256_round_key_store_powervsx_(odd_round_key_u8x16, schedule + 5 * 4);
    even_round_key_u8x16 = sz_aes256_key_fold_powervsx_(
        even_round_key_u8x16, sz_aes256_key_rotate_substitute_powervsx_(odd_round_key_u8x16, 0x04u));
    sz_aes256_round_key_store_powervsx_(even_round_key_u8x16, schedule + 6 * 4);
    odd_round_key_u8x16 = sz_aes256_key_fold_powervsx_(odd_round_key_u8x16,
                                                       sz_aes256_key_substitute_powervsx_(even_round_key_u8x16));
    sz_aes256_round_key_store_powervsx_(odd_round_key_u8x16, schedule + 7 * 4);
    even_round_key_u8x16 = sz_aes256_key_fold_powervsx_(
        even_round_key_u8x16, sz_aes256_key_rotate_substitute_powervsx_(odd_round_key_u8x16, 0x08u));
    sz_aes256_round_key_store_powervsx_(even_round_key_u8x16, schedule + 8 * 4);
    odd_round_key_u8x16 = sz_aes256_key_fold_powervsx_(odd_round_key_u8x16,
                                                       sz_aes256_key_substitute_powervsx_(even_round_key_u8x16));
    sz_aes256_round_key_store_powervsx_(odd_round_key_u8x16, schedule + 9 * 4);
    even_round_key_u8x16 = sz_aes256_key_fold_powervsx_(
        even_round_key_u8x16, sz_aes256_key_rotate_substitute_powervsx_(odd_round_key_u8x16, 0x10u));
    sz_aes256_round_key_store_powervsx_(even_round_key_u8x16, schedule + 10 * 4);
    odd_round_key_u8x16 = sz_aes256_key_fold_powervsx_(odd_round_key_u8x16,
                                                       sz_aes256_key_substitute_powervsx_(even_round_key_u8x16));
    sz_aes256_round_key_store_powervsx_(odd_round_key_u8x16, schedule + 11 * 4);
    even_round_key_u8x16 = sz_aes256_key_fold_powervsx_(
        even_round_key_u8x16, sz_aes256_key_rotate_substitute_powervsx_(odd_round_key_u8x16, 0x20u));
    sz_aes256_round_key_store_powervsx_(even_round_key_u8x16, schedule + 12 * 4);
    odd_round_key_u8x16 = sz_aes256_key_fold_powervsx_(odd_round_key_u8x16,
                                                       sz_aes256_key_substitute_powervsx_(even_round_key_u8x16));
    sz_aes256_round_key_store_powervsx_(odd_round_key_u8x16, schedule + 13 * 4);
    even_round_key_u8x16 = sz_aes256_key_fold_powervsx_(
        even_round_key_u8x16, sz_aes256_key_rotate_substitute_powervsx_(odd_round_key_u8x16, 0x40u));
    sz_aes256_round_key_store_powervsx_(even_round_key_u8x16, schedule + 14 * 4);
}

#pragma endregion // Key Schedule

#pragma region Block Encryption

/** @brief Applies one middle round to all eight chains, so the eight issue back to back. */
SZ_HELPER_INLINE void sz_aes256_blocks_round_powervsx_(__vector unsigned char *blocks_u8x16,
                                                       __vector unsigned char round_key_u8x16) {
    blocks_u8x16[0] = vec_cipher_be(blocks_u8x16[0], round_key_u8x16);
    blocks_u8x16[1] = vec_cipher_be(blocks_u8x16[1], round_key_u8x16);
    blocks_u8x16[2] = vec_cipher_be(blocks_u8x16[2], round_key_u8x16);
    blocks_u8x16[3] = vec_cipher_be(blocks_u8x16[3], round_key_u8x16);
    blocks_u8x16[4] = vec_cipher_be(blocks_u8x16[4], round_key_u8x16);
    blocks_u8x16[5] = vec_cipher_be(blocks_u8x16[5], round_key_u8x16);
    blocks_u8x16[6] = vec_cipher_be(blocks_u8x16[6], round_key_u8x16);
    blocks_u8x16[7] = vec_cipher_be(blocks_u8x16[7], round_key_u8x16);
}

/**
 *  @brief Encrypts one block with the expanded schedule.
 *  @param key The expanded schedule.
 *  @param block_u8x16 The plaintext block in big-endian register order.
 *  @return The ciphertext block in big-endian register order.
 */
SZ_HELPER_INLINE __vector unsigned char sz_aes256_block_encrypt_powervsx_(sz_aes256_key_t const *key,
                                                                          __vector unsigned char block_u8x16) {
    block_u8x16 = vec_xor(block_u8x16, sz_aes256_round_key_powervsx_(key, 0));
    block_u8x16 = vec_cipher_be(block_u8x16, sz_aes256_round_key_powervsx_(key, 1));
    block_u8x16 = vec_cipher_be(block_u8x16, sz_aes256_round_key_powervsx_(key, 2));
    block_u8x16 = vec_cipher_be(block_u8x16, sz_aes256_round_key_powervsx_(key, 3));
    block_u8x16 = vec_cipher_be(block_u8x16, sz_aes256_round_key_powervsx_(key, 4));
    block_u8x16 = vec_cipher_be(block_u8x16, sz_aes256_round_key_powervsx_(key, 5));
    block_u8x16 = vec_cipher_be(block_u8x16, sz_aes256_round_key_powervsx_(key, 6));
    block_u8x16 = vec_cipher_be(block_u8x16, sz_aes256_round_key_powervsx_(key, 7));
    block_u8x16 = vec_cipher_be(block_u8x16, sz_aes256_round_key_powervsx_(key, 8));
    block_u8x16 = vec_cipher_be(block_u8x16, sz_aes256_round_key_powervsx_(key, 9));
    block_u8x16 = vec_cipher_be(block_u8x16, sz_aes256_round_key_powervsx_(key, 10));
    block_u8x16 = vec_cipher_be(block_u8x16, sz_aes256_round_key_powervsx_(key, 11));
    block_u8x16 = vec_cipher_be(block_u8x16, sz_aes256_round_key_powervsx_(key, 12));
    block_u8x16 = vec_cipher_be(block_u8x16, sz_aes256_round_key_powervsx_(key, 13));
    return vec_cipherlast_be(block_u8x16, sz_aes256_round_key_powervsx_(key, 14));
}

/**
 *  @brief Encrypts eight blocks with the expanded schedule.
 *  @param key The expanded schedule.
 *  @param blocks_u8x16 The eight plaintext blocks, replaced by their ciphertexts.
 *
 *  One round instruction has several cycles of latency and issues every cycle, so a single chain of
 *  fourteen dependent rounds leaves most of that throughput idle. Eight independent chains fill it, and
 *  eight is also the number of subkey powers the key carries, so the counter and the hash advance in step.
 *
 *  The lanes are written out rather than counted through, because a loop over the group keeps its
 *  subscript live and the compiler then holds the group in memory, storing and reloading all eight blocks
 *  once per round. Constant subscripts let it scalarize the group into eight registers instead. The
 *  rounds are written out for the same reason the lanes are, and because a loop leaves the unrolling to
 *  the optimizer, which only does it at `-O3`.
 */
SZ_HELPER_INLINE void sz_aes256_blocks_encrypt_powervsx_(sz_aes256_key_t const *key,
                                                         __vector unsigned char *blocks_u8x16) {
    __vector unsigned char round_key_u8x16 = sz_aes256_round_key_powervsx_(key, 0);

    blocks_u8x16[0] = vec_xor(blocks_u8x16[0], round_key_u8x16);
    blocks_u8x16[1] = vec_xor(blocks_u8x16[1], round_key_u8x16);
    blocks_u8x16[2] = vec_xor(blocks_u8x16[2], round_key_u8x16);
    blocks_u8x16[3] = vec_xor(blocks_u8x16[3], round_key_u8x16);
    blocks_u8x16[4] = vec_xor(blocks_u8x16[4], round_key_u8x16);
    blocks_u8x16[5] = vec_xor(blocks_u8x16[5], round_key_u8x16);
    blocks_u8x16[6] = vec_xor(blocks_u8x16[6], round_key_u8x16);
    blocks_u8x16[7] = vec_xor(blocks_u8x16[7], round_key_u8x16);
    sz_aes256_blocks_round_powervsx_(blocks_u8x16, sz_aes256_round_key_powervsx_(key, 1));
    sz_aes256_blocks_round_powervsx_(blocks_u8x16, sz_aes256_round_key_powervsx_(key, 2));
    sz_aes256_blocks_round_powervsx_(blocks_u8x16, sz_aes256_round_key_powervsx_(key, 3));
    sz_aes256_blocks_round_powervsx_(blocks_u8x16, sz_aes256_round_key_powervsx_(key, 4));
    sz_aes256_blocks_round_powervsx_(blocks_u8x16, sz_aes256_round_key_powervsx_(key, 5));
    sz_aes256_blocks_round_powervsx_(blocks_u8x16, sz_aes256_round_key_powervsx_(key, 6));
    sz_aes256_blocks_round_powervsx_(blocks_u8x16, sz_aes256_round_key_powervsx_(key, 7));
    sz_aes256_blocks_round_powervsx_(blocks_u8x16, sz_aes256_round_key_powervsx_(key, 8));
    sz_aes256_blocks_round_powervsx_(blocks_u8x16, sz_aes256_round_key_powervsx_(key, 9));
    sz_aes256_blocks_round_powervsx_(blocks_u8x16, sz_aes256_round_key_powervsx_(key, 10));
    sz_aes256_blocks_round_powervsx_(blocks_u8x16, sz_aes256_round_key_powervsx_(key, 11));
    sz_aes256_blocks_round_powervsx_(blocks_u8x16, sz_aes256_round_key_powervsx_(key, 12));
    sz_aes256_blocks_round_powervsx_(blocks_u8x16, sz_aes256_round_key_powervsx_(key, 13));
    round_key_u8x16 = sz_aes256_round_key_powervsx_(key, 14);
    blocks_u8x16[0] = vec_cipherlast_be(blocks_u8x16[0], round_key_u8x16);
    blocks_u8x16[1] = vec_cipherlast_be(blocks_u8x16[1], round_key_u8x16);
    blocks_u8x16[2] = vec_cipherlast_be(blocks_u8x16[2], round_key_u8x16);
    blocks_u8x16[3] = vec_cipherlast_be(blocks_u8x16[3], round_key_u8x16);
    blocks_u8x16[4] = vec_cipherlast_be(blocks_u8x16[4], round_key_u8x16);
    blocks_u8x16[5] = vec_cipherlast_be(blocks_u8x16[5], round_key_u8x16);
    blocks_u8x16[6] = vec_cipherlast_be(blocks_u8x16[6], round_key_u8x16);
    blocks_u8x16[7] = vec_cipherlast_be(blocks_u8x16[7], round_key_u8x16);
}

#pragma endregion // Block Encryption

#pragma region Counter Mode

/**
 *  @brief Places the twelve nonce bytes in a counter block whose trailing index word is zero.
 *  @param nonce The twelve nonce bytes.
 *  @return The counter block for index zero, in big-endian register order.
 */
SZ_HELPER_INLINE __vector unsigned char sz_aes256_counter_base_powervsx_(sz_u8_t const *nonce) {
    // Only twelve bytes are readable, and Power9's length-limited load reads exactly that many and
    // zeroes the rest. It loads in memory order, so the little-endian build reverses into register order.
    __vector unsigned char const loaded_u8x16 = vec_xl_len((unsigned char *)nonce, 12);
#if SZ_IS_BIG_ENDIAN_
    return loaded_u8x16;
#else
    return vec_reve(loaded_u8x16);
#endif
}

/**
 *  @brief Completes a counter block with a big-endian block index in its trailing word.
 *  @param base_u8x16 A counter block carrying the nonce, whose trailing word is overwritten.
 *  @param block_index The block index.
 *  @return The counter block for that index, in big-endian register order.
 *
 *  A broadcast index lands in every word, and the selection mask keeps only the last of them. Building the
 *  mask by sliding an all-ones register twelve bytes out keeps it free of any element numbering, so the
 *  same expression is correct on both endiannesses.
 */
SZ_HELPER_INLINE __vector unsigned char sz_aes256_counter_block_powervsx_(__vector unsigned char base_u8x16,
                                                                          sz_u32_t block_index) {
    __vector unsigned char const trailing_mask_u8x16 = vec_sld(sz_aes256_zero_powervsx_(),
                                                               vec_splats((unsigned char)0xFF), 4);
    __vector unsigned char const index_u8x16 = (__vector unsigned char)vec_splats((unsigned int)block_index);
    return vec_sel(base_u8x16, index_u8x16, trailing_mask_u8x16);
}

/**
 *  @brief Reads the block index back out of a counter block's trailing four bytes.
 *  @param counter The counter block, in the order the cipher defines it.
 *  @return The block index.
 */
SZ_HELPER_INLINE sz_u32_t sz_aes256_counter_index_powervsx_(sz_u8_t const *counter) {
    return ((sz_u32_t)counter[12] << 24) | ((sz_u32_t)counter[13] << 16) | ((sz_u32_t)counter[14] << 8) |
           (sz_u32_t)counter[15];
}

/**
 *  @brief Exclusive-ors one keystream block into one block of the message.
 *  @param input The sixteen input bytes.
 *  @param output Receives the sixteen output bytes; may alias @p input.
 *  @param keystream_u8x16 The keystream block.
 */
SZ_HELPER_INLINE void sz_aes256_ctr_lane_powervsx_(sz_u8_t const *input, sz_u8_t *output,
                                                   __vector unsigned char keystream_u8x16) {
    sz_aes256_block_store_powervsx_(vec_xor(sz_aes256_block_load_powervsx_(input), keystream_u8x16), output);
}

SZ_API_COMPTIME void sz_aes256_ctr_xor_powervsx(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                                sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length,
                                                sz_ptr_t output) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    __vector unsigned char const counter_base_u8x16 = sz_aes256_counter_base_powervsx_(nonce);
    sz_u32_t block_index = (sz_u32_t)(byte_offset / SZ_AES_BLOCK_LENGTH);
    sz_size_t within_block = (sz_size_t)(byte_offset % SZ_AES_BLOCK_LENGTH);
    sz_size_t produced = 0;

    //  A start that is not block aligned generates its first block whole and discards the leading bytes.
    if (within_block != 0 && length != 0) {
        sz_u128_vec_t keystream_vec;
        __vector unsigned char const counter_u8x16 = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index);
        sz_aes256_block_store_powervsx_(sz_aes256_block_encrypt_powervsx_(key, counter_u8x16), keystream_vec.u8s);
        for (; within_block != SZ_AES_BLOCK_LENGTH && produced != length; ++within_block, ++produced)
            output_bytes[produced] = (sz_u8_t)(input_bytes[produced] ^ keystream_vec.u8s[within_block]);
        ++block_index;
    }

    for (; produced + 8 * SZ_AES_BLOCK_LENGTH <= length; produced += 8 * SZ_AES_BLOCK_LENGTH) {
        __vector unsigned char keystream_u8x16[8];
        keystream_u8x16[0] = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index + 0u);
        keystream_u8x16[1] = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index + 1u);
        keystream_u8x16[2] = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index + 2u);
        keystream_u8x16[3] = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index + 3u);
        keystream_u8x16[4] = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index + 4u);
        keystream_u8x16[5] = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index + 5u);
        keystream_u8x16[6] = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index + 6u);
        keystream_u8x16[7] = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index + 7u);
        block_index += 8;
        sz_aes256_blocks_encrypt_powervsx_(key, keystream_u8x16);
        sz_aes256_ctr_lane_powervsx_(input_bytes + produced + 0 * SZ_AES_BLOCK_LENGTH,
                                     output_bytes + produced + 0 * SZ_AES_BLOCK_LENGTH, keystream_u8x16[0]);
        sz_aes256_ctr_lane_powervsx_(input_bytes + produced + 1 * SZ_AES_BLOCK_LENGTH,
                                     output_bytes + produced + 1 * SZ_AES_BLOCK_LENGTH, keystream_u8x16[1]);
        sz_aes256_ctr_lane_powervsx_(input_bytes + produced + 2 * SZ_AES_BLOCK_LENGTH,
                                     output_bytes + produced + 2 * SZ_AES_BLOCK_LENGTH, keystream_u8x16[2]);
        sz_aes256_ctr_lane_powervsx_(input_bytes + produced + 3 * SZ_AES_BLOCK_LENGTH,
                                     output_bytes + produced + 3 * SZ_AES_BLOCK_LENGTH, keystream_u8x16[3]);
        sz_aes256_ctr_lane_powervsx_(input_bytes + produced + 4 * SZ_AES_BLOCK_LENGTH,
                                     output_bytes + produced + 4 * SZ_AES_BLOCK_LENGTH, keystream_u8x16[4]);
        sz_aes256_ctr_lane_powervsx_(input_bytes + produced + 5 * SZ_AES_BLOCK_LENGTH,
                                     output_bytes + produced + 5 * SZ_AES_BLOCK_LENGTH, keystream_u8x16[5]);
        sz_aes256_ctr_lane_powervsx_(input_bytes + produced + 6 * SZ_AES_BLOCK_LENGTH,
                                     output_bytes + produced + 6 * SZ_AES_BLOCK_LENGTH, keystream_u8x16[6]);
        sz_aes256_ctr_lane_powervsx_(input_bytes + produced + 7 * SZ_AES_BLOCK_LENGTH,
                                     output_bytes + produced + 7 * SZ_AES_BLOCK_LENGTH, keystream_u8x16[7]);
    }

    for (; produced + SZ_AES_BLOCK_LENGTH <= length; produced += SZ_AES_BLOCK_LENGTH, ++block_index) {
        __vector unsigned char const counter_u8x16 = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index);
        __vector unsigned char const original_u8x16 = sz_aes256_block_load_powervsx_(input_bytes + produced);
        __vector unsigned char const keystream_u8x16 = sz_aes256_block_encrypt_powervsx_(key, counter_u8x16);
        sz_aes256_block_store_powervsx_(vec_xor(original_u8x16, keystream_u8x16), output_bytes + produced);
    }

    if (produced != length) {
        sz_u128_vec_t keystream_vec;
        __vector unsigned char const counter_u8x16 = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index);
        sz_aes256_block_store_powervsx_(sz_aes256_block_encrypt_powervsx_(key, counter_u8x16), keystream_vec.u8s);
        for (within_block = 0; produced != length; ++within_block, ++produced)
            output_bytes[produced] = (sz_u8_t)(input_bytes[produced] ^ keystream_vec.u8s[within_block]);
    }
}

#pragma endregion // Counter Mode

#pragma region Galois Hashing

/**
 *  @brief Accumulates the 256-bit carry-less product of two blocks into a running triple.
 *  @param block_u8x16 One operand, in big-endian register order.
 *  @param subkey_u8x16 The other operand, one of the subkey powers.
 *  @param low_u8x16 Accumulates the product of the two low halves.
 *  @param middle_u8x16 Accumulates both cross products.
 *  @param high_u8x16 Accumulates the product of the two high halves.
 *
 *  The polynomial multiplier here sums two doubleword products rather than producing one, which is a
 *  different shape from the single-product multipliers the other backends use. Isolating a half-product
 *  means blanking the other half of one operand, and the two cross products fall out of a single
 *  instruction once one operand's halves are exchanged, so three instructions still cover the schoolbook
 *  expansion. Blanking the block rather than the subkey is what keeps the eight powers in eight registers.
 *
 *  Splitting the expansion from its reduction is what lets several blocks share one reduction: the field
 *  is linear, so the partial products of a whole group may be summed before folding once.
 */
SZ_HELPER_INLINE void sz_ghash_accumulate_powervsx_(__vector unsigned char block_u8x16,
                                                    __vector unsigned char subkey_u8x16,
                                                    __vector unsigned char *low_u8x16,
                                                    __vector unsigned char *middle_u8x16,
                                                    __vector unsigned char *high_u8x16) {
    __vector unsigned char const zero_u8x16 = sz_aes256_zero_powervsx_();
    __vector unsigned char const exchanged_u8x16 = vec_sld(block_u8x16, block_u8x16, 8);
    __vector unsigned char const high_only_u8x16 = vec_sld(exchanged_u8x16, zero_u8x16, 8);
    __vector unsigned char const low_only_u8x16 = vec_sld(zero_u8x16, exchanged_u8x16, 8);

    *low_u8x16 = vec_xor(*low_u8x16, (__vector unsigned char)vec_pmsum_be((__vector unsigned long long)low_only_u8x16,
                                                                          (__vector unsigned long long)subkey_u8x16));
    *middle_u8x16 = vec_xor(*middle_u8x16,
                            (__vector unsigned char)vec_pmsum_be((__vector unsigned long long)exchanged_u8x16,
                                                                 (__vector unsigned long long)subkey_u8x16));
    *high_u8x16 = vec_xor(*high_u8x16,
                          (__vector unsigned char)vec_pmsum_be((__vector unsigned long long)high_only_u8x16,
                                                               (__vector unsigned long long)subkey_u8x16));
}

/**
 *  @brief Folds an accumulated triple into one block.
 *  @param low_u8x16 The accumulated low halves.
 *  @param middle_u8x16 The accumulated cross products.
 *  @param high_u8x16 The accumulated high halves.
 *  @return The reduced product, in big-endian register order.
 *
 *  The cross products straddle the halves, so they are split and merged first. The tag's byte order puts
 *  the whole 256-bit value one bit below where the reduction expects it, hence the leftward shift, and
 *  what remains is the standard fold modulo `x^128 + x^7 + x^2 + x + 1`, done in two passes because the
 *  three low taps are close enough together to share a shift chain.
 */
SZ_HELPER_INLINE __vector unsigned char sz_ghash_reduce_powervsx_(__vector unsigned char low_u8x16,
                                                                  __vector unsigned char middle_u8x16,
                                                                  __vector unsigned char high_u8x16) {
    __vector unsigned char const zero_u8x16 = sz_aes256_zero_powervsx_();
    __vector unsigned int const one_u32x4 = vec_splats((unsigned int)1);
    __vector unsigned int const two_u32x4 = vec_splats((unsigned int)2);
    __vector unsigned int const shift_seven_u32x4 = vec_splats((unsigned int)7);
    __vector unsigned int const twenty_five_u32x4 = vec_splats((unsigned int)25);
    __vector unsigned int const thirty_u32x4 = vec_splats((unsigned int)30);
    __vector unsigned int const thirty_one_u32x4 = vec_splats((unsigned int)31);
    __vector unsigned char low_carries_u8x16, high_carries_u8x16, crossing_carry_u8x16, folded_u8x16,
        folded_spill_u8x16, reduced_taps_u8x16;

    low_u8x16 = vec_xor(low_u8x16, vec_sld(middle_u8x16, zero_u8x16, 8));
    high_u8x16 = vec_xor(high_u8x16, vec_sld(zero_u8x16, middle_u8x16, 8));

    low_carries_u8x16 = (__vector unsigned char)vec_sr((__vector unsigned int)low_u8x16, thirty_one_u32x4);
    high_carries_u8x16 = (__vector unsigned char)vec_sr((__vector unsigned int)high_u8x16, thirty_one_u32x4);
    crossing_carry_u8x16 = vec_sld(zero_u8x16, low_carries_u8x16, 4);
    low_u8x16 = vec_or((__vector unsigned char)vec_sl((__vector unsigned int)low_u8x16, one_u32x4),
                       vec_sld(low_carries_u8x16, zero_u8x16, 4));
    high_u8x16 = vec_or((__vector unsigned char)vec_sl((__vector unsigned int)high_u8x16, one_u32x4),
                        vec_sld(high_carries_u8x16, zero_u8x16, 4));
    high_u8x16 = vec_or(high_u8x16, crossing_carry_u8x16);

    folded_u8x16 = (__vector unsigned char)vec_sl((__vector unsigned int)low_u8x16, thirty_one_u32x4);
    folded_u8x16 = vec_xor(folded_u8x16,
                           (__vector unsigned char)vec_sl((__vector unsigned int)low_u8x16, thirty_u32x4));
    folded_u8x16 = vec_xor(folded_u8x16,
                           (__vector unsigned char)vec_sl((__vector unsigned int)low_u8x16, twenty_five_u32x4));
    folded_spill_u8x16 = vec_sld(zero_u8x16, folded_u8x16, 12);
    low_u8x16 = vec_xor(low_u8x16, vec_sld(folded_u8x16, zero_u8x16, 12));

    reduced_taps_u8x16 = (__vector unsigned char)vec_sr((__vector unsigned int)low_u8x16, one_u32x4);
    reduced_taps_u8x16 = vec_xor(reduced_taps_u8x16,
                                 (__vector unsigned char)vec_sr((__vector unsigned int)low_u8x16, two_u32x4));
    reduced_taps_u8x16 = vec_xor(reduced_taps_u8x16,
                                 (__vector unsigned char)vec_sr((__vector unsigned int)low_u8x16, shift_seven_u32x4));
    reduced_taps_u8x16 = vec_xor(reduced_taps_u8x16, folded_spill_u8x16);
    return vec_xor(high_u8x16, vec_xor(low_u8x16, reduced_taps_u8x16));
}

/**
 *  @brief Multiplies two blocks in the Galois field the tag is built over.
 *  @param accumulator_u8x16 The running hash.
 *  @param subkey_u8x16 One of the subkey powers.
 *  @return The reduced product, in big-endian register order.
 */
SZ_HELPER_INLINE __vector unsigned char sz_ghash_multiply_powervsx_(__vector unsigned char accumulator_u8x16,
                                                                    __vector unsigned char subkey_u8x16) {
    __vector unsigned char low_u8x16 = sz_aes256_zero_powervsx_();
    __vector unsigned char middle_u8x16 = low_u8x16;
    __vector unsigned char high_u8x16 = low_u8x16;
    sz_ghash_accumulate_powervsx_(accumulator_u8x16, subkey_u8x16, &low_u8x16, &middle_u8x16, &high_u8x16);
    return sz_ghash_reduce_powervsx_(low_u8x16, middle_u8x16, high_u8x16);
}

/**
 *  @brief Absorbs one block into the running hash.
 *  @param accumulator_u8x16 The running hash.
 *  @param block_u8x16 The block to absorb, in big-endian register order.
 *  @param subkey_u8x16 The hash subkey.
 *  @return The updated running hash.
 */
SZ_HELPER_INLINE __vector unsigned char sz_ghash_absorb_powervsx_(__vector unsigned char accumulator_u8x16,
                                                                  __vector unsigned char block_u8x16,
                                                                  __vector unsigned char subkey_u8x16) {
    return sz_ghash_multiply_powervsx_(vec_xor(accumulator_u8x16, block_u8x16), subkey_u8x16);
}

SZ_API_COMPTIME void sz_aes256_gcm_key_init_powervsx(sz_aes256_gcm_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    __vector unsigned char subkey_u8x16, power_u8x16;
    sz_size_t power_index;

    sz_aes256_key_init_powervsx(&key->block, secret);
    subkey_u8x16 = sz_aes256_block_encrypt_powervsx_(&key->block, sz_aes256_zero_powervsx_());
    power_u8x16 = subkey_u8x16;
    sz_aes256_block_store_powervsx_(power_u8x16, &key->powers[0]);
    for (power_index = 1; power_index != 8; ++power_index) {
        power_u8x16 = sz_ghash_multiply_powervsx_(power_u8x16, subkey_u8x16);
        sz_aes256_block_store_powervsx_(power_u8x16, &key->powers[power_index * SZ_AES_BLOCK_LENGTH]);
    }
}

/**
 *  @brief Loads the eight subkey powers, descending, so power `j` pairs with block `j` of a group.
 *  @param powers The eight subkey powers, ascending.
 *  @param powers_u8x16 Receives `H^8` down to `H^1`.
 */
SZ_HELPER_INLINE void sz_ghash_descending_powers_powervsx_(sz_u8_t const *powers,
                                                           __vector unsigned char *powers_u8x16) {
    powers_u8x16[0] = sz_aes256_block_load_powervsx_(powers + 7 * SZ_AES_BLOCK_LENGTH);
    powers_u8x16[1] = sz_aes256_block_load_powervsx_(powers + 6 * SZ_AES_BLOCK_LENGTH);
    powers_u8x16[2] = sz_aes256_block_load_powervsx_(powers + 5 * SZ_AES_BLOCK_LENGTH);
    powers_u8x16[3] = sz_aes256_block_load_powervsx_(powers + 4 * SZ_AES_BLOCK_LENGTH);
    powers_u8x16[4] = sz_aes256_block_load_powervsx_(powers + 3 * SZ_AES_BLOCK_LENGTH);
    powers_u8x16[5] = sz_aes256_block_load_powervsx_(powers + 2 * SZ_AES_BLOCK_LENGTH);
    powers_u8x16[6] = sz_aes256_block_load_powervsx_(powers + 1 * SZ_AES_BLOCK_LENGTH);
    powers_u8x16[7] = sz_aes256_block_load_powervsx_(powers + 0 * SZ_AES_BLOCK_LENGTH);
}

/**
 *  @brief Absorbs eight blocks into the running hash with a single field reduction.
 *  @param accumulator_u8x16 The running hash.
 *  @param blocks_u8x16 The eight blocks, in the order they arrived.
 *  @param powers_u8x16 The powers `H^8` through `H^1`.
 *  @return The updated running hash.
 *
 *  Eight absorbed blocks expand to `(Y ^ X1) H^8 ^ X2 H^7 ^ ... ^ X8 H`, which needs eight multiplies
 *  either way but only one reduction instead of eight.
 */
SZ_HELPER_INLINE __vector unsigned char sz_ghash_absorb_eight_powervsx_(__vector unsigned char accumulator_u8x16,
                                                                        __vector unsigned char const *blocks_u8x16,
                                                                        __vector unsigned char const *powers_u8x16) {
    __vector unsigned char low_u8x16 = sz_aes256_zero_powervsx_();
    __vector unsigned char middle_u8x16 = low_u8x16;
    __vector unsigned char high_u8x16 = low_u8x16;

    sz_ghash_accumulate_powervsx_(vec_xor(accumulator_u8x16, blocks_u8x16[0]), powers_u8x16[0], &low_u8x16,
                                  &middle_u8x16, &high_u8x16);
    sz_ghash_accumulate_powervsx_(blocks_u8x16[1], powers_u8x16[1], &low_u8x16, &middle_u8x16, &high_u8x16);
    sz_ghash_accumulate_powervsx_(blocks_u8x16[2], powers_u8x16[2], &low_u8x16, &middle_u8x16, &high_u8x16);
    sz_ghash_accumulate_powervsx_(blocks_u8x16[3], powers_u8x16[3], &low_u8x16, &middle_u8x16, &high_u8x16);
    sz_ghash_accumulate_powervsx_(blocks_u8x16[4], powers_u8x16[4], &low_u8x16, &middle_u8x16, &high_u8x16);
    sz_ghash_accumulate_powervsx_(blocks_u8x16[5], powers_u8x16[5], &low_u8x16, &middle_u8x16, &high_u8x16);
    sz_ghash_accumulate_powervsx_(blocks_u8x16[6], powers_u8x16[6], &low_u8x16, &middle_u8x16, &high_u8x16);
    sz_ghash_accumulate_powervsx_(blocks_u8x16[7], powers_u8x16[7], &low_u8x16, &middle_u8x16, &high_u8x16);
    return sz_ghash_reduce_powervsx_(low_u8x16, middle_u8x16, high_u8x16);
}

#pragma endregion // Galois Hashing

#pragma region Streaming Interface

/**
 *  @brief Overwrites a finished state so the key schedule it embeds does not outlive the call.
 *
 *  The size is known at compile time, so this is a straight-line fill rather than a length-driven loop.
 *  A loop spanning the whole state is recognized as a byte fill and lowered to a library call, which is
 *  the one shape to avoid: zeroing a dead local through such a call is exactly the store a compiler may
 *  drop. Twenty-nine whole vector stores cover the state's first 464 bytes and the last eight merge into
 *  a single scalar store, so nothing is written past the state either. Writes are ordinary stores
 *  followed by one barrier: routing them through a `volatile` view instead would forbid vectorization
 *  and cost 472 single-byte stores on every one-shot call.
 *
 *  The value stored is zero, so the big-endian register order the rest of this file keeps does not enter
 *  into it, and the plain store rather than the reversing one serves on both endiannesses.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_state_scrub_powervsx_(sz_aes256_gcm_state_t *state) {
    __vector unsigned char const zero_u8x16 = sz_aes256_zero_powervsx_();
    unsigned char *const bytes = (unsigned char *)state;
    sz_size_t byte_index;

    vec_xst(zero_u8x16, 0 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 1 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 2 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 3 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 4 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 5 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 6 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 7 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 8 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 9 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 10 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 11 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 12 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 13 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 14 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 15 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 16 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 17 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 18 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 19 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 20 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 21 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 22 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 23 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 24 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 25 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 26 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 27 * SZ_AES_BLOCK_LENGTH, bytes);
    vec_xst(zero_u8x16, 28 * SZ_AES_BLOCK_LENGTH, bytes);
    for (byte_index = 29 * SZ_AES_BLOCK_LENGTH; byte_index != sizeof(*state); ++byte_index) bytes[byte_index] = 0;
    sz_keep_alive_(state);
}

/** @brief Prepares the payload both directions share: counter block, tag mask and empty carries. */
SZ_HELPER_INLINE void sz_aes256_gcm_begin_powervsx_(sz_aes256_gcm_state_t *state, sz_aes256_gcm_key_t const *key,
                                                    sz_u8_t const nonce[sz_at_least_(12)]) {
    __vector unsigned char const zero_u8x16 = sz_aes256_zero_powervsx_();
    __vector unsigned char initial_u8x16;

    state->key = *key;

    //  With a twelve-byte nonce the initial counter block is the nonce followed by a one, and the value
    //  encrypted under it masks the finished hash. Data blocks start one past it.
    initial_u8x16 = sz_aes256_counter_block_powervsx_(sz_aes256_counter_base_powervsx_(nonce), 1u);
    sz_aes256_block_store_powervsx_(sz_aes256_block_encrypt_powervsx_(&state->key.block, initial_u8x16),
                                    state->tag_mask);
    sz_aes256_block_store_powervsx_(initial_u8x16, state->counter);
    sz_aes256_block_store_powervsx_(zero_u8x16, state->accumulator);
    sz_aes256_block_store_powervsx_(zero_u8x16, state->partial);
    sz_aes256_block_store_powervsx_(zero_u8x16, state->keystream);

    state->associated_length = 0;
    state->text_length = 0;
    state->buffered = 0;
    state->keystream_used = SZ_AES_BLOCK_LENGTH; // ? Forces the first message byte to derive a fresh block
}

/** @brief Absorbs associated data into the payload both directions share. */
SZ_HELPER_AUTO void sz_aes256_gcm_associate_powervsx_(sz_aes256_gcm_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    __vector unsigned char const subkey_u8x16 = sz_aes256_block_load_powervsx_(state->key.powers);
    __vector unsigned char powers_u8x16[8];
    __vector unsigned char accumulator_u8x16;
    sz_size_t consumed = 0, byte_index;

    if (length == 0) return;
    sz_ghash_descending_powers_powervsx_(state->key.powers, powers_u8x16);
    accumulator_u8x16 = sz_aes256_block_load_powervsx_(state->accumulator);
    state->associated_length += length;

    //  Associated data is hashed but never encrypted, so a partial block is completed in place.
    if (state->buffered != 0) {
        sz_size_t const available_bytes = SZ_AES_BLOCK_LENGTH - state->buffered;
        sz_size_t const taken_bytes = length < available_bytes ? length : available_bytes;
        for (byte_index = 0; byte_index != taken_bytes; ++byte_index)
            state->partial[state->buffered + byte_index] = input_bytes[byte_index];
        state->buffered = (sz_u8_t)(state->buffered + taken_bytes);
        consumed = taken_bytes;
        if (state->buffered == SZ_AES_BLOCK_LENGTH) {
            accumulator_u8x16 = sz_ghash_absorb_powervsx_(accumulator_u8x16,
                                                          sz_aes256_block_load_powervsx_(state->partial), subkey_u8x16);
            state->buffered = 0;
        }
    }

    for (; consumed + 8 * SZ_AES_BLOCK_LENGTH <= length; consumed += 8 * SZ_AES_BLOCK_LENGTH) {
        __vector unsigned char blocks_u8x16[8];
        blocks_u8x16[0] = sz_aes256_block_load_powervsx_(input_bytes + consumed + 0 * SZ_AES_BLOCK_LENGTH);
        blocks_u8x16[1] = sz_aes256_block_load_powervsx_(input_bytes + consumed + 1 * SZ_AES_BLOCK_LENGTH);
        blocks_u8x16[2] = sz_aes256_block_load_powervsx_(input_bytes + consumed + 2 * SZ_AES_BLOCK_LENGTH);
        blocks_u8x16[3] = sz_aes256_block_load_powervsx_(input_bytes + consumed + 3 * SZ_AES_BLOCK_LENGTH);
        blocks_u8x16[4] = sz_aes256_block_load_powervsx_(input_bytes + consumed + 4 * SZ_AES_BLOCK_LENGTH);
        blocks_u8x16[5] = sz_aes256_block_load_powervsx_(input_bytes + consumed + 5 * SZ_AES_BLOCK_LENGTH);
        blocks_u8x16[6] = sz_aes256_block_load_powervsx_(input_bytes + consumed + 6 * SZ_AES_BLOCK_LENGTH);
        blocks_u8x16[7] = sz_aes256_block_load_powervsx_(input_bytes + consumed + 7 * SZ_AES_BLOCK_LENGTH);
        accumulator_u8x16 = sz_ghash_absorb_eight_powervsx_(accumulator_u8x16, blocks_u8x16, powers_u8x16);
    }
    for (; consumed + SZ_AES_BLOCK_LENGTH <= length; consumed += SZ_AES_BLOCK_LENGTH)
        accumulator_u8x16 = sz_ghash_absorb_powervsx_(
            accumulator_u8x16, sz_aes256_block_load_powervsx_(input_bytes + consumed), subkey_u8x16);
    for (; consumed != length; ++consumed) state->partial[state->buffered++] = input_bytes[consumed];

    sz_aes256_block_store_powervsx_(accumulator_u8x16, state->accumulator);
}

/**
 *  @brief Absorbs whatever the hash block holds, zero padded to a full block.
 *  @param partial The pending hash bytes.
 *  @param buffered How many of them are valid.
 *  @param accumulator_u8x16 The running hash.
 *  @param subkey_u8x16 The hash subkey.
 *  @return The updated running hash.
 *
 *  Emptying the block is left to the caller, because the digest reads a state it may not modify while the
 *  transform owns one it must.
 */
SZ_HELPER_INLINE __vector unsigned char sz_aes256_gcm_flush_partial_powervsx_(sz_u8_t const *partial,
                                                                              sz_size_t buffered,
                                                                              __vector unsigned char accumulator_u8x16,
                                                                              __vector unsigned char subkey_u8x16) {
    // A length-limited load reads the live bytes and zeroes the rest, replacing sixteen per-byte
    // selects. `partial` is a sixteen-byte field, so the shorter read is safe by construction.
    __vector unsigned char const loaded_u8x16 = vec_xl_len((unsigned char *)partial, (sz_size_t)buffered);
#if SZ_IS_BIG_ENDIAN_
    __vector unsigned char const padded_u8x16 = loaded_u8x16;
#else
    __vector unsigned char const padded_u8x16 = vec_reve(loaded_u8x16);
#endif
    return sz_ghash_absorb_powervsx_(accumulator_u8x16, padded_u8x16, subkey_u8x16);
}

/**
 *  @brief Spends what is left of the keystream block the state carries, one byte at a time.
 *  @param state The state, whose two sixteen-byte counters advance together here.
 *  @param input The bytes to transform.
 *  @param output Receives the transformed bytes.
 *  @param count Bytes to consume, never more than the keystream block has left.
 *  @param accumulator_u8x16 The running hash.
 *  @param subkey_u8x16 The hash subkey.
 *  @param direction Which buffer the hash absorbs.
 *  @return The updated running hash.
 *
 *  Through the message the keystream offset and the hash offset are the same number, because every byte
 *  spends one of each, so a chunk that ends mid block leaves both mid block and this resumes both.
 */
SZ_HELPER_INLINE __vector unsigned char sz_aes256_gcm_spend_powervsx_(sz_aes256_gcm_state_t *state,
                                                                      sz_u8_t const *input, sz_u8_t *output,
                                                                      sz_size_t count,
                                                                      __vector unsigned char accumulator_u8x16,
                                                                      __vector unsigned char subkey_u8x16,
                                                                      sz_aes256_gcm_direction_t direction) {
    sz_size_t byte_index;

    //  The hash always eats ciphertext, which is the output when encrypting and the input when
    //  decrypting, and the byte is captured before the store because a caller may pass one pointer.
    for (byte_index = 0; byte_index != count; ++byte_index) {
        sz_u8_t const plaintext_byte = input[byte_index];
        sz_u8_t const transformed = (sz_u8_t)(plaintext_byte ^ state->keystream[state->keystream_used]);
        sz_u8_t const ciphertext_byte = direction == sz_aes256_gcm_decrypting_k ? plaintext_byte : transformed;
        output[byte_index] = transformed;
        state->partial[state->buffered] = ciphertext_byte;
        ++state->buffered;
        ++state->keystream_used;
        if (state->buffered == SZ_AES_BLOCK_LENGTH) {
            accumulator_u8x16 = sz_ghash_absorb_powervsx_(accumulator_u8x16,
                                                          sz_aes256_block_load_powervsx_(state->partial), subkey_u8x16);
            state->buffered = 0;
        }
    }
    return accumulator_u8x16;
}

/**
 *  @brief Transforms one whole block and hands back the ciphertext the hash must absorb.
 *  @param input The sixteen input bytes.
 *  @param output Receives the sixteen output bytes; may alias @p input.
 *  @param keystream_u8x16 The keystream block.
 *  @param cipher_mask_u8x16 All ones when decrypting, all zeros when encrypting.
 *  @return The ciphertext, which is the output when encrypting and the input when decrypting.
 *
 *  The ciphertext leaves in a register rather than being read back from @p output, because a caller may
 *  pass one pointer for both and the tag would then be built over plaintext.
 */
SZ_HELPER_INLINE __vector unsigned char sz_aes256_gcm_lane_powervsx_(sz_u8_t const *input, sz_u8_t *output,
                                                                     __vector unsigned char keystream_u8x16,
                                                                     __vector unsigned char cipher_mask_u8x16) {
    __vector unsigned char const original_u8x16 = sz_aes256_block_load_powervsx_(input);
    __vector unsigned char const transformed_u8x16 = vec_xor(original_u8x16, keystream_u8x16);
    sz_aes256_block_store_powervsx_(transformed_u8x16, output);
    return vec_sel(transformed_u8x16, original_u8x16, cipher_mask_u8x16);
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
 *  whole blocks eight at a time, then a trailing block that the next chunk will resume.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_transform_powervsx_(sz_aes256_gcm_state_t *state, sz_cptr_t text, sz_size_t length,
                                                        sz_ptr_t output, sz_aes256_gcm_direction_t direction) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    __vector unsigned char const subkey_u8x16 = sz_aes256_block_load_powervsx_(state->key.powers);
    __vector unsigned char const cipher_mask_u8x16 = direction == sz_aes256_gcm_decrypting_k
                                                         ? vec_splats((unsigned char)0xFF)
                                                         : vec_splats((unsigned char)0x00);
    __vector unsigned char powers_u8x16[8];
    __vector unsigned char counter_base_u8x16, accumulator_u8x16;
    sz_u32_t block_index;
    sz_size_t produced = 0;

    if (length == 0) return;
    sz_ghash_descending_powers_powervsx_(state->key.powers, powers_u8x16);
    accumulator_u8x16 = sz_aes256_block_load_powervsx_(state->accumulator);

    //  Associated data ends the moment the first message byte arrives, and its tail needs padding.
    if (state->text_length == 0 && state->buffered != 0) {
        accumulator_u8x16 = sz_aes256_gcm_flush_partial_powervsx_(state->partial, state->buffered, accumulator_u8x16,
                                                                  subkey_u8x16);
        state->buffered = 0;
    }

    counter_base_u8x16 = sz_aes256_block_load_powervsx_(state->counter);
    block_index = sz_aes256_counter_index_powervsx_(state->counter);

    if (state->keystream_used != SZ_AES_BLOCK_LENGTH) {
        sz_size_t const available_bytes = SZ_AES_BLOCK_LENGTH - state->keystream_used;
        sz_size_t const taken_bytes = length < available_bytes ? length : available_bytes;
        accumulator_u8x16 = sz_aes256_gcm_spend_powervsx_(state, input_bytes, output_bytes, taken_bytes,
                                                          accumulator_u8x16, subkey_u8x16, direction);
        produced = taken_bytes;
    }

    for (; produced + 8 * SZ_AES_BLOCK_LENGTH <= length; produced += 8 * SZ_AES_BLOCK_LENGTH) {
        __vector unsigned char keystream_u8x16[8], ciphertext_u8x16[8];
        sz_u8_t const *lane_input = input_bytes + produced;
        sz_u8_t *lane_output = output_bytes + produced;
        keystream_u8x16[0] = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index + 1u);
        keystream_u8x16[1] = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index + 2u);
        keystream_u8x16[2] = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index + 3u);
        keystream_u8x16[3] = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index + 4u);
        keystream_u8x16[4] = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index + 5u);
        keystream_u8x16[5] = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index + 6u);
        keystream_u8x16[6] = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index + 7u);
        keystream_u8x16[7] = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index + 8u);
        block_index += 8;
        sz_aes256_blocks_encrypt_powervsx_(&state->key.block, keystream_u8x16);
        ciphertext_u8x16[0] = sz_aes256_gcm_lane_powervsx_(lane_input + 0 * SZ_AES_BLOCK_LENGTH,
                                                           lane_output + 0 * SZ_AES_BLOCK_LENGTH, keystream_u8x16[0],
                                                           cipher_mask_u8x16);
        ciphertext_u8x16[1] = sz_aes256_gcm_lane_powervsx_(lane_input + 1 * SZ_AES_BLOCK_LENGTH,
                                                           lane_output + 1 * SZ_AES_BLOCK_LENGTH, keystream_u8x16[1],
                                                           cipher_mask_u8x16);
        ciphertext_u8x16[2] = sz_aes256_gcm_lane_powervsx_(lane_input + 2 * SZ_AES_BLOCK_LENGTH,
                                                           lane_output + 2 * SZ_AES_BLOCK_LENGTH, keystream_u8x16[2],
                                                           cipher_mask_u8x16);
        ciphertext_u8x16[3] = sz_aes256_gcm_lane_powervsx_(lane_input + 3 * SZ_AES_BLOCK_LENGTH,
                                                           lane_output + 3 * SZ_AES_BLOCK_LENGTH, keystream_u8x16[3],
                                                           cipher_mask_u8x16);
        ciphertext_u8x16[4] = sz_aes256_gcm_lane_powervsx_(lane_input + 4 * SZ_AES_BLOCK_LENGTH,
                                                           lane_output + 4 * SZ_AES_BLOCK_LENGTH, keystream_u8x16[4],
                                                           cipher_mask_u8x16);
        ciphertext_u8x16[5] = sz_aes256_gcm_lane_powervsx_(lane_input + 5 * SZ_AES_BLOCK_LENGTH,
                                                           lane_output + 5 * SZ_AES_BLOCK_LENGTH, keystream_u8x16[5],
                                                           cipher_mask_u8x16);
        ciphertext_u8x16[6] = sz_aes256_gcm_lane_powervsx_(lane_input + 6 * SZ_AES_BLOCK_LENGTH,
                                                           lane_output + 6 * SZ_AES_BLOCK_LENGTH, keystream_u8x16[6],
                                                           cipher_mask_u8x16);
        ciphertext_u8x16[7] = sz_aes256_gcm_lane_powervsx_(lane_input + 7 * SZ_AES_BLOCK_LENGTH,
                                                           lane_output + 7 * SZ_AES_BLOCK_LENGTH, keystream_u8x16[7],
                                                           cipher_mask_u8x16);
        accumulator_u8x16 = sz_ghash_absorb_eight_powervsx_(accumulator_u8x16, ciphertext_u8x16, powers_u8x16);
    }

    for (; produced + SZ_AES_BLOCK_LENGTH <= length; produced += SZ_AES_BLOCK_LENGTH) {
        __vector unsigned char counter_u8x16, keystream_u8x16, ciphertext_u8x16;
        block_index += 1;
        counter_u8x16 = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index);
        keystream_u8x16 = sz_aes256_block_encrypt_powervsx_(&state->key.block, counter_u8x16);
        ciphertext_u8x16 = sz_aes256_gcm_lane_powervsx_(input_bytes + produced, output_bytes + produced,
                                                        keystream_u8x16, cipher_mask_u8x16);
        accumulator_u8x16 = sz_ghash_absorb_powervsx_(accumulator_u8x16, ciphertext_u8x16, subkey_u8x16);
    }

    if (produced != length) {
        __vector unsigned char counter_u8x16, keystream_u8x16;
        block_index += 1;
        counter_u8x16 = sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index);
        keystream_u8x16 = sz_aes256_block_encrypt_powervsx_(&state->key.block, counter_u8x16);
        sz_aes256_block_store_powervsx_(keystream_u8x16, state->keystream);
        state->keystream_used = 0;
        accumulator_u8x16 = sz_aes256_gcm_spend_powervsx_(state, input_bytes + produced, output_bytes + produced,
                                                          length - produced, accumulator_u8x16, subkey_u8x16,
                                                          direction);
    }

    state->text_length += length;
    sz_aes256_block_store_powervsx_(accumulator_u8x16, state->accumulator);
    sz_aes256_block_store_powervsx_(sz_aes256_counter_block_powervsx_(counter_base_u8x16, block_index), state->counter);
}

/**
 *  @brief Finishes the payload both directions share, emitting the masked hash.
 *  @param state The state, left unmodified so a caller may keep appending.
 *  @param tag Receives the sixteen tag bytes.
 */
SZ_HELPER_AUTO void sz_aes256_gcm_digest_powervsx_(sz_aes256_gcm_state_t const *state, sz_u8_t tag[sz_at_least_(16)]) {
    __vector unsigned char const subkey_u8x16 = sz_aes256_block_load_powervsx_(state->key.powers);
    __vector unsigned char accumulator_u8x16 = sz_aes256_block_load_powervsx_(state->accumulator);
    sz_u128_vec_t lengths_vec;

    //  Whatever is still pending gets padded here: an associated-data tail when the message was empty,
    //  or the message's own trailing ciphertext bytes otherwise.
    if (state->buffered != 0)
        accumulator_u8x16 = sz_aes256_gcm_flush_partial_powervsx_(state->partial, state->buffered, accumulator_u8x16,
                                                                  subkey_u8x16);

    sz_aes256_gcm_lengths_serial_(&lengths_vec, state->associated_length, state->text_length);
    accumulator_u8x16 = sz_ghash_absorb_powervsx_(accumulator_u8x16, sz_aes256_block_load_powervsx_(lengths_vec.u8s),
                                                  subkey_u8x16);

    sz_aes256_block_store_powervsx_(vec_xor(accumulator_u8x16, sz_aes256_block_load_powervsx_(state->tag_mask)), tag);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_init_powervsx(sz_aes256_gcm_encryptor_t *encryptor,
                                                           sz_aes256_gcm_key_t const *key,
                                                           sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_aes256_gcm_begin_powervsx_(&encryptor->state, key, nonce);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_associate_powervsx(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                sz_size_t length) {
    sz_aes256_gcm_associate_powervsx_(&encryptor->state, text, length);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_update_powervsx(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                             sz_size_t length, sz_ptr_t output) {
    sz_aes256_gcm_transform_powervsx_(&encryptor->state, text, length, output, sz_aes256_gcm_encrypting_k);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_digest_powervsx(sz_aes256_gcm_encryptor_t const *encryptor,
                                                             sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_digest_powervsx_(&encryptor->state, tag);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_init_powervsx(sz_aes256_gcm_decryptor_t *decryptor,
                                                           sz_aes256_gcm_key_t const *key,
                                                           sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_aes256_gcm_begin_powervsx_(&decryptor->state, key, nonce);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_associate_powervsx(sz_aes256_gcm_decryptor_t *decryptor, sz_cptr_t text,
                                                                sz_size_t length) {
    sz_aes256_gcm_associate_powervsx_(&decryptor->state, text, length);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_update_unverified_powervsx(sz_aes256_gcm_decryptor_t *decryptor,
                                                                        sz_cptr_t text, sz_size_t length,
                                                                        sz_ptr_t output) {
    sz_aes256_gcm_transform_powervsx_(&decryptor->state, text, length, output, sz_aes256_gcm_decrypting_k);
}

SZ_API_COMPTIME sz_status_t sz_aes256_gcm_decryptor_verify_powervsx(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                    sz_u8_t const tag[sz_at_least_(16)]) {
    sz_u128_vec_t expected_vec;
    sz_aes256_gcm_digest_powervsx_(&decryptor->state, expected_vec.u8s);
    return sz_aes256_tag_equal_serial_(expected_vec.u8s, tag) == sz_true_k ? sz_success_k : sz_authentication_failed_k;
}

#pragma endregion // Streaming Interface

#pragma region One Shot Interface

SZ_API_COMPTIME void sz_aes256_gcm_encrypt_powervsx(sz_aes256_gcm_key_t const *key,
                                                    sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                    sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                    sz_ptr_t output, sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_encryptor_t encryptor;
    sz_aes256_gcm_encryptor_init_powervsx(&encryptor, key, nonce);
    if (associated_length) sz_aes256_gcm_encryptor_associate_powervsx(&encryptor, associated, associated_length);
    sz_aes256_gcm_encryptor_update_powervsx(&encryptor, text, length, output);
    sz_aes256_gcm_encryptor_digest_powervsx(&encryptor, tag);
    sz_aes256_gcm_state_scrub_powervsx_(&encryptor.state);
}

SZ_API_COMPTIME sz_status_t sz_aes256_gcm_decrypt_powervsx(sz_aes256_gcm_key_t const *key,
                                                           sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                           sz_size_t associated_length, sz_cptr_t text,
                                                           sz_size_t length, sz_ptr_t output,
                                                           sz_u8_t const tag[sz_at_least_(16)]) {
    sz_aes256_gcm_decryptor_t decryptor;
    sz_status_t verdict;
    sz_aes256_gcm_decryptor_init_powervsx(&decryptor, key, nonce);
    if (associated_length) sz_aes256_gcm_decryptor_associate_powervsx(&decryptor, associated, associated_length);
    sz_aes256_gcm_decryptor_update_unverified_powervsx(&decryptor, text, length, output);
    verdict = sz_aes256_gcm_decryptor_verify_powervsx(&decryptor, tag);
    sz_aes256_gcm_state_scrub_powervsx_(&decryptor.state);

    //  A caller who drops the status must still be unable to act on forged plaintext.
    if (verdict != sz_success_k) sz_fill(output, length, 0);
    return verdict;
}

#pragma endregion // One Shot Interface

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_POWERVSX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_CIPHER_POWERVSX_H_
