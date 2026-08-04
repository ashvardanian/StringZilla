/**
 *  @brief Ice Lake (AVX-512 + VAES + VPCLMULQDQ) backend for AES-256 encryption in counter and Galois/counter modes.
 *  @file include/stringzilla/cipher/icelake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/cipher.h
 *
 *  The fourteen rounds are written out rather than looped, and the block groups are named registers rather
 *  than an indexed array.
 */
#ifndef STRINGZILLA_CIPHER_ICELAKE_H_
#define STRINGZILLA_CIPHER_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/cipher/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_ICELAKE
/*  `avx512vbmi` carries the byte permute the counter-mode head uses. The tier already implies it, and
 *  `hash/icelake.h` and `utf8_graphemes/icelake.h` both name it; this string was short, not narrower. */
#if defined(__clang__)
#pragma clang attribute push(                                                                                         \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,bmi,bmi2,aes,vaes,pclmul,vpclmulqdq"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,bmi,bmi2,aes,vaes,pclmul,vpclmulqdq")
#endif

/*  Four counter blocks share one register, and counter mode keeps four such registers in flight, so
 *  sixteen blocks are encrypted per pass and the fourteen dependent rounds of AES-256 never stall on their
 *  own latency. The Galois hash reaches the same width through `vpclmulqdq`, whose four independent
 *  128-bit lanes let eight blocks accumulate as unreduced carry-less products before a single reduction
 *  closes them out. Authenticated mode runs both in one loop, because they contend for different ports.
 *
 *  The counter is held with its trailing 32-bit field little-endian, because advancing four lanes is then
 *  one `vpaddd`, and byte-reversed into the big-endian block NIST specifies immediately before the cipher
 *  sees it. Both halves of that sentence are load-bearing: a keystream generated from an unswapped
 *  counter looks perfectly random and is simply not AES-CTR.
 */

#pragma region Key Schedule

/**
 *  @brief Folds one schedule step: the running exclusive-or of four words, then the substituted word.
 *  @param previous_u8x16 The four schedule words eight positions back.
 *  @param assisted_u8x16 The substituted word, already broadcast across all four lanes.
 *  @return The next four schedule words.
 *
 *  FIPS 197 writes `w[i] = w[i - 8] ^ temp` with `temp` carried forward through the quadruple, which is the
 *  cumulative exclusive-or that three byte-wise doublings of `_mm_slli_si128` produce.
 */
SZ_HELPER_INLINE __m128i sz_aes256_key_fold_icelake_(__m128i previous_u8x16, __m128i assisted_u8x16) {
    previous_u8x16 = _mm_xor_si128(previous_u8x16, _mm_slli_si128(previous_u8x16, 4));
    previous_u8x16 = _mm_xor_si128(previous_u8x16, _mm_slli_si128(previous_u8x16, 4));
    previous_u8x16 = _mm_xor_si128(previous_u8x16, _mm_slli_si128(previous_u8x16, 4));
    return _mm_xor_si128(previous_u8x16, assisted_u8x16);
}

SZ_API_COMPTIME void sz_aes256_key_init_icelake(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    __m128i secret_low_words_u8x16 = _mm_loadu_epi8(secret);
    __m128i secret_high_words_u8x16 = _mm_loadu_epi8(secret + 16);

    //  `AESKEYGENASSIST` places `RotWord(SubWord(w)) ^ RCON` in the top word and `SubWord(w)` in the third,
    //  so a broadcast of one or the other serves the two shapes the AES-256 schedule alternates between.
    _mm_storeu_epi32(key->round_keys + 0, secret_low_words_u8x16);
    _mm_storeu_epi32(key->round_keys + 4, secret_high_words_u8x16);
    secret_low_words_u8x16 = sz_aes256_key_fold_icelake_(
        secret_low_words_u8x16, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_high_words_u8x16, 0x01), 0xFF));
    _mm_storeu_epi32(key->round_keys + 8, secret_low_words_u8x16);
    secret_high_words_u8x16 = sz_aes256_key_fold_icelake_(
        secret_high_words_u8x16, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_low_words_u8x16, 0x00), 0xAA));
    _mm_storeu_epi32(key->round_keys + 12, secret_high_words_u8x16);
    secret_low_words_u8x16 = sz_aes256_key_fold_icelake_(
        secret_low_words_u8x16, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_high_words_u8x16, 0x02), 0xFF));
    _mm_storeu_epi32(key->round_keys + 16, secret_low_words_u8x16);
    secret_high_words_u8x16 = sz_aes256_key_fold_icelake_(
        secret_high_words_u8x16, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_low_words_u8x16, 0x00), 0xAA));
    _mm_storeu_epi32(key->round_keys + 20, secret_high_words_u8x16);
    secret_low_words_u8x16 = sz_aes256_key_fold_icelake_(
        secret_low_words_u8x16, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_high_words_u8x16, 0x04), 0xFF));
    _mm_storeu_epi32(key->round_keys + 24, secret_low_words_u8x16);
    secret_high_words_u8x16 = sz_aes256_key_fold_icelake_(
        secret_high_words_u8x16, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_low_words_u8x16, 0x00), 0xAA));
    _mm_storeu_epi32(key->round_keys + 28, secret_high_words_u8x16);
    secret_low_words_u8x16 = sz_aes256_key_fold_icelake_(
        secret_low_words_u8x16, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_high_words_u8x16, 0x08), 0xFF));
    _mm_storeu_epi32(key->round_keys + 32, secret_low_words_u8x16);
    secret_high_words_u8x16 = sz_aes256_key_fold_icelake_(
        secret_high_words_u8x16, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_low_words_u8x16, 0x00), 0xAA));
    _mm_storeu_epi32(key->round_keys + 36, secret_high_words_u8x16);
    secret_low_words_u8x16 = sz_aes256_key_fold_icelake_(
        secret_low_words_u8x16, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_high_words_u8x16, 0x10), 0xFF));
    _mm_storeu_epi32(key->round_keys + 40, secret_low_words_u8x16);
    secret_high_words_u8x16 = sz_aes256_key_fold_icelake_(
        secret_high_words_u8x16, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_low_words_u8x16, 0x00), 0xAA));
    _mm_storeu_epi32(key->round_keys + 44, secret_high_words_u8x16);
    secret_low_words_u8x16 = sz_aes256_key_fold_icelake_(
        secret_low_words_u8x16, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_high_words_u8x16, 0x20), 0xFF));
    _mm_storeu_epi32(key->round_keys + 48, secret_low_words_u8x16);
    secret_high_words_u8x16 = sz_aes256_key_fold_icelake_(
        secret_high_words_u8x16, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_low_words_u8x16, 0x00), 0xAA));
    _mm_storeu_epi32(key->round_keys + 52, secret_high_words_u8x16);
    secret_low_words_u8x16 = sz_aes256_key_fold_icelake_(
        secret_low_words_u8x16, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_high_words_u8x16, 0x40), 0xFF));
    _mm_storeu_epi32(key->round_keys + 56, secret_low_words_u8x16);
}

#pragma endregion // Key Schedule

#pragma region Counter Mode

/**
 *  @brief Encrypts one 16-byte block with the expanded schedule.
 *  @param key The expanded schedule.
 *  @param block_u8x16 The plaintext block.
 *  @return The ciphertext block.
 */
SZ_HELPER_INLINE __m128i sz_aes256_block_encrypt_icelake_(sz_aes256_key_t const *key, __m128i block_u8x16) {
    block_u8x16 = _mm_xor_si128(block_u8x16, _mm_loadu_epi32(key->round_keys + 0));
    block_u8x16 = _mm_aesenc_si128(block_u8x16, _mm_loadu_epi32(key->round_keys + 4));
    block_u8x16 = _mm_aesenc_si128(block_u8x16, _mm_loadu_epi32(key->round_keys + 8));
    block_u8x16 = _mm_aesenc_si128(block_u8x16, _mm_loadu_epi32(key->round_keys + 12));
    block_u8x16 = _mm_aesenc_si128(block_u8x16, _mm_loadu_epi32(key->round_keys + 16));
    block_u8x16 = _mm_aesenc_si128(block_u8x16, _mm_loadu_epi32(key->round_keys + 20));
    block_u8x16 = _mm_aesenc_si128(block_u8x16, _mm_loadu_epi32(key->round_keys + 24));
    block_u8x16 = _mm_aesenc_si128(block_u8x16, _mm_loadu_epi32(key->round_keys + 28));
    block_u8x16 = _mm_aesenc_si128(block_u8x16, _mm_loadu_epi32(key->round_keys + 32));
    block_u8x16 = _mm_aesenc_si128(block_u8x16, _mm_loadu_epi32(key->round_keys + 36));
    block_u8x16 = _mm_aesenc_si128(block_u8x16, _mm_loadu_epi32(key->round_keys + 40));
    block_u8x16 = _mm_aesenc_si128(block_u8x16, _mm_loadu_epi32(key->round_keys + 44));
    block_u8x16 = _mm_aesenc_si128(block_u8x16, _mm_loadu_epi32(key->round_keys + 48));
    block_u8x16 = _mm_aesenc_si128(block_u8x16, _mm_loadu_epi32(key->round_keys + 52));
    return _mm_aesenclast_si128(block_u8x16, _mm_loadu_epi32(key->round_keys + 56));
}

/**
 *  @brief Broadcasts each of the fifteen round keys across the four lanes of its own register.
 *  @param key The expanded schedule.
 *  @param wide_keys_vec Receives fifteen registers, one per round.
 */
SZ_HELPER_INLINE void sz_aes256_round_keys_wide_icelake_(sz_aes256_key_t const *key, sz_u512_vec_t *wide_keys_vec) {
    wide_keys_vec[0].zmm = _mm512_broadcast_i32x4(_mm_loadu_epi32(key->round_keys + 0));
    wide_keys_vec[1].zmm = _mm512_broadcast_i32x4(_mm_loadu_epi32(key->round_keys + 4));
    wide_keys_vec[2].zmm = _mm512_broadcast_i32x4(_mm_loadu_epi32(key->round_keys + 8));
    wide_keys_vec[3].zmm = _mm512_broadcast_i32x4(_mm_loadu_epi32(key->round_keys + 12));
    wide_keys_vec[4].zmm = _mm512_broadcast_i32x4(_mm_loadu_epi32(key->round_keys + 16));
    wide_keys_vec[5].zmm = _mm512_broadcast_i32x4(_mm_loadu_epi32(key->round_keys + 20));
    wide_keys_vec[6].zmm = _mm512_broadcast_i32x4(_mm_loadu_epi32(key->round_keys + 24));
    wide_keys_vec[7].zmm = _mm512_broadcast_i32x4(_mm_loadu_epi32(key->round_keys + 28));
    wide_keys_vec[8].zmm = _mm512_broadcast_i32x4(_mm_loadu_epi32(key->round_keys + 32));
    wide_keys_vec[9].zmm = _mm512_broadcast_i32x4(_mm_loadu_epi32(key->round_keys + 36));
    wide_keys_vec[10].zmm = _mm512_broadcast_i32x4(_mm_loadu_epi32(key->round_keys + 40));
    wide_keys_vec[11].zmm = _mm512_broadcast_i32x4(_mm_loadu_epi32(key->round_keys + 44));
    wide_keys_vec[12].zmm = _mm512_broadcast_i32x4(_mm_loadu_epi32(key->round_keys + 48));
    wide_keys_vec[13].zmm = _mm512_broadcast_i32x4(_mm_loadu_epi32(key->round_keys + 52));
    wide_keys_vec[14].zmm = _mm512_broadcast_i32x4(_mm_loadu_epi32(key->round_keys + 56));
}

/**
 *  @brief Applies the thirteen middle rounds and the last round to one four-block group.
 *
 *  The rounds are written out rather than looped.
 */
SZ_HELPER_INLINE __m512i sz_aes256_rounds_wide_icelake_(__m512i block_u8x64, sz_u512_vec_t const *wide_keys_vec) {
    block_u8x64 = _mm512_aesenc_epi128(block_u8x64, wide_keys_vec[1].zmm);
    block_u8x64 = _mm512_aesenc_epi128(block_u8x64, wide_keys_vec[2].zmm);
    block_u8x64 = _mm512_aesenc_epi128(block_u8x64, wide_keys_vec[3].zmm);
    block_u8x64 = _mm512_aesenc_epi128(block_u8x64, wide_keys_vec[4].zmm);
    block_u8x64 = _mm512_aesenc_epi128(block_u8x64, wide_keys_vec[5].zmm);
    block_u8x64 = _mm512_aesenc_epi128(block_u8x64, wide_keys_vec[6].zmm);
    block_u8x64 = _mm512_aesenc_epi128(block_u8x64, wide_keys_vec[7].zmm);
    block_u8x64 = _mm512_aesenc_epi128(block_u8x64, wide_keys_vec[8].zmm);
    block_u8x64 = _mm512_aesenc_epi128(block_u8x64, wide_keys_vec[9].zmm);
    block_u8x64 = _mm512_aesenc_epi128(block_u8x64, wide_keys_vec[10].zmm);
    block_u8x64 = _mm512_aesenc_epi128(block_u8x64, wide_keys_vec[11].zmm);
    block_u8x64 = _mm512_aesenc_epi128(block_u8x64, wide_keys_vec[12].zmm);
    block_u8x64 = _mm512_aesenc_epi128(block_u8x64, wide_keys_vec[13].zmm);
    return _mm512_aesenclast_epi128(block_u8x64, wide_keys_vec[14].zmm);
}

/** @brief Lane identity `{0, 1, ..., 15}`, the base a byte-offset permutation is built from. */
SZ_HELPER_INLINE __m128i sz_aes256_lane_iota_icelake_(void) {
    return _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}

/** @brief Reverses the trailing four bytes of each 128-bit lane, its own inverse. */
SZ_HELPER_INLINE sz_u8_t const *sz_aes256_counter_swap_icelake_(void) {
    static sz_align_(64) sz_u8_t const swap[64] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, //
                                                   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, //
                                                   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, //
                                                   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12};
    return &swap[0];
}

/**
 *  @brief Builds four consecutive counter blocks, the trailing 32-bit field kept little-endian.
 *  @param nonce The 12 nonce bytes.
 *  @param block_index Index of the first of the four blocks.
 *  @return Counters for blocks @p block_index through `block_index + 3`.
 */
SZ_HELPER_INLINE __m512i sz_aes256_counters_icelake_(sz_u8_t const *nonce, sz_u32_t block_index) {
    // Only twelve bytes are readable, and a masked load suppresses the fault on the rest while zeroing
    // them, which is the trailing index word this wants anyway.
    __m128i const nonce_u8x16 = _mm_maskz_loadu_epi8((__mmask16)0x0FFFu, nonce);
    return _mm512_add_epi32(_mm512_broadcast_i32x4(nonce_u8x16),
                            _mm512_add_epi32(_mm512_maskz_set1_epi32((__mmask16)0x8888u, (int)block_index),
                                             _mm512_set_epi32(3, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)));
}

/**
 *  @brief Restores the big-endian counter block held in the first lane of a little-endian counter register.
 *  @param counters_u8x64 Four counter blocks with their trailing field little-endian.
 *  @param counter_swap_u8x64 The trailing-field reversal pattern.
 *  @return The block NIST's counter mode feeds to the cipher.
 */
SZ_HELPER_INLINE __m128i sz_aes256_counter_block_icelake_(__m512i counters_u8x64, __m512i counter_swap_u8x64) {
    return _mm512_castsi512_si128(_mm512_shuffle_epi8(counters_u8x64, counter_swap_u8x64));
}

/**
 *  @brief Exclusive-ors @p length bytes against the keystream, starting on a block boundary.
 *  @param counters_u8x64 Four consecutive counter blocks, the trailing field little-endian.
 *  @param wide_keys_vec The fifteen round keys, each broadcast across four lanes.
 *  @param text The input bytes.
 *  @param output Receives @p length bytes; may equal @p text.
 *  @param length Bytes to transform, not necessarily a multiple of the block length.
 *  @return The counters advanced past every block the call consumed.
 *
 *  Sixteen blocks in flight is what saturates the two AES ports; the 64-byte and masked passes below it exist
 *  only to land the tail, and a caller reaching them has already run out of work to hide latency in.
 */
SZ_HELPER_INLINE __m512i sz_aes256_ctr_stride_icelake_(__m512i counters_u8x64, sz_u512_vec_t const *wide_keys_vec,
                                                       sz_u8_t const *text, sz_u8_t *output, sz_size_t length) {
    __m512i const counter_swap_u8x64 = _mm512_load_si512(sz_aes256_counter_swap_icelake_());
    __m512i const step_four_u8x64 = _mm512_maskz_set1_epi32((__mmask16)0x8888u, 4);
    __m512i const step_sixteen_u8x64 = _mm512_maskz_set1_epi32((__mmask16)0x8888u, 16);
    sz_size_t produced = 0;

    if (length >= 256) {
        __m512i first_group_u8x64 = counters_u8x64;
        __m512i second_group_u8x64 = _mm512_add_epi32(first_group_u8x64, step_four_u8x64);
        __m512i third_group_u8x64 = _mm512_add_epi32(second_group_u8x64, step_four_u8x64);
        __m512i fourth_group_u8x64 = _mm512_add_epi32(third_group_u8x64, step_four_u8x64);
        for (; produced + 256 <= length; produced += 256) {
            // Four groups stay in named registers rather than an indexed array, so the four independent
            // round chains interleave in the scheduler instead of spilling to the stack.
            __m512i first_keystream_u8x64 = _mm512_xor_si512(_mm512_shuffle_epi8(first_group_u8x64, counter_swap_u8x64),
                                                             wide_keys_vec[0].zmm);
            __m512i second_keystream_u8x64 = _mm512_xor_si512(
                _mm512_shuffle_epi8(second_group_u8x64, counter_swap_u8x64), wide_keys_vec[0].zmm);
            __m512i third_keystream_u8x64 = _mm512_xor_si512(_mm512_shuffle_epi8(third_group_u8x64, counter_swap_u8x64),
                                                             wide_keys_vec[0].zmm);
            __m512i fourth_keystream_u8x64 = _mm512_xor_si512(
                _mm512_shuffle_epi8(fourth_group_u8x64, counter_swap_u8x64), wide_keys_vec[0].zmm);
            first_keystream_u8x64 = sz_aes256_rounds_wide_icelake_(first_keystream_u8x64, wide_keys_vec);
            second_keystream_u8x64 = sz_aes256_rounds_wide_icelake_(second_keystream_u8x64, wide_keys_vec);
            third_keystream_u8x64 = sz_aes256_rounds_wide_icelake_(third_keystream_u8x64, wide_keys_vec);
            fourth_keystream_u8x64 = sz_aes256_rounds_wide_icelake_(fourth_keystream_u8x64, wide_keys_vec);
            _mm512_storeu_si512(output + produced + 0,
                                _mm512_xor_si512(first_keystream_u8x64, _mm512_loadu_si512(text + produced + 0)));
            _mm512_storeu_si512(output + produced + 64,
                                _mm512_xor_si512(second_keystream_u8x64, _mm512_loadu_si512(text + produced + 64)));
            _mm512_storeu_si512(output + produced + 128,
                                _mm512_xor_si512(third_keystream_u8x64, _mm512_loadu_si512(text + produced + 128)));
            _mm512_storeu_si512(output + produced + 192,
                                _mm512_xor_si512(fourth_keystream_u8x64, _mm512_loadu_si512(text + produced + 192)));
            first_group_u8x64 = _mm512_add_epi32(first_group_u8x64, step_sixteen_u8x64);
            second_group_u8x64 = _mm512_add_epi32(second_group_u8x64, step_sixteen_u8x64);
            third_group_u8x64 = _mm512_add_epi32(third_group_u8x64, step_sixteen_u8x64);
            fourth_group_u8x64 = _mm512_add_epi32(fourth_group_u8x64, step_sixteen_u8x64);
        }
        counters_u8x64 = first_group_u8x64;
    }

    for (; produced + 64 <= length; produced += 64) {
        __m512i keystream_u8x64 = _mm512_xor_si512(_mm512_shuffle_epi8(counters_u8x64, counter_swap_u8x64),
                                                   wide_keys_vec[0].zmm);
        keystream_u8x64 = sz_aes256_rounds_wide_icelake_(keystream_u8x64, wide_keys_vec);
        _mm512_storeu_si512(output + produced, _mm512_xor_si512(keystream_u8x64, _mm512_loadu_si512(text + produced)));
        counters_u8x64 = _mm512_add_epi32(counters_u8x64, step_four_u8x64);
    }

    if (produced != length) {
        sz_size_t const remaining = length - produced;
        __mmask64 const tail_m64 = sz_u64_mask_until_(remaining);
        __m512i keystream_u8x64 = _mm512_xor_si512(_mm512_shuffle_epi8(counters_u8x64, counter_swap_u8x64),
                                                   wide_keys_vec[0].zmm);
        keystream_u8x64 = sz_aes256_rounds_wide_icelake_(keystream_u8x64, wide_keys_vec);
        _mm512_mask_storeu_epi8(output + produced, tail_m64,
                                _mm512_xor_si512(keystream_u8x64, _mm512_maskz_loadu_epi8(tail_m64, text + produced)));
        counters_u8x64 = _mm512_add_epi32(
            counters_u8x64, _mm512_maskz_set1_epi32((__mmask16)0x8888u, (int)((remaining + SZ_AES_BLOCK_LENGTH - 1) /
                                                                              SZ_AES_BLOCK_LENGTH)));
    }
    return counters_u8x64;
}

SZ_API_COMPTIME void sz_aes256_ctr_xor_icelake(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                               sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length,
                                               sz_ptr_t output) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    sz_u32_t block_index = (sz_u32_t)(byte_offset / SZ_AES_BLOCK_LENGTH);
    sz_size_t within_block = (sz_size_t)(byte_offset % SZ_AES_BLOCK_LENGTH);
    sz_size_t produced = 0;

    if (length == 0) return;

    //  A seek that lands mid-block spends the leading bytes of one keystream block and nothing else.
    if (within_block != 0) {
        __m512i const counter_swap_u8x64 = _mm512_load_si512(sz_aes256_counter_swap_icelake_());
        __m128i const counter_u8x16 = sz_aes256_counter_block_icelake_(sz_aes256_counters_icelake_(nonce, block_index),
                                                                       counter_swap_u8x64);
        // A partial first block, entered mid-stream by a seeking caller. Shifting the keystream down to
        // the resume offset covers the run in three masked instructions.
        __m128i const keystream_u8x16 = sz_aes256_block_encrypt_icelake_(key, counter_u8x16);
        sz_size_t const head_bytes = sz_min_of_two(length, SZ_AES_BLOCK_LENGTH - within_block);
        __mmask16 const head_m16 = sz_u16_mask_until_(head_bytes);
        __m128i const shifted_keystream_u8x16 = _mm_maskz_permutexvar_epi8(
            head_m16, _mm_add_epi8(sz_aes256_lane_iota_icelake_(), _mm_set1_epi8((char)within_block)), keystream_u8x16);
        _mm_mask_storeu_epi8(output_bytes, head_m16,
                             _mm_xor_si128(_mm_maskz_loadu_epi8(head_m16, input_bytes), shifted_keystream_u8x16));
        produced += head_bytes, within_block += head_bytes;
        ++block_index;
    }

    if (produced != length) {
        sz_u512_vec_t wide_keys_vec[15];
        sz_aes256_round_keys_wide_icelake_(key, wide_keys_vec);
        sz_aes256_ctr_stride_icelake_(sz_aes256_counters_icelake_(nonce, block_index), wide_keys_vec,
                                      input_bytes + produced, output_bytes + produced, length - produced);
    }
}

#pragma endregion // Counter Mode

#pragma region Galois Hashing

/** @brief Reverses all sixteen bytes of each 128-bit lane, its own inverse. */
SZ_HELPER_INLINE sz_u8_t const *sz_ghash_byte_reverse_icelake_(void) {
    static sz_align_(64) sz_u8_t const reversal[64] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, //
                                                       15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, //
                                                       15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, //
                                                       15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    return &reversal[0];
}

/**
 *  @brief Reduces a carry-less product into the field the authentication tag is built over.
 *  @param product_low_u8x16 The low 128 bits of the schoolbook product.
 *  @param product_middle_u8x16 The two cross terms, already exclusive-ored together.
 *  @param product_high_u8x16 The high 128 bits of the schoolbook product.
 *  @return The product modulo `x^128 + x^7 + x^2 + x + 1`, byte reversed like its operands.
 *
 *  Galois/counter mode numbers the bits of a block backwards relative to the way `pclmulqdq` reads them, which
 *  leaves the 256-bit product one position low; the shift by one puts it back.
 */
SZ_HELPER_INLINE __m128i sz_ghash_reduce_icelake_(__m128i product_low_u8x16, __m128i product_middle_u8x16,
                                                  __m128i product_high_u8x16) {
    __m128i const polynomial_u8x16 = _mm_set_epi64x((sz_i64_t)0xC200000000000000ull, 0);
    __m128i low_u8x16 = _mm_xor_si128(product_low_u8x16, _mm_slli_si128(product_middle_u8x16, 8));
    __m128i high_u8x16 = _mm_xor_si128(product_high_u8x16, _mm_srli_si128(product_middle_u8x16, 8));
    __m128i const low_carry_u8x16 = _mm_srli_epi32(low_u8x16, 31);
    __m128i const high_carry_u8x16 = _mm_srli_epi32(high_u8x16, 31);
    __m128i folded_u8x16;

    low_u8x16 = _mm_or_si128(_mm_slli_epi32(low_u8x16, 1), _mm_slli_si128(low_carry_u8x16, 4));
    high_u8x16 = _mm_or_si128(_mm_slli_epi32(high_u8x16, 1), _mm_slli_si128(high_carry_u8x16, 4));
    high_u8x16 = _mm_or_si128(high_u8x16, _mm_srli_si128(low_carry_u8x16, 12));

    folded_u8x16 = _mm_clmulepi64_si128(low_u8x16, polynomial_u8x16, 0x10);
    low_u8x16 = _mm_xor_si128(_mm_shuffle_epi32(low_u8x16, 0x4E), folded_u8x16);
    folded_u8x16 = _mm_clmulepi64_si128(low_u8x16, polynomial_u8x16, 0x10);
    low_u8x16 = _mm_xor_si128(_mm_shuffle_epi32(low_u8x16, 0x4E), folded_u8x16);
    return _mm_xor_si128(high_u8x16, low_u8x16);
}

/**
 *  @brief Multiplies two byte-reversed field elements.
 *  @param first_operand_u8x16 One operand, byte reversed.
 *  @param second_operand_u8x16 The other operand, byte reversed.
 *  @return Their product, byte reversed.
 */
SZ_HELPER_INLINE __m128i sz_ghash_multiply_icelake_(__m128i first_operand_u8x16, __m128i second_operand_u8x16) {
    __m128i const product_low_u8x16 = _mm_clmulepi64_si128(first_operand_u8x16, second_operand_u8x16, 0x00);
    __m128i const product_high_u8x16 = _mm_clmulepi64_si128(first_operand_u8x16, second_operand_u8x16, 0x11);
    __m128i const product_middle_u8x16 = _mm_xor_si128(
        _mm_clmulepi64_si128(first_operand_u8x16, second_operand_u8x16, 0x10),
        _mm_clmulepi64_si128(first_operand_u8x16, second_operand_u8x16, 0x01));
    return sz_ghash_reduce_icelake_(product_low_u8x16, product_middle_u8x16, product_high_u8x16);
}

SZ_API_COMPTIME void sz_aes256_gcm_key_init_icelake(sz_aes256_gcm_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    __m128i const reverse_u8x16 = _mm512_castsi512_si128(_mm512_load_si512(sz_ghash_byte_reverse_icelake_()));
    __m128i subkey_u8x16, power_u8x16;
    sz_size_t power_index;

    sz_aes256_key_init_icelake(&key->block, secret);
    subkey_u8x16 = _mm_shuffle_epi8(sz_aes256_block_encrypt_icelake_(&key->block, _mm_setzero_si128()), reverse_u8x16);
    power_u8x16 = subkey_u8x16;
    for (power_index = 0; power_index != 8; ++power_index) {
        _mm_mask_storeu_epi8(key->powers + power_index * SZ_AES_BLOCK_LENGTH, (__mmask16)0xFFFFu,
                             _mm_shuffle_epi8(power_u8x16, reverse_u8x16));
        power_u8x16 = sz_ghash_multiply_icelake_(power_u8x16, subkey_u8x16);
    }
}

/** @brief Exclusive-ors the four 128-bit lanes of a register into one, which the reduction then closes. */
SZ_HELPER_INLINE __m128i sz_ghash_fold_lanes_icelake_(__m512i value_u8x64) {
    __m512i folded_u8x64 = _mm512_xor_si512(value_u8x64, _mm512_shuffle_i64x2(value_u8x64, value_u8x64, 0x4E));
    folded_u8x64 = _mm512_xor_si512(folded_u8x64, _mm512_shuffle_i64x2(folded_u8x64, folded_u8x64, 0xB1));
    return _mm512_castsi512_si128(folded_u8x64);
}

/**
 *  @brief Absorbs eight byte-reversed blocks and the running hash in one reduction.
 *  @param accumulator_u8x16 The running hash, byte reversed.
 *  @param blocks_low_u8x64 Blocks one through four, byte reversed.
 *  @param blocks_high_u8x64 Blocks five through eight, byte reversed.
 *  @param powers_high_u8x64 `H^8` through `H^5`, byte reversed, one per lane.
 *  @param powers_low_u8x64 `H^4` through `H^1`, byte reversed, one per lane.
 *  @param eighth_power_u8x16 `H^8` alone, byte reversed.
 *  @return The running hash after all eight blocks, byte reversed.
 *
 *  The hash is a chain, `Y = (Y ^ X) * H`, and a chain of reductions would run at the latency of one multiply
 *  per block.
 */
SZ_HELPER_INLINE __m128i sz_ghash_stage_icelake_(__m128i accumulator_u8x16, __m512i blocks_low_u8x64,
                                                 __m512i blocks_high_u8x64, __m512i powers_high_u8x64,
                                                 __m512i powers_low_u8x64, __m128i eighth_power_u8x16) {
    __m512i const product_low_u8x64 = _mm512_xor_si512(
        _mm512_clmulepi64_epi128(blocks_low_u8x64, powers_high_u8x64, 0x00),
        _mm512_clmulepi64_epi128(blocks_high_u8x64, powers_low_u8x64, 0x00));
    __m512i const product_high_u8x64 = _mm512_xor_si512(
        _mm512_clmulepi64_epi128(blocks_low_u8x64, powers_high_u8x64, 0x11),
        _mm512_clmulepi64_epi128(blocks_high_u8x64, powers_low_u8x64, 0x11));
    __m512i const product_middle_u8x64 = _mm512_xor_si512(
        _mm512_xor_si512(_mm512_clmulepi64_epi128(blocks_low_u8x64, powers_high_u8x64, 0x01),
                         _mm512_clmulepi64_epi128(blocks_low_u8x64, powers_high_u8x64, 0x10)),
        _mm512_xor_si512(_mm512_clmulepi64_epi128(blocks_high_u8x64, powers_low_u8x64, 0x01),
                         _mm512_clmulepi64_epi128(blocks_high_u8x64, powers_low_u8x64, 0x10)));
    __m128i const carried_low_u8x16 = _mm_clmulepi64_si128(accumulator_u8x16, eighth_power_u8x16, 0x00);
    __m128i const carried_high_u8x16 = _mm_clmulepi64_si128(accumulator_u8x16, eighth_power_u8x16, 0x11);
    __m128i const carried_middle_u8x16 = _mm_xor_si128(
        _mm_clmulepi64_si128(accumulator_u8x16, eighth_power_u8x16, 0x10),
        _mm_clmulepi64_si128(accumulator_u8x16, eighth_power_u8x16, 0x01));
    return sz_ghash_reduce_icelake_(
        _mm_xor_si128(sz_ghash_fold_lanes_icelake_(product_low_u8x64), carried_low_u8x16),
        _mm_xor_si128(sz_ghash_fold_lanes_icelake_(product_middle_u8x64), carried_middle_u8x16),
        _mm_xor_si128(sz_ghash_fold_lanes_icelake_(product_high_u8x64), carried_high_u8x16));
}

/**
 *  @brief Loads `H^8` through `H^1` descending and byte reversed, two lanes-worth per register.
 *  @param key The expanded key holding the powers ascending.
 *  @param powers_high_u8x64 Receives `H^8`, `H^7`, `H^6`, `H^5`.
 *  @param powers_low_u8x64 Receives `H^4`, `H^3`, `H^2`, `H^1`.
 */
SZ_HELPER_INLINE void sz_ghash_powers_icelake_(sz_aes256_gcm_key_t const *key, __m512i *powers_high_u8x64,
                                               __m512i *powers_low_u8x64) {
    __m512i const reverse_u8x64 = _mm512_load_si512(sz_ghash_byte_reverse_icelake_());
    __m512i const ascending_low_u8x64 = _mm512_loadu_si512(key->powers);
    __m512i const ascending_high_u8x64 = _mm512_loadu_si512(key->powers + 4 * SZ_AES_BLOCK_LENGTH);
    *powers_high_u8x64 = _mm512_shuffle_epi8(_mm512_shuffle_i64x2(ascending_high_u8x64, ascending_high_u8x64, 0x1B),
                                             reverse_u8x64);
    *powers_low_u8x64 = _mm512_shuffle_epi8(_mm512_shuffle_i64x2(ascending_low_u8x64, ascending_low_u8x64, 0x1B),
                                            reverse_u8x64);
}

/**
 *  @brief Absorbs whole blocks into a byte-reversed running hash.
 *  @param accumulator_u8x16 The running hash, byte reversed.
 *  @param blocks The bytes to absorb, `count * 16` of them.
 *  @param count Number of whole blocks.
 *  @param powers_high_u8x64 `H^8` through `H^5`, byte reversed, one per lane.
 *  @param powers_low_u8x64 `H^4` through `H^1`, byte reversed, one per lane.
 *  @return The running hash after every block, byte reversed.
 */
SZ_HELPER_INLINE __m128i sz_ghash_absorb_blocks_icelake_(__m128i accumulator_u8x16, sz_u8_t const *blocks,
                                                         sz_size_t count, __m512i powers_high_u8x64,
                                                         __m512i powers_low_u8x64) {
    __m512i const reverse_u8x64 = _mm512_load_si512(sz_ghash_byte_reverse_icelake_());
    __m128i const eighth_power_u8x16 = _mm512_castsi512_si128(powers_high_u8x64);
    __m128i const subkey_u8x16 = _mm512_extracti64x2_epi64(powers_low_u8x64, 3);
    sz_size_t absorbed = 0;

    for (; absorbed + 8 <= count; absorbed += 8) {
        __m512i const blocks_low_u8x64 = _mm512_shuffle_epi8(_mm512_loadu_si512(blocks + absorbed * 16), reverse_u8x64);
        __m512i const blocks_high_u8x64 = _mm512_shuffle_epi8(_mm512_loadu_si512(blocks + absorbed * 16 + 64),
                                                              reverse_u8x64);
        accumulator_u8x16 = sz_ghash_stage_icelake_(accumulator_u8x16, blocks_low_u8x64, blocks_high_u8x64,
                                                    powers_high_u8x64, powers_low_u8x64, eighth_power_u8x16);
    }

    for (; absorbed != count; ++absorbed) {
        __m128i const block_u8x16 = _mm_shuffle_epi8(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, blocks + absorbed * 16),
                                                     _mm512_castsi512_si128(reverse_u8x64));
        accumulator_u8x16 = sz_ghash_multiply_icelake_(_mm_xor_si128(accumulator_u8x16, block_u8x16), subkey_u8x16);
    }
    return accumulator_u8x16;
}

#pragma endregion // Galois Hashing

#pragma region Streaming Interface

/**
 *  @brief Overwrites a finished state so the key schedule it embeds does not outlive the call.
 *
 *  The size is known at compile time, so this is seven full-width stores and one masked tail rather than a
 *  length-driven loop.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_state_scrub_icelake_(sz_aes256_gcm_state_t *state) {
    sz_u8_t *const bytes = (sz_u8_t *)state;
    __m512i const zeros_u8x64 = _mm512_setzero_si512();
    sz_size_t offset = 0;
    for (; offset + 64 <= sizeof(*state); offset += 64) _mm512_storeu_si512(bytes + offset, zeros_u8x64);
    _mm512_mask_storeu_epi8(bytes + offset, sz_u64_mask_until_(sizeof(*state) - offset), zeros_u8x64);
    sz_keep_alive_(state);
}

/** @brief Compares two tags in constant time; `sz_true_k` when all sixteen bytes match. */
SZ_HELPER_INLINE sz_bool_t sz_aes256_tag_equal_icelake_(sz_u8_t const *first, sz_u8_t const *second) {
    __m128i const first_u8x16 = _mm_loadu_si128((__m128i const *)first);
    __m128i const second_u8x16 = _mm_loadu_si128((__m128i const *)second);
    __mmask16 const differing_m16 = _mm_cmpneq_epi8_mask(first_u8x16, second_u8x16);
    return differing_m16 == 0 ? sz_true_k : sz_false_k;
}

/** @brief Prepares the payload both directions share: counter block, tag mask and empty carries. */
SZ_HELPER_INLINE void sz_aes256_gcm_begin_icelake_(sz_aes256_gcm_state_t *state, sz_aes256_gcm_key_t const *key,
                                                   sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_u128_vec_t initial_vec;
    sz_size_t byte_index;

    state->key = *key;
    _mm_mask_storeu_epi8(state->accumulator, (__mmask16)0xFFFFu, _mm_setzero_si128());
    _mm_mask_storeu_epi8(state->partial, (__mmask16)0xFFFFu, _mm_setzero_si128());
    _mm_mask_storeu_epi8(state->keystream, (__mmask16)0xFFFFu, _mm_setzero_si128());

    //  With a twelve-byte nonce the initial counter block is the nonce followed by a one, and the value
    //  encrypted under it masks the finished hash. Data blocks start one past it.
    for (byte_index = 0; byte_index != 12; ++byte_index) initial_vec.u8s[byte_index] = nonce[byte_index];
    initial_vec.u8s[12] = 0, initial_vec.u8s[13] = 0, initial_vec.u8s[14] = 0, initial_vec.u8s[15] = 1;
    _mm_mask_storeu_epi8(state->tag_mask, (__mmask16)0xFFFFu,
                         sz_aes256_block_encrypt_icelake_(&state->key.block, initial_vec.xmm));
    _mm_mask_storeu_epi8(state->counter, (__mmask16)0xFFFFu, initial_vec.xmm);

    state->associated_length = 0;
    state->text_length = 0;
    state->buffered = 0;
    state->keystream_used = SZ_AES_BLOCK_LENGTH; // ? Forces the first message byte to derive a fresh block
}

/** @brief Absorbs associated data into the payload both directions share. */
SZ_HELPER_AUTO void sz_aes256_gcm_associate_icelake_(sz_aes256_gcm_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    __m128i const reverse_u8x16 = _mm512_castsi512_si128(_mm512_load_si512(sz_ghash_byte_reverse_icelake_()));
    __m512i powers_high_u8x64, powers_low_u8x64;
    __m128i accumulator_u8x16, subkey_u8x16;
    sz_size_t consumed = 0, byte_index;

    if (length == 0) return;
    sz_ghash_powers_icelake_(&state->key, &powers_high_u8x64, &powers_low_u8x64);
    subkey_u8x16 = _mm512_extracti64x2_epi64(powers_low_u8x64, 3);
    accumulator_u8x16 = _mm_shuffle_epi8(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->accumulator), reverse_u8x16);

    //  Associated data is hashed but never encrypted, so a partial block is completed in place.
    if (state->buffered != 0) {
        sz_size_t const wanted_bytes = SZ_AES_BLOCK_LENGTH - state->buffered;
        sz_size_t const taken_bytes = length < wanted_bytes ? length : wanted_bytes;
        for (byte_index = 0; byte_index != taken_bytes; ++byte_index)
            state->partial[state->buffered + byte_index] = input_bytes[byte_index];
        state->buffered = (sz_u8_t)(state->buffered + taken_bytes);
        state->associated_length += taken_bytes;
        consumed = taken_bytes;
        if (state->buffered == SZ_AES_BLOCK_LENGTH) {
            __m128i const block_u8x16 = _mm_shuffle_epi8(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->partial),
                                                         reverse_u8x16);
            accumulator_u8x16 = sz_ghash_multiply_icelake_(_mm_xor_si128(accumulator_u8x16, block_u8x16), subkey_u8x16);
            state->buffered = 0;
        }
    }

    if (state->buffered == 0 && length - consumed >= SZ_AES_BLOCK_LENGTH) {
        sz_size_t const blocks = (length - consumed) / SZ_AES_BLOCK_LENGTH;
        accumulator_u8x16 = sz_ghash_absorb_blocks_icelake_(accumulator_u8x16, input_bytes + consumed, blocks,
                                                            powers_high_u8x64, powers_low_u8x64);
        consumed += blocks * SZ_AES_BLOCK_LENGTH;
        state->associated_length += blocks * SZ_AES_BLOCK_LENGTH;
    }

    for (; consumed != length; ++consumed) {
        state->partial[state->buffered++] = input_bytes[consumed];
        ++state->associated_length;
    }
    _mm_mask_storeu_epi8(state->accumulator, (__mmask16)0xFFFFu, _mm_shuffle_epi8(accumulator_u8x16, reverse_u8x16));
}

/** @brief Absorbs whatever `partial` holds, zero padded to a full block, and empties it. */
SZ_HELPER_AUTO void sz_aes256_gcm_flush_partial_icelake_(sz_aes256_gcm_state_t *state) {
    __m128i reverse_u8x16, subkey_u8x16, padded_u8x16, accumulator_u8x16;
    if (state->buffered == 0) return;
    reverse_u8x16 = _mm512_castsi512_si128(_mm512_load_si512(sz_ghash_byte_reverse_icelake_()));
    subkey_u8x16 = _mm_shuffle_epi8(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->key.powers), reverse_u8x16);
    padded_u8x16 = _mm_shuffle_epi8(_mm_maskz_loadu_epi8(sz_u16_mask_until_(state->buffered), state->partial),
                                    reverse_u8x16);
    accumulator_u8x16 = _mm_shuffle_epi8(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->accumulator), reverse_u8x16);
    accumulator_u8x16 = sz_ghash_multiply_icelake_(_mm_xor_si128(accumulator_u8x16, padded_u8x16), subkey_u8x16);
    _mm_mask_storeu_epi8(state->accumulator, (__mmask16)0xFFFFu, _mm_shuffle_epi8(accumulator_u8x16, reverse_u8x16));
    state->buffered = 0;
}

/**
 *  @brief Transforms bytes one at a time, absorbing the ciphertext side into @p accumulator_u8x16.
 *  @param state The state, whose keystream block and hash block both advance byte by byte.
 *  @param accumulator_u8x16 The running hash, byte reversed, updated in place.
 *  @param subkey_u8x16 `H^1`, byte reversed.
 *  @param reverse_u8x16 The byte reversal pattern.
 *  @param text The bytes to transform.
 *  @param length Number of bytes.
 *  @param output Receives @p length transformed bytes.
 *  @param direction Which side of the transformation the hash absorbs.
 *
 *  Serves the two edges of a chunk: the keystream block a previous call left half spent, and the trailing
 *  bytes of this one that do not fill a block.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_bytes_icelake_(sz_aes256_gcm_state_t *state, __m128i *accumulator_u8x16,
                                                   __m128i subkey_u8x16, __m128i reverse_u8x16, sz_u8_t const *text,
                                                   sz_size_t length, sz_u8_t *output,
                                                   sz_aes256_gcm_direction_t direction) {
    __m128i const lane_iota_u8x16 = sz_aes256_lane_iota_icelake_();
    sz_size_t produced = 0;
    while (produced != length) {
        if (state->keystream_used == SZ_AES_BLOCK_LENGTH) {
            sz_u128_vec_t counter_vec;
            counter_vec.xmm = _mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->counter);
            counter_vec.u32s[3] = sz_u32_bytes_reverse(sz_u32_bytes_reverse(counter_vec.u32s[3]) + 1u);
            _mm_mask_storeu_epi8(state->counter, (__mmask16)0xFFFFu, counter_vec.xmm);
            _mm_mask_storeu_epi8(state->keystream, (__mmask16)0xFFFFu,
                                 sz_aes256_block_encrypt_icelake_(&state->key.block, counter_vec.xmm));
            state->keystream_used = 0;
        }

        // The run is what fits before either sixteen-byte rhythm turns over, transformed and staged in
        // registers rather than a byte at a time.
        sz_size_t const keystream_left = SZ_AES_BLOCK_LENGTH - state->keystream_used;
        sz_size_t const block_left = SZ_AES_BLOCK_LENGTH - state->buffered;
        sz_size_t const remaining = length - produced;
        sz_size_t run = remaining < keystream_left ? remaining : keystream_left;
        if (block_left < run) run = block_left;
        __mmask16 const run_m16 = sz_u16_mask_until_(run);

        __m128i const spent_keystream_u8x16 = _mm_maskz_permutexvar_epi8(
            run_m16, _mm_add_epi8(lane_iota_u8x16, _mm_set1_epi8((char)state->keystream_used)),
            _mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->keystream));
        __m128i const plaintext_u8x16 = _mm_maskz_loadu_epi8(run_m16, text + produced);
        __m128i const transformed_u8x16 = _mm_xor_si128(plaintext_u8x16, spent_keystream_u8x16);
        _mm_mask_storeu_epi8(output + produced, run_m16, transformed_u8x16);

        // Captured before the store because a caller may pass one pointer for both sides, then slid up
        // to the offset the pending block already holds.
        __m128i const ciphertext_u8x16 = direction == sz_aes256_gcm_decrypting_k ? plaintext_u8x16 : transformed_u8x16;
        __m128i const staged_u8x16 = _mm_permutexvar_epi8(
            _mm_sub_epi8(lane_iota_u8x16, _mm_set1_epi8((char)state->buffered)), ciphertext_u8x16);
        _mm_mask_storeu_epi8(state->partial, (__mmask16)(run_m16 << state->buffered), staged_u8x16);

        state->buffered = (sz_u8_t)(state->buffered + run);
        state->keystream_used = (sz_u8_t)(state->keystream_used + run);
        state->text_length += run;
        produced += run;

        if (state->buffered == SZ_AES_BLOCK_LENGTH) {
            __m128i const block_u8x16 = _mm_shuffle_epi8(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->partial),
                                                         reverse_u8x16);
            *accumulator_u8x16 = sz_ghash_multiply_icelake_(_mm_xor_si128(*accumulator_u8x16, block_u8x16),
                                                            subkey_u8x16);
            state->buffered = 0;
        }
    }
}

/**
 *  @brief Encrypts whole blocks and absorbs their ciphertext, the hash trailing the cipher by one stage.
 *  @param counters_out_u8x64 Four consecutive counter blocks, advanced past everything the call consumed.
 *  @param accumulator_u8x16 The running hash, byte reversed, updated in place.
 *  @param wide_keys_vec The fifteen round keys, each broadcast across four lanes.
 *  @param powers_high_u8x64 `H^8` through `H^5`, byte reversed, one per lane.
 *  @param powers_low_u8x64 `H^4` through `H^1`, byte reversed, one per lane.
 *  @param cipher_m8 All ones when the hash absorbs the output, zero when it absorbs the input.
 *  @param text The input blocks.
 *  @param output Receives the transformed blocks; may equal @p text.
 *  @param blocks Number of whole blocks, at least one.
 *
 *  The hash of a stage cannot begin until its ciphertext exists, so it is held one stage back and the
 *  carry-less multiplies then sit in the shadow of the next stage's substitution rounds, which occupy a
 *  different port and would otherwise stall on the reduction's latency.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_stride_icelake_(__m512i *counters_out_u8x64, __m128i *accumulator_u8x16,
                                                    sz_u512_vec_t const *wide_keys_vec, __m512i powers_high_u8x64,
                                                    __m512i powers_low_u8x64, __mmask8 cipher_m8, sz_u8_t const *text,
                                                    sz_u8_t *output, sz_size_t blocks) {
    __m512i const counter_swap_u8x64 = _mm512_load_si512(sz_aes256_counter_swap_icelake_());
    __m512i const reverse_u8x64 = _mm512_load_si512(sz_ghash_byte_reverse_icelake_());
    __m512i const step_four_u8x64 = _mm512_maskz_set1_epi32((__mmask16)0x8888u, 4);
    __m512i const step_eight_u8x64 = _mm512_maskz_set1_epi32((__mmask16)0x8888u, 8);
    __m128i const eighth_power_u8x16 = _mm512_castsi512_si128(powers_high_u8x64);
    sz_size_t const stages = blocks / 8;
    sz_size_t stage_index, produced = 0;

    __m512i first_counter_u8x64 = *counters_out_u8x64;
    __m512i second_counter_u8x64 = _mm512_add_epi32(first_counter_u8x64, step_four_u8x64);
    __m512i first_pending_u8x64 = _mm512_setzero_si512(), second_pending_u8x64 = _mm512_setzero_si512();

    for (stage_index = 0; stage_index != stages; ++stage_index) {
        __m512i first_keystream_u8x64 = _mm512_xor_si512(_mm512_shuffle_epi8(first_counter_u8x64, counter_swap_u8x64),
                                                         wide_keys_vec[0].zmm);
        __m512i second_keystream_u8x64 = _mm512_xor_si512(_mm512_shuffle_epi8(second_counter_u8x64, counter_swap_u8x64),
                                                          wide_keys_vec[0].zmm);
        // The previous stage's hash lands between the first round key and the rounds themselves, so the
        // carry-less multipliers and the cipher units issue against each other rather than in sequence.
        if (stage_index != 0)
            *accumulator_u8x16 = sz_ghash_stage_icelake_(*accumulator_u8x16, first_pending_u8x64, second_pending_u8x64,
                                                         powers_high_u8x64, powers_low_u8x64, eighth_power_u8x16);
        first_keystream_u8x64 = sz_aes256_rounds_wide_icelake_(first_keystream_u8x64, wide_keys_vec);
        second_keystream_u8x64 = sz_aes256_rounds_wide_icelake_(second_keystream_u8x64, wide_keys_vec);

        __m512i const first_loaded_u8x64 = _mm512_loadu_si512(text + produced + 0);
        __m512i const first_stored_u8x64 = _mm512_xor_si512(first_loaded_u8x64, first_keystream_u8x64);
        _mm512_storeu_si512(output + produced + 0, first_stored_u8x64);
        first_pending_u8x64 = _mm512_shuffle_epi8(
            _mm512_mask_blend_epi64(cipher_m8, first_loaded_u8x64, first_stored_u8x64), reverse_u8x64);
        first_counter_u8x64 = _mm512_add_epi32(first_counter_u8x64, step_eight_u8x64);

        __m512i const second_loaded_u8x64 = _mm512_loadu_si512(text + produced + 64);
        __m512i const second_stored_u8x64 = _mm512_xor_si512(second_loaded_u8x64, second_keystream_u8x64);
        _mm512_storeu_si512(output + produced + 64, second_stored_u8x64);
        second_pending_u8x64 = _mm512_shuffle_epi8(
            _mm512_mask_blend_epi64(cipher_m8, second_loaded_u8x64, second_stored_u8x64), reverse_u8x64);
        second_counter_u8x64 = _mm512_add_epi32(second_counter_u8x64, step_eight_u8x64);
        produced += 128;
    }
    if (stages != 0)
        *accumulator_u8x16 = sz_ghash_stage_icelake_(*accumulator_u8x16, first_pending_u8x64, second_pending_u8x64,
                                                     powers_high_u8x64, powers_low_u8x64, eighth_power_u8x16);

    if (blocks % 8 != 0) {
        sz_size_t const tail_blocks = blocks % 8;
        sz_size_t const tail_bytes = tail_blocks * SZ_AES_BLOCK_LENGTH;
        sz_align_(64) sz_u8_t staged_bytes[128];
        __mmask64 const first_m64 = sz_u64_clamp_mask_until_(tail_bytes);
        __m512i first_keystream_u8x64 = _mm512_xor_si512(_mm512_shuffle_epi8(first_counter_u8x64, counter_swap_u8x64),
                                                         wide_keys_vec[0].zmm);
        first_keystream_u8x64 = sz_aes256_rounds_wide_icelake_(first_keystream_u8x64, wide_keys_vec);
        {
            __m512i const loaded_u8x64 = _mm512_maskz_loadu_epi8(first_m64, text + produced);
            __m512i const stored_u8x64 = _mm512_xor_si512(loaded_u8x64, first_keystream_u8x64);
            _mm512_mask_storeu_epi8(output + produced, first_m64, stored_u8x64);
            _mm512_store_si512(staged_bytes, _mm512_mask_blend_epi64(cipher_m8, loaded_u8x64, stored_u8x64));
        }
        // A tail of five blocks or more spills into a second group; four or fewer never does, and
        // encrypting a group that stores nothing would cost thirteen rounds for no bytes.
        if (tail_bytes > 64) {
            __mmask64 const second_m64 = sz_u64_clamp_mask_until_(tail_bytes - 64);
            __m512i second_keystream_u8x64 = _mm512_xor_si512(
                _mm512_shuffle_epi8(second_counter_u8x64, counter_swap_u8x64), wide_keys_vec[0].zmm);
            second_keystream_u8x64 = sz_aes256_rounds_wide_icelake_(second_keystream_u8x64, wide_keys_vec);
            __m512i const loaded_u8x64 = _mm512_maskz_loadu_epi8(second_m64, text + produced + 64);
            __m512i const stored_u8x64 = _mm512_xor_si512(loaded_u8x64, second_keystream_u8x64);
            _mm512_mask_storeu_epi8(output + produced + 64, second_m64, stored_u8x64);
            _mm512_store_si512(staged_bytes + 64, _mm512_mask_blend_epi64(cipher_m8, loaded_u8x64, stored_u8x64));
        }
        *accumulator_u8x16 = sz_ghash_absorb_blocks_icelake_(*accumulator_u8x16, staged_bytes, tail_blocks,
                                                             powers_high_u8x64, powers_low_u8x64);
        first_counter_u8x64 = _mm512_add_epi32(first_counter_u8x64,
                                               _mm512_maskz_set1_epi32((__mmask16)0x8888u, (int)tail_blocks));
    }
    *counters_out_u8x64 = first_counter_u8x64;
}

/**
 *  @brief Transforms a chunk and absorbs its ciphertext, whichever side of the call that is.
 *  @param state The state.
 *  @param text The chunk to transform.
 *  @param length Bytes in the chunk.
 *  @param output Receives the transformed bytes.
 *  @param direction Which side of the transformation the hash absorbs.
 *
 *  Two sixteen-byte rhythms run underneath a caller's arbitrary chunk sizes, and neither may restart at a
 *  chunk boundary, so the wide path is entered only once both stand at a block boundary and left with the same
 *  property.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_transform_icelake_(sz_aes256_gcm_state_t *state, sz_cptr_t text, sz_size_t length,
                                                       sz_ptr_t output, sz_aes256_gcm_direction_t direction) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    __m512i const counter_swap_u8x64 = _mm512_load_si512(sz_aes256_counter_swap_icelake_());
    __m128i const reverse_u8x16 = _mm512_castsi512_si128(_mm512_load_si512(sz_ghash_byte_reverse_icelake_()));
    __m512i powers_high_u8x64, powers_low_u8x64;
    __m128i accumulator_u8x16, subkey_u8x16;
    sz_size_t produced, head_bytes;

    //  Associated data ends the moment the first message byte arrives, and its tail needs padding.
    if (state->text_length == 0 && length != 0) sz_aes256_gcm_flush_partial_icelake_(state);
    if (length == 0) return;

    sz_ghash_powers_icelake_(&state->key, &powers_high_u8x64, &powers_low_u8x64);
    subkey_u8x16 = _mm512_extracti64x2_epi64(powers_low_u8x64, 3);
    accumulator_u8x16 = _mm_shuffle_epi8(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->accumulator), reverse_u8x16);

    head_bytes = (sz_size_t)(SZ_AES_BLOCK_LENGTH - state->keystream_used);
    if (head_bytes > length) head_bytes = length;
    sz_aes256_gcm_bytes_icelake_(state, &accumulator_u8x16, subkey_u8x16, reverse_u8x16, input_bytes, head_bytes,
                                 output_bytes, direction);
    produced = head_bytes;

    if (state->keystream_used == SZ_AES_BLOCK_LENGTH && state->buffered == 0 &&
        length - produced >= SZ_AES_BLOCK_LENGTH) {
        sz_size_t const blocks = (length - produced) / SZ_AES_BLOCK_LENGTH;
        __mmask8 const cipher_m8 = direction == sz_aes256_gcm_decrypting_k ? (__mmask8)0x00u : (__mmask8)0xFFu;
        sz_u512_vec_t wide_keys_vec[15];
        __m512i counters_u8x64;
        sz_aes256_round_keys_wide_icelake_(&state->key.block, wide_keys_vec);
        counters_u8x64 = _mm512_add_epi32(
            _mm512_shuffle_epi8(_mm512_broadcast_i32x4(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->counter)),
                                counter_swap_u8x64),
            _mm512_set_epi32(4, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0));
        sz_aes256_gcm_stride_icelake_(&counters_u8x64, &accumulator_u8x16, wide_keys_vec, powers_high_u8x64,
                                      powers_low_u8x64, cipher_m8, input_bytes + produced, output_bytes + produced,
                                      blocks);
        produced += blocks * SZ_AES_BLOCK_LENGTH;
        state->text_length += blocks * SZ_AES_BLOCK_LENGTH;
        _mm_mask_storeu_epi8(
            state->counter, (__mmask16)0xFFFFu,
            sz_aes256_counter_block_icelake_(
                _mm512_sub_epi32(counters_u8x64, _mm512_maskz_set1_epi32((__mmask16)0x8888u, 1)), counter_swap_u8x64));
    }

    sz_aes256_gcm_bytes_icelake_(state, &accumulator_u8x16, subkey_u8x16, reverse_u8x16, input_bytes + produced,
                                 length - produced, output_bytes + produced, direction);
    _mm_mask_storeu_epi8(state->accumulator, (__mmask16)0xFFFFu, _mm_shuffle_epi8(accumulator_u8x16, reverse_u8x16));
}

/**
 *  @brief Closes the hash over whatever is still pending and masks it into the authentication tag.
 *  @param state The state, left untouched.
 *  @param tag Receives the sixteen tag bytes.
 */
SZ_HELPER_AUTO void sz_aes256_gcm_digest_icelake_(sz_aes256_gcm_state_t const *state, sz_u8_t tag[sz_at_least_(16)]) {
    __m128i const reverse_u8x16 = _mm512_castsi512_si128(_mm512_load_si512(sz_ghash_byte_reverse_icelake_()));
    __m128i const subkey_u8x16 = _mm_shuffle_epi8(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->key.powers),
                                                  reverse_u8x16);
    __m128i accumulator_u8x16 = _mm_shuffle_epi8(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->accumulator),
                                                 reverse_u8x16);
    sz_u128_vec_t lengths_vec;

    //  Whatever is still pending gets padded here: an associated-data tail when the message was empty,
    //  or the message's own trailing ciphertext bytes otherwise.
    if (state->buffered != 0) {
        __m128i const padded_u8x16 = _mm_shuffle_epi8(
            _mm_maskz_loadu_epi8(sz_u16_mask_until_(state->buffered), state->partial), reverse_u8x16);
        accumulator_u8x16 = sz_ghash_multiply_icelake_(_mm_xor_si128(accumulator_u8x16, padded_u8x16), subkey_u8x16);
    }

    sz_aes256_gcm_lengths_serial_(&lengths_vec, state->associated_length, state->text_length);
    accumulator_u8x16 = sz_ghash_multiply_icelake_(
        _mm_xor_si128(accumulator_u8x16, _mm_shuffle_epi8(lengths_vec.xmm, reverse_u8x16)), subkey_u8x16);
    _mm_mask_storeu_epi8(tag, (__mmask16)0xFFFFu,
                         _mm_xor_si128(_mm_shuffle_epi8(accumulator_u8x16, reverse_u8x16),
                                       _mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->tag_mask)));
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_init_icelake(sz_aes256_gcm_encryptor_t *encryptor,
                                                          sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_aes256_gcm_begin_icelake_(&encryptor->state, key, nonce);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_associate_icelake(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                               sz_size_t length) {
    sz_aes256_gcm_associate_icelake_(&encryptor->state, text, length);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_update_icelake(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                            sz_size_t length, sz_ptr_t output) {
    sz_aes256_gcm_transform_icelake_(&encryptor->state, text, length, output, sz_aes256_gcm_encrypting_k);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_digest_icelake(sz_aes256_gcm_encryptor_t const *encryptor,
                                                            sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_digest_icelake_(&encryptor->state, tag);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_init_icelake(sz_aes256_gcm_decryptor_t *decryptor,
                                                          sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_aes256_gcm_begin_icelake_(&decryptor->state, key, nonce);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_associate_icelake(sz_aes256_gcm_decryptor_t *decryptor, sz_cptr_t text,
                                                               sz_size_t length) {
    sz_aes256_gcm_associate_icelake_(&decryptor->state, text, length);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_update_unverified_icelake(sz_aes256_gcm_decryptor_t *decryptor,
                                                                       sz_cptr_t text, sz_size_t length,
                                                                       sz_ptr_t output) {
    sz_aes256_gcm_transform_icelake_(&decryptor->state, text, length, output, sz_aes256_gcm_decrypting_k);
}

SZ_API_COMPTIME sz_status_t sz_aes256_gcm_decryptor_verify_icelake(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                   sz_u8_t const tag[sz_at_least_(16)]) {
    sz_u8_t expected[SZ_AES_BLOCK_LENGTH];
    sz_aes256_gcm_digest_icelake_(&decryptor->state, expected);
    return sz_aes256_tag_equal_icelake_(expected, tag) == sz_true_k ? sz_success_k : sz_authentication_failed_k;
}

#pragma endregion // Streaming Interface

#pragma region One Shot Interface

SZ_API_COMPTIME void sz_aes256_gcm_encrypt_icelake(sz_aes256_gcm_key_t const *key,
                                                   sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                   sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                   sz_ptr_t output, sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_encryptor_t encryptor;
    sz_aes256_gcm_encryptor_init_icelake(&encryptor, key, nonce);
    if (associated_length) sz_aes256_gcm_encryptor_associate_icelake(&encryptor, associated, associated_length);
    sz_aes256_gcm_encryptor_update_icelake(&encryptor, text, length, output);
    sz_aes256_gcm_encryptor_digest_icelake(&encryptor, tag);
    sz_aes256_gcm_state_scrub_icelake_(&encryptor.state);
}

SZ_API_COMPTIME sz_status_t sz_aes256_gcm_decrypt_icelake(sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                          sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                          sz_ptr_t output, sz_u8_t const tag[sz_at_least_(16)]) {
    sz_aes256_gcm_decryptor_t decryptor;
    sz_status_t verdict;
    sz_aes256_gcm_decryptor_init_icelake(&decryptor, key, nonce);
    if (associated_length) sz_aes256_gcm_decryptor_associate_icelake(&decryptor, associated, associated_length);
    sz_aes256_gcm_decryptor_update_unverified_icelake(&decryptor, text, length, output);
    verdict = sz_aes256_gcm_decryptor_verify_icelake(&decryptor, tag);
    sz_aes256_gcm_state_scrub_icelake_(&decryptor.state);

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
#endif // SZ_USE_ICELAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_CIPHER_ICELAKE_H_
