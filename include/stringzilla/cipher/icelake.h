/**
 *  @brief Ice Lake (AVX-512 + VAES + VPCLMULQDQ) backend for AES-256 encryption in counter and Galois/counter modes.
 *  @file include/stringzilla/cipher/icelake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/cipher.h
 *
 *  The fourteen AES rounds are written out here rather than driven by a loop over the round index, and
 *  the block groups are named registers rather than an indexed array. Both are deliberate. A round loop
 *  leaves the unrolling to the optimizer, which only does it reliably at `-O3`: GCC 15 emitted 35 AES
 *  round instructions across this file at `-O2` against 390 at `-O3`, because the thirteen-iteration
 *  loop body exceeds the completely-peel budget at the lower level. An indexed group array then spills
 *  to the stack instead of staying in `zmm` registers, so every round reloads its operands.
 *
 *  The cost was 2.1x to 2.9x of counter-mode throughput on a Sapphire Rapids core at `-O2`, and that
 *  matters here because `SZ_API_COMPTIME` kernels are `static inline` and compile into whichever
 *  translation unit consumes them, at whatever flags that consumer was built with. Writing the rounds
 *  out costs fourteen lines and makes the kernel's throughput a property of the code rather than of the
 *  caller's build. The `-O3` numbers were unchanged by the rewrite, which is the evidence that this is
 *  about removing a dependency on the optimizer and not about a new algorithm.
 */
#ifndef STRINGZILLA_CIPHER_ICELAKE_H_
#define STRINGZILLA_CIPHER_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/cipher/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_ICELAKE
#if defined(__clang__)
#pragma clang attribute push(                                                                              \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,bmi,bmi2,aes,vaes,pclmul,vpclmulqdq"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx,avx512f,avx512vl,avx512bw,avx512dq,bmi,bmi2,aes,vaes,pclmul,vpclmulqdq")
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
 *  @param previous_xmm The four schedule words eight positions back.
 *  @param assisted_xmm The substituted word, already broadcast across all four lanes.
 *  @return The next four schedule words.
 *
 *  FIPS 197 writes `w[i] = w[i - 8] ^ temp` with `temp` carried forward through the quadruple, which is
 *  the cumulative exclusive-or that three byte-wise doublings of `_mm_slli_si128` produce.
 */
SZ_HELPER_INLINE __m128i sz_aes256_key_fold_icelake_(__m128i previous_xmm, __m128i assisted_xmm) {
    previous_xmm = _mm_xor_si128(previous_xmm, _mm_slli_si128(previous_xmm, 4));
    previous_xmm = _mm_xor_si128(previous_xmm, _mm_slli_si128(previous_xmm, 4));
    previous_xmm = _mm_xor_si128(previous_xmm, _mm_slli_si128(previous_xmm, 4));
    return _mm_xor_si128(previous_xmm, assisted_xmm);
}

SZ_API_COMPTIME void sz_aes256_key_init_icelake(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    __m128i secret_low_words_xmm = _mm_loadu_epi8(secret);
    __m128i secret_high_words_xmm = _mm_loadu_epi8(secret + 16);

    //  `AESKEYGENASSIST` places `RotWord(SubWord(w)) ^ RCON` in the top word and `SubWord(w)` in the third,
    //  so a broadcast of one or the other serves the two shapes the AES-256 schedule alternates between.
    _mm_storeu_epi32(key->round_keys + 0, secret_low_words_xmm);
    _mm_storeu_epi32(key->round_keys + 4, secret_high_words_xmm);
    secret_low_words_xmm = sz_aes256_key_fold_icelake_(
        secret_low_words_xmm, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_high_words_xmm, 0x01), 0xFF));
    _mm_storeu_epi32(key->round_keys + 8, secret_low_words_xmm);
    secret_high_words_xmm = sz_aes256_key_fold_icelake_(
        secret_high_words_xmm, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_low_words_xmm, 0x00), 0xAA));
    _mm_storeu_epi32(key->round_keys + 12, secret_high_words_xmm);
    secret_low_words_xmm = sz_aes256_key_fold_icelake_(
        secret_low_words_xmm, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_high_words_xmm, 0x02), 0xFF));
    _mm_storeu_epi32(key->round_keys + 16, secret_low_words_xmm);
    secret_high_words_xmm = sz_aes256_key_fold_icelake_(
        secret_high_words_xmm, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_low_words_xmm, 0x00), 0xAA));
    _mm_storeu_epi32(key->round_keys + 20, secret_high_words_xmm);
    secret_low_words_xmm = sz_aes256_key_fold_icelake_(
        secret_low_words_xmm, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_high_words_xmm, 0x04), 0xFF));
    _mm_storeu_epi32(key->round_keys + 24, secret_low_words_xmm);
    secret_high_words_xmm = sz_aes256_key_fold_icelake_(
        secret_high_words_xmm, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_low_words_xmm, 0x00), 0xAA));
    _mm_storeu_epi32(key->round_keys + 28, secret_high_words_xmm);
    secret_low_words_xmm = sz_aes256_key_fold_icelake_(
        secret_low_words_xmm, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_high_words_xmm, 0x08), 0xFF));
    _mm_storeu_epi32(key->round_keys + 32, secret_low_words_xmm);
    secret_high_words_xmm = sz_aes256_key_fold_icelake_(
        secret_high_words_xmm, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_low_words_xmm, 0x00), 0xAA));
    _mm_storeu_epi32(key->round_keys + 36, secret_high_words_xmm);
    secret_low_words_xmm = sz_aes256_key_fold_icelake_(
        secret_low_words_xmm, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_high_words_xmm, 0x10), 0xFF));
    _mm_storeu_epi32(key->round_keys + 40, secret_low_words_xmm);
    secret_high_words_xmm = sz_aes256_key_fold_icelake_(
        secret_high_words_xmm, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_low_words_xmm, 0x00), 0xAA));
    _mm_storeu_epi32(key->round_keys + 44, secret_high_words_xmm);
    secret_low_words_xmm = sz_aes256_key_fold_icelake_(
        secret_low_words_xmm, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_high_words_xmm, 0x20), 0xFF));
    _mm_storeu_epi32(key->round_keys + 48, secret_low_words_xmm);
    secret_high_words_xmm = sz_aes256_key_fold_icelake_(
        secret_high_words_xmm, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_low_words_xmm, 0x00), 0xAA));
    _mm_storeu_epi32(key->round_keys + 52, secret_high_words_xmm);
    secret_low_words_xmm = sz_aes256_key_fold_icelake_(
        secret_low_words_xmm, _mm_shuffle_epi32(_mm_aeskeygenassist_si128(secret_high_words_xmm, 0x40), 0xFF));
    _mm_storeu_epi32(key->round_keys + 56, secret_low_words_xmm);
}

#pragma endregion // Key Schedule

#pragma region Counter Mode

/**
 *  @brief Encrypts one 16-byte block with the expanded schedule.
 *  @param key The expanded schedule.
 *  @param block_xmm The plaintext block.
 *  @return The ciphertext block.
 */
SZ_HELPER_INLINE __m128i sz_aes256_block_encrypt_icelake_(sz_aes256_key_t const *key, __m128i block_xmm) {
    block_xmm = _mm_xor_si128(block_xmm, _mm_loadu_epi32(key->round_keys + 0));
    block_xmm = _mm_aesenc_si128(block_xmm, _mm_loadu_epi32(key->round_keys + 4));
    block_xmm = _mm_aesenc_si128(block_xmm, _mm_loadu_epi32(key->round_keys + 8));
    block_xmm = _mm_aesenc_si128(block_xmm, _mm_loadu_epi32(key->round_keys + 12));
    block_xmm = _mm_aesenc_si128(block_xmm, _mm_loadu_epi32(key->round_keys + 16));
    block_xmm = _mm_aesenc_si128(block_xmm, _mm_loadu_epi32(key->round_keys + 20));
    block_xmm = _mm_aesenc_si128(block_xmm, _mm_loadu_epi32(key->round_keys + 24));
    block_xmm = _mm_aesenc_si128(block_xmm, _mm_loadu_epi32(key->round_keys + 28));
    block_xmm = _mm_aesenc_si128(block_xmm, _mm_loadu_epi32(key->round_keys + 32));
    block_xmm = _mm_aesenc_si128(block_xmm, _mm_loadu_epi32(key->round_keys + 36));
    block_xmm = _mm_aesenc_si128(block_xmm, _mm_loadu_epi32(key->round_keys + 40));
    block_xmm = _mm_aesenc_si128(block_xmm, _mm_loadu_epi32(key->round_keys + 44));
    block_xmm = _mm_aesenc_si128(block_xmm, _mm_loadu_epi32(key->round_keys + 48));
    block_xmm = _mm_aesenc_si128(block_xmm, _mm_loadu_epi32(key->round_keys + 52));
    return _mm_aesenclast_si128(block_xmm, _mm_loadu_epi32(key->round_keys + 56));
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
 *  The rounds are written out rather than looped. A `for` loop over the round index leaves the
 *  unrolling to the optimizer, and the optimizer only does it reliably at `-O3`: GCC 15 emits 35 AES
 *  round instructions across this file at `-O2` against 390 at `-O3`, so a looped kernel makes its own
 *  throughput a property of the caller's build flags rather than of the code.
 */
SZ_HELPER_INLINE __m512i sz_aes256_rounds_wide_icelake_(__m512i block_zmm, sz_u512_vec_t const *wide_keys_vec) {
    block_zmm = _mm512_aesenc_epi128(block_zmm, wide_keys_vec[1].zmm);
    block_zmm = _mm512_aesenc_epi128(block_zmm, wide_keys_vec[2].zmm);
    block_zmm = _mm512_aesenc_epi128(block_zmm, wide_keys_vec[3].zmm);
    block_zmm = _mm512_aesenc_epi128(block_zmm, wide_keys_vec[4].zmm);
    block_zmm = _mm512_aesenc_epi128(block_zmm, wide_keys_vec[5].zmm);
    block_zmm = _mm512_aesenc_epi128(block_zmm, wide_keys_vec[6].zmm);
    block_zmm = _mm512_aesenc_epi128(block_zmm, wide_keys_vec[7].zmm);
    block_zmm = _mm512_aesenc_epi128(block_zmm, wide_keys_vec[8].zmm);
    block_zmm = _mm512_aesenc_epi128(block_zmm, wide_keys_vec[9].zmm);
    block_zmm = _mm512_aesenc_epi128(block_zmm, wide_keys_vec[10].zmm);
    block_zmm = _mm512_aesenc_epi128(block_zmm, wide_keys_vec[11].zmm);
    block_zmm = _mm512_aesenc_epi128(block_zmm, wide_keys_vec[12].zmm);
    block_zmm = _mm512_aesenc_epi128(block_zmm, wide_keys_vec[13].zmm);
    return _mm512_aesenclast_epi128(block_zmm, wide_keys_vec[14].zmm);
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
    sz_u128_vec_t nonce_vec;
    sz_size_t byte_index;
    for (byte_index = 0; byte_index != 12; ++byte_index) nonce_vec.u8s[byte_index] = nonce[byte_index];
    nonce_vec.u32s[3] = 0;
    return _mm512_add_epi32(_mm512_broadcast_i32x4(nonce_vec.xmm),
                            _mm512_add_epi32(_mm512_maskz_set1_epi32((__mmask16)0x8888u, (int)block_index),
                                             _mm512_set_epi32(3, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)));
}

/**
 *  @brief Restores the big-endian counter block held in the first lane of a little-endian counter register.
 *  @param counters Four counter blocks with their trailing field little-endian.
 *  @param counter_swap The trailing-field reversal pattern.
 *  @return The block NIST's counter mode feeds to the cipher.
 */
SZ_HELPER_INLINE __m128i sz_aes256_counter_block_icelake_(__m512i counters, __m512i counter_swap) {
    return _mm512_castsi512_si128(_mm512_shuffle_epi8(counters, counter_swap));
}

/**
 *  @brief Exclusive-ors @p length bytes against the keystream, starting on a block boundary.
 *  @param counters Four consecutive counter blocks, the trailing field little-endian.
 *  @param wide_keys_vec The fifteen round keys, each broadcast across four lanes.
 *  @param text The input bytes.
 *  @param output Receives @p length bytes; may equal @p text.
 *  @param length Bytes to transform, not necessarily a multiple of the block length.
 *  @return The counters advanced past every block the call consumed.
 *
 *  Sixteen blocks in flight is what saturates the two AES ports; the 64-byte and masked passes below it
 *  exist only to land the tail, and a caller reaching them has already run out of work to hide latency in.
 */
SZ_HELPER_INLINE __m512i sz_aes256_ctr_stride_icelake_(__m512i counters, sz_u512_vec_t const *wide_keys_vec,
                                                       sz_u8_t const *text, sz_u8_t *output, sz_size_t length) {
    __m512i const counter_swap = _mm512_load_si512(sz_aes256_counter_swap_icelake_());
    __m512i const step_four = _mm512_maskz_set1_epi32((__mmask16)0x8888u, 4);
    __m512i const step_sixteen = _mm512_maskz_set1_epi32((__mmask16)0x8888u, 16);
    sz_size_t produced = 0;

    if (length >= 256) {
        __m512i first_group = counters;
        __m512i second_group = _mm512_add_epi32(first_group, step_four);
        __m512i third_group = _mm512_add_epi32(second_group, step_four);
        __m512i fourth_group = _mm512_add_epi32(third_group, step_four);
        for (; produced + 256 <= length; produced += 256) {
            // Four groups stay in named registers rather than an indexed array, so the four independent
            // round chains interleave in the scheduler instead of spilling to the stack.
            __m512i first_keystream = _mm512_xor_si512(_mm512_shuffle_epi8(first_group, counter_swap),
                                                       wide_keys_vec[0].zmm);
            __m512i second_keystream = _mm512_xor_si512(_mm512_shuffle_epi8(second_group, counter_swap),
                                                        wide_keys_vec[0].zmm);
            __m512i third_keystream = _mm512_xor_si512(_mm512_shuffle_epi8(third_group, counter_swap),
                                                       wide_keys_vec[0].zmm);
            __m512i fourth_keystream = _mm512_xor_si512(_mm512_shuffle_epi8(fourth_group, counter_swap),
                                                        wide_keys_vec[0].zmm);
            first_keystream = sz_aes256_rounds_wide_icelake_(first_keystream, wide_keys_vec);
            second_keystream = sz_aes256_rounds_wide_icelake_(second_keystream, wide_keys_vec);
            third_keystream = sz_aes256_rounds_wide_icelake_(third_keystream, wide_keys_vec);
            fourth_keystream = sz_aes256_rounds_wide_icelake_(fourth_keystream, wide_keys_vec);
            _mm512_storeu_si512(output + produced + 0,
                                _mm512_xor_si512(first_keystream, _mm512_loadu_si512(text + produced + 0)));
            _mm512_storeu_si512(output + produced + 64,
                                _mm512_xor_si512(second_keystream, _mm512_loadu_si512(text + produced + 64)));
            _mm512_storeu_si512(output + produced + 128,
                                _mm512_xor_si512(third_keystream, _mm512_loadu_si512(text + produced + 128)));
            _mm512_storeu_si512(output + produced + 192,
                                _mm512_xor_si512(fourth_keystream, _mm512_loadu_si512(text + produced + 192)));
            first_group = _mm512_add_epi32(first_group, step_sixteen);
            second_group = _mm512_add_epi32(second_group, step_sixteen);
            third_group = _mm512_add_epi32(third_group, step_sixteen);
            fourth_group = _mm512_add_epi32(fourth_group, step_sixteen);
        }
        counters = first_group;
    }

    for (; produced + 64 <= length; produced += 64) {
        __m512i keystream = _mm512_xor_si512(_mm512_shuffle_epi8(counters, counter_swap), wide_keys_vec[0].zmm);
        keystream = sz_aes256_rounds_wide_icelake_(keystream, wide_keys_vec);
        _mm512_storeu_si512(output + produced, _mm512_xor_si512(keystream, _mm512_loadu_si512(text + produced)));
        counters = _mm512_add_epi32(counters, step_four);
    }

    if (produced != length) {
        sz_size_t const remaining = length - produced;
        __mmask64 const tail_mask = sz_u64_mask_until_(remaining);
        __m512i keystream = _mm512_xor_si512(_mm512_shuffle_epi8(counters, counter_swap), wide_keys_vec[0].zmm);
        keystream = sz_aes256_rounds_wide_icelake_(keystream, wide_keys_vec);
        _mm512_mask_storeu_epi8(output + produced, tail_mask,
                                _mm512_xor_si512(keystream, _mm512_maskz_loadu_epi8(tail_mask, text + produced)));
        counters = _mm512_add_epi32(
            counters, _mm512_maskz_set1_epi32((__mmask16)0x8888u,
                                              (int)((remaining + SZ_AES_BLOCK_LENGTH - 1) / SZ_AES_BLOCK_LENGTH)));
    }
    return counters;
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
        __m512i const counter_swap = _mm512_load_si512(sz_aes256_counter_swap_icelake_());
        __m128i const counter_xmm = sz_aes256_counter_block_icelake_(sz_aes256_counters_icelake_(nonce, block_index),
                                                                     counter_swap);
        sz_u128_vec_t keystream_vec;
        keystream_vec.xmm = sz_aes256_block_encrypt_icelake_(key, counter_xmm);
        for (; within_block != SZ_AES_BLOCK_LENGTH && produced != length; ++within_block, ++produced)
            output_bytes[produced] = (sz_u8_t)(input_bytes[produced] ^ keystream_vec.u8s[within_block]);
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
 *  @param product_low_xmm The low 128 bits of the schoolbook product.
 *  @param product_middle_xmm The two cross terms, already exclusive-ored together.
 *  @param product_high_xmm The high 128 bits of the schoolbook product.
 *  @return The product modulo `x^128 + x^7 + x^2 + x + 1`, byte reversed like its operands.
 *
 *  Galois/counter mode numbers the bits of a block backwards relative to the way `pclmulqdq` reads them,
 *  which leaves the 256-bit product one position low; the shift by one puts it back. The fold that follows
 *  drives the low half through the polynomial twice with the same instruction that produced the product,
 *  which is a third of the operations the equivalent chain of shifts and exclusive-ors needs. Splitting the
 *  three halves out as parameters is what lets the wide path exclusive-or eight blocks' worth of products
 *  together first and pay this reduction once.
 */
SZ_HELPER_INLINE __m128i sz_ghash_reduce_icelake_(__m128i product_low_xmm, __m128i product_middle_xmm,
                                                  __m128i product_high_xmm) {
    __m128i const polynomial_xmm = _mm_set_epi64x((sz_i64_t)0xC200000000000000ull, 0);
    __m128i low_xmm = _mm_xor_si128(product_low_xmm, _mm_slli_si128(product_middle_xmm, 8));
    __m128i high_xmm = _mm_xor_si128(product_high_xmm, _mm_srli_si128(product_middle_xmm, 8));
    __m128i const low_carry_xmm = _mm_srli_epi32(low_xmm, 31);
    __m128i const high_carry_xmm = _mm_srli_epi32(high_xmm, 31);
    __m128i folded_xmm;

    low_xmm = _mm_or_si128(_mm_slli_epi32(low_xmm, 1), _mm_slli_si128(low_carry_xmm, 4));
    high_xmm = _mm_or_si128(_mm_slli_epi32(high_xmm, 1), _mm_slli_si128(high_carry_xmm, 4));
    high_xmm = _mm_or_si128(high_xmm, _mm_srli_si128(low_carry_xmm, 12));

    folded_xmm = _mm_clmulepi64_si128(low_xmm, polynomial_xmm, 0x10);
    low_xmm = _mm_xor_si128(_mm_shuffle_epi32(low_xmm, 0x4E), folded_xmm);
    folded_xmm = _mm_clmulepi64_si128(low_xmm, polynomial_xmm, 0x10);
    low_xmm = _mm_xor_si128(_mm_shuffle_epi32(low_xmm, 0x4E), folded_xmm);
    return _mm_xor_si128(high_xmm, low_xmm);
}

/**
 *  @brief Multiplies two byte-reversed field elements.
 *  @param first_operand_xmm One operand, byte reversed.
 *  @param second_operand_xmm The other operand, byte reversed.
 *  @return Their product, byte reversed.
 */
SZ_HELPER_INLINE __m128i sz_ghash_multiply_icelake_(__m128i first_operand_xmm, __m128i second_operand_xmm) {
    __m128i const product_low_xmm = _mm_clmulepi64_si128(first_operand_xmm, second_operand_xmm, 0x00);
    __m128i const product_high_xmm = _mm_clmulepi64_si128(first_operand_xmm, second_operand_xmm, 0x11);
    __m128i const product_middle_xmm = _mm_xor_si128(_mm_clmulepi64_si128(first_operand_xmm, second_operand_xmm, 0x10),
                                                     _mm_clmulepi64_si128(first_operand_xmm, second_operand_xmm, 0x01));
    return sz_ghash_reduce_icelake_(product_low_xmm, product_middle_xmm, product_high_xmm);
}

SZ_API_COMPTIME void sz_aes256_gcm_key_init_icelake(sz_aes256_gcm_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    __m128i const reverse_xmm = _mm512_castsi512_si128(_mm512_load_si512(sz_ghash_byte_reverse_icelake_()));
    __m128i subkey_xmm, power_xmm;
    sz_size_t power_index;

    sz_aes256_key_init_icelake(&key->block, secret);
    subkey_xmm = _mm_shuffle_epi8(sz_aes256_block_encrypt_icelake_(&key->block, _mm_setzero_si128()), reverse_xmm);
    power_xmm = subkey_xmm;
    for (power_index = 0; power_index != 8; ++power_index) {
        _mm_mask_storeu_epi8(key->powers + power_index * SZ_AES_BLOCK_LENGTH, (__mmask16)0xFFFFu,
                             _mm_shuffle_epi8(power_xmm, reverse_xmm));
        power_xmm = sz_ghash_multiply_icelake_(power_xmm, subkey_xmm);
    }
}

/** @brief Exclusive-ors the four 128-bit lanes of a register into one, which the reduction then closes. */
SZ_HELPER_INLINE __m128i sz_ghash_fold_lanes_icelake_(__m512i value) {
    __m512i folded = _mm512_xor_si512(value, _mm512_shuffle_i64x2(value, value, 0x4E));
    folded = _mm512_xor_si512(folded, _mm512_shuffle_i64x2(folded, folded, 0xB1));
    return _mm512_castsi512_si128(folded);
}

/**
 *  @brief Absorbs eight byte-reversed blocks and the running hash in one reduction.
 *  @param accumulator_xmm The running hash, byte reversed.
 *  @param blocks_low Blocks one through four, byte reversed.
 *  @param blocks_high Blocks five through eight, byte reversed.
 *  @param powers_high `H^8` through `H^5`, byte reversed, one per lane.
 *  @param powers_low `H^4` through `H^1`, byte reversed, one per lane.
 *  @param eighth_power_xmm `H^8` alone, byte reversed.
 *  @return The running hash after all eight blocks, byte reversed.
 *
 *  The hash is a chain, `Y = (Y ^ X) * H`, and a chain of reductions would run at the latency of one
 *  multiply per block. Expanding eight steps gives `Y * H^8 ^ X1 * H^8 ^ ... ^ X8 * H^1`, whose terms are
 *  independent, so their unreduced products exclusive-or together and one reduction closes all eight.
 *  `Y * H^8` is kept out of the wide half deliberately: eight blocks of message are known long before the
 *  previous reduction lands, so folding them separately leaves only one narrow multiply, one exclusive-or
 *  and the reduction on the chain that actually carries between iterations.
 */
SZ_HELPER_INLINE __m128i sz_ghash_stage_icelake_(__m128i accumulator_xmm, __m512i blocks_low, __m512i blocks_high,
                                                 __m512i powers_high, __m512i powers_low, __m128i eighth_power_xmm) {
    __m512i const product_low = _mm512_xor_si512(_mm512_clmulepi64_epi128(blocks_low, powers_high, 0x00),
                                                 _mm512_clmulepi64_epi128(blocks_high, powers_low, 0x00));
    __m512i const product_high = _mm512_xor_si512(_mm512_clmulepi64_epi128(blocks_low, powers_high, 0x11),
                                                  _mm512_clmulepi64_epi128(blocks_high, powers_low, 0x11));
    __m512i const product_middle = _mm512_xor_si512(
        _mm512_xor_si512(_mm512_clmulepi64_epi128(blocks_low, powers_high, 0x01),
                         _mm512_clmulepi64_epi128(blocks_low, powers_high, 0x10)),
        _mm512_xor_si512(_mm512_clmulepi64_epi128(blocks_high, powers_low, 0x01),
                         _mm512_clmulepi64_epi128(blocks_high, powers_low, 0x10)));
    __m128i const carried_low_xmm = _mm_clmulepi64_si128(accumulator_xmm, eighth_power_xmm, 0x00);
    __m128i const carried_high_xmm = _mm_clmulepi64_si128(accumulator_xmm, eighth_power_xmm, 0x11);
    __m128i const carried_middle_xmm = _mm_xor_si128(_mm_clmulepi64_si128(accumulator_xmm, eighth_power_xmm, 0x10),
                                                     _mm_clmulepi64_si128(accumulator_xmm, eighth_power_xmm, 0x01));
    return sz_ghash_reduce_icelake_(_mm_xor_si128(sz_ghash_fold_lanes_icelake_(product_low), carried_low_xmm),
                                    _mm_xor_si128(sz_ghash_fold_lanes_icelake_(product_middle), carried_middle_xmm),
                                    _mm_xor_si128(sz_ghash_fold_lanes_icelake_(product_high), carried_high_xmm));
}

/**
 *  @brief Loads `H^8` through `H^1` descending and byte reversed, two lanes-worth per register.
 *  @param key The expanded key holding the powers ascending.
 *  @param powers_high Receives `H^8`, `H^7`, `H^6`, `H^5`.
 *  @param powers_low Receives `H^4`, `H^3`, `H^2`, `H^1`.
 */
SZ_HELPER_INLINE void sz_ghash_powers_icelake_(sz_aes256_gcm_key_t const *key, __m512i *powers_high,
                                               __m512i *powers_low) {
    __m512i const reverse = _mm512_load_si512(sz_ghash_byte_reverse_icelake_());
    __m512i const ascending_low = _mm512_loadu_si512(key->powers);
    __m512i const ascending_high = _mm512_loadu_si512(key->powers + 4 * SZ_AES_BLOCK_LENGTH);
    *powers_high = _mm512_shuffle_epi8(_mm512_shuffle_i64x2(ascending_high, ascending_high, 0x1B), reverse);
    *powers_low = _mm512_shuffle_epi8(_mm512_shuffle_i64x2(ascending_low, ascending_low, 0x1B), reverse);
}

/**
 *  @brief Absorbs whole blocks into a byte-reversed running hash.
 *  @param accumulator_xmm The running hash, byte reversed.
 *  @param blocks The bytes to absorb, `count * 16` of them.
 *  @param count Number of whole blocks.
 *  @param powers_high `H^8` through `H^5`, byte reversed, one per lane.
 *  @param powers_low `H^4` through `H^1`, byte reversed, one per lane.
 *  @return The running hash after every block, byte reversed.
 */
SZ_HELPER_INLINE __m128i sz_ghash_absorb_blocks_icelake_(__m128i accumulator_xmm, sz_u8_t const *blocks,
                                                         sz_size_t count, __m512i powers_high, __m512i powers_low) {
    __m512i const reverse = _mm512_load_si512(sz_ghash_byte_reverse_icelake_());
    __m128i const eighth_power_xmm = _mm512_castsi512_si128(powers_high);
    __m128i const subkey_xmm = _mm512_extracti64x2_epi64(powers_low, 3);
    sz_size_t absorbed = 0;

    for (; absorbed + 8 <= count; absorbed += 8) {
        __m512i const blocks_low = _mm512_shuffle_epi8(_mm512_loadu_si512(blocks + absorbed * 16), reverse);
        __m512i const blocks_high = _mm512_shuffle_epi8(_mm512_loadu_si512(blocks + absorbed * 16 + 64), reverse);
        accumulator_xmm = sz_ghash_stage_icelake_(accumulator_xmm, blocks_low, blocks_high, powers_high, powers_low,
                                                  eighth_power_xmm);
    }

    for (; absorbed != count; ++absorbed) {
        __m128i const block_xmm = _mm_shuffle_epi8(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, blocks + absorbed * 16),
                                                   _mm512_castsi512_si128(reverse));
        accumulator_xmm = sz_ghash_multiply_icelake_(_mm_xor_si128(accumulator_xmm, block_xmm), subkey_xmm);
    }
    return accumulator_xmm;
}

#pragma endregion // Galois Hashing

#pragma region Streaming Interface

/**
 *  @brief Overwrites a finished state so the key schedule it embeds does not outlive the call.
 *
 *  The size is known at compile time, so this is seven full-width stores and one masked tail rather than
 *  a length-driven loop. Writes are ordinary stores followed by one barrier: routing them through a
 *  `volatile` view instead would forbid vectorization and cost 472 single-byte stores on every one-shot
 *  call.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_state_scrub_icelake_(sz_aes256_gcm_state_t *state) {
    sz_u8_t *const bytes = (sz_u8_t *)state;
    __m512i const zeros = _mm512_setzero_si512();
    sz_size_t offset = 0;
    for (; offset + 64 <= sizeof(*state); offset += 64) _mm512_storeu_si512(bytes + offset, zeros);
    _mm512_mask_storeu_epi8(bytes + offset, sz_u64_mask_until_(sizeof(*state) - offset), zeros);
    sz_keep_alive_(state);
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
    __m128i const reverse_xmm = _mm512_castsi512_si128(_mm512_load_si512(sz_ghash_byte_reverse_icelake_()));
    __m512i powers_high, powers_low;
    __m128i accumulator_xmm, subkey_xmm;
    sz_size_t consumed = 0, byte_index;

    if (length == 0) return;
    sz_ghash_powers_icelake_(&state->key, &powers_high, &powers_low);
    subkey_xmm = _mm512_extracti64x2_epi64(powers_low, 3);
    accumulator_xmm = _mm_shuffle_epi8(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->accumulator), reverse_xmm);

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
            __m128i const block_xmm = _mm_shuffle_epi8(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->partial),
                                                       reverse_xmm);
            accumulator_xmm = sz_ghash_multiply_icelake_(_mm_xor_si128(accumulator_xmm, block_xmm), subkey_xmm);
            state->buffered = 0;
        }
    }

    if (state->buffered == 0 && length - consumed >= SZ_AES_BLOCK_LENGTH) {
        sz_size_t const blocks = (length - consumed) / SZ_AES_BLOCK_LENGTH;
        accumulator_xmm = sz_ghash_absorb_blocks_icelake_(accumulator_xmm, input_bytes + consumed, blocks, powers_high,
                                                          powers_low);
        consumed += blocks * SZ_AES_BLOCK_LENGTH;
        state->associated_length += blocks * SZ_AES_BLOCK_LENGTH;
    }

    for (; consumed != length; ++consumed) {
        state->partial[state->buffered++] = input_bytes[consumed];
        ++state->associated_length;
    }
    _mm_mask_storeu_epi8(state->accumulator, (__mmask16)0xFFFFu, _mm_shuffle_epi8(accumulator_xmm, reverse_xmm));
}

/** @brief Absorbs whatever `partial` holds, zero padded to a full block, and empties it. */
SZ_HELPER_AUTO void sz_aes256_gcm_flush_partial_icelake_(sz_aes256_gcm_state_t *state) {
    __m128i reverse_xmm, subkey_xmm, padded_xmm, accumulator_xmm;
    if (state->buffered == 0) return;
    reverse_xmm = _mm512_castsi512_si128(_mm512_load_si512(sz_ghash_byte_reverse_icelake_()));
    subkey_xmm = _mm_shuffle_epi8(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->key.powers), reverse_xmm);
    padded_xmm = _mm_shuffle_epi8(_mm_maskz_loadu_epi8(sz_u16_mask_until_(state->buffered), state->partial),
                                  reverse_xmm);
    accumulator_xmm = _mm_shuffle_epi8(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->accumulator), reverse_xmm);
    accumulator_xmm = sz_ghash_multiply_icelake_(_mm_xor_si128(accumulator_xmm, padded_xmm), subkey_xmm);
    _mm_mask_storeu_epi8(state->accumulator, (__mmask16)0xFFFFu, _mm_shuffle_epi8(accumulator_xmm, reverse_xmm));
    state->buffered = 0;
}

/**
 *  @brief Transforms bytes one at a time, absorbing the ciphertext side into @p accumulator_xmm.
 *  @param state The state, whose keystream block and hash block both advance byte by byte.
 *  @param accumulator_xmm The running hash, byte reversed, updated in place.
 *  @param subkey_xmm `H^1`, byte reversed.
 *  @param reverse_xmm The byte reversal pattern.
 *  @param text The bytes to transform.
 *  @param length Number of bytes.
 *  @param output Receives @p length transformed bytes.
 *  @param direction Which side of the transformation the hash absorbs.
 *
 *  Serves the two edges of a chunk: the keystream block a previous call left half spent, and the trailing
 *  bytes of this one that do not fill a block. The ciphertext byte is captured before the store because a
 *  caller may pass one pointer for both sides.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_bytes_icelake_(sz_aes256_gcm_state_t *state, __m128i *accumulator_xmm,
                                                   __m128i subkey_xmm, __m128i reverse_xmm, sz_u8_t const *text,
                                                   sz_size_t length, sz_u8_t *output,
                                                   sz_aes256_gcm_direction_t direction) {
    sz_size_t produced = 0;
    for (; produced != length; ++produced) {
        if (state->keystream_used == SZ_AES_BLOCK_LENGTH) {
            sz_u128_vec_t counter_vec;
            counter_vec.xmm = _mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->counter);
            counter_vec.u32s[3] = sz_u32_bytes_reverse(sz_u32_bytes_reverse(counter_vec.u32s[3]) + 1u);
            _mm_mask_storeu_epi8(state->counter, (__mmask16)0xFFFFu, counter_vec.xmm);
            _mm_mask_storeu_epi8(state->keystream, (__mmask16)0xFFFFu,
                                 sz_aes256_block_encrypt_icelake_(&state->key.block, counter_vec.xmm));
            state->keystream_used = 0;
        }
        {
            sz_u8_t const plaintext_byte = text[produced];
            sz_u8_t const transformed = (sz_u8_t)(plaintext_byte ^ state->keystream[state->keystream_used]);
            sz_u8_t const ciphertext = direction == sz_aes256_gcm_decrypting_k ? plaintext_byte : transformed;
            output[produced] = transformed;
            state->partial[state->buffered++] = ciphertext;
            ++state->keystream_used;
            ++state->text_length;
            if (state->buffered == SZ_AES_BLOCK_LENGTH) {
                __m128i const block_xmm = _mm_shuffle_epi8(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->partial),
                                                           reverse_xmm);
                *accumulator_xmm = sz_ghash_multiply_icelake_(_mm_xor_si128(*accumulator_xmm, block_xmm), subkey_xmm);
                state->buffered = 0;
            }
        }
    }
}

/**
 *  @brief Encrypts whole blocks and absorbs their ciphertext, the hash trailing the cipher by one stage.
 *  @param counters Four consecutive counter blocks, advanced past everything the call consumed.
 *  @param accumulator_xmm The running hash, byte reversed, updated in place.
 *  @param wide_keys_vec The fifteen round keys, each broadcast across four lanes.
 *  @param powers_high `H^8` through `H^5`, byte reversed, one per lane.
 *  @param powers_low `H^4` through `H^1`, byte reversed, one per lane.
 *  @param cipher_mask All ones when the hash absorbs the output, zero when it absorbs the input.
 *  @param text The input blocks.
 *  @param output Receives the transformed blocks; may equal @p text.
 *  @param blocks Number of whole blocks, at least one.
 *
 *  The hash of a stage cannot begin until its ciphertext exists, so it is held one stage back and the
 *  carry-less multiplies then sit in the shadow of the next stage's substitution rounds, which occupy a
 *  different port and would otherwise stall on the reduction's latency. The ciphertext reaches the hash in
 *  registers rather than through the output buffer, which is what lets a decryption transform its input in
 *  place.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_stride_icelake_(__m512i *counters_out, __m128i *accumulator_xmm,
                                                    sz_u512_vec_t const *wide_keys_vec, __m512i powers_high,
                                                    __m512i powers_low, __mmask8 cipher_mask, sz_u8_t const *text,
                                                    sz_u8_t *output, sz_size_t blocks) {
    __m512i const counter_swap = _mm512_load_si512(sz_aes256_counter_swap_icelake_());
    __m512i const reverse = _mm512_load_si512(sz_ghash_byte_reverse_icelake_());
    __m512i const step_four = _mm512_maskz_set1_epi32((__mmask16)0x8888u, 4);
    __m512i const step_eight = _mm512_maskz_set1_epi32((__mmask16)0x8888u, 8);
    __m128i const eighth_power_xmm = _mm512_castsi512_si128(powers_high);
    sz_size_t const stages = blocks / 8;
    sz_size_t stage_index, produced = 0;

    __m512i first_counter = *counters_out;
    __m512i second_counter = _mm512_add_epi32(first_counter, step_four);
    __m512i first_pending = _mm512_setzero_si512(), second_pending = _mm512_setzero_si512();

    for (stage_index = 0; stage_index != stages; ++stage_index) {
        __m512i first_keystream = _mm512_xor_si512(_mm512_shuffle_epi8(first_counter, counter_swap),
                                                   wide_keys_vec[0].zmm);
        __m512i second_keystream = _mm512_xor_si512(_mm512_shuffle_epi8(second_counter, counter_swap),
                                                    wide_keys_vec[0].zmm);
        // The previous stage's hash lands between the first round key and the rounds themselves, so the
        // carry-less multipliers and the cipher units issue against each other rather than in sequence.
        if (stage_index != 0)
            *accumulator_xmm = sz_ghash_stage_icelake_(*accumulator_xmm, first_pending, second_pending, powers_high,
                                                       powers_low, eighth_power_xmm);
        first_keystream = sz_aes256_rounds_wide_icelake_(first_keystream, wide_keys_vec);
        second_keystream = sz_aes256_rounds_wide_icelake_(second_keystream, wide_keys_vec);

        __m512i const first_loaded = _mm512_loadu_si512(text + produced + 0);
        __m512i const first_stored = _mm512_xor_si512(first_loaded, first_keystream);
        _mm512_storeu_si512(output + produced + 0, first_stored);
        first_pending = _mm512_shuffle_epi8(_mm512_mask_blend_epi64(cipher_mask, first_loaded, first_stored), reverse);
        first_counter = _mm512_add_epi32(first_counter, step_eight);

        __m512i const second_loaded = _mm512_loadu_si512(text + produced + 64);
        __m512i const second_stored = _mm512_xor_si512(second_loaded, second_keystream);
        _mm512_storeu_si512(output + produced + 64, second_stored);
        second_pending = _mm512_shuffle_epi8(_mm512_mask_blend_epi64(cipher_mask, second_loaded, second_stored),
                                             reverse);
        second_counter = _mm512_add_epi32(second_counter, step_eight);
        produced += 128;
    }
    if (stages != 0)
        *accumulator_xmm = sz_ghash_stage_icelake_(*accumulator_xmm, first_pending, second_pending, powers_high,
                                                   powers_low, eighth_power_xmm);

    if (blocks % 8 != 0) {
        sz_size_t const tail_blocks = blocks % 8;
        sz_size_t const tail_bytes = tail_blocks * SZ_AES_BLOCK_LENGTH;
        sz_align_(64) sz_u8_t staged_bytes[128];
        __mmask64 const first_mask = sz_u64_clamp_mask_until_(tail_bytes);
        __m512i first_keystream = _mm512_xor_si512(_mm512_shuffle_epi8(first_counter, counter_swap),
                                                   wide_keys_vec[0].zmm);
        first_keystream = sz_aes256_rounds_wide_icelake_(first_keystream, wide_keys_vec);
        {
            __m512i const loaded = _mm512_maskz_loadu_epi8(first_mask, text + produced);
            __m512i const stored = _mm512_xor_si512(loaded, first_keystream);
            _mm512_mask_storeu_epi8(output + produced, first_mask, stored);
            _mm512_store_si512(staged_bytes, _mm512_mask_blend_epi64(cipher_mask, loaded, stored));
        }
        // A tail of five blocks or more spills into a second group; four or fewer never does, and
        // encrypting a group that stores nothing would cost thirteen rounds for no bytes.
        if (tail_bytes > 64) {
            __mmask64 const second_mask = sz_u64_clamp_mask_until_(tail_bytes - 64);
            __m512i second_keystream = _mm512_xor_si512(_mm512_shuffle_epi8(second_counter, counter_swap),
                                                        wide_keys_vec[0].zmm);
            second_keystream = sz_aes256_rounds_wide_icelake_(second_keystream, wide_keys_vec);
            __m512i const loaded = _mm512_maskz_loadu_epi8(second_mask, text + produced + 64);
            __m512i const stored = _mm512_xor_si512(loaded, second_keystream);
            _mm512_mask_storeu_epi8(output + produced + 64, second_mask, stored);
            _mm512_store_si512(staged_bytes + 64, _mm512_mask_blend_epi64(cipher_mask, loaded, stored));
        }
        *accumulator_xmm = sz_ghash_absorb_blocks_icelake_(*accumulator_xmm, staged_bytes, tail_blocks, powers_high,
                                                           powers_low);
        first_counter = _mm512_add_epi32(first_counter, _mm512_maskz_set1_epi32((__mmask16)0x8888u, (int)tail_blocks));
    }
    *counters_out = first_counter;
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
 *  chunk boundary, so the wide path is entered only once both stand at a block boundary and left with the
 *  same property. The bytes on either side of it spend one keystream block between them and are the only
 *  place a chunk boundary costs anything.
 *
 *  Both call sites pass a constant direction into this always-inlined body, so the lane mask the wide
 *  path blends with folds away and neither loop carries a branch or a mask register on its account.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_transform_icelake_(sz_aes256_gcm_state_t *state, sz_cptr_t text, sz_size_t length,
                                                       sz_ptr_t output, sz_aes256_gcm_direction_t direction) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    __m512i const counter_swap = _mm512_load_si512(sz_aes256_counter_swap_icelake_());
    __m128i const reverse_xmm = _mm512_castsi512_si128(_mm512_load_si512(sz_ghash_byte_reverse_icelake_()));
    __m512i powers_high, powers_low;
    __m128i accumulator_xmm, subkey_xmm;
    sz_size_t produced, head_bytes;

    //  Associated data ends the moment the first message byte arrives, and its tail needs padding.
    if (state->text_length == 0 && length != 0) sz_aes256_gcm_flush_partial_icelake_(state);
    if (length == 0) return;

    sz_ghash_powers_icelake_(&state->key, &powers_high, &powers_low);
    subkey_xmm = _mm512_extracti64x2_epi64(powers_low, 3);
    accumulator_xmm = _mm_shuffle_epi8(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->accumulator), reverse_xmm);

    head_bytes = (sz_size_t)(SZ_AES_BLOCK_LENGTH - state->keystream_used);
    if (head_bytes > length) head_bytes = length;
    sz_aes256_gcm_bytes_icelake_(state, &accumulator_xmm, subkey_xmm, reverse_xmm, input_bytes, head_bytes,
                                 output_bytes, direction);
    produced = head_bytes;

    if (state->keystream_used == SZ_AES_BLOCK_LENGTH && state->buffered == 0 &&
        length - produced >= SZ_AES_BLOCK_LENGTH) {
        sz_size_t const blocks = (length - produced) / SZ_AES_BLOCK_LENGTH;
        __mmask8 const cipher_mask = direction == sz_aes256_gcm_decrypting_k ? (__mmask8)0x00u : (__mmask8)0xFFu;
        sz_u512_vec_t wide_keys_vec[15];
        __m512i counters;
        sz_aes256_round_keys_wide_icelake_(&state->key.block, wide_keys_vec);
        counters = _mm512_add_epi32(
            _mm512_shuffle_epi8(_mm512_broadcast_i32x4(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->counter)),
                                counter_swap),
            _mm512_set_epi32(4, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0));
        sz_aes256_gcm_stride_icelake_(&counters, &accumulator_xmm, wide_keys_vec, powers_high, powers_low, cipher_mask,
                                      input_bytes + produced, output_bytes + produced, blocks);
        produced += blocks * SZ_AES_BLOCK_LENGTH;
        state->text_length += blocks * SZ_AES_BLOCK_LENGTH;
        _mm_mask_storeu_epi8(
            state->counter, (__mmask16)0xFFFFu,
            sz_aes256_counter_block_icelake_(_mm512_sub_epi32(counters, _mm512_maskz_set1_epi32((__mmask16)0x8888u, 1)),
                                             counter_swap));
    }

    sz_aes256_gcm_bytes_icelake_(state, &accumulator_xmm, subkey_xmm, reverse_xmm, input_bytes + produced,
                                 length - produced, output_bytes + produced, direction);
    _mm_mask_storeu_epi8(state->accumulator, (__mmask16)0xFFFFu, _mm_shuffle_epi8(accumulator_xmm, reverse_xmm));
}

/**
 *  @brief Closes the hash over whatever is still pending and masks it into the authentication tag.
 *  @param state The state, left untouched.
 *  @param tag Receives the sixteen tag bytes.
 */
SZ_HELPER_AUTO void sz_aes256_gcm_digest_icelake_(sz_aes256_gcm_state_t const *state, sz_u8_t tag[sz_at_least_(16)]) {
    __m128i const reverse_xmm = _mm512_castsi512_si128(_mm512_load_si512(sz_ghash_byte_reverse_icelake_()));
    __m128i const subkey_xmm = _mm_shuffle_epi8(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->key.powers),
                                                reverse_xmm);
    __m128i accumulator_xmm = _mm_shuffle_epi8(_mm_maskz_loadu_epi8((__mmask16)0xFFFFu, state->accumulator),
                                               reverse_xmm);
    sz_u128_vec_t lengths_vec;

    //  Whatever is still pending gets padded here: an associated-data tail when the message was empty,
    //  or the message's own trailing ciphertext bytes otherwise.
    if (state->buffered != 0) {
        __m128i const padded_xmm = _mm_shuffle_epi8(
            _mm_maskz_loadu_epi8(sz_u16_mask_until_(state->buffered), state->partial), reverse_xmm);
        accumulator_xmm = sz_ghash_multiply_icelake_(_mm_xor_si128(accumulator_xmm, padded_xmm), subkey_xmm);
    }

    sz_aes256_gcm_lengths_serial_(&lengths_vec, state->associated_length, state->text_length);
    accumulator_xmm = sz_ghash_multiply_icelake_(
        _mm_xor_si128(accumulator_xmm, _mm_shuffle_epi8(lengths_vec.xmm, reverse_xmm)), subkey_xmm);
    _mm_mask_storeu_epi8(tag, (__mmask16)0xFFFFu,
                         _mm_xor_si128(_mm_shuffle_epi8(accumulator_xmm, reverse_xmm),
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
    return sz_aes256_tag_equal_serial_(expected, tag) == sz_true_k ? sz_success_k : sz_authentication_failed_k;
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
