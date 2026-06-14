/**
 *  @brief Ice Lake (AVX-512 VBMI+VAES) backend for substring & byte-set search.
 *  @file include/stringzilla/find/icelake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_ICELAKE_H_
#define STRINGZILLA_FIND_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/find/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  AVX512 implementation of the string search algorithms for Ice Lake and newer CPUs.
 *  Includes extensions:
 *      - 2017 Skylake: F, CD, ER, PF, VL, DQ, BW,
 *      - 2018 CannonLake: IFMA, VBMI,
 *      - 2019 Ice Lake: VPOPCNTDQ, VNNI, VBMI2, BITALG, GFNI, VPCLMULQDQ, VAES.
 *
 *  We are going to use VBMI2 for `_mm256_maskz_compress_epi8`.
 */
#if SZ_USE_ICELAKE
#if defined(__clang__)
#pragma clang attribute push(                                                                          \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2")
#endif

SZ_PUBLIC sz_cptr_t sz_find_byteset_icelake(sz_cptr_t text, sz_size_t length, sz_byteset_t const *filter) {

    // Before initializing the AVX-512 vectors, we may want to run the sequential code for the first few bytes.
    // In practice, that only hurts, even when we have matches every 5-ish bytes.
    //
    //      if (length < SZ_SWAR_THRESHOLD) return sz_find_byteset_serial(text, length, filter);
    //      sz_cptr_t early_result = sz_find_byteset_serial(text, SZ_SWAR_THRESHOLD, filter);
    //      if (early_result) return early_result;
    //      text += SZ_SWAR_THRESHOLD;
    //      length -= SZ_SWAR_THRESHOLD;
    //
    // Let's unzip even and odd elements and replicate them into both lanes of the YMM register.
    // That way when we invoke `_mm512_shuffle_epi8` we can use the same mask for both lanes.
    sz_u512_vec_t filter_even_vec, filter_odd_vec;
    __m256i filter_ymm = _mm256_lddqu_si256((__m256i const *)filter);
    // There are a few way to initialize filters without having native strided loads.
    // In the chronological order of experiments:
    // - serial code initializing 128 bytes of odd and even mask
    // - using several shuffles
    // - using `_mm512_permutexvar_epi8`
    // - using `_mm512_broadcast_i32x4(_mm256_castsi256_si128(_mm256_maskz_compress_epi8(0x55555555, filter_ymm)))`
    //   and `_mm512_broadcast_i32x4(_mm256_castsi256_si128(_mm256_maskz_compress_epi8(0xaaaaaaaa, filter_ymm)))`
    filter_even_vec.zmm = _mm512_broadcast_i32x4(_mm256_castsi256_si128( // broadcast __m128i to __m512i
        _mm256_maskz_compress_epi8(0x55555555, filter_ymm)));
    filter_odd_vec.zmm = _mm512_broadcast_i32x4(_mm256_castsi256_si128( // broadcast __m128i to __m512i
        _mm256_maskz_compress_epi8(0xaaaaaaaa, filter_ymm)));
    // After the unzipping operation, we can validate the contents of the vectors like this:
    //
    //      for (sz_size_t i = 0; i != 16; ++i) {
    //          sz_assert_(filter_even_vec.u8s[i] == filter->_u8s[i * 2]);
    //          sz_assert_(filter_odd_vec.u8s[i] == filter->_u8s[i * 2 + 1]);
    //          sz_assert_(filter_even_vec.u8s[i + 16] == filter->_u8s[i * 2]);
    //          sz_assert_(filter_odd_vec.u8s[i + 16] == filter->_u8s[i * 2 + 1]);
    //          sz_assert_(filter_even_vec.u8s[i + 32] == filter->_u8s[i * 2]);
    //          sz_assert_(filter_odd_vec.u8s[i + 32] == filter->_u8s[i * 2 + 1]);
    //          sz_assert_(filter_even_vec.u8s[i + 48] == filter->_u8s[i * 2]);
    //          sz_assert_(filter_odd_vec.u8s[i + 48] == filter->_u8s[i * 2 + 1]);
    //      }
    //
    sz_u512_vec_t text_vec;
    sz_u512_vec_t lower_nibbles_vec, higher_nibbles_vec;
    sz_u512_vec_t bitset_even_vec, bitset_odd_vec;
    sz_u512_vec_t bitmask_vec, bitmask_lookup_vec;
    bitmask_lookup_vec.zmm = _mm512_set_epi8(                       //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1);

    while (length) {
        // The following algorithm is a transposed equivalent of the "SIMDized check which bytes are in a set"
        // solutions by Wojciech Muła. We populate the bitmask differently and target newer CPUs, so
        // StrinZilla uses a somewhat different approach.
        // http://0x80.pl/articles/simd-byte-lookup.html#alternative-implementation-new
        //
        //      sz_u8_t input = *(sz_u8_t const *)text;
        //      sz_u8_t lo_nibble = input & 0x0f;
        //      sz_u8_t hi_nibble = input >> 4;
        //      sz_u8_t bitset_even = filter_even_vec.u8s[hi_nibble];
        //      sz_u8_t bitset_odd = filter_odd_vec.u8s[hi_nibble];
        //      sz_u8_t bitmask = (1 << (lo_nibble & 0x7));
        //      sz_u8_t bitset = lo_nibble < 8 ? bitset_even : bitset_odd;
        //      if ((bitset & bitmask) != 0) return text;
        //      else { length--, text++; }
        //
        // The nice part about this, loading the strided data is vey easy with Arm NEON,
        // while with x86 CPUs after AVX, shuffles within 256 bits shouldn't be an issue either.
        sz_size_t load_length = sz_min_of_two(length, 64);
        __mmask64 load_mask = sz_u64_mask_until_(load_length);
        text_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, text);
        lower_nibbles_vec.zmm = _mm512_and_si512(text_vec.zmm, _mm512_set1_epi8(0x0f));
        bitmask_vec.zmm = _mm512_shuffle_epi8(bitmask_lookup_vec.zmm, lower_nibbles_vec.zmm);
        //
        // At this point we can validate the `bitmask_vec` contents like this:
        //
        //      for (sz_size_t i = 0; i != load_length; ++i) {
        //          sz_u8_t input = *(sz_u8_t const *)(text + i);
        //          sz_u8_t lo_nibble = input & 0x0f;
        //          sz_u8_t bitmask = (1 << (lo_nibble & 0x7));
        //          sz_assert_(bitmask_vec.u8s[i] == bitmask);
        //      }
        //
        // Shift right every byte by 4 bits.
        // There is no `_mm512_srli_epi8` intrinsic, so we have to use `_mm512_srli_epi16`
        // and combine it with a mask to clear the higher bits.
        higher_nibbles_vec.zmm = _mm512_and_si512(_mm512_srli_epi16(text_vec.zmm, 4), _mm512_set1_epi8(0x0f));
        bitset_even_vec.zmm = _mm512_shuffle_epi8(filter_even_vec.zmm, higher_nibbles_vec.zmm);
        bitset_odd_vec.zmm = _mm512_shuffle_epi8(filter_odd_vec.zmm, higher_nibbles_vec.zmm);
        //
        // At this point we can validate the `bitset_even_vec` and `bitset_odd_vec` contents like this:
        //
        //      for (sz_size_t i = 0; i != load_length; ++i) {
        //          sz_u8_t input = *(sz_u8_t const *)(text + i);
        //          sz_u8_t const *bitset_pointer = &filter->_u8s[0];
        //          sz_u8_t hi_nibble = input >> 4;
        //          sz_u8_t bitset_even = bitset_pointer[hi_nibble * 2];
        //          sz_u8_t bitset_odd = bitset_pointer[hi_nibble * 2 + 1];
        //          sz_assert_(bitset_even_vec.u8s[i] == bitset_even);
        //          sz_assert_(bitset_odd_vec.u8s[i] == bitset_odd);
        //      }
        //
        // TODO: Is this a good place for ternary logic?
        __mmask64 take_first = _mm512_cmplt_epi8_mask(lower_nibbles_vec.zmm, _mm512_set1_epi8(8));
        bitset_even_vec.zmm = _mm512_mask_blend_epi8(take_first, bitset_odd_vec.zmm, bitset_even_vec.zmm);
        __mmask64 matches_mask = _mm512_mask_test_epi8_mask(load_mask, bitset_even_vec.zmm, bitmask_vec.zmm);
        if (matches_mask) {
            int offset = sz_u64_ctz(matches_mask);
            return text + offset;
        }
        else { text += load_length, length -= load_length; }
    }

    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_rfind_byteset_icelake(sz_cptr_t text, sz_size_t length, sz_byteset_t const *filter) {
    return sz_rfind_byteset_serial(text, length, filter);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_ICELAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_ICELAKE_H_
