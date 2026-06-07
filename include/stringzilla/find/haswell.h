/**
 *  @brief Haswell (AVX2) backend for substring & byte-set search.
 *  @file include/stringzilla/find/haswell.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_HASWELL_H_
#define STRINGZILLA_FIND_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/find/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  AVX2 implementation of the string search algorithms for Haswell processors and newer.
 *  Very minimalistic (compared to AVX-512), but still faster than the serial implementation.
 */
#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2")
#endif

SZ_PUBLIC sz_cptr_t sz_find_byte_haswell(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    int mask;
    sz_u256_vec_t h_vec, n_vec;
    n_vec.ymm = _mm256_set1_epi8(n[0]);

    while (h_length >= 32) {
        h_vec.ymm = _mm256_lddqu_si256((__m256i const *)h);
        mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_vec.ymm, n_vec.ymm));
        if (mask) return h + sz_u32_ctz(mask);
        h += 32, h_length -= 32;
    }

    return sz_find_byte_serial(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byte_haswell(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    int mask;
    sz_u256_vec_t h_vec, n_vec;
    n_vec.ymm = _mm256_set1_epi8(n[0]);

    while (h_length >= 32) {
        h_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h + h_length - 32));
        mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_vec.ymm, n_vec.ymm));
        if (mask) return h + h_length - 1 - sz_u32_clz(mask);
        h_length -= 32;
    }

    return sz_rfind_byte_serial(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_find_haswell(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_find_byte_haswell(h, h_length, n);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into YMM registers.
    sz_u32_vec_t matches_vec;
    sz_u256_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.ymm = _mm256_set1_epi8(n[offset_first]);
    n_mid_vec.ymm = _mm256_set1_epi8(n[offset_mid]);
    n_last_vec.ymm = _mm256_set1_epi8(n[offset_last]);

    // Scan through the string.
    for (; h_length >= n_length + 32; h += 32, h_length -= 32) {
        h_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h + offset_first));
        h_mid_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h + offset_mid));
        h_last_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h + offset_last));
        matches_vec.i32 = //
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_first_vec.ymm, n_first_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_mid_vec.ymm, n_mid_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_last_vec.ymm, n_last_vec.ymm));
        while (matches_vec.u32) {
            int potential_offset = sz_u32_ctz(matches_vec.u32);
            if (sz_equal_haswell(h + potential_offset, n, n_length)) return h + potential_offset;
            matches_vec.u32 &= matches_vec.u32 - 1;
        }
    }

    return sz_find_serial(h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_rfind_haswell(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_rfind_byte_haswell(h, h_length, n);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into YMM registers.
    sz_u32_vec_t matches_vec;
    sz_u256_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.ymm = _mm256_set1_epi8(n[offset_first]);
    n_mid_vec.ymm = _mm256_set1_epi8(n[offset_mid]);
    n_last_vec.ymm = _mm256_set1_epi8(n[offset_last]);

    // Scan through the string.
    sz_cptr_t h_reversed;
    for (; h_length >= n_length + 32; h_length -= 32) {
        h_reversed = h + h_length - n_length - 32 + 1;
        h_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h_reversed + offset_first));
        h_mid_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h_reversed + offset_mid));
        h_last_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h_reversed + offset_last));
        matches_vec.i32 = //
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_first_vec.ymm, n_first_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_mid_vec.ymm, n_mid_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_last_vec.ymm, n_last_vec.ymm));
        while (matches_vec.u32) {
            int potential_offset = sz_u32_clz(matches_vec.u32);
            if (sz_equal_haswell(h + h_length - n_length - potential_offset, n, n_length))
                return h + h_length - n_length - potential_offset;
            matches_vec.u32 &= ~(1u << (31 - potential_offset));
        }
    }

    return sz_rfind_serial(h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_find_byteset_haswell(sz_cptr_t text, sz_size_t length, sz_byteset_t const *filter) {

    // Let's unzip even and odd elements and replicate them into both lanes of the YMM register.
    // That way when we invoke `_mm256_shuffle_epi8` we can use the same mask for both lanes.
    // Load the 32-byte filter as two 16-byte halves, separate even/odd bytes, pack, and broadcast to YMM.
    sz_u128_vec_t byte_mask_vec;
    sz_u128_vec_t filter_lo_vec, filter_hi_vec;
    sz_u128_vec_t lo_evens_vec, hi_evens_vec;
    sz_u128_vec_t lo_odds_vec, hi_odds_vec;
    sz_u128_vec_t evens_xmm_vec, odds_xmm_vec;
    sz_u256_vec_t filter_even_vec, filter_odd_vec;

    sz_u256_vec_t text_vec;
    sz_u256_vec_t matches_vec;
    sz_u256_vec_t lower_nibbles_vec, higher_nibbles_vec;
    sz_u256_vec_t bitset_even_vec, bitset_odd_vec;
    sz_u256_vec_t bitmask_vec, bitmask_lookup_vec;

    byte_mask_vec.xmm = _mm_set1_epi16(0x00ff);

    filter_lo_vec.xmm = _mm_lddqu_si128((__m128i const *)(filter));
    filter_hi_vec.xmm = _mm_lddqu_si128((__m128i const *)(filter) + 1);
    lo_evens_vec.xmm = _mm_and_si128(filter_lo_vec.xmm, byte_mask_vec.xmm);
    hi_evens_vec.xmm = _mm_and_si128(filter_hi_vec.xmm, byte_mask_vec.xmm);
    lo_odds_vec.xmm = _mm_srli_epi16(filter_lo_vec.xmm, 8);
    hi_odds_vec.xmm = _mm_srli_epi16(filter_hi_vec.xmm, 8);

    evens_xmm_vec.xmm = _mm_packus_epi16(lo_evens_vec.xmm, hi_evens_vec.xmm);
    odds_xmm_vec.xmm = _mm_packus_epi16(lo_odds_vec.xmm, hi_odds_vec.xmm);
    filter_even_vec.ymm = _mm256_set_m128i(evens_xmm_vec.xmm, evens_xmm_vec.xmm);
    filter_odd_vec.ymm = _mm256_set_m128i(odds_xmm_vec.xmm, odds_xmm_vec.xmm);

    bitmask_lookup_vec.ymm = _mm256_set_epi8(                       //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1);

    while (length >= 32) {
        // The following algorithm is a transposed equivalent of the "SIMD-ized check which bytes are in a set"
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
        text_vec.ymm = _mm256_lddqu_si256((__m256i const *)text);
        lower_nibbles_vec.ymm = _mm256_and_si256(text_vec.ymm, _mm256_set1_epi8(0x0f));
        bitmask_vec.ymm = _mm256_shuffle_epi8(bitmask_lookup_vec.ymm, lower_nibbles_vec.ymm);
        //
        // At this point we can validate the `bitmask_vec` contents like this:
        //
        //      for (sz_size_t i = 0; i != 32; ++i) {
        //          sz_u8_t input = *(sz_u8_t const *)(text + i);
        //          sz_u8_t lo_nibble = input & 0x0f;
        //          sz_u8_t bitmask = (1 << (lo_nibble & 0x7));
        //          sz_assert_(bitmask_vec.u8s[i] == bitmask);
        //      }
        //
        // Shift right every byte by 4 bits.
        // There is no `_mm256_srli_epi8` intrinsic, so we have to use `_mm256_srli_epi16`
        // and combine it with a mask to clear the higher bits.
        higher_nibbles_vec.ymm = _mm256_and_si256(_mm256_srli_epi16(text_vec.ymm, 4), _mm256_set1_epi8(0x0f));
        bitset_even_vec.ymm = _mm256_shuffle_epi8(filter_even_vec.ymm, higher_nibbles_vec.ymm);
        bitset_odd_vec.ymm = _mm256_shuffle_epi8(filter_odd_vec.ymm, higher_nibbles_vec.ymm);
        //
        // At this point we can validate the `bitset_even_vec` and `bitset_odd_vec` contents like this:
        //
        //      for (sz_size_t i = 0; i != 32; ++i) {
        //          sz_u8_t input = *(sz_u8_t const *)(text + i);
        //          sz_u8_t const *bitset_ptr = &filter->_u8s[0];
        //          sz_u8_t hi_nibble = input >> 4;
        //          sz_u8_t bitset_even = bitset_ptr[hi_nibble * 2];
        //          sz_u8_t bitset_odd = bitset_ptr[hi_nibble * 2 + 1];
        //          sz_assert_(bitset_even_vec.u8s[i] == bitset_even);
        //          sz_assert_(bitset_odd_vec.u8s[i] == bitset_odd);
        //      }
        //
        __m256i take_first = _mm256_cmpgt_epi8(_mm256_set1_epi8(8), lower_nibbles_vec.ymm);
        bitset_even_vec.ymm = _mm256_blendv_epi8(bitset_odd_vec.ymm, bitset_even_vec.ymm, take_first);

        // It would have been great to have an instruction that tests the bits and then broadcasts
        // the matching bit into all bits in that byte. But we don't have that, so we have to
        // `and`, `cmpeq`, `movemask`, and then invert at the end...
        matches_vec.ymm = _mm256_and_si256(bitset_even_vec.ymm, bitmask_vec.ymm);
        matches_vec.ymm = _mm256_cmpeq_epi8(matches_vec.ymm, _mm256_setzero_si256());
        int matches_mask = ~_mm256_movemask_epi8(matches_vec.ymm);
        if (matches_mask) {
            int offset = sz_u32_ctz(matches_mask);
            return text + offset;
        }
        else { text += 32, length -= 32; }
    }

    return sz_find_byteset_serial(text, length, filter);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byteset_haswell(sz_cptr_t text, sz_size_t length, sz_byteset_t const *filter) {
    return sz_rfind_byteset_serial(text, length, filter);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_HASWELL_H_
