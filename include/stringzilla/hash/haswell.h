/**
 *  @brief Haswell (AVX2) backend for string hashing and checksums.
 *  @file include/stringzilla/hash/haswell.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_HASWELL_H_
#define STRINGZILLA_HASH_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/hash/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2")
#endif

SZ_PUBLIC sz_u64_t sz_bytesum_haswell(sz_cptr_t text, sz_size_t length) {
    // The naive implementation of this function is very simple.
    // It assumes the CPU is great at handling unaligned "loads".
    //
    // A typical AWS Skylake instance can have 32 KB x 2 blocks of L1 data cache per core,
    // 1 MB x 2 blocks of L2 cache per core, and one shared L3 cache buffer.
    // For now, let's avoid the cases beyond the L2 size.
    int is_huge = length > 1ull * 1024ull * 1024ull;

    // When the buffer is small, there isn't much to innovate.
    if (length <= 32) { return sz_bytesum_serial(text, length); }
    else if (!is_huge) {
        sz_u256_vec_t text_vec, sums_vec;
        sums_vec.ymm = _mm256_setzero_si256();
        for (; length >= 32; text += 32, length -= 32) {
            text_vec.ymm = _mm256_lddqu_si256((__m256i const *)text);
            sums_vec.ymm = _mm256_add_epi64(sums_vec.ymm, _mm256_sad_epu8(text_vec.ymm, _mm256_setzero_si256()));
        }
        // We can also avoid the final serial loop by fetching 32 bytes from end, in reverse direction,
        // and shifting the data within the register to zero-out the duplicate bytes.

        // Accumulating 256 bits is harder, as we need to extract the 128-bit sums first.
        __m128i low_xmm = _mm256_castsi256_si128(sums_vec.ymm);
        __m128i high_xmm = _mm256_extracti128_si256(sums_vec.ymm, 1);
        __m128i sums_xmm = _mm_add_epi64(low_xmm, high_xmm);
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_xmm);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_xmm, 1);
        sz_u64_t result = low + high;
        if (length) result += sz_bytesum_serial(text, length);
        return result;
    }
    // For gigantic buffers, exceeding typical L1 cache sizes, there are other tricks we can use.
    // Most notably, we can avoid populating the cache with the entire buffer, and instead traverse it in 2 directions.
    else {
        sz_size_t head_length = (32 - ((sz_size_t)text % 32)) % 32; // 31 or less.
        sz_size_t tail_length = (sz_size_t)(text + length) % 32;    // 31 or less.
        sz_size_t body_length = length - head_length - tail_length; // Multiple of 32.
        sz_u64_t result = 0;

        // Handle the tail before we start updating the `text` pointer
        while (tail_length) result += text[length - (tail_length--)];
        // Handle the head
        while (head_length--) result += *text++;

        sz_u256_vec_t text_vec, sums_vec;
        sums_vec.ymm = _mm256_setzero_si256();
        // Fill the aligned body of the buffer.
        if (!is_huge) {
            for (; body_length >= 32; text += 32, body_length -= 32) {
                text_vec.ymm = _mm256_stream_load_si256((__m256i const *)text);
                sums_vec.ymm = _mm256_add_epi64(sums_vec.ymm, _mm256_sad_epu8(text_vec.ymm, _mm256_setzero_si256()));
            }
        }
        // When the buffer is huge, we can traverse it in 2 directions.
        else {
            sz_u256_vec_t text_reversed_vec, sums_reversed_vec;
            sums_reversed_vec.ymm = _mm256_setzero_si256();
            for (; body_length >= 64; text += 32, body_length -= 64) {
                text_vec.ymm = _mm256_stream_load_si256((__m256i *)(text));
                sums_vec.ymm = _mm256_add_epi64(sums_vec.ymm, _mm256_sad_epu8(text_vec.ymm, _mm256_setzero_si256()));
                text_reversed_vec.ymm = _mm256_stream_load_si256((__m256i *)(text + body_length - 32));
                sums_reversed_vec.ymm = _mm256_add_epi64(
                    sums_reversed_vec.ymm, _mm256_sad_epu8(text_reversed_vec.ymm, _mm256_setzero_si256()));
            }
            if (body_length >= 32) {
                sz_assert_(body_length == 32);
                text_vec.ymm = _mm256_stream_load_si256((__m256i *)(text));
                sums_vec.ymm = _mm256_add_epi64(sums_vec.ymm, _mm256_sad_epu8(text_vec.ymm, _mm256_setzero_si256()));
                text += 32;
            }
            sums_vec.ymm = _mm256_add_epi64(sums_vec.ymm, sums_reversed_vec.ymm);
        }

        // Accumulating 256 bits is harder, as we need to extract the 128-bit sums first.
        __m128i low_xmm = _mm256_castsi256_si128(sums_vec.ymm);
        __m128i high_xmm = _mm256_extracti128_si256(sums_vec.ymm, 1);
        __m128i sums_xmm = _mm_add_epi64(low_xmm, high_xmm);
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_xmm);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_xmm, 1);
        result += low + high;
        return result;
    }
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

#endif // STRINGZILLA_HASH_HASWELL_H_
