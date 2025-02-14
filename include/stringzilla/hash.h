/**
 *  @brief  Hardware-accelerated string hashing and checksums.
 *  @file   hash.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs:
 *
 *  - `sz_checksum` - for byte-level 64-bit unsigned checksums.
 *  - `sz_hash` - for 64-bit single-shot hashing.
 *  - `sz_generate` - populating buffers with random data.
 */
#ifndef STRINGZILLA_HASH_H_
#define STRINGZILLA_HASH_H_

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief  Computes the 64-bit check-sum of bytes in a string.
 *          Similar to `std::ranges::accumulate`.
 *
 *  @param text     String to aggregate.
 *  @param length   Number of bytes in the text.
 *  @return         64-bit unsigned value.
 */
SZ_DYNAMIC sz_u64_t sz_checksum(sz_cptr_t text, sz_size_t length);

/**
 *  @brief  Computes the 64-bit unsigned hash of a string. Fairly fast for short strings,
 *          simple implementation, and supports rolling computation, reused in other APIs.
 *          Similar to `std::hash` in C++.
 *
 *  @param text     String to hash.
 *  @param length   Number of bytes in the text.
 *  @return         64-bit hash value.
 *
 *  @see    sz_hashes, sz_hashes_fingerprint, sz_hashes_intersection
 */
SZ_PUBLIC sz_u64_t sz_hash(sz_cptr_t text, sz_size_t length) {
    sz_unused(text && length);
    return 0;
}

/**
 *  @brief  Generates a random string for a given alphabet, avoiding integer division and modulo operations.
 *          Similar to `text[i] = alphabet[rand() % cardinality]`.
 *
 *  The modulo operation is expensive, and should be avoided in performance-critical code.
 *  We avoid it using small lookup tables and replacing it with a multiplication and shifts, similar to `libdivide`.
 *  Alternative algorithms would include:
 *      - Montgomery form: https://en.algorithmica.org/hpc/number-theory/montgomery/
 *      - Barret reduction: https://www.nayuki.io/page/barrett-reduction-algorithm
 *      - Lemire's trick: https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
 *
 *  @param alphabet     Set of characters to sample from.
 *  @param cardinality  Number of characters to sample from.
 *  @param text         Output string, can point to the same address as ::text.
 *  @param generate     Callback producing random numbers given the generator state.
 *  @param generator    Generator state, can be a pointer to a seed, or a pointer to a random number generator.
 */
SZ_DYNAMIC void sz_generate(sz_cptr_t alphabet, sz_size_t cardinality, sz_ptr_t text, sz_size_t length,
                            sz_random_generator_t generate, void *generator);

/** @copydoc sz_checksum */
SZ_PUBLIC sz_u64_t sz_checksum_serial(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hash */
SZ_PUBLIC sz_u64_t sz_hash_serial(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_generate */
SZ_PUBLIC void sz_generate_serial( //
    sz_cptr_t alphabet, sz_size_t cardinality, sz_ptr_t text, sz_size_t length, sz_random_generator_t generate,
    void *generator) {
    sz_unused(alphabet && cardinality && text && length && generate && generator);
}

#pragma endregion // Core API

#pragma region Serial Implementation

SZ_PUBLIC sz_u64_t sz_checksum_serial(sz_cptr_t text, sz_size_t length) {
    sz_u64_t checksum = 0;
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *text_end = text_u8 + length;
    for (; text_u8 != text_end; ++text_u8) checksum += *text_u8;
    return checksum;
}

/*
 *  One hardware-accelerated way of mixing hashes can be CRC, but it's only implemented for 32-bit values.
 *  Using a Boost-like mixer works very poorly in such case:
 *
 *       hash_first ^ (hash_second + 0x517cc1b727220a95 + (hash_first << 6) + (hash_first >> 2));
 *
 *  Let's stick to the Fibonacci hash trick using the golden ratio.
 *  https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/
 */
#define _sz_hash_mix(first, second) ((first * 11400714819323198485ull) ^ (second * 11400714819323198485ull))
#define _sz_shift_low(x) (x)
#define _sz_shift_high(x) ((x + 77ull) & 0xFFull)
#define _sz_prime_mod(x) (x % SZ_U64_MAX_PRIME)

SZ_PUBLIC sz_u64_t sz_hash_serial(sz_cptr_t start, sz_size_t length) {

    sz_u64_t hash_low = 0;
    sz_u64_t hash_high = 0;
    sz_u8_t const *text = (sz_u8_t const *)start;
    sz_u8_t const *text_end = text + length;

    switch (length) {
    case 0: return 0;

    // Texts under 7 bytes long are definitely below the largest prime.
    case 1:
        hash_low = _sz_shift_low(text[0]);
        hash_high = _sz_shift_high(text[0]);
        break;
    case 2:
        hash_low = _sz_shift_low(text[0]) * 31ull + _sz_shift_low(text[1]);
        hash_high = _sz_shift_high(text[0]) * 257ull + _sz_shift_high(text[1]);
        break;
    case 3:
        hash_low = _sz_shift_low(text[0]) * 31ull * 31ull + //
                   _sz_shift_low(text[1]) * 31ull +         //
                   _sz_shift_low(text[2]);
        hash_high = _sz_shift_high(text[0]) * 257ull * 257ull + //
                    _sz_shift_high(text[1]) * 257ull +          //
                    _sz_shift_high(text[2]);
        break;
    case 4:
        hash_low = _sz_shift_low(text[0]) * 31ull * 31ull * 31ull + //
                   _sz_shift_low(text[1]) * 31ull * 31ull +         //
                   _sz_shift_low(text[2]) * 31ull +                 //
                   _sz_shift_low(text[3]);
        hash_high = _sz_shift_high(text[0]) * 257ull * 257ull * 257ull + //
                    _sz_shift_high(text[1]) * 257ull * 257ull +          //
                    _sz_shift_high(text[2]) * 257ull +                   //
                    _sz_shift_high(text[3]);
        break;
    case 5:
        hash_low = _sz_shift_low(text[0]) * 31ull * 31ull * 31ull * 31ull + //
                   _sz_shift_low(text[1]) * 31ull * 31ull * 31ull +         //
                   _sz_shift_low(text[2]) * 31ull * 31ull +                 //
                   _sz_shift_low(text[3]) * 31ull +                         //
                   _sz_shift_low(text[4]);
        hash_high = _sz_shift_high(text[0]) * 257ull * 257ull * 257ull * 257ull + //
                    _sz_shift_high(text[1]) * 257ull * 257ull * 257ull +          //
                    _sz_shift_high(text[2]) * 257ull * 257ull +                   //
                    _sz_shift_high(text[3]) * 257ull +                            //
                    _sz_shift_high(text[4]);
        break;
    case 6:
        hash_low = _sz_shift_low(text[0]) * 31ull * 31ull * 31ull * 31ull * 31ull + //
                   _sz_shift_low(text[1]) * 31ull * 31ull * 31ull * 31ull +         //
                   _sz_shift_low(text[2]) * 31ull * 31ull * 31ull +                 //
                   _sz_shift_low(text[3]) * 31ull * 31ull +                         //
                   _sz_shift_low(text[4]) * 31ull +                                 //
                   _sz_shift_low(text[5]);
        hash_high = _sz_shift_high(text[0]) * 257ull * 257ull * 257ull * 257ull * 257ull + //
                    _sz_shift_high(text[1]) * 257ull * 257ull * 257ull * 257ull +          //
                    _sz_shift_high(text[2]) * 257ull * 257ull * 257ull +                   //
                    _sz_shift_high(text[3]) * 257ull * 257ull +                            //
                    _sz_shift_high(text[4]) * 257ull +                                     //
                    _sz_shift_high(text[5]);
        break;
    case 7:
        hash_low = _sz_shift_low(text[0]) * 31ull * 31ull * 31ull * 31ull * 31ull * 31ull + //
                   _sz_shift_low(text[1]) * 31ull * 31ull * 31ull * 31ull * 31ull +         //
                   _sz_shift_low(text[2]) * 31ull * 31ull * 31ull * 31ull +                 //
                   _sz_shift_low(text[3]) * 31ull * 31ull * 31ull +                         //
                   _sz_shift_low(text[4]) * 31ull * 31ull +                                 //
                   _sz_shift_low(text[5]) * 31ull +                                         //
                   _sz_shift_low(text[6]);
        hash_high = _sz_shift_high(text[0]) * 257ull * 257ull * 257ull * 257ull * 257ull * 257ull + //
                    _sz_shift_high(text[1]) * 257ull * 257ull * 257ull * 257ull * 257ull +          //
                    _sz_shift_high(text[2]) * 257ull * 257ull * 257ull * 257ull +                   //
                    _sz_shift_high(text[3]) * 257ull * 257ull * 257ull +                            //
                    _sz_shift_high(text[4]) * 257ull * 257ull +                                     //
                    _sz_shift_high(text[5]) * 257ull +                                              //
                    _sz_shift_high(text[6]);
        break;
    default:
        // Unroll the first seven cycles:
        hash_low = hash_low * 31ull + _sz_shift_low(text[0]);
        hash_high = hash_high * 257ull + _sz_shift_high(text[0]);
        hash_low = hash_low * 31ull + _sz_shift_low(text[1]);
        hash_high = hash_high * 257ull + _sz_shift_high(text[1]);
        hash_low = hash_low * 31ull + _sz_shift_low(text[2]);
        hash_high = hash_high * 257ull + _sz_shift_high(text[2]);
        hash_low = hash_low * 31ull + _sz_shift_low(text[3]);
        hash_high = hash_high * 257ull + _sz_shift_high(text[3]);
        hash_low = hash_low * 31ull + _sz_shift_low(text[4]);
        hash_high = hash_high * 257ull + _sz_shift_high(text[4]);
        hash_low = hash_low * 31ull + _sz_shift_low(text[5]);
        hash_high = hash_high * 257ull + _sz_shift_high(text[5]);
        hash_low = hash_low * 31ull + _sz_shift_low(text[6]);
        hash_high = hash_high * 257ull + _sz_shift_high(text[6]);
        text += 7;

        // Iterate throw the rest with the modulus:
        for (; text != text_end; ++text) {
            hash_low = hash_low * 31ull + _sz_shift_low(text[0]);
            hash_high = hash_high * 257ull + _sz_shift_high(text[0]);
            // Wrap the hashes around:
            hash_low = _sz_prime_mod(hash_low);
            hash_high = _sz_prime_mod(hash_high);
        }
        break;
    }

    return _sz_hash_mix(hash_low, hash_high);
}

#undef _sz_shift_low
#undef _sz_shift_high
#undef _sz_hash_mix
#undef _sz_prime_mod

#pragma endregion // Serial Implementation

/*  AVX2 implementation of the string search algorithms for Haswell processors and newer.
 *  Very minimalistic (compared to AVX-512), but still faster than the serial implementation.
 */
#pragma region Haswell Implementation
#if SZ_USE_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2")
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)

SZ_PUBLIC sz_u64_t sz_checksum_haswell(sz_cptr_t text, sz_size_t length) {
    // The naive implementation of this function is very simple.
    // It assumes the CPU is great at handling unaligned "loads".
    //
    // A typical AWS Skylake instance can have 32 KB x 2 blocks of L1 data cache per core,
    // 1 MB x 2 blocks of L2 cache per core, and one shared L3 cache buffer.
    // For now, let's avoid the cases beyond the L2 size.
    int is_huge = length > 1ull * 1024ull * 1024ull;

    // When the buffer is small, there isn't much to innovate.
    if (length <= 32) { return sz_checksum_serial(text, length); }
    else if (!is_huge) {
        sz_u256_vec_t text_vec, sums_vec;
        sums_vec.ymm = _mm256_setzero_si256();
        for (; length >= 32; text += 32, length -= 32) {
            text_vec.ymm = _mm256_lddqu_si256((__m256i const *)text);
            sums_vec.ymm = _mm256_add_epi64(sums_vec.ymm, _mm256_sad_epu8(text_vec.ymm, _mm256_setzero_si256()));
        }
        // Accumulating 256 bits is harder, as we need to extract the 128-bit sums first.
        __m128i low_xmm = _mm256_castsi256_si128(sums_vec.ymm);
        __m128i high_xmm = _mm256_extracti128_si256(sums_vec.ymm, 1);
        __m128i sums_xmm = _mm_add_epi64(low_xmm, high_xmm);
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_xmm);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_xmm, 1);
        sz_u64_t result = low + high;
        if (length) result += sz_checksum_serial(text, length);
        return result;
    }
    // For gigantic buffers, exceeding typical L1 cache sizes, there are other tricks we can use.
    // Most notably, we can avoid populating the cache with the entire buffer, and instead traverse it in 2 directions.
    else {
        sz_size_t head_length = (32 - ((sz_size_t)text % 32)) % 32; // 31 or less.
        sz_size_t tail_length = (sz_size_t)(text + length) % 32;    // 31 or less.
        sz_size_t body_length = length - head_length - tail_length; // Multiple of 32.
        sz_u64_t result = 0;

        // Handle the head
        while (head_length--) result += *text++;
        // Handle the tail
        while (tail_length) result += text[length - (tail_length--) - 1];

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
                _sz_assert(body_length == 32);
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

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_HASWELL
#pragma endregion // Haswell Implementation

/*  AVX512 implementation of the string hashing algorithms for Skylake and newer CPUs.
 *  Includes extensions: F, CD, ER, PF, VL, DQ, BW.
 *
 *  This is the "starting level" for the advanced algorithms using K-mask registers on x86.
 */
#pragma region Skylake Implementation
#if SZ_USE_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2")
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2"))), apply_to = function)

SZ_PUBLIC sz_u64_t sz_checksum_skylake(sz_cptr_t text, sz_size_t length) {
    // The naive implementation of this function is very simple.
    // It assumes the CPU is great at handling unaligned "loads".
    //
    // A typical AWS Sapphire Rapids instance can have 48 KB x 2 blocks of L1 data cache per core,
    // 2 MB x 2 blocks of L2 cache per core, and one shared 60 MB buffer of L3 cache.
    // With two strings, we may consider the overall workload huge, if each exceeds 1 MB in length.
    int const is_huge = length >= 1ull * 1024ull * 1024ull;
    sz_u512_vec_t text_vec, sums_vec;

    // When the buffer is small, there isn't much to innovate.
    // Separately handling even smaller payloads doesn't increase performance even on synthetic benchmarks.
    if (length <= 16) {
        __mmask16 mask = _sz_u16_mask_until(length);
        text_vec.xmms[0] = _mm_maskz_loadu_epi8(mask, text);
        sums_vec.xmms[0] = _mm_sad_epu8(text_vec.xmms[0], _mm_setzero_si128());
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_vec.xmms[0]);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_vec.xmms[0], 1);
        return low + high;
    }
    else if (length <= 32) {
        __mmask32 mask = _sz_u32_mask_until(length);
        text_vec.ymms[0] = _mm256_maskz_loadu_epi8(mask, text);
        sums_vec.ymms[0] = _mm256_sad_epu8(text_vec.ymms[0], _mm256_setzero_si256());
        // Accumulating 256 bits is harder, as we need to extract the 128-bit sums first.
        __m128i low_xmm = _mm256_castsi256_si128(sums_vec.ymms[0]);
        __m128i high_xmm = _mm256_extracti128_si256(sums_vec.ymms[0], 1);
        __m128i sums_xmm = _mm_add_epi64(low_xmm, high_xmm);
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_xmm);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_xmm, 1);
        return low + high;
    }
    else if (length <= 64) {
        __mmask64 mask = _sz_u64_mask_until(length);
        text_vec.zmm = _mm512_maskz_loadu_epi8(mask, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512());
        return _mm512_reduce_add_epi64(sums_vec.zmm);
    }
    // For large buffers, fitting into L1 cache sizes, there are other tricks we can use.
    //
    // 1. Moving in both directions to maximize the throughput, when fetching from multiple
    //    memory pages. Also helps with cache set-associativity issues, as we won't always
    //    be fetching the same buckets in the lookup table.
    //
    // Bidirectional traversal generally adds about 10% to such algorithms.
    else if (!is_huge) {
        sz_size_t head_length = (64 - ((sz_size_t)text % 64)) % 64; // 63 or less.
        sz_size_t tail_length = (sz_size_t)(text + length) % 64;    // 63 or less.
        sz_size_t body_length = length - head_length - tail_length; // Multiple of 64.
        _sz_assert(body_length % 64 == 0 && head_length < 64 && tail_length < 64);
        __mmask64 head_mask = _sz_u64_mask_until(head_length);
        __mmask64 tail_mask = _sz_u64_mask_until(tail_length);

        text_vec.zmm = _mm512_maskz_loadu_epi8(head_mask, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512());
        for (text += head_length; body_length >= 64; text += 64, body_length -= 64) {
            text_vec.zmm = _mm512_load_si512((__m512i const *)text);
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
        }
        text_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask, text);
        sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
        return _mm512_reduce_add_epi64(sums_vec.zmm);
    }
    // For gigantic buffers, exceeding typical L1 cache sizes, there are other tricks we can use.
    //
    // 1. Using non-temporal loads to avoid polluting the cache.
    // 2. Prefetching the next cache line, to avoid stalling the CPU. This generally useless
    //    for predictable patterns, so disregard this advice.
    //
    // Bidirectional traversal generally adds about 10% to such algorithms.
    else {
        sz_u512_vec_t text_reversed_vec, sums_reversed_vec;
        sz_size_t head_length = (64 - ((sz_size_t)text % 64)) % 64;
        sz_size_t tail_length = (sz_size_t)(text + length) % 64;
        sz_size_t body_length = length - head_length - tail_length;
        __mmask64 head_mask = _sz_u64_mask_until(head_length);
        __mmask64 tail_mask = _sz_u64_mask_until(tail_length);

        text_vec.zmm = _mm512_maskz_loadu_epi8(head_mask, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512());
        text_reversed_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask, text + head_length + body_length);
        sums_reversed_vec.zmm = _mm512_sad_epu8(text_reversed_vec.zmm, _mm512_setzero_si512());

        // Now in the main loop, we can use non-temporal loads, performing the operation in both directions.
        for (text += head_length; body_length >= 128; text += 64, body_length -= 128) {
            text_vec.zmm = _mm512_stream_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
            text_reversed_vec.zmm = _mm512_stream_load_si512((__m512i *)(text + body_length - 64));
            sums_reversed_vec.zmm =
                _mm512_add_epi64(sums_reversed_vec.zmm, _mm512_sad_epu8(text_reversed_vec.zmm, _mm512_setzero_si512()));
        }
        if (body_length >= 64) {
            text_vec.zmm = _mm512_stream_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
        }

        return _mm512_reduce_add_epi64(_mm512_add_epi64(sums_vec.zmm, sums_reversed_vec.zmm));
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_SKYLAKE
#pragma endregion // Skylake Implementation

/*  AVX512 implementation of the string search algorithms for Ice Lake and newer CPUs.
 *  Includes extensions:
 *  - 2017 Skylake: F, CD, ER, PF, VL, DQ, BW,
 *  - 2018 CannonLake: IFMA, VBMI,
 *  - 2019 Ice Lake: VPOPCNTDQ, VNNI, VBMI2, BITALG, GFNI, VPCLMULQDQ, VAES.
 */
#pragma region Ice Lake Implementation
#if SZ_USE_ICE
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vnni", "bmi", "bmi2")
#pragma clang attribute push(                                                                         \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vnni,bmi,bmi2"))), \
    apply_to = function)

SZ_PUBLIC sz_u64_t sz_checksum_ice(sz_cptr_t text, sz_size_t length) {
    // The naive implementation of this function is very simple.
    // It assumes the CPU is great at handling unaligned "loads".
    //
    // A typical AWS Sapphire Rapids instance can have 48 KB x 2 blocks of L1 data cache per core,
    // 2 MB x 2 blocks of L2 cache per core, and one shared 60 MB buffer of L3 cache.
    // With two strings, we may consider the overall workload huge, if each exceeds 1 MB in length.
    int const is_huge = length >= 1ull * 1024ull * 1024ull;

    // When the buffer is small, there isn't much to innovate.
    // Separately handling even smaller payloads doesn't increase performance even on synthetic benchmarks.
    if (length <= 16) {
        sz_u512_vec_t text_vec, sums_vec;
        __mmask16 mask = _sz_u16_mask_until(length);
        text_vec.xmms[0] = _mm_maskz_loadu_epi8(mask, text);
        sums_vec.xmms[0] = _mm_sad_epu8(text_vec.xmms[0], _mm_setzero_si128());
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_vec.xmms[0]);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_vec.xmms[0], 1);
        return low + high;
    }
    else if (length <= 32) {
        sz_u512_vec_t text_vec, sums_vec;
        __mmask32 mask = _sz_u32_mask_until(length);
        text_vec.ymms[0] = _mm256_maskz_loadu_epi8(mask, text);
        sums_vec.ymms[0] = _mm256_sad_epu8(text_vec.ymms[0], _mm256_setzero_si256());
        // Accumulating 256 bits is harder, as we need to extract the 128-bit sums first.
        __m128i low_xmm = _mm256_castsi256_si128(sums_vec.ymms[0]);
        __m128i high_xmm = _mm256_extracti128_si256(sums_vec.ymms[0], 1);
        __m128i sums_xmm = _mm_add_epi64(low_xmm, high_xmm);
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_xmm);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_xmm, 1);
        return low + high;
    }
    else if (length <= 64) {
        sz_u512_vec_t text_vec, sums_vec;
        __mmask64 mask = _sz_u64_mask_until(length);
        text_vec.zmm = _mm512_maskz_loadu_epi8(mask, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512());
        return _mm512_reduce_add_epi64(sums_vec.zmm);
    }
    // For large buffers, fitting into L1 cache sizes, there are other tricks we can use.
    // The naive idea may be to use `_mm512_unpacklo_epi8`-like operations to upcast to larger integers,
    // and then accumulating with `_mm512_madd_epi16`-like addition. This however, may require multiple levels
    // of upcasting, and bottlenecks port 5 on most Intel machines.
    //
    // There is, however, a way to further optimize our `_mm512_sad_epu8`-based solution:
    //
    // 1. Moving in both directions to maximize the throughput, when fetching from multiple
    //    memory pages. Also helps with cache set-associativity issues, as we won't always
    //    be fetching the same buckets in the lookup table.
    // 2. Port-level parallelism, can be used to hide the latency of expensive SIMD instructions.
    //    - `VPSADBW (ZMM, ZMM, ZMM)` combination with `VPADDQ (ZMM, ZMM, ZMM)`:
    //        - On Ice Lake, the `VPSADBW` is 3 cycles on port 5; the `VPADDQ` is 1 cycle on ports 0/5.
    //        - On Zen 4, the `VPSADBW` is 3 cycles on ports 0/1; the `VPADDQ` is 1 cycle on ports 0/1/2/3.
    //    - `VPDPBUSDS (ZMM, ZMM, ZMM)`:
    //        - On Ice Lake, the `VPDPBUSDS` is 5 cycles on port 0.
    //        - On Zen 4, the `VPDPBUSDS` is 4 cycles on ports 0/1.
    //
    // Bidirectional traversal generally adds about 10% to such algorithms.
    // Port level parallelism can yield more, but remember that one of the instructions accumulates
    // with 32-bit integers and the other one will be using 64-bit integers.
    //
    // The problem, however, is that the element-wise `VPADDQ` may also be dispatched on port 0.
    // So we can use one more trick! We can delay the element-wise addition by using `VPERMILPS` on port 5,
    // and introducing a dependency chain that will make the CPU scheduler delay the `VPADDQ` instruction.
    else if (!is_huge) {
        sz_size_t head_length = (64 - ((sz_size_t)text % 64)) % 64; // 63 or less.
        sz_size_t tail_length = (sz_size_t)(text + length) % 64;    // 63 or less.
        sz_size_t body_length = length - head_length - tail_length; // Multiple of 64.
        _sz_assert(body_length % 64 == 0 && head_length < 64 && tail_length < 64);
        __mmask64 head_mask = _sz_u64_mask_until(head_length);
        __mmask64 tail_mask = _sz_u64_mask_until(tail_length);
        __m512i swap_nearby = _mm512_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
        __mmask16 interleave_mask = 0b1010101010101010;

        sz_u512_vec_t zeros_vec, ones_vec;
        zeros_vec.zmm = _mm512_setzero_si512();
        ones_vec.zmm = _mm512_set1_epi8(1);

        // Take care of the unaligned head and tail!
        sz_u512_vec_t text0_vec, text1_vec, text2_vec, text3_vec, sums02_i32even_vec, sums13_i32_vec;
        text0_vec.zmm = _mm512_maskz_loadu_epi8(head_mask, text);
        sums02_i32even_vec.zmm = _mm512_sad_epu8(text0_vec.zmm, zeros_vec.zmm);
        text3_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask, text + head_length + body_length);
        sums13_i32_vec.zmm = _mm512_dpbusds_epi32(zeros_vec.zmm, text3_vec.zmm, ones_vec.zmm);

        // We will also need 2 temporary registers to store partial 64-bit sums for the normal direction.
        sz_u512_vec_t sums0_i32_vec, sums2_i32_vec;

        // Now in the main loop, we can use aligned loads, performing the operation in both directions.
        for (text += head_length; body_length >= 256; text += 128, body_length -= 256) {
            // Fetch 2 chunks of 64 bytes in each direction ~ 256 bytes per cycle.
            text0_vec.zmm = _mm512_load_si512((__m512i *)(text));
            text1_vec.zmm = _mm512_load_si512((__m512i *)(text + 64));
            text2_vec.zmm = _mm512_load_si512((__m512i *)(text + body_length - 128));
            text3_vec.zmm = _mm512_load_si512((__m512i *)(text + body_length - 64));
            // § Port 0 section
            //
            // Multiply each byte in `text2_vec` and `text3_vec` by 1 and accumulate into 32-bit sums.
            sums13_i32_vec.zmm = _mm512_dpbusds_epi32(sums13_i32_vec.zmm, text1_vec.zmm, ones_vec.zmm);
            sums13_i32_vec.zmm = _mm512_dpbusds_epi32(sums13_i32_vec.zmm, text3_vec.zmm, ones_vec.zmm);
            // § Port 5 section
            //
            // The `_mm512_sad_epu8` will take place on port 5.
            // It will aggregate absolute byte differences into 8x 16-bit integers.
            // It will keep a quarter of bytes in `sums0_i32_vec` and `sums2_i32_vec` free.
            // Let's accumulate them as 32-bit integers, leaving half of 32-bit words free.
            //
            // The subsequent `_mm512_add_epi32` or `_mm512_add_epi64` can take place on port 0 or 5,
            // so we need to fool the scheduler so he doesn't even consider it 😎
            // To delay addition we can make the operation dependent on more instructions bound to retire
            // on port 5! That can be something like a `_mm512_permute_ps`, which will take just 1 cycle.
            sums0_i32_vec.zmm = _mm512_sad_epu8(text0_vec.zmm, zeros_vec.zmm);
            sums2_i32_vec.zmm = _mm512_sad_epu8(text2_vec.zmm, zeros_vec.zmm);
            // This is practically a no-op, because we simply swap nearby values in the register,
            // that will be anyways accumulated down the road. But the difference between having it and not is:
            // - 11.25 GB/s for `VPSADBW`-only Skylake version
            // - 11.63 GB/s with `VPDPBUSDS`
            sums0_i32_vec.zmm = _mm512_castps_si512(_mm512_mask_permutevar_ps( //
                _mm512_castsi512_ps(sums0_i32_vec.zmm), interleave_mask,       //
                _mm512_castsi512_ps(sums2_i32_vec.zmm), swap_nearby));
            sums02_i32even_vec.zmm = _mm512_add_epi32(sums02_i32even_vec.zmm, sums0_i32_vec.zmm);
        }
        // There may be an aligned chunk of 64 bytes left.
        for (; body_length >= 64; text += 64, body_length -= 64) {
            text0_vec.zmm = _mm512_load_si512((__m512i *)(text));
            sums02_i32even_vec.zmm =
                _mm512_add_epi32(sums02_i32even_vec.zmm, _mm512_sad_epu8(text0_vec.zmm, zeros_vec.zmm));
        }

        return _mm512_reduce_add_epi32(sums02_i32even_vec.zmm) + _mm512_reduce_add_epi32(sums13_i32_vec.zmm);
    }
    // For gigantic buffers, exceeding typical L1 cache sizes, there are other tricks we can use.
    //
    // 1. Using non-temporal loads to avoid polluting the cache.
    //    That `VMOVNTDQA (ZMM, M512)` instruction has the "1*p05+1*p23" port signature on Ice Lake.
    // 2. Prefetching the next cache line, to avoid stalling the CPU. This generally useless
    //    for predictable patterns, so disregard this advice.
    //
    // Bidirectional traversal generally adds about 10% to such algorithms.
    else {
        sz_u512_vec_t text_reversed_vec, sums_reversed_vec;
        sz_size_t head_length = (64 - ((sz_size_t)text % 64)) % 64;
        sz_size_t tail_length = (sz_size_t)(text + length) % 64;
        sz_size_t body_length = length - head_length - tail_length;
        __mmask64 head_mask = _sz_u64_mask_until(head_length);
        __mmask64 tail_mask = _sz_u64_mask_until(tail_length);

        sz_u512_vec_t text_vec, sums_vec;
        text_vec.zmm = _mm512_maskz_loadu_epi8(head_mask, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512());
        text_reversed_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask, text + head_length + body_length);
        sums_reversed_vec.zmm = _mm512_sad_epu8(text_reversed_vec.zmm, _mm512_setzero_si512());

        // Now in the main loop, we can use non-temporal loads, performing the operation in both directions.
        for (text += head_length; body_length >= 128; text += 64, body_length -= 128) {
            text_vec.zmm = _mm512_stream_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
            text_reversed_vec.zmm = _mm512_stream_load_si512((__m512i *)(text + body_length - 64));
            sums_reversed_vec.zmm =
                _mm512_add_epi64(sums_reversed_vec.zmm, _mm512_sad_epu8(text_reversed_vec.zmm, _mm512_setzero_si512()));
        }
        if (body_length >= 64) {
            text_vec.zmm = _mm512_stream_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
        }

        return _mm512_reduce_add_epi64(_mm512_add_epi64(sums_vec.zmm, sums_reversed_vec.zmm));
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_ICE
#pragma endregion // Ice Lake Implementation

/*  Implementation of the string hashing algorithms using the Arm NEON instruction set, available on 64-bit
 *  Arm processors. Covers billions of mobile CPUs worldwide, including Apple's A-series, and Qualcomm's Snapdragon.
 */
#pragma region NEON Implementation
#if SZ_USE_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)

SZ_PUBLIC sz_u64_t sz_checksum_neon(sz_cptr_t text, sz_size_t length) {
    uint64x2_t sum_vec = vdupq_n_u64(0);

    // Process 16 bytes (128 bits) at a time
    for (; length >= 16; text += 16, length -= 16) {
        uint8x16_t vec = vld1q_u8((sz_u8_t const *)text);      // Load 16 bytes
        uint16x8_t pairwise_sum1 = vpaddlq_u8(vec);            // Pairwise add lower and upper 8 bits
        uint32x4_t pairwise_sum2 = vpaddlq_u16(pairwise_sum1); // Pairwise add 16-bit results
        uint64x2_t pairwise_sum3 = vpaddlq_u32(pairwise_sum2); // Pairwise add 32-bit results
        sum_vec = vaddq_u64(sum_vec, pairwise_sum3);           // Accumulate the sum
    }

    // Final reduction of `sum_vec` to a single scalar
    sz_u64_t sum = vgetq_lane_u64(sum_vec, 0) + vgetq_lane_u64(sum_vec, 1);
    if (length) sum += sz_checksum_serial(text, length);
    return sum;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_NEON
#pragma endregion // NEON Implementation

/*  Implementation of the string search algorithms using the Arm SVE variable-length registers,
 *  available in Arm v9 processors, like in Apple M4+ and Graviton 3+ CPUs.
 */
#pragma region SVE Implementation
#if SZ_USE_SVE
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_SVE
#pragma endregion // SVE Implementation

/*  Pick the right implementation for the string search algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_u64_t sz_checksum(sz_cptr_t text, sz_size_t length) {
#if SZ_USE_ICE
    return sz_checksum_ice(text, length);
#elif SZ_USE_SKYLAKE
    return sz_checksum_skylake(text, length);
#elif SZ_USE_HASWELL
    return sz_checksum_haswell(text, length);
#elif SZ_USE_NEON
    return sz_checksum_neon(text, length);
#else
    return sz_checksum_serial(text, length);
#endif
}

SZ_DYNAMIC void sz_generate(sz_cptr_t alphabet, sz_size_t alphabet_size, sz_ptr_t result, sz_size_t result_length,
                            sz_random_generator_t generator, void *generator_user_data) {
    sz_generate_serial(alphabet, alphabet_size, result, result_length, generator, generator_user_data);
}

#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_HASH_H_
