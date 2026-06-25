/**
 *  @brief Skylake (AVX-512) backend for string hashing and checksums.
 *  @file include/stringzilla/hash/skylake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_SKYLAKE_H_
#define STRINGZILLA_HASH_SKYLAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/hash/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2,aes"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2", "aes")
#endif

SZ_PUBLIC sz_u64_t sz_bytesum_skylake(sz_cptr_t text, sz_size_t length) {
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
        __mmask16 mask = sz_u16_mask_until_(length);
        text_vec.xmms[0] = _mm_maskz_loadu_epi8(mask, text);
        sums_vec.xmms[0] = _mm_sad_epu8(text_vec.xmms[0], _mm_setzero_si128());
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_vec.xmms[0]);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_vec.xmms[0], 1);
        return low + high;
    }
    else if (length <= 32) {
        __mmask32 mask = sz_u32_mask_until_(length);
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
        __mmask64 mask = sz_u64_mask_until_(length);
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
        sz_assert_(body_length % 64 == 0 && head_length < 64 && tail_length < 64);
        __mmask64 head_mask = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask = sz_u64_mask_until_(tail_length);

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
        __mmask64 head_mask = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask = sz_u64_mask_until_(tail_length);

        text_vec.zmm = _mm512_maskz_loadu_epi8(head_mask, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512());
        text_reversed_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask, text + head_length + body_length);
        sums_reversed_vec.zmm = _mm512_sad_epu8(text_reversed_vec.zmm, _mm512_setzero_si512());

        // Now in the main loop, we can use non-temporal loads, performing the operation in both directions.
        for (text += head_length; body_length >= 128; text += 64, body_length -= 128) {
            text_vec.zmm = _mm512_stream_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
            text_reversed_vec.zmm = _mm512_stream_load_si512((__m512i *)(text + body_length - 64));
            sums_reversed_vec.zmm = _mm512_add_epi64(sums_reversed_vec.zmm,
                                                     _mm512_sad_epu8(text_reversed_vec.zmm, _mm512_setzero_si512()));
        }
        if (body_length >= 64) {
            text_vec.zmm = _mm512_stream_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
        }

        return _mm512_reduce_add_epi64(_mm512_add_epi64(sums_vec.zmm, sums_reversed_vec.zmm));
    }
}

SZ_PUBLIC void sz_hash_state_init_skylake(sz_hash_state_t *state, sz_u64_t seed) {
    // The key is made from the seed and half of it will be mixed with the length in the end
    __m512i seed_vec = _mm512_set1_epi64(seed);
    // ! In this kernel, assuming it may be called on arbitrarily misaligned `state`,
    // ! we must use `_mm_storeu_si128` stores to update the state.
    _mm_storeu_si128((__m128i *)state->key, _mm512_castsi512_si128(seed_vec));

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    __m512i const pi0 = _mm512_load_epi64((__m512i const *)(pi));
    __m512i const pi1 = _mm512_load_epi64((__m512i const *)(pi + 8));
    _mm512_storeu_si512((__m512i *)state->aes, _mm512_xor_si512(seed_vec, pi0));
    _mm512_storeu_si512((__m512i *)state->sum, _mm512_xor_si512(seed_vec, pi1));

    // The inputs are zeroed out at the beginning
    _mm512_storeu_si512((__m512i *)state->ins, _mm512_setzero_si512());
    state->ins_length = 0;
}

SZ_PUBLIC SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_skylake(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {

    if (length <= 16) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_state_aligned_for_short_t_ state;
        sz_hash_state_short_init_westmere_aligned_(&state, seed);

        // Load the data and update the state
        sz_u128_vec_t data_vec;
        data_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length), start);

        // Shuffle with the same mask
        __m128i const order = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_state_short_update_westmere_aligned_(&state, data_vec.xmm, order);
        return sz_hash_state_short_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 32) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_state_aligned_for_short_t_ state;
        sz_hash_state_short_init_westmere_aligned_(&state, seed);

        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length - 16), start + 16);

        // Shuffle with the same mask
        __m128i const order = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_state_short_update_westmere_aligned_(&state, data0_vec.xmm, order);
        sz_hash_state_short_update_westmere_aligned_(&state, data1_vec.xmm, order);
        return sz_hash_state_short_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 48) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_state_aligned_for_short_t_ state;
        sz_hash_state_short_init_westmere_aligned_(&state, seed);

        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 16));
        data2_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length - 32), start + 32);

        // Shuffle with the same mask
        __m128i const order = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_state_short_update_westmere_aligned_(&state, data0_vec.xmm, order);
        sz_hash_state_short_update_westmere_aligned_(&state, data1_vec.xmm, order);
        sz_hash_state_short_update_westmere_aligned_(&state, data2_vec.xmm, order);
        return sz_hash_state_short_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 64) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_state_aligned_for_short_t_ state;
        sz_hash_state_short_init_westmere_aligned_(&state, seed);

        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 16));
        data2_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 32));
        data3_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length - 48), start + 48);

        // Shuffle with the same mask
        __m128i const order = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_state_short_update_westmere_aligned_(&state, data0_vec.xmm, order);
        sz_hash_state_short_update_westmere_aligned_(&state, data1_vec.xmm, order);
        sz_hash_state_short_update_westmere_aligned_(&state, data2_vec.xmm, order);
        sz_hash_state_short_update_westmere_aligned_(&state, data3_vec.xmm, order);
        return sz_hash_state_short_finalize_westmere_aligned_(&state, length);
    }
    // Skylake has no VAES, so its four-lane AES-NI absorb has no throughput edge over Westmere; in 512-bit form it
    // is in fact slower (per-lane `vextracti128` contends on the shuffle port). Skylake's win is the masked-load
    // short path above, so inputs over one block defer to the full-clock pure-SSE Westmere kernel.
    return sz_hash_westmere(start, length, seed);
}

SZ_PUBLIC void sz_hash_state_update_skylake(sz_hash_state_t *state_ptr, sz_cptr_t text, sz_size_t length) {
    // Skylake has AVX-512BW but neither VBMI (no `vpermb` byte slide) nor VAES, so the absorb stays the four-lane
    // AES-NI Westmere kernel. What Westmere lacks is a masked load: it merges incoming bytes one at a time into
    // `ins.u8s[...]`, then reads `ins` back as wide lanes - a ~12-cycle store-forwarding stall on every cross-call
    // merge, the dominant cost for short streamed tokens. AVX-512 removes it without VBMI: a single fault-suppressed
    // masked load from `text - buffered` lands the incoming bytes directly at their buffer offset. The masked-off
    // low `buffered` lanes alias the bytes before `text` and are never accessed; the set lanes
    // [buffered, buffered + to_copy) read exactly [text, text + to_copy). One `vpblendmb` drops them into `ins`;
    // bit-identical digest to the per-byte copy.
    sz_hash_state_aligned_t_ state = sz_hash_state_load_westmere_(state_ptr);
    sz_size_t buffered = state.ins_length % 64;
    if (buffered == 0 && state.ins_length) buffered = 64;
    while (length) {
        if (buffered == 64) { // the deferred block is now interior - absorb it (4x AES-NI) and re-zero the buffer
            sz_hash_state_update_westmere_(&state);
            state.ins.zmm = _mm512_setzero_si512();
            buffered = 0;
        }
        sz_size_t const to_copy = sz_min_of_two(length, (sz_size_t)64 - buffered);
        __mmask64 const place_mask = _cvtu64_mask64(sz_u64_mask_until_(to_copy) << buffered);
        __m512i const incoming_u8x64 = _mm512_maskz_loadu_epi8(place_mask, (void const *)(text - buffered));
        state.ins.zmm = _mm512_mask_blend_epi8(place_mask, state.ins.zmm, incoming_u8x64);
        buffered += to_copy, text += to_copy, length -= to_copy, state.ins_length += to_copy;
    }
    sz_hash_state_store_westmere_(state_ptr, &state);
}

SZ_PUBLIC sz_u64_t sz_hash_state_digest_skylake(sz_hash_state_t const *state) {
    // ? We don't know a better way to fold the state on Ice Lake, than to use the Haswell implementation.
    return sz_hash_state_digest_westmere(state);
}

SZ_PUBLIC void sz_fill_random_skylake(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_fill_random_westmere(text, length, nonce);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SKYLAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_SKYLAKE_H_
