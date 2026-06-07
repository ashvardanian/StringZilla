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
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_westmere_aligned_(&state, seed);

        // Load the data and update the state
        sz_u128_vec_t data_vec;
        data_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length), start);

        // Shuffle with the same mask
        __m128i const order = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_minimal_update_westmere_aligned_(&state, data_vec.xmm, order);
        return sz_hash_minimal_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 32) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_westmere_aligned_(&state, seed);

        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length - 16), start + 16);

        // Shuffle with the same mask
        __m128i const order = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_minimal_update_westmere_aligned_(&state, data0_vec.xmm, order);
        sz_hash_minimal_update_westmere_aligned_(&state, data1_vec.xmm, order);
        return sz_hash_minimal_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 48) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_westmere_aligned_(&state, seed);

        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 16));
        data2_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length - 32), start + 32);

        // Shuffle with the same mask
        __m128i const order = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_minimal_update_westmere_aligned_(&state, data0_vec.xmm, order);
        sz_hash_minimal_update_westmere_aligned_(&state, data1_vec.xmm, order);
        sz_hash_minimal_update_westmere_aligned_(&state, data2_vec.xmm, order);
        return sz_hash_minimal_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 64) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_westmere_aligned_(&state, seed);

        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 16));
        data2_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 32));
        data3_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length - 48), start + 48);

        // Shuffle with the same mask
        __m128i const order = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_minimal_update_westmere_aligned_(&state, data0_vec.xmm, order);
        sz_hash_minimal_update_westmere_aligned_(&state, data1_vec.xmm, order);
        sz_hash_minimal_update_westmere_aligned_(&state, data2_vec.xmm, order);
        sz_hash_minimal_update_westmere_aligned_(&state, data3_vec.xmm, order);
        return sz_hash_minimal_finalize_westmere_aligned_(&state, length);
    }
    else {
        sz_align_(64) sz_hash_state_internal_t_ state;
        sz_hash_state_init_skylake((sz_hash_state_t *)&state, seed);

        // Shuffle with the same mask
        __m128i const order = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        for (; state.ins_length + 64 <= length; state.ins_length += 64) {
            state.ins.zmm = _mm512_loadu_epi8(start + state.ins_length);
            state.aes.xmms[0] = _mm_aesenc_si128(state.aes.xmms[0], state.ins.xmms[0]);
            state.aes.xmms[1] = _mm_aesenc_si128(state.aes.xmms[1], state.ins.xmms[1]);
            state.aes.xmms[2] = _mm_aesenc_si128(state.aes.xmms[2], state.ins.xmms[2]);
            state.aes.xmms[3] = _mm_aesenc_si128(state.aes.xmms[3], state.ins.xmms[3]);
            state.sum.xmms[0] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[0], order), state.ins.xmms[0]);
            state.sum.xmms[1] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[1], order), state.ins.xmms[1]);
            state.sum.xmms[2] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[2], order), state.ins.xmms[2]);
            state.sum.xmms[3] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[3], order), state.ins.xmms[3]);
        }
        if (state.ins_length < length) {
            state.ins.zmm = _mm512_maskz_loadu_epi8( //
                sz_u64_mask_until_(length - state.ins_length), start + state.ins_length);
            state.aes.xmms[0] = _mm_aesenc_si128(state.aes.xmms[0], state.ins.xmms[0]);
            state.aes.xmms[1] = _mm_aesenc_si128(state.aes.xmms[1], state.ins.xmms[1]);
            state.aes.xmms[2] = _mm_aesenc_si128(state.aes.xmms[2], state.ins.xmms[2]);
            state.aes.xmms[3] = _mm_aesenc_si128(state.aes.xmms[3], state.ins.xmms[3]);
            state.sum.xmms[0] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[0], order), state.ins.xmms[0]);
            state.sum.xmms[1] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[1], order), state.ins.xmms[1]);
            state.sum.xmms[2] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[2], order), state.ins.xmms[2]);
            state.sum.xmms[3] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[3], order), state.ins.xmms[3]);
            state.ins_length = length;
        }
        return sz_hash_state_finalize_westmere_((sz_hash_state_t const *)&state);
    }
}

SZ_PUBLIC void sz_hash_state_update_skylake(sz_hash_state_t *state_ptr, sz_cptr_t text, sz_size_t length) {

    // The worst usage pattern... that we should ironically handle first - is updating the state
    // with a very small chunk of data, potentially, one byte at a time. In such cases, we won't
    // even bother using AVX-512 masked loads to avoid tripping the CPU state.
    sz_size_t const current_block_index = state_ptr->ins_length / 64;
    sz_size_t const final_block_index = (state_ptr->ins_length + length) / 64;
    int const stays_in_the_block = current_block_index == final_block_index;
    int const fills_the_block = (state_ptr->ins_length + length) % 64 == 0;
    if (stays_in_the_block && !fills_the_block) {
        for (; length; --length, ++state_ptr->ins_length, ++text) state_ptr->ins[state_ptr->ins_length % 64] = *text;
        return;
    }

    // Now we know that our "text" parts will end up in different blocks.
    // It's a good idea to pull the state into registers, as well definitely mix them at block boundaries.
    sz_size_t const progress_in_block = state_ptr->ins_length % 64;
    sz_size_t const head_length = (64 - progress_in_block) % 64;
    sz_size_t const tail_length = (state_ptr->ins_length + length) % 64;
    sz_size_t const body_length = length - head_length - tail_length;
    sz_assert_(body_length % 64 == 0 && head_length < 64 && tail_length < 64);

    // Lets keep a local copy of the state for one or more updates.
    sz_align_(64) sz_hash_state_internal_t_ state;
    state.aes.zmm = _mm512_loadu_si512((__m512i const *)state_ptr->aes);
    state.sum.zmm = _mm512_loadu_si512((__m512i const *)state_ptr->sum);
    state.ins.zmm = _mm512_loadu_si512((__m512i const *)state_ptr->ins);
    state.ins_length = state_ptr->ins_length;

    // Handle the head first, to align to the next block boundary
    __m128i const order = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
    if (head_length) {
        __mmask64 progress_mask = _knot_mask64(sz_u64_mask_until_(progress_in_block));
        state.ins.zmm = _mm512_mask_loadu_epi8(state.ins.zmm, progress_mask, text - progress_in_block);
        state.aes.xmms[0] = _mm_aesenc_si128(state.aes.xmms[0], state.ins.xmms[0]);
        state.aes.xmms[1] = _mm_aesenc_si128(state.aes.xmms[1], state.ins.xmms[1]);
        state.aes.xmms[2] = _mm_aesenc_si128(state.aes.xmms[2], state.ins.xmms[2]);
        state.aes.xmms[3] = _mm_aesenc_si128(state.aes.xmms[3], state.ins.xmms[3]);
        state.sum.xmms[0] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[0], order), state.ins.xmms[0]);
        state.sum.xmms[1] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[1], order), state.ins.xmms[1]);
        state.sum.xmms[2] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[2], order), state.ins.xmms[2]);
        state.sum.xmms[3] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[3], order), state.ins.xmms[3]);
        state.ins_length += head_length;
        text += head_length;
        length -= head_length;
    }

    // Now handle the body
    for (; length >= 64; state.ins_length += 64, text += 64, length -= 64) {
        state.ins.zmm = _mm512_loadu_epi8(text);
        state.aes.xmms[0] = _mm_aesenc_si128(state.aes.xmms[0], state.ins.xmms[0]);
        state.aes.xmms[1] = _mm_aesenc_si128(state.aes.xmms[1], state.ins.xmms[1]);
        state.aes.xmms[2] = _mm_aesenc_si128(state.aes.xmms[2], state.ins.xmms[2]);
        state.aes.xmms[3] = _mm_aesenc_si128(state.aes.xmms[3], state.ins.xmms[3]);
        state.sum.xmms[0] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[0], order), state.ins.xmms[0]);
        state.sum.xmms[1] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[1], order), state.ins.xmms[1]);
        state.sum.xmms[2] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[2], order), state.ins.xmms[2]);
        state.sum.xmms[3] = _mm_add_epi64(_mm_shuffle_epi8(state.sum.xmms[3], order), state.ins.xmms[3]);
    }

    // The tail is the last part we need to handle
    if (tail_length) {
        __mmask64 tail_mask = sz_u64_mask_until_(tail_length);
        state.ins.zmm = _mm512_maskz_loadu_epi8(tail_mask, text);
        state.ins_length += tail_length;
    }

    // Save the state back to the memory
    _mm512_storeu_si512((__m512i *)state_ptr->aes, state.aes.zmm);
    _mm512_storeu_si512((__m512i *)state_ptr->sum, state.sum.zmm);
    _mm512_storeu_si512((__m512i *)state_ptr->ins, state.ins.zmm);
    state_ptr->ins_length = state.ins_length;
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
