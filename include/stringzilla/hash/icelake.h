/**
 *  @brief Ice Lake (AVX-512 + VAES) backend for string hashing and checksums.
 *  @file include/stringzilla/hash/icelake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_ICELAKE_H_
#define STRINGZILLA_HASH_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/hash/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_ICELAKE
#if defined(__clang__)
#pragma clang attribute push(                                                                                      \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vnni,bmi,bmi2,aes,vaes,sha"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vnni", "bmi", "bmi2", \
                   "aes", "vaes", "sha")
#endif

SZ_API_COMPTIME sz_u64_t sz_bytesum_icelake(sz_cptr_t text, sz_size_t length) {
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
    else if (!is_huge) {
        sz_size_t head_length = (64 - ((sz_size_t)text % 64)) % 64; // 63 or less.
        sz_size_t tail_length = (sz_size_t)(text + length) % 64;    // 63 or less.
        sz_size_t body_length = length - head_length - tail_length; // Multiple of 64.
        sz_assert_(body_length % 64 == 0 && head_length < 64 && tail_length < 64);
        __mmask64 head_mask = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask = sz_u64_mask_until_(tail_length);

        sz_u512_vec_t zeros_vec, ones_vec;
        zeros_vec.zmm = _mm512_setzero_si512();
        ones_vec.zmm = _mm512_set1_epi8(1);

        // Take care of the unaligned head and tail!
        sz_u512_vec_t text_reversed_vec, sums_reversed_vec;
        text_vec.zmm = _mm512_maskz_loadu_epi8(head_mask, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, zeros_vec.zmm);
        text_reversed_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask, text + head_length + body_length);
        sums_reversed_vec.zmm = _mm512_dpbusds_epi32(zeros_vec.zmm, text_reversed_vec.zmm, ones_vec.zmm);

        // Now in the main loop, we can use aligned loads, performing the operation in both directions.
        for (text += head_length; body_length >= 128; text += 64, body_length -= 128) {
            text_reversed_vec.zmm = _mm512_load_si512((__m512i *)(text + body_length - 64));
            sums_reversed_vec.zmm = _mm512_dpbusds_epi32(sums_reversed_vec.zmm, text_reversed_vec.zmm, ones_vec.zmm);
            text_vec.zmm = _mm512_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, zeros_vec.zmm));
        }
        // There may be an aligned chunk of 64 bytes left.
        if (body_length >= 64) {
            sz_assert_(body_length == 64);
            text_vec.zmm = _mm512_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, zeros_vec.zmm));
        }

        return _mm512_reduce_add_epi64(sums_vec.zmm) + _mm512_reduce_add_epi32(sums_reversed_vec.zmm);
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

SZ_API_COMPTIME SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_icelake(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {

    // For short strings the "masked loads" are identical to Skylake-X and
    // the "logic" is identical to Haswell.
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
    // This is where the logic differs from Skylake-X and other pre-Ice Lake CPUs:
    else {
        sz_align_(64) sz_hash_state_aligned_t_ state;
        sz_hash_state_init_skylake((sz_hash_state_t *)&state, seed);

        // Absorb every full 64-byte block EXCEPT the last; the final block (a full 64 or a partial tail) stays
        // buffered in `ins` for `sz_hash_state_finalize_westmere_` to fold - the same deferral the streaming path uses.
        __m512i const order = _mm512_load_si512((__m512i const *)sz_hash_u8x16x4_shuffle_());
        for (; state.ins_length + 64 < length; state.ins_length += 64) {
            state.ins.zmm = _mm512_loadu_epi8(start + state.ins_length);
            state.aes.zmm = _mm512_aesenc_epi128(state.aes.zmm, state.ins.zmm);
            state.sum.zmm = _mm512_add_epi64(_mm512_shuffle_epi8(state.sum.zmm, order), state.ins.zmm);
        }
        // Stage the final [ins_length, length) bytes (1..64) into a zero-padded buffer; finalize folds them.
        state.ins.zmm = _mm512_maskz_loadu_epi8(sz_u64_mask_until_(length - state.ins_length),
                                                start + state.ins_length);
        state.ins_length = length;
        return sz_hash_state_finalize_westmere_(&state);
    }
}

SZ_API_COMPTIME void sz_hash_state_init_icelake(sz_hash_state_t *state, sz_u64_t seed) {
    sz_hash_state_init_skylake(state, seed);
}

/** @brief Loads the packed public state into the aligned twin (one `_mm512_loadu_si512` per 64-byte field). */
SZ_HELPER_AUTO sz_hash_state_aligned_t_ sz_hash_state_load_icelake_(sz_hash_state_t const *packed) {
    sz_hash_state_aligned_t_ state;
    state.aes.zmm = _mm512_loadu_si512((__m512i const *)packed->aes);
    state.sum.zmm = _mm512_loadu_si512((__m512i const *)packed->sum);
    state.ins.zmm = _mm512_loadu_si512((__m512i const *)packed->ins);
    state.key.xmm = _mm_lddqu_si128((__m128i const *)packed->key);
    state.ins_length = packed->ins_length;
    return state;
}

/** @brief Stores the aligned twin back into the packed public state (one `_mm512_storeu_si512` per field). */
SZ_HELPER_AUTO void sz_hash_state_store_icelake_(sz_hash_state_t *packed, sz_hash_state_aligned_t_ const *state) {
    _mm512_storeu_si512((__m512i *)packed->aes, state->aes.zmm);
    _mm512_storeu_si512((__m512i *)packed->sum, state->sum.zmm);
    _mm512_storeu_si512((__m512i *)packed->ins, state->ins.zmm);
    _mm_storeu_si128((__m128i *)packed->key, state->key.xmm);
    packed->ins_length = state->ins_length;
}

/** @brief Absorbs the buffered 64-byte block into the aligned state with a single VAES `VAESENC` over four lanes. */
SZ_HELPER_AUTO void sz_hash_state_update_icelake_(sz_hash_state_aligned_t_ *state) {
    __m512i const order = _mm512_load_si512((__m512i const *)sz_hash_u8x16x4_shuffle_());
    state->aes.zmm = _mm512_aesenc_epi128(state->aes.zmm, state->ins.zmm);
    state->sum.zmm = _mm512_add_epi64(_mm512_shuffle_epi8(state->sum.zmm, order), state->ins.zmm);
}

SZ_API_COMPTIME void sz_hash_state_update_icelake(sz_hash_state_t *state_ptr, sz_cptr_t text, sz_size_t length) {

    // Load the packed public state (any alignment) into an aligned twin once, buffer/absorb on it, then store back.
    // `ins` is exactly one 64-byte block (one ZMM), so buffering is just: track how many bytes it holds, absorb
    // it only once it becomes interior (more bytes arrive - the deferral `digest` needs to choose minimal/full by
    // total length), and append incoming bytes. The deferred trailing block reads back as
    // `ins_length % 64 == 0 && ins_length != 0`; treat that as `buffered == 64`. Re-zeroing `ins` after each absorb
    // keeps the high lanes zero-padded for `finalize` to fold.
    sz_hash_state_aligned_t_ state = sz_hash_state_load_icelake_(state_ptr);
    sz_size_t buffered = state.ins_length % 64;
    if (buffered == 0 && state.ins_length) buffered = 64;
    //  Per-lane identity {0, 1, ..., 63} used to slide incoming bytes to the buffer offset entirely in-register.
    sz_align_(64) static sz_u8_t const lane_iota[64] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                                        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                                        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                                        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
    __m512i const lane_iota_vec = _mm512_load_si512((__m512i const *)lane_iota);
    while (length) {
        if (buffered == 64) { // the deferred block is now interior - absorb it and re-zero the buffer
            sz_hash_state_update_icelake_(&state);
            state.ins.zmm = _mm512_setzero_si512();
            buffered = 0;
        }
        sz_size_t const to_copy = sz_min_of_two(length, (sz_size_t)64 - buffered);
        //  Merge `to_copy` incoming bytes into `ins` at lane offset `buffered` WITHOUT a stack round-trip: poking
        //  individual bytes into the just-written `ins` ZMM stalls on store-forwarding (~12 cy) every cross-call
        //  merge - the dominant cost for short streamed tokens. `vpermb` slides the masked-loaded bytes to the
        //  offset and `vpblendmb` drops them into place; bit-identical digest to the per-byte copy.
        __mmask64 const copy_mask = _cvtu64_mask64(sz_u64_mask_until_(to_copy));
        __m512i const incoming = _mm512_maskz_loadu_epi8(copy_mask, text);
        __m512i const slide = _mm512_sub_epi8(lane_iota_vec, _mm512_set1_epi8((char)buffered));
        __m512i const shifted = _mm512_permutexvar_epi8(slide, incoming);
        state.ins.zmm = _mm512_mask_blend_epi8(_cvtu64_mask64(sz_u64_mask_until_(to_copy) << buffered), state.ins.zmm,
                                               shifted);
        buffered += to_copy, text += to_copy, length -= to_copy, state.ins_length += to_copy;
    }
    sz_hash_state_store_icelake_(state_ptr, &state);
}

SZ_API_COMPTIME sz_u64_t sz_hash_state_digest_icelake(sz_hash_state_t const *state) {
    // ? We don't know a better way to fold the state on Ice Lake, than to use the Haswell implementation.
    return sz_hash_state_digest_westmere(state);
}

SZ_API_COMPTIME void sz_fill_random_icelake(sz_ptr_t output, sz_size_t length, sz_u64_t nonce) {
    if (length <= 16) {
        __m128i input = _mm_set1_epi64x(nonce);
        __m128i pi = _mm_load_si128((__m128i const *)sz_hash_pi_constants_());
        __m128i key = _mm_xor_si128(_mm_set1_epi64x(nonce), pi);
        __m128i generated = _mm_aesenc_si128(input, key);
        __mmask16 store_mask = sz_u16_mask_until_(length);
        _mm_mask_storeu_epi8((void *)output, store_mask, generated);
    }
    // Assuming the YMM register contains two 128-bit blocks, the input to the generator
    // will be more complex, containing the sum of the nonce and the block number.
    else if (length <= 32) {
        __m256i input = _mm256_set_epi64x(nonce + 1, nonce + 1, nonce, nonce);
        __m256i pi = _mm256_load_si256((__m256i const *)sz_hash_pi_constants_());
        __m256i key = _mm256_xor_si256(_mm256_set1_epi64x(nonce), pi);
        __m256i generated = _mm256_aesenc_epi128(input, key);
        __mmask32 store_mask = sz_u32_mask_until_(length);
        _mm256_mask_storeu_epi8((void *)output, store_mask, generated);
    }
    // The last special case we handle outside of the primary loop is for buffers up to 64 bytes long.
    else if (length <= 64) {
        __m512i input = _mm512_set_epi64(               //
            nonce + 3, nonce + 3, nonce + 2, nonce + 2, //
            nonce + 1, nonce + 1, nonce, nonce);
        __m512i pi = _mm512_load_si512((__m512i const *)sz_hash_pi_constants_());
        __m512i key = _mm512_xor_si512(_mm512_set1_epi64(nonce), pi);
        __m512i generated = _mm512_aesenc_epi128(input, key);
        __mmask64 store_mask = sz_u64_mask_until_(length);
        _mm512_mask_storeu_epi8((void *)output, store_mask, generated);
    }
    // The final part of the function is the primary loop, which processes the buffer in 64-byte chunks.
    else {
        __m512i const increment = _mm512_set1_epi64(4);
        __m512i input = _mm512_set_epi64(               //
            nonce + 3, nonce + 3, nonce + 2, nonce + 2, //
            nonce + 1, nonce + 1, nonce, nonce);
        __m512i const pi = _mm512_load_si512((__m512i const *)sz_hash_pi_constants_());
        __m512i const key = _mm512_xor_si512(_mm512_set1_epi64(nonce), pi);

        // Produce the output, fixing the key and enumerating input chunks.
        sz_size_t byte_index = 0;
        for (; byte_index + 64 <= length; byte_index += 64) {
            __m512i generated = _mm512_aesenc_epi128(input, key);
            _mm512_storeu_epi8((void *)(output + byte_index), generated);
            input = _mm512_add_epi64(input, increment);
        }

        // Handle the tail of the buffer.
        __m512i generated = _mm512_aesenc_epi128(input, key);
        __mmask64 store_mask = sz_u64_mask_until_(length - byte_index);
        _mm512_mask_storeu_epi8((void *)(output + byte_index), store_mask, generated);
    }
}

/**
 *  @brief A wider parallel analog of `sz_hash_state_aligned_for_short_t_`, which is not used for computing individual hashes,
 *         but for parallel hashing of @b short 4x separate strings under 16 bytes long.
 *         Useful for higher-level Database and Machine Learning operations.
 */
typedef struct sz_hash_state_aligned_for_short_x4_t_ {
    sz_u512_vec_t aes;
    sz_u512_vec_t sum;
    sz_u512_vec_t key;
} sz_hash_state_aligned_for_short_x4_t_;

/**
 *  @brief Initializes the 4-wide parallel minimal hash state using VAES and AVX-512 intrinsics.
 *  @param state Pointer to the 4-wide minimal hash state to initialize.
 *  @param seed 64-bit seed XOR-ed with Pi constants replicated across all four 128-bit lanes.
 */
SZ_HELPER_AUTO void sz_hash_state_short_x4_init_icelake_(sz_hash_state_aligned_for_short_x4_t_ *state, sz_u64_t seed) {

    // The key is made from the seed and half of it will be mixed with the length in the end
    __m512i seed_vec = _mm512_set1_epi64(seed);
    state->key.zmm = seed_vec; //! This will definitely be aligned

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    __m512i pi0 = _mm512_load_si512((__m512i const *)(pi));
    __m512i pi1 = _mm512_load_si512((__m512i const *)(pi + 8));
    // We will load the entire 512-bit values, but will only use the first 128 bits,
    // replicating it 4x times across the register. The `_mm512_shuffle_i64x2` is supposed to
    // be faster than `_mm512_broadcast_i64x2` on Ice Lake.
    pi0 = _mm512_shuffle_i64x2(pi0, pi0, 0);
    pi1 = _mm512_shuffle_i64x2(pi1, pi1, 0);
    __m512i aes_state_key = _mm512_xor_si512(seed_vec, pi0);
    __m512i sum_state_key = _mm512_xor_si512(seed_vec, pi1);

    // The first 128 bits of the "sum" and "AES" blocks are the same for the "minimal" and full state
    state->aes.zmm = aes_state_key;
    state->sum.zmm = sum_state_key;
}

/**
 *  @brief Finalizes the 4-wide parallel minimal hash state, returning four 64-bit digests packed in a 256-bit vector.
 *  @param state Pointer to the (const) 4-wide minimal hash state.
 *  @param length0 Total byte count for the first 128-bit lane.
 *  @param length1 Total byte count for the second 128-bit lane.
 *  @param length2 Total byte count for the third 128-bit lane.
 *  @param length3 Total byte count for the fourth 128-bit lane.
 *  @return 256-bit vector containing four 64-bit hash values (one per lane).
 */
SZ_HELPER_AUTO __m256i sz_hash_state_short_x4_finalize_icelake_(sz_hash_state_aligned_for_short_x4_t_ const *state, //
                                                                sz_size_t length0, sz_size_t length1, sz_size_t length2,
                                                                sz_size_t length3) {
    __m512i const padded_lengths = _mm512_set_epi64(0, length3, 0, length2, 0, length1, 0, length0);
    // Mix the length into the key
    __m512i key_with_length = _mm512_add_epi64(state->key.zmm, padded_lengths);
    // Combine the "sum" and the "AES" blocks
    __m512i mixed = _mm512_aesenc_epi128(state->sum.zmm, state->aes.zmm);
    // Make sure the "key" mixes enough with the state,
    // as with less than 2 rounds - SMHasher fails
    __m512i mixed_in_register = _mm512_aesenc_epi128(_mm512_aesenc_epi128(mixed, key_with_length), mixed);
    // Extract the low 64 bits from each 128-bit lane - weirdly using the `permutexvar` instruction
    // is cheaper than compressing instructions like `_mm512_maskz_compress_epi64`.
    return _mm512_castsi512_si256(
        _mm512_permutexvar_epi64(_mm512_set_epi64(0, 0, 0, 0, 6, 4, 2, 0), mixed_in_register));
}

/**
 *  @brief Absorbs four 128-bit blocks (one per hash lane) into the 4-wide minimal hash state.
 *  @param state Pointer to the 4-wide minimal hash state.
 *  @param blocks 512-bit register containing four 128-bit data blocks, one per lane.
 */
SZ_HELPER_AUTO void sz_hash_state_short_x4_update_icelake_(sz_hash_state_aligned_for_short_x4_t_ *state,
                                                           __m512i blocks) {
    __m512i const order = _mm512_load_si512((__m512i const *)sz_hash_u8x16x4_shuffle_());
    state->aes.zmm = _mm512_aesenc_epi128(state->aes.zmm, blocks);
    state->sum.zmm = _mm512_add_epi64(_mm512_shuffle_epi8(state->sum.zmm, order), blocks);
}

/**
 *  @brief Initializes the 4-wide parallel minimal state with four @b distinct seeds (one per lane).
 *         Unlike `sz_hash_state_short_x4_init_icelake_`, which seeds all four lanes identically to hash
 *         four different strings, this seeds each lane differently to hash one string under four seeds.
 *  @param state Pointer to the 4-wide minimal hash state to initialize.
 *  @param seeds_vec Four seeds spread as `[s0,s0,s1,s1,s2,s2,s3,s3]` across the 512-bit register.
 */
SZ_HELPER_AUTO void sz_hash_multiseed_x4_init_icelake_(sz_hash_state_aligned_for_short_x4_t_ *state,
                                                       __m512i seeds_vec) {
    state->key.zmm = seeds_vec;
    // Replicate the first 128 bits of each Pi half across all four lanes, then XOR the per-lane seeds.
    sz_u64_t const *pi = sz_hash_pi_constants_();
    __m512i pi0 = _mm512_shuffle_i64x2(_mm512_load_si512((__m512i const *)(pi)), //
                                       _mm512_load_si512((__m512i const *)(pi)), 0);
    __m512i pi1 = _mm512_shuffle_i64x2(_mm512_load_si512((__m512i const *)(pi + 8)), //
                                       _mm512_load_si512((__m512i const *)(pi + 8)), 0);
    state->aes.zmm = _mm512_xor_si512(seeds_vec, pi0);
    state->sum.zmm = _mm512_xor_si512(seeds_vec, pi1);
}

/**
 *  @brief Finalizes the 4-wide parallel minimal state, folding in the input length.
 *  @param state Pointer to the (const) 4-wide minimal hash state.
 *  @param lengths The length addend broadcast into each seed-lane's key low half, i.e.
 *                 `[0, length, 0, length, 0, length, 0, length]`. Seed-independent, so the caller
 *                 builds it once and reuses it across all seed groups.
 *  @return 256-bit vector with four 64-bit hashes, one per lane.
 */
SZ_HELPER_AUTO __m256i sz_hash_multiseed_x4_finalize_icelake_(sz_hash_state_aligned_for_short_x4_t_ const *state,
                                                              __m512i lengths) {
    __m512i key_with_length = _mm512_add_epi64(state->key.zmm, lengths);
    __m512i mixed = _mm512_aesenc_epi128(state->sum.zmm, state->aes.zmm);
    __m512i mixed_in_register = _mm512_aesenc_epi128(_mm512_aesenc_epi128(mixed, key_with_length), mixed);
    return _mm512_castsi512_si256(
        _mm512_permutexvar_epi64(_mm512_set_epi64(0, 0, 0, 0, 6, 4, 2, 0), mixed_in_register));
}

SZ_API_COMPTIME void sz_hash_multiseed_icelake(sz_cptr_t text, sz_size_t length,             //
                                               sz_u64_t const *seeds, sz_size_t seeds_count, //
                                               sz_u64_t *hashes) {
    // Trivial counts don't benefit from sharing a normalization pass - go straight to the single-shot.
    if (seeds_count == 0) return;
    if (seeds_count == 1) {
        hashes[0] = sz_hash_icelake(text, length, seeds[0]);
        return;
    }
    // Long strings gain nothing from seed-packing - the AES work scales with the byte count regardless.
    if (length > 64) {
        for (sz_size_t seed_index = 0; seed_index < seeds_count; ++seed_index)
            hashes[seed_index] = sz_hash_icelake(text, length, seeds[seed_index]);
        return;
    }

    // One branchless masked load pulls the whole <= 64 byte input into a single ZMM; its four 128-bit
    // halves ARE the de-interleaved text-lanes (each up to 16 contiguous, low-justified, zero-padded
    // bytes), so no scalar normalization is needed. Each text-lane is then broadcast to all four
    // @b seed-lanes of a ZMM, so a single `VAESENC` advances four seeds at once; the text-lanes are
    // consumed one after another, never as a single 512-bit value.
    sz_u512_vec_t text_lanes;
    text_lanes.zmm = _mm512_maskz_loadu_epi8(sz_u64_mask_until_(length), text);
    sz_size_t const text_lanes_count = length <= 16 ? 1 : sz_size_divide_round_up(length, 16);
    __m512i broadcast_text_lanes[4];
    for (sz_size_t lane_index = 0; lane_index < text_lanes_count; ++lane_index)
        broadcast_text_lanes[lane_index] = _mm512_broadcast_i32x4(text_lanes.xmms[lane_index]);

    // Both addends below are seed-independent, so hoist them out of the per-group loop:
    // - `lengths` folds the input length into each seed-lane's key half at finalization;
    // - `seed_spread` turns a contiguous `[s0,s1,s2,s3]` load into per-seed-lane `[s0,s0,s1,s1,s2,s2,s3,s3]`.
    __m512i const lengths = _mm512_set_epi64(0, (sz_i64_t)length, 0, (sz_i64_t)length, //
                                             0, (sz_i64_t)length, 0, (sz_i64_t)length);
    __m512i const seed_spread = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);

    sz_size_t seed_index = 0;
    for (; seed_index + 4 <= seeds_count; seed_index += 4) {
        sz_hash_state_aligned_for_short_x4_t_ state;
        __m512i const seeds_vec = _mm512_permutexvar_epi64(
            seed_spread, _mm512_castsi256_si512(_mm256_loadu_si256((__m256i const *)(seeds + seed_index))));
        sz_hash_multiseed_x4_init_icelake_(&state, seeds_vec);
        for (sz_size_t lane_index = 0; lane_index < text_lanes_count; ++lane_index)
            sz_hash_state_short_x4_update_icelake_(&state, broadcast_text_lanes[lane_index]);
        _mm256_storeu_si256((__m256i *)(hashes + seed_index), sz_hash_multiseed_x4_finalize_icelake_(&state, lengths));
    }

    // Tail: a final partial group of 1..3 seeds, masked so we never read or write past the arrays
    // and stay entirely within VAES instead of dropping to a scalar AES-NI loop.
    if (seed_index < seeds_count) {
        __mmask8 const seed_mask = (__mmask8)((1u << (seeds_count - seed_index)) - 1);
        sz_hash_state_aligned_for_short_x4_t_ state;
        __m512i const seeds_vec = _mm512_permutexvar_epi64(
            seed_spread, _mm512_castsi256_si512(_mm256_maskz_loadu_epi64(seed_mask, seeds + seed_index)));
        sz_hash_multiseed_x4_init_icelake_(&state, seeds_vec);
        for (sz_size_t lane_index = 0; lane_index < text_lanes_count; ++lane_index)
            sz_hash_state_short_x4_update_icelake_(&state, broadcast_text_lanes[lane_index]);
        _mm256_mask_storeu_epi64(hashes + seed_index, seed_mask,
                                 sz_hash_multiseed_x4_finalize_icelake_(&state, lengths));
    }
}

/**
 *  @brief Process a single 512-bit (64-byte) block of data using SHA256 with SHA-NI and AVX-512.
 *  @param hash Pointer to 8x 32-bit hash values, modified in place.
 *  @param block Pointer to 64-byte message block.
 */
SZ_HELPER_AUTO void sz_sha256_process_block_icelake_(sz_u32_t hash[sz_at_least_(8)],
                                                     sz_u8_t const block[sz_at_least_(64)]) {
    sz_u32_t const *round_constants = sz_sha256_round_constants_();

    // Load entire 64-byte block with single 512-bit load and byte-swap
    sz_u512_vec_t block_vec;
    block_vec.zmm = _mm512_loadu_si512(block);

    // Byte-swap using 512-bit shuffle (reverse byte order within each 32-bit word)
    __m512i const bswap_mask_512 = _mm512_set_epi8(                     //
        60, 61, 62, 63, 56, 57, 58, 59, 52, 53, 54, 55, 48, 49, 50, 51, //
        44, 45, 46, 47, 40, 41, 42, 43, 36, 37, 38, 39, 32, 33, 34, 35, //
        28, 29, 30, 31, 24, 25, 26, 27, 20, 21, 22, 23, 16, 17, 18, 19, //
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
    block_vec.zmm = _mm512_shuffle_epi8(block_vec.zmm, bswap_mask_512);

    // Extract 128-bit message words (SHA-NI operates on 128-bit registers)
    __m128i msg0 = block_vec.xmms[0];
    __m128i msg1 = block_vec.xmms[1];
    __m128i msg2 = block_vec.xmms[2];
    __m128i msg3 = block_vec.xmms[3];

    // Pre-load round constants into 512-bit registers for efficient access
    sz_u128_vec_t k0_3, k4_7, k8_11, k12_15, k16_19, k20_23, k24_27, k28_31;
    sz_u128_vec_t k32_35, k36_39, k40_43, k44_47, k48_51, k52_55, k56_59, k60_63;
    k0_3.xmm = _mm_lddqu_si128((__m128i const *)&round_constants[0]);
    k4_7.xmm = _mm_lddqu_si128((__m128i const *)&round_constants[4]);
    k8_11.xmm = _mm_lddqu_si128((__m128i const *)&round_constants[8]);
    k12_15.xmm = _mm_lddqu_si128((__m128i const *)&round_constants[12]);
    k16_19.xmm = _mm_lddqu_si128((__m128i const *)&round_constants[16]);
    k20_23.xmm = _mm_lddqu_si128((__m128i const *)&round_constants[20]);
    k24_27.xmm = _mm_lddqu_si128((__m128i const *)&round_constants[24]);
    k28_31.xmm = _mm_lddqu_si128((__m128i const *)&round_constants[28]);
    k32_35.xmm = _mm_lddqu_si128((__m128i const *)&round_constants[32]);
    k36_39.xmm = _mm_lddqu_si128((__m128i const *)&round_constants[36]);
    k40_43.xmm = _mm_lddqu_si128((__m128i const *)&round_constants[40]);
    k44_47.xmm = _mm_lddqu_si128((__m128i const *)&round_constants[44]);
    k48_51.xmm = _mm_lddqu_si128((__m128i const *)&round_constants[48]);
    k52_55.xmm = _mm_lddqu_si128((__m128i const *)&round_constants[52]);
    k56_59.xmm = _mm_lddqu_si128((__m128i const *)&round_constants[56]);
    k60_63.xmm = _mm_lddqu_si128((__m128i const *)&round_constants[60]);

    // Pack state into SHA-NI format (ABEF/CDGH)
    __m128i state0 = _mm_lddqu_si128((__m128i const *)&hash[0]); // A B C D
    __m128i state1 = _mm_lddqu_si128((__m128i const *)&hash[4]); // E F G H
    __m128i perm_u32x4 = _mm_shuffle_epi32(state0, 0xB1);        // CDAB
    state1 = _mm_shuffle_epi32(state1, 0x1B);                    // HGFE
    state0 = _mm_alignr_epi8(perm_u32x4, state1, 8);             // ABEF
    state1 = _mm_blend_epi16(state1, perm_u32x4, 0xF0);          // CDGH

    __m128i state0_saved = state0;
    __m128i state1_saved = state1;

    // Rounds 0-3
    __m128i round_input = _mm_add_epi32(msg0, k0_3.xmm);
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);

    // Rounds 4-7
    round_input = _mm_add_epi32(msg1, k4_7.xmm);
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);

    // Rounds 8-11
    round_input = _mm_add_epi32(msg2, k8_11.xmm);
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);

    // Rounds 12-15
    round_input = _mm_add_epi32(msg3, k12_15.xmm);
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);

    // Rounds 16-19
    round_input = _mm_add_epi32(msg0, k16_19.xmm);
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);

    // Rounds 20-23
    round_input = _mm_add_epi32(msg1, k20_23.xmm);
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);

    // Rounds 24-27
    round_input = _mm_add_epi32(msg2, k24_27.xmm);
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);

    // Rounds 28-31
    round_input = _mm_add_epi32(msg3, k28_31.xmm);
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);

    // Rounds 32-35
    round_input = _mm_add_epi32(msg0, k32_35.xmm);
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);

    // Rounds 36-39
    round_input = _mm_add_epi32(msg1, k36_39.xmm);
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);

    // Rounds 40-43
    round_input = _mm_add_epi32(msg2, k40_43.xmm);
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);

    // Rounds 44-47
    round_input = _mm_add_epi32(msg3, k44_47.xmm);
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);

    // Rounds 48-51
    round_input = _mm_add_epi32(msg0, k48_51.xmm);
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);

    // Rounds 52-55
    round_input = _mm_add_epi32(msg1, k52_55.xmm);
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);

    // Rounds 56-59
    round_input = _mm_add_epi32(msg2, k56_59.xmm);
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);
    msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);

    // Rounds 60-63
    round_input = _mm_add_epi32(msg3, k60_63.xmm);
    state1 = _mm_sha256rnds2_epu32(state1, state0, round_input);
    round_input = _mm_shuffle_epi32(round_input, 0x0E);
    state0 = _mm_sha256rnds2_epu32(state0, state1, round_input);

    // Add compressed chunk to hash
    state0 = _mm_add_epi32(state0, state0_saved);
    state1 = _mm_add_epi32(state1, state1_saved);

    // Unpack from SHA-NI format back to ABCD/EFGH
    perm_u32x4 = _mm_shuffle_epi32(state0, 0x1B);       // FEBA
    state1 = _mm_shuffle_epi32(state1, 0xB1);           // GHCD
    state0 = _mm_blend_epi16(perm_u32x4, state1, 0xF0); // ABCD
    state1 = _mm_alignr_epi8(state1, perm_u32x4, 8);    // EFGH

    // Store results
    _mm_storeu_si128((__m128i *)&hash[0], state0);
    _mm_storeu_si128((__m128i *)&hash[4], state1);
}

SZ_API_COMPTIME void sz_sha256_state_init_icelake(sz_sha256_state_t *state_ptr) {
    // Vectorize the load/store of 8x u32s using 1x 256-bit AVX load
    sz_u32_t const *initial_hash = sz_sha256_initial_hash_();
    _mm256_storeu_si256((__m256i *)state_ptr->hash, _mm256_lddqu_si256((__m256i const *)initial_hash));
    state_ptr->block_length = 0, state_ptr->total_length = 0;
}

SZ_API_COMPTIME void sz_sha256_state_update_icelake(sz_sha256_state_t *state_ptr, sz_cptr_t data, sz_size_t length) {
    sz_u8_t const *input = (sz_u8_t const *)data;
    sz_size_t const current_block_index = state_ptr->block_length / 64;
    sz_size_t const final_block_index = (state_ptr->block_length + length) / 64;
    int const stays_in_the_block = current_block_index == final_block_index;
    int const fills_the_block = (state_ptr->block_length + length) % 64 == 0;

    state_ptr->total_length += length;

    // Fast path: stays in same block and doesn't fill it
    if (stays_in_the_block && !fills_the_block) {
        for (; length; --length, ++state_ptr->block_length, ++input) state_ptr->block[state_ptr->block_length] = *input;
        return;
    }

    // Calculate head, body, and tail lengths
    sz_size_t const head_length = (64 - state_ptr->block_length) % 64;
    sz_size_t const tail_length = (state_ptr->block_length + length) % 64;
    sz_size_t const body_length = length - head_length - tail_length;

    // Copy hash to aligned local buffer
    sz_align_(32) sz_u32_t hash[8];
    _mm256_store_si256((__m256i *)hash, _mm256_lddqu_si256((__m256i const *)state_ptr->hash));

    // Process head to complete the current block
    if (head_length) {
        __mmask64 head_mask = sz_u64_clamp_mask_until_(head_length);
        _mm512_mask_storeu_epi8(&state_ptr->block[state_ptr->block_length], head_mask,
                                _mm512_maskz_loadu_epi8(head_mask, input));
        state_ptr->block_length += head_length;
        sz_sha256_process_block_icelake_(hash, state_ptr->block);
        state_ptr->block_length = 0;
        input += head_length;
    }

    // Process body (complete aligned blocks)
    for (sz_size_t processed = 0; processed < body_length; processed += 64, input += 64)
        sz_sha256_process_block_icelake_(hash, input);

    // Process tail (remaining bytes into block buffer)
    if (tail_length) {
        __mmask64 tail_mask = sz_u64_clamp_mask_until_(tail_length);
        _mm512_mask_storeu_epi8(state_ptr->block, tail_mask, _mm512_maskz_loadu_epi8(tail_mask, input));
        state_ptr->block_length = tail_length;
    }

    // Copy hash back
    _mm256_storeu_si256((__m256i *)state_ptr->hash, _mm256_load_si256((__m256i const *)hash));
}

SZ_API_COMPTIME void sz_sha256_state_digest_icelake(sz_sha256_state_t const *state_ptr,
                                                    sz_u8_t digest[sz_at_least_(32)]) {
    // Create a copy of the state for padding
    sz_sha256_state_t state = *state_ptr;

    // Append the '1' bit (0x80 byte) after the message
    state.block[state.block_length++] = 0x80;

    // If there's not enough room for the 64-bit length, pad this block and process it
    if (state.block_length > 56) {
        // Zero remaining bytes using AVX-512 masked store
        sz_size_t remaining = 64 - state.block_length;
        __mmask64 remaining_mask = sz_u64_clamp_mask_until_(remaining);
        _mm512_mask_storeu_epi8(&state.block[state.block_length], remaining_mask, _mm512_setzero_si512());
        sz_sha256_process_block_icelake_(state.hash, state.block);
        state.block_length = 0;
    }

    // Pad with zeros until we have 56 bytes using AVX-512 masked store
    sz_size_t remaining = 56 - state.block_length;
    __mmask64 remaining_mask = sz_u64_clamp_mask_until_(remaining);
    _mm512_mask_storeu_epi8(&state.block[state.block_length], remaining_mask, _mm512_setzero_si512());

    // Append the 64-bit length in bits (big-endian)
    sz_u64_t bit_length = state.total_length * 8;
    state.block[56] = (sz_u8_t)(bit_length >> 56);
    state.block[57] = (sz_u8_t)(bit_length >> 48);
    state.block[58] = (sz_u8_t)(bit_length >> 40);
    state.block[59] = (sz_u8_t)(bit_length >> 32);
    state.block[60] = (sz_u8_t)(bit_length >> 24);
    state.block[61] = (sz_u8_t)(bit_length >> 16);
    state.block[62] = (sz_u8_t)(bit_length >> 8);
    state.block[63] = (sz_u8_t)(bit_length >> 0);

    // Process the final block
    sz_sha256_process_block_icelake_(state.hash, state.block);

    // Produce the final hash digest in big-endian format
    for (sz_size_t lane_index = 0; lane_index < 8; ++lane_index) {
        digest[lane_index * 4 + 0] = (sz_u8_t)(state.hash[lane_index] >> 24);
        digest[lane_index * 4 + 1] = (sz_u8_t)(state.hash[lane_index] >> 16);
        digest[lane_index * 4 + 2] = (sz_u8_t)(state.hash[lane_index] >> 8);
        digest[lane_index * 4 + 3] = (sz_u8_t)(state.hash[lane_index] >> 0);
    }
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

#endif // STRINGZILLA_HASH_ICELAKE_H_
