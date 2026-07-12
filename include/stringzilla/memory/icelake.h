/**
 *  @brief AVX-512 VBMI backend for hardware-accelerated memory operations on Ice Lake and newer x86 CPUs.
 *  @file include/stringzilla/memory/icelake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_ICELAKE_H_
#define STRINGZILLA_MEMORY_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_ICELAKE
#if defined(__clang__) && SZ_CLANG_HAS_EVEX512_
#pragma clang attribute push(                                                                      \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,bmi,bmi2,evex512"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "bmi", "bmi2")
#endif

SZ_API_COMPTIME void sz_lookup_icelake(sz_ptr_t target, sz_size_t length, sz_cptr_t source,
                                       char const lut[sz_at_least_(256)]) {

    // If the input is tiny (especially smaller than the look-up table itself), we may end up paying
    // more for organizing the SIMD registers and changing the CPU state, than for the actual computation.
    // But if at least 3 cache lines are touched, the AVX-512 implementation should be faster.
    if (length <= 128) {
        // Inline the serial scalar tail (one indexed byte-LUT copy) rather than paying a separate call hop on the
        // short path - the dispatched entry already cost one indirect call, and this loop is trivial.
        sz_u8_t const *lut_u8 = (sz_u8_t const *)lut;
        sz_u8_t const *source_u8 = (sz_u8_t const *)source;
        sz_u8_t *target_u8 = (sz_u8_t *)target;
        for (sz_size_t byte_index = 0; byte_index < length; ++byte_index)
            target_u8[byte_index] = lut_u8[source_u8[byte_index]];
        return;
    }

    // When the buffer is over 64 bytes, it's guaranteed to touch at least two cache lines - the head and tail,
    // and may include more cache-lines in-between. Knowing this, we can avoid expensive unaligned stores
    // by computing 2 masks - for the head and tail, using masked stores for the head and tail, and unmasked
    // for the body.
    sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64; // 63 or less.
    sz_size_t tail_length = (sz_size_t)(target + length) % 64;    // 63 or less.
    __mmask64 head_mask = sz_u64_mask_until_(head_length);
    __mmask64 tail_mask = sz_u64_mask_until_(tail_length);

    // We use VPERMI2B (`_mm512_permutex2var_epi8`) to perform 256-entry lookups efficiently.
    // VPERMI2B uses bit 6 of each index to select between two 64-byte tables, allowing us to
    // cover 128 entries per instruction (2 instructions for all 256 entries).
    //
    // For the high-bit (bit 7) selection, we use VPMOVB2M (`_mm512_movepi8_mask`) which extracts
    // the sign bit of each byte directly to a mask register. This goes to port 0 on Intel,
    // avoiding the port 5 bottleneck that VPTESTMB would cause.
    sz_u512_vec_t lut_0_to_63_vec, lut_64_to_127_vec, lut_128_to_191_vec, lut_192_to_255_vec;
    lut_0_to_63_vec.zmm = _mm512_loadu_si512((lut));
    lut_64_to_127_vec.zmm = _mm512_loadu_si512((lut + 64));
    lut_128_to_191_vec.zmm = _mm512_loadu_si512((lut + 128));
    lut_192_to_255_vec.zmm = _mm512_loadu_si512((lut + 192));

    __mmask64 high_bit_mask;
    sz_u512_vec_t source_vec, low_half_vec, high_half_vec, result_vec;

    // Handling the head.
    if (head_length) {
        source_vec.zmm = _mm512_maskz_loadu_epi8(head_mask, source);
        // VPERMI2B: bit 6 selects between the two tables, bits 0-5 index within each
        low_half_vec.zmm = _mm512_permutex2var_epi8(lut_0_to_63_vec.zmm, source_vec.zmm, lut_64_to_127_vec.zmm);
        high_half_vec.zmm = _mm512_permutex2var_epi8(lut_128_to_191_vec.zmm, source_vec.zmm, lut_192_to_255_vec.zmm);
        // VPMOVB2M: extract bit 7 (sign bit) of each byte directly to mask - uses port 0, not port 5
        high_bit_mask = _mm512_movepi8_mask(source_vec.zmm);
        result_vec.zmm = _mm512_mask_blend_epi8(high_bit_mask, low_half_vec.zmm, high_half_vec.zmm);
        _mm512_mask_storeu_epi8(target, head_mask, result_vec.zmm);
        source += head_length, target += head_length, length -= head_length;
    }

    // Handling the body in 64-byte chunks aligned to cache-line boundaries with respect to `target`.
    while (length >= 64) {
        source_vec.zmm = _mm512_loadu_si512(source);
        low_half_vec.zmm = _mm512_permutex2var_epi8(lut_0_to_63_vec.zmm, source_vec.zmm, lut_64_to_127_vec.zmm);
        high_half_vec.zmm = _mm512_permutex2var_epi8(lut_128_to_191_vec.zmm, source_vec.zmm, lut_192_to_255_vec.zmm);
        high_bit_mask = _mm512_movepi8_mask(source_vec.zmm);
        result_vec.zmm = _mm512_mask_blend_epi8(high_bit_mask, low_half_vec.zmm, high_half_vec.zmm);
        _mm512_store_si512(target, result_vec.zmm); //! Aligned store, our main weapon!
        source += 64, target += 64, length -= 64;
    }

    // Handling the tail.
    if (tail_length) {
        source_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask, source);
        low_half_vec.zmm = _mm512_permutex2var_epi8(lut_0_to_63_vec.zmm, source_vec.zmm, lut_64_to_127_vec.zmm);
        high_half_vec.zmm = _mm512_permutex2var_epi8(lut_128_to_191_vec.zmm, source_vec.zmm, lut_192_to_255_vec.zmm);
        high_bit_mask = _mm512_movepi8_mask(source_vec.zmm);
        result_vec.zmm = _mm512_mask_blend_epi8(high_bit_mask, low_half_vec.zmm, high_half_vec.zmm);
        _mm512_mask_storeu_epi8(target, tail_mask, result_vec.zmm);
        source += tail_length, target += tail_length, length -= tail_length;
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

#endif // STRINGZILLA_MEMORY_ICELAKE_H_
