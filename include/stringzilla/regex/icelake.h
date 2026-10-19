/**
 *  @file include/stringzilla/regex/icelake.h
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief AVX-512 (Ice Lake) backend for regular expressions: idle skips
 *      through the VBMI2 byte-set search.
 *
 *  Both automata stay scalar: a data-dependent chase has no vector form that beats eight scalar
 *  chains. While the start pass sits on its idle state, this tier's reverse byte-set search jumps
 *  to the next byte that leaves it, and its forward search carries end walks through the runs their
 *  states stay put on, 64 bytes per compare; every other stage is the serial one.
 *  Where every match starts with a few literal bytes, nibble lookups at each offset find the
 *  candidates instead of the start pass, 64 positions per step. The lookaround table's one-byte
 *  lookarounds fill by byte permutations, 64 positions per step.
 *
 *  @sa include/stringzilla/regex.h
 */
#ifndef STRINGZILLA_REGEX_ICELAKE_H_
#define STRINGZILLA_REGEX_ICELAKE_H_

#include "stringzilla/types.h"

#include "stringzilla/find/icelake.h" // `sz_find_byteset_icelake_`, `sz_rfind_byteset_icelake_`
#include "stringzilla/regex/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_X8664_
#if STRINGZILLA_TARGET_ICELAKE
#if defined(__clang__) && STRINGZILLA_HAS_CLANG_EVEX512_
#pragma clang attribute push(                                                                                        \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,lzcnt,evex512"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(                                                                                \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,lzcnt"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2", \
                   "lzcnt")
#endif

#pragma region Ice Lake

/** One offset of @ref sz_regex_find_prefixes_icelake_: the buckets each byte's two nibbles keep. */
STRINGZILLA_INLINE __m512i sz_regex_prefix_buckets_icelake_(__m512i text_u8x64, __m512i low_u8x64, __m512i high_u8x64) {
    __m512i const nibble_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i const low_nibbles_u8x64 = _mm512_and_si512(text_u8x64, nibble_u8x64);
    __m512i const high_nibbles_u8x64 = _mm512_and_si512(_mm512_srli_epi16(text_u8x64, 4), nibble_u8x64);
    return _mm512_and_si512(_mm512_shuffle_epi8(low_u8x64, low_nibbles_u8x64),
                            _mm512_shuffle_epi8(high_u8x64, high_nibbles_u8x64));
}

/** @ref sz_regex_find_prefixes_serial_ testing nibbles, 64 positions per step; offsets past the
 *  text read zeros, which only reject positions too close to its end to start a match. */
STRINGZILLA_INLINE sz_cptr_t sz_regex_find_prefixes_icelake_(sz_cptr_t text, sz_size_t length,
                                                             sz_regex_prefixes_t const *prefixes) {
    __m512i low_u8x64[sz_regex_prefixes_depth_k], high_u8x64[sz_regex_prefixes_depth_k];
    sz_size_t position = 0;
    for (sz_size_t offset = 0; offset != sz_regex_prefixes_depth_k; ++offset)
        low_u8x64[offset] = _mm512_broadcast_i32x4(_mm_loadu_si128((__m128i const *)prefixes->nibbles[offset])),
        high_u8x64[offset] = _mm512_broadcast_i32x4(_mm_loadu_si128((__m128i const *)prefixes->nibbles[offset] + 1));
    for (; position + 64 + sz_regex_prefixes_depth_k - 1 <= length; position += 64) {
        __m512i const first_u8x64 = sz_regex_prefix_buckets_icelake_(_mm512_loadu_si512(text + position), low_u8x64[0],
                                                                     high_u8x64[0]);
        __m512i const second_u8x64 = sz_regex_prefix_buckets_icelake_(_mm512_loadu_si512(text + position + 1),
                                                                      low_u8x64[1], high_u8x64[1]);
        __m512i const third_u8x64 = sz_regex_prefix_buckets_icelake_(_mm512_loadu_si512(text + position + 2),
                                                                     low_u8x64[2], high_u8x64[2]);
        __m512i const kept_u8x64 = _mm512_ternarylogic_epi64(first_u8x64, second_u8x64, third_u8x64, 0x80);
        __mmask64 const kept_m64 = _mm512_test_epi8_mask(kept_u8x64, kept_u8x64);
        if (kept_m64) return text + position + _tzcnt_u64(kept_m64);
    }
    for (; position < length; position += 64) {
        __m512i kept_u8x64 = _mm512_set1_epi8(-1);
        __mmask64 kept_m64;
        for (sz_size_t offset = 0; offset != sz_regex_prefixes_depth_k; ++offset) {
            sz_size_t const loaded = length - position > offset ? length - position - offset : 0;
            __mmask64 const load_m64 = loaded >= 64 ? ~(__mmask64)0 : _bzhi_u64(~(sz_u64_t)0, (unsigned)loaded);
            kept_u8x64 = _mm512_and_si512(kept_u8x64, sz_regex_prefix_buckets_icelake_(
                                                          _mm512_maskz_loadu_epi8(load_m64, text + position + offset),
                                                          low_u8x64[offset], high_u8x64[offset]));
        }
        kept_m64 = _mm512_test_epi8_mask(kept_u8x64, kept_u8x64);
        if (length - position < 64) kept_m64 &= _bzhi_u64(~(sz_u64_t)0, (unsigned)(length - position));
        if (kept_m64) return text + position + _tzcnt_u64(kept_m64);
    }
    return STRINGZILLA_NULL;
}

/** Each byte's entry of a 256-entry byte table @p table, for any byte of @p text_u8x64. */
STRINGZILLA_INLINE __m512i sz_regex_lookup_icelake_(__m512i const *table, __m512i text_u8x64) {
    __m512i const low_u8x64 = _mm512_permutex2var_epi8(table[0], text_u8x64, table[1]);
    __m512i const high_u8x64 = _mm512_permutex2var_epi8(table[2], text_u8x64, table[3]);
    return _mm512_mask_blend_epi8(_mm512_movepi8_mask(text_u8x64), low_u8x64, high_u8x64);
}

/**
 *  @brief @ref sz_regex_fill_looks_serial_ by byte permutations, 64 positions per step, where every
 *      lookaround it writes is among the first eight.
 *
 *  The two tables narrow to bytes, each byte's entry one permutation of its low seven bits per half
 *  and a blend on its top bit; the entries widen back to look bits on the store.
 */
STRINGZILLA_C_INLINE_ void sz_regex_fill_looks_icelake_(sz_cptr_t text, sz_size_t length, sz_u32_t const *looks_bytes,
                                                        sz_u32_t flipped, sz_u32_t *looks) {
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    sz_u32_t const *const after = looks_bytes + 1, *const before = after + 256;
    __m512i const flipped_u8x64 = _mm512_set1_epi8((char)(flipped / sz_regex_look_lookaround_k));
    __m512i after_u8x64[4], before_u8x64[4];
    sz_u8_t narrowed[512];
    if (((looks_bytes[0] | flipped) / sz_regex_look_lookaround_k) >> 8 || !length) {
        sz_regex_fill_looks_serial_(text, length, looks_bytes, flipped, looks);
        return;
    }
    for (sz_size_t entry = 0; entry != 512; entry += 16)
        _mm_storeu_si128((__m128i *)(narrowed + entry),
                         _mm512_cvtepi32_epi8(_mm512_srli_epi32(_mm512_loadu_si512(after + entry), 6)));
    for (sz_size_t quarter = 0; quarter != 4; ++quarter)
        after_u8x64[quarter] = _mm512_loadu_si512(narrowed + quarter * 64),
        before_u8x64[quarter] = _mm512_loadu_si512(narrowed + 256 + quarter * 64);
    // Position zero has no byte before it, and the table's last entry no byte at it.
    looks[0] = before[bytes[0]] ^ flipped;
    for (sz_size_t position = 1; position < length; position += 64) {
        sz_size_t const count = sz_min_of_two(length - position, (sz_size_t)64);
        __mmask64 const load_m64 = _bzhi_u64(~(sz_u64_t)0, (unsigned)count);
        __m512i const at_u8x64 = _mm512_maskz_loadu_epi8(load_m64, bytes + position);
        __m512i const previous_u8x64 = _mm512_maskz_loadu_epi8(load_m64, bytes + position - 1);
        // 0x56 is `(a | b) ^ c`.
        __m512i const held_u8x64 = _mm512_ternarylogic_epi64(sz_regex_lookup_icelake_(after_u8x64, previous_u8x64),
                                                             sz_regex_lookup_icelake_(before_u8x64, at_u8x64),
                                                             flipped_u8x64, 0x56);
        __m128i const quarters_u8x16[4] = {_mm512_castsi512_si128(held_u8x64), _mm512_extracti32x4_epi32(held_u8x64, 1),
                                           _mm512_extracti32x4_epi32(held_u8x64, 2),
                                           _mm512_extracti32x4_epi32(held_u8x64, 3)};
        for (sz_size_t quarter = 0; quarter != 4 && quarter * 16 < count; ++quarter)
            _mm512_mask_storeu_epi32(looks + position + quarter * 16,
                                     (__mmask16)_bzhi_u32(0xFFFF, (unsigned)(count - quarter * 16)),
                                     _mm512_slli_epi32(_mm512_cvtepu8_epi32(quarters_u8x16[quarter]), 6));
    }
    looks[length] = after[bytes[length - 1]] ^ flipped;
}

STRINGZILLA_API sz_status_t sz_regex_engine_init_icelake(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                         sz_case_sensitivity_t case_sensitivity,
                                                         sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                         sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_engine_compile_(engine, patterns, case_sensitivity, malformed_index, sz_cap_icelake_k, allocator,
                                    sz_regex_reverse_budget_k, sz_regex_forward_budget_k, stream);
}

STRINGZILLA_API sz_status_t sz_regex_counts_icelake(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                    sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_counts_with_(engine, haystacks, counts, counts_stride, &sz_rfind_byteset_icelake_,
                                 &sz_find_byteset_icelake_, &sz_regex_find_prefixes_icelake_,
                                 &sz_regex_fill_looks_icelake_);
}

STRINGZILLA_API sz_status_t sz_regex_find_icelake(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                  sz_match_t *matches, sz_size_t matches_capacity,
                                                  sz_size_t *matches_offsets, sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_find_with_(engine, haystacks, matches, matches_capacity, matches_offsets,
                               &sz_rfind_byteset_icelake_, &sz_find_byteset_icelake_, &sz_regex_find_prefixes_icelake_,
                               &sz_regex_fill_looks_icelake_);
}

STRINGZILLA_API sz_status_t sz_regex_replace_icelake(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                     sz_strings_view_t const *replacements, sz_ptr_t target,
                                                     sz_size_t target_capacity, sz_size_t *offsets,
                                                     sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_replace_with_(engine, haystacks, replacements, target, target_capacity, offsets,
                                  &sz_rfind_byteset_icelake_, &sz_find_byteset_icelake_,
                                  &sz_regex_find_prefixes_icelake_, &sz_regex_fill_looks_icelake_);
}

STRINGZILLA_API sz_status_t sz_regex_extract_icelake(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                     sz_size_t pattern_index, sz_match_t *groups,
                                                     sz_size_t groups_capacity, sz_size_t *groups_offsets,
                                                     sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_extract_with_(engine, haystacks, pattern_index, groups, groups_capacity, groups_offsets,
                                  &sz_rfind_byteset_icelake_, &sz_find_byteset_icelake_,
                                  &sz_regex_find_prefixes_icelake_, &sz_regex_fill_looks_icelake_);
}

#pragma endregion Ice Lake

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_ICELAKE
#endif // STRINGZILLA_ARCH_X8664_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_REGEX_ICELAKE_H_
