/**
 *  @file include/stringzilla/regex/haswell.h
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief AVX2 (Haswell) backend for regular expressions: idle skips
 *      through the AVX2 byte-set search.
 *
 *  Both automata stay scalar: a data-dependent chase has no vector form that beats eight scalar
 *  chains. While the start pass sits on its idle state, this tier's reverse byte-set search jumps
 *  to the next byte that leaves it, and its forward search carries end walks through the runs their
 *  states stay put on, 32 bytes per compare; every other stage is the serial one.
 *  Where every match starts with a few literal bytes, nibble lookups at each offset find the
 *  candidates instead of the start pass, 32 positions per step.
 *
 *  @sa include/stringzilla/regex.h
 */
#ifndef STRINGZILLA_REGEX_HASWELL_H_
#define STRINGZILLA_REGEX_HASWELL_H_

#include "stringzilla/types.h"

#include "stringzilla/find/haswell.h" // `sz_find_byteset_haswell_`, `sz_rfind_byteset_haswell_`
#include "stringzilla/regex/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_X8664_
#if STRINGZILLA_ARCH_X8664_HASWELL_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,lzcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "bmi", "bmi2", "lzcnt")
#endif

#pragma region Haswell

/** One offset of @ref sz_regex_find_prefixes_haswell_: the buckets each byte's two nibbles keep. */
STRINGZILLA_INLINE __m256i sz_regex_prefix_buckets_haswell_(__m256i text_u8x32, __m256i low_u8x32, __m256i high_u8x32) {
    __m256i const nibble_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i const low_nibbles_u8x32 = _mm256_and_si256(text_u8x32, nibble_u8x32);
    __m256i const high_nibbles_u8x32 = _mm256_and_si256(_mm256_srli_epi16(text_u8x32, 4), nibble_u8x32);
    return _mm256_and_si256(_mm256_shuffle_epi8(low_u8x32, low_nibbles_u8x32),
                            _mm256_shuffle_epi8(high_u8x32, high_nibbles_u8x32));
}

/** @ref sz_regex_find_prefixes_serial_ testing nibbles, 32 positions per step, the last few
 *  positions a byte at a time. */
STRINGZILLA_INLINE sz_cptr_t sz_regex_find_prefixes_haswell_(sz_cptr_t text, sz_size_t length,
                                                             sz_regex_prefixes_t const *prefixes) {
    __m256i low_u8x32[sz_regex_prefixes_depth_k], high_u8x32[sz_regex_prefixes_depth_k];
    __m256i const zero_u8x32 = _mm256_setzero_si256();
    sz_size_t position = 0;
    for (sz_size_t offset = 0; offset != sz_regex_prefixes_depth_k; ++offset)
        low_u8x32[offset] = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i const *)prefixes->nibbles[offset])),
        high_u8x32[offset] = _mm256_broadcastsi128_si256(
            _mm_loadu_si128((__m128i const *)prefixes->nibbles[offset] + 1));
    for (; position + 32 + sz_regex_prefixes_depth_k - 1 <= length; position += 32) {
        __m256i const first_u8x32 = sz_regex_prefix_buckets_haswell_(
            _mm256_loadu_si256((__m256i const *)(text + position)), low_u8x32[0], high_u8x32[0]);
        __m256i const second_u8x32 = sz_regex_prefix_buckets_haswell_(
            _mm256_loadu_si256((__m256i const *)(text + position + 1)), low_u8x32[1], high_u8x32[1]);
        __m256i const third_u8x32 = sz_regex_prefix_buckets_haswell_(
            _mm256_loadu_si256((__m256i const *)(text + position + 2)), low_u8x32[2], high_u8x32[2]);
        __m256i const kept_u8x32 = _mm256_and_si256(_mm256_and_si256(first_u8x32, second_u8x32), third_u8x32);
        sz_u32_t const kept = ~(sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(kept_u8x32, zero_u8x32));
        if (kept) return text + position + _tzcnt_u32(kept);
    }
    return sz_regex_find_prefixes_serial_(text + position, length - position, prefixes);
}

#if STRINGZILLA_TARGET_HASWELL

STRINGZILLA_API sz_status_t sz_regex_engine_init_haswell(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                         sz_case_sensitivity_t case_sensitivity,
                                                         sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                         sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_engine_compile_(engine, patterns, case_sensitivity, malformed_index, sz_cap_haswell_k, allocator,
                                    sz_regex_reverse_budget_k, sz_regex_forward_budget_k, stream);
}

STRINGZILLA_API sz_status_t sz_regex_contains_haswell(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                      sz_u1x8_t *contained, sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_contains_with_(engine, haystacks, contained, &sz_rfind_byteset_haswell_, &sz_find_byteset_haswell_,
                                   &sz_regex_find_prefixes_haswell_, &sz_regex_fill_looks_serial_);
}

STRINGZILLA_API sz_status_t sz_regex_counts_haswell(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                    sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_counts_with_(engine, haystacks, counts, counts_stride, &sz_rfind_byteset_haswell_,
                                 &sz_find_byteset_haswell_, &sz_regex_find_prefixes_haswell_,
                                 &sz_regex_fill_looks_serial_);
}

STRINGZILLA_API sz_status_t sz_regex_find_haswell(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                  sz_match_t *matches, sz_size_t matches_capacity,
                                                  sz_size_t *matches_offsets, sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_find_with_(engine, haystacks, matches, matches_capacity, matches_offsets,
                               &sz_rfind_byteset_haswell_, &sz_find_byteset_haswell_, &sz_regex_find_prefixes_haswell_,
                               &sz_regex_fill_looks_serial_);
}

STRINGZILLA_API sz_status_t sz_regex_replace_haswell(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                     sz_strings_view_t const *replacements, sz_ptr_t target,
                                                     sz_size_t target_capacity, sz_size_t *offsets,
                                                     sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_replace_with_(engine, haystacks, replacements, target, target_capacity, offsets,
                                  &sz_rfind_byteset_haswell_, &sz_find_byteset_haswell_,
                                  &sz_regex_find_prefixes_haswell_, &sz_regex_fill_looks_serial_);
}

STRINGZILLA_API sz_status_t sz_regex_extract_haswell(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                     sz_size_t pattern_index, sz_match_t *groups,
                                                     sz_size_t groups_capacity, sz_size_t *groups_offsets,
                                                     sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_extract_with_(engine, haystacks, pattern_index, groups, groups_capacity, groups_offsets,
                                  &sz_rfind_byteset_haswell_, &sz_find_byteset_haswell_,
                                  &sz_regex_find_prefixes_haswell_, &sz_regex_fill_looks_serial_);
}

#endif // STRINGZILLA_TARGET_HASWELL

#pragma endregion Haswell

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_X8664_HASWELL_
#endif // STRINGZILLA_ARCH_X8664_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_REGEX_HASWELL_H_
