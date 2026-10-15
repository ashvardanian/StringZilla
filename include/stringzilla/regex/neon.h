/**
 *  @file include/stringzilla/regex/neon.h
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief NEON backend for regular expressions: idle skips through the NEON byte-set search.
 *
 *  Both automata stay scalar: a data-dependent chase has no vector form that beats eight scalar
 *  chains. While the start pass sits on its idle state, this tier's reverse byte-set search jumps
 *  to the next byte that leaves it, and its forward search carries end walks through the runs their
 *  states stay put on, 16 bytes per compare; every other stage is the serial one.
 *  Where every match starts with a few literal bytes, nibble lookups at each offset find the
 *  candidates instead of the start pass, 16 positions per step.
 *
 *  @sa include/stringzilla/regex.h
 */
#ifndef STRINGZILLA_REGEX_NEON_H_
#define STRINGZILLA_REGEX_NEON_H_

#include "stringzilla/types.h"

#include "stringzilla/find/neon.h" // `sz_find_byteset_neon_`, `sz_rfind_byteset_neon_`
#include "stringzilla/regex/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_ARM64_
#if STRINGZILLA_ARCH_ARM64_NEON_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

#pragma region NEON

/** One offset of @ref sz_regex_find_prefixes_neon_: the buckets each byte's two nibbles keep. */
STRINGZILLA_INLINE uint8x16_t sz_regex_prefix_buckets_neon_(uint8x16_t text_u8x16, uint8x16_t low_u8x16,
                                                            uint8x16_t high_u8x16) {
    return vandq_u8(vqtbl1q_u8(low_u8x16, vandq_u8(text_u8x16, vdupq_n_u8(0x0F))),
                    vqtbl1q_u8(high_u8x16, vshrq_n_u8(text_u8x16, 4)));
}

/** @ref sz_regex_find_prefixes_serial_ testing nibbles, 16 positions per step, the last few
 *  positions a byte at a time. */
STRINGZILLA_INLINE sz_cptr_t sz_regex_find_prefixes_neon_(sz_cptr_t text, sz_size_t length,
                                                          sz_regex_prefixes_t const *prefixes) {
    uint8x16_t low_u8x16[sz_regex_prefixes_depth_k], high_u8x16[sz_regex_prefixes_depth_k];
    sz_size_t position = 0;
    for (sz_size_t offset = 0; offset != sz_regex_prefixes_depth_k; ++offset)
        low_u8x16[offset] = vld1q_u8(prefixes->nibbles[offset]),
        high_u8x16[offset] = vld1q_u8(prefixes->nibbles[offset] + 16);
    for (; position + 16 + sz_regex_prefixes_depth_k - 1 <= length; position += 16) {
        sz_u8_t const *const bytes = (sz_u8_t const *)text + position;
        uint8x16_t const kept_u8x16 = vandq_u8(
            vandq_u8(sz_regex_prefix_buckets_neon_(vld1q_u8(bytes), low_u8x16[0], high_u8x16[0]),
                     sz_regex_prefix_buckets_neon_(vld1q_u8(bytes + 1), low_u8x16[1], high_u8x16[1])),
            sz_regex_prefix_buckets_neon_(vld1q_u8(bytes + 2), low_u8x16[2], high_u8x16[2]));
        sz_u64_t const kept = sz_find_vreinterpretq_u8_u4_(vtstq_u8(kept_u8x16, kept_u8x16));
        if (kept) return text + position + sz_u64_ctz_neon_(kept) / 4;
    }
    return sz_regex_find_prefixes_serial_(text + position, length - position, prefixes);
}

#if STRINGZILLA_TARGET_NEON

STRINGZILLA_API sz_status_t sz_regex_engine_init_neon(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                      sz_case_sensitivity_t case_sensitivity,
                                                      sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                      sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_engine_compile_(engine, patterns, case_sensitivity, malformed_index, sz_cap_neon_k, allocator,
                                    sz_regex_reverse_budget_k, sz_regex_forward_budget_k, stream);
}

STRINGZILLA_API sz_status_t sz_regex_counts_neon(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                 sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_counts_with_(engine, haystacks, counts, counts_stride, &sz_rfind_byteset_neon_,
                                 &sz_find_byteset_neon_, &sz_regex_find_prefixes_neon_);
}

STRINGZILLA_API sz_status_t sz_regex_find_neon(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                               sz_match_t *matches, sz_size_t matches_capacity,
                                               sz_size_t *matches_offsets, sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_find_with_(engine, haystacks, matches, matches_capacity, matches_offsets, &sz_rfind_byteset_neon_,
                               &sz_find_byteset_neon_, &sz_regex_find_prefixes_neon_);
}

STRINGZILLA_API sz_status_t sz_regex_replace_neon(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                  sz_strings_view_t const *replacements, sz_ptr_t target,
                                                  sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_replace_with_(engine, haystacks, replacements, target, target_capacity, offsets,
                                  &sz_rfind_byteset_neon_, &sz_find_byteset_neon_, &sz_regex_find_prefixes_neon_);
}

STRINGZILLA_API sz_status_t sz_regex_extract_neon(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                  sz_size_t pattern_index, sz_match_t *groups,
                                                  sz_size_t groups_capacity, sz_size_t *groups_offsets,
                                                  sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_extract_with_(engine, haystacks, pattern_index, groups, groups_capacity, groups_offsets,
                                  &sz_rfind_byteset_neon_, &sz_find_byteset_neon_, &sz_regex_find_prefixes_neon_);
}

#endif // STRINGZILLA_TARGET_NEON

#pragma endregion NEON

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_ARM64_NEON_
#endif // STRINGZILLA_ARCH_ARM64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_REGEX_NEON_H_
