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
 *  candidates instead of the start pass, 16 positions per step. The lookaround table's one-byte
 *  lookarounds fill by four-register table lookups, 16 positions per step.
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

/** Each byte's entry of a 256-entry byte table @p table, for any byte of @p text_u8x16; indices
 *  past each quarter's 64 entries look up zero. */
STRINGZILLA_INLINE uint8x16_t sz_regex_lookup_neon_(uint8x16x4_t const *table, uint8x16_t text_u8x16) {
    uint8x16_t const quarter_u8x16 = vdupq_n_u8(64);
    uint8x16_t held_u8x16 = vqtbl4q_u8(table[0], text_u8x16);
    for (sz_size_t quarter = 1; quarter != 4; ++quarter)
        text_u8x16 = vsubq_u8(text_u8x16, quarter_u8x16),
        held_u8x16 = vorrq_u8(held_u8x16, vqtbl4q_u8(table[quarter], text_u8x16));
    return held_u8x16;
}

/** Narrows 256 look-bit entries to the bytes of their eight lowest lookarounds, into @p table. */
STRINGZILLA_INLINE void sz_regex_narrow_looks_neon_(sz_u32_t const *entries, uint8x16x4_t *table) {
    sz_u8_t narrowed[256];
    for (sz_size_t entry = 0; entry != 256; entry += 8) {
        uint16x8_t const looks_u16x8 = vcombine_u16(vmovn_u32(vshrq_n_u32(vld1q_u32(entries + entry), 6)),
                                                    vmovn_u32(vshrq_n_u32(vld1q_u32(entries + entry + 4), 6)));
        vst1_u8(narrowed + entry, vmovn_u16(looks_u16x8));
    }
    for (sz_size_t quarter = 0; quarter != 4; ++quarter) table[quarter] = vld1q_u8_x4(narrowed + quarter * 64);
}

/**
 *  @brief @ref sz_regex_fill_looks_serial_ by table lookups, 16 positions per step, where every
 *      lookaround it writes is among the first eight.
 *
 *  The two tables narrow to bytes, each byte's entry four lookups of 64 entries; the entries widen
 *  back to look bits on the store, and the last few positions take the serial tables.
 */
STRINGZILLA_INLINE void sz_regex_fill_looks_neon_(sz_cptr_t text, sz_size_t length, sz_u32_t const *looks_bytes,
                                                  sz_u32_t flipped, sz_u32_t *looks) {
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    sz_u32_t const *const after = looks_bytes + 1, *const before = after + 256;
    uint8x16x4_t after_u8x16x4[4], before_u8x16x4[4];
    uint8x16_t const flipped_u8x16 = vdupq_n_u8((sz_u8_t)(flipped / sz_regex_look_lookaround_k));
    sz_size_t position = 1;
    if (((looks_bytes[0] | flipped) / sz_regex_look_lookaround_k) >> 8 || !length) {
        sz_regex_fill_looks_serial_(text, length, looks_bytes, flipped, looks);
        return;
    }
    sz_regex_narrow_looks_neon_(after, after_u8x16x4);
    sz_regex_narrow_looks_neon_(before, before_u8x16x4);
    // Position zero has no byte before it, and the table's last entry no byte at it.
    looks[0] = before[bytes[0]] ^ flipped;
    for (; position + 16 <= length; position += 16) {
        uint8x16_t const held_u8x16 = veorq_u8(
            vorrq_u8(sz_regex_lookup_neon_(after_u8x16x4, vld1q_u8(bytes + position - 1)),
                     sz_regex_lookup_neon_(before_u8x16x4, vld1q_u8(bytes + position))),
            flipped_u8x16);
        uint16x8_t const low_u16x8 = vmovl_u8(vget_low_u8(held_u8x16));
        uint16x8_t const high_u16x8 = vmovl_u8(vget_high_u8(held_u8x16));
        vst1q_u32(looks + position, vshlq_n_u32(vmovl_u16(vget_low_u16(low_u16x8)), 6));
        vst1q_u32(looks + position + 4, vshlq_n_u32(vmovl_u16(vget_high_u16(low_u16x8)), 6));
        vst1q_u32(looks + position + 8, vshlq_n_u32(vmovl_u16(vget_low_u16(high_u16x8)), 6));
        vst1q_u32(looks + position + 12, vshlq_n_u32(vmovl_u16(vget_high_u16(high_u16x8)), 6));
    }
    for (; position != length; ++position)
        looks[position] = (after[bytes[position - 1]] | before[bytes[position]]) ^ flipped;
    looks[length] = after[bytes[length - 1]] ^ flipped;
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
                                 &sz_find_byteset_neon_, &sz_regex_find_prefixes_neon_, &sz_regex_fill_looks_neon_);
}

STRINGZILLA_API sz_status_t sz_regex_find_neon(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                               sz_match_t *matches, sz_size_t matches_capacity,
                                               sz_size_t *matches_offsets, sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_find_with_(engine, haystacks, matches, matches_capacity, matches_offsets, &sz_rfind_byteset_neon_,
                               &sz_find_byteset_neon_, &sz_regex_find_prefixes_neon_, &sz_regex_fill_looks_neon_);
}

STRINGZILLA_API sz_status_t sz_regex_replace_neon(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                  sz_strings_view_t const *replacements, sz_ptr_t target,
                                                  sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_replace_with_(engine, haystacks, replacements, target, target_capacity, offsets,
                                  &sz_rfind_byteset_neon_, &sz_find_byteset_neon_, &sz_regex_find_prefixes_neon_,
                                  &sz_regex_fill_looks_neon_);
}

STRINGZILLA_API sz_status_t sz_regex_extract_neon(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                  sz_size_t pattern_index, sz_match_t *groups,
                                                  sz_size_t groups_capacity, sz_size_t *groups_offsets,
                                                  sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_extract_with_(engine, haystacks, pattern_index, groups, groups_capacity, groups_offsets,
                                  &sz_rfind_byteset_neon_, &sz_find_byteset_neon_, &sz_regex_find_prefixes_neon_,
                                  &sz_regex_fill_looks_neon_);
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
