/**
 *  @file include/stringzilla/regex/powervsx.h
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief IBM Power VSX backend for regular expressions: idle skips via the VSX byte-set search.
 *
 *  Both automata stay scalar: a data-dependent chase has no vector form that beats eight scalar
 *  chains. While the start pass sits on its idle state, this tier's reverse byte-set search jumps
 *  to the next byte that leaves it, and its forward search carries end walks through the runs their
 *  states stay put on, 16 bytes per compare; every other stage is the serial one.
 *  Where every match starts with a few literal bytes, nibble permutes at each offset find the
 *  candidates instead of the start pass, 16 positions per step. The lookaround table's one-byte
 *  lookarounds fill by sixteen-register permutes, 16 positions per step.
 *
 *  @sa c/stringzilla/regex.h
 */
#ifndef STRINGZILLA_REGEX_POWERVSX_H_
#define STRINGZILLA_REGEX_POWERVSX_H_

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_PPC64_
#if STRINGZILLA_ARCH_PPC64_POWERVSX_

#include "stringzilla/find/powervsx.h" // `sz_find_byteset_powervsx_`, `sz_rfind_byteset_powervsx_`
#include "stringzilla/regex/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

#pragma region Power VSX

/** One offset of @ref sz_regex_find_prefixes_powervsx_: the buckets each byte's two nibbles keep.
 *  A permute of one register with itself reads the low four index bits, so no mask is needed. */
STRINGZILLA_INLINE __vector unsigned char sz_regex_prefix_buckets_powervsx_(__vector unsigned char text_u8x16,
                                                                            __vector unsigned char low_u8x16,
                                                                            __vector unsigned char high_u8x16) {
    return vec_and(vec_perm(low_u8x16, low_u8x16, text_u8x16),
                   vec_perm(high_u8x16, high_u8x16, vec_sr(text_u8x16, vec_splats((unsigned char)4))));
}

/** @ref sz_regex_find_prefixes_serial_ testing nibbles, 16 positions per step, the last few
 *  positions a byte at a time. */
STRINGZILLA_INLINE sz_cptr_t sz_regex_find_prefixes_powervsx_(sz_cptr_t text, sz_size_t length,
                                                              sz_regex_prefixes_t const *prefixes) {
    __vector unsigned char low_u8x16[sz_regex_prefixes_depth_k], high_u8x16[sz_regex_prefixes_depth_k];
    sz_size_t position = 0;
    for (sz_size_t offset = 0; offset != sz_regex_prefixes_depth_k; ++offset)
        low_u8x16[offset] = vec_xl(0, prefixes->nibbles[offset]),
        high_u8x16[offset] = vec_xl(16, prefixes->nibbles[offset]);
    for (; position + 16 + sz_regex_prefixes_depth_k - 1 <= length; position += 16) {
        sz_u8_t const *const bytes = (sz_u8_t const *)text + position;
        __vector unsigned char const kept_u8x16 = vec_and(
            vec_and(sz_regex_prefix_buckets_powervsx_(vec_xl(0, bytes), low_u8x16[0], high_u8x16[0]),
                    sz_regex_prefix_buckets_powervsx_(vec_xl(1, bytes), low_u8x16[1], high_u8x16[1])),
            sz_regex_prefix_buckets_powervsx_(vec_xl(2, bytes), low_u8x16[2], high_u8x16[2]));
        // POWER9 counts the leading lanes without a match in one `vclzlsbb`, sixteen when none.
        sz_size_t const first = (sz_size_t)vec_cntlz_lsbb(
            (__vector unsigned char)vec_cmpne(kept_u8x16, vec_splats((unsigned char)0)));
        if (first != 16) return text + position + first;
    }
    return sz_regex_find_prefixes_serial_(text + position, length - position, prefixes);
}

/** Each byte's entry of the 256-entry byte table held in @p table_u8x16 's sixteen registers:
 *  eight two-register permutes look up the low five bits, and three rounds of selects on the top
 *  three bits pick among them. */
STRINGZILLA_INLINE __vector unsigned char sz_regex_lookup_powervsx_(__vector unsigned char const *table_u8x16,
                                                                    __vector unsigned char text_u8x16) {
    __vector unsigned char const seven_u8x16 = vec_splats((unsigned char)7);
    __vector unsigned char eighths_u8x16[8];
    for (sz_size_t eighth = 0; eighth != 8; ++eighth)
        eighths_u8x16[eighth] = vec_perm(table_u8x16[2 * eighth], table_u8x16[2 * eighth + 1], text_u8x16);
    // Shifting a bit into the sign and back arithmetically fills its byte, ready for a select.
    for (sz_size_t bit = 5, count = 8; bit != 8; ++bit, count /= 2) {
        __vector unsigned char const selector_u8x16 = (__vector unsigned char)vec_sra(
            (__vector signed char)vec_sl(text_u8x16, vec_splats((unsigned char)(7 - bit))), seven_u8x16);
        for (sz_size_t half = 0; half != count / 2; ++half)
            eighths_u8x16[half] = vec_sel(eighths_u8x16[2 * half], eighths_u8x16[2 * half + 1], selector_u8x16);
    }
    return eighths_u8x16[0];
}

/** Narrows 256 look-bit entries to the bytes of their eight lowest lookarounds, into the sixteen
 *  registers of @p table_u8x16, packing in registers with no pass through memory. */
STRINGZILLA_INLINE void sz_regex_narrow_looks_powervsx_(sz_u32_t const *entries, __vector unsigned char *table_u8x16) {
    __vector unsigned int const shift_u32x4 = vec_splats(6u);
    for (sz_size_t sixteenth = 0; sixteenth != 16; ++sixteenth) {
        sz_u32_t const *const block = entries + sixteenth * 16;
        __vector unsigned short const low_u16x8 = vec_pack(vec_sr(vec_xl(0, block), shift_u32x4),
                                                           vec_sr(vec_xl(16, block), shift_u32x4));
        __vector unsigned short const high_u16x8 = vec_pack(vec_sr(vec_xl(32, block), shift_u32x4),
                                                            vec_sr(vec_xl(48, block), shift_u32x4));
        table_u8x16[sixteenth] = vec_pack(low_u16x8, high_u16x8);
    }
}

/**
 *  @brief @ref sz_regex_fill_looks_serial_ by permutes, 16 positions per step, where every
 *      lookaround it writes is among the first eight.
 *
 *  The two tables narrow to bytes, each byte's entry eight permutes and seven selects; the entries
 *  widen back to look bits on the store, and the last few positions take the serial tables.
 */
STRINGZILLA_INLINE void sz_regex_fill_looks_powervsx_(sz_cptr_t text, sz_size_t length, sz_u32_t const *looks_bytes,
                                                      sz_u32_t flipped, sz_u32_t *looks) {
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    sz_u32_t const *const after = looks_bytes + 1, *const before = after + 256;
    __vector unsigned char after_u8x16[16], before_u8x16[16];
    __vector unsigned char const flipped_u8x16 = vec_splats((unsigned char)(flipped / sz_regex_look_lookaround_k));
    __vector unsigned char const zero_u8x16 = vec_splats((unsigned char)0);
    __vector unsigned int const shift_u32x4 = vec_splats(6u);
    // Each word's low byte indexes a narrowed entry and its other three the zero register, and a
    // word constant lays its low byte where the target's word keeps it, on either byte order.
    __vector unsigned int const widen_u32x4 = {0x10101000u, 0x10101001u, 0x10101002u, 0x10101003u};
    sz_size_t position = 1;
    if (((looks_bytes[0] | flipped) / sz_regex_look_lookaround_k) >> 8 || !length) {
        sz_regex_fill_looks_serial_(text, length, looks_bytes, flipped, looks);
        return;
    }
    sz_regex_narrow_looks_powervsx_(after, after_u8x16);
    sz_regex_narrow_looks_powervsx_(before, before_u8x16);
    // Position zero has no byte before it, and the table's last entry no byte at it.
    looks[0] = before[bytes[0]] ^ flipped;
    for (; position + 16 <= length; position += 16) {
        __vector unsigned char const held_u8x16 = vec_xor(
            vec_or(sz_regex_lookup_powervsx_(after_u8x16, vec_xl(0, bytes + position - 1)),
                   sz_regex_lookup_powervsx_(before_u8x16, vec_xl(0, bytes + position))),
            flipped_u8x16);
        for (sz_size_t quarter = 0; quarter != 4; ++quarter) {
            __vector unsigned char const indices_u8x16 = (__vector unsigned char)vec_add(
                widen_u32x4, vec_splats((unsigned int)(quarter * 4)));
            __vector unsigned int const widened_u32x4 = (__vector unsigned int)vec_perm(held_u8x16, zero_u8x16,
                                                                                        indices_u8x16);
            vec_xst(vec_sl(widened_u32x4, shift_u32x4), 0, looks + position + quarter * 4);
        }
    }
    for (; position != length; ++position)
        looks[position] = (after[bytes[position - 1]] | before[bytes[position]]) ^ flipped;
    looks[length] = after[bytes[length - 1]] ^ flipped;
}

#if STRINGZILLA_TARGET_POWERVSX

STRINGZILLA_API sz_status_t sz_regex_engine_init_powervsx(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                          sz_case_sensitivity_t case_sensitivity,
                                                          sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                          sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_engine_compile_(engine, patterns, case_sensitivity, malformed_index, sz_cap_powervsx_k, allocator,
                                    sz_regex_reverse_budget_k, sz_regex_forward_budget_k, stream);
}

STRINGZILLA_API sz_status_t sz_regex_contains_powervsx(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                       sz_u1x8_t *contained, sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_contains_with_(engine, haystacks, contained, &sz_rfind_byteset_powervsx_,
                                   &sz_find_byteset_powervsx_, &sz_regex_find_prefixes_powervsx_,
                                   &sz_regex_fill_looks_powervsx_);
}

STRINGZILLA_API sz_status_t sz_regex_counts_powervsx(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                     sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_counts_with_(engine, haystacks, counts, counts_stride, &sz_rfind_byteset_powervsx_,
                                 &sz_find_byteset_powervsx_, &sz_regex_find_prefixes_powervsx_,
                                 &sz_regex_fill_looks_powervsx_);
}

STRINGZILLA_API sz_status_t sz_regex_find_powervsx(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                   sz_match_t *matches, sz_size_t matches_capacity,
                                                   sz_size_t *matches_offsets, sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_find_with_(engine, haystacks, matches, matches_capacity, matches_offsets,
                               &sz_rfind_byteset_powervsx_, &sz_find_byteset_powervsx_,
                               &sz_regex_find_prefixes_powervsx_, &sz_regex_fill_looks_powervsx_);
}

STRINGZILLA_API sz_status_t sz_regex_replace_powervsx(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                      sz_strings_view_t const *replacements, sz_ptr_t target,
                                                      sz_size_t target_capacity, sz_size_t *offsets,
                                                      sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_replace_with_(engine, haystacks, replacements, target, target_capacity, offsets,
                                  &sz_rfind_byteset_powervsx_, &sz_find_byteset_powervsx_,
                                  &sz_regex_find_prefixes_powervsx_, &sz_regex_fill_looks_powervsx_);
}

STRINGZILLA_API sz_status_t sz_regex_extract_powervsx(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                      sz_size_t pattern_index, sz_match_t *groups,
                                                      sz_size_t groups_capacity, sz_size_t *groups_offsets,
                                                      sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_extract_with_(engine, haystacks, pattern_index, groups, groups_capacity, groups_offsets,
                                  &sz_rfind_byteset_powervsx_, &sz_find_byteset_powervsx_,
                                  &sz_regex_find_prefixes_powervsx_, &sz_regex_fill_looks_powervsx_);
}

#pragma endregion Power VSX

#endif // STRINGZILLA_TARGET_POWERVSX

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // STRINGZILLA_ARCH_PPC64_POWERVSX_
#endif // STRINGZILLA_ARCH_PPC64_

#endif // STRINGZILLA_REGEX_POWERVSX_H_
