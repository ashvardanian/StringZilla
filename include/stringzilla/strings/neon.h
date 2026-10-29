/**
 *  @file include/stringzilla/strings/neon.h
 *  @author Ash Vardanian
 *  @date October 3, 2024
 *  @brief NEON backend for collections of strings on 64-bit Arm CPUs.
 *
 *  @sa c/stringzilla/strings.h
 */
#ifndef STRINGZILLA_STRINGS_NEON_H_
#define STRINGZILLA_STRINGS_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/find/neon.h" // `sz_find_byteset_neon_`
#include "stringzilla/utf8_norm.h" // `sz_utf8_norm_neon`, `sz_utf8_find_denormalized_neon`
#include "stringzilla/strings/serial.h"

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

STRINGZILLA_INLINE void sz_strings_lookup_bytes_neon_(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                                      char const lut[sz_at_least_(256)]) {
    sz_assert_no_overlap_(target, length, source, length);

    // Loading the table into registers only pays off for inputs longer than the table itself.
    if (length <= 128) {
        sz_strings_lookup_bytes_serial_(target, source, length, lut);
        return;
    }

    sz_size_t head_length = (16 - ((sz_size_t)target % 16)) % 16; // 15 or less.
    sz_size_t tail_length = (sz_size_t)(target + length) % 16;    // 15 or less.
    sz_size_t body_length = length - head_length - tail_length;

    // The table takes 16 of the 32 NEON registers; a 4-table lookup costs 6 cycles on Neoverse V2.
    uint8x16x4_t lut_0_to_63_vec, lut_64_to_127_vec, lut_128_to_191_vec, lut_192_to_255_vec;
    lut_0_to_63_vec = vld1q_u8_x4((sz_u8_t const *)(lut + 0));
    lut_64_to_127_vec = vld1q_u8_x4((sz_u8_t const *)(lut + 64));
    lut_128_to_191_vec = vld1q_u8_x4((sz_u8_t const *)(lut + 128));
    lut_192_to_255_vec = vld1q_u8_x4((sz_u8_t const *)(lut + 192));

    sz_u128_vec_t source_vec;
    sz_u128_vec_t lookup_0_to_63_vec, lookup_64_to_127_vec, lookup_128_to_191_vec, lookup_192_to_255_vec;
    sz_u128_vec_t blended_0_to_255_vec;

    // Process the head with serial code
    for (; head_length; target += 1, source += 1, head_length -= 1) *target = lut[*(sz_u8_t const *)source];

    // Each `vqtbl4q_u8` covers 64 entries and zeroes out-of-range lanes, so XOR-shifted indices
    // pick each quarter and an OR merges them, as in Lemire's arbitrary byte-to-byte maps:
    // https://lemire.me/blog/2019/07/23/arbitrary-byte-to-byte-maps-using-arm-neon/
    for (; body_length >= 16; source += 16, target += 16, body_length -= 16) {
        source_vec.u8x16 = vld1q_u8((sz_u8_t const *)source);
        lookup_0_to_63_vec.u8x16 = vqtbl4q_u8(lut_0_to_63_vec, source_vec.u8x16);
        lookup_64_to_127_vec.u8x16 = vqtbl4q_u8(lut_64_to_127_vec, veorq_u8(source_vec.u8x16, vdupq_n_u8(0x40)));
        lookup_128_to_191_vec.u8x16 = vqtbl4q_u8(lut_128_to_191_vec, veorq_u8(source_vec.u8x16, vdupq_n_u8(0x80)));
        lookup_192_to_255_vec.u8x16 = vqtbl4q_u8(lut_192_to_255_vec, veorq_u8(source_vec.u8x16, vdupq_n_u8(0xc0)));
        blended_0_to_255_vec.u8x16 = vorrq_u8( //
            vorrq_u8(lookup_0_to_63_vec.u8x16, lookup_64_to_127_vec.u8x16),
            vorrq_u8(lookup_128_to_191_vec.u8x16, lookup_192_to_255_vec.u8x16));
        vst1q_u8((sz_u8_t *)target, blended_0_to_255_vec.u8x16);
    }

    // Process the tail with serial code
    for (; tail_length; target += 1, source += 1, tail_length -= 1) *target = lut[*(sz_u8_t const *)source];
}

/** The bytes of @p text_u8x16 in the byteset @p set_u8x16x2 holds, as all-ones lanes. */
STRINGZILLA_INLINE uint8x16_t sz_strings_classify_neon_(uint8x16_t text_u8x16, uint8x16x2_t set_u8x16x2) {
    uint8x16_t const bits_u8x16 = vqtbl2q_u8(set_u8x16x2, vshrq_n_u8(text_u8x16, 3));
    uint8x16_t const bit_u8x16 = vshlq_u8(vdupq_n_u8(1), vreinterpretq_s8_u8(vandq_u8(text_u8x16, vdupq_n_u8(7))));
    return vtstq_u8(bits_u8x16, bit_u8x16);
}

/** The bits of four 0x00/0xFF masks of 16 lanes, lane @c i of the 64 in bit @c i. */
STRINGZILLA_INLINE sz_u64_t sz_strings_bits_neon_(uint8x16x4_t masks_u8x16x4) {
    static sz_u8_t const weights[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t const weights_u8x16 = vld1q_u8(weights);
    uint8x16_t const first_u8x16 = vpaddq_u8(vandq_u8(masks_u8x16x4.val[0], weights_u8x16),
                                             vandq_u8(masks_u8x16x4.val[1], weights_u8x16));
    uint8x16_t const second_u8x16 = vpaddq_u8(vandq_u8(masks_u8x16x4.val[2], weights_u8x16),
                                              vandq_u8(masks_u8x16x4.val[3], weights_u8x16));
    uint8x16_t const quarters_u8x16 = vpaddq_u8(first_u8x16, second_u8x16);
    return vgetq_lane_u64(vreinterpretq_u64_u8(vpaddq_u8(quarters_u8x16, quarters_u8x16)), 0);
}

/** The bytes among the 64 at @p source in the byteset @p set_u8x16x2 holds, as bits. */
STRINGZILLA_INLINE sz_u64_t sz_strings_split_removed_neon_(sz_u8_t const *source, uint8x16x2_t set_u8x16x2) {
    uint8x16x4_t masks_u8x16x4 = vld1q_u8_x4(source);
    masks_u8x16x4.val[0] = sz_strings_classify_neon_(masks_u8x16x4.val[0], set_u8x16x2);
    masks_u8x16x4.val[1] = sz_strings_classify_neon_(masks_u8x16x4.val[1], set_u8x16x2);
    masks_u8x16x4.val[2] = sz_strings_classify_neon_(masks_u8x16x4.val[2], set_u8x16x2);
    masks_u8x16x4.val[3] = sz_strings_classify_neon_(masks_u8x16x4.val[3], set_u8x16x2);
    return sz_strings_bits_neon_(masks_u8x16x4);
}

/** The bytes of line breaks among the 64 at @p source into @p removed, and the first byte of each
 *  break into @p ends. @p previous holds the CR, NEL and LS-or-PS masks of the block before and
 *  takes this one's; the 2 bytes after the block must be readable, as longer breaks look ahead. */
STRINGZILLA_INLINE void sz_strings_lines_classify_neon_(sz_u8_t const *source, sz_u64_t previous[3], sz_u64_t *removed,
                                                        sz_u64_t *ends) {
    uint8x16x4_t const text_u8x16x4 = vld1q_u8_x4(source);
    uint8x16x4_t singles_u8x16x4, returns_u8x16x4, feeds_u8x16x4;
    for (int chunk = 0; chunk != 4; ++chunk) {
        uint8x16_t const text_u8x16 = text_u8x16x4.val[chunk];
        // LF, VT, FF and CR run from 0x0A, and FS, GS and RS from 0x1C.
        singles_u8x16x4.val[chunk] = vorrq_u8(vcltq_u8(vsubq_u8(text_u8x16, vdupq_n_u8(0x0A)), vdupq_n_u8(4)),
                                              vcltq_u8(vsubq_u8(text_u8x16, vdupq_n_u8(0x1C)), vdupq_n_u8(3)));
        returns_u8x16x4.val[chunk] = vceqq_u8(text_u8x16, vdupq_n_u8('\r'));
        feeds_u8x16x4.val[chunk] = vceqq_u8(text_u8x16, vdupq_n_u8('\n'));
    }
    sz_u64_t const singles = sz_strings_bits_neon_(singles_u8x16x4), returns = sz_strings_bits_neon_(returns_u8x16x4);
    sz_u64_t const feeds_after_returns = sz_strings_bits_neon_(feeds_u8x16x4) & (returns << 1 | previous[0] >> 63);
    sz_u64_t nels = 0, separators = 0;
    uint8x16_t const highest_u8x16 = vmaxq_u8(vmaxq_u8(text_u8x16x4.val[0], text_u8x16x4.val[1]),
                                              vmaxq_u8(text_u8x16x4.val[2], text_u8x16x4.val[3]));
    // NEL starts with C2 and LS and PS with E2, so blocks below C2 hold no multi-byte break.
    if (vmaxvq_u8(highest_u8x16) >= 0xC2) {
        uint8x16x4_t const next1_u8x16x4 = vld1q_u8_x4(source + 1), next2_u8x16x4 = vld1q_u8_x4(source + 2);
        uint8x16x4_t nels_u8x16x4, separators_u8x16x4;
        for (int chunk = 0; chunk != 4; ++chunk) {
            uint8x16_t const text_u8x16 = text_u8x16x4.val[chunk];
            nels_u8x16x4.val[chunk] = vandq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0xC2)),
                                               vceqq_u8(next1_u8x16x4.val[chunk], vdupq_n_u8(0x85)));
            // LS and PS end in A8 and A9, which differ in the lowest bit alone.
            separators_u8x16x4.val[chunk] = vandq_u8(
                vandq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0xE2)), vceqq_u8(next1_u8x16x4.val[chunk], vdupq_n_u8(0x80))),
                vceqq_u8(vorrq_u8(next2_u8x16x4.val[chunk], vdupq_n_u8(1)), vdupq_n_u8(0xA9)));
        }
        nels = sz_strings_bits_neon_(nels_u8x16x4), separators = sz_strings_bits_neon_(separators_u8x16x4);
    }
    sz_u64_t const carried = previous[1] >> 63 | previous[2] >> 62 | previous[2] >> 63;
    *removed = singles | nels | nels << 1 | separators | separators << 1 | separators << 2 | carried;
    *ends = (singles & ~feeds_after_returns) | nels | separators;
    previous[0] = returns, previous[1] = nels, previous[2] = separators;
}

/** Appends the 64 bytes at @p source but its @p removed lanes at @p target, which has room for
 *  128, and sets the end offset, @p written plus the bytes kept before it, of each part a lane of
 *  @p ends ends. A removed lane ending no part directly follows one that does, so rewriting its
 *  slot is harmless. */
STRINGZILLA_INLINE void sz_strings_split_emit_neon_(sz_u8_t const *source, sz_u64_t removed, sz_u64_t ends,
                                                    sz_u8_t *target, sz_size_t written, sz_strings_span_t *output,
                                                    sz_size_t *ended) {
    vst1q_u8_x4(target, vld1q_u8_x4(source));
    sz_size_t skipped = 0;
    for (sz_u64_t rest = removed; rest; rest &= rest - 1, ++skipped) {
        sz_size_t const lane = (sz_size_t)sz_u64_ctz_neon_(rest);
        sz_u64_t const later = rest & (rest - 1);
        sz_size_t const next = later ? (sz_size_t)sz_u64_ctz_neon_(later) : 64;
        *ended += ends >> lane & 1;
        sz_strings_offset_set_(output, *ended, written + lane - skipped);
        // The bytes up to the next removed lane slide down, a longer run taking the block's rest.
        vst1q_u8(target + lane - skipped, vld1q_u8(source + lane + 1));
        if (next - lane <= 17) continue;
        vst1q_u8(target + lane - skipped + 16, vld1q_u8(source + lane + 17));
        vst1q_u8(target + lane - skipped + 32, vld1q_u8(source + lane + 33));
        vst1q_u8(target + lane - skipped + 48, vld1q_u8(source + lane + 49));
    }
}

/** Copies the @p length bytes of @p source to <tt>target + written</tt>: up to 16 in one load and store
 *  when 16 bytes stay @p readable and fit before @p end, and exactly otherwise. */
STRINGZILLA_INLINE void sz_strings_join_piece_neon_(sz_u8_t *target, sz_size_t written, sz_size_t end,
                                                    sz_u8_t const *source, sz_size_t length, sz_size_t readable) {
    if (length <= 16 && readable >= 16 && end - written >= 16) {
        vst1q_u8(target + written, vld1q_u8(source));
        return;
    }
    if (length < 16) {
        for (sz_size_t copied = 0; copied != length; ++copied) target[written + copied] = source[copied];
        return;
    }
    for (sz_size_t copied = 0; copied + 16 < length; copied += 16)
        vst1q_u8(target + written + copied, vld1q_u8(source + copied));
    vst1q_u8(target + written + length - 16, vld1q_u8(source + length - 16));
}

/** The NEON stages. */
STRINGZILLA_INLINE sz_strings_walks_t sz_strings_walks_neon_(void) {
    sz_strings_walks_t walks;
    walks.find_byteset = &sz_find_byteset_neon_;
    walks.lookup_bytes = &sz_strings_lookup_bytes_neon_;
    return walks;
}

/** The bytes escaping adds per byte of @p text_u8x16, from 0 to 5: JSON, HTML or URL escapes. */
STRINGZILLA_INLINE uint8x16_t sz_strings_escape_extras_neon_(sz_escape_kind_t kind, uint8x16_t text_u8x16) {
    // Every byte outside `A-Z a-z 0-9 - . _ ~`
    static sz_byteset_t const url_set = {{0xFC009FFFFFFFFFFFull, 0xB800000178000001ull, ~0ull, ~0ull}};
    // `\u00XX` for the controls, but `\b`, `\t`, `\n`, `\f` and `\r`
    static sz_u8_t const json_controls[32] = {5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 5, 1, 1, 5, 5,
                                              5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
    // `&quot;` at 0x22, `&amp;` and `&#39;` at 0x26 and 0x27, `&lt;` and `&gt;` at 0x3C and 0x3E
    static sz_u8_t const html_extras[64] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
                                            0, 0, 5, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, //
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0};
    if (kind == sz_escape_url_k)
        return vandq_u8(sz_strings_classify_neon_(text_u8x16, vld1q_u8_x2(url_set._u8s)), vdupq_n_u8(2));
    if (kind == sz_escape_html_k) return vqtbl4q_u8(vld1q_u8_x4(html_extras), text_u8x16);
    uint8x16_t const quoted_u8x16 = vorrq_u8(vceqq_u8(text_u8x16, vdupq_n_u8('"')),
                                             vceqq_u8(text_u8x16, vdupq_n_u8('\\')));
    return vorrq_u8(vqtbl2q_u8(vld1q_u8_x2(json_controls), text_u8x16), vandq_u8(quoted_u8x16, vdupq_n_u8(1)));
}

/** The sum of the first @p lanes of the 64 bytes @p extras_u8x16x4 holds. */
STRINGZILLA_INLINE sz_size_t sz_strings_escape_prefix_neon_(uint8x16x4_t extras_u8x16x4, sz_size_t lanes) {
    static sz_u8_t const iota[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint8x16_t const iota_u8x16 = vld1q_u8(iota), lanes_u8x16 = vdupq_n_u8((sz_u8_t)lanes);
    uint8x16_t sum_u8x16 = vdupq_n_u8(0);
    for (int chunk = 0; chunk != 4; ++chunk) {
        uint8x16_t const lane_u8x16 = vaddq_u8(iota_u8x16, vdupq_n_u8((sz_u8_t)(chunk * 16)));
        sum_u8x16 = vaddq_u8(sum_u8x16, vandq_u8(extras_u8x16x4.val[chunk], vcltq_u8(lane_u8x16, lanes_u8x16)));
    }
    return vaddlvq_u8(sum_u8x16);
}

/** Escapes the 64 bytes of @p text_u8x16x4 for @p kind into @p target, those of @p escaped as their
 *  sequences, storing up to 64 bytes past the output. Every byte is read from a local copy of the
 *  block and of @p next_u8x16x4 after it, so in place no store overwrites a byte not yet read. */
STRINGZILLA_INLINE void sz_strings_escape_emit_neon_(sz_escape_kind_t kind, uint8x16x4_t text_u8x16x4,
                                                     uint8x16x4_t next_u8x16x4, sz_u64_t escaped, sz_u8_t *target) {
    vst1q_u8_x4(target, text_u8x16x4);
    if (!escaped) return;
    sz_u8_t window[128];
    vst1q_u8_x4(window, text_u8x16x4), vst1q_u8_x4(window + 64, next_u8x16x4);
    sz_size_t extra = 0;
    for (sz_u64_t rest = escaped; rest; rest &= rest - 1) {
        sz_size_t const lane = (sz_size_t)sz_u64_ctz_neon_(rest);
        sz_u64_t const later = rest & (rest - 1);
        sz_size_t const next = later ? (sz_size_t)sz_u64_ctz_neon_(later) : 64;
        sz_u8_t *const output = target + lane + extra;
        sz_u64_t const sequence = sz_strings_escape_sequence_(kind, window[lane], &extra);
        memcpy(output, &sequence, sizeof(sequence));
        // The bytes up to the next escape follow, a longer run taking the block's rest.
        sz_u8_t *const after = target + lane + extra + 1;
        vst1q_u8(after, vld1q_u8(window + lane + 1));
        if (next - lane <= 17) continue;
        vst1q_u8(after + 16, vld1q_u8(window + lane + 17));
        vst1q_u8(after + 32, vld1q_u8(window + lane + 33));
        vst1q_u8(after + 48, vld1q_u8(window + lane + 49));
    }
}

/**
 *  @brief Escapes the @p count strings of a tape, read in its own order as one run of bytes, for a
 *      @p kind whose bytes escape alone: JSON, HTML or URL escapes.
 *
 *  Each block is read before the one behind it is written, and no store reaches 64 bytes past its
 *  block's output, so a tape escaped where it lies never overwrites a byte still to be read.
 */
STRINGZILLA_INLINE sz_status_t sz_strings_escape_run_neon_(sz_strings_view_t const *strings, sz_size_t count,
                                                           sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                           sz_strings_span_t *tape) {
    sz_size_t const start = sz_strings_view_offset_(strings, 0);
    sz_size_t const length = sz_strings_view_offset_(strings, count) - start;
    sz_cptr_t text = sz_strings_view_data_(strings) + start;
    // The last block and the one after it read a copy padded with a byte no kind escapes.
    sz_size_t const copied_from = length & ~(sz_size_t)63;
    sz_u8_t tail[128];
    memset(tail, 'a', sizeof(tail));
    if (length != copied_from) memcpy(tail, text + copied_from, length - copied_from);

    sz_size_t payload = length;
    for (sz_size_t block = 0; block < length; block += 64) {
        sz_u8_t const *const source = block < copied_from ? (sz_u8_t const *)text + block : tail;
        uint8x16x4_t const text_u8x16x4 = vld1q_u8_x4(source);
        uint8x16x4_t extras_u8x16x4;
        for (int chunk = 0; chunk != 4; ++chunk)
            extras_u8x16x4.val[chunk] = sz_strings_escape_extras_neon_(kind, text_u8x16x4.val[chunk]);
        payload += sz_strings_escape_prefix_neon_(extras_u8x16x4, 64);
    }

    sz_strings_span_t output;
    sz_status_t const status = sz_strings_escapes_in_place_(strings, tape, count, payload)
                                   ? sz_strings_escape_grow_(tape, payload, allocator, &output, &text)
                                   : sz_strings_acquire_(strings, tape, count, payload, allocator, &output,
                                                         STRINGZILLA_NULL);
    if (status != sz_success_k) return status;
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    sz_u8_t *const data = (sz_u8_t *)sz_strings_payload_(&output);
    sz_u8_t bounce[448];
    sz_size_t written = 0, row = 0;
    uint8x16x4_t text_u8x16x4 = vld1q_u8_x4(copied_from ? bytes : tail);
    for (sz_size_t block = 0; block < length; block += 64) {
        sz_size_t const after = block + 64;
        uint8x16x4_t const next_u8x16x4 = vld1q_u8_x4(after < copied_from ? bytes + after
                                                                          : tail + (after - copied_from));
        uint8x16x4_t extras_u8x16x4, escaped_u8x16x4;
        for (int chunk = 0; chunk != 4; ++chunk) {
            extras_u8x16x4.val[chunk] = sz_strings_escape_extras_neon_(kind, text_u8x16x4.val[chunk]);
            escaped_u8x16x4.val[chunk] = vtstq_u8(extras_u8x16x4.val[chunk], extras_u8x16x4.val[chunk]);
        }
        sz_size_t const produced = sz_min_of_two(length - block, (sz_size_t)64) +
                                   sz_strings_escape_prefix_neon_(extras_u8x16x4, 64);
        // Near the payload's end the block is escaped aside and copied out exactly.
        sz_bool_t const aside = (sz_bool_t)(written + sizeof(bounce) > payload);
        sz_strings_escape_emit_neon_(kind, text_u8x16x4, next_u8x16x4, sz_strings_bits_neon_(escaped_u8x16x4),
                                     aside ? bounce : data + written);
        if (aside) memcpy(data + written, bounce, produced);
        for (sz_size_t end; row != count && (end = sz_strings_view_offset_(strings, row + 1) - start) < after; ++row)
            sz_strings_offset_set_(&output, row + 1,
                                   written + end - block + sz_strings_escape_prefix_neon_(extras_u8x16x4, end - block));
        written += produced, text_u8x16x4 = next_u8x16x4;
    }
    for (; row != count; ++row) sz_strings_offset_set_(&output, row + 1, payload);
    sz_strings_offset_set_(&output, 0, 0);
    sz_strings_commit_(tape, &output, allocator, STRINGZILLA_NULL);
    return sz_success_k;
}

/** Escapes the @p count strings of a tape, read in its own order as one run of bytes, as
 *  hexadecimal digit pairs: every 16 bytes are read before their 32 are written, which in place
 *  end at the next unread byte at the latest. */
STRINGZILLA_INLINE sz_status_t sz_strings_escape_hex_neon_(sz_strings_view_t const *strings, sz_size_t count,
                                                           sz_allocator_t *allocator, sz_strings_span_t *tape) {
    static sz_u8_t const digits[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    sz_size_t const start = sz_strings_view_offset_(strings, 0);
    sz_size_t const length = sz_strings_view_offset_(strings, count) - start, payload = length * 2;
    sz_cptr_t text = sz_strings_view_data_(strings) + start;
    sz_strings_span_t output;
    sz_status_t const status = sz_strings_escapes_in_place_(strings, tape, count, payload)
                                   ? sz_strings_escape_grow_(tape, payload, allocator, &output, &text)
                                   : sz_strings_acquire_(strings, tape, count, payload, allocator, &output,
                                                         STRINGZILLA_NULL);
    if (status != sz_success_k) return status;
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    sz_u8_t *const data = (sz_u8_t *)sz_strings_payload_(&output);
    uint8x16_t const digits_u8x16 = vld1q_u8(digits);
    sz_size_t offset = 0;
    for (; offset + 16 <= length; offset += 16) {
        uint8x16_t const text_u8x16 = vld1q_u8(bytes + offset);
        uint8x16x2_t pairs_u8x16x2;
        pairs_u8x16x2.val[0] = vqtbl1q_u8(digits_u8x16, vshrq_n_u8(text_u8x16, 4));
        pairs_u8x16x2.val[1] = vqtbl1q_u8(digits_u8x16, vandq_u8(text_u8x16, vdupq_n_u8(15)));
        vst2q_u8(data + offset * 2, pairs_u8x16x2);
    }
    for (; offset != length; ++offset) {
        sz_u8_t const byte = bytes[offset];
        data[offset * 2] = digits[byte >> 4], data[offset * 2 + 1] = digits[byte & 15];
    }
    for (sz_size_t row = 0; row <= count; ++row)
        sz_strings_offset_set_(&output, row, (sz_strings_view_offset_(strings, row) - start) * 2);
    sz_strings_commit_(tape, &output, allocator, STRINGZILLA_NULL);
    return sz_success_k;
}

/** 0xFF in the lanes of @p digits_u8x16 that are hexadecimal digits of either case. */
STRINGZILLA_INLINE uint8x16_t sz_strings_hex_digits_neon_(uint8x16_t digits_u8x16) {
    uint8x16_t const decimal_u8x16 = vsubq_u8(digits_u8x16, vdupq_n_u8('0'));
    uint8x16_t const letter_u8x16 = vsubq_u8(vorrq_u8(digits_u8x16, vdupq_n_u8(0x20)), vdupq_n_u8('a'));
    return vorrq_u8(vcltq_u8(decimal_u8x16, vdupq_n_u8(10)), vcltq_u8(letter_u8x16, vdupq_n_u8(6)));
}

/** The <tt>\%</tt> lanes among the 64 bytes at @p source, as bits. */
STRINGZILLA_INLINE sz_u64_t sz_strings_percents_neon_(sz_u8_t const *source) {
    uint8x16x4_t percents_u8x16x4 = vld1q_u8_x4(source);
    for (int chunk = 0; chunk != 4; ++chunk)
        percents_u8x16x4.val[chunk] = vceqq_u8(percents_u8x16x4.val[chunk], vdupq_n_u8('%'));
    return sz_strings_bits_neon_(percents_u8x16x4);
}

/** Decodes the 64 bytes at @p source, whose <tt>\%</tt> lanes are @p percents and whose first @p skipped
 *  lanes are digits of an escape of the block before, into @p target, reading up to 130 bytes and
 *  storing up to 128. */
STRINGZILLA_INLINE void sz_strings_unescape_url_emit_neon_(sz_u8_t const *source, sz_u64_t percents, sz_size_t skipped,
                                                           sz_u8_t *target) {
    vst1q_u8_x4(target, vld1q_u8_x4(source + skipped));
    sz_size_t dropped = skipped;
    for (sz_u64_t rest = percents; rest; rest &= rest - 1, dropped += 2) {
        sz_size_t const lane = (sz_size_t)sz_u64_ctz_neon_(rest);
        sz_u64_t const later = rest & (rest - 1);
        sz_size_t const next = later ? (sz_size_t)sz_u64_ctz_neon_(later) : 64;
        sz_u8_t *const output = target + lane - dropped;
        *output = (sz_u8_t)(sz_strings_hex_digit_((char)source[lane + 1]) << 4 |
                            sz_strings_hex_digit_((char)source[lane + 2]));
        // The bytes up to the next escape follow, a longer run taking the block's rest.
        vst1q_u8(output + 1, vld1q_u8(source + lane + 3));
        if (next - lane <= 19) continue;
        vst1q_u8(output + 17, vld1q_u8(source + lane + 19));
        vst1q_u8(output + 33, vld1q_u8(source + lane + 35));
        vst1q_u8(output + 49, vld1q_u8(source + lane + 51));
    }
}

/**
 *  @brief Unescapes the @p count percent-encoded strings of a tape, read in its own order as one
 *      run of bytes, into fresh blocks.
 *  @return @c sz_invalid_utf8_k with nothing written when a <tt>\%</tt> lacks two hexadecimal digits after
 *      it in its string, for the driver to name that string.
 */
STRINGZILLA_INLINE sz_status_t sz_strings_unescape_url_run_neon_(sz_strings_view_t const *strings, sz_size_t count,
                                                                 sz_allocator_t *allocator, sz_strings_span_t *tape) {
    sz_size_t const start = sz_strings_view_offset_(strings, 0);
    sz_size_t const length = sz_strings_view_offset_(strings, count) - start;
    sz_u8_t const *const text = (sz_u8_t const *)sz_strings_view_data_(strings) + start;
    // Blocks within 130 bytes of the end read a zeroed copy, as decoding one reads 130 bytes.
    sz_size_t const copied_from = length < 130 ? 0 : ((length - 130) / 64 + 1) * 64;
    sz_u8_t tail[256] = {0};
    if (length != copied_from) memcpy(tail, text + copied_from, length - copied_from);

    sz_u64_t malformed = 0, owed = 0;
    sz_size_t percents_count = 0, row = 0;
    for (sz_size_t block = 0; block < length; block += 64) {
        sz_u8_t const *const source = block < copied_from ? text + block : tail + (block - copied_from);
        uint8x16x4_t digits_u8x16x4 = vld1q_u8_x4(source);
        for (int chunk = 0; chunk != 4; ++chunk)
            digits_u8x16x4.val[chunk] = sz_strings_hex_digits_neon_(digits_u8x16x4.val[chunk]);
        sz_u64_t const percents = sz_strings_percents_neon_(source);
        sz_u64_t const needed = percents << 1 | percents << 2 | owed;
        malformed |= needed & ~sz_strings_bits_neon_(digits_u8x16x4);
        // A string ending where a digit is due cuts its escape short.
        for (sz_size_t end; row != count && (end = sz_strings_view_offset_(strings, row + 1) - start) < block + 64;
             ++row)
            malformed |= needed >> (end - block) & 1;
        owed = percents >> 63 | percents >> 62, percents_count += (sz_size_t)sz_u64_popcount_neon_(percents);
    }
    if (malformed | owed) return sz_invalid_utf8_k;

    sz_size_t const payload = length - 2 * percents_count;
    sz_strings_span_t output;
    sz_status_t const status = sz_strings_acquire_(strings, tape, count, payload, allocator, &output, STRINGZILLA_NULL);
    if (status != sz_success_k) return status;
    sz_u8_t *const data = (sz_u8_t *)sz_strings_payload_(&output);
    sz_u8_t bounce[128];
    sz_size_t written = 0;
    owed = 0, percents_count = 0, row = 0;
    for (sz_size_t block = 0; block < length; block += 64) {
        sz_u8_t const *const source = block < copied_from ? text + block : tail + (block - copied_from);
        sz_u64_t const percents = sz_strings_percents_neon_(source);
        sz_u64_t const valid = sz_u64_mask_until_serial_(sz_min_of_two(length - block, (sz_size_t)64));
        sz_size_t const kept = (sz_size_t)sz_u64_popcount_neon_(~(percents << 1 | percents << 2 | owed) & valid);
        // Near the payload's end the block is decoded aside and copied out exactly.
        sz_bool_t const aside = (sz_bool_t)(written + sizeof(bounce) > payload);
        sz_strings_unescape_url_emit_neon_(source, percents, (sz_size_t)sz_u64_popcount_neon_(owed),
                                           aside ? bounce : data + written);
        if (aside && kept) memcpy(data + written, bounce, kept);
        for (sz_size_t end; row != count && (end = sz_strings_view_offset_(strings, row + 1) - start) < block + 64;
             ++row) {
            sz_u64_t const before = percents & sz_u64_mask_until_serial_(end - block);
            sz_strings_offset_set_(&output, row + 1,
                                   end - 2 * (percents_count + (sz_size_t)sz_u64_popcount_neon_(before)));
        }
        owed = percents >> 63 | percents >> 62, percents_count += (sz_size_t)sz_u64_popcount_neon_(percents);
        written += kept;
    }
    for (; row != count; ++row) sz_strings_offset_set_(&output, row + 1, payload);
    sz_strings_offset_set_(&output, 0, 0);
    sz_strings_commit_(tape, &output, allocator, STRINGZILLA_NULL);
    return sz_success_k;
}

/** Escapes strings for @p kind, a tape read in its own order as one run where each byte escapes
 *  alone, and anything else string by string. */
STRINGZILLA_INLINE sz_status_t sz_strings_escape_neon_(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_escape_kind_t kind,
                                                       sz_allocator_t *allocator, sz_strings_span_t *tape) {
    sz_bool_t const run = (sz_bool_t)(!order && strings->layout != sz_strings_accessor_k);
    if (run && kind == sz_escape_hex_k) return sz_strings_escape_hex_neon_(strings, count, allocator, tape);
    if (run && kind == sz_escape_json_k)
        return sz_strings_escape_run_neon_(strings, count, sz_escape_json_k, allocator, tape);
    if (run && kind == sz_escape_html_k)
        return sz_strings_escape_run_neon_(strings, count, sz_escape_html_k, allocator, tape);
    if (run && kind == sz_escape_url_k)
        return sz_strings_escape_run_neon_(strings, count, sz_escape_url_k, allocator, tape);
    sz_strings_walks_t const walks = sz_strings_walks_neon_();
    return sz_strings_escape_with_(strings, order, count, kind, allocator, tape, &walks);
}

/** Unescapes strings from @p kind, a tape of URLs read in its own order as one run, and anything
 *  else, or a run holding a malformed string, string by string. */
STRINGZILLA_INLINE sz_status_t sz_strings_unescape_neon_(sz_strings_view_t const *strings, sz_size_t const *order,
                                                         sz_size_t count, sz_escape_kind_t kind,
                                                         sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                         sz_size_t *malformed_index) {
    sz_bool_t const run = (sz_bool_t)(!order && strings->layout != sz_strings_accessor_k && kind == sz_escape_url_k);
    sz_status_t const status = run ? sz_strings_unescape_url_run_neon_(strings, count, allocator, tape)
                                   : sz_invalid_utf8_k;
    if (status != sz_invalid_utf8_k) return status;
    sz_strings_walks_t const walks = sz_strings_walks_neon_();
    return sz_strings_unescape_with_(strings, order, count, kind, allocator, tape, malformed_index, &walks);
}

#if STRINGZILLA_TARGET_NEON

STRINGZILLA_API sz_status_t sz_strings_to_tape_neon(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                    sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_tape_with_(strings, order, count, allocator, tape);
}

STRINGZILLA_API sz_status_t sz_strings_to_rows_neon(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_strings_rows_t const *rows, sz_size_t *longest,
                                                    sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_rows_with_(strings, order, count, rows, longest);
}

STRINGZILLA_API sz_status_t sz_strings_escape_neon(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                   sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_escape_neon_(strings, order, count, kind, allocator, tape);
}

STRINGZILLA_API sz_status_t sz_strings_unescape_neon(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_size_t *malformed_index,
                                                     sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_unescape_neon_(strings, order, count, kind, allocator, tape, malformed_index);
}

STRINGZILLA_API sz_status_t sz_strings_lookup_neon(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, char const lut[sz_at_least_(256)],
                                                   sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                   sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_neon_();
    return sz_strings_lookup_with_(strings, order, count, lut, allocator, tape, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_normalize_neon(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_normal_form_t form, sz_allocator_t *allocator,
                                                      sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_normalize_with_(strings, order, count, form, allocator, tape, &sz_utf8_norm_neon);
}

STRINGZILLA_API sz_status_t sz_strings_split_neon(sz_cptr_t text, sz_size_t length, sz_byteset_t const *delimiters,
                                                  sz_split_kind_t kind, sz_allocator_t *allocator,
                                                  sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    uint8x16x2_t const set_u8x16x2 = vld1q_u8_x2(delimiters->_u8s);
    sz_u64_t const runs = kind == sz_split_runs_k ? ~(sz_u64_t)0 : 0;
    // Blocks within 128 bytes of the end read a zeroed copy, as emitting one reads 64 bytes past.
    sz_size_t const copied_from = length < 128 ? 0 : ((length - 128) / 64 + 1) * 64;
    sz_u8_t tail[192] = {0};
    if (length != copied_from) memcpy(tail, bytes + copied_from, length - copied_from);

    sz_size_t removed_count = 0, ends_count = 0;
    sz_u64_t previous_removed = 1;
    for (sz_size_t block = 0; block < length; block += 64) {
        sz_u8_t const *const source = block < copied_from ? bytes + block : tail + (block - copied_from);
        sz_size_t const lanes = sz_min_of_two(length - block, (sz_size_t)64);
        sz_u64_t const removed = sz_strings_split_removed_neon_(source, set_u8x16x2) & sz_u64_mask_until_serial_(lanes);
        removed_count += (sz_size_t)sz_u64_popcount_neon_(removed);
        ends_count += (sz_size_t)sz_u64_popcount_neon_(removed & ~((removed << 1 | previous_removed) & runs));
        previous_removed = removed >> (lanes - 1) & 1;
    }
    sz_size_t const payload = length - removed_count;
    sz_size_t const parts = runs ? ends_count + (previous_removed ^ 1) : removed_count + 1;

    sz_strings_span_t output;
    sz_status_t const status = sz_strings_split_acquire_(text, length, tape, parts, payload, allocator, &output);
    if (status != sz_success_k) return status;
    sz_u8_t *const data = (sz_u8_t *)sz_strings_payload_(&output);
    sz_u8_t bounce[128];
    sz_size_t written = 0, ended = 0;
    previous_removed = 1;
    for (sz_size_t block = 0; block < length; block += 64) {
        sz_u8_t const *const source = block < copied_from ? bytes + block : tail + (block - copied_from);
        sz_size_t const lanes = sz_min_of_two(length - block, (sz_size_t)64);
        sz_u64_t const valid = sz_u64_mask_until_serial_(lanes);
        sz_u64_t const removed = sz_strings_split_removed_neon_(source, set_u8x16x2) & valid;
        sz_u64_t const ends = removed & ~((removed << 1 | previous_removed) & runs);
        sz_size_t const kept = (sz_size_t)sz_u64_popcount_neon_(~removed & valid);
        // Near the payload's end the block is emitted aside and copied out exactly.
        sz_bool_t const aside = (sz_bool_t)(written + 128 > payload);
        sz_strings_split_emit_neon_(source, removed, ends, aside ? bounce : data + written, written, &output, &ended);
        if (aside && kept) memcpy(data + written, bounce, kept);
        written += kept, previous_removed = removed >> 63;
    }
    sz_strings_split_commit_(tape, &output, payload, allocator);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_strings_split_lines_neon(sz_cptr_t text, sz_size_t length, sz_allocator_t *allocator,
                                                        sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    sz_size_t const copied_from = length < 128 ? 0 : ((length - 128) / 64 + 1) * 64;
    sz_u8_t tail[192] = {0};
    if (length != copied_from) memcpy(tail, bytes + copied_from, length - copied_from);

    sz_size_t removed_count = 0, ends_count = 0;
    sz_u64_t previous[3] = {0, 0, 0}, last_removed = 1, removed, ends;
    for (sz_size_t block = 0; block < length; block += 64) {
        sz_u8_t const *const source = block < copied_from ? bytes + block : tail + (block - copied_from);
        sz_size_t const lanes = sz_min_of_two(length - block, (sz_size_t)64);
        sz_strings_lines_classify_neon_(source, previous, &removed, &ends);
        removed &= sz_u64_mask_until_serial_(lanes), ends &= sz_u64_mask_until_serial_(lanes);
        removed_count += (sz_size_t)sz_u64_popcount_neon_(removed),
            ends_count += (sz_size_t)sz_u64_popcount_neon_(ends);
        last_removed = removed >> (lanes - 1) & 1;
    }
    sz_size_t const payload = length - removed_count, parts = ends_count + (last_removed ^ 1);

    sz_strings_span_t output;
    sz_status_t const status = sz_strings_split_acquire_(text, length, tape, parts, payload, allocator, &output);
    if (status != sz_success_k) return status;
    sz_u8_t *const data = (sz_u8_t *)sz_strings_payload_(&output);
    sz_u8_t bounce[128];
    sz_size_t written = 0, ended = 0;
    previous[0] = previous[1] = previous[2] = 0;
    for (sz_size_t block = 0; block < length; block += 64) {
        sz_u8_t const *const source = block < copied_from ? bytes + block : tail + (block - copied_from);
        sz_u64_t const valid = sz_u64_mask_until_serial_(sz_min_of_two(length - block, (sz_size_t)64));
        sz_strings_lines_classify_neon_(source, previous, &removed, &ends);
        removed &= valid, ends &= valid;
        sz_size_t const kept = (sz_size_t)sz_u64_popcount_neon_(~removed & valid);
        sz_bool_t const aside = (sz_bool_t)(written + 128 > payload);
        sz_strings_split_emit_neon_(source, removed, ends, aside ? bounce : data + written, written, &output, &ended);
        if (aside && kept) memcpy(data + written, bounce, kept);
        written += kept;
    }
    sz_strings_split_commit_(tape, &output, payload, allocator);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_strings_join_neon(sz_strings_view_t const *columns, sz_size_t columns_count,
                                                 sz_size_t rows, sz_string_view_t const *frames,
                                                 sz_string_view_t between, sz_ptr_t output, sz_size_t *length,
                                                 sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    if (!output) return sz_strings_join_measure_(columns, columns_count, rows, frames, between, length);
    // Accessors, frames longer than a register and more columns than slots join cell by cell.
    sz_size_t by_cell = columns_count > 15 || between.length > 16;
    for (sz_size_t column = 0; column <= columns_count && !by_cell; ++column)
        by_cell |= frames[column].length > 16 ||
                   (column != columns_count && columns[column].layout == sz_strings_accessor_k);
    if (by_cell) return sz_strings_join_rows_serial_(columns, columns_count, rows, frames, between, output);

    // Frames and the separator come from zero-padded slots, whose padding later pieces overwrite.
    sz_u8_t slots[17][16];
    sz_u8_t const *payloads[15];
    sz_size_t payload_ends[15];
    memset(slots, 0, sizeof(slots));
    if (between.length) memcpy(slots[16], between.start, between.length);
    for (sz_size_t column = 0; column <= columns_count; ++column)
        if (frames[column].length) memcpy(slots[column], frames[column].start, frames[column].length);
    for (sz_size_t column = 0; column != columns_count; ++column)
        payloads[column] = (sz_u8_t const *)sz_strings_view_data_(columns + column),
        payload_ends[column] = sz_strings_view_offset_(columns + column, columns[column].count);

    sz_u8_t *const target = (sz_u8_t *)output;
    sz_size_t const end = *length;
    sz_size_t written = 0;
    for (sz_size_t row = 0; row != rows; ++row) {
        for (sz_size_t column = 0; column != columns_count; ++column) {
            sz_strings_join_piece_neon_(target, written, end, slots[column], frames[column].length, 16);
            written += frames[column].length;
            sz_size_t const start = sz_strings_view_offset_(columns + column, row);
            sz_size_t const cell = sz_strings_view_offset_(columns + column, row + 1) - start;
            sz_strings_join_piece_neon_(target, written, end, payloads[column] + start, cell,
                                        payload_ends[column] - start);
            written += cell;
        }
        sz_size_t const separator = row + 1 != rows ? between.length : 0;
        sz_strings_join_piece_neon_(target, written, end, slots[columns_count], frames[columns_count].length, 16);
        written += frames[columns_count].length;
        sz_strings_join_piece_neon_(target, written, end, slots[16], separator, 16);
        written += separator;
    }
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_NEON

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

#endif // STRINGZILLA_STRINGS_NEON_H_
