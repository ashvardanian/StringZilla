/**
 *  @brief RISC-V Vector backend for UTF-8 codepoint mechanics.
 *  @file include/stringzilla/utf8_codepoints/rvv.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_CODEPOINTS_RVV_H_
#define STRINGZILLA_UTF8_CODEPOINTS_RVV_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes.h" // `sz_rune_parse_unchecked`
#include "stringzilla/utf8_codepoints/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_RVV
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

/**
 *  @brief Decode UTF-8 to UTF-32 codepoints, byte-for-byte equivalent to `sz_utf8_unpack_chunk_serial`.
 *
 *  Vector-classifies the maximal ASCII prefix of each strip (`vmsgtu` high bit, `vfirst`) and widens it
 *  straight into the rune output; any non-ASCII byte takes a single serial `sz_rune_parse` step.
 */
/** @brief Widen `count` ASCII bytes (u8 -> u16 -> u32) and store them as runes. */
SZ_INTERNAL void sz_utf8_rvv_store_ascii_run_(sz_rune_t *runes_out, sz_u8_t const *src, sz_size_t count) {
    sz_size_t done = 0;
    while (done < count) {
        sz_size_t widened_vector_length = __riscv_vsetvl_e8m2(count - done);
        vuint8m2_t ascii_u8m2 = __riscv_vle8_v_u8m2(src + done, widened_vector_length);
        vuint16m4_t ascii_u16m4 = __riscv_vwcvtu_x_x_v_u16m4(ascii_u8m2, widened_vector_length);
        vuint32m8_t ascii_u32m8 = __riscv_vwcvtu_x_x_v_u32m8(ascii_u16m4, widened_vector_length);
        __riscv_vse32_v_u32m8(runes_out + done, ascii_u32m8, widened_vector_length);
        done += widened_vector_length;
    }
}

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_rvv(   //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {

    sz_u8_t const *text_cursor = (sz_u8_t const *)text;
    sz_u8_t const *text_end = text_cursor + length;
    sz_size_t runes_written = 0;

    while (text_cursor < text_end && runes_written < runes_capacity) {
        // Consume a maximal ASCII run, bounded by remaining input and remaining capacity.
        sz_size_t bytes_left = (sz_size_t)(text_end - text_cursor);
        sz_size_t cap_left = runes_capacity - runes_written;
        sz_size_t want = bytes_left < cap_left ? bytes_left : cap_left;

        sz_size_t ascii_run = 0;
        while (ascii_run < want) {
            sz_size_t vector_length = __riscv_vsetvl_e8m8(want - ascii_run);
            vuint8m8_t bytes_u8m8 = __riscv_vle8_v_u8m8(text_cursor + ascii_run, vector_length);
            vbool1_t non_ascii = __riscv_vmsgtu_vx_u8m8_b1(bytes_u8m8, 0x7F, vector_length);
            long first_match_index = __riscv_vfirst_m_b1(non_ascii, vector_length);
            sz_size_t take = first_match_index < 0 ? vector_length : (sz_size_t)first_match_index;
            if (take) sz_utf8_rvv_store_ascii_run_(runes + runes_written + ascii_run, text_cursor + ascii_run, take);
            ascii_run += take;
            if (first_match_index >= 0) break; // Hit a non-ASCII byte.
        }
        runes_written += ascii_run;
        text_cursor += ascii_run;
        if (runes_written >= runes_capacity || text_cursor >= text_end) break;

        // Now `*text_cursor` is a multi-byte lead (or a stray continuation). Decode exactly like serial.
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_parse_unchecked((sz_cptr_t)text_cursor, &rune);
        if (text_cursor + rune_length > text_end) break; // Incomplete sequence at buffer boundary.
        runes[runes_written++] = rune;
        text_cursor += rune_length;
    }

    *runes_unpacked = runes_written;
    return (sz_cptr_t)text_cursor;
}

/** @brief Count UTF-8 codepoints: `vcpop` the leading (non-continuation) bytes per strip. */
SZ_PUBLIC sz_size_t sz_utf8_count_rvv(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0;
    while (length) {
        sz_size_t vector_length = __riscv_vsetvl_e8m8(length);
        vuint8m8_t bytes_u8m8 = __riscv_vle8_v_u8m8(text_u8, vector_length);
        // Leading byte iff `(c & 0xC0) != 0x80`.
        vuint8m8_t masked = __riscv_vand_vx_u8m8(bytes_u8m8, 0xC0, vector_length);
        vbool1_t lead_mask = __riscv_vmsne_vx_u8m8_b1(masked, 0x80, vector_length);
        count += (sz_size_t)__riscv_vcpop_m_b1(lead_mask, vector_length);
        text_u8 += vector_length, length -= vector_length;
    }
    return count;
}

/**
 *  @brief Locate the start of the n-th codepoint, byte-for-byte equivalent to the serial baseline.
 *
 *  Skips whole strips by `vcpop` of leading bytes, then locates the wanted lead in-register with `viota`
 *  (the lane that is a lead and whose prefix count equals the target). Runs at `e8m4` so the `vbool2`
 *  lead mask pairs with a `u16m8` iota whose lane count never overflows the prefix counts.
 */
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_rvv(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t seen = 0;
    while (length) {
        sz_size_t vector_length = __riscv_vsetvl_e8m4(length);
        vuint8m4_t bytes_u8m4 = __riscv_vle8_v_u8m4(text_u8, vector_length);
        vbool2_t lead_mask = __riscv_vmsne_vx_u8m4_b2(__riscv_vand_vx_u8m4(bytes_u8m4, 0xC0, vector_length), 0x80,
                                                      vector_length);
        sz_size_t strip_leads = (sz_size_t)__riscv_vcpop_m_b2(lead_mask, vector_length);
        if (seen + strip_leads > n) {
            sz_u16_t target = (sz_u16_t)(n - seen); // 0-based index of the wanted lead byte within this strip
            vuint16m8_t prefix = __riscv_viota_m_u16m8(lead_mask, vector_length);
            vbool2_t hit = __riscv_vmand_mm_b2(__riscv_vmseq_vx_u16m8_b2(prefix, target, vector_length), lead_mask,
                                               vector_length);
            return (sz_cptr_t)(text_u8 + __riscv_vfirst_m_b2(hit, vector_length));
        }
        seen += strip_leads;
        text_u8 += vector_length, length -= vector_length;
    }
    return SZ_NULL_CHAR;
}

/*  Multistep newline / whitespace iteration (RVV 1.0).
 *
 *  Fully masked / streaming: each iteration loads `vl = vsetvl_e8m4(length - position)` lanes, so the final
 *  iteration is the masked tail (carry bytes past `length` read as 0, so a truncated delimiter at EOF never
 *  matches) — no serial tail call. A small per-set classifier builds the per-lane byte-length vector; one
 *  shared driver (`sz_utf8_iterate_multistep_rvv_`) handles the window/carry/trusted-lane/peel scaffolding. */

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_RVV

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_CODEPOINTS_RVV_H_
