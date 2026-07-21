/**
 *  @brief SVE backend for the single-pass Unicode normalizer (NFD / NFC / NFKD / NFKC).
 *  @file include/stringzilla/utf8_norm/sve.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_norm.h
 *
 *  This backend overrides exactly one point of the shared engine: the scan primitive
 *  `sz_utf8_norm_classify_sve_`, which locates the first non-inert byte for a form. The two public
 *  entry points (`sz_utf8_norm_sve` / `sz_utf8_find_denormalized_sve`) reuse the force-inlined engines
 *  from `serial.h`, passing this scanner as the constant function address that devirtualizes the call.
 *
 *  The scanner is the vector-length-agnostic sibling of the NEON kernel: a `svtbl_u8` lead-classify
 *  behind a per-vector ASCII/inert gate, with a cold per-codepoint verify. Because SVE's `svtbl_u8`
 *  indexes only the bottom slice of a scalable vector, the 64-entry `sz_utf8_norm_lead_lut_` is
 *  resolved with the proven 4-way `svsel_u8` range-blend from `memory/sve.h`: four sub-tables for
 *  index ranges 0-15 / 16-31 / 32-47 / 48-63, blended by `svcmplt` / `svcmpge` predicates. For a
 *  64-byte (or wider) vector the upper predicates fold away to a single `svtbl_u8`.
 */
#ifndef STRINGZILLA_UTF8_NORM_SVE_H_
#define STRINGZILLA_UTF8_NORM_SVE_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_norm/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve")
#endif

/**
 *  @brief Scan primitive (SVE): first byte that begins a non-inert codepoint for @p form, else NULL.
 *
 *  Matches `sz_utf8_norm_classify_serial_` semantics, computed from the unified props trie. The hot loop
 *  is vector-length-agnostic: a per-vector ASCII gate plus a `svtbl_u8` lead-classify (resolved with the
 *  4-way range-blend so the 64-entry LUT spans any VL); the cold per-codepoint verify carries the
 *  combining class across vectors and reports order or quick-check violations exactly.
 */
SZ_HELPER_NOINLINE sz_cptr_t sz_utf8_norm_classify_sve_(sz_cptr_t text, sz_size_t length, sz_normal_form_t form) {
    sz_u8_t const *const text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *const end = text_u8 + length;
    sz_u8_t const form_flag = sz_utf8_norm_form_flag_(form);
    sz_size_t const step = svcntb();

    // Load the 64-byte LUT once into four sub-tables, each indexed by the bottom 4 bits of the index.
    // Each load is bounded to its 16-lane row so wide vectors (VL >= 256) never over-read the 64-byte table.
    svbool_t const lut_row_b8x = svwhilelt_b8((sz_u64_t)0, (sz_u64_t)16);
    svuint8_t const lut_0_to_15_u8x = svld1_u8(lut_row_b8x, sz_utf8_norm_lead_lut_ + 0);
    svuint8_t const lut_16_to_31_u8x = svld1_u8(lut_row_b8x, sz_utf8_norm_lead_lut_ + 16);
    svuint8_t const lut_32_to_47_u8x = svld1_u8(lut_row_b8x, sz_utf8_norm_lead_lut_ + 32);
    svuint8_t const lut_48_to_63_u8x = svld1_u8(lut_row_b8x, sz_utf8_norm_lead_lut_ + 48);

    sz_u8_t previous_canonical_combining_class = 0;
    for (sz_size_t offset = 0; offset < length; offset += step) {
        svbool_t const active_b8x = svwhilelt_b8((sz_u64_t)offset, (sz_u64_t)length);
        svuint8_t const bytes_u8x = svld1_u8(active_b8x, text_u8 + offset);

        // ASCII gate: a vector with no high-bit byte is wholly inert - reset the carry and continue.
        svbool_t const non_ascii_b8x = svcmpge_n_u8(active_b8x, bytes_u8x, 0x80);
        if (!svptest_any(active_b8x, non_ascii_b8x)) {
            previous_canonical_combining_class = 0;
            continue;
        }

        // Lead bytes only (non-ASCII and not a 10xxxxxx continuation), classified via the 64-entry LUT.
        svbool_t const continuation_b8x = svcmplt_n_u8(active_b8x, svsub_n_u8_x(active_b8x, bytes_u8x, 0x80), 0x40);
        svbool_t const is_lead_b8x = svbic_b_z(active_b8x, non_ascii_b8x, continuation_b8x);

        // 64-entry LUT via a 4-way range-blend over the bottom 6 bits of each byte. Each sub-table holds
        // 16 entries at lanes [0,16), so every quadrant is indexed by `index` minus its base - `svtbl_u8`
        // yields 0 for out-of-range lanes, which `svsel_u8` then discards via the range predicate.
        svuint8_t const index_u8x = svand_n_u8_x(active_b8x, bytes_u8x, 0x3F);
        svuint8_t const index_low_nibble_u8x = svand_n_u8_x(active_b8x, index_u8x, 0x0F);
        svbool_t const range_0_to_15_b8x = svcmplt_n_u8(active_b8x, index_u8x, 16);
        svbool_t const range_16_to_31_b8x = svand_b_z(active_b8x, svcmpge_n_u8(active_b8x, index_u8x, 16),
                                                      svcmplt_n_u8(active_b8x, index_u8x, 32));
        svbool_t const range_32_to_47_b8x = svand_b_z(active_b8x, svcmpge_n_u8(active_b8x, index_u8x, 32),
                                                      svcmplt_n_u8(active_b8x, index_u8x, 48));
        svbool_t const range_48_to_63_b8x = svcmpge_n_u8(active_b8x, index_u8x, 48);
        svuint8_t families_u8x = svsel_u8(range_0_to_15_b8x, svtbl_u8(lut_0_to_15_u8x, index_low_nibble_u8x),
                                          svdup_n_u8(0));
        families_u8x = svsel_u8(range_16_to_31_b8x, svtbl_u8(lut_16_to_31_u8x, index_low_nibble_u8x), families_u8x);
        families_u8x = svsel_u8(range_32_to_47_b8x, svtbl_u8(lut_32_to_47_u8x, index_low_nibble_u8x), families_u8x);
        families_u8x = svsel_u8(range_48_to_63_b8x, svtbl_u8(lut_48_to_63_u8x, index_low_nibble_u8x), families_u8x);

        // Flagged leads: a lead byte whose family carries the form's quick-check bit.
        svbool_t const has_flag_b8x = svcmpne_n_u8(active_b8x, svand_n_u8_x(active_b8x, families_u8x, form_flag), 0);
        svbool_t const flagged_b8x = svand_b_z(active_b8x, is_lead_b8x, has_flag_b8x);
        if (!svptest_any(active_b8x, flagged_b8x)) {
            // No flagged lead in this vector: inert for the form. Reset the carry and realign onto a
            // codepoint boundary - a vector step can land mid-sequence, and the straddling codepoint's
            // lead was already classified inert in the vector we are leaving.
            previous_canonical_combining_class = 0;
            sz_u8_t const *boundary = text_u8 + offset + step;
            if (boundary > end) boundary = end;
            while (boundary < end && (*boundary & 0xC0) == 0x80) ++boundary;
            offset = (sz_size_t)(boundary - text_u8) - step; // loop adds `step` back
            continue;
        }

        // The vector has a possibly-non-inert lead; verify exactly, codepoint by codepoint.
        sz_u8_t const *position = text_u8 + offset;
        sz_u8_t const *block_end = text_u8 + offset + step;
        if (block_end > end) block_end = end;
        sz_cptr_t violation = sz_utf8_norm_verify_block_(&position, block_end, end, form_flag,
                                                         &previous_canonical_combining_class);
        if (violation) return violation;
        // `verify_block` advances `position` to `block_end`; resume the loop from there.
        offset = (sz_size_t)(position - text_u8) - step; // loop adds `step` back
    }
    return SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_size_t sz_utf8_norm_sve(sz_cptr_t source, sz_size_t length, sz_normal_form_t form,
                                           sz_ptr_t destination) {
    return sz_utf8_norm_engine_(source, length, form, destination, &sz_utf8_norm_classify_sve_);
}

SZ_API_COMPTIME sz_cptr_t sz_utf8_find_denormalized_sve(sz_cptr_t source, sz_size_t length, sz_normal_form_t form) {
    return sz_utf8_find_denormalized_engine_(source, length, form, &sz_utf8_norm_classify_sve_);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_NORM_SVE_H_
