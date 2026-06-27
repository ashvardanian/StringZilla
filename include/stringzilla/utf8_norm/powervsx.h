/**
 *  @brief IBM Power VSX (128-bit) backend for the single-pass Unicode normalizer (NFD / NFC / NFKD / NFKC).
 *  @file include/stringzilla/utf8_norm/powervsx.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_norm.h
 *
 *  This backend overrides exactly one point of the shared engine: the scan primitive
 *  `sz_utf8_norm_classify_powervsx_`, which locates the first non-inert byte for a form. The two public
 *  entry points (`sz_utf8_norm_powervsx` / `sz_utf8_find_denormalized_powervsx`) reuse the force-inlined
 *  engines from `serial.h`, passing this scanner as the constant function address that devirtualizes the call.
 *
 *  The scanner mirrors the NEON and Skylake scanners over a 16-byte window: an all-ASCII gate behind one
 *  horizontal reduction, a lead-byte classify over the shared `sz_utf8_norm_lead_lut_`, then the shared cold
 *  per-codepoint verify (`sz_utf8_norm_verify_block_`) on any block that survives the gate. VSX has no
 *  256-entry shuffle, so the 64-entry lookup is covered by two `vec_perm` selections (lut[0..31] and
 *  lut[32..63]) merged with `vec_sel` on the index's bit 5. The LUT bytes load naturally; ppc64le is
 *  little-endian and `vec_xl` handles the load, so no endian fixups are needed for the table or the window.
 */
#ifndef STRINGZILLA_UTF8_NORM_POWERVSX_H_
#define STRINGZILLA_UTF8_NORM_POWERVSX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_norm/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

/**
 *  @brief Scan primitive (Power VSX): first byte that begins a non-inert codepoint for @p form, else NULL.
 *
 *  Matches `sz_utf8_norm_classify_serial_` semantics, computed from the unified props trie. The hot loop
 *  uses a 16-byte all-ASCII gate plus a `vec_perm` lead-classify; the cold per-codepoint verify carries the
 *  combining class across chunks and reports order or quick-check violations exactly.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_norm_classify_powervsx_(sz_cptr_t text, sz_size_t length, sz_normal_form_t form) {
    sz_u8_t const *position = (sz_u8_t const *)text;
    sz_u8_t const *const end = position + length;
    sz_u8_t const form_flag = sz_utf8_norm_form_flag_(form);
    sz_u8_t previous_canonical_combining_class = 0;

    __vector unsigned char const table_low_vec = vec_xl(0, sz_utf8_norm_lead_lut_ + 0);   // entries [0,31]
    __vector unsigned char const table_high_vec = vec_xl(0, sz_utf8_norm_lead_lut_ + 32); // entries [32,63]
    __vector unsigned char const high_bit_vec = vec_splats((unsigned char)0x80);
    __vector unsigned char const continuation_mask_vec = vec_splats((unsigned char)0xC0);
    __vector unsigned char const continuation_pattern_vec = vec_splats((unsigned char)0x80);
    __vector unsigned char const low_six_bits_vec = vec_splats((unsigned char)0x3F);
    __vector unsigned char const quadrant_select_vec = vec_splats((unsigned char)32);
    __vector unsigned char const form_flag_vec = vec_splats((unsigned char)form_flag);
    __vector unsigned char const zero_vec = vec_splats((unsigned char)0);

    while (position + 16 <= end) {
        __vector unsigned char bytes_vec = vec_xl(0, position);
        // All-ASCII gate: one horizontal reduction per 16B for the (overwhelmingly common) inert case.
        if (!vec_any_ge(bytes_vec, high_bit_vec)) {
            position += 16, previous_canonical_combining_class = 0;
            continue;
        }
        // Lead bytes only (non-ASCII and not a 10xxxxxx continuation), classified via the 64-entry LUT.
        __vector unsigned char non_ascii_vec = (__vector unsigned char)vec_cmpge(bytes_vec, high_bit_vec);
        __vector unsigned char continuation_vec = (__vector unsigned char)vec_cmpeq(
            vec_and(bytes_vec, continuation_mask_vec), continuation_pattern_vec);
        __vector unsigned char is_lead_vec = vec_andc(non_ascii_vec, continuation_vec);
        // 64-entry lookup: `vec_perm` selects from a 32-byte concatenation by the low 5 bits of each index,
        // so cover [0,31] and [32,63] with two `vec_perm` and merge on the index's bit 5 (>= 32) via `vec_sel`.
        __vector unsigned char index_vec = vec_and(bytes_vec, low_six_bits_vec);
        __vector unsigned char families_low_vec = vec_perm(table_low_vec, table_low_vec, index_vec);
        __vector unsigned char families_high_vec = vec_perm(table_high_vec, table_high_vec, index_vec);
        __vector unsigned char select_high_vec = (__vector unsigned char)vec_cmpge(index_vec, quadrant_select_vec);
        __vector unsigned char families_vec = vec_sel(families_low_vec, families_high_vec, select_high_vec);
        __vector unsigned char flagged_vec = vec_and(
            is_lead_vec, (__vector unsigned char)vec_cmpgt(vec_and(families_vec, form_flag_vec), zero_vec));
        if (!vec_any_ne(flagged_vec, zero_vec)) { // 16 bytes inert for the form
            position += 16, previous_canonical_combining_class = 0;
            // Realign onto a codepoint boundary: a 16-byte step can land mid-sequence, and the straddling
            // codepoint's lead was already classified inert in the chunk we are leaving.
            while (position < end && (*position & 0xC0) == 0x80) ++position;
            continue;
        }
        // The chunk has a possibly-non-inert lead; verify exactly, codepoint by codepoint.
        sz_cptr_t violation = sz_utf8_norm_verify_block_(&position, position + 16, end, form_flag,
                                                         &previous_canonical_combining_class);
        if (violation) return violation;
    }

    // Tail (< 16 bytes): the shared scalar verify carries the combining class across the final boundary.
    return sz_utf8_norm_verify_block_(&position, end, end, form_flag, &previous_canonical_combining_class);
}

SZ_PUBLIC sz_size_t sz_utf8_norm_powervsx(sz_cptr_t source, sz_size_t length, sz_normal_form_t form,
                                          sz_ptr_t destination) {
    return sz_utf8_norm_engine_(source, length, form, destination, &sz_utf8_norm_classify_powervsx_);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_denormalized_powervsx(sz_cptr_t source, sz_size_t length, sz_normal_form_t form) {
    return sz_utf8_find_denormalized_engine_(source, length, form, &sz_utf8_norm_classify_powervsx_);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_POWERVSX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_NORM_POWERVSX_H_
