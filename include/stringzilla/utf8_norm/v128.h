/**
 *  @brief WebAssembly SIMD128 backend for the single-pass Unicode normalizer (NFD / NFC / NFKD / NFKC).
 *  @file include/stringzilla/utf8_norm/v128.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_norm.h
 *
 *  This backend overrides exactly one point of the shared engine: the scan primitive
 *  `sz_utf8_norm_classify_v128_`, which locates the first non-inert byte for a form. The two public
 *  entry points (`sz_utf8_norm_v128` / `sz_utf8_find_denormalized_v128`) reuse the force-inlined engines
 *  from `serial.h`, passing this scanner as the constant function address that devirtualizes the call.
 *
 *  WASM SIMD128 is 16 bytes wide like NEON, but it has only a 16-entry `wasm_i8x16_swizzle`, so the
 *  64-entry `sz_utf8_norm_lead_lut_` is split into 4x16 sub-tables selected by the index's high two
 *  bits - the `sz_utf8_gather64_v128_` helper that the case-folding backend already uses. The hot loop
 *  runs an all-ASCII gate plus this lead-classify over one 16-byte window; any window that survives the
 *  gate is resolved by the shared cold per-codepoint verify (`sz_utf8_norm_verify_block_`).
 */
#ifndef STRINGZILLA_UTF8_NORM_V128_H_
#define STRINGZILLA_UTF8_NORM_V128_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_uncased_fold/v128.h" // `sz_utf8_gather64_v128_`
#include "stringzilla/utf8_norm/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

#pragma region simd128

/**
 *  @brief Scan primitive (SIMD128): first byte that begins a non-inert codepoint for @p form, else NULL.
 *
 *  Matches `sz_utf8_norm_classify_serial_` semantics, computed from the unified props trie. The hot loop
 *  uses a 16-byte all-ASCII gate plus a 4x16-swizzle lead-classify; the cold per-codepoint verify
 *  carries the canonical combining class across windows and reports order or quick-check violations exactly.
 */
SZ_HELPER_NOINLINE sz_cptr_t sz_utf8_norm_classify_v128_(sz_cptr_t text, sz_size_t length, sz_normal_form_t form) {
    sz_u8_t const *position = (sz_u8_t const *)text;
    sz_u8_t const *const end = position + length;
    sz_u8_t const form_flag = sz_utf8_norm_form_flag_(form);
    sz_u8_t previous_canonical_combining_class = 0;

    v128_t const lut0 = wasm_v128_load(sz_utf8_norm_lead_lut_ + 0);
    v128_t const lut1 = wasm_v128_load(sz_utf8_norm_lead_lut_ + 16);
    v128_t const lut2 = wasm_v128_load(sz_utf8_norm_lead_lut_ + 32);
    v128_t const lut3 = wasm_v128_load(sz_utf8_norm_lead_lut_ + 48);
    v128_t const high_bit = wasm_u8x16_splat(0x80);
    v128_t const continuation_span = wasm_u8x16_splat(0x40);
    v128_t const low6_mask = wasm_u8x16_splat(0x3F);
    v128_t const flag_vector = wasm_u8x16_splat(form_flag);
    v128_t const zero = wasm_u8x16_splat(0);

    while (position + 16 <= end) {
        v128_t bytes = wasm_v128_load(position);

        // 16-byte all-ASCII gate: an inert window of single-byte runes skips with zero LUT work.
        v128_t non_ascii = wasm_u8x16_ge(bytes, high_bit);
        if (!wasm_v128_any_true(non_ascii)) {
            position += 16, previous_canonical_combining_class = 0;
            continue;
        }

        // Lead bytes only (non-ASCII and not a 10xxxxxx continuation), classified via the 64-entry LUT.
        v128_t continuation = wasm_u8x16_lt(wasm_i8x16_sub(bytes, high_bit), continuation_span);
        v128_t is_lead = wasm_v128_andnot(non_ascii, continuation);
        v128_t index = wasm_v128_and(bytes, low6_mask);
        v128_t families = sz_utf8_gather64_v128_(lut0, lut1, lut2, lut3, index);
        v128_t flagged = wasm_v128_and(is_lead, wasm_i8x16_ne(wasm_v128_and(families, flag_vector), zero));

        if (!wasm_v128_any_true(flagged)) { // no flagged lead: inert for the form
            // After the skip, realign to a codepoint boundary - a 16-byte step can land mid-sequence, and
            // the straddling codepoint's lead was already classified inert in the window we are leaving.
            position += 16, previous_canonical_combining_class = 0;
            while (position < end && (*position & 0xC0) == 0x80) ++position;
            continue;
        }

        // The window has a possibly-non-inert lead; verify exactly, codepoint by codepoint.
        sz_cptr_t violation = sz_utf8_norm_verify_block_(&position, position + 16, end, form_flag,
                                                         &previous_canonical_combining_class);
        if (violation) return violation;
    }

    // Tail (< 16 bytes), carrying the canonical combining class across the last window boundary.
    return sz_utf8_norm_verify_block_(&position, end, end, form_flag, &previous_canonical_combining_class);
}

SZ_API_COMPTIME sz_size_t sz_utf8_norm_v128(sz_cptr_t source, sz_size_t length, sz_normal_form_t form,
                                            sz_ptr_t destination) {
    return sz_utf8_norm_engine_(source, length, form, destination, &sz_utf8_norm_classify_v128_);
}

SZ_API_COMPTIME sz_cptr_t sz_utf8_find_denormalized_v128(sz_cptr_t source, sz_size_t length, sz_normal_form_t form) {
    return sz_utf8_find_denormalized_engine_(source, length, form, &sz_utf8_norm_classify_v128_);
}

#pragma endregion simd128

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_NORM_V128_H_
