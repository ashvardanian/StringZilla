/**
 *  @brief NEON backend for the single-pass Unicode normalizer (NFD / NFC / NFKD / NFKC).
 *  @file include/stringzilla/utf8_norm/neon.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_norm.h
 *
 *  This backend overrides exactly one point of the shared engine: the scan primitive
 *  `sz_utf8_norm_classify_neon_`, which locates the first non-inert byte for a form. The two public
 *  entry points (`sz_utf8_norm_neon` / `sz_utf8_norm_violation_neon`) reuse the force-inlined engines
 *  from `serial.h`, passing this scanner as the constant function address that devirtualizes the call.
 *
 *  The scanner is a clean port of the proven `utf8_tokens/sz_utf8_find_denormalized_neon` hot loop -
 *  a `vqtbl4q_u8` lead-classify behind a 64-byte ASCII/inert gate, with a cold per-codepoint exact
 *  verify - reading the unified `utf8_norm/tables.h` record set instead of the legacy denorm tables.
 */
#ifndef STRINGZILLA_UTF8_NORM_NEON_H_
#define STRINGZILLA_UTF8_NORM_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_norm/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

/** @brief Per-16B classify: nonzero lanes mark lead bytes that are candidate-non-inert for the form. */
SZ_INTERNAL uint8x16_t sz_utf8_norm_classify_neon_lead_(uint8x16_t v, uint8x16x4_t lut, uint8x16_t flag_vec) {
    uint8x16_t non_ascii = vcgeq_u8(v, vdupq_n_u8(0x80));
    uint8x16_t continuation = vcltq_u8(vsubq_u8(v, vdupq_n_u8(0x80)), vdupq_n_u8(0x40));
    uint8x16_t is_lead = vbicq_u8(non_ascii, continuation);
    uint8x16_t families = vqtbl4q_u8(lut, vandq_u8(v, vdupq_n_u8(0x3F)));
    return vandq_u8(vandq_u8(families, flag_vec), is_lead);
}

/**
 *  @brief Scan primitive (NEON): first byte that begins a non-inert codepoint for @p form, else NULL.
 *
 *  Matches `sz_utf8_norm_classify_serial_` semantics, computed from the unified props trie. The hot loop
 *  uses a 64-byte superchunk gate plus a `vqtbl4q_u8` lead-classify; the cold per-codepoint verify
 *  carries the combining class across chunks and reports order or QC violations exactly.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_norm_classify_neon_(sz_cptr_t text, sz_size_t length, sz_normal_form_t form) {
    sz_u8_t const *ptr = (sz_u8_t const *)text;
    sz_u8_t const *const end = ptr + length;
    sz_u8_t const flag = sz_utf8_norm_form_flag_(form);

    uint8x16x4_t lut;
    lut.val[0] = vld1q_u8(sz_utf8_norm_lead_lut_ + 0);
    lut.val[1] = vld1q_u8(sz_utf8_norm_lead_lut_ + 16);
    lut.val[2] = vld1q_u8(sz_utf8_norm_lead_lut_ + 32);
    lut.val[3] = vld1q_u8(sz_utf8_norm_lead_lut_ + 48);
    uint8x16_t const flag_vec = vdupq_n_u8(flag);
    sz_u8_t previous_canonical_combining_class = 0;

    // 64-byte superchunk gate: one horizontal reduction per 64B for the (overwhelmingly common) inert
    // case. All-ASCII blocks skip with zero LUT work; otherwise the flagged-lead OR is reduced once.
    while (ptr + 64 <= end) {
        uint8x16_t v0 = vld1q_u8(ptr), v1 = vld1q_u8(ptr + 16);
        uint8x16_t v2 = vld1q_u8(ptr + 32), v3 = vld1q_u8(ptr + 48);
        uint8x16_t any = vorrq_u8(vorrq_u8(v0, v1), vorrq_u8(v2, v3));
        if (vmaxvq_u8(any) < 0x80) {
            ptr += 64, previous_canonical_combining_class = 0;
            continue;
        } // all 64 bytes ASCII: inert
        uint8x16_t flagged = vorrq_u8(vorrq_u8(sz_utf8_norm_classify_neon_lead_(v0, lut, flag_vec),
                                               sz_utf8_norm_classify_neon_lead_(v1, lut, flag_vec)),
                                      vorrq_u8(sz_utf8_norm_classify_neon_lead_(v2, lut, flag_vec),
                                               sz_utf8_norm_classify_neon_lead_(v3, lut, flag_vec)));
        if (vmaxvq_u8(flagged) == 0) { // 64 bytes inert for the form
            ptr += 64, previous_canonical_combining_class = 0;
            while (ptr < end && (*ptr & 0xC0) == 0x80) ++ptr;
            continue;
        }
        // A candidate lead is somewhere in the block; verify exactly, codepoint by codepoint.
        sz_cptr_t violation = sz_utf8_norm_verify_block_(&ptr, ptr + 64, end, flag,
                                                         &previous_canonical_combining_class);
        if (violation) return violation;
    }

    while (ptr + 16 <= end) {
        uint8x16_t v = vld1q_u8(ptr);
        // After a skip, realign to a codepoint boundary - a 16-byte step can land mid-sequence, and the
        // straddling codepoint's lead was already classified inert in the chunk we are leaving.
        if (vmaxvq_u8(v) < 0x80) {
            ptr += 16, previous_canonical_combining_class = 0;
            while (ptr < end && (*ptr & 0xC0) == 0x80) ++ptr;
            continue; // all-ASCII: inert
        }
        // Lead bytes only (non-ASCII and not a 10xxxxxx continuation), classified via the 64-entry LUT.
        uint8x16_t non_ascii = vcgeq_u8(v, vdupq_n_u8(0x80));
        uint8x16_t continuation = vcltq_u8(vsubq_u8(v, vdupq_n_u8(0x80)), vdupq_n_u8(0x40));
        uint8x16_t is_lead = vbicq_u8(non_ascii, continuation);
        uint8x16_t families = vqtbl4q_u8(lut, vandq_u8(v, vdupq_n_u8(0x3F)));
        uint8x16_t flagged = vandq_u8(vandq_u8(families, flag_vec), is_lead);
        if (vmaxvq_u8(flagged) == 0) { // no flagged lead: inert for form
            ptr += 16, previous_canonical_combining_class = 0;
            while (ptr < end && (*ptr & 0xC0) == 0x80) ++ptr;
            continue;
        }

        // The chunk has a possibly-non-inert lead; verify exactly, codepoint by codepoint.
        sz_cptr_t violation = sz_utf8_norm_verify_block_(&ptr, ptr + 16, end, flag,
                                                         &previous_canonical_combining_class);
        if (violation) return violation;
    }

    // Tail (< 16 bytes), carrying the canonical combining class across the last chunk boundary.
    return sz_utf8_norm_verify_block_(&ptr, end, end, flag, &previous_canonical_combining_class);
}

SZ_PUBLIC sz_size_t sz_utf8_norm_neon(sz_cptr_t source, sz_size_t length, sz_normal_form_t form, sz_ptr_t destination) {
    return sz_utf8_norm_engine_(source, length, form, destination, &sz_utf8_norm_classify_neon_);
}

SZ_PUBLIC sz_cptr_t sz_utf8_norm_violation_neon(sz_cptr_t source, sz_size_t length, sz_normal_form_t form) {
    return sz_utf8_norm_violation_engine_(source, length, form, &sz_utf8_norm_classify_neon_);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_NORM_NEON_H_
