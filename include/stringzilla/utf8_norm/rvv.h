/**
 *  @brief RISC-V Vector (RVV 1.0) backend for the single-pass Unicode normalizer (NFD / NFC / NFKD / NFKC).
 *  @file include/stringzilla/utf8_norm/rvv.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_norm.h
 *
 *  Like every other backend, this overrides exactly one point of the shared engine: the scan primitive
 *  `sz_utf8_norm_classify_rvv_`, which locates the first non-inert byte for a form. The two public entry
 *  points (`sz_utf8_norm_rvv` / `sz_utf8_find_denormalized_rvv`) reuse the force-inlined engines from
 *  `serial.h`, passing this scanner as the constant function address that devirtualizes the call.
 *
 *  RVV is the cleanest backend of the family: the 64-entry lead lookup `sz_utf8_norm_lead_lut_` needs no
 *  table split (no `vpshufb` 16-lane window, no `vqtbl4q` register quad). A single indexed gather
 *  (`vluxei8`) over `byte & 0x3F` reads the family bits straight from memory at any `VLEN`, the same
 *  idiom the word-break classifier in `utf8_words/rvv.h` uses for its ASCII property table. A per-strip
 *  ASCII gate (`vmsgtu` for the high bit, reduced with `vfirst`) keeps the dominant inert case off the
 *  LUT entirely, and any strip that survives the gate is handed to the shared scalar verify.
 */
#ifndef STRINGZILLA_UTF8_NORM_RVV_H_
#define STRINGZILLA_UTF8_NORM_RVV_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_norm/serial.h"

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
 *  @brief Scan primitive (RVV): first byte that begins a non-inert codepoint for @p form, else NULL.
 *
 *  Matches `sz_utf8_norm_classify_serial_` semantics. The hot loop classifies whole `e8m8` strips: a
 *  `vfirst` over the high-bit mask skips all-ASCII strips with zero LUT work, and a `vluxei8` gather over
 *  `sz_utf8_norm_lead_lut_[byte & 0x3F]` masked by the form flag identifies any candidate-non-inert lead.
 *  The cold per-codepoint verify (`sz_utf8_norm_verify_block_`) carries the combining class across strips
 *  and reports order / quick-check violations exactly, including the partial final strip.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_norm_classify_rvv_(sz_cptr_t text, sz_size_t length, sz_normal_form_t form) {
    sz_u8_t const *position = (sz_u8_t const *)text;
    sz_u8_t const *const end = position + length;
    sz_u8_t previous_canonical_combining_class = 0;
    sz_u8_t const form_flag = sz_utf8_norm_form_flag_(form);

    while (position < end) {
        sz_size_t vector_length = __riscv_vsetvl_e8m8((sz_size_t)(end - position));
        vuint8m8_t bytes = __riscv_vle8_v_u8m8(position, vector_length);

        // ASCII gate: a strip with no high-bit byte is wholly inert for every form.
        vbool1_t non_ascii = __riscv_vmsgtu_vx_u8m8_b1(bytes, 0x7F, vector_length);
        if (__riscv_vfirst_m_b1(non_ascii, vector_length) < 0) {
            position += vector_length, previous_canonical_combining_class = 0;
            continue;
        }

        // Lead bytes only: non-ASCII and not a `10xxxxxx` continuation.
        vbool1_t continuation = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(bytes, 0xC0, vector_length), 0x80,
                                                         vector_length);
        vbool1_t is_lead = __riscv_vmandn_mm_b1(non_ascii, continuation, vector_length);

        // 64-entry lead lookup by indexed gather over `byte & 0x3F` - no table split at any VLEN.
        vuint8m8_t index = __riscv_vand_vx_u8m8(bytes, 0x3F, vector_length);
        vuint8m8_t families = __riscv_vluxei8_v_u8m8(sz_utf8_norm_lead_lut_, index, vector_length);
        vbool1_t has_flag = __riscv_vmsne_vx_u8m8_b1(__riscv_vand_vx_u8m8(families, form_flag, vector_length), 0,
                                                     vector_length);
        vbool1_t flagged = __riscv_vmand_mm_b1(is_lead, has_flag, vector_length);

        if (__riscv_vfirst_m_b1(flagged, vector_length) < 0) { // strip inert for the form: skip and realign
            position += vector_length, previous_canonical_combining_class = 0;
            while (position < end && (*position & 0xC0) == 0x80) ++position;
            continue;
        }

        // A candidate lead is somewhere in the strip; verify exactly, codepoint by codepoint.
        sz_u8_t const *block_end = position + vector_length > end ? end : position + vector_length;
        sz_cptr_t violation = sz_utf8_norm_verify_block_(&position, block_end, end, form_flag,
                                                         &previous_canonical_combining_class);
        if (violation) return violation;
    }
    // The verify handles the partial final strip in-loop, so a clean exit means the span is inert.
    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_size_t sz_utf8_norm_rvv(sz_cptr_t source, sz_size_t length, sz_normal_form_t form, sz_ptr_t destination) {
    return sz_utf8_norm_engine_(source, length, form, destination, &sz_utf8_norm_classify_rvv_);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_denormalized_rvv(sz_cptr_t source, sz_size_t length, sz_normal_form_t form) {
    return sz_utf8_find_denormalized_engine_(source, length, form, &sz_utf8_norm_classify_rvv_);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_RVV

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_NORM_RVV_H_
