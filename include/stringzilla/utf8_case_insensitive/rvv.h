/**
 *  @brief RISC-V Vector (RVV 1.0) case-insensitive UTF-8 search, comparison & invariance backend.
 *  @file include/stringzilla/utf8_case_insensitive/rvv.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_case_insensitive.h
 */
#ifndef STRINGZILLA_UTF8_CASE_INSENSITIVE_RVV_H_
#define STRINGZILLA_UTF8_CASE_INSENSITIVE_RVV_H_

#include "stringzilla/utf8_case_fold/rvv.h" // `sz_utf8_fold_latin_c4_deltas_rvv_` & co
#include "stringzilla/utf8_case_insensitive/serial.h"

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

/*  The per-script fold/alarm strips are dispatched through constant function pointers from the force-inlined
 *  driver. GCC 14 devirtualizes & inlines those constant callbacks into the driver and miscompiles the result
 *  (a vsetvl / register-allocation pass interaction that silently drops matches); keeping them as genuine
 *  out-of-line calls — their intended shape, since they ARE invoked via pointer — is what makes -O2/-O3 (and
 *  -O0) byte-exact with serial. They are declared `static` + `noinline` directly rather than decorating
 *  `SZ_INTERNAL` (which expands to `inline static`, and GCC silently ignores `noinline` on an `inline`
 *  function while warning about it), so the out-of-line shape actually takes effect.
 *
 *  KNOWN LIMITATION: at -O1 specifically the same GCC 14 RVV codegen defect still drops matches in the
 *  alarm-routed (length-changing-fold) paths even with these out-of-line callbacks — differential fuzzing
 *  confirms -O0/-O2/-O3 are clean at all VLENs while -O1 is not, and no individual `-fno-*` pass toggle nor
 *  a per-function `optimize("O2")` attribute rescues -O1. StringZilla ships at -O2/-O3, so this is a
 *  toolchain artifact rather than a logic gap; do not "fix" it by re-inlining the callbacks. */
#if defined(__GNUC__)
#define SZ_CI_RVV_NOINLINE_ static __attribute__((noinline))
#else
#define SZ_CI_RVV_NOINLINE_ static
#endif

/*  Forward declaration: the substring dispatcher uses the invariance check (defined further below) to take
 *  the exact-search fast path for case-less needles, matching `sz_utf8_case_insensitive_find_serial`. */
SZ_PUBLIC sz_bool_t sz_utf8_case_invariant_rvv(sz_cptr_t str, sz_size_t length);

#pragma region Substring Search

/*  The RVV substring port mirrors the NEON architecture: a shared, force-inlined "scripted" driver walks the
 *  haystack in `e8m8` strips; each per-script kernel supplies two callbacks: a `fold_strip` that case-folds a
 *  length-preserving strip of haystack in place (so a folded haystack byte aligns one-to-one with its source
 *  byte), and an `alarm_strip` that flags length-CHANGING folds (ß→ss, ligatures, Kelvin/Angstrom, …). The
 *  fold logic itself is lifted verbatim from `utf8_case_fold/rvv.h`'s `_strip_rvv_` handlers, but trimmed of
 *  the "stop and serial" machinery: in a clean (un-alarmed) region every fold is one-to-one, so the strip is
 *  folded whole and the probe filter runs straight on it.
 *
 *  Candidate filtering reuses the `sz_find_rvv` idiom: four probe bytes (first / probe_second / probe_third /
 *  last of the folded needle window) are broadcast and compared against the matching offset views of the
 *  folded haystack strip, AND-ed into one predicate, then walked low-to-high with `vfirst`/`vmsif`/`vmandn`.
 *  Every surviving candidate re-folds its <=16-byte window, compares it byte-exact against the needle window,
 *  and defers to the value-exact serial `sz_utf8_case_insensitive_verify_match_`. Alarmed strips and the
 *  sub-window tail are handed to the serial `sz_utf8_case_insensitive_find_in_danger_zone_`. The result
 *  (pointer AND `*matched_length`) is therefore byte-identical to `sz_utf8_case_insensitive_find_serial`. */

/** @brief Folds a length-preserving strip of haystack bytes in place using script-specific rules. */
typedef void (*sz_utf8_ci_fold_strip_rvv_t_)(sz_u8_t const *source_ptr, sz_size_t vl, sz_u8_t *destination_ptr);

/** @brief Scans a strip for length-changing fold characters; returns the offset of the first, or -1. */
typedef long (*sz_utf8_ci_alarm_strip_rvv_t_)(sz_u8_t const *source_ptr, sz_size_t vl);

#pragma region Per-Script Fold Strips

/*  Each `_fold_strip_rvv_` folds exactly `vl` bytes from `source_ptr` into `destination_ptr`, one-to-one.
 *  They are only ever called on regions the matching alarm has cleared of length-changing folds, so the
 *  irregular continuation bytes the case-fold strips routed to serial never appear here; the fold math is
 *  the length-preserving subset of `utf8_case_fold/rvv.h`. Slides fill lane 0's predecessor with zero,
 *  matching the "zero predecessor" convention the candidate re-fold uses, so both agree at every match. */

/*  ASCII: `c + ((c - 'A' <= 25) * 0x20)`. Pure ASCII never changes byte width, so this is also the
 *  Georgian-Mkhedruli fold (Mkhedruli is caseless). */
SZ_CI_RVV_NOINLINE_ void sz_utf8_ci_fold_ascii_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vl,
                                                          sz_u8_t *destination_ptr) {
    vl = __riscv_vsetvl_e8m4(vl); // configure the vector unit (out-of-line callbacks inherit no config)
    vuint8m4_t source = __riscv_vle8_v_u8m4(source_ptr, vl);
    vbool2_t is_upper = __riscv_vmsleu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(source, 'A', vl), 25, vl);
    vuint8m4_t folded = __riscv_vmerge_vvm_u8m4(source, __riscv_vadd_vx_u8m4(source, 0x20, vl), is_upper, vl);
    __riscv_vse8_v_u8m4(destination_ptr, folded, vl);
}

/*  Western Europe: ASCII A-Z, Latin-1 Supplement 'À'-'Þ' (C3 80-9E, excluding '×' 0x97) +0x20, and the
 *  in-place ß→"ss" (both bytes of C3 9F become 's'). Length-changing folds are routed to the alarm. */
SZ_CI_RVV_NOINLINE_ void sz_utf8_ci_fold_western_europe_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vl,
                                                                   sz_u8_t *destination_ptr) {
    vl = __riscv_vsetvl_e8m4(vl); // configure the vector unit (out-of-line callbacks inherit no config)
    vuint8m4_t source = __riscv_vle8_v_u8m4(source_ptr, vl);
    vuint8m4_t previous = __riscv_vslide1up_vx_u8m4(source, 0, vl);
    sz_u8_t next_carry = source_ptr[vl]; // padded buffer guarantees one readable byte past `vl`
    vuint8m4_t next = __riscv_vslide1down_vx_u8m4(source, next_carry, vl);

    vbool2_t after_c3 = __riscv_vmseq_vx_u8m4_b2(previous, 0xC3, vl);
    vbool2_t is_upper = __riscv_vmsleu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(source, 'A', vl), 25, vl);
    vuint8m4_t folded = __riscv_vmerge_vvm_u8m4(source, __riscv_vadd_vx_u8m4(source, 0x20, vl), is_upper, vl);

    // Latin-1 'À'-'Þ' (C3 80-9E, excluding '×' 0x97) get +0x20.
    vbool2_t latin1_range = __riscv_vmsltu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(source, 0x80, vl), 0x1F, vl);
    vbool2_t is_latin1_upper = __riscv_vmandn_mm_b2(__riscv_vmand_mm_b2(after_c3, latin1_range, vl),
                                                    __riscv_vmseq_vx_u8m4_b2(source, 0x97, vl), vl);
    folded = __riscv_vadd_vx_u8m4_m(is_latin1_upper, folded, 0x20, vl);
    // ß (C3 9F) -> "ss": both bytes become 's'.
    vbool2_t eszett_lead = __riscv_vmand_mm_b2(__riscv_vmseq_vx_u8m4_b2(source, 0xC3, vl),
                                               __riscv_vmseq_vx_u8m4_b2(next, 0x9F, vl), vl);
    vbool2_t eszett_second = __riscv_vmand_mm_b2(after_c3, __riscv_vmseq_vx_u8m4_b2(source, 0x9F, vl), vl);
    folded = __riscv_vmerge_vxm_u8m4(folded, 's', __riscv_vmor_mm_b2(eszett_lead, eszett_second, vl), vl);
    __riscv_vse8_v_u8m4(destination_ptr, folded, vl);
}

/*  Central Europe: Latin-1 Supplement +0x20 (C3 80-9E, except '×' 0x97) and Latin Extended-A +1 parity
 *  deltas from the C4/C5 LUTs. Irregulars ('İ', 'ŉ', 'Ŀ', 'Ÿ', 'ſ', …) are alarm-routed. */
SZ_CI_RVV_NOINLINE_ void sz_utf8_ci_fold_central_europe_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vl,
                                                                   sz_u8_t *destination_ptr) {
    vl = __riscv_vsetvl_e8m4(vl); // configure the vector unit (out-of-line callbacks inherit no config)
    vuint8m4_t source = __riscv_vle8_v_u8m4(source_ptr, vl);
    vuint8m4_t previous = __riscv_vslide1up_vx_u8m4(source, 0, vl);

    vbool2_t is_continuation = __riscv_vmseq_vx_u8m4_b2(__riscv_vand_vx_u8m4(source, 0xC0, vl), 0x80, vl);
    vbool2_t after_c3 = __riscv_vmseq_vx_u8m4_b2(previous, 0xC3, vl);
    vbool2_t after_c4 = __riscv_vmseq_vx_u8m4_b2(previous, 0xC4, vl);
    vbool2_t after_c5 = __riscv_vmseq_vx_u8m4_b2(previous, 0xC5, vl);

    vbool2_t is_upper = __riscv_vmsleu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(source, 'A', vl), 25, vl);
    vuint8m4_t folded = __riscv_vmerge_vvm_u8m4(source, __riscv_vadd_vx_u8m4(source, 0x20, vl), is_upper, vl);

    vbool2_t latin1_range = __riscv_vmsltu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(source, 0x80, vl), 0x1F, vl);
    vbool2_t is_latin1_upper = __riscv_vmandn_mm_b2(__riscv_vmand_mm_b2(after_c3, latin1_range, vl),
                                                    __riscv_vmseq_vx_u8m4_b2(source, 0x97, vl), vl);
    folded = __riscv_vadd_vx_u8m4_m(is_latin1_upper, folded, 0x20, vl);

    // Latin Extended-A/B +1 parity deltas, keyed by the continuation byte's low 6 bits, gathered per lead.
    vuint8m4_t low6 = __riscv_vand_vx_u8m4(source, 0x3F, vl);
    vuint8m4_t c4_lut = __riscv_vle8_v_u8m4(sz_utf8_fold_latin_c4_deltas_rvv_, 64);
    vuint8m4_t c5_lut = __riscv_vle8_v_u8m4(sz_utf8_fold_latin_c5_deltas_rvv_, 64);
    vuint8m4_t delta = __riscv_vmv_v_x_u8m4(0, vl);
    delta = __riscv_vmerge_vvm_u8m4(delta, __riscv_vrgather_vv_u8m4(c4_lut, low6, vl), after_c4, vl);
    delta = __riscv_vmerge_vvm_u8m4(delta, __riscv_vrgather_vv_u8m4(c5_lut, low6, vl), after_c5, vl);
    delta = __riscv_vmerge_vvm_u8m4(__riscv_vmv_v_x_u8m4(0, vl), delta, is_continuation, vl);
    // Clear the 0x80 irregular flag (alarm already routed those positions away): keep only the +1 bit.
    delta = __riscv_vand_vx_u8m4(delta, 0x01, vl);
    folded = __riscv_vadd_vv_u8m4(folded, delta, vl);
    __riscv_vse8_v_u8m4(destination_ptr, folded, vl);
}

/*  Cyrillic: basic D0/D1. Second-byte offset by high nibble after a D0 lead (8→+0x10, 9→+0x20, A→−0x20)
 *  plus the masked D0→D1 (+1) lead rewrite. Extended Cyrillic is banned at needle-analysis time. */
SZ_CI_RVV_NOINLINE_ void sz_utf8_ci_fold_cyrillic_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vl,
                                                             sz_u8_t *destination_ptr) {
    static sz_u8_t const second_byte_offsets[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x20, 0xE0, 0, 0, 0, 0, 0};
    vl = __riscv_vsetvl_e8m4(vl); // configure the vector unit (out-of-line callbacks inherit no config)
    vuint8m4_t source = __riscv_vle8_v_u8m4(source_ptr, vl);
    vuint8m4_t previous = __riscv_vslide1up_vx_u8m4(source, 0, vl);
    sz_u8_t next_carry = source_ptr[vl];
    vuint8m4_t next = __riscv_vslide1down_vx_u8m4(source, next_carry, vl);

    vbool2_t after_d0 = __riscv_vmseq_vx_u8m4_b2(previous, 0xD0, vl);
    vbool2_t is_d0 = __riscv_vmseq_vx_u8m4_b2(source, 0xD0, vl);

    vbool2_t is_upper = __riscv_vmsleu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(source, 'A', vl), 25, vl);
    vuint8m4_t folded = __riscv_vmerge_vvm_u8m4(source, __riscv_vadd_vx_u8m4(source, 0x20, vl), is_upper, vl);

    vuint8m4_t offsets_lut = __riscv_vle8_v_u8m4(second_byte_offsets, 16);
    vuint8m4_t offset = __riscv_vrgather_vv_u8m4(offsets_lut, __riscv_vsrl_vx_u8m4(source, 4, vl), vl);
    offset = __riscv_vmerge_vvm_u8m4(__riscv_vmv_v_x_u8m4(0, vl), offset, after_d0, vl);
    folded = __riscv_vadd_vv_u8m4(folded, offset, vl);

    vbool2_t next_80_8f = __riscv_vmsltu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(next, 0x80, vl), 0x10, vl);
    vbool2_t next_a0_af = __riscv_vmsltu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(next, 0xA0, vl), 0x10, vl);
    vbool2_t needs_d1 = __riscv_vmand_mm_b2(is_d0, __riscv_vmor_mm_b2(next_80_8f, next_a0_af, vl), vl);
    folded = __riscv_vadd_vx_u8m4_m(needs_d1, folded, 1, vl);
    __riscv_vse8_v_u8m4(destination_ptr, folded, vl);
}

/*  Monotonic-Greek second-byte fold deltas after a CE lead, indexed by `text & 0x3F` (same table the NEON
 *  Greek fold uses): 'Ά' (86) +0x26, 'Έ'-'Ί' (88-8A) +0x25, 'Ύ'/'Ώ' (8E-8F) −1, 'Α'-'Ο' (91-9F) +0x20,
 *  'Π'-'Ω'/'Ϊ'/'Ϋ' (A0-AB) −0x20. 'Ό' (8C) keeps its byte (lead-only change). */
static sz_u8_t const sz_utf8_ci_greek_ce_deltas_rvv_[64] = {
    // clang-format off
    0, 0, 0, 0, 0, 0, 0x26, 0, 0x25, 0x25, 0x25, 0, 0, 0, 0xFF, 0xFF, // CE 80-8F
    0, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, // CE 90-9F
    0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0, 0, 0, 0, // CE A0-AF
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // CE B0-BF
    // clang-format on
};
/*  CE→CF (+1) lead-promotion flags for the second-byte classes whose lowercase lands in the CF block:
 *  'Ό' (8C), 'Ύ'/'Ώ' (8E-8F), 'Π'-'Ω'/'Ϊ'/'Ϋ' (A0-AB). 'Α'-'Ο' (91-9F) stay under CE. */
static sz_u8_t const sz_utf8_ci_greek_ce_promotes_rvv_[64] = {
    // clang-format off
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, // CE 80-8F: 8C, 8E, 8F promote
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // CE 90-9F: stay under CE
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, // CE A0-AB promote
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // CE B0-BF
    // clang-format on
};

/*  Greek: monotonic CE/CF + micro sign. Second-byte deltas after CE come from `ce_deltas` (gather over the
 *  64-entry LUT), the CE→CF lead promotion from `ce_promotes` carried one lane back, final sigma 'ς' (CF 82)
 *  +1, and 'µ' (C2 B5)→'μ' (CE BC). Accented 'ΐ'/'ΰ', symbols, polytonic & archaic are alarm-routed. */
SZ_CI_RVV_NOINLINE_ void sz_utf8_ci_fold_greek_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vl,
                                                          sz_u8_t *destination_ptr) {
    vl = __riscv_vsetvl_e8m4(vl); // configure the vector unit (out-of-line callbacks inherit no config)
    vuint8m4_t source = __riscv_vle8_v_u8m4(source_ptr, vl);
    vuint8m4_t previous = __riscv_vslide1up_vx_u8m4(source, 0, vl);

    vbool2_t is_continuation = __riscv_vmseq_vx_u8m4_b2(__riscv_vand_vx_u8m4(source, 0xC0, vl), 0x80, vl);
    vbool2_t after_ce = __riscv_vmseq_vx_u8m4_b2(previous, 0xCE, vl);
    vbool2_t after_cf = __riscv_vmseq_vx_u8m4_b2(previous, 0xCF, vl);
    vbool2_t after_c2 = __riscv_vmseq_vx_u8m4_b2(previous, 0xC2, vl);
    vbool2_t after_ce_cont = __riscv_vmand_mm_b2(after_ce, is_continuation, vl);

    vbool2_t is_upper = __riscv_vmsleu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(source, 'A', vl), 25, vl);
    vuint8m4_t folded = __riscv_vmerge_vvm_u8m4(source, __riscv_vadd_vx_u8m4(source, 0x20, vl), is_upper, vl);

    // Second-byte CE deltas and CE->CF promotion flags via gather (continuation-gated to avoid aliasing).
    vuint8m4_t low6 = __riscv_vand_vx_u8m4(source, 0x3F, vl);
    vuint8m4_t deltas_lut = __riscv_vle8_v_u8m4(sz_utf8_ci_greek_ce_deltas_rvv_, 64);
    vuint8m4_t promotes_lut = __riscv_vle8_v_u8m4(sz_utf8_ci_greek_ce_promotes_rvv_, 64);
    vuint8m4_t ce_delta = __riscv_vmerge_vvm_u8m4(__riscv_vmv_v_x_u8m4(0, vl),
                                                  __riscv_vrgather_vv_u8m4(deltas_lut, low6, vl), after_ce_cont, vl);
    vuint8m4_t promote_second = __riscv_vmerge_vvm_u8m4(
        __riscv_vmv_v_x_u8m4(0, vl), __riscv_vrgather_vv_u8m4(promotes_lut, low6, vl), after_ce_cont, vl);
    folded = __riscv_vadd_vv_u8m4(folded, ce_delta, vl);

    // Final sigma 'ς' (CF 82) +1; micro sign 'µ' (C2 B5) second byte -> 0xBC (μ).
    vbool2_t final_sigma = __riscv_vmand_mm_b2(after_cf, __riscv_vmseq_vx_u8m4_b2(source, 0x82, vl), vl);
    folded = __riscv_vadd_vx_u8m4_m(final_sigma, folded, 0x01, vl);
    vbool2_t micro_second = __riscv_vmand_mm_b2(after_c2, __riscv_vmseq_vx_u8m4_b2(source, 0xB5, vl), vl);
    folded = __riscv_vmerge_vxm_u8m4(folded, 0xBC, micro_second, vl);

    // Lead rewrites carried one lane back from the second byte: CE->CF (+1) and C2->CE (+0x0C, micro sign).
    vuint8m4_t micro_second_byte = __riscv_vmerge_vxm_u8m4(__riscv_vmv_v_x_u8m4(0, vl), 1, micro_second, vl);
    vuint8m4_t promote_lead = __riscv_vslide1down_vx_u8m4(promote_second, 0, vl);
    vuint8m4_t micro_lead = __riscv_vslide1down_vx_u8m4(micro_second_byte, 0, vl);
    folded = __riscv_vadd_vx_u8m4_m(__riscv_vmsne_vx_u8m4_b2(promote_lead, 0, vl), folded, 0x01, vl);
    folded = __riscv_vadd_vx_u8m4_m(__riscv_vmsne_vx_u8m4_b2(micro_lead, 0, vl), folded, 0x0C, vl);
    __riscv_vse8_v_u8m4(destination_ptr, folded, vl);
}

/*  Armenian: D4/D5/D6. Disjoint second-byte offsets (D4 B1-BF and D5 90-96 fold −0x10, D5 80-8F folds
 *  +0x30) plus the lead +1 rewrites D4→D5 (next B1-BF) and D5→D6 (next 90-96). 'և' is alarm-routed. */
SZ_CI_RVV_NOINLINE_ void sz_utf8_ci_fold_armenian_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vl,
                                                             sz_u8_t *destination_ptr) {
    vl = __riscv_vsetvl_e8m4(vl); // configure the vector unit (out-of-line callbacks inherit no config)
    vuint8m4_t source = __riscv_vle8_v_u8m4(source_ptr, vl);
    vuint8m4_t previous = __riscv_vslide1up_vx_u8m4(source, 0, vl);
    sz_u8_t next_carry = source_ptr[vl];
    vuint8m4_t next = __riscv_vslide1down_vx_u8m4(source, next_carry, vl);

    vbool2_t is_d4 = __riscv_vmseq_vx_u8m4_b2(source, 0xD4, vl);
    vbool2_t is_d5 = __riscv_vmseq_vx_u8m4_b2(source, 0xD5, vl);
    vbool2_t after_d4 = __riscv_vmseq_vx_u8m4_b2(previous, 0xD4, vl);
    vbool2_t after_d5 = __riscv_vmseq_vx_u8m4_b2(previous, 0xD5, vl);

    vbool2_t is_upper = __riscv_vmsleu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(source, 'A', vl), 25, vl);
    vuint8m4_t folded = __riscv_vmerge_vvm_u8m4(source, __riscv_vadd_vx_u8m4(source, 0x20, vl), is_upper, vl);

    vbool2_t d4_second = __riscv_vmand_mm_b2(after_d4, __riscv_vmsgeu_vx_u8m4_b2(source, 0xB1, vl), vl);
    vbool2_t d5_plus30 = __riscv_vmand_mm_b2(
        after_d5, __riscv_vmsltu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(source, 0x80, vl), 0x10, vl), vl);
    vbool2_t d5_minus10 = __riscv_vmand_mm_b2(
        after_d5, __riscv_vmsltu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(source, 0x90, vl), 0x07, vl), vl);
    folded = __riscv_vadd_vx_u8m4_m(__riscv_vmor_mm_b2(d4_second, d5_minus10, vl), folded, 0xF0, vl); // -0x10
    folded = __riscv_vadd_vx_u8m4_m(d5_plus30, folded, 0x30, vl);
    vbool2_t promotes_d4 = __riscv_vmand_mm_b2(is_d4, __riscv_vmsgeu_vx_u8m4_b2(next, 0xB1, vl), vl);
    vbool2_t promotes_d5 = __riscv_vmand_mm_b2(
        is_d5, __riscv_vmsltu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(next, 0x90, vl), 0x07, vl), vl);
    folded = __riscv_vadd_vx_u8m4_m(__riscv_vmor_mm_b2(promotes_d4, promotes_d5, vl), folded, 0x01, vl);
    __riscv_vse8_v_u8m4(destination_ptr, folded, vl);
}

/*  Vietnamese: Latin-1 Supplement +0x20, Latin Extended-A parity, 'Ơ'/'Ư' (C6 A0/AF) +1, and Latin
 *  Extended Additional (E1 B8-BB) even-third +1, all length-preserving. Expanding folds are alarm-routed.
 *  Reuses the C4/C5/C6 delta LUTs from `utf8_case_fold/rvv.h`, masked to the +1 bit. */
SZ_CI_RVV_NOINLINE_ void sz_utf8_ci_fold_vietnamese_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vl,
                                                               sz_u8_t *destination_ptr) {
    vl = __riscv_vsetvl_e8m4(vl); // configure the vector unit (out-of-line callbacks inherit no config)
    vuint8m4_t source = __riscv_vle8_v_u8m4(source_ptr, vl);
    vuint8m4_t previous = __riscv_vslide1up_vx_u8m4(source, 0, vl);
    vuint8m4_t previous2 = __riscv_vslide1up_vx_u8m4(previous, 0, vl);

    vbool2_t is_continuation = __riscv_vmseq_vx_u8m4_b2(__riscv_vand_vx_u8m4(source, 0xC0, vl), 0x80, vl);
    vbool2_t after_c3 = __riscv_vmseq_vx_u8m4_b2(previous, 0xC3, vl);
    vbool2_t after_c4 = __riscv_vmseq_vx_u8m4_b2(previous, 0xC4, vl);
    vbool2_t after_c5 = __riscv_vmseq_vx_u8m4_b2(previous, 0xC5, vl);
    vbool2_t after_c6 = __riscv_vmseq_vx_u8m4_b2(previous, 0xC6, vl);

    vbool2_t is_upper = __riscv_vmsleu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(source, 'A', vl), 25, vl);
    vuint8m4_t folded = __riscv_vmerge_vvm_u8m4(source, __riscv_vadd_vx_u8m4(source, 0x20, vl), is_upper, vl);

    // Latin-1 'À'-'Þ' +0x20 (except '×' 0x97).
    vbool2_t latin1_range = __riscv_vmsltu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(source, 0x80, vl), 0x1F, vl);
    vbool2_t is_latin1_upper = __riscv_vmandn_mm_b2(__riscv_vmand_mm_b2(after_c3, latin1_range, vl),
                                                    __riscv_vmseq_vx_u8m4_b2(source, 0x97, vl), vl);
    folded = __riscv_vadd_vx_u8m4_m(is_latin1_upper, folded, 0x20, vl);

    // Latin Extended-A/B +1 parity deltas from the shared C4/C5/C6 LUTs (keep only the +1 bit).
    vuint8m4_t low6 = __riscv_vand_vx_u8m4(source, 0x3F, vl);
    vuint8m4_t c4_lut = __riscv_vle8_v_u8m4(sz_utf8_fold_latin_c4_deltas_rvv_, 64);
    vuint8m4_t c5_lut = __riscv_vle8_v_u8m4(sz_utf8_fold_latin_c5_deltas_rvv_, 64);
    vuint8m4_t c6_lut = __riscv_vle8_v_u8m4(sz_utf8_fold_latin_c6_deltas_rvv_, 64);
    vuint8m4_t delta = __riscv_vmv_v_x_u8m4(0, vl);
    delta = __riscv_vmerge_vvm_u8m4(delta, __riscv_vrgather_vv_u8m4(c4_lut, low6, vl), after_c4, vl);
    delta = __riscv_vmerge_vvm_u8m4(delta, __riscv_vrgather_vv_u8m4(c5_lut, low6, vl), after_c5, vl);
    delta = __riscv_vmerge_vvm_u8m4(delta, __riscv_vrgather_vv_u8m4(c6_lut, low6, vl), after_c6, vl);
    delta = __riscv_vmerge_vvm_u8m4(__riscv_vmv_v_x_u8m4(0, vl), delta, is_continuation, vl);
    delta = __riscv_vand_vx_u8m4(delta, 0x01, vl);
    folded = __riscv_vadd_vv_u8m4(folded, delta, vl);

    // Latin Extended Additional E1 B8-BB: even third byte +1 (excluding the E1 BA 96-9F expanding block,
    // which is alarm-routed). Third byte sits two lanes after the E1 lead and one after the B8-BB second.
    vbool2_t after_e1_pair = __riscv_vmand_mm_b2(
        __riscv_vmseq_vx_u8m4_b2(previous2, 0xE1, vl),
        __riscv_vmsltu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(previous, 0xB8, vl), 0x04, vl), vl);
    vbool2_t third_even = __riscv_vmseq_vx_u8m4_b2(__riscv_vand_vx_u8m4(source, 0x01, vl), 0, vl);
    vbool2_t fold_e1 = __riscv_vmand_mm_b2(after_e1_pair, third_even, vl);
    folded = __riscv_vadd_vx_u8m4_m(fold_e1, folded, 0x01, vl);
    __riscv_vse8_v_u8m4(destination_ptr, folded, vl);
}

#pragma endregion // Per-Script Fold Strips

#pragma region Per-Script Alarm Strips

/*  Each `_alarm_strip_rvv_` returns the lane offset of the FIRST length-changing fold character in the
 *  strip, or -1 if the strip is clean. Anchored at the SECOND byte of each multi-byte pattern (lead read
 *  from the `previous` slide, third byte from the `next` slide), then `vslide1up` carries the flag back
 *  one lane onto the lead so the returned offset is the sequence START — matching how the danger-zone
 *  handler expects to begin scanning. The slides zero-fill the strip edges; padded loads keep range
 *  compares on `next` safe-negative past the real data. */

/*  Danger lanes accumulate as a 0/1 byte vector (one `vor` per rule), keeping the rule algebra free of the
 *  nested mask-intrinsic arity that the mask-domain form invites. `eq` builds a 0/1 byte from a byte compare;
 *  `_to_lead_` shifts the second-byte danger flags back one lane onto the lead and reports the first set. */
SZ_INTERNAL vuint8m4_t sz_utf8_ci_eq_byte_(vuint8m4_t bytes, sz_u8_t value, sz_size_t vl) {
    return __riscv_vmerge_vxm_u8m4(__riscv_vmv_v_x_u8m4(0, vl), 1, __riscv_vmseq_vx_u8m4_b2(bytes, value, vl), vl);
}

SZ_INTERNAL vuint8m4_t sz_utf8_ci_in_range_byte_(vuint8m4_t bytes, sz_u8_t start, sz_u8_t length, sz_size_t vl) {
    vbool2_t in_range = __riscv_vmsltu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(bytes, start, vl), length, vl);
    return __riscv_vmerge_vxm_u8m4(__riscv_vmv_v_x_u8m4(0, vl), 1, in_range, vl);
}

SZ_INTERNAL long sz_utf8_ci_alarm_to_lead_(vuint8m4_t danger_at_second, sz_size_t vl) {
    // Carry the second-byte flags back one lane onto the lead (the lower index), then report the first set
    // lane. The lead precedes its second byte, so this is a slide-DOWN (`dst[i] = src[i+1]`).
    vuint8m4_t flag_at_lead = __riscv_vslide1down_vx_u8m4(danger_at_second, 0, vl);
    return __riscv_vfirst_m_b2(__riscv_vmsne_vx_u8m4_b2(flag_at_lead, 0, vl), vl);
}

SZ_CI_RVV_NOINLINE_ long sz_utf8_ci_alarm_western_europe_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vl) {
    vl = __riscv_vsetvl_e8m4(vl); // configure the vector unit (out-of-line callbacks inherit no config)
    vuint8m4_t source = __riscv_vle8_v_u8m4(source_ptr, vl);
    vuint8m4_t previous = __riscv_vslide1up_vx_u8m4(source, 0, vl);
    vuint8m4_t next = __riscv_vslide1down_vx_u8m4(source, source_ptr[vl], vl);

    vuint8m4_t after_c3 = sz_utf8_ci_eq_byte_(previous, 0xC3, vl);
    vuint8m4_t after_c5 = sz_utf8_ci_eq_byte_(previous, 0xC5, vl);
    // 'ẞ' & co (E1 BA 96-9E): expand to ASCII-led sequences.
    vuint8m4_t danger = __riscv_vand_vv_u8m4(
        __riscv_vand_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0xBA, vl), sz_utf8_ci_eq_byte_(previous, 0xE1, vl), vl),
        sz_utf8_ci_in_range_byte_(next, 0x96, 0x09, vl), vl);
    // Kelvin/Angstrom (E2 84 AA/AB).
    danger = __riscv_vor_vv_u8m4(
        danger,
        __riscv_vand_vv_u8m4(
            __riscv_vand_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0x84, vl), sz_utf8_ci_eq_byte_(previous, 0xE2, vl), vl),
            __riscv_vor_vv_u8m4(sz_utf8_ci_eq_byte_(next, 0xAA, vl), sz_utf8_ci_eq_byte_(next, 0xAB, vl), vl), vl),
        vl);
    // Ligatures (EF AC xx).
    danger = __riscv_vor_vv_u8m4(
        danger,
        __riscv_vand_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0xAC, vl), sz_utf8_ci_eq_byte_(previous, 0xEF, vl), vl), vl);
    // Long S (C5 BF) and 'Ÿ' (C5 B8).
    danger = __riscv_vor_vv_u8m4(danger, __riscv_vand_vv_u8m4(after_c5, sz_utf8_ci_eq_byte_(source, 0xBF, vl), vl), vl);
    danger = __riscv_vor_vv_u8m4(danger, __riscv_vand_vv_u8m4(after_c5, sz_utf8_ci_eq_byte_(source, 0xB8, vl), vl), vl);
    // Sharp S (C3 9F).
    danger = __riscv_vor_vv_u8m4(danger, __riscv_vand_vv_u8m4(after_c3, sz_utf8_ci_eq_byte_(source, 0x9F, vl), vl), vl);
    return sz_utf8_ci_alarm_to_lead_(danger, vl);
}

SZ_CI_RVV_NOINLINE_ long sz_utf8_ci_alarm_central_europe_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vl) {
    vl = __riscv_vsetvl_e8m4(vl); // configure the vector unit (out-of-line callbacks inherit no config)
    vuint8m4_t source = __riscv_vle8_v_u8m4(source_ptr, vl);
    vuint8m4_t previous = __riscv_vslide1up_vx_u8m4(source, 0, vl);

    vuint8m4_t after_c3 = sz_utf8_ci_eq_byte_(previous, 0xC3, vl);
    vuint8m4_t after_c4 = sz_utf8_ci_eq_byte_(previous, 0xC4, vl);
    vuint8m4_t after_c5 = sz_utf8_ci_eq_byte_(previous, 0xC5, vl);
    // Kelvin (E2 84).
    vuint8m4_t danger = __riscv_vand_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0x84, vl),
                                             sz_utf8_ci_eq_byte_(previous, 0xE2, vl), vl);
    // Sharp S (C3 9F).
    danger = __riscv_vor_vv_u8m4(danger, __riscv_vand_vv_u8m4(after_c3, sz_utf8_ci_eq_byte_(source, 0x9F, vl), vl), vl);
    // Dotted I (C4 B0) and 'Ŀ' (C4 BF).
    danger = __riscv_vor_vv_u8m4(danger, __riscv_vand_vv_u8m4(after_c4, sz_utf8_ci_eq_byte_(source, 0xB0, vl), vl), vl);
    danger = __riscv_vor_vv_u8m4(danger, __riscv_vand_vv_u8m4(after_c4, sz_utf8_ci_eq_byte_(source, 0xBF, vl), vl), vl);
    // Long S (C5 BF) and 'Ÿ' (C5 B8).
    danger = __riscv_vor_vv_u8m4(danger, __riscv_vand_vv_u8m4(after_c5, sz_utf8_ci_eq_byte_(source, 0xBF, vl), vl), vl);
    danger = __riscv_vor_vv_u8m4(danger, __riscv_vand_vv_u8m4(after_c5, sz_utf8_ci_eq_byte_(source, 0xB8, vl), vl), vl);
    // Ligatures (EF AC xx).
    danger = __riscv_vor_vv_u8m4(
        danger,
        __riscv_vand_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0xAC, vl), sz_utf8_ci_eq_byte_(previous, 0xEF, vl), vl), vl);
    return sz_utf8_ci_alarm_to_lead_(danger, vl);
}

SZ_CI_RVV_NOINLINE_ long sz_utf8_ci_alarm_cyrillic_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vl) {
    vl = __riscv_vsetvl_e8m4(vl); // configure the vector unit (out-of-line callbacks inherit no config)
    vuint8m4_t source = __riscv_vle8_v_u8m4(source_ptr, vl);
    vuint8m4_t previous = __riscv_vslide1up_vx_u8m4(source, 0, vl);
    vuint8m4_t next = __riscv_vslide1down_vx_u8m4(source, source_ptr[vl], vl);

    // Cyrillic Extended-C (E1 B2 80-88) folds into basic 2-byte Cyrillic letters.
    vuint8m4_t danger = __riscv_vand_vv_u8m4(
        __riscv_vand_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0xB2, vl), sz_utf8_ci_eq_byte_(previous, 0xE1, vl), vl),
        sz_utf8_ci_in_range_byte_(next, 0x80, 0x09, vl), vl);
    return sz_utf8_ci_alarm_to_lead_(danger, vl);
}

SZ_CI_RVV_NOINLINE_ long sz_utf8_ci_alarm_greek_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vl) {
    vl = __riscv_vsetvl_e8m4(vl); // configure the vector unit (out-of-line callbacks inherit no config)
    vuint8m4_t source = __riscv_vle8_v_u8m4(source_ptr, vl);
    vuint8m4_t previous = __riscv_vslide1up_vx_u8m4(source, 0, vl);

    vuint8m4_t after_ce = sz_utf8_ci_eq_byte_(previous, 0xCE, vl);
    vuint8m4_t after_cf = sz_utf8_ci_eq_byte_(previous, 0xCF, vl);
    // 'ΐ','ΰ' (CE 90 / CE B0), plus the unassigned U+03A2 (CE A2) which folds to itself in serial but the
    // strip's promoting-range delta would mis-fold to 'ς' (CF 82) — route it to the serial scanner instead.
    vuint8m4_t second_90_b0_a2 = __riscv_vor_vv_u8m4(
        __riscv_vor_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0x90, vl), sz_utf8_ci_eq_byte_(source, 0xB0, vl), vl),
        sz_utf8_ci_eq_byte_(source, 0xA2, vl), vl);
    vuint8m4_t danger = __riscv_vand_vv_u8m4(after_ce, second_90_b0_a2, vl);
    // Greek symbols (CF 90/91/95/96, CF B0/B1/B4/B5).
    vuint8m4_t second_9x = __riscv_vor_vv_u8m4(
        __riscv_vor_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0x90, vl), sz_utf8_ci_eq_byte_(source, 0x91, vl), vl),
        __riscv_vor_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0x95, vl), sz_utf8_ci_eq_byte_(source, 0x96, vl), vl), vl);
    vuint8m4_t second_bx = __riscv_vor_vv_u8m4(
        __riscv_vor_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0xB0, vl), sz_utf8_ci_eq_byte_(source, 0xB1, vl), vl),
        __riscv_vor_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0xB4, vl), sz_utf8_ci_eq_byte_(source, 0xB5, vl), vl), vl);
    danger = __riscv_vor_vv_u8m4(danger,
                                 __riscv_vand_vv_u8m4(after_cf, __riscv_vor_vv_u8m4(second_9x, second_bx, vl), vl), vl);
    // Ohm sign (E2 84).
    danger = __riscv_vor_vv_u8m4(
        danger,
        __riscv_vand_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0x84, vl), sz_utf8_ci_eq_byte_(previous, 0xE2, vl), vl), vl);
    long lead_danger = sz_utf8_ci_alarm_to_lead_(danger, vl);

    // Blanket polytonic & archaic LEADS (E1 / CD): these are flagged at their own lane, not the second byte.
    vbool2_t lead_blanket = __riscv_vmor_mm_b2(__riscv_vmseq_vx_u8m4_b2(source, 0xE1, vl),
                                               __riscv_vmseq_vx_u8m4_b2(source, 0xCD, vl), vl);
    long blanket_danger = __riscv_vfirst_m_b2(lead_blanket, vl);

    if (lead_danger < 0) return blanket_danger;
    if (blanket_danger < 0) return lead_danger;
    return lead_danger < blanket_danger ? lead_danger : blanket_danger;
}

SZ_CI_RVV_NOINLINE_ long sz_utf8_ci_alarm_armenian_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vl) {
    vl = __riscv_vsetvl_e8m4(vl); // configure the vector unit (out-of-line callbacks inherit no config)
    vuint8m4_t source = __riscv_vle8_v_u8m4(source_ptr, vl);
    vuint8m4_t previous = __riscv_vslide1up_vx_u8m4(source, 0, vl);

    // Ech-Yiwn (D6 87).
    vuint8m4_t danger = __riscv_vand_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0x87, vl),
                                             sz_utf8_ci_eq_byte_(previous, 0xD6, vl), vl);
    // Ligatures (EF AC xx).
    danger = __riscv_vor_vv_u8m4(
        danger,
        __riscv_vand_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0xAC, vl), sz_utf8_ci_eq_byte_(previous, 0xEF, vl), vl), vl);
    return sz_utf8_ci_alarm_to_lead_(danger, vl);
}

SZ_CI_RVV_NOINLINE_ long sz_utf8_ci_alarm_vietnamese_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vl) {
    vl = __riscv_vsetvl_e8m4(vl); // configure the vector unit (out-of-line callbacks inherit no config)
    vuint8m4_t source = __riscv_vle8_v_u8m4(source_ptr, vl);
    vuint8m4_t previous = __riscv_vslide1up_vx_u8m4(source, 0, vl);
    vuint8m4_t next = __riscv_vslide1down_vx_u8m4(source, source_ptr[vl], vl);

    // E1 BA 96-9F (expanding third byte).
    vuint8m4_t danger = __riscv_vand_vv_u8m4(
        __riscv_vand_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0xBA, vl), sz_utf8_ci_eq_byte_(previous, 0xE1, vl), vl),
        sz_utf8_ci_in_range_byte_(next, 0x96, 0x0A, vl), vl);
    // Sharp S (C3 9F).
    danger = __riscv_vor_vv_u8m4(
        danger,
        __riscv_vand_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0x9F, vl), sz_utf8_ci_eq_byte_(previous, 0xC3, vl), vl), vl);
    // Long S (C5 BF).
    danger = __riscv_vor_vv_u8m4(
        danger,
        __riscv_vand_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0xBF, vl), sz_utf8_ci_eq_byte_(previous, 0xC5, vl), vl), vl);
    // Ligatures (EF AC xx).
    danger = __riscv_vor_vv_u8m4(
        danger,
        __riscv_vand_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0xAC, vl), sz_utf8_ci_eq_byte_(previous, 0xEF, vl), vl), vl);
    // Kelvin (E2 84).
    danger = __riscv_vor_vv_u8m4(
        danger,
        __riscv_vand_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0x84, vl), sz_utf8_ci_eq_byte_(previous, 0xE2, vl), vl), vl);
    // Cross-block Latin Extended folds the in-place strip can't do (e.g. 'Ŀ' C4 BF -> C5 80, 'Ÿ' C5 B8 ->
    // C3 BF, and many C6 letters): the shared C4/C5/C6 delta LUTs flag those continuation bytes with 0x80.
    // Such characters are still Vietnamese-SAFE per the needle classifier, so the haystack must route them
    // to the serial scanner instead of folding them wrong. Anchored at the second byte like the rest.
    vuint8m4_t after_c4 = sz_utf8_ci_eq_byte_(previous, 0xC4, vl);
    vuint8m4_t after_c5 = sz_utf8_ci_eq_byte_(previous, 0xC5, vl);
    vuint8m4_t after_c6 = sz_utf8_ci_eq_byte_(previous, 0xC6, vl);
    vuint8m4_t low6 = __riscv_vand_vx_u8m4(source, 0x3F, vl);
    vuint8m4_t c4_lut = __riscv_vle8_v_u8m4(sz_utf8_fold_latin_c4_deltas_rvv_, 64);
    vuint8m4_t c5_lut = __riscv_vle8_v_u8m4(sz_utf8_fold_latin_c5_deltas_rvv_, 64);
    vuint8m4_t c6_lut = __riscv_vle8_v_u8m4(sz_utf8_fold_latin_c6_deltas_rvv_, 64);
    vuint8m4_t delta = __riscv_vmv_v_x_u8m4(0, vl);
    delta = __riscv_vmerge_vvm_u8m4(delta, __riscv_vrgather_vv_u8m4(c4_lut, low6, vl),
                                    __riscv_vmsne_vx_u8m4_b2(after_c4, 0, vl), vl);
    delta = __riscv_vmerge_vvm_u8m4(delta, __riscv_vrgather_vv_u8m4(c5_lut, low6, vl),
                                    __riscv_vmsne_vx_u8m4_b2(after_c5, 0, vl), vl);
    delta = __riscv_vmerge_vvm_u8m4(delta, __riscv_vrgather_vv_u8m4(c6_lut, low6, vl),
                                    __riscv_vmsne_vx_u8m4_b2(after_c6, 0, vl), vl);
    vuint8m4_t irregular = sz_utf8_ci_eq_byte_(__riscv_vand_vx_u8m4(delta, 0x80, vl), 0x80, vl);
    danger = __riscv_vor_vv_u8m4(danger, irregular, vl);
    return sz_utf8_ci_alarm_to_lead_(danger, vl);
}

SZ_CI_RVV_NOINLINE_ long sz_utf8_ci_alarm_georgian_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t vl) {
    vl = __riscv_vsetvl_e8m4(vl); // configure the vector unit (out-of-line callbacks inherit no config)
    vuint8m4_t source = __riscv_vle8_v_u8m4(source_ptr, vl);
    vuint8m4_t previous = __riscv_vslide1up_vx_u8m4(source, 0, vl);
    vuint8m4_t next = __riscv_vslide1down_vx_u8m4(source, source_ptr[vl], vl);

    vuint8m4_t after_e1 = sz_utf8_ci_eq_byte_(previous, 0xE1, vl);
    // Mtavruli (E1 B2).
    vuint8m4_t danger = __riscv_vand_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0xB2, vl), after_e1, vl);
    // Asomtavruli (E1 82 A0-E5).
    danger = __riscv_vor_vv_u8m4(
        danger,
        __riscv_vand_vv_u8m4(__riscv_vand_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0x82, vl), after_e1, vl),
                             sz_utf8_ci_in_range_byte_(next, 0xA0, 0x46, vl), vl),
        vl);
    // Nuskhuri (E2 B4).
    danger = __riscv_vor_vv_u8m4(
        danger,
        __riscv_vand_vv_u8m4(sz_utf8_ci_eq_byte_(source, 0xB4, vl), sz_utf8_ci_eq_byte_(previous, 0xE2, vl), vl), vl);
    return sz_utf8_ci_alarm_to_lead_(danger, vl);
}

#pragma endregion // Per-Script Alarm Strips

#pragma region Scripted Driver

/*  Loads up to `length` bytes from `source` into a zeroed 64-byte scratch buffer, returning a pointer into a
 *  caller-provided buffer whose first `length` bytes are the data and the rest zero. The padding lets the
 *  fold/alarm strips read `source_ptr[vl]` for their `next` carry and keeps range compares safe-negative. */
SZ_INTERNAL sz_u8_t const *sz_utf8_ci_load_padded_rvv_(sz_cptr_t source, sz_size_t length, sz_u8_t *buffer,
                                                       sz_size_t buffer_capacity) {
    for (sz_size_t byte_index = 0; byte_index < buffer_capacity; ++byte_index) buffer[byte_index] = 0;
    for (sz_size_t byte_index = 0; byte_index < length; ++byte_index) buffer[byte_index] = (sz_u8_t)source[byte_index];
    return buffer;
}

/*  Shared scan loop behind every script-specific case-insensitive search, mirroring the NEON driver.
 *  Walks the haystack in `e8m8` strips; the `fold`/`alarm` callbacks resolve to direct calls because the
 *  driver is force-inlined into each thin wrapper. Alarmed strips and the sub-window tail go to the serial
 *  danger-zone handler; clean strips are folded and probe-filtered, with each survivor re-folded, byte-
 *  compared against the needle window, and verified by `sz_utf8_case_insensitive_verify_match_`. */
SZ_FORCE_INLINE sz_cptr_t sz_utf8_case_insensitive_find_rvv_scripted_( //
    sz_utf8_ci_fold_strip_rvv_t_ fold,                                 //
    sz_utf8_ci_alarm_strip_rvv_t_ alarm,                               //
    sz_cptr_t haystack, sz_size_t haystack_length,                     //
    sz_cptr_t needle, sz_size_t needle_length,                         //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata, //
    sz_size_t *matched_length) {

    sz_assert_(needle_metadata && "needle_metadata must be provided");
    sz_assert_(needle_metadata->folded_slice_length > 0 && "folded window must be non-empty");

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_assert_(folded_window_length <= 16 && "expect folded needle part to fit in one register");
    sz_cptr_t const haystack_end = haystack + haystack_length;

    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;
    sz_u8_t const probe_first = needle_metadata->folded_slice[0];
    sz_u8_t const probe_second = needle_metadata->folded_slice[offset_second];
    sz_u8_t const probe_third = needle_metadata->folded_slice[offset_third];
    sz_u8_t const probe_last = needle_metadata->folded_slice[offset_last];

    // The first folded rune of the safe window, for the serial danger-zone pivot.
    sz_rune_t needle_first_safe_folded_rune = 0;
    if (alarm) {
        sz_rune_length_t rune_byte_length;
        sz_rune_parse_unchecked((sz_cptr_t)needle_metadata->folded_slice, &needle_first_safe_folded_rune,
                                &rune_byte_length);
    }

    // Scratch space: a folded-strip buffer (e8m8 strip + slack) and a per-candidate window buffer.
    // RVV 1.0 caps VLEN at 64 Kib, so an e8m8 strip is at most VLMAX = 8*VLEN/8 = 8192 bytes; we cap the
    // strip at a fixed, generous size to keep the buffer on the stack and the fold/store fully covered.
    enum { strip_capacity_k = 4096, fold_pad_k = 8 };
    sz_u8_t source_buffer[strip_capacity_k + fold_pad_k];
    sz_u8_t folded_buffer[strip_capacity_k + fold_pad_k];

    sz_cptr_t haystack_ptr = haystack;
    while (haystack_ptr < haystack_end) {
        sz_size_t const available = (sz_size_t)(haystack_end - haystack_ptr);
        if (available < folded_window_length) break;

        // A strip covers `chunk_size` bytes but only its first `valid_starts` are candidate START positions.
        // Cap `chunk_size` to the e8m4 VLMAX the fold/alarm strips can process in one pass: those call
        // `vsetvl_e8m4(chunk_size)` internally and would silently fold fewer bytes if `chunk_size` exceeded it.
        sz_size_t chunk_request = available < strip_capacity_k ? available : strip_capacity_k;
        sz_size_t chunk_size = __riscv_vsetvl_e8m4(chunk_request);
        sz_size_t const valid_starts = chunk_size - folded_window_length + 1;
        // With danger characters, retreat the step by `folded_window_length - 1` so a multi-byte pattern
        // straddling the strip edge stays fully visible in the next strip; otherwise advance by all starts.
        sz_size_t const step = !alarm                                ? valid_starts
                               : valid_starts > folded_window_length ? valid_starts - (folded_window_length - 1)
                                                                     : 1;

        // Load the strip into a zero-padded buffer so the fold/alarm strips can read one carry byte past it.
        sz_u8_t const *source = sz_utf8_ci_load_padded_rvv_(haystack_ptr, chunk_size, source_buffer,
                                                            strip_capacity_k + fold_pad_k);

        if (alarm) {
            long danger_offset = alarm(source, chunk_size);
            if (danger_offset >= 0) {
                // Scan the WHOLE strip serially: an expanding fold makes the haystack span shorter than the
                // folded window, so a real match can begin within the window's length of the strip's end.
                sz_cptr_t match = sz_utf8_case_insensitive_find_in_danger_zone_( //
                    haystack, haystack_length,                                   //
                    needle, needle_length,                                       //
                    haystack_ptr, chunk_size,                                    //
                    needle_first_safe_folded_rune,                               //
                    needle_metadata->offset_in_unfolded,                         //
                    matched_length);
                if (match) return match;
                haystack_ptr += step;
                continue;
            }
        }

        // Fold the clean strip in place (one-to-one), then run the 4-probe filter over candidate starts.
        fold(source, chunk_size, folded_buffer);

        // The probe filter runs at the SAME LMUL (e8m4) as the fold/alarm strips and the `chunk_size`
        // configuration above, keeping one consistent vector config across the whole driver body.
        sz_size_t position = 0;
        while (position < valid_starts) {
            size_t vl = __riscv_vsetvl_e8m4(valid_starts - position);
            vuint8m4_t first_view = __riscv_vle8_v_u8m4(folded_buffer + position, vl);
            vuint8m4_t second_view = __riscv_vle8_v_u8m4(folded_buffer + position + offset_second, vl);
            vuint8m4_t third_view = __riscv_vle8_v_u8m4(folded_buffer + position + offset_third, vl);
            vuint8m4_t last_view = __riscv_vle8_v_u8m4(folded_buffer + position + offset_last, vl);
            vbool2_t match_mask = __riscv_vmseq_vx_u8m4_b2(first_view, probe_first, vl);
            match_mask = __riscv_vmand_mm_b2(match_mask, __riscv_vmseq_vx_u8m4_b2(second_view, probe_second, vl), vl);
            match_mask = __riscv_vmand_mm_b2(match_mask, __riscv_vmseq_vx_u8m4_b2(third_view, probe_third, vl), vl);
            match_mask = __riscv_vmand_mm_b2(match_mask, __riscv_vmseq_vx_u8m4_b2(last_view, probe_last, vl), vl);

            for (long match_index = __riscv_vfirst_m_b2(match_mask, vl); match_index >= 0;
                 match_index = __riscv_vfirst_m_b2(match_mask, vl)) {
                sz_size_t const candidate_offset = position + (sz_size_t)match_index;
                sz_cptr_t const haystack_candidate_ptr = haystack_ptr + candidate_offset;

                // Re-fold the candidate's folded window and compare it byte-exact against the needle window.
                // The folded strip is one-to-one with the source, so `folded_buffer + candidate_offset`
                // already holds the folded window; verify all `folded_window_length` bytes match.
                int window_ok = 1;
                for (sz_size_t byte_index = 0; byte_index < folded_window_length; ++byte_index)
                    if (folded_buffer[candidate_offset + byte_index] != needle_metadata->folded_slice[byte_index]) {
                        window_ok = 0;
                        break;
                    }
                if (window_ok) {
                    sz_cptr_t match = sz_utf8_case_insensitive_verify_match_(    //
                        haystack, haystack_length,                               //
                        needle, needle_length,                                   //
                        haystack_candidate_ptr - haystack, folded_window_length, //
                        needle_metadata->offset_in_unfolded,                     //
                        needle_length - needle_metadata->offset_in_unfolded -    //
                            needle_metadata->length_in_unfolded,                 //
                        matched_length);
                    if (match) return match;
                }
                match_mask = __riscv_vmandn_mm_b2(match_mask, __riscv_vmsif_m_b2(match_mask, vl), vl);
            }
            position += vl;
        }
        haystack_ptr += step;
    }

    // Expanding danger characters make the haystack span shorter than the folded window, so a match can
    // still start in the sub-window tail the main loop never probes; the tail is short, so serial is cheap.
    if (alarm && haystack_ptr < haystack_end) {
        sz_cptr_t match = sz_utf8_case_insensitive_find_in_danger_zone_( //
            haystack, haystack_length,                                   //
            needle, needle_length,                                       //
            haystack_ptr, (sz_size_t)(haystack_end - haystack_ptr),      //
            needle_first_safe_folded_rune,                               //
            needle_metadata->offset_in_unfolded,                         //
            matched_length);
        if (match) return match;
    }

    return SZ_NULL_CHAR;
}

#pragma endregion // Scripted Driver

#pragma region Per-Script Kernels

SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_rvv_ascii_( //
    sz_cptr_t haystack, sz_size_t haystack_length,              //
    sz_cptr_t needle, sz_size_t needle_length,                  //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_case_insensitive_find_rvv_scripted_(
        sz_utf8_ci_fold_ascii_strip_rvv_, (sz_utf8_ci_alarm_strip_rvv_t_)SZ_NULL, haystack, haystack_length, needle,
        needle_length, needle_metadata, matched_length);
}

SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_rvv_western_europe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                       //
    sz_cptr_t needle, sz_size_t needle_length,                           //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_case_insensitive_find_rvv_scripted_(
        sz_utf8_ci_fold_western_europe_strip_rvv_, sz_utf8_ci_alarm_western_europe_strip_rvv_, haystack,
        haystack_length, needle, needle_length, needle_metadata, matched_length);
}

SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_rvv_central_europe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                       //
    sz_cptr_t needle, sz_size_t needle_length,                           //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_case_insensitive_find_rvv_scripted_(
        sz_utf8_ci_fold_central_europe_strip_rvv_, sz_utf8_ci_alarm_central_europe_strip_rvv_, haystack,
        haystack_length, needle, needle_length, needle_metadata, matched_length);
}

SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_rvv_cyrillic_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                 //
    sz_cptr_t needle, sz_size_t needle_length,                     //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_case_insensitive_find_rvv_scripted_(sz_utf8_ci_fold_cyrillic_strip_rvv_,
                                                       sz_utf8_ci_alarm_cyrillic_strip_rvv_, haystack, haystack_length,
                                                       needle, needle_length, needle_metadata, matched_length);
}

SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_rvv_greek_( //
    sz_cptr_t haystack, sz_size_t haystack_length,              //
    sz_cptr_t needle, sz_size_t needle_length,                  //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_case_insensitive_find_rvv_scripted_(sz_utf8_ci_fold_greek_strip_rvv_,
                                                       sz_utf8_ci_alarm_greek_strip_rvv_, haystack, haystack_length,
                                                       needle, needle_length, needle_metadata, matched_length);
}

SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_rvv_armenian_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                 //
    sz_cptr_t needle, sz_size_t needle_length,                     //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_case_insensitive_find_rvv_scripted_(sz_utf8_ci_fold_armenian_strip_rvv_,
                                                       sz_utf8_ci_alarm_armenian_strip_rvv_, haystack, haystack_length,
                                                       needle, needle_length, needle_metadata, matched_length);
}

SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_rvv_vietnamese_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                   //
    sz_cptr_t needle, sz_size_t needle_length,                       //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_case_insensitive_find_rvv_scripted_(
        sz_utf8_ci_fold_vietnamese_strip_rvv_, sz_utf8_ci_alarm_vietnamese_strip_rvv_, haystack, haystack_length,
        needle, needle_length, needle_metadata, matched_length);
}

SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_rvv_georgian_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                 //
    sz_cptr_t needle, sz_size_t needle_length,                     //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata, sz_size_t *matched_length) {
    // Mkhedruli is caseless, so the fold is the bare ASCII fold; the alarm watches the historical scripts.
    return sz_utf8_case_insensitive_find_rvv_scripted_(sz_utf8_ci_fold_ascii_strip_rvv_,
                                                       sz_utf8_ci_alarm_georgian_strip_rvv_, haystack, haystack_length,
                                                       needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Per-Script Kernels

SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_rvv( //
    sz_cptr_t haystack, sz_size_t haystack_length,     //
    sz_cptr_t needle, sz_size_t needle_length,         //
    sz_utf8_case_insensitive_needle_metadata_t *needle_metadata, sz_size_t *matched_length) {

    // Handle the obvious edge cases first.
    if (needle_length == 0) {
        *matched_length = 0;
        return haystack;
    }

    // If the needle is entirely case-less, perform a direct (exact) substring search.
    int const is_unknown = needle_metadata->kernel_id == sz_utf8_case_rune_unknown_k;
    int const known_agnostic = needle_metadata->kernel_id == sz_utf8_case_rune_case_invariant_k;
    if (known_agnostic || (is_unknown && sz_utf8_case_invariant_rvv(needle, needle_length))) {
        sz_cptr_t result = sz_find(haystack, haystack_length, needle, needle_length);
        *matched_length = result ? needle_length : 0;
        return result;
    }

    // Analyze the needle to find the best safe window and kernel.
    if (is_unknown) {
        sz_utf8_case_insensitive_needle_metadata_(needle, needle_length, needle_metadata);
        if (needle_metadata->kernel_id == sz_utf8_case_rune_fallback_serial_k)
            return sz_utf8_case_insensitive_find_serial(haystack, haystack_length, needle, needle_length,
                                                        needle_metadata, matched_length);
    }

    switch (needle_metadata->kernel_id) {
    case sz_utf8_case_rune_ascii_invariant_k:
        return sz_utf8_case_insensitive_find_rvv_ascii_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    case sz_utf8_case_rune_safe_western_europe_k:
        return sz_utf8_case_insensitive_find_rvv_western_europe_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    case sz_utf8_case_rune_safe_central_europe_k:
        return sz_utf8_case_insensitive_find_rvv_central_europe_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    case sz_utf8_case_rune_safe_cyrillic_k:
        return sz_utf8_case_insensitive_find_rvv_cyrillic_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    case sz_utf8_case_rune_safe_greek_k:
        return sz_utf8_case_insensitive_find_rvv_greek_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    case sz_utf8_case_rune_safe_armenian_k:
        return sz_utf8_case_insensitive_find_rvv_armenian_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    case sz_utf8_case_rune_safe_vietnamese_k:
        return sz_utf8_case_insensitive_find_rvv_vietnamese_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    case sz_utf8_case_rune_safe_georgian_k:
        return sz_utf8_case_insensitive_find_rvv_georgian_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    default: break;
    }

    // No suitable SIMD path (complex Unicode needle), fall back to serial.
    needle_metadata->kernel_id = sz_utf8_case_rune_fallback_serial_k;
    return sz_utf8_case_insensitive_find_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                                matched_length);
}

#pragma endregion // Substring Search

/*  Byte-for-byte equivalent to `sz_utf8_case_invariant_serial`. A string is NOT case-invariant the moment it
 *  contains a case-participating character. ASCII letters (`A`-`Z`, `a`-`z`) occupy `0x41-0x7A`, byte values
 *  that can never appear inside a multi-byte sequence, so a vector scan for "ASCII letter OR any non-ASCII
 *  byte" is exact: an ASCII-letter hit means not invariant; a non-ASCII hit (always a lead, since the scan
 *  starts on a codepoint boundary) is decoded and checked by the value-exact serial `sz_rune_is_case_invariant_`.
 *  Caseless ASCII (digits, punctuation, control) is skipped a whole vector strip at a time. */
SZ_PUBLIC sz_bool_t sz_utf8_case_invariant_rvv(sz_cptr_t str, sz_size_t length) {
    sz_u8_t const *cursor = (sz_u8_t const *)str;
    sz_u8_t const *end = cursor + length;
    while (cursor < end) {
        // Skip caseless ASCII in strips; stop at the first ASCII letter or non-ASCII lead.
        sz_size_t remaining = (sz_size_t)(end - cursor);
        sz_size_t skip = 0;
        while (skip < remaining) {
            size_t vl = __riscv_vsetvl_e8m8(remaining - skip);
            vuint8m8_t bytes = __riscv_vle8_v_u8m8(cursor + skip, vl);
            // ASCII letter: (byte | 0x20) in [0x61, 0x7A]; folds to lowercase so it never false-matches >= 0x80.
            vbool1_t is_letter = __riscv_vmsleu_vx_u8m8_b1(
                __riscv_vsub_vx_u8m8(__riscv_vor_vx_u8m8(bytes, 0x20, vl), 0x61, vl), 0x19, vl);
            vbool1_t is_non_ascii = __riscv_vmsgtu_vx_u8m8_b1(bytes, 0x7F, vl);
            long first = __riscv_vfirst_m_b1(__riscv_vmor_mm_b1(is_letter, is_non_ascii, vl), vl);
            if (first < 0) {
                skip += vl;
                continue;
            }
            skip += (sz_size_t)first;
            break;
        }
        cursor += skip;
        if (cursor >= end) break;
        if (*cursor < 0x80) return sz_false_k; // an ASCII letter participates in case

        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse((sz_cptr_t)cursor, (sz_cptr_t)end, &rune, &rune_length);
        if (sz_rune_is_case_invariant_(rune) == sz_false_k) return sz_false_k;
        cursor += rune_length;
    }
    return sz_true_k;
}

SZ_PUBLIC sz_ordering_t sz_utf8_case_insensitive_order_rvv(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                           sz_size_t b_length) {
    return sz_utf8_case_insensitive_order_serial(a, a_length, b, b_length);
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

#endif // STRINGZILLA_UTF8_CASE_INSENSITIVE_RVV_H_
