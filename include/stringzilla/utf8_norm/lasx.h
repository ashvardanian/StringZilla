/**
 *  @brief LoongArch LASX (256-bit) backend for the single-pass Unicode normalizer (NFD / NFC / NFKD / NFKC).
 *  @file include/stringzilla/utf8_norm/lasx.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_norm.h
 *
 *  This backend overrides exactly one point of the shared engine: the scan primitive
 *  `sz_utf8_norm_classify_lasx_`, which locates the first non-inert byte for a form. The two public
 *  entry points (`sz_utf8_norm_lasx` / `sz_utf8_find_denormalized_lasx`) reuse the force-inlined engines
 *  from `serial.h`, passing this scanner as the constant function address that devirtualizes the call.
 *
 *  The scanner mirrors the Skylake structure at 256-bit width: a 32-byte all-ASCII gate via the
 *  `sz_xvmovemask_b_utf8_norm_lasx_` reduction, then a lead-byte classify over the shared 64-entry
 *  `sz_utf8_norm_lead_lut_`, then the shared cold per-codepoint verify (`sz_utf8_norm_verify_block_`).
 *  LASX has no 64-entry permute, so the lookup uses the even/odd two-table `__lasx_xvshuf_b` blend
 *  from `find/lasx.h`: `xvshuf_b(hi, lo, index)` is a 32-entry (two 16-byte tables) per-128-bit-lane
 *  select keyed on the low five index bits, and two such selects (index < 32 vs >= 32) are blended by
 *  `__lasx_xvbitsel_v` on bit five of the index to cover all 64 entries.
 */
#ifndef STRINGZILLA_UTF8_NORM_LASX_H_
#define STRINGZILLA_UTF8_NORM_LASX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_norm/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_LASX

/*  See `utf8_runes/lasx.h`: `__lasx_xvmskltz_b` packs each byte's sign bit into a per-128-bit-lane
 *  16-bit mask (word 0 = low lane, word 4 = high lane), recombined to match AVX2's `_mm256_movemask_epi8`. */
SZ_HELPER_INLINE sz_u32_t sz_xvmovemask_b_utf8_norm_lasx_(__m256i sign_extended) {
    __m256i collected = __lasx_xvmskltz_b(sign_extended);
    sz_u32_t low = (sz_u32_t)__lasx_xvpickve2gr_wu(collected, 0);
    sz_u32_t high = (sz_u32_t)__lasx_xvpickve2gr_wu(collected, 4);
    return (low & 0xFFFFu) | ((high & 0xFFFFu) << 16);
}

/**
 *  @brief 64-entry lead lookup without a wide permute: two even/odd `__lasx_xvshuf_b` selects over the
 *         broadcast LUT halves, blended on bit five of the index. `families & form_flag` then identifies
 *         the form. @p index holds `byte & 0x3F` per lane.
 *
 *  `__lasx_xvshuf_b(high, low, index)` selects, per 128-bit lane, `low[index]` for `index` in [0, 16) and
 *  `high[index - 16]` for `index` in [16, 32), reading only the low five index bits. So one select covers
 *  index range [0, 32) (tables 0..15 and 16..31), a second covers [32, 64) because the discarded bit five
 *  re-bases the index onto [0, 32) over tables 32..47 and 48..63. `__lasx_xvbitsel_v` then picks the high
 *  select wherever bit five of the index is set (index >= 32).
 */
SZ_HELPER_INLINE __m256i sz_utf8_norm_lead_lookup_lasx_(__m256i index, __m256i table_low_0, __m256i table_low_1,
                                                        __m256i table_high_0, __m256i table_high_1) {
    __m256i families_low = __lasx_xvshuf_b(table_low_1, table_low_0, index);
    __m256i families_high = __lasx_xvshuf_b(table_high_1, table_high_0, index);
    // Bit five (value 0x20) of the index is set exactly for index >= 32: select the high half there.
    __m256i select_high = __lasx_xvslt_bu(__lasx_xvreplgr2vr_b(0x1F), index);
    return __lasx_xvbitsel_v(families_low, families_high, select_high);
}

/**
 *  @brief Scan primitive (LASX): first byte that begins a non-inert codepoint for @p form, else NULL.
 *
 *  Matches `sz_utf8_norm_classify_serial_` semantics, computed from the unified props trie. The hot loop
 *  uses a 32-byte window gate plus the two-table `__lasx_xvshuf_b` lead-classify; the cold per-codepoint
 *  verify carries the combining class across windows and reports order or quick-check violations exactly.
 */
SZ_HELPER_NOINLINE sz_cptr_t sz_utf8_norm_classify_lasx_(sz_cptr_t text, sz_size_t length, sz_normal_form_t form) {
    sz_u8_t const *position = (sz_u8_t const *)text;
    sz_u8_t const *const end = position + length;
    sz_u8_t const form_flag = sz_utf8_norm_form_flag_(form);
    sz_u8_t previous_canonical_combining_class = 0;

    // The 64-entry LUT split into four 16-byte tables, each duplicated into both 128-bit lanes so the
    // per-lane `__lasx_xvshuf_b` selects the same table for either half (mirroring `find/lasx.h`).
    sz_u8_t table_low_0_bytes[32], table_low_1_bytes[32], table_high_0_bytes[32], table_high_1_bytes[32];
    for (sz_size_t lane = 0; lane != 16; ++lane) {
        table_low_0_bytes[lane] = table_low_0_bytes[lane + 16] = sz_utf8_norm_lead_lut_[lane + 0];
        table_low_1_bytes[lane] = table_low_1_bytes[lane + 16] = sz_utf8_norm_lead_lut_[lane + 16];
        table_high_0_bytes[lane] = table_high_0_bytes[lane + 16] = sz_utf8_norm_lead_lut_[lane + 32];
        table_high_1_bytes[lane] = table_high_1_bytes[lane + 16] = sz_utf8_norm_lead_lut_[lane + 48];
    }
    __m256i const table_low_0 = __lasx_xvld(table_low_0_bytes, 0);
    __m256i const table_low_1 = __lasx_xvld(table_low_1_bytes, 0);
    __m256i const table_high_0 = __lasx_xvld(table_high_0_bytes, 0);
    __m256i const table_high_1 = __lasx_xvld(table_high_1_bytes, 0);

    while (position + 32 <= end) {
        __m256i bytes = __lasx_xvld(position, 0);
        // All-ASCII gate: a high (sign) bit marks a non-ASCII byte. An ASCII-only window is inert.
        sz_u32_t non_ascii = sz_xvmovemask_b_utf8_norm_lasx_(bytes);
        if (non_ascii == 0) {
            position += 32, previous_canonical_combining_class = 0;
            continue;
        }
        // Lead bytes only: non-ASCII and not a 10xxxxxx continuation byte.
        __m256i non_ascii_vec = __lasx_xvslt_b(bytes, __lasx_xvreplgr2vr_b(0));
        __m256i continuation_vec = __lasx_xvseq_b(__lasx_xvand_v(bytes, __lasx_xvreplgr2vr_b((char)0xC0)),
                                                  __lasx_xvreplgr2vr_b((char)0x80));
        __m256i is_lead_vec = __lasx_xvandn_v(continuation_vec, non_ascii_vec);
        // Classify each lead via the 64-entry LUT (index = byte & 0x3F), then keep the flagged form bit.
        __m256i index = __lasx_xvand_v(bytes, __lasx_xvreplgr2vr_b(0x3F));
        __m256i families = sz_utf8_norm_lead_lookup_lasx_(index, table_low_0, table_low_1, table_high_0, table_high_1);
        __m256i has_flag = __lasx_xvslt_bu(__lasx_xvreplgr2vr_b(0),
                                           __lasx_xvand_v(families, __lasx_xvreplgr2vr_b((char)form_flag)));
        __m256i flagged_vec = __lasx_xvand_v(is_lead_vec, has_flag);
        if (sz_xvmovemask_b_utf8_norm_lasx_(flagged_vec) == 0) {
            // 32 bytes inert for the form: skip, then realign onto a codepoint boundary.
            position += 32, previous_canonical_combining_class = 0;
            while (position < end && (*position & 0xC0) == 0x80) ++position;
            continue;
        }
        // A candidate lead is somewhere in the window; verify exactly, codepoint by codepoint.
        sz_cptr_t violation = sz_utf8_norm_verify_block_(&position, position + 32, end, form_flag,
                                                         &previous_canonical_combining_class);
        if (violation) return violation;
    }
    // Tail (< 32 bytes): the shared scalar verify carries the combining class across the final boundary.
    return sz_utf8_norm_verify_block_(&position, end, end, form_flag, &previous_canonical_combining_class);
}

SZ_API_COMPTIME sz_size_t sz_utf8_norm_lasx(sz_cptr_t source, sz_size_t length, sz_normal_form_t form,
                                            sz_ptr_t destination) {
    return sz_utf8_norm_engine_(source, length, form, destination, &sz_utf8_norm_classify_lasx_);
}

SZ_API_COMPTIME sz_cptr_t sz_utf8_find_denormalized_lasx(sz_cptr_t source, sz_size_t length, sz_normal_form_t form) {
    return sz_utf8_find_denormalized_engine_(source, length, form, &sz_utf8_norm_classify_lasx_);
}

#endif // SZ_USE_LASX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_NORM_LASX_H_
