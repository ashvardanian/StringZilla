/**
 *  @brief Arm NEON backend for UTF-8 case folding.
 *  @file include/stringzilla/utf8_case_fold/neon.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_case_fold.h
 */
#ifndef STRINGZILLA_UTF8_CASE_FOLD_NEON_H_
#define STRINGZILLA_UTF8_CASE_FOLD_NEON_H_

#include "stringzilla/utf8_case_fold/serial.h"

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

/**
 *  Bit flags describing which UTF-8 lead-byte families occur in a chunk - the same values and
 *  semantics as the Ice Lake `sz_utf8_fold_lead_family_t_`, kept under a NEON-suffixed name so
 *  both backends can coexist in one translation unit. Each family shares one folding strategy,
 *  so the union of flags picks the chunk handler in a single dispatch - instead of sequentially
 *  probing per-script fast paths, which degrades on mixed-script text.
 */
enum sz_utf8_fold_neon_lead_family_t_ {
    sz_utf8_fold_neon_lead_caseless_flag_ = 1 << 0,       // D7-DF, E0, E3-E9, EB-EE: scripts with no case
    sz_utf8_fold_neon_lead_latin_flag_ = 1 << 1,          // C2-C3: Latin-1 Supplement
    sz_utf8_fold_neon_lead_latin_extended_flag_ = 1 << 2, // C4-C6: Latin Extended-A and the Ext-B +1 pairs
    sz_utf8_fold_neon_lead_cyrillic_flag_ = 1 << 3,       // D0-D1: basic Cyrillic
    sz_utf8_fold_neon_lead_greek_flag_ = 1 << 4,          // CE-CF: basic Greek
    sz_utf8_fold_neon_lead_e1_flag_ = 1 << 5,             // E1: Latin Ext Additional, Georgian, Greek Extended
    sz_utf8_fold_neon_lead_guarded_flag_ = 1 << 6,        // E2, EA, EF: case-awareness depends on the second byte
    sz_utf8_fold_neon_lead_complex_flag_ = 1 << 7,        // C0-C1, C7-CD, D2-D6, F0-FF: decode or serial paths
};

/**
 *  Lead-byte family table indexed by the low 6 bits of the lead byte: leads 0xC0-0xFF map onto
 *  indices 0x00-0x3F injectively, and ASCII/continuation bytes are masked out before the lookup.
 *  Byte-for-byte the same 64 values as the Ice Lake `lead_families_lut`, but laid out in
 *  ascending index order: `vqtbl4q_u8` reads its `uint8x16x4_t` table in memory order, whereas
 *  `_mm512_set_epi8` lists lanes 0x3F → 0x00.
 */
static sz_u8_t const sz_utf8_fold_neon_lead_families_lut_[64] = {
    // clang-format off
    // Indices 0x00-0x0F (leads C0-CF): C2-C3 Latin, C4-C6 Latin Ext, C7-CD complex, CE-CF Greek
    0x80, 0x80, 0x02, 0x02, 0x04, 0x04, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x10, 0x10,
    // Indices 0x10-0x1F (leads D0-DF): D0-D1 Cyrillic, D2-D6 complex, D7-DF caseless
    0x08, 0x08, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    // Indices 0x20-0x2F (leads E0-EF): E1 special, E2/EA/EF guarded, the rest caseless
    0x01, 0x20, 0x40, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x40, 0x01, 0x01, 0x01, 0x01, 0x40,
    // Indices 0x30-0x3F (leads F0-FF): supplementary planes and invalid leads
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    // clang-format on
};

/**
 *  Per-codepoint deltas for 2-byte Latin Extended sequences, indexed by the continuation byte's
 *  low 6 bits: 0x00 = identity, 0x01 = fold by +1, 0x80 = irregular (serial fallback). The same
 *  64 values as the Ice Lake `c4_deltas_lut`, but in ascending index order: `vqtbl4q_u8` reads
 *  its table in memory order, whereas `_mm512_set_epi8` lists lanes 0x3F → 0x00.
 *  Generated from Unicode full case folding; verified against the serial reference in tests.
 */
static sz_u8_t const sz_utf8_fold_neon_c4_deltas_lut_[64] = {
    // Latin Ext-A U+0100-013F: 'Ā'-'Ŀ'
    // clang-format off
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, // 0x00-0x0F: even-parity pairs
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, // 0x10-0x1F
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, // 0x20-0x2F
    // 0x30-0x3F: 'İ' (U+0130, C4 B0) expands to "i" + combining dot, and 'Ŀ' (U+013F, C4 BF)
    // folds to 'ŀ' (U+0140, C5 80) - the +1 crosses into the next lead byte, so neither can
    // fold in place and both are flagged irregular; 'ĸ' (U+0138) is caseless, parity flips after.
    0x80, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0x80,
    // clang-format on
};
static sz_u8_t const sz_utf8_fold_neon_c5_deltas_lut_[64] = {
    // Latin Ext-A U+0140-017F: 'ŀ'-'ſ'
    // clang-format off
    0, 1, 0, 1, 0, 1, 0, 1, 0, 0x80, 1, 0, 1, 0, 1, 0, // 0x00-0x0F: odd head, 'ŉ' (U+0149) irregular
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,    // 0x10-0x1F
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,    // 0x20-0x2F
    // 0x30-0x3F: 'Ÿ' (U+0178) folds down to 'ÿ' (U+00FF) and 'ſ' (U+017F) folds down to 's' -
    // both cross blocks, so both are flagged irregular.
    1, 0, 1, 0, 1, 0, 1, 0, 0x80, 1, 0, 1, 0, 1, 0, 0x80,
    // clang-format on
};
static sz_u8_t const sz_utf8_fold_neon_c6_deltas_lut_[64] = {
    // Latin Ext-B U+0180-01BF: 'ƀ'-'ƿ'
    // clang-format off
    // Latin Extended-B mixes +1 parity pairs with uppercase letters whose lowercase lives in
    // the IPA Extensions block (U+0250+) - those 20 cross-block folds are flagged irregular.
    0, 0x80, 1, 0, 1, 0, 0x80, 1, 0, 0x80, 0x80, 1, 0, 0, 0x80, 0x80, // 0x00-0x0F
    0x80, 1, 0, 0x80, 0x80, 0, 0x80, 0x80, 1, 0, 0, 0, 0x80, 0x80, 0, 0x80, // 0x10-0x1F
    1, 0, 1, 0, 1, 0, 0x80, 1, 0, 0x80, 0, 0, 1, 0, 0x80, 1, // 0x20-0x2F
    0, 0x80, 0x80, 1, 0, 1, 0, 0x80, 1, 0, 0, 0, 1, 0, 0, 0, // 0x30-0x3F
    // clang-format on
};

/** @brief Folds ASCII A-Z down to a-z in one register, leaving every other byte unchanged. */
SZ_INTERNAL uint8x16_t sz_utf8_fold_neon_ascii_(uint8x16_t source_u8x16) {
    // Unsigned wrap-around turns the two-sided 'A' ≤ x ≤ 'Z' test into one compare: bytes below
    // 'A' wrap past 0xE5 and bytes above 'Z' land at 26+, so only A-Z stay under 26.
    uint8x16_t is_ascii_upper_u8x16 = vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8('A')), vdupq_n_u8(26));
    return vaddq_u8(source_u8x16, vandq_u8(is_ascii_upper_u8x16, vdupq_n_u8(0x20)));
}

/**
 *  @brief Produces a movemask-style 64-bit value from a NEON comparison result.
 *      Each matching byte sets one bit (bit spacing is 4 bits per byte, at positions 3, 7, ..., 63).
 *
 *  NEON has no `movemask`; `vshrn` narrowing each 16-bit lane by 4 packs one nibble per byte,
 *  which a single `vget_lane_u64` moves to a scalar register - that vector → GPR hop is the
 *  serializing edge, so callers extract once per chunk and do the rest in scalar arithmetic.
 *  https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
 */
SZ_INTERNAL sz_u64_t sz_utf8_fold_neon_nibble_mask_(uint8x16_t mask_u8x16) {
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(mask_u8x16), 4)), 0) &
           0x8888888888888888ull;
}

/** @brief OR-reduces all 16 byte lanes of one register into a single byte of accumulated flags. */
SZ_INTERNAL sz_u8_t sz_utf8_fold_neon_reduce_or_u8_(uint8x16_t flags_u8x16) {
    uint8x8_t flags_u8x8 = vorr_u8(vget_low_u8(flags_u8x16), vget_high_u8(flags_u8x16));
    sz_u64_t flags_u64 = vget_lane_u64(vreinterpret_u64_u8(flags_u8x8), 0);
    flags_u64 |= flags_u64 >> 32, flags_u64 |= flags_u64 >> 16, flags_u64 |= flags_u64 >> 8;
    return (sz_u8_t)flags_u64;
}

/**
 *  @brief Maps every lead byte in one register onto its folding-family flag; non-leads map to zero.
 *      One `vqtbl4q_u8` covers the full 64-entry table - the NEON twin of Ice Lake's single VPERMB.
 */
SZ_INTERNAL uint8x16_t sz_utf8_fold_neon_classify_(uint8x16_t source_u8x16, uint8x16x4_t lead_families_lut_u8x16x4) {
    uint8x16_t is_non_ascii_u8x16 = vcgeq_u8(source_u8x16, vdupq_n_u8(0x80));
    // Continuations are 10xxxxxx, i.e. exactly the [0x80, 0xBF] range - one wrap-around compare
    uint8x16_t is_continuation_u8x16 = vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0x80)), vdupq_n_u8(0x40));
    uint8x16_t is_lead_u8x16 = vbicq_u8(is_non_ascii_u8x16, is_continuation_u8x16);
    uint8x16_t lead_families_u8x16 = vqtbl4q_u8(lead_families_lut_u8x16x4, vandq_u8(source_u8x16, vdupq_n_u8(0x3F)));
    return vandq_u8(lead_families_u8x16, is_lead_u8x16);
}

/**
 *  @brief Folds a 64-byte superchunk containing only caseless multi-byte scripts mixed with ASCII.
 *      Folds ASCII A-Z in place and copies everything else, trimming an incomplete trailing sequence.
 *
 *  Mirrors `sz_utf8_case_fold_icelake_caseless_chunk_` at a fixed 64-byte chunk size: an
 *  incomplete sequence can only be a 2-byte lead in the last byte or a 3-byte lead in the last
 *  two bytes (4-byte leads carry the complex flag and never reach this handler), so only the
 *  last register's lead masks matter and one nibble-mask extraction covers both checks.
 *
 *  @return Bytes consumed; always 62..64, never zero - 62 bytes of any valid UTF-8 cover at
 *      least one complete sequence, so the superchunk cannot start with an incomplete one.
 */
SZ_INTERNAL sz_size_t sz_utf8_case_fold_neon_caseless_chunk_(uint8x16x4_t source_u8x16x4, sz_ptr_t target) {

    uint8x16_t last_u8x16 = source_u8x16x4.val[3];
    uint8x16_t is_two_byte_lead_u8x16 = vcltq_u8(vsubq_u8(last_u8x16, vdupq_n_u8(0xC0)), vdupq_n_u8(0x20));
    uint8x16_t is_three_byte_lead_u8x16 = vceqq_u8(vandq_u8(last_u8x16, vdupq_n_u8(0xF0)), vdupq_n_u8(0xE0));

    // A 2-byte lead is incomplete only in the last lane; a 3-byte lead in the last two lanes
    uint8x16_t last_lane_u8x16 = vsetq_lane_u8(0xFF, vdupq_n_u8(0x00), 15);
    uint8x16_t last_two_lanes_u8x16 = vsetq_lane_u8(0xFF, last_lane_u8x16, 14);
    uint8x16_t is_incomplete_u8x16 = vorrq_u8(vandq_u8(is_two_byte_lead_u8x16, last_lane_u8x16),
                                              vandq_u8(is_three_byte_lead_u8x16, last_two_lanes_u8x16));
    sz_u64_t incomplete_nibbles = sz_utf8_fold_neon_nibble_mask_(is_incomplete_u8x16);

    // Nibble positions are byte positions ×4; lanes 14-15 of the last register are bytes 62-63
    sz_size_t copy_length = incomplete_nibbles ? 48 + (sz_size_t)(sz_u64_ctz(incomplete_nibbles) / 4) : 64;

    // Caseless folding is length-preserving, so full-register stores are exact for the consumed
    // prefix; the 0-2 trailing bytes written past `copy_length` sit inside the caller-guaranteed
    // output headroom and are rewritten by the next iteration.
    vst1q_u8((sz_u8_t *)target + 0, sz_utf8_fold_neon_ascii_(source_u8x16x4.val[0]));
    vst1q_u8((sz_u8_t *)target + 16, sz_utf8_fold_neon_ascii_(source_u8x16x4.val[1]));
    vst1q_u8((sz_u8_t *)target + 32, sz_utf8_fold_neon_ascii_(source_u8x16x4.val[2]));
    vst1q_u8((sz_u8_t *)target + 48, sz_utf8_fold_neon_ascii_(source_u8x16x4.val[3]));
    return copy_length;
}

/**
 *  @brief Folds a 64-byte superchunk of Latin text in place: ASCII, Latin-1 Supplement (C2-C3),
 *      Latin Extended-A/B (C4-C6), and Latin Extended Additional (E1 B8-BB) - the working set
 *      of German, Czech, Vietnamese, and most other Latin-script languages.
 *
 *  Mirrors `sz_utf8_case_fold_icelake_latin_chunk_` at a fixed 64-byte chunk size. Latin
 *  Extended folding is parity-based: uppercase codepoints are even and fold to the next odd
 *  codepoint, and the codepoint's low bit lives in the last byte of its UTF-8 sequence, so the
 *  fold is an in-place masked +1. Per-codepoint deltas come from `vqtbl4q_u8` tables indexed by
 *  the continuation byte's low 6 bits; table entries flag the irregular codepoints (those that
 *  expand, shrink, or fold across blocks), which truncate the superchunk and route one rune to
 *  the serial fallback.
 *
 *  After-lead positions come from `vextq_u8(previous_mask, mask, 15)` - the previous register's
 *  lane 15 slides into lane 0, so sequences straddling a register boundary still see their
 *  lead's classification. A zero register seeds lane 0 of the first register, which is correct
 *  because superchunks always start on a character boundary.
 *
 *  @return Bytes consumed and written, or zero if the first character needs the serial path.
 */
SZ_INTERNAL sz_size_t sz_utf8_case_fold_neon_latin_chunk_(uint8x16x4_t source_u8x16x4, sz_cptr_t source,
                                                          sz_ptr_t target) {

    uint8x16x4_t const c4_deltas_lut_u8x16x4 = vld1q_u8_x4(sz_utf8_fold_neon_c4_deltas_lut_);
    uint8x16x4_t const c5_deltas_lut_u8x16x4 = vld1q_u8_x4(sz_utf8_fold_neon_c5_deltas_lut_);
    uint8x16x4_t const c6_deltas_lut_u8x16x4 = vld1q_u8_x4(sz_utf8_fold_neon_c6_deltas_lut_);
    uint8x16_t const zero_u8x16 = vdupq_n_u8(0x00);

    // Lead masks carried across loop iterations so after-lead positions can straddle registers
    uint8x16_t previous_is_c2_u8x16 = zero_u8x16, previous_is_c3_u8x16 = zero_u8x16;
    uint8x16_t previous_is_c4_u8x16 = zero_u8x16, previous_is_c5_u8x16 = zero_u8x16;
    uint8x16_t previous_is_c6_u8x16 = zero_u8x16, previous_is_e1_u8x16 = zero_u8x16;
    uint8x16_t previous_is_additional_second_u8x16 = zero_u8x16, previous_is_ba_second_u8x16 = zero_u8x16;

    // Stops (irregular or foreign-to-this-handler sequences) are rare, and the vector → scalar
    // hop is the serializing edge on NEON: the loop only accumulates per-register stop masks,
    // and a single `vmaxvq_u8` after it decides whether any nibble-mask extraction is needed.
    uint8x16_t stop_masks_u8x16[4];
    uint8x16_t any_stop_u8x16 = zero_u8x16;

    for (sz_size_t register_index = 0; register_index != 4; ++register_index) {
        uint8x16_t source_u8x16 = source_u8x16x4.val[register_index];
        uint8x16_t next_register_u8x16 = register_index != 3 ? source_u8x16x4.val[register_index + 1] : zero_u8x16;
        // A view of the same data shifted one byte forward lets lead positions test their own
        // continuation byte, instead of propagating result masks backwards across registers.
        // The zero filler after the last register is safe: a lead in byte 63 is an incomplete
        // sequence, and the trailing trim below excludes it from the consumed prefix anyway.
        uint8x16_t next_byte_u8x16 = vextq_u8(source_u8x16, next_register_u8x16, 1);

        // Lead byte detection: C2/C3 Latin-1, C4-C6 Latin Extended, E1 Latin Extended Additional
        uint8x16_t is_c2_u8x16 = vceqq_u8(source_u8x16, vdupq_n_u8(0xC2));
        uint8x16_t is_c3_u8x16 = vceqq_u8(source_u8x16, vdupq_n_u8(0xC3));
        uint8x16_t is_c4_u8x16 = vceqq_u8(source_u8x16, vdupq_n_u8(0xC4));
        uint8x16_t is_c5_u8x16 = vceqq_u8(source_u8x16, vdupq_n_u8(0xC5));
        uint8x16_t is_c6_u8x16 = vceqq_u8(source_u8x16, vdupq_n_u8(0xC6));
        uint8x16_t is_e1_u8x16 = vceqq_u8(source_u8x16, vdupq_n_u8(0xE1));
        uint8x16_t is_continuation_u8x16 = vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0x80)), vdupq_n_u8(0x40));

        uint8x16_t after_c2_u8x16 = vextq_u8(previous_is_c2_u8x16, is_c2_u8x16, 15);
        uint8x16_t after_c3_u8x16 = vextq_u8(previous_is_c3_u8x16, is_c3_u8x16, 15);
        uint8x16_t after_c4_u8x16 = vextq_u8(previous_is_c4_u8x16, is_c4_u8x16, 15);
        uint8x16_t after_c5_u8x16 = vextq_u8(previous_is_c5_u8x16, is_c5_u8x16, 15);
        uint8x16_t after_c6_u8x16 = vextq_u8(previous_is_c6_u8x16, is_c6_u8x16, 15);
        uint8x16_t after_e1_u8x16 = vextq_u8(previous_is_e1_u8x16, is_e1_u8x16, 15);

        // E1 sequences qualify only when the second byte is B8-BB (Latin Extended Additional);
        // other E1 sub-families (Georgian, Greek Extended) route to the generic paths.
        uint8x16_t is_additional_second_u8x16 = vandq_u8(
            after_e1_u8x16, vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0xB8)), vdupq_n_u8(0x04)));
        uint8x16_t foreign_e1_second_u8x16 = vbicq_u8(after_e1_u8x16, is_additional_second_u8x16);
        uint8x16_t is_ba_second_u8x16 = vandq_u8(is_additional_second_u8x16, vceqq_u8(source_u8x16, vdupq_n_u8(0xBA)));
        uint8x16_t additional_third_u8x16 = vextq_u8(previous_is_additional_second_u8x16, is_additional_second_u8x16,
                                                     15);
        uint8x16_t ba_third_u8x16 = vextq_u8(previous_is_ba_second_u8x16, is_ba_second_u8x16, 15);

        // Latin Extended Additional irregulars: E1 BA 96-9E expand or shrink when folded
        uint8x16_t irregular_additional_u8x16 = vandq_u8(
            ba_third_u8x16, vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0x96)), vdupq_n_u8(0x09)));

        // Latin Extended 2-byte deltas and their irregular flags
        uint8x16_t delta_indices_u8x16 = vandq_u8(source_u8x16, vdupq_n_u8(0x3F));
        uint8x16_t extended_deltas_u8x16 = vorrq_u8(
            vorrq_u8(vandq_u8(vqtbl4q_u8(c4_deltas_lut_u8x16x4, delta_indices_u8x16), after_c4_u8x16),
                     vandq_u8(vqtbl4q_u8(c5_deltas_lut_u8x16x4, delta_indices_u8x16), after_c5_u8x16)),
            vandq_u8(vqtbl4q_u8(c6_deltas_lut_u8x16x4, delta_indices_u8x16), after_c6_u8x16));
        extended_deltas_u8x16 = vandq_u8(extended_deltas_u8x16, is_continuation_u8x16);
        uint8x16_t irregular_extended_u8x16 = vtstq_u8(extended_deltas_u8x16, vdupq_n_u8(0x80));

        uint8x16_t stop_u8x16 = vorrq_u8(vorrq_u8(irregular_extended_u8x16, foreign_e1_second_u8x16),
                                         irregular_additional_u8x16);
        stop_masks_u8x16[register_index] = stop_u8x16;
        any_stop_u8x16 = vorrq_u8(any_stop_u8x16, stop_u8x16);

        // 1. ASCII A-Z
        uint8x16_t folded_u8x16 = sz_utf8_fold_neon_ascii_(source_u8x16);

        // 2. Latin-1 Supplement: 'À'-'Þ' (C3 80-9E, excluding '×' at 0x97) get +0x20
        uint8x16_t is_latin1_upper_u8x16 = vandq_u8(
            after_c3_u8x16, vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0x80)), vdupq_n_u8(0x1F)));
        is_latin1_upper_u8x16 = vbicq_u8(is_latin1_upper_u8x16, vceqq_u8(source_u8x16, vdupq_n_u8(0x97)));
        folded_u8x16 = vaddq_u8(folded_u8x16, vandq_u8(is_latin1_upper_u8x16, vdupq_n_u8(0x20)));

        // 3. 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73): replace both bytes in place
        uint8x16_t is_eszett_u8x16 = vorrq_u8(vandq_u8(after_c3_u8x16, vceqq_u8(source_u8x16, vdupq_n_u8(0x9F))),
                                              vandq_u8(is_c3_u8x16, vceqq_u8(next_byte_u8x16, vdupq_n_u8(0x9F))));
        folded_u8x16 = vbslq_u8(is_eszett_u8x16, vdupq_n_u8('s'), folded_u8x16);

        // 4. 'µ' (U+00B5, C2 B5) → 'μ' (U+03BC, CE BC): both bytes replaced in place
        uint8x16_t is_micro_lead_u8x16 = vandq_u8(is_c2_u8x16, vceqq_u8(next_byte_u8x16, vdupq_n_u8(0xB5)));
        uint8x16_t is_micro_second_u8x16 = vandq_u8(after_c2_u8x16, vceqq_u8(source_u8x16, vdupq_n_u8(0xB5)));
        folded_u8x16 = vbslq_u8(is_micro_lead_u8x16, vdupq_n_u8(0xCE), folded_u8x16);
        folded_u8x16 = vbslq_u8(is_micro_second_u8x16, vdupq_n_u8(0xBC), folded_u8x16);

        // 5. Latin Extended-A/B: masked +1 on the continuation byte. Irregular 0x80 entries
        //    corrupt their lanes, but those lanes sit at or past the truncation point below
        //    and never become part of the consumed prefix.
        folded_u8x16 = vaddq_u8(folded_u8x16, extended_deltas_u8x16);

        // 6. Latin Extended Additional: +1 on even third bytes of E1 B8-BB sequences
        uint8x16_t is_even_codepoint_u8x16 = vceqq_u8(vandq_u8(source_u8x16, vdupq_n_u8(0x01)), zero_u8x16);
        folded_u8x16 = vaddq_u8(folded_u8x16,
                                vandq_u8(vandq_u8(additional_third_u8x16, is_even_codepoint_u8x16), vdupq_n_u8(0x01)));

        // Latin folding is length-preserving for everything this handler keeps, so full-register
        // stores are exact for the consumed prefix; bytes past it sit inside the caller-guaranteed
        // output headroom and are rewritten by the next iteration.
        vst1q_u8((sz_u8_t *)target + register_index * 16, folded_u8x16);

        previous_is_c2_u8x16 = is_c2_u8x16, previous_is_c3_u8x16 = is_c3_u8x16;
        previous_is_c4_u8x16 = is_c4_u8x16, previous_is_c5_u8x16 = is_c5_u8x16;
        previous_is_c6_u8x16 = is_c6_u8x16, previous_is_e1_u8x16 = is_e1_u8x16;
        previous_is_additional_second_u8x16 = is_additional_second_u8x16;
        previous_is_ba_second_u8x16 = is_ba_second_u8x16;
    }

    // Truncate at the first irregular codepoint or foreign E1 sub-family: nibble positions are
    // byte positions ×4, and register k adds 16k bytes. The flagged byte is a continuation -
    // walk back to the lead (at most 2 steps for E1 thirds) so the consumed prefix ends on a
    // character boundary; landing on byte 0 returns zero and routes one rune to serial.
    if (vmaxvq_u8(any_stop_u8x16)) {
        sz_size_t first_flagged_position = 64;
        for (sz_size_t register_index = 0; register_index != 4; ++register_index) {
            sz_u64_t stop_nibbles = sz_utf8_fold_neon_nibble_mask_(stop_masks_u8x16[register_index]);
            if (!stop_nibbles) continue;
            first_flagged_position = register_index * 16 + (sz_size_t)(sz_u64_ctz(stop_nibbles) / 4);
            break;
        }
        while (first_flagged_position && ((sz_u8_t)source[first_flagged_position] & 0xC0) == 0x80)
            --first_flagged_position;
        return first_flagged_position; // A stop boundary is a character boundary - no trim needed
    }

    // Don't split a trailing incomplete sequence across superchunks: a 2-byte lead (C2-C6) in
    // the last byte, or an E1 lead - the only 3-byte family dispatched here - in the last two.
    // Lanes 14-15 of the last register are superchunk bytes 62-63, so the result is 62..64.
    uint8x16_t last_u8x16 = source_u8x16x4.val[3];
    uint8x16_t is_two_byte_lead_u8x16 = vcltq_u8(vsubq_u8(last_u8x16, vdupq_n_u8(0xC2)), vdupq_n_u8(0x05));
    uint8x16_t is_e1_lead_u8x16 = vceqq_u8(last_u8x16, vdupq_n_u8(0xE1));
    uint8x16_t last_lane_u8x16 = vsetq_lane_u8(0xFF, zero_u8x16, 15);
    uint8x16_t last_two_lanes_u8x16 = vsetq_lane_u8(0xFF, last_lane_u8x16, 14);
    uint8x16_t is_incomplete_u8x16 = vorrq_u8(vandq_u8(is_two_byte_lead_u8x16, last_lane_u8x16),
                                              vandq_u8(is_e1_lead_u8x16, last_two_lanes_u8x16));
    sz_u64_t incomplete_nibbles = sz_utf8_fold_neon_nibble_mask_(is_incomplete_u8x16);
    return incomplete_nibbles ? 48 + (sz_size_t)(sz_u64_ctz(incomplete_nibbles) / 4) : 64;
}

SZ_PUBLIC sz_size_t sz_utf8_case_fold_neon(sz_cptr_t source, sz_size_t source_length, sz_ptr_t target) {
    // The main loop processes a 64-byte logical superchunk - four 16-byte NEON registers - so
    // chunk decisions stay comparable to the Ice Lake kernel on the same input, which makes
    // three-way differential debugging (serial / icelake / neon) tractable.
    //
    // Store policy: the output buffer holds at least 3× the source length, and everything written
    // so far is at most 3× the bytes consumed so far, so whenever `source_length` ≥ 64 the
    // remaining headroom is ≥ 192 bytes - full 16-byte stores with partial advance are safe
    // anywhere inside the superchunk path, and no masked stores are needed.
    sz_ptr_t const target_start = target;

    uint8x16x4_t const lead_families_lut_u8x16x4 = vld1q_u8_x4(sz_utf8_fold_neon_lead_families_lut_);

    while (source_length >= 64) {
        uint8x16x4_t source_u8x16x4 = vld1q_u8_x4((sz_u8_t const *)source);

        // FAST PATH: pure ASCII - one OR-tree and one `vmaxvq_u8` decide for all 64 bytes,
        // before any classification masks are computed. Most common case for English text.
        uint8x16_t any_byte_u8x16 = vorrq_u8(vorrq_u8(source_u8x16x4.val[0], source_u8x16x4.val[1]),
                                             vorrq_u8(source_u8x16x4.val[2], source_u8x16x4.val[3]));
        if (vmaxvq_u8(any_byte_u8x16) < 0x80) {
            vst1q_u8((sz_u8_t *)target + 0, sz_utf8_fold_neon_ascii_(source_u8x16x4.val[0]));
            vst1q_u8((sz_u8_t *)target + 16, sz_utf8_fold_neon_ascii_(source_u8x16x4.val[1]));
            vst1q_u8((sz_u8_t *)target + 32, sz_utf8_fold_neon_ascii_(source_u8x16x4.val[2]));
            vst1q_u8((sz_u8_t *)target + 48, sz_utf8_fold_neon_ascii_(source_u8x16x4.val[3]));
            target += 64, source += 64, source_length -= 64;
            continue;
        }

        // Classify lead bytes once: four table lookups map every lead onto a family flag, and an
        // OR-reduction summarizes which families occur in this superchunk. The dispatch below
        // routes uniform chunks straight to their handler instead of probing per-script fast
        // paths sequentially, which degrades on mixed-script text.
        uint8x16_t lead_families_u8x16 = vorrq_u8(
            vorrq_u8(sz_utf8_fold_neon_classify_(source_u8x16x4.val[0], lead_families_lut_u8x16x4),
                     sz_utf8_fold_neon_classify_(source_u8x16x4.val[1], lead_families_lut_u8x16x4)),
            vorrq_u8(sz_utf8_fold_neon_classify_(source_u8x16x4.val[2], lead_families_lut_u8x16x4),
                     sz_utf8_fold_neon_classify_(source_u8x16x4.val[3], lead_families_lut_u8x16x4)));
        sz_u8_t lead_families = sz_utf8_fold_neon_reduce_or_u8_(lead_families_u8x16);

        if (!(lead_families & ~sz_utf8_fold_neon_lead_caseless_flag_)) {
            sz_size_t handled = sz_utf8_case_fold_neon_caseless_chunk_(source_u8x16x4, target);
            target += handled, source += handled, source_length -= handled;
            continue;
        }
        // Latin union chunk: nothing outside (latin | latin_extended | e1). Unlike Ice Lake,
        // pure Latin-1 subsets route here too - the union handler folds them correctly, and
        // skipping a dedicated Latin-1 path keeps the dispatch chain one test shorter.
        if (!(lead_families & ~(sz_utf8_fold_neon_lead_latin_flag_ | sz_utf8_fold_neon_lead_latin_extended_flag_ |
                                sz_utf8_fold_neon_lead_e1_flag_))) {
            sz_size_t handled = sz_utf8_case_fold_neon_latin_chunk_(source_u8x16x4, source, target);
            if (handled) {
                target += handled, source += handled, source_length -= handled;
                continue;
            }
        }
        // Dispatch slots for the remaining families, mirroring the Ice Lake subset-test chain:
        //  • Cyrillic, Greek, guarded (E2/EA/EF) chunks follow the same subset-test shape.
        // Until those land, every other non-caseless superchunk - and any superchunk whose
        // handler truncated to zero - advances by exactly ONE rune through the serial logic
        // below: byte-for-byte the serial reference output, just slower.

        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse(source, &rune, &rune_length);
        sz_rune_t folded_runes[3]; // Unicode case folding produces at most 3 runes
        sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
        for (sz_size_t rune_index = 0; rune_index != folded_count; ++rune_index)
            target += sz_rune_export(folded_runes[rune_index], (sz_u8_t *)target);
        // With ≥ 64 bytes remaining, valid UTF-8 guarantees the rune fits the buffer
        source += rune_length, source_length -= rune_length;
    }

    // 16..63 bytes left: single-register ASCII chunks, with remaining headroom ≥ 3× the
    // remaining source, full 16-byte stores stay within bounds. The first non-ASCII register
    // hands the whole tail to the serial epilogue.
    while (source_length >= 16) {
        uint8x16_t source_u8x16 = vld1q_u8((sz_u8_t const *)source);
        if (vmaxvq_u8(source_u8x16) >= 0x80) break;
        vst1q_u8((sz_u8_t *)target, sz_utf8_fold_neon_ascii_(source_u8x16));
        target += 16, source += 16, source_length -= 16;
    }

    // Serial epilogue: handles the sub-register tail and any trailing incomplete sequence
    // exactly like the scalar reference, keeping the two backends byte-for-byte identical.
    if (source_length) target += sz_utf8_case_fold_serial(source, source_length, target);
    return (sz_size_t)(target - target_start);
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

#endif // STRINGZILLA_UTF8_CASE_FOLD_NEON_H_
