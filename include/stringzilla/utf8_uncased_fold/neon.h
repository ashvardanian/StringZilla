/**
 *  @brief Arm NEON backend for UTF-8 case folding.
 *  @file include/stringzilla/utf8_uncased_fold/neon.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased_fold.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_FOLD_NEON_H_
#define STRINGZILLA_UTF8_UNCASED_FOLD_NEON_H_

#include "stringzilla/utf8_uncased_fold/serial.h"

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

/** @brief Folds ASCII A-Z down to a-z in one register, leaving every other byte unchanged. */
SZ_HELPER_AUTO uint8x16_t sz_utf8_fold_neon_ascii_(uint8x16_t source_u8x16) {
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
SZ_HELPER_INLINE sz_u64_t sz_utf8_fold_neon_nibble_mask_(uint8x16_t mask_u8x16) {
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(mask_u8x16), 4)), 0) &
           0x8888888888888888ull;
}

/** @brief OR-reduces all 16 byte lanes of one register into a single byte of accumulated flags. */
SZ_HELPER_INLINE sz_u8_t sz_utf8_fold_neon_reduce_or_u8_(uint8x16_t flags_u8x16) {
    uint8x8_t flags_u8x8 = vorr_u8(vget_low_u8(flags_u8x16), vget_high_u8(flags_u8x16));
    sz_u64_t flags_u64 = vget_lane_u64(vreinterpret_u64_u8(flags_u8x8), 0);
    flags_u64 |= flags_u64 >> 32, flags_u64 |= flags_u64 >> 16, flags_u64 |= flags_u64 >> 8;
    return (sz_u8_t)flags_u64;
}

/**
 *  @brief Maps every lead byte in one register onto its folding-family flag; non-leads map to zero.
 *      One `vqtbl4q_u8` covers the full 64-entry table - the NEON twin of Ice Lake's single VPERMB.
 */
SZ_HELPER_AUTO uint8x16_t sz_utf8_fold_neon_classify_(uint8x16_t source_u8x16, uint8x16x4_t lead_families_lut_u8x16x4) {
    uint8x16_t is_non_ascii_u8x16 = vcgeq_u8(source_u8x16, vdupq_n_u8(0x80));
    // Continuations are 10xxxxxx, i.e. exactly the [0x80, 0xBF] range - one wrap-around compare
    uint8x16_t is_continuation_u8x16 = vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0x80)), vdupq_n_u8(0x40));
    uint8x16_t is_lead_u8x16 = vbicq_u8(is_non_ascii_u8x16, is_continuation_u8x16);
    uint8x16_t lead_families_u8x16 = vqtbl4q_u8(lead_families_lut_u8x16x4, vandq_u8(source_u8x16, vdupq_n_u8(0x3F)));
    return vandq_u8(lead_families_u8x16, is_lead_u8x16);
}

/**
 *  @brief Per-lead well-formedness mirror of `sz_rune_decode`, computed branchlessly so every
 *      family handler can treat overlong, surrogate, truncated, and out-of-range leads as foreign and
 *      resync one byte at a time - byte-for-byte with the serial reference. The NEON twin of the
 *      Haswell `well_formed_lead_mask`: a lead is well-formed iff its declared continuations follow
 *      (the 2/3/4-byte-lead masks ANDed with the next 1/2/3 bytes all being continuations) AND it is
 *      not in the bad-special set - C0/C1, F5..FF, E0 with 2nd < 0xA0 (overlong), ED with 2nd >= 0xA0
 *      (surrogate), F0 with 2nd < 0x90 (overlong), F4 with 2nd >= 0x90 (> U+10FFFF). C0/C1 and F5..FF
 *      carry no width bit, so they never enter the width-keyed accept set and need no subtraction.
 *
 *      `next_register_u8x16` supplies the bytes following this register's last lanes; at the
 *      superchunk's final lane it is zero-filled, so a multi-byte lead whose continuations spill past
 *      the superchunk reads as malformed - coinciding exactly with the existing incomplete-sequence
 *      trim, so valid output is unchanged.
 *
 *  @return Per-byte mask (0xFF) set on every lead byte that does NOT begin a well-formed rune.
 */
SZ_HELPER_AUTO uint8x16_t sz_utf8_fold_neon_malformed_lead_(uint8x16_t source_u8x16, uint8x16_t next_register_u8x16) {
    uint8x16_t const continuation_low_u8x16 = vdupq_n_u8(0x80);
    uint8x16_t const continuation_span_u8x16 = vdupq_n_u8(0x40);

    uint8x16_t next1_u8x16 = vextq_u8(source_u8x16, next_register_u8x16, 1);
    uint8x16_t next2_u8x16 = vextq_u8(source_u8x16, next_register_u8x16, 2);
    uint8x16_t next3_u8x16 = vextq_u8(source_u8x16, next_register_u8x16, 3);

    uint8x16_t is_continuation_u8x16 = vcltq_u8(vsubq_u8(source_u8x16, continuation_low_u8x16),
                                                continuation_span_u8x16);
    uint8x16_t is_non_ascii_u8x16 = vcgeq_u8(source_u8x16, continuation_low_u8x16);
    uint8x16_t is_lead_u8x16 = vbicq_u8(is_non_ascii_u8x16, is_continuation_u8x16);
    uint8x16_t next1_is_continuation_u8x16 = vcltq_u8(vsubq_u8(next1_u8x16, continuation_low_u8x16),
                                                      continuation_span_u8x16);
    uint8x16_t next2_is_continuation_u8x16 = vcltq_u8(vsubq_u8(next2_u8x16, continuation_low_u8x16),
                                                      continuation_span_u8x16);
    uint8x16_t next3_is_continuation_u8x16 = vcltq_u8(vsubq_u8(next3_u8x16, continuation_low_u8x16),
                                                      continuation_span_u8x16);

    // Width leads: C2-DF (2-byte), E0-EF (3-byte), F0-F4 (4-byte). C0/C1 and F5-FF are excluded here
    // and therefore never reach the accept set - they stay malformed.
    uint8x16_t is_two_byte_lead_u8x16 = vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0xC2)), vdupq_n_u8(0x1E));
    uint8x16_t is_three_byte_lead_u8x16 = vceqq_u8(vandq_u8(source_u8x16, vdupq_n_u8(0xF0)), vdupq_n_u8(0xE0));
    uint8x16_t is_four_byte_lead_u8x16 = vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0xF0)), vdupq_n_u8(0x05));

    uint8x16_t two_byte_complete_u8x16 = vandq_u8(is_two_byte_lead_u8x16, next1_is_continuation_u8x16);
    uint8x16_t three_byte_complete_u8x16 = vandq_u8(is_three_byte_lead_u8x16,
                                                    vandq_u8(next1_is_continuation_u8x16, next2_is_continuation_u8x16));
    uint8x16_t four_byte_complete_u8x16 = vandq_u8(
        is_four_byte_lead_u8x16,
        vandq_u8(next1_is_continuation_u8x16, vandq_u8(next2_is_continuation_u8x16, next3_is_continuation_u8x16)));

    // Bad-special set keyed by the lead and its second byte, mirroring `sz_rune_decode`
    uint8x16_t e0_bad_u8x16 = vandq_u8(vceqq_u8(source_u8x16, vdupq_n_u8(0xE0)),
                                       vcltq_u8(next1_u8x16, vdupq_n_u8(0xA0)));
    uint8x16_t ed_bad_u8x16 = vandq_u8(vceqq_u8(source_u8x16, vdupq_n_u8(0xED)),
                                       vcgeq_u8(next1_u8x16, vdupq_n_u8(0xA0)));
    uint8x16_t f0_bad_u8x16 = vandq_u8(vceqq_u8(source_u8x16, vdupq_n_u8(0xF0)),
                                       vcltq_u8(next1_u8x16, vdupq_n_u8(0x90)));
    uint8x16_t f4_bad_u8x16 = vandq_u8(vceqq_u8(source_u8x16, vdupq_n_u8(0xF4)),
                                       vcgeq_u8(next1_u8x16, vdupq_n_u8(0x90)));
    uint8x16_t bad_special_u8x16 = vorrq_u8(vorrq_u8(e0_bad_u8x16, ed_bad_u8x16), vorrq_u8(f0_bad_u8x16, f4_bad_u8x16));

    uint8x16_t well_formed_u8x16 = vbicq_u8(
        vorrq_u8(two_byte_complete_u8x16, vorrq_u8(three_byte_complete_u8x16, four_byte_complete_u8x16)),
        bad_special_u8x16);
    return vbicq_u8(is_lead_u8x16, well_formed_u8x16);
}

/**
 *  @brief Folds a 64-byte superchunk containing only caseless multi-byte scripts mixed with ASCII.
 *      Folds ASCII A-Z in place and copies everything else, trimming an incomplete trailing sequence.
 *
 *  Mirrors `sz_utf8_uncased_fold_icelake_caseless_chunk_` at a fixed 64-byte chunk size: an
 *  incomplete sequence can only be a 2-byte lead in the last byte or a 3-byte lead in the last
 *  two bytes (4-byte leads carry the complex flag and never reach this handler), so only the
 *  last register's lead masks matter and one nibble-mask extraction covers both checks.
 *
 *  @return Bytes consumed; always 62..64, never zero - 62 bytes of any valid UTF-8 cover at
 *      least one complete sequence, so the superchunk cannot start with an incomplete one.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_neon_caseless_chunk_(uint8x16x4_t source_u8x16x4, sz_ptr_t target) {

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
 *  Mirrors `sz_utf8_uncased_fold_icelake_latin_chunk_` at a fixed 64-byte chunk size. Latin
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
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_neon_latin_chunk_(uint8x16x4_t source_u8x16x4, sz_cptr_t source,
                                                                sz_ptr_t target) {

    uint8x16x4_t const c4_deltas_lut_u8x16x4 = vld1q_u8_x4(sz_utf8_fold_c4_deltas_lut_);
    uint8x16x4_t const c5_deltas_lut_u8x16x4 = vld1q_u8_x4(sz_utf8_fold_c5_deltas_lut_);
    uint8x16x4_t const c6_deltas_lut_u8x16x4 = vld1q_u8_x4(sz_utf8_fold_c6_deltas_lut_);
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

        // Foreign leads stop the chunk: the handler triggers on family PRESENCE, so a German
        // chunk mixing Latin words with E2 quotation marks folds its Latin prefix here and lets
        // the guarded handler take the next chunk - measured a sixth faster on such mixes than
        // pure subset dispatch on the AVX2 port.
        uint8x16_t is_non_ascii_u8x16 = vcgeq_u8(source_u8x16, vdupq_n_u8(0x80));
        uint8x16_t is_lead_u8x16 = vbicq_u8(is_non_ascii_u8x16, is_continuation_u8x16);
        uint8x16_t is_family_lead_u8x16 = vorrq_u8(vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0xC2)), vdupq_n_u8(0x05)),
                                                   is_e1_u8x16);
        uint8x16_t is_foreign_lead_u8x16 = vbicq_u8(is_lead_u8x16, is_family_lead_u8x16);

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

        // Malformed leads (overlong, surrogate, truncated, out-of-range, C0/C1, F5..FF) are foreign
        // to this handler: treating them as stops truncates the fold before them so the per-rune
        // fallback copies one byte and resyncs. For valid text the malformed mask is empty, so the
        // handler behaves exactly as before.
        uint8x16_t malformed_lead_u8x16 = sz_utf8_fold_neon_malformed_lead_(source_u8x16, next_register_u8x16);
        uint8x16_t stop_u8x16 = vorrq_u8(
            vorrq_u8(irregular_extended_u8x16, foreign_e1_second_u8x16),
            vorrq_u8(vorrq_u8(irregular_additional_u8x16, is_foreign_lead_u8x16), malformed_lead_u8x16));
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

/**
 *  @brief Folds a 64-byte superchunk of basic Cyrillic mixed with ASCII.
 *
 *  All three uppercase sub-ranges are keyed by the second byte's high nibble, so one
 *  `vqtbl1q_u8` over a 16-entry table yields the offset (8 → +0x10, 9 → +0x20, A → −0x20)
 *  and the D0 → D1 lead rewrite is a masked +1 wherever the next byte falls in the two
 *  lead-changing ranges. Cyrillic Extended-A (D1 A0+) and any out-of-family lead stop the
 *  chunk; the consumed prefix always ends on a character boundary.
 *
 *  @return Bytes consumed and written, or zero if the first character needs another path.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_neon_cyrillic_chunk_(uint8x16x4_t source_u8x16x4, sz_cptr_t source,
                                                                   sz_ptr_t target) {
    static sz_u8_t const second_byte_offsets_lut_[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x20, 0xE0, 0, 0, 0, 0, 0};
    uint8x16_t const offsets_lut_u8x16 = vld1q_u8(second_byte_offsets_lut_);
    uint8x16_t const zero_u8x16 = vdupq_n_u8(0x00);

    uint8x16_t previous_is_d0_u8x16 = zero_u8x16;
    uint8x16_t stop_masks_u8x16[4];
    uint8x16_t any_stop_u8x16 = zero_u8x16;

    for (sz_size_t register_index = 0; register_index != 4; ++register_index) {
        uint8x16_t source_u8x16 = source_u8x16x4.val[register_index];
        uint8x16_t next_register_u8x16 = register_index != 3 ? source_u8x16x4.val[register_index + 1] : zero_u8x16;
        uint8x16_t next_byte_u8x16 = vextq_u8(source_u8x16, next_register_u8x16, 1);

        uint8x16_t is_d0_u8x16 = vceqq_u8(source_u8x16, vdupq_n_u8(0xD0));
        uint8x16_t is_d1_u8x16 = vceqq_u8(source_u8x16, vdupq_n_u8(0xD1));
        uint8x16_t is_continuation_u8x16 = vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0x80)), vdupq_n_u8(0x40));
        uint8x16_t is_lead_u8x16 = vbicq_u8(vcgeq_u8(source_u8x16, vdupq_n_u8(0x80)), is_continuation_u8x16);
        uint8x16_t is_foreign_lead_u8x16 = vbicq_u8(is_lead_u8x16, vorrq_u8(is_d0_u8x16, is_d1_u8x16));

        // Cyrillic Extended-A ('Ѡ'+, D1 A0+) folds by parity and stays on the serial path
        uint8x16_t is_extended_u8x16 = vandq_u8(is_d1_u8x16, vcgeq_u8(next_byte_u8x16, vdupq_n_u8(0xA0)));
        // Malformed leads are foreign to this handler - see the Latin handler for the rationale
        uint8x16_t malformed_lead_u8x16 = sz_utf8_fold_neon_malformed_lead_(source_u8x16, next_register_u8x16);
        uint8x16_t stop_u8x16 = vorrq_u8(vorrq_u8(is_foreign_lead_u8x16, is_extended_u8x16), malformed_lead_u8x16);
        stop_masks_u8x16[register_index] = stop_u8x16;
        any_stop_u8x16 = vorrq_u8(any_stop_u8x16, stop_u8x16);

        // Second-byte offsets: 'Ѐ'-'Џ' (D0 80-8F) → +0x10, 'А'-'П' (90-9F) → +0x20,
        // 'Р'-'Я' (A0-AF) → −0x20; lowercase B0+ maps to zero through the table.
        uint8x16_t after_d0_u8x16 = vextq_u8(previous_is_d0_u8x16, is_d0_u8x16, 15);
        uint8x16_t offsets_u8x16 = vandq_u8(vqtbl1q_u8(offsets_lut_u8x16, vshrq_n_u8(source_u8x16, 4)), after_d0_u8x16);
        uint8x16_t folded_u8x16 = vaddq_u8(sz_utf8_fold_neon_ascii_(source_u8x16), offsets_u8x16);

        // Lead rewrite D0 → D1 (+1) where the lowercase lives in the next block:
        // seconds 80-8F ('Ѐ'-'Џ' → 'ѐ'-'џ') and A0-AF ('Р'-'Я' → 'р'-'я')
        uint8x16_t needs_d1_u8x16 = vandq_u8(
            is_d0_u8x16, vorrq_u8(vcltq_u8(vsubq_u8(next_byte_u8x16, vdupq_n_u8(0x80)), vdupq_n_u8(0x10)),
                                  vcltq_u8(vsubq_u8(next_byte_u8x16, vdupq_n_u8(0xA0)), vdupq_n_u8(0x10))));
        folded_u8x16 = vaddq_u8(folded_u8x16, vandq_u8(needs_d1_u8x16, vdupq_n_u8(0x01)));

        vst1q_u8((sz_u8_t *)target + register_index * 16, folded_u8x16);
        previous_is_d0_u8x16 = is_d0_u8x16;
    }

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
        return first_flagged_position;
    }

    // Only a D0/D1 lead in the last byte can be incomplete - the 2-byte trim degenerates to one test
    sz_u8_t last_byte = (sz_u8_t)source[63];
    return (last_byte == 0xD0 || last_byte == 0xD1) ? 63 : 64;
}

/**
 *  @brief Folds a 64-byte superchunk of basic Greek mixed with ASCII.
 *
 *  'Α'-'Ο' (CE 91-9F) fold with +0x20 in place; 'Π'-'Ρ' and 'Σ'-'Ϋ' (CE A0-A1, A3-AB - 'Ϣ'-class
 *  A2 is unassigned) fold with −0x20 plus a CE → CF lead promotion (+1); final sigma 'ς' (CF 82)
 *  folds to 'σ' with +1. The fold-side exclusions are NARROWER than the finder's: accented
 *  uppercase (CE 84-90), 'ΰ' (CE B0, expands when folded), and the cross-block Greek symbols
 *  (CF 8F+: 'Ϗ' and the archaic letter pairs) stop the chunk for the serial path, as does any
 *  out-of-family lead. The accented lowercase vowels 'ό' 'ύ' 'ώ' (CF 8C-8E, identity-folding and
 *  very common in real Greek text) stay in the fast path - only CF 8F begins the symbol folds.
 *
 *  @return Bytes consumed and written, or zero if the first character needs another path.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_neon_greek_chunk_(uint8x16x4_t source_u8x16x4, sz_cptr_t source,
                                                                sz_ptr_t target) {
    uint8x16_t const zero_u8x16 = vdupq_n_u8(0x00);
    uint8x16_t previous_is_ce_u8x16 = zero_u8x16, previous_is_cf_u8x16 = zero_u8x16;
    uint8x16_t stop_masks_u8x16[4];
    uint8x16_t any_stop_u8x16 = zero_u8x16;

    for (sz_size_t register_index = 0; register_index != 4; ++register_index) {
        uint8x16_t source_u8x16 = source_u8x16x4.val[register_index];
        uint8x16_t next_register_u8x16 = register_index != 3 ? source_u8x16x4.val[register_index + 1] : zero_u8x16;
        uint8x16_t next_byte_u8x16 = vextq_u8(source_u8x16, next_register_u8x16, 1);

        uint8x16_t is_ce_u8x16 = vceqq_u8(source_u8x16, vdupq_n_u8(0xCE));
        uint8x16_t is_cf_u8x16 = vceqq_u8(source_u8x16, vdupq_n_u8(0xCF));
        uint8x16_t is_continuation_u8x16 = vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0x80)), vdupq_n_u8(0x40));
        uint8x16_t is_lead_u8x16 = vbicq_u8(vcgeq_u8(source_u8x16, vdupq_n_u8(0x80)), is_continuation_u8x16);
        uint8x16_t is_foreign_lead_u8x16 = vbicq_u8(is_lead_u8x16, vorrq_u8(is_ce_u8x16, is_cf_u8x16));

        // Exclusions, tested at the LEAD position so the walk-back lands on a boundary
        uint8x16_t ce_excluded_u8x16 = vandq_u8(is_ce_u8x16, vorrq_u8(vcltq_u8(next_byte_u8x16, vdupq_n_u8(0x91)),
                                                                      vceqq_u8(next_byte_u8x16, vdupq_n_u8(0xB0))));
        uint8x16_t cf_excluded_u8x16 = vandq_u8(is_cf_u8x16, vcgeq_u8(next_byte_u8x16, vdupq_n_u8(0x8F)));
        // Malformed leads are foreign to this handler - see the Latin handler for the rationale
        uint8x16_t malformed_lead_u8x16 = sz_utf8_fold_neon_malformed_lead_(source_u8x16, next_register_u8x16);
        uint8x16_t stop_u8x16 = vorrq_u8(
            vorrq_u8(is_foreign_lead_u8x16, vorrq_u8(ce_excluded_u8x16, cf_excluded_u8x16)), malformed_lead_u8x16);
        stop_masks_u8x16[register_index] = stop_u8x16;
        any_stop_u8x16 = vorrq_u8(any_stop_u8x16, stop_u8x16);

        uint8x16_t after_ce_u8x16 = vextq_u8(previous_is_ce_u8x16, is_ce_u8x16, 15);
        uint8x16_t after_cf_u8x16 = vextq_u8(previous_is_cf_u8x16, is_cf_u8x16, 15);

        // 'Α'-'Ο': +0x20 in place
        uint8x16_t is_basic_upper_u8x16 = vandq_u8(
            after_ce_u8x16, vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0x91)), vdupq_n_u8(0x0F)));
        // 'Π'-'Ρ' and 'Σ'-'Ϋ': −0x20 with the lead moving to CF
        uint8x16_t is_promoting_upper_u8x16 = vandq_u8(
            after_ce_u8x16, vorrq_u8(vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0xA0)), vdupq_n_u8(0x02)),
                                     vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0xA3)), vdupq_n_u8(0x09))));
        // Final sigma 'ς' → 'σ': +1
        uint8x16_t is_final_sigma_u8x16 = vandq_u8(after_cf_u8x16, vceqq_u8(source_u8x16, vdupq_n_u8(0x82)));

        uint8x16_t folded_u8x16 = sz_utf8_fold_neon_ascii_(source_u8x16);
        folded_u8x16 = vaddq_u8(folded_u8x16, vandq_u8(is_basic_upper_u8x16, vdupq_n_u8(0x20)));
        folded_u8x16 = vaddq_u8(folded_u8x16, vandq_u8(is_promoting_upper_u8x16, vdupq_n_u8(0xE0)));
        folded_u8x16 = vaddq_u8(folded_u8x16, vandq_u8(is_final_sigma_u8x16, vdupq_n_u8(0x01)));

        // CE → CF lead promotion (+1) keyed by the next byte's membership in the promoting ranges
        uint8x16_t promotes_lead_u8x16 = vandq_u8(
            is_ce_u8x16, vorrq_u8(vcltq_u8(vsubq_u8(next_byte_u8x16, vdupq_n_u8(0xA0)), vdupq_n_u8(0x02)),
                                  vcltq_u8(vsubq_u8(next_byte_u8x16, vdupq_n_u8(0xA3)), vdupq_n_u8(0x09))));
        folded_u8x16 = vaddq_u8(folded_u8x16, vandq_u8(promotes_lead_u8x16, vdupq_n_u8(0x01)));

        vst1q_u8((sz_u8_t *)target + register_index * 16, folded_u8x16);
        previous_is_ce_u8x16 = is_ce_u8x16, previous_is_cf_u8x16 = is_cf_u8x16;
    }

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
        return first_flagged_position;
    }

    sz_u8_t last_byte = (sz_u8_t)source[63];
    return (last_byte == 0xCE || last_byte == 0xCF) ? 63 : 64;
}

/**
 *  @brief Folds a 64-byte superchunk of Armenian (D4-D6 leads) mixed with ASCII.
 *
 *  Armenian uppercase spans two lead bytes and folds into three target blocks, reusing the exact
 *  math verified in the NEON finder's `sz_utf8_uncased_search_neon_armenian_fold_u8x16x2_`:
 *  - D4 B1-BF: 'Ա'-'Ձ' → D5 A1-AF 'ա'-'ձ' (second −0x10, lead D4 → D5)
 *  - D5 80-8F: 'Ղ'-'Տ' → D5 B0-BF 'ղ'-'տ' (second +0x30, lead unchanged)
 *  - D5 90-96: 'Ր'-'Ֆ' → D6 80-86 'ր'-'ֆ' (second −0x10, lead D5 → D6)
 *  Both lead rewrites are a +1 increment, decided at the LEAD position from its own next byte; the
 *  three second-byte offsets fall on disjoint lanes, so they merge into one masked add. The −0x10
 *  is added as +0xF0 (two's-complement wrap), exactly as the finder does.
 *
 *  Everything else in D4-D6 either passes through as identity (D5 97-BF reserved/punctuation and
 *  lowercase, all of D6 except the ligature, Hebrew D6 90-BF) or stops the chunk for the serial
 *  path: D4 80-B0 is the Cyrillic Supplement (U+0500-052F parity folds plus reserved U+0530, none
 *  of which this Armenian handler folds), and 'և' (U+0587, D6 87) folds to two runes "եւ" - the
 *  only length-changing fold in the block, so it cannot fold in place. Any out-of-family complex
 *  lead (D2-D3 Cyrillic Extended-C/Supplement, C7-CD, F0+) also stops. A stop at byte 0 returns
 *  zero, routing the leading non-Armenian rune to the serial fallback.
 *
 *  @return Bytes consumed and written, or zero if the first character needs another path.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_neon_armenian_chunk_(uint8x16x4_t source_u8x16x4, sz_cptr_t source,
                                                                   sz_ptr_t target) {
    uint8x16_t const zero_u8x16 = vdupq_n_u8(0x00);
    uint8x16_t previous_is_d4_u8x16 = zero_u8x16, previous_is_d5_u8x16 = zero_u8x16;
    uint8x16_t stop_masks_u8x16[4];
    uint8x16_t any_stop_u8x16 = zero_u8x16;

    for (sz_size_t register_index = 0; register_index != 4; ++register_index) {
        uint8x16_t source_u8x16 = source_u8x16x4.val[register_index];
        uint8x16_t next_register_u8x16 = register_index != 3 ? source_u8x16x4.val[register_index + 1] : zero_u8x16;
        uint8x16_t next_byte_u8x16 = vextq_u8(source_u8x16, next_register_u8x16, 1);

        uint8x16_t is_d4_u8x16 = vceqq_u8(source_u8x16, vdupq_n_u8(0xD4));
        uint8x16_t is_d5_u8x16 = vceqq_u8(source_u8x16, vdupq_n_u8(0xD5));
        uint8x16_t is_d6_u8x16 = vceqq_u8(source_u8x16, vdupq_n_u8(0xD6));
        uint8x16_t is_continuation_u8x16 = vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0x80)), vdupq_n_u8(0x40));
        uint8x16_t is_lead_u8x16 = vbicq_u8(vcgeq_u8(source_u8x16, vdupq_n_u8(0x80)), is_continuation_u8x16);
        uint8x16_t is_armenian_lead_u8x16 = vorrq_u8(vorrq_u8(is_d4_u8x16, is_d5_u8x16), is_d6_u8x16);
        uint8x16_t is_foreign_lead_u8x16 = vbicq_u8(is_lead_u8x16, is_armenian_lead_u8x16);

        // Stops tested at the LEAD position so the walk-back lands on a character boundary:
        // D4 with a Cyrillic-Supplement/reserved second byte (≤ B0) and the 'և' ligature (D6 87).
        uint8x16_t is_d4_stop_u8x16 = vandq_u8(is_d4_u8x16, vcltq_u8(next_byte_u8x16, vdupq_n_u8(0xB1)));
        uint8x16_t is_ligature_stop_u8x16 = vandq_u8(is_d6_u8x16, vceqq_u8(next_byte_u8x16, vdupq_n_u8(0x87)));
        // Malformed leads are foreign to this handler - see the Latin handler for the rationale
        uint8x16_t malformed_lead_u8x16 = sz_utf8_fold_neon_malformed_lead_(source_u8x16, next_register_u8x16);
        uint8x16_t stop_u8x16 = vorrq_u8(
            vorrq_u8(is_foreign_lead_u8x16, vorrq_u8(is_d4_stop_u8x16, is_ligature_stop_u8x16)), malformed_lead_u8x16);
        stop_masks_u8x16[register_index] = stop_u8x16;
        any_stop_u8x16 = vorrq_u8(any_stop_u8x16, stop_u8x16);

        uint8x16_t after_d4_u8x16 = vextq_u8(previous_is_d4_u8x16, is_d4_u8x16, 15);
        uint8x16_t after_d5_u8x16 = vextq_u8(previous_is_d5_u8x16, is_d5_u8x16, 15);

        // Second-byte offsets on disjoint lanes: D4 B1-BF → −0x10 (+0xF0 wrap), D5 80-8F → +0x30,
        // D5 90-96 → −0x10. The D4 range only needs a lower-bound check - valid continuations ≤ BF.
        uint8x16_t is_d4_armenian_second_u8x16 = vandq_u8(after_d4_u8x16, vcgeq_u8(source_u8x16, vdupq_n_u8(0xB1)));
        uint8x16_t is_d5_plus30_second_u8x16 = vandq_u8(
            after_d5_u8x16, vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0x80)), vdupq_n_u8(0x10)));
        uint8x16_t is_d5_minus10_second_u8x16 = vandq_u8(
            after_d5_u8x16, vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0x90)), vdupq_n_u8(0x07)));

        uint8x16_t offsets_u8x16 = vandq_u8(vorrq_u8(is_d4_armenian_second_u8x16, is_d5_minus10_second_u8x16),
                                            vdupq_n_u8(0xF0));
        offsets_u8x16 = vorrq_u8(offsets_u8x16, vandq_u8(is_d5_plus30_second_u8x16, vdupq_n_u8(0x30)));
        uint8x16_t folded_u8x16 = vaddq_u8(sz_utf8_fold_neon_ascii_(source_u8x16), offsets_u8x16);

        // Lead +1 rewrites decided from the lead's own next byte: D4 → D5 where next is B1-BF,
        // D5 → D6 where next is 90-96. D5 80-8F keeps its lead.
        uint8x16_t promotes_d4_u8x16 = vandq_u8(is_d4_u8x16, vcgeq_u8(next_byte_u8x16, vdupq_n_u8(0xB1)));
        uint8x16_t promotes_d5_u8x16 = vandq_u8(
            is_d5_u8x16, vcltq_u8(vsubq_u8(next_byte_u8x16, vdupq_n_u8(0x90)), vdupq_n_u8(0x07)));
        folded_u8x16 = vaddq_u8(folded_u8x16,
                                vandq_u8(vorrq_u8(promotes_d4_u8x16, promotes_d5_u8x16), vdupq_n_u8(0x01)));

        vst1q_u8((sz_u8_t *)target + register_index * 16, folded_u8x16);
        previous_is_d4_u8x16 = is_d4_u8x16, previous_is_d5_u8x16 = is_d5_u8x16;
    }

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
        return first_flagged_position;
    }

    // Only a D4/D5/D6 lead in the last byte can be incomplete - the 2-byte trim is one test
    sz_u8_t last_byte = (sz_u8_t)source[63];
    return (last_byte == 0xD4 || last_byte == 0xD5 || last_byte == 0xD6) ? 63 : 64;
}

/**
 *  @brief Folds a 64-byte superchunk of Georgian (E1 82/83 content) mixed with ASCII.
 *
 *  Ordered AFTER the Latin handler: that one folds E1 B8-BB (Latin Extended Additional) and stops
 *  at E1 82/83; this one picks up Georgian and stops at E1 BC-BF (Greek Extended) and every other
 *  non-Georgian E1 second byte. Ports the verified Ice Lake "3.1. Georgian fast path" - a uniform
 *  3-byte → 3-byte fold with a lead-byte rewrite:
 *  - E1 82 A0-BF: 'Ⴀ'-'Ⴟ' (U+10A0-10BF) → E2 B4 80-9F (lead E1→E2, second 82→B4, third −0x20)
 *  - E1 83 80-85: 'Ⴠ'-'Ⴥ' (U+10C0-10C5) → E2 B4 A0-A5 (lead E1→E2, second 83→B4, third +0x20)
 *  - E1 83 87:    'Ⴧ' (U+10C7)           → E2 B4 A7      (same rewrite, third +0x20)
 *  - E1 83 8D:    'Ⴭ' (U+10CD)           → E2 B4 AD      (same rewrite, third +0x20)
 *  Lowercase Mkhedruli and reserved Georgian (E1 82 80-9F, E1 83 86/88-8C/8E-BF) copy unchanged.
 *
 *  The lead/second rewrites must fire only for the UPPERCASE subset, which is decided by the THIRD
 *  byte - so the lead position reads two bytes forward through `vextq_u8(.., .., 2)` and the
 *  uppercase flag is carried forward one lane to the second-byte rewrite and a second lane to the
 *  third-byte offset. The two third-byte offsets land on disjoint sequences, so −0x20 (added as
 *  +0xE0 wrap) and +0x20 merge into one add. Every fold here is length-preserving, so full-register
 *  stores are exact for the consumed prefix.
 *
 *  @return Bytes consumed and written, or zero if the first character needs another path.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_neon_georgian_chunk_(uint8x16x4_t source_u8x16x4, sz_cptr_t source,
                                                                   sz_ptr_t target) {
    uint8x16_t const zero_u8x16 = vdupq_n_u8(0x00);
    uint8x16_t previous_is_82_upper_lead_u8x16 = zero_u8x16, previous_is_83_upper_lead_u8x16 = zero_u8x16;
    uint8x16_t previous_is_82_upper_second_u8x16 = zero_u8x16, previous_is_83_upper_second_u8x16 = zero_u8x16;
    uint8x16_t stop_masks_u8x16[4];
    uint8x16_t any_stop_u8x16 = zero_u8x16;

    for (sz_size_t register_index = 0; register_index != 4; ++register_index) {
        uint8x16_t source_u8x16 = source_u8x16x4.val[register_index];
        uint8x16_t next_register_u8x16 = register_index != 3 ? source_u8x16x4.val[register_index + 1] : zero_u8x16;
        // The lead position needs both its second (byte+1) and third (byte+2) bytes to classify a
        // Georgian sequence, so two shifted views are built across the register boundary. A lead in
        // the last two lanes is an incomplete sequence excluded by the trailing trim below.
        uint8x16_t next_byte_u8x16 = vextq_u8(source_u8x16, next_register_u8x16, 1);
        uint8x16_t next_next_byte_u8x16 = vextq_u8(source_u8x16, next_register_u8x16, 2);

        uint8x16_t is_e1_u8x16 = vceqq_u8(source_u8x16, vdupq_n_u8(0xE1));
        uint8x16_t is_continuation_u8x16 = vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0x80)), vdupq_n_u8(0x40));
        uint8x16_t is_lead_u8x16 = vbicq_u8(vcgeq_u8(source_u8x16, vdupq_n_u8(0x80)), is_continuation_u8x16);
        uint8x16_t is_foreign_lead_u8x16 = vbicq_u8(is_lead_u8x16, is_e1_u8x16);

        // Georgian second bytes are 82 or 83; any other E1 second byte stops the chunk at the lead
        uint8x16_t is_82_lead_u8x16 = vandq_u8(is_e1_u8x16, vceqq_u8(next_byte_u8x16, vdupq_n_u8(0x82)));
        uint8x16_t is_83_lead_u8x16 = vandq_u8(is_e1_u8x16, vceqq_u8(next_byte_u8x16, vdupq_n_u8(0x83)));
        uint8x16_t is_georgian_lead_u8x16 = vorrq_u8(is_82_lead_u8x16, is_83_lead_u8x16);
        uint8x16_t is_foreign_e1_u8x16 = vbicq_u8(is_e1_u8x16, is_georgian_lead_u8x16);
        // Malformed leads are foreign to this handler - see the Latin handler for the rationale
        uint8x16_t malformed_lead_u8x16 = sz_utf8_fold_neon_malformed_lead_(source_u8x16, next_register_u8x16);
        uint8x16_t stop_u8x16 = vorrq_u8(vorrq_u8(is_foreign_lead_u8x16, is_foreign_e1_u8x16), malformed_lead_u8x16);
        stop_masks_u8x16[register_index] = stop_u8x16;
        any_stop_u8x16 = vorrq_u8(any_stop_u8x16, stop_u8x16);

        // Uppercase Georgian is keyed by the third byte: E1 82 third ≥ A0, E1 83 third in 80-85/87/8D
        uint8x16_t is_82_upper_lead_u8x16 = vandq_u8(is_82_lead_u8x16,
                                                     vcgeq_u8(next_next_byte_u8x16, vdupq_n_u8(0xA0)));
        uint8x16_t is_83_third_range_u8x16 = vcltq_u8(vsubq_u8(next_next_byte_u8x16, vdupq_n_u8(0x80)),
                                                      vdupq_n_u8(0x06));
        uint8x16_t is_83_third_extra_u8x16 = vorrq_u8(vceqq_u8(next_next_byte_u8x16, vdupq_n_u8(0x87)),
                                                      vceqq_u8(next_next_byte_u8x16, vdupq_n_u8(0x8D)));
        uint8x16_t is_83_upper_lead_u8x16 = vandq_u8(is_83_lead_u8x16,
                                                     vorrq_u8(is_83_third_range_u8x16, is_83_third_extra_u8x16));

        // Carry the uppercase flag forward: lane +1 for the second-byte rewrite, lane +2 for the third
        uint8x16_t is_82_upper_second_u8x16 = vextq_u8(previous_is_82_upper_lead_u8x16, is_82_upper_lead_u8x16, 15);
        uint8x16_t is_83_upper_second_u8x16 = vextq_u8(previous_is_83_upper_lead_u8x16, is_83_upper_lead_u8x16, 15);
        uint8x16_t is_82_upper_third_u8x16 = vextq_u8(previous_is_82_upper_second_u8x16, is_82_upper_second_u8x16, 15);
        uint8x16_t is_83_upper_third_u8x16 = vextq_u8(previous_is_83_upper_second_u8x16, is_83_upper_second_u8x16, 15);

        uint8x16_t is_upper_lead_u8x16 = vorrq_u8(is_82_upper_lead_u8x16, is_83_upper_lead_u8x16);
        uint8x16_t is_upper_second_u8x16 = vorrq_u8(is_82_upper_second_u8x16, is_83_upper_second_u8x16);

        // Rewrites: lead E1 → E2, second 82/83 → B4, third −0x20 (E1 82, added as +0xE0) or +0x20 (E1 83)
        uint8x16_t folded_u8x16 = sz_utf8_fold_neon_ascii_(source_u8x16);
        folded_u8x16 = vbslq_u8(is_upper_lead_u8x16, vdupq_n_u8(0xE2), folded_u8x16);
        folded_u8x16 = vbslq_u8(is_upper_second_u8x16, vdupq_n_u8(0xB4), folded_u8x16);
        uint8x16_t third_offsets_u8x16 = vorrq_u8(vandq_u8(is_82_upper_third_u8x16, vdupq_n_u8(0xE0)),
                                                  vandq_u8(is_83_upper_third_u8x16, vdupq_n_u8(0x20)));
        folded_u8x16 = vaddq_u8(folded_u8x16, third_offsets_u8x16);

        vst1q_u8((sz_u8_t *)target + register_index * 16, folded_u8x16);
        previous_is_82_upper_lead_u8x16 = is_82_upper_lead_u8x16;
        previous_is_83_upper_lead_u8x16 = is_83_upper_lead_u8x16;
        previous_is_82_upper_second_u8x16 = is_82_upper_second_u8x16;
        previous_is_83_upper_second_u8x16 = is_83_upper_second_u8x16;
    }

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
        return first_flagged_position;
    }

    // Don't split a trailing E1 sequence: a lead in the last two lanes (bytes 62-63) is incomplete
    sz_u8_t second_to_last_byte = (sz_u8_t)source[62], last_byte = (sz_u8_t)source[63];
    if (second_to_last_byte == 0xE1) return 62;
    if (last_byte == 0xE1) return 63;
    return 64;
}

/**
 *  @brief Folds a 64-byte superchunk of caseless scripts mixed with SAFE guarded punctuation.
 *      Folds ASCII and copies the rest. The guarded leads are case-aware only for some second
 *      bytes: E2 is safe for 80-83 (General Punctuation quotes and dashes), EA for everything
 *      except 99-9F and AD-AE (Cyrillic Extended-B and Cherokee Supplement); E1 and EF always
 *      stop, as does any 2-byte lead outside the caseless D7-DF run and any 4-byte lead.
 *
 *  @return Bytes consumed and written, or zero if the first character needs another path.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_neon_guarded_chunk_(uint8x16x4_t source_u8x16x4, sz_cptr_t source,
                                                                  sz_ptr_t target) {
    uint8x16_t const zero_u8x16 = vdupq_n_u8(0x00);
    uint8x16_t stop_masks_u8x16[4];
    uint8x16_t any_stop_u8x16 = zero_u8x16;

    for (sz_size_t register_index = 0; register_index != 4; ++register_index) {
        uint8x16_t source_u8x16 = source_u8x16x4.val[register_index];
        uint8x16_t next_register_u8x16 = register_index != 3 ? source_u8x16x4.val[register_index + 1] : zero_u8x16;
        uint8x16_t next_byte_u8x16 = vextq_u8(source_u8x16, next_register_u8x16, 1);

        uint8x16_t is_caseless_two_byte_u8x16 = vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0xD7)), vdupq_n_u8(0x09));
        uint8x16_t is_three_byte_u8x16 = vceqq_u8(vandq_u8(source_u8x16, vdupq_n_u8(0xF0)), vdupq_n_u8(0xE0));
        uint8x16_t is_continuation_u8x16 = vcltq_u8(vsubq_u8(source_u8x16, vdupq_n_u8(0x80)), vdupq_n_u8(0x40));
        uint8x16_t is_lead_u8x16 = vbicq_u8(vcgeq_u8(source_u8x16, vdupq_n_u8(0x80)), is_continuation_u8x16);
        uint8x16_t is_foreign_lead_u8x16 = vbicq_u8(is_lead_u8x16,
                                                    vorrq_u8(is_caseless_two_byte_u8x16, is_three_byte_u8x16));

        // E1 and EF are always case-aware; E2 and EA only for some second bytes
        uint8x16_t is_e2_u8x16 = vceqq_u8(source_u8x16, vdupq_n_u8(0xE2));
        uint8x16_t is_ea_u8x16 = vceqq_u8(source_u8x16, vdupq_n_u8(0xEA));
        uint8x16_t stop_u8x16 = vorrq_u8(is_foreign_lead_u8x16, vorrq_u8(vceqq_u8(source_u8x16, vdupq_n_u8(0xE1)),
                                                                         vceqq_u8(source_u8x16, vdupq_n_u8(0xEF))));
        stop_u8x16 = vorrq_u8(
            stop_u8x16, vbicq_u8(is_e2_u8x16, vcltq_u8(vsubq_u8(next_byte_u8x16, vdupq_n_u8(0x80)), vdupq_n_u8(0x04))));
        stop_u8x16 = vorrq_u8(
            stop_u8x16,
            vandq_u8(is_ea_u8x16, vorrq_u8(vcltq_u8(vsubq_u8(next_byte_u8x16, vdupq_n_u8(0x99)), vdupq_n_u8(0x07)),
                                           vcltq_u8(vsubq_u8(next_byte_u8x16, vdupq_n_u8(0xAD)), vdupq_n_u8(0x02)))));
        // Malformed leads are foreign to this handler - see the Latin handler for the rationale
        stop_u8x16 = vorrq_u8(stop_u8x16, sz_utf8_fold_neon_malformed_lead_(source_u8x16, next_register_u8x16));
        stop_masks_u8x16[register_index] = stop_u8x16;
        any_stop_u8x16 = vorrq_u8(any_stop_u8x16, stop_u8x16);

        vst1q_u8((sz_u8_t *)target + register_index * 16, sz_utf8_fold_neon_ascii_(source_u8x16));
    }

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
        return first_flagged_position;
    }

    // Trailing trim mirrors the caseless handler: 2-byte lead in the last byte, 3-byte in the last two
    sz_u8_t second_to_last_byte = (sz_u8_t)source[62], last_byte = (sz_u8_t)source[63];
    if ((second_to_last_byte & 0xF0) == 0xE0) return 62;
    if ((last_byte & 0xF0) == 0xE0 || (sz_u8_t)(last_byte - 0xD7) < 0x09) return 63;
    return 64;
}

SZ_API_COMPTIME sz_size_t sz_utf8_uncased_fold_neon(sz_cptr_t source, sz_size_t source_length, sz_ptr_t target) {
    // The main loop processes a 64-byte logical superchunk - four 16-byte NEON registers - so
    // chunk decisions stay comparable to the Ice Lake kernel on the same input, which makes
    // three-way differential debugging (serial / icelake / neon) tractable.
    //
    // Store policy: the output buffer holds at least 3× the source length, and everything written
    // so far is at most 3× the bytes consumed so far, so whenever `source_length` ≥ 64 the
    // remaining headroom is ≥ 192 bytes - full 16-byte stores with partial advance are safe
    // anywhere inside the superchunk path, and no masked stores are needed.
    sz_ptr_t const target_start = target;

    uint8x16x4_t const lead_families_lut_u8x16x4 = vld1q_u8_x4(sz_utf8_fold_lead_families_lut_);

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

        if (!(lead_families & ~sz_utf8_fold_lead_caseless_flag_)) {
            sz_size_t handled = sz_utf8_uncased_fold_neon_caseless_chunk_(source_u8x16x4, target);
            target += handled, source += handled, source_length -= handled;
            continue;
        }
        // Handlers trigger on family PRESENCE in priority order and truncate at the first
        // out-of-family lead (validated on the AVX2 port: a sixth faster on Russian than pure
        // subset dispatch, because quotes and dashes no longer poison whole superchunks into
        // one-rune serial steps). A handler that cannot fold the first character returns zero
        // and the chunk falls through to the next family - or to the serial rune below.
        if (lead_families & (sz_utf8_fold_lead_latin_flag_ | sz_utf8_fold_lead_latin_extended_flag_ |
                             sz_utf8_fold_lead_e1_flag_)) {
            sz_size_t handled = sz_utf8_uncased_fold_neon_latin_chunk_(source_u8x16x4, source, target);
            if (handled) {
                target += handled, source += handled, source_length -= handled;
                continue;
            }
        }
        // Georgian shares the E1 lead with Latin Extended Additional, so it runs second: the Latin
        // handler folds E1 B8-BB and returns zero on a leading E1 82/83, and this picks those up.
        if (lead_families & sz_utf8_fold_lead_e1_flag_) {
            sz_size_t handled = sz_utf8_uncased_fold_neon_georgian_chunk_(source_u8x16x4, source, target);
            if (handled) {
                target += handled, source += handled, source_length -= handled;
                continue;
            }
        }
        if (lead_families & sz_utf8_fold_lead_cyrillic_flag_) {
            sz_size_t handled = sz_utf8_uncased_fold_neon_cyrillic_chunk_(source_u8x16x4, source, target);
            if (handled) {
                target += handled, source += handled, source_length -= handled;
                continue;
            }
        }
        if (lead_families & sz_utf8_fold_lead_greek_flag_) {
            sz_size_t handled = sz_utf8_uncased_fold_neon_greek_chunk_(source_u8x16x4, source, target);
            if (handled) {
                target += handled, source += handled, source_length -= handled;
                continue;
            }
        }
        if (lead_families & sz_utf8_fold_lead_guarded_flag_) {
            sz_size_t handled = sz_utf8_uncased_fold_neon_guarded_chunk_(source_u8x16x4, source, target);
            if (handled) {
                target += handled, source += handled, source_length -= handled;
                continue;
            }
        }
        // Armenian (D4-D6) plus the Cyrillic Supplement that shares the D4 lead both fall in the
        // complex family; this handler folds them and truncates at any non-Armenian complex lead.
        if (lead_families & sz_utf8_fold_lead_complex_flag_) {
            sz_size_t handled = sz_utf8_uncased_fold_neon_armenian_chunk_(source_u8x16x4, source, target);
            if (handled) {
                target += handled, source += handled, source_length -= handled;
                continue;
            }
        }
        // Complex leads (4-byte emoji, rare 2-byte blocks) and zero-returning handlers advance
        // by exactly ONE rune through the serial logic below: byte-for-byte the serial
        // reference output, just slower. Georgian, fullwidth, and supplementary copy handlers
        // are candidates for the ARM-hardware tuning round if profiles justify them.
        //
        // `sz_rune_decode` is the parse-side authority: a malformed lead - the only reason
        // the vector handlers decline a multi-byte sequence after the malformed-mask fix - copies
        // one byte through unchanged and resyncs, byte-for-byte with the serial reference.
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_decode(source, source + source_length, &rune);
        if (rune_length == sz_rune_invalid_k) {
            *(sz_u8_t *)target = *(sz_u8_t const *)source; // Maximal-subpart resync: copy one byte, advance one
            target += 1, source += 1, source_length -= 1;
            continue;
        }
        sz_rune_t folded_runes[3]; // Unicode case folding produces at most 3 runes
        sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
        for (sz_size_t rune_index = 0; rune_index != folded_count; ++rune_index)
            target += sz_rune_encode(folded_runes[rune_index], (sz_u8_t *)target);
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
    if (source_length) target += sz_utf8_uncased_fold_serial(source, source_length, target);
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

#endif // STRINGZILLA_UTF8_UNCASED_FOLD_NEON_H_
