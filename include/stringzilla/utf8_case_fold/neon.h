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
SZ_INTERNAL uint8x16_t sz_utf8_fold_neon_classify_(uint8x16_t source_u8x16,
                                                   uint8x16x4_t lead_families_lut_u8x16x4) {
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
        uint8x16_t lead_families_u8x16 =
            vorrq_u8(vorrq_u8(sz_utf8_fold_neon_classify_(source_u8x16x4.val[0], lead_families_lut_u8x16x4),
                              sz_utf8_fold_neon_classify_(source_u8x16x4.val[1], lead_families_lut_u8x16x4)),
                     vorrq_u8(sz_utf8_fold_neon_classify_(source_u8x16x4.val[2], lead_families_lut_u8x16x4),
                              sz_utf8_fold_neon_classify_(source_u8x16x4.val[3], lead_families_lut_u8x16x4)));
        sz_u8_t lead_families = sz_utf8_fold_neon_reduce_or_u8_(lead_families_u8x16);

        if (!(lead_families & ~sz_utf8_fold_neon_lead_caseless_flag_)) {
            sz_size_t handled = sz_utf8_case_fold_neon_caseless_chunk_(source_u8x16x4, target);
            target += handled, source += handled, source_length -= handled;
            continue;
        }
        // Dispatch slots for the remaining families, mirroring the Ice Lake subset-test chain:
        //  • Latin union chunk: (latin_extended | e1) present, nothing outside
        //    (latin | latin_extended | e1) - delta-LUT folding lands here.
        //  • Cyrillic, Greek, guarded (E2/EA/EF) chunks follow the same subset-test shape.
        // Until those land, every non-caseless superchunk advances by exactly ONE rune through
        // the serial logic below - byte-for-byte the serial reference output, just slower.

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
