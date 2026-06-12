/**
 *  @brief Haswell (AVX2) backend for UTF-8 case folding.
 *  @file include/stringzilla/utf8_case_fold/haswell.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_case_fold.h
 */
#ifndef STRINGZILLA_UTF8_CASE_FOLD_HASWELL_H_
#define STRINGZILLA_UTF8_CASE_FOLD_HASWELL_H_

#include "stringzilla/utf8_case_fold/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "bmi", "bmi2")
#endif

/**
 *  Bit flags describing which UTF-8 lead-byte families occur in a chunk, mirroring the Ice Lake
 *  `sz_utf8_fold_lead_family_t_` semantics. Each family shares one folding strategy, so the union
 *  of flags picks the chunk handler in a single dispatch - instead of sequentially probing
 *  per-script fast paths, which degrades on mixed-script text. Without `VPERMB`, the flags are
 *  produced by a compare tree: one unsigned range compare per family, reduced to a 32-bit
 *  `VPMOVMSKB` integer - `movemask` is single-uop on port 0, so eight of them beat any
 *  multi-shuffle emulation of a 64-entry byte LUT.
 */
enum sz_utf8_fold_haswell_lead_family_t_ {
    sz_utf8_fold_haswell_lead_caseless_flag_ = 1 << 0,       // D7-DF, E0, E3-E9, EB-EE: scripts with no case
    sz_utf8_fold_haswell_lead_latin_flag_ = 1 << 1,          // C2-C3: Latin-1 Supplement
    sz_utf8_fold_haswell_lead_latin_extended_flag_ = 1 << 2, // C4-C6: Latin Extended-A and Ext-B with +1 pairs
    sz_utf8_fold_haswell_lead_cyrillic_flag_ = 1 << 3,       // D0-D1: basic Cyrillic
    sz_utf8_fold_haswell_lead_greek_flag_ = 1 << 4,          // CE-CF: basic Greek
    sz_utf8_fold_haswell_lead_e1_flag_ = 1 << 5,             // E1: Latin Ext Additional, Georgian, Greek Extended
    sz_utf8_fold_haswell_lead_guarded_flag_ = 1 << 6,        // E2, EA, EF: case-awareness depends on the second byte
    sz_utf8_fold_haswell_lead_complex_flag_ = 1 << 7,        // C0-C1, C7-CD, D2-D6, F0-FF: decode or serial paths
};

/**
 *  @brief Detects bytes in the unsigned range [range_start, range_start + range_length).
 *      AVX2 has no unsigned byte compares, so `(x − start) ≤ limit` is realized as
 *      `min_epu8(x − start, limit) == x − start`: the wrap-around subtraction maps the range
 *      onto [0, limit] and `VPMINUB` + `VPCMPEQB` realize the unsigned `≤` in two single-uop
 *      instructions - cheaper and clearer than the sign-flip `VPXOR` + `VPCMPGTB` alternative.
 */
SZ_INTERNAL __m256i sz_haswell_in_byte_range_(__m256i values_ymm, sz_u8_t range_start, sz_u8_t range_length) {
    __m256i offsets_ymm = _mm256_sub_epi8(values_ymm, _mm256_set1_epi8((char)range_start));
    return _mm256_cmpeq_epi8(_mm256_min_epu8(offsets_ymm, _mm256_set1_epi8((char)(range_length - 1))), offsets_ymm);
}

/** @brief Folds ASCII A-Z to a-z across the whole vector via a masked +0x20 (no `VPBLENDVB` needed). */
SZ_INTERNAL __m256i sz_haswell_fold_ascii_(__m256i source_ymm) {
    __m256i is_ascii_upper_ymm = sz_haswell_in_byte_range_(source_ymm, 'A', 26);
    return _mm256_add_epi8(source_ymm, _mm256_and_si256(is_ascii_upper_ymm, _mm256_set1_epi8(0x20)));
}

/**
 *  @brief Shifts the 32 source bytes right by @p byte_offset lanes, so lane `i` holds byte `i − offset`.
 *      Positions before the chunk receive zeros - safe, because the main loop always advances by whole
 *      characters, so a chunk never starts with a continuation byte that would need its true predecessor.
 *      AVX2 `VPALIGNR` works per 128-bit lane, so a `VPERM2I128` first materializes the cross-lane carry.
 */
SZ_INTERNAL __m256i sz_haswell_previous_bytes_(__m256i source_ymm, int byte_offset) {
    __m256i carry_ymm = _mm256_permute2x128_si256(source_ymm, source_ymm, 0x08); // [zero, source.low]
    return byte_offset == 1 ? _mm256_alignr_epi8(source_ymm, carry_ymm, 15)
                            : _mm256_alignr_epi8(source_ymm, carry_ymm, 14);
}

/**
 *  @brief Shifts the 32 source bytes left by one lane, so lane `i` holds byte `i + 1`.
 *      Lane 31 receives zero; any 2-byte lead there is trimmed as incomplete before folding anyway.
 */
SZ_INTERNAL __m256i sz_haswell_next_bytes_(__m256i source_ymm) {
    __m256i carry_ymm = _mm256_permute2x128_si256(source_ymm, source_ymm, 0x81); // [source.high, zero]
    return _mm256_alignr_epi8(carry_ymm, source_ymm, 1);
}

/** @brief First N bits set; BZHI keeps `n == 32` defined, unlike the `(1 << n) − 1` idiom. */
SZ_INTERNAL sz_u32_t sz_haswell_mask_until_(sz_size_t n) { return (sz_u32_t)_bzhi_u32(0xFFFFFFFFu, (unsigned)n); }

/**
 *  @brief Folds a 32-byte chunk of caseless multi-byte scripts mixed with ASCII.
 *      Folds ASCII A-Z in place and copies everything else, truncating before the first lead of
 *      a case-aware family and trimming incomplete trailing sequences. Stores a full 32-byte
 *      vector and reports how many bytes were consumed: the destination is contractually ≥ 3×
 *      the source, so the overshoot is always in bounds and the next chunk's store (or the
 *      serial tail) overwrites it.
 *  @param is_foreign_lead_mask Lead bytes outside the caseless family: the chunk is truncated
 *      before the first such lead, so Hebrew or CJK text with an embedded Latin or Cyrillic
 *      word still copies its longest caseless prefix vectorized.
 *  @return Bytes consumed and written, or zero if the first character needs another handler.
 */
SZ_INTERNAL sz_size_t sz_utf8_case_fold_haswell_caseless_chunk_( //
    __m256i source_ymm, sz_u32_t is_two_byte_lead_mask, sz_u32_t is_three_byte_lead_mask,
    sz_u32_t is_foreign_lead_mask, sz_ptr_t target) {

    sz_size_t fold_length = is_foreign_lead_mask ? (sz_size_t)sz_u32_ctz(is_foreign_lead_mask) : 32;

    // Don't split a trailing 2-byte or 3-byte sequence across chunks
    sz_u32_t incomplete_mask = //
        (is_two_byte_lead_mask & ~sz_haswell_mask_until_(fold_length > 1 ? fold_length - 1 : 0)) |
        (is_three_byte_lead_mask & ~sz_haswell_mask_until_(fold_length > 2 ? fold_length - 2 : 0));
    incomplete_mask &= sz_haswell_mask_until_(fold_length);
    if (incomplete_mask) fold_length = (sz_size_t)sz_u32_ctz(incomplete_mask);
    if (fold_length == 0) return 0;

    _mm256_storeu_si256((__m256i *)target, sz_haswell_fold_ascii_(source_ymm));
    return fold_length;
}

/**
 *  @brief Folds a 32-byte chunk of Latin text in place: ASCII, Latin-1 Supplement (C2-C3),
 *      Latin Extended-A/B (C4-C6), and Latin Extended Additional (E1 B8-BB) - the working set
 *      of German, Czech, Vietnamese, and most other Latin-script languages.
 *
 *  Latin Extended folding is parity-based: uppercase codepoints fold to the adjacent codepoint,
 *  and the codepoint's low bit lives in the last byte of its UTF-8 sequence, so the fold is an
 *  in-place masked +1. Where Ice Lake reads per-codepoint deltas from `VPERMB` tables, AVX2
 *  re-derives them arithmetically: within C4 and C5 the foldable continuations form a few
 *  contiguous even-parity or odd-parity runs, so two or three range compares per lead replace the
 *  table. C6 (Latin Ext-B 'ƀ'-'ƿ') is too fragmented for ranges - 16 scattered +1 pairs and 20
 *  irregulars - so its two 64-entry membership bitsets are decomposed into 16-entry `VPSHUFB`
 *  nibble lookups: `bitmap[low_nibble] & (1 << high_quadrant)` tests membership in five ops,
 *  versus ~12 compares for the explicit ranges. Irregular codepoints (those that expand, shrink,
 *  or fold across lead bytes) and foreign E1 sub-families (Georgian, Greek Extended) truncate the
 *  chunk before the character's lead and route one rune to the serial fallback.
 *
 *  All folds are applied across the full vector and a full 32-byte store is issued; lanes at or
 *  beyond the consumed length hold garbage that the next chunk's store overwrites (the destination
 *  is contractually ≥ 3× the source, so the overshoot stays in bounds).
 *
 *  @param is_foreign_lead_mask Lead bytes outside this handler's families: the chunk is truncated
 *      before the first such lead, so mixed-script chunks still fold their longest pure prefix
 *      vectorized instead of degrading to one-rune serial steps per chunk.
 *  @return Bytes consumed and written, or zero if the first character needs the serial path.
 */
SZ_INTERNAL sz_size_t sz_utf8_case_fold_haswell_latin_chunk_( //
    __m256i source_ymm, sz_u32_t is_continuation_mask, sz_u32_t is_three_byte_lead_mask,
    sz_u32_t is_foreign_lead_mask, sz_ptr_t target) {

    // Shifted views of the data replace Ice Lake's k-mask shifts: comparing `previous_bytes`
    // against a lead value marks the continuation lanes directly, with no vector→GPR round-trip.
    __m256i previous_bytes_ymm = sz_haswell_previous_bytes_(source_ymm, 1);
    __m256i second_previous_bytes_ymm = sz_haswell_previous_bytes_(source_ymm, 2);
    __m256i next_bytes_ymm = sz_haswell_next_bytes_(source_ymm);

    __m256i is_after_c2_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xC2));
    __m256i is_after_c3_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xC3));
    __m256i is_after_c4_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xC4));
    __m256i is_after_c5_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xC5));
    __m256i is_after_c6_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xC6));
    __m256i is_after_e1_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xE1));

    // The codepoint's low bit lives in the continuation byte's low bit, deciding the +1 parity fold
    __m256i is_odd_byte_ymm = _mm256_cmpeq_epi8(_mm256_and_si256(source_ymm, _mm256_set1_epi8(0x01)),
                                                _mm256_set1_epi8(0x01));

    // Latin Ext-A first half after C4 ('Ā' U+0100 - 'Ŀ' U+013F): even continuations fold across
    // 80-B7, odd ones across B9-BD ('Ĺ' U+0139 - 'Ľ' U+013D). 'İ' (U+0130, C4 B0) expands and
    // 'Ŀ' (U+013F, C4 BF) folds to 'ŀ' (U+0140, C5 80) - the +1 would cross lead bytes - so both
    // are irregular.
    __m256i c4_fold_ymm = _mm256_and_si256(
        is_after_c4_ymm, _mm256_or_si256(_mm256_andnot_si256(is_odd_byte_ymm, //
                                                             sz_haswell_in_byte_range_(source_ymm, 0x80, 0x38)),
                                         _mm256_and_si256(is_odd_byte_ymm, //
                                                          sz_haswell_in_byte_range_(source_ymm, 0xB9, 0x05))));
    __m256i c4_irregular_ymm = _mm256_and_si256(
        is_after_c4_ymm, _mm256_or_si256(_mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0xB0)),
                                         _mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0xBF))));

    // Latin Ext-A second half after C5 ('ŀ' U+0140 - 'ſ' U+017F): odd continuations fold across
    // 81-87 ('Ł'-'Ň') and B9-BD ('Ź'-'Ž'), even ones across 8A-B6 ('Ŋ'-'Ŷ'). 'ŉ' (U+0149, C5 89)
    // expands, 'Ÿ' (U+0178, C5 B8) folds down to 'ÿ' (U+00FF), and 'ſ' (U+017F, C5 BF) folds to
    // ASCII 's' - all three are irregular.
    __m256i c5_fold_ymm = _mm256_and_si256(
        is_after_c5_ymm,
        _mm256_or_si256(_mm256_and_si256(is_odd_byte_ymm, //
                                         _mm256_or_si256(sz_haswell_in_byte_range_(source_ymm, 0x81, 0x07),
                                                         sz_haswell_in_byte_range_(source_ymm, 0xB9, 0x05))),
                        _mm256_andnot_si256(is_odd_byte_ymm, //
                                            sz_haswell_in_byte_range_(source_ymm, 0x8A, 0x2D))));
    __m256i c5_irregular_ymm = _mm256_and_si256(
        is_after_c5_ymm, _mm256_or_si256(_mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0x89)),
                                         _mm256_or_si256(_mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0xB8)),
                                                         _mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0xBF)))));

    // Latin Ext-B after C6 ('ƀ' U+0180 - 'ƿ' U+01BF): membership bitsets from the Ice Lake delta
    // LUT, decomposed into nibble lookups. Each table byte holds, for one continuation low-nibble,
    // a 4-bit map across the four 16-codepoint quadrants of the block.
    __m256i const c6_fold_bitmap_lut = _mm256_setr_epi8(  //
        4, 2, 5, 8, 5, 8, 0, 5, 10, 0, 0, 1, 12, 0, 0, 4, //
        4, 2, 5, 8, 5, 8, 0, 5, 10, 0, 0, 1, 12, 0, 0, 4);
    __m256i const c6_irregular_bitmap_lut = _mm256_setr_epi8( //
        2, 9, 8, 2, 2, 0, 7, 10, 0, 5, 1, 0, 2, 2, 5, 3,      //
        2, 9, 8, 2, 2, 0, 7, 10, 0, 5, 1, 0, 2, 2, 5, 3);
    __m256i const quadrant_bit_lut = _mm256_setr_epi8(  //
        1, 2, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
        1, 2, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i low_nibbles_ymm = _mm256_and_si256(source_ymm, _mm256_set1_epi8(0x0F));
    __m256i quadrant_indices_ymm = _mm256_and_si256(_mm256_srli_epi16(source_ymm, 4), _mm256_set1_epi8(0x03));
    __m256i quadrant_bits_ymm = _mm256_shuffle_epi8(quadrant_bit_lut, quadrant_indices_ymm);
    __m256i c6_fold_ymm = _mm256_and_si256(
        is_after_c6_ymm,
        _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(c6_fold_bitmap_lut, low_nibbles_ymm), quadrant_bits_ymm),
                          quadrant_bits_ymm));
    __m256i c6_irregular_ymm = _mm256_and_si256(
        is_after_c6_ymm, _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(c6_irregular_bitmap_lut, //
                                                                                low_nibbles_ymm),
                                                            quadrant_bits_ymm),
                                           quadrant_bits_ymm));

    // E1 sequences qualify only when the second byte is B8-BB (Latin Extended Additional);
    // other E1 sub-families (Georgian, Greek Extended) route to the serial path.
    __m256i is_b8_bb_second_ymm = sz_haswell_in_byte_range_(source_ymm, 0xB8, 0x04);
    __m256i foreign_e1_second_ymm = _mm256_andnot_si256(is_b8_bb_second_ymm, is_after_e1_ymm);
    __m256i e1_latin_third_ymm = _mm256_and_si256( //
        _mm256_cmpeq_epi8(second_previous_bytes_ymm, _mm256_set1_epi8((char)0xE1)),
        sz_haswell_in_byte_range_(previous_bytes_ymm, 0xB8, 0x04));
    // Latin Extended Additional irregulars: E1 BA 96-9E ('ẖ'-'ẞ') expand or shrink when folded
    __m256i e1_irregular_ymm = _mm256_and_si256(
        _mm256_and_si256(e1_latin_third_ymm, _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xBA))),
        sz_haswell_in_byte_range_(source_ymm, 0x96, 0x09));

    // Truncate at the first irregular codepoint, foreign E1 sub-family, or foreign-family lead -
    // an irregular-flagged lane is a continuation, so walk back over continuations to the start
    // of that sequence (foreign leads are already at a sequence start, so the walk is a no-op)
    sz_u32_t stop_mask = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_or_si256(_mm256_or_si256(c4_irregular_ymm, c5_irregular_ymm),
                        _mm256_or_si256(c6_irregular_ymm, _mm256_or_si256(e1_irregular_ymm, foreign_e1_second_ymm))));
    stop_mask |= is_foreign_lead_mask;
    sz_size_t fold_length = 32;
    if (stop_mask) {
        sz_size_t first_flagged_position = (sz_size_t)sz_u32_ctz(stop_mask);
        while (first_flagged_position && ((is_continuation_mask >> first_flagged_position) & 1))
            --first_flagged_position;
        fold_length = first_flagged_position;
    }

    // Don't split a trailing 2-byte or 3-byte sequence across chunks; C2-C6 are this family's
    // only 2-byte leads, so one range compare covers them all
    sz_u32_t is_two_byte_lead_mask = (sz_u32_t)_mm256_movemask_epi8(sz_haswell_in_byte_range_(source_ymm, 0xC2, 0x05));
    sz_u32_t incomplete_mask = //
        (is_two_byte_lead_mask & ~sz_haswell_mask_until_(fold_length > 1 ? fold_length - 1 : 0)) |
        (is_three_byte_lead_mask & ~sz_haswell_mask_until_(fold_length > 2 ? fold_length - 2 : 0));
    incomplete_mask &= sz_haswell_mask_until_(fold_length);
    if (incomplete_mask) fold_length = (sz_size_t)sz_u32_ctz(incomplete_mask);
    if (fold_length == 0) return 0;

    // 1. ASCII A-Z
    __m256i folded_ymm = sz_haswell_fold_ascii_(source_ymm);

    // 2. Latin-1 Supplement: 'À'-'Þ' (C3 80-9E, excluding '×' at 0x97) get +0x20
    __m256i is_latin1_upper_ymm = _mm256_andnot_si256(
        _mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0x97)),
        _mm256_and_si256(is_after_c3_ymm, sz_haswell_in_byte_range_(source_ymm, 0x80, 0x1F)));
    folded_ymm = _mm256_add_epi8(folded_ymm, _mm256_and_si256(is_latin1_upper_ymm, _mm256_set1_epi8(0x20)));

    // 3. 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73): both bytes become 's' via masked
    //    subtractions (C3 − 0x50 = 9F − 0x2C = 0x73), keeping the no-`VPBLENDVB` discipline
    __m256i is_eszett_second_ymm = _mm256_and_si256(is_after_c3_ymm,
                                                    _mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0x9F)));
    __m256i is_eszett_lead_ymm = _mm256_and_si256(_mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0xC3)),
                                                  _mm256_cmpeq_epi8(next_bytes_ymm, _mm256_set1_epi8((char)0x9F)));
    folded_ymm = _mm256_sub_epi8(folded_ymm, _mm256_and_si256(is_eszett_lead_ymm, _mm256_set1_epi8(0x50)));
    folded_ymm = _mm256_sub_epi8(folded_ymm, _mm256_and_si256(is_eszett_second_ymm, _mm256_set1_epi8(0x2C)));

    // 4. 'µ' (U+00B5, C2 B5) → 'μ' (U+03BC, CE BC): masked additions (C2 + 0x0C = CE, B5 + 7 = BC)
    __m256i is_micro_second_ymm = _mm256_and_si256(is_after_c2_ymm,
                                                   _mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0xB5)));
    __m256i is_micro_lead_ymm = _mm256_and_si256(_mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0xC2)),
                                                 _mm256_cmpeq_epi8(next_bytes_ymm, _mm256_set1_epi8((char)0xB5)));
    folded_ymm = _mm256_add_epi8(folded_ymm, _mm256_and_si256(is_micro_lead_ymm, _mm256_set1_epi8(0x0C)));
    folded_ymm = _mm256_add_epi8(folded_ymm, _mm256_and_si256(is_micro_second_ymm, _mm256_set1_epi8(0x07)));

    // 5. Latin Extended-A/B masked +1, and Latin Extended Additional (+1 on even third bytes of
    //    E1 B8-BB sequences); irregular lanes may fold wrongly here, but they sit at or beyond
    //    `fold_length`, so the next chunk's store replaces them
    __m256i extended_fold_ymm = _mm256_or_si256(_mm256_or_si256(c4_fold_ymm, c5_fold_ymm), c6_fold_ymm);
    __m256i e1_fold_ymm = _mm256_andnot_si256(is_odd_byte_ymm, e1_latin_third_ymm);
    folded_ymm = _mm256_add_epi8(
        folded_ymm, _mm256_and_si256(_mm256_or_si256(extended_fold_ymm, e1_fold_ymm), _mm256_set1_epi8(0x01)));

    _mm256_storeu_si256((__m256i *)target, folded_ymm);
    return fold_length;
}

/**
 *  @brief Folds a 32-byte chunk of basic Cyrillic (D0/D1 leads, U+0400-045F) mixed with ASCII -
 *      the working set of Russian, Ukrainian, Bulgarian, and Serbian text.
 *
 *  The uppercase block maps onto the lowercase block with offsets keyed purely by the second
 *  byte's high nibble, so one `VPSHUFB` over a 16-entry table replaces three range compares
 *  plus three masked adds:
 *
 *  Input Range | Codepoints        | Output           | Transform
 *  D0 80-8F    | Ѐ-Џ (U+0400-040F) | D1 90-9F (ѐ-џ)   | second +0x10, lead D0 → D1
 *  D0 90-9F    | А-П (U+0410-041F) | D0 B0-BF (а-п)   | second +0x20
 *  D0 A0-AF    | Р-Я (U+0420-042F) | D1 80-8F (р-я)   | second −0x20, lead D0 → D1
 *  D0 B0-BF    | а-п lowercase     | unchanged        | high nibble B → offset 0
 *  D1 80-9F    | р-џ lowercase     | unchanged        | only after-D0 lanes take offsets
 *  D1 A0+      | Ext-A (U+0460+)   | → serial         | +1 parity folds, not modeled here
 *
 *  The D0 → D1 lead fixup is a masked +1 (D0 + 1 = D1), keeping the no-`VPBLENDVB` discipline.
 *  Cyrillic Extended-A and foreign-family leads (Russian quotes «» are C2, dashes are E2)
 *  truncate the chunk before the offending lead, so mixed chunks still fold their longest
 *  pure prefix vectorized. A full 32-byte store is issued; lanes at or beyond the consumed
 *  length are overwritten by the next chunk's store (the destination is ≥ 3× the source).
 *
 *  @return Bytes consumed and written, or zero if the first character needs the serial path.
 */
SZ_INTERNAL sz_size_t sz_utf8_case_fold_haswell_cyrillic_chunk_( //
    __m256i source_ymm, sz_u32_t is_foreign_lead_mask, sz_ptr_t target) {

    __m256i previous_bytes_ymm = sz_haswell_previous_bytes_(source_ymm, 1);
    __m256i next_bytes_ymm = sz_haswell_next_bytes_(source_ymm);
    __m256i is_after_d0_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xD0));
    __m256i is_after_d1_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xD1));

    // Cyrillic Extended-A ('Ѡ' U+0460 onward) starts at D1 A0 and folds by +1 parity - its
    // flagged continuation maps back to the lead one lane before via `>> 1`; foreign-family
    // leads are already at a sequence start
    sz_u32_t is_extended_second_mask = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_and_si256(is_after_d1_ymm, sz_haswell_in_byte_range_(source_ymm, 0xA0, 0x20)));
    sz_u32_t stop_mask = is_foreign_lead_mask | (is_extended_second_mask >> 1);
    sz_size_t fold_length = stop_mask ? (sz_size_t)sz_u32_ctz(stop_mask) : 32;

    // Don't split a trailing D0/D1 lead across chunks - this family's only multi-byte leads
    sz_u32_t is_cyrillic_lead_mask = (sz_u32_t)_mm256_movemask_epi8(sz_haswell_in_byte_range_(source_ymm, 0xD0, 0x02));
    sz_u32_t incomplete_mask = is_cyrillic_lead_mask & ~sz_haswell_mask_until_(fold_length > 1 ? fold_length - 1 : 0);
    incomplete_mask &= sz_haswell_mask_until_(fold_length);
    if (incomplete_mask) fold_length = (sz_size_t)sz_u32_ctz(incomplete_mask);
    if (fold_length == 0) return 0;

    // The second byte's high nibble keys the fold offset: 8 → +0x10, 9 → +0x20, A → −0x20 (0xE0),
    // B → 0. `VPSHUFB` works per 128-bit lane, so the 16-entry table is mirrored into both halves.
    __m256i const cyrillic_offset_lut = _mm256_setr_epi8(            //
        0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x20, (char)0xE0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x20, (char)0xE0, 0, 0, 0, 0, 0);
    __m256i high_nibbles_ymm = _mm256_and_si256(_mm256_srli_epi16(source_ymm, 4), _mm256_set1_epi8(0x0F));
    __m256i offsets_ymm = _mm256_shuffle_epi8(cyrillic_offset_lut, high_nibbles_ymm);

    __m256i folded_ymm = sz_haswell_fold_ascii_(source_ymm);
    folded_ymm = _mm256_add_epi8(folded_ymm, _mm256_and_si256(is_after_d0_ymm, offsets_ymm));

    // Lead fixup: Ѐ-Џ (seconds 80-8F) and Р-Я (seconds A0-AF) land in the D1 block, so their
    // D0 lead takes a masked +1; А-П (seconds 90-9F) stay under D0
    __m256i is_d0_ymm = _mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0xD0));
    __m256i needs_d1_ymm = _mm256_and_si256(
        is_d0_ymm, _mm256_or_si256(sz_haswell_in_byte_range_(next_bytes_ymm, 0x80, 0x10),
                                   sz_haswell_in_byte_range_(next_bytes_ymm, 0xA0, 0x10)));
    folded_ymm = _mm256_add_epi8(folded_ymm, _mm256_and_si256(needs_d1_ymm, _mm256_set1_epi8(0x01)));

    _mm256_storeu_si256((__m256i *)target, folded_ymm);
    return fold_length;
}

/**
 *  @brief Folds a 32-byte chunk of basic Greek (CE/CF leads, U+0370-03FF) mixed with ASCII.
 *
 *  Greek CE/CF uppercase → lowercase transformations:
 *
 *  Input Range  | Codepoints        | Output           | Transform
 *  CE 91-9F     | Α-Ο (U+0391-039F) | CE B1-BF (α-ο)   | second +0x20
 *  CE A0-A1     | Π-Ρ (U+03A0-03A1) | CF 80-81 (π-ρ)   | second −0x20, lead CE → CF
 *  CE A3-AB     | Σ-Ϋ (U+03A3-03AB) | CF 83-8B (σ-ϋ)   | second −0x20, lead CE → CF
 *  CE B1-BF     | α-ο lowercase     | unchanged        | —
 *  CF 80-8B     | π-ϋ lowercase     | unchanged        | —
 *  CF 82        | ς final sigma     | CF 83 (σ)        | second +1
 *
 *  CE A2 (U+03A2, unassigned) is deliberately neither folded nor flagged. The fold-side
 *  exclusion set is narrower than the finder's: only sequences whose FOLD is irregular leave
 *  the fast path - CE 80-90 (tonos and accented capitals with non-uniform offsets), CE B0
 *  ('ΰ', which expands to three codepoints), and CF 8C+ (accented lowercase and archaic
 *  symbols like 'ϐ'/'ϑ' that fold onto basic letters) and foreign-family leads truncate the
 *  chunk before the offending lead, so mixed chunks still fold their longest pure prefix
 *  vectorized. The CE → CF lead fixup is a masked +1.
 *
 *  @return Bytes consumed and written, or zero if the first character needs the serial path.
 */
SZ_INTERNAL sz_size_t sz_utf8_case_fold_haswell_greek_chunk_( //
    __m256i source_ymm, sz_u32_t is_foreign_lead_mask, sz_ptr_t target) {

    __m256i previous_bytes_ymm = sz_haswell_previous_bytes_(source_ymm, 1);
    __m256i next_bytes_ymm = sz_haswell_next_bytes_(source_ymm);
    __m256i is_after_ce_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xCE));
    __m256i is_after_cf_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xCF));

    // Irregular folds truncate the chunk: CE 80-90 tonos/accents, CE B0 'ΰ' expansion, CF 8C+
    // accented lowercase and archaic symbols. The flagged lane is a continuation, mapping back
    // to its lead one lane before via `>> 1`; foreign leads are already at a sequence start.
    __m256i ce_irregular_ymm = _mm256_and_si256(
        is_after_ce_ymm, _mm256_or_si256(sz_haswell_in_byte_range_(source_ymm, 0x80, 0x11),
                                         _mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0xB0))));
    __m256i cf_irregular_ymm = _mm256_and_si256(is_after_cf_ymm, sz_haswell_in_byte_range_(source_ymm, 0x8C, 0x34));
    sz_u32_t is_irregular_second_mask = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_or_si256(ce_irregular_ymm, cf_irregular_ymm));
    sz_u32_t stop_mask = is_foreign_lead_mask | (is_irregular_second_mask >> 1);
    sz_size_t fold_length = stop_mask ? (sz_size_t)sz_u32_ctz(stop_mask) : 32;

    // Don't split a trailing CE/CF lead across chunks - this family's only multi-byte leads
    sz_u32_t is_greek_lead_mask = (sz_u32_t)_mm256_movemask_epi8(sz_haswell_in_byte_range_(source_ymm, 0xCE, 0x02));
    sz_u32_t incomplete_mask = is_greek_lead_mask & ~sz_haswell_mask_until_(fold_length > 1 ? fold_length - 1 : 0);
    incomplete_mask &= sz_haswell_mask_until_(fold_length);
    if (incomplete_mask) fold_length = (sz_size_t)sz_u32_ctz(incomplete_mask);
    if (fold_length == 0) return 0;

    __m256i folded_ymm = sz_haswell_fold_ascii_(source_ymm);

    // Α-Ο (CE 91-9F): second byte +0x20, lead stays CE
    __m256i plus_fold_ymm = _mm256_and_si256(is_after_ce_ymm, sz_haswell_in_byte_range_(source_ymm, 0x91, 0x0F));
    folded_ymm = _mm256_add_epi8(folded_ymm, _mm256_and_si256(plus_fold_ymm, _mm256_set1_epi8(0x20)));

    // Π-Ρ (CE A0-A1) and Σ-Ϋ (CE A3-AB): second byte −0x20, lead CE → CF via masked +1;
    // A2 is skipped because U+03A2 is unassigned
    __m256i minus_second_ymm = _mm256_or_si256(sz_haswell_in_byte_range_(source_ymm, 0xA0, 0x02),
                                               sz_haswell_in_byte_range_(source_ymm, 0xA3, 0x09));
    __m256i minus_fold_ymm = _mm256_and_si256(is_after_ce_ymm, minus_second_ymm);
    folded_ymm = _mm256_sub_epi8(folded_ymm, _mm256_and_si256(minus_fold_ymm, _mm256_set1_epi8(0x20)));
    __m256i is_ce_ymm = _mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0xCE));
    __m256i needs_cf_ymm = _mm256_and_si256(
        is_ce_ymm, _mm256_or_si256(sz_haswell_in_byte_range_(next_bytes_ymm, 0xA0, 0x02),
                                   sz_haswell_in_byte_range_(next_bytes_ymm, 0xA3, 0x09)));
    folded_ymm = _mm256_add_epi8(folded_ymm, _mm256_and_si256(needs_cf_ymm, _mm256_set1_epi8(0x01)));

    // Final sigma: 'ς' (U+03C2, CF 82) → 'σ' (U+03C3, CF 83) is a +1 on the second byte
    __m256i is_final_sigma_ymm = _mm256_and_si256(is_after_cf_ymm,
                                                  _mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0x82)));
    folded_ymm = _mm256_add_epi8(folded_ymm, _mm256_and_si256(is_final_sigma_ymm, _mm256_set1_epi8(0x01)));

    _mm256_storeu_si256((__m256i *)target, folded_ymm);
    return fold_length;
}

/**
 *  @brief Folds a 32-byte chunk mixing ASCII with caseless and guarded 3-byte scripts -
 *      CJK with E2 punctuation, German quotes („…“), Korean, and similar real-world blends.
 *
 *  The guarded leads (E2, EA) are case-aware only for specific second bytes, so the chunk
 *  is a fold-ASCII-and-copy as long as every guarded sequence is provably caseless:
 *    - E2 is safe for seconds 80-83 (General Punctuation - quotes, dashes, ellipses);
 *      other E2 blocks (84 Letterlike Kelvin/Angstrom, B0-B3 Glagolitic/Coptic, …) fold;
 *    - EA is safe except seconds 99-9F (Cyrillic Ext-B, Latin Ext-D) and AD-AE (Cherokee
 *      Supplement) - Hangul (EA B0+) passes untouched;
 *    - EF is never safe: fullwidth A-Z lives under EF BC and folds by +0x20.
 *  The first unsafe or foreign-family lead truncates the chunk in place - no walk-back
 *  needed, the flagged lane IS the lead - and one rune goes to the serial fallback.
 *
 *  @return Bytes consumed and written, or zero if the first character needs the serial path.
 */
SZ_INTERNAL sz_size_t sz_utf8_case_fold_haswell_guarded_chunk_( //
    __m256i source_ymm, sz_u32_t is_two_byte_lead_mask, sz_u32_t is_three_byte_lead_mask,
    sz_u32_t is_foreign_lead_mask, sz_ptr_t target) {

    __m256i next_bytes_ymm = sz_haswell_next_bytes_(source_ymm);
    __m256i is_e2_ymm = _mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0xE2));
    __m256i is_ea_ymm = _mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0xEA));
    __m256i is_ef_ymm = _mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0xEF));

    // An E2 lead in lane 31 sees a zero `next` byte, fails the 80-83 test, and truncates -
    // which doubles as the incomplete-sequence trim for that lane
    __m256i unsafe_e2_ymm = _mm256_andnot_si256(sz_haswell_in_byte_range_(next_bytes_ymm, 0x80, 0x04), is_e2_ymm);
    __m256i unsafe_ea_ymm = _mm256_and_si256(
        is_ea_ymm, _mm256_or_si256(sz_haswell_in_byte_range_(next_bytes_ymm, 0x99, 0x07),
                                   sz_haswell_in_byte_range_(next_bytes_ymm, 0xAD, 0x02)));
    sz_u32_t stop_mask = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_or_si256(_mm256_or_si256(unsafe_e2_ymm, unsafe_ea_ymm), is_ef_ymm));
    stop_mask |= is_foreign_lead_mask;
    sz_size_t fold_length = stop_mask ? (sz_size_t)sz_u32_ctz(stop_mask) : 32;

    // Don't split trailing sequences: the caseless family contributes 2-byte leads (D7-DF)
    // and both families contribute 3-byte leads
    sz_u32_t incomplete_mask = //
        (is_two_byte_lead_mask & ~sz_haswell_mask_until_(fold_length > 1 ? fold_length - 1 : 0)) |
        (is_three_byte_lead_mask & ~sz_haswell_mask_until_(fold_length > 2 ? fold_length - 2 : 0));
    incomplete_mask &= sz_haswell_mask_until_(fold_length);
    if (incomplete_mask) fold_length = (sz_size_t)sz_u32_ctz(incomplete_mask);
    if (fold_length == 0) return 0;

    _mm256_storeu_si256((__m256i *)target, sz_haswell_fold_ascii_(source_ymm));
    return fold_length;
}

/**
 *  @brief Folds a 32-byte chunk of supplementary-plane 4-byte sequences (F0-F4 leads) mixed
 *      with ASCII - emoji-rich text and historic scripts.
 *
 *  Every supplementary-plane codepoint WITH case folding (Deseret, Osage, Warang Citi, Adlam,
 *  Garay, Medefaidrin, archaic Latin/Greek extensions, …) lives below U+1F000, i.e. its UTF-8
 *  second byte is < 0x9F under an F0 lead. Sequences whose second byte is ≥ 0x9F - all emoji
 *  and everything in planes 2+ - are caseless and copy through unchanged, with only the mixed
 *  ASCII folded. Folding 4-byte sequences, the complex family's 2-byte leads (C0-C1, C7-CD,
 *  D2-D6: Latin Ext-B tails, IPA, extended Cyrillic, Armenian), and foreign-family leads
 *  truncate the chunk at their lead and route one rune to the serial fallback.
 *
 *  @return Bytes consumed and written, or zero if the first character needs the serial path.
 */
SZ_INTERNAL sz_size_t sz_utf8_case_fold_haswell_supplementary_chunk_( //
    __m256i source_ymm, sz_u32_t is_complex_lead_mask, sz_u32_t is_four_byte_lead_mask,
    sz_u32_t is_foreign_lead_mask, sz_ptr_t target) {

    __m256i next_bytes_ymm = sz_haswell_next_bytes_(source_ymm);
    __m256i is_four_byte_lead_ymm = sz_haswell_in_byte_range_(source_ymm, 0xF0, 0x08);

    // A 4-byte lead in lane 31 sees a zero `next` byte, fails the ≥ 0x9F test, and truncates -
    // partially covering the incomplete-sequence trim; lanes 29-30 still need the trim below
    __m256i folding_four_byte_ymm = _mm256_andnot_si256(sz_haswell_in_byte_range_(next_bytes_ymm, 0x9F, 0x61),
                                                        is_four_byte_lead_ymm);
    sz_u32_t stop_mask = (is_complex_lead_mask & ~is_four_byte_lead_mask) | is_foreign_lead_mask |
                         (sz_u32_t)_mm256_movemask_epi8(folding_four_byte_ymm);
    sz_size_t fold_length = stop_mask ? (sz_size_t)sz_u32_ctz(stop_mask) : 32;

    // Don't split a trailing 4-byte sequence: a lead in lanes 29-31 lacks its continuations
    sz_u32_t incomplete_mask = is_four_byte_lead_mask &
                               ~sz_haswell_mask_until_(fold_length > 3 ? fold_length - 3 : 0);
    incomplete_mask &= sz_haswell_mask_until_(fold_length);
    if (incomplete_mask) fold_length = (sz_size_t)sz_u32_ctz(incomplete_mask);
    if (fold_length == 0) return 0;

    _mm256_storeu_si256((__m256i *)target, sz_haswell_fold_ascii_(source_ymm));
    return fold_length;
}

SZ_PUBLIC sz_size_t sz_utf8_case_fold_haswell(sz_cptr_t source, sz_size_t source_length, sz_ptr_t target) {
    // The 32-byte port of the Ice Lake classify-once design: every full chunk is classified into
    // lead-byte families with a compare tree, and uniform chunks route straight to their handler.
    // Chunks that mix families beyond the handlers below fold one rune serially and re-enter the
    // vector loop; the final sub-32-byte tail is delegated to the serial kernel wholesale, which
    // also keeps the incomplete-trailing-sequence behavior bit-identical to the baseline.
    sz_ptr_t const target_start = target;

    while (source_length >= 32) {
        // Prefetch ahead to hide memory latency on large datasets that overflow the caches
        _mm_prefetch(source + 512, _MM_HINT_T0);

        __m256i source_ymm = _mm256_lddqu_si256((__m256i const *)source);
        sz_u32_t is_non_ascii_mask = (sz_u32_t)_mm256_movemask_epi8(source_ymm);

        // FAST PATH: pure ASCII chunks - the most common case for English and many other
        // Latin-script texts - skip all classification work
        if (is_non_ascii_mask == 0) {
            _mm256_storeu_si256((__m256i *)target, sz_haswell_fold_ascii_(source_ymm));
            target += 32, source += 32, source_length -= 32;
            continue;
        }

        // Lead bytes are non-ASCII bytes outside the continuation range 10xxxxxx (80-BF).
        // Every family range starts at 0xC2 or above, so the range compares below cannot
        // misfire on ASCII or continuation bytes and can run on the raw source vector.
        __m256i is_continuation_ymm = sz_haswell_in_byte_range_(source_ymm, 0x80, 0x40);
        sz_u32_t is_continuation_mask = (sz_u32_t)_mm256_movemask_epi8(is_continuation_ymm);
        sz_u32_t is_lead_mask = is_non_ascii_mask & ~is_continuation_mask;
        sz_u32_t is_three_byte_lead_mask = (sz_u32_t)_mm256_movemask_epi8(
            sz_haswell_in_byte_range_(source_ymm, 0xE0, 0x10));
        sz_u32_t is_four_byte_lead_mask = (sz_u32_t)_mm256_movemask_epi8(
            sz_haswell_in_byte_range_(source_ymm, 0xF0, 0x08));

        // Classify lead bytes once: one range compare per family, each reduced via `VPMOVMSKB`,
        // then OR-folded into the same flag byte the Ice Lake `VPERMB` LUT produces. The caseless
        // family merges D7-DF and E0 into one contiguous D7-E0 span.
        sz_u32_t is_caseless_lead_mask = (sz_u32_t)_mm256_movemask_epi8(
            _mm256_or_si256(sz_haswell_in_byte_range_(source_ymm, 0xD7, 0x0A),
                            _mm256_or_si256(sz_haswell_in_byte_range_(source_ymm, 0xE3, 0x07),
                                            sz_haswell_in_byte_range_(source_ymm, 0xEB, 0x04))));
        sz_u32_t is_latin_lead_mask = (sz_u32_t)_mm256_movemask_epi8(sz_haswell_in_byte_range_(source_ymm, 0xC2, 0x02));
        sz_u32_t is_latin_extended_lead_mask = (sz_u32_t)_mm256_movemask_epi8(
            sz_haswell_in_byte_range_(source_ymm, 0xC4, 0x03));
        sz_u32_t is_cyrillic_lead_mask = (sz_u32_t)_mm256_movemask_epi8(
            sz_haswell_in_byte_range_(source_ymm, 0xD0, 0x02));
        sz_u32_t is_greek_lead_mask = (sz_u32_t)_mm256_movemask_epi8(sz_haswell_in_byte_range_(source_ymm, 0xCE, 0x02));
        sz_u32_t is_e1_lead_mask = (sz_u32_t)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0xE1)));
        sz_u32_t is_guarded_lead_mask = (sz_u32_t)_mm256_movemask_epi8(
            _mm256_or_si256(_mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0xE2)),
                            _mm256_or_si256(_mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0xEA)),
                                            _mm256_cmpeq_epi8(source_ymm, _mm256_set1_epi8((char)0xEF)))));
        sz_u32_t is_complex_lead_mask = is_lead_mask & ~(is_caseless_lead_mask | is_latin_lead_mask |
                                                         is_latin_extended_lead_mask | is_cyrillic_lead_mask |
                                                         is_greek_lead_mask | is_e1_lead_mask | is_guarded_lead_mask);
        sz_u8_t lead_families = (sz_u8_t)( //
            ((is_caseless_lead_mask != 0) << 0) | ((is_latin_lead_mask != 0) << 1) |
            ((is_latin_extended_lead_mask != 0) << 2) | ((is_cyrillic_lead_mask != 0) << 3) |
            ((is_greek_lead_mask != 0) << 4) | ((is_e1_lead_mask != 0) << 5) | ((is_guarded_lead_mask != 0) << 6) |
            ((is_complex_lead_mask != 0) << 7));

        // Handler dispatch triggers on family PRESENCE, not exclusivity: every handler below
        // truncates at the first lead outside its families, so a mixed chunk still folds its
        // longest matching prefix vectorized. Without this, one foreign byte - a « quote in
        // Russian, an — dash in German - poisons ~16 consecutive windows into one-rune serial
        // steps, halving real-corpus throughput. A handler that cannot fold even the first
        // character returns zero and falls through to the next family or the serial fallback.
        sz_size_t handled = 0;
        sz_u32_t is_two_byte_lead_mask = is_lead_mask & ~is_three_byte_lead_mask & ~is_four_byte_lead_mask;
        if (lead_families & sz_utf8_fold_haswell_lead_caseless_flag_)
            handled = sz_utf8_case_fold_haswell_caseless_chunk_(source_ymm, is_two_byte_lead_mask,
                                                                is_three_byte_lead_mask,
                                                                is_lead_mask & ~is_caseless_lead_mask, target);
        // Unlike Ice Lake, pure Latin-1 chunks (German, French) take this handler too: it covers
        // their C2/C3 folds exactly, and there is no separate Latin-1 cascade to fall back onto
        if (!handled && (lead_families & (sz_utf8_fold_haswell_lead_latin_flag_ |
                                          sz_utf8_fold_haswell_lead_latin_extended_flag_ |
                                          sz_utf8_fold_haswell_lead_e1_flag_)))
            handled = sz_utf8_case_fold_haswell_latin_chunk_(
                source_ymm, is_continuation_mask, is_three_byte_lead_mask,
                is_lead_mask & ~(is_latin_lead_mask | is_latin_extended_lead_mask | is_e1_lead_mask), target);
        // Basic Cyrillic + ASCII - the common case for Russian, Ukrainian, and Bulgarian
        if (!handled && (lead_families & sz_utf8_fold_haswell_lead_cyrillic_flag_))
            handled = sz_utf8_case_fold_haswell_cyrillic_chunk_(source_ymm, is_lead_mask & ~is_cyrillic_lead_mask,
                                                                target);
        // Basic Greek + ASCII
        if (!handled && (lead_families & sz_utf8_fold_haswell_lead_greek_flag_))
            handled = sz_utf8_case_fold_haswell_greek_chunk_(source_ymm, is_lead_mask & ~is_greek_lead_mask, target);
        // Guarded 3-byte leads mixed with caseless scripts - CJK or Hangul with E2 punctuation;
        // the handler verifies the guarded seconds and truncates at the first folding sequence
        if (!handled && (lead_families & sz_utf8_fold_haswell_lead_guarded_flag_))
            handled = sz_utf8_case_fold_haswell_guarded_chunk_(
                source_ymm, is_two_byte_lead_mask, is_three_byte_lead_mask,
                is_lead_mask & ~(is_caseless_lead_mask | is_guarded_lead_mask), target);
        // Complex chunks are usually emoji runs: 4-byte sequences with caseless second bytes
        // copy through; anything else in the family truncates to the serial path
        if (!handled && (lead_families & sz_utf8_fold_haswell_lead_complex_flag_))
            handled = sz_utf8_case_fold_haswell_supplementary_chunk_(
                source_ymm, is_complex_lead_mask, is_four_byte_lead_mask, is_lead_mask & ~is_complex_lead_mask, target);
        if (handled) {
            target += handled, source += handled, source_length -= handled;
            continue;
        }

        // Mixed or complex chunks fold one rune serially and rejoin the vector loop; with ≥ 32
        // source bytes left, a complete (≤ 4-byte) sequence is always available
        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse(source, &rune, &rune_length);
        sz_rune_t folded_runes[3]; // Unicode case folding produces at most 3 runes
        sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
        for (sz_size_t rune_index = 0; rune_index != folded_count; ++rune_index)
            target += sz_rune_export(folded_runes[rune_index], (sz_u8_t *)target);
        source += rune_length;
        source_length -= rune_length;
    }

    // The sub-32-byte tail goes through the serial kernel, inheriting its handling of
    // incomplete or invalid trailing sequences byte-for-byte
    target += sz_utf8_case_fold_serial(source, source_length, target);
    return (sz_size_t)(target - target_start);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_CASE_FOLD_HASWELL_H_
