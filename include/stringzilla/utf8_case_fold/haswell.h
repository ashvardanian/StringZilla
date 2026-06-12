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
 *  @brief Folds a 32-byte chunk containing only caseless multi-byte scripts mixed with ASCII.
 *      Folds ASCII A-Z in place and copies everything else, trimming incomplete trailing sequences.
 *      Stores a full 32-byte vector and reports how many bytes were consumed: the destination is
 *      contractually ≥ 3× the source, so the ≤ 2-byte overshoot is always in bounds and the next
 *      chunk's store (or the serial tail) overwrites it.
 *  @return Bytes consumed and written - at least 30, since only the last two lanes can host an
 *      incomplete 2-byte or 3-byte lead in a full chunk.
 */
SZ_INTERNAL sz_size_t sz_utf8_case_fold_haswell_caseless_chunk_( //
    __m256i source_ymm, sz_u32_t is_two_byte_lead_mask, sz_u32_t is_three_byte_lead_mask, sz_ptr_t target) {

    // Don't split a trailing 2-byte or 3-byte sequence across chunks: in a full 32-byte chunk
    // only a 2-byte lead in lane 31 or a 3-byte lead in lanes 30-31 can be incomplete.
    sz_u32_t incomplete_mask = (is_two_byte_lead_mask & 0x80000000u) | (is_three_byte_lead_mask & 0xC0000000u);
    sz_size_t copy_length = incomplete_mask ? (sz_size_t)sz_u32_ctz(incomplete_mask) : 32;
    _mm256_storeu_si256((__m256i *)target, sz_haswell_fold_ascii_(source_ymm));
    return copy_length;
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
 *  @return Bytes consumed and written, or zero if the first character needs the serial path.
 */
SZ_INTERNAL sz_size_t sz_utf8_case_fold_haswell_latin_chunk_( //
    __m256i source_ymm, sz_u32_t is_continuation_mask, sz_u32_t is_three_byte_lead_mask, sz_ptr_t target) {

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

    // Truncate at the first irregular codepoint or foreign E1 sub-family - the flagged lane is a
    // continuation, so walk back over continuations to the start of that sequence
    sz_u32_t stop_mask = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_or_si256(_mm256_or_si256(c4_irregular_ymm, c5_irregular_ymm),
                        _mm256_or_si256(c6_irregular_ymm, _mm256_or_si256(e1_irregular_ymm, foreign_e1_second_ymm))));
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

        if (!(lead_families & ~sz_utf8_fold_haswell_lead_caseless_flag_)) {
            sz_u32_t is_two_byte_lead_mask = is_lead_mask & ~is_three_byte_lead_mask & ~is_four_byte_lead_mask;
            sz_size_t handled = sz_utf8_case_fold_haswell_caseless_chunk_(source_ymm, is_two_byte_lead_mask,
                                                                          is_three_byte_lead_mask, target);
            if (handled) {
                target += handled, source += handled, source_length -= handled;
                continue;
            }
        }
        // Unlike Ice Lake, pure Latin-1 chunks (German, French) take this handler too: it covers
        // their C2/C3 folds exactly, and there is no separate Latin-1 cascade to fall back onto
        else if (!(lead_families &
                   ~(sz_utf8_fold_haswell_lead_latin_flag_ | sz_utf8_fold_haswell_lead_latin_extended_flag_ |
                     sz_utf8_fold_haswell_lead_e1_flag_))) {
            sz_size_t handled = sz_utf8_case_fold_haswell_latin_chunk_(source_ymm, is_continuation_mask,
                                                                       is_three_byte_lead_mask, target);
            if (handled) {
                target += handled, source += handled, source_length -= handled;
                continue;
            }
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
