/**
 *  @brief Ice Lake (AVX-512) backend for UTF-8 case folding.
 *  @file include/stringzilla/utf8_case_fold/icelake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_case_fold.h
 */
#ifndef STRINGZILLA_UTF8_CASE_FOLD_ICELAKE_H_
#define STRINGZILLA_UTF8_CASE_FOLD_ICELAKE_H_

#include "stringzilla/utf8_case_fold/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_ICELAKE
#if defined(__clang__)
#pragma clang attribute push(                                                                          \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2")
#endif

/**
 *  Helper macros to reduce code duplication in fast-paths.
 *  Detect ASCII uppercase A-Z: returns mask where bytes are in range 0x41-0x5A.
 */
#define sz_icelake_is_ascii_upper_(src_zmm) \
    _mm512_cmplt_epu8_mask(_mm512_sub_epi8((src_zmm), a_upper_vec), subtract26_vec)

/** Apply ASCII case folding (+0x20) to masked positions. */
#define sz_icelake_fold_ascii_(src_zmm, upper_mask, apply_mask) \
    _mm512_mask_add_epi8((src_zmm), (upper_mask) & (apply_mask), (src_zmm), ascii_case_offset)

/** Fold ASCII A-Z in source vector within a prefix mask, returning folded result. */
#define sz_icelake_fold_ascii_in_prefix_(src_zmm, prefix_mask) \
    _mm512_mask_add_epi8((src_zmm), sz_icelake_is_ascii_upper_(src_zmm) & (prefix_mask), (src_zmm), ascii_case_offset)

/**
 *  Georgian uppercase transformation: E1 82/83 XX → E2 B4 YY.
 *  Applies to lead byte positions (sets E2), second byte positions (sets B4),
 *  and adjusts third bytes (-0x20 for 82 sequences, +0x20 for 83 sequences).
 */
#define sz_icelake_transform_georgian_(folded, georgian_leads, is_82_upper, is_83_upper, prefix_mask)          \
    do {                                                                                                       \
        (folded) = _mm512_mask_blend_epi8((georgian_leads), (folded), _mm512_set1_epi8((char)0xE2));           \
        (folded) = _mm512_mask_blend_epi8((georgian_leads) << 1, (folded), _mm512_set1_epi8((char)0xB4));      \
        (folded) = _mm512_mask_sub_epi8((folded), (is_82_upper) & (prefix_mask), (folded), ascii_case_offset); \
        (folded) = _mm512_mask_add_epi8((folded), (is_83_upper) & (prefix_mask), (folded), ascii_case_offset); \
    } while (0)

/**
 *  @brief Find the first invalid position within load_mask, returning chunk_size if all valid.
 *
 *  This safely handles the edge case where all 64 bytes are valid (i.e., ctz(0) which is undefined).
 *  The mask `~is_valid | ~load_mask` marks positions that are either invalid OR outside the loaded chunk.
 *  When all loaded bytes are valid, this mask becomes zero, so we return chunk_size instead of calling ctz(0).
 */
SZ_INTERNAL sz_size_t sz_icelake_first_invalid_(sz_u64_t is_valid, sz_u64_t load_mask, sz_size_t chunk_size) {
    sz_u64_t invalid_mask = ~is_valid | ~load_mask;
    return invalid_mask ? (sz_size_t)sz_u64_ctz(invalid_mask) : chunk_size;
}

SZ_PUBLIC sz_size_t sz_utf8_case_fold_icelake(sz_cptr_t source, sz_size_t source_length, sz_ptr_t target) {
    // This algorithm exploits the idea, that most text in a single ZMM register is either:
    //
    // 1. All ASCII single-byte codepoints
    //    Fast-path: detect A-Z (0x41-0x5A) and add 0x20 to convert to lowercase.
    //
    // 2. Mixture of 2-byte codepoints in one language (continuous range) and 1-byte ASCII codepoints
    //
    //    2.1. Latin-1 Supplement (C3 80-BF): À-ß and à-ÿ.
    //         Folding is trivial +32 to second byte for 80-9E (except × at 0x97).
    //         Special case: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) - handled via mask blend.
    //
    //    2.2. Basic Cyrillic (D0/D1): Ѐ-џ and А-я (U+0400-U+045F).
    //         Folding maps uppercase → lowercase:
    //         - D0 80-8F (Ѐ-Џ) → D1 90-9F (ѐ-џ): +0x10, D0→D1
    //         - D0 90-9F (А-П) → D0 B0-BF (а-п): +0x20
    //         - D0 A0-AF (Р-Я) → D1 80-8F (р-я): −0x20, D0→D1
    //         Excludes Extended-A (D1 A0+) which uses parity (+1) folding.
    //
    //    2.3. Greek (CE/CF): Basic Greek letters Α-Ω and α-ω.
    //         Uppercase is in CE, but lowercase spans both CE and CF:
    //         - CE 91-9F (Α-Ο) → CE B1-BF (α-ο) (+0x20)
    //         - CE A0-AB (Π-Ϋ, skipping A2) → CF 80-8B (π-ϋ) (CE→CF and −0x20)
    //         - CF 82 ('ς' U+03C2) → CF 83 ('σ' U+03C3) (+1)
    //         Excludes archaic letters, tonos/diacritics, and symbol variants needing special handling.
    //
    // 3. Groups of 3-byte codepoints - split into caseless and case-aware paths
    //
    //    3.1. Caseless 3-byte content: CJK (E4-E9), Hangul (EA with safe seconds),
    //         most punctuation (E2 80-83), Thai, Hindi, etc.
    //         Fast-path: copy directly without transformation.
    //
    //    3.2. Georgian uppercase (E1 82 A0-BF, E1 83 80-85/87/8D) → lowercase (E2 B4 80-AF).
    //         Full 3-byte transformation: lead E1→E2, second byte→B4,
    //         third byte ±0x20 depending on original second byte (82 vs 83).
    //
    //    3.3. Fullwidth Latin:
    //         - Ａ-Ｚ: 'Ａ' (U+FF21, EF BC A1) → 'ａ' (U+FF41, EF BD 81) (third byte +0x20, second byte BC→BD)
    //
    // 4. Groups of 4-byte codepoints (emoji, historic scripts)
    //    Generally caseless, but Deseret/Warang Citi have folding rules.
    //    Falls back to serial code for rare case-aware sequences.
    //
    // Within these assumptions, this kernel vectorizes case folding for simple cases, when
    // a sequence of bytes in a certain range can be folded by a single constant addition/subtraction.
    // For more complex cases it periodically switches to serial code.
    //
    // UTF-8 encoding quick reference:
    //
    //   Bytes | Range          | Lead Byte | Pattern
    //   1     | U+0000-007F    | 0xxxxxxx  | ASCII
    //   2     | U+0080-07FF    | 110xxxxx  | C2-DF lead + 1 continuation
    //   3     | U+0800-FFFF    | 1110xxxx  | E0-EF lead + 2 continuations
    //   4     | U+10000-10FFFF | 11110xxx  | F0-F4 lead + 3 continuations
    //
    // Continuation bytes are 10xxxxxx (0x80-0xBF). Lead-byte detection: (byte & 0xC0) == 0x80 marks a
    // continuation, (byte & 0xF0) == 0xE0 a 3-byte lead, and (byte & 0xF8) == 0xF0 a 4-byte lead.
    //
    sz_u512_vec_t source_vec;
    sz_ptr_t target_start = target;

    // Pre-compute constants used in multiple places
    __m512i const indices_vec = _mm512_set_epi8(                        //
        63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, //
        47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, //
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, //
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m512i const a_upper_vec = _mm512_set1_epi8('A');
    __m512i const subtract26_vec = _mm512_set1_epi8(26);
    __m512i const ascii_case_offset = _mm512_set1_epi8(0x20); // Difference between 'A' and 'a'

    // UTF-8 lead byte detection constants:
    // Continuation bytes match: (byte & 0xC0) == 0x80  → pattern 10xxxxxx
    // 3-byte leads match:       (byte & 0xF0) == 0xE0  → pattern 1110xxxx
    // 4-byte leads match:       (byte & 0xF8) == 0xF0  → pattern 11110xxx
    __m512i const utf8_cont_test_mask = _mm512_set1_epi8((char)0xC0);
    __m512i const utf8_cont_pattern = _mm512_set1_epi8((char)0x80);
    __m512i const utf8_3byte_test_mask = _mm512_set1_epi8((char)0xF0);
    __m512i const utf8_3byte_pattern = _mm512_set1_epi8((char)0xE0);
    __m512i const utf8_4byte_test_mask = _mm512_set1_epi8((char)0xF8);

    while (source_length) {
        // Prefetch ahead to hide memory latency on large datasets.
        // This helps when processing multi-GB files that don't fit in cache.
        _mm_prefetch(source + 1024, _MM_HINT_T1); // To L2: 16 cache lines ahead
        _mm_prefetch(source + 512, _MM_HINT_T0);  // To L1: 8 cache lines ahead
        _mm_prefetch(source + 576, _MM_HINT_T0);  // To L1: 9 cache lines ahead
        _mm_prefetch(source + 640, _MM_HINT_T0);  // To L1: 10 cache lines ahead

        sz_size_t chunk_size = sz_min_of_two(source_length, 64);
        __mmask64 load_mask = sz_u64_mask_until_(chunk_size);
        source_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, source);
        __mmask64 is_non_ascii = _mm512_movepi8_mask(source_vec.zmm);

        // FAST PATH: Check for pure ASCII FIRST, before computing any masks.
        // This is the most common case for English and many other Latin-script texts.
        // Avoids computing 6+ masks that would be wasted on pure ASCII chunks.
        if (is_non_ascii == 0) {
            _mm512_mask_storeu_epi8(target, load_mask, sz_icelake_fold_ascii_in_prefix_(source_vec.zmm, load_mask));
            target += chunk_size, source += chunk_size, source_length -= chunk_size;
            continue;
        }

        // Compute lead byte masks only for non-ASCII chunks.
        //
        // Optimization: Use VPTERNLOGD (imm8=0x80 = A & B) instead of VPAND.
        // - VPAND executes on p0/p5 only (2 ports)
        // - VPTERNLOGD executes on p0/p1/p5 (3 ports)
        // - VPCMPEQ-to-mask is p5-only (bottleneck)
        // By moving AND operations to VPTERNLOGD, we reduce p5 contention, allowing
        // the subsequent CMPEQ operations to execute with less stalling.
        //
        // Original pattern:
        //   _mm512_cmpeq_epi8_mask(_mm512_and_si512(src, mask), pattern)
        // Optimized pattern:
        //   _mm512_cmpeq_epi8_mask(_mm512_ternarylogic_epi64(src, mask, mask, 0x80), pattern)
        //
        __m512i masked_cont = _mm512_ternarylogic_epi64(source_vec.zmm, utf8_cont_test_mask, utf8_cont_test_mask, 0x80);
        __m512i masked_3byte =
            _mm512_ternarylogic_epi64(source_vec.zmm, utf8_3byte_test_mask, utf8_3byte_test_mask, 0x80);
        __m512i masked_4byte =
            _mm512_ternarylogic_epi64(source_vec.zmm, utf8_4byte_test_mask, utf8_4byte_test_mask, 0x80);
        __mmask64 is_cont_mask = _mm512_cmpeq_epi8_mask(masked_cont, utf8_cont_pattern);
        __mmask64 is_three_byte_lead_mask = _mm512_cmpeq_epi8_mask(masked_3byte, utf8_3byte_pattern);
        __mmask64 is_four_byte_lead_mask = _mm512_cmpeq_epi8_mask(masked_4byte, utf8_3byte_test_mask);

        // Early fast path: Pure 3-byte content (no ASCII, no 2-byte, no 4-byte)
        // This is common for CJK, Hindi (Devanagari), Thai, etc.
        // Check if all non-ASCII bytes are 3-byte leads or continuations
        {
            __mmask64 is_valid_pure_3byte_mask = is_three_byte_lead_mask | is_cont_mask;
            // Quick check: if all bits match, we have pure 3-byte content
            if ((is_valid_pure_3byte_mask & load_mask) == (is_non_ascii & load_mask) && !is_four_byte_lead_mask) {
                // Check for problematic leads that have case folding:
                //   - E1: Georgian, Greek Extended, Latin Extended Additional
                //   - E2: Glagolitic (B0-B1), Coptic (B2-B3), Letterlike (84 = Kelvin/Angstrom)
                //   - EF: Fullwidth A-Z
                // E2 80-83 are safe (General Punctuation quotes); other E2 ranges have folding
                // EA is mostly safe (Hangul B0-BF) but some second bytes have folding:
                //   - 0x99-0x9F: Cyrillic Ext-B, Latin Ext-D (A640-A7FF)
                //   - 0xAD-0xAE: Cherokee Supplement (AB70-ABBF)
                // Optimization: Range compression for E1/E2 detection.
                // E1 (0xE1) and E2 (0xE2) are adjacent, so instead of 2 CMPEQ operations:
                //   is_e1_mask = CMPEQ(src, E1)  // p5-only
                //   is_e2_mask = CMPEQ(src, E2)  // p5-only
                // We use 1 range check + 1 CMPEQ:
                //   is_e1_e2_mask = (src - E1) < 2  // SUB (any port) + CMPLT (p5)
                //   is_e1_mask = CMPEQ(src, E1)     // p5 (still needed for final check)
                //   is_e2_mask = is_e1_e2_mask & ~is_e1_mask  // KANDN (k-unit, not p5)
                // Net effect: saves 1 p5 operation and enables early-out on combined mask.
                //
                // Original pattern:
                //   is_e1_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE1));
                //   is_e2_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE2));
                //
                __mmask64 is_e1_e2_mask = _mm512_cmplt_epu8_mask(
                    _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xE1)), _mm512_set1_epi8(2));
                __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE1));
                __mmask64 is_ef_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEF));
                // For E2, only allow 80-83 (General Punctuation quotes) through - many other E2 ranges have folding
                // (84 Letterlike, 93 Enclosed Alphanumerics, B0-B3 Glagolitic/Coptic, etc.)
                __mmask64 is_e2_mask = _kand_mask64(is_e1_e2_mask, _knot_mask64(is_e1_mask));
                __mmask64 e2_second_byte_positions = is_e2_mask << 1;
                // E2 folding needed if second byte is NOT in 80-83 range
                __mmask64 is_e2_folding_mask =
                    e2_second_byte_positions &
                    ~_mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x80)),
                                            _mm512_set1_epi8(0x04)); // NOT 80-83
                // For EA, check if second byte is in problematic ranges
                __mmask64 is_ea_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEA));
                __mmask64 ea_second_byte_positions = is_ea_mask << 1;
                // EA 99-9F (second byte 0x99-0x9F) or EA AD-AE (second byte 0xAD-0xAE)
                __mmask64 is_ea_folding_mask =
                    ea_second_byte_positions &
                    (_mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x99)),
                                            _mm512_set1_epi8(0x07)) | // 0x99-0x9F
                     _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xAD)),
                                            _mm512_set1_epi8(0x02))); // 0xAD-0xAE
                if (!(is_e1_mask | is_e2_folding_mask | is_ea_folding_mask | is_ef_mask)) {
                    // Safe 3-byte content (E0, E3-E9, EB-EE) - no 3-byte case folding needed
                    // But ASCII mixed in still needs folding! Use sz_icelake_fold_ascii_in_prefix_.
                    // Just need to avoid splitting a 3-byte sequence at the end.
                    sz_size_t copy_len = chunk_size;
                    // Check if last 1-2 bytes are an incomplete sequence
                    __mmask64 leads_in_chunk_mask = is_three_byte_lead_mask & load_mask;
                    if (leads_in_chunk_mask) {
                        int last_lead_pos = 63 - sz_u64_clz(leads_in_chunk_mask);
                        if (last_lead_pos + 3 > (int)copy_len) copy_len = last_lead_pos;
                    }
                    if (copy_len > 0) {
                        __mmask64 copy_mask = sz_u64_mask_until_(copy_len);
                        _mm512_mask_storeu_epi8(target, copy_mask,
                                                sz_icelake_fold_ascii_in_prefix_(source_vec.zmm, copy_mask));
                        target += copy_len, source += copy_len, source_length -= copy_len;
                        continue;
                    }
                }
            }
        }

        // 2. Two-byte UTF-8 sequences (valid lead bytes C2-DF)
        //
        // 2.1. Latin-1 Supplement (C3 80 - C3 BF) mixed with ASCII
        __mmask64 is_latin1_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xC3));
        __mmask64 latin1_second_byte_positions = is_latin1_lead << 1;
        __mmask64 is_valid_latin1_mix = ~is_non_ascii | is_latin1_lead | latin1_second_byte_positions;
        sz_size_t latin1_length = sz_icelake_first_invalid_(is_valid_latin1_mix, load_mask, chunk_size);
        latin1_length -= latin1_length && ((is_latin1_lead >> (latin1_length - 1)) & 1); // Don't split 2-byte seq

        if (latin1_length >= 2) {
            __mmask64 prefix_mask = sz_u64_mask_until_(latin1_length);
            __mmask64 latin1_second_bytes = latin1_second_byte_positions & prefix_mask;

            // ASCII A-Z (0x41-0x5A) and Latin-1 À-Þ (second byte 0x80-0x9E excl. ×=0x97) both get +0x20
            __mmask64 is_upper_ascii = sz_icelake_is_ascii_upper_(source_vec.zmm);
            __mmask64 is_latin1_upper = _mm512_mask_cmplt_epu8_mask(
                latin1_second_bytes, _mm512_sub_epi8(source_vec.zmm, utf8_cont_pattern), _mm512_set1_epi8(0x1F));
            is_latin1_upper ^= _mm512_mask_cmpeq_epi8_mask(is_latin1_upper, source_vec.zmm,
                                                           _mm512_set1_epi8((char)0x97)); // Exclude ×
            __m512i folded = sz_icelake_fold_ascii_(source_vec.zmm, is_upper_ascii | is_latin1_upper, prefix_mask);

            // 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73): replace both bytes with 's'
            __mmask64 is_eszett_mask =
                _mm512_mask_cmpeq_epi8_mask(latin1_second_bytes, source_vec.zmm, _mm512_set1_epi8((char)0x9F));
            folded = _mm512_mask_set1_epi8(folded, is_eszett_mask | (is_eszett_mask >> 1), 's');

            _mm512_mask_storeu_epi8(target, prefix_mask, folded);
            target += latin1_length, source += latin1_length, source_length -= latin1_length;
            continue;
        }

        // 2.2. Cyrillic fast path (D0/D1 lead bytes for basic Cyrillic 0x0400-0x045F)
        //
        // Cyrillic D0/D1 uppercase → lowercase transformations:
        // Input Range      | Codepoints            | Output              | Transform
        // D0 80-8F         | Ѐ-Џ (U+0400-040F)     | D1 90-9F (ѐ-џ)      | +0x10, D0→D1
        // D0 90-9F         | А-П (U+0410-041F)     | D0 B0-BF (а-п)      | +0x20
        // D0 A0-AF         | Р-Я (U+0420-042F)     | D1 80-8F (р-я)      | −0x20, D0→D1
        // D0 B0-BF         | а-п (lowercase)       | unchanged           | —
        // D1 80-9F         | р-џ (lowercase)       | unchanged           | —
        // D1 A0+           | Extended-A (U+0460+)  | → serial path       | +1 parity
        //
        // EXCLUDED from fast path: Cyrillic Extended-A (0x0460-04FF) which starts at D1 A0.
        // These use +1 folding for even codepoints and must go through the general 2-byte path.
        {
            // Optimization: Range compression for D0/D1 Cyrillic detection.
            // D0 (0xD0) and D1 (0xD1) are adjacent, so instead of 2 CMPEQ + OR:
            //   is_d0_mask = CMPEQ(src, D0)  // p5-only
            //   is_d1_mask = CMPEQ(src, D1)  // p5-only
            //   is_cyrillic_lead_mask = is_d0_mask | is_d1_mask
            // We use 1 range check + 1 CMPEQ:
            //   is_cyrillic_lead_mask = (src - D0) < 2  // SUB (any port) + CMPLT (p5)
            //   is_d0_mask = CMPEQ(src, D0)             // p5 (still needed for is_after_d0_mask)
            //   is_d1_mask = is_cyrillic_lead_mask & ~is_d0_mask  // KANDN (k-unit, not p5)
            // Net effect: saves 1 p5 operation.
            //
            // Original pattern:
            //   is_d0_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD0));
            //   is_d1_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD1));
            //   is_cyrillic_lead_mask = is_d0_mask | is_d1_mask;
            //
            __mmask64 is_cyrillic_lead_mask = _mm512_cmplt_epu8_mask(
                _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xD0)), _mm512_set1_epi8(2));
            __mmask64 is_d0_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD0));
            __mmask64 is_d1_mask = _kand_mask64(is_cyrillic_lead_mask, _knot_mask64(is_d0_mask));
            __mmask64 cyrillic_second_byte_positions = is_cyrillic_lead_mask << 1;

            // Exclude Cyrillic Extended-A: D1 with second byte >= 0xA0 (U+0460+)
            // These have +1 folding rules, not the basic Cyrillic patterns
            __mmask64 is_d1_extended_mask =
                (is_d1_mask << 1) & _mm512_cmpge_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xA0));

            // Check for pure basic Cyrillic + ASCII mix (no extended)
            __mmask64 is_valid_cyrillic_mix_mask =
                ~is_non_ascii | is_cyrillic_lead_mask | cyrillic_second_byte_positions;
            is_valid_cyrillic_mix_mask &= ~is_d1_extended_mask; // Stop at Cyrillic Extended
            sz_size_t cyrillic_length = sz_icelake_first_invalid_(is_valid_cyrillic_mix_mask, load_mask, chunk_size);
            cyrillic_length -= cyrillic_length && ((is_cyrillic_lead_mask >> (cyrillic_length - 1)) & 1);

            if (cyrillic_length >= 2) {
                __mmask64 prefix_mask = sz_u64_mask_until_(cyrillic_length);
                __mmask64 is_after_d0_mask = (is_d0_mask << 1) & prefix_mask;

                // Start with source, apply ASCII folding
                __m512i folded = sz_icelake_fold_ascii_in_prefix_(source_vec.zmm, prefix_mask);

                // D0 second bytes: apply Cyrillic uppercase folding
                // Range 0x80-0x8F (Ѐ-Џ): second byte += 0x10, lead becomes D1
                // Range 0x90-0x9F (А-П): second byte += 0x20, lead stays D0
                // Range 0xA0-0xAF (Р-Я): second byte -= 0x20, lead becomes D1
                __mmask64 is_d0_upper1_mask =
                    _mm512_mask_cmplt_epu8_mask(is_after_d0_mask, source_vec.zmm, _mm512_set1_epi8((char)0x90));
                __mmask64 is_d0_upper2_mask = _mm512_mask_cmplt_epu8_mask(
                    is_after_d0_mask, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x90)),
                    _mm512_set1_epi8(0x10));
                __mmask64 is_d0_upper3_mask = _mm512_mask_cmplt_epu8_mask(
                    is_after_d0_mask, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA0)),
                    _mm512_set1_epi8(0x10));

                // Apply transformations
                // Ѐ-Џ (0x80-0x8F): +0x10 to second byte
                folded = _mm512_mask_add_epi8(folded, is_d0_upper1_mask, folded, _mm512_set1_epi8(0x10));
                // А-П (0x90-0x9F): +0x20 to second byte
                folded = _mm512_mask_add_epi8(folded, is_d0_upper2_mask, folded, _mm512_set1_epi8(0x20));
                // Р-Я (0xA0-0xAF): -0x20 to second byte (wraps to 0x80-0x8F)
                folded = _mm512_mask_sub_epi8(folded, is_d0_upper3_mask, folded, _mm512_set1_epi8(0x20));

                // Fix lead bytes: Ѐ-Џ and Р-Я need D0→D1
                __mmask64 needs_d1_mask = ((is_d0_upper1_mask | is_d0_upper3_mask) >> 1) & (is_d0_mask & prefix_mask);
                folded = _mm512_mask_mov_epi8(folded, needs_d1_mask, _mm512_set1_epi8((char)0xD1));

                _mm512_mask_storeu_epi8(target, prefix_mask, folded);
                target += cyrillic_length, source += cyrillic_length, source_length -= cyrillic_length;
                continue;
            }
        }

        // 2.3. Greek fast path (CE/CF lead bytes for basic Greek 0x0370-0x03FF)
        //
        // Greek CE/CF uppercase → lowercase transformations:
        // Input Range      | Codepoints            | Output              | Transform
        // CE 91-9F         | Α-Ο (U+0391-039F)     | CE B1-BF (α-ο)      | +0x20
        // CE A0-A1         | Π-Ρ (U+03A0-03A1)     | CF 80-81 (π-ρ)      | −0x20, CE→CF
        // CE A3-AB         | Σ-Ϋ (U+03A3-03AB)     | CF 83-8B (σ-ϋ)      | −0x20, CE→CF
        // CE B1-BF         | α-ο (lowercase)       | unchanged           | —
        // CF 80-8B         | π-ϋ (lowercase)       | unchanged           | —
        // CF 82            | ς (U+03C2, final sigma) | CF 83 (σ U+03C3)   | +1
        //
        // EXCLUDED from fast path: Greek tonos/diacritics (CE 84-90) and extended symbols (CF 8C+).
        // These have irregular folding rules and must go through the general 2-byte path.
        {
            __mmask64 is_ce_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xCE));
            __mmask64 is_cf_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xCF));
            __mmask64 is_greek_lead_mask = is_ce_mask | is_cf_mask;
            __mmask64 greek_second_byte_positions = is_greek_lead_mask << 1;

            // Exclude problematic Greek ranges that need serial handling:
            // - CE 84-90: tonos/diacritics with irregular folding (U+0384-0390)
            // - CE 8C, 8E-8F: Ό, Ύ, Ώ with non-standard offsets
            // - CE B0: ΰ (U+03B0) expands to 3 codepoints
            // - CF 8C+: symbols and extended Greek with irregular folding
            // Check second bytes after CE leads
            __mmask64 is_ce_problematic =
                (is_ce_mask << 1) &
                (_mm512_cmplt_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0x91)) | // < 0x91
                 _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xB0))); // == 0xB0 (ΰ expands)
            // Check second bytes after CF leads (only allow 80-8B for basic lowercase + final sigma)
            __mmask64 is_cf_problematic =
                (is_cf_mask << 1) & _mm512_cmpge_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0x8C));

            // Check for pure basic Greek + ASCII mix (no problematic ranges)
            __mmask64 is_valid_greek_mix_mask = ~is_non_ascii | is_greek_lead_mask | greek_second_byte_positions;
            is_valid_greek_mix_mask &= ~(is_ce_problematic | is_cf_problematic);
            sz_size_t greek_length = sz_icelake_first_invalid_(is_valid_greek_mix_mask, load_mask, chunk_size);
            greek_length -= greek_length && ((is_greek_lead_mask >> (greek_length - 1)) & 1);

            if (greek_length >= 2) {
                __mmask64 prefix_mask = sz_u64_mask_until_(greek_length);
                __mmask64 is_after_ce_mask = (is_ce_mask << 1) & prefix_mask;
                __mmask64 is_after_cf_mask = (is_cf_mask << 1) & prefix_mask;

                // Start with source, apply ASCII folding
                __m512i folded = sz_icelake_fold_ascii_in_prefix_(source_vec.zmm, prefix_mask);

                // CE second bytes: Range 91-9F gets +0x20, Range A0-A1 and A3-AB gets -0x20 (lead changes)
                // Note: A2 is unassigned (U+03A2) and must be excluded!
                // 91-9F: (second - 0x91) < 0x0F
                __mmask64 is_ce_upper1_mask = _mm512_mask_cmplt_epu8_mask(
                    is_after_ce_mask, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x91)),
                    _mm512_set1_epi8(0x0F));
                // A0-A1: Π-Ρ (2 chars)
                __mmask64 is_ce_upper2a_mask = _mm512_mask_cmplt_epu8_mask(
                    is_after_ce_mask, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA0)),
                    _mm512_set1_epi8(0x02));
                // A3-AB: Σ-Ϋ (9 chars) - skip A2 which is unassigned
                __mmask64 is_ce_upper2b_mask = _mm512_mask_cmplt_epu8_mask(
                    is_after_ce_mask, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA3)),
                    _mm512_set1_epi8(0x09));
                __mmask64 is_ce_upper2_mask = is_ce_upper2a_mask | is_ce_upper2b_mask;

                // Apply +0x20 for CE 91-9F
                folded = _mm512_mask_add_epi8(folded, is_ce_upper1_mask, folded, _mm512_set1_epi8(0x20));
                // Apply -0x20 for CE A0-AB
                folded = _mm512_mask_sub_epi8(folded, is_ce_upper2_mask, folded, _mm512_set1_epi8(0x20));

                // Fix lead bytes: CE A0-AB need CE→CF
                __mmask64 needs_cf_mask = (is_ce_upper2_mask >> 1) & (is_ce_mask & prefix_mask);
                folded = _mm512_mask_mov_epi8(folded, needs_cf_mask, _mm512_set1_epi8((char)0xCF));

                // Handle final sigma: 'ς' (U+03C2, CF 82) → 'σ' (U+03C3, CF 83)
                __mmask64 is_final_sigma =
                    is_after_cf_mask & _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0x82));
                folded = _mm512_mask_add_epi8(folded, is_final_sigma, folded, _mm512_set1_epi8(1));

                _mm512_mask_storeu_epi8(target, prefix_mask, folded);
                target += greek_length, source += greek_length, source_length -= greek_length;
                continue;
            }
        }

        // 2.4. Fast path for 2-byte scripts without case folding (Hebrew, Arabic, Syriac, etc.)
        //
        // Lead bytes D7-DF cover Hebrew (D7), Arabic (D8-DB), Syriac (DC-DD), Thaana/NKo (DE-DF).
        // None of these scripts have case distinctions, so we can just copy them unchanged.
        // Note: D5/D6 cover Armenian which HAS case folding (including U+0587 which expands).
        __mmask64 is_caseless_2byte = _mm512_cmpge_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD7)) &
                                      _mm512_cmple_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xDF));
        if (is_caseless_2byte) {
            __mmask64 is_caseless_second = is_caseless_2byte << 1;
            __mmask64 is_valid_caseless = ~is_non_ascii | is_caseless_2byte | is_caseless_second;
            sz_size_t caseless_length = sz_icelake_first_invalid_(is_valid_caseless, load_mask, chunk_size);
            caseless_length -= caseless_length && ((is_caseless_2byte >> (caseless_length - 1)) & 1);

            if (caseless_length >= 2) {
                __mmask64 prefix_mask = sz_u64_mask_until_(caseless_length);
                // Fold only ASCII A-Z, copy 2-byte unchanged
                _mm512_mask_storeu_epi8(target, prefix_mask,
                                        sz_icelake_fold_ascii_in_prefix_(source_vec.zmm, prefix_mask));
                target += caseless_length, source += caseless_length, source_length -= caseless_length;
                continue;
            }
        }

        // 2.5. Other 2-byte scripts (Latin Extended, Greek, Cyrillic, Armenian)
        //
        // Unlike Latin-1 where folding is a simple +0x20 to the second byte in-place, these scripts
        // require unpacking to 32-bit codepoints because:
        //   - Different scripts have different folding offsets (+0x20 for Cyrillic/Greek, +0x30 for Armenian, +0x50 for
        //   Cyrillic Ѐ-Џ)
        //   - Latin Extended-A uses parity-based rules (even codepoints → +1, odd → +1 in different ranges)
        //   - Some codepoints expand (İ→iı, ŉ→ʼn) requiring serial handling
        //
        // Strategy: Compress character start positions, gather lead/continuation bytes, expand to 32-bit,
        // decode UTF-8 to codepoints, apply vectorized folding rules, re-encode to UTF-8, scatter back
        // using expand instructions. Processes up to 16 characters per iteration.
        //
        // Detect any 2-byte lead (110xxxxx = 0xC0-0xDF) except C3 (Latin-1, already handled above)
        __mmask64 is_two_byte_lead = _mm512_cmpeq_epi8_mask(
            _mm512_and_si512(source_vec.zmm, _mm512_set1_epi8((char)0xE0)), _mm512_set1_epi8((char)0xC0));
        is_two_byte_lead &= ~is_latin1_lead; // Exclude C3
        __mmask64 two_byte_second_positions = is_two_byte_lead << 1;

        // Accept ALL 2-byte sequences; we'll detect singletons after decoding
        __mmask64 is_valid_two_byte_mix = ~is_non_ascii | is_two_byte_lead | two_byte_second_positions;
        sz_size_t two_byte_length = sz_icelake_first_invalid_(is_valid_two_byte_mix, load_mask, chunk_size);
        two_byte_length -= two_byte_length && ((is_two_byte_lead >> (two_byte_length - 1)) & 1);

        if (two_byte_length >= 2) {
            __mmask64 prefix_mask = sz_u64_mask_until_(two_byte_length);
            __mmask64 is_char_start = (~is_non_ascii | is_two_byte_lead) & prefix_mask;
            sz_size_t num_chars = (sz_size_t)sz_u64_popcount(is_char_start);

            // Compress character start positions, gather first/second bytes
            sz_u512_vec_t char_indices;
            char_indices.zmm = _mm512_maskz_compress_epi8(is_char_start, indices_vec);

            // We can only process 16 chars at a time (one ZMM register of 32-bit values)
            // If we have more, truncate the prefix to only include those 16 chars
            if (num_chars > 16) {
                sz_u8_t last_char_idx = char_indices.u8s[15]; // 16th char's byte position
                two_byte_length = last_char_idx + ((is_two_byte_lead >> last_char_idx) & 1 ? 2 : 1);
                prefix_mask = sz_u64_mask_until_(two_byte_length);
                is_char_start &= prefix_mask;
                num_chars = 16;
            }

            sz_u512_vec_t first_bytes, second_bytes;
            first_bytes.zmm = _mm512_permutexvar_epi8(char_indices.zmm, source_vec.zmm);
            second_bytes.zmm =
                _mm512_permutexvar_epi8(_mm512_add_epi8(char_indices.zmm, _mm512_set1_epi8(1)), source_vec.zmm);

            // Expand to 32-bit for arithmetic
            __m512i first_wide = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(first_bytes.zmm));
            __m512i second_wide = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(second_bytes.zmm));
            __mmask16 is_two_byte_char = (__mmask16)_pext_u64(is_two_byte_lead & prefix_mask, is_char_start);

            // Decode: ASCII as-is, 2-byte as ((first & 0x1F) << 6) | (second & 0x3F)
            __m512i decoded =
                _mm512_or_si512(_mm512_slli_epi32(_mm512_and_si512(first_wide, _mm512_set1_epi32(0x1F)), 6),
                                _mm512_and_si512(second_wide, _mm512_set1_epi32(0x3F)));
            __m512i codepoints = _mm512_mask_blend_epi32(is_two_byte_char, first_wide, decoded);

            // Detect codepoints that need serial handling - ONLY ranges with case folding that
            // our vectorized rules don't handle correctly. Ranges without case folding (Arabic,
            // Hebrew, etc.) can pass through unchanged since no folding rules will match.
            __mmask16 needs_serial =
                // Specific singletons/expanders in Latin Extended-A:
                _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x0130)) | // İ (expands)
                _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x0149)) | // ŉ (expands)
                _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x0178)) | // Ÿ → ÿ (0x00FF)
                _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x017F)) | // ſ → s (ASCII)
                // Latin Extended-B: 0x0180-0x024F (irregular case folding patterns)
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0180)),
                                        _mm512_set1_epi32(0x00D0)) | // 0x0180-0x024F
                // Greek singletons with non-standard folding: 0x0345-0x0390
                // (Basic Greek Α-Ω 0x0391-0x03A9 and α-ω 0x03B1-0x03C9 are handled by vectorized +0x20 rules)
                // (Final sigma ς 0x03C2 → σ is also vectorized)
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0345)),
                                        _mm512_set1_epi32(0x4C)) | // 0x0345-0x0390
                // Greek ΰ (0x03B0) expands to 3 codepoints
                _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x03B0)) |
                // Greek symbols with non-standard folding: 0x03CF-0x03FF
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x03CF)),
                                        _mm512_set1_epi32(0x31)) | // 0x03CF-0x03FF
                // Cyrillic Extended: 0x0460-0x052F (has case folding, not covered)
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0460)),
                                        _mm512_set1_epi32(0x00D0)) | // 0x0460-0x052F
                // Armenian ligature that expands: U+0587 (և → եւ, ech + yiwn)
                _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x0587));

            // Only consider characters we're actually processing
            needs_serial &= (__mmask16)((1u << num_chars) - 1);

            // Handle serial-needed characters by processing them one at a time (assumes valid UTF-8)
            if (needs_serial) {
                sz_size_t first_special = (sz_size_t)sz_u64_ctz((sz_u64_t)needs_serial);
                if (first_special == 0) {
                    // First character needs serial - process it and continue the main loop
                    sz_rune_t rune;
                    sz_rune_length_t rune_length;
                    sz_rune_parse(source, &rune, &rune_length);
                    sz_rune_t folded_runes[3]; // Unicode case folding produces at most 3 runes
                    sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
                    for (sz_size_t i = 0; i != folded_count; ++i)
                        target += sz_rune_export(folded_runes[i], (sz_u8_t *)target);
                    source += rune_length;
                    source_length -= rune_length;
                    continue;
                }
                // Truncate to only process characters before the special one
                num_chars = first_special;
                // Recalculate byte length from character positions
                sz_u8_t last_char_idx = char_indices.u8s[num_chars - 1];
                two_byte_length = last_char_idx + ((is_two_byte_lead >> last_char_idx) & 1 ? 2 : 1);
                prefix_mask = sz_u64_mask_until_(two_byte_length);
                is_char_start &= prefix_mask;
                is_two_byte_char = (__mmask16)_pext_u64(is_two_byte_lead & prefix_mask, is_char_start);
            }

            // Apply folding rules - all use range check: (cp - base) < size
            __m512i folded = codepoints;

            // ASCII A-Z: 0x0041-0x005A → +0x20
            folded = _mm512_mask_add_epi32(
                folded,
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0041)), _mm512_set1_epi32(26)),
                folded, _mm512_set1_epi32(0x20));
            // Cyrillic А-Я: 0x0410-0x042F → +0x20
            folded =
                _mm512_mask_add_epi32(folded,
                                      _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0410)),
                                                              _mm512_set1_epi32(0x20)),
                                      folded, _mm512_set1_epi32(0x20));
            // Cyrillic Ѐ-Џ: 0x0400-0x040F → +0x50
            folded =
                _mm512_mask_add_epi32(folded,
                                      _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0400)),
                                                              _mm512_set1_epi32(0x10)),
                                      folded, _mm512_set1_epi32(0x50));
            // Greek Α-Ρ: 0x0391-0x03A1 → +0x20
            folded =
                _mm512_mask_add_epi32(folded,
                                      _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0391)),
                                                              _mm512_set1_epi32(0x11)),
                                      folded, _mm512_set1_epi32(0x20));
            // Greek Σ-Ϋ: 0x03A3-0x03AB → +0x20
            folded =
                _mm512_mask_add_epi32(folded,
                                      _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x03A3)),
                                                              _mm512_set1_epi32(0x09)),
                                      folded, _mm512_set1_epi32(0x20));
            // Armenian Ա-Ֆ: 0x0531-0x0556 → +0x30
            folded =
                _mm512_mask_add_epi32(folded,
                                      _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0531)),
                                                              _mm512_set1_epi32(0x26)),
                                      folded, _mm512_set1_epi32(0x30));
            // Latin Extended-A/B: complex alternating pattern with varying parity
            // Use _mm512_test_epi32_mask for cleaner parity check: returns 1 where (a & b) != 0
            __mmask16 is_odd = _mm512_test_epi32_mask(codepoints, _mm512_set1_epi32(1));
            __mmask16 is_even = ~is_odd;
            // Ranges where EVEN is uppercase (even → +1):
            // 0x0100-0x012F, 0x0132-0x0137, 0x014A-0x0177
            __mmask16 is_latin_even_upper =
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0100)),
                                        _mm512_set1_epi32(0x30)) |
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0132)),
                                        _mm512_set1_epi32(0x06)) |
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x014A)),
                                        _mm512_set1_epi32(0x2E));
            folded = _mm512_mask_add_epi32(folded, is_latin_even_upper & is_even, folded, _mm512_set1_epi32(1));
            // Ranges where ODD is uppercase (odd → +1):
            // 0x0139-0x0148, 0x0179-0x017E
            __mmask16 is_latin_odd_upper =
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0139)),
                                        _mm512_set1_epi32(0x10)) |
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0179)),
                                        _mm512_set1_epi32(0x06));
            folded = _mm512_mask_add_epi32(folded, is_latin_odd_upper & is_odd, folded, _mm512_set1_epi32(1));
            // Special case: µ (U+00B5 MICRO SIGN) → μ (U+03BC GREEK SMALL LETTER MU)
            folded = _mm512_mask_mov_epi32(folded, _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x00B5)),
                                           _mm512_set1_epi32(0x03BC));
            // Special case: ς (U+03C2 GREEK SMALL LETTER FINAL SIGMA) → σ (U+03C3)
            folded = _mm512_mask_mov_epi32(folded, _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x03C2)),
                                           _mm512_set1_epi32(0x03C3));

            // Re-encode to UTF-8: lead = 0xC0 | (cp >> 6), second = 0x80 | (cp & 0x3F)
            __m512i new_lead = _mm512_or_si512(_mm512_set1_epi32(0xC0), _mm512_srli_epi32(folded, 6));
            __m512i new_second =
                _mm512_or_si512(_mm512_set1_epi32(0x80), _mm512_and_si512(folded, _mm512_set1_epi32(0x3F)));
            // ASCII stays single-byte
            __mmask16 is_ascii_out = _mm512_cmplt_epu32_mask(folded, _mm512_set1_epi32(0x80));
            new_lead = _mm512_mask_blend_epi32(is_ascii_out, new_lead, folded);

            // Scatter back using expand (inverse of compress) - no scalar loop needed!
            // Pack 32-bit results to bytes
            __m512i lead_zmm = _mm512_zextsi128_si512(_mm512_cvtepi32_epi8(new_lead));
            __m512i second_zmm = _mm512_zextsi128_si512(_mm512_cvtepi32_epi8(new_second));

            // Expand lead bytes to char start positions
            __m512i result = _mm512_mask_expand_epi8(source_vec.zmm, is_char_start, lead_zmm);

            // For second bytes: compress only 2-byte chars' seconds, then expand to continuation positions
            __m512i second_compressed = _mm512_maskz_compress_epi8((__mmask64)is_two_byte_char, second_zmm);
            result = _mm512_mask_expand_epi8(result, two_byte_second_positions & prefix_mask, second_compressed);

            _mm512_mask_storeu_epi8(target, prefix_mask, result);
            target += two_byte_length, source += two_byte_length, source_length -= two_byte_length;
            continue;
        }

        // 3. Handle 3-byte sequences (E0-EF leads), possibly mixed with ASCII
        //
        // Most 3-byte codepoints have NO case folding, making this a fast-path copy:
        //   - CJK Unified Ideographs (U+4E00-9FFF, U+3400-4DBF): E4-E9, some E3
        //   - Hangul Syllables (U+AC00-D7AF): EA-ED
        //   - General Punctuation, Symbols, Currency, etc.
        //
        // Exceptions requiring special handling:
        //   - Latin Extended Additional (U+1E00-1EFF = E1 B8-BB): +1 folding for even → vectorized
        //   - Greek Extended (U+1F00-1FFF = E1 BC-BF): Complex expansions → serial
        //   - Capital Eszett (U+1E9E = E1 BA 9E): Expands to "ss" → serial
        //   - Fullwidth A-Z (U+FF21-FF3A = EF BC A1-BA): +0x20 to third byte → vectorized
        //   - Cyrillic Ext-B, Cherokee (EA 99-9F, AD-AE): +1 folding → serial
        //
        // Strategy:
        //   - Fast path: ASCII + safe 3-byte (E0, E3-E9, EB-EE, most of EA) → fold ASCII, copy 3-byte
        //   - E1 path: Check for Latin Ext Add (vectorized) vs Greek Extended (serial)
        //   - EF path: Check for Fullwidth A-Z (vectorized)
        {
            // Check for any byte that prevents the fast path
            // Problematic bytes: C0-DF (2-byte), E1, E2, EF, F0-FF (4-byte)
            // EA is mostly safe (Hangul B0-BF) but some second bytes have folding:
            //   - 0x99-0x9F: Cyrillic Ext-B, Latin Ext-D (A640-A7FF)
            //   - 0xAD-0xAE: Cherokee Supplement (AB70-ABBF)
            __mmask64 is_two_byte_lead =
                _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xC0)),
                                       _mm512_set1_epi8(0x20)); // C0-DF
            __mmask64 is_e1_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE1));
            __mmask64 is_e2_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE2));
            __mmask64 is_ef_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEF));
            // For EA, only flag as complex if second byte is in problematic ranges
            __mmask64 is_ea_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEA));
            __mmask64 ea_second_byte_positions = is_ea_lead << 1;
            __mmask64 is_ea_complex =
                ea_second_byte_positions &
                (_mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x99)),
                                        _mm512_set1_epi8(0x07)) | // 0x99-0x9F
                 _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xAD)),
                                        _mm512_set1_epi8(0x02))); // 0xAD-0xAE
            __mmask64 has_complex =
                (is_two_byte_lead | is_four_byte_lead_mask | is_e1_lead | is_e2_lead | is_ea_complex | is_ef_lead) &
                load_mask;

            // Fast path: No complex bytes - just fold ASCII A-Z and copy everything else
            if (!has_complex) {
                // All bytes are ASCII or safe 3-byte (E0, E3-E9, EB-EE range)
                // Accept: ASCII, safe 3-byte leads (E0, E3-E9, EB-EE), continuations
                __mmask64 is_valid = ~is_non_ascii | is_three_byte_lead_mask | is_cont_mask;
                sz_size_t valid_length = sz_icelake_first_invalid_(is_valid, load_mask, chunk_size);

                // Don't split a 3-byte sequence at chunk boundary.
                // Note: `>= 1` is an optimization, not redundant! When valid_length == 0, the mask
                // operations below are no-ops, but skipping them saves ~5% throughput on mixed content.
                if (valid_length >= 1) {
                    __mmask64 all_leads = is_three_byte_lead_mask & sz_u64_mask_until_(valid_length);
                    __mmask64 safe_mask = valid_length >= 3 ? sz_u64_mask_until_(valid_length - 2) : 0;
                    __mmask64 unsafe = all_leads & ~safe_mask;
                    if (unsafe) valid_length = sz_u64_ctz(unsafe);
                }

                if (valid_length >= 2) {
                    __mmask64 mask = sz_u64_mask_until_(valid_length);
                    // Fold ASCII A-Z, copy everything else unchanged
                    _mm512_mask_storeu_epi8(target, mask, sz_icelake_fold_ascii_in_prefix_(source_vec.zmm, mask));
                    target += valid_length, source += valid_length, source_length -= valid_length;
                    continue;
                }
            }

            // 3.1. Georgian fast path: handles E1 82/83 content
            //
            // Georgian script uses E1 82 and E1 83 lead sequences:
            //   - E1 82 A0-BF: Uppercase (Ⴀ-Ⴟ) - folds to E2 B4 80-9F
            //   - E1 83 80-85: Uppercase (Ⴠ-Ⴥ) - folds to E2 B4 A0-A5
            //   - E1 83 86-BF: Lowercase/other (ა-ჿ) - no folding needed
            //
            // We include ALL E1 82/83 content in the fast path, but only transform uppercase.
            if (is_e1_lead && source_length >= 3) {
                // Check if E1 leads have Georgian second bytes (82 or 83)
                __m512i georgian_second_bytes =
                    _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(1)), source_vec.zmm);

                // Check for Georgian second bytes: 82 or 83
                // Only check positions where second byte is within chunk (positions 0-62 for 64-byte chunk)
                // Position 63's second byte would wrap around in the permutation
                __mmask64 safe_e1_mask = is_e1_lead & (load_mask >> 1);
                __mmask64 is_82_at_e1 =
                    _mm512_mask_cmpeq_epi8_mask(safe_e1_mask, georgian_second_bytes, _mm512_set1_epi8((char)0x82));
                __mmask64 is_83_at_e1 =
                    _mm512_mask_cmpeq_epi8_mask(safe_e1_mask, georgian_second_bytes, _mm512_set1_epi8((char)0x83));
                __mmask64 is_georgian_e1 = is_82_at_e1 | is_83_at_e1;

                // Check if all E1 leads are Georgian (82/83) with no mixed content.
                // Note: Both conditions are necessary, not redundant!
                // - `!non_georgian_e1`: ensures ALL E1 leads are Georgian (no Greek Extended, etc.)
                // - `is_georgian_e1`: ensures at least ONE Georgian exists (handles empty safe_e1_mask)
                __mmask64 non_georgian_e1 = safe_e1_mask & ~is_georgian_e1;
                if (!non_georgian_e1 && is_georgian_e1) {
                    // All Georgian 3-byte sequences are valid (E1 82 80-BF, E1 83 80-BF)
                    // We only transform the uppercase subset, rest passes through

                    // Find uppercase positions that need transformation:
                    // - E1 82 A0-BF: third byte in A0-BF range (U+10A0-10BF)
                    // - E1 83 80-85: third byte in 80-85 range (U+10C0-10C5)
                    // - E1 83 87: third byte = 87 (U+10C7)
                    // - E1 83 8D: third byte = 8D (U+10CD)
                    __mmask64 third_pos_82 = is_82_at_e1 << 2;
                    __mmask64 third_pos_83 = is_83_at_e1 << 2;

                    __mmask64 is_82_uppercase = _mm512_mask_cmplt_epu8_mask(
                        third_pos_82 & load_mask, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA0)),
                        _mm512_set1_epi8(0x20));
                    // For E1 83: check 80-85, 87, or 8D
                    __mmask64 is_83_range = _mm512_mask_cmplt_epu8_mask(
                        third_pos_83 & load_mask, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x80)),
                        _mm512_set1_epi8(0x06));
                    __mmask64 is_83_c7 = _mm512_mask_cmpeq_epi8_mask(third_pos_83 & load_mask, source_vec.zmm,
                                                                     _mm512_set1_epi8((char)0x87));
                    __mmask64 is_83_cd = _mm512_mask_cmpeq_epi8_mask(third_pos_83 & load_mask, source_vec.zmm,
                                                                     _mm512_set1_epi8((char)0x8D));
                    __mmask64 is_83_uppercase = is_83_range | is_83_c7 | is_83_cd;

                    // Include ASCII, ALL Georgian E1 (not just uppercase), safe E2, continuations, safe EA
                    // E2 80-83 are safe (General Punctuation), others have case folding (Glagolitic, Coptic, etc.)
                    // Also include C2 leads (Latin-1 Supplement: U+0080-00BF) - no case folding needed
                    __mmask64 is_safe_ea = is_ea_lead & ~(is_ea_complex >> 1);
                    __mmask64 is_c2_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xC2));
                    // E2 is only safe if second byte is 80-83 (General Punctuation quotes)
                    __m512i second_bytes_for_e2 =
                        _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(1)), source_vec.zmm);
                    __mmask64 safe_e2_positions = is_e2_lead & (load_mask >> 1); // Ensure second byte is in chunk
                    __mmask64 is_safe_e2 =
                        safe_e2_positions &
                        _mm512_mask_cmplt_epu8_mask(safe_e2_positions,
                                                    _mm512_sub_epi8(second_bytes_for_e2, _mm512_set1_epi8((char)0x80)),
                                                    _mm512_set1_epi8(0x04)); // 80-83
                    __mmask64 is_valid_georgian_mix =
                        ~is_non_ascii | is_georgian_e1 | is_safe_e2 | is_cont_mask | is_safe_ea | is_c2_lead;
                    // Exclude other 2-byte leads (C3-DF may need folding), 4-byte, EF, and unsafe E2
                    __mmask64 is_foldable_2byte = is_two_byte_lead & ~is_c2_lead;
                    __mmask64 is_unsafe_e2 = is_e2_lead & ~is_safe_e2;
                    is_valid_georgian_mix &= ~(is_foldable_2byte | is_four_byte_lead_mask | is_ef_lead | is_unsafe_e2);
                    sz_size_t georgian_length = sz_icelake_first_invalid_(is_valid_georgian_mix, load_mask, chunk_size);

                    // Don't split multi-byte sequences (2-byte C2, 3-byte E1/E2/EA).
                    // Note: `>= 1` is an optimization - skips mask ops when nothing valid.
                    if (georgian_length >= 1) {
                        __mmask64 prefix = sz_u64_mask_until_(georgian_length);
                        // Check for incomplete 3-byte sequences (leads in last 2 positions)
                        __mmask64 leads3_in_prefix = is_three_byte_lead_mask & prefix;
                        __mmask64 safe3_mask = georgian_length >= 3 ? sz_u64_mask_until_(georgian_length - 2) : 0;
                        __mmask64 unsafe3 = leads3_in_prefix & ~safe3_mask;
                        // Check for incomplete 2-byte sequences (leads in last position)
                        __mmask64 leads2_in_prefix = is_c2_lead & prefix;
                        __mmask64 safe2_mask = georgian_length >= 2 ? sz_u64_mask_until_(georgian_length - 1) : 0;
                        __mmask64 unsafe2 = leads2_in_prefix & ~safe2_mask;
                        __mmask64 unsafe = unsafe3 | unsafe2;
                        if (unsafe) georgian_length = sz_u64_ctz(unsafe);
                    }

                    if (georgian_length >= 2) {
                        __mmask64 prefix_mask = sz_u64_mask_until_(georgian_length);

                        // Find uppercase leads that need transformation within prefix
                        __mmask64 uppercase_leads = ((is_82_uppercase | is_83_uppercase) >> 2) & is_georgian_e1;
                        uppercase_leads &= prefix_mask;

                        // Transform only uppercase Georgian: E1 82/83 XX → E2 B4 YY
                        __m512i folded = source_vec.zmm;

                        // Set lead bytes to E2 where uppercase Georgian
                        folded = _mm512_mask_blend_epi8(uppercase_leads, folded, _mm512_set1_epi8((char)0xE2));

                        // Set second bytes to B4 where uppercase Georgian
                        __mmask64 uppercase_second_pos = uppercase_leads << 1;
                        folded = _mm512_mask_blend_epi8(uppercase_second_pos, folded, _mm512_set1_epi8((char)0xB4));

                        // Adjust third bytes for uppercase only: -0x20 for 82, +0x20 for 83
                        __mmask64 prefix_82_upper = is_82_uppercase & prefix_mask;
                        __mmask64 prefix_83_upper = is_83_uppercase & prefix_mask;
                        folded = _mm512_mask_sub_epi8(folded, prefix_82_upper, folded, _mm512_set1_epi8(0x20));
                        folded = _mm512_mask_add_epi8(folded, prefix_83_upper, folded, _mm512_set1_epi8(0x20));

                        // Also fold ASCII A-Z
                        folded = _mm512_mask_add_epi8(folded, sz_icelake_is_ascii_upper_(source_vec.zmm) & prefix_mask,
                                                      folded, ascii_case_offset);

                        // Fold Micro Sign: 'µ' (U+00B5, C2 B5) → 'μ' (U+03BC, CE BC)
                        __mmask64 c2_in_prefix = is_c2_lead & prefix_mask;
                        __mmask64 c2_second_pos = c2_in_prefix << 1;
                        __mmask64 is_micro_second =
                            c2_second_pos & _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xB5));
                        __mmask64 is_micro_lead = is_micro_second >> 1;
                        folded = _mm512_mask_blend_epi8(is_micro_lead, folded, _mm512_set1_epi8((char)0xCE));
                        folded = _mm512_mask_blend_epi8(is_micro_second, folded, _mm512_set1_epi8((char)0xBC));

                        _mm512_mask_storeu_epi8(target, prefix_mask, folded);
                        target += georgian_length, source += georgian_length, source_length -= georgian_length;
                        continue;
                    }
                }
            }

            // 3.2. Latin Extended Additional fast path: E1 B8-BB (U+1E00-1EFF)
            //
            // Vietnamese and other Latin Extended Additional characters use parity folding:
            //   - Even codepoints are uppercase → +1 to get lowercase
            //   - Odd codepoints are lowercase → no change
            // UTF-8 third byte determines parity: even third byte = even codepoint.
            //
            // Exceptions that need serial handling:
            //   - U+1E96–U+1E9E (E1 BA 96-9E):
            //     - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
            //     - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
            //     - 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
            //     - 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
            //     - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
            //     - 'ẛ' (U+1E9B, E1 BA 9B) → 'ṡ' (U+1E61, E1 B9 A1)
            //     - 'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73)
            if (is_e1_lead && source_length >= 3) {
                __m512i latin_ext_second_bytes =
                    _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(1)), source_vec.zmm);
                __m512i latin_ext_third_bytes =
                    _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(2)), source_vec.zmm);

                // Check for Latin Ext Add second bytes: B8, B9, BA, BB
                __mmask64 safe_e1_mask = is_e1_lead & (load_mask >> 1);
                __mmask64 is_latin_ext_e1 = _mm512_mask_cmplt_epu8_mask(
                    safe_e1_mask, _mm512_sub_epi8(latin_ext_second_bytes, _mm512_set1_epi8((char)0xB8)),
                    _mm512_set1_epi8(0x04)); // 0xB8-0xBB

                // Check for U+1E96-U+1E9E (E1 BA 96-9E) which expand to multiple codepoints
                __mmask64 is_ba_second =
                    _mm512_mask_cmpeq_epi8_mask(safe_e1_mask, latin_ext_second_bytes, _mm512_set1_epi8((char)0xBA));
                // Third byte in range 0x96-0x9E (9 values: 0x96 + 0..8)
                // Note: latin_ext_third_bytes holds third byte value at lead positions, so compare at is_ba_second
                // position
                __mmask64 is_special_third =
                    is_ba_second &
                    _mm512_mask_cmplt_epu8_mask(is_ba_second,
                                                _mm512_sub_epi8(latin_ext_third_bytes, _mm512_set1_epi8((char)0x96)),
                                                _mm512_set1_epi8(0x09));

                // All E1 leads must be Latin Ext Add (no Greek Ext BC-BF, Georgian 82/83)
                __mmask64 non_latin_ext_e1 = safe_e1_mask & ~is_latin_ext_e1;
                if (!non_latin_ext_e1 && is_latin_ext_e1 && !is_special_third) {
                    // Pure Latin Extended Additional content
                    // Accept ASCII + Latin Ext Add E1 + continuations
                    __mmask64 is_valid_latin_ext = ~is_non_ascii | is_latin_ext_e1 | is_cont_mask;
                    is_valid_latin_ext &= ~(is_four_byte_lead_mask | is_ef_lead);
                    sz_size_t latin_ext_length = sz_icelake_first_invalid_(is_valid_latin_ext, load_mask, chunk_size);

                    // Don't split 3-byte sequences
                    if (latin_ext_length >= 1) {
                        __mmask64 prefix = sz_u64_mask_until_(latin_ext_length);
                        __mmask64 leads_in_prefix = is_three_byte_lead_mask & prefix;
                        __mmask64 safe_mask = latin_ext_length >= 3 ? sz_u64_mask_until_(latin_ext_length - 2) : 0;
                        __mmask64 unsafe = leads_in_prefix & ~safe_mask;
                        if (unsafe) latin_ext_length = sz_u64_ctz(unsafe);
                    }

                    if (latin_ext_length >= 3) {
                        __mmask64 prefix_mask = sz_u64_mask_until_(latin_ext_length);

                        // Find third byte positions of Latin Ext Add chars
                        __mmask64 third_positions = (is_latin_ext_e1 & prefix_mask) << 2;
                        third_positions &= prefix_mask;

                        // Check parity: even third byte = uppercase (fold +1)
                        __mmask64 is_even_third =
                            ~_mm512_test_epi8_mask(source_vec.zmm, _mm512_set1_epi8(0x01)) & third_positions;

                        // Apply folding
                        __m512i folded = source_vec.zmm;
                        folded = _mm512_mask_add_epi8(folded, is_even_third, folded, _mm512_set1_epi8(0x01));

                        // Also fold ASCII A-Z
                        folded = _mm512_mask_add_epi8(folded, sz_icelake_is_ascii_upper_(source_vec.zmm) & prefix_mask,
                                                      folded, ascii_case_offset);

                        _mm512_mask_storeu_epi8(target, prefix_mask, folded);
                        target += latin_ext_length, source += latin_ext_length, source_length -= latin_ext_length;
                        continue;
                    }
                }
            }

            // 3.4. Slow Path: Mixed 3-byte content with E1/E2/EF leads
            // This path handles remaining 3-byte content that couldn't use a fast path.
            // Sub-sections:
            //   • E1 Greek Extended / Capital Eszett detection → serial fallback
            //   • E2 validation (Glagolitic, Coptic, Letterlike) → serial fallback
            //   • Latin Extended Additional parity folding (vectorized)
            //   • Georgian transformation (vectorized)
            //   • Fullwidth A-Z (EF BC A1-BA) (vectorized)
            //
            // EA with problematic second bytes (is_ea_complex) also needs special handling
            // But plain EA (Hangul) is safe
            __mmask64 is_ea_lead_complex = is_ea_complex >> 1; // Shift back to lead position
            __mmask64 is_safe_three_byte_lead =
                is_three_byte_lead_mask & ~is_e1_lead & ~is_e2_lead & ~is_ea_lead_complex & ~is_ef_lead;
            __mmask64 is_valid_mixed = ~is_non_ascii | is_safe_three_byte_lead | is_cont_mask;
            is_valid_mixed &= ~is_four_byte_lead_mask;
            sz_size_t three_byte_length = sz_icelake_first_invalid_(is_valid_mixed, load_mask, chunk_size);

            // Don't split a 3-byte sequence: find first incomplete lead and truncate there.
            // Note: `>= 1` is an optimization - skips mask ops when nothing valid.
            if (three_byte_length >= 1) {
                __mmask64 all_leads = is_three_byte_lead_mask & sz_u64_mask_until_(three_byte_length);
                __mmask64 safe_leads_mask = three_byte_length >= 3 ? sz_u64_mask_until_(three_byte_length - 2) : 0;
                __mmask64 unsafe_leads = all_leads & ~safe_leads_mask;
                if (unsafe_leads) three_byte_length = sz_u64_ctz(unsafe_leads);
            }

            // Need at least 2 bytes
            if (three_byte_length >= 2) {
                __mmask64 prefix_mask_3 = sz_u64_mask_until_(three_byte_length);
                __mmask64 three_byte_leads_in_prefix = is_three_byte_lead_mask & prefix_mask_3;

                // Check for problematic lead bytes in this prefix

                // E2 leads: E2 80-83 are safe (General Punctuation), but others need folding.
                // For safety and simplicity, treating ALL E2s here as unsafe forces serial fallback,
                // which handles both folding and identity cases correctly.
                __mmask64 is_unsafe_e2 = is_e2_lead & three_byte_leads_in_prefix;
                __mmask64 problematic_leads = (is_e1_lead | is_ef_lead | is_unsafe_e2) & three_byte_leads_in_prefix;

                if (!problematic_leads) {
                    // No E1 or EF leads in prefix - fold ASCII A-Z, copy 3-byte chars unchanged
                    _mm512_mask_storeu_epi8(target, prefix_mask_3,
                                            sz_icelake_fold_ascii_in_prefix_(source_vec.zmm, prefix_mask_3));
                    target += three_byte_length, source += three_byte_length, source_length -= three_byte_length;
                    continue;
                }

                // Check if E1 leads need special handling:
                // - Greek Extended (E1 BC-BF = 1F00-1FFF): complex expansions
                // - Capital Eszett (E1 BA 9E = U+1E9E): expands to "ss"
                __mmask64 is_e1_in_prefix = is_e1_lead & three_byte_leads_in_prefix;
                if (is_e1_in_prefix) {
                    __m512i e1_second_bytes =
                        _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(1)), source_vec.zmm);
                    __m512i e1_third_bytes =
                        _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(2)), source_vec.zmm);

                    // Check for Greek Extended (second byte BC-BF)
                    __mmask64 is_greek_ext = _mm512_mask_cmplt_epu8_mask(
                        is_e1_in_prefix, _mm512_sub_epi8(e1_second_bytes, _mm512_set1_epi8((char)0xBC)),
                        _mm512_set1_epi8(0x04));

                    // Check for U+1E96-U+1E9E (E1 BA 96-9E) which expand to multiple codepoints
                    __mmask64 is_ba_second =
                        _mm512_mask_cmpeq_epi8_mask(is_e1_in_prefix, e1_second_bytes, _mm512_set1_epi8((char)0xBA));
                    // Third byte in range 0x96-0x9E (9 values)
                    __mmask64 is_special_third = _mm512_mask_cmplt_epu8_mask(
                        is_ba_second, _mm512_sub_epi8(e1_third_bytes, _mm512_set1_epi8((char)0x96)),
                        _mm512_set1_epi8(0x09));
                    __mmask64 needs_serial_e1 = is_greek_ext | is_special_third;

                    if (needs_serial_e1) {
                        sz_size_t first_special = sz_u64_ctz(needs_serial_e1);
                        if (first_special == 0) {
                            // First char needs serial processing (assumes valid UTF-8)
                            sz_rune_t rune;
                            sz_rune_length_t rune_length;
                            sz_rune_parse(source, &rune, &rune_length);
                            sz_rune_t folded_runes[3]; // Unicode case folding produces at most 3 runes
                            sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
                            for (sz_size_t i = 0; i != folded_count; ++i)
                                target += sz_rune_export(folded_runes[i], (sz_u8_t *)target);
                            source += rune_length;
                            source_length -= rune_length;
                            continue;
                        }
                        three_byte_length = first_special;
                        prefix_mask_3 = sz_u64_mask_until_(three_byte_length);
                        three_byte_leads_in_prefix = is_three_byte_lead_mask & prefix_mask_3;
                        is_ef_lead &= prefix_mask_3;
                        is_e1_in_prefix = is_e1_lead & three_byte_leads_in_prefix;
                    }

                    // Vectorized Latin Extended Additional (E1 B8-BB = U+1E00-1EFF)
                    // This covers Vietnamese letters which use +1 folding for even codepoints.
                    // Check for E1 B8/B9/BA/BB (second byte 0xB8-0xBB)
                    __mmask64 is_latin_ext_add = _mm512_mask_cmplt_epu8_mask(
                        is_e1_in_prefix, _mm512_sub_epi8(e1_second_bytes, _mm512_set1_epi8((char)0xB8)),
                        _mm512_set1_epi8(0x04)); // 0xB8-0xBB

                    if (is_latin_ext_add) {
                        // For Latin Ext Add, even codepoints fold to +1
                        // The third byte determines parity: even third byte = even codepoint
                        // Third bytes 80, 82, 84... are even codepoints (uppercase)
                        // Note: U+1E96-1E9E (E1 BA 96-9E) are already filtered above via is_special_third
                        __mmask64 third_positions = is_latin_ext_add << 2;
                        // Check if third byte is even (bit 0 = 0)
                        __mmask64 is_even_third = ~_mm512_test_epi8_mask(source_vec.zmm, _mm512_set1_epi8(0x01));
                        __mmask64 fold_third = third_positions & is_even_third & prefix_mask_3;

                        // Also fold ASCII A-Z (special handling: Latin Ext gets +1, ASCII gets +0x20)
                        __mmask64 is_upper_ascii = sz_icelake_is_ascii_upper_(source_vec.zmm);
                        // First apply +1 to both Latin Ext (fold_third) and ASCII positions
                        __m512i folded =
                            _mm512_mask_add_epi8(source_vec.zmm, (fold_third | is_upper_ascii) & prefix_mask_3,
                                                 source_vec.zmm, _mm512_set1_epi8(0x01));
                        // Then add remaining +0x1F to ASCII only (total +0x20)
                        folded = _mm512_mask_add_epi8(folded, is_upper_ascii & prefix_mask_3, folded,
                                                      _mm512_set1_epi8(0x1F));

                        _mm512_mask_storeu_epi8(target, prefix_mask_3, folded);
                        target += three_byte_length, source += three_byte_length, source_length -= three_byte_length;
                        continue;
                    }

                    // Handle Georgian uppercase (E1 82/83) → lowercase (E2 B4)
                    // Georgian Mkhedruli: U+10A0-10C5 (E1 82 A0 - E1 83 85) → U+2D00-2D25 (E2 B4 80 - E2 B4 A5)
                    // This transformation changes the UTF-8 lead byte from E1 to E2, requiring special handling.
                    __mmask64 is_82_second =
                        _mm512_mask_cmpeq_epi8_mask(is_e1_in_prefix, e1_second_bytes, _mm512_set1_epi8((char)0x82));
                    __mmask64 is_83_second =
                        _mm512_mask_cmpeq_epi8_mask(is_e1_in_prefix, e1_second_bytes, _mm512_set1_epi8((char)0x83));
                    __mmask64 is_georgian_second = is_82_second | is_83_second;

                    if (is_georgian_second) {
                        // Validate third byte range for Georgian uppercase:
                        // - E1 82 A0-BF: U+10A0-10BF (32 chars)
                        // - E1 83 80-85: U+10C0-10C5 (6 chars)
                        __mmask64 third_pos_82 = is_82_second << 2;
                        __mmask64 third_pos_83 = is_83_second << 2;

                        // For E1 82: third byte must be A0-BF
                        __mmask64 is_82_valid = _mm512_mask_cmplt_epu8_mask(
                            third_pos_82, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA0)),
                            _mm512_set1_epi8(0x20));

                        // For E1 83: third byte must be 80-85
                        __mmask64 is_83_valid = _mm512_mask_cmplt_epu8_mask(
                            third_pos_83, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x80)),
                            _mm512_set1_epi8(0x06));

                        __mmask64 georgian_leads = ((is_82_valid | is_83_valid) >> 2) & is_e1_in_prefix;

                        if (georgian_leads) {
                            // Transform: E1 82/83 XX → E2 B4 YY
                            // For 82: YY = XX - 0x20 (A0-BF → 80-9F)
                            // For 83: YY = XX + 0x20 (80-85 → A0-A5)

                            // Start with source, then apply transformations
                            __m512i folded = source_vec.zmm;

                            // Set lead bytes to E2 where Georgian
                            folded = _mm512_mask_blend_epi8(georgian_leads, folded, _mm512_set1_epi8((char)0xE2));

                            // Set second bytes to B4 where Georgian
                            __mmask64 georgian_second_pos = georgian_leads << 1;
                            folded = _mm512_mask_blend_epi8(georgian_second_pos, folded, _mm512_set1_epi8((char)0xB4));

                            // Adjust third bytes based on original second byte
                            // -0x20 for sequences that had 82, +0x20 for sequences that had 83
                            folded = _mm512_mask_sub_epi8(folded, is_82_valid, folded, _mm512_set1_epi8(0x20));
                            folded = _mm512_mask_add_epi8(folded, is_83_valid, folded, _mm512_set1_epi8(0x20));

                            // Also fold any ASCII A-Z that might be mixed in
                            folded =
                                _mm512_mask_add_epi8(folded, sz_icelake_is_ascii_upper_(source_vec.zmm) & prefix_mask_3,
                                                     folded, ascii_case_offset);

                            _mm512_mask_storeu_epi8(target, prefix_mask_3, folded);
                            target += three_byte_length, source += three_byte_length,
                                source_length -= three_byte_length;
                            continue;
                        }
                    }
                }

                // Handle EF leads - check for Fullwidth A-Z (FF21-FF3A = EF BC A1 - EF BC BA)
                __mmask64 is_ef_in_prefix = is_ef_lead & three_byte_leads_in_prefix;
                if (is_ef_in_prefix) {
                    __m512i ef_second_bytes =
                        _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(1)), source_vec.zmm);
                    __m512i ef_third_bytes =
                        _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(2)), source_vec.zmm);

                    // Check for EF BC (Fullwidth block FF00-FF3F)
                    __mmask64 is_ef_bc =
                        _mm512_mask_cmpeq_epi8_mask(is_ef_in_prefix, ef_second_bytes, _mm512_set1_epi8((char)0xBC));

                    // Check third byte in A1-BA range (Fullwidth A-Z = FF21-FF3A)
                    // Third byte A1-BA corresponds to codepoints FF21-FF3A
                    __mmask64 is_fullwidth_az = _mm512_mask_cmplt_epu8_mask(
                        is_ef_bc, _mm512_sub_epi8(ef_third_bytes, _mm512_set1_epi8((char)0xA1)),
                        _mm512_set1_epi8(0x1A));

                    if (is_fullwidth_az) {
                        // Has Fullwidth A-Z - apply +0x20 to third byte for those positions
                        // Also fold any ASCII A-Z in the mixed content
                        __mmask64 third_byte_positions = is_fullwidth_az << 2;
                        __mmask64 fold_mask =
                            (third_byte_positions | sz_icelake_is_ascii_upper_(source_vec.zmm)) & prefix_mask_3;
                        __m512i folded =
                            _mm512_mask_add_epi8(source_vec.zmm, fold_mask, source_vec.zmm, ascii_case_offset);
                        _mm512_mask_storeu_epi8(target, prefix_mask_3, folded);
                        target += three_byte_length, source += three_byte_length, source_length -= three_byte_length;
                        continue;
                    }
                }

                // No special 3-byte cases found - fold ASCII A-Z, copy 3-byte unchanged
                // But do NOT copy if we detected unsafe E2s that weren't handled!
                if (!is_unsafe_e2) {
                    _mm512_mask_storeu_epi8(target, prefix_mask_3,
                                            sz_icelake_fold_ascii_in_prefix_(source_vec.zmm, prefix_mask_3));
                    target += three_byte_length, source += three_byte_length, source_length -= three_byte_length;
                    continue;
                }
            }
        }

        // 4. Handle 4-byte sequences (emoji, rare scripts): detect lead bytes (11110xxx = F0-F7)
        {
            __mmask64 is_valid_four_byte_only = is_four_byte_lead_mask | is_cont_mask;
            sz_size_t four_byte_length = sz_icelake_first_invalid_(is_valid_four_byte_only, load_mask, chunk_size);

            // Don't split a 4-byte sequence: find first incomplete lead and truncate there.
            // Note: `>= 1` is an optimization - skips mask ops when nothing valid.
            if (four_byte_length >= 1) {
                __mmask64 all_leads = is_four_byte_lead_mask & sz_u64_mask_until_(four_byte_length);
                __mmask64 safe_leads_mask = four_byte_length >= 4 ? sz_u64_mask_until_(four_byte_length - 3) : 0;
                __mmask64 unsafe_leads = all_leads & ~safe_leads_mask;
                if (unsafe_leads) four_byte_length = sz_u64_ctz(unsafe_leads);
            }

            if (four_byte_length >= 4) {
                // Check if all 4-byte chars are in non-folding ranges (emoji: second byte >= 0x9F)
                __m512i f0_second_bytes =
                    _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(1)), source_vec.zmm);
                __mmask64 is_emoji_lead = _mm512_cmpge_epu8_mask(f0_second_bytes, _mm512_set1_epi8((char)0x9F));
                __mmask64 prefix_mask_4 = sz_u64_mask_until_(four_byte_length);
                __mmask64 four_byte_leads_in_prefix = is_four_byte_lead_mask & prefix_mask_4;

                // All 4-byte leads are emoji (second byte >= 0x9F) - no case folding needed
                if ((four_byte_leads_in_prefix & ~is_emoji_lead) == 0) {
                    _mm512_mask_storeu_epi8(target, prefix_mask_4, source_vec.zmm);
                    target += four_byte_length, source += four_byte_length, source_length -= four_byte_length;
                    continue;
                }
            }
        }

        // Mixed content or expanding characters - process one character serially inline
        {
            // Check for incomplete sequence at end of buffer before parsing
            sz_u8_t lead = (sz_u8_t)*source;
            sz_size_t expected_length = 1;
            if ((lead & 0xE0) == 0xC0) expected_length = 2;
            else if ((lead & 0xF0) == 0xE0)
                expected_length = 3;
            else if ((lead & 0xF8) == 0xF0)
                expected_length = 4;

            if (expected_length > source_length) {
                // Incomplete sequence at end - copy remaining bytes as-is
                while (source_length) {
                    *target++ = *source++;
                    source_length--;
                }
                break;
            }
            // Serial fallback for remaining bytes (assumes valid UTF-8)
            sz_rune_t rune;
            sz_rune_length_t rune_length;
            sz_rune_parse(source, &rune, &rune_length);

            sz_rune_t folded_runes[3]; // Unicode case folding produces at most 3 runes
            sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);

            for (sz_size_t i = 0; i != folded_count; ++i) target += sz_rune_export(folded_runes[i], (sz_u8_t *)target);
            source += rune_length;
            source_length -= rune_length;
        }
    }

    return (sz_size_t)(target - target_start);
}

/*  Undefine local helper macros to avoid namespace pollution */
#undef sz_icelake_is_ascii_upper_
#undef sz_icelake_fold_ascii_
#undef sz_icelake_fold_ascii_in_prefix_
#undef sz_icelake_transform_georgian_

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // SZ_USE_ICELAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_CASE_FOLD_ICELAKE_H_
