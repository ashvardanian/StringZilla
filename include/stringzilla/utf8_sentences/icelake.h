/**
 *  @brief Ice Lake backend for UAX-29 sentence boundaries.
 *  @file include/stringzilla/utf8_sentences/icelake.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_SENTENCES_ICELAKE_H_
#define STRINGZILLA_UTF8_SENTENCES_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_sentences/tables.h"
#include "stringzilla/utf8_sentences/serial.h"
#include "stringzilla/utf8_runes/icelake.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_ICELAKE
#if defined(__clang__) && SZ_CLANG_HAS_EVEX512_
#pragma clang attribute push(                                                                                         \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,evex512,popcnt"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(                                                                                 \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,popcnt"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2", \
                   "popcnt")
#endif

#pragma region Sentence_Break classifier

/** @brief  Start-compacting flat-lookup classify for the COLD `0x800..0xFFFF` residue (Devanagari, Bengali, Thai,
 *          Tamil, ...). Every cold lane is a 3-byte codepoint START, so a 64-byte window holds at most 21 of them:
 *          their `high`/`low` bytes are `vpcompressb`-compacted into the low lanes, widened to full 32-bit codepoints
 *          in up to two 16-lane registers, resolved by the shared @ref sz_utf8_rune_flat_lookup_icelake_ (page LUT
 *          `vpermb` + one `vpgatherdd` each), then `vpexpandb`-scattered back onto @p classes_u8x64 at their original
 *          byte-lane positions. The second half only runs when more than sixteen cold starts are present. Every other
 *          lane keeps its prior value. */
SZ_HELPER_AUTO __m512i sz_utf8_sentence_break_cold_compact_icelake_( //
    __m512i classes_u8x64, __m512i high_bytes_u8x64, __m512i low_bytes_u8x64, sz_u64_t cold_starts) {
    __mmask64 const cold_start_mask = _cvtu64_mask64(cold_starts);
    __m512i const high_packed_u8x64 = _mm512_maskz_compress_epi8(cold_start_mask, high_bytes_u8x64);
    __m512i const low_packed_u8x64 = _mm512_maskz_compress_epi8(cold_start_mask, low_bytes_u8x64);

    //  Unused (zeroed) compacted lanes decode to codepoint 0, a safe in-bounds flat index whose class is discarded.
    __m512i const codepoints_first_u32x16 = _mm512_or_si512(
        _mm512_slli_epi32(_mm512_cvtepu8_epi32(_mm512_castsi512_si128(high_packed_u8x64)), 8),
        _mm512_cvtepu8_epi32(_mm512_castsi512_si128(low_packed_u8x64)));
    __m128i const classes_first_u8x16 = _mm512_cvtepi32_epi8(sz_utf8_rune_flat_lookup_icelake_(
        sz_utf8_sentence_break_bmp_page_lut_, sz_utf8_sentence_break_flat_bmp_, codepoints_first_u32x16));
    __m512i classes_packed_u8x64 = _mm512_castsi128_si512(classes_first_u8x16);

    if (_mm_popcnt_u64(cold_starts) > 16) {
        __m512i const codepoints_second_u32x16 = _mm512_or_si512(
            _mm512_slli_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(high_packed_u8x64, 1)), 8),
            _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(low_packed_u8x64, 1)));
        __m128i const classes_second_u8x16 = _mm512_cvtepi32_epi8(sz_utf8_rune_flat_lookup_icelake_(
            sz_utf8_sentence_break_bmp_page_lut_, sz_utf8_sentence_break_flat_bmp_, codepoints_second_u32x16));
        classes_packed_u8x64 = _mm512_inserti32x4(classes_packed_u8x64, classes_second_u8x16, 1);
    }
    return _mm512_mask_expand_epi8(classes_u8x64, cold_start_mask, classes_packed_u8x64);
}

/**
 *  @brief  Classify up to 64 codepoints (held one-per-lane in the decoded @p high / @p low byte halves) into
 *          per-lane Sentence_Break properties. The dominant `cp < 0x800` region (Latin/Greek/Cyrillic/Arabic/Hebrew)
 *          is resolved by an in-register `vpermi2b` page network over `sz_utf8_sentence_break_flat_lut_0800_`; the big
 *          homogeneous OLetter blocks (CJK/Hangul/...) by arithmetic range compares (zero data); the cold 3-byte-BMP
 *          residue by a page-compressed flat table read with one `vpgatherdd` per sixteen lanes (see
 *          @ref sz_utf8_sentence_break_cold_compact_icelake_). No scalar per-lane loop, no stack round-trip. Astral
 *          (4-byte) lanes are reconstructed to full 21-bit codepoints in register (four 16-lane chunks) and resolved
 *          through the canonical sorted astral range list with arithmetic compares, all 64 lanes uniformly.
 *
 *  @param  raw_window         The raw 64 input bytes (codepoint lead/continuation bytes, one lane each here).
 *  @param  raw_next1          Byte at lane+1 (first continuation), @p raw_next2 lane+2, @p raw_next3 lane+3.
 *  @param  high               Per-lane high byte of the reconstructed codepoint (`cp >> 8`, BMP) from the driver.
 *  @param  low                Per-lane low byte of the reconstructed codepoint (`cp & 0xFF`) from the driver.
 *  @param  four_byte_starts   Lanes that begin a 4-byte UTF-8 sequence (gates the >= 0x10000 astral test).
 *  @param  codepoint_starts   Lanes that begin any codepoint (non-continuation, in range).
 */
SZ_HELPER_AUTO __m512i sz_utf8_sentence_break_classify_window_icelake_(          //
    __m512i raw_window, __m512i raw_next1, __m512i raw_next2, __m512i raw_next3, //
    __m512i high, __m512i low,                                                   //
    __mmask64 four_byte_starts, __mmask64 codepoint_starts) {

    // Partition the lanes FIRST so the expensive classify paths only run for lanes that need them.
    // Dispatch by codepoint VALUE, not byte-length, so overlong / malformed sequences classify exactly like the
    // serial reference: an n-byte lead whose blind value lands in a shorter range is resolved by that range's table.
    // `is_astral` selects 4-byte leads whose value is truly >= 0x10000 (cp bit 16+ set, from `(b0 & 7) | (b1 & 0x30)`);
    // every other lead is a BMP lane routed to the page LUT (high < 0x08) or the flat table gather (high >= 0x08).
    // For well-formed UTF-8 the value split is identical to the byte-length split, so this leaves conformance
    // untouched.
    __mmask64 const is_astral = four_byte_starts & (_mm512_test_epi8_mask(raw_window, _mm512_set1_epi8(0x07)) |
                                                    _mm512_test_epi8_mask(raw_next1, _mm512_set1_epi8(0x30)));
    __mmask64 const bmp_starts = codepoint_starts & ~is_astral;
    __mmask64 const small_lanes = bmp_starts & _mm512_cmplt_epu8_mask(high, _mm512_set1_epi8(0x08));
    __mmask64 const cold_lanes = bmp_starts & _mm512_cmp_epu8_mask(high, _mm512_set1_epi8(0x08), _MM_CMPINT_NLT);

    // Big homogeneous OLetter ranges (CJK, Kana, ...) as arithmetic compares (zero data) over the per-lane
    // (high<<8|low). These resolve the vast majority of CJK / Kana with zero table reads, so a pure-CJK window
    // skips the flat-table gather entirely below.
    __mmask64 oletter = 0;
    for (int range = 0; range < sz_utf8_sentence_break_big_oletter_count_k; ++range) {
        sz_u32_t const lo = sz_utf8_sentence_break_big_oletter_lo_[range];
        sz_u32_t const hi = sz_utf8_sentence_break_big_oletter_hi_[range];
        if (lo >= 0x10000u) continue; // astral OLetter handled by the astral sweep below
        __m512i const lo_hi = _mm512_set1_epi8((char)(sz_u8_t)(lo >> 8));
        __m512i const lo_lo = _mm512_set1_epi8((char)(sz_u8_t)lo);
        __m512i const hi_hi = _mm512_set1_epi8((char)(sz_u8_t)(hi >> 8));
        __m512i const hi_lo = _mm512_set1_epi8((char)(sz_u8_t)hi);
        __mmask64 const ge = _mm512_cmpgt_epu8_mask(high, lo_hi) |
                             (_mm512_cmpeq_epi8_mask(high, lo_hi) & _mm512_cmp_epu8_mask(low, lo_lo, _MM_CMPINT_NLT));
        __mmask64 const le = _mm512_cmplt_epu8_mask(high, hi_hi) |
                             (_mm512_cmpeq_epi8_mask(high, hi_hi) & _mm512_cmp_epu8_mask(low, hi_lo, _MM_CMPINT_LE));
        oletter |= ge & le;
    }
    oletter &= bmp_starts;

    __m512i classes = _mm512_set1_epi8((char)sz_sentence_break_other_k);

    // Page LUT for codepoint < 0x800 (`high[2:0]:low[7:0]` via a `vpermi2b` segment-pair cascade), gated on a small
    // lane being present so a pure-3-byte (CJK) window pays nothing here.
    if (small_lanes) {
        __m512i const in_seven = _mm512_and_si512(low, _mm512_set1_epi8(0x7F));
        __m512i const low_high_bit = sz_utf8_srl8_icelake_(low, 7, 0x01);
        __m512i const page = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(high, _mm512_set1_epi8(0x07)), 1),
                                             low_high_bit);
        __m512i small_class = _mm512_setzero_si512();
        for (int pair = 0; pair < 16; ++pair) {
            __m512i const seg_lo = _mm512_loadu_si512(sz_utf8_sentence_break_flat_lut_0800_ + (2 * pair) * 64);
            __m512i const seg_hi = _mm512_loadu_si512(sz_utf8_sentence_break_flat_lut_0800_ + (2 * pair + 1) * 64);
            __m512i const permuted = _mm512_permutex2var_epi8(seg_lo, in_seven, seg_hi);
            __mmask64 const hit = _mm512_cmpeq_epi8_mask(page, _mm512_set1_epi8((char)pair));
            small_class = _mm512_mask_mov_epi8(small_class, hit, permuted);
        }
        classes = _mm512_mask_mov_epi8(classes, small_lanes, small_class);
    }

    // Flat page-LUT + gather lookup for the 0x800..0xFFFF residue, gated on a 3-byte lane the OLetter ranges did NOT
    // already resolve, so CJK / Kana windows skip it entirely. When every 3-byte lane is OLetter the lookup's output
    // would be overwritten by the OLetter overlay anyway, so skipping it is byte-identical.
    __mmask64 const cold_residual = cold_lanes & ~oletter;
    if (cold_residual)
        classes = sz_utf8_sentence_break_cold_compact_icelake_(classes, high, low, _cvtmask64_u64(cold_residual));

    classes = _mm512_mask_mov_epi8(classes, oletter, _mm512_set1_epi8((char)sz_sentence_break_oletter_k));

    // Astral (4-byte) lanes: reconstruct the full 21-bit codepoint per lane and resolve through the sorted astral
    // range list with arithmetic compares (no table lookup at all). cp = ((b0&7)<<18)|((b1&0x3F)<<12)|((b2&0x3F)<<6)|(b3&0x3F).
    // The 21-bit value exceeds a byte lane, so we widen to 32-bit lanes in four 16-lane chunks and compare each
    // chunk against every astral range, blending the matching class back. The whole block only fires when a
    // 4-byte lead is present (corpus astral residue < 0.1%), keeping the common path branch-free.
    if (is_astral) {
        __m512i const b0 = _mm512_and_si512(raw_window, _mm512_set1_epi8(0x07));
        __m512i const b1 = _mm512_and_si512(raw_next1, _mm512_set1_epi8(0x3F));
        __m512i const b2 = _mm512_and_si512(raw_next2, _mm512_set1_epi8(0x3F));
        __m512i const b3 = _mm512_and_si512(raw_next3, _mm512_set1_epi8(0x3F));
        __m512i const lane_identity = sz_utf8_lane_identity_icelake_();
        for (int chunk = 0; chunk < 4; ++chunk) {
            // Bring this chunk's 16 bytes to the low 16 lanes via a runtime permute, then widen to 32-bit lanes.
            __m512i const select = _mm512_add_epi8(lane_identity, _mm512_set1_epi8((char)(chunk * 16)));
            __m512i const w0 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_permutexvar_epi8(select, b0)));
            __m512i const w1 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_permutexvar_epi8(select, b1)));
            __m512i const w2 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_permutexvar_epi8(select, b2)));
            __m512i const w3 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_permutexvar_epi8(select, b3)));
            __m512i const cp = _mm512_or_si512(_mm512_or_si512(_mm512_slli_epi32(w0, 18), _mm512_slli_epi32(w1, 12)),
                                               _mm512_or_si512(_mm512_slli_epi32(w2, 6), w3));
            __mmask16 const lane_is_four = (__mmask16)((is_astral >> (chunk * 16)) & 0xFFFFu);
            __m512i astral_class = _mm512_setzero_si512();
            __mmask16 matched = 0;
            // Big homogeneous OLetter blocks that live above the BMP (CJK Extension B+, astral Hangul/Kana, ...) are
            // skipped by the BMP OLetter loop above and are NOT duplicated in the astral range list; the serial
            // reference resolves them from `big_oletter` FIRST, so check them here before the astral list with the
            // same first-match-wins precedence.
            for (int range = 0; range < sz_utf8_sentence_break_big_oletter_count_k; ++range) {
                sz_u32_t const lo = sz_utf8_sentence_break_big_oletter_lo_[range];
                sz_u32_t const hi = sz_utf8_sentence_break_big_oletter_hi_[range];
                if (lo < 0x10000u) continue; // BMP OLetter blocks are resolved by the page LUT / flat gather path
                __mmask16 const in_range = _mm512_cmpge_epu32_mask(cp, _mm512_set1_epi32((int)lo)) &
                                           _mm512_cmple_epu32_mask(cp, _mm512_set1_epi32((int)hi)) & lane_is_four &
                                           ~matched;
                astral_class = _mm512_mask_mov_epi32(astral_class, in_range,
                                                     _mm512_set1_epi32((int)sz_sentence_break_oletter_k));
                matched |= in_range;
            }
            for (int range = 0; range < sz_utf8_sentence_break_astral_count_k; ++range) {
                sz_u32_t const lo = sz_utf8_sentence_break_astral_lo_[range];
                sz_u32_t const hi = sz_utf8_sentence_break_astral_hi_[range];
                if (hi < 0x10000u) continue; // BMP ranges are already resolved by the page LUT / flat gather / OLetter
                __mmask16 const in_range = _mm512_cmpge_epu32_mask(cp, _mm512_set1_epi32((int)lo)) &
                                           _mm512_cmple_epu32_mask(cp, _mm512_set1_epi32((int)hi)) & lane_is_four &
                                           ~matched;
                astral_class = _mm512_mask_mov_epi32(astral_class, in_range,
                                                     _mm512_set1_epi32((int)sz_utf8_sentence_break_astral_cls_[range]));
                matched |= in_range;
            }
            // Narrow the 16 32-bit astral classes to 16 bytes (`vpmovdb`), broadcast them back to this chunk's byte
            // lanes via `vpexpandb`, and blend - fully vectorized, no scalar per-lane writeback.
            __m128i const astral_bytes = _mm512_cvtepi32_epi8(astral_class);
            __m512i const astral_broadcast = _mm512_maskz_expand_epi8(_cvtu64_mask64(0xFFFFull << (chunk * 16)),
                                                                      _mm512_castsi128_si512(astral_bytes));
            classes = _mm512_mask_mov_epi8(classes, _cvtu64_mask64((sz_u64_t)matched << (chunk * 16)),
                                           astral_broadcast);
        }
    }
    return classes;
}

#pragma endregion Sentence_Break classifier

#pragma region Sentence_Break boundary algebra

/**
 *  @brief  Ice Lake extractor for the portable dense rule engine: build the per-class membership frame from the dense
 *          codepoint-class register with fifteen `vpcmpeqb` masks (no scalar pass), store the dense class bytes for the
 *          engine's three trailing-context reads, then delegate every SB3-SB998 decision to
 *          @ref sz_utf8_sentence_break_decide_block_. The `__m512i`->mask contact lives here; the rule algebra is
 *          intrinsic-free and shared verbatim with serial / haswell. Force-inlined so the 24-byte window result stays
 *          in registers instead of spilling through an `sret`.
 */
SZ_HELPER_INLINE sz_utf8_sentence_break_window_t sz_utf8_sentence_break_block_breaks_( //
    __m512i classes, sz_size_t count, sz_utf8_sentence_break_carry_t *carry, sz_bool_t more_text) {
    sz_u64_t const valid = (count >= 64) ? ~0ull : ((1ull << count) - 1);
    sz_utf8_sentence_break_frame_t frame;
    for (int cls = 0; cls < 15; ++cls)
        frame.by_class[cls] = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(classes, _mm512_set1_epi8((char)cls))) & valid;
    sz_u8_t dense_classes[64];
    _mm512_storeu_si512((void *)dense_classes, classes);
    return sz_utf8_sentence_break_decide_block_(&frame, dense_classes, count, carry, more_text);
}

#pragma endregion Sentence_Break boundary algebra

#pragma region Sentence_Break forward driver

SZ_API_COMPTIME sz_size_t sz_utf8_sentences_icelake(         //
    sz_cptr_t text, sz_size_t length,                        //
    sz_size_t *sentence_starts, sz_size_t *sentence_lengths, //
    sz_size_t sentences_capacity, sz_size_t *bytes_consumed) {

    sz_size_t sentences = 0;
    if (length == 0 || sentences_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t sentence_start = 0;
    sz_size_t position = 0;
    __m512i const lane_identity = sz_utf8_lane_identity_icelake_();

    sz_utf8_sentence_break_carry_t carry;
    carry.have_prev = 0;
    carry.prev_raw = carry.prev_eff = carry.prev_prev_eff = (sz_u8_t)sz_sentence_break_other_k;
    carry.in_shadow = carry.shadow_aterm = carry.shadow_saw_sp = 0;
    carry.sb8_pending = 0;

    sz_size_t sb8_pending_position = 0;
    int sb8_pending_active = 0;

    while (position < length) {
        sz_utf8_rune_window_t const decoded = sz_utf8_rune_decode_window_icelake_(text_u8 + position, length - position,
                                                                                  lane_identity);
        sz_size_t const loaded = decoded.loaded;
        __mmask64 const lead_continuation = (position == 0) ? (__mmask64)1 : (__mmask64)0;
        __mmask64 const codepoint_starts = decoded.codepoint_starts | lead_continuation;
        sz_u64_t const start_bytes = _cvtmask64_u64(codepoint_starts);

        __m512i const window = decoded.window;
        __mmask64 const keep1 = sz_u64_mask_until_(loaded >= 1 ? loaded - 1 : 0);
        __mmask64 const keep2 = sz_u64_mask_until_(loaded >= 2 ? loaded - 2 : 0);
        __mmask64 const keep3 = sz_u64_mask_until_(loaded >= 3 ? loaded - 3 : 0);
        __m512i const next1 = _mm512_maskz_permutexvar_epi8(keep1, _mm512_add_epi8(lane_identity, _mm512_set1_epi8(1)),
                                                            window);
        __m512i const next2 = _mm512_maskz_permutexvar_epi8(keep2, _mm512_add_epi8(lane_identity, _mm512_set1_epi8(2)),
                                                            window);
        __m512i const next3 = _mm512_maskz_permutexvar_epi8(keep3, _mm512_add_epi8(lane_identity, _mm512_set1_epi8(3)),
                                                            window);

        __m512i const two_high = sz_utf8_srl8_icelake_(_mm512_and_si512(window, _mm512_set1_epi8(0x1F)), 2, 0x07);
        __m512i const two_low = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(window, _mm512_set1_epi8(0x03)), 6),
                                                _mm512_and_si512(next1, _mm512_set1_epi8(0x3F)));
        __m512i const three_high = _mm512_or_si512(
            _mm512_slli_epi16(_mm512_and_si512(window, _mm512_set1_epi8(0x0F)), 4),
            sz_utf8_srl8_icelake_(next1, 2, 0x0F));
        __m512i const three_low = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(next1, _mm512_set1_epi8(0x03)), 6),
                                                  _mm512_and_si512(next2, _mm512_set1_epi8(0x3F)));
        __m512i const four_low = _mm512_or_si512(_mm512_and_si512(next3, _mm512_set1_epi8(0x3F)),
                                                 _mm512_slli_epi16(_mm512_and_si512(next2, _mm512_set1_epi8(0x03)), 6));
        __m512i const four_high = _mm512_or_si512(
            sz_utf8_srl8_icelake_(next2, 2, 0x0F),
            _mm512_slli_epi16(_mm512_and_si512(next1, _mm512_set1_epi8(0x0F)), 4));
        __m512i high = _mm512_setzero_si512();
        high = _mm512_mask_mov_epi8(high, decoded.two_byte_starts, two_high);
        high = _mm512_mask_mov_epi8(high, decoded.three_byte_starts, three_high);
        high = _mm512_mask_mov_epi8(high, decoded.four_byte_starts, four_high);
        __m512i low = window;
        low = _mm512_mask_mov_epi8(low, decoded.two_byte_starts, two_low);
        low = _mm512_mask_mov_epi8(low, decoded.three_byte_starts, three_low);
        low = _mm512_mask_mov_epi8(low, decoded.four_byte_starts, four_low);
        __m512i const classes = sz_utf8_sentence_break_classify_window_icelake_(
            window, next1, next2, next3, high, low, decoded.four_byte_starts, codepoint_starts);

        sz_bool_t const more_text = (position + loaded < length) ? sz_true_k : sz_false_k;

        // Effective-window<64 trim (identical to the shipped driver): exclude every straddling trailing lead and a
        // stray continuation at the window edge, so only fully-decoded codepoints enter the dense stream.
        sz_size_t complete_limit = loaded;
        if (more_text && start_bytes) {
            sz_u64_t const valid_lanes = sz_u64_mask_until_(loaded);
            sz_u64_t const two = _cvtmask64_u64(decoded.two_byte_starts) & valid_lanes;
            sz_u64_t const three = _cvtmask64_u64(decoded.three_byte_starts) & valid_lanes;
            sz_u64_t const four = _cvtmask64_u64(decoded.four_byte_starts) & valid_lanes;
            sz_u64_t const straddle = ((two & ~sz_u64_mask_until_(loaded > 1 ? loaded - 1 : 0)) |
                                       (three & ~sz_u64_mask_until_(loaded > 2 ? loaded - 2 : 0)) |
                                       (four & ~sz_u64_mask_until_(loaded > 3 ? loaded - 3 : 0))) &
                                      valid_lanes;
            sz_size_t limit = straddle ? (sz_size_t)sz_u64_ctz(straddle) : loaded;
            if ((text_u8[position + loaded] & 0xC0) == 0x80) {
                sz_size_t const last_lead = (sz_size_t)(63 - sz_u64_clz(start_bytes));
                if (last_lead < limit) limit = last_lead;
            }
            if (limit > 0) complete_limit = limit;
        }

        // Dense compaction: keep only codepoint starts inside the complete region, compress their class bytes to a
        // dense codepoint stream (one vpcompressb), and remember the byte lanes for the scatter back.
        sz_u64_t const complete_mask = sz_u64_mask_until_(complete_limit);
        sz_u64_t const dense_start_lanes = start_bytes & complete_mask;
        sz_size_t const dense_count = (sz_size_t)_mm_popcnt_u64(dense_start_lanes);
        __m512i const dense_classes = _mm512_maskz_compress_epi8(_cvtu64_mask64(dense_start_lanes), classes);

        sz_utf8_sentence_break_carry_t carry_full = carry;
        sz_utf8_sentence_break_window_t const win = sz_utf8_sentence_break_block_breaks_(dense_classes, dense_count,
                                                                                         &carry_full, more_text);

        // Resolve a previously deferred SB8 boundary before any of this window's boundaries.
        if (sb8_pending_active && win.sb8_resolution != 0) {
            if (win.sb8_resolution == 1) {
                if (sentences == sentences_capacity) {
                    if (bytes_consumed) *bytes_consumed = sentence_start;
                    return sentences;
                }
                sentence_starts[sentences] = sentence_start;
                sentence_lengths[sentences] = sb8_pending_position - sentence_start;
                ++sentences;
                sentence_start = sb8_pending_position;
            }
            sb8_pending_active = 0;
        }

        // Scatter the trusted dense breaks back to byte lanes. `win.resolved` is dense; deposit the trusted dense
        // bits into the dense start lanes via `_pdep_u64`, then mask to the byte lanes of the trusted dense prefix.
        sz_size_t const dense_adv = win.resolved;
        sz_u64_t const dense_breaks = win.breaks & sz_u64_mask_until_(dense_adv);
        sz_u64_t boundary_lanes = _pdep_u64(dense_breaks, dense_start_lanes);
        if (!carry.have_prev) boundary_lanes &= ~1ull;

        // The byte-domain advance is the byte offset just past the last trusted dense codepoint. When the whole
        // complete region is trusted (dense_adv == dense_count), advance by the complete region; otherwise advance
        // to the byte lane of the first untrusted dense codepoint (the SB8 clamp re-anchor point).
        sz_size_t byte_adv;
        if (dense_adv >= dense_count) { byte_adv = complete_limit ? complete_limit : loaded; }
        else {
            // Byte offset of dense codepoint `dense_adv`: the (dense_adv)-th set bit of `dense_start_lanes`.
            sz_u64_t const upto = _pdep_u64((1ull << dense_adv), dense_start_lanes);
            byte_adv = (sz_size_t)sz_u64_ctz(upto);
        }

        sentences = sz_utf8_rune_drain_forward_(boundary_lanes, position, lane_identity, sentence_starts,
                                                sentence_lengths, sentences, sentences_capacity, &sentence_start);
        if (sentences == sentences_capacity) {
            if (bytes_consumed) *bytes_consumed = sentence_start;
            return sentences;
        }

        if (dense_adv >= dense_count) {
            carry = carry_full;
            position += byte_adv;
        }
        else if (dense_adv > 0) {
            // SB8 clamp strictly before the dense edge: rebuild the carry to the clamp point.
            sz_utf8_sentence_break_carry_t carry_to_edge = carry;
            sz_utf8_sentence_break_block_breaks_(dense_classes, dense_adv, &carry_to_edge, sz_true_k);
            carry = carry_to_edge;
            position += byte_adv;
        }
        else {
            // Clamp at dense lane 0: defer the ATerm SB8 boundary and step past the whole complete span.
            if (!sb8_pending_active) {
                sb8_pending_active = 1;
                sb8_pending_position = position;
            }
            carry = carry_full;
            carry.sb8_pending = 1;
            position += complete_limit ? complete_limit : loaded;
        }
    }

    if (sb8_pending_active) {
        if (sentences == sentences_capacity) {
            if (bytes_consumed) *bytes_consumed = sentence_start;
            return sentences;
        }
        sentence_starts[sentences] = sentence_start;
        sentence_lengths[sentences] = sb8_pending_position - sentence_start;
        ++sentences;
        sentence_start = sb8_pending_position;
    }

    if (sentences == sentences_capacity) {
        if (bytes_consumed) *bytes_consumed = sentence_start;
        return sentences;
    }
    sentence_starts[sentences] = sentence_start;
    sentence_lengths[sentences] = length - sentence_start;
    ++sentences;
    if (bytes_consumed) *bytes_consumed = length;
    return sentences;
}

#pragma endregion Sentence_Break forward driver
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_ICELAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_SENTENCES_ICELAKE_H_
