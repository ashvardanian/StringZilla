/**
 *  @brief SVE2 backend for UTF-8 codepoint mechanics.
 *  @file include/stringzilla/utf8_runes/sve2.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_RUNES_SVE2_H_
#define STRINGZILLA_UTF8_RUNES_SVE2_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SVE2
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve+sve2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve+sve2")
#endif

SZ_PUBLIC sz_size_t sz_utf8_count_sve2(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const step = svcntb();
    sz_size_t char_count = 0;

    // Count bytes that are NOT continuation bytes: (byte & 0xC0) != 0x80
    for (sz_size_t offset = 0; offset < length; offset += step) {
        svbool_t pg = svwhilelt_b8((sz_u64_t)offset, (sz_u64_t)length);
        svuint8_t text_vec = svld1_u8(pg, text_u8 + offset);
        svbool_t is_start = svcmpne_n_u8(pg, svand_n_u8_x(pg, text_vec, 0xC0), 0x80);
        char_count += svcntp_b8(pg, is_start);
    }
    return char_count;
}

/** @brief  Return a pointer to the start byte of the `n`-th UTF-8 codepoint, or `SZ_NULL_CHAR` if absent.
 *
 *  Single-window O(1) locate mirroring the RVV backend: each window zero-extends `svcntw()` bytes into 32-bit
 *  lanes (SVE2 has no `svcompact_u8`, so the index domain is 32-bit), skips whole windows by lead popcount, then
 *  `svcompact_u32` packs the lead-lane iota and `svlastb` reads the `n`-th packed lane. Byte-exact to serial. */
SZ_PUBLIC sz_cptr_t sz_utf8_seek_sve2(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const window_bytes = svcntw(); // one byte per 32-bit lane
    svuint32_t const lane_iota = svindex_u32(0, 1);
    while (length) {
        svbool_t const pg = svwhilelt_b32_u64(0, (sz_u64_t)length);
        svuint32_t const bytes_u32 = svld1ub_u32(pg, text_u8);
        // Leading byte iff `(byte & 0xC0) != 0x80`.
        svbool_t const lead = svcmpne_n_u32(pg, svand_n_u32_x(pg, bytes_u32, 0xC0), 0x80);
        sz_size_t const lead_count = svcntp_b32(pg, lead);
        if (n >= lead_count) {
            n -= lead_count;
            text_u8 += window_bytes, length -= sz_min_of_two(window_bytes, length);
            continue;
        }
        svuint32_t const packed = svcompact_u32(lead, lane_iota);
        sz_size_t const lane = svlastb_u32(svwhilelt_b32_u64(0, (sz_u64_t)(n + 1)), packed);
        return (sz_cptr_t)(text_u8 + lane);
    }
    return SZ_NULL_CHAR;
}

/*  Multistep newline / whitespace iteration (SVE2 / AArch64 scalable). Each tile classifies delimiter starts
 *  branchlessly into a per-lane start predicate + byte-length vector; the masked tail reads zero so a truncated
 *  multi-byte delimiter at EOF never matches (no serial tail). Starts are trusted in `[0, tile - 3]` on a full
 *  tile (2-/3-byte views fully loaded) and in every valid lane on the final tile. Output is capacity-cut to
 *  `matches_capacity`. For newlines a CRLF is one 2-byte match: a `text[pos-1] == '\r'` carry suppresses the
 *  straddling LF and a post-loop skip mirrors it; whitespace does no CRLF merging. */

#pragma region Codepoint unpack

/** @brief  Right-shift a per-byte-lane value vector by @p amount lanes (toward lower indices), zero-filling the
 *          high edge - the SVE2 value-domain twin of the icelake `mask >> amount`. Lanes that would read a wrapped
 *          negative or `>= VL` index resolve to 0 under `svtbl`, exactly "no source lane there". */
SZ_INTERNAL svuint8_t sz_utf8_shift_value_down_sve2_(svuint8_t value, sz_u8_t amount, svuint8_t lane_iota) {
    return svtbl_u8(value, svadd_n_u8_x(svptrue_b8(), lane_iota, amount));
}

/** @brief  Left-shift a per-byte-lane value vector by @p amount lanes (toward higher indices), zero-filling the
 *          low edge - the SVE2 value-domain twin of the icelake `mask << amount`. */
SZ_INTERNAL svuint8_t sz_utf8_shift_value_up_sve2_(svuint8_t value, sz_u8_t amount, svuint8_t lane_iota) {
    return svtbl_u8(value, svsub_n_u8_x(svptrue_b8(), lane_iota, amount));
}

/**
 *  @brief  Decode the dense set of emitted-start lanes @p emit_starts (a 32-bit predicate over the first `svcntw()`
 *          byte lanes of @p bytes) into sequential UTF-32 runes via ONE `svcompact_u32` - the rune-valued SVE2 twin
 *          of @ref sz_utf8_rune_drain_icelake_. The start byte-indices ride a `svindex_u32(0,1)` iota,
 *          one `svcompact_u32(emit_starts, iota)` packs the K start indices, `svtbl_u8` gathers each start's lead +
 *          up to three trailing bytes, a 1/2/3/4-byte width-blend assembles the values, and ill-formed lanes are
 *          overwritten with U+FFFD before one masked `svst1_u32`. The cursor is the last packed start's offset plus
 *          its compacted maximal-subpart length, so an ill-formed trailing lane never skips bytes owing their own
 *          next U+FFFD.
 *
 *  @note   Per-window K is capped by `svcntw()` (the 32-bit compaction width): the caller bounds the decodable span
 *          to the first `svcntw()` byte lanes. That is an accepted per-window throughput ceiling, not a correctness
 *          limit - the driver simply takes more windows.
 *  @return Number of runes emitted; sets @p consumed_bytes to the byte span they cover (the resume cursor delta).
 */
SZ_INTERNAL sz_size_t sz_utf8_rune_drain_sve2_(                                    //
    svuint8_t bytes, svuint8_t next1, svuint8_t next2, svuint8_t next3,            //
    svbool_t emit_starts_b32, svuint8_t ill_formed_v, svuint8_t consumed_length_v, //
    sz_size_t emit_count, sz_rune_t *runes, sz_size_t capacity, sz_size_t *consumed_bytes) {

    svbool_t const all = svptrue_b32();
    svuint32_t const iota = svindex_u32(0, 1);

    // ONE compaction: pack the K start byte-indices, and (in lockstep) every per-start datum we still need.
    svuint32_t const start_offsets = svcompact_u32(emit_starts_b32, iota);
    // The lead / trailing bytes and the ill-formed flag + maximal-subpart length are byte-lane vectors; widen the
    // low `svcntw()` lanes to 32-bit (their byte index is the iota) and compact by the very same predicate, so lane
    // `j` of every compacted vector belongs to the `j`-th emitted start.
    svuint32_t const lead_word = svunpklo_u32(svunpklo_u16(bytes));
    svuint32_t const next1_word = svunpklo_u32(svunpklo_u16(next1));
    svuint32_t const next2_word = svunpklo_u32(svunpklo_u16(next2));
    svuint32_t const next3_word = svunpklo_u32(svunpklo_u16(next3));
    svuint32_t const ill_word = svunpklo_u32(svunpklo_u16(ill_formed_v));
    svuint32_t const len_word = svunpklo_u32(svunpklo_u16(consumed_length_v));

    svuint32_t const b0 = svcompact_u32(emit_starts_b32, lead_word);
    svuint32_t const b1 = svcompact_u32(emit_starts_b32, next1_word);
    svuint32_t const b2 = svcompact_u32(emit_starts_b32, next2_word);
    svuint32_t const b3 = svcompact_u32(emit_starts_b32, next3_word);
    svuint32_t const ill_compacted = svcompact_u32(emit_starts_b32, ill_word);
    svuint32_t const len_compacted = svcompact_u32(emit_starts_b32, len_word);

    sz_size_t const want = emit_count < capacity ? emit_count : capacity;
    svbool_t const emit = svwhilelt_b32_u64(0, (sz_u64_t)want);

    svuint32_t codepoints = b0; // ASCII keeps the lead.
    // 2-byte (0xC0..0xDF): ((b0 & 0x1F) << 6) | (b1 & 0x3F).
    svbool_t const is_two = svand_b_z(all, svcmpge_n_u32(all, b0, 0xC0), svcmplt_n_u32(all, b0, 0xE0));
    svuint32_t const two_byte = svorr_u32_x(all, svlsl_n_u32_x(all, svand_n_u32_x(all, b0, 0x1F), 6),
                                            svand_n_u32_x(all, b1, 0x3F));
    codepoints = svsel_u32(is_two, two_byte, codepoints);
    // 3-byte (0xE0..0xEF): ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F).
    svbool_t const is_three = svand_b_z(all, svcmpge_n_u32(all, b0, 0xE0), svcmplt_n_u32(all, b0, 0xF0));
    svuint32_t const three_byte = svorr_u32_x(all,
                                              svorr_u32_x(all, svlsl_n_u32_x(all, svand_n_u32_x(all, b0, 0x0F), 12),
                                                          svlsl_n_u32_x(all, svand_n_u32_x(all, b1, 0x3F), 6)),
                                              svand_n_u32_x(all, b2, 0x3F));
    codepoints = svsel_u32(is_three, three_byte, codepoints);
    // 4-byte (0xF0..0xF4): ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F).
    svbool_t const is_four = svcmpge_n_u32(all, b0, 0xF0);
    svuint32_t const four_byte = svorr_u32_x(
        all,
        svorr_u32_x(all, svlsl_n_u32_x(all, svand_n_u32_x(all, b0, 0x07), 18),
                    svlsl_n_u32_x(all, svand_n_u32_x(all, b1, 0x3F), 12)),
        svorr_u32_x(all, svlsl_n_u32_x(all, svand_n_u32_x(all, b2, 0x3F), 6), svand_n_u32_x(all, b3, 0x3F)));
    codepoints = svsel_u32(is_four, four_byte, codepoints);

    // Every ill-formed start (bad lead, broken continuation chain, overlong / surrogate / range, orphan continuation)
    // collapses to one U+FFFD over its maximal ill-formed subpart; the width-blend value for that lane is discarded.
    svbool_t const ill_mask = svcmpne_n_u32(all, ill_compacted, 0);
    codepoints = svsel_u32(ill_mask, svdup_n_u32((sz_u32_t)sz_rune_replacement_k), codepoints);

    svst1_u32(emit, runes, codepoints);

    // Resume cursor delta = last emitted start's offset + its compacted maximal-subpart length. For a well-formed
    // lane that length is the declared rune length; for an ill-formed lane it is the 1-3 byte subpart, so the cursor
    // never skips bytes that must become their own next-window U+FFFD.
    svbool_t const last = svwhilelt_b32_u64(0, (sz_u64_t)want);
    sz_u32_t const last_offset = svlastb_u32(last, start_offsets);
    sz_u32_t const last_length = svlastb_u32(last, len_compacted);
    *consumed_bytes = (sz_size_t)last_offset + (sz_size_t)last_length;
    return want;
}

/**
 *  @brief  Decode one window of @p text into dense UTF-32 @p runes by the uniform "classify -> per-lane well-formed +
 *          orphan promotion -> compress emitted starts -> gather -> width-blend -> blend U+FFFD" path, emitting at
 *          most @p runes_capacity runes and returning the resume cursor. The SVE2 twin of
 *          @ref sz_utf8_decode_once_icelake_: classification runs in the byte domain over a `min(svcntb(), 64)`
 *          window (so neighbours of every start are in register), then a single `svcompact_u32` drains the K starts
 *          that fall in the first `svcntw()` byte lanes. The decode is TOTAL: clean and dirty bytes are handled
 *          in-vector, one U+FFFD per maximal ill-formed subpart (Unicode 17.0 §3.9 / W3C), bit-exact with
 *          @ref sz_utf8_decode_serial. The step declines (`*runes_unpacked == 0`, cursor unchanged) ONLY when
 *          the first lead's declared sequence crosses the window edge (a boundary truncation), which the public
 *          entry finalizes without a serial re-decode.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_decode_once_sve2_( //
    sz_cptr_t text, sz_size_t length,            //
    sz_rune_t *runes, sz_size_t runes_capacity,  //
    sz_size_t *runes_unpacked) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const tile = svcntb() < 64 ? svcntb() : 64; // byte-per-lane window, capped so u8 indices stay in range
    sz_size_t const window = tile < length ? tile : length;
    // The 32-bit compaction packs at most `svcntw()` starts per window: bound the emitted span to the first
    // `svcntw()` byte lanes (an accepted per-window throughput ceiling; the driver simply takes more windows).
    sz_size_t const emit_span = svcntw() < window ? svcntw() : window;
    svbool_t const all = svptrue_b8();
    svbool_t const loaded = svwhilelt_b8_u64(0, (sz_u64_t)window);
    svuint8_t const lane_iota = svindex_u8(0, 1);
    svuint8_t const zeros = svdup_n_u8(0);

    // One byte per lane (masked tail reads zero); neighbours come from in-register `svext` fixed shifts: high lanes
    // (and out-of-window lanes) read zero, so a truncated trailing sequence never fabricates continuation bytes.
    svuint8_t const bytes = svld1_u8(loaded, text_u8);
    svuint8_t const next1 = svext_u8(bytes, zeros, 1);
    svuint8_t const next2 = svext_u8(bytes, zeros, 2);
    svuint8_t const next3 = svext_u8(bytes, zeros, 3);

    // ASCII fast lane: a window with the sign bit clear in every loaded lane widens the low `svcntw()` bytes directly
    // (one `svunpklo` cascade, no classification). Like the general drain it emits at most `svcntw()` runes per
    // window - the same per-window ceiling, no sub-block loop.
    svbool_t const high_bit = svcmplt_n_s8(loaded, svreinterpret_s8_u8(bytes), 0);
    if (!svptest_any(loaded, high_bit)) {
        sz_size_t const ascii_span = emit_span < runes_capacity ? emit_span : runes_capacity;
        svbool_t const store = svwhilelt_b32_u64(0, (sz_u64_t)ascii_span);
        svst1_u32(store, runes, svunpklo_u32(svunpklo_u16(bytes)));
        *runes_unpacked = ascii_span;
        return text + ascii_span;
    }

    // Single-source classification: per-lane length from the lead's high nibble via a 16-entry `svtbl_u8` LUT
    // (`{1x12,2,2,3,4}`, the SAME table the serial / NEON references use), so a lead and its length can never
    // disagree and 0xF8..0xFF map to length 4 and cannot slip the gate.
    svbool_t const is_continuation = svcmpeq_n_u8(loaded, svand_n_u8_x(loaded, bytes, 0xC0), 0x80);
    svbool_t const starts = svand_b_z(loaded, loaded, svnot_b_z(loaded, is_continuation));
    svuint8_t const length_lut = svdupq_n_u8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4);
    svuint8_t const high_nibble = svlsr_n_u8_x(all, bytes, 4);
    svuint8_t const length_per_lane = svtbl_u8(length_lut, high_nibble);

    // Defer EVERY start whose declared sequence would reach past the EMIT span (the first `emit_span` byte lanes):
    // the FIRST overrunning start bounds the decodable prefix (well-formed text has only the trailing one; a
    // malformed `E0 C0`-style lead-in-lead overruns earlier). Its bytes resume in the next window or finalize at the
    // edge -> the cursor can never exceed `text + length`.
    svuint8_t const sequence_end = svadd_u8_x(all, lane_iota, length_per_lane);
    svbool_t const overruns = svand_b_z(starts, starts, svcmpgt_n_u8(all, sequence_end, (sz_u8_t)emit_span));
    sz_size_t decodable_end = emit_span;
    if (svptest_any(starts, overruns)) {
        // Lowest overrunning lane index = min-reduction of `overrun ? lane : emit_span` (emit_span <= 64 fits a byte).
        svuint8_t const big = svdup_n_u8((sz_u8_t)emit_span);
        svuint8_t const overrun_lane = svsel_u8(overruns, lane_iota, big);
        decodable_end = (sz_size_t)svminv_u8(all, overrun_lane);
    }
    svbool_t const decodable = svwhilelt_b8_u64(0, (sz_u64_t)decodable_end);
    svbool_t const decodable_starts = svand_b_z(decodable, starts, decodable);

    // Per-lane validity, classifying every lane uniformly (no per-lane loop, no decline). The window is decoded
    // TOTAL: well-formed leads decode to their value, ill-formed leads (bad lead, broken continuation chain,
    // overlong / surrogate / out-of-range first continuation) and orphan continuation bytes each collapse to one
    // U+FFFD over the maximal ill-formed subpart (Unicode 17.0 §3.9 / W3C), bit-exact with the serial reference.
    svbool_t const len_ge2 = svand_b_z(starts, starts, svcmpge_n_u8(all, length_per_lane, 2));
    svbool_t const len_ge3 = svand_b_z(starts, starts, svcmpge_n_u8(all, length_per_lane, 3));
    svbool_t const len_ge4 = svand_b_z(starts, starts, svcmpge_n_u8(all, length_per_lane, 4));
    svbool_t const len_eq2 = svbic_b_z(all, len_ge2, len_ge3);
    int const has_three = svptest_any(starts, len_ge3);

    // Continuation availability vector (1 where lane is a continuation), shifted DOWN so lane `i` of `cont_k` is 1
    // iff lane `i + k` is a continuation - the SVE2 value-domain twin of icelake `continuation_bits >> k`.
    svuint8_t const cont_v = svdup_u8_z(is_continuation, 1);
    svuint8_t const starts_v = svdup_u8_z(starts, 1);
    svuint8_t const cont1_v = svand_u8_x(all, starts_v, sz_utf8_shift_value_down_sve2_(cont_v, 1, lane_iota));
    svuint8_t const cont2_v = svand_u8_x(all, starts_v, sz_utf8_shift_value_down_sve2_(cont_v, 2, lane_iota));
    svuint8_t const cont3_v = svand_u8_x(all, starts_v, sz_utf8_shift_value_down_sve2_(cont_v, 3, lane_iota));
    svbool_t const cont1 = svcmpne_n_u8(all, cont1_v, 0);
    svbool_t const cont2 = svcmpne_n_u8(all, cont2_v, 0);
    svbool_t const cont3 = svcmpne_n_u8(all, cont3_v, 0);

    // Bad lead: 2-byte lead < 0xC2 (C0/C1 overlong), or 4-byte lead > 0xF4 (out of range, incl. F5..FF).
    svbool_t const len_eq4 = svand_b_z(starts, starts, svcmpeq_n_u8(all, length_per_lane, 4));
    svbool_t const bad_lead = svorr_b_z(starts, svand_b_z(starts, len_eq2, svcmplt_n_u8(all, bytes, 0xC2)),
                                        svand_b_z(starts, len_eq4, svcmpgt_n_u8(all, bytes, 0xF4)));

    // First-continuation range violations for E0/ED/F0/F4 (overlong 3/4-byte, surrogate, > U+10FFFF); computed only
    // when a 3-/4-byte lead is present, else empty so `b1_range_ok` keeps every lane.
    svbool_t b1_range_bad = svpfalse_b();
    if (has_three) {
        svbool_t const e0_low = svand_b_z(all, svcmpeq_n_u8(all, bytes, 0xE0), svcmplt_n_u8(all, next1, 0xA0));
        svbool_t const ed_high = svand_b_z(all, svcmpeq_n_u8(all, bytes, 0xED), svcmpge_n_u8(all, next1, 0xA0));
        svbool_t const f0_low = svand_b_z(all, svcmpeq_n_u8(all, bytes, 0xF0), svcmplt_n_u8(all, next1, 0x90));
        svbool_t const f4_high = svand_b_z(all, svcmpeq_n_u8(all, bytes, 0xF4), svcmpge_n_u8(all, next1, 0x90));
        b1_range_bad = svand_b_z(all, starts,
                                 svorr_b_z(all, svorr_b_z(all, e0_low, ed_high), svorr_b_z(all, f0_low, f4_high)));
    }
    svbool_t const b1_range_ok = svbic_b_z(starts, starts, b1_range_bad);
    // A valid first continuation: present, range-valid, and the lead is not itself bad (C0/C1, F5..FF).
    svbool_t const first_ok = svand_b_z(starts, svand_b_z(starts, cont1, b1_range_ok), svnot_b_z(starts, bad_lead));

    // Well-formed leads (vectorized `sz_rune_decode` success), restricted to the decodable span.
    svbool_t const wf1 = svbic_b_z(starts, starts, len_ge2);
    svbool_t const wf2 = svand_b_z(starts, svand_b_z(starts, len_eq2, cont1), svnot_b_z(starts, bad_lead));
    svbool_t const wf3 = svand_b_z(starts, svand_b_z(starts, len_ge3, b1_range_ok), svand_b_z(starts, cont1, cont2));
    svbool_t const wf3_only = svbic_b_z(starts, wf3, len_ge4);
    svbool_t const wf4 = svand_b_z(
        starts, svand_b_z(starts, len_ge4, b1_range_ok),
        svand_b_z(starts, svand_b_z(starts, cont1, cont2), svand_b_z(starts, cont3, svnot_b_z(starts, bad_lead))));
    svbool_t const well_formed = svand_b_z(
        decodable, svorr_b_z(all, svorr_b_z(all, wf1, wf2), svorr_b_z(all, wf3_only, wf4)), decodable);

    // Per-lane maximal-subpart steps (mirror of `sz_utf8_maximal_subpart_`): start at 1 and extend across each
    // continuation slot a well-formed sequence would still accept. For well-formed lanes this reaches the declared
    // length; for ill-formed lanes it is the 1-3 byte subpart that one U+FFFD consumes.
    svbool_t const step2 = svand_b_z(starts, len_ge2, first_ok);
    svbool_t const step3 = svand_b_z(starts, svand_b_z(starts, step2, len_ge3), cont2);
    svbool_t const step4 = svand_b_z(starts, svand_b_z(starts, step3, len_ge4), cont3);

    // Orphan promotion: a continuation byte not covered by ANY lead's maximal-subpart span becomes its own 1-byte
    // U+FFFD. The subpart spans are exactly the continuation slots `step2/3/4` reached, so coverage is those steps
    // (restricted to decodable) smeared UP by their offset - the value-domain twin of icelake `(step << k)`.
    svuint8_t const step2_v = svdup_u8_z(svand_b_z(decodable, step2, decodable), 1);
    svuint8_t const step3_v = svdup_u8_z(svand_b_z(decodable, step3, decodable), 1);
    svuint8_t const step4_v = svdup_u8_z(svand_b_z(decodable, step4, decodable), 1);
    svuint8_t const covered_v = svorr_u8_x(all,
                                           svorr_u8_x(all, sz_utf8_shift_value_up_sve2_(step2_v, 1, lane_iota),
                                                      sz_utf8_shift_value_up_sve2_(step3_v, 2, lane_iota)),
                                           sz_utf8_shift_value_up_sve2_(step4_v, 3, lane_iota));
    svbool_t const covered = svcmpne_n_u8(all, covered_v, 0);
    svbool_t const orphan = svbic_b_z(decodable, svand_b_z(decodable, is_continuation, decodable), covered);
    svbool_t const emit_starts = svand_b_z(decodable, svorr_b_z(decodable, decodable_starts, orphan), decodable);
    sz_size_t const emit_count = svcntp_b8(decodable, emit_starts);
    if (emit_count == 0) { return *runes_unpacked = 0, text; } // Nothing decodable -> window-edge finalize in driver.
    // Orphans are continuations, never well-formed -> ill_formed = emit_starts & ~well_formed.
    svbool_t const ill_formed = svbic_b_z(decodable, emit_starts, well_formed);

    // Per-lane maximal-subpart length = 1 + step2 + step3 + step4 (in window order), needed for the resume cursor.
    svuint8_t consumed_length = svdup_n_u8(1);
    consumed_length = svadd_u8_m(step2, consumed_length, svdup_n_u8(1));
    consumed_length = svadd_u8_m(step3, consumed_length, svdup_n_u8(1));
    consumed_length = svadd_u8_m(step4, consumed_length, svdup_n_u8(1));

    // The drain needs the emit predicate, ill-formed flag, and length as inputs over the FIRST `svcntw()` byte lanes.
    // Build the 32-bit emit predicate from the byte mask: a lane is emitted iff it is a non-zero compacted flag.
    svuint8_t const emit_flags_v = svdup_u8_z(emit_starts, 1);
    svbool_t const emit_starts_b32 = svcmpne_n_u32(svptrue_b32(), svunpklo_u32(svunpklo_u16(emit_flags_v)), 0);
    svuint8_t const ill_flags_v = svdup_u8_z(ill_formed, 1);

    sz_size_t consumed = 0;
    sz_size_t const produced = sz_utf8_rune_drain_sve2_(bytes, next1, next2, next3, emit_starts_b32, ill_flags_v,
                                                        consumed_length, emit_count, runes, runes_capacity, &consumed);
    *runes_unpacked = produced;
    return text + consumed;
}

/**
 *  @brief  Decode @p text into dense UTF-32 @p runes (SVE2). Drives @ref sz_utf8_decode_once_sve2_ window by
 *          window. The in-vector step decodes its whole decodable span; `step_unpacked == 0` only when the very
 *          first lead declares a sequence crossing the window edge (a boundary truncation). A resumable truncation
 *          breaks and awaits more bytes; a bad/overlong truncated lead at the edge finalizes to one U+FFFD over its
 *          maximal ill-formed subpart - a bounded <=3-byte finalize, never a serial window re-decode.
 */
SZ_PUBLIC sz_cptr_t sz_utf8_decode_sve2(        //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {

    sz_cptr_t cursor = text;
    sz_cptr_t const end = text + length;
    sz_size_t runes_written = 0;
    while (runes_written < runes_capacity && cursor < end) {
        sz_size_t step_unpacked = 0;
        sz_cptr_t next = sz_utf8_decode_once_sve2_(cursor, (sz_size_t)(end - cursor), runes + runes_written,
                                                   runes_capacity - runes_written, &step_unpacked);
        if (step_unpacked) {
            runes_written += step_unpacked;
            cursor = next;
            continue;
        }
        if (sz_utf8_incomplete_tail_(cursor, end)) break;
        runes[runes_written++] = (sz_rune_t)sz_rune_replacement_k;
        cursor += sz_utf8_maximal_subpart_(cursor, end);
    }
    *runes_unpacked = runes_written;
    return cursor;
}

#pragma endregion Codepoint unpack

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE2

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_RUNES_SVE2_H_
