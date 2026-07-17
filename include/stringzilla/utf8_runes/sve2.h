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

SZ_API_COMPTIME sz_size_t sz_utf8_count_sve2(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const step = svcntb();
    sz_size_t char_count = 0;

    // Count bytes that are NOT continuation bytes: (byte & 0xC0) != 0x80
    for (sz_size_t offset = 0; offset < length; offset += step) {
        svbool_t pg_b8x = svwhilelt_b8((sz_u64_t)offset, (sz_u64_t)length);
        svuint8_t text_vec_u8x = svld1_u8(pg_b8x, text_u8 + offset);
        svbool_t is_start_b8x = svcmpne_n_u8(pg_b8x, svand_n_u8_x(pg_b8x, text_vec_u8x, 0xC0), 0x80);
        char_count += svcntp_b8(pg_b8x, is_start_b8x);
    }
    return char_count;
}

/** @brief  Return a pointer to the start byte of the `n`-th UTF-8 codepoint, or `SZ_NULL_CHAR` if absent.
 *
 *  Full byte-vector windows skip by lead popcount at `svcntb()` bytes per step; the window holding the target
 *  walks its 32-bit quarters (SVE2 has no `svcompact_u8`), compacting the lead-lane iota of the one quarter the
 *  `n`-th lead falls in and reading it with `svlastb`. Byte-exact to serial. */
SZ_API_COMPTIME sz_cptr_t sz_utf8_seek_sve2(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const window_bytes = svcntb();
    sz_size_t const quarter_lanes = svcntw();
    while (length) {
        svbool_t const pg_b8x = svwhilelt_b8_u64(0, (sz_u64_t)length);
        svuint8_t const bytes_u8x = svld1_u8(pg_b8x, text_u8);
        // Leading byte iff `(byte & 0xC0) != 0x80`.
        svbool_t const lead_b8x = svcmpne_n_u8(pg_b8x, svand_n_u8_x(pg_b8x, bytes_u8x, 0xC0), 0x80);
        sz_size_t const lead_count = svcntp_b8(pg_b8x, lead_b8x);
        if (n >= lead_count) {
            n -= lead_count;
            text_u8 += window_bytes, length -= sz_min_of_two(window_bytes, length);
            continue;
        }
        svuint8_t const lead_flags_u8x = svdup_u8_z(lead_b8x, 1);
        svuint16_t const flags_lo_u16x = svunpklo_u16(lead_flags_u8x), flags_hi_u16x = svunpkhi_u16(lead_flags_u8x);
        for (int quarter = 0; quarter < 4; ++quarter) {
            svuint16_t const half_u16x = (quarter & 2) ? flags_hi_u16x : flags_lo_u16x;
            svuint32_t const quarter_u32x = (quarter & 1) ? svunpkhi_u32(half_u16x) : svunpklo_u32(half_u16x);
            svbool_t const quarter_lead_b32x = svcmpne_n_u32(svptrue_b32(), quarter_u32x, 0);
            sz_size_t const quarter_count = svcntp_b32(svptrue_b32(), quarter_lead_b32x);
            if (n >= quarter_count) {
                n -= quarter_count;
                continue;
            }
            svuint32_t const packed_u32x = svcompact_u32(quarter_lead_b32x,
                                                         svindex_u32((sz_u32_t)(quarter * quarter_lanes), 1));
            sz_size_t const lane = svlastb_u32(svwhilelt_b32_u64(0, (sz_u64_t)(n + 1)), packed_u32x);
            return (sz_cptr_t)(text_u8 + lane);
        }
    }
    return SZ_NULL_CHAR;
}

#pragma region Codepoint unpack

/** @brief  Lower a byte predicate over up to 64 lanes to a `sz_u64_t` lane mask, fully in-register: eight 0/1 bytes
 *          pack into each 64-bit lane's top byte via one multiply (every partial product lands on a distinct bit, so
 *          no carries), and a lane-indexed shift + `svaddv` folds the disjoint per-lane bytes into one scalar. */
SZ_HELPER_INLINE sz_u64_t sz_utf8_rune_pred_to_u64_sve2_(svbool_t mask_b8x) {
    svbool_t const all_b64x = svptrue_b64();
    svuint64_t const flag_bytes_u64x = svreinterpret_u64_u8(svdup_u8_z(mask_b8x, 1));
    svuint64_t const packed_u64x = svlsr_n_u64_x(all_b64x,
                                                 svmul_n_u64_x(all_b64x, flag_bytes_u64x, 0x0102040810204080ull), 56);
    return svaddv_u64(all_b64x, svlsl_u64_x(all_b64x, packed_u64x, svindex_u64(0, 8)));
}

/** @brief  Left-shift a per-byte-lane value vector by @p amount lanes (toward higher indices) across a chunk
 *          boundary: low lanes fill from the top of @p carry (the previous chunk), the SVE2 value-domain twin of
 *          the icelake `mask << amount` with a cross-window carry. Out-of-range `svtbl` indices resolve to 0, so
 *          each lane reads exactly one of the two sources. */
SZ_HELPER_INLINE svuint8_t sz_utf8_shift_value_up_pair_sve2_(svuint8_t carry, svuint8_t value, sz_u8_t amount,
                                                             svuint8_t lane_iota, sz_u8_t chunk_lanes) {
    svbool_t const all_b8x = svptrue_b8();
    return svorr_u8_x(all_b8x, svtbl_u8(value, svsub_n_u8_x(all_b8x, lane_iota, amount)),
                      svtbl_u8(carry, svadd_n_u8_x(all_b8x, lane_iota, (sz_u8_t)(chunk_lanes - amount))));
}

/**
 *  @brief  One widened quarter of @ref sz_utf8_rune_flat_lookup_sve2_: two chained `svld1ub_gather_u32offset_u32`
 *          resolve `flat[page_lut[high] * 256 + low]` for `svcntw()` lanes on the load pipes. Every offset is
 *          in-bounds by construction (the page offset is a byte, and `page * 256 + low` stays under `pages * 256`),
 *          so `svptrue_b32()` governs even the tail lanes.
 */
SZ_HELPER_INLINE svuint32_t sz_utf8_rune_flat_lookup_quarter_sve2_( //
    sz_u8_t const *page_lut, sz_u8_t const *flat, svuint32_t high_bytes_u32x, svuint32_t low_bytes_u32x) {
    svbool_t const all_words_b32x = svptrue_b32();
    svuint32_t const page_indices_u32x = svld1ub_gather_u32offset_u32(all_words_b32x, page_lut, high_bytes_u32x);
    svuint32_t const flat_offsets_u32x = svadd_u32_x(
        all_words_b32x, svlsl_n_u32_x(all_words_b32x, page_indices_u32x, 8), low_bytes_u32x);
    return svld1ub_gather_u32offset_u32(all_words_b32x, flat, flat_offsets_u32x);
}

/**
 *  @brief  Class byte per lane from a page-compressed flat table: `page_lut[cp >> 8]` selects a 256-byte page, then
 *          `flat[page * 256 + (cp & 0xFF)]` yields the descriptor, both stages gathered by real `LD1B` byte gathers.
 *          The SVE2 twin of @ref sz_utf8_rune_flat_lookup_haswell_ and @ref sz_utf8_rune_flat_lookup_icelake_. An
 *          `svtbl_u8` scan of the 256-entry page LUT would cost sixteen shuffle-pipe lookups at the architectural
 *          minimum vector length, while gathering both stages stays length-agnostic with no VL-dependent branch.
 */
SZ_HELPER_AUTO svuint8_t sz_utf8_rune_flat_lookup_sve2_( //
    sz_u8_t const *page_lut, sz_u8_t const *flat, svuint8_t high_bytes_u8x, svuint8_t low_bytes_u8x) {

    // Widen both byte-lane vectors into four 32-bit-lane quarters each, preserving lane order. SVE vector types
    // are sizeless and cannot be arrayed, so the quarters stay individually named and the walk is unrolled.
    svuint16_t const high_bytes_low_half_u16x = svunpklo_u16(high_bytes_u8x);
    svuint16_t const high_bytes_high_half_u16x = svunpkhi_u16(high_bytes_u8x);
    svuint16_t const low_bytes_low_half_u16x = svunpklo_u16(low_bytes_u8x);
    svuint16_t const low_bytes_high_half_u16x = svunpkhi_u16(low_bytes_u8x);

    svuint32_t const first_quarter_u32x = sz_utf8_rune_flat_lookup_quarter_sve2_(
        page_lut, flat, svunpklo_u32(high_bytes_low_half_u16x), svunpklo_u32(low_bytes_low_half_u16x));
    svuint32_t const second_quarter_u32x = sz_utf8_rune_flat_lookup_quarter_sve2_(
        page_lut, flat, svunpkhi_u32(high_bytes_low_half_u16x), svunpkhi_u32(low_bytes_low_half_u16x));
    svuint32_t const third_quarter_u32x = sz_utf8_rune_flat_lookup_quarter_sve2_(
        page_lut, flat, svunpklo_u32(high_bytes_high_half_u16x), svunpklo_u32(low_bytes_high_half_u16x));
    svuint32_t const fourth_quarter_u32x = sz_utf8_rune_flat_lookup_quarter_sve2_(
        page_lut, flat, svunpkhi_u32(high_bytes_high_half_u16x), svunpkhi_u32(low_bytes_high_half_u16x));

    // Narrow the four 32-bit-lane quarters back into one byte-lane vector: `uzp1` on the 16-bit view keeps each
    // word's low half, then `uzp1` on the 8-bit view keeps each half-word's low byte. Values are `< 256`, so
    // both truncations are lossless and the original lane order is restored.
    svuint16_t const packed_low_half_u16x = svuzp1_u16(svreinterpret_u16_u32(first_quarter_u32x),
                                                       svreinterpret_u16_u32(second_quarter_u32x));
    svuint16_t const packed_high_half_u16x = svuzp1_u16(svreinterpret_u16_u32(third_quarter_u32x),
                                                        svreinterpret_u16_u32(fourth_quarter_u32x));
    return svuzp1_u8(svreinterpret_u8_u16(packed_low_half_u16x), svreinterpret_u8_u16(packed_high_half_u16x));
}

/** @brief  Read up to 256 LUT entries by per-lane u8 index via overlapping `svtbl_u8` chunks (gather-free); lanes
 *          outside a chunk's span select zero and OR away. Serves the astral cascade stage-1 tables. */
SZ_HELPER_AUTO svuint8_t sz_utf8_rune_lut_sve2_(sz_u8_t const *table, int count, svuint8_t index_u8x) {
    svbool_t const all_bytes_b8x = svptrue_b8();
    int const vector_length = (int)svcntb();
    svuint8_t result_u8x = svdup_n_u8(0);
    for (int base = 0; base < count; base += vector_length) {
        int const chunk_length = (count - base) < vector_length ? (count - base) : vector_length;
        svbool_t const load_b8x = svwhilelt_b8_u64(0, (sz_u64_t)chunk_length);
        svuint8_t const chunk_u8x = svld1_u8(load_b8x, table + base);
        svuint8_t const within_chunk_u8x = svsub_n_u8_x(all_bytes_b8x, index_u8x, (sz_u8_t)base);
        result_u8x = svorr_u8_x(all_bytes_b8x, result_u8x, svtbl_u8(chunk_u8x, within_chunk_u8x));
    }
    return result_u8x;
}

/** @brief  Select one of `tile_count` 16-entry rows by `selector` and index it by `within` (nibble cascade tile),
 *          serving the astral cascade stages on every property. */
SZ_HELPER_AUTO svuint8_t sz_utf8_rune_cascade_sve2_(sz_u8_t const *table, int tile_count, svuint8_t selector_u8x,
                                                    svuint8_t within_u8x) {
    svbool_t const all_bytes_b8x = svptrue_b8();
    svbool_t const row_b8x = svwhilelt_b8_u64(0, 16);
    svuint8_t result_u8x = svdup_n_u8(0);
    for (int tile = 0; tile < tile_count; ++tile) {
        svuint8_t const picked_u8x = svtbl_u8(svld1_u8(row_b8x, table + tile * 16), within_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(all_bytes_b8x, selector_u8x, (sz_u8_t)tile), picked_u8x, result_u8x);
    }
    return result_u8x;
}

/** @brief  Select and widen one 32-bit quarter of a byte-lane vector given its two unpacked 16-bit halves. */
SZ_HELPER_INLINE svuint32_t sz_utf8_rune_quarter_words_sve2_(svuint16_t half_lo_u16x, svuint16_t half_hi_u16x,
                                                             int high_half, int high_quarter) {
    svuint16_t const half_u16x = high_half ? half_hi_u16x : half_lo_u16x;
    return high_quarter ? svunpkhi_u32(half_u16x) : svunpklo_u32(half_u16x);
}

/**
 *  @brief  Emit the classified starts of one chunk as sequential UTF-32 runes, one 32-bit quarter at a time - the
 *          rune-valued SVE2 twin of @ref sz_utf8_rune_drain_icelake_. Each quarter compacts only two vectors: the
 *          start byte-offsets and the packed `ill << 7 | subpart_length` flags. Each start's lead and three
 *          trailing bytes come back in ONE `svtbl2` over the (chunk, peek) register pair - `offset * 0x01010101 +
 *          0x03020100` spreads every offset into its four byte indices - so the width-blend reads all operands
 *          from a single gathered little-endian word. Ill-formed lanes collapse to U+FFFD over their maximal
 *          ill-formed subpart. The cursor delta is the last emitted start's offset plus its subpart length, so an
 *          ill-formed trailing lane never skips bytes owing their own next U+FFFD.
 *
 *  @return Number of runes emitted; sets @p consumed_bytes to the byte span they cover (the resume cursor delta).
 */
SZ_HELPER_INLINE sz_size_t sz_utf8_rune_drain_sve2_( //
    svuint8_t bytes_u8x, svuint8_t peek_u8x,         //
    svbool_t emit_starts_b8x, svuint8_t flags_u8x,   //
    sz_rune_t *runes, sz_size_t capacity, sz_size_t *consumed_bytes) {

    svbool_t const all_b32x = svptrue_b32();
    sz_size_t const quarter_lanes = svcntw();
    svuint8x2_t const window_pair_u8x2 = svcreate2_u8(bytes_u8x, peek_u8x);
    svuint8_t const emit_flags_u8x = svdup_u8_z(emit_starts_b8x, 1);

    svuint16_t const emit_lo_u16x = svunpklo_u16(emit_flags_u8x), emit_hi_u16x = svunpkhi_u16(emit_flags_u8x);
    svuint16_t const flags_lo_u16x = svunpklo_u16(flags_u8x), flags_hi_u16x = svunpkhi_u16(flags_u8x);

    sz_size_t produced = 0, consumed = 0;
    for (int quarter = 0; quarter < 4 && produced < capacity; ++quarter) {
        int const high_half = quarter & 2, high_quarter = quarter & 1;
        svuint32_t const emit_q_u32x = sz_utf8_rune_quarter_words_sve2_(emit_lo_u16x, emit_hi_u16x, high_half,
                                                                        high_quarter);
        svbool_t const emit_b32x = svcmpne_n_u32(all_b32x, emit_q_u32x, 0);
        sz_size_t const found = svcntp_b32(all_b32x, emit_b32x);
        if (found == 0) continue;

        svuint32_t const offsets_u32x = svcompact_u32(emit_b32x, svindex_u32((sz_u32_t)(quarter * quarter_lanes), 1));
        svuint32_t const flags_q_u32x = svcompact_u32(
            emit_b32x, sz_utf8_rune_quarter_words_sve2_(flags_lo_u16x, flags_hi_u16x, high_half, high_quarter));

        svuint32_t const byte_indices_u32x = svadd_n_u32_x(all_b32x, svmul_n_u32_x(all_b32x, offsets_u32x, 0x01010101u),
                                                           0x03020100u);
        svuint32_t const quad_u32x = svreinterpret_u32_u8(
            svtbl2_u8(window_pair_u8x2, svreinterpret_u8_u32(byte_indices_u32x)));

        svuint32_t const trail1_u32x = svand_n_u32_x(all_b32x, svlsr_n_u32_x(all_b32x, quad_u32x, 8), 0x3F);
        svuint32_t const trail2_u32x = svand_n_u32_x(all_b32x, svlsr_n_u32_x(all_b32x, quad_u32x, 16), 0x3F);
        svuint32_t const trail3_u32x = svand_n_u32_x(all_b32x, svlsr_n_u32_x(all_b32x, quad_u32x, 24), 0x3F);
        svuint32_t const length_u32x = svand_n_u32_x(all_b32x, flags_q_u32x, 0x7F);

        // Well-formed lanes have subpart length == declared length, so the flags length picks the blend width;
        // ill-formed lanes blend garbage that the final U+FFFD select discards.
        svuint32_t codepoints_u32x = svand_n_u32_x(all_b32x, quad_u32x, 0xFF); // 1-byte keeps the lead
        svuint32_t const two_u32x = svorr_u32_x(
            all_b32x, svlsl_n_u32_x(all_b32x, svand_n_u32_x(all_b32x, quad_u32x, 0x1F), 6), trail1_u32x);
        codepoints_u32x = svsel_u32(svcmpeq_n_u32(all_b32x, length_u32x, 2), two_u32x, codepoints_u32x);
        svuint32_t const three_u32x = svorr_u32_x(
            all_b32x,
            svorr_u32_x(all_b32x, svlsl_n_u32_x(all_b32x, svand_n_u32_x(all_b32x, quad_u32x, 0x0F), 12),
                        svlsl_n_u32_x(all_b32x, trail1_u32x, 6)),
            trail2_u32x);
        codepoints_u32x = svsel_u32(svcmpeq_n_u32(all_b32x, length_u32x, 3), three_u32x, codepoints_u32x);
        svuint32_t const four_u32x = svorr_u32_x(
            all_b32x,
            svorr_u32_x(all_b32x, svlsl_n_u32_x(all_b32x, svand_n_u32_x(all_b32x, quad_u32x, 0x07), 18),
                        svlsl_n_u32_x(all_b32x, trail1_u32x, 12)),
            svorr_u32_x(all_b32x, svlsl_n_u32_x(all_b32x, trail2_u32x, 6), trail3_u32x));
        codepoints_u32x = svsel_u32(svcmpeq_n_u32(all_b32x, length_u32x, 4), four_u32x, codepoints_u32x);
        codepoints_u32x = svsel_u32(svcmpge_n_u32(all_b32x, flags_q_u32x, 0x80),
                                    svdup_n_u32((sz_u32_t)sz_rune_replacement_k), codepoints_u32x);

        sz_size_t const room = capacity - produced;
        sz_size_t const want = found < room ? found : room;
        svbool_t const store_b32x = svwhilelt_b32_u64(0, (sz_u64_t)want);
        svst1_u32(store_b32x, runes + produced, codepoints_u32x);

        consumed = (sz_size_t)svlastb_u32(store_b32x, offsets_u32x) + (sz_size_t)svlastb_u32(store_b32x, length_u32x);
        produced += want;
        if (want < found) break; // capacity filled mid-quarter
    }
    *consumed_bytes = consumed;
    return produced;
}

/**
 *  @brief  Emit precomputed 16-bit codepoints for one clean chunk (the ASCII+2-byte and ASCII+3-byte fast lanes):
 *          each quarter compacts just two vectors - the packed codepoint halves and `offset | length << 8` - with
 *          no per-start byte gathering and no width blend.
 *  @return Number of runes emitted; sets @p consumed_bytes to the byte span they cover (the resume cursor delta).
 */
SZ_HELPER_INLINE sz_size_t sz_utf8_rune_drain_packed_sve2_(                                   //
    svuint8_t cp_lo_u8x, svuint8_t cp_hi_u8x, svuint8_t length_u8x, svbool_t emit_starts_b8x, //
    sz_rune_t *runes, sz_size_t capacity, sz_size_t full_consumed, sz_size_t *consumed_bytes) {

    svbool_t const all_b32x = svptrue_b32();
    sz_size_t const quarter_lanes = svcntw();
    sz_size_t const emit_total = svcntp_b8(svptrue_b8(), emit_starts_b8x);
    svuint8_t const emit_flags_u8x = svdup_u8_z(emit_starts_b8x, 1);
    svuint16_t const emit_lo_u16x = svunpklo_u16(emit_flags_u8x), emit_hi_u16x = svunpkhi_u16(emit_flags_u8x);
    svuint16_t const cplo_lo_u16x = svunpklo_u16(cp_lo_u8x), cplo_hi_u16x = svunpkhi_u16(cp_lo_u8x);
    svuint16_t const cphi_lo_u16x = svunpklo_u16(cp_hi_u8x), cphi_hi_u16x = svunpkhi_u16(cp_hi_u8x);

    sz_size_t produced = 0, consumed = 0;
    for (int quarter = 0; quarter < 4 && produced < capacity; ++quarter) {
        int const high_half = quarter & 2, high_quarter = quarter & 1;
        svuint32_t const emit_q_u32x = sz_utf8_rune_quarter_words_sve2_(emit_lo_u16x, emit_hi_u16x, high_half,
                                                                        high_quarter);
        svbool_t const emit_b32x = svcmpne_n_u32(all_b32x, emit_q_u32x, 0);
        sz_size_t const found = svcntp_b32(all_b32x, emit_b32x);
        if (found == 0) continue;

        svuint32_t const cp_q_u32x = svorr_u32_x(
            all_b32x, sz_utf8_rune_quarter_words_sve2_(cplo_lo_u16x, cplo_hi_u16x, high_half, high_quarter),
            svlsl_n_u32_x(all_b32x,
                          sz_utf8_rune_quarter_words_sve2_(cphi_lo_u16x, cphi_hi_u16x, high_half, high_quarter), 8));
        svuint32_t const codepoints_u32x = svcompact_u32(emit_b32x, cp_q_u32x);

        sz_size_t const room = capacity - produced;
        sz_size_t const want = found < room ? found : room;
        svbool_t const store_b32x = svwhilelt_b32_u64(0, (sz_u64_t)want);
        svst1_u32(store_b32x, runes + produced, codepoints_u32x);

        // The `offset | length << 8` pack is only compacted when this quarter may be the capacity cut - a full
        // drain's span is the caller-precomputed `full_consumed` (chunk edge plus trailing-sequence overhang).
        if (want < found || produced + want >= capacity) {
            svuint16_t const len_lo_u16x = svunpklo_u16(length_u8x), len_hi_u16x = svunpkhi_u16(length_u8x);
            svuint32_t const pack_q_u32x = svorr_u32_x(
                all_b32x, svindex_u32((sz_u32_t)(quarter * quarter_lanes), 1),
                svlsl_n_u32_x(all_b32x,
                              sz_utf8_rune_quarter_words_sve2_(len_lo_u16x, len_hi_u16x, high_half, high_quarter), 8));
            svuint32_t const pack_u32x = svcompact_u32(emit_b32x, pack_q_u32x);
            sz_u32_t const last_pack = svlastb_u32(store_b32x, pack_u32x);
            consumed = (sz_size_t)(last_pack & 0xFF) + (sz_size_t)(last_pack >> 8);
        }
        produced += want;
        if (want < found) break; // capacity filled mid-quarter
    }
    *consumed_bytes = produced == emit_total ? full_consumed : consumed;
    return produced;
}

/**
 *  @brief  Decode one multi-chunk window of @p text into dense UTF-32 @p runes by the uniform "classify -> per-lane
 *          well-formed + orphan promotion -> compact emitted starts -> gather -> width-blend -> blend U+FFFD" path,
 *          emitting at most @p runes_capacity runes and returning the resume cursor. The window is 64 bytes walked
 *          as `64 / svcntb()` streaming register chunks (4 at VL=128, 2 at 256, 1 at 512+), each peeking one vector
 *          ahead so `svext` supplies cross-chunk neighbours and maximal-subpart coverage carries across the edge in
 *          the value domain. The decode is TOTAL: clean and dirty bytes are handled in-vector, one U+FFFD per
 *          maximal ill-formed subpart (Unicode 17.0 §3.9 / W3C), bit-exact with @ref sz_utf8_decode_serial. Only a
 *          start whose declared sequence crosses the WINDOW edge defers to the next call; the step declines
 *          (`*runes_unpacked == 0`, cursor unchanged) ONLY when that happens on the very first lead (a boundary
 *          truncation), which the public entry finalizes without a serial re-decode.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_decode_once_sve2_( //
    sz_cptr_t text, sz_size_t length,               //
    sz_rune_t *runes, sz_size_t runes_capacity,     //
    sz_size_t *runes_unpacked) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const vector_bytes = svcntb();
    sz_size_t const chunk_bytes = vector_bytes < 64 ? vector_bytes : 64; // u8 lane-index domain cap
    sz_size_t const window_capacity = (64 / chunk_bytes) * chunk_bytes;  // one 64-byte window, or one giant chunk
    sz_size_t const window = window_capacity < length ? window_capacity : length;
    svbool_t const all_b8x = svptrue_b8();
    svuint8_t const lane_iota_u8x = svindex_u8(0, 1);
    svuint8_t const zeros_u8x = svdup_n_u8(0);
    svuint8_t const length_lut_u8x = svdupq_n_u8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4);
    sz_size_t const quarter_lanes = svcntw();

    sz_size_t produced_total = 0, consumed_total = 0;
    svuint8_t carry_step2_u8x = svdup_n_u8(0), carry_step3_u8x = svdup_n_u8(0), carry_step4_u8x = svdup_n_u8(0);

    sz_size_t chunk_base = 0;
    while (chunk_base < window && produced_total < runes_capacity) {
        sz_size_t const ahead = window - chunk_base;

        // Pure 3-byte tile: when the cursor sits on an E0..EF lead - the steady state of CJK, Hangul, and Thai
        // prose - one `svld3_u8` structure load deinterleaves a whole vector of (lead, cont1, cont2) triples, so a
        // handful of range checks validates `svcntb()` runes at once and the codepoints assemble and store with no
        // compaction. Any irregularity falls through to the chunked engine below at the same cursor.
        if ((text_u8[chunk_base] & 0xF0) == 0xE0) {
            sz_size_t tile_runes = (length - chunk_base) / 3; // tiles use no lane offsets, so the window cap waives
            sz_size_t const tile_room = runes_capacity - produced_total;
            if (tile_runes > vector_bytes) tile_runes = vector_bytes;
            if (tile_runes > tile_room) tile_runes = tile_room;
            if (tile_runes) {
                svbool_t const rune_b8x = svwhilelt_b8_u64(0, (sz_u64_t)tile_runes);
                svuint8x3_t const triple_u8x3 = svld3_u8(rune_b8x, text_u8 + chunk_base);
                svuint8_t const b0_u8x = svget3_u8(triple_u8x3, 0);
                svuint8_t const b1_u8x = svget3_u8(triple_u8x3, 1);
                svuint8_t const b2_u8x = svget3_u8(triple_u8x3, 2);
                svbool_t const shape_b8x = svand_b_z(
                    rune_b8x, svcmpeq_n_u8(rune_b8x, svand_n_u8_x(rune_b8x, b0_u8x, 0xF0), 0xE0),
                    svand_b_z(rune_b8x, svcmpeq_n_u8(rune_b8x, svand_n_u8_x(rune_b8x, b1_u8x, 0xC0), 0x80),
                              svcmpeq_n_u8(rune_b8x, svand_n_u8_x(rune_b8x, b2_u8x, 0xC0), 0x80)));
                svbool_t const bounds_bad_b8x = svorr_b_z(
                    rune_b8x,
                    svand_b_z(rune_b8x, svcmpeq_n_u8(rune_b8x, b0_u8x, 0xE0), svcmplt_n_u8(rune_b8x, b1_u8x, 0xA0)),
                    svand_b_z(rune_b8x, svcmpeq_n_u8(rune_b8x, b0_u8x, 0xED), svcmpge_n_u8(rune_b8x, b1_u8x, 0xA0)));
                // Emit the clean 3-byte prefix when it covers the whole attempt (a window- or capacity-clamped
                // tile) or is long enough to out-run the chunked engine; a short prefix cut by an irregular byte
                // means separator-dense text (Hangul, Devanagari), which the chunked ASCII+3-byte lane below
                // digests together with its separators.
                svbool_t const bad_b8x = svorr_b_z(rune_b8x, svbic_b_z(rune_b8x, rune_b8x, shape_b8x), bounds_bad_b8x);
                sz_size_t const clean_runes = svcntp_b8(rune_b8x, svbrkb_b_z(rune_b8x, bad_b8x));
                int const tile_pays = clean_runes == tile_runes || clean_runes >= (vector_bytes >> 2);
                tile_runes = clean_runes;
                if (tile_runes && tile_pays) {
                    svbool_t const all_pair_b8x = svptrue_b8();
                    svuint8_t const cp_lo_u8x = svorr_u8_x(
                        all_pair_b8x, svand_n_u8_x(all_pair_b8x, svlsl_n_u8_x(all_pair_b8x, b1_u8x, 6), 0xC0),
                        svand_n_u8_x(all_pair_b8x, b2_u8x, 0x3F));
                    svuint8_t const cp_hi_u8x = svorr_u8_x(
                        all_pair_b8x, svlsl_n_u8_x(all_pair_b8x, svand_n_u8_x(all_pair_b8x, b0_u8x, 0x0F), 4),
                        svand_n_u8_x(all_pair_b8x, svlsr_n_u8_x(all_pair_b8x, b1_u8x, 2), 0x0F));
                    svuint16_t const first_half_u16x = svreinterpret_u16_u8(svzip1_u8(cp_lo_u8x, cp_hi_u8x));
                    svuint16_t const second_half_u16x = svreinterpret_u16_u8(svzip2_u8(cp_lo_u8x, cp_hi_u8x));
                    for (sz_size_t emitted = 0; emitted < tile_runes; emitted += quarter_lanes) {
                        int const quarter = (int)(emitted / quarter_lanes);
                        svuint32_t const words_u32x = sz_utf8_rune_quarter_words_sve2_(
                            first_half_u16x, second_half_u16x, quarter & 2, quarter & 1);
                        svbool_t const store_b32x = svwhilelt_b32_u64(0, (sz_u64_t)(tile_runes - emitted));
                        svst1_u32(store_b32x, runes + produced_total + emitted, words_u32x);
                    }
                    produced_total += tile_runes;
                    consumed_total = chunk_base + tile_runes * 3;
                    carry_step2_u8x = carry_step3_u8x = carry_step4_u8x = zeros_u8x;
                    chunk_base += tile_runes * 3;
                    continue;
                }
            }
        }

        sz_size_t const chunk_span = ahead < chunk_bytes ? ahead : chunk_bytes;
        sz_size_t const peek_span = ahead > chunk_bytes
                                        ? (ahead - chunk_bytes < chunk_bytes ? ahead - chunk_bytes : chunk_bytes)
                                        : 0;
        svbool_t const loaded_b8x = svwhilelt_b8_u64(0, (sz_u64_t)chunk_span);
        svbool_t const peek_b8x = svwhilelt_b8_u64(0, (sz_u64_t)peek_span);
        svuint8_t const bytes_u8x = svld1_u8(loaded_b8x, text_u8 + chunk_base);
        svuint8_t const peek_u8x = svld1_u8(peek_b8x, text_u8 + chunk_base + (peek_span ? chunk_bytes : 0));

        // ASCII chunk fast lane: the sign bit clear in every loaded lane widens each byte quarter directly (an
        // `svunpk` cascade per quarter, no classification). Chunk granularity matters: prose interleaves short
        // ASCII spans (spaces, digits, markup) between multibyte runs, and each such chunk skips the whole engine.
        svbool_t const high_bit_b8x = svcmplt_n_s8(loaded_b8x, svreinterpret_s8_u8(bytes_u8x), 0);
        if (!svptest_any(loaded_b8x, high_bit_b8x)) {
            sz_size_t const room = runes_capacity - produced_total;
            sz_size_t const ascii_span = chunk_span < room ? chunk_span : room;
            svuint16_t const ascii_lo_u16x = svunpklo_u16(bytes_u8x), ascii_hi_u16x = svunpkhi_u16(bytes_u8x);
            if (ascii_span == vector_bytes) { // full chunk, full room: four unmasked quarter stores
                svbool_t const all_b32x = svptrue_b32();
                svst1_u32(all_b32x, runes + produced_total, svunpklo_u32(ascii_lo_u16x));
                svst1_u32(all_b32x, runes + produced_total + quarter_lanes, svunpkhi_u32(ascii_lo_u16x));
                svst1_u32(all_b32x, runes + produced_total + 2 * quarter_lanes, svunpklo_u32(ascii_hi_u16x));
                svst1_u32(all_b32x, runes + produced_total + 3 * quarter_lanes, svunpkhi_u32(ascii_hi_u16x));
            }
            else
                for (sz_size_t emitted = 0; emitted < ascii_span; emitted += quarter_lanes) {
                    int const quarter = (int)(emitted / quarter_lanes);
                    svuint32_t const words_u32x = sz_utf8_rune_quarter_words_sve2_(ascii_lo_u16x, ascii_hi_u16x,
                                                                                   quarter & 2, quarter & 1);
                    svbool_t const store_b32x = svwhilelt_b32_u64(0, (sz_u64_t)(ascii_span - emitted));
                    svst1_u32(store_b32x, runes + produced_total + emitted, words_u32x);
                }
            produced_total += ascii_span;
            consumed_total = chunk_base + ascii_span;
            carry_step2_u8x = carry_step3_u8x = carry_step4_u8x = zeros_u8x;
            chunk_base += chunk_bytes;
            continue;
        }

        // Single-source classification: per-lane length from the lead's high nibble via a 16-entry `svtbl_u8` LUT
        // (`{1x12,2,2,3,4}`, the SAME table the serial / NEON references use), so a lead and its length can never
        // disagree and 0xF8..0xFF map to length 4 and cannot slip the gate.
        svbool_t const is_continuation_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xC0), 0x80);
        svbool_t const starts_b8x = svbic_b_z(loaded_b8x, loaded_b8x, is_continuation_b8x);
        svuint8_t const high_nibble_u8x = svlsr_n_u8_x(all_b8x, bytes_u8x, 4);
        svuint8_t const length_per_lane_u8x = svtbl_u8(length_lut_u8x, high_nibble_u8x);

        // Defer EVERY start whose declared sequence would cross the WINDOW edge (never a mere chunk edge - the peek
        // vector covers those); the FIRST overrunning start bounds the decodable prefix and the loop breaks after
        // this chunk so its bytes resume in the next call. Overruns can only exist within reach of the window edge.
        sz_size_t decodable_end = chunk_span;
        svbool_t decodable_b8x = loaded_b8x;
        if (ahead <= chunk_bytes + 3) {
            svuint8_t const sequence_end_u8x = svadd_u8_x(all_b8x, lane_iota_u8x, length_per_lane_u8x);
            svbool_t const overruns_b8x = svand_b_z(starts_b8x, starts_b8x,
                                                    svcmpgt_n_u8(all_b8x, sequence_end_u8x, (sz_u8_t)ahead));
            if (svptest_any(starts_b8x, overruns_b8x)) {
                svuint8_t const overrun_lane_u8x = svsel_u8(overruns_b8x, lane_iota_u8x,
                                                            svdup_n_u8((sz_u8_t)chunk_span));
                decodable_end = (sz_size_t)svminv_u8(all_b8x, overrun_lane_u8x);
                decodable_b8x = svwhilelt_b8_u64(0, (sz_u64_t)decodable_end);
            }
        }
        svbool_t const decodable_starts_b8x = svand_b_z(decodable_b8x, starts_b8x, decodable_b8x);

        // Continuation availability across the chunk edge: `svext` on the 0/1 value vectors so lane `i` of
        // `cont_k` reflects lane `i + k`, even when that lane lives in the peeked next chunk.
        svuint8_t const cont_cur_u8x = svdup_u8_z(is_continuation_b8x, 1);
        svbool_t const peek_cont_b8x = svcmpeq_n_u8(peek_b8x, svand_n_u8_x(peek_b8x, peek_u8x, 0xC0), 0x80);
        svuint8_t const cont_peek_u8x = svdup_u8_z(peek_cont_b8x, 1);
        svuint8_t const down1_u8x = svext_u8(cont_cur_u8x, cont_peek_u8x, 1);
        svuint8_t const next1_u8x = svext_u8(bytes_u8x, peek_u8x, 1);

        // Fast lane "ASCII + 2-byte": every non-ASCII byte belongs to a well-formed C2..DF + continuation pair -
        // the shape of Latin-script, Cyrillic, Greek, Hebrew, and Arabic prose with ASCII separators. Structure is
        // proven in three predicate tests: no 3-byte-or-wider or overlong lead anywhere, the continuation pattern
        // equals the lead pattern smeared one lane up (with the previous chunk's carry), and every decodable lead
        // has its continuation present (the peeked lane covers the chunk edge). Codepoints then assemble in the
        // byte domain and drain without gathers.
        svbool_t const lead2_b8x = svand_b_z(loaded_b8x, svcmpge_n_u8(loaded_b8x, bytes_u8x, 0xC2),
                                             svcmple_n_u8(loaded_b8x, bytes_u8x, 0xDF));
        int const any_wide = svptest_any(loaded_b8x, svcmpge_n_u8(loaded_b8x, bytes_u8x, 0xE0));
        if (!any_wide &&
            !svptest_any(loaded_b8x, svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xFE), 0xC0))) {
            svuint8_t const lead2_u8x = svdup_u8_z(lead2_b8x, 1);
            svuint8_t const covered2_u8x = svorr_u8_x(
                all_b8x,
                svorr_u8_x(all_b8x,
                           sz_utf8_shift_value_up_pair_sve2_(carry_step2_u8x, lead2_u8x, 1, lane_iota_u8x,
                                                             (sz_u8_t)chunk_bytes),
                           sz_utf8_shift_value_up_pair_sve2_(carry_step3_u8x, zeros_u8x, 2, lane_iota_u8x,
                                                             (sz_u8_t)chunk_bytes)),
                sz_utf8_shift_value_up_pair_sve2_(carry_step4_u8x, zeros_u8x, 3, lane_iota_u8x, (sz_u8_t)chunk_bytes));
            svbool_t const mismatch_b8x = sveor_b_z(decodable_b8x, is_continuation_b8x,
                                                    svcmpne_n_u8(all_b8x, covered2_u8x, 0));
            svbool_t const incomplete_b8x = svand_b_z(decodable_b8x, lead2_b8x, svcmpeq_n_u8(all_b8x, down1_u8x, 0));
            if (!svptest_any(decodable_b8x, mismatch_b8x) && !svptest_any(decodable_b8x, incomplete_b8x)) {
                svuint8_t const cp_lo_u8x = svsel_u8(
                    lead2_b8x,
                    svorr_u8_x(all_b8x, svand_n_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, bytes_u8x, 6), 0xC0),
                               svand_n_u8_x(all_b8x, next1_u8x, 0x3F)),
                    bytes_u8x);
                svuint8_t const cp_hi_u8x = svand_n_u8_z(lead2_b8x, svlsr_n_u8_x(all_b8x, bytes_u8x, 2), 0x07);
                svuint8_t const length2_u8x = svadd_u8_x(all_b8x, svdup_n_u8(1), lead2_u8x);
                if (svptest_any(decodable_b8x, decodable_starts_b8x)) {
                    sz_size_t const full_consumed = decodable_end + (sz_size_t)svlastb_u8(decodable_b8x, lead2_u8x);
                    sz_size_t consumed_local = 0;
                    sz_size_t const produced = sz_utf8_rune_drain_packed_sve2_(
                        cp_lo_u8x, cp_hi_u8x, length2_u8x, decodable_starts_b8x, runes + produced_total,
                        runes_capacity - produced_total, full_consumed, &consumed_local);
                    produced_total += produced;
                    consumed_total = chunk_base + consumed_local;
                }
                carry_step2_u8x = svdup_u8_z(svand_b_z(decodable_b8x, lead2_b8x, decodable_b8x), 1);
                carry_step3_u8x = carry_step4_u8x = zeros_u8x;
                if (decodable_end < chunk_span) break;
                chunk_base += chunk_bytes;
                continue;
            }
        }

        // Fast lane "ASCII + 3-byte": every decodable start is ASCII or an E0..EF lead with both continuations -
        // the shape of CJK, Hangul, Devanagari, and Thai prose with ASCII separators. Same three structural tests
        // plus the E0 overlong / ED surrogate first-continuation bounds.
        svbool_t const lead3_b8x = svand_b_z(starts_b8x, starts_b8x,
                                             svcmpeq_n_u8(all_b8x, svand_n_u8_x(all_b8x, bytes_u8x, 0xF0), 0xE0));
        svbool_t const ascii_b8x = svbic_b_z(loaded_b8x, loaded_b8x, high_bit_b8x);
        if (any_wide && !svptest_any(decodable_b8x,
                                     svbic_b_z(decodable_b8x, svbic_b_z(decodable_b8x, decodable_starts_b8x, lead3_b8x),
                                               ascii_b8x))) {
            svuint8_t const lead3_u8x = svdup_u8_z(lead3_b8x, 1);
            svuint8_t const covered3_u8x = svorr_u8_x(
                all_b8x,
                svorr_u8_x(all_b8x,
                           sz_utf8_shift_value_up_pair_sve2_(carry_step2_u8x, lead3_u8x, 1, lane_iota_u8x,
                                                             (sz_u8_t)chunk_bytes),
                           sz_utf8_shift_value_up_pair_sve2_(carry_step3_u8x, lead3_u8x, 2, lane_iota_u8x,
                                                             (sz_u8_t)chunk_bytes)),
                sz_utf8_shift_value_up_pair_sve2_(carry_step4_u8x, zeros_u8x, 3, lane_iota_u8x, (sz_u8_t)chunk_bytes));
            svbool_t const mismatch_b8x = sveor_b_z(decodable_b8x, is_continuation_b8x,
                                                    svcmpne_n_u8(all_b8x, covered3_u8x, 0));
            svuint8_t const down2_u8x = svext_u8(cont_cur_u8x, cont_peek_u8x, 2);
            svbool_t const incomplete_b8x = svand_b_z(
                decodable_b8x, lead3_b8x,
                svorr_b_z(all_b8x, svcmpeq_n_u8(all_b8x, down1_u8x, 0), svcmpeq_n_u8(all_b8x, down2_u8x, 0)));
            svbool_t const bounds_bad_b8x = svorr_b_z(decodable_b8x,
                                                      svand_b_z(decodable_b8x, svcmpeq_n_u8(all_b8x, bytes_u8x, 0xE0),
                                                                svcmplt_n_u8(all_b8x, next1_u8x, 0xA0)),
                                                      svand_b_z(decodable_b8x, svcmpeq_n_u8(all_b8x, bytes_u8x, 0xED),
                                                                svcmpge_n_u8(all_b8x, next1_u8x, 0xA0)));
            if (!svptest_any(decodable_b8x, mismatch_b8x) && !svptest_any(decodable_b8x, incomplete_b8x) &&
                !svptest_any(decodable_b8x, bounds_bad_b8x)) {
                svuint8_t const next2_u8x = svext_u8(bytes_u8x, peek_u8x, 2);
                svuint8_t const three_lo_u8x = svorr_u8_x(
                    all_b8x, svand_n_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, next1_u8x, 6), 0xC0),
                    svand_n_u8_x(all_b8x, next2_u8x, 0x3F));
                svuint8_t const three_hi_u8x = svorr_u8_x(
                    all_b8x, svlsl_n_u8_x(all_b8x, svand_n_u8_x(all_b8x, bytes_u8x, 0x0F), 4),
                    svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, next1_u8x, 2), 0x0F));
                svuint8_t const cp_lo_u8x = svsel_u8(lead3_b8x, three_lo_u8x, bytes_u8x);
                svuint8_t const cp_hi_u8x = svand_u8_z(lead3_b8x, three_hi_u8x, three_hi_u8x);
                svuint8_t const length3_u8x = svadd_u8_x(all_b8x, svdup_n_u8(1),
                                                         svadd_u8_x(all_b8x, lead3_u8x, lead3_u8x));
                if (svptest_any(decodable_b8x, decodable_starts_b8x)) {
                    // Trailing overhang: a 3-byte lead in the last lane owes two peeked continuations, one in the
                    // second-to-last lane owes one; a clean chunk can hold at most one of the two.
                    sz_size_t const trailing_lead = (sz_size_t)svlastb_u8(decodable_b8x, lead3_u8x);
                    sz_size_t const second_lead =
                        decodable_end > 1
                            ? (sz_size_t)svlastb_u8(svwhilelt_b8_u64(0, (sz_u64_t)(decodable_end - 1)), lead3_u8x)
                            : 0;
                    sz_size_t const full_consumed = decodable_end + trailing_lead * 2 + second_lead;
                    sz_size_t consumed_local = 0;
                    sz_size_t const produced = sz_utf8_rune_drain_packed_sve2_(
                        cp_lo_u8x, cp_hi_u8x, length3_u8x, decodable_starts_b8x, runes + produced_total,
                        runes_capacity - produced_total, full_consumed, &consumed_local);
                    produced_total += produced;
                    consumed_total = chunk_base + consumed_local;
                }
                carry_step2_u8x = carry_step3_u8x = svdup_u8_z(svand_b_z(decodable_b8x, lead3_b8x, decodable_b8x), 1);
                carry_step4_u8x = zeros_u8x;
                if (decodable_end < chunk_span) break;
                chunk_base += chunk_bytes;
                continue;
            }
        }

        svbool_t const cont2_b8x = svcmpne_n_u8(all_b8x, svext_u8(cont_cur_u8x, cont_peek_u8x, 2), 0);
        svbool_t const cont3_b8x = svcmpne_n_u8(all_b8x, svext_u8(cont_cur_u8x, cont_peek_u8x, 3), 0);

        // First-continuation bounds per lead: default [0x80, 0xBF] (== "is a continuation"), E0 lifts the floor to
        // 0xA0 and F0 to 0x90 (overlong), ED drops the ceiling to 0x9F (surrogates) and F4 to 0x8F (> U+10FFFF).
        // One in-range test subsumes the presence check AND the E0/ED/F0/F4 range rules.
        svuint8_t min1_u8x = svdup_n_u8(0x80);
        min1_u8x = svorr_n_u8_m(svcmpeq_n_u8(all_b8x, bytes_u8x, 0xE0), min1_u8x, 0x20);
        min1_u8x = svorr_n_u8_m(svcmpeq_n_u8(all_b8x, bytes_u8x, 0xF0), min1_u8x, 0x10);
        svuint8_t max1_u8x = svdup_n_u8(0xBF);
        max1_u8x = sveor_n_u8_m(svcmpeq_n_u8(all_b8x, bytes_u8x, 0xED), max1_u8x, 0x20); // 0xBF ^ 0x20 = 0x9F
        max1_u8x = sveor_n_u8_m(svcmpeq_n_u8(all_b8x, bytes_u8x, 0xF4), max1_u8x, 0x30); // 0xBF ^ 0x30 = 0x8F
        svbool_t const b1_ok_b8x = svand_b_z(all_b8x, svcmpge_u8(all_b8x, next1_u8x, min1_u8x),
                                             svcmple_u8(all_b8x, next1_u8x, max1_u8x));

        svbool_t const len_ge2_b8x = svand_b_z(starts_b8x, starts_b8x, svcmpge_n_u8(all_b8x, length_per_lane_u8x, 2));
        svbool_t const len_ge3_b8x = svand_b_z(starts_b8x, starts_b8x, svcmpge_n_u8(all_b8x, length_per_lane_u8x, 3));
        svbool_t const len_ge4_b8x = svand_b_z(starts_b8x, starts_b8x, svcmpge_n_u8(all_b8x, length_per_lane_u8x, 4));
        svbool_t const len_eq2_b8x = svbic_b_z(all_b8x, len_ge2_b8x, len_ge3_b8x);

        // Bad lead: 2-byte lead < 0xC2 (C0/C1 overlong), or 4-byte lead > 0xF4 (out of range, incl. F5..FF).
        svbool_t const bad_lead_b8x = svorr_b_z(
            starts_b8x, svand_b_z(starts_b8x, len_eq2_b8x, svcmplt_n_u8(all_b8x, bytes_u8x, 0xC2)),
            svand_b_z(starts_b8x, len_ge4_b8x, svcmpgt_n_u8(all_b8x, bytes_u8x, 0xF4)));
        svbool_t const first_ok_b8x = svand_b_z(starts_b8x, svand_b_z(starts_b8x, len_ge2_b8x, b1_ok_b8x),
                                                svnot_b_z(starts_b8x, bad_lead_b8x));

        // Well-formed leads (vectorized `sz_rune_decode` success), restricted to the decodable span.
        svbool_t const wf1_b8x = svbic_b_z(starts_b8x, starts_b8x, len_ge2_b8x);
        svbool_t const wf2_b8x = svbic_b_z(starts_b8x, svand_b_z(starts_b8x, len_eq2_b8x, first_ok_b8x), len_ge3_b8x);
        svbool_t const wf3_b8x = svand_b_z(starts_b8x, svand_b_z(starts_b8x, len_ge3_b8x, first_ok_b8x), cont2_b8x);
        svbool_t const wf3_only_b8x = svbic_b_z(starts_b8x, wf3_b8x, len_ge4_b8x);
        svbool_t const wf4_b8x = svand_b_z(starts_b8x, svand_b_z(starts_b8x, len_ge4_b8x, wf3_b8x), cont3_b8x);
        svbool_t const well_formed_b8x = svand_b_z(
            decodable_b8x,
            svorr_b_z(all_b8x, svorr_b_z(all_b8x, wf1_b8x, wf2_b8x), svorr_b_z(all_b8x, wf3_only_b8x, wf4_b8x)),
            decodable_b8x);

        // Per-lane maximal-subpart steps (mirror of `sz_utf8_maximal_subpart_`): start at 1 and extend across each
        // continuation slot a well-formed sequence would still accept.
        svbool_t const step2_b8x = svand_b_z(starts_b8x, len_ge2_b8x, first_ok_b8x);
        svbool_t const step3_b8x = svand_b_z(starts_b8x, svand_b_z(starts_b8x, step2_b8x, len_ge3_b8x), cont2_b8x);
        svbool_t const step4_b8x = svand_b_z(starts_b8x, svand_b_z(starts_b8x, step3_b8x, len_ge4_b8x), cont3_b8x);

        // Orphan promotion: a continuation byte not covered by ANY lead's maximal-subpart span becomes its own
        // 1-byte U+FFFD. Coverage is the decodable steps smeared UP by their offset, with the previous chunk's
        // steps carried across the edge so a sequence straddling two chunks covers its trailing bytes.
        svuint8_t const step2_u8x = svdup_u8_z(svand_b_z(decodable_b8x, step2_b8x, decodable_b8x), 1);
        svuint8_t const step3_u8x = svdup_u8_z(svand_b_z(decodable_b8x, step3_b8x, decodable_b8x), 1);
        svuint8_t const step4_u8x = svdup_u8_z(svand_b_z(decodable_b8x, step4_b8x, decodable_b8x), 1);
        svuint8_t const covered_u8x = svorr_u8_x(
            all_b8x,
            svorr_u8_x(
                all_b8x,
                sz_utf8_shift_value_up_pair_sve2_(carry_step2_u8x, step2_u8x, 1, lane_iota_u8x, (sz_u8_t)chunk_bytes),
                sz_utf8_shift_value_up_pair_sve2_(carry_step3_u8x, step3_u8x, 2, lane_iota_u8x, (sz_u8_t)chunk_bytes)),
            sz_utf8_shift_value_up_pair_sve2_(carry_step4_u8x, step4_u8x, 3, lane_iota_u8x, (sz_u8_t)chunk_bytes));
        svbool_t const covered_b8x = svcmpne_n_u8(all_b8x, covered_u8x, 0);
        svbool_t const orphan_b8x = svbic_b_z(
            decodable_b8x, svand_b_z(decodable_b8x, is_continuation_b8x, decodable_b8x), covered_b8x);
        svbool_t const emit_starts_b8x = svand_b_z(
            decodable_b8x, svorr_b_z(decodable_b8x, decodable_starts_b8x, orphan_b8x), decodable_b8x);
        carry_step2_u8x = step2_u8x, carry_step3_u8x = step3_u8x, carry_step4_u8x = step4_u8x;

        if (svptest_any(decodable_b8x, emit_starts_b8x)) {
            // Orphans are continuations, never well-formed -> ill_formed = emit_starts & ~well_formed. The packed
            // per-lane flags carry `ill << 7 | subpart_length` where length = 1 + step2 + step3 + step4.
            svbool_t const ill_formed_b8x = svbic_b_z(decodable_b8x, emit_starts_b8x, well_formed_b8x);
            svuint8_t flags_u8x = svdup_n_u8(1);
            flags_u8x = svadd_n_u8_m(step2_b8x, flags_u8x, 1);
            flags_u8x = svadd_n_u8_m(step3_b8x, flags_u8x, 1);
            flags_u8x = svadd_n_u8_m(step4_b8x, flags_u8x, 1);
            flags_u8x = svorr_n_u8_m(ill_formed_b8x, flags_u8x, 0x80);

            sz_size_t consumed_local = 0;
            sz_size_t const produced = sz_utf8_rune_drain_sve2_(bytes_u8x, peek_u8x, emit_starts_b8x, flags_u8x,
                                                                runes + produced_total, runes_capacity - produced_total,
                                                                &consumed_local);
            produced_total += produced;
            consumed_total = chunk_base + consumed_local;
        }
        if (decodable_end < chunk_span) break; // the deferred sequence resumes as the next call's first lead
        chunk_base += chunk_bytes;
    }
    *runes_unpacked = produced_total;
    return text + consumed_total;
}

/**
 *  @brief  Decode @p text into dense UTF-32 @p runes (SVE2). Drives @ref sz_utf8_decode_once_sve2_ window by
 *          window. The in-vector step decodes its whole decodable span; `step_unpacked == 0` only when the very
 *          first lead declares a sequence crossing the window edge (a boundary truncation). A resumable truncation
 *          breaks and awaits more bytes; a bad/overlong truncated lead at the edge finalizes to one U+FFFD over its
 *          maximal ill-formed subpart - a bounded <=3-byte finalize, never a serial window re-decode.
 */
SZ_API_COMPTIME sz_cptr_t sz_utf8_decode_sve2(  //
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
