/**
 *  @file   include/stringzilla/utf8_graphemes/neon.h
 *  @author Ash Vardanian
 *  @brief  NEON (AArch64) backend for UAX-29 extended grapheme cluster boundaries, fully vectorized end-to-end.
 *
 *  The NEON twin of the Haswell (AVX2) grapheme kernel. Each 64-byte window lives as four `uint8x16_t` quarters; every
 *  lane is decoded through the shared codepoint substrate (`sz_utf8_rune_decode_window_neon_`), classified
 *  gather-free into one packed Grapheme_Cluster_Break descriptor per lane by a register-resident `vqtbl` nibble cascade
 *  (the NEON twin of the AVX2 `vpshufb` BMP trie + 4-stage astral trie), compacted to a codepoint-dense descriptor byte
 *  buffer, and resolved by the SHARED portable boundary engine (`sz_grapheme_window_boundaries_`). No `vpgatherdd` /
 *  table gather, no `_pext_u64` / `_pdep_u64` (NEON lacks them — replaced by scalar sparse bit shuffles), no per-lane
 *  scalar rule loop, no serial deferral. The classifier algorithm, tables, and rule engine are reused verbatim.
 */
#ifndef STRINGZILLA_UTF8_GRAPHEMES_NEON_H_
#define STRINGZILLA_UTF8_GRAPHEMES_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_graphemes/tables.h"
#include "stringzilla/utf8_graphemes/serial.h"
#include "stringzilla/utf8_runes/neon.h"

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

#pragma region Branchless bit gather and scatter

/** @brief  Precomputed routing masks for the Hacker's-Delight bit-compress network — the BMI2-free `pext`/`pdep`.
 *          NEON has no `pext`/`pdep`, and one grapheme window applies the @b same `start_lanes` selector to 18 class
 *          masks plus the boundary scatter, so the six routing masks are built once and shared. This replaces the old
 *          O(popcount) per-call scalar loops with a branchless, constant-time apply. Bit-exact with BMI2. */
typedef struct sz_grapheme_bit_route_t {
    sz_u64_t selector;
    sz_u64_t move_masks[6];
} sz_grapheme_bit_route_t;

/** @brief  Build the per-selector routing plan once; reuse it for every gather/scatter sharing this @p selector. */
SZ_HELPER_AUTO sz_grapheme_bit_route_t sz_grapheme_bit_route_build_(sz_u64_t selector) {
    sz_grapheme_bit_route_t route;
    route.selector = selector;
    sz_u64_t routing_selector = selector;
    sz_u64_t shifted_complement = ~selector << 1;
    for (int step = 0; step < 6; ++step) {
        sz_u64_t parallel_prefix = shifted_complement ^ (shifted_complement << 1);
        parallel_prefix ^= parallel_prefix << 2;
        parallel_prefix ^= parallel_prefix << 4;
        parallel_prefix ^= parallel_prefix << 8;
        parallel_prefix ^= parallel_prefix << 16;
        parallel_prefix ^= parallel_prefix << 32;
        sz_u64_t const movable_selector = parallel_prefix & routing_selector;
        route.move_masks[step] = movable_selector;
        routing_selector = (routing_selector ^ movable_selector) | (movable_selector >> (1u << step));
        shifted_complement &= ~parallel_prefix;
    }
    return route;
}

/** @brief  `pext` via a precomputed @p route: gather the bits of @p value at the selector positions to the low end. */
SZ_HELPER_AUTO sz_u64_t sz_grapheme_bit_gather_(sz_u64_t value, sz_grapheme_bit_route_t const *route) {
    value &= route->selector;
    for (int step = 0; step < 6; ++step) {
        sz_u64_t const movable_value = value & route->move_masks[step];
        value = (value ^ movable_value) | (movable_value >> (1u << step));
    }
    return value;
}

/** @brief  `pdep` via a precomputed @p route: scatter the low bits of @p value into the selector positions. */
SZ_HELPER_AUTO sz_u64_t sz_grapheme_bit_scatter_(sz_u64_t value, sz_grapheme_bit_route_t const *route) {
    for (int step = 5; step >= 0; --step) {
        sz_u64_t const movable_value = value << (1u << step);
        value = (value & ~route->move_masks[step]) | (movable_value & route->move_masks[step]);
    }
    return value & route->selector;
}

/** @brief  Drop-in branchless `_pext_u64` (builds a one-shot route). Prefer `sz_grapheme_bit_gather_` when one
 *          selector serves several values, as inside `sz_grapheme_build_masks_neon_`. Bit-exact with BMI2. */
SZ_HELPER_AUTO sz_u64_t sz_grapheme_pext_neon_(sz_u64_t value, sz_u64_t selector) {
    sz_grapheme_bit_route_t const route = sz_grapheme_bit_route_build_(selector);
    return sz_grapheme_bit_gather_(value, &route);
}

/** @brief  Drop-in branchless `_pdep_u64`. Bit-exact with BMI2. */
SZ_HELPER_AUTO sz_u64_t sz_grapheme_pdep_neon_(sz_u64_t value, sz_u64_t selector) {
    sz_grapheme_bit_route_t const route = sz_grapheme_bit_route_build_(selector);
    return sz_grapheme_bit_scatter_(value, &route);
}

#pragma endregion Branchless bit gather and scatter

#pragma region Gather free Grapheme_Cluster_Break classifier

/** @brief  Packed descriptor byte for sixteen BMP codepoints (per-lane high = cp>>8, low = cp&0xFF) via a
 *          register-resident 3-stage `vqtbl` nibble cascade, the NEON twin of the AVX2 `vpshufb` full-BMP trie. The
 *          cascade emits the descriptor directly (the serial `id_to_desc` permute is folded into the leaf tables).
 *          Gather-free; bit-exact with `sz_rune_grapheme_break_property` over the whole BMP (Hangul included). Addresses
 *          ONE quarter; the caller iterates the four quarters. */
SZ_HELPER_AUTO uint8x16_t sz_grapheme_bmp_descriptor_neon_(uint8x16_t high, uint8x16_t low) {
    // The 3-stage BMP trie is a pure function of (page, low) - `stage2`/`stage3` only ever see `page = stage1[high]`
    // and the two `low` nibbles - so the composition is materialized as ONE flat generated table (page_count x 256,
    // bit-exact by construction): one vectorized `lut256` for the page, one bounded L1 gather for the descriptor,
    // replacing three chained gathers. See the flat table's generator note in tables.h.
    uint8x16_t const page = sz_utf8_rune_lut256_neon_(sz_utf8_grapheme_break_haswell_stage1_, high);
    return sz_utf8_rune_leaf_gather_neon_(sz_utf8_grapheme_break_flat_bmp_,
                                          (int)sz_utf8_grapheme_break_haswell_stage2_lo_count_k / 16, page, low);
}

/** @brief  Packed descriptor byte for sixteen ASTRAL codepoints over offset = cp - 0x10000 (5-nibble cascade), the
 *          NEON twin of the AVX2 astral trie. Per-lane bytes: @p plane = (offset>>16)&0xFF (low nibble meaningful),
 *          @p high = (offset>>8)&0xFF, @p low = offset&0xFF. Gather-free; bit-exact. Addresses ONE quarter. */
SZ_HELPER_AUTO uint8x16_t sz_grapheme_astral_descriptor_neon_(uint8x16_t plane, uint8x16_t high, uint8x16_t low) {
    uint8x16_t const low_nibble_mask = vdupq_n_u8(0x0F);
    uint8x16_t const n4 = vandq_u8(plane, low_nibble_mask);
    uint8x16_t const n3 = vandq_u8(vshrq_n_u8(high, 4), low_nibble_mask);
    uint8x16_t const stage1_index = vorrq_u8(vshlq_n_u8(n4, 4), n3);
    uint8x16_t const page = sz_utf8_rune_lut256_neon_(sz_utf8_grapheme_break_haswell_astral_stage1_, stage1_index);
    uint8x16_t const n2 = vandq_u8(high, low_nibble_mask);
    // leaf2 ids fit in a byte (n_leaf2 <= 255), so only the lo plane carries information; the stage3 cascade selects
    // on `leaf2` directly with a tile_count covering every leaf2 id (the hi plane is all-zero and unused).
    uint8x16_t const leaf2 = sz_utf8_rune_cascade_stage_neon_(
        sz_utf8_grapheme_break_haswell_astral_stage2_lo_, sz_utf8_grapheme_break_haswell_astral_stage2_lo_count_k / 16,
        page, n2);
    uint8x16_t const n1 = vandq_u8(vshrq_n_u8(low, 4), low_nibble_mask);
    uint8x16_t const leaf_lo = sz_utf8_rune_cascade_stage_neon_(
        sz_utf8_grapheme_break_haswell_astral_stage3_lo_, sz_utf8_grapheme_break_haswell_astral_stage3_lo_count_k / 16,
        leaf2, n1);
    uint8x16_t const leaf_hi = sz_utf8_rune_cascade_stage_neon_(
        sz_utf8_grapheme_break_haswell_astral_stage3_hi_, sz_utf8_grapheme_break_haswell_astral_stage3_hi_count_k / 16,
        leaf2, n1);
    uint8x16_t const n0 = vandq_u8(low, low_nibble_mask);
    uint8x16_t const leaf_group = vorrq_u8(vandq_u8(vshrq_n_u8(leaf_lo, 4), low_nibble_mask), vshlq_n_u8(leaf_hi, 4));
    uint8x16_t const leaf_low_nibble = vandq_u8(leaf_lo, low_nibble_mask);
    uint8x16_t const stage4_lut_index = vorrq_u8(vshlq_n_u8(leaf_low_nibble, 4), n0);
    return sz_utf8_rune_leaf_gather_neon_(sz_utf8_grapheme_break_haswell_astral_stage4_groups_,
                                          (int)sz_utf8_grapheme_break_haswell_astral_leaf_groups_k, leaf_group,
                                          stage4_lut_index);
}

/** @brief  Per-quarter unsigned `value >= bound` boolean mask (0x00/0xFF lanes), the NEON `vcgeq_u8` twin of the AVX2
 *          `max_epu8(value,bound)==value` idiom. */
SZ_HELPER_INLINE uint8x16_t sz_grapheme_cmpge_epu8_neon_(uint8x16_t value, uint8x16_t bound) {
    return vcgeq_u8(value, bound);
}

/** @brief  Build a per-quarter byte-boolean selector (0x00/0xFF) from the 16 lane bits of @p bits at offset @p shift,
 *          the NEON twin of `sz_utf8_byte_mask_from_bits_haswell_` confined to one quarter. */
SZ_HELPER_INLINE uint8x16_t sz_grapheme_byte_mask_from_bits_neon_(sz_u64_t bits, int shift) {
    static sz_u8_t const bit_position_lanes[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    static sz_u8_t const lane_half[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    sz_u8_t const low_byte = (sz_u8_t)((bits >> shift) & 0xFF);
    sz_u8_t const high_byte = (sz_u8_t)((bits >> (shift + 8)) & 0xFF);
    uint8x16_t const byte = vbslq_u8(vld1q_u8(lane_half), vdupq_n_u8(high_byte), vdupq_n_u8(low_byte));
    uint8x16_t const position = vld1q_u8(bit_position_lanes);
    return vceqq_u8(vandq_u8(byte, position), position);
}

/** @brief  Per-window per-lane descriptors as four `uint8x16_t` quarters, plus the codepoint-start lane mask and
 *          geometry. The NEON twin of the AVX2 `sz_grapheme_classified_haswell_t` outputs. */
typedef struct sz_grapheme_classified_neon_t {
    uint8x16_t descriptors[4]; /**< Packed descriptor per byte-lane (valid only on start lanes). */
    sz_u64_t start_lanes;      /**< Codepoint-start lanes within the effective span (trimmed to `byte_span`). */
    sz_size_t codepoint_count; /**< Number of codepoint starts resolved (<= 64). */
    sz_size_t byte_span;       /**< Bytes consumed by the resolved codepoints (offset of the next start). */
} sz_grapheme_classified_neon_t;

/**
 *  @brief  Decode a 64-byte window, classify every lane to a packed descriptor gather-free, and emit the per-lane
 *          descriptor quarters with the trimmed codepoint-start geometry. Mirrors the haswell decode/classify path
 *          (value-based blind reconstruction so malformed input agrees byte-for-byte) without table gather.
 */
SZ_HELPER_AUTO sz_grapheme_classified_neon_t sz_grapheme_classify_window_neon_( //
    sz_u8_t const *text, sz_size_t length, sz_size_t base) {

    sz_utf8_rune_window_neon_t const decoded = sz_utf8_rune_decode_window_neon_(text + base, length - base);
    sz_size_t const loaded = decoded.loaded;
    sz_u64_t const loaded_mask = sz_u64_mask_until_serial_(loaded);
    sz_u64_t start_lanes = decoded.codepoint_starts;

    // Effective-window trim: a multi-byte lead near the 64-byte edge whose span runs past `loaded` would decode against
    // a wrapped neighbour; defer it to the next window. Branchless union of overrunning starts, take the lowest.
    sz_size_t byte_span = loaded;
    if (loaded >= 64) {
        sz_u64_t const two = decoded.two_byte_starts;
        sz_u64_t const three = decoded.three_byte_starts;
        sz_u64_t const four = decoded.four_byte_starts;
        sz_u64_t const overrun = (two & ~sz_u64_mask_until_serial_(loaded - 1)) |
                                 (three & ~sz_u64_mask_until_serial_(loaded - 2)) |
                                 (four & ~sz_u64_mask_until_serial_(loaded - 3));
        byte_span = overrun ? (sz_size_t)sz_u64_ctz(overrun) : loaded;
        start_lanes &= sz_u64_mask_until_serial_(byte_span);
    }

    uint8x16_t raw[4];
    for (int quarter = 0; quarter < 4; ++quarter) raw[quarter] = decoded.window[quarter];
    uint8x16_t next1[4], next2[4], next3[4];
    sz_utf8_forward_neighbours_neon_(raw, next1, next2, next3);

    // Zero each `next k` lane whose source byte index reaches `loaded` (short final window only), matching the haswell
    // maskz neighbour fetch / serial blind decode that pads out-of-window continuation bytes with 0. On a FULL window
    // (`loaded == 64`) `loaded_mask` is all-ones so this is inert and the only touched lanes (truncated edge leads) are
    // already deferred by the effective-window trim — so skip the neighbour mask builds entirely on the hot path.
    if (loaded < 64) {
        for (int quarter = 0; quarter < 4; ++quarter) {
            next1[quarter] = vandq_u8(next1[quarter],
                                      sz_grapheme_byte_mask_from_bits_neon_(loaded_mask >> 1, quarter * 16));
            next2[quarter] = vandq_u8(next2[quarter],
                                      sz_grapheme_byte_mask_from_bits_neon_(loaded_mask >> 2, quarter * 16));
            next3[quarter] = vandq_u8(next3[quarter],
                                      sz_grapheme_byte_mask_from_bits_neon_(loaded_mask >> 3, quarter * 16));
        }
    }

    // Reconstruct per-lane high/low (BMP) and plane/mid/low (astral) BLINDLY from the raw lead + zeroed neighbours,
    // mirroring `sz_grapheme_classify_window_haswell_` byte-for-byte. ASCII lanes take identity (high=0, low=raw).
    uint8x16_t const c03 = vdupq_n_u8(0x03), c07 = vdupq_n_u8(0x07), c0f = vdupq_n_u8(0x0F), c1f = vdupq_n_u8(0x1F),
                     c3f = vdupq_n_u8(0x3F), c80 = vdupq_n_u8(0x80);
    sz_u64_t const three_byte = decoded.three_byte_starts;
    sz_u64_t const four_byte = decoded.four_byte_starts;

    uint8x16_t high[4], low[4], plane[4], mid[4], alo[4], ascii_desc[4], desc[4];
    uint8x16_t ascii_bool[4];
    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const here = raw[quarter];
        uint8x16_t const n1 = next1[quarter], n2 = next2[quarter], n3 = next3[quarter];

        // 2-byte: high = ((lead & 0x1F) >> 2) & 0x07; low = ((lead & 0x03) << 6) | (next1 & 0x3F).
        uint8x16_t const two_high = sz_utf8_srl8_neon_(vandq_u8(here, c1f), 2, 0x07);
        uint8x16_t const two_low = vorrq_u8(vshlq_n_u8(vandq_u8(here, c03), 6), vandq_u8(n1, c3f));
        // 3-byte: high = ((lead & 0x0F) << 4) | ((next1 >> 2) & 0x0F); low = ((next1 & 0x03) << 6) | (next2 & 0x3F).
        uint8x16_t const three_high = vorrq_u8(vshlq_n_u8(vandq_u8(here, c0f), 4), sz_utf8_srl8_neon_(n1, 2, 0x0F));
        uint8x16_t const three_low = vorrq_u8(vshlq_n_u8(vandq_u8(n1, c03), 6), vandq_u8(n2, c3f));

        uint8x16_t const three_select = sz_grapheme_byte_mask_from_bits_neon_(three_byte, quarter * 16);
        uint8x16_t const ascii_select = vceqq_u8(vandq_u8(here, c80), vdupq_n_u8(0));
        ascii_bool[quarter] = ascii_select;
        uint8x16_t const non_ascii_high = vbslq_u8(three_select, three_high, two_high);
        uint8x16_t const non_ascii_low = vbslq_u8(three_select, three_low, two_low);
        // high = ascii ? 0 : non_ascii_high; low = ascii ? raw : non_ascii_low.
        high[quarter] = vbicq_u8(non_ascii_high, ascii_select);
        low[quarter] = vbslq_u8(ascii_select, here, non_ascii_low);

        // Astral plane/mid/low (offset domain): plane = ((b0&7)<<2)|((next1>>4)&3);
        // mid = ((next1&F)<<4)|((next2>>2)&F); lo = ((next2&3)<<6)|(next3&3F).
        plane[quarter] = vorrq_u8(vshlq_n_u8(vandq_u8(here, c07), 2), vandq_u8(vshrq_n_u8(n1, 4), c03));
        mid[quarter] = vorrq_u8(vshlq_n_u8(vandq_u8(n1, c0f), 4), sz_utf8_srl8_neon_(n2, 2, 0x0F));
        alo[quarter] = vorrq_u8(vshlq_n_u8(vandq_u8(n2, c03), 6), vandq_u8(n3, c3f));

        // A 4-byte lead's blind codepoint is `(plane<<16)|(mid<<8)|alo`, NOT the 2-byte fold seated in high/low above.
        // Override its BMP high/low to mid/alo so a plane-0 (overlong) 4-byte lead resolves on the BMP path exactly as
        // serial/haswell do — those carry the full 21-bit value and split BMP-vs-astral on `cp < 0x10000`, so a plane-0
        // 4-byte lane (e.g. `F0 87 ..`) is a BMP codepoint, not astral. (The astral overwrite below is then gated on a
        // non-zero in-range plane, mirroring haswell's `is_astral`.)
        uint8x16_t const four_sel_blend = sz_grapheme_byte_mask_from_bits_neon_(four_byte, quarter * 16);
        high[quarter] = vbslq_u8(four_sel_blend, mid[quarter], high[quarter]);
        low[quarter] = vbslq_u8(four_sel_blend, alo[quarter], low[quarter]);

        // ASCII descriptor (cp < 0x80) via a single 256-LUT over the raw byte — the cheap gated fast path so a
        // pure-ASCII window never pays the full BMP nibble cascade. Mirrors haswell's `ascii_desc` LUT.
        ascii_desc[quarter] = sz_utf8_rune_lut256_neon_(sz_utf8_grapheme_break_haswell_ascii_desc_, here);
        desc[quarter] = ascii_desc[quarter];
    }

    sz_u64_t const ascii = sz_utf8_mask_combine_neon_(ascii_bool[0], ascii_bool[1], ascii_bool[2], ascii_bool[3]);

    // The flat gather IS the classifier - no arithmetic script gates. A quarter runs it only when it holds a
    // non-ASCII codepoint-START lane (descriptors are read on start lanes only); the flat table returns the exact
    // descriptor for every BMP codepoint, so pure-ASCII quarters keep the cheap `ascii_desc` LUT untouched. Measured
    // on M5: this beats every gated variant (CJK/euro/Hangul range tests cost more than the 16 L1 loads they skip).
    sz_u64_t const non_ascii_starts = loaded_mask & ~ascii & start_lanes;
    if (non_ascii_starts) {
        for (int quarter = 0; quarter < 4; ++quarter) {
            if (((non_ascii_starts >> (quarter * 16)) & 0xFFFFull) == 0) continue;
            uint8x16_t const bmp = sz_grapheme_bmp_descriptor_neon_(high[quarter], low[quarter]);
            desc[quarter] = vbslq_u8(ascii_bool[quarter], ascii_desc[quarter], bmp);
        }
    }

    // A 4-byte lead splits three ways on its blind plane = bits[16..20] of cp, matching haswell/serial value dispatch
    // (`is_bmp = cp < 0x10000`, `is_astral = 0x10000 ≤ cp < 0x110000`, else Other):
    //   plane == 0      → BMP codepoint (cp = (mid<<8)|alo); already resolved through the BMP path above, since the
    //                      high/low override seated `mid`/`alo` for these lanes (e.g. overlong `F0 87 ..`).
    //   plane in [1,16] → genuine astral (cp in 0x10000..0x10FFFF); overwrite with the 4-byte trie descriptor.
    //   plane ≥ 17      → cp ≥ 0x110000 (e.g. overlong `F4 90 80 80`); neither BMP nor astral, force Other (0).
    // `plane[q]` carries the 5-bit plane per 4-byte-lead lane (junk on other lanes, gated out by `four_byte`).
    uint8x16_t const plane_one = vdupq_n_u8(1), plane_sixteen = vdupq_n_u8(0x10);
    uint8x16_t plane_nonzero_q[4], plane_le16_q[4];
    for (int quarter = 0; quarter < 4; ++quarter) {
        plane_nonzero_q[quarter] = vcgeq_u8(plane[quarter], plane_one);
        plane_le16_q[quarter] = sz_grapheme_cmpge_epu8_neon_(plane_sixteen, plane[quarter]);
    }
    sz_u64_t const plane_nonzero = sz_utf8_mask_combine_neon_(plane_nonzero_q[0], plane_nonzero_q[1],
                                                              plane_nonzero_q[2], plane_nonzero_q[3]);
    sz_u64_t const plane_le_16 = sz_utf8_mask_combine_neon_(plane_le16_q[0], plane_le16_q[1], plane_le16_q[2],
                                                            plane_le16_q[3]);
    sz_u64_t const is_astral = four_byte & plane_nonzero & plane_le_16 & loaded_mask;
    sz_u64_t const is_overrange = four_byte & plane_nonzero & ~plane_le_16 & loaded_mask;
    if (is_astral) {
        // offset = cp - 0x10000; high16/low16 unchanged, plane nibble = cp_plane - 1.
        for (int quarter = 0; quarter < 4; ++quarter) {
            uint8x16_t const four_sel = sz_grapheme_byte_mask_from_bits_neon_(is_astral, quarter * 16);
            uint8x16_t const plane_off = vsubq_u8(vandq_u8(four_sel, plane[quarter]), vdupq_n_u8(1));
            uint8x16_t const astral = sz_grapheme_astral_descriptor_neon_(plane_off, mid[quarter], alo[quarter]);
            desc[quarter] = vbslq_u8(four_sel, astral, desc[quarter]);
        }
    }
    if (is_overrange) {
        // cp ≥ 0x110000: clear to Other (0); the BMP-path value seated on these lanes is meaningless.
        for (int quarter = 0; quarter < 4; ++quarter)
            desc[quarter] = vbicq_u8(desc[quarter], sz_grapheme_byte_mask_from_bits_neon_(is_overrange, quarter * 16));
    }

    // `0xF8..0xFF` begin no valid sequence and match no lead-length mask; force their start lanes to the Other
    // descriptor (0) so they classify neighbour-independently, matching serial/haswell (§6.2 U+FFFD -> Other).
    uint8x16_t invalid_bool[4];
    for (int quarter = 0; quarter < 4; ++quarter) invalid_bool[quarter] = vcgeq_u8(raw[quarter], vdupq_n_u8(0xF8));
    sz_u64_t const invalid_lead = sz_utf8_mask_combine_neon_(invalid_bool[0], invalid_bool[1], invalid_bool[2],
                                                             invalid_bool[3]);
    for (int quarter = 0; quarter < 4; ++quarter)
        desc[quarter] = vbicq_u8(desc[quarter], sz_grapheme_byte_mask_from_bits_neon_(invalid_lead, quarter * 16));

    sz_grapheme_classified_neon_t result;
    for (int quarter = 0; quarter < 4; ++quarter) result.descriptors[quarter] = desc[quarter];
    result.start_lanes = start_lanes;
    result.codepoint_count = (sz_size_t)sz_u64_popcount(start_lanes);
    result.byte_span = byte_span;
    return result;
}

#pragma endregion Gather free Grapheme_Cluster_Break classifier

#pragma region Boundary algebra extractor

/**
 *  @brief  Build the codepoint-dense per-class window masks from the per-lane descriptor quarters at the codepoint-start
 *          lanes — the NEON twin of `sz_grapheme_build_masks_haswell_` feeding the SAME portable engine. Each per-class
 *          membership mask is built in the BYTE-lane domain (`vceqq_u8` quarters -> `mask_combine`), then compacted to
 *          the codepoint-dense domain by a single software-`pext` over the start lanes (the BMI2-free analogue of the
 *          haswell `_pext_u64`). No scalar per-lane loop, no table gather, no rule control flow.
 */
SZ_HELPER_INLINE sz_grapheme_window_masks_t sz_grapheme_build_masks_neon_(sz_grapheme_classified_neon_t classified,
                                                                          sz_u64_t valid) {
    sz_u64_t const starts = classified.start_lanes;
    // Every class compaction below gathers the byte-lane mask down to the codepoint-dense domain over the SAME
    // `starts` selector, so build the bit-route once and reuse it for all 18 gathers (the branchless PEXT).
    sz_grapheme_bit_route_t const start_route = sz_grapheme_bit_route_build_(starts);
    uint8x16_t const nibble_mask = vdupq_n_u8(0x0F);
    uint8x16_t class_q[4], desc_q[4];
    for (int quarter = 0; quarter < 4; ++quarter) {
        desc_q[quarter] = classified.descriptors[quarter];
        class_q[quarter] = vandq_u8(desc_q[quarter], nibble_mask);
    }

    sz_grapheme_window_masks_t masks;

    // Trivial-window fast path (the CJK/Other analogue of the driver's ASCII gate). When every start-lane descriptor is
    // a simple break class - Other/CR/LF/Control, i.e. `desc <= 3`: no Extend/ZWJ/Regional_Indicator/Prepend/
    // SpacingMark/Hangul {L,V,T,LV,LVT} in the low nibble, no Extended_Pictographic (bit 0x40), no Indic (bits [5:4]) -
    // only CR/LF/Control drive the boundary (GB3/GB4/GB5) and the other 15 class masks are provably all-zero. Build just
    // those three, skipping ~15 of 18 software-`pext` gathers. Bit-exact. This dominates Han/Kana text and any diacritic-
    // free 2-byte script (Cyrillic/Greek/Arabic/Hebrew base letters are all GCB=Other), where `build_masks` is the top
    // cost once the cascade is gated away. The one extra `vcgtq`+combine per window is <1/18 of the full extraction.
    uint8x16_t const three_vec = vdupq_n_u8(3);
    sz_u64_t const nonsimple = sz_utf8_mask_combine_neon_(
                                   vcgtq_u8(desc_q[0], three_vec), vcgtq_u8(desc_q[1], three_vec),
                                   vcgtq_u8(desc_q[2], three_vec), vcgtq_u8(desc_q[3], three_vec)) &
                               starts;
    if (nonsimple == 0) {
        for (int class_index = 0; class_index < 14; ++class_index) masks.class_bit[class_index] = 0;
        masks.extended_pictographic = 0;
        masks.indic_consonant = masks.indic_extend = masks.indic_linker = 0;
        for (int class_index = (int)sz_grapheme_break_cr_k; class_index <= (int)sz_grapheme_break_control_k;
             ++class_index) {
            uint8x16_t const class_id_vec = vdupq_n_u8((sz_u8_t)class_index);
            sz_u64_t const byte_mask = sz_utf8_mask_combine_neon_(
                vceqq_u8(class_q[0], class_id_vec), vceqq_u8(class_q[1], class_id_vec),
                vceqq_u8(class_q[2], class_id_vec), vceqq_u8(class_q[3], class_id_vec));
            masks.class_bit[class_index] = sz_grapheme_bit_gather_(byte_mask, &start_route) & valid;
        }
        return masks;
    }

    for (int class_index = 0; class_index < 14; ++class_index) {
        uint8x16_t const class_id_vec = vdupq_n_u8((sz_u8_t)class_index);
        sz_u64_t const byte_mask = sz_utf8_mask_combine_neon_(
            vceqq_u8(class_q[0], class_id_vec), vceqq_u8(class_q[1], class_id_vec), vceqq_u8(class_q[2], class_id_vec),
            vceqq_u8(class_q[3], class_id_vec));
        masks.class_bit[class_index] = sz_grapheme_bit_gather_(byte_mask, &start_route) & valid;
    }
    uint8x16_t const ext_bit = vdupq_n_u8(0x40);
    uint8x16_t ext_q[4];
    for (int quarter = 0; quarter < 4; ++quarter)
        ext_q[quarter] = vceqq_u8(vandq_u8(desc_q[quarter], ext_bit), ext_bit);
    sz_u64_t const ext_byte = sz_utf8_mask_combine_neon_(ext_q[0], ext_q[1], ext_q[2], ext_q[3]);
    masks.extended_pictographic = sz_grapheme_bit_gather_(ext_byte, &start_route) & valid;

    uint8x16_t const incb_consonant = vdupq_n_u8((sz_u8_t)sz_grapheme_incb_consonant_k);
    uint8x16_t const incb_extend = vdupq_n_u8((sz_u8_t)sz_grapheme_incb_extend_k);
    uint8x16_t const incb_linker = vdupq_n_u8((sz_u8_t)sz_grapheme_incb_linker_k);
    uint8x16_t consonant_q[4], extend_q[4], linker_q[4];
    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const incb = vandq_u8(vshrq_n_u8(desc_q[quarter], 4), vdupq_n_u8(0x03));
        consonant_q[quarter] = vceqq_u8(incb, incb_consonant);
        extend_q[quarter] = vceqq_u8(incb, incb_extend);
        linker_q[quarter] = vceqq_u8(incb, incb_linker);
    }
    masks.indic_consonant = sz_grapheme_bit_gather_(sz_utf8_mask_combine_neon_(consonant_q[0], consonant_q[1],
                                                                               consonant_q[2], consonant_q[3]),
                                                    &start_route) &
                            valid;
    masks.indic_extend = sz_grapheme_bit_gather_(
                             sz_utf8_mask_combine_neon_(extend_q[0], extend_q[1], extend_q[2], extend_q[3]),
                             &start_route) &
                         valid;
    masks.indic_linker = sz_grapheme_bit_gather_(
                             sz_utf8_mask_combine_neon_(linker_q[0], linker_q[1], linker_q[2], linker_q[3]),
                             &start_route) &
                         valid;
    return masks;
}

#pragma endregion Boundary algebra extractor

#pragma region Grapheme forward driver

SZ_API_COMPTIME sz_size_t sz_utf8_graphemes_neon(          //
    sz_cptr_t text, sz_size_t length,                      //
    sz_size_t *cluster_starts, sz_size_t *cluster_lengths, //
    sz_size_t clusters_capacity, sz_size_t *bytes_consumed) {

    sz_size_t clusters = 0;
    if (length == 0 || clusters_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_grapheme_carry_t carry = sz_grapheme_carry_empty_();
    sz_size_t cluster_start = 0;
    sz_size_t base = 0;

    while (base < length) {
        // ASCII whole-window fast path (x86-style tiering): a window with no byte >= 0x80 decodes 1:1 to codepoints,
        // and only CR/LF/Control affect grapheme boundaries (GB3, GB4/GB5) - every other ASCII byte is `Other`
        // (GB999 break). Build just those three class masks and run the SAME rule engine (bit-exact, carry handled),
        // skipping the UTF-8 decode and the 18-class `build_masks` that dominate Latin text (measured ~+49% on M5).
        sz_size_t const ascii_bytes = length - base < 64 ? length - base : 64;
        uint8x16_t ascii_q[4];
        if (length - base >= 64) {
            ascii_q[0] = vld1q_u8(text_u8 + base), ascii_q[1] = vld1q_u8(text_u8 + base + 16);
            ascii_q[2] = vld1q_u8(text_u8 + base + 32), ascii_q[3] = vld1q_u8(text_u8 + base + 48);
        }
        else {
            sz_align_(16) sz_u8_t padded[64];
            for (sz_size_t byte_index = 0; byte_index < 64; ++byte_index)
                padded[byte_index] = byte_index < ascii_bytes ? text_u8[base + byte_index] : 0;
            ascii_q[0] = vld1q_u8(padded), ascii_q[1] = vld1q_u8(padded + 16);
            ascii_q[2] = vld1q_u8(padded + 32), ascii_q[3] = vld1q_u8(padded + 48);
        }
        uint8x16_t const ascii_any = vorrq_u8(vorrq_u8(ascii_q[0], ascii_q[1]), vorrq_u8(ascii_q[2], ascii_q[3]));
        if ((vmaxvq_u8(ascii_any) & 0x80u) == 0) {
            int const codepoint_count = (int)ascii_bytes;
            sz_u64_t const valid = codepoint_count >= 64 ? ~(sz_u64_t)0 : (((sz_u64_t)1 << codepoint_count) - 1);
            uint8x16_t const cr_vec = vdupq_n_u8(0x0D), lf_vec = vdupq_n_u8(0x0A);
            uint8x16_t const space_vec = vdupq_n_u8(0x20), del_vec = vdupq_n_u8(0x7F);
            uint8x16_t cr_q[4], lf_q[4], control_q[4];
            for (int quarter = 0; quarter < 4; ++quarter) {
                cr_q[quarter] = vceqq_u8(ascii_q[quarter], cr_vec);
                lf_q[quarter] = vceqq_u8(ascii_q[quarter], lf_vec);
                uint8x16_t const c0_or_del = vorrq_u8(vcltq_u8(ascii_q[quarter], space_vec),
                                                      vceqq_u8(ascii_q[quarter], del_vec));
                control_q[quarter] = vbicq_u8(vbicq_u8(c0_or_del, cr_q[quarter]), lf_q[quarter]);
            }
            sz_grapheme_window_masks_t ascii_masks;
            for (int class_index = 0; class_index < 14; ++class_index) ascii_masks.class_bit[class_index] = 0;
            ascii_masks.extended_pictographic = 0;
            ascii_masks.indic_consonant = ascii_masks.indic_extend = ascii_masks.indic_linker = 0;
            ascii_masks.class_bit[sz_grapheme_break_cr_k] =
                sz_utf8_mask_combine_neon_(cr_q[0], cr_q[1], cr_q[2], cr_q[3]) & valid;
            ascii_masks.class_bit[sz_grapheme_break_lf_k] =
                sz_utf8_mask_combine_neon_(lf_q[0], lf_q[1], lf_q[2], lf_q[3]) & valid;
            ascii_masks.class_bit[sz_grapheme_break_control_k] =
                sz_utf8_mask_combine_neon_(control_q[0], control_q[1], control_q[2], control_q[3]) & valid;
            // ASCII: every byte is a codepoint-start, so the byte-domain boundary equals the dense boundary (pdep is
            // the identity) and no software-pext scatter is needed.
            sz_u64_t boundary = sz_grapheme_window_boundaries_(&ascii_masks, codepoint_count, valid, &carry);
            if (base == 0) boundary &= ~(sz_u64_t)1;
            clusters = sz_utf8_rune_drain_forward_neon_(boundary, base, cluster_starts, cluster_lengths, clusters,
                                                        clusters_capacity, &cluster_start);
            if (clusters == clusters_capacity) {
                if (bytes_consumed) *bytes_consumed = cluster_start;
                return clusters;
            }
            base += ascii_bytes;
            continue;
        }

        sz_grapheme_classified_neon_t const window = sz_grapheme_classify_window_neon_(text_u8, length, base);
        if (window.codepoint_count == 0) break; // defensive: cannot happen for valid input

        int const codepoint_count = (int)window.codepoint_count;
        sz_u64_t const valid = (codepoint_count >= 64) ? ~0ull : ((1ull << codepoint_count) - 1);
        sz_grapheme_window_masks_t const masks = sz_grapheme_build_masks_neon_(window, valid);
        sz_u64_t const dense_boundary = sz_grapheme_window_boundaries_(&masks, codepoint_count, valid, &carry);
        // Scatter dense boundary bits back to codepoint-start byte lanes (software `pdep` deposits the j-th dense bit
        // into the j-th set start lane), recovering the byte-domain boundary mask the drain consumes.
        sz_u64_t boundary = sz_grapheme_pdep_neon_(dense_boundary, window.start_lanes);

        // GB1 anchor at byte 0 of the first window is the open cluster's own start, not a new break: clear it.
        if (base == 0) boundary &= ~1ull;
        clusters = sz_utf8_rune_drain_forward_neon_(boundary, base, cluster_starts, cluster_lengths, clusters,
                                                    clusters_capacity, &cluster_start);
        if (clusters == clusters_capacity) {
            if (bytes_consumed) *bytes_consumed = cluster_start;
            return clusters;
        }
        base += window.byte_span;
    }

    cluster_starts[clusters] = cluster_start;
    cluster_lengths[clusters] = length - cluster_start;
    ++clusters;
    if (bytes_consumed) *bytes_consumed = length;
    return clusters;
}

#pragma endregion Grapheme forward driver

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_GRAPHEMES_NEON_H_
