/**
 *  @brief WebAssembly SIMD128 backend for UTF-8 codepoint mechanics.
 *  @file include/stringzilla/utf8_runes/v128.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_RUNES_V128_H_
#define STRINGZILLA_UTF8_RUNES_V128_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

SZ_API_COMPTIME sz_size_t sz_utf8_count_v128(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    // A continuation byte satisfies `(byte & 0xC0) == 0x80`, i.e. it lies in `[0x80, 0xBF]`, which as a
    // SIGNED int8 is `[-128, -65]` — exactly the values strictly less than `0xC0` (= -64). So a single
    // signed compare `byte < 0xC0` flags continuation bytes, replacing the `and` + `eq` pair.
    v128_t continuation_threshold_vec = wasm_i8x16_splat((sz_i8_t)0xC0);

    // The UTF-8 code-point count equals (#bytes - #continuation bytes). Instead of a per-block
    // `bitmask` + `popcount` horizontal reduction, we keep a per-lane counter in a vector register:
    // the compare yields 0xFF (= -1) for each continuation lane, so SUBTRACTING it adds +1 per match.
    // A u8 lane can hold up to 255 matches, so we flush into a wider u64 accumulator every 255 blocks
    // and reduce ONCE at the end.
    sz_size_t total_bytes = (length / 16) * 16;
    sz_u128_vec_t cont8_vec, cont64_vec;
    cont8_vec.v128 = wasm_u64x2_splat(0);  // 16x u8 per-lane match counters
    cont64_vec.v128 = wasm_u64x2_splat(0); // 2x u64 accumulated counters

    while (length >= 16) {
        sz_size_t blocks = length / 16;
        if (blocks > 255) blocks = 255;
        length -= blocks * 16;
        for (sz_size_t i = 0; i < blocks; ++i, text_u8 += 16) {
            v128_t text_vec = wasm_v128_load(text_u8);
            v128_t continuation_vec = wasm_i8x16_lt(text_vec, continuation_threshold_vec);
            // `continuation_vec` lane is 0xFF (-1) on a match; subtract to increment the counter.
            cont8_vec.v128 = wasm_i8x16_sub(cont8_vec.v128, continuation_vec);
        }
        // Flush: sum the 16x u8 counters into 2x u64 lanes via two pairwise widenings.
        v128_t cont16 = wasm_u16x8_extadd_pairwise_u8x16(cont8_vec.v128);
        v128_t cont32 = wasm_u32x4_extadd_pairwise_u16x8(cont16);
        cont64_vec.v128 = wasm_i64x2_add( //
            cont64_vec.v128, wasm_i64x2_add(wasm_u64x2_extend_low_u32x4(cont32), wasm_u64x2_extend_high_u32x4(cont32)));
        cont8_vec.v128 = wasm_u64x2_splat(0);
    }

    sz_size_t continuation_count = (sz_size_t)(cont64_vec.u64s[0] + cont64_vec.u64s[1]);
    sz_size_t char_count = total_bytes - continuation_count;
    if (length) char_count += sz_utf8_count_serial((sz_cptr_t)text_u8, length);
    return char_count;
}

/** @brief  Locate the @p n-th code-point start (a non-continuation byte) via a per-tile count + nth-set-bit. */
SZ_API_COMPTIME sz_cptr_t sz_utf8_seek_v128(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    // A continuation byte is exactly a value `< 0xC0` as a signed int8 (see `sz_utf8_count_v128`), so a single
    // signed compare flags continuation lanes; the code-point starts are the complement.
    v128_t const continuation_threshold_vec = wasm_i8x16_splat((sz_i8_t)0xC0);

    while (length >= 16) {
        v128_t const window = wasm_v128_load(text_u8);
        v128_t const continuation = wasm_i8x16_lt(window, continuation_threshold_vec);
        sz_u32_t const start_bits = ~(sz_u32_t)wasm_i8x16_bitmask(continuation) & 0xFFFFu;
        sz_size_t const start_count = (sz_size_t)sz_u32_popcount(start_bits);
        if (n >= start_count) {
            n -= start_count, text_u8 += 16, length -= 16;
            continue;
        }
        return (sz_cptr_t)(text_u8 + sz_u32_nth_set_bit(start_bits, n));
    }

    return sz_utf8_seek_serial((sz_cptr_t)text_u8, length, n);
}

/*  `sz_utf8_decode_v128` (the public UTF-8 -> UTF-32 entry) is defined AFTER the shared SIMD leaf substrate below,
 *  since it drives `sz_utf8_decode_once_v128_` which reuses that substrate's window classifiers and drain leaves. */

#pragma region Shared SIMD leaf substrate

/*  Family-agnostic WebAssembly SIMD128 (v128) leaf helpers shared by the UTF-8 segmentation kernels. The v128 twin
 *  of the NEON @ref sz_utf8_rune_window_neon_t substrate: a 64-byte window held as four `v128_t` quarters, masks
 *  reduced to one-bit-per-byte `sz_u64_t` lane masks (bit `i` <=> lane `i`) via a single `wasm_i8x16_bitmask`, table
 *  reads by bounded scalar L1 walks (`wasm_i8x16_swizzle` reaches only 16 B, so no in-register `vqtbl4q` ports), and
 *  first-set extraction by the `63 - clz(mask & -mask)` idiom so no `ctz` / `popcount` builtin is ever needed. The
 *  portable rule algebra in `serial.h` consumes the `sz_u64_t` masks unchanged across every backend. */

/** @brief  The decoded 64-byte window for the v128 backend. The 64 bytes live as four `v128_t` quarters
 *          (`window[0]` = lanes [0, 16), ... `window[3]` = lanes [48, 64)); the per-lane byte-domain codepoint
 *          halves `high` / `low` share that shape. Masks are `sz_u64_t` (one bit per byte-lane: the v128 bitmask of
 *          each quarter placed at bit positions [0,16)/[16,32)/[32,48)/[48,64)). Field names and semantics match
 *          @ref sz_utf8_rune_window_neon_t so the portable rule algebra is unchanged. */
typedef struct sz_utf8_rune_window_v128_t {
    v128_t window[4];           /**< Raw input bytes for lanes [16*q, 16*q+16). */
    v128_t high[4];             /**< Per-lane `codepoint >> 8`. */
    v128_t low[4];              /**< Per-lane `codepoint & 0xFF`. */
    sz_u64_t continuation;      /**< Bit `i` => lane `i` is a continuation byte `10xxxxxx`. */
    sz_u64_t codepoint_starts;  /**< Bit `i` => lane `i` begins a codepoint (loaded, non-continuation). */
    sz_u64_t two_byte_starts;   /**< Bit `i` => lane `i` is a 2-byte lead `110xxxxx`. */
    sz_u64_t three_byte_starts; /**< Bit `i` => lane `i` is a 3-byte lead `1110xxxx`. */
    sz_u64_t four_byte_starts;  /**< Bit `i` => lane `i` is a 4-byte lead `11110xxx`. */
    sz_size_t loaded;           /**< Number of bytes actually loaded (<= 64). */
} sz_utf8_rune_window_v128_t;

/** @brief  Per-byte logical right shift by @p shift keeping the low @p keep bits — the v128 twin of `srl8_`. */
SZ_HELPER_INLINE v128_t sz_utf8_srl8_v128_(v128_t value_u8x16, int shift, sz_u8_t keep) {
    return wasm_v128_and(wasm_u8x16_shr(value_u8x16, (sz_u32_t)shift), wasm_i8x16_splat((sz_i8_t)keep));
}

/** @brief  Reduce one `v128_t` whose lanes are 0x00/0xFF booleans into a 16-bit lane mask (bit `i` <=> lane `i`) with
 *          ONE native `wasm_i8x16_bitmask` — simpler than NEON's `vaddv` reduction. */
SZ_HELPER_INLINE sz_u64_t sz_utf8_movemask16_v128_(v128_t boolean_lanes_u8x16) {
    return (sz_u64_t)((sz_u32_t)wasm_i8x16_bitmask(boolean_lanes_u8x16) & 0xFFFFu);
}

/** @brief  Combine the four per-quarter bitmasks into one 64-bit lane mask: quarter `q` -> bits [16*q, 16*q+16). The
 *          v128 twin of @ref sz_utf8_mask_combine_neon_. */
SZ_HELPER_INLINE sz_u64_t sz_utf8_mask_combine_v128_( //
    v128_t quarter0_u8x16, v128_t quarter1_u8x16, v128_t quarter2_u8x16, v128_t quarter3_u8x16) {
    sz_u64_t mask = sz_utf8_movemask16_v128_(quarter0_u8x16);
    mask |= sz_utf8_movemask16_v128_(quarter1_u8x16) << 16;
    mask |= sz_utf8_movemask16_v128_(quarter2_u8x16) << 32;
    mask |= sz_utf8_movemask16_v128_(quarter3_u8x16) << 48;
    return mask;
}

/** @brief  Masked 64-byte load into four quarters; bytes [loaded, 64) read as zero. A zero-initialized vector union
 *          stages the partial tail so we never read past `text + loaded`. Mirrors @ref sz_utf8_load_window_neon_. */
SZ_HELPER_AUTO void sz_utf8_rune_load_window_v128_(sz_u8_t const *text, sz_size_t loaded, v128_t *out) {
    if (loaded >= 64) {
        out[0] = wasm_v128_load(text + 0);
        out[1] = wasm_v128_load(text + 16);
        out[2] = wasm_v128_load(text + 32);
        out[3] = wasm_v128_load(text + 48);
        return;
    }
    sz_u512_vec_t window_vec;
    window_vec.u128s[0].v128 = window_vec.u128s[1].v128 = window_vec.u128s[2].v128 = window_vec.u128s[3].v128 =
        wasm_i8x16_splat(0);
    for (sz_size_t i = 0; i < loaded; ++i) window_vec.u8s[i] = text[i];
    out[0] = window_vec.u128s[0].v128;
    out[1] = window_vec.u128s[1].v128;
    out[2] = window_vec.u128s[2].v128;
    out[3] = window_vec.u128s[3].v128;
}

/**
 *  @brief  Forward neighbours `next1[i] = window[i+1]`, `next2[i] = window[i+2]`, `next3[i] = window[i+3]` over all 64
 *          lanes, with the lanes past the window WRAPPING modulo 64 to match Ice Lake's `_mm512_permutexvar_epi8`
 *          (so `next1[63]==window[0]`, etc.). `wasm_i8x16_shuffle` takes IMMEDIATE indices, so per quarter
 *          `here = window[q]` is concatenated with `succ = window[(q+1)&3]` (quarter 3 wraps to 0) and the shifted
 *          span extracted. The three neighbour distances are provided because the family classifiers need up to
 *          `next3` (4-byte sequences).
 */
SZ_HELPER_AUTO void sz_utf8_forward_neighbours_v128_( //
    v128_t const *window, v128_t *next1, v128_t *next2, v128_t *next3) {
    for (int quarter = 0; quarter < 4; ++quarter) {
        v128_t const here_u8x16 = window[quarter];
        v128_t const successor_u8x16 = window[(quarter + 1) & 3];
        next1[quarter] = wasm_i8x16_shuffle(here_u8x16, successor_u8x16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                            15, 16);
        next2[quarter] = wasm_i8x16_shuffle(here_u8x16, successor_u8x16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                            16, 17);
        next3[quarter] = wasm_i8x16_shuffle(here_u8x16, successor_u8x16, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                            16, 17, 18);
    }
}

/** @brief  Load up to 64 bytes (masked tail) and decode every lane into byte-domain halves — the v128 twin of
 *          @ref sz_utf8_rune_decode_window_neon_, bit-identical to it on every lane. */
SZ_HELPER_AUTO sz_utf8_rune_window_v128_t sz_utf8_rune_decode_window_v128_( //
    sz_u8_t const *text, sz_size_t available) {
    sz_utf8_rune_window_v128_t result;
    result.loaded = available < 64 ? available : 64;

    v128_t window[4];
    sz_utf8_rune_load_window_v128_(text, result.loaded, window);
    for (int quarter = 0; quarter < 4; ++quarter) result.window[quarter] = window[quarter];

    v128_t next1[4], next2[4], next3[4];
    sz_utf8_forward_neighbours_v128_(window, next1, next2, next3);

    sz_u64_t const loaded_mask = sz_u64_mask_until_serial_(result.loaded);

    // Lead-class detection over loaded lanes: (byte & mask) == pattern, AND-clamped to loaded lanes.
    v128_t continuation_bool[4], two_byte_bool[4], three_byte_bool[4], four_byte_bool[4];
    v128_t const mask_continuation_u8x16 = wasm_i8x16_splat((sz_i8_t)0xC0),
                 pattern_continuation_u8x16 = wasm_i8x16_splat((sz_i8_t)0x80);
    v128_t const mask_two_u8x16 = wasm_i8x16_splat((sz_i8_t)0xE0), pattern_two_u8x16 = wasm_i8x16_splat((sz_i8_t)0xC0);
    v128_t const mask_three_u8x16 = wasm_i8x16_splat((sz_i8_t)0xF0),
                 pattern_three_u8x16 = wasm_i8x16_splat((sz_i8_t)0xE0);
    v128_t const mask_four_u8x16 = wasm_i8x16_splat((sz_i8_t)0xF8),
                 pattern_four_u8x16 = wasm_i8x16_splat((sz_i8_t)0xF0);
    for (int quarter = 0; quarter < 4; ++quarter) {
        v128_t const here_u8x16 = window[quarter];
        continuation_bool[quarter] = wasm_i8x16_eq(wasm_v128_and(here_u8x16, mask_continuation_u8x16),
                                                   pattern_continuation_u8x16);
        two_byte_bool[quarter] = wasm_i8x16_eq(wasm_v128_and(here_u8x16, mask_two_u8x16), pattern_two_u8x16);
        three_byte_bool[quarter] = wasm_i8x16_eq(wasm_v128_and(here_u8x16, mask_three_u8x16), pattern_three_u8x16);
        four_byte_bool[quarter] = wasm_i8x16_eq(wasm_v128_and(here_u8x16, mask_four_u8x16), pattern_four_u8x16);
    }
    result.continuation = sz_utf8_mask_combine_v128_(continuation_bool[0], continuation_bool[1], continuation_bool[2],
                                                     continuation_bool[3]) &
                          loaded_mask;
    result.codepoint_starts = loaded_mask & ~result.continuation;
    result.two_byte_starts = sz_utf8_mask_combine_v128_(two_byte_bool[0], two_byte_bool[1], two_byte_bool[2],
                                                        two_byte_bool[3]) &
                             loaded_mask;
    result.three_byte_starts = sz_utf8_mask_combine_v128_(three_byte_bool[0], three_byte_bool[1], three_byte_bool[2],
                                                          three_byte_bool[3]) &
                               loaded_mask;
    result.four_byte_starts = sz_utf8_mask_combine_v128_(four_byte_bool[0], four_byte_bool[1], four_byte_bool[2],
                                                         four_byte_bool[3]) &
                              loaded_mask;

    v128_t const low_five_bits_u8x16 = wasm_i8x16_splat(0x1F);
    v128_t const low_six_bits_u8x16 = wasm_i8x16_splat(0x3F);
    v128_t const low_two_bits_u8x16 = wasm_i8x16_splat(0x03);
    v128_t const low_four_bits_u8x16 = wasm_i8x16_splat(0x0F);

    for (int quarter = 0; quarter < 4; ++quarter) {
        v128_t const here_u8x16 = window[quarter];
        v128_t const next_byte_u8x16 = next1[quarter];
        v128_t const next_byte2_u8x16 = next2[quarter];

        // 2-byte: codepoint = ((b0 & 0x1F) << 6) | (b1 & 0x3F); high = codepoint >> 8, low = codepoint & 0xFF.
        v128_t const high_two_u8x16 = sz_utf8_srl8_v128_(wasm_v128_and(here_u8x16, low_five_bits_u8x16), 2, 0x07);
        v128_t const low_two_u8x16 = wasm_v128_or(wasm_i8x16_shl(wasm_v128_and(here_u8x16, low_two_bits_u8x16), 6),
                                                  wasm_v128_and(next_byte_u8x16, low_six_bits_u8x16));

        // 3-byte: codepoint = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F).
        v128_t const high_three_u8x16 = wasm_v128_or(wasm_i8x16_shl(wasm_v128_and(here_u8x16, low_four_bits_u8x16), 4),
                                                     sz_utf8_srl8_v128_(next_byte_u8x16, 2, 0x0F));
        v128_t const low_three_u8x16 = wasm_v128_or(
            wasm_i8x16_shl(wasm_v128_and(next_byte_u8x16, low_two_bits_u8x16), 6),
            wasm_v128_and(next_byte2_u8x16, low_six_bits_u8x16));

        // Blend 2-byte vs 3-byte per lane: select the 3-byte value where this lane is a 3-byte lead.
        v128_t const three_select_u8x16 = three_byte_bool[quarter];
        result.high[quarter] = wasm_v128_bitselect(high_three_u8x16, high_two_u8x16, three_select_u8x16);
        result.low[quarter] = wasm_v128_bitselect(low_three_u8x16, low_two_u8x16, three_select_u8x16);
    }
    return result;
}

/** @brief  One nibble-cascade stage with a sub-256 selector:
 *          `result[lane] = table[selector_u8x16[lane]*16 + within_u8x16[lane]]` when `selector_u8x16 < tile_count`,
 *          else 0. Each 16-byte row (one selector value) fits `wasm_i8x16_swizzle`'s 16-B reach, so each row is loaded
 *          and shuffled by @p within_u8x16, then blended in where @p selector_u8x16 picks it; the final `wasm_u8x16_lt`
 *          clamp reproduces the all-zero result for selectors past the table. @p within_u8x16 is a nibble by
 *          construction. The v128 twin of @ref sz_utf8_rune_cascade_stage_neon_. */
SZ_HELPER_AUTO v128_t sz_utf8_rune_cascade_stage_v128_( //
    sz_u8_t const *table, int tile_count, v128_t selector_u8x16, v128_t within_u8x16) {
    v128_t result_u8x16 = wasm_i8x16_splat(0);
    for (int tile = 0; tile < tile_count; ++tile) {
        v128_t const row_u8x16 = wasm_v128_load(table + tile * 16);
        v128_t const shuffled_u8x16 = wasm_i8x16_swizzle(row_u8x16, within_u8x16);
        v128_t const here_u8x16 = wasm_i8x16_eq(selector_u8x16, wasm_i8x16_splat((sz_i8_t)tile));
        result_u8x16 = wasm_v128_bitselect(shuffled_u8x16, result_u8x16, here_u8x16);
    }
    return wasm_v128_and(result_u8x16, wasm_u8x16_lt(selector_u8x16, wasm_i8x16_splat((sz_i8_t)tile_count)));
}

/** @brief  256-entry byte LUT addressed by a per-lane byte index in `[0,256)`:
 *          `result[lane] = group_base[index_u8x16[lane]]`. `wasm_i8x16_swizzle` reaches only 16 B, so the read is a
 *          bounded scalar L1 walk (the v128 twin of the substrate `lut256` leaf); the index byte is total over the
 *          256-entry table by construction. */
SZ_HELPER_AUTO v128_t sz_utf8_rune_lut256_v128_(sz_u8_t const *group_base, v128_t index_u8x16) {
    sz_align_(16) sz_u8_t index_lanes[16], out_lanes[16];
    wasm_v128_store(index_lanes, index_u8x16);
    for (int lane = 0; lane < 16; ++lane) out_lanes[lane] = group_base[index_lanes[lane]];
    return wasm_v128_load(out_lanes);
}

/** @brief  Class byte per lane from a page-compressed flat table: `page_lut[high]` selects one 256-byte page, then
 *          `flat[page * 256 + low]` is read per lane. The page LUT resolves via the in-register scalar `lut256` walk;
 *          the leaf read is a bounded scalar L1 walk over fused 16-bit indices `(page << 8) | low`. Lanes whose page
 *          index reaches @p page_count return zero. The v128 twin of @ref sz_utf8_rune_flat_lookup_neon_. */
SZ_HELPER_AUTO v128_t sz_utf8_rune_flat_lookup_v128_( //
    sz_u8_t const *page_lut, sz_u8_t const *flat, int page_count, v128_t high_bytes_u8x16, v128_t low_bytes_u8x16) {
    v128_t const page_indices_u8x16 = sz_utf8_rune_lut256_v128_(page_lut, high_bytes_u8x16);
    v128_t const in_range_u8x16 = wasm_u8x16_lt(page_indices_u8x16, wasm_i8x16_splat((sz_i8_t)page_count));
    v128_t const page_clamped_u8x16 = wasm_v128_and(page_indices_u8x16, in_range_u8x16);
    sz_align_(16) sz_u8_t page_lanes[16], low_lanes[16], class_lanes[16];
    wasm_v128_store(page_lanes, page_clamped_u8x16);
    wasm_v128_store(low_lanes, low_bytes_u8x16);
    for (int lane = 0; lane < 16; ++lane)
        class_lanes[lane] = flat[((sz_size_t)page_lanes[lane] << 8) | (sz_size_t)low_lanes[lane]];
    return wasm_v128_and(wasm_v128_load(class_lanes), in_range_u8x16);
}

/** @brief  v128 forward drain — the `vpcompressb`-free twin of @ref sz_utf8_rune_drain_forward_. Emits one
 *          (start, length) per set boundary lane (ascending), honoring @p capacity and the carried previous-boundary
 *          via @p previous_io; bit-exact with the Ice Lake leaf. Consumption is inherently scalar (one output pair
 *          per lane), so each set lane is isolated with the `63 - clz(mask & -mask)` first-set idiom and cleared with
 *          `mask & (mask - 1)` — the cost scales with the boundary count, and no `ctz` / `popcount` builtin is used. */
SZ_HELPER_AUTO sz_size_t sz_utf8_rune_drain_forward_v128_( //
    sz_u64_t boundary, sz_size_t base, sz_size_t *starts, sz_size_t *lengths, sz_size_t produced, sz_size_t capacity,
    sz_size_t *previous_io) {
    sz_size_t previous = *previous_io;
    sz_u64_t remaining = boundary;
    while (remaining && produced < capacity) {
        sz_size_t const lane = (sz_size_t)(63 - sz_u64_clz(remaining & (~remaining + 1ull)));
        remaining &= remaining - 1ull;
        sz_size_t const position = base + lane;
        starts[produced] = previous;
        lengths[produced] = position - previous;
        previous = position;
        ++produced;
    }
    *previous_io = previous;
    return produced;
}

#pragma endregion Shared SIMD leaf substrate

#pragma region Vectorized rune decode

/** @brief  Gather 16 bytes from the 64-byte window held in @p regs at the per-lane byte offsets @p index (in [0,64)),
 *          gather-free. `wasm_i8x16_swizzle` reaches only one 16-byte register (index >= 16 -> 0), so each quarter is
 *          swizzled by `index - 16*q` and the four disjoint results OR-ed: for any lane exactly one quarter holds the
 *          offset in range, the others contribute 0. Offsets past the window (>= 64) read as 0 (discarded downstream).
 *          The v128 twin of @ref sz_utf8_gather16_powervsx_. */
SZ_HELPER_INLINE v128_t sz_utf8_rune_gather16_v128_(v128_t const *regs, v128_t index_u8x16) {
    v128_t gathered_u8x16 = wasm_i8x16_swizzle(regs[0], index_u8x16);
    gathered_u8x16 = wasm_v128_or(gathered_u8x16,
                                  wasm_i8x16_swizzle(regs[1], wasm_i8x16_sub(index_u8x16, wasm_i8x16_splat(16))));
    gathered_u8x16 = wasm_v128_or(gathered_u8x16,
                                  wasm_i8x16_swizzle(regs[2], wasm_i8x16_sub(index_u8x16, wasm_i8x16_splat(32))));
    gathered_u8x16 = wasm_v128_or(gathered_u8x16,
                                  wasm_i8x16_swizzle(regs[3], wasm_i8x16_sub(index_u8x16, wasm_i8x16_splat(48))));
    return gathered_u8x16;
}

/** @brief  Zero-extend the four byte lanes [4*quarter, 4*quarter+4) of @p bytes_u8x16 into a `u32x4`. The base selector
 *          `{0,-,-,-,1,-,-,-,2,-,-,-,3,-,-,-}` (unused lanes 0x10 -> swizzle 0) is biased by `4*quarter`; the biased
 *          filler lanes stay >= 16 so they still swizzle to 0, so one `wasm_i8x16_swizzle` widens the quarter. */
SZ_HELPER_INLINE v128_t sz_utf8_rune_widen4_v128_(v128_t bytes_u8x16, int quarter, v128_t base_selector_u8x16) {
    v128_t const selector_u8x16 = wasm_i8x16_add(base_selector_u8x16, wasm_i8x16_splat((sz_i8_t)(quarter * 4)));
    return wasm_i8x16_swizzle(bytes_u8x16, selector_u8x16);
}

/**
 *  @brief  Decode the emitted start lanes @p emit_starts of a classified 64-byte window (held in the four 16-byte
 *          @p regs) into sequential UTF-32 runes, the v128 sibling of @ref sz_utf8_rune_drain_powervsx_. The start
 *          byte-offsets are left-packed by one first-set (`63 - clz(mask & -mask)`) bit walk — no `ctz` / `popcount`
 *          builtin — then a 16-lane-block loop gathers the lead + up to three trailing bytes with
 *          @ref sz_utf8_rune_gather16_v128_ and width-blends each codepoint in `u32x4` lanes; the wider 3rd/4th bytes
 *          are gathered and blended CONDITIONALLY (@p has_three / @p has_four sibling `if`s). Every @p ill_formed start
 *          (its maximal ill-formed subpart) is overwritten with U+FFFD, and the resume cursor reads the last emitted
 *          start's offset + its compacted @p consumed_length, so an ill-formed trailing lane never skips bytes owed
 *          their own next U+FFFD.
 *  @return Number of runes emitted; sets @p consumed_bytes to the byte span they cover (the resume cursor delta).
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_rune_drain_v128_( //
    v128_t const *regs, sz_u64_t emit_starts, sz_u64_t ill_formed, sz_u8_t const *consumed_length, int has_three,
    int has_four, sz_size_t capacity, sz_rune_t *runes, sz_size_t *consumed_bytes) {

    // Left-pack the emitted start byte-offsets (ascending) by isolating each set bit with `mask & -mask` and reading
    // its lane via `63 - clz`; clearing it with `mask & (mask - 1)`. The count falls out as the loop trip count, so no
    // separate `popcount` is needed and the cost scales with the emitted-start count.
    sz_u8_t packed_offsets[64];
    sz_size_t emit_count = 0;
    for (sz_u64_t remaining = emit_starts; remaining; remaining &= remaining - 1ull) {
        sz_size_t const lane = (sz_size_t)(63 - sz_u64_clz(remaining & (~remaining + 1ull)));
        packed_offsets[emit_count++] = (sz_u8_t)lane;
    }

    sz_size_t const want = emit_count < capacity ? emit_count : capacity;
    v128_t const base_selector_u8x16 = wasm_i8x16_make(0, 16, 16, 16, 1, 16, 16, 16, 2, 16, 16, 16, 3, 16, 16, 16);
    v128_t const mask_word_3f_u32x4 = wasm_u32x4_splat(0x3F);
    v128_t const mask_word_c0_u32x4 = wasm_u32x4_splat(0xC0), mask_word_e0_u32x4 = wasm_u32x4_splat(0xE0),
                 mask_word_f0_u32x4 = wasm_u32x4_splat(0xF0);
    sz_size_t produced = 0;
    for (sz_size_t block_start = 0; block_start < want; block_start += 16) {
        // Load up to 16 packed offsets into an index vector; trailing lanes get 0xFF (>= 64 -> gathers 0, discarded).
        sz_u128_vec_t index_vec;
        for (int lane = 0; lane < 16; ++lane) {
            sz_size_t const source = block_start + (sz_size_t)lane;
            index_vec.u8s[lane] = source < want ? packed_offsets[source] : (sz_u8_t)0xFF;
        }
        v128_t const index_u8x16 = index_vec.v128;
        v128_t const lead_byte_u8x16 = sz_utf8_rune_gather16_v128_(regs, index_u8x16);
        v128_t const continuation_byte_1_u8x16 = sz_utf8_rune_gather16_v128_(
            regs, wasm_i8x16_add(index_u8x16, wasm_i8x16_splat(1)));
        v128_t const continuation_byte_2_u8x16 = has_three ? sz_utf8_rune_gather16_v128_(
                                                                 regs, wasm_i8x16_add(index_u8x16, wasm_i8x16_splat(2)))
                                                           : wasm_i8x16_splat(0);
        v128_t const continuation_byte_3_u8x16 = has_four ? sz_utf8_rune_gather16_v128_(
                                                                regs, wasm_i8x16_add(index_u8x16, wasm_i8x16_splat(3)))
                                                          : wasm_i8x16_splat(0);

        for (int quarter = 0; quarter < 4; ++quarter) {
            sz_size_t const out_base = block_start + (sz_size_t)quarter * 4;
            if (out_base >= want) break;
            v128_t const lead_u32x4 = sz_utf8_rune_widen4_v128_(lead_byte_u8x16, quarter, base_selector_u8x16);
            v128_t const continuation_byte_1_u32x4 = sz_utf8_rune_widen4_v128_(continuation_byte_1_u8x16, quarter,
                                                                               base_selector_u8x16);
            v128_t const continuation_byte_2_u32x4 = sz_utf8_rune_widen4_v128_(continuation_byte_2_u8x16, quarter,
                                                                               base_selector_u8x16);
            v128_t const continuation_byte_3_u32x4 = sz_utf8_rune_widen4_v128_(continuation_byte_3_u8x16, quarter,
                                                                               base_selector_u8x16);

            // 2-byte: lead in [C0,E0): ((b0 & 0x1F) << 6) | (b1 & 0x3F). ASCII (< C0) keeps the raw lead value.
            v128_t const two_byte_mask_u32x4 = wasm_v128_and(wasm_u32x4_ge(lead_u32x4, mask_word_c0_u32x4),
                                                             wasm_u32x4_lt(lead_u32x4, mask_word_e0_u32x4));
            v128_t const two_byte_codepoint_u32x4 = wasm_v128_or(
                wasm_i32x4_shl(wasm_v128_and(lead_u32x4, wasm_u32x4_splat(0x1F)), 6),
                wasm_v128_and(continuation_byte_1_u32x4, mask_word_3f_u32x4));
            v128_t decoded_codepoint_u32x4 = wasm_v128_bitselect(two_byte_codepoint_u32x4, lead_u32x4,
                                                                 two_byte_mask_u32x4);
            // The 3-/4-byte width blends run CONDITIONALLY via two sibling (depth-1) `if`s gated by `has_three` /
            // `has_four`, mirroring the VSX drain: a window with no 3-/4-byte lead skips the wider gather + assembly.
            // `has_four => has_three`, so the 4-byte sibling always sees a populated 3rd byte. Bit-exact with the gated
            // form: without a 3-/4-byte lead the corresponding width mask is all-false in every emitted lane.
            if (has_three) {
                v128_t const three_byte_mask_u32x4 = wasm_v128_and(wasm_u32x4_ge(lead_u32x4, mask_word_e0_u32x4),
                                                                   wasm_u32x4_lt(lead_u32x4, mask_word_f0_u32x4));
                v128_t const three_byte_codepoint_u32x4 = wasm_v128_or(
                    wasm_v128_or(wasm_i32x4_shl(wasm_v128_and(lead_u32x4, wasm_u32x4_splat(0x0F)), 12),
                                 wasm_i32x4_shl(wasm_v128_and(continuation_byte_1_u32x4, mask_word_3f_u32x4), 6)),
                    wasm_v128_and(continuation_byte_2_u32x4, mask_word_3f_u32x4));
                decoded_codepoint_u32x4 = wasm_v128_bitselect(three_byte_codepoint_u32x4, decoded_codepoint_u32x4,
                                                              three_byte_mask_u32x4);
            }
            if (has_four) { // SIBLING, not nested; has_four => has_three so the 3rd byte is already gathered.
                v128_t const four_byte_mask_u32x4 = wasm_u32x4_ge(lead_u32x4, mask_word_f0_u32x4);
                v128_t const four_byte_codepoint_u32x4 = wasm_v128_or(
                    wasm_v128_or(wasm_i32x4_shl(wasm_v128_and(lead_u32x4, wasm_u32x4_splat(0x07)), 18),
                                 wasm_i32x4_shl(wasm_v128_and(continuation_byte_1_u32x4, mask_word_3f_u32x4), 12)),
                    wasm_v128_or(wasm_i32x4_shl(wasm_v128_and(continuation_byte_2_u32x4, mask_word_3f_u32x4), 6),
                                 wasm_v128_and(continuation_byte_3_u32x4, mask_word_3f_u32x4)));
                decoded_codepoint_u32x4 = wasm_v128_bitselect(four_byte_codepoint_u32x4, decoded_codepoint_u32x4,
                                                              four_byte_mask_u32x4);
            }

            sz_u128_vec_t lane_values_vec;
            lane_values_vec.v128 = decoded_codepoint_u32x4;
            sz_size_t lanes = want - out_base;
            if (lanes > 4) lanes = 4;
            for (sz_size_t lane = 0; lane < lanes; ++lane) {
                sz_size_t const flat = out_base + lane;
                int const offset = packed_offsets[flat];
                sz_rune_t value = (sz_rune_t)lane_values_vec.u32s[lane];
                if ((ill_formed >> offset) & 1) value = (sz_rune_t)sz_rune_replacement_k;
                runes[flat] = value;
            }
            produced += lanes;
        }
    }

    sz_size_t const last_offset = packed_offsets[produced - 1];
    *consumed_bytes = (sz_size_t)last_offset + (sz_size_t)consumed_length[last_offset];
    return produced;
}

/**
 *  @brief  Decode one <=64-byte window of @p text into dense UTF-32 @p runes by the uniform classify -> per-lane
 *          well-formed + orphan promotion -> compress emitted starts -> gather -> width-blend -> blend U+FFFD path,
 *          the v128 sibling of @ref sz_utf8_decode_once_powervsx_. The decode is TOTAL: clean and dirty bytes are
 *          handled in-vector, one U+FFFD per maximal ill-formed subpart (Unicode 17.0 §3.9 / W3C), bit-exact with
 *          @ref sz_utf8_decode_serial. The step declines (`*runes_unpacked == 0`, cursor unchanged) ONLY when the
 *          first decodable lead's declared sequence crosses the window edge (a boundary truncation), which the public
 *          entry finalizes without a serial re-decode.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_decode_once_v128_( //
    sz_cptr_t text, sz_size_t length,               //
    sz_rune_t *runes, sz_size_t runes_capacity,     //
    sz_size_t *runes_unpacked) {

    sz_size_t const chunk = length < 64 ? length : 64;
    sz_u64_t const loaded_mask = sz_u64_mask_until_serial_(chunk);

    v128_t regs[4];
    sz_utf8_rune_load_window_v128_((sz_u8_t const *)text, chunk, regs);

    // ASCII fast lane: a window with no high bit set widens directly (the bulk-ASCII case is peeled by the caller, so
    // this only fires for a short all-ASCII residue and can store scalar). `wasm_i8x16_bitmask` flags high-bit lanes.
    v128_t nonascii_bool[4];
    for (int quarter = 0; quarter < 4; ++quarter)
        nonascii_bool[quarter] = wasm_u8x16_lt(wasm_i8x16_splat((sz_i8_t)0x7F), regs[quarter]);
    sz_u64_t const nonascii = sz_utf8_mask_combine_v128_(nonascii_bool[0], nonascii_bool[1], nonascii_bool[2],
                                                         nonascii_bool[3]) &
                              loaded_mask;
    if (nonascii == 0) {
        sz_size_t const want = chunk < runes_capacity ? chunk : runes_capacity;
        for (sz_size_t index = 0; index < want; ++index) runes[index] = (sz_rune_t)(sz_u8_t)text[index];
        *runes_unpacked = want;
        return text + want;
    }

    // Single-source classification via the high-nibble length LUT (the SAME table the serial reference uses), so a
    // lead and its declared length can never disagree - crucially, a stray lead such as 0xF8 maps to length 4 and is
    // rejected by the bad-lead gate rather than mistaken for a 1-byte start. Per-quarter neighbours feed the range gate.
    v128_t next1[4], next2[4], next3[4];
    sz_utf8_forward_neighbours_v128_(regs, next1, next2, next3);
    v128_t const length_lut_u8x16 = wasm_i8x16_make(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4);
    v128_t const iota_u8x16 = wasm_i8x16_make(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    v128_t const chunk_u8x16 = wasm_i8x16_splat((sz_i8_t)chunk);

    v128_t continuation_bool[4], length_2_bool[4], length_3_bool[4], length_4_bool[4];
    v128_t overrun_bool[4], bad_lead_bool[4], range_bad_bool[4];
    for (int quarter = 0; quarter < 4; ++quarter) {
        v128_t const here_u8x16 = regs[quarter];
        v128_t const next_u8x16 = next1[quarter];
        v128_t const high_nibble_u8x16 = wasm_v128_and(wasm_u8x16_shr(here_u8x16, 4), wasm_i8x16_splat(0x0F));
        v128_t const length_u8x16 = wasm_i8x16_swizzle(length_lut_u8x16, high_nibble_u8x16);

        continuation_bool[quarter] = wasm_i8x16_eq(wasm_v128_and(here_u8x16, wasm_i8x16_splat((sz_i8_t)0xC0)),
                                                   wasm_i8x16_splat((sz_i8_t)0x80));
        v128_t const is_len2 = wasm_i8x16_eq(length_u8x16, wasm_i8x16_splat(2));
        v128_t const is_len4 = wasm_i8x16_eq(length_u8x16, wasm_i8x16_splat(4));
        length_2_bool[quarter] = is_len2;
        length_3_bool[quarter] = wasm_i8x16_eq(length_u8x16, wasm_i8x16_splat(3));
        length_4_bool[quarter] = is_len4;

        // Overrun: global lane offset + declared length past `chunk` (unsigned byte compare).
        v128_t const global_offset_u8x16 = wasm_i8x16_add(iota_u8x16, wasm_i8x16_splat((sz_i8_t)(quarter * 16)));
        v128_t const sequence_end_u8x16 = wasm_i8x16_add(global_offset_u8x16, length_u8x16);
        overrun_bool[quarter] = wasm_u8x16_gt(sequence_end_u8x16, chunk_u8x16);

        // Bad lead: 0xC0/0xC1 (overlong 2-byte by the LUT) or 0xF5..0xFF (out of range, length-4 by the LUT).
        v128_t const lead_lt_c2 = wasm_u8x16_lt(here_u8x16, wasm_i8x16_splat((sz_i8_t)0xC2));
        v128_t const lead_gt_f4 = wasm_u8x16_gt(here_u8x16, wasm_i8x16_splat((sz_i8_t)0xF4));
        bad_lead_bool[quarter] = wasm_v128_or(wasm_v128_and(is_len2, lead_lt_c2), wasm_v128_and(is_len4, lead_gt_f4));

        // First-continuation range violations for E0/ED/F0/F4 (overlong 3/4-byte, surrogate, > U+10FFFF).
        v128_t const e0_bad = wasm_v128_and(wasm_i8x16_eq(here_u8x16, wasm_i8x16_splat((sz_i8_t)0xE0)),
                                            wasm_u8x16_lt(next_u8x16, wasm_i8x16_splat((sz_i8_t)0xA0)));
        v128_t const ed_bad = wasm_v128_and(wasm_i8x16_eq(here_u8x16, wasm_i8x16_splat((sz_i8_t)0xED)),
                                            wasm_u8x16_ge(next_u8x16, wasm_i8x16_splat((sz_i8_t)0xA0)));
        v128_t const f0_bad = wasm_v128_and(wasm_i8x16_eq(here_u8x16, wasm_i8x16_splat((sz_i8_t)0xF0)),
                                            wasm_u8x16_lt(next_u8x16, wasm_i8x16_splat((sz_i8_t)0x90)));
        v128_t const f4_bad = wasm_v128_and(wasm_i8x16_eq(here_u8x16, wasm_i8x16_splat((sz_i8_t)0xF4)),
                                            wasm_u8x16_ge(next_u8x16, wasm_i8x16_splat((sz_i8_t)0x90)));
        range_bad_bool[quarter] = wasm_v128_or(wasm_v128_or(e0_bad, ed_bad), wasm_v128_or(f0_bad, f4_bad));
    }

    sz_u64_t const continuation_bits = sz_utf8_mask_combine_v128_(continuation_bool[0], continuation_bool[1],
                                                                  continuation_bool[2], continuation_bool[3]) &
                                       loaded_mask;
    sz_u64_t const starts_bits = loaded_mask & ~continuation_bits;
    sz_u64_t const len2_bits = sz_utf8_mask_combine_v128_(length_2_bool[0], length_2_bool[1], length_2_bool[2],
                                                          length_2_bool[3]);
    sz_u64_t const len3_bits = sz_utf8_mask_combine_v128_(length_3_bool[0], length_3_bool[1], length_3_bool[2],
                                                          length_3_bool[3]);
    sz_u64_t const len4_bits = sz_utf8_mask_combine_v128_(length_4_bool[0], length_4_bool[1], length_4_bool[2],
                                                          length_4_bool[3]);
    sz_u64_t const length_ge_two_starts = (len2_bits | len3_bits | len4_bits) & starts_bits;
    sz_u64_t const length_ge_three_starts = (len3_bits | len4_bits) & starts_bits;
    sz_u64_t const length_ge_four_starts = len4_bits & starts_bits;

    // Branchless overrun defer: the FIRST overrunning start bounds the decodable prefix (well-formed text overruns only
    // at the trailing truncation; a malformed `E0 C0` overruns earlier). First-set via `63 - clz(mask & -mask)`.
    sz_u64_t const overruns =
        sz_utf8_mask_combine_v128_(overrun_bool[0], overrun_bool[1], overrun_bool[2], overrun_bool[3]) & starts_bits;
    sz_size_t const decodable_end = overruns ? (sz_size_t)(63 - sz_u64_clz(overruns & (~overruns + 1ull))) : chunk;
    sz_u64_t const decodable_mask = sz_u64_mask_until_serial_(decodable_end);

    sz_u64_t const bad_lead_bits = sz_utf8_mask_combine_v128_(bad_lead_bool[0], bad_lead_bool[1], bad_lead_bool[2],
                                                              bad_lead_bool[3]) &
                                   starts_bits;
    sz_u64_t const overlong_or_surrogate_or_range_bits =
        sz_utf8_mask_combine_v128_(range_bad_bool[0], range_bad_bool[1], range_bad_bool[2], range_bad_bool[3]) &
        starts_bits;

    int const has_three = length_ge_three_starts != 0;
    int const has_four = length_ge_four_starts != 0;

    // Per-lane continuation availability at the declared trailing slots, evaluated at the lead lane (mirror icelake).
    sz_u64_t const cont1 = starts_bits & (continuation_bits >> 1);
    sz_u64_t const cont2 = starts_bits & (continuation_bits >> 2);
    sz_u64_t const cont3 = starts_bits & (continuation_bits >> 3);
    sz_u64_t const b1_range_ok = starts_bits & ~overlong_or_surrogate_or_range_bits;
    sz_u64_t const first_ok = cont1 & b1_range_ok & ~bad_lead_bits;

    sz_u64_t const length_one_bits = starts_bits & ~length_ge_two_starts;
    sz_u64_t const length_two_bits = length_ge_two_starts & ~length_ge_three_starts;
    sz_u64_t const wf1 = length_one_bits;
    sz_u64_t const wf2 = length_two_bits & ~bad_lead_bits & cont1;
    sz_u64_t const wf3 = (length_ge_three_starts & ~length_ge_four_starts) & b1_range_ok & cont1 & cont2;
    sz_u64_t const wf4 = length_ge_four_starts & ~bad_lead_bits & b1_range_ok & cont1 & cont2 & cont3;
    sz_u64_t const well_formed = (wf1 | wf2 | wf3 | wf4) & decodable_mask;

    // Per-lane maximal-subpart length: extend across each continuation slot a well-formed sequence would accept.
    sz_u64_t const step2 = length_ge_two_starts & first_ok;
    sz_u64_t const step3 = step2 & length_ge_three_starts & cont2;
    sz_u64_t const step4 = step3 & length_ge_four_starts & cont3;

    // Orphan promotion: a continuation byte covered by NO lead's maximal-subpart span becomes its own 1-byte U+FFFD.
    sz_u64_t const covered = ((step2 & decodable_mask) << 1) | ((step3 & decodable_mask) << 2) |
                             ((step4 & decodable_mask) << 3);
    sz_u64_t const orphan = continuation_bits & decodable_mask & ~covered;
    sz_u64_t const emit_starts = (starts_bits | orphan) & decodable_mask;
    if (emit_starts == 0) { return *runes_unpacked = 0, text; } // Window-edge truncation -> driver finalize.
    sz_u64_t const ill_formed = emit_starts & ~well_formed;

    // Per-lane (window-order) maximal-subpart length: 1 + the continuation slots the steps reached.
    sz_u512_vec_t consumed_length_vec;
    for (int lane = 0; lane < 64; ++lane) {
        sz_u8_t length_here = 1;
        length_here = (sz_u8_t)(length_here + (sz_u8_t)((step2 >> lane) & 1));
        length_here = (sz_u8_t)(length_here + (sz_u8_t)((step3 >> lane) & 1));
        length_here = (sz_u8_t)(length_here + (sz_u8_t)((step4 >> lane) & 1));
        consumed_length_vec.u8s[lane] = length_here;
    }

    sz_size_t consumed = 0;
    sz_size_t const produced = sz_utf8_rune_drain_v128_(regs, emit_starts, ill_formed, consumed_length_vec.u8s,
                                                        has_three, has_four, runes_capacity, runes, &consumed);
    *runes_unpacked = produced;
    return text + consumed;
}

/*  Decodes UTF-8 into UTF-32 runes. The pure-ASCII prefix (every byte `< 0x80`) widens 16 bytes straight into 16 u32
 *  runes via zero-extends; the multi-byte / malformed / truncation remainder is decoded in-register, one 64-byte window
 *  at a time, by @ref sz_utf8_decode_once_v128_ (no serial-stride re-decode). A step that declines a window-edge
 *  truncation is finalized by the shared bounded <=3-byte maximal-subpart helper, exactly like the other backends. */
SZ_API_COMPTIME sz_cptr_t sz_utf8_decode_v128(  //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {

    sz_u8_t const *text_cursor = (sz_u8_t const *)text;
    sz_size_t runes_written = 0;

    // Vectorized pure-ASCII prefix: 16 single-byte runes per window.
    while (length - (sz_size_t)(text_cursor - (sz_u8_t const *)text) >= 16 && runes_written + 16 <= runes_capacity) {
        v128_t bytes = wasm_v128_load(text_cursor);
        // Any byte >= 0x80 has its sign bit set; if present, hand the window to the general decoder.
        if (wasm_i8x16_bitmask(bytes) != 0) break;
        // Widen 16x u8 → 16x u32 and store as runes (zero-extension == the ASCII code point value).
        v128_t low_u16x8 = wasm_u16x8_extend_low_u8x16(bytes);
        v128_t high_u16x8 = wasm_u16x8_extend_high_u8x16(bytes);
        wasm_v128_store(runes + runes_written + 0, wasm_u32x4_extend_low_u16x8(low_u16x8));
        wasm_v128_store(runes + runes_written + 4, wasm_u32x4_extend_high_u16x8(low_u16x8));
        wasm_v128_store(runes + runes_written + 8, wasm_u32x4_extend_low_u16x8(high_u16x8));
        wasm_v128_store(runes + runes_written + 12, wasm_u32x4_extend_high_u16x8(high_u16x8));
        runes_written += 16;
        text_cursor += 16;
    }

    // Multi-byte / malformed / truncation remainder: in-register window decode, declining-window -> bounded finalize.
    sz_cptr_t cursor = (sz_cptr_t)text_cursor;
    sz_cptr_t const end = text + length;
    while (runes_written < runes_capacity && cursor < end) {
        sz_size_t step_unpacked = 0;
        sz_cptr_t next = sz_utf8_decode_once_v128_(cursor, (sz_size_t)(end - cursor), runes + runes_written,
                                                   runes_capacity - runes_written, &step_unpacked);
        if (step_unpacked) {
            runes_written += step_unpacked;
            cursor = next;
            continue;
        }
        // `step_unpacked == 0` only when the first decodable lead declares a sequence crossing the window edge (a
        // boundary truncation). A resumable truncation breaks and awaits more bytes; a bad/overlong truncated lead at
        // the edge finalizes to one U+FFFD over its maximal ill-formed subpart - a bounded <=3-byte finalize, never a
        // serial window re-decode.
        if (sz_utf8_incomplete_tail_(cursor, end)) break;
        runes[runes_written++] = (sz_rune_t)sz_rune_replacement_k;
        cursor += sz_utf8_maximal_subpart_(cursor, end);
    }
    *runes_unpacked = runes_written;
    return cursor;
}

#pragma endregion Vectorized rune decode

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_RUNES_V128_H_
