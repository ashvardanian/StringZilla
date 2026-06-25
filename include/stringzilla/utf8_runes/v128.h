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

SZ_PUBLIC sz_size_t sz_utf8_count_v128(sz_cptr_t text, sz_size_t length) {
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
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_v128(sz_cptr_t text, sz_size_t length, sz_size_t n) {
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

    return sz_utf8_find_nth_serial((sz_cptr_t)text_u8, length, n);
}

/*  Decodes UTF-8 into UTF-32 runes. The pure-ASCII prefix (every byte `< 0x80`) widens 16 bytes straight
 *  into 16 u32 runes via zero-extends; the first non-ASCII byte or exhausted capacity defers the remainder
 *  to `sz_utf8_decode_serial`, which owns all multi-byte / malformed / truncation logic. */
SZ_PUBLIC sz_cptr_t sz_utf8_decode_v128(        //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {

    sz_u8_t const *text_cursor = (sz_u8_t const *)text;
    sz_size_t runes_written = 0;

    // Vectorized pure-ASCII prefix: 16 single-byte runes per window.
    while (length - (sz_size_t)(text_cursor - (sz_u8_t const *)text) >= 16 && runes_written + 16 <= runes_capacity) {
        v128_t bytes = wasm_v128_load(text_cursor);
        // Any byte >= 0x80 has its sign bit set; if present, defer the whole window to serial.
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

    // Remainder (multi-byte sequences, ragged tail, or exhausted capacity) goes through serial.
    sz_size_t bytes_consumed = (sz_size_t)(text_cursor - (sz_u8_t const *)text);
    sz_size_t tail_unpacked = 0;
    sz_cptr_t tail_end = sz_utf8_decode_serial((sz_cptr_t)text_cursor, length - bytes_consumed, runes + runes_written,
                                               runes_capacity - runes_written, &tail_unpacked);
    *runes_unpacked = runes_written + tail_unpacked;
    return tail_end;
}

/*  UAX-29 word-boundary detection is a stateful machine. We accelerate only its dominant cost: scanning
 *  the all-ASCII interior of long words/numbers, where every UAX-29 rule resolves to "do NOT break" - so an
 *  all-ASCII window resolves its trusted lanes [2,14] branchlessly. Any non-ASCII window or edge is handed
 *  to `sz_utf8_is_word_boundary_serial` verbatim, keeping the result bit-for-bit identical to serial. */

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_RUNES_V128_H_
