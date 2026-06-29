/**
 *  @brief WebAssembly SIMD128 backend for find.
 *  @file include/stringzilla/find/v128.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_V128_H_
#define STRINGZILLA_FIND_V128_H_

#include "stringzilla/types.h"
#include "stringzilla/find/serial.h"
#include "stringzilla/compare.h" // `sz_equal`

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128

/*  WebAssembly SIMD128 has a TRUE movemask via `wasm_i8x16_bitmask`, producing one bit per byte
 *  (the most significant bit of each lane). Combined with `sz_u32_ctz`/`sz_u32_clz` we get the
 *  SSE/Westmere-style search. The fixed register width is 16 bytes, like Arm NEON. */
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

/**
 *  @brief Return a pointer to the first matching byte in a 16-byte block, or NULL if none.
 *  @param match_vec A 16-byte comparison result (0xFF where matched, 0x00 otherwise).
 *  @param block_start Pointer to the first byte of the block.
 *  @return Pointer to the first match, or SZ_NULL_CHAR if none.
 */
SZ_HELPER_INLINE sz_cptr_t sz_locate_first_v128_(v128_t match_vec, sz_cptr_t block_start) {
    sz_u32_t matches = (sz_u32_t)wasm_i8x16_bitmask(match_vec);
    if (!matches) return SZ_NULL_CHAR;
    return block_start + sz_u32_ctz(matches);
}

/**
 *  @brief Return a pointer to the last matching byte in a 16-byte block, or NULL if none.
 *  @param match_vec A 16-byte comparison result (0xFF where matched, 0x00 otherwise).
 *  @param block_start Pointer to the first byte of the block.
 *  @return Pointer to the last match, or SZ_NULL_CHAR if none.
 */
SZ_HELPER_INLINE sz_cptr_t sz_locate_last_v128_(v128_t match_vec, sz_cptr_t block_start) {
    sz_u32_t matches = (sz_u32_t)wasm_i8x16_bitmask(match_vec);
    if (!matches) return SZ_NULL_CHAR;
    // `matches` occupies the low 16 bits, so `31 - clz` is the index of its highest set bit.
    return block_start + (31 - sz_u32_clz(matches));
}

SZ_API_COMPTIME sz_cptr_t sz_find_byte_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    v128_t needle_vec = wasm_i8x16_splat(*(sz_i8_t const *)needle);

    // Scan 64 bytes per iteration: OR four equality masks and gate with a single `any_true`, only
    // descending into the per-block locate when one of the windows actually contains a match.
    while (haystack_length >= 64) {
        v128_t match_at_0_vec = wasm_i8x16_eq(wasm_v128_load(haystack + 0), needle_vec);
        v128_t match_at_16_vec = wasm_i8x16_eq(wasm_v128_load(haystack + 16), needle_vec);
        v128_t match_at_32_vec = wasm_i8x16_eq(wasm_v128_load(haystack + 32), needle_vec);
        v128_t match_at_48_vec = wasm_i8x16_eq(wasm_v128_load(haystack + 48), needle_vec);
        v128_t any_match_vec = wasm_v128_or(wasm_v128_or(match_at_0_vec, match_at_16_vec),
                                            wasm_v128_or(match_at_32_vec, match_at_48_vec));
        if (wasm_v128_any_true(any_match_vec)) {
            sz_cptr_t found;
            if ((found = sz_locate_first_v128_(match_at_0_vec, haystack + 0))) return found;
            if ((found = sz_locate_first_v128_(match_at_16_vec, haystack + 16))) return found;
            if ((found = sz_locate_first_v128_(match_at_32_vec, haystack + 32))) return found;
            if ((found = sz_locate_first_v128_(match_at_48_vec, haystack + 48))) return found;
        }
        haystack += 64, haystack_length -= 64;
    }

    while (haystack_length >= 16) {
        sz_cptr_t found = sz_locate_first_v128_(wasm_i8x16_eq(wasm_v128_load(haystack), needle_vec), haystack);
        if (found) return found;
        haystack += 16, haystack_length -= 16;
    }

    return sz_find_byte_serial(haystack, haystack_length, needle);
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    v128_t needle_vec = wasm_i8x16_splat(*(sz_i8_t const *)needle);

    // Scan the trailing 64 bytes per iteration; on a hit, locate the LAST match by walking the four
    // sub-windows from the highest one down (each block reports its own latest match).
    while (haystack_length >= 64) {
        sz_cptr_t window_start = haystack + haystack_length - 64;
        v128_t match_at_0_vec = wasm_i8x16_eq(wasm_v128_load(window_start + 0), needle_vec);
        v128_t match_at_16_vec = wasm_i8x16_eq(wasm_v128_load(window_start + 16), needle_vec);
        v128_t match_at_32_vec = wasm_i8x16_eq(wasm_v128_load(window_start + 32), needle_vec);
        v128_t match_at_48_vec = wasm_i8x16_eq(wasm_v128_load(window_start + 48), needle_vec);
        v128_t any_match_vec = wasm_v128_or(wasm_v128_or(match_at_0_vec, match_at_16_vec),
                                            wasm_v128_or(match_at_32_vec, match_at_48_vec));
        if (wasm_v128_any_true(any_match_vec)) {
            sz_cptr_t found;
            if ((found = sz_locate_last_v128_(match_at_48_vec, window_start + 48))) return found;
            if ((found = sz_locate_last_v128_(match_at_32_vec, window_start + 32))) return found;
            if ((found = sz_locate_last_v128_(match_at_16_vec, window_start + 16))) return found;
            if ((found = sz_locate_last_v128_(match_at_0_vec, window_start + 0))) return found;
        }
        haystack_length -= 64;
    }

    while (haystack_length >= 16) {
        sz_cptr_t window_start = haystack + haystack_length - 16;
        sz_cptr_t found = sz_locate_last_v128_(wasm_i8x16_eq(wasm_v128_load(window_start), needle_vec), window_start);
        if (found) return found;
        haystack_length -= 16;
    }

    return sz_rfind_byte_serial(haystack, haystack_length, needle);
}

/**
 *  @brief Compute a per-lane match mask for bytes present in `set`, split across two 16-byte half-tables.
 *  @param haystack_vec The 16-byte input register.
 *  @param set_top_vec Top half of the byteset (byte indices 0..15).
 *  @param set_bottom_vec Bottom half of the byteset (byte indices 16..31).
 *  @return 0xFF per lane where the byte belongs to the set, 0x00 otherwise.
 */
SZ_HELPER_AUTO v128_t sz_find_byteset_match_v128_(v128_t haystack_vec, v128_t set_top_vec, v128_t set_bottom_vec) {
    // Serial equivalent per byte `c`: `(set->_u8s[c >> 3] & (1u << (c & 7u))) != 0`.
    v128_t byte_index_vec = wasm_u8x16_shr(haystack_vec, 3); // c >> 3, in [0, 31]
    // The bit mask `1 << (c & 7)` is produced via a swizzle into a tiny power-of-two table.
    v128_t bit_table = wasm_i8x16_make(1, 2, 4, 8, 16, 32, 64, (sz_i8_t)128, 0, 0, 0, 0, 0, 0, 0, 0);
    v128_t byte_mask_vec = wasm_i8x16_swizzle(bit_table, wasm_v128_and(haystack_vec, wasm_i8x16_splat(7)));
    // `swizzle` returns 0 for indices >= 16 (or with the high bit set). For the bottom half we shift
    // the index down by 16: values originally < 16 underflow into [240, 256) and yield zeros there.
    v128_t matches_top_vec = wasm_i8x16_swizzle(set_top_vec, byte_index_vec);
    v128_t matches_bottom_vec = wasm_i8x16_swizzle(set_bottom_vec,
                                                   wasm_i8x16_sub(byte_index_vec, wasm_i8x16_splat(16)));
    v128_t matches_vec = wasm_v128_or(matches_top_vec, matches_bottom_vec);
    // Keep a lane (0xFF) only if its selected bit is set.
    return wasm_i8x16_ne(wasm_v128_and(matches_vec, byte_mask_vec), wasm_i8x16_splat(0));
}

SZ_API_COMPTIME sz_cptr_t sz_find_byteset_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *set) {
    v128_t set_top_vec = wasm_v128_load(&set->_u8s[0]);
    v128_t set_bottom_vec = wasm_v128_load(&set->_u8s[16]);

    while (haystack_length >= 64) {
        v128_t match_at_0_vec = sz_find_byteset_match_v128_(wasm_v128_load(haystack + 0), set_top_vec, set_bottom_vec);
        v128_t match_at_16_vec = sz_find_byteset_match_v128_(wasm_v128_load(haystack + 16), set_top_vec,
                                                             set_bottom_vec);
        v128_t match_at_32_vec = sz_find_byteset_match_v128_(wasm_v128_load(haystack + 32), set_top_vec,
                                                             set_bottom_vec);
        v128_t match_at_48_vec = sz_find_byteset_match_v128_(wasm_v128_load(haystack + 48), set_top_vec,
                                                             set_bottom_vec);
        v128_t any_match_vec = wasm_v128_or(wasm_v128_or(match_at_0_vec, match_at_16_vec),
                                            wasm_v128_or(match_at_32_vec, match_at_48_vec));
        if (wasm_v128_any_true(any_match_vec)) {
            sz_cptr_t found;
            if ((found = sz_locate_first_v128_(match_at_0_vec, haystack + 0))) return found;
            if ((found = sz_locate_first_v128_(match_at_16_vec, haystack + 16))) return found;
            if ((found = sz_locate_first_v128_(match_at_32_vec, haystack + 32))) return found;
            if ((found = sz_locate_first_v128_(match_at_48_vec, haystack + 48))) return found;
        }
        haystack += 64, haystack_length -= 64;
    }

    for (; haystack_length >= 16; haystack += 16, haystack_length -= 16) {
        v128_t match_vec = sz_find_byteset_match_v128_(wasm_v128_load(haystack), set_top_vec, set_bottom_vec);
        sz_cptr_t found = sz_locate_first_v128_(match_vec, haystack);
        if (found) return found;
    }

    return sz_find_byteset_serial(haystack, haystack_length, set);
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_v128(sz_cptr_t haystack, sz_size_t haystack_length,
                                                sz_byteset_t const *set) {
    v128_t set_top_vec = wasm_v128_load(&set->_u8s[0]);
    v128_t set_bottom_vec = wasm_v128_load(&set->_u8s[16]);

    while (haystack_length >= 64) {
        sz_cptr_t window_start = haystack + haystack_length - 64;
        v128_t match_at_0_vec = sz_find_byteset_match_v128_(wasm_v128_load(window_start + 0), set_top_vec,
                                                            set_bottom_vec);
        v128_t match_at_16_vec = sz_find_byteset_match_v128_(wasm_v128_load(window_start + 16), set_top_vec,
                                                             set_bottom_vec);
        v128_t match_at_32_vec = sz_find_byteset_match_v128_(wasm_v128_load(window_start + 32), set_top_vec,
                                                             set_bottom_vec);
        v128_t match_at_48_vec = sz_find_byteset_match_v128_(wasm_v128_load(window_start + 48), set_top_vec,
                                                             set_bottom_vec);
        v128_t any_match_vec = wasm_v128_or(wasm_v128_or(match_at_0_vec, match_at_16_vec),
                                            wasm_v128_or(match_at_32_vec, match_at_48_vec));
        if (wasm_v128_any_true(any_match_vec)) {
            sz_cptr_t found;
            if ((found = sz_locate_last_v128_(match_at_48_vec, window_start + 48))) return found;
            if ((found = sz_locate_last_v128_(match_at_32_vec, window_start + 32))) return found;
            if ((found = sz_locate_last_v128_(match_at_16_vec, window_start + 16))) return found;
            if ((found = sz_locate_last_v128_(match_at_0_vec, window_start + 0))) return found;
        }
        haystack_length -= 64;
    }

    for (; haystack_length >= 16; haystack_length -= 16) {
        sz_cptr_t window_start = haystack + haystack_length - 16;
        v128_t match_vec = sz_find_byteset_match_v128_(wasm_v128_load(window_start), set_top_vec, set_bottom_vec);
        sz_cptr_t found = sz_locate_last_v128_(match_vec, window_start);
        if (found) return found;
    }

    return sz_rfind_byteset_serial(haystack, haystack_length, set);
}

/**
 *  @brief Candidate mask for a substring window: lanes where the first/mid/last needle bytes all line up.
 *  @param haystack_start Base pointer for the 16-byte window.
 *  @param offset_first Offset of the first anomaly byte in the needle.
 *  @param offset_mid Offset of the mid anomaly byte in the needle.
 *  @param offset_last Offset of the last anomaly byte in the needle.
 *  @param needle_first_vec Broadcasted first-anomaly needle byte.
 *  @param needle_mid_vec Broadcasted mid-anomaly needle byte.
 *  @param needle_last_vec Broadcasted last-anomaly needle byte.
 *  @return 0xFF per lane where all three needle bytes match.
 */
SZ_HELPER_INLINE v128_t sz_find_substr_match_v128_(                                                //
    sz_cptr_t haystack_start, sz_size_t offset_first, sz_size_t offset_mid, sz_size_t offset_last, //
    v128_t needle_first_vec, v128_t needle_mid_vec, v128_t needle_last_vec) {
    return wasm_v128_and(                                                                   //
        wasm_v128_and(                                                                      //
            wasm_i8x16_eq(wasm_v128_load(haystack_start + offset_first), needle_first_vec), //
            wasm_i8x16_eq(wasm_v128_load(haystack_start + offset_mid), needle_mid_vec)),    //
        wasm_i8x16_eq(wasm_v128_load(haystack_start + offset_last), needle_last_vec));
}

/**
 *  @brief Walk a window's candidates low-to-high, returning the first that verifies via `sz_equal`.
 *  @param match_vec The 16-byte candidate mask.
 *  @param window_start Base pointer for this 16-byte window.
 *  @param needle The full needle to verify against.
 *  @param needle_length Length of `needle` in bytes.
 *  @return Pointer to the first verified match, or SZ_NULL_CHAR if none.
 */
SZ_HELPER_AUTO sz_cptr_t sz_locate_substr_first_v128_( //
    v128_t match_vec, sz_cptr_t window_start, sz_cptr_t needle, sz_size_t needle_length) {
    sz_u32_t matches = (sz_u32_t)wasm_i8x16_bitmask(match_vec);
    while (matches) {
        int candidate_offset = sz_u32_ctz(matches);
        if (sz_equal_v128(window_start + candidate_offset, needle, needle_length))
            return window_start + candidate_offset;
        matches &= matches - 1; // clear the lowest set bit
    }
    return SZ_NULL_CHAR;
}

/**
 *  @brief Walk a window's candidates high-to-low, returning the last that verifies via `sz_equal`.
 *  @param match_vec The 16-byte candidate mask.
 *  @param window_start Base pointer for this 16-byte window.
 *  @param needle The full needle to verify against.
 *  @param needle_length Length of `needle` in bytes.
 *  @return Pointer to the last verified match, or SZ_NULL_CHAR if none.
 */
SZ_HELPER_AUTO sz_cptr_t sz_locate_substr_last_v128_( //
    v128_t match_vec, sz_cptr_t window_start, sz_cptr_t needle, sz_size_t needle_length) {
    sz_u32_t matches = (sz_u32_t)wasm_i8x16_bitmask(match_vec);
    while (matches) {
        int candidate_offset = 31 - sz_u32_clz(matches);
        if (sz_equal_v128(window_start + candidate_offset, needle, needle_length))
            return window_start + candidate_offset;
        matches &= ~((sz_u32_t)1 << candidate_offset); // clear the highest set bit
    }
    return SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_cptr_t sz_find_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                       sz_size_t needle_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (haystack_length < needle_length || !needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_find_byte_v128(haystack, haystack_length, needle);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);

    v128_t needle_first_vec = wasm_i8x16_splat(*(sz_i8_t const *)&needle[offset_first]);
    v128_t needle_mid_vec = wasm_i8x16_splat(*(sz_i8_t const *)&needle[offset_mid]);
    v128_t needle_last_vec = wasm_i8x16_splat(*(sz_i8_t const *)&needle[offset_last]);

    // Evaluate four 16-byte windows per iteration; only descend into the candidate-verify loop when
    // at least one of them flagged a possible match.
    while (haystack_length >= needle_length + 64) {
        v128_t match_at_0_vec = sz_find_substr_match_v128_(haystack + 0, offset_first, offset_mid, offset_last,
                                                           needle_first_vec, needle_mid_vec, needle_last_vec);
        v128_t match_at_16_vec = sz_find_substr_match_v128_(haystack + 16, offset_first, offset_mid, offset_last,
                                                            needle_first_vec, needle_mid_vec, needle_last_vec);
        v128_t match_at_32_vec = sz_find_substr_match_v128_(haystack + 32, offset_first, offset_mid, offset_last,
                                                            needle_first_vec, needle_mid_vec, needle_last_vec);
        v128_t match_at_48_vec = sz_find_substr_match_v128_(haystack + 48, offset_first, offset_mid, offset_last,
                                                            needle_first_vec, needle_mid_vec, needle_last_vec);
        v128_t any_match_vec = wasm_v128_or(wasm_v128_or(match_at_0_vec, match_at_16_vec),
                                            wasm_v128_or(match_at_32_vec, match_at_48_vec));
        if (wasm_v128_any_true(any_match_vec)) {
            sz_cptr_t found;
            if ((found = sz_locate_substr_first_v128_(match_at_0_vec, haystack + 0, needle, needle_length)))
                return found;
            if ((found = sz_locate_substr_first_v128_(match_at_16_vec, haystack + 16, needle, needle_length)))
                return found;
            if ((found = sz_locate_substr_first_v128_(match_at_32_vec, haystack + 32, needle, needle_length)))
                return found;
            if ((found = sz_locate_substr_first_v128_(match_at_48_vec, haystack + 48, needle, needle_length)))
                return found;
        }
        haystack += 64, haystack_length -= 64;
    }

    for (; haystack_length >= needle_length + 16; haystack += 16, haystack_length -= 16) {
        v128_t match_vec = sz_find_substr_match_v128_( //
            haystack, offset_first, offset_mid, offset_last, needle_first_vec, needle_mid_vec, needle_last_vec);
        sz_cptr_t found = sz_locate_substr_first_v128_(match_vec, haystack, needle, needle_length);
        if (found) return found;
    }

    return sz_find_serial(haystack, haystack_length, needle, needle_length);
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                        sz_size_t needle_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (haystack_length < needle_length || !needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_rfind_byte_v128(haystack, haystack_length, needle);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);

    v128_t needle_first_vec = wasm_i8x16_splat(*(sz_i8_t const *)&needle[offset_first]);
    v128_t needle_mid_vec = wasm_i8x16_splat(*(sz_i8_t const *)&needle[offset_mid]);
    v128_t needle_last_vec = wasm_i8x16_splat(*(sz_i8_t const *)&needle[offset_last]);

    // For a window whose anchors are loaded at `window_start`, lane `k` corresponds to a candidate
    // needle start at `window_start + k`. Scanning the four sub-windows from the highest base down,
    // each from its highest set bit, walks candidates latest-first as `rfind` requires.
    while (haystack_length >= needle_length + 64) {
        sz_cptr_t window_start = haystack + haystack_length - needle_length - 15; // highest sub-window's load base
        v128_t match_at_0_vec = sz_find_substr_match_v128_(window_start - 0, offset_first, offset_mid, offset_last,
                                                           needle_first_vec, needle_mid_vec, needle_last_vec);
        v128_t match_at_16_vec = sz_find_substr_match_v128_(window_start - 16, offset_first, offset_mid, offset_last,
                                                            needle_first_vec, needle_mid_vec, needle_last_vec);
        v128_t match_at_32_vec = sz_find_substr_match_v128_(window_start - 32, offset_first, offset_mid, offset_last,
                                                            needle_first_vec, needle_mid_vec, needle_last_vec);
        v128_t match_at_48_vec = sz_find_substr_match_v128_(window_start - 48, offset_first, offset_mid, offset_last,
                                                            needle_first_vec, needle_mid_vec, needle_last_vec);
        v128_t any_match_vec = wasm_v128_or(wasm_v128_or(match_at_0_vec, match_at_16_vec),
                                            wasm_v128_or(match_at_32_vec, match_at_48_vec));
        if (wasm_v128_any_true(any_match_vec)) {
            sz_cptr_t found;
            if ((found = sz_locate_substr_last_v128_(match_at_0_vec, window_start - 0, needle, needle_length)))
                return found;
            if ((found = sz_locate_substr_last_v128_(match_at_16_vec, window_start - 16, needle, needle_length)))
                return found;
            if ((found = sz_locate_substr_last_v128_(match_at_32_vec, window_start - 32, needle, needle_length)))
                return found;
            if ((found = sz_locate_substr_last_v128_(match_at_48_vec, window_start - 48, needle, needle_length)))
                return found;
        }
        haystack_length -= 64;
    }

    for (; haystack_length >= needle_length + 16; haystack_length -= 16) {
        sz_cptr_t window_start = haystack + haystack_length - needle_length - 15;
        v128_t match_vec = sz_find_substr_match_v128_( //
            window_start, offset_first, offset_mid, offset_last, needle_first_vec, needle_mid_vec, needle_last_vec);
        sz_cptr_t found = sz_locate_substr_last_v128_(match_vec, window_start, needle, needle_length);
        if (found) return found;
    }

    return sz_rfind_serial(haystack, haystack_length, needle, needle_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_V128_H_
