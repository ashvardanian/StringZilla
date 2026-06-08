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

/** @brief First matching byte in a 16-byte block given its 0xFF/0x00 match mask, or NULL if none. */
SZ_INTERNAL sz_cptr_t sz_locate_first_v128_(v128_t match_vec, sz_cptr_t block_start) {
    sz_u32_t matches = (sz_u32_t)wasm_i8x16_bitmask(match_vec);
    if (!matches) return SZ_NULL_CHAR;
    return block_start + sz_u32_ctz(matches);
}

/** @brief Last matching byte in a 16-byte block given its 0xFF/0x00 match mask, or NULL if none. */
SZ_INTERNAL sz_cptr_t sz_locate_last_v128_(v128_t match_vec, sz_cptr_t block_start) {
    sz_u32_t matches = (sz_u32_t)wasm_i8x16_bitmask(match_vec);
    if (!matches) return SZ_NULL_CHAR;
    // `matches` occupies the low 16 bits, so `31 - clz` is the index of its highest set bit.
    return block_start + (31 - sz_u32_clz(matches));
}

SZ_PUBLIC sz_cptr_t sz_find_byte_v128(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    v128_t needle_vec = wasm_i8x16_splat(*(sz_i8_t const *)n);

    // Scan 64 bytes per iteration: OR four equality masks and gate with a single `any_true`, only
    // descending into the per-block locate when one of the windows actually contains a match.
    while (h_length >= 64) {
        v128_t match_at_0_vec = wasm_i8x16_eq(wasm_v128_load(h + 0), needle_vec);
        v128_t match_at_16_vec = wasm_i8x16_eq(wasm_v128_load(h + 16), needle_vec);
        v128_t match_at_32_vec = wasm_i8x16_eq(wasm_v128_load(h + 32), needle_vec);
        v128_t match_at_48_vec = wasm_i8x16_eq(wasm_v128_load(h + 48), needle_vec);
        v128_t any_match_vec =
            wasm_v128_or(wasm_v128_or(match_at_0_vec, match_at_16_vec), wasm_v128_or(match_at_32_vec, match_at_48_vec));
        if (wasm_v128_any_true(any_match_vec)) {
            sz_cptr_t found;
            if ((found = sz_locate_first_v128_(match_at_0_vec, h + 0))) return found;
            if ((found = sz_locate_first_v128_(match_at_16_vec, h + 16))) return found;
            if ((found = sz_locate_first_v128_(match_at_32_vec, h + 32))) return found;
            if ((found = sz_locate_first_v128_(match_at_48_vec, h + 48))) return found;
        }
        h += 64, h_length -= 64;
    }

    while (h_length >= 16) {
        sz_cptr_t found = sz_locate_first_v128_(wasm_i8x16_eq(wasm_v128_load(h), needle_vec), h);
        if (found) return found;
        h += 16, h_length -= 16;
    }

    return sz_find_byte_serial(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byte_v128(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    v128_t needle_vec = wasm_i8x16_splat(*(sz_i8_t const *)n);

    // Scan the trailing 64 bytes per iteration; on a hit, locate the LAST match by walking the four
    // sub-windows from the highest one down (each block reports its own latest match).
    while (h_length >= 64) {
        sz_cptr_t window_start = h + h_length - 64;
        v128_t match_at_0_vec = wasm_i8x16_eq(wasm_v128_load(window_start + 0), needle_vec);
        v128_t match_at_16_vec = wasm_i8x16_eq(wasm_v128_load(window_start + 16), needle_vec);
        v128_t match_at_32_vec = wasm_i8x16_eq(wasm_v128_load(window_start + 32), needle_vec);
        v128_t match_at_48_vec = wasm_i8x16_eq(wasm_v128_load(window_start + 48), needle_vec);
        v128_t any_match_vec =
            wasm_v128_or(wasm_v128_or(match_at_0_vec, match_at_16_vec), wasm_v128_or(match_at_32_vec, match_at_48_vec));
        if (wasm_v128_any_true(any_match_vec)) {
            sz_cptr_t found;
            if ((found = sz_locate_last_v128_(match_at_48_vec, window_start + 48))) return found;
            if ((found = sz_locate_last_v128_(match_at_32_vec, window_start + 32))) return found;
            if ((found = sz_locate_last_v128_(match_at_16_vec, window_start + 16))) return found;
            if ((found = sz_locate_last_v128_(match_at_0_vec, window_start + 0))) return found;
        }
        h_length -= 64;
    }

    while (h_length >= 16) {
        sz_cptr_t window_start = h + h_length - 16;
        sz_cptr_t found = sz_locate_last_v128_(wasm_i8x16_eq(wasm_v128_load(window_start), needle_vec), window_start);
        if (found) return found;
        h_length -= 16;
    }

    return sz_rfind_byte_serial(h, h_length, n);
}

/** @brief Match mask (0xFF per lane) for bytes present in `set`, split across two 16-byte half-tables. */
SZ_INTERNAL v128_t sz_find_byteset_match_v128_(v128_t h_vec, v128_t set_top_vec, v128_t set_bottom_vec) {
    // Serial equivalent per byte `c`: `(set->_u8s[c >> 3] & (1u << (c & 7u))) != 0`.
    v128_t byte_index_vec = wasm_u8x16_shr(h_vec, 3); // c >> 3, in [0, 31]
    // The bit mask `1 << (c & 7)` is produced via a swizzle into a tiny power-of-two table.
    v128_t bit_table = wasm_i8x16_make(1, 2, 4, 8, 16, 32, 64, (sz_i8_t)128, 0, 0, 0, 0, 0, 0, 0, 0);
    v128_t byte_mask_vec = wasm_i8x16_swizzle(bit_table, wasm_v128_and(h_vec, wasm_i8x16_splat(7)));
    // `swizzle` returns 0 for indices >= 16 (or with the high bit set). For the bottom half we shift
    // the index down by 16: values originally < 16 underflow into [240, 256) and yield zeros there.
    v128_t matches_top_vec = wasm_i8x16_swizzle(set_top_vec, byte_index_vec);
    v128_t matches_bottom_vec =
        wasm_i8x16_swizzle(set_bottom_vec, wasm_i8x16_sub(byte_index_vec, wasm_i8x16_splat(16)));
    v128_t matches_vec = wasm_v128_or(matches_top_vec, matches_bottom_vec);
    // Keep a lane (0xFF) only if its selected bit is set.
    return wasm_i8x16_ne(wasm_v128_and(matches_vec, byte_mask_vec), wasm_i8x16_splat(0));
}

SZ_PUBLIC sz_cptr_t sz_find_byteset_v128(sz_cptr_t h, sz_size_t h_length, sz_byteset_t const *set) {
    v128_t set_top_vec = wasm_v128_load(&set->_u8s[0]);
    v128_t set_bottom_vec = wasm_v128_load(&set->_u8s[16]);

    while (h_length >= 64) {
        v128_t match_at_0_vec = sz_find_byteset_match_v128_(wasm_v128_load(h + 0), set_top_vec, set_bottom_vec);
        v128_t match_at_16_vec = sz_find_byteset_match_v128_(wasm_v128_load(h + 16), set_top_vec, set_bottom_vec);
        v128_t match_at_32_vec = sz_find_byteset_match_v128_(wasm_v128_load(h + 32), set_top_vec, set_bottom_vec);
        v128_t match_at_48_vec = sz_find_byteset_match_v128_(wasm_v128_load(h + 48), set_top_vec, set_bottom_vec);
        v128_t any_match_vec =
            wasm_v128_or(wasm_v128_or(match_at_0_vec, match_at_16_vec), wasm_v128_or(match_at_32_vec, match_at_48_vec));
        if (wasm_v128_any_true(any_match_vec)) {
            sz_cptr_t found;
            if ((found = sz_locate_first_v128_(match_at_0_vec, h + 0))) return found;
            if ((found = sz_locate_first_v128_(match_at_16_vec, h + 16))) return found;
            if ((found = sz_locate_first_v128_(match_at_32_vec, h + 32))) return found;
            if ((found = sz_locate_first_v128_(match_at_48_vec, h + 48))) return found;
        }
        h += 64, h_length -= 64;
    }

    for (; h_length >= 16; h += 16, h_length -= 16) {
        v128_t match_vec = sz_find_byteset_match_v128_(wasm_v128_load(h), set_top_vec, set_bottom_vec);
        sz_cptr_t found = sz_locate_first_v128_(match_vec, h);
        if (found) return found;
    }

    return sz_find_byteset_serial(h, h_length, set);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byteset_v128(sz_cptr_t h, sz_size_t h_length, sz_byteset_t const *set) {
    v128_t set_top_vec = wasm_v128_load(&set->_u8s[0]);
    v128_t set_bottom_vec = wasm_v128_load(&set->_u8s[16]);

    while (h_length >= 64) {
        sz_cptr_t window_start = h + h_length - 64;
        v128_t match_at_0_vec = sz_find_byteset_match_v128_(wasm_v128_load(window_start + 0), set_top_vec, set_bottom_vec);
        v128_t match_at_16_vec = sz_find_byteset_match_v128_(wasm_v128_load(window_start + 16), set_top_vec, set_bottom_vec);
        v128_t match_at_32_vec = sz_find_byteset_match_v128_(wasm_v128_load(window_start + 32), set_top_vec, set_bottom_vec);
        v128_t match_at_48_vec = sz_find_byteset_match_v128_(wasm_v128_load(window_start + 48), set_top_vec, set_bottom_vec);
        v128_t any_match_vec =
            wasm_v128_or(wasm_v128_or(match_at_0_vec, match_at_16_vec), wasm_v128_or(match_at_32_vec, match_at_48_vec));
        if (wasm_v128_any_true(any_match_vec)) {
            sz_cptr_t found;
            if ((found = sz_locate_last_v128_(match_at_48_vec, window_start + 48))) return found;
            if ((found = sz_locate_last_v128_(match_at_32_vec, window_start + 32))) return found;
            if ((found = sz_locate_last_v128_(match_at_16_vec, window_start + 16))) return found;
            if ((found = sz_locate_last_v128_(match_at_0_vec, window_start + 0))) return found;
        }
        h_length -= 64;
    }

    for (; h_length >= 16; h_length -= 16) {
        sz_cptr_t window_start = h + h_length - 16;
        v128_t match_vec = sz_find_byteset_match_v128_(wasm_v128_load(window_start), set_top_vec, set_bottom_vec);
        sz_cptr_t found = sz_locate_last_v128_(match_vec, window_start);
        if (found) return found;
    }

    return sz_rfind_byteset_serial(h, h_length, set);
}

/** @brief Candidate mask for a substring window: lanes where the first/mid/last needle bytes all line up. */
SZ_INTERNAL v128_t sz_find_substr_match_v128_(                                            //
    sz_cptr_t h, sz_size_t offset_first, sz_size_t offset_mid, sz_size_t offset_last,     //
    v128_t needle_first_vec, v128_t needle_mid_vec, v128_t needle_last_vec) {
    return wasm_v128_and(                                                      //
        wasm_v128_and(                                                         //
            wasm_i8x16_eq(wasm_v128_load(h + offset_first), needle_first_vec), //
            wasm_i8x16_eq(wasm_v128_load(h + offset_mid), needle_mid_vec)),    //
        wasm_i8x16_eq(wasm_v128_load(h + offset_last), needle_last_vec));
}

/** @brief Walk a window's candidates low-to-high, returning the first that verifies via `sz_equal`. */
SZ_INTERNAL sz_cptr_t sz_locate_substr_first_v128_( //
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

/** @brief Walk a window's candidates high-to-low, returning the first that verifies via `sz_equal`. */
SZ_INTERNAL sz_cptr_t sz_locate_substr_last_v128_( //
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

SZ_PUBLIC sz_cptr_t sz_find_v128(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_find_byte_v128(h, h_length, n);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    v128_t needle_first_vec = wasm_i8x16_splat(*(sz_i8_t const *)&n[offset_first]);
    v128_t needle_mid_vec = wasm_i8x16_splat(*(sz_i8_t const *)&n[offset_mid]);
    v128_t needle_last_vec = wasm_i8x16_splat(*(sz_i8_t const *)&n[offset_last]);

    // Evaluate four 16-byte windows per iteration; only descend into the candidate-verify loop when
    // at least one of them flagged a possible match.
    while (h_length >= n_length + 64) {
        v128_t match_at_0_vec =
            sz_find_substr_match_v128_(h + 0, offset_first, offset_mid, offset_last, needle_first_vec, needle_mid_vec, needle_last_vec);
        v128_t match_at_16_vec =
            sz_find_substr_match_v128_(h + 16, offset_first, offset_mid, offset_last, needle_first_vec, needle_mid_vec, needle_last_vec);
        v128_t match_at_32_vec =
            sz_find_substr_match_v128_(h + 32, offset_first, offset_mid, offset_last, needle_first_vec, needle_mid_vec, needle_last_vec);
        v128_t match_at_48_vec =
            sz_find_substr_match_v128_(h + 48, offset_first, offset_mid, offset_last, needle_first_vec, needle_mid_vec, needle_last_vec);
        v128_t any_match_vec =
            wasm_v128_or(wasm_v128_or(match_at_0_vec, match_at_16_vec), wasm_v128_or(match_at_32_vec, match_at_48_vec));
        if (wasm_v128_any_true(any_match_vec)) {
            sz_cptr_t found;
            if ((found = sz_locate_substr_first_v128_(match_at_0_vec, h + 0, n, n_length))) return found;
            if ((found = sz_locate_substr_first_v128_(match_at_16_vec, h + 16, n, n_length))) return found;
            if ((found = sz_locate_substr_first_v128_(match_at_32_vec, h + 32, n, n_length))) return found;
            if ((found = sz_locate_substr_first_v128_(match_at_48_vec, h + 48, n, n_length))) return found;
        }
        h += 64, h_length -= 64;
    }

    for (; h_length >= n_length + 16; h += 16, h_length -= 16) {
        v128_t match_vec = sz_find_substr_match_v128_( //
            h, offset_first, offset_mid, offset_last, needle_first_vec, needle_mid_vec, needle_last_vec);
        sz_cptr_t found = sz_locate_substr_first_v128_(match_vec, h, n, n_length);
        if (found) return found;
    }

    return sz_find_serial(h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_rfind_v128(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_rfind_byte_v128(h, h_length, n);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    v128_t needle_first_vec = wasm_i8x16_splat(*(sz_i8_t const *)&n[offset_first]);
    v128_t needle_mid_vec = wasm_i8x16_splat(*(sz_i8_t const *)&n[offset_mid]);
    v128_t needle_last_vec = wasm_i8x16_splat(*(sz_i8_t const *)&n[offset_last]);

    // For a window whose anchors are loaded at `window_start`, lane `k` corresponds to a candidate
    // needle start at `window_start + k`. Scanning the four sub-windows from the highest base down,
    // each from its highest set bit, walks candidates latest-first as `rfind` requires.
    while (h_length >= n_length + 64) {
        sz_cptr_t window_start = h + h_length - n_length - 15; // highest sub-window's load base
        v128_t match_at_0_vec = sz_find_substr_match_v128_(
            window_start - 0, offset_first, offset_mid, offset_last, needle_first_vec, needle_mid_vec, needle_last_vec);
        v128_t match_at_16_vec = sz_find_substr_match_v128_(
            window_start - 16, offset_first, offset_mid, offset_last, needle_first_vec, needle_mid_vec, needle_last_vec);
        v128_t match_at_32_vec = sz_find_substr_match_v128_(
            window_start - 32, offset_first, offset_mid, offset_last, needle_first_vec, needle_mid_vec, needle_last_vec);
        v128_t match_at_48_vec = sz_find_substr_match_v128_(
            window_start - 48, offset_first, offset_mid, offset_last, needle_first_vec, needle_mid_vec, needle_last_vec);
        v128_t any_match_vec =
            wasm_v128_or(wasm_v128_or(match_at_0_vec, match_at_16_vec), wasm_v128_or(match_at_32_vec, match_at_48_vec));
        if (wasm_v128_any_true(any_match_vec)) {
            sz_cptr_t found;
            if ((found = sz_locate_substr_last_v128_(match_at_0_vec, window_start - 0, n, n_length))) return found;
            if ((found = sz_locate_substr_last_v128_(match_at_16_vec, window_start - 16, n, n_length))) return found;
            if ((found = sz_locate_substr_last_v128_(match_at_32_vec, window_start - 32, n, n_length))) return found;
            if ((found = sz_locate_substr_last_v128_(match_at_48_vec, window_start - 48, n, n_length))) return found;
        }
        h_length -= 64;
    }

    for (; h_length >= n_length + 16; h_length -= 16) {
        sz_cptr_t window_start = h + h_length - n_length - 15;
        v128_t match_vec = sz_find_substr_match_v128_( //
            window_start, offset_first, offset_mid, offset_last, needle_first_vec, needle_mid_vec, needle_last_vec);
        sz_cptr_t found = sz_locate_substr_last_v128_(match_vec, window_start, n, n_length);
        if (found) return found;
    }

    return sz_rfind_serial(h, h_length, n, n_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_V128_H_
