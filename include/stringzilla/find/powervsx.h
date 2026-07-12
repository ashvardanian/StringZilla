/**
 *  @brief IBM Power VSX backend for find.
 *  @file include/stringzilla/find/powervsx.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_POWERVSX_H_
#define STRINGZILLA_FIND_POWERVSX_H_

#include "stringzilla/types.h"
#include "stringzilla/find/serial.h"
#include "stringzilla/compare.h" // `sz_equal`

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

/**
 *  @brief x86-`movemask`-equivalent for VSX: gathers the MSB of each of the 16 bytes into a
 *         16-bit mask where bit `i` reflects byte `i` (lowest-addressed byte -> bit 0).
 *
 *  Implemented with `vec_vbpermq`, which bit-gathers from a source vector using the supplied
 *  big-endian bit indices. On little-endian targets the gathered bits land in element `[1]`.
 *  The indices `(15 - i) * 8` select the MSB (bit 7) of byte `i`, yielding an SSE-like ordering.
 *
 *  @param cmp_u8x16 A comparison result vector (0xFF where matched, 0x00 otherwise).
 *  @return 64-bit value with low 16 bits forming the movemask.
 */
SZ_HELPER_INLINE sz_u64_t sz_movemask_powervsx_(__vector unsigned char cmp_u8x16) {
    __vector unsigned char const indices = {120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0};
    __vector unsigned long long const gathered = vec_vbpermq(cmp_u8x16, indices);
#if SZ_IS_BIG_ENDIAN_
    return (sz_u64_t)gathered[0] & 0xFFFFull;
#else
    return (sz_u64_t)gathered[1] & 0xFFFFull;
#endif
}

SZ_API_COMPTIME sz_cptr_t sz_find_byte_powervsx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    __vector unsigned char needle_u8x16 = vec_splats(*(unsigned char const *)needle);
    while (haystack_length >= 16) {
        __vector unsigned char haystack_u8x16 = vec_xl(0, (unsigned char const *)haystack);
        // On Power10 `vec_first_match_index` returns the (little-endian) index of the first equal
        // byte directly — 16 when none match — fusing the compare, the bit-gather movemask, and the
        // trailing-zero count into one instruction. The Power9 fallback keeps the movemask + ctz path.
#if defined(_ARCH_PWR10)
        unsigned int match_index = vec_first_match_index(haystack_u8x16, needle_u8x16);
        if (match_index != 16) return haystack + match_index;
#else
        __vector unsigned char cmp_u8x16 = (__vector unsigned char)vec_cmpeq(haystack_u8x16, needle_u8x16);
        sz_u64_t matches = sz_movemask_powervsx_(cmp_u8x16);
        if (matches) return haystack + sz_u64_ctz(matches);
#endif
        haystack += 16, haystack_length -= 16;
    }
    return sz_find_byte_serial(haystack, haystack_length, needle);
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_powervsx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    __vector unsigned char needle_u8x16 = vec_splats(*(unsigned char const *)needle);
    while (haystack_length >= 16) {
        __vector unsigned char haystack_u8x16 = vec_xl(0, (unsigned char const *)(haystack + haystack_length - 16));
        // On Power10 we mirror the forward kernel's `vec_first_match_index`: reversing the window
        // turns the last-match query into a first-match one, fusing compare + movemask + count into
        // one instruction (after the byte-reverse permute). The Power9 fallback keeps movemask + clz.
#if defined(_ARCH_PWR10)
        __vector unsigned char reversed_u8x16 = vec_reve(haystack_u8x16);
        unsigned int match_index = vec_first_match_index(reversed_u8x16, needle_u8x16);
        if (match_index != 16) return haystack + haystack_length - 1 - match_index;
#else
        __vector unsigned char cmp_u8x16 = (__vector unsigned char)vec_cmpeq(haystack_u8x16, needle_u8x16);
        sz_u64_t matches = sz_movemask_powervsx_(cmp_u8x16);
        // The mask occupies the low 16 bits; the highest set bit is the last match in the window.
        if (matches) return haystack + haystack_length - 1 - (sz_u64_clz(matches) - 48);
#endif
        haystack_length -= 16;
    }
    return sz_rfind_byte_serial(haystack, haystack_length, needle);
}

SZ_API_COMPTIME sz_cptr_t sz_find_powervsx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                           sz_size_t needle_length) {

    // Empty needle matches at the start, like `strstr`.
    if (!needle_length) return haystack;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_find_byte_powervsx(haystack, haystack_length, needle);

    // Pick the parts of the needle that are worth comparing (first, middle, last).
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);

    __vector unsigned char needle_first_u8x16 = vec_splats(*(unsigned char const *)&needle[offset_first]);
    __vector unsigned char needle_mid_u8x16 = vec_splats(*(unsigned char const *)&needle[offset_mid]);
    __vector unsigned char needle_last_u8x16 = vec_splats(*(unsigned char const *)&needle[offset_last]);

    __vector unsigned char const zero_u8x16 = vec_splats((unsigned char)0);

    // Unroll two 16-byte blocks per iteration. A single OR-reduced gate skips both blocks at once
    // on the common no-match path, halving the movemask + ctz bookkeeping over dense haystacks.
    while (haystack_length >= needle_length + 32) {
        __vector unsigned char comparison_low_u8x16 = vec_and( //
            vec_and((__vector unsigned char)vec_cmpeq(vec_xl(0, (unsigned char const *)(haystack + offset_first)),
                                                      needle_first_u8x16),
                    (__vector unsigned char)vec_cmpeq(vec_xl(0, (unsigned char const *)(haystack + offset_mid)),
                                                      needle_mid_u8x16)),
            (__vector unsigned char)vec_cmpeq(vec_xl(0, (unsigned char const *)(haystack + offset_last)),
                                              needle_last_u8x16));
        __vector unsigned char comparison_high_u8x16 = vec_and( //
            vec_and((__vector unsigned char)vec_cmpeq(vec_xl(0, (unsigned char const *)(haystack + 16 + offset_first)),
                                                      needle_first_u8x16),
                    (__vector unsigned char)vec_cmpeq(vec_xl(0, (unsigned char const *)(haystack + 16 + offset_mid)),
                                                      needle_mid_u8x16)),
            (__vector unsigned char)vec_cmpeq(vec_xl(0, (unsigned char const *)(haystack + 16 + offset_last)),
                                              needle_last_u8x16));
        // Cheap "any match across either block?" gate before locating exact offsets.
        if (!vec_all_eq(vec_or(comparison_low_u8x16, comparison_high_u8x16), zero_u8x16)) {
            sz_u64_t matches = sz_movemask_powervsx_(comparison_low_u8x16);
            while (matches) {
                int potential_offset = sz_u64_ctz(matches);
                if (sz_equal_powervsx(haystack + potential_offset, needle, needle_length))
                    return haystack + potential_offset;
                matches &= matches - 1;
            }
            matches = sz_movemask_powervsx_(comparison_high_u8x16);
            while (matches) {
                int potential_offset = sz_u64_ctz(matches);
                if (sz_equal_powervsx(haystack + 16 + potential_offset, needle, needle_length))
                    return haystack + 16 + potential_offset;
                matches &= matches - 1;
            }
        }
        haystack += 32, haystack_length -= 32;
    }

    for (; haystack_length >= needle_length + 16; haystack += 16, haystack_length -= 16) {
        __vector unsigned char haystack_first_u8x16 = vec_xl(0, (unsigned char const *)(haystack + offset_first));
        __vector unsigned char haystack_mid_u8x16 = vec_xl(0, (unsigned char const *)(haystack + offset_mid));
        __vector unsigned char haystack_last_u8x16 = vec_xl(0, (unsigned char const *)(haystack + offset_last));
        __vector unsigned char cmp_u8x16 = vec_and( //
            vec_and((__vector unsigned char)vec_cmpeq(haystack_first_u8x16, needle_first_u8x16),
                    (__vector unsigned char)vec_cmpeq(haystack_mid_u8x16, needle_mid_u8x16)),
            (__vector unsigned char)vec_cmpeq(haystack_last_u8x16, needle_last_u8x16));
        sz_u64_t matches = sz_movemask_powervsx_(cmp_u8x16);
        while (matches) {
            int potential_offset = sz_u64_ctz(matches);
            if (sz_equal_powervsx(haystack + potential_offset, needle, needle_length))
                return haystack + potential_offset;
            matches &= matches - 1;
        }
    }

    return sz_find_serial(haystack, haystack_length, needle, needle_length);
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_powervsx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                            sz_size_t needle_length) {

    // Empty needle matches at the end.
    if (!needle_length) return haystack + haystack_length;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_rfind_byte_powervsx(haystack, haystack_length, needle);

    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);

    __vector unsigned char needle_first_u8x16 = vec_splats(*(unsigned char const *)&needle[offset_first]);
    __vector unsigned char needle_mid_u8x16 = vec_splats(*(unsigned char const *)&needle[offset_mid]);
    __vector unsigned char needle_last_u8x16 = vec_splats(*(unsigned char const *)&needle[offset_last]);

    __vector unsigned char const zero_u8x16 = vec_splats((unsigned char)0);

    // Unroll two 16-byte windows per iteration, scanning the higher (right-most) window first so the
    // first match found is still the right-most. One OR-reduced gate skips both windows at once.
    while (haystack_length >= needle_length + 32) {
        sz_cptr_t cursor_hi = haystack + haystack_length - needle_length - 16 + 1;
        sz_cptr_t cursor_lo = cursor_hi - 16;
        __vector unsigned char comparison_high_u8x16 = vec_and( //
            vec_and((__vector unsigned char)vec_cmpeq(vec_xl(0, (unsigned char const *)(cursor_hi + offset_first)),
                                                      needle_first_u8x16),
                    (__vector unsigned char)vec_cmpeq(vec_xl(0, (unsigned char const *)(cursor_hi + offset_mid)),
                                                      needle_mid_u8x16)),
            (__vector unsigned char)vec_cmpeq(vec_xl(0, (unsigned char const *)(cursor_hi + offset_last)),
                                              needle_last_u8x16));
        __vector unsigned char comparison_low_u8x16 = vec_and( //
            vec_and((__vector unsigned char)vec_cmpeq(vec_xl(0, (unsigned char const *)(cursor_lo + offset_first)),
                                                      needle_first_u8x16),
                    (__vector unsigned char)vec_cmpeq(vec_xl(0, (unsigned char const *)(cursor_lo + offset_mid)),
                                                      needle_mid_u8x16)),
            (__vector unsigned char)vec_cmpeq(vec_xl(0, (unsigned char const *)(cursor_lo + offset_last)),
                                              needle_last_u8x16));
        if (!vec_all_eq(vec_or(comparison_high_u8x16, comparison_low_u8x16), zero_u8x16)) {
            sz_u64_t matches = sz_movemask_powervsx_(comparison_high_u8x16);
            while (matches) {
                int potential_offset = (int)(sz_u64_clz(matches) - 48);
                if (sz_equal_powervsx(haystack + haystack_length - needle_length - potential_offset, needle,
                                      needle_length))
                    return haystack + haystack_length - needle_length - potential_offset;
                matches &= ~(1ull << (15 - potential_offset));
            }
            matches = sz_movemask_powervsx_(comparison_low_u8x16);
            while (matches) {
                int potential_offset = (int)(sz_u64_clz(matches) - 48) + 16;
                if (sz_equal_powervsx(haystack + haystack_length - needle_length - potential_offset, needle,
                                      needle_length))
                    return haystack + haystack_length - needle_length - potential_offset;
                matches &= ~(1ull << (31 - potential_offset));
            }
        }
        haystack_length -= 32;
    }

    for (; haystack_length >= needle_length + 16; haystack_length -= 16) {
        sz_cptr_t haystack_cursor = haystack + haystack_length - needle_length - 16 + 1;
        __vector unsigned char haystack_first_u8x16 = vec_xl(0,
                                                             (unsigned char const *)(haystack_cursor + offset_first));
        __vector unsigned char haystack_mid_u8x16 = vec_xl(0, (unsigned char const *)(haystack_cursor + offset_mid));
        __vector unsigned char haystack_last_u8x16 = vec_xl(0, (unsigned char const *)(haystack_cursor + offset_last));
        __vector unsigned char cmp_u8x16 = vec_and( //
            vec_and((__vector unsigned char)vec_cmpeq(haystack_first_u8x16, needle_first_u8x16),
                    (__vector unsigned char)vec_cmpeq(haystack_mid_u8x16, needle_mid_u8x16)),
            (__vector unsigned char)vec_cmpeq(haystack_last_u8x16, needle_last_u8x16));
        sz_u64_t matches = sz_movemask_powervsx_(cmp_u8x16);
        while (matches) {
            // Highest set bit within the low 16 bits is the right-most candidate.
            int potential_offset = (int)(sz_u64_clz(matches) - 48);
            if (sz_equal_powervsx(haystack + haystack_length - needle_length - potential_offset, needle, needle_length))
                return haystack + haystack_length - needle_length - potential_offset;
            matches &= ~(1ull << (15 - potential_offset));
        }
    }

    return sz_rfind_serial(haystack, haystack_length, needle, needle_length);
}

SZ_API_COMPTIME sz_cptr_t sz_find_byteset_powervsx(sz_cptr_t haystack, sz_size_t haystack_length,
                                                   sz_byteset_t const *set) {
    // Two halves of the 256-bit set, each 16 bytes, covering byte indices [0,16) and [16,32).
    __vector unsigned char set_low_u8x16 = vec_xl(0, (unsigned char const *)&set->_u8s[0]);
    __vector unsigned char set_high_u8x16 = vec_xl(0, (unsigned char const *)&set->_u8s[16]);

    for (; haystack_length >= 16; haystack += 16, haystack_length -= 16) {
        __vector unsigned char haystack_u8x16 = vec_xl(0, (unsigned char const *)haystack);
        __vector unsigned char byte_index_u8x16 = vec_sr(haystack_u8x16,
                                                         vec_splats((unsigned char)3)); // c >> 3, in [0,32)
        __vector unsigned char bit_position_u8x16 = vec_and(haystack_u8x16, vec_splats((unsigned char)7));
        __vector unsigned char byte_mask_u8x16 = vec_sl(vec_splats((unsigned char)1), bit_position_u8x16);
        // `vec_perm` with two distinct registers indexes the full 32-byte set using the low 5 bits.
        __vector unsigned char looked_up_u8x16 = vec_perm(set_low_u8x16, set_high_u8x16, byte_index_u8x16);
        __vector unsigned char anded_u8x16 = vec_and(looked_up_u8x16, byte_mask_u8x16);
        __vector unsigned char matched_u8x16 = (__vector unsigned char)vec_cmpne(anded_u8x16,
                                                                                 vec_splats((unsigned char)0));
#if defined(_ARCH_PWR10)
        // `matched_u8x16` is 0xFF in matched lanes; the first such lane is the first set member.
        unsigned int match_index = vec_first_match_index(matched_u8x16, vec_splats((unsigned char)0xFF));
        if (match_index != 16) return haystack + match_index;
#else
        sz_u64_t matches = sz_movemask_powervsx_(matched_u8x16);
        if (matches) return haystack + sz_u64_ctz(matches);
#endif
    }

    return sz_find_byteset_serial(haystack, haystack_length, set);
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_powervsx(sz_cptr_t haystack, sz_size_t haystack_length,
                                                    sz_byteset_t const *set) {
    __vector unsigned char set_low_u8x16 = vec_xl(0, (unsigned char const *)&set->_u8s[0]);
    __vector unsigned char set_high_u8x16 = vec_xl(0, (unsigned char const *)&set->_u8s[16]);

    for (; haystack_length >= 16; haystack_length -= 16) {
        __vector unsigned char haystack_u8x16 = vec_xl(0, (unsigned char const *)(haystack + haystack_length - 16));
        __vector unsigned char byte_index_u8x16 = vec_sr(haystack_u8x16, vec_splats((unsigned char)3));
        __vector unsigned char bit_position_u8x16 = vec_and(haystack_u8x16, vec_splats((unsigned char)7));
        __vector unsigned char byte_mask_u8x16 = vec_sl(vec_splats((unsigned char)1), bit_position_u8x16);
        __vector unsigned char looked_up_u8x16 = vec_perm(set_low_u8x16, set_high_u8x16, byte_index_u8x16);
        __vector unsigned char anded_u8x16 = vec_and(looked_up_u8x16, byte_mask_u8x16);
        __vector unsigned char matched_u8x16 = (__vector unsigned char)vec_cmpne(anded_u8x16,
                                                                                 vec_splats((unsigned char)0));
        sz_u64_t matches = sz_movemask_powervsx_(matched_u8x16);
        if (matches) return haystack + haystack_length - 1 - (sz_u64_clz(matches) - 48);
    }

    return sz_rfind_byteset_serial(haystack, haystack_length, set);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_POWERVSX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_POWERVSX_H_
