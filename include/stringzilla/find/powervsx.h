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
 *         16-bit mask where bit `i` reflects byte `i` (lowest-addressed byte → bit 0).
 *
 *  Implemented with `vec_vbpermq`, which bit-gathers from a source vector using the supplied
 *  big-endian bit indices. On little-endian targets the gathered bits land in element `[1]`.
 *  The indices `(15 - i) * 8` select the MSB (bit 7) of byte `i`, yielding an SSE-like ordering.
 *  @param cmp  A comparison result vector (0xFF where matched, 0x00 otherwise).
 */
SZ_INTERNAL sz_u64_t sz_movemask_powervsx_(__vector unsigned char cmp) {
    __vector unsigned char const indices = {120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0};
    __vector unsigned long long const gathered = vec_vbpermq(cmp, indices);
#if SZ_IS_BIG_ENDIAN_
    return (sz_u64_t)gathered[0] & 0xFFFFull;
#else
    return (sz_u64_t)gathered[1] & 0xFFFFull;
#endif
}

SZ_PUBLIC sz_cptr_t sz_find_byte_powervsx(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    __vector unsigned char n_vec = vec_splats(*(unsigned char const *)n);
    while (h_length >= 16) {
        __vector unsigned char h_vec = vec_xl(0, (unsigned char const *)h);
        // On Power10 `vec_first_match_index` returns the (little-endian) index of the first equal
        // byte directly — 16 when none match — fusing the compare, the bit-gather movemask, and the
        // trailing-zero count into one instruction. The Power9 fallback keeps the movemask + ctz path.
#if defined(_ARCH_PWR10)
        unsigned int idx = vec_first_match_index(h_vec, n_vec);
        if (idx != 16) return h + idx;
#else
        __vector unsigned char cmp = (__vector unsigned char)vec_cmpeq(h_vec, n_vec);
        sz_u64_t matches = sz_movemask_powervsx_(cmp);
        if (matches) return h + sz_u64_ctz(matches);
#endif
        h += 16, h_length -= 16;
    }
    return sz_find_byte_serial(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byte_powervsx(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    __vector unsigned char n_vec = vec_splats(*(unsigned char const *)n);
    while (h_length >= 16) {
        __vector unsigned char h_vec = vec_xl(0, (unsigned char const *)(h + h_length - 16));
        __vector unsigned char cmp = (__vector unsigned char)vec_cmpeq(h_vec, n_vec);
        sz_u64_t matches = sz_movemask_powervsx_(cmp);
        // The mask occupies the low 16 bits; the highest set bit is the last match in the window.
        if (matches) return h + h_length - 1 - (sz_u64_clz(matches) - 48);
        h_length -= 16;
    }
    return sz_rfind_byte_serial(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_find_powervsx(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_find_byte_powervsx(h, h_length, n);

    // Pick the parts of the needle that are worth comparing (first, middle, last).
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    __vector unsigned char n_first_vec = vec_splats(*(unsigned char const *)&n[offset_first]);
    __vector unsigned char n_mid_vec = vec_splats(*(unsigned char const *)&n[offset_mid]);
    __vector unsigned char n_last_vec = vec_splats(*(unsigned char const *)&n[offset_last]);

    for (; h_length >= n_length + 16; h += 16, h_length -= 16) {
        __vector unsigned char h_first_vec = vec_xl(0, (unsigned char const *)(h + offset_first));
        __vector unsigned char h_mid_vec = vec_xl(0, (unsigned char const *)(h + offset_mid));
        __vector unsigned char h_last_vec = vec_xl(0, (unsigned char const *)(h + offset_last));
        __vector unsigned char cmp = vec_and( //
            vec_and((__vector unsigned char)vec_cmpeq(h_first_vec, n_first_vec),
                    (__vector unsigned char)vec_cmpeq(h_mid_vec, n_mid_vec)),
            (__vector unsigned char)vec_cmpeq(h_last_vec, n_last_vec));
        sz_u64_t matches = sz_movemask_powervsx_(cmp);
        while (matches) {
            int potential_offset = sz_u64_ctz(matches);
            if (sz_equal_powervsx(h + potential_offset, n, n_length)) return h + potential_offset;
            matches &= matches - 1;
        }
    }

    return sz_find_serial(h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_rfind_powervsx(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_rfind_byte_powervsx(h, h_length, n);

    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    __vector unsigned char n_first_vec = vec_splats(*(unsigned char const *)&n[offset_first]);
    __vector unsigned char n_mid_vec = vec_splats(*(unsigned char const *)&n[offset_mid]);
    __vector unsigned char n_last_vec = vec_splats(*(unsigned char const *)&n[offset_last]);

    for (; h_length >= n_length + 16; h_length -= 16) {
        sz_cptr_t h_reversed = h + h_length - n_length - 16 + 1;
        __vector unsigned char h_first_vec = vec_xl(0, (unsigned char const *)(h_reversed + offset_first));
        __vector unsigned char h_mid_vec = vec_xl(0, (unsigned char const *)(h_reversed + offset_mid));
        __vector unsigned char h_last_vec = vec_xl(0, (unsigned char const *)(h_reversed + offset_last));
        __vector unsigned char cmp = vec_and( //
            vec_and((__vector unsigned char)vec_cmpeq(h_first_vec, n_first_vec),
                    (__vector unsigned char)vec_cmpeq(h_mid_vec, n_mid_vec)),
            (__vector unsigned char)vec_cmpeq(h_last_vec, n_last_vec));
        sz_u64_t matches = sz_movemask_powervsx_(cmp);
        while (matches) {
            // Highest set bit within the low 16 bits is the right-most candidate.
            int potential_offset = (int)(sz_u64_clz(matches) - 48);
            if (sz_equal_powervsx(h + h_length - n_length - potential_offset, n, n_length))
                return h + h_length - n_length - potential_offset;
            matches &= ~(1ull << (15 - potential_offset));
        }
    }

    return sz_rfind_serial(h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_find_byteset_powervsx(sz_cptr_t h, sz_size_t h_length, sz_byteset_t const *set) {
    // Two halves of the 256-bit set, each 16 bytes, covering byte indices [0,16) and [16,32).
    __vector unsigned char set_lo = vec_xl(0, (unsigned char const *)&set->_u8s[0]);
    __vector unsigned char set_hi = vec_xl(0, (unsigned char const *)&set->_u8s[16]);

    for (; h_length >= 16; h += 16, h_length -= 16) {
        __vector unsigned char h_vec = vec_xl(0, (unsigned char const *)h);
        __vector unsigned char byte_index = vec_sr(h_vec, vec_splats((unsigned char)3)); // c >> 3, in [0,32)
        __vector unsigned char bit_pos = vec_and(h_vec, vec_splats((unsigned char)7));
        __vector unsigned char byte_mask = vec_sl(vec_splats((unsigned char)1), bit_pos);
        // `vec_perm` with two distinct registers indexes the full 32-byte set using the low 5 bits.
        __vector unsigned char looked_up = vec_perm(set_lo, set_hi, byte_index);
        __vector unsigned char anded = vec_and(looked_up, byte_mask);
        __vector unsigned char matched = (__vector unsigned char)vec_cmpne(anded, vec_splats((unsigned char)0));
#if defined(_ARCH_PWR10)
        // `matched` is 0xFF in matched lanes; the first such lane is the first set member.
        unsigned int idx = vec_first_match_index(matched, vec_splats((unsigned char)0xFF));
        if (idx != 16) return h + idx;
#else
        sz_u64_t matches = sz_movemask_powervsx_(matched);
        if (matches) return h + sz_u64_ctz(matches);
#endif
    }

    return sz_find_byteset_serial(h, h_length, set);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byteset_powervsx(sz_cptr_t h, sz_size_t h_length, sz_byteset_t const *set) {
    __vector unsigned char set_lo = vec_xl(0, (unsigned char const *)&set->_u8s[0]);
    __vector unsigned char set_hi = vec_xl(0, (unsigned char const *)&set->_u8s[16]);

    for (; h_length >= 16; h_length -= 16) {
        __vector unsigned char h_vec = vec_xl(0, (unsigned char const *)(h + h_length - 16));
        __vector unsigned char byte_index = vec_sr(h_vec, vec_splats((unsigned char)3));
        __vector unsigned char bit_pos = vec_and(h_vec, vec_splats((unsigned char)7));
        __vector unsigned char byte_mask = vec_sl(vec_splats((unsigned char)1), bit_pos);
        __vector unsigned char looked_up = vec_perm(set_lo, set_hi, byte_index);
        __vector unsigned char anded = vec_and(looked_up, byte_mask);
        __vector unsigned char matched = (__vector unsigned char)vec_cmpne(anded, vec_splats((unsigned char)0));
        sz_u64_t matches = sz_movemask_powervsx_(matched);
        if (matches) return h + h_length - 1 - (sz_u64_clz(matches) - 48);
    }

    return sz_rfind_byteset_serial(h, h_length, set);
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
