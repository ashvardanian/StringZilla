/**
 *  @brief IBM Power VSX backend for memory.
 *  @file include/stringzilla/memory/powervsx.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_POWERVSX_H_
#define STRINGZILLA_MEMORY_POWERVSX_H_

#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h"

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

SZ_PUBLIC void sz_copy_powervsx(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    for (; length >= 16; target += 16, source += 16, length -= 16)
        vec_xst(vec_xl(0, (unsigned char const *)source), 0, (unsigned char *)target);
    if (length) sz_copy_serial(target, source, length);
}

SZ_PUBLIC void sz_move_powervsx(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    if (target < source || target >= source + length) {
        // Non-overlapping (or target before source) — copy forward.
        sz_copy_powervsx(target, source, length);
    }
    else {
        // Overlapping — walk backwards to preserve the source bytes.
        target += length, source += length;
        while (length >= 16) {
            target -= 16, source -= 16, length -= 16;
            vec_xst(vec_xl(0, (unsigned char const *)source), 0, (unsigned char *)target);
        }
        while (length) {
            target -= 1, source -= 1, length -= 1;
            *target = *source;
        }
    }
}

SZ_PUBLIC void sz_fill_powervsx(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    __vector unsigned char fill_u8x16 = vec_splats(value);
    for (; length >= 16; target += 16, length -= 16) vec_xst(fill_u8x16, 0, (unsigned char *)target);
    if (length) sz_fill_serial(target, length, value);
}

SZ_PUBLIC void sz_lookup_powervsx(sz_ptr_t target, sz_size_t length, sz_cptr_t source,
                                  char const lut[sz_at_least_(256)]) {
    // Small inputs aren't worth the SIMD setup cost — defer to the serial path.
    if (length <= 128) {
        sz_lookup_serial(target, length, source, lut);
        return;
    }

    // VSX `vec_perm` selects from a 32-byte (two-register) space using the low 5 bits of each index.
    // We replicate the byte-to-byte map by loading the 256-entry table into 16 chunks of 16 bytes,
    // permuting each chunk with the low nibble of the source, and blending in the chunk that the
    // high nibble of each source byte selects.
    __vector unsigned char lut_vecs[16];
    for (sz_size_t sub_table_index = 0; sub_table_index < 16; ++sub_table_index)
        lut_vecs[sub_table_index] = vec_xl(0, (unsigned char const *)(lut + sub_table_index * 16));

    __vector unsigned char const low_nibble_mask = vec_splats((unsigned char)0x0F);
    sz_size_t byte_index = 0;
    for (; byte_index + 16 <= length; byte_index += 16) {
        __vector unsigned char source_u8x16 = vec_xl(0, (unsigned char const *)(source + byte_index));
        __vector unsigned char low_nibble_index = vec_and(source_u8x16, low_nibble_mask);
        __vector unsigned char high_nibble_index = vec_sr(source_u8x16,
                                                          vec_splats((unsigned char)4)); // 0..15 chunk selector
        __vector unsigned char result = vec_splats((unsigned char)0);
        for (sz_size_t chunk_index = 0; chunk_index < 16; ++chunk_index) {
            __vector unsigned char permuted = vec_perm(lut_vecs[chunk_index], lut_vecs[chunk_index], low_nibble_index);
            __vector unsigned char selector = (__vector unsigned char)vec_cmpeq(high_nibble_index,
                                                                                vec_splats((unsigned char)chunk_index));
            result = vec_sel(result, permuted, selector);
        }
        vec_xst(result, 0, (unsigned char *)(target + byte_index));
    }

    // Handle the remaining tail serially.
    if (byte_index < length) sz_lookup_serial(target + byte_index, length - byte_index, source + byte_index, lut);
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

#endif // STRINGZILLA_MEMORY_POWERVSX_H_
