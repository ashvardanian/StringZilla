/**
 *  @brief RISC-V Vector (RVV 1.0) backend for memory.
 *  @file include/stringzilla/memory/rvv.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_RVV_H_
#define STRINGZILLA_MEMORY_RVV_H_

#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_RVV
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

SZ_PUBLIC void sz_copy_rvv(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    sz_u8_t *target_cursor = (sz_u8_t *)target;
    sz_u8_t const *source_cursor = (sz_u8_t const *)source;
    while (length) {
        sz_size_t vector_length = __riscv_vsetvl_e8m8(length);
        vuint8m8_t data_u8m8 = __riscv_vle8_v_u8m8(source_cursor, vector_length);
        __riscv_vse8_v_u8m8(target_cursor, data_u8m8, vector_length);
        target_cursor += vector_length, source_cursor += vector_length, length -= vector_length;
    }
}

SZ_PUBLIC void sz_move_rvv(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    if (target < source || target >= source + length) { sz_copy_rvv(target, source, length); }
    else {
        // Overlapping with `target > source`: walk from the end backwards.
        sz_u8_t *target_cursor = (sz_u8_t *)target + length;
        sz_u8_t const *source_cursor = (sz_u8_t const *)source + length;
        while (length) {
            sz_size_t vector_length = __riscv_vsetvl_e8m8(length);
            target_cursor -= vector_length, source_cursor -= vector_length;
            vuint8m8_t data_u8m8 = __riscv_vle8_v_u8m8(source_cursor, vector_length);
            __riscv_vse8_v_u8m8(target_cursor, data_u8m8, vector_length);
            length -= vector_length;
        }
    }
}

SZ_PUBLIC void sz_fill_rvv(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    sz_u8_t *target_cursor = (sz_u8_t *)target;
    while (length) {
        sz_size_t vector_length = __riscv_vsetvl_e8m8(length);
        vuint8m8_t fill_u8m8 = __riscv_vmv_v_x_u8m8(value, vector_length);
        __riscv_vse8_v_u8m8(target_cursor, fill_u8m8, vector_length);
        target_cursor += vector_length, length -= vector_length;
    }
}

/*  Byte-wise table lookup via an indexed (gather) memory load. `vluxei8` reads `lut[index[i]]` from
 *  the in-memory 256-entry table for every lane, so any input byte in [0, 255] is a valid index at any
 *  `VLEN` — no register-group capacity ceiling, hence no half-split and no scalar fallback. The
 *  `vsetvl`-driven loop handles every length including the sub-`VLMAX` tail, so there is no serial path. */
SZ_PUBLIC void sz_lookup_rvv(sz_ptr_t target, sz_size_t length, sz_cptr_t source, char const lut[sz_at_least_(256)]) {
    sz_u8_t *target_cursor = (sz_u8_t *)target;
    sz_u8_t const *source_cursor = (sz_u8_t const *)source;
    sz_u8_t const *lut_u8 = (sz_u8_t const *)lut;
    while (length) {
        sz_size_t vector_length = __riscv_vsetvl_e8m8(length);
        vuint8m8_t index_u8m8 = __riscv_vle8_v_u8m8(source_cursor, vector_length);
        vuint8m8_t result_u8m8 = __riscv_vluxei8_v_u8m8(lut_u8, index_u8m8, vector_length);
        __riscv_vse8_v_u8m8(target_cursor, result_u8m8, vector_length);
        target_cursor += vector_length, source_cursor += vector_length, length -= vector_length;
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_RVV

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_MEMORY_RVV_H_
