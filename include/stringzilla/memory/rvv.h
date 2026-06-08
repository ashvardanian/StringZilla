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
        size_t vector_length = __riscv_vsetvl_e8m8(length);
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
            size_t vector_length = __riscv_vsetvl_e8m8(length);
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
        size_t vector_length = __riscv_vsetvl_e8m8(length);
        vuint8m8_t fill_u8m8 = __riscv_vmv_v_x_u8m8(value, vector_length);
        __riscv_vse8_v_u8m8(target_cursor, fill_u8m8, vector_length);
        target_cursor += vector_length, length -= vector_length;
    }
}

/*  Byte-wise table lookup. RVV has no bounded `vqtbl`-style instruction, but `vrgather` performs
 *  an arbitrary gather within a register group. For very short inputs the serial path avoids the
 *  LUT-load overhead, matching the NEON heuristic.
 *
 *  If the hardware can hold all 256 LUT entries in a single `u8m8` group (`VLEN >= 256`) we gather
 *  directly: any input byte in [0, 255] is a valid index, so no masking is needed.
 *
 *  On `VLEN == 128` a `u8m8` group only spans 128 bytes, so a single gather cannot reach the upper
 *  half of the table. Instead of falling back to scalar, we split the LUT into a low half (indices
 *  0..127) and a high half (128..255), gather both with the low 7 index bits, and select per lane by
 *  the input byte's top bit via `vmsgeu` + `vmerge`. This keeps `VLEN == 128` parts (a large share of
 *  RVV 1.0 hardware) on the vector path. */
SZ_PUBLIC void sz_lookup_rvv(sz_ptr_t target, sz_size_t length, sz_cptr_t source, char const lut[sz_at_least_(256)]) {
    if (length <= 128) {
        sz_lookup_serial(target, length, source, lut);
        return;
    }

    sz_u8_t *target_cursor = (sz_u8_t *)target;
    sz_u8_t const *source_cursor = (sz_u8_t const *)source;
    sz_u8_t const *lut_u8 = (sz_u8_t const *)lut;

    size_t lut_vector_length = __riscv_vsetvl_e8m8(256);
    if (lut_vector_length >= 256) {
        vuint8m8_t lut_u8m8 = __riscv_vle8_v_u8m8(lut_u8, 256);
        while (length) {
            size_t vector_length = __riscv_vsetvl_e8m8(length);
            vuint8m8_t index_u8m8 = __riscv_vle8_v_u8m8(source_cursor, vector_length);
            vuint8m8_t result_u8m8 = __riscv_vrgather_vv_u8m8(lut_u8m8, index_u8m8, vector_length);
            __riscv_vse8_v_u8m8(target_cursor, result_u8m8, vector_length);
            target_cursor += vector_length, source_cursor += vector_length, length -= vector_length;
        }
    }
    else if (__riscv_vsetvl_e8m8(128) >= 128) {
        // Two-half gather for `VLEN == 128` (and any case where a group holds >= 128 but < 256 bytes).
        vuint8m8_t lut_low_u8m8 = __riscv_vle8_v_u8m8(lut_u8, 128);
        vuint8m8_t lut_high_u8m8 = __riscv_vle8_v_u8m8(lut_u8 + 128, 128);
        while (length) {
            size_t vector_length = __riscv_vsetvl_e8m8(length);
            vuint8m8_t index_u8m8 = __riscv_vle8_v_u8m8(source_cursor, vector_length);
            vuint8m8_t index_low7_u8m8 = __riscv_vand_vx_u8m8(index_u8m8, 0x7F,
                                                              vector_length); // low 7 bits index either half
            vuint8m8_t result_low_u8m8 = __riscv_vrgather_vv_u8m8(lut_low_u8m8, index_low7_u8m8, vector_length);
            vuint8m8_t result_high_u8m8 = __riscv_vrgather_vv_u8m8(lut_high_u8m8, index_low7_u8m8, vector_length);
            vbool1_t high_half_mask = __riscv_vmsgeu_vx_u8m8_b1(index_u8m8, 128,
                                                                vector_length); // top bit set -> upper half
            vuint8m8_t result_u8m8 = __riscv_vmerge_vvm_u8m8(result_low_u8m8, result_high_u8m8, high_half_mask,
                                                             vector_length);
            __riscv_vse8_v_u8m8(target_cursor, result_u8m8, vector_length);
            target_cursor += vector_length, source_cursor += vector_length, length -= vector_length;
        }
    }
    else { sz_lookup_serial(target, length, source, lut); }
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
