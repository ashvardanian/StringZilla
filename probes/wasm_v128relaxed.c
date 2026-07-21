/* StringZilla ISA probe: WebAssembly relaxed SIMD (compile with `-msimd128 -mrelaxed-simd`), mirroring
 * `include/stringzilla/memory/v128relaxed.h` */
#include <wasm_simd128.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

static v128_t sz_probe_swizzle_(v128_t table, v128_t indices) { return wasm_i8x16_relaxed_swizzle(table, indices); }
static v128_t sz_probe_dot_(v128_t a, v128_t b, v128_t acc) {
    return wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a, b, acc);
}

int main(void) {
    v128_t gathered = sz_probe_swizzle_(wasm_i8x16_splat(1), wasm_i8x16_splat(2));
    v128_t summed = sz_probe_dot_(gathered, gathered, wasm_i32x4_splat(0));
    return wasm_i32x4_extract_lane(summed, 0) != 0 ? 0 : 1;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
