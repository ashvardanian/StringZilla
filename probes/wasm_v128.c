/* StringZilla ISA probe: WebAssembly SIMD128 (compile with `-msimd128`), mirroring `include/stringzilla/find/v128.h` */
#include <wasm_simd128.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

static v128_t sz_probe_swizzle_(v128_t table, v128_t indices) { return wasm_i8x16_swizzle(table, indices); }

int main(void) {
    v128_t gathered = sz_probe_swizzle_(wasm_i8x16_splat(1), wasm_i8x16_splat(2));
    return wasm_i8x16_extract_lane(gathered, 0) != 0 ? 0 : 1;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
