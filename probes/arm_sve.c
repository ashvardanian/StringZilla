/* StringZilla ISA probe: SVE (Armv8.2-A scalable vectors), mirroring `include/stringzilla/find/sve.h`.
 * SVE values cross function boundaries on purpose: the scalable-vector ABI is where fragile toolchains
 * break (some Clang builds ICE emitting it for Mach-O), which a body-only snippet would not expose. */
#if defined(_MSC_VER)
#error "SVE target attributes and intrinsics are unavailable under MSVC"
#endif
#if defined(__APPLE__)
#error "No Apple CPU implements SVE - kernels could compile but never run"
#endif
#include <arm_sve.h>
#include <stdint.h> // `uint64_t` picks the exact `svwhilelt_b8` overload on both LP64 and LLP64 targets

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve")
#endif

static svuint8_t sz_probe_select_(svbool_t mask, svuint8_t on, svuint8_t off) { return svsel_u8(mask, on, off); }

int main(void) {
    svbool_t head = svwhilelt_b8((uint64_t)0, svcntb());
    svuint8_t picked = sz_probe_select_(head, svdup_n_u8(1), svdup_n_u8(2));
    return svlasta_u8(svpfalse_b(), picked) != 0 ? 0 : 1;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
