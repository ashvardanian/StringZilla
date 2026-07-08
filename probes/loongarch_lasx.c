/* StringZilla ISA probe: LoongArch LASX 256-bit SIMD (compile with `-mlasx`), mirroring
 * `include/stringzilla/find/lasx.h` — the LASX kernels carry no per-function target attributes,
 * so the whole translation unit must be built with LASX enabled, and this probe matches that. */
#include <lasxintrin.h>

static __m256i sz_probe_ltz_(__m256i bytes) { return __lasx_xvmskltz_b(bytes); }

int main(void) {
    __m256i mask = sz_probe_ltz_(__lasx_xvreplgr2vr_b(-1));
    return __lasx_xvpickve2gr_wu(mask, 0) != 0 ? 0 : 1;
}
