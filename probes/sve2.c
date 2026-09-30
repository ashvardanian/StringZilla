/**
 *  @file probes/sve2.c
 *  @author Ash Vardanian
 *  @date July 9, 2026
 *  @brief Whether the toolchain builds the SVE2 kernels, the Armv9-A scalable vectors kit.
 */
#define STRINGZILLA_HEADER_ONLY 1
#define STRINGZILLA_TARGET_SVE2 1
#include <stringzilla/find/sve2.h>

/** Passes vectors between calls as the uncased kernels do, which LLVM 20 can't emit for Mach-O. */
__attribute__((target("+sve+sve2"), noinline)) svuint8_t sz_probe_increment_(svuint8_t bytes) {
    return svadd_n_u8_x(svptrue_b8(), bytes, 1);
}
__attribute__((target("+sve+sve2"))) static int sz_probe_passes_vectors_(void) {
    return svaddv_u8(svptrue_b8(), sz_probe_increment_(svdup_n_u8(0))) != 0;
}

int main(void) {
    char haystack[64] = {'x'};
    sz_byteset_t set;
    sz_cptr_t match = 0;
    sz_byteset_init(&set);
    sz_byteset_add(&set, 'x');
    sz_find_byteset_sve2(haystack, sizeof(haystack), &set, &match, 0);
    return match != haystack || !sz_probe_passes_vectors_();
}
