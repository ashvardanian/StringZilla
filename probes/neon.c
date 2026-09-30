/**
 *  @file probes/neon.c
 *  @author Ash Vardanian
 *  @date July 9, 2026
 *  @brief Whether the toolchain builds the NEON kernels, the AArch64 baseline SIMD kit.
 */
#define STRINGZILLA_HEADER_ONLY 1
#define STRINGZILLA_TARGET_NEON 1
#include <stringzilla/find/neon.h>

int main(void) {
    char haystack[64] = {'x'};
    sz_cptr_t match = 0;
    sz_find_byte_neon(haystack, sizeof(haystack), "x", &match, 0);
    return match != haystack;
}
