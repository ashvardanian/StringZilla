/**
 *  @file probes/skylake.c
 *  @author Ash Vardanian
 *  @date July 9, 2026
 *  @brief Whether the toolchain builds the Skylake-X kernels, the x86-64 AVX-512 F, VL, and BW kit.
 */
#define STRINGZILLA_HEADER_ONLY 1
#define STRINGZILLA_TARGET_SKYLAKE 1
#include <stringzilla/find/skylake.h>

int main(void) {
    char haystack[64] = {'x'};
    sz_cptr_t match = 0;
    sz_find_byte_skylake(haystack, sizeof(haystack), "x", &match, 0);
    return match != haystack;
}
