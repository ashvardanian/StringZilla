/**
 *  @file probes/haswell.c
 *  @author Ash Vardanian
 *  @date July 9, 2026
 *  @brief Whether the toolchain builds the Haswell kernels, the x86-64 AVX2 and BMI2 kit.
 */
#define STRINGZILLA_HEADER_ONLY 1
#define STRINGZILLA_ARCH_X8664_HASWELL_ 1
#include "stringzilla/find.h" // the hub, whose header-only tail defines the tier

int main(void) {
    char haystack[64] = {'x'};
    sz_cptr_t match = 0;
    sz_find_byte_haswell(haystack, sizeof(haystack), "x", &match, 0);
    return match != haystack;
}
