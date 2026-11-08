/**
 *  @file probes/powervsx.c
 *  @author Ash Vardanian
 *  @date July 9, 2026
 *  @brief Whether the toolchain builds the Power VSX kernels, the IBM POWER9 vector kit.
 */
#define STRINGZILLA_HEADER_ONLY 1
#define STRINGZILLA_TARGET_POWERVSX 1
#include "stringzilla/find/powervsx.h"
#include "stringzilla/cipher/powervsx.h" // Needs the POWER9 half of `altivec.h`, closed to Clang at the floor

int main(void) {
    char haystack[64] = {'x'};
    sz_cptr_t match = 0;
    sz_find_byte_powervsx(haystack, sizeof(haystack), "x", &match, 0);
    return match != haystack;
}
