/**
 *  @file probes/loongsonasx.c
 *  @author Ash Vardanian
 *  @date July 9, 2026
 *  @brief Whether the toolchain builds the LASX kernels, the LoongArch 256-bit SIMD kit.
 */
#define STRINGZILLA_HEADER_ONLY 1
#define STRINGZILLA_ARCH_LOONGARCH64_LOONGSONASX_ 1
#include "stringzilla/find.h" // the hub, whose header-only tail defines the tier

int main(void) {
    char haystack[64] = {'x'};
    sz_cptr_t match = 0;
    sz_find_byte_loongsonasx(haystack, sizeof(haystack), "x", &match, 0);
    return match != haystack;
}
