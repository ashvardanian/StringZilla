/**
 *  @file probes/sve.c
 *  @author Ash Vardanian
 *  @date July 9, 2026
 *  @brief Whether the toolchain builds the SVE kernels, the Armv8.2-A scalable vectors kit.
 */
#define STRINGZILLA_HEADER_ONLY 1
#define STRINGZILLA_TARGET_SVE 1
#include <stringzilla/find/sve.h>

int main(void) {
    char haystack[64] = {'x'};
    sz_cptr_t match = 0;
    sz_find_byte_sve(haystack, sizeof(haystack), "x", &match, 0);
    return match != haystack;
}
