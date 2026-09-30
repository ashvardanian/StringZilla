/**
 *  @file probes/sve2.c
 *  @author Ash Vardanian
 *  @date July 9, 2026
 *  @brief Whether the toolchain builds the SVE2 kernels, the Armv9-A scalable vectors kit.
 */
#define STRINGZILLA_HEADER_ONLY 1
#define STRINGZILLA_TARGET_SVE2 1
#include <stringzilla/find/sve2.h>

int main(void) {
    char haystack[64] = {'x'};
    sz_byteset_t set;
    sz_cptr_t match = 0;
    sz_byteset_init(&set);
    sz_byteset_add(&set, 'x');
    sz_find_byteset_sve2(haystack, sizeof(haystack), &set, &match, 0);
    return match != haystack;
}
