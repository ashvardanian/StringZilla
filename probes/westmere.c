/**
 *  @file probes/westmere.c
 *  @author Ash Vardanian
 *  @date July 9, 2026
 *  @brief Whether the toolchain builds the Westmere kernels, the x86-64 SSE4.2 and AES-NI kit.
 */
#define STRINGZILLA_HEADER_ONLY 1
#define STRINGZILLA_TARGET_WESTMERE 1
#include <stringzilla/hash/westmere.h>

int main(void) {
    char text[64] = {'x'};
    sz_u64_t hash = 0;
    sz_hash_westmere(text, sizeof(text), 42, &hash, 0);
    return hash == 0;
}
