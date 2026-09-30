/**
 *  @file probes/neonaes.c
 *  @author Ash Vardanian
 *  @date July 9, 2026
 *  @brief Whether the toolchain builds the NEON AES kernels, from the Armv8-A crypto extension.
 */
#define STRINGZILLA_HEADER_ONLY 1
#define STRINGZILLA_TARGET_NEONAES 1
#include <stringzilla/hash/neonaes.h>

int main(void) {
    char text[64] = {'x'};
    sz_u64_t hash = 0;
    sz_hash_neonaes(text, sizeof(text), 42, &hash, 0);
    return hash == 0;
}
