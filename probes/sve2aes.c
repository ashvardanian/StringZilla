/**
 *  @file probes/sve2aes.c
 *  @author Ash Vardanian
 *  @date July 9, 2026
 *  @brief Whether the toolchain builds the SVE2 AES kernels, from the Armv9-A crypto extension.
 */
#define STRINGZILLA_HEADER_ONLY 1
#define STRINGZILLA_TARGET_SVE2AES 1
#include <stringzilla/hash/sve2aes.h>

int main(void) {
    // Up to 16 bytes the kernel runs SVE2 AES, where longer inputs take the NEON AES one.
    char text[16] = {'x'};
    sz_u64_t hash = 0;
    sz_hash_sve2aes(text, sizeof(text), 42, &hash, 0);
    return hash == 0;
}
