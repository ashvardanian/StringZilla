/**
 *  @file probes/icelake.c
 *  @author Ash Vardanian
 *  @date July 9, 2026
 *  @brief Whether the toolchain builds the Ice Lake kernels, x86-64 AVX-512 with VBMI, VNNI, VAES.
 */
#define STRINGZILLA_HEADER_ONLY 1
#define STRINGZILLA_TARGET_ICELAKE 1
#include <stringzilla/hash/icelake.h>

int main(void) {
    char text[64] = {'x'};
    sz_u64_t hash = 0;
    sz_hash_icelake(text, sizeof(text), 42, &hash, 0);
    return hash == 0;
}
