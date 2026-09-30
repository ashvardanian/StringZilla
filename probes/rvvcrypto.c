/**
 *  @file probes/rvvcrypto.c
 *  @author Ash Vardanian
 *  @date July 9, 2026
 *  @brief Whether the toolchain builds the RVV Crypto kernels, Zvkned AES and Zvknhb SHA-2.
 */
#define STRINGZILLA_HEADER_ONLY 1
#define STRINGZILLA_TARGET_RVVCRYPTO 1
#include <stringzilla/hash/rvvcrypto.h>

int main(void) {
    char text[64] = {'x'};
    sz_u64_t hash = 0;
    sz_hash_rvvcrypto(text, sizeof(text), 42, &hash, 0);
    return hash == 0;
}
