/**
 *  @file probes/goldmont.c
 *  @author Ash Vardanian
 *  @date July 9, 2026
 *  @brief Whether the toolchain builds the Goldmont kernels, the x86-64 SHA-NI kit.
 */
#define STRINGZILLA_HEADER_ONLY 1
#define STRINGZILLA_TARGET_GOLDMONT 1
#include <stringzilla/hash/goldmont.h>

int main(void) {
    char text[64] = {'x'};
    sz_sha256_state_t state;
    sz_sha256_state_init_goldmont(&state, 0);
    sz_sha256_state_update_goldmont(&state, text, sizeof(text), 0);
    return state.hash[0] == 0;
}
