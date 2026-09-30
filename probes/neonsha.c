/**
 *  @file probes/neonsha.c
 *  @author Ash Vardanian
 *  @date July 9, 2026
 *  @brief Whether the toolchain builds the NEON SHA2 kernels, from the Armv8-A crypto extension.
 */
#define STRINGZILLA_HEADER_ONLY 1
#define STRINGZILLA_TARGET_NEONSHA 1
#include <stringzilla/hash/neonsha.h>

int main(void) {
    char text[64] = {'x'};
    sz_sha256_state_t state;
    sz_sha256_state_init_neonsha(&state, 0);
    sz_sha256_state_update_neonsha(&state, text, sizeof(text), 0);
    return state.hash[0] == 0;
}
