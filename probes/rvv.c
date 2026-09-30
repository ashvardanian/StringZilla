/**
 *  @file probes/rvv.c
 *  @author Ash Vardanian
 *  @date July 9, 2026
 *  @brief Whether the toolchain builds the RVV kernels, the RISC-V Vector 1.0 kit.
 *
 *  Counting runes reaches 64-bit lanes, which Clang 20 hides from a unit compiled without the V
 *  extension, while a byte search never does.
 */
#define STRINGZILLA_HEADER_ONLY 1
#define STRINGZILLA_TARGET_RVV 1
#include <stringzilla/utf8_runes/rvv.h>

int main(void) {
    char text[64] = {'x'};
    sz_size_t count = 0;
    sz_utf8_count_rvv(text, sizeof(text), &count, 0);
    return count == 0;
}
