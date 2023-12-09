#include <stringzilla/stringzilla.h>

#if defined(__SSE4_2__)
#include <x86intrin.h>

SZ_EXPORT sz_u32_t sz_crc32_sse42(sz_cptr_t start, sz_size_t length) {
    sz_u32_t crc = 0xFFFFFFFF;
    sz_cptr_t const end = start + length;

    // Align the input to the word boundary
    while (((unsigned long)start & 7ull) && start != end) { crc = _mm_crc32_u8(crc, *start), start++; }

    // Process the body 8 bytes at a time
    while (start + 8 <= end) { crc = (sz_u32_t)_mm_crc32_u64(crc, *(unsigned long long *)start), start += 8; }

    // Process the tail bytes
    if (start + 4 <= end) { crc = _mm_crc32_u32(crc, *(unsigned int *)start), start += 4; }
    if (start + 2 <= end) { crc = _mm_crc32_u16(crc, *(unsigned short *)start), start += 2; }
    if (start < end) { crc = _mm_crc32_u8(crc, *start); }
    return crc ^ 0xFFFFFFFF;
}
#endif
