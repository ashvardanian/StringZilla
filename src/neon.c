#include <stringzilla/stringzilla.h>

#if SZ_USE_ARM_NEON
#include <arm_neon.h>

/**
 *  @brief  Substring-search implementation, leveraging Arm Neon intrinsics and speculative
 *          execution capabilities on modern CPUs. Performing 4 unaligned vector loads per cycle
 *          was practically more efficient than loading once and shifting around, as introduces
 *          less data dependencies.
 */
SZ_EXPORT sz_cptr_t sz_find_neon(sz_cptr_t const haystack, sz_size_t const haystack_length, sz_cptr_t const needle,
                                 sz_size_t const needle_length) {

    // Precomputed constants
    sz_cptr_t const end = haystack + haystack_length;
    _sz_anomaly_t anomaly;
    _sz_anomaly_t mask;
    sz_export_prefix_u32(needle, needle_length, &anomaly, &mask);
    uint32x4_t const anomalies = vld1q_dup_u32(&anomaly.u32);
    uint32x4_t const masks = vld1q_dup_u32(&mask.u32);
    uint32x4_t matches, matches0, matches1, matches2, matches3;

    sz_cptr_t text = haystack;
    while (text + needle_length + 16 <= end) {

        // Each of the following `matchesX` contains only 4 relevant bits - one per word.
        // Each signifies a match at the given offset.
        matches0 = vceqq_u32(vandq_u32(vreinterpretq_u32_u8(vld1q_u8((unsigned char *)text + 0)), masks), anomalies);
        matches1 = vceqq_u32(vandq_u32(vreinterpretq_u32_u8(vld1q_u8((unsigned char *)text + 1)), masks), anomalies);
        matches2 = vceqq_u32(vandq_u32(vreinterpretq_u32_u8(vld1q_u8((unsigned char *)text + 2)), masks), anomalies);
        matches3 = vceqq_u32(vandq_u32(vreinterpretq_u32_u8(vld1q_u8((unsigned char *)text + 3)), masks), anomalies);
        matches = vorrq_u32(vorrq_u32(matches0, matches1), vorrq_u32(matches2, matches3));

        if (vmaxvq_u32(matches)) {
            // Let's isolate the match from every word
            matches0 = vandq_u32(matches0, vdupq_n_u32(0x00000001));
            matches1 = vandq_u32(matches1, vdupq_n_u32(0x00000002));
            matches2 = vandq_u32(matches2, vdupq_n_u32(0x00000004));
            matches3 = vandq_u32(matches3, vdupq_n_u32(0x00000008));
            matches = vorrq_u32(vorrq_u32(matches0, matches1), vorrq_u32(matches2, matches3));

            // By now, every 32-bit word of `matches` no more than 4 set bits.
            // Meaning that we can narrow it down to a single 16-bit word.
            uint16x4_t matches_u16x4 = vmovn_u32(matches);
            uint16_t matches_u16 =                       //
                (vget_lane_u16(matches_u16x4, 0) << 0) | //
                (vget_lane_u16(matches_u16x4, 1) << 4) | //
                (vget_lane_u16(matches_u16x4, 2) << 8) | //
                (vget_lane_u16(matches_u16x4, 3) << 12);

            // Find the first match
            sz_size_t first_match_offset = sz_ctz64(matches_u16);
            if (needle_length > 4) {
                if (sz_equal(text + first_match_offset + 4, needle + 4, needle_length - 4)) {
                    return text + first_match_offset;
                }
                else { text += first_match_offset + 1; }
            }
            else { return text + first_match_offset; }
        }
        else { text += 16; }
    }

    // Don't forget the last (up to 16+3=19) characters.
    return sz_find_serial(text, end - text, needle, needle_length);
}

#endif // Arm Neon

#if SZ_USE_ARM_CRC32
#include <arm_acle.h>

SZ_EXPORT sz_u32_t sz_crc32_arm(sz_cptr_t start, sz_size_t length) {
    sz_u32_t crc = 0xFFFFFFFF;
    sz_cptr_t const end = start + length;

    // Align the input to the word boundary
    while (((unsigned long)start & 7ull) && start != end) { crc = __crc32cb(crc, *start), start++; }

    // Process the body 8 bytes at a time
    while (start + 8 <= end) { crc = __crc32cd(crc, *(unsigned long long *)start), start += 8; }

    // Process the tail bytes
    if (start + 4 <= end) { crc = __crc32cw(crc, *(unsigned int *)start), start += 4; }
    if (start + 2 <= end) { crc = __crc32ch(crc, *(unsigned short *)start), start += 2; }
    if (start < end) { crc = __crc32cb(crc, *start); }
    return crc ^ 0xFFFFFFFF;
}
#endif