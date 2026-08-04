/**
 *  @brief SVE backend for substring & byte-set search.
 *  @file include/stringzilla/find/sve.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_SVE_H_
#define STRINGZILLA_FIND_SVE_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/find/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  Implementation of the string search algorithms using the Arm SVE variable-length registers,
 *  available in Arm v9 processors, like in Apple M4+ and Graviton 3+ CPUs.
 */
#if SZ_USE_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve")
#endif

SZ_API_COMPTIME sz_cptr_t sz_find_byte_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    sz_u8_t const n_scalar = *needle;
    // Determine the number of bytes in an SVE vector.
    sz_size_t const vector_bytes = svcntb();
    sz_size_t progress = 0;
    do {
        svbool_t progress_mask_b8x = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)haystack_length);
        svuint8_t haystack_u8x = svld1(progress_mask_b8x, (sz_u8_t const *)(haystack + progress));
        // Compare: generate a predicate marking lanes where haystack[i] == needle
        svbool_t equal_b8x = svcmpeq_n_u8(progress_mask_b8x, haystack_u8x, n_scalar);
        if (svptest_any(progress_mask_b8x, equal_b8x)) {
            sz_size_t forward_offset_in_register = svcntp_b8(progress_mask_b8x,
                                                             svbrkb_b_z(progress_mask_b8x, equal_b8x));
            return haystack + progress + forward_offset_in_register;
        }
        progress += vector_bytes;
    } while (progress < haystack_length);
    // No match found.
    return SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    sz_u8_t const n_scalar = *needle;
    // Determine the number of bytes in an SVE vector.
    sz_size_t const vector_bytes = svcntb();
    sz_size_t progress = 0;
    do {
        svbool_t progress_mask_b8x = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)haystack_length);
        svbool_t backward_mask_b8x = svrev_b8(progress_mask_b8x);
        svuint8_t haystack_u8x = svld1(backward_mask_b8x,
                                       (sz_u8_t const *)(haystack + haystack_length - progress - vector_bytes));
        // Compare: generate a predicate marking lanes where haystack[i] == needle
        svbool_t equal_b8x = svcmpeq_n_u8(backward_mask_b8x, haystack_u8x, n_scalar);
        if (svptest_any(backward_mask_b8x, equal_b8x)) {
            sz_size_t backward_offset_in_register = svcntp_b8(progress_mask_b8x,
                                                              svbrkb_b_z(progress_mask_b8x, svrev_b8(equal_b8x)));
            return haystack + haystack_length - progress - backward_offset_in_register - 1;
        }
        progress += vector_bytes;
    } while (progress < haystack_length);
    // No match found.
    return SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_cptr_t sz_find_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                      sz_size_t needle_length) {
    // Empty needle matches at the start, like `strstr`.
    if (!needle_length) return haystack;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_find_byte_sve(haystack, haystack_length, needle);

    // Determine the number of bytes in an SVE vector.
    sz_size_t const vector_bytes = svcntb();
    sz_size_t progress = 0;

    if (needle_length == 2) {
        // Broadcast needle characters.
        sz_u8_t n0 = ((sz_u8_t *)needle)[0];
        sz_u8_t n1 = ((sz_u8_t *)needle)[1];
        do {
            // We must avoid overrunning the haystack for the second byte.
            svbool_t pred_b8x = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)(haystack_length - 1));
            // Load two adjacent vectors.
            svuint8_t hay0_u8x = svld1(pred_b8x, (sz_u8_t const *)(haystack + progress));
            svuint8_t hay1_u8x = svld1(pred_b8x, (sz_u8_t const *)(haystack + progress + 1));
            svbool_t cmp0_b8x = svcmpeq_n_u8(pred_b8x, hay0_u8x, n0);
            svbool_t cmp1_b8x = svcmpeq_n_u8(pred_b8x, hay1_u8x, n1);
            svbool_t matches_b8x = svmov_b_z(cmp0_b8x, cmp1_b8x); //? Practically a bitwise AND
            if (svptest_any(pred_b8x, matches_b8x))
                return haystack + progress + svcntp_b8(pred_b8x, svbrkb_b_z(pred_b8x, matches_b8x));
            progress += vector_bytes;
        } while (progress < (haystack_length - 1));
        return SZ_NULL_CHAR;
    }
    else if (needle_length == 3) {
        // Broadcast needle characters.
        sz_u8_t n0 = ((sz_u8_t *)needle)[0];
        sz_u8_t n1 = ((sz_u8_t *)needle)[1];
        sz_u8_t n2 = ((sz_u8_t *)needle)[2];
        do {
            // Prevent overrunning for the 3rd byte.
            svbool_t pred_b8x = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)(haystack_length - 2));
            svuint8_t hay0_u8x = svld1(pred_b8x, (sz_u8_t const *)(haystack + progress));
            svuint8_t hay1_u8x = svld1(pred_b8x, (sz_u8_t const *)(haystack + progress + 1));
            svuint8_t hay2_u8x = svld1(pred_b8x, (sz_u8_t const *)(haystack + progress + 2));
            svbool_t cmp0_b8x = svcmpeq_n_u8(pred_b8x, hay0_u8x, n0);
            svbool_t cmp1_b8x = svcmpeq_n_u8(pred_b8x, hay1_u8x, n1);
            svbool_t cmp2_b8x = svcmpeq_n_u8(pred_b8x, hay2_u8x, n2);
            svbool_t matches_b8x = svand_b_z(cmp0_b8x, cmp1_b8x, cmp2_b8x); //? Practically a 3-way AND.
            if (svptest_any(pred_b8x, matches_b8x))
                return haystack + progress + svcntp_b8(pred_b8x, svbrkb_b_z(pred_b8x, matches_b8x));
            progress += vector_bytes;
        } while (progress < (haystack_length - 2));
        return SZ_NULL_CHAR;
    }
    else {
        // For longer needles we first pick "anomalies" (i.e. informative offsets)
        sz_size_t offset_first, offset_mid, offset_last;
        sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);
        // Broadcast the selected needle bytes.
        sz_u8_t n_first = ((sz_u8_t *)needle)[offset_first];
        sz_u8_t n_mid = ((sz_u8_t *)needle)[offset_mid];
        sz_u8_t n_last = ((sz_u8_t *)needle)[offset_last];
        do {
            // Make sure the predicate does not run off the end.
            svbool_t pred_b8x = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)(haystack_length - needle_length + 1));
            // Load haystack bytes at the chosen offsets.
            svuint8_t hay_first_u8x = svld1(pred_b8x, (sz_u8_t const *)(haystack + progress + offset_first));
            svuint8_t hay_mid_u8x = svld1(pred_b8x, (sz_u8_t const *)(haystack + progress + offset_mid));
            svuint8_t hay_last_u8x = svld1(pred_b8x, (sz_u8_t const *)(haystack + progress + offset_last));
            svbool_t cmp0_b8x = svcmpeq_n_u8(pred_b8x, hay_first_u8x, n_first);
            svbool_t cmp1_b8x = svcmpeq_n_u8(pred_b8x, hay_mid_u8x, n_mid);
            svbool_t cmp2_b8x = svcmpeq_n_u8(pred_b8x, hay_last_u8x, n_last);
            svbool_t matches_b8x = svand_b_z(cmp0_b8x, cmp1_b8x, cmp2_b8x); //? Practically a 3-way AND.
            // There might be multiple candidate positions, so we need to iterate over them.
            while (svptest_any(pred_b8x, matches_b8x)) {
                svbool_t pred_to_skip_b8x = svbrkb_b_z(pred_b8x, matches_b8x);
                sz_size_t forward_offset_in_register = svcntp_b8(pred_b8x, pred_to_skip_b8x);
                if (sz_equal_sve(haystack + progress + forward_offset_in_register, needle, needle_length))
                    return haystack + progress + forward_offset_in_register;
                // If it doesn't match - clear the first bit and continue
                svbool_t first_match_b8x = svpnext_b8(svptrue_b8(), pred_to_skip_b8x);
                sz_assert_(svcntp_b8(svptrue_b8(), first_match_b8x) == 1);
                matches_b8x = svbic_b_z(svptrue_b8(), matches_b8x, first_match_b8x);
            }
            progress += vector_bytes;
        } while (progress < haystack_length - (needle_length - 1));
        return SZ_NULL_CHAR;
    }
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                       sz_size_t needle_length) {
    // Empty needle matches at the end.
    if (!needle_length) return haystack + haystack_length;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_rfind_byte_sve(haystack, haystack_length, needle);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);
    sz_u8_t const n_first = ((sz_u8_t *)needle)[offset_first];
    sz_u8_t const n_mid = ((sz_u8_t *)needle)[offset_mid];
    sz_u8_t const n_last = ((sz_u8_t *)needle)[offset_last];

    // Walk candidate blocks from the end; within a block the matches are consumed in REVERSED lane order, so
    // the candidate loop is the forward `sz_find_sve` loop over the reversed match predicate.
    sz_size_t const vector_bytes = svcntb();
    sz_size_t const candidates = haystack_length - needle_length + 1;
    sz_size_t progress = 0;
    do {
        svbool_t const pred_b8x = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)candidates);
        svbool_t const backward_mask_b8x = svrev_b8(pred_b8x);
        sz_cptr_t const block = haystack + candidates - progress - vector_bytes;
        svuint8_t const hay_first_u8x = svld1(backward_mask_b8x, (sz_u8_t const *)(block + offset_first));
        svuint8_t const hay_mid_u8x = svld1(backward_mask_b8x, (sz_u8_t const *)(block + offset_mid));
        svuint8_t const hay_last_u8x = svld1(backward_mask_b8x, (sz_u8_t const *)(block + offset_last));
        svbool_t const cmp0_b8x = svcmpeq_n_u8(backward_mask_b8x, hay_first_u8x, n_first);
        svbool_t const cmp1_b8x = svcmpeq_n_u8(backward_mask_b8x, hay_mid_u8x, n_mid);
        svbool_t const cmp2_b8x = svcmpeq_n_u8(backward_mask_b8x, hay_last_u8x, n_last);
        svbool_t matches_b8x = svrev_b8(
            svand_b_z(cmp0_b8x, cmp1_b8x, cmp2_b8x)); //? A 3-way AND, reversed to walk back-to-front.
        while (svptest_any(pred_b8x, matches_b8x)) {
            svbool_t const pred_to_skip_b8x = svbrkb_b_z(pred_b8x, matches_b8x);
            sz_size_t const backward_offset = svcntp_b8(pred_b8x, pred_to_skip_b8x);
            sz_size_t const candidate = candidates - progress - backward_offset - 1;
            if (sz_equal_sve(haystack + candidate, needle, needle_length)) return haystack + candidate;
            // If it doesn't match - clear the first (i.e. rightmost) bit and continue.
            svbool_t const first_match_b8x = svpnext_b8(svptrue_b8(), pred_to_skip_b8x);
            matches_b8x = svbic_b_z(svptrue_b8(), matches_b8x, first_match_b8x);
        }
        progress += vector_bytes;
    } while (progress < candidates);
    return SZ_NULL_CHAR;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_SVE_H_
