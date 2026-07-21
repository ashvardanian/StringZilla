/**
 *  @brief SVE2 backend for byte-set search.
 *  @file include/stringzilla/find/sve2.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_SVE2_H_
#define STRINGZILLA_FIND_SVE2_H_

#include "stringzilla/types.h"
#include "stringzilla/find/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SVE2
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve+sve2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve+sve2")
#endif

/** @brief  Byte-set membership for one predicated vector: the serial `set->_u8s[c >> 3] & (1 << (c & 7))` test
 *          rides two 16-byte `svtbl` tables — the second addressed at `index - 16`, where the wrap past the
 *          zero-padded table reads zero at any vector length. */
SZ_HELPER_INLINE svbool_t sz_find_byteset_matches_sve2_(svbool_t progress_b8x, svuint8_t haystack_u8x,
                                                        svuint8_t set_low_u8x, svuint8_t set_high_u8x) {
    svuint8_t const byte_index_u8x = svlsr_n_u8_x(progress_b8x, haystack_u8x, 3);
    svuint8_t const bit_mask_u8x = svlsl_u8_x(progress_b8x, svdup_n_u8(1), svand_n_u8_x(progress_b8x, haystack_u8x, 7));
    svuint8_t const bitmap_u8x = svorr_u8_x(progress_b8x, svtbl_u8(set_low_u8x, byte_index_u8x),
                                            svtbl_u8(set_high_u8x, svsub_n_u8_x(progress_b8x, byte_index_u8x, 16)));
    return svcmpne_n_u8(progress_b8x, svand_u8_x(progress_b8x, bitmap_u8x, bit_mask_u8x), 0);
}

SZ_API_COMPTIME sz_cptr_t sz_find_byteset_sve2(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *set) {
    sz_size_t const vector_bytes = svcntb();
    sz_size_t progress = 0;
    if (!haystack_length) return SZ_NULL_CHAR;

    // A set of up to 16 distinct bytes fits one 128-bit needle segment, and one `svmatch_u8` compares every
    // haystack byte against all of them; the tail pads with the first member so no foreign byte can match.
    // Extracting the members walks the whole 256-bit set, so short scans - the all-matches iteration pattern
    // re-enters with just a few bytes to the next hit - skip straight to the load-and-go table path.
    sz_align_(16) sz_u8_t members[16];
    sz_size_t const member_count = haystack_length >= vector_bytes * 4 ? sz_byteset_population_serial_(set)
                                                                       : SZ_SIZE_MAX;
    svbool_t const all_b8x = svptrue_b8();
    if (member_count <= 16) {
        if (member_count == 0) return SZ_NULL_CHAR;
        sz_byteset_members_serial_(set, members);
        for (sz_size_t lane = member_count; lane < 16; ++lane) members[lane] = members[0];
        svuint8_t const needles_u8x = svld1rq_u8(all_b8x, members);
        for (; progress + vector_bytes <= haystack_length; progress += vector_bytes) {
            svuint8_t const haystack_u8x = svld1_u8(all_b8x, (sz_u8_t const *)(haystack + progress));
            svbool_t const matches_b8x = svmatch_u8(all_b8x, haystack_u8x, needles_u8x);
            if (svptest_any(all_b8x, matches_b8x))
                return haystack + progress + svcntp_b8(all_b8x, svbrkb_b_z(all_b8x, matches_b8x));
        }
        if (progress < haystack_length) {
            svbool_t const tail_b8x = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)haystack_length);
            svuint8_t const haystack_u8x = svld1_u8(tail_b8x, (sz_u8_t const *)(haystack + progress));
            svbool_t const matches_b8x = svmatch_u8(tail_b8x, haystack_u8x, needles_u8x);
            if (svptest_any(tail_b8x, matches_b8x))
                return haystack + progress + svcntp_b8(tail_b8x, svbrkb_b_z(tail_b8x, matches_b8x));
        }
        return SZ_NULL_CHAR;
    }

    svuint8_t const set_low_u8x = svld1_u8(svwhilelt_b8_u64(0, 16), set->_u8s);
    svuint8_t const set_high_u8x = svld1_u8(svwhilelt_b8_u64(0, 16), set->_u8s + 16);
    for (; progress + vector_bytes <= haystack_length; progress += vector_bytes) {
        svuint8_t const haystack_u8x = svld1_u8(all_b8x, (sz_u8_t const *)(haystack + progress));
        svbool_t const matches_b8x = sz_find_byteset_matches_sve2_(all_b8x, haystack_u8x, set_low_u8x, set_high_u8x);
        if (svptest_any(all_b8x, matches_b8x))
            return haystack + progress + svcntp_b8(all_b8x, svbrkb_b_z(all_b8x, matches_b8x));
    }
    if (progress < haystack_length) {
        svbool_t const tail_b8x = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)haystack_length);
        svuint8_t const haystack_u8x = svld1_u8(tail_b8x, (sz_u8_t const *)(haystack + progress));
        svbool_t const matches_b8x = sz_find_byteset_matches_sve2_(tail_b8x, haystack_u8x, set_low_u8x, set_high_u8x);
        if (svptest_any(tail_b8x, matches_b8x))
            return haystack + progress + svcntp_b8(tail_b8x, svbrkb_b_z(tail_b8x, matches_b8x));
    }
    return SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_sve2(sz_cptr_t haystack, sz_size_t haystack_length,
                                                sz_byteset_t const *set) {
    sz_size_t const vector_bytes = svcntb();
    sz_size_t progress = 0;
    if (!haystack_length) return SZ_NULL_CHAR;

    sz_align_(16) sz_u8_t members[16];
    sz_size_t const member_count = haystack_length >= vector_bytes * 4 ? sz_byteset_population_serial_(set)
                                                                       : SZ_SIZE_MAX;
    svbool_t const all_b8x = svptrue_b8();
    if (member_count <= 16) {
        if (member_count == 0) return SZ_NULL_CHAR;
        sz_byteset_members_serial_(set, members);
        for (sz_size_t lane = member_count; lane < 16; ++lane) members[lane] = members[0];
        svuint8_t const needles_u8x = svld1rq_u8(all_b8x, members);
        for (; progress + vector_bytes <= haystack_length; progress += vector_bytes) {
            svuint8_t const haystack_u8x = svld1_u8(
                all_b8x, (sz_u8_t const *)(haystack + haystack_length - progress - vector_bytes));
            svbool_t const matches_b8x = svmatch_u8(all_b8x, haystack_u8x, needles_u8x);
            if (svptest_any(all_b8x, matches_b8x)) {
                sz_size_t const backward_offset = svcntp_b8(all_b8x, svbrkb_b_z(all_b8x, svrev_b8(matches_b8x)));
                return haystack + haystack_length - progress - backward_offset - 1;
            }
        }
        if (progress < haystack_length) {
            svbool_t const progress_b8x = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)haystack_length);
            svbool_t const backward_b8x = svrev_b8(progress_b8x);
            svuint8_t const haystack_u8x = svld1_u8(
                backward_b8x, (sz_u8_t const *)(haystack + haystack_length - progress - vector_bytes));
            svbool_t const matches_b8x = svmatch_u8(backward_b8x, haystack_u8x, needles_u8x);
            if (svptest_any(backward_b8x, matches_b8x)) {
                sz_size_t const backward_offset = svcntp_b8(progress_b8x,
                                                            svbrkb_b_z(progress_b8x, svrev_b8(matches_b8x)));
                return haystack + haystack_length - progress - backward_offset - 1;
            }
        }
        return SZ_NULL_CHAR;
    }

    svuint8_t const set_low_u8x = svld1_u8(svwhilelt_b8_u64(0, 16), set->_u8s);
    svuint8_t const set_high_u8x = svld1_u8(svwhilelt_b8_u64(0, 16), set->_u8s + 16);
    for (; progress + vector_bytes <= haystack_length; progress += vector_bytes) {
        svuint8_t const haystack_u8x = svld1_u8(
            all_b8x, (sz_u8_t const *)(haystack + haystack_length - progress - vector_bytes));
        svbool_t const matches_b8x = sz_find_byteset_matches_sve2_(all_b8x, haystack_u8x, set_low_u8x, set_high_u8x);
        if (svptest_any(all_b8x, matches_b8x)) {
            sz_size_t const backward_offset = svcntp_b8(all_b8x, svbrkb_b_z(all_b8x, svrev_b8(matches_b8x)));
            return haystack + haystack_length - progress - backward_offset - 1;
        }
    }
    if (progress < haystack_length) {
        svbool_t const progress_b8x = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)haystack_length);
        svbool_t const backward_b8x = svrev_b8(progress_b8x);
        svuint8_t const haystack_u8x = svld1_u8(
            backward_b8x, (sz_u8_t const *)(haystack + haystack_length - progress - vector_bytes));
        svbool_t const matches_b8x = sz_find_byteset_matches_sve2_(backward_b8x, haystack_u8x, set_low_u8x,
                                                                   set_high_u8x);
        if (svptest_any(backward_b8x, matches_b8x)) {
            sz_size_t const backward_offset = svcntp_b8(progress_b8x, svbrkb_b_z(progress_b8x, svrev_b8(matches_b8x)));
            return haystack + haystack_length - progress - backward_offset - 1;
        }
    }
    return SZ_NULL_CHAR;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE2

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_SVE2_H_
