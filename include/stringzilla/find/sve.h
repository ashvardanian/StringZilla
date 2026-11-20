/**
 *  @file include/stringzilla/find/sve.h
 *  @author Ash Vardanian
 *  @date October 3, 2024
 *  @brief SVE backend for substring & byte-set search.
 *
 *  @sa c/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_SVE_H_
#define STRINGZILLA_FIND_SVE_H_

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_ARM64_
#if STRINGZILLA_ARCH_ARM64_SVE_

#include "stringzilla/compare/sve.h" // `sz_equal_sve_`
#include "stringzilla/find/serial.h"
#include "stringzilla/find/neon.h" // `sz_find_neon_`, `sz_rfind_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

/*  Implementation of the string search algorithms using the Arm SVE variable-length registers,
 *  available in Arm v9 processors, like in Apple M4+ and Graviton 3+ CPUs. */

/** How far from the start (or, backwards, the end) of the haystack the substring search keeps the
 *  low-latency loop before switching to the high-throughput one. Matches of frequent needles fall
 *  within it, and on long scans of rare needles it is negligible. Graviton 4 and 5 measurements. */
#define STRINGZILLA_FIND_SVE_NEAR_BYTES 512
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve")
#endif

/** XOR of the haystack bytes at up to three probe offsets with the needle bytes there, OR-ed
 *  together: a lane is zero exactly where all probes match. Staying in the vector domain keeps a
 *  single predicate-producing compare per register, instead of one per probe plus predicate ANDs,
 *  which is what limits the predicate-heavy loop on Neoverse V1 (Graviton 3). */
STRINGZILLA_INLINE svuint8_t sz_find_probes_diff_sve_(svbool_t pred_b8x, sz_u8_t const *text,
                                                      sz_size_t const offsets[3], svuint8_t first_u8x,
                                                      svuint8_t second_u8x, svuint8_t third_u8x, int probes) {
    svuint8_t diff_u8x = sveor_u8_x(pred_b8x, svld1_u8(pred_b8x, text + offsets[0]), first_u8x);
    if (probes > 1)
        diff_u8x = svorr_u8_x(pred_b8x, diff_u8x,
                              sveor_u8_x(pred_b8x, svld1_u8(pred_b8x, text + offsets[1]), second_u8x));
    if (probes > 2)
        diff_u8x = svorr_u8_x(pred_b8x, diff_u8x,
                              sveor_u8_x(pred_b8x, svld1_u8(pred_b8x, text + offsets[2]), third_u8x));
    return diff_u8x;
}

/** Lanes of the block at @p text where all probes match. One probe compares directly. */
STRINGZILLA_INLINE svbool_t sz_find_probes_match_sve_(svbool_t pred_b8x, sz_u8_t const *text,
                                                      sz_size_t const offsets[3], svuint8_t first_u8x,
                                                      svuint8_t second_u8x, svuint8_t third_u8x, int probes) {
    if (probes == 1) return svcmpeq_u8(pred_b8x, svld1_u8(pred_b8x, text + offsets[0]), first_u8x);
    return svcmpeq_n_u8(pred_b8x,
                        sz_find_probes_diff_sve_(pred_b8x, text, offsets, first_u8x, second_u8x, third_u8x, probes), 0);
}

/** The first candidate of @p matches_b8x in the block at @p block that holds the whole needle, or
 *  NULL. With @p verify false the probes cover the needle and @p matches_b8x must not be empty. */
STRINGZILLA_INLINE sz_cptr_t sz_find_probes_resolve_sve_(svbool_t pred_b8x, svbool_t matches_b8x, sz_cptr_t block,
                                                         sz_cptr_t needle, sz_size_t needle_length, int verify) {
    if (!verify) return block + svcntp_b8(pred_b8x, svbrkb_b_z(pred_b8x, matches_b8x));
    while (svptest_any(pred_b8x, matches_b8x)) {
        svbool_t const before_b8x = svbrkb_b_z(pred_b8x, matches_b8x);
        sz_cptr_t const candidate = block + svcntp_b8(pred_b8x, before_b8x);
        if (sz_equal_sve_(candidate, needle, needle_length)) return candidate;
        // Clear the first match and continue.
        matches_b8x = svbic_b_z(pred_b8x, matches_b8x, svpnext_b8(pred_b8x, before_b8x));
    }
    return STRINGZILLA_NULL_CHAR;
}

/**
 *  @brief Forward search comparing up to three probe bytes of the needle in every candidate
 *      position, verifying the whole needle on probe matches when @p verify is set.
 *
 *  One predicated register goes first, as most matches are close and many haystacks are short.
 *  Then two full registers per iteration, testing both at once with a single compare of their
 *  minimum against zero. A predicated register at a time covers the rest.
 */
STRINGZILLA_INLINE sz_cptr_t sz_find_probes_sve_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                 sz_size_t needle_length, sz_size_t const offsets[3], int probes,
                                                 int verify) {
    sz_u8_t const *const text = (sz_u8_t const *)haystack;
    sz_u8_t const *const needle_bytes = (sz_u8_t const *)needle;
    svuint8_t const first_u8x = svdup_n_u8(needle_bytes[offsets[0]]);
    svuint8_t const second_u8x = svdup_n_u8(needle_bytes[offsets[1]]);
    svuint8_t const third_u8x = svdup_n_u8(needle_bytes[offsets[2]]);
    sz_size_t const vector_bytes = svcntb();
    sz_size_t const candidates = haystack_length - needle_length + 1;
    svbool_t const all_b8x = svptrue_b8();
    sz_cptr_t found;

    svbool_t pred_b8x = svwhilelt_b8((sz_u64_t)0, (sz_u64_t)candidates);
    svbool_t matches_b8x = sz_find_probes_match_sve_(pred_b8x, text, offsets, first_u8x, second_u8x, third_u8x, probes);
    if (svptest_any(pred_b8x, matches_b8x) &&
        (!verify ||
         (found = sz_find_probes_resolve_sve_(pred_b8x, matches_b8x, haystack, needle, needle_length, verify))))
        return verify ? found : haystack + svcntp_b8(pred_b8x, svbrkb_b_z(pred_b8x, matches_b8x));
    if (candidates <= vector_bytes) return STRINGZILLA_NULL_CHAR;

    sz_size_t progress = vector_bytes;
    // Near the start, two full registers per iteration, each tested with its own flag-setting
    // compare. The register holding a match is known without extra work, which keeps the latency
    // from a match to the result short: for frequent needles that latency, not the scan
    // throughput, bounds the next search.
    sz_size_t const near_end = candidates < STRINGZILLA_FIND_SVE_NEAR_BYTES ? candidates
                                                                            : STRINGZILLA_FIND_SVE_NEAR_BYTES;
    for (; progress + 2 * vector_bytes <= near_end; progress += 2 * vector_bytes) {
        svbool_t const matches_first_b8x = svcmpeq_n_u8(
            all_b8x,
            sz_find_probes_diff_sve_(all_b8x, text + progress, offsets, first_u8x, second_u8x, third_u8x, probes), 0);
        svbool_t const matches_second_b8x = svcmpeq_n_u8(
            all_b8x,
            sz_find_probes_diff_sve_(all_b8x, text + progress + vector_bytes, offsets, first_u8x, second_u8x, third_u8x,
                                     probes),
            0);
        if (svptest_any(all_b8x, matches_first_b8x)) {
            if (!verify) return haystack + progress + svcntp_b8(all_b8x, svbrkb_b_z(all_b8x, matches_first_b8x));
            if ((found = sz_find_probes_resolve_sve_(all_b8x, matches_first_b8x, haystack + progress, needle,
                                                     needle_length, verify)))
                return found;
        }
        if (svptest_any(all_b8x, matches_second_b8x)) {
            if (!verify)
                return haystack + progress + vector_bytes + svcntp_b8(all_b8x, svbrkb_b_z(all_b8x, matches_second_b8x));
            if ((found = sz_find_probes_resolve_sve_(all_b8x, matches_second_b8x, haystack + progress + vector_bytes,
                                                     needle, needle_length, verify)))
                return found;
        }
    }

    // Further on, throughput matters more: one compare of the minimum of
    // both registers against zero.
    for (; progress + 2 * vector_bytes <= candidates; progress += 2 * vector_bytes) {
        svuint8_t const diff_first_u8x = sz_find_probes_diff_sve_(all_b8x, text + progress, offsets, first_u8x,
                                                                  second_u8x, third_u8x, probes);
        svuint8_t const diff_second_u8x = sz_find_probes_diff_sve_(all_b8x, text + progress + vector_bytes, offsets,
                                                                   first_u8x, second_u8x, third_u8x, probes);
        if (!svptest_any(all_b8x, svcmpeq_n_u8(all_b8x, svmin_u8_x(all_b8x, diff_first_u8x, diff_second_u8x), 0)))
            continue;
        // Pick the first candidate of the two registers without a branch on which of them holds it.
        svbool_t const matches_first_b8x = svcmpeq_n_u8(all_b8x, diff_first_u8x, 0);
        svbool_t const matches_second_b8x = svcmpeq_n_u8(all_b8x, diff_second_u8x, 0);
        sz_size_t const skip_first = svcntp_b8(all_b8x, svbrkb_b_z(all_b8x, matches_first_b8x));
        sz_size_t const skip_second = svcntp_b8(all_b8x, svbrkb_b_z(all_b8x, matches_second_b8x));
        sz_cptr_t const candidate = haystack + progress +
                                    (skip_first < vector_bytes ? skip_first : vector_bytes + skip_second);
        if (!verify || sz_equal_sve_(candidate, needle, needle_length)) return candidate;
        if ((found = sz_find_probes_resolve_sve_(all_b8x, matches_first_b8x, haystack + progress, needle, needle_length,
                                                 verify)))
            return found;
        if ((found = sz_find_probes_resolve_sve_(all_b8x, matches_second_b8x, haystack + progress + vector_bytes,
                                                 needle, needle_length, verify)))
            return found;
    }

    // Fewer than two registers remain: one predicated register, then one full register ending at
    // the last candidate, without a loop, so no trip count to mispredict. The head returned if
    // one register covered every candidate, so the last register starts within the haystack. It
    // may overlap positions already checked, which hold no match, so its first match is still
    // the first one.
    pred_b8x = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)candidates);
    matches_b8x = sz_find_probes_match_sve_(pred_b8x, text + progress, offsets, first_u8x, second_u8x, third_u8x,
                                            probes);
    if (svptest_any(pred_b8x, matches_b8x) &&
        (!verify || (found = sz_find_probes_resolve_sve_(pred_b8x, matches_b8x, haystack + progress, needle,
                                                         needle_length, verify))))
        return verify ? found : haystack + progress + svcntp_b8(pred_b8x, svbrkb_b_z(pred_b8x, matches_b8x));
    progress = candidates - vector_bytes;
    matches_b8x = sz_find_probes_match_sve_(all_b8x, text + progress, offsets, first_u8x, second_u8x, third_u8x,
                                            probes);
    if (svptest_any(all_b8x, matches_b8x) &&
        (!verify || (found = sz_find_probes_resolve_sve_(all_b8x, matches_b8x, haystack + progress, needle,
                                                         needle_length, verify))))
        return verify ? found : haystack + progress + svcntp_b8(all_b8x, svbrkb_b_z(all_b8x, matches_b8x));
    return STRINGZILLA_NULL_CHAR;
}

/** The last candidate of @p matches_b8x in the block at @p block that holds the whole needle, or
 *  NULL. With @p verify false the probes cover the needle and @p matches_b8x must not be empty. */
STRINGZILLA_INLINE sz_cptr_t sz_rfind_probes_resolve_sve_(svbool_t matches_b8x, sz_cptr_t block, sz_cptr_t needle,
                                                          sz_size_t needle_length, int verify) {
    svbool_t const all_b8x = svptrue_b8();
    sz_size_t const last_lane = svcntb() - 1;
    if (!verify) return block + last_lane - svcntp_b8(all_b8x, svbrkb_b_z(all_b8x, svrev_b8(matches_b8x)));
    while (svptest_any(all_b8x, matches_b8x)) {
        svbool_t const after_b8x = svbrkb_b_z(all_b8x, svrev_b8(matches_b8x)); //? Lanes after the last match, reversed.
        sz_cptr_t const candidate = block + last_lane - svcntp_b8(all_b8x, after_b8x);
        if (sz_equal_sve_(candidate, needle, needle_length)) return candidate;
        // Clear the last match and continue.
        matches_b8x = svbic_b_z(all_b8x, matches_b8x, svrev_b8(svpnext_b8(all_b8x, after_b8x)));
    }
    return STRINGZILLA_NULL_CHAR;
}

/** Backward search mirroring @c sz_find_probes_sve_: one predicated register at the end of the
 *  haystack, then two full registers per iteration, then a predicated register at a time down
 *  to the start of the haystack. */
STRINGZILLA_INLINE sz_cptr_t sz_rfind_probes_sve_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                  sz_size_t needle_length, sz_size_t const offsets[3], int probes,
                                                  int verify) {
    sz_u8_t const *const text = (sz_u8_t const *)haystack;
    sz_u8_t const *const needle_bytes = (sz_u8_t const *)needle;
    svuint8_t const first_u8x = svdup_n_u8(needle_bytes[offsets[0]]);
    svuint8_t const second_u8x = svdup_n_u8(needle_bytes[offsets[1]]);
    svuint8_t const third_u8x = svdup_n_u8(needle_bytes[offsets[2]]);
    sz_size_t const vector_bytes = svcntb();
    svbool_t const all_b8x = svptrue_b8();
    sz_cptr_t found;

    // Candidates in `[0, end)` remain unchecked.
    sz_size_t end = haystack_length - needle_length + 1;
    sz_size_t start = end > vector_bytes ? end - vector_bytes : 0;
    svbool_t pred_b8x = svwhilelt_b8((sz_u64_t)0, (sz_u64_t)(end - start));
    svbool_t matches_b8x = sz_find_probes_match_sve_(pred_b8x, text + start, offsets, first_u8x, second_u8x, third_u8x,
                                                     probes);
    if (svptest_any(pred_b8x, matches_b8x) &&
        (!verify ||
         (found = sz_rfind_probes_resolve_sve_(matches_b8x, haystack + start, needle, needle_length, verify))))
        return verify ? found : sz_rfind_probes_resolve_sve_(matches_b8x, haystack + start, needle, needle_length, 0);
    if (!start) return STRINGZILLA_NULL_CHAR;
    end = start;

    // Near the end, two full registers per iteration, each tested with its own flag-setting
    // compare, then the minimum of both, as in `sz_find_probes_sve_`.
    sz_size_t const near_start = end > STRINGZILLA_FIND_SVE_NEAR_BYTES ? end - STRINGZILLA_FIND_SVE_NEAR_BYTES : 0;
    for (; end >= near_start + 2 * vector_bytes; end -= 2 * vector_bytes) {
        svbool_t const matches_high_b8x = svcmpeq_n_u8(
            all_b8x,
            sz_find_probes_diff_sve_(all_b8x, text + end - vector_bytes, offsets, first_u8x, second_u8x, third_u8x,
                                     probes),
            0);
        svbool_t const matches_low_b8x = svcmpeq_n_u8(
            all_b8x,
            sz_find_probes_diff_sve_(all_b8x, text + end - 2 * vector_bytes, offsets, first_u8x, second_u8x, third_u8x,
                                     probes),
            0);
        if (svptest_any(all_b8x, matches_high_b8x) &&
            (found = sz_rfind_probes_resolve_sve_(matches_high_b8x, haystack + end - vector_bytes, needle,
                                                  needle_length, verify)))
            return found;
        if (svptest_any(all_b8x, matches_low_b8x) &&
            (found = sz_rfind_probes_resolve_sve_(matches_low_b8x, haystack + end - 2 * vector_bytes, needle,
                                                  needle_length, verify)))
            return found;
    }
    for (; end >= 2 * vector_bytes; end -= 2 * vector_bytes) {
        svuint8_t const diff_high_u8x = sz_find_probes_diff_sve_(all_b8x, text + end - vector_bytes, offsets, first_u8x,
                                                                 second_u8x, third_u8x, probes);
        svuint8_t const diff_low_u8x = sz_find_probes_diff_sve_(all_b8x, text + end - 2 * vector_bytes, offsets,
                                                                first_u8x, second_u8x, third_u8x, probes);
        if (!svptest_any(all_b8x, svcmpeq_n_u8(all_b8x, svmin_u8_x(all_b8x, diff_high_u8x, diff_low_u8x), 0))) continue;
        svbool_t const matches_high_b8x = svcmpeq_n_u8(all_b8x, diff_high_u8x, 0);
        svbool_t const matches_low_b8x = svcmpeq_n_u8(all_b8x, diff_low_u8x, 0);
        sz_size_t const skip_high = svcntp_b8(all_b8x, svbrkb_b_z(all_b8x, svrev_b8(matches_high_b8x)));
        sz_size_t const skip_low = svcntp_b8(all_b8x, svbrkb_b_z(all_b8x, svrev_b8(matches_low_b8x)));
        sz_cptr_t const candidate = haystack + end - 1 -
                                    (skip_high < vector_bytes ? skip_high : vector_bytes + skip_low);
        if (!verify || sz_equal_sve_(candidate, needle, needle_length)) return candidate;
        if ((found = sz_rfind_probes_resolve_sve_(matches_high_b8x, haystack + end - vector_bytes, needle,
                                                  needle_length, verify)))
            return found;
        if ((found = sz_rfind_probes_resolve_sve_(matches_low_b8x, haystack + end - 2 * vector_bytes, needle,
                                                  needle_length, verify)))
            return found;
    }

    // Fewer than two registers remain: two predicated steps without a loop,
    // so there is no trip count to mispredict.
    for (int step = 0; step != 2; ++step, end = start) {
        start = end > vector_bytes ? end - vector_bytes : 0;
        pred_b8x = svwhilelt_b8((sz_u64_t)0, (sz_u64_t)(end - start));
        matches_b8x = sz_find_probes_match_sve_(pred_b8x, text + start, offsets, first_u8x, second_u8x, third_u8x,
                                                probes);
        if (svptest_any(pred_b8x, matches_b8x) &&
            (!verify ||
             (found = sz_rfind_probes_resolve_sve_(matches_b8x, haystack + start, needle, needle_length, verify))))
            return verify ? found
                          : sz_rfind_probes_resolve_sve_(matches_b8x, haystack + start, needle, needle_length, 0);
    }
    return STRINGZILLA_NULL_CHAR;
}

STRINGZILLA_INLINE sz_cptr_t sz_find_byte_sve_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    sz_size_t const offsets[3] = {0, 0, 0};
    if (!haystack_length) return STRINGZILLA_NULL_CHAR;
    return sz_find_probes_sve_(haystack, haystack_length, needle, 1, offsets, 1, 0);
}

STRINGZILLA_INLINE sz_cptr_t sz_rfind_byte_sve_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    sz_size_t const offsets[3] = {0, 0, 0};
    if (!haystack_length) return STRINGZILLA_NULL_CHAR;
    return sz_rfind_probes_sve_(haystack, haystack_length, needle, 1, offsets, 1, 0);
}

STRINGZILLA_INLINE sz_cptr_t sz_find_sve_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length) {
    // Empty needle matches at the start, like `strstr`.
    if (!needle_length) return haystack;
    if (haystack_length < needle_length) return STRINGZILLA_NULL_CHAR;

    // Needles of up to 3 bytes are probed whole and need no verification.
    if (needle_length <= 3) {
        sz_size_t const offsets[3] = {0, needle_length > 1, needle_length - 1};
        if (needle_length == 1) return sz_find_probes_sve_(haystack, haystack_length, needle, 1, offsets, 1, 0);
        if (needle_length == 2) return sz_find_probes_sve_(haystack, haystack_length, needle, 2, offsets, 2, 0);
        return sz_find_probes_sve_(haystack, haystack_length, needle, 3, offsets, 3, 0);
    }

    // For longer needles we first pick "anomalies" (i.e. informative offsets).
    sz_size_t offsets[3];
    sz_locate_needle_anomalies_(needle, needle_length, &offsets[0], &offsets[1], &offsets[2]);
    return sz_find_probes_sve_(haystack, haystack_length, needle, needle_length, offsets, 3, 1);
}

STRINGZILLA_INLINE sz_cptr_t sz_rfind_sve_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                           sz_size_t needle_length) {
    // Empty needle matches at the end.
    if (!needle_length) return haystack + haystack_length;
    if (haystack_length < needle_length) return STRINGZILLA_NULL_CHAR;

    // Needles of up to 3 bytes are probed whole and need no verification.
    if (needle_length <= 3) {
        sz_size_t const offsets[3] = {0, needle_length > 1, needle_length - 1};
        if (needle_length == 1) return sz_rfind_probes_sve_(haystack, haystack_length, needle, 1, offsets, 1, 0);
        if (needle_length == 2) return sz_rfind_probes_sve_(haystack, haystack_length, needle, 2, offsets, 2, 0);
        return sz_rfind_probes_sve_(haystack, haystack_length, needle, 3, offsets, 3, 0);
    }

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offsets[3];
    sz_locate_needle_anomalies_(needle, needle_length, &offsets[0], &offsets[1], &offsets[2]);
    return sz_rfind_probes_sve_(haystack, haystack_length, needle, needle_length, offsets, 3, 1);
}

#if STRINGZILLA_TARGET_SVE

STRINGZILLA_API sz_status_t sz_find_byte_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                             sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_find_byte_sve_(haystack, haystack_length, needle);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_rfind_byte_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                              sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_rfind_byte_sve_(haystack, haystack_length, needle);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_find_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                        sz_size_t needle_length, sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    // Graviton 5: the scalable substring search only outruns NEON on registers wider than 128 bits.
    *match = svcntb() <= 16 ? sz_find_neon_(haystack, haystack_length, needle, needle_length)
                            : sz_find_sve_(haystack, haystack_length, needle, needle_length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_rfind_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                         sz_size_t needle_length, sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    // Graviton 5: the scalable substring search only outruns NEON on registers wider than 128 bits.
    *match = svcntb() <= 16 ? sz_rfind_neon_(haystack, haystack_length, needle, needle_length)
                            : sz_rfind_sve_(haystack, haystack_length, needle, needle_length);
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_SVE

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // STRINGZILLA_ARCH_ARM64_SVE_
#endif // STRINGZILLA_ARCH_ARM64_

#endif // STRINGZILLA_FIND_SVE_H_
