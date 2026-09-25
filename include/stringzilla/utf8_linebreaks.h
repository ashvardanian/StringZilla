/**
 *  @file include/stringzilla/utf8_linebreaks.h
 *  @author Ash Vardanian
 *  @date June 20, 2026
 *  @brief Hardware-accelerated UAX-14 line break segmentation.
 */
#ifndef STRINGZILLA_UTF8_LINEBREAKS_H_
#define STRINGZILLA_UTF8_LINEBREAKS_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Segment UTF-8 text into UAX-14 line-break opportunities in one pass (dispatch function).
 *
 *  Walks the whole input left-to-right and writes one entry per line segment into two parallel
 *  output arrays: `line_starts[i]` is the byte offset of the i-th segment and `line_lengths[i]`
 *  its byte length. Segments are the spans between consecutive UAX-14 break opportunities, so a
 *  single call segments the entire input without the caller having to loop and restart a scan
 *  for every break.
 *
 *  This emits every wrap opportunity (both the mandatory LB4/LB5 hard breaks and the allowed
 *  soft-wrap points). To split only on the hard breaks (the @c str.splitlines behavior) use
 *  @c sz_utf8_newlines instead, which enumerates exactly the LB4/LB5 break positions.
 *
 *  @param[in] text UTF-8 encoded text.
 *  @param[in] length Byte length of @p text.
 *  @param[out] line_starts Segment byte offsets, at least @p lines_capacity entries.
 *  @param[out] line_lengths Segment byte lengths, at least @p lines_capacity entries.
 *  @param[in] lines_capacity Capacity of the output arrays, in entries.
 *  @param[out] bytes_consumed Optional byte offset up to which the input was segmented: @p length
 *      when everything fit, else the start of the first segment that did not fit (a break
 *      opportunity), so the caller may resume from `text + *bytes_consumed`.
 *  @return Number of segments written (at most @p lines_capacity).
 *
 *  @note No zero-length segments are emitted; @p length == 0 returns 0.
 *  @note Line segmentation is forward-only.
 */
SZ_API_RUNTIME sz_size_t sz_utf8_linebreaks(         //
    sz_cptr_t text, sz_size_t length,                //
    sz_size_t *line_starts, sz_size_t *line_lengths, //
    sz_size_t lines_capacity, sz_size_t *bytes_consumed);

#pragma endregion

#pragma region Platform Specific Backends

/** @copydoc sz_utf8_linebreaks */
SZ_API_COMPTIME sz_size_t sz_utf8_linebreaks_serial(sz_cptr_t text, sz_size_t length, sz_size_t *line_starts,
                                                    sz_size_t *line_lengths, sz_size_t lines_capacity,
                                                    sz_size_t *bytes_consumed);

#if SZ_USE_HASWELL
/** @copydoc sz_utf8_linebreaks */
SZ_API_COMPTIME sz_size_t sz_utf8_linebreaks_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *line_starts,
                                                     sz_size_t *line_lengths, sz_size_t lines_capacity,
                                                     sz_size_t *bytes_consumed);
#endif

#if SZ_USE_NEON
/** @copydoc sz_utf8_linebreaks */
SZ_API_COMPTIME sz_size_t sz_utf8_linebreaks_neon(sz_cptr_t text, sz_size_t length, sz_size_t *line_starts,
                                                  sz_size_t *line_lengths, sz_size_t lines_capacity,
                                                  sz_size_t *bytes_consumed);
#endif

#if SZ_USE_ICELAKE
/** @copydoc sz_utf8_linebreaks */
SZ_API_COMPTIME sz_size_t sz_utf8_linebreaks_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *line_starts,
                                                     sz_size_t *line_lengths, sz_size_t lines_capacity,
                                                     sz_size_t *bytes_consumed);
#endif

#if SZ_USE_SVE2
/** @copydoc sz_utf8_linebreaks */
SZ_API_COMPTIME sz_size_t sz_utf8_linebreaks_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *starts,
                                                  sz_size_t *lengths, sz_size_t capacity, sz_size_t *bytes_consumed);
#endif

#pragma endregion

/*  Implementation Section - each ISA backend lives in its own header, included serial-first. */
#include "stringzilla/utf8_linebreaks/serial.h"
#include "stringzilla/utf8_linebreaks/haswell.h"
#include "stringzilla/utf8_linebreaks/neon.h"
#include "stringzilla/utf8_linebreaks/icelake.h"
#include "stringzilla/utf8_linebreaks/sve2.h"

#pragma region Dynamic Dispatch

#if !SZ_DYNAMIC_DISPATCH

SZ_API_RUNTIME sz_size_t sz_utf8_linebreaks(sz_cptr_t text, sz_size_t length, sz_size_t *line_starts,
                                            sz_size_t *line_lengths, sz_size_t lines_capacity,
                                            sz_size_t *bytes_consumed) {
#if SZ_USE_ICELAKE
    return sz_utf8_linebreaks_icelake(text, length, line_starts, line_lengths, lines_capacity, bytes_consumed);
#elif SZ_USE_HASWELL
    return sz_utf8_linebreaks_haswell(text, length, line_starts, line_lengths, lines_capacity, bytes_consumed);
#elif SZ_USE_SVE2 && SZ_SVE_WIDER_THAN_NEON_
    return sz_utf8_linebreaks_sve2(text, length, line_starts, line_lengths, lines_capacity, bytes_consumed);
#elif SZ_USE_NEON
    return sz_utf8_linebreaks_neon(text, length, line_starts, line_lengths, lines_capacity, bytes_consumed);
#else
    return sz_utf8_linebreaks_serial(text, length, line_starts, line_lengths, lines_capacity, bytes_consumed);
#endif
}

#endif // !SZ_DYNAMIC_DISPATCH

#pragma endregion

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_LINEBREAKS_H_
