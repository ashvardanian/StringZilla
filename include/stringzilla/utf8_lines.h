/**
 *  @brief Hardware-accelerated UAX-14 line break segmentation.
 *  @file utf8_lines.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_LINES_H_
#define STRINGZILLA_UTF8_LINES_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Segment UTF-8 text into UAX-14 line-break opportunities in a single pass (dispatch function).
 *
 *  Walks the whole input left-to-right and writes one entry per line segment into two parallel output arrays:
 *  `line_starts[i]` is the byte offset of the i-th segment and `line_lengths[i]` its byte length. Segments are
 *  the spans between consecutive UAX-14 break opportunities, so a single call segments the entire input without
 *  the caller having to loop and restart a scan for every break.
 *
 *  Unlike the word/grapheme/sentence kernels, line break distinguishes @b mandatory breaks (UAX-14 rules LB4/LB5
 *  - hard breaks after BK/CR/LF/NL that the caller must honor) from allowed wrap opportunities. `mandatory[i]`
 *  is set to a non-zero flag when the break @b ending segment @p i (i.e. the boundary at
 *  `line_starts[i] + line_lengths[i]`) is mandatory, and zero when it is merely an allowed wrap point.
 *
 *  @param text UTF-8 encoded text.
 *  @param length Byte length of @p text.
 *  @param line_starts Output array of segment byte offsets (at least @p lines_capacity entries).
 *  @param line_lengths Output array of segment byte lengths (at least @p lines_capacity entries).
 *  @param mandatory Output array of mandatory-break flags (at least @p lines_capacity entries): non-zero marks a
 *         UAX-14 LB4/LB5 hard break, zero an allowed wrap opportunity.
 *  @param lines_capacity Capacity of the output arrays, in entries.
 *  @param bytes_consumed Optional output: byte offset up to which the input was segmented. Equals @p length
 *         when everything fit; otherwise it is the start of the first segment that did not fit (a break
 *         opportunity), so the caller may resume from @c text+*bytes_consumed.
 *  @return Number of segments written (at most @p lines_capacity).
 *
 *  @note No zero-length segments are emitted; @p length == 0 returns 0. Line segmentation is forward-only.
 */
SZ_DYNAMIC sz_size_t sz_utf8_line_find_boundaries(                       //
    sz_cptr_t text, sz_size_t length,                                    //
    sz_size_t *line_starts, sz_size_t *line_lengths, sz_u8_t *mandatory, //
    sz_size_t lines_capacity, sz_size_t *bytes_consumed);

#pragma endregion

#pragma region Platform-Specific Backends

/** @copydoc sz_utf8_line_find_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_line_find_boundaries_serial(sz_cptr_t text, sz_size_t length, sz_size_t *line_starts,
                                                        sz_size_t *line_lengths, sz_u8_t *mandatory,
                                                        sz_size_t lines_capacity, sz_size_t *bytes_consumed);

#if SZ_USE_ICELAKE
/** @copydoc sz_utf8_line_find_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_line_find_boundaries_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *line_starts,
                                                         sz_size_t *line_lengths, sz_u8_t *mandatory,
                                                         sz_size_t lines_capacity, sz_size_t *bytes_consumed);
#endif

#pragma endregion

/*  Implementation Section - each ISA backend lives in its own header, included serial-first. */
#include "stringzilla/utf8_lines/serial.h"
#include "stringzilla/utf8_lines/icelake.h"

#pragma region Dynamic Dispatch

#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_size_t sz_utf8_line_find_boundaries(sz_cptr_t text, sz_size_t length, sz_size_t *line_starts,
                                                  sz_size_t *line_lengths, sz_u8_t *mandatory, sz_size_t lines_capacity,
                                                  sz_size_t *bytes_consumed) {
#if SZ_USE_ICELAKE
    return sz_utf8_line_find_boundaries_icelake(text, length, line_starts, line_lengths, mandatory, lines_capacity,
                                                bytes_consumed);
#else
    return sz_utf8_line_find_boundaries_serial(text, length, line_starts, line_lengths, mandatory, lines_capacity,
                                               bytes_consumed);
#endif
}

#endif // !SZ_DYNAMIC_DISPATCH

#pragma endregion

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_LINES_H_
