/**
 *  @file include/stringzilla/regex.h
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Multi-pattern regular expressions: one compiled pair of automata over many haystacks.
 *
 *  A vocabulary of RE2-syntax patterns compiles once into two DFAs over one byte-class map, and
 *  every haystack then streams through them whatever the pattern count. Matches are leftmost-first
 *  across the union of patterns, ties going to the lower pattern index, which is what Rust's
 *  @c regex, RE2, Python's @c re and Arrow report.
 *
 *  The start-finding automaton walks a window right to left and marks every position a match can
 *  start at; the end-finding one walks forward from one marked start until it dies. A search takes
 *  the first marked start at or past its cursor, walks to its end, and moves the cursor there, so
 *  the bytes it walks are the window once plus each match and its lookahead. A window that does not
 *  end the haystack needs its entry state: a cased DFA walk on CPUs guesses it from the bytes right
 *  of its edge and settles the guess against the window right of it once both are walked; other
 *  walks prime it, every state walking the bytes right of the edge together until one remains.
 *
 *  @section regex_api Public API
 *
 *  - @ref sz_regex_engine_init → compiles a vocabulary for the best capability of a mask;
 *  - @ref sz_regex_counts → how many matches each haystack holds;
 *  - @ref sz_regex_find → every match, located by haystack, pattern and byte span;
 *  - @ref sz_regex_replace → the haystacks rewritten with a replacement per pattern, naming groups;
 *  - @ref sz_regex_extract → the groups of each haystack's first match of one pattern, as matches;
 *  - @ref sz_regex_find_kernel → the kernel any of them would run, for a caller that keeps it.
 *
 *  @section regex_syntax Syntax
 *
 *  Patterns are RE2's: UTF-8 literals, escaped metacharacters, `\n \t \r \f \v \a`, `\xHH` and
 *  `\x{H…}` naming codepoints, `.` for any codepoint but a line feed, bracketed classes with
 *  ranges, negation and ASCII POSIX names, the ASCII Perl classes `\d \D \w \W \s \S`, the anchors
 *  `^ $ \A \z`, the ASCII word boundaries `\b \B`, plain, non-capturing and named groups, `|`,
 *  greedy and lazy `* + ? {n} {n,} {n,m}`, and the flags `(?s)`, `(?m)` and `(?U)`, set for the
 *  rest of a group or scoped as `(?s:…)`. Plain and named groups capture, numbered from one per
 *  pattern by their opening parenthesis, as @ref sz_regex_engine_t::patterns_groups records.
 *
 *  Beyond RE2, the lookaheads `(?=…)` and `(?!…)` and the lookbehinds `(?<=…)` and `(?<!…)` test
 *  whether their body matches from or up to a position, whatever its width, and nest freely; a body
 *  holds no capturing group, and a vocabulary holds at most 26 of them. Up to six, nested ones
 *  included, become columns of both DFAs; a vocabulary of more walks slower automata instead.
 *
 *  Refused with @c sz_invalid_syntax_k: backreferences, a capturing group inside a lookaround or
 *  more lookarounds than that, inline `(?i)`, `\p` and `\P`, possessive and atomic forms, `\Z`,
 *  `\G`, `\C`, `\Q`, malformed UTF-8, unbalanced groups, more than a thousand nested counted
 *  repetitions, and the forms RE2 and `regex-syntax` read apart: a brace that opens no counted
 *  repetition, a quantifier on a quantifier, and a class holding a nested class or the `&&`, `--`
 *  and `~~` operators.
 *
 *  @section regex_uncased Case Folding
 *
 *  An @c sz_uncased_k engine folds its patterns once and every haystack as it walks, with the
 *  full `CaseFolding.txt` folding @c substrings uses, so `ß` matches @c ss and the Kelvin sign
 *  matches @c k. A class holds the folds of its members, a member folding to several codepoints
 *  joining as an alternative after it, so `[ß]` matches @c ss; a negated class, `.` among them,
 *  complements the single-codepoint folds of its members, so it matches one folded codepoint.
 *  Anchors, `\b` and lookarounds are judged on the folded text, lookarounds only between whole
 *  codepoints, as none holds inside the folds of one. A match starting or ending inside the folds
 *  of one codepoint snaps outward to the whole codepoint, the search resuming past it, and an
 *  empty match is never reported there.
 */
#ifndef STRINGZILLA_REGEX_H_
#define STRINGZILLA_REGEX_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`
#include "stringzilla/regex/serial.h" // `sz_regex_engine_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Compiles @p patterns for the best capability of @p capabilities that has an init kernel.
 *
 *  @param[out] engine Left untouched unless the call succeeds.
 *  @param[in] patterns The vocabulary, read on the host; a pattern's index is the @c needle_index
 *      its matches report, and the lower index wins between patterns matching at one position.
 *  @param[in] case_sensitivity Whether patterns match bytes as written, or both sides are folded
 *      first, as the section on case folding describes.
 *  @param[out] malformed_index Where the index of the first refused pattern is written, only when
 *      the call returns @c sz_invalid_syntax_k; may be @c STRINGZILLA_NULL.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] allocator Where both blocks come from, or @c STRINGZILLA_NULL for the tier's default:
 *      the heap on the CPU, or the device's unified allocator. Stored by value, so
 *      @ref sz_regex_engine_free needs none and cannot get the wrong one.
 *  @param[in] stream Null on the CPU, or the device stream the call is enqueued on.
 *  @return @c sz_success_k once the vocabulary compiled; @c sz_invalid_syntax_k for an empty
 *      vocabulary or a pattern outside the syntax above; @c sz_missing_kernel_k when no capability
 *      of the mask has an init; @c sz_bad_alloc_k if memory allocation failed;
 *      @c sz_overflow_risk_k if an NFA exceeds a 32-bit state id; or
 *      @c sz_device_memory_mismatch_k for a @p stream or @p allocator its capability cannot use.
 */
STRINGZILLA_API sz_status_t sz_regex_engine_init(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                 sz_case_sensitivity_t case_sensitivity, sz_size_t *malformed_index,
                                                 sz_capability_t capabilities, sz_allocator_t *allocator,
                                                 sz_stream_t stream);

/** Returns both of the engine's blocks to the allocator that built them, once the work queued on
 *  @p stream is done with them, emptying @p engine. */
STRINGZILLA_API void sz_regex_engine_free(sz_regex_engine_t *engine, sz_stream_t stream);

/**
 *  @brief Counts the vocabulary's leftmost-first matches in every haystack, one count per haystack.
 *
 *  @param[in] engine A compiled vocabulary, whose round scratch grows to the longest haystack.
 *  @param[in] haystacks The texts to search, in any layout; their bytes are read in place.
 *      Malformed UTF-8 is searched as is: no class or literal matches a malformed byte.
 *  @param[out] counts The per-haystack counts, haystack h at `counts[h * stride]`.
 *  @param[in] counts_stride Entries from one haystack's count to the next, at least one, so a
 *      strided call writes one column of a @b [haystacks, vocabularies] feature matrix.
 *  @param[in] stream Null on the CPU, or the device stream the call is enqueued on.
 *  @return @c sz_success_k once the haystacks were counted; @c sz_unexpected_dimensions_k if
 *      @p counts_stride is zero; @c sz_missing_kernel_k for an empty engine;
 *      @c sz_bad_alloc_k if the round scratch could not grow; or
 *      @c sz_device_memory_mismatch_k for a @p stream the engine's capability cannot use.
 */
STRINGZILLA_API sz_status_t sz_regex_counts(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                            sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream);

/**
 *  @brief Reports every leftmost-first match of the vocabulary in every haystack.
 *
 *  Within a haystack, matches share no bytes and come in ascending order of start; an empty match
 *  is never reported where the previous match ended, as Rust's @c find_iter does, nor before a
 *  UTF-8 continuation byte, so none splits a codepoint.
 *
 *  @param[in] engine A compiled vocabulary.
 *  @param[in] haystacks The texts to search; their bytes are read in place and never copied.
 *  @param[out] matches Room for @p matches_capacity matches, ascending by haystack, then by start;
 *      @c STRINGZILLA_NULL together with a zero capacity makes the call a pure size query.
 *  @param[in] matches_capacity Entries @p matches holds; only the first that many are written.
 *  @param[out] matches_offsets One boundary per haystack into the full list of matches plus one,
 *      the last being the total; true whether or not the matches fit, which sizes the next call.
 *  @param[in] stream Null on the CPU, or the device stream the call is enqueued on.
 *  @return @c sz_success_k once every match was counted and those that fit were written;
 *      @c sz_missing_kernel_k for an empty engine; @c sz_bad_alloc_k if the round scratch could not
 *      grow; or @c sz_device_memory_mismatch_k for a @p stream the engine's capability cannot use.
 */
STRINGZILLA_API sz_status_t sz_regex_find(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                          sz_match_t *matches, sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                          sz_stream_t stream);

/**
 *  @brief Rewrites every haystack, substituting each match with its pattern's replacement,
 *      into one @p target.
 *
 *  A replacement is copied as written, except that `$$` stands for one dollar and `$N`, `${N}` and
 *  `${name}` for the text of that group of the match's pattern, `$0` being the whole match; a group
 *  that took no part in the match or does not exist is empty, and a dollar starting no reference
 *  stands for itself, so `$1a` is group one followed by @c a. Groups are read only for the matches
 *  whose replacement names them, by a PikeVM over each such match's own span.
 *
 *  @param[in] engine A compiled vocabulary.
 *  @param[in] haystacks The texts to rewrite; their bytes are read in place and never copied.
 *  @param[in] replacements One replacement per pattern, indexed by pattern; an empty one deletes.
 *  @param[out] target Room for @p target_capacity bytes, or @c STRINGZILLA_NULL with zero capacity
 *      to size only.
 *  @param[in] target_capacity Bytes @p target holds; only the first that many are written.
 *  @param[out] offsets One rewritten boundary per haystack plus one, the last being the total;
 *      true whether or not the target held the result, which sizes the next call.
 *  @param[in] stream Null on the CPU, or the device stream the call is enqueued on.
 *  @return @c sz_success_k once every haystack was rewritten; @c sz_unexpected_dimensions_k if
 *      @p replacements lacks one entry per pattern; @c sz_missing_kernel_k for an empty engine;
 *      @c sz_bad_alloc_k if the round scratch could not grow; or @c sz_device_memory_mismatch_k for
 *      a @p stream the engine's capability cannot use.
 */
STRINGZILLA_API sz_status_t sz_regex_replace(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                             sz_strings_view_t const *replacements, sz_ptr_t target,
                                             sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream);

/**
 *  @brief Locates the groups of each haystack's first match of @p pattern_index, as Pandas
 *      `str.extract` reads them.
 *
 *  A haystack with such a match gets one row for the whole match, @c needle_index zero, then one
 *  row per group that took part, @c needle_index naming the group; a haystack without one gets no
 *  rows, so a group that took no part is told apart from an empty one. The first match of a pattern
 *  is the first the vocabulary reports with that @c needle_index.
 *
 *  @param[in] engine A compiled vocabulary.
 *  @param[in] haystacks The texts to search; their bytes are read in place and never copied.
 *  @param[in] pattern_index The pattern whose groups to locate.
 *  @param[out] groups Room for @p groups_capacity rows, or @c STRINGZILLA_NULL with zero capacity
 *      to size only.
 *  @param[in] groups_capacity Rows @p groups holds; only the first that many are written.
 *  @param[out] groups_offsets One boundary per haystack plus one, the last being the total;
 *      true whether or not @p groups held them, which sizes the next call.
 *  @param[in] stream Null on the CPU, or the device stream the call is enqueued on.
 *  @return @c sz_success_k once every row was counted and those that fit were written;
 *      @c sz_unexpected_dimensions_k if @p pattern_index is past the vocabulary;
 *      @c sz_missing_kernel_k for an empty engine; @c sz_bad_alloc_k if the round scratch could not
 *      grow; or @c sz_device_memory_mismatch_k for a @p stream the engine's capability cannot use.
 */
STRINGZILLA_API sz_status_t sz_regex_extract(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                             sz_size_t pattern_index, sz_match_t *groups, sz_size_t groups_capacity,
                                             sz_size_t *groups_offsets, sz_stream_t stream);

/** @copydoc sz_regex_engine_init */
STRINGZILLA_API sz_status_t sz_regex_engine_init_serial(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                        sz_case_sensitivity_t case_sensitivity,
                                                        sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                        sz_stream_t stream);
/** @copydoc sz_regex_counts */
STRINGZILLA_API sz_status_t sz_regex_counts_serial(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                   sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream);
/** @copydoc sz_regex_find */
STRINGZILLA_API sz_status_t sz_regex_find_serial(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                 sz_match_t *matches, sz_size_t matches_capacity,
                                                 sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_regex_replace */
STRINGZILLA_API sz_status_t sz_regex_replace_serial(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                    sz_strings_view_t const *replacements, sz_ptr_t target,
                                                    sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream);
/** @copydoc sz_regex_extract */
STRINGZILLA_API sz_status_t sz_regex_extract_serial(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                    sz_size_t pattern_index, sz_match_t *groups,
                                                    sz_size_t groups_capacity, sz_size_t *groups_offsets,
                                                    sz_stream_t stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_regex_engine_init */
STRINGZILLA_API sz_status_t sz_regex_engine_init_haswell(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                         sz_case_sensitivity_t case_sensitivity,
                                                         sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                         sz_stream_t stream);
/** @copydoc sz_regex_counts */
STRINGZILLA_API sz_status_t sz_regex_counts_haswell(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                    sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream);
/** @copydoc sz_regex_find */
STRINGZILLA_API sz_status_t sz_regex_find_haswell(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                  sz_match_t *matches, sz_size_t matches_capacity,
                                                  sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_regex_replace */
STRINGZILLA_API sz_status_t sz_regex_replace_haswell(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                     sz_strings_view_t const *replacements, sz_ptr_t target,
                                                     sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream);
/** @copydoc sz_regex_extract */
STRINGZILLA_API sz_status_t sz_regex_extract_haswell(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                     sz_size_t pattern_index, sz_match_t *groups,
                                                     sz_size_t groups_capacity, sz_size_t *groups_offsets,
                                                     sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_regex_engine_init */
STRINGZILLA_API sz_status_t sz_regex_engine_init_icelake(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                         sz_case_sensitivity_t case_sensitivity,
                                                         sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                         sz_stream_t stream);
/** @copydoc sz_regex_counts */
STRINGZILLA_API sz_status_t sz_regex_counts_icelake(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                    sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream);
/** @copydoc sz_regex_find */
STRINGZILLA_API sz_status_t sz_regex_find_icelake(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                  sz_match_t *matches, sz_size_t matches_capacity,
                                                  sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_regex_replace */
STRINGZILLA_API sz_status_t sz_regex_replace_icelake(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                     sz_strings_view_t const *replacements, sz_ptr_t target,
                                                     sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream);
/** @copydoc sz_regex_extract */
STRINGZILLA_API sz_status_t sz_regex_extract_icelake(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                     sz_size_t pattern_index, sz_match_t *groups,
                                                     sz_size_t groups_capacity, sz_size_t *groups_offsets,
                                                     sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_regex_engine_init */
STRINGZILLA_API sz_status_t sz_regex_engine_init_neon(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                      sz_case_sensitivity_t case_sensitivity,
                                                      sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                      sz_stream_t stream);
/** @copydoc sz_regex_counts */
STRINGZILLA_API sz_status_t sz_regex_counts_neon(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                 sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream);
/** @copydoc sz_regex_find */
STRINGZILLA_API sz_status_t sz_regex_find_neon(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                               sz_match_t *matches, sz_size_t matches_capacity,
                                               sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_regex_replace */
STRINGZILLA_API sz_status_t sz_regex_replace_neon(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                  sz_strings_view_t const *replacements, sz_ptr_t target,
                                                  sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream);
/** @copydoc sz_regex_extract */
STRINGZILLA_API sz_status_t sz_regex_extract_neon(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                  sz_size_t pattern_index, sz_match_t *groups,
                                                  sz_size_t groups_capacity, sz_size_t *groups_offsets,
                                                  sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_CUDA
/** @copydoc sz_regex_engine_init */
STRINGZILLA_API sz_status_t sz_regex_engine_init_cuda(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                      sz_case_sensitivity_t case_sensitivity,
                                                      sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                      sz_stream_t stream);
/** @copydoc sz_regex_counts */
STRINGZILLA_API sz_status_t sz_regex_counts_cuda(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                 sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream);
/** @copydoc sz_regex_find */
STRINGZILLA_API sz_status_t sz_regex_find_cuda(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                               sz_match_t *matches, sz_size_t matches_capacity,
                                               sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_regex_replace */
STRINGZILLA_API sz_status_t sz_regex_replace_cuda(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                  sz_strings_view_t const *replacements, sz_ptr_t target,
                                                  sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream);
/** @copydoc sz_regex_extract */
STRINGZILLA_API sz_status_t sz_regex_extract_cuda(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                  sz_size_t pattern_index, sz_match_t *groups,
                                                  sz_size_t groups_capacity, sz_size_t *groups_offsets,
                                                  sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ROCM
/** @copydoc sz_regex_engine_init */
STRINGZILLA_API sz_status_t sz_regex_engine_init_rocm(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                      sz_case_sensitivity_t case_sensitivity,
                                                      sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                      sz_stream_t stream);
/** @copydoc sz_regex_counts */
STRINGZILLA_API sz_status_t sz_regex_counts_rocm(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                 sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream);
/** @copydoc sz_regex_find */
STRINGZILLA_API sz_status_t sz_regex_find_rocm(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                               sz_match_t *matches, sz_size_t matches_capacity,
                                               sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_regex_replace */
STRINGZILLA_API sz_status_t sz_regex_replace_rocm(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                  sz_strings_view_t const *replacements, sz_ptr_t target,
                                                  sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream);
/** @copydoc sz_regex_extract */
STRINGZILLA_API sz_status_t sz_regex_extract_rocm(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                  sz_size_t pattern_index, sz_match_t *groups,
                                                  sz_size_t groups_capacity, sz_size_t *groups_offsets,
                                                  sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_METAL
/** @copydoc sz_regex_engine_init */
STRINGZILLA_API sz_status_t sz_regex_engine_init_metal(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                       sz_case_sensitivity_t case_sensitivity,
                                                       sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                       sz_stream_t stream);
/** @copydoc sz_regex_counts */
STRINGZILLA_API sz_status_t sz_regex_counts_metal(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                  sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream);
/** @copydoc sz_regex_find */
STRINGZILLA_API sz_status_t sz_regex_find_metal(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                sz_match_t *matches, sz_size_t matches_capacity,
                                                sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_regex_replace */
STRINGZILLA_API sz_status_t sz_regex_replace_metal(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                   sz_strings_view_t const *replacements, sz_ptr_t target,
                                                   sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream);
/** @copydoc sz_regex_extract */
STRINGZILLA_API sz_status_t sz_regex_extract_metal(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                   sz_size_t pattern_index, sz_match_t *groups,
                                                   sz_size_t groups_capacity, sz_size_t *groups_offsets,
                                                   sz_stream_t stream);
#endif

/**
 *  @brief Finds the regex kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_regex_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                 sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion Core API

#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/regex/haswell.h"
#include "stringzilla/regex/icelake.h"
#include "stringzilla/regex/neon.h"
#include "stringzilla/regex/cuda.cuh"
#include "stringzilla/regex/rocm.cuh"
#include "stringzilla/regex/metal.h"
#endif // STRINGZILLA_HEADER_ONLY

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_regex_engine_init(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                 sz_case_sensitivity_t case_sensitivity, sz_size_t *malformed_index,
                                                 sz_capability_t capabilities, sz_allocator_t *allocator,
                                                 sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(patterns), sz_unused_(case_sensitivity), sz_unused_(malformed_index),
        sz_unused_(capabilities), sz_unused_(allocator), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API void sz_regex_engine_free(sz_regex_engine_t *engine, sz_stream_t stream) {
    sz_regex_engine_free_(engine, stream);
}

STRINGZILLA_API sz_status_t sz_regex_counts(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                            sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(haystacks), sz_unused_(counts), sz_unused_(counts_stride), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_regex_find(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                          sz_match_t *matches, sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                          sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(haystacks), sz_unused_(matches), sz_unused_(matches_capacity),
        sz_unused_(matches_offsets), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_regex_replace(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                             sz_strings_view_t const *replacements, sz_ptr_t target,
                                             sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(haystacks), sz_unused_(replacements), sz_unused_(target),
        sz_unused_(target_capacity), sz_unused_(offsets), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_regex_extract(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                             sz_size_t pattern_index, sz_match_t *groups, sz_size_t groups_capacity,
                                             sz_size_t *groups_offsets, sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(haystacks), sz_unused_(pattern_index), sz_unused_(groups),
        sz_unused_(groups_capacity), sz_unused_(groups_offsets), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_regex_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                 sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_REGEX_H_
