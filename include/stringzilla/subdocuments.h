/**
 *  @file include/stringzilla/subdocuments.h
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Path queries: one compiled engine pulling what many paths select out of many documents.
 *
 *  A vocabulary of paths compiles once into steps, one per segment, and every document is then
 *  walked once, whatever the path count, advancing every path together as a bitset of live steps.
 *  The walk never builds a tree. In JSON it classifies each 64-byte block into masks of structural
 *  bytes, visits only those inside containers a live step can enter, and steps over every other
 *  container by counting its brackets a block at a time. In XML, whose lexing needs no state
 *  across blocks, it searches for the one byte its lexer needs next, `<` in character data, and
 *  reads only the tags.
 *
 *  A match is the subdocument's exact bytes in its document: in JSON a string with its quotes, a
 *  number or literal without the blank space around it, or an object or array through its closing
 *  bracket; in XML an element from `<` through its end tag, an attribute's value without its
 *  quotes, or a text node's character data and CDATA sections. A caller reads it in place or hands
 *  it to another parser.
 *
 *  @section subdocuments_api Public API
 *
 *  - @ref sz_subdocuments_engine_init → compiles a path vocabulary for a mask's best capability;
 *  - @ref sz_subdocuments_contains → which documents hold any such node, one bit each;
 *  - @ref sz_subdocuments_counts → how many nodes each document holds that some path selects;
 *  - @ref sz_subdocuments_find → every such node, located by document, by path and by byte span;
 *  - @ref sz_subdocuments_find_kernel → the kernel any of them would run, for a caller to keep.
 *
 *  Each verb runs the kernel of the capability its engine was compiled for. The engine lives in
 *  `subdocuments/serial.h`.
 */
#ifndef STRINGZILLA_SUBDOCUMENTS_H_
#define STRINGZILLA_SUBDOCUMENTS_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`

/** The array positions one selector picks, RFC 9535's `[begin:end:step]`, an index @c n being
 *  `[n:n+1:1]`, or `[-1::1]` for the last element; negative bounds count from the array's end. */
typedef struct sz_subdocuments_slice_t {

    /** The first position, or 2⁵³ past the end that a negative step walks from when omitted. */
    sz_i64_t begin;

    /** The position the slice stops before, or 2⁵³ past the end its step walks to when omitted. */
    sz_i64_t end;

    /** Distance between selected positions, negative walking backward; zero selects none. */
    sz_i64_t step;
} sz_subdocuments_slice_t;

#include "stringzilla/subdocuments/serial.h" // `sz_subdocuments_engine_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Compiles @p paths for the best capability of @p capabilities that has an init kernel.
 *
 *  JSON paths follow RFC 9535: the root `$`, child segments `.name` and `.*`, and bracketed unions
 *  of names `'name'` or `"name"`, wildcards `*`, indices @c n, slices `begin:end:step` and filters
 *  `?expression`, the same after a descendant `..`, with blank space where the RFC allows it.
 *  Negative indices and bounds count from the array's end, and a negative step walks backward,
 *  though matches still come in document order. Filters hold comparisons, existence tests of
 *  relative queries, `&&`, `||`, `!` and parentheses, and the functions `length()`, `count()`,
 *  `value()`, `match()` and `search()`, whose I-Regexp patterns compile into one regex engine owned
 *  by this one. Each path compiles into one step per segment, numbered path after path, each
 *  filter's queries taking steps of their own. A path may also be a bare filter `?expression`,
 *  which tests the document's root as `$[?expression]` tests an array's elements.
 *
 *  XML paths follow XPath 1.0 along forward axes: absolute paths of `/` and `//` steps, each a name
 *  test, `*`, `@name`, `@*` or `text()` with an optional position `[n]`, which counts the siblings
 *  passing the same test. Name tests compare QNames as the document spells them, prefix included,
 *  and namespace declarations are no attributes. No blank space is allowed.
 *
 *  @param[out] engine Left untouched unless the call succeeds.
 *  @param[in] paths The vocabulary, one path per string, read on the host.
 *  @param[in] format The grammar of the documents every verb of this engine reads.
 *  @param[in] options How JSON walks read what RFC 8259 leaves open, or
 *      @c sz_subdocuments_options_none_k for the default.
 *  @param[out] malformed_index The first path to break the grammar or leave the subset, on failure.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] allocator Where both blocks come from, or @c STRINGZILLA_NULL for
 *      @ref sz_allocator_init_unified_best of @p capabilities. Stored by value, so
 *      @ref sz_subdocuments_engine_free needs none and cannot get the wrong one. Scratch grows to
 *      the widest call seen, so a warm engine allocates nothing.
 *  @param[in] stream Null on the CPU.
 *  @return @c sz_success_k once the paths compiled, or one of the failures below.
 *
 *  The call returns @c sz_invalid_syntax_k for anything outside the grammar, and for what no
 *  walk evaluates: an absolute query inside a filter, a comparison with no literal,
 *  `length()` or `count()` on either side, a pattern other than a string literal or one the
 *  regex family refuses, and in XML a reverse axis or a predicate other than a position. It
 *  returns @c sz_missing_library_k for `match()` or `search()` in a header-only build, whose
 *  regex dispatch links no library; @c sz_missing_kernel_k when no capability of the mask has
 *  an init, for a @p format no kernel reads yet, or on a GPU for non-zero @p options or a bare
 *  filter, which @p malformed_index then names; @c sz_unexpected_dimensions_k for an empty
 *  vocabulary; @c sz_overflow_risk_k past 32-bit step or name offsets; or
 *  @c sz_bad_alloc_k if memory allocation failed.
 */
STRINGZILLA_API sz_status_t sz_subdocuments_engine_init(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *paths, sz_subdocuments_format_t format,
                                                        sz_subdocuments_options_t options, sz_size_t *malformed_index,
                                                        sz_capability_t capabilities, sz_allocator_t *allocator,
                                                        sz_stream_t stream);

/** Returns both of the engine's blocks to the allocator that built them, once the work queued on
 *  @p stream is done with them, emptying @p engine. */
STRINGZILLA_API void sz_subdocuments_engine_free(sz_subdocuments_engine_t *engine, sz_stream_t stream);

/**
 *  @brief Marks which documents hold at least one node some path selects, stopping each walk at its
 *      first match, the lightest verb of the engine.
 *
 *  A walk validates nothing past the match it stops at, so a document malformed only after its
 *  first match is marked rather than refused. Device walks run whole, as the counts' do, so on a
 *  device that document fails the call.
 *
 *  @param[in] engine Compiled paths, serving one call at a time.
 *  @param[in] documents One document per string; their bytes are read in place and never copied.
 *  @param[out] contained One bit per document, eight to a byte, the lowest bit first: set when that
 *      document matches, with the bits past the last document written as zero.
 *  @param[out] malformed_index The first malformed document, on failure.
 *  @param[in] stream Null on the CPU.
 *  @return @c sz_success_k once every document was marked; @c sz_invalid_syntax_k for a malformed
 *      document, as @ref sz_subdocuments_counts defines it, leaving @p contained unspecified;
 *      @c sz_missing_kernel_k for an empty engine; or @c sz_bad_alloc_k if the walk's depth stack
 *      could not grow to the document's nesting.
 */
STRINGZILLA_API sz_status_t sz_subdocuments_contains(sz_subdocuments_engine_t *engine,
                                                     sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                     sz_size_t *malformed_index, sz_stream_t stream);

/**
 *  @brief Counts the nodes of every document that some path selects, one count per document.
 *
 *  A node several paths select counts once per path, as @ref sz_subdocuments_find reports it.
 *
 *  @param[in] engine Compiled paths; the walk grows the engine's depth stack, so one engine serves
 *      one call at a time.
 *  @param[in] documents One document per string; their bytes are read in place and never copied.
 *  @param[out] counts The per-document counts, document d at `counts[d * stride]`.
 *  @param[in] counts_stride Entries from one document's count to the next, at least one.
 *  @param[out] malformed_index The first malformed document, on failure.
 *  @param[in] stream Null on the CPU.
 *  @return @c sz_success_k once every document was counted; @c sz_unexpected_dimensions_k if
 *      @p counts_stride is zero; @c sz_missing_kernel_k for an empty engine; @c sz_bad_alloc_k if
 *      the depth stack could not grow; or @c sz_invalid_syntax_k for a malformed document.
 *
 *  A document is malformed with unbalanced brackets or tags, an unterminated string, attribute
 *  value, comment or CDATA section, a `/` outside the strings of JSON, broken structure where the
 *  walk reads it, or an XML DOCTYPE with an internal subset, and leaves the counts unspecified. A
 *  container or element no live step can enter is skipped by counting its nesting, so nothing else
 *  it holds is validated.
 */
STRINGZILLA_API sz_status_t sz_subdocuments_counts(sz_subdocuments_engine_t *engine, sz_strings_view_t const *documents,
                                                   sz_size_t *counts, sz_size_t counts_stride,
                                                   sz_size_t *malformed_index, sz_stream_t stream);

/**
 *  @brief Reports every node of every document that some path selects.
 *
 *  Matches come by document, then by start offset, then by path, so a container comes before the
 *  nodes inside it, and a node is reported once per path that selects it, however many of the
 *  path's selectors or descendant routes reach it. Each match spans the node's exact bytes: in JSON
 *  a string with its quotes, a number, @c true, @c false or @c null without the blank space around
 *  it, or an object or array through its closing bracket; in XML an element through its end tag, an
 *  attribute's value without its quotes, or a text node, whose CDATA sections stay as written.
 *
 *  @param[in] engine Compiled paths, serving one call at a time.
 *  @param[in] documents One document per string; their bytes are read in place and never copied.
 *  @param[out] matches Room for @p matches_capacity matches; @c STRINGZILLA_NULL together with a
 *      zero capacity makes the call a pure size query.
 *  @param[in] matches_capacity Entries @p matches holds.
 *  @param[out] matches_offsets One boundary per document into the matches plus one, the last being
 *      the total; filled whether or not the matches fit, which is what sizes the next call.
 *  @param[out] malformed_index The first malformed document, on failure.
 *  @param[in] stream Null on the CPU.
 *  @return @c sz_success_k once every match was reported, those past @p matches_capacity only
 *      counted; @c sz_invalid_syntax_k for a malformed document, as @ref sz_subdocuments_counts
 *      defines it, leaving the outputs unspecified; @c sz_missing_kernel_k for an empty engine; or
 *      @c sz_bad_alloc_k if the depth stack could not grow.
 */
STRINGZILLA_API sz_status_t sz_subdocuments_find(sz_subdocuments_engine_t *engine, sz_strings_view_t const *documents,
                                                 sz_match_t *matches, sz_size_t matches_capacity,
                                                 sz_size_t *matches_offsets, sz_size_t *malformed_index,
                                                 sz_stream_t stream);

/** @copydoc sz_subdocuments_engine_init */
STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_serial(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_subdocuments_contains */
STRINGZILLA_API sz_status_t sz_subdocuments_contains_serial(sz_subdocuments_engine_t *engine,
                                                            sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                            sz_size_t *malformed_index, sz_stream_t stream);
/** @copydoc sz_subdocuments_counts */
STRINGZILLA_API sz_status_t sz_subdocuments_counts_serial(sz_subdocuments_engine_t *engine,
                                                          sz_strings_view_t const *documents, sz_size_t *counts,
                                                          sz_size_t counts_stride, sz_size_t *malformed_index,
                                                          sz_stream_t stream);
/** @copydoc sz_subdocuments_find */
STRINGZILLA_API sz_status_t sz_subdocuments_find_serial(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *documents, sz_match_t *matches,
                                                        sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                        sz_size_t *malformed_index, sz_stream_t stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_subdocuments_engine_init */
STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_haswell(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_subdocuments_contains */
STRINGZILLA_API sz_status_t sz_subdocuments_contains_haswell(sz_subdocuments_engine_t *engine,
                                                             sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                             sz_size_t *malformed_index, sz_stream_t stream);
/** @copydoc sz_subdocuments_counts */
STRINGZILLA_API sz_status_t sz_subdocuments_counts_haswell(sz_subdocuments_engine_t *engine,
                                                           sz_strings_view_t const *documents, sz_size_t *counts,
                                                           sz_size_t counts_stride, sz_size_t *malformed_index,
                                                           sz_stream_t stream);
/** @copydoc sz_subdocuments_find */
STRINGZILLA_API sz_status_t sz_subdocuments_find_haswell(sz_subdocuments_engine_t *engine,
                                                         sz_strings_view_t const *documents, sz_match_t *matches,
                                                         sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                         sz_size_t *malformed_index, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_subdocuments_engine_init */
STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_icelake(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_subdocuments_contains */
STRINGZILLA_API sz_status_t sz_subdocuments_contains_icelake(sz_subdocuments_engine_t *engine,
                                                             sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                             sz_size_t *malformed_index, sz_stream_t stream);
/** @copydoc sz_subdocuments_counts */
STRINGZILLA_API sz_status_t sz_subdocuments_counts_icelake(sz_subdocuments_engine_t *engine,
                                                           sz_strings_view_t const *documents, sz_size_t *counts,
                                                           sz_size_t counts_stride, sz_size_t *malformed_index,
                                                           sz_stream_t stream);
/** @copydoc sz_subdocuments_find */
STRINGZILLA_API sz_status_t sz_subdocuments_find_icelake(sz_subdocuments_engine_t *engine,
                                                         sz_strings_view_t const *documents, sz_match_t *matches,
                                                         sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                         sz_size_t *malformed_index, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_subdocuments_engine_init */
STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_neon(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_subdocuments_contains */
STRINGZILLA_API sz_status_t sz_subdocuments_contains_neon(sz_subdocuments_engine_t *engine,
                                                          sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                          sz_size_t *malformed_index, sz_stream_t stream);
/** @copydoc sz_subdocuments_counts */
STRINGZILLA_API sz_status_t sz_subdocuments_counts_neon(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *documents, sz_size_t *counts,
                                                        sz_size_t counts_stride, sz_size_t *malformed_index,
                                                        sz_stream_t stream);
/** @copydoc sz_subdocuments_find */
STRINGZILLA_API sz_status_t sz_subdocuments_find_neon(sz_subdocuments_engine_t *engine,
                                                      sz_strings_view_t const *documents, sz_match_t *matches,
                                                      sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                      sz_size_t *malformed_index, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_SVE2
/** @copydoc sz_subdocuments_engine_init */
STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_sve2(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_subdocuments_contains */
STRINGZILLA_API sz_status_t sz_subdocuments_contains_sve2(sz_subdocuments_engine_t *engine,
                                                          sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                          sz_size_t *malformed_index, sz_stream_t stream);
/** @copydoc sz_subdocuments_counts */
STRINGZILLA_API sz_status_t sz_subdocuments_counts_sve2(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *documents, sz_size_t *counts,
                                                        sz_size_t counts_stride, sz_size_t *malformed_index,
                                                        sz_stream_t stream);
/** @copydoc sz_subdocuments_find */
STRINGZILLA_API sz_status_t sz_subdocuments_find_sve2(sz_subdocuments_engine_t *engine,
                                                      sz_strings_view_t const *documents, sz_match_t *matches,
                                                      sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                      sz_size_t *malformed_index, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_RVV
/** @copydoc sz_subdocuments_engine_init */
STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_rvv(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_subdocuments_contains */
STRINGZILLA_API sz_status_t sz_subdocuments_contains_rvv(sz_subdocuments_engine_t *engine,
                                                         sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                         sz_size_t *malformed_index, sz_stream_t stream);
/** @copydoc sz_subdocuments_counts */
STRINGZILLA_API sz_status_t sz_subdocuments_counts_rvv(sz_subdocuments_engine_t *engine,
                                                       sz_strings_view_t const *documents, sz_size_t *counts,
                                                       sz_size_t counts_stride, sz_size_t *malformed_index,
                                                       sz_stream_t stream);
/** @copydoc sz_subdocuments_find */
STRINGZILLA_API sz_status_t sz_subdocuments_find_rvv(sz_subdocuments_engine_t *engine,
                                                     sz_strings_view_t const *documents, sz_match_t *matches,
                                                     sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                     sz_size_t *malformed_index, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128
/** @copydoc sz_subdocuments_engine_init */
STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_v128(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_subdocuments_contains */
STRINGZILLA_API sz_status_t sz_subdocuments_contains_v128(sz_subdocuments_engine_t *engine,
                                                          sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                          sz_size_t *malformed_index, sz_stream_t stream);
/** @copydoc sz_subdocuments_counts */
STRINGZILLA_API sz_status_t sz_subdocuments_counts_v128(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *documents, sz_size_t *counts,
                                                        sz_size_t counts_stride, sz_size_t *malformed_index,
                                                        sz_stream_t stream);
/** @copydoc sz_subdocuments_find */
STRINGZILLA_API sz_status_t sz_subdocuments_find_v128(sz_subdocuments_engine_t *engine,
                                                      sz_strings_view_t const *documents, sz_match_t *matches,
                                                      sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                      sz_size_t *malformed_index, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_LOONGSONASX
/** @copydoc sz_subdocuments_engine_init */
STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_loongsonasx(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_subdocuments_contains */
STRINGZILLA_API sz_status_t sz_subdocuments_contains_loongsonasx(sz_subdocuments_engine_t *engine,
                                                                 sz_strings_view_t const *documents,
                                                                 sz_u1x8_t *contained, sz_size_t *malformed_index,
                                                                 sz_stream_t stream);
/** @copydoc sz_subdocuments_counts */
STRINGZILLA_API sz_status_t sz_subdocuments_counts_loongsonasx(sz_subdocuments_engine_t *engine,
                                                               sz_strings_view_t const *documents, sz_size_t *counts,
                                                               sz_size_t counts_stride, sz_size_t *malformed_index,
                                                               sz_stream_t stream);
/** @copydoc sz_subdocuments_find */
STRINGZILLA_API sz_status_t sz_subdocuments_find_loongsonasx(sz_subdocuments_engine_t *engine,
                                                             sz_strings_view_t const *documents, sz_match_t *matches,
                                                             sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                             sz_size_t *malformed_index, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX
/** @copydoc sz_subdocuments_engine_init */
STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_powervsx(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_subdocuments_contains */
STRINGZILLA_API sz_status_t sz_subdocuments_contains_powervsx(sz_subdocuments_engine_t *engine,
                                                              sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                              sz_size_t *malformed_index, sz_stream_t stream);
/** @copydoc sz_subdocuments_counts */
STRINGZILLA_API sz_status_t sz_subdocuments_counts_powervsx(sz_subdocuments_engine_t *engine,
                                                            sz_strings_view_t const *documents, sz_size_t *counts,
                                                            sz_size_t counts_stride, sz_size_t *malformed_index,
                                                            sz_stream_t stream);
/** @copydoc sz_subdocuments_find */
STRINGZILLA_API sz_status_t sz_subdocuments_find_powervsx(sz_subdocuments_engine_t *engine,
                                                          sz_strings_view_t const *documents, sz_match_t *matches,
                                                          sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                          sz_size_t *malformed_index, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_CUDA
/** @copydoc sz_subdocuments_engine_init */
STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_cuda(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_subdocuments_contains */
STRINGZILLA_API sz_status_t sz_subdocuments_contains_cuda(sz_subdocuments_engine_t *engine,
                                                          sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                          sz_size_t *malformed_index, sz_stream_t stream);
/** @copydoc sz_subdocuments_counts */
STRINGZILLA_API sz_status_t sz_subdocuments_counts_cuda(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *documents, sz_size_t *counts,
                                                        sz_size_t counts_stride, sz_size_t *malformed_index,
                                                        sz_stream_t stream);
/** @copydoc sz_subdocuments_find */
STRINGZILLA_API sz_status_t sz_subdocuments_find_cuda(sz_subdocuments_engine_t *engine,
                                                      sz_strings_view_t const *documents, sz_match_t *matches,
                                                      sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                      sz_size_t *malformed_index, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ROCM
/** @copydoc sz_subdocuments_engine_init */
STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_rocm(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_subdocuments_contains */
STRINGZILLA_API sz_status_t sz_subdocuments_contains_rocm(sz_subdocuments_engine_t *engine,
                                                          sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                          sz_size_t *malformed_index, sz_stream_t stream);
/** @copydoc sz_subdocuments_counts */
STRINGZILLA_API sz_status_t sz_subdocuments_counts_rocm(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *documents, sz_size_t *counts,
                                                        sz_size_t counts_stride, sz_size_t *malformed_index,
                                                        sz_stream_t stream);
/** @copydoc sz_subdocuments_find */
STRINGZILLA_API sz_status_t sz_subdocuments_find_rocm(sz_subdocuments_engine_t *engine,
                                                      sz_strings_view_t const *documents, sz_match_t *matches,
                                                      sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                      sz_size_t *malformed_index, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_METAL
/** @copydoc sz_subdocuments_engine_init */
STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_metal(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_subdocuments_counts */
STRINGZILLA_API sz_status_t sz_subdocuments_counts_metal(sz_subdocuments_engine_t *engine,
                                                         sz_strings_view_t const *documents, sz_size_t *counts,
                                                         sz_size_t counts_stride, sz_size_t *malformed_index,
                                                         sz_stream_t stream);
/** @copydoc sz_subdocuments_find */
STRINGZILLA_API sz_status_t sz_subdocuments_find_metal(sz_subdocuments_engine_t *engine,
                                                       sz_strings_view_t const *documents, sz_match_t *matches,
                                                       sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                       sz_size_t *malformed_index, sz_stream_t stream);
#endif

/**
 *  @brief Finds the subdocuments kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_subdocuments_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                        sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion Core API

#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/subdocuments/haswell.h"
#include "stringzilla/subdocuments/icelake.h"
#include "stringzilla/subdocuments/neon.h"
#include "stringzilla/subdocuments/sve2.h"
#include "stringzilla/subdocuments/rvv.h"
#include "stringzilla/subdocuments/v128.h"
#include "stringzilla/subdocuments/loongsonasx.h"
#include "stringzilla/subdocuments/powervsx.h"
#include "stringzilla/subdocuments/cuda.cuh"
#include "stringzilla/subdocuments/rocm.cuh"
#include "stringzilla/subdocuments/metal.h"
#endif // STRINGZILLA_HEADER_ONLY

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *paths, sz_subdocuments_format_t format,
                                                        sz_subdocuments_options_t options, sz_size_t *malformed_index,
                                                        sz_capability_t capabilities, sz_allocator_t *allocator,
                                                        sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(paths), sz_unused_(format), sz_unused_(options), sz_unused_(malformed_index),
        sz_unused_(capabilities), sz_unused_(allocator), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API void sz_subdocuments_engine_free(sz_subdocuments_engine_t *engine, sz_stream_t stream) {
    sz_subdocuments_engine_free_(engine, stream);
}

STRINGZILLA_API sz_status_t sz_subdocuments_contains(sz_subdocuments_engine_t *engine,
                                                     sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                     sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(documents), sz_unused_(contained), sz_unused_(malformed_index), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts(sz_subdocuments_engine_t *engine, sz_strings_view_t const *documents,
                                                   sz_size_t *counts, sz_size_t counts_stride,
                                                   sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(documents), sz_unused_(counts), sz_unused_(counts_stride),
        sz_unused_(malformed_index), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_subdocuments_find(sz_subdocuments_engine_t *engine, sz_strings_view_t const *documents,
                                                 sz_match_t *matches, sz_size_t matches_capacity,
                                                 sz_size_t *matches_offsets, sz_size_t *malformed_index,
                                                 sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(documents), sz_unused_(matches), sz_unused_(matches_capacity),
        sz_unused_(matches_offsets), sz_unused_(malformed_index), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                        sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_SUBDOCUMENTS_H_
