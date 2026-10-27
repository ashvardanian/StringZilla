/**
 *  @file include/stringzilla/strings.h
 *  @author Ash Vardanian
 *  @date October 7, 2023
 *  @brief Hardware-accelerated conversions and byte transforms over strings and their collections.
 *
 *  Every collection operation reads a @c sz_strings_view_t in any of its three layouts, optionally
 *  reordered or sampled through indices, or one text, and writes NUL-padded rows, one buffer, or a
 *  @c sz_strings_span_t tape:
 *
 *  - @c sz_strings_to_tape_best - gathers strings into a tape, reusing its blocks when they fit.
 *  - @c sz_strings_to_rows_best - writes strings into NUL-padded rows, truncating longer ones.
 *  - @c sz_strings_escape_best - escapes strings for JSON, CSV, hex, HTML, URLs or terminals.
 *  - @c sz_strings_unescape_best - reverses that escaping, naming the first malformed string.
 *  - @c sz_strings_lookup_best - maps every byte through a 256-entry Look-Up Table @b (LUT).
 *  - @c sz_strings_normalize_best - brings strings into a Unicode form, NFKC_Casefold included.
 *  - @c sz_strings_split_best - splits a text at the bytes of a set into a tape of its parts.
 *  - @c sz_strings_split_lines_best - splits a text at Unicode line breaks into a tape of lines.
 *  - @c sz_strings_join_best - joins columns of strings into one buffer, framing every row.
 *  - @c sz_allocator_init_unified_best - an allocator of blocks the host and a device share.
 *
 *  Unlike many other libraries focusing on trivial SIMD transformations, like converting lowercase
 *  to uppercase, StringZilla generalizes those to basic lookup table transforms. For typical ASCII
 *  conversions, you can use the following @b LUT initialization functions:
 *
 *  - @c sz_strings_lookup_init_lower for transforms like @c tolower
 *  - @c sz_strings_lookup_init_upper for transforms like @c toupper
 *  - @c sz_strings_lookup_init_ascii for transforms like @c isascii
 */
#ifndef STRINGZILLA_STRINGS_H_
#define STRINGZILLA_STRINGS_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/** The text formats that the escape and unescape families convert strings to and from. */
typedef enum sz_escape_kind_t {

    /** RFC 8259 string bodies: <tt>\\"</tt>, <tt>\\\\</tt>, short forms like <tt>\\n</tt>, and <tt>\\u00XX</tt> for other bytes
     *  below 0x20. */
    sz_escape_json_k = 0,

    /** Two lowercase hexadecimal digits per byte, decoding either case. */
    sz_escape_hex_k = 1,

    /** HTML and XML text and attribute values: <tt>&amp;amp;</tt>, <tt>&amp;lt;</tt>, <tt>&amp;gt;</tt>, <tt>&amp;quot;</tt> and <tt>&amp;#39;</tt>. */
    sz_escape_html_k = 2,

    /** RFC 3986 percent-encoding of every byte outside <tt>A-Z a-z 0-9 - _ ~</tt> and the dot. */
    sz_escape_url_k = 3,

    /** Terminal-safe logs: short forms like <tt>\\n</tt>, <tt>\\\\</tt> and <tt>\\"</tt>, <tt>\\xHH</tt> for other C0 control
     *  bytes, DEL and ill-formed UTF-8, and <tt>\\uXXXX</tt> for C1 controls and the bidirectional
     *  overrides U+202A to U+202E and U+2066 to U+2069, which reorder what a reader sees. */
    sz_escape_console_k = 4,

    /** RFC 4180 fields, quoted only when holding a comma, a quote, CR or LF, with every inner
     *  quote doubled. */
    sz_escape_csv_k = 5,

    /** RFC 8259 string bodies in ASCII alone, as Python's <tt>json.dumps</tt> writes them by default:
     *  every codepoint outside printable ASCII without a short form, DEL included, becomes
     *  <tt>\\uXXXX</tt>, a surrogate pair beyond the Basic Multilingual Plane, and every ill-formed byte
     *  becomes the escape of U+FFFD. */
    sz_escape_json_ascii_k = 6,
} sz_escape_kind_t;

/** Signature of the @c sz_strings_escape_best kernels, kept beside the enum it takes. */
typedef sz_status_t (*sz_kernel_strings_escape_t)(sz_strings_view_t const *, sz_size_t const *, sz_size_t,
                                                  sz_escape_kind_t, sz_allocator_t *, sz_strings_span_t *, sz_stream_t);

/** Signature of the @c sz_strings_unescape_best kernels, kept beside the enum it takes. */
typedef sz_status_t (*sz_kernel_strings_unescape_t)(sz_strings_view_t const *, sz_size_t const *, sz_size_t,
                                                    sz_escape_kind_t, sz_allocator_t *, sz_strings_span_t *,
                                                    sz_size_t *, sz_stream_t);

/** How a split ends its parts. */
typedef enum sz_split_kind_t {

    /** Every delimiter ends a part, so neighbouring delimiters leave empty parts, as
     *  <tt>str.split(sep)</tt> does. */
    sz_split_each_k = 0,

    /** A run of delimiters ends one part and empty parts are skipped, as <tt>str.split()</tt> does. */
    sz_split_runs_k = 1,
} sz_split_kind_t;

/** Signature of the @c sz_strings_split_best kernels, kept beside the enum it takes. */
typedef sz_status_t (*sz_kernel_strings_split_t)(sz_cptr_t, sz_size_t, sz_byteset_t const *, sz_split_kind_t,
                                                 sz_allocator_t *, sz_strings_span_t *, sz_stream_t);

/** Signature of the @c sz_strings_split_lines_best kernels. */
typedef sz_status_t (*sz_kernel_strings_split_lines_t)(sz_cptr_t, sz_size_t, sz_allocator_t *, sz_strings_span_t *,
                                                       sz_stream_t);

/** Signature of the @c sz_strings_join_best kernels. */
typedef sz_status_t (*sz_kernel_strings_join_t)(sz_strings_view_t const *, sz_size_t, sz_size_t,
                                                sz_string_view_t const *, sz_string_view_t, sz_ptr_t, sz_size_t *,
                                                sz_stream_t);

/**
 *  @brief Gathers <tt>strings[order[i]]</tt>, or the first @p count strings without @p order, into a tape.
 *
 *  Reads any layout, so it turns accessor-backed strings, slices and permutations into a tape that
 *  GPU kernels and Apache Arrow read directly. The payload and the offsets are two blocks, exactly
 *  as large as they must be, with 32-bit offsets while the payload fits in 4 GiB and 64-bit beyond.
 *
 *  Every collection verb is atomic: it succeeds and leaves @p tape holding the whole result, or it
 *  fails and leaves @p tape, its blocks and their bytes as they were. Blocks of exactly the sizes
 *  the result needs are written in place, so a borrowed span of the right sizes needs no
 *  allocator; any other @p tape receives fresh blocks from @p allocator, and its old ones go back
 *  only once the result is written. A whole tape gathered onto itself is left as it is.
 *
 *  @param[in] strings The strings to gather, in any layout.
 *  @param[in] order Indices into @p strings, possibly repeated, or null for the first @p count.
 *  @param[in] count Number of strings to write.
 *  @param[in] allocator Where the blocks of @p tape come from and return to.
 *  @param[in,out] tape A zeroed span, one an earlier call produced, or the one @p strings views.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *
 *  @note A GPU call reads the payload size back before allocating, synchronizing @p stream once.
 */
STRINGZILLA_API sz_status_t sz_strings_to_tape_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                    sz_capability_t capabilities, sz_stream_t stream);

/** @copydoc sz_strings_to_tape_best */
STRINGZILLA_API sz_status_t sz_strings_to_tape_serial(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_allocator_t *allocator,
                                                      sz_strings_span_t *tape, sz_stream_t stream);

#if STRINGZILLA_TARGET_HASWELL

/** @copydoc sz_strings_to_tape_best */
STRINGZILLA_API sz_status_t sz_strings_to_tape_haswell(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_allocator_t *allocator,
                                                       sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE

/** @copydoc sz_strings_to_tape_best */
STRINGZILLA_API sz_status_t sz_strings_to_tape_icelake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_allocator_t *allocator,
                                                       sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_strings_to_tape_best */
STRINGZILLA_API sz_status_t sz_strings_to_tape_neon(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                    sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_SVE

/** @copydoc sz_strings_to_tape_best */
STRINGZILLA_API sz_status_t sz_strings_to_tape_sve(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                   sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_RVV

/** @copydoc sz_strings_to_tape_best */
STRINGZILLA_API sz_status_t sz_strings_to_tape_rvv(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                   sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128

/** @copydoc sz_strings_to_tape_best */
STRINGZILLA_API sz_status_t sz_strings_to_tape_v128(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                    sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128RELAXED

/** @copydoc sz_strings_to_tape_best */
STRINGZILLA_API sz_status_t sz_strings_to_tape_v128relaxed(sz_strings_view_t const *strings, sz_size_t const *order,
                                                           sz_size_t count, sz_allocator_t *allocator,
                                                           sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_LOONGSONASX

/** @copydoc sz_strings_to_tape_best */
STRINGZILLA_API sz_status_t sz_strings_to_tape_loongsonasx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                           sz_size_t count, sz_allocator_t *allocator,
                                                           sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX

/** @copydoc sz_strings_to_tape_best */
STRINGZILLA_API sz_status_t sz_strings_to_tape_powervsx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                        sz_size_t count, sz_allocator_t *allocator,
                                                        sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_CUDA

/** @copydoc sz_strings_to_tape_best */
STRINGZILLA_API sz_status_t sz_strings_to_tape_cuda(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                    sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ROCM

/** @copydoc sz_strings_to_tape_best */
STRINGZILLA_API sz_status_t sz_strings_to_tape_rocm(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                    sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_METAL

/** @copydoc sz_strings_to_tape_best */
STRINGZILLA_API sz_status_t sz_strings_to_tape_metal(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_stream_t stream);
#endif

/**
 *  @brief Writes <tt>strings[order[i]]</tt>, or the first @p count strings, into NUL-padded rows.
 *  @param[in] strings The strings to write, in any layout.
 *  @param[in] order Indices into @p strings, possibly repeated, or null for the first @p count.
 *  @param[in] count Number of rows to write.
 *  @param[in] rows Room for @p count rows, each truncating a longer string at its width.
 *  @param[out] longest Length of the longest string, to retry with wider rows.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *
 *  @par Statuses:
 *  - @c sz_missing_kernel_k No capability in @p capabilities has it.
 */
STRINGZILLA_API sz_status_t sz_strings_to_rows_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_strings_rows_t const *rows, sz_size_t *longest,
                                                    sz_capability_t capabilities, sz_stream_t stream);

/** @copydoc sz_strings_to_rows_best */
STRINGZILLA_API sz_status_t sz_strings_to_rows_serial(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_strings_rows_t const *rows,
                                                      sz_size_t *longest, sz_stream_t stream);

#if STRINGZILLA_TARGET_HASWELL

/** @copydoc sz_strings_to_rows_best */
STRINGZILLA_API sz_status_t sz_strings_to_rows_haswell(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_strings_rows_t const *rows,
                                                       sz_size_t *longest, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE

/** @copydoc sz_strings_to_rows_best */
STRINGZILLA_API sz_status_t sz_strings_to_rows_icelake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_strings_rows_t const *rows,
                                                       sz_size_t *longest, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_strings_to_rows_best */
STRINGZILLA_API sz_status_t sz_strings_to_rows_neon(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_strings_rows_t const *rows, sz_size_t *longest,
                                                    sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_SVE

/** @copydoc sz_strings_to_rows_best */
STRINGZILLA_API sz_status_t sz_strings_to_rows_sve(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, sz_strings_rows_t const *rows, sz_size_t *longest,
                                                   sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_RVV

/** @copydoc sz_strings_to_rows_best */
STRINGZILLA_API sz_status_t sz_strings_to_rows_rvv(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, sz_strings_rows_t const *rows, sz_size_t *longest,
                                                   sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128

/** @copydoc sz_strings_to_rows_best */
STRINGZILLA_API sz_status_t sz_strings_to_rows_v128(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_strings_rows_t const *rows, sz_size_t *longest,
                                                    sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128RELAXED

/** @copydoc sz_strings_to_rows_best */
STRINGZILLA_API sz_status_t sz_strings_to_rows_v128relaxed(sz_strings_view_t const *strings, sz_size_t const *order,
                                                           sz_size_t count, sz_strings_rows_t const *rows,
                                                           sz_size_t *longest, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_LOONGSONASX

/** @copydoc sz_strings_to_rows_best */
STRINGZILLA_API sz_status_t sz_strings_to_rows_loongsonasx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                           sz_size_t count, sz_strings_rows_t const *rows,
                                                           sz_size_t *longest, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX

/** @copydoc sz_strings_to_rows_best */
STRINGZILLA_API sz_status_t sz_strings_to_rows_powervsx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                        sz_size_t count, sz_strings_rows_t const *rows,
                                                        sz_size_t *longest, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_CUDA

/** @copydoc sz_strings_to_rows_best */
STRINGZILLA_API sz_status_t sz_strings_to_rows_cuda(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_strings_rows_t const *rows, sz_size_t *longest,
                                                    sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ROCM

/** @copydoc sz_strings_to_rows_best */
STRINGZILLA_API sz_status_t sz_strings_to_rows_rocm(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_strings_rows_t const *rows, sz_size_t *longest,
                                                    sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_METAL

/** @copydoc sz_strings_to_rows_best */
STRINGZILLA_API sz_status_t sz_strings_to_rows_metal(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_strings_rows_t const *rows, sz_size_t *longest,
                                                     sz_stream_t stream);
#endif

/**
 *  @brief Escapes <tt>strings[order[i]]</tt>, or the first @p count strings, for @p kind into a tape.
 *
 *  Allocates and reuses @p tape like @c sz_strings_to_tape_best. When @p strings views all of
 *  @p tape, its payload block grows first, through @c reallocate when the allocator has one, then
 *  its bytes move to the end and escape front to back, the writer never overtaking unread bytes.
 *
 *  @param[in] strings The strings to escape, in any layout.
 *  @param[in] order Indices into @p strings, possibly repeated, or null for the first @p count.
 *  @param[in] count Number of strings to write.
 *  @param[in] kind The format to escape for.
 *  @param[in] allocator Where the blocks of @p tape come from and return to.
 *  @param[in,out] tape A zeroed span, one an earlier call produced, or the one @p strings views.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 */
STRINGZILLA_API sz_status_t sz_strings_escape_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                   sz_strings_span_t *tape, sz_capability_t capabilities,
                                                   sz_stream_t stream);

/** @copydoc sz_strings_escape_best */
STRINGZILLA_API sz_status_t sz_strings_escape_serial(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_stream_t stream);

#if STRINGZILLA_TARGET_HASWELL

/** @copydoc sz_strings_escape_best */
STRINGZILLA_API sz_status_t sz_strings_escape_haswell(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                      sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE

/** @copydoc sz_strings_escape_best */
STRINGZILLA_API sz_status_t sz_strings_escape_icelake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                      sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_strings_escape_best */
STRINGZILLA_API sz_status_t sz_strings_escape_neon(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                   sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_SVE

/** @copydoc sz_strings_escape_best */
STRINGZILLA_API sz_status_t sz_strings_escape_sve(sz_strings_view_t const *strings, sz_size_t const *order,
                                                  sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                  sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_RVV

/** @copydoc sz_strings_escape_best */
STRINGZILLA_API sz_status_t sz_strings_escape_rvv(sz_strings_view_t const *strings, sz_size_t const *order,
                                                  sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                  sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128

/** @copydoc sz_strings_escape_best */
STRINGZILLA_API sz_status_t sz_strings_escape_v128(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                   sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128RELAXED

/** @copydoc sz_strings_escape_best */
STRINGZILLA_API sz_status_t sz_strings_escape_v128relaxed(sz_strings_view_t const *strings, sz_size_t const *order,
                                                          sz_size_t count, sz_escape_kind_t kind,
                                                          sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                          sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_LOONGSONASX

/** @copydoc sz_strings_escape_best */
STRINGZILLA_API sz_status_t sz_strings_escape_loongsonasx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                          sz_size_t count, sz_escape_kind_t kind,
                                                          sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                          sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX

/** @copydoc sz_strings_escape_best */
STRINGZILLA_API sz_status_t sz_strings_escape_powervsx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_escape_kind_t kind,
                                                       sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                       sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_CUDA

/** @copydoc sz_strings_escape_best */
STRINGZILLA_API sz_status_t sz_strings_escape_cuda(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                   sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ROCM

/** @copydoc sz_strings_escape_best */
STRINGZILLA_API sz_status_t sz_strings_escape_rocm(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                   sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_METAL

/** @copydoc sz_strings_escape_best */
STRINGZILLA_API sz_status_t sz_strings_escape_metal(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                    sz_strings_span_t *tape, sz_stream_t stream);
#endif

/**
 *  @brief Unescapes <tt>strings[order[i]]</tt>, or the first @p count strings, from @p kind into a tape.
 *
 *  Allocates and reuses @p tape like @c sz_strings_to_tape_best. Every string is checked before
 *  anything is written, so one malformed string leaves @p tape as it was.
 *
 *  @param[in] strings The strings to unescape, in any layout.
 *  @param[in] order Indices into @p strings, possibly repeated, or null for the first @p count.
 *  @param[in] count Number of strings to write.
 *  @param[in] kind The format to unescape from.
 *  @param[in] allocator Where the blocks of @p tape come from and return to.
 *  @param[in,out] tape A zeroed span, one an earlier call produced, or the one @p strings views.
 *  @param[out] malformed_index The first string holding a malformed sequence, on failure.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *
 *  @par Statuses:
 *  - @c sz_invalid_utf8_k A malformed sequence.
 */
STRINGZILLA_API sz_status_t sz_strings_unescape_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_size_t *malformed_index,
                                                     sz_capability_t capabilities, sz_stream_t stream);

/** @copydoc sz_strings_unescape_best */
STRINGZILLA_API sz_status_t sz_strings_unescape_serial(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_escape_kind_t kind,
                                                       sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                       sz_size_t *malformed_index, sz_stream_t stream);

#if STRINGZILLA_TARGET_HASWELL

/** @copydoc sz_strings_unescape_best */
STRINGZILLA_API sz_status_t sz_strings_unescape_haswell(sz_strings_view_t const *strings, sz_size_t const *order,
                                                        sz_size_t count, sz_escape_kind_t kind,
                                                        sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                        sz_size_t *malformed_index, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE

/** @copydoc sz_strings_unescape_best */
STRINGZILLA_API sz_status_t sz_strings_unescape_icelake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                        sz_size_t count, sz_escape_kind_t kind,
                                                        sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                        sz_size_t *malformed_index, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_strings_unescape_best */
STRINGZILLA_API sz_status_t sz_strings_unescape_neon(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_size_t *malformed_index,
                                                     sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_SVE

/** @copydoc sz_strings_unescape_best */
STRINGZILLA_API sz_status_t sz_strings_unescape_sve(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                    sz_strings_span_t *tape, sz_size_t *malformed_index,
                                                    sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_RVV

/** @copydoc sz_strings_unescape_best */
STRINGZILLA_API sz_status_t sz_strings_unescape_rvv(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                    sz_strings_span_t *tape, sz_size_t *malformed_index,
                                                    sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128

/** @copydoc sz_strings_unescape_best */
STRINGZILLA_API sz_status_t sz_strings_unescape_v128(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_size_t *malformed_index,
                                                     sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128RELAXED

/** @copydoc sz_strings_unescape_best */
STRINGZILLA_API sz_status_t sz_strings_unescape_v128relaxed(sz_strings_view_t const *strings, sz_size_t const *order,
                                                            sz_size_t count, sz_escape_kind_t kind,
                                                            sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                            sz_size_t *malformed_index, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_LOONGSONASX

/** @copydoc sz_strings_unescape_best */
STRINGZILLA_API sz_status_t sz_strings_unescape_loongsonasx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                            sz_size_t count, sz_escape_kind_t kind,
                                                            sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                            sz_size_t *malformed_index, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX

/** @copydoc sz_strings_unescape_best */
STRINGZILLA_API sz_status_t sz_strings_unescape_powervsx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                         sz_size_t count, sz_escape_kind_t kind,
                                                         sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                         sz_size_t *malformed_index, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_CUDA

/** @copydoc sz_strings_unescape_best */
STRINGZILLA_API sz_status_t sz_strings_unescape_cuda(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_size_t *malformed_index,
                                                     sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ROCM

/** @copydoc sz_strings_unescape_best */
STRINGZILLA_API sz_status_t sz_strings_unescape_rocm(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_size_t *malformed_index,
                                                     sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_METAL

/** @copydoc sz_strings_unescape_best */
STRINGZILLA_API sz_status_t sz_strings_unescape_metal(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                      sz_strings_span_t *tape, sz_size_t *malformed_index,
                                                      sz_stream_t stream);
#endif

/**
 *  @brief Maps every byte of <tt>strings[order[i]]</tt>, or the first @p count strings, through a
 *      256-entry Look-Up Table @b (LUT) into a tape, the same as <tt>for (char &c : text) c = lut[c]</tt>.
 *
 *  Allocates and reuses @p tape like @c sz_strings_to_tape_best. The mapping keeps every length, so
 *  a span over the caller's own buffers of matching sizes needs no allocator, and a span that
 *  @p strings views is mapped in place.
 *
 *  Can be used to implement some form of string normalization, partially masking punctuation marks,
 *  or converting between different character sets, like uppercase or lowercase. Surprisingly, also
 *  has broad implications in image processing, where channel transformations often use LUTs.
 *
 *  Lowercasing a buffer in place, as a tape of one string:
 *
 *  @code{.c}
 *     #include <ctype.h> // for tolower
 *     #include <stringzilla/strings.h>
 *     int main() {
 *          sz_capability_t capabilities;
 *          char to_lower_lut[256];
 *          for (int i = 0; i < 256; ++i) to_lower_lut[i] = tolower(i);
 *          char buffer[3] = {'A', 'B', 'C'};
 *          sz_u32_t offsets[2] = {0, 3};
 *          sz_strings_span_t tape;
 *          tape.layout = sz_strings_u32_tape_k, tape.count = 1;
 *          tape.data.u32_tape.data = buffer, tape.data.u32_tape.offsets = offsets;
 *          sz_strings_view_t strings;
 *          sz_strings_from_span(&tape, &strings);
 *          sz_capabilities_enabled_cpu(&capabilities);
 *          sz_strings_lookup_best(&strings, NULL, 1, to_lower_lut, NULL, &tape, capabilities, NULL);
 *          return buffer[0] == 'a' && buffer[1] == 'b' && buffer[2] == 'c' ? 0 : 1;
 *     }
 *  @endcode
 *
 *  @param[in] strings The strings to map, in any layout.
 *  @param[in] order Indices into @p strings, possibly repeated, or null for the first @p count.
 *  @param[in] count Number of strings to write.
 *  @param[in] lut The table to map through, exactly @b 256 bytes long.
 *  @param[in] allocator Where the blocks of @p tape come from and return to.
 *  @param[in,out] tape A zeroed span, one an earlier call produced, or the one @p strings views.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *
 *  @see Lookup table: https://en.wikipedia.org/wiki/Lookup_table
 */
STRINGZILLA_API sz_status_t sz_strings_lookup_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, char const lut[sz_at_least_(256)],
                                                   sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                   sz_capability_t capabilities, sz_stream_t stream);

/** @copydoc sz_strings_lookup_best */
STRINGZILLA_API sz_status_t sz_strings_lookup_serial(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, char const lut[sz_at_least_(256)],
                                                     sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                     sz_stream_t stream);

#if STRINGZILLA_TARGET_HASWELL

/** @copydoc sz_strings_lookup_best */
STRINGZILLA_API sz_status_t sz_strings_lookup_haswell(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, char const lut[sz_at_least_(256)],
                                                      sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                      sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE

/** @copydoc sz_strings_lookup_best */
STRINGZILLA_API sz_status_t sz_strings_lookup_icelake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, char const lut[sz_at_least_(256)],
                                                      sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                      sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_strings_lookup_best */
STRINGZILLA_API sz_status_t sz_strings_lookup_neon(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, char const lut[sz_at_least_(256)],
                                                   sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                   sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_SVE

/** @copydoc sz_strings_lookup_best */
STRINGZILLA_API sz_status_t sz_strings_lookup_sve(sz_strings_view_t const *strings, sz_size_t const *order,
                                                  sz_size_t count, char const lut[sz_at_least_(256)],
                                                  sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                  sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_RVV

/** @copydoc sz_strings_lookup_best */
STRINGZILLA_API sz_status_t sz_strings_lookup_rvv(sz_strings_view_t const *strings, sz_size_t const *order,
                                                  sz_size_t count, char const lut[sz_at_least_(256)],
                                                  sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                  sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128

/** @copydoc sz_strings_lookup_best */
STRINGZILLA_API sz_status_t sz_strings_lookup_v128(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, char const lut[sz_at_least_(256)],
                                                   sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                   sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128RELAXED

/** @copydoc sz_strings_lookup_best */
STRINGZILLA_API sz_status_t sz_strings_lookup_v128relaxed(sz_strings_view_t const *strings, sz_size_t const *order,
                                                          sz_size_t count, char const lut[sz_at_least_(256)],
                                                          sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                          sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_LOONGSONASX

/** @copydoc sz_strings_lookup_best */
STRINGZILLA_API sz_status_t sz_strings_lookup_loongsonasx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                          sz_size_t count, char const lut[sz_at_least_(256)],
                                                          sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                          sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX

/** @copydoc sz_strings_lookup_best */
STRINGZILLA_API sz_status_t sz_strings_lookup_powervsx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, char const lut[sz_at_least_(256)],
                                                       sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                       sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_CUDA

/** @copydoc sz_strings_lookup_best */
STRINGZILLA_API sz_status_t sz_strings_lookup_cuda(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, char const lut[sz_at_least_(256)],
                                                   sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                   sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ROCM

/** @copydoc sz_strings_lookup_best */
STRINGZILLA_API sz_status_t sz_strings_lookup_rocm(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, char const lut[sz_at_least_(256)],
                                                   sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                   sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_METAL

/** @copydoc sz_strings_lookup_best */
STRINGZILLA_API sz_status_t sz_strings_lookup_metal(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, char const lut[sz_at_least_(256)],
                                                    sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                    sz_stream_t stream);
#endif

/**
 *  @brief Brings <tt>strings[order[i]]</tt>, or the first @p count strings, into the Unicode @p form,
 *      writing them into a tape.
 *
 *  @c sz_normal_form_nfkc_casefold_k gives the keys search engines match on: NFKC with full case
 *  folding and without default-ignorable codepoints, the Unicode toNFKC_Casefold.
 *
 *  On the CPU every string is normalized once, into blocks of the call's own from @p allocator: a
 *  payload starting at the total input length, grown through @c reallocate when the allocator has
 *  one, and fitted to its exact size at the end. That needs an allocator even when @p tape holds
 *  blocks of exactly the final sizes, which are then filled rather than replaced. Like every
 *  collection verb it is atomic. GPUs measure first and read only tapes, a string to a thread.
 *
 *  @param[in] strings The strings to normalize, in any layout.
 *  @param[in] order Indices into @p strings, possibly repeated, or null for the first @p count.
 *  @param[in] count Number of strings to write.
 *  @param[in] form The normalization form to bring every string into.
 *  @param[in] allocator Where the blocks of @p tape come from and return to.
 *  @param[in,out] tape A zeroed span, one an earlier call produced, or the one @p strings views.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *
 *  @see UAX #15 Unicode Normalization Forms: https://www.unicode.org/reports/tr15/
 */
STRINGZILLA_API sz_status_t sz_strings_normalize_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_normal_form_t form, sz_allocator_t *allocator,
                                                      sz_strings_span_t *tape, sz_capability_t capabilities,
                                                      sz_stream_t stream);

/** @copydoc sz_strings_normalize_best */
STRINGZILLA_API sz_status_t sz_strings_normalize_serial(sz_strings_view_t const *strings, sz_size_t const *order,
                                                        sz_size_t count, sz_normal_form_t form,
                                                        sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                        sz_stream_t stream);

#if STRINGZILLA_TARGET_HASWELL

/** @copydoc sz_strings_normalize_best */
STRINGZILLA_API sz_status_t sz_strings_normalize_haswell(sz_strings_view_t const *strings, sz_size_t const *order,
                                                         sz_size_t count, sz_normal_form_t form,
                                                         sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                         sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE

/** @copydoc sz_strings_normalize_best */
STRINGZILLA_API sz_status_t sz_strings_normalize_icelake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                         sz_size_t count, sz_normal_form_t form,
                                                         sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                         sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_strings_normalize_best */
STRINGZILLA_API sz_status_t sz_strings_normalize_neon(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_normal_form_t form, sz_allocator_t *allocator,
                                                      sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_SVE

/** @copydoc sz_strings_normalize_best */
STRINGZILLA_API sz_status_t sz_strings_normalize_sve(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_normal_form_t form, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_RVV

/** @copydoc sz_strings_normalize_best */
STRINGZILLA_API sz_status_t sz_strings_normalize_rvv(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_normal_form_t form, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128

/** @copydoc sz_strings_normalize_best */
STRINGZILLA_API sz_status_t sz_strings_normalize_v128(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_normal_form_t form, sz_allocator_t *allocator,
                                                      sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128RELAXED

/** @copydoc sz_strings_normalize_best */
STRINGZILLA_API sz_status_t sz_strings_normalize_v128relaxed(sz_strings_view_t const *strings, sz_size_t const *order,
                                                             sz_size_t count, sz_normal_form_t form,
                                                             sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                             sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_LOONGSONASX

/** @copydoc sz_strings_normalize_best */
STRINGZILLA_API sz_status_t sz_strings_normalize_loongsonasx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                             sz_size_t count, sz_normal_form_t form,
                                                             sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                             sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX

/** @copydoc sz_strings_normalize_best */
STRINGZILLA_API sz_status_t sz_strings_normalize_powervsx(sz_strings_view_t const *strings, sz_size_t const *order,
                                                          sz_size_t count, sz_normal_form_t form,
                                                          sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                          sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_CUDA

/** @copydoc sz_strings_normalize_best */
STRINGZILLA_API sz_status_t sz_strings_normalize_cuda(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_normal_form_t form, sz_allocator_t *allocator,
                                                      sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ROCM

/** @copydoc sz_strings_normalize_best */
STRINGZILLA_API sz_status_t sz_strings_normalize_rocm(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_normal_form_t form, sz_allocator_t *allocator,
                                                      sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_METAL

/** @copydoc sz_strings_normalize_best */
STRINGZILLA_API sz_status_t sz_strings_normalize_metal(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_normal_form_t form,
                                                       sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                       sz_stream_t stream);
#endif

/**
 *  @brief Splits @p text at the bytes of @p delimiters into a tape of the parts between them.
 *
 *  With @c sz_split_each_k a text of @c n delimiters always makes @c n + 1 parts, like Python's
 *  <tt>bytes.split(sep)</tt> for a one-byte @c sep, generalized to a set. With @c sz_split_runs_k leading
 *  and trailing runs make no parts, like <tt>bytes.split()</tt> when the set is ASCII whitespace. Measures
 *  the parts first, then writes blocks of exactly their sizes, reusing those of @p tape like
 *  @c sz_strings_to_tape_best unless @p text lies in them; like every collection verb it is atomic.
 *
 *  @param[in] text The bytes to split, null only when @p length is zero.
 *  @param[in] length Number of bytes in @p text.
 *  @param[in] delimiters The bytes that separate the parts.
 *  @param[in] kind Whether every delimiter ends a part, or every run of them.
 *  @param[in] allocator Where the blocks of @p tape come from and return to.
 *  @param[in,out] tape A zeroed span, or one an earlier call produced.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null, as only CPUs split.
 */
STRINGZILLA_API sz_status_t sz_strings_split_best(sz_cptr_t text, sz_size_t length, sz_byteset_t const *delimiters,
                                                  sz_split_kind_t kind, sz_allocator_t *allocator,
                                                  sz_strings_span_t *tape, sz_capability_t capabilities,
                                                  sz_stream_t stream);

/** @copydoc sz_strings_split_best */
STRINGZILLA_API sz_status_t sz_strings_split_serial(sz_cptr_t text, sz_size_t length, sz_byteset_t const *delimiters,
                                                    sz_split_kind_t kind, sz_allocator_t *allocator,
                                                    sz_strings_span_t *tape, sz_stream_t stream);

#if STRINGZILLA_TARGET_ICELAKE

/** @copydoc sz_strings_split_best */
STRINGZILLA_API sz_status_t sz_strings_split_icelake(sz_cptr_t text, sz_size_t length, sz_byteset_t const *delimiters,
                                                     sz_split_kind_t kind, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_strings_split_best */
STRINGZILLA_API sz_status_t sz_strings_split_neon(sz_cptr_t text, sz_size_t length, sz_byteset_t const *delimiters,
                                                  sz_split_kind_t kind, sz_allocator_t *allocator,
                                                  sz_strings_span_t *tape, sz_stream_t stream);
#endif

/**
 *  @brief Splits @p text at Unicode line breaks, CR LF as one, into a tape of lines without breaks.
 *
 *  The breaks are those of Python's <tt>str.splitlines()</tt> over UTF-8: LF, VT, FF, CR, FS, GS, RS,
 *  CR LF, NEL, LS and PS, and a final break leaves no empty line after it. Allocates and reuses
 *  @p tape like @c sz_strings_split_best, and like every collection verb it is atomic.
 *
 *  @param[in] text The UTF-8 bytes to split, null only when @p length is zero.
 *  @param[in] length Number of bytes in @p text.
 *  @param[in] allocator Where the blocks of @p tape come from and return to.
 *  @param[in,out] tape A zeroed span, or one an earlier call produced.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null, as only CPUs split.
 *
 *  @see Python's <tt>str.splitlines</tt>: https://docs.python.org/3/library/stdtypes.html
 */
STRINGZILLA_API sz_status_t sz_strings_split_lines_best(sz_cptr_t text, sz_size_t length, sz_allocator_t *allocator,
                                                        sz_strings_span_t *tape, sz_capability_t capabilities,
                                                        sz_stream_t stream);

/** @copydoc sz_strings_split_lines_best */
STRINGZILLA_API sz_status_t sz_strings_split_lines_serial(sz_cptr_t text, sz_size_t length, sz_allocator_t *allocator,
                                                          sz_strings_span_t *tape, sz_stream_t stream);

#if STRINGZILLA_TARGET_ICELAKE

/** @copydoc sz_strings_split_lines_best */
STRINGZILLA_API sz_status_t sz_strings_split_lines_icelake(sz_cptr_t text, sz_size_t length, sz_allocator_t *allocator,
                                                           sz_strings_span_t *tape, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_strings_split_lines_best */
STRINGZILLA_API sz_status_t sz_strings_split_lines_neon(sz_cptr_t text, sz_size_t length, sz_allocator_t *allocator,
                                                        sz_strings_span_t *tape, sz_stream_t stream);
#endif

/**
 *  @brief Writes row r of @p columns as
 *      <tt>frames[0] + columns[0][r] + frames[1] + … + columns[n-1][r] + frames[n]</tt>, with @p between
 *      written between rows, into @p output; a null @p output only measures into @p length.
 *
 *  Serializes columns as CSV lines, JSON arrays or SQL value lists without a string per cell, and
 *  joins one column with an empty frame on each side like Python's <tt>str.join</tt>.
 *
 *  @param[in] columns The @p columns_count columns of @p rows strings each, in any layouts.
 *  @param[in] columns_count Number of columns, one less than the number of @p frames.
 *  @param[in] rows Number of strings in every column.
 *  @param[in] frames The strings written before, between and after the cells of every row.
 *  @param[in] between The string written between consecutive rows.
 *  @param[out] output Room for exactly @p length bytes, or null to measure them.
 *  @param[in,out] length The bytes the join takes: written when @p output is null, read otherwise.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null, as only CPUs join.
 *
 *  @par Statuses:
 *  - @c sz_bad_alloc_k Measuring overflows the length.
 */
STRINGZILLA_API sz_status_t sz_strings_join_best(sz_strings_view_t const *columns, sz_size_t columns_count,
                                                 sz_size_t rows, sz_string_view_t const *frames,
                                                 sz_string_view_t between, sz_ptr_t output, sz_size_t *length,
                                                 sz_capability_t capabilities, sz_stream_t stream);

/** @copydoc sz_strings_join_best */
STRINGZILLA_API sz_status_t sz_strings_join_serial(sz_strings_view_t const *columns, sz_size_t columns_count,
                                                   sz_size_t rows, sz_string_view_t const *frames,
                                                   sz_string_view_t between, sz_ptr_t output, sz_size_t *length,
                                                   sz_stream_t stream);

#if STRINGZILLA_TARGET_ICELAKE

/** @copydoc sz_strings_join_best */
STRINGZILLA_API sz_status_t sz_strings_join_icelake(sz_strings_view_t const *columns, sz_size_t columns_count,
                                                    sz_size_t rows, sz_string_view_t const *frames,
                                                    sz_string_view_t between, sz_ptr_t output, sz_size_t *length,
                                                    sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_strings_join_best */
STRINGZILLA_API sz_status_t sz_strings_join_neon(sz_strings_view_t const *columns, sz_size_t columns_count,
                                                 sz_size_t rows, sz_string_view_t const *frames,
                                                 sz_string_view_t between, sz_ptr_t output, sz_size_t *length,
                                                 sz_stream_t stream);
#endif

/**
 *  @brief Finds the strings kernel of @p kind, from the best of @p capabilities.
 *  @param[in] kind The verb whose kernel to find.
 *  @param[in] capabilities The mask to pick the best capability from.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *
 *  @par Statuses:
 *  - @c sz_missing_kernel_k
 *  - @c sz_missing_library_k Header-only.
 */
STRINGZILLA_API sz_status_t sz_strings_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                   sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion Core API

#pragma region Memory Management

/**
 *  @brief Initializes an allocator whose blocks both the host and the device of @p capabilities
 *      address: the host heap on the CPU, managed memory on CUDA and ROCm, shared buffers on Metal.
 *
 *  The allocator is stateless, its @c handle null, and each call allocates on the device of the
 *  stream it is given, so one allocator serves every device of its vendor.
 *
 *  @param[out] allocator The allocator to initialize.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cuda reports;
 *      its group picks the CPU or a GPU vendor.
 *
 *  @par Statuses:
 *  - @c sz_missing_gpu_k A GPU vendor this library was built without.
 */
STRINGZILLA_API sz_status_t sz_allocator_init_unified_best(sz_allocator_t *allocator, sz_capability_t capabilities);

/** Initializes device storage on CUDA and ROCm, or unified storage on Metal.
 *  Each callback uses its stream's device. CPU returns @c sz_missing_kernel_k; GPU vendors omitted
 *  from the build return @c sz_missing_gpu_k. */
STRINGZILLA_API sz_status_t sz_allocator_init_device_best(sz_allocator_t *allocator, sz_capability_t capabilities);

/** Initializes page-locked CUDA or ROCm host storage, using each callback's stream.
 *  CPU and Metal return @c sz_missing_kernel_k; unbuilt GPU vendors return @c sz_missing_gpu_k. */
STRINGZILLA_API sz_status_t sz_allocator_init_pinned_best(sz_allocator_t *allocator, sz_capability_t capabilities);

/** @copydoc sz_allocator_init_unified_best */
STRINGZILLA_API sz_status_t sz_allocator_init_unified_serial(sz_allocator_t *allocator);

#if STRINGZILLA_ARCH_CUDA_
/** @copydoc sz_allocator_init_unified_best */
STRINGZILLA_API sz_status_t sz_allocator_init_unified_cuda(sz_allocator_t *allocator);
/** @copydoc sz_allocator_init_device_best */
STRINGZILLA_API sz_status_t sz_allocator_init_device_cuda(sz_allocator_t *allocator);
/** @copydoc sz_allocator_init_pinned_best */
STRINGZILLA_API sz_status_t sz_allocator_init_pinned_cuda(sz_allocator_t *allocator);
#endif

#if STRINGZILLA_ARCH_ROCM_
/** @copydoc sz_allocator_init_unified_best */
STRINGZILLA_API sz_status_t sz_allocator_init_unified_rocm(sz_allocator_t *allocator);
/** @copydoc sz_allocator_init_device_best */
STRINGZILLA_API sz_status_t sz_allocator_init_device_rocm(sz_allocator_t *allocator);
/** @copydoc sz_allocator_init_pinned_best */
STRINGZILLA_API sz_status_t sz_allocator_init_pinned_rocm(sz_allocator_t *allocator);
#endif

#if STRINGZILLA_ARCH_METAL_
/** @copydoc sz_allocator_init_unified_best */
STRINGZILLA_API sz_status_t sz_allocator_init_unified_metal(sz_allocator_t *allocator);
#endif

#pragma endregion Memory Management

#pragma region Helper API

/**
 *  @brief Initializes a lookup table for converting ASCII characters to lowercase.
 *  @param[out] lut Lookup table to be initialized. Must be exactly 256 bytes long.
 *  @see SWAR swap case: http://0x80.pl/notesen/2016-01-06-swar-swap-case.html
 *
 *  ASCII characters [A, Z] map to decimals [65, 90], and [a, z] map to [97, 122]. So there are 26
 *  English letters, shifted by 32 values, meaning that a conversion can be done by flipping the 5th
 *  bit of each inappropriate character byte. This, however, breaks for extended ASCII, so a
 *  different solution is needed.
 */
STRINGZILLA_INLINE void sz_strings_lookup_init_lower(char lut[sz_at_least_(256)]) {
    static sz_u8_t const lowered[256] = {
        0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  //
        16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  //
        32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  //
        48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  //
        64,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, //
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 91,  92,  93,  94,  95,  //
        96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, //
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, //
        128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, //
        144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, //
        160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, //
        176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, //
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, //
        240, 241, 242, 243, 244, 245, 246, 215, 248, 249, 250, 251, 252, 253, 254, 223, //
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, //
        240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, //
    };
    for (sz_size_t byte_index = 0; byte_index < 256; ++byte_index) lut[byte_index] = lowered[byte_index];
}

/**
 *  @brief Initializes a lookup table for converting ASCII characters to uppercase.
 *  @param[out] lut Lookup table to be initialized. Must be exactly 256 bytes long.
 *  @see SWAR swap case: http://0x80.pl/notesen/2016-01-06-swar-swap-case.html
 *
 *  ASCII characters [A, Z] map to decimals [65, 90], and [a, z] map to [97, 122]. So there are 26
 *  English letters, shifted by 32 values, meaning that a conversion can be done by flipping the 5th
 *  bit of each inappropriate character byte. This, however, breaks for extended ASCII, so a
 *  different solution is needed.
 */
STRINGZILLA_INLINE void sz_strings_lookup_init_upper(char lut[sz_at_least_(256)]) {
    static sz_u8_t const upped[256] = {
        0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  //
        16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  //
        32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  //
        48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  //
        64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  //
        80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  //
        96,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  //
        80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  123, 124, 125, 126, 127, //
        128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, //
        144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, //
        160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, //
        176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, //
        192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, //
        208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, //
        192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, //
        208, 209, 210, 211, 212, 213, 214, 247, 216, 217, 218, 219, 220, 221, 222, 255, //
    };
    for (sz_size_t byte_index = 0; byte_index < 256; ++byte_index) lut[byte_index] = upped[byte_index];
}

/**
 *  @brief Initializes a lookup table for converting bytes to ASCII characters.
 *  @param[out] lut Lookup table to be initialized. Must be exactly 256 bytes long.
 */
STRINGZILLA_CONSTEXPR void sz_strings_lookup_init_ascii(char lut[sz_at_least_(256)]) {
    for (sz_size_t byte_index = 0; byte_index < 256; ++byte_index) lut[byte_index] = (sz_u8_t)(byte_index & 0x7F);
}

#pragma endregion Helper API

#include "stringzilla/strings/serial.h"
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/strings/haswell.h"
#include "stringzilla/strings/icelake.h"
#include "stringzilla/strings/neon.h"
#include "stringzilla/strings/sve.h"
#include "stringzilla/strings/v128relaxed.h"
#include "stringzilla/strings/v128.h"
#include "stringzilla/strings/rvv.h"
#include "stringzilla/strings/loongsonasx.h"
#include "stringzilla/strings/powervsx.h"
#include "stringzilla/strings/cuda.cuh"
#include "stringzilla/strings/rocm.cuh"
#include "stringzilla/strings/metal.h"
#endif // STRINGZILLA_HEADER_ONLY

/*  Header-only builds link no library, so their dispatch points only report it missing. */
#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_strings_to_tape_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                    sz_capability_t capabilities, sz_stream_t stream) {
    sz_unused_(strings), sz_unused_(order), sz_unused_(count), sz_unused_(allocator), sz_unused_(tape),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_strings_to_rows_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_strings_rows_t const *rows, sz_size_t *longest,
                                                    sz_capability_t capabilities, sz_stream_t stream) {
    sz_unused_(strings), sz_unused_(order), sz_unused_(count), sz_unused_(rows), sz_unused_(longest),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_strings_escape_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                   sz_strings_span_t *tape, sz_capability_t capabilities,
                                                   sz_stream_t stream) {
    sz_unused_(strings), sz_unused_(order), sz_unused_(count), sz_unused_(kind), sz_unused_(allocator),
        sz_unused_(tape), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_strings_unescape_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_size_t *malformed_index,
                                                     sz_capability_t capabilities, sz_stream_t stream) {
    sz_unused_(strings), sz_unused_(order), sz_unused_(count), sz_unused_(kind), sz_unused_(allocator),
        sz_unused_(tape), sz_unused_(malformed_index), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_strings_lookup_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, char const lut[sz_at_least_(256)],
                                                   sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                   sz_capability_t capabilities, sz_stream_t stream) {
    sz_unused_(strings), sz_unused_(order), sz_unused_(count), sz_unused_(lut), sz_unused_(allocator), sz_unused_(tape),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_strings_normalize_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_normal_form_t form, sz_allocator_t *allocator,
                                                      sz_strings_span_t *tape, sz_capability_t capabilities,
                                                      sz_stream_t stream) {
    sz_unused_(strings), sz_unused_(order), sz_unused_(count), sz_unused_(form), sz_unused_(allocator),
        sz_unused_(tape), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_strings_split_best(sz_cptr_t text, sz_size_t length, sz_byteset_t const *delimiters,
                                                  sz_split_kind_t kind, sz_allocator_t *allocator,
                                                  sz_strings_span_t *tape, sz_capability_t capabilities,
                                                  sz_stream_t stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(delimiters), sz_unused_(kind), sz_unused_(allocator),
        sz_unused_(tape), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_strings_split_lines_best(sz_cptr_t text, sz_size_t length, sz_allocator_t *allocator,
                                                        sz_strings_span_t *tape, sz_capability_t capabilities,
                                                        sz_stream_t stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(allocator), sz_unused_(tape), sz_unused_(capabilities),
        sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_strings_join_best(sz_strings_view_t const *columns, sz_size_t columns_count,
                                                 sz_size_t rows, sz_string_view_t const *frames,
                                                 sz_string_view_t between, sz_ptr_t output, sz_size_t *length,
                                                 sz_capability_t capabilities, sz_stream_t stream) {
    sz_unused_(columns), sz_unused_(columns_count), sz_unused_(rows), sz_unused_(frames), sz_unused_(between),
        sz_unused_(output), sz_unused_(length), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_strings_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                   sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_unified_best(sz_allocator_t *allocator, sz_capability_t capabilities) {
    sz_unused_(allocator), sz_unused_(capabilities);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_device_best(sz_allocator_t *allocator, sz_capability_t capabilities) {
    sz_unused_(allocator), sz_unused_(capabilities);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_pinned_best(sz_allocator_t *allocator, sz_capability_t capabilities) {
    sz_unused_(allocator), sz_unused_(capabilities);
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_STRINGS_H_
