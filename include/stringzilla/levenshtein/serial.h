/**
 *  @file include/stringzilla/levenshtein/serial.h
 *  @author Ash Vardanian
 *  @date September 6, 2023
 *  @brief Serial backend for Levenshtein edit distances with Myers' bit-parallel algorithm.
 *
 *  It scores a batch of prepared queries against a batch of candidates, over bytes or over UTF-8
 *  runes, and holds the cross-product engine every backend scores through and the step primitives
 *  every SIMD backend mirrors.
 *
 *  @sa include/stringzilla/levenshtein.h
 */
#ifndef STRINGZILLA_LEVENSHTEIN_SERIAL_H_
#define STRINGZILLA_LEVENSHTEIN_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h" // `sz_utf8_next_rune_`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Generic Public Helpers

/** One query's bit-parallel state, prepared once and read by every candidate: every symbol maps to
 *  a class that indexes the match masks, a view over caller-owned memory of @c classes × stride
 *  entries with no length ceiling. */
typedef struct sz_levenshtein_query_t {

    /** Row @c class * stride + word, bit @c i: symbol @c word * 64 + i is that class. */
    sz_u64_t const *masks;

    /** Bytes only: the mask row each of the 256 byte values reads. */
    sz_u8_t const *byte_to_class;

    /** UTF-8 only: class row per 256-rune page, zero for a page the query lacks. */
    sz_u16_t const *page_rows;

    /** UTF-8 only: the rows the page table indexes, 256 classes each. */
    sz_u32_t const *class_rows;

    /** Mask rows: one per distinct symbol, plus the row an absent symbol reads. */
    sz_size_t classes;

    /** Words from one class's row to the next: the query's words, warp padded. */
    sz_size_t stride;

    /** In symbols: bytes, or runes for a UTF-8 query. */
    sz_size_t length;
} sz_levenshtein_query_t;

/** Byte values, which is both the byte-to-class map's length and the most classes a byte
 *  query can take. */
enum { sz_levenshtein_byte_classes_k = 256 };

/** Words a class row is padded to: a warp's width, so a lane reading its own word skewed owns
 *  its own bank. */
enum { sz_levenshtein_words_stride_k = 32 };

/** Words a query of @p length symbols spans - every class's row holds that many, one
 *  bit per symbol. */
STRINGZILLA_HELPER_AUTO sz_size_t sz_levenshtein_query_words(sz_size_t length) { return (length + 63) / 64; }

/** Words from one class's mask row to the next: the query's words, padded
 *  to @c sz_levenshtein_words_stride_k. */
STRINGZILLA_HELPER_AUTO sz_size_t sz_levenshtein_query_stride(sz_size_t length) {
    return (sz_levenshtein_query_words(length) + sz_levenshtein_words_stride_k - 1) &
           ~(sz_size_t)(sz_levenshtein_words_stride_k - 1);
}

/** Mask entries a byte query of @p length bytes can take: a row per distinct byte, plus the
 *  absent byte's row. */
STRINGZILLA_HELPER_AUTO sz_size_t sz_levenshtein_query_mask_entries(sz_size_t length) {
    return sz_min_of_two(length + 1, (sz_size_t)sz_levenshtein_byte_classes_k) * sz_levenshtein_query_stride(length);
}

/** Mask entries a UTF-8 query of @p runes runes can take: a row per distinct rune, plus the
 *  absent rune's row. */
STRINGZILLA_HELPER_AUTO sz_size_t sz_levenshtein_query_mask_entries_utf8(sz_size_t runes) {
    return (runes + 1) * sz_levenshtein_query_stride(runes);
}

/**
 *  @brief Builds the match masks of the byte string @p text into @p masks and points
 *      @p query at them.
 *  @param[out] masks Caller-owned, @c sz_levenshtein_query_mask_entries(length) entries, of which
 *      the first @c classes × stride are written.
 *  @param[out] byte_to_class Caller-owned, @c sz_levenshtein_byte_classes_k entries; the
 *      transposes read it.
 *  @return @c sz_success_k, or @c sz_unexpected_dimensions_k for an empty query, whose distance is
 *      every candidate's length.
 */
STRINGZILLA_HELPER_AUTO sz_status_t sz_levenshtein_query_prepare(sz_cptr_t text, sz_size_t length, sz_u64_t *masks,
                                                                 sz_u8_t *byte_to_class,
                                                                 sz_levenshtein_query_t *query) {
    if (length == 0) return sz_unexpected_dimensions_k;
    // First pass: flag the byte values the query holds, then hand them dense classes in byte order. The row past
    // them is what a byte the query lacks reads, and a query holding all 256 values has no such byte.
    for (sz_size_t byte = 0; byte != sz_levenshtein_byte_classes_k; ++byte) byte_to_class[byte] = 0;
    for (sz_size_t position = 0; position != length; ++position) byte_to_class[(sz_u8_t)text[position]] = 1;
    sz_size_t distinct = 0;
    for (sz_size_t byte = 0; byte != sz_levenshtein_byte_classes_k; ++byte) distinct += byte_to_class[byte];
    sz_size_t const classes = sz_min_of_two(distinct + 1, (sz_size_t)sz_levenshtein_byte_classes_k);
    for (sz_size_t byte = 0, next_class = 0; byte != sz_levenshtein_byte_classes_k; ++byte)
        byte_to_class[byte] = byte_to_class[byte] ? (sz_u8_t)next_class++ : (sz_u8_t)(classes - 1);
    // Second pass: the row stride is known, so every byte sets its bit in its class's row.
    sz_size_t const stride = sz_levenshtein_query_stride(length);
    for (sz_size_t entry = 0; entry != classes * stride; ++entry) masks[entry] = 0;
    for (sz_size_t position = 0; position != length; ++position) {
        sz_u64_t *const row = masks + (sz_size_t)byte_to_class[(sz_u8_t)text[position]] * stride;
        row[position >> 6] |= (sz_u64_t)1 << (position & 63);
    }
    query->masks = masks;
    query->byte_to_class = byte_to_class;
    query->page_rows = STRINGZILLA_NULL;
    query->class_rows = STRINGZILLA_NULL;
    query->classes = classes;
    query->stride = stride;
    query->length = length;
    return sz_success_k;
}

/** Unicode as 256-rune pages: the page table a UTF-8 query indexes by @c rune >> 8. Its
 *  @c u16 entries fill a whole number of cache lines, so the class rows follow it at a
 *  constant aligned offset. */
enum { sz_levenshtein_utf8_pages_k = 0x110000 / 256 };
sz_static_assert_(sz_levenshtein_utf8_pages_k * sizeof(sz_u16_t) % 64 == 0,
                  sz_levenshtein_utf8_page_table_fills_cache_lines);

/** The class rows behind the page table: 256 classes per row, row zero all zeros. */
STRINGZILLA_API_COMPTIME sz_u32_t const *sz_levenshtein_utf8_class_rows_(sz_levenshtein_query_t const *query) {
    return query->class_rows;
}

/** The class of @p rune under a UTF-8 @p query: two loads through the page table, zero for a rune
 *  the query lacks. */
STRINGZILLA_HELPER_AUTO sz_u32_t sz_levenshtein_utf8_class(sz_levenshtein_query_t const *query, sz_rune_t rune) {
    return query->class_rows[(sz_size_t)query->page_rows[rune >> 8] * 256 + (rune & 255)];
}

/** Runes in @p text under the decoding the family applies - one @c U+FFFD per ill-formed byte, the
 *  grid the segmenters use - which differs from @c sz_utf8_count on ill-formed input. */
STRINGZILLA_API_COMPTIME sz_size_t sz_levenshtein_utf8_runes(sz_cptr_t text, sz_size_t length) {
    sz_size_t runes = 0;
    for (sz_size_t position = 0; position < length; ++runes) sz_utf8_next_rune_(text, length, &position);
    return runes;
}

/** Scratch the page table of a @p runes -rune query takes: the page index, then one class row per
 *  page the query can touch plus the all-zero row. */
STRINGZILLA_API_COMPTIME sz_size_t sz_levenshtein_utf8_pages_bytes(sz_size_t runes) {
    sz_size_t const rows = sz_min_of_two(runes, (sz_size_t)sz_levenshtein_utf8_pages_k) + 1;
    return sz_levenshtein_utf8_pages_k * sizeof(sz_u16_t) + rows * 256 * sizeof(sz_u32_t);
}

/**
 *  @brief Builds the match masks of the UTF-8 string @p text into @p masks, its rune classes into
 *      the page table at @p pages, and points @p query at both. Classes follow first appearance.
 *  @param[out] masks Caller-owned, @c sz_levenshtein_query_mask_entries_utf8(runes) entries for a
 *      query of @c runes runes, of which the first @c classes × stride are written.
 *  @param[out] pages Caller-owned, @c sz_levenshtein_utf8_pages_bytes(runes)
 *      bytes, cache-line aligned.
 *  @return @c sz_success_k, or @c sz_unexpected_dimensions_k for an empty query, whose distance is
 *      every candidate's rune count.
 */
STRINGZILLA_API_COMPTIME sz_status_t sz_levenshtein_query_prepare_utf8(sz_cptr_t text, sz_size_t length,
                                                                       sz_u64_t *masks, void *pages,
                                                                       sz_levenshtein_query_t *query) {
    if (length == 0) return sz_unexpected_dimensions_k;
    sz_u16_t *const page_rows = (sz_u16_t *)pages;
    sz_u32_t *const class_rows = (sz_u32_t *)(page_rows + sz_levenshtein_utf8_pages_k);
    for (sz_size_t page = 0; page != sz_levenshtein_utf8_pages_k; ++page) page_rows[page] = 0;
    for (sz_size_t entry = 0; entry != 256; ++entry) class_rows[entry] = 0;
    // First pass: a fresh row for every page first seen, a fresh class for every rune first seen.
    sz_size_t count = 0, rows = 1, classes = 1;
    for (sz_size_t position = 0; position < length; ++count) {
        sz_rune_t const rune = sz_utf8_next_rune_(text, length, &position);
        if (page_rows[rune >> 8] == 0) {
            page_rows[rune >> 8] = (sz_u16_t)rows++;
            for (sz_size_t entry = 0; entry != 256; ++entry) class_rows[page_rows[rune >> 8] * 256 + entry] = 0;
        }
        sz_u32_t *const class_slot = class_rows + (sz_size_t)page_rows[rune >> 8] * 256 + (rune & 255);
        if (*class_slot == 0) *class_slot = (sz_u32_t)classes++;
    }
    query->page_rows = page_rows;
    query->class_rows = class_rows;
    query->classes = classes;
    sz_size_t const stride = sz_levenshtein_query_stride(count);
    for (sz_size_t entry = 0; entry != classes * stride; ++entry) masks[entry] = 0;
    // Second pass: the row stride is known, so every rune sets its bit in its class's row.
    sz_size_t rune_index = 0;
    for (sz_size_t position = 0; position < length; ++rune_index) {
        sz_rune_t const rune = sz_utf8_next_rune_(text, length, &position);
        sz_u64_t *const row = masks + (sz_size_t)sz_levenshtein_utf8_class(query, rune) * stride;
        row[rune_index >> 6] |= (sz_u64_t)1 << (rune_index & 63);
    }
    query->masks = masks;
    query->byte_to_class = STRINGZILLA_NULL;
    query->stride = stride;
    query->length = count;
    return sz_success_k;
}

/** Positions one transpose spans: a candidate's classes are emitted this many at
 *  a time, transposed. */
enum { sz_levenshtein_positions_per_transpose_k = 256 };

/** How wide a transpose's class ids are: bytes are their own classes and fit a byte, rune
 *  classes need four. */
typedef enum sz_levenshtein_classes_width_t {
    sz_levenshtein_classes_u8_k = 1,
    sz_levenshtein_classes_u32_k = 4,
} sz_levenshtein_classes_width_t;

/**
 *  @brief Transposes the next @p positions symbols of each of @p candidates texts into class ids.
 *
 *  The symbol at transpose position @c p of candidate @c c lands at p × candidates + c, and a
 *  candidate past its text reads as class zero. Returns the positions filled, below @p positions
 *  only once every candidate has run dry.
 *
 *  A candidate that runs dry inside the transpose writes its exact symbol count to
 *  @p symbol_counts; until then the entry keeps the byte count the caller seeded.
 *
 *  @param[inout] cursors Byte offsets into each candidate's text, advanced past what was emitted.
 *  @param[in] transpose_start The absolute position of the transpose's first symbol.
 *  @param[out] transpose_classes @c positions × candidates class ids, @c sz_u32_t for
 *      the transposes here.
 */
typedef sz_size_t (*sz_levenshtein_transpose_t)(sz_levenshtein_query_t const *query, sz_cptr_t const *texts,
                                                sz_u64_t const *byte_counts, sz_size_t candidates, sz_size_t *cursors,
                                                sz_u64_t *symbol_counts, sz_size_t transpose_start, sz_size_t positions,
                                                void *transpose_classes);

/** The byte transpose: every byte takes the class the query gave it, and the byte count is
 *  the symbol count. */
STRINGZILLA_API_COMPTIME sz_size_t sz_levenshtein_transpose(sz_levenshtein_query_t const *query, sz_cptr_t const *texts,
                                                            sz_u64_t const *byte_counts, sz_size_t candidates,
                                                            sz_size_t *cursors, sz_u64_t *symbol_counts,
                                                            sz_size_t transpose_start, sz_size_t positions,
                                                            void *transpose_classes) {
    sz_unused_(symbol_counts), sz_unused_(transpose_start);
    sz_u8_t const *const byte_to_class = query->byte_to_class;
    sz_u32_t *const classes = (sz_u32_t *)transpose_classes;
    sz_size_t filled = 0;
    for (sz_size_t candidate = 0; candidate != candidates; ++candidate)
        filled = sz_max_of_two(filled,
                               sz_min_of_two(positions, (sz_size_t)byte_counts[candidate] - cursors[candidate]));
    for (sz_size_t position = 0; position != filled; ++position)
        for (sz_size_t candidate = 0; candidate != candidates; ++candidate)
            classes[position * candidates + candidate] =
                cursors[candidate] < byte_counts[candidate]
                    ? byte_to_class[(sz_u8_t)texts[candidate][cursors[candidate]++]]
                    : 0;
    return filled;
}

/** The UTF-8 transpose: every rune decodes and takes the class the query gave it. */
STRINGZILLA_API_COMPTIME sz_size_t sz_levenshtein_transpose_utf8(sz_levenshtein_query_t const *query,
                                                                 sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                                 sz_size_t candidates, sz_size_t *cursors,
                                                                 sz_u64_t *symbol_counts, sz_size_t transpose_start,
                                                                 sz_size_t positions, void *transpose_classes) {
    sz_u32_t *const classes = (sz_u32_t *)transpose_classes;
    sz_size_t filled = 0;
    for (sz_size_t candidate = 0; candidate != candidates; ++candidate) {
        sz_size_t emitted = 0;
        for (; emitted != positions && cursors[candidate] < byte_counts[candidate]; ++emitted)
            classes[emitted * candidates + candidate] = sz_levenshtein_utf8_class(
                query, sz_utf8_next_rune_(texts[candidate], byte_counts[candidate], cursors + candidate));
        if (emitted != 0 && cursors[candidate] == byte_counts[candidate])
            symbol_counts[candidate] = transpose_start + emitted;
        for (sz_size_t position = emitted; position != positions; ++position)
            classes[position * candidates + candidate] = 0;
        filled = sz_max_of_two(filled, emitted);
    }
    return filled;
}

#pragma endregion Generic Public Helpers

#pragma region Generic Internal Helpers

/** The bit of the query's last symbol within its last word: where the score deltas are read. */
STRINGZILLA_HELPER_AUTO sz_u64_t sz_levenshtein_last_symbol_bit_(sz_size_t length) {
    return (sz_u64_t)1 << ((length - 1) & 63);
}

/** The shift that turns that bit into a one. */
STRINGZILLA_HELPER_AUTO sz_size_t sz_levenshtein_last_symbol_shift_(sz_size_t length) { return (length - 1) & 63; }

/** When a sweep must next read scores, and whose: one compare a position, and a walk
 *  with no test. */
typedef struct sz_levenshtein_deadline_t {

    /** The earliest position, in the sweep's own coordinates, at which a candidate ends. */
    sz_size_t position;

    /** The candidates whose text ends at that position, one bit each, at most one per lane. */
    sz_u64_t retiring;
} sz_levenshtein_deadline_t;

/** The next deadline over the candidates still @p unread, whose symbol counts @p counts holds. */
STRINGZILLA_HELPER_INLINE sz_levenshtein_deadline_t sz_levenshtein_deadline_(sz_u64_t unread, sz_u64_t const *counts) {
    sz_levenshtein_deadline_t deadline;
    deadline.position = STRINGZILLA_SIZE_MAX, deadline.retiring = 0;
    for (sz_u64_t pending = unread; pending; pending &= pending - 1) {
        sz_size_t const candidate = (sz_size_t)sz_u64_ctz(pending);
        sz_size_t const ends_at = (sz_size_t)counts[candidate];
        if (ends_at < deadline.position) deadline.position = ends_at, deadline.retiring = (sz_u64_t)1 << candidate;
        else if (ends_at == deadline.position) deadline.retiring |= (sz_u64_t)1 << candidate;
    }
    return deadline;
}

/** Rounds @p bytes up to a cache line, so every scratch area below starts aligned. */
STRINGZILLA_HELPER_AUTO sz_size_t sz_levenshtein_align64_(sz_size_t bytes) { return (bytes + 63) & ~(sz_size_t)63; }

/** Eight bytes as their eight classes, packed the same way, so a transpose's byte transposes
 *  stay byte transposes. */
STRINGZILLA_HELPER_INLINE sz_u64_t sz_levenshtein_octet_classes_(sz_u8_t const *byte_to_class, sz_u64_t octet) {
    sz_u64_t classes = 0;
    for (sz_size_t byte = 0; byte != 8; ++byte)
        classes |= (sz_u64_t)byte_to_class[(sz_u8_t)(octet >> (byte * 8))] << (byte * 8);
    return classes;
}

/** Every candidate's distance to an empty byte query is its byte count. */
STRINGZILLA_HELPER_AUTO void sz_levenshtein_byte_counts_as_distances_(sz_sequence_t const *candidates,
                                                                      sz_size_t *distances) {
    for (sz_size_t index = 0; index != candidates->count; ++index)
        distances[index] = candidates->get_length(candidates->handle, index);
}

/** Every candidate's distance to an empty UTF-8 query is its rune count. */
STRINGZILLA_HELPER_AUTO void sz_levenshtein_rune_counts_as_distances_(sz_sequence_t const *candidates,
                                                                      sz_size_t *distances) {
    for (sz_size_t index = 0; index != candidates->count; ++index)
        distances[index] = sz_levenshtein_utf8_runes(candidates->get_start(candidates->handle, index),
                                                     candidates->get_length(candidates->handle, index));
}

#pragma endregion Generic Internal Helpers

#pragma region Cross Product Engine

/**
 *  @brief A batch of prepared queries, the block it lives in, and the round's scratch beside it.
 *
 *  The first four members are @ref sz_levenshtein_query_t in tensor form: every query's mask plane
 *  back to back, which a kernel indexes by arithmetic where an array of structs would need a
 *  pointer chase. The launch geometry lives in @c memory 's head, which only the tier named by
 *  @c capability reads.
 */
typedef struct sz_levenshtein_engine_t {

    /** Every query's mask plane, back to back, @c masks_offsets addressing them. */
    sz_u64_t const *masks;

    /** The @b [count+1] word offsets into @c masks, the last being its length. */
    sz_size_t const *masks_offsets;

    /** The mask row each symbol reads, laid out as @c symbol spells it. */
    void const *symbol_to_class;

    /** The @b [count] symbols each query spans, which seeds its score bit. */
    sz_u32_t const *lengths;

    /** Queries prepared, which is the first axis of every output. */
    sz_size_t count;

    /** The alphabet the batch was prepared over, and how to read the class map. */
    sz_levenshtein_symbol_t symbol;

    /** The tier @c _init_* resolved, and the only one that may score with it. */
    sz_capability_t capability;

    /** What built both blocks below and what grows the second. */
    sz_memory_allocator_t alloc;

    /** The batch's block, fixed for the engine's life, its head tier-private. */
    void *memory;

    /** Bytes of that block. */
    sz_size_t memory_bytes;

    /** The round's block, grown by a compute verb and never shrunk. */
    void *scratch;

    /** Bytes of that block, zero until the first round sizes it. */
    sz_size_t scratch_bytes;
} sz_levenshtein_engine_t;

/** One query's shape, measured before the block that holds it exists. */
typedef struct sz_levenshtein_engine_shape_t {

    /** Symbols the query spans: bytes, or runes under @c sz_levenshtein_runes_k. */
    sz_u32_t length;

    /** Mask rows its plane takes, the row an absent symbol reads included, zero if empty. */
    sz_u32_t classes;

    /** Class rows its page table takes, which is zero under @c sz_levenshtein_bytes_k. */
    sz_u32_t rows;
} sz_levenshtein_engine_shape_t;

/** Where a batch's tensors sit inside one block, so the sizing pass and the filling pass
 *  agree by construction. */
typedef struct sz_levenshtein_engine_layout_t {

    /** Bytes the block opens with, which only the tier that asked for them reads. */
    sz_size_t head_bytes;

    /** Byte offset of the mask planes, which is the head rounded to a cache line. */
    sz_size_t masks_offset;

    /** Words every plane spans together. */
    sz_size_t masks_words;

    /** Byte offset of the @b [count+1] plane offsets. */
    sz_size_t offsets_offset;

    /** Byte offset of the @b [count] symbol counts. */
    sz_size_t lengths_offset;

    /** Byte offset of the class map, laid out as the alphabet spells it. */
    sz_size_t classes_offset;

    /** Bytes that map spans. */
    sz_size_t classes_bytes;

    /** Bytes the whole block takes. */
    sz_size_t total_bytes;
} sz_levenshtein_engine_layout_t;

/**
 *  @brief The tier a batch of @p symbol scores on, given @p caps, resolved once so no
 *      round asks again.
 *
 *  Ice Lake's byte lanes have no rune arm, so a rune batch stops at Skylake however capable
 *  the machine is.
 */
STRINGZILLA_HELPER_AUTO sz_capability_t sz_levenshtein_tier_for(sz_capability_t caps, sz_levenshtein_symbol_t symbol) {
#if STRINGZILLA_TARGET_ICELAKE
    if ((caps & sz_cap_icelake_k) != 0 && symbol == sz_levenshtein_bytes_k) return sz_cap_icelake_k;
#endif
#if STRINGZILLA_TARGET_SKYLAKE
    if ((caps & sz_cap_skylake_k) != 0) return sz_cap_skylake_k;
#endif
#if STRINGZILLA_TARGET_HASWELL
    if ((caps & sz_cap_haswell_k) != 0) return sz_cap_haswell_k;
#endif
    return sz_unused_(caps), sz_unused_(symbol), sz_cap_serial_k;
}

/** The page table and the class rows of query @p index, which the rune alphabet keeps one
 *  block per query. */
STRINGZILLA_HELPER_INLINE void sz_levenshtein_engine_pages_(sz_levenshtein_engine_t const *engine, sz_size_t index,
                                                            sz_u16_t const **page_rows, sz_u32_t const **class_rows) {
    sz_size_t const *const pages_offsets = (sz_size_t const *)engine->symbol_to_class;
    sz_u16_t const *const pages = (sz_u16_t const *)((sz_cptr_t)engine->symbol_to_class + pages_offsets[index]);
    *page_rows = pages;
    *class_rows = (sz_u32_t const *)(pages + sz_levenshtein_utf8_pages_k);
}

/** Materializes one row of the batch as the kit's own query type: base pointers and
 *  arithmetic, no storage. */
STRINGZILLA_HELPER_AUTO sz_levenshtein_query_t sz_levenshtein_engine_row_(sz_levenshtein_engine_t const *engine,
                                                                          sz_size_t index) {
    sz_levenshtein_query_t row = {STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL, 0, 0, 0};
    row.masks = engine->masks + engine->masks_offsets[index];
    row.length = engine->lengths[index];
    row.stride = sz_levenshtein_query_stride(row.length);
    row.classes =
        row.stride ? (engine->masks_offsets[index + 1] - engine->masks_offsets[index]) / row.stride : 0;
    if (engine->symbol == sz_levenshtein_bytes_k) {
        row.byte_to_class = (sz_u8_t const *)engine->symbol_to_class + index * sz_levenshtein_byte_classes_k;
        row.page_rows = STRINGZILLA_NULL, row.class_rows = STRINGZILLA_NULL;
    }
    else {
        row.byte_to_class = STRINGZILLA_NULL;
        sz_levenshtein_engine_pages_(engine, index, &row.page_rows, &row.class_rows);
    }
    return row;
}

/** The widest word count the batch holds, which is what one round's verticals are sized for. */
STRINGZILLA_HELPER_AUTO sz_size_t sz_levenshtein_engine_words_max_(sz_levenshtein_engine_t const *engine) {
    sz_size_t words = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        words = sz_max_of_two(words, sz_levenshtein_query_words(engine->lengths[index]));
    return words;
}

/** Every candidate's distance to an empty query, which is its own symbol count in
 *  the batch's alphabet. */
STRINGZILLA_HELPER_AUTO void sz_levenshtein_engine_empty_row_(sz_levenshtein_engine_t const *engine,
                                                              sz_sequence_t const *candidates, sz_size_t *row) {
    if (engine->symbol == sz_levenshtein_bytes_k) sz_levenshtein_byte_counts_as_distances_(candidates, row);
    else sz_levenshtein_rune_counts_as_distances_(candidates, row);
}

/** Bytes one round's verticals take: a cache line to align on, then one set per
 *  register of candidates. */
STRINGZILLA_HELPER_AUTO sz_size_t sz_levenshtein_engine_verticals_bytes_(sz_size_t registers_per_position,
                                                                         sz_size_t words, sz_size_t vertical_bytes) {
    return 64 + registers_per_position * words * vertical_bytes;
}

/** The round's verticals, cache-line aligned inside the block the round grew for them. */
STRINGZILLA_API_COMPTIME void *sz_levenshtein_engine_verticals_(sz_levenshtein_engine_t const *engine) {
    return (void *)sz_levenshtein_align64_((sz_size_t)engine->scratch);
}

/** Grows @p engine 's round block to @p bytes, which a round does once and never undoes. */
STRINGZILLA_API_COMPTIME sz_status_t sz_levenshtein_engine_scratch_(sz_levenshtein_engine_t *engine, sz_size_t bytes) {
    if (engine->scratch_bytes >= bytes) return sz_success_k;
    void *const grown = engine->alloc.allocate(bytes, engine->alloc.handle);
    if (!grown) return sz_bad_alloc_k;
    if (engine->scratch) engine->alloc.free(engine->scratch, engine->scratch_bytes, engine->alloc.handle);
    engine->scratch = grown, engine->scratch_bytes = bytes;
    return sz_success_k;
}

/** Flags the distinct byte values of @p text into @p seen and answers the mask rows they take, the
 *  absent one too. */
STRINGZILLA_API_COMPTIME sz_size_t sz_levenshtein_engine_byte_classes_(sz_cptr_t text, sz_size_t length,
                                                                       sz_u8_t *seen) {
    sz_size_t distinct = 0;
    for (sz_size_t byte = 0; byte != sz_levenshtein_byte_classes_k; ++byte) seen[byte] = 0;
    for (sz_size_t position = 0; position != length; ++position) seen[(sz_u8_t)text[position]] = 1;
    for (sz_size_t byte = 0; byte != sz_levenshtein_byte_classes_k; ++byte) distinct += seen[byte];
    return sz_min_of_two(distinct + 1, (sz_size_t)sz_levenshtein_byte_classes_k);
}

/**
 *  @brief Counts the runes, the classes and the page rows of @p text without building a mask
 *      plane for it.
 *  @param[out] page_rows Caller-owned, @c sz_levenshtein_utf8_pages_k entries, rewritten
 *      on every call.
 *  @param[out] class_rows Caller-owned, one row of 256 per page the text can touch plus
 *      the all-zero row.
 *  @param[out] shape The counts a batch is sized from, all three zero for an empty text.
 */
STRINGZILLA_API_COMPTIME void sz_levenshtein_engine_rune_classes_(sz_cptr_t text, sz_size_t length, sz_u16_t *page_rows,
                                                                  sz_u32_t *class_rows,
                                                                  sz_levenshtein_engine_shape_t *shape) {
    for (sz_size_t page = 0; page != sz_levenshtein_utf8_pages_k; ++page) page_rows[page] = 0;
    for (sz_size_t entry = 0; entry != 256; ++entry) class_rows[entry] = 0;
    sz_size_t runes = 0, rows = 1, classes = 1;
    for (sz_size_t position = 0; position < length; ++runes) {
        sz_rune_t const rune = sz_utf8_next_rune_(text, length, &position);
        if (page_rows[rune >> 8] == 0) {
            page_rows[rune >> 8] = (sz_u16_t)rows++;
            for (sz_size_t entry = 0; entry != 256; ++entry) class_rows[page_rows[rune >> 8] * 256 + entry] = 0;
        }
        sz_u32_t *const class_slot = class_rows + (sz_size_t)page_rows[rune >> 8] * 256 + (rune & 255);
        if (*class_slot == 0) *class_slot = (sz_u32_t)classes++;
    }
    shape->length = (sz_u32_t)runes;
    shape->classes = runes != 0 ? (sz_u32_t)classes : 0;
    shape->rows = runes != 0 ? (sz_u32_t)rows : 0;
}

/** Measures every query of @p queries, which is what sizes the block they are
 *  then prepared into. */
STRINGZILLA_API_COMPTIME sz_status_t sz_levenshtein_engine_measure_(sz_sequence_t const *queries,
                                                                    sz_levenshtein_symbol_t symbol,
                                                                    sz_memory_allocator_t const *alloc,
                                                                    sz_levenshtein_engine_shape_t *shapes) {
    if (queries->count == 0) return sz_success_k;
    if (symbol == sz_levenshtein_bytes_k) {
        sz_u8_t seen[sz_levenshtein_byte_classes_k];
        for (sz_size_t index = 0; index != queries->count; ++index) {
            sz_size_t const length = queries->get_length(queries->handle, index);
            sz_cptr_t const text = queries->get_start(queries->handle, index);
            shapes[index].length = (sz_u32_t)length;
            shapes[index].classes = length != 0
                                        ? (sz_u32_t)sz_levenshtein_engine_byte_classes_(text, length, seen)
                                        : 0;
            shapes[index].rows = 0;
        }
        return sz_success_k;
    }
    // Runes never outnumber bytes, so the longest text bounds the page table every measurement reuses.
    sz_size_t longest = 0;
    for (sz_size_t index = 0; index != queries->count; ++index)
        longest = sz_max_of_two(longest, queries->get_length(queries->handle, index));
    sz_size_t const pages_bytes = sz_levenshtein_utf8_pages_bytes(longest);
    sz_u16_t *const page_rows = (sz_u16_t *)alloc->allocate(pages_bytes, alloc->handle);
    if (!page_rows) return sz_bad_alloc_k;
    sz_u32_t *const class_rows = (sz_u32_t *)(page_rows + sz_levenshtein_utf8_pages_k);
    for (sz_size_t index = 0; index != queries->count; ++index)
        sz_levenshtein_engine_rune_classes_(queries->get_start(queries->handle, index),
                                            queries->get_length(queries->handle, index), page_rows, class_rows,
                                            shapes + index);
    alloc->free(page_rows, pages_bytes, alloc->handle);
    return sz_success_k;
}

/** Lays @p count queries of the given @p shapes out inside one block, after a tier's
 *  own @p head_bytes. */
STRINGZILLA_HELPER_AUTO void sz_levenshtein_engine_layout_(sz_size_t count, sz_levenshtein_symbol_t symbol,
                                                           sz_levenshtein_engine_shape_t const *shapes,
                                                           sz_size_t head_bytes,
                                                           sz_levenshtein_engine_layout_t *layout) {
    sz_size_t words = 0, rows = 0;
    for (sz_size_t index = 0; index != count; ++index) {
        words += (sz_size_t)shapes[index].classes * sz_levenshtein_query_stride(shapes[index].length);
        rows += shapes[index].rows;
    }
    layout->head_bytes = sz_levenshtein_align64_(head_bytes);
    layout->masks_offset = layout->head_bytes;
    layout->masks_words = words;
    layout->offsets_offset = layout->masks_offset + sz_levenshtein_align64_(words * sizeof(sz_u64_t));
    layout->lengths_offset = layout->offsets_offset + sz_levenshtein_align64_((count + 1) * sizeof(sz_size_t));
    layout->classes_offset = layout->lengths_offset + sz_levenshtein_align64_(count * sizeof(sz_u32_t));
    layout->classes_bytes = symbol == sz_levenshtein_bytes_k
                                ? count * sz_levenshtein_byte_classes_k
                                : (count + 1) * sizeof(sz_size_t) +
                                      count * sz_levenshtein_utf8_pages_k * sizeof(sz_u16_t) +
                                      rows * 256 * sizeof(sz_u32_t);
    layout->total_bytes = layout->classes_offset + sz_levenshtein_align64_(layout->classes_bytes);
}

/** Points @p engine 's tensors into its block and writes the offsets and lengths the
 *  @p shapes imply. */
STRINGZILLA_API_COMPTIME void sz_levenshtein_engine_bind_(sz_levenshtein_engine_t *engine,
                                                          sz_levenshtein_engine_layout_t const *layout,
                                                          sz_levenshtein_engine_shape_t const *shapes) {
    sz_ptr_t const block = (sz_ptr_t)engine->memory;
    sz_size_t *const offsets = (sz_size_t *)(block + layout->offsets_offset);
    sz_u32_t *const lengths = (sz_u32_t *)(block + layout->lengths_offset);
    sz_size_t words = 0;
    for (sz_size_t index = 0; index != engine->count; ++index) {
        offsets[index] = words;
        lengths[index] = shapes[index].length;
        words += (sz_size_t)shapes[index].classes * sz_levenshtein_query_stride(shapes[index].length);
    }
    offsets[engine->count] = words;
    engine->masks = (sz_u64_t const *)(block + layout->masks_offset);
    engine->masks_offsets = offsets;
    engine->lengths = lengths;
    engine->symbol_to_class = block + layout->classes_offset;
    if (engine->symbol == sz_levenshtein_bytes_k) return;
    // A rune query's page table and its class rows sit together, which is the block `_prepare_utf8` fills.
    sz_size_t *const pages_offsets = (sz_size_t *)(block + layout->classes_offset);
    sz_size_t taken = (engine->count + 1) * sizeof(sz_size_t);
    for (sz_size_t index = 0; index != engine->count; ++index) {
        pages_offsets[index] = taken;
        taken += sz_levenshtein_utf8_pages_k * sizeof(sz_u16_t) +
                 (sz_size_t)shapes[index].rows * 256 * sizeof(sz_u32_t);
    }
    pages_offsets[engine->count] = taken;
}

/** Builds every query's plane on the host, into the block
 *  @ref sz_levenshtein_engine_bind_ addressed. */
STRINGZILLA_API_COMPTIME void sz_levenshtein_engine_fill_(sz_levenshtein_engine_t *engine,
                                                          sz_sequence_t const *queries) {
    sz_u64_t *const masks = (sz_u64_t *)engine->masks;
    for (sz_size_t index = 0; index != engine->count; ++index) {
        if (engine->lengths[index] == 0) continue;
        sz_cptr_t const text = queries->get_start(queries->handle, index);
        sz_size_t const bytes = queries->get_length(queries->handle, index);
        sz_u64_t *const plane = masks + engine->masks_offsets[index];
        sz_levenshtein_query_t prepared = {
            STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL, 0, 0, 0};
        if (engine->symbol == sz_levenshtein_bytes_k) {
            sz_u8_t *const byte_to_class = (sz_u8_t *)engine->symbol_to_class +
                                           index * sz_levenshtein_byte_classes_k;
            sz_levenshtein_query_prepare(text, bytes, plane, byte_to_class, &prepared);
            continue;
        }
        sz_u16_t const *page_rows = STRINGZILLA_NULL;
        sz_u32_t const *class_rows = STRINGZILLA_NULL;
        sz_levenshtein_engine_pages_(engine, index, &page_rows, &class_rows);
        sz_levenshtein_query_prepare_utf8(text, bytes, plane, (void *)page_rows, &prepared);
    }
}

/** Sizes @p engine 's batch block through @p alloc and binds its tensors, leaving
 *  the planes unwritten. */
STRINGZILLA_API_COMPTIME sz_status_t sz_levenshtein_engine_build_(sz_sequence_t const *queries,
                                                                  sz_levenshtein_symbol_t symbol, sz_size_t head_bytes,
                                                                  sz_memory_allocator_t const *alloc,
                                                                  sz_levenshtein_engine_t *engine) {
    sz_size_t const count = queries->count;
    sz_size_t const shapes_bytes = (count != 0 ? count : 1) * sizeof(sz_levenshtein_engine_shape_t);
    sz_levenshtein_engine_shape_t *const shapes =
        (sz_levenshtein_engine_shape_t *)alloc->allocate(shapes_bytes, alloc->handle);
    if (!shapes) return sz_bad_alloc_k;
    sz_status_t const measured = sz_levenshtein_engine_measure_(queries, symbol, alloc, shapes);
    if (measured != sz_success_k) {
        alloc->free(shapes, shapes_bytes, alloc->handle);
        return measured;
    }

    sz_levenshtein_engine_layout_t layout;
    sz_levenshtein_engine_layout_(count, symbol, shapes, head_bytes, &layout);
    void *const block = alloc->allocate(layout.total_bytes, alloc->handle);
    if (!block) {
        alloc->free(shapes, shapes_bytes, alloc->handle);
        return sz_bad_alloc_k;
    }

    engine->count = count;
    engine->symbol = symbol;
    engine->capability = sz_cap_serial_k;
    engine->alloc = *alloc;
    engine->memory = block;
    engine->memory_bytes = layout.total_bytes;
    engine->scratch = STRINGZILLA_NULL;
    engine->scratch_bytes = 0;
    sz_levenshtein_engine_bind_(engine, &layout, shapes);
    alloc->free(shapes, shapes_bytes, alloc->handle);
    return sz_success_k;
}

/** Prepares @p queries on the host and records the tier @p caps and @p symbol
 *  resolve between them. */
STRINGZILLA_API_COMPTIME sz_status_t sz_levenshtein_engine_init_cpu_(sz_sequence_t const *queries,
                                                                     sz_levenshtein_symbol_t symbol,
                                                                     sz_capability_t caps, sz_memory_allocator_t *alloc,
                                                                     sz_levenshtein_engine_t *engine) {
    sz_memory_allocator_t host;
    if (alloc) host = *alloc;
    else sz_memory_allocator_init_default(&host);
    sz_status_t const built = sz_levenshtein_engine_build_(queries, symbol, 0, &host, engine);
    if (built != sz_success_k) return built;
    sz_levenshtein_engine_fill_(engine, queries);
    engine->capability = sz_levenshtein_tier_for(caps, symbol);
    return sz_success_k;
}

/** Returns both of @p engine 's blocks to the allocator they were built with, and
 *  leaves it empty. */
STRINGZILLA_API_COMPTIME void sz_levenshtein_engine_free_(sz_levenshtein_engine_t *engine) {
    if (engine->memory) engine->alloc.free(engine->memory, engine->memory_bytes, engine->alloc.handle);
    if (engine->scratch) engine->alloc.free(engine->scratch, engine->scratch_bytes, engine->alloc.handle);
    engine->masks = STRINGZILLA_NULL, engine->masks_offsets = STRINGZILLA_NULL;
    engine->symbol_to_class = STRINGZILLA_NULL, engine->lengths = STRINGZILLA_NULL;
    engine->count = 0;
    engine->memory = STRINGZILLA_NULL, engine->memory_bytes = 0;
    engine->scratch = STRINGZILLA_NULL, engine->scratch_bytes = 0;
}

#pragma endregion Cross Product Engine

#pragma region Serial Implementation

/** One candidate's running score. */
typedef struct sz_levenshtein_u64x1_state_serial_t {

    /** The running edit distance. */
    sz_u64_t score;
} sz_levenshtein_u64x1_state_serial_t;

/** One query word of one candidate's Myers state - its vertical deltas. */
typedef struct sz_levenshtein_u64x1_vertical_serial_t {

    /** Myers' VP. */
    sz_u64_t positive;

    /** Myers' VN. */
    sz_u64_t negative;
} sz_levenshtein_u64x1_vertical_serial_t;

/** Starts one candidate: the score at the query's length, and @p words verticals at
 *  the top boundary. */
STRINGZILLA_HELPER_AUTO void sz_levenshtein_u64x1_init_serial(sz_levenshtein_u64x1_state_serial_t *state,
                                                              sz_levenshtein_u64x1_vertical_serial_t *verticals,
                                                              sz_size_t words, sz_levenshtein_query_t const *query) {
    state->score = query->length;
    for (sz_size_t word = 0; word != words; ++word)
        verticals[word].positive = ~(sz_u64_t)0, verticals[word].negative = 0;
}

/**
 *  @brief Advances one candidate by one symbol through exactly @p words verticals, chaining the
 *      horizontal deltas from each word into the next; the score moves on the last word.
 *
 *  There is no candidate mask: a candidate past its text keeps stepping whatever class the
 *  transpose emits, and its score is read where its text ends.
 *
 *  @param[in] words Exactly `sz_levenshtein_query_words(query->length)`; a constant keeps
 *      verticals in registers.
 *  @param[in] class_id The candidate's class at this position, as the transpose emitted it.
 */
STRINGZILLA_HELPER_AUTO void sz_levenshtein_u64x1_step_serial(sz_levenshtein_u64x1_state_serial_t *state,
                                                              sz_levenshtein_u64x1_vertical_serial_t *verticals,
                                                              sz_size_t words, sz_levenshtein_query_t const *query,
                                                              sz_u32_t class_id) {
    sz_u64_t const *const masks = query->masks + (sz_size_t)class_id * query->stride;
    sz_u64_t const last_symbol_bit = sz_levenshtein_last_symbol_bit_(query->length);
    // The top boundary: the row above the first word is one edit higher than the cell to its left.
    sz_u64_t positive_carry = 1, negative_carry = 0;
    for (sz_size_t word = 0; word != words; ++word) {
        sz_levenshtein_u64x1_vertical_serial_t *const vertical = verticals + word;
        sz_u64_t const equality = masks[word];
        sz_u64_t const vertical_carry = equality | vertical->negative;
        sz_u64_t const matched = equality | negative_carry;
        sz_u64_t const diagonal = (((matched & vertical->positive) + vertical->positive) ^ vertical->positive) |
                                  matched;
        sz_u64_t horizontal_positive = vertical->negative | ~(diagonal | vertical->positive);
        sz_u64_t horizontal_negative = vertical->positive & diagonal;
        if (word + 1 == words) {
            state->score += (horizontal_positive & last_symbol_bit) != 0;
            state->score -= (horizontal_negative & last_symbol_bit) != 0;
        }
        sz_u64_t const next_positive_carry = horizontal_positive >> 63;
        sz_u64_t const next_negative_carry = horizontal_negative >> 63;
        horizontal_positive = (horizontal_positive << 1) | positive_carry;
        horizontal_negative = (horizontal_negative << 1) | negative_carry;
        positive_carry = next_positive_carry, negative_carry = next_negative_carry;
        vertical->positive = horizontal_negative | ~(vertical_carry | horizontal_positive);
        vertical->negative = horizontal_positive & vertical_carry;
    }
}

/** Whether the candidate can still come under @p radius at @p position: its score falls by
 *  at most one per remaining symbol. Monotone, so once false it stays false;
 *  @c STRINGZILLA_SSIZE_MAX bounds nothing. */
STRINGZILLA_API_COMPTIME sz_bool_t sz_levenshtein_u64x1_any_active_serial(
    sz_levenshtein_u64x1_state_serial_t const *state, sz_u64_t symbol_count, sz_size_t position, sz_ssize_t radius) {
    if (position >= symbol_count) return sz_false_k;
    return (sz_ssize_t)state->score - (sz_ssize_t)(symbol_count - position) <= radius ? sz_true_k : sz_false_k;
}

/** The running score of @p candidate, read at the position where that candidate's text ends. */
STRINGZILLA_HELPER_AUTO sz_size_t sz_levenshtein_u64x1_score_serial(sz_levenshtein_u64x1_state_serial_t const *state,
                                                                    sz_size_t candidate) {
    return sz_unused_(candidate), state->score;
}

/** Scalar candidates advanced per position: one word's recurrence is a dependency chain, and only
 *  independent candidates fill the pipeline behind it. */
enum {
    sz_levenshtein_serial_u64x1_candidates_per_step_k = 1,
    sz_levenshtein_serial_u64x1_registers_per_position_k = 8
};

/** Sweeps up to eight candidates through every transpose with @p words verticals each - a constant
 *  keeps a one- or two-word query's verticals register-resident. Every score is read at the
 *  position where its text ends. */
STRINGZILLA_HELPER_INLINE void sz_levenshtein_serial_u64x1_sweep_(sz_levenshtein_query_t const *shared_query,
                                                                  sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                                  sz_u64_t *symbol_counts, sz_size_t sweep_count,
                                                                  sz_levenshtein_transpose_t transpose,
                                                                  sz_levenshtein_u64x1_vertical_serial_t *verticals,
                                                                  sz_size_t words, sz_size_t *distances) {
    enum {
        candidates_per_position_k = sz_levenshtein_serial_u64x1_candidates_per_step_k *
                                    sz_levenshtein_serial_u64x1_registers_per_position_k,
        positions_per_transpose_k = sz_levenshtein_positions_per_transpose_k,
    };
    // A local copy: nothing stored through the verticals can alias it, so the step keeps its fields in registers.
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_query_t const *const query = &local_query;
    sz_size_t cursors[candidates_per_position_k] = {0};
    sz_levenshtein_u64x1_state_serial_t states[candidates_per_position_k];
    for (sz_size_t candidate = 0; candidate != candidates_per_position_k; ++candidate)
        sz_levenshtein_u64x1_init_serial(&states[candidate], verticals + candidate * words, words, query);
    sz_u32_t transpose_classes[positions_per_transpose_k][candidates_per_position_k];
    sz_size_t end = 0;
    for (sz_size_t transpose_start = 0, filled = positions_per_transpose_k; filled == positions_per_transpose_k;
         transpose_start += filled) {
        filled = transpose(query, texts, byte_counts, candidates_per_position_k, cursors, symbol_counts,
                           transpose_start, positions_per_transpose_k, &transpose_classes[0][0]);
        end = transpose_start + filled;
        // A local copy: the transpose may refine the counts, and stores into `distances` must not force reloads.
        sz_u64_t counts[candidates_per_position_k];
        for (sz_size_t candidate = 0; candidate != candidates_per_position_k; ++candidate)
            counts[candidate] = symbol_counts[candidate];
        for (sz_size_t position = 0; position != filled; ++position)
            for (sz_size_t candidate = 0; candidate != candidates_per_position_k; ++candidate) {
                // A candidate past its text skips for free, and the first skipped position is where its score is read.
                if (transpose_start + position < counts[candidate])
                    sz_levenshtein_u64x1_step_serial(&states[candidate], verticals + candidate * words, words, query,
                                                     transpose_classes[position][candidate]);
                else if (transpose_start + position == counts[candidate] && candidate < sweep_count)
                    distances[candidate] = sz_levenshtein_u64x1_score_serial(&states[candidate], 0);
            }
    }
    // Candidates as long as the sweep itself end at the position the transposes never reached.
    for (sz_size_t candidate = 0; candidate != sweep_count; ++candidate)
        if (symbol_counts[candidate] == end)
            distances[candidate] = sz_levenshtein_u64x1_score_serial(&states[candidate], 0);
}

/** Streams every candidate through a prepared @p query, @c candidates_per_position_k at a time,
 *  with @p transpose emitting their classes; @p verticals holds enough for a runtime word count. */
STRINGZILLA_HELPER_INLINE void sz_levenshtein_serial_u64x1_distances_(sz_levenshtein_query_t const *query,
                                                                      sz_sequence_t const *candidates,
                                                                      sz_levenshtein_transpose_t transpose,
                                                                      sz_levenshtein_u64x1_vertical_serial_t *verticals,
                                                                      sz_size_t *distances) {
    enum {
        candidates_per_position_k = sz_levenshtein_serial_u64x1_candidates_per_step_k *
                                    sz_levenshtein_serial_u64x1_registers_per_position_k
    };
    sz_size_t const words = sz_levenshtein_query_words(query->length);
    sz_levenshtein_u64x1_vertical_serial_t resident_verticals[candidates_per_position_k * 2];
    for (sz_size_t sweep_first = 0; sweep_first < candidates->count; sweep_first += candidates_per_position_k) {
        sz_size_t const sweep_count = sz_min_of_two((sz_size_t)candidates_per_position_k,
                                                    candidates->count - sweep_first);
        sz_cptr_t texts[candidates_per_position_k] = {0};
        sz_u64_t byte_counts[candidates_per_position_k] = {0}, symbol_counts[candidates_per_position_k] = {0};
        for (sz_size_t candidate = 0; candidate != sweep_count; ++candidate) {
            texts[candidate] = candidates->get_start(candidates->handle, sweep_first + candidate);
            byte_counts[candidate] = candidates->get_length(candidates->handle, sweep_first + candidate);
            symbol_counts[candidate] = byte_counts[candidate];
        }
        if (words == 1)
            sz_levenshtein_serial_u64x1_sweep_(query, texts, byte_counts, symbol_counts, sweep_count, transpose,
                                               resident_verticals, 1, distances + sweep_first);
        else if (words == 2)
            sz_levenshtein_serial_u64x1_sweep_(query, texts, byte_counts, symbol_counts, sweep_count, transpose,
                                               resident_verticals, 2, distances + sweep_first);
        else
            sz_levenshtein_serial_u64x1_sweep_(query, texts, byte_counts, symbol_counts, sweep_count, transpose,
                                               verticals, words, distances + sweep_first);
    }
}

STRINGZILLA_API_COMPTIME sz_status_t sz_levenshtein_distances_serial(sz_levenshtein_engine_t *engine,
                                                                     sz_sequence_t const *candidates,
                                                                     sz_size_t *distances, sz_size_t distances_stride) {
    sz_assert_((engine->capability & sz_caps_cpus_k) != 0 &&
               "A host tier never scores a device-prepared engine, whose head only its GPU tier reads");
    enum { registers_k = sz_levenshtein_serial_u64x1_registers_per_position_k };
    if (distances_stride < candidates->count) return sz_unexpected_dimensions_k;
    sz_levenshtein_transpose_t const transpose = engine->symbol == sz_levenshtein_bytes_k
                                                     ? sz_levenshtein_transpose
                                                     : sz_levenshtein_transpose_utf8;
    sz_status_t const grown = sz_levenshtein_engine_scratch_(
        engine, sz_levenshtein_engine_verticals_bytes_(registers_k, sz_levenshtein_engine_words_max_(engine),
                                                       sizeof(sz_levenshtein_u64x1_vertical_serial_t)));
    if (grown != sz_success_k) return grown;
    sz_levenshtein_u64x1_vertical_serial_t *const verticals =
        (sz_levenshtein_u64x1_vertical_serial_t *)sz_levenshtein_engine_verticals_(engine);

    for (sz_size_t index = 0; index != engine->count; ++index) {
        sz_size_t *const row = distances + index * distances_stride;
        if (engine->lengths[index] == 0) {
            sz_levenshtein_engine_empty_row_(engine, candidates, row);
            continue;
        }
        sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(engine, index);
        sz_levenshtein_serial_u64x1_distances_(&query, candidates, transpose, verticals, row);
    }
    return sz_success_k;
}

#pragma endregion Serial Implementation

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_LEVENSHTEIN_SERIAL_H_
