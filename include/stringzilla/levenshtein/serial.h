/**
 *  @brief Serial backend for Levenshtein edit distances: Myers' bit-parallel algorithm, one query against one
 *      or many candidates, over bytes or over UTF-8 runes, plus the shared query state and the step primitives
 *      every SIMD backend mirrors.
 *  @file include/stringzilla/levenshtein/serial.h
 *  @author Ash Vardanian
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

/** One query's bit-parallel state, prepared once and read by every candidate: every symbol maps to a class that
 *  indexes the match masks, a view over caller-owned memory of @c words × classes entries with no length ceiling. */
typedef struct sz_levenshtein_query_t {
    sz_u64_t const *masks;     /**< Row @c word * classes + class, bit @c i: symbol @c word * 64 + i has that class. */
    sz_u16_t const *page_rows; /**< UTF-8 only: class row per 256-rune page, zero for a page the query lacks. */
    sz_size_t classes;         /**< Mask rows per word: 256 for bytes, distinct runes plus one for UTF-8. */
    sz_size_t length;          /**< In symbols: bytes, or runes for a UTF-8 query. */
} sz_levenshtein_query_t;

/** Words a query of @p length symbols spans - the mask table holds that many rows of @c classes entries. */
SZ_HELPER_AUTO sz_size_t sz_levenshtein_query_words(sz_size_t length) { return (length + 63) / 64; }

/**
 *  @brief Builds the match masks of the byte string @p text into @p masks and points @p query at them.
 *  @param[out] masks Caller-owned, @c sz_levenshtein_query_words(length) × 256 entries; zeroed here.
 *  @retval sz_unexpected_dimensions_k for an empty query, whose distance is every candidate's length.
 */
SZ_HELPER_AUTO sz_status_t sz_levenshtein_query_prepare(sz_cptr_t text, sz_size_t length, sz_u64_t *masks,
                                                        sz_levenshtein_query_t *query) {
    if (length == 0) return sz_unexpected_dimensions_k;
    sz_size_t const words = sz_levenshtein_query_words(length);
    for (sz_size_t entry = 0; entry != words * 256; ++entry) masks[entry] = 0;
    for (sz_size_t position = 0; position != length; ++position)
        masks[(position >> 6) * 256 + (sz_u8_t)text[position]] |= (sz_u64_t)1 << (position & 63);
    query->masks = masks;
    query->page_rows = SZ_NULL;
    query->classes = 256;
    query->length = length;
    return sz_success_k;
}

/** Unicode as 256-rune pages: the page table a UTF-8 query indexes by @c rune >> 8. Its @c u16 entries fill
 *  a whole number of cache lines, so the class rows follow it at a constant aligned offset. */
enum { sz_levenshtein_utf8_pages_k = 0x110000 / 256 };
sz_static_assert(sz_levenshtein_utf8_pages_k * sizeof(sz_u16_t) % 64 == 0,
                 sz_levenshtein_utf8_page_table_fills_cache_lines);

/** The class rows behind the page table: 256 classes per row, row zero all zeros. */
SZ_API_COMPTIME sz_u32_t const *sz_levenshtein_utf8_class_rows_(sz_levenshtein_query_t const *query) {
    return (sz_u32_t const *)(query->page_rows + sz_levenshtein_utf8_pages_k);
}

/** The class of @p rune under a UTF-8 @p query: two loads through the page table, zero for a rune the query lacks. */
SZ_API_COMPTIME sz_u32_t sz_levenshtein_utf8_class(sz_levenshtein_query_t const *query, sz_rune_t rune) {
    return sz_levenshtein_utf8_class_rows_(query)[(sz_size_t)query->page_rows[rune >> 8] * 256 + (rune & 255)];
}

/** Runes in @p text under the decoding the family applies - one @c U+FFFD per ill-formed byte, the grid the
 *  segmenters use - which differs from @c sz_utf8_count on ill-formed input. */
SZ_API_COMPTIME sz_size_t sz_levenshtein_utf8_runes(sz_cptr_t text, sz_size_t length) {
    sz_size_t runes = 0;
    for (sz_size_t position = 0; position < length; ++runes) sz_utf8_next_rune_(text, length, &position);
    return runes;
}

/** Scratch the page table of a @p runes -rune query takes: the page index, then one class row per page the
 *  query can touch plus the all-zero row. */
SZ_API_COMPTIME sz_size_t sz_levenshtein_utf8_pages_bytes(sz_size_t runes) {
    sz_size_t const rows = sz_min_of_two(runes, (sz_size_t)sz_levenshtein_utf8_pages_k) + 1;
    return sz_levenshtein_utf8_pages_k * sizeof(sz_u16_t) + rows * 256 * sizeof(sz_u32_t);
}

/**
 *  @brief Builds the match masks of the UTF-8 string @p text into @p masks, its rune classes into the page
 *      table at @p pages, and points @p query at both. Classes follow first appearance.
 *  @param[out] masks Caller-owned, @c sz_levenshtein_query_words(runes) × (runes + 1) entries for a query of
 *      @c runes runes, of which the first @c words × classes are written.
 *  @param[out] pages Caller-owned, @c sz_levenshtein_utf8_pages_bytes(runes) bytes, cache-line aligned.
 *  @retval sz_unexpected_dimensions_k for an empty query, whose distance is every candidate's rune count.
 */
SZ_API_COMPTIME sz_status_t sz_levenshtein_query_prepare_utf8(sz_cptr_t text, sz_size_t length, sz_u64_t *masks,
                                                              void *pages, sz_levenshtein_query_t *query) {
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
    query->classes = classes;
    sz_size_t const words = sz_levenshtein_query_words(count);
    for (sz_size_t entry = 0; entry != words * classes; ++entry) masks[entry] = 0;
    // Second pass: the row stride is known, so every rune sets its bit in its class's row.
    sz_size_t rune_index = 0;
    for (sz_size_t position = 0; position < length; ++rune_index) {
        sz_rune_t const rune = sz_utf8_next_rune_(text, length, &position);
        masks[(rune_index >> 6) * classes + sz_levenshtein_utf8_class(query, rune)] |= (sz_u64_t)1 << (rune_index & 63);
    }
    query->masks = masks;
    query->length = count;
    return sz_success_k;
}

/** Positions one stripe spans: a candidate's classes are emitted this many at a time, transposed. */
enum { sz_levenshtein_positions_per_stripe_k = 256 };

/** @brief How wide a stripe's class ids are: bytes are their own classes and fit a byte, rune classes need four. */
typedef enum sz_levenshtein_classes_width_t {
    sz_levenshtein_classes_u8_k = 1,
    sz_levenshtein_classes_u32_k = 4,
} sz_levenshtein_classes_width_t;

/**
 *  @brief Emits the next @p positions symbols of each of @p candidates texts as class ids, transposed: the symbol
 *      at stripe position @c p of candidate @c c lands at @c p · candidates + c, and a candidate past its text
 *      reads as class zero. Returns the positions filled - below @p positions only once every candidate has run dry.
 *
 *  A candidate that runs dry inside the stripe writes its exact symbol count to @p symbol_counts; until then the
 *  entry keeps the byte count the caller seeded.
 *
 *  @param[inout] cursors Byte offsets into each candidate's text, advanced past what was emitted.
 *  @param[in] stripe_start The absolute position of the stripe's first symbol.
 *  @param[out] stripe_classes @c positions × candidates class ids, @c sz_u32_t for the stripes here.
 */
typedef sz_size_t (*sz_levenshtein_stripe_t)(sz_levenshtein_query_t const *query, sz_cptr_t const *texts,
                                             sz_u64_t const *byte_counts, sz_size_t candidates, sz_size_t *cursors,
                                             sz_u64_t *symbol_counts, sz_size_t stripe_start, sz_size_t positions,
                                             void *stripe_classes);

/** The byte stripe: every byte is its own class, and the byte count is the symbol count. */
SZ_API_COMPTIME sz_size_t sz_levenshtein_stripe(sz_levenshtein_query_t const *query, sz_cptr_t const *texts,
                                                sz_u64_t const *byte_counts, sz_size_t candidates, sz_size_t *cursors,
                                                sz_u64_t *symbol_counts, sz_size_t stripe_start, sz_size_t positions,
                                                void *stripe_classes) {
    sz_unused_(query), sz_unused_(symbol_counts), sz_unused_(stripe_start);
    sz_u32_t *const classes = (sz_u32_t *)stripe_classes;
    sz_size_t filled = 0;
    for (sz_size_t candidate = 0; candidate != candidates; ++candidate)
        filled = sz_max_of_two(filled,
                               sz_min_of_two(positions, (sz_size_t)byte_counts[candidate] - cursors[candidate]));
    for (sz_size_t position = 0; position != filled; ++position)
        for (sz_size_t candidate = 0; candidate != candidates; ++candidate)
            classes[position * candidates + candidate] = cursors[candidate] < byte_counts[candidate]
                                                             ? (sz_u8_t)texts[candidate][cursors[candidate]++]
                                                             : 0;
    return filled;
}

/** The UTF-8 stripe: every rune decodes and takes the class the query gave it. */
SZ_API_COMPTIME sz_size_t sz_levenshtein_stripe_utf8(sz_levenshtein_query_t const *query, sz_cptr_t const *texts,
                                                     sz_u64_t const *byte_counts, sz_size_t candidates,
                                                     sz_size_t *cursors, sz_u64_t *symbol_counts,
                                                     sz_size_t stripe_start, sz_size_t positions,
                                                     void *stripe_classes) {
    sz_u32_t *const classes = (sz_u32_t *)stripe_classes;
    sz_size_t filled = 0;
    for (sz_size_t candidate = 0; candidate != candidates; ++candidate) {
        sz_size_t emitted = 0;
        for (; emitted != positions && cursors[candidate] < byte_counts[candidate]; ++emitted)
            classes[emitted * candidates + candidate] = sz_levenshtein_utf8_class(
                query, sz_utf8_next_rune_(texts[candidate], byte_counts[candidate], cursors + candidate));
        if (emitted != 0 && cursors[candidate] == byte_counts[candidate])
            symbol_counts[candidate] = stripe_start + emitted;
        for (sz_size_t position = emitted; position != positions; ++position)
            classes[position * candidates + candidate] = 0;
        filled = sz_max_of_two(filled, emitted);
    }
    return filled;
}

#pragma endregion Generic Public Helpers

#pragma region Generic Internal Helpers

/** The bit of the query's last symbol within its last word: where the score deltas are read. */
SZ_HELPER_AUTO sz_u64_t sz_levenshtein_last_symbol_bit_(sz_size_t length) { return (sz_u64_t)1 << ((length - 1) & 63); }

/** The shift that turns that bit into a one. */
SZ_HELPER_AUTO sz_size_t sz_levenshtein_last_symbol_shift_(sz_size_t length) { return (length - 1) & 63; }

/** Rounds @p bytes up to a cache line, so every scratch area below starts aligned. */
SZ_HELPER_AUTO sz_size_t sz_levenshtein_align64_(sz_size_t bytes) { return (bytes + 63) & ~(sz_size_t)63; }

/** Scratch one call takes from the allocator: the mask table, the UTF-8 page table if any, then the
 *  verticals, each starting on a cache line. */
SZ_HELPER_AUTO sz_size_t sz_levenshtein_distances_scratch_bytes_(sz_size_t mask_entries, sz_size_t pages_bytes,
                                                                 sz_size_t registers_per_position, sz_size_t words,
                                                                 sz_size_t vertical_bytes) {
    return 64 + sz_levenshtein_align64_(mask_entries * sizeof(sz_u64_t)) + sz_levenshtein_align64_(pages_bytes) +
           registers_per_position * words * vertical_bytes;
}

/** Every candidate's distance to an empty byte query is its byte count. */
SZ_HELPER_AUTO void sz_levenshtein_byte_counts_as_distances_(sz_sequence_t const *candidates, sz_size_t *distances) {
    for (sz_size_t index = 0; index != candidates->count; ++index)
        distances[index] = candidates->get_length(candidates->handle, index);
}

/** Every candidate's distance to an empty UTF-8 query is its rune count. */
SZ_HELPER_AUTO void sz_levenshtein_rune_counts_as_distances_(sz_sequence_t const *candidates, sz_size_t *distances) {
    for (sz_size_t index = 0; index != candidates->count; ++index)
        distances[index] = sz_levenshtein_utf8_runes(candidates->get_start(candidates->handle, index),
                                                     candidates->get_length(candidates->handle, index));
}

#pragma endregion Generic Internal Helpers

#pragma region Serial Implementation

/** One candidate's running score. */
typedef struct sz_levenshtein_u64x1_state_serial_t {
    sz_u64_t score; /**< The running edit distance. */
} sz_levenshtein_u64x1_state_serial_t;

/** One query word of one candidate's Myers state - its vertical deltas. */
typedef struct sz_levenshtein_u64x1_vertical_serial_t {
    sz_u64_t positive; /**< Myers' VP. */
    sz_u64_t negative; /**< Myers' VN. */
} sz_levenshtein_u64x1_vertical_serial_t;

/** Starts one candidate: the score at the query's length, and @p words verticals at the top boundary. */
SZ_HELPER_AUTO void sz_levenshtein_u64x1_init_serial(sz_levenshtein_u64x1_state_serial_t *state,
                                                     sz_levenshtein_u64x1_vertical_serial_t *verticals, sz_size_t words,
                                                     sz_levenshtein_query_t const *query) {
    state->score = query->length;
    for (sz_size_t word = 0; word != words; ++word)
        verticals[word].positive = ~(sz_u64_t)0, verticals[word].negative = 0;
}

/**
 *  @brief Advances one candidate by one symbol through exactly @p words verticals, chaining the horizontal
 *      deltas from each word into the next; the score moves on the last word.
 *
 *  There is no candidate mask: a candidate past its text keeps stepping whatever class the stripe emits, and its
 *  score is read where its text ends.
 *
 *  @param[in] words Exactly @c sz_levenshtein_query_words(query->length); a constant keeps verticals in registers.
 *  @param[in] class_id The candidate's class at this position, as the stripe emitted it.
 */
SZ_HELPER_AUTO void sz_levenshtein_u64x1_step_serial(sz_levenshtein_u64x1_state_serial_t *state,
                                                     sz_levenshtein_u64x1_vertical_serial_t *verticals, sz_size_t words,
                                                     sz_levenshtein_query_t const *query, sz_u32_t class_id) {
    sz_u64_t const *const masks = query->masks + class_id;
    sz_u64_t const last_symbol_bit = sz_levenshtein_last_symbol_bit_(query->length);
    // The top boundary: the row above the first word is one edit higher than the cell to its left.
    sz_u64_t positive_carry = 1, negative_carry = 0;
    for (sz_size_t word = 0; word != words; ++word) {
        sz_levenshtein_u64x1_vertical_serial_t *const vertical = verticals + word;
        sz_u64_t const equality = masks[word * query->classes];
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

/** @brief Whether the candidate can still come under @p radius at @p position: its score falls by at most one per
 *      remaining symbol. Monotone, so once false it stays false; @c SZ_SSIZE_MAX bounds nothing. */
SZ_API_COMPTIME sz_bool_t sz_levenshtein_u64x1_any_active_serial(sz_levenshtein_u64x1_state_serial_t const *state,
                                                                 sz_u64_t symbol_count, sz_size_t position,
                                                                 sz_ssize_t radius) {
    if (position >= symbol_count) return sz_false_k;
    return (sz_ssize_t)state->score - (sz_ssize_t)(symbol_count - position) <= radius ? sz_true_k : sz_false_k;
}

/** The running score of @p candidate, read at the position where that candidate's text ends. */
SZ_HELPER_AUTO sz_size_t sz_levenshtein_u64x1_score_serial(sz_levenshtein_u64x1_state_serial_t const *state,
                                                           sz_size_t candidate) {
    return sz_unused_(candidate), state->score;
}

/** Scalar candidates advanced per position: one word's recurrence is a dependency chain, and only
 *  independent candidates fill the pipeline behind it. */
enum {
    sz_levenshtein_serial_u64x1_candidates_per_step_k = 1,
    sz_levenshtein_serial_u64x1_registers_per_position_k = 8
};

/** Sweeps up to eight candidates through every stripe with @p words verticals each - a constant keeps a one- or
 *  two-word query's verticals register-resident. Every score is read at the position where its text ends. */
SZ_HELPER_INLINE void sz_levenshtein_serial_u64x1_sweep_(sz_levenshtein_query_t const *shared_query,
                                                         sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                         sz_u64_t *symbol_counts, sz_size_t sweep_count,
                                                         sz_levenshtein_stripe_t stripe,
                                                         sz_levenshtein_u64x1_vertical_serial_t *verticals,
                                                         sz_size_t words, sz_size_t *distances) {
    enum {
        candidates_per_position_k = sz_levenshtein_serial_u64x1_candidates_per_step_k *
                                    sz_levenshtein_serial_u64x1_registers_per_position_k,
        positions_per_stripe_k = sz_levenshtein_positions_per_stripe_k,
    };
    // A local copy: nothing stored through the verticals can alias it, so the step keeps its fields in registers.
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_query_t const *const query = &local_query;
    sz_size_t cursors[candidates_per_position_k] = {0};
    sz_levenshtein_u64x1_state_serial_t states[candidates_per_position_k];
    for (sz_size_t candidate = 0; candidate != candidates_per_position_k; ++candidate)
        sz_levenshtein_u64x1_init_serial(&states[candidate], verticals + candidate * words, words, query);
    sz_u32_t stripe_classes[positions_per_stripe_k][candidates_per_position_k];
    sz_size_t end = 0;
    for (sz_size_t stripe_start = 0, filled = positions_per_stripe_k; filled == positions_per_stripe_k;
         stripe_start += filled) {
        filled = stripe(query, texts, byte_counts, candidates_per_position_k, cursors, symbol_counts, stripe_start,
                        positions_per_stripe_k, &stripe_classes[0][0]);
        end = stripe_start + filled;
        // A local copy: the stripe may refine the counts, and stores into `distances` must not force reloads.
        sz_u64_t counts[candidates_per_position_k];
        for (sz_size_t candidate = 0; candidate != candidates_per_position_k; ++candidate)
            counts[candidate] = symbol_counts[candidate];
        for (sz_size_t position = 0; position != filled; ++position)
            for (sz_size_t candidate = 0; candidate != candidates_per_position_k; ++candidate) {
                // A candidate past its text skips for free, and the first skipped position is where its score is read.
                if (stripe_start + position < counts[candidate])
                    sz_levenshtein_u64x1_step_serial(&states[candidate], verticals + candidate * words, words, query,
                                                     stripe_classes[position][candidate]);
                else if (stripe_start + position == counts[candidate] && candidate < sweep_count)
                    distances[candidate] = sz_levenshtein_u64x1_score_serial(&states[candidate], 0);
            }
    }
    // Candidates as long as the sweep itself end at the position the stripes never reached.
    for (sz_size_t candidate = 0; candidate != sweep_count; ++candidate)
        if (symbol_counts[candidate] == end)
            distances[candidate] = sz_levenshtein_u64x1_score_serial(&states[candidate], 0);
}

/** Streams every candidate through a prepared @p query, @c candidates_per_position_k at a time, with
 *  @p stripe emitting their classes; @p verticals holds enough for a runtime word count. */
SZ_HELPER_INLINE void sz_levenshtein_serial_u64x1_distances_(sz_levenshtein_query_t const *query,
                                                             sz_sequence_t const *candidates,
                                                             sz_levenshtein_stripe_t stripe,
                                                             sz_levenshtein_u64x1_vertical_serial_t *verticals,
                                                             sz_size_t *distances) {
    enum {
        candidates_per_position_k = sz_levenshtein_serial_u64x1_candidates_per_step_k *
                                    sz_levenshtein_serial_u64x1_registers_per_position_k
    };
    sz_size_t const words = sz_levenshtein_query_words(query->length);
    sz_levenshtein_u64x1_vertical_serial_t resident_verticals[candidates_per_position_k * 2];
    for (sz_size_t sweep_first = 0; sweep_first < candidates->count; sweep_first += candidates_per_position_k) {
        sz_size_t const sweep_count = sz_min_of_two(candidates_per_position_k, candidates->count - sweep_first);
        sz_cptr_t texts[candidates_per_position_k] = {0};
        sz_u64_t byte_counts[candidates_per_position_k] = {0}, symbol_counts[candidates_per_position_k] = {0};
        for (sz_size_t candidate = 0; candidate != sweep_count; ++candidate) {
            texts[candidate] = candidates->get_start(candidates->handle, sweep_first + candidate);
            byte_counts[candidate] = candidates->get_length(candidates->handle, sweep_first + candidate);
            symbol_counts[candidate] = byte_counts[candidate];
        }
        if (words == 1)
            sz_levenshtein_serial_u64x1_sweep_(query, texts, byte_counts, symbol_counts, sweep_count, stripe,
                                               resident_verticals, 1, distances + sweep_first);
        else if (words == 2)
            sz_levenshtein_serial_u64x1_sweep_(query, texts, byte_counts, symbol_counts, sweep_count, stripe,
                                               resident_verticals, 2, distances + sweep_first);
        else
            sz_levenshtein_serial_u64x1_sweep_(query, texts, byte_counts, symbol_counts, sweep_count, stripe, verticals,
                                               words, distances + sweep_first);
    }
}

SZ_API_COMPTIME sz_status_t sz_levenshtein_distance_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                           sz_size_t b_length, sz_memory_allocator_t *alloc,
                                                           sz_size_t *distance) {
    // The shorter side is the pattern: fewer words per vertical, and unit-cost edit distance is symmetric.
    if (a_length > b_length) {
        sz_cptr_t const swapped_text = a;
        sz_size_t const swapped_length = a_length;
        a = b, a_length = b_length, b = swapped_text, b_length = swapped_length;
    }
    if (a_length == 0) return *distance = b_length, sz_success_k;
    sz_size_t const words = sz_levenshtein_query_words(a_length);
    sz_size_t const scratch_bytes = sz_levenshtein_distances_scratch_bytes_(
        words * 256, 0, 1, words, sizeof(sz_levenshtein_u64x1_vertical_serial_t));
    sz_ptr_t const scratch = (sz_ptr_t)alloc->allocate(scratch_bytes, alloc->handle);
    if (!scratch) return sz_bad_alloc_k;
    sz_u64_t *const masks = (sz_u64_t *)sz_levenshtein_align64_((sz_size_t)scratch);
    sz_levenshtein_u64x1_vertical_serial_t *const verticals =
        (sz_levenshtein_u64x1_vertical_serial_t *)((sz_ptr_t)masks +
                                                   sz_levenshtein_align64_(words * 256 * sizeof(sz_u64_t)));

    sz_levenshtein_query_t query;
    sz_levenshtein_query_prepare(a, a_length, masks, &query);
    sz_levenshtein_u64x1_state_serial_t state;
    sz_levenshtein_u64x1_init_serial(&state, verticals, words, &query);
    for (sz_size_t position = 0; position != b_length; ++position)
        sz_levenshtein_u64x1_step_serial(&state, verticals, words, &query, (sz_u8_t)b[position]);
    *distance = state.score;
    alloc->free(scratch, scratch_bytes, alloc->handle);
    return sz_success_k;
}

SZ_API_COMPTIME sz_status_t sz_levenshtein_distance_utf8_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                                sz_size_t b_length, sz_memory_allocator_t *alloc,
                                                                sz_size_t *distance) {
    // Either side is a valid pattern, so fewer bytes stands in for fewer runes: a heuristic, never the answer.
    if (a_length > b_length) {
        sz_cptr_t const swapped_text = a;
        sz_size_t const swapped_length = a_length;
        a = b, a_length = b_length, b = swapped_text, b_length = swapped_length;
    }
    if (a_length == 0) return *distance = sz_levenshtein_utf8_runes(b, b_length), sz_success_k;
    sz_size_t const a_runes = sz_levenshtein_utf8_runes(a, a_length);
    sz_size_t const words = sz_levenshtein_query_words(a_runes);
    sz_size_t const pages_bytes = sz_levenshtein_utf8_pages_bytes(a_runes);
    sz_size_t const scratch_bytes = sz_levenshtein_distances_scratch_bytes_(
        words * (a_runes + 1), pages_bytes, 1, words, sizeof(sz_levenshtein_u64x1_vertical_serial_t));
    sz_ptr_t const scratch = (sz_ptr_t)alloc->allocate(scratch_bytes, alloc->handle);
    if (!scratch) return sz_bad_alloc_k;
    sz_u64_t *const masks = (sz_u64_t *)sz_levenshtein_align64_((sz_size_t)scratch);
    sz_ptr_t const pages = (sz_ptr_t)masks + sz_levenshtein_align64_(words * (a_runes + 1) * sizeof(sz_u64_t));
    sz_levenshtein_u64x1_vertical_serial_t *const verticals =
        (sz_levenshtein_u64x1_vertical_serial_t *)(pages + sz_levenshtein_align64_(pages_bytes));

    sz_levenshtein_query_t query;
    sz_levenshtein_query_prepare_utf8(a, a_length, masks, pages, &query);
    sz_levenshtein_u64x1_state_serial_t state;
    sz_levenshtein_u64x1_init_serial(&state, verticals, words, &query);
    for (sz_size_t cursor = 0; cursor < b_length;)
        sz_levenshtein_u64x1_step_serial(&state, verticals, words, &query,
                                         sz_levenshtein_utf8_class(&query, sz_utf8_next_rune_(b, b_length, &cursor)));
    *distance = state.score;
    alloc->free(scratch, scratch_bytes, alloc->handle);
    return sz_success_k;
}

SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_serial(sz_cptr_t query_text, sz_size_t query_length,
                                                            sz_sequence_t const *candidates,
                                                            sz_memory_allocator_t *alloc, sz_size_t *distances) {
    enum { registers_k = sz_levenshtein_serial_u64x1_registers_per_position_k };
    if (query_length == 0) return sz_levenshtein_byte_counts_as_distances_(candidates, distances), sz_success_k;
    sz_size_t const words = sz_levenshtein_query_words(query_length);
    sz_size_t const scratch_bytes = sz_levenshtein_distances_scratch_bytes_(
        words * 256, 0, registers_k, words, sizeof(sz_levenshtein_u64x1_vertical_serial_t));
    sz_ptr_t const scratch = (sz_ptr_t)alloc->allocate(scratch_bytes, alloc->handle);
    if (!scratch) return sz_bad_alloc_k;
    sz_u64_t *const masks = (sz_u64_t *)sz_levenshtein_align64_((sz_size_t)scratch);
    sz_levenshtein_u64x1_vertical_serial_t *const verticals =
        (sz_levenshtein_u64x1_vertical_serial_t *)((sz_ptr_t)masks +
                                                   sz_levenshtein_align64_(words * 256 * sizeof(sz_u64_t)));

    sz_levenshtein_query_t query;
    sz_levenshtein_query_prepare(query_text, query_length, masks, &query);
    sz_levenshtein_serial_u64x1_distances_(&query, candidates, sz_levenshtein_stripe, verticals, distances);
    alloc->free(scratch, scratch_bytes, alloc->handle);
    return sz_success_k;
}

SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_utf8_serial(sz_cptr_t query_text, sz_size_t query_length,
                                                                 sz_sequence_t const *candidates,
                                                                 sz_memory_allocator_t *alloc, sz_size_t *distances) {
    enum { registers_k = sz_levenshtein_serial_u64x1_registers_per_position_k };
    sz_size_t const runes = sz_levenshtein_utf8_runes(query_text, query_length);
    if (runes == 0) return sz_levenshtein_rune_counts_as_distances_(candidates, distances), sz_success_k;
    sz_size_t const words = sz_levenshtein_query_words(runes);
    sz_size_t const pages_bytes = sz_levenshtein_utf8_pages_bytes(runes);
    sz_size_t const scratch_bytes = sz_levenshtein_distances_scratch_bytes_(
        words * (runes + 1), pages_bytes, registers_k, words, sizeof(sz_levenshtein_u64x1_vertical_serial_t));
    sz_ptr_t const scratch = (sz_ptr_t)alloc->allocate(scratch_bytes, alloc->handle);
    if (!scratch) return sz_bad_alloc_k;
    sz_u64_t *const masks = (sz_u64_t *)sz_levenshtein_align64_((sz_size_t)scratch);
    sz_ptr_t const pages = (sz_ptr_t)masks + sz_levenshtein_align64_(words * (runes + 1) * sizeof(sz_u64_t));
    sz_levenshtein_u64x1_vertical_serial_t *const verticals =
        (sz_levenshtein_u64x1_vertical_serial_t *)(pages + sz_levenshtein_align64_(pages_bytes));

    sz_levenshtein_query_t query;
    sz_levenshtein_query_prepare_utf8(query_text, query_length, masks, pages, &query);
    sz_levenshtein_serial_u64x1_distances_(&query, candidates, sz_levenshtein_stripe_utf8, verticals, distances);
    alloc->free(scratch, scratch_bytes, alloc->handle);
    return sz_success_k;
}

#pragma endregion Serial Implementation

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_LEVENSHTEIN_SERIAL_H_
