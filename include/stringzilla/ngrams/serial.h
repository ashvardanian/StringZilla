/**
 *  @file include/stringzilla/ngrams/serial.h
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief Serial backend for N-grams: the modulus and the residue of a window at any width and
 *      offset, the prepared-query tables every candidate probes, the one-permutation sketch, and
 *      the constants and accessors every backend shares.
 *
 *  @sa c/stringzilla/ngrams.h
 */
#ifndef STRINGZILLA_NGRAMS_SERIAL_H_
#define STRINGZILLA_NGRAMS_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`, `sz_cap_cpus_k`

#ifdef __cplusplus
extern "C" {
#endif

/** The 32-bit prime every N-gram residue is reduced by; the SIMD backends hold it as an @c f64 and
 *  reduce through a rounded reciprocal. Its cheapest signed base-256 multiple within a machine word
 *  has digit weight twenty-four. */
static sz_u32_t const sz_ngrams_modulus_k = 4026525731u;

/** 256ᵏ mod p for k ∈ [0, 8]: the multipliers that carry a prefix k bytes on. */
static sz_f64_t const sz_ngrams_powers_of_256_k[9] = {1.0,         256.0,       65536.0,     16777216.0,  268441565.0,
                                                      270103213.0, 695485101.0, 877053692.0, 3066829947.0};

/** A slot's layout, high to low: the window's residue in 32 bits, its first byte in the query in
 *  24, its width's index in 6, then the repeat bit and the occupied bit; an empty slot is zero.
 *  The layout is the family's own: read a slot through @ref sz_ngrams_slot_first and
 *  @ref sz_ngrams_slot_repeated. */
enum {
    sz_ngrams_slot_first_shift_k = 8,
    sz_ngrams_slot_width_shift_k = 2,
    sz_ngrams_slot_repeat_k = 2,
    sz_ngrams_slot_occupied_k = 1,
};

/** The bits two slots of one window share: the residue, the width's index and the occupied bit. */
static sz_u64_t const sz_ngrams_slot_key_mask_k = 0xFFFFFFFF000000FDull;

/** Widths one engine or sketch holds at most, as a slot spends six bits on the width's index. */
enum { sz_ngrams_widths_max_k = 64 };

/** Bytes one query spans at most, as a slot spends 24 bits on a window's first byte. */
enum { sz_ngrams_query_bytes_max_k = 1 << 24 };

/** Candidate chains advanced together in the scoring loops: one modular multiply-add step is about
 *  fifty cycles of latency against about a dozen of issue, so four independent chains fill the pipe
 *  and a fifth gains nothing. */
enum { sz_ngrams_interleaved_chains_k = 4 };

/** Window starts one segment of a sketch covers, ranked together. */
enum { sz_ngrams_sketch_block_k = 64 };

/** The widest window a sketch hashes from its segments' own chains, each chain covering a segment's
 *  starts and this many bytes past them, so four chains stay within eight kilobytes of the stack
 *  and rehash at most four bytes in each one they cover; wider windows take two chains each. */
enum { sz_ngrams_sketch_reach_k = 192 };

#pragma region Public Helpers

/** 256 raised to @p width, mod p: the multiplier that shifts a prefix past that many bytes. */
STRINGZILLA_CONSTEXPR sz_u32_t sz_ngrams_power(sz_size_t width) {
    sz_u64_t const prime = sz_ngrams_modulus_k;
    sz_u64_t result = 1, base = 256ull % prime, exponent = width;
    while (exponent) {
        if (exponent & 1) result = result * base % prime;
        base = base * base % prime;
        exponent >>= 1;
    }
    return (sz_u32_t)result;
}

/** N-grams a text of @p length bytes holds, summed over @p widths_count @p widths; a width of zero
 *  or past the text holds none. */
STRINGZILLA_CONSTEXPR sz_size_t sz_ngrams_count(sz_size_t length, sz_size_t const *widths, sz_size_t widths_count) {
    sz_size_t ngrams = 0;
    for (sz_size_t index = 0; index != widths_count; ++index)
        if (widths[index] && widths[index] <= length) ngrams += length - widths[index] + 1;
    return ngrams;
}

/**
 *  @brief Mixes 32 bits, so every bit of the result depends on every bit of @p value.
 *
 *  A residue is a window's big-endian integer under a prime, so a window of up to four bytes is its
 *  own residue and its low byte is the window's last byte. The mixer is the two-round
 *  multiply-xorshift finalizer whose shifts of 16, 15 and 16 and multipliers @c 0x7FEB352D and
 *  @c 0x846CA68B have the lowest avalanche bias a search of that shape found, below MurmurHash3's.
 *  It is a bijection of 32 bits, so distinct inputs stay distinct.
 */
STRINGZILLA_CONSTEXPR sz_u32_t sz_ngrams_mix_(sz_u32_t value) {
    value ^= value >> 16;
    value *= 0x7FEB352Du;
    value ^= value >> 15;
    value *= 0x846CA68Bu;
    return value ^ (value >> 16);
}

/** Unequal bytes of the words @p a_word and @p b_word, four slots of two sketches: a byte's high
 *  bit is raised where any of its bits differ, without a carry into its neighbour, and one
 *  multiply sums the four. */
STRINGZILLA_CONSTEXPR sz_u32_t sz_ngrams_sketch_word_distance(sz_u32_t a_word, sz_u32_t b_word) {
    sz_u32_t const differing = a_word ^ b_word;
    sz_u32_t const raised = (((differing & 0x7F7F7F7Fu) + 0x7F7F7F7Fu) | differing) & 0x80808080u;
    return ((raised >> 7) * 0x01010101u) >> 24;
}

/** Slots a table of @p ngrams N-grams takes: a power of two, at least twice @p ngrams and at
 *  least 64, so a probe meets an empty slot within a few steps. */
STRINGZILLA_CONSTEXPR sz_size_t sz_ngrams_slots_count_(sz_size_t ngrams) {
    sz_size_t slots_count = 64;
    while (slots_count < 2 * ngrams) slots_count *= 2;
    return slots_count;
}

/** The bit of a table's miss filter, eight bits a slot, that a window of @p residue sets in a table
 *  of @p mask plus one slots: the high bits of one multiply, as the low bits of a short window's
 *  residue are its last byte's. Its bits past the low three name the slot a probe starts at. */
STRINGZILLA_CONSTEXPR sz_size_t sz_ngrams_filter_bit_(sz_u32_t residue, sz_size_t mask) {
    return (sz_size_t)(((sz_u64_t)(sz_u32_t)(residue * 2654435761u) * (mask + 1)) >> 29);
}

/** The slot of @p slots_count @p slots holding @p residue at width @p width_index, or zero when the
 *  query lacks that window, which its @p filter tells for most such N-grams by one load. */
STRINGZILLA_CONSTEXPR sz_u64_t sz_ngrams_slot_probe_(sz_u64_t const *slots, sz_u32_t const *filter,
                                                     sz_size_t slots_count, sz_u32_t residue, sz_size_t width_index) {
    sz_u64_t const wanted = ((sz_u64_t)residue << 32) | ((sz_u64_t)width_index << sz_ngrams_slot_width_shift_k) |
                            sz_ngrams_slot_occupied_k;
    sz_size_t const mask = slots_count - 1, bit = sz_ngrams_filter_bit_(residue, mask);
    if (!((filter[bit / 32] >> (bit % 32)) & 1)) return 0;
    for (sz_size_t slot = bit >> 3;; slot = (slot + 1) & mask) {
        sz_u64_t const entry = slots[slot];
        if (!entry || ((entry ^ wanted) & sz_ngrams_slot_key_mask_k) == 0) return entry;
    }
}

/** Files the window starting at byte @p first of width @p width_index with @p residue into @p slots
 *  and its bit into @p filter, marking its slot repeated when the window is already there; answers
 *  one for a new window and zero for a repeat. */
STRINGZILLA_INLINE sz_size_t sz_ngrams_slot_insert_(sz_u64_t *slots, sz_u32_t *filter, sz_size_t slots_count,
                                                    sz_u32_t residue, sz_size_t width_index, sz_size_t first) {
    sz_u64_t const wanted = ((sz_u64_t)residue << 32) | ((sz_u64_t)width_index << sz_ngrams_slot_width_shift_k) |
                            sz_ngrams_slot_occupied_k;
    sz_size_t const mask = slots_count - 1, bit = sz_ngrams_filter_bit_(residue, mask);
    filter[bit / 32] |= (sz_u32_t)1 << (bit % 32);
    for (sz_size_t slot = bit >> 3;; slot = (slot + 1) & mask) {
        sz_u64_t const entry = slots[slot];
        if (!entry) {
            slots[slot] = wanted | ((sz_u64_t)first << sz_ngrams_slot_first_shift_k);
            return 1;
        }
        if (((entry ^ wanted) & sz_ngrams_slot_key_mask_k) == 0) {
            slots[slot] = entry | sz_ngrams_slot_repeat_k;
            return 0;
        }
    }
}

/** One prepared query's table of distinct N-grams at every width, and what a candidate's share
 *  needs beside it: what @ref sz_ngrams_engine_query reads off an engine, or what a device walk
 *  prepares inside a span its caller owns. */
typedef struct sz_ngrams_query_t {

    /** Slots of the query's distinct N-grams, zero if empty, read by @ref sz_ngrams_query_probe. */
    sz_u64_t const *slots;

    /** A power of two, at least twice @c ngrams and at least 64. */
    sz_size_t slots_count;

    /** The miss filter after the slots, eight bits a slot, so most N-grams the query lacks are told
     *  by one load. */
    sz_u32_t const *filter;

    /** The @b [widths_count] window widths, in bytes. */
    sz_size_t const *widths;

    /** The @b [widths_count] @ref sz_ngrams_power values. */
    sz_u32_t const *powers;

    /** Entries in @c widths and @c powers. */
    sz_size_t widths_count;

    /** The query's N-grams summed over every width, repeats included, which is the number of bits
     *  a candidate's hits span. */
    sz_size_t ngrams;

    /** The query's distinct N-grams, the share's denominator. */
    sz_size_t distinct;

    /** The query's bytes, so a width past them probes nothing. */
    sz_size_t length;
} sz_ngrams_query_t;

/** Bytes of the span a device walk prepares one query of @p length bytes into at @p widths: the
 *  table's slots over every width's N-grams, its miss filter, then a power per width. */
STRINGZILLA_CONSTEXPR sz_size_t sz_ngrams_query_span_bytes(sz_size_t length, sz_size_t const *widths,
                                                           sz_size_t widths_count) {
    return sz_ngrams_slots_count_(sz_ngrams_count(length, widths, widths_count)) * (sizeof(sz_u64_t) + 1) +
           widths_count * sizeof(sz_u32_t);
}

/** The slot of @p query holding the window of @p residue at width @p width_index, or zero when the
 *  query lacks that window. */
STRINGZILLA_CONSTEXPR sz_u64_t sz_ngrams_query_probe(sz_ngrams_query_t const *query, sz_u32_t residue,
                                                     sz_size_t width_index) {
    return sz_ngrams_slot_probe_(query->slots, query->filter, query->slots_count, residue, width_index);
}

/** The first byte in its query of the window a nonzero @p slot holds. */
STRINGZILLA_CONSTEXPR sz_size_t sz_ngrams_slot_first(sz_u64_t slot) {
    return (sz_size_t)((slot >> sz_ngrams_slot_first_shift_k) & 0xFFFFFFu);
}

/** Whether the query holds the window a nonzero @p slot holds more than once, so it names no single
 *  first byte. */
STRINGZILLA_CONSTEXPR sz_bool_t sz_ngrams_slot_repeated(sz_u64_t slot) {
    return (sz_bool_t)((slot & sz_ngrams_slot_repeat_k) != 0);
}

/** The share of the query's distinct N-grams a pair may miss within @p radius, rounded up to 24
 *  fractional bits and full at 1 or more, so a pair's budget of misses is this times the query's
 *  distinct N-grams, shifted down by 24, in integers alone. */
STRINGZILLA_INLINE sz_u32_t sz_ngrams_misses_bound_(sz_f32_t radius) {
    if (!(radius < 1.0f)) return 1u << 24;
    if (!(radius > 0.0f)) return 0;
    sz_f32_t const scaled = radius * 16777216.0f;
    return (sz_u32_t)scaled + ((sz_f32_t)(sz_u32_t)scaled != scaled);
}

#pragma endregion Public Helpers

#pragma region Sketch Helpers

/** Refuses a sketch of no slots or 2³² or more, more widths than a slot can name, and a width of
 *  zero, which no window can span, or of 2³² or more. */
STRINGZILLA_CONSTEXPR sz_status_t sz_ngrams_sketch_dimensions_(sz_size_t const *widths, sz_size_t widths_count,
                                                               sz_size_t slots) {
    // Slots index gathers as signed 32-bit lanes, and four-byte ranks must fit a `sz_size_t` block.
    if (!slots || (sz_u64_t)slots > 0x7FFFFFFFull || slots > STRINGZILLA_SIZE_MAX / sizeof(sz_u32_t) ||
        widths_count > sz_ngrams_widths_max_k)
        return sz_unexpected_dimensions_k;
    for (sz_size_t width_index = 0; width_index != widths_count; ++width_index)
        if (!widths[width_index] || widths[width_index] > 0xFFFFFFFFu) return sz_unexpected_dimensions_k;
    return sz_success_k;
}

/** Lowers the least @p ranks at the @p count slots of @p slots_of to the ranks of @p ranks_of. */
STRINGZILLA_INLINE void sz_ngrams_sketch_scatter_(sz_u32_t const *slots_of, sz_u32_t const *ranks_of, sz_size_t count,
                                                  sz_u32_t *ranks) {
    for (sz_size_t window = 0; window != count; ++window) {
        sz_u32_t *const rank = ranks + slots_of[window];
        *rank = ranks_of[window] < *rank ? ranks_of[window] : *rank;
    }
}

/** The slot and the rank of each of @p count N-grams of one @p width from their @p residues: a
 *  window's mixed value picks one of @p slots by a multiply-high and ranks it by a second mix. */
STRINGZILLA_INLINE void sz_ngrams_sketch_hash_serial_(sz_u32_t const *residues, sz_size_t count, sz_size_t width,
                                                      sz_size_t slots, sz_u32_t *slots_of, sz_u32_t *ranks_of) {
    for (sz_size_t window = 0; window != count; ++window) {
        sz_u32_t const mixed = sz_ngrams_mix_(residues[window] + (sz_u32_t)width);
        slots_of[window] = (sz_u32_t)(((sz_u64_t)mixed * slots) >> 32);
        ranks_of[window] = sz_ngrams_mix_(mixed ^ 0x9E3779B9u);
    }
}

/** Ranks @p count N-grams of one @p width by their @p residues into the @p slots least @p ranks. */
STRINGZILLA_INLINE void sz_ngrams_sketch_block_(sz_u32_t const *residues, sz_size_t count, sz_size_t width,
                                                sz_u32_t *ranks, sz_size_t slots) {
    sz_u32_t slots_of[sz_ngrams_sketch_block_k], ranks_of[sz_ngrams_sketch_block_k];
    sz_ngrams_sketch_hash_serial_(residues, count, width, slots, slots_of, ranks_of);
    sz_ngrams_sketch_scatter_(slots_of, ranks_of, count, ranks);
}

/**
 *  @brief Densifies the @p slots least @p ranks and writes each one's mixed low byte to @p sketch.
 *
 *  A slot no window reached, still all ones, takes the rank of the nearest reached slot after it,
 *  wrapping past the last, mixed with its distance, so two texts agree there about as often as at
 *  the slot it borrowed from. A text with no N-grams keeps all ones everywhere, and every byte of
 *  its sketch is @c 0x4A, the low byte of all ones mixed.
 */
STRINGZILLA_INLINE void sz_ngrams_sketch_close_(sz_u32_t *ranks, sz_size_t slots, sz_u8_t *sketch) {
    sz_size_t last = slots;
    while (last != 0 && ranks[last - 1] == 0xFFFFFFFFu) --last;
    if (last != 0) {
        // Below the last reached slot, each unreached one borrows from the reached one above it;
        // above it, from the first reached slot, past the wrap.
        // Branchless, as reached and unreached slots interleave at random.
        sz_size_t next = --last;
        for (sz_size_t slot = last; slot-- != 0;) {
            sz_u32_t const own = ranks[slot], borrowed = sz_ngrams_mix_(ranks[next] + (sz_u32_t)(next - slot));
            sz_bool_t const reached = (sz_bool_t)(own != 0xFFFFFFFFu);
            ranks[slot] = reached ? own : borrowed;
            next = reached ? slot : next;
        }
        for (sz_size_t slot = last + 1; slot != slots; ++slot)
            ranks[slot] = sz_ngrams_mix_(ranks[next] + (sz_u32_t)(next + slots - slot));
    }
    for (sz_size_t slot = 0; slot != slots; ++slot) sketch[slot] = (sz_u8_t)sz_ngrams_mix_(ranks[slot]);
}

/** The lowest slot a window reached into @p lowest, answering whether any did. */
STRINGZILLA_INLINE sz_bool_t sz_ngrams_sketch_lowest_(sz_u32_t const *ranks, sz_size_t slots, sz_size_t *lowest) {
    for (sz_size_t slot = 0; slot != slots; ++slot)
        if (ranks[slot] != 0xFFFFFFFFu) {
            *lowest = slot;
            return sz_true_k;
        }
    return sz_false_k;
}

/** For each slot from @p low below @p high, walking down, the nearest reached slot after it into
 *  @p nexts, starting from @p next, the nearest reached slot at or past @p high, wrapping, and
 *  leaving there the nearest one at or past @p low. A densifying pass over a block of slots then
 *  borrows from these by a gather, with no chain from slot to slot. */
STRINGZILLA_INLINE void sz_ngrams_sketch_nexts_(sz_u32_t const *ranks, sz_size_t low, sz_size_t high, sz_size_t *next,
                                                sz_u32_t *nexts) {
    sz_size_t borrowed = *next;
    for (sz_size_t slot = high; slot-- != low;) {
        nexts[slot - low] = (sz_u32_t)borrowed;
        borrowed = ranks[slot] != 0xFFFFFFFFu ? slot : borrowed;
    }
    *next = borrowed;
}

/** The prefix residue of the first @p count bytes of @p text, which primes a chain. */
STRINGZILLA_CONSTEXPR sz_f64_t sz_ngrams_prefix_serial_(sz_cptr_t text, sz_size_t count) {
    sz_u64_t prior = 0;
    for (sz_size_t position = 0; position != count; ++position)
        prior = (prior * 256 + (sz_u8_t)text[position]) % sz_ngrams_modulus_k;
    return (sz_f64_t)prior;
}

#pragma endregion Sketch Helpers

#pragma region Engine

/**
 *  @brief Every prepared query's table over one block, the widths every one of them holds, and the
 *      scratch both verbs grow.
 *
 *  Only @c count, @c widths_count, @c capability and @c distinct_counts are public; the rest is the
 *  family's own layout, read through @ref sz_ngrams_engine_query.
 */
typedef struct sz_ngrams_engine_t {

    /** Every query's slots, back to back, @c slots_offsets addressing them. */
    sz_u64_t const *slots;

    /** The @b [count+1] slot offsets into @c slots, the last being its length. */
    sz_size_t const *slots_offsets;

    /** Every query's miss filter, eight bits a slot, the query @c i 's from word
     *  @c slots_offsets[i]/4. */
    sz_u32_t const *filters;

    /** The @b [count,widths_count] distinct N-grams each query holds at each width, which is a
     *  share's denominator. */
    sz_u32_t const *distinct_counts;

    /** The @b [widths_count] window widths, in bytes. */
    sz_size_t const *widths;

    /** The @b [widths_count] @ref sz_ngrams_power values. */
    sz_u32_t const *powers;

    /** The @b [count] bytes each query spans. */
    sz_u32_t const *lengths;

    /** Queries prepared, which is the first axis of every overlap output; zero for an engine that
     *  only sketches. */
    sz_size_t count;

    /** Window widths, the last axis of an overlap output. */
    sz_size_t widths_count;

    /** The capability whose init kernel prepared the engine, which picks every verb's kernel. */
    sz_capability_t capability;

    /** What built both blocks below and what grows the second. */
    sz_allocator_t allocator;

    /** The tables' block, fixed for the engine's life, its head tier-private. */
    void *memory;

    /** Bytes of that block. */
    sz_size_t memory_bytes;

    /** A round's chains, residues and hit words, or a sketch's ranks, grown and never shrunk. */
    void *scratch;

    /** Bytes of that block, zero until a verb sizes it. */
    sz_size_t scratch_bytes;
} sz_ngrams_engine_t;

/** The tier's own record at the head of @p engine 's block, where @ref sz_ngrams_engine_open_
 *  reserved it. */
STRINGZILLA_INLINE void *sz_ngrams_engine_head_(sz_ngrams_engine_t const *engine) {
    return (void *)(((sz_size_t)engine->memory + 63) & ~(sz_size_t)63);
}

/** The distinct N-grams query @p index of @p engine holds over every width. */
STRINGZILLA_CONSTEXPR sz_size_t sz_ngrams_engine_distinct_(sz_ngrams_engine_t const *engine, sz_size_t index) {
    sz_size_t distinct = 0;
    for (sz_size_t width_index = 0; width_index != engine->widths_count; ++width_index)
        distinct += engine->distinct_counts[index * engine->widths_count + width_index];
    return distinct;
}

/** Query @p index of a host @p engine as one table, for a caller probing it window by window. */
STRINGZILLA_CONSTEXPR sz_ngrams_query_t sz_ngrams_engine_query(sz_ngrams_engine_t const *engine, sz_size_t index) {
    sz_size_t const first = engine->slots_offsets[index];
    sz_ngrams_query_t query = {STRINGZILLA_NULL, 0, STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL, 0, 0, 0, 0};
    query.slots = engine->slots + first, query.slots_count = engine->slots_offsets[index + 1] - first;
    query.filter = engine->filters + first / 4;
    query.widths = engine->widths, query.powers = engine->powers, query.widths_count = engine->widths_count;
    query.length = engine->lengths[index];
    query.ngrams = sz_ngrams_count(query.length, engine->widths, engine->widths_count);
    query.distinct = sz_ngrams_engine_distinct_(engine, index);
    return query;
}

/** The most N-grams any query of @p engine holds, which sizes a candidate's hit words. */
STRINGZILLA_INLINE sz_size_t sz_ngrams_engine_widest_ngrams_(sz_ngrams_engine_t const *engine) {
    sz_size_t widest = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        widest = sz_max_of_two(widest, sz_ngrams_count(engine->lengths[index], engine->widths, engine->widths_count));
    return widest;
}

/** The length of the longest of @p strings, which sizes a round's chains. */
STRINGZILLA_INLINE sz_size_t sz_ngrams_longest_(sz_strings_view_t const *strings) {
    sz_size_t longest = 0;
    sz_string_view_t views[64];
    for (sz_size_t block_start = 0; block_start < strings->count; block_start += 64) {
        sz_size_t const block_count = sz_strings_get_block_(strings, block_start, views);
        for (sz_size_t position = 0; position != block_count; ++position)
            longest = sz_max_of_two(longest, views[position].length);
    }
    return longest;
}

/**
 *  @brief Takes one block for the tables of @p queries and points every view of @p engine into it.
 *
 *  Leaves every slot, filter bit and @c distinct_counts zero: a tier fills them with its own
 *  hashing, and records the capability it did so at.
 *
 *  @param[in] queries The queries, or null for an engine that only sketches.
 *  @param[in] head_bytes Bytes the tier keeps for itself before the tables, rounded up to
 *      a cache line.
 *  @param[in] allocator Whose residency every view of the engine inherits; stored by value, so
 *      @c _free needs no other.
 *
 *  @return @c sz_success_k; @c sz_unexpected_dimensions_k for zero widths, more than
 *      @ref sz_ngrams_widths_max_k of them, a width of zero or past 32 bits, or a query past
 *      @ref sz_ngrams_query_bytes_max_k; or @c sz_bad_alloc_k when the block cannot be taken.
 */
STRINGZILLA_INLINE sz_status_t sz_ngrams_engine_open_(sz_strings_view_t const *queries, sz_size_t const *window_widths,
                                                      sz_size_t window_widths_count, sz_size_t head_bytes,
                                                      sz_allocator_t const *allocator, sz_stream_t stream,
                                                      sz_ngrams_engine_t *engine) {
    if (!window_widths_count || window_widths_count > sz_ngrams_widths_max_k) return sz_unexpected_dimensions_k;
    for (sz_size_t width_index = 0; width_index != window_widths_count; ++width_index)
        if (!window_widths[width_index] || window_widths[width_index] > 0xFFFFFFFFu) return sz_unexpected_dimensions_k;
    sz_size_t const count = queries ? queries->count : 0;
    sz_size_t const head = (head_bytes + 63) & ~(sz_size_t)63;

    sz_size_t slots_count = 0;
    sz_string_view_t views[64];
    for (sz_size_t block_start = 0; block_start < count; block_start += 64) {
        sz_size_t const block_count = sz_strings_get_block_(queries, block_start, views);
        for (sz_size_t position = 0; position != block_count; ++position) {
            sz_size_t const length = views[position].length;
            if (length > sz_ngrams_query_bytes_max_k) return sz_unexpected_dimensions_k;
            sz_size_t ngrams = 0;
            for (sz_size_t width_index = 0; width_index != window_widths_count; ++width_index)
                ngrams += window_widths[width_index] <= length ? length - window_widths[width_index] + 1 : 0;
            slots_count += sz_ngrams_slots_count_(ngrams);
        }
    }

    sz_size_t const offsets_bytes = sz_size_divide_round_up((count + 1 + window_widths_count) * sizeof(sz_size_t), 8) *
                                    8;
    sz_size_t const total_bytes = 63 + head + offsets_bytes + slots_count * (sizeof(sz_u64_t) + 1) +
                                  (count * window_widths_count + count + window_widths_count) * sizeof(sz_u32_t);
    sz_ptr_t const allocation = (sz_ptr_t)allocator->allocate(total_bytes, allocator->handle, stream);
    if (!allocation) return sz_bad_alloc_k;

    sz_ptr_t const aligned = (sz_ptr_t)(((sz_size_t)allocation + 63) & ~(sz_size_t)63);
    sz_size_t *const slots_offsets = (sz_size_t *)(aligned + head);
    sz_size_t *const widths = slots_offsets + count + 1;
    sz_u64_t *const slots = (sz_u64_t *)((sz_ptr_t)slots_offsets + offsets_bytes);
    sz_u32_t *const filters = (sz_u32_t *)(slots + slots_count);
    sz_u32_t *const distinct_counts = filters + slots_count / 4;
    sz_u32_t *const lengths = distinct_counts + count * window_widths_count;
    sz_u32_t *const powers = lengths + count;

    sz_size_t written = 0;
    for (sz_size_t block_start = 0; block_start < count; block_start += 64) {
        sz_size_t const block_count = sz_strings_get_block_(queries, block_start, views);
        for (sz_size_t position = 0; position != block_count; ++position) {
            sz_size_t const index = block_start + position;
            sz_size_t const length = views[position].length;
            sz_size_t ngrams = 0;
            for (sz_size_t width_index = 0; width_index != window_widths_count; ++width_index)
                ngrams += window_widths[width_index] <= length ? length - window_widths[width_index] + 1 : 0;
            slots_offsets[index] = written;
            written += sz_ngrams_slots_count_(ngrams);
            lengths[index] = (sz_u32_t)length;
        }
    }
    slots_offsets[count] = written;
    for (sz_size_t slot = 0; slot != slots_count; ++slot) slots[slot] = 0;
    for (sz_size_t word = 0; word != slots_count / 4; ++word) filters[word] = 0;
    for (sz_size_t entry = 0; entry != count * window_widths_count; ++entry) distinct_counts[entry] = 0;
    for (sz_size_t index = 0; index != window_widths_count; ++index) {
        widths[index] = window_widths[index];
        powers[index] = sz_ngrams_power(window_widths[index]);
    }

    engine->slots = slots, engine->slots_offsets = slots_offsets, engine->filters = filters;
    engine->distinct_counts = distinct_counts;
    engine->widths = widths, engine->powers = powers, engine->lengths = lengths;
    engine->count = count, engine->widths_count = window_widths_count;
    engine->capability = 0;
    engine->allocator = *allocator;
    engine->memory = allocation, engine->memory_bytes = total_bytes;
    engine->scratch = STRINGZILLA_NULL, engine->scratch_bytes = 0;
    return sz_success_k;
}

/** Grows @p engine 's scratch block to @p bytes on the device of @p stream, keeping whatever it
 *  already holds when that is enough. */
STRINGZILLA_INLINE sz_status_t sz_ngrams_engine_grow_(sz_ngrams_engine_t *engine, sz_size_t bytes, sz_stream_t stream) {
    if (engine->scratch_bytes >= bytes) return sz_success_k;
    if (engine->scratch)
        engine->allocator.free(engine->scratch, engine->scratch_bytes, engine->allocator.handle, stream);
    engine->scratch = engine->allocator.allocate(bytes, engine->allocator.handle, stream);
    engine->scratch_bytes = engine->scratch ? bytes : 0;
    return engine->scratch ? sz_success_k : sz_bad_alloc_k;
}

/** Returns both of @p engine 's blocks to the allocator they were built with, once the work queued
 *  on @p stream is done with them, and leaves it empty. */
STRINGZILLA_INLINE void sz_ngrams_engine_close_(sz_ngrams_engine_t *engine, sz_stream_t stream) {
    if (engine->scratch)
        engine->allocator.free(engine->scratch, engine->scratch_bytes, engine->allocator.handle, stream);
    if (engine->memory) engine->allocator.free(engine->memory, engine->memory_bytes, engine->allocator.handle, stream);
    engine->slots = STRINGZILLA_NULL, engine->slots_offsets = STRINGZILLA_NULL, engine->filters = STRINGZILLA_NULL;
    engine->distinct_counts = STRINGZILLA_NULL;
    engine->widths = STRINGZILLA_NULL, engine->powers = STRINGZILLA_NULL, engine->lengths = STRINGZILLA_NULL;
    engine->count = 0, engine->widths_count = 0, engine->capability = 0;
    engine->memory = STRINGZILLA_NULL, engine->memory_bytes = 0;
    engine->scratch = STRINGZILLA_NULL, engine->scratch_bytes = 0;
}

/** Opens @p engine for the host from @p allocator, or from the heap when it is null. */
STRINGZILLA_INLINE sz_status_t sz_ngrams_engine_open_host_(sz_strings_view_t const *queries,
                                                           sz_size_t const *window_widths,
                                                           sz_size_t window_widths_count, sz_allocator_t *allocator,
                                                           sz_ngrams_engine_t *engine) {
    sz_allocator_t host;
    if (allocator) host = *allocator;
    else {
        sz_status_t const status = sz_allocator_init_heap(&host);
        if (status != sz_success_k) return status;
    }
    return sz_ngrams_engine_open_(queries, window_widths, window_widths_count, 0, &host, STRINGZILLA_NULL, engine);
}

/**
 *  @brief Grows @p engine 's scratch for the residues of its longest query on the host.
 *  @param[out] prefixes The @b [longest+1] prefix residues, at the head of the block.
 *  @param[out] residues The longest query's N-gram residues at every width, after them.
 *  @return @c sz_success_k or @c sz_bad_alloc_k, having closed @p engine.
 */
STRINGZILLA_INLINE sz_status_t sz_ngrams_engine_query_scratch_(sz_ngrams_engine_t *engine, sz_stream_t stream,
                                                               sz_f64_t **prefixes, sz_u32_t **residues) {
    sz_size_t longest_query = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        longest_query = sz_max_of_two(longest_query, (sz_size_t)engine->lengths[index]);
    sz_size_t const ngrams = sz_ngrams_count(longest_query, engine->widths, engine->widths_count);
    sz_status_t const grown = sz_ngrams_engine_grow_(
        engine, (longest_query + 1) * sizeof(sz_f64_t) + (ngrams + 1) * sizeof(sz_u32_t), stream);
    if (grown != sz_success_k) {
        sz_ngrams_engine_close_(engine, stream);
        return grown;
    }
    *prefixes = (sz_f64_t *)engine->scratch;
    *residues = (sz_u32_t *)(*prefixes + longest_query + 1);
    return sz_success_k;
}

/** Files query @p index 's window @p residues, width after width in position order, into its slots
 *  of @p engine, so a slot keeps its window's first byte, and counts distinct N-grams per width. */
STRINGZILLA_INLINE void sz_ngrams_engine_fill_row_(sz_ngrams_engine_t *engine, sz_size_t index,
                                                   sz_u32_t const *residues) {
    // The tables and counts stay writable until the engine is handed back; readers see them const.
    sz_u64_t *const slots = (sz_u64_t *)engine->slots + engine->slots_offsets[index];
    sz_u32_t *const filter = (sz_u32_t *)engine->filters + engine->slots_offsets[index] / 4;
    sz_u32_t *const distinct_counts = (sz_u32_t *)engine->distinct_counts + index * engine->widths_count;
    sz_size_t const slots_count = engine->slots_offsets[index + 1] - engine->slots_offsets[index];
    sz_size_t const length = engine->lengths[index];
    for (sz_size_t width_index = 0; width_index != engine->widths_count; ++width_index) {
        sz_size_t const width = engine->widths[width_index];
        sz_size_t const ngrams = width <= length ? length - width + 1 : 0;
        sz_size_t distinct = 0;
        for (sz_size_t window = 0; window != ngrams; ++window)
            distinct += sz_ngrams_slot_insert_(slots, filter, slots_count, residues[window], width_index, window);
        distinct_counts[width_index] = (sz_u32_t)distinct;
        residues += ngrams;
    }
}

/** Bytes one round's @p chains interleaved chains need over its longest candidate, its window
 *  residues at every width of @p engine, width after width, the hit words of its widest query, and
 *  the @p order_entries entries its hashing order is built in. */
STRINGZILLA_INLINE sz_size_t sz_ngrams_engine_round_bytes_(sz_ngrams_engine_t const *engine,
                                                           sz_size_t longest_candidate, sz_size_t chains,
                                                           sz_size_t order_entries) {
    sz_size_t const residues_bytes = (longest_candidate + 1) * chains * sizeof(sz_f64_t) +
                                     (sz_ngrams_count(longest_candidate, engine->widths, engine->widths_count) + 1) *
                                         sizeof(sz_u32_t);
    sz_size_t const hit_words = sz_size_divide_round_up(sz_ngrams_engine_widest_ngrams_(engine), 64);
    return sz_size_divide_round_up(residues_bytes, sizeof(sz_u64_t)) * sizeof(sz_u64_t) + hit_words * sizeof(sz_u64_t) +
           order_entries * sizeof(sz_size_t);
}

/** Refuses a stride under the axis it spans, so no row can be written into its neighbour's. */
STRINGZILLA_INLINE sz_status_t sz_ngrams_engine_strides_(sz_ngrams_engine_t const *engine, sz_size_t candidates_count,
                                                         sz_size_t matches_query_stride,
                                                         sz_size_t matches_candidate_stride) {
    if (matches_candidate_stride < engine->widths_count) return sz_unexpected_dimensions_k;
    if (matches_query_stride < candidates_count * matches_candidate_stride) return sz_unexpected_dimensions_k;
    return sz_success_k;
}

/**
 *  @brief Counts one query's distinct N-grams that one candidate holds, at every width, as
 *      @ref sz_ngrams_overlaps counts them.
 *
 *  Every window of the candidate probes the query's table, and a hit sets the bit of the window's
 *  first byte in the query, offset by the N-grams of the widths before, so a window the candidate
 *  repeats, or one the query repeats, counts once. The pair may miss @p misses_bound of the query's
 *  distinct N-grams, as @c sz_ngrams_misses_bound_ rounds its radius, and a candidate with too few
 *  N-grams to come within it probes none.
 *
 *  @param[in] residues The candidate's N-gram residues, as @ref sz_ngrams_count counts them.
 *  @param[in] hit_words Scratch of at least the query's N-grams in bits, overwritten.
 *  @param[out] row The pair's count at width zero, the other widths following it.
 */
STRINGZILLA_INLINE void sz_ngrams_pair_overlaps_(sz_ngrams_engine_t const *engine, sz_size_t query,
                                                 sz_u32_t const *residues, sz_size_t length, sz_u32_t misses_bound,
                                                 sz_u64_t *hit_words, sz_u32_t *row) {
    sz_size_t const widths_count = engine->widths_count;
    sz_size_t const query_length = engine->lengths[query];
    sz_size_t const distinct = sz_ngrams_engine_distinct_(engine, query);
    sz_u64_t const *const slots = engine->slots + engine->slots_offsets[query];
    sz_u32_t const *const filter = engine->filters + engine->slots_offsets[query] / 4;
    sz_size_t const slots_count = engine->slots_offsets[query + 1] - engine->slots_offsets[query];
    sz_size_t reachable = 0, held = 0, width_index;
    for (width_index = 0; width_index != widths_count; ++width_index) {
        sz_size_t const width = engine->widths[width_index];
        sz_size_t const ngrams = width <= length ? length - width + 1 : 0;
        reachable += width <= query_length ? ngrams : 0;
        row[width_index] = 0;
    }

    sz_size_t const bound = (sz_size_t)(((sz_u64_t)misses_bound * distinct) >> 24);
    if (distinct - sz_min_of_two(distinct, reachable) > bound) return;

    sz_size_t const query_ngrams = sz_ngrams_count(query_length, engine->widths, widths_count);
    for (sz_size_t word = 0; word != sz_size_divide_round_up(query_ngrams, 64); ++word) hit_words[word] = 0;
    sz_u32_t const *keys = residues;
    sz_size_t base = 0;
    for (width_index = 0; width_index != widths_count; ++width_index) {
        sz_size_t const width = engine->widths[width_index];
        sz_size_t const ngrams = width <= length ? length - width + 1 : 0;
        sz_size_t const own_ngrams = width <= query_length ? query_length - width + 1 : 0;
        sz_size_t found = 0;
        for (sz_size_t window = 0; own_ngrams && window != ngrams; ++window) {
            sz_u64_t const entry = sz_ngrams_slot_probe_(slots, filter, slots_count, keys[window], width_index);
            if (!entry) continue;
            sz_size_t const bit = base + sz_ngrams_slot_first(entry);
            sz_u64_t const mask = (sz_u64_t)1 << (bit & 63);
            found += (hit_words[bit >> 6] & mask) == 0;
            hit_words[bit >> 6] |= mask;
        }
        row[width_index] = (sz_u32_t)found;
        held += found;
        keys += ngrams;
        base += own_ngrams;
    }
    // Past its budget after counting, every width counts zero.
    if (distinct - held > bound)
        for (width_index = 0; width_index != widths_count; ++width_index) row[width_index] = 0;
}

/** What one round hashes, and where it counts each hashed candidate against every query. */
typedef struct sz_ngrams_round_t {

    /** The candidates' indices in the order the round hashes them. */
    sz_size_t const *order;

    /** Candidates the round hashes. */
    sz_size_t count;

    /** The misses the radius allows, as @c sz_ngrams_misses_bound_ rounds it. */
    sz_u32_t misses_bound;

    /** One pair's hit words, as many as the widest query's N-grams in bits. */
    sz_u64_t *hit_words;

    /** Where the counts go. */
    sz_u32_t *matches;

    /** Entries from one query's plane of @c matches to the next. */
    sz_size_t matches_query_stride;

    /** Entries from one candidate's row of @c matches to the next. */
    sz_size_t matches_candidate_stride;
} sz_ngrams_round_t;

/** Counts @p candidate of @p round 's batch against every query, from its @p residues. */
STRINGZILLA_INLINE void sz_ngrams_round_score_(sz_ngrams_engine_t const *engine, sz_ngrams_round_t const *round,
                                               sz_size_t candidate, sz_u32_t const *residues, sz_size_t length) {
    for (sz_size_t query = 0; query != engine->count; ++query)
        sz_ngrams_pair_overlaps_(
            engine, query, residues, length, round->misses_bound, round->hit_words,
            round->matches + query * round->matches_query_stride + candidate * round->matches_candidate_stride);
}

/**
 *  @brief Grows @p engine 's scratch for the cross product of @p candidates and lays out a
 *      @p round that hashes them ascending by length, so the chains one group interleaves end
 *      together and no long remainder steps alone.
 *  @param[out] longest_candidate The longest of @p candidates, which strides the chains.
 *  @return @c sz_success_k, with an empty @p round when either batch is empty; @c sz_bad_alloc_k;
 *      or @c sz_unexpected_dimensions_k when either stride is under the axis it spans.
 */
STRINGZILLA_INLINE sz_status_t sz_ngrams_overlaps_round_(sz_ngrams_engine_t *engine,
                                                         sz_strings_view_t const *candidates, sz_f32_t radius,
                                                         sz_u32_t *matches, sz_size_t matches_query_stride,
                                                         sz_size_t matches_candidate_stride, sz_size_t chains,
                                                         sz_stream_t stream, sz_ngrams_round_t *round,
                                                         sz_size_t *longest_candidate) {
    sz_status_t const dimensions = sz_ngrams_engine_strides_(engine, candidates->count, matches_query_stride,
                                                             matches_candidate_stride);
    if (dimensions != sz_success_k) return dimensions;
    round->count = 0;
    if (!candidates->count || !engine->count) return sz_success_k;
    *longest_candidate = sz_ngrams_longest_(candidates);
    sz_size_t const round_bytes = sz_ngrams_engine_round_bytes_(engine, *longest_candidate, chains, 0);
    sz_size_t const hit_words_offset =
        round_bytes - sz_size_divide_round_up(sz_ngrams_engine_widest_ngrams_(engine), 64) * sizeof(sz_u64_t);
    sz_status_t const grown = sz_ngrams_engine_grow_(
        engine, sz_ngrams_engine_round_bytes_(engine, *longest_candidate, chains, 3 * candidates->count), stream);
    if (grown != sz_success_k) return grown;
    sz_size_t *const area = (sz_size_t *)((sz_ptr_t)engine->scratch + round_bytes);
    round->order = sz_strings_order_by_length_(candidates, area);
    round->count = candidates->count;
    round->misses_bound = sz_ngrams_misses_bound_(radius);
    round->hit_words = (sz_u64_t *)((sz_ptr_t)engine->scratch + hit_words_offset);
    round->matches = matches;
    round->matches_query_stride = matches_query_stride, round->matches_candidate_stride = matches_candidate_stride;
    return sz_success_k;
}

/** Refuses a sketch batch whose stride is under its slots, then grows @p engine 's scratch for the
 *  ranks of @p slots slots, 4 bytes a slot. */
STRINGZILLA_INLINE sz_status_t sz_ngrams_sketches_scratch_(sz_ngrams_engine_t *engine, sz_size_t slots,
                                                           sz_size_t sketches_stride, sz_u32_t **ranks) {
    sz_status_t const dimensions = sz_ngrams_sketch_dimensions_(engine->widths, engine->widths_count, slots);
    if (dimensions != sz_success_k) return dimensions;
    if (sketches_stride < slots) return sz_unexpected_dimensions_k;
    sz_status_t const grown = sz_ngrams_engine_grow_(engine, slots * sizeof(sz_u32_t), STRINGZILLA_NULL);
    *ranks = (sz_u32_t *)engine->scratch;
    return grown;
}

#pragma endregion Engine

#pragma region Serial

/**
 *  @brief (multiplier · multiplicand + addend) mod p, in [0, p), exact for every input below 2³²:
 *      the product fits one @c u64.
 *  @see https://ashvardanian.com/posts/stringwars-on-gpus/#52-bit-math-for-bioinformatics
 */
STRINGZILLA_CONSTEXPR sz_f64_t sz_ngrams_multiply_add_serial_(sz_f64_t multiplier, sz_f64_t multiplicand,
                                                              sz_f64_t addend) {
    sz_u64_t const product = (sz_u64_t)multiplier * (sz_u64_t)multiplicand + (sz_u64_t)addend;
    return (sz_f64_t)(product % (sz_u64_t)sz_ngrams_modulus_k);
}

/** Advances a prefix chain over one byte, writing @c P(k+1) and answering it for the next step. */
STRINGZILLA_INLINE sz_f64_t sz_ngrams_f64x1_prefix_step_serial_(sz_f64_t prior, sz_cptr_t text, sz_f64_t *prefixes) {
    sz_f64_t const next = sz_ngrams_multiply_add_serial_(prior, sz_ngrams_powers_of_256_k[1],
                                                         (sz_f64_t)(sz_u8_t)text[0]);
    prefixes[0] = next;
    return next;
}

/** One window's residue, R(i, w) = P(i + w) − P(i) · bʷ mod p, from the prefix residues at its
 *  first byte and past its last. */
STRINGZILLA_CONSTEXPR void sz_ngrams_f64x1_residue_step_serial_(sz_f64_t const *starts, sz_f64_t const *ends,
                                                                sz_f64_t power, sz_u32_t *residues) {
    sz_f64_t const shifted = sz_ngrams_multiply_add_serial_(starts[0], power, 0.0);
    sz_f64_t residue = ends[0] - shifted;
    if (residue < 0.0) residue += (sz_f64_t)sz_ngrams_modulus_k;
    residues[0] = (sz_u32_t)residue;
}

/** The residues of @p count N-grams of one width, from the prefix residues at their first bytes,
 *  @p starts, and past their last, @p ends. */
STRINGZILLA_INLINE void sz_ngrams_residues_span_serial_(sz_f64_t const *starts, sz_f64_t const *ends, sz_f64_t power,
                                                        sz_size_t count, sz_u32_t *residues) {
    for (sz_size_t window = 0; window != count; ++window)
        sz_ngrams_f64x1_residue_step_serial_(starts + window, ends + window, power, residues + window);
}

/** Writes the @p length + 1 prefix residues of @p text into @p prefixes, from zero. */
STRINGZILLA_INLINE void sz_ngrams_prefixes_serial_(sz_cptr_t text, sz_size_t length, sz_f64_t *prefixes) {
    sz_f64_t prior = prefixes[0] = 0.0;
    for (sz_size_t position = 0; position != length; ++position)
        prior = sz_ngrams_f64x1_prefix_step_serial_(prior, text + position, prefixes + position + 1);
}

/** The residues of every window of @p text at every width into @p residues, width after width,
 *  leaving its prefix residues in @p prefixes. */
STRINGZILLA_INLINE void sz_ngrams_residues_serial_(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                                   sz_size_t widths_count, sz_f64_t *prefixes, sz_u32_t *residues) {
    sz_ngrams_prefixes_serial_(text, length, prefixes);
    for (sz_size_t width_index = 0; width_index != widths_count; ++width_index) {
        sz_size_t const width = widths[width_index];
        if (!width || width > length) continue;
        sz_f64_t const power = (sz_f64_t)sz_ngrams_power(width);
        sz_size_t const ngrams = length - width + 1;
        sz_ngrams_residues_span_serial_(prefixes, prefixes + width, power, ngrams, residues);
        residues += ngrams;
    }
}

/**
 *  @brief Hashes every query of @p queries and files its N-grams into its table, on the host.
 *
 *  The device tiers fill their tables this way too, as filing a few thousand N-grams per query is
 *  no launch's worth; only where the tables live differs.
 *
 *  @param[out] prefixes Host working space of the longest query's length plus one entries.
 *  @param[out] residues Host working space of the longest query's N-grams.
 */
STRINGZILLA_INLINE void sz_ngrams_engine_fill_serial_(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                      sz_f64_t *prefixes, sz_u32_t *residues) {
    sz_string_view_t views[64];
    for (sz_size_t index = 0; index != engine->count; ++index) {
        if (index % 64 == 0) sz_strings_get_block_(queries, index, views);
        sz_ngrams_residues_serial_(views[index % 64].start, engine->lengths[index], engine->widths,
                                   engine->widths_count, prefixes, residues);
        sz_ngrams_engine_fill_row_(engine, index, residues);
    }
}

/** Ranks every window of @p text at the @p widths past @ref sz_ngrams_sketch_reach_k into @p ranks,
 *  two chains a width in lockstep, one at the N-grams' first bytes and one past their last. */
STRINGZILLA_INLINE void sz_ngrams_sketch_wide_serial_(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                                      sz_size_t widths_count, sz_u32_t *ranks, sz_size_t slots) {
    sz_f64_t starts[sz_ngrams_sketch_block_k + 1], ends[sz_ngrams_sketch_block_k];
    sz_u32_t residues[sz_ngrams_sketch_block_k];
    for (sz_size_t width_index = 0; width_index != widths_count; ++width_index) {
        sz_size_t const width = widths[width_index];
        if (width <= sz_ngrams_sketch_reach_k || width > length) continue;
        sz_size_t const ngrams = length - width + 1;
        sz_f64_t const power = (sz_f64_t)sz_ngrams_power(width);
        sz_f64_t trailing = 0.0, leading = sz_ngrams_prefix_serial_(text, width - 1);
        for (sz_size_t first = 0, count; first != ngrams; first += count) {
            count = sz_min_of_two(ngrams - first, (sz_size_t)sz_ngrams_sketch_block_k);
            starts[0] = trailing;
            for (sz_size_t window = 0; window != count; ++window) {
                trailing = sz_ngrams_f64x1_prefix_step_serial_(trailing, text + first + window, starts + window + 1);
                leading = sz_ngrams_f64x1_prefix_step_serial_(leading, text + first + width - 1 + window,
                                                              ends + window);
            }
            sz_ngrams_residues_span_serial_(starts, ends, power, count, residues);
            sz_ngrams_sketch_block_(residues, count, width, ranks, slots);
        }
    }
}

/** The widest of @p widths within both @p length and @ref sz_ngrams_sketch_reach_k, and each one's
 *  power into @p powers, so the segments' chains know how far to reach. */
STRINGZILLA_INLINE sz_size_t sz_ngrams_sketch_reach_(sz_size_t length, sz_size_t const *widths, sz_size_t widths_count,
                                                     sz_f64_t *powers) {
    sz_size_t reach = 0;
    for (sz_size_t width_index = 0; width_index != widths_count; ++width_index) {
        sz_size_t const width = widths[width_index];
        powers[width_index] = (sz_f64_t)sz_ngrams_power(width);
        if (width <= length && width <= sz_ngrams_sketch_reach_k) reach = sz_max_of_two(reach, width);
    }
    return reach;
}

/** Ranks the N-grams of one segment of @p text, whose prefix residues from its first start are at
 *  @p prefixes, at every width within @ref sz_ngrams_sketch_reach_k, through @p residues. */
STRINGZILLA_INLINE void sz_ngrams_sketch_segment_serial_(sz_size_t first, sz_size_t length, sz_size_t const *widths,
                                                         sz_size_t widths_count, sz_f64_t const *powers,
                                                         sz_f64_t const *prefixes, sz_u32_t *residues, sz_u32_t *ranks,
                                                         sz_size_t slots) {
    for (sz_size_t width_index = 0; width_index != widths_count; ++width_index) {
        sz_size_t const width = widths[width_index];
        if (width > sz_ngrams_sketch_reach_k || first + width > length) continue;
        sz_size_t const count = sz_min_of_two(length - width + 1 - first, (sz_size_t)sz_ngrams_sketch_block_k);
        sz_ngrams_residues_span_serial_(prefixes, prefixes + width, powers[width_index], count, residues);
        sz_ngrams_sketch_block_(residues, count, width, ranks, slots);
    }
}

#if STRINGZILLA_TARGET_SERIAL

STRINGZILLA_API void sz_ngrams_residues_serial(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                               sz_size_t widths_count, sz_f64_t *prefixes, sz_u32_t *residues) {
    sz_ngrams_residues_serial_(text, length, widths, widths_count, prefixes, residues);
}

STRINGZILLA_API sz_status_t sz_ngrams_engine_init_serial(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                         sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                         sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                         sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_unused_(candidates_budget);
    sz_status_t const opened = sz_ngrams_engine_open_host_(queries, window_widths, window_widths_count, allocator,
                                                           engine);
    if (opened != sz_success_k) return opened;
    sz_f64_t *prefixes;
    sz_u32_t *residues;
    sz_status_t const grown = sz_ngrams_engine_query_scratch_(engine, stream, &prefixes, &residues);
    if (grown != sz_success_k) return grown;
    sz_ngrams_engine_fill_serial_(engine, queries, prefixes, residues);
    engine->capability = sz_cap_serial_k;
    return sz_success_k;
}

/** Hashes the candidates of @p round, four chains at a time, and scores each as @p round says. */
STRINGZILLA_INLINE void sz_ngrams_round_serial_(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                sz_size_t longest_candidate, sz_ngrams_round_t const *round) {
    sz_size_t const chains = sz_ngrams_interleaved_chains_k;
    sz_size_t const chain_stride = longest_candidate + 1;
    sz_f64_t *const prefixes = (sz_f64_t *)engine->scratch;
    sz_u32_t *const residues = (sz_u32_t *)(prefixes + chain_stride * chains);

    // One chain per candidate, not one per candidate per width, and four chains at a time: each
    // step's modular multiply-add waits on the last, so four independent chains fill the pipeline
    // one alone leaves idle. A short last group repeats its first candidate in the spare chains, so
    // every group runs the same four-chain loop.
    for (sz_size_t first = 0; first < round->count; first += chains) {
        sz_size_t const interleaved = sz_min_of_two(round->count - first, chains);
        sz_cptr_t texts[sz_ngrams_interleaved_chains_k];
        sz_size_t lengths[sz_ngrams_interleaved_chains_k];
        sz_f64_t priors[sz_ngrams_interleaved_chains_k];
        sz_size_t indices[sz_ngrams_interleaved_chains_k];
        sz_string_view_t views[sz_ngrams_interleaved_chains_k];
        sz_size_t chain, shortest = STRINGZILLA_SIZE_MAX;
        for (chain = 0; chain != chains; ++chain)
            indices[chain] = round->order[first + (chain < interleaved ? chain : 0)];
        sz_strings_get_(candidates, indices, chains, views);
        for (chain = 0; chain != chains; ++chain) {
            texts[chain] = views[chain].start, lengths[chain] = views[chain].length;
            priors[chain] = prefixes[chain * chain_stride] = 0.0;
            shortest = sz_min_of_two(shortest, lengths[chain]);
        }
        sz_size_t position = 0;
        for (; position != shortest; ++position)
            for (chain = 0; chain != chains; ++chain)
                priors[chain] = sz_ngrams_f64x1_prefix_step_serial_(priors[chain], texts[chain] + position,
                                                                    prefixes + chain * chain_stride + position + 1);

        // One candidate's residues at every width serve every query's table, so the hashing runs
        // once here and the probes run once per query over what it wrote.
        for (chain = 0; chain != interleaved; ++chain) {
            sz_f64_t *const chain_prefixes = prefixes + chain * chain_stride;
            sz_size_t const length = lengths[chain];
            sz_f64_t running = priors[chain];
            for (sz_size_t walked = position; walked != length; ++walked)
                running = sz_ngrams_f64x1_prefix_step_serial_(running, texts[chain] + walked,
                                                              chain_prefixes + walked + 1);
            sz_u32_t *keys = residues;
            for (sz_size_t width_index = 0; width_index != engine->widths_count; ++width_index) {
                sz_size_t const width = engine->widths[width_index];
                sz_size_t const ngrams = width <= length ? length - width + 1 : 0;
                sz_f64_t const power = (sz_f64_t)engine->powers[width_index];
                sz_ngrams_residues_span_serial_(chain_prefixes, chain_prefixes + width, power, ngrams, keys);
                keys += ngrams;
            }
            sz_ngrams_round_score_(engine, round, indices[chain], residues, length);
        }
    }
}

STRINGZILLA_API sz_status_t sz_ngrams_overlaps_serial(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                      sz_f32_t radius, sz_u32_t *matches,
                                                      sz_size_t matches_query_stride,
                                                      sz_size_t matches_candidate_stride, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_ngrams_round_t round;
    sz_size_t longest_candidate = 0;
    sz_status_t const prepared = sz_ngrams_overlaps_round_(engine, candidates, radius, matches, matches_query_stride,
                                                           matches_candidate_stride, sz_ngrams_interleaved_chains_k,
                                                           stream, &round, &longest_candidate);
    if (prepared != sz_success_k) return prepared;
    sz_ngrams_round_serial_(engine, candidates, longest_candidate, &round);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_ngrams_sketch_serial(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                                    sz_size_t widths_count, sz_u32_t *ranks, sz_size_t slots,
                                                    sz_u8_t *sketch) {
    sz_status_t const status = sz_ngrams_sketch_dimensions_(widths, widths_count, slots);
    if (status != sz_success_k) return status;
    for (sz_size_t slot = 0; slot != slots; ++slot) ranks[slot] = 0xFFFFFFFFu;
    sz_size_t const chains = sz_ngrams_interleaved_chains_k, block = sz_ngrams_sketch_block_k;
    sz_f64_t powers[sz_ngrams_widths_max_k];
    sz_f64_t prefixes[sz_ngrams_interleaved_chains_k][sz_ngrams_sketch_block_k + sz_ngrams_sketch_reach_k];
    sz_u32_t residues[sz_ngrams_sketch_block_k];
    sz_size_t const reach = sz_ngrams_sketch_reach_(length, widths, widths_count, powers);

    // Four segments of window starts a round, each chain from zero at its segment's first start, so
    // every window within the reach is a difference of one chain and the chains step independently.
    for (sz_size_t base = 0; reach && base < length; base += chains * block) {
        sz_size_t firsts[sz_ngrams_interleaved_chains_k], covered[sz_ngrams_interleaved_chains_k];
        sz_f64_t priors[sz_ngrams_interleaved_chains_k];
        sz_size_t chain, shortest = STRINGZILLA_SIZE_MAX;
        for (chain = 0; chain != chains; ++chain) {
            firsts[chain] = sz_min_of_two(base + chain * block, length);
            covered[chain] = sz_min_of_two(block + reach - 1, length - firsts[chain]);
            priors[chain] = prefixes[chain][0] = 0.0;
            shortest = sz_min_of_two(shortest, covered[chain]);
        }
        sz_size_t walked = 0;
        for (; walked != shortest; ++walked)
            for (chain = 0; chain != chains; ++chain)
                priors[chain] = sz_ngrams_f64x1_prefix_step_serial_(priors[chain], text + firsts[chain] + walked,
                                                                    prefixes[chain] + walked + 1);
        for (chain = 0; chain != chains; ++chain) {
            for (sz_size_t own = walked; own != covered[chain]; ++own)
                priors[chain] = sz_ngrams_f64x1_prefix_step_serial_(priors[chain], text + firsts[chain] + own,
                                                                    prefixes[chain] + own + 1);
            sz_ngrams_sketch_segment_serial_(firsts[chain], length, widths, widths_count, powers, prefixes[chain],
                                             residues, ranks, slots);
        }
    }
    sz_ngrams_sketch_wide_serial_(text, length, widths, widths_count, ranks, slots);
    sz_ngrams_sketch_close_(ranks, slots, sketch);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_ngrams_sketches_serial(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                      sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                      sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_unused_(stream);
    sz_u32_t *ranks;
    sz_status_t const prepared = sz_ngrams_sketches_scratch_(engine, slots, sketches_stride, &ranks);
    if (prepared != sz_success_k) return prepared;
    sz_string_view_t views[64];
    for (sz_size_t block_start = 0; block_start < texts->count; block_start += 64) {
        sz_size_t const block_count = sz_strings_get_block_(texts, block_start, views);
        for (sz_size_t position = 0; position != block_count; ++position)
            sz_ngrams_sketch_serial(views[position].start, views[position].length, engine->widths, engine->widths_count,
                                    ranks, slots, sketches + (block_start + position) * sketches_stride);
    }
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_SERIAL

#pragma endregion Serial

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_NGRAMS_SERIAL_H_
