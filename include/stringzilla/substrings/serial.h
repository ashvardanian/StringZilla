/**
 *  @brief Serial backend for multi-pattern search: the automaton's construction, its transition, and the
 *      walks and verb drivers every other CPU tier reuses.
 *  @file include/stringzilla/substrings/serial.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/substrings.h
 *
 *  Construction is a trie, a breadth-first failure pass, and a two-tier publication. The trie is stored as
 *  sibling lists rather than as an edge pool with an index beside it: a tree gives every state exactly one
 *  incoming edge, so a state @b is its edge, and the byte, the first child and the next sibling are three
 *  fields of the state itself. That removes the hash table an edge pool would need, and makes the edge
 *  count follow the state count instead of being bounded separately.
 *
 *  The walk is one data-dependent load per byte, so a single chain leaves the load ports idle for that
 *  whole latency. Byte-exact walks therefore step @ref SZ_SUBSTRINGS_CHAINS disjoint slices at once, each
 *  primed by the bytes before it; a walk reporting in haystack order does so in rounds of windows whose
 *  matches it buffers and flushes in window order. A SIMD tier replaces the stages in
 *  @ref sz_substrings_walks_t and inherits the verb drivers.
 */
#ifndef STRINGZILLA_SUBSTRINGS_SERIAL_H_
#define STRINGZILLA_SUBSTRINGS_SERIAL_H_

#include "stringzilla/types.h"

#include "stringzilla/memory.h"                   // `sz_copy`
#include "stringzilla/utf8_runes/serial.h"        // `sz_rune_decode`, `sz_rune_encode`
#include "stringzilla/utf8_uncased_fold/serial.h" // `sz_unicode_fold_codepoint_`, the folded iterators

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Vocabulary

/** A state id no state ever takes, marking an empty slot, an absent child and an unplaced state alike. */
#define SZ_SUBSTRINGS_NO_STATE ((sz_u32_t)0xFFFFFFFFu)

/** Independent transition chains one counting walk keeps in flight, so their loads overlap. */
#define SZ_SUBSTRINGS_CHAINS (8)

/** Interior vacancies one double-array row may reject before it settles on the arena frontier instead. */
#define SZ_SUBSTRINGS_MAX_INTERIOR_PROBES (256)

/** Bytes of hot rows an automaton keeps when the caller names no count, which is the last-level cache it
 *  assumes; a caller that can measure its own host passes @c hot_states instead. */
#define SZ_SUBSTRINGS_HOT_BYTES_DEFAULT (4u << 20)

/**
 *  @brief Whether a walk's reports arrive in the order the haystack spells them, or in any order at all.
 *
 *  One chain finishes every match ending at a position before it steps to the next, so its reported ends
 *  never decrease - the order a leftmost cover drains its undecided starts in. Several chains stand at
 *  several positions of one haystack at once, which is what overlaps their loads, so nothing orders their
 *  reports against each other.
 */
typedef enum sz_substrings_report_order_t {
    /** Non-decreasing match end within one haystack; what a leftmost cover reads. */
    sz_substrings_ascending_ends_k = 0,
    /** Every match once, in no order at all; what a tally or a collector reads. */
    sz_substrings_unordered_k = 1,
} sz_substrings_report_order_t;

/** Whether a walk carries on, or the consumer has seen everything it wanted. */
typedef enum sz_substrings_walk_t {
    /** Keep stepping the haystack. */
    sz_substrings_continue_k = 0,
    /** Stop where the walk stands; a leftmost cover then clears whatever it left undecided. */
    sz_substrings_stop_k = 1,
} sz_substrings_walk_t;

/**
 *  @brief Receives one match, located inside the haystack the walk was handed.
 *  @param[in] context Whatever the caller bound; the walks never inspect it.
 *  @param[in] needle_index Which needle of the vocabulary matched.
 *  @param[in] byte_offset Where the match starts inside that haystack.
 *  @param[in] byte_length Haystack bytes the match spans.
 */
typedef sz_substrings_walk_t (*sz_substrings_reporter_t)(void *context, sz_size_t needle_index, sz_size_t byte_offset,
                                                         sz_size_t byte_length);

/**
 *  @brief Whether @p challenger outranks @p incumbent among matches sharing one start position.
 *  @param[in] incumbent A zero @c source_match_bytes means no match has claimed that start yet.
 *
 *  The only place either leftmost policy is consulted, and it runs once per discovered match rather than
 *  once per byte. Lengths compared here are source bytes, since a needle that folds shorter can still
 *  outspan a rival whose folded form is longer.
 */
SZ_HELPER_AUTO sz_bool_t sz_substrings_leftmost_wins(sz_substrings_pending_start_t challenger,
                                                     sz_substrings_pending_start_t incumbent,
                                                     sz_substrings_overlap_policy_t policy) {
    if (incumbent.source_match_bytes == 0) return sz_true_k;
    if (policy == sz_substrings_leftmost_longest_k && challenger.source_match_bytes != incumbent.source_match_bytes)
        return (sz_bool_t)(challenger.source_match_bytes > incumbent.source_match_bytes);
    return (sz_bool_t)(challenger.needle_index < incumbent.needle_index);
}

/**
 *  @brief How many starts a leftmost walk can hold undecided, given the longest match in the vocabulary.
 *  @param[in] max_source_match_bytes The longest match in @b source bytes, which is what the slots are keyed
 *             by; passing the folded bound instead would undersize the ring wherever a fold contracts.
 *
 *  A start can no longer be outbid once the walk is that many bytes past it, so that many slots suffice.
 *  Rounded up to a power of two, which turns the slot lookup into a mask rather than a runtime division.
 */
SZ_HELPER_AUTO sz_size_t sz_substrings_pending_starts_width(sz_size_t max_source_match_bytes) {
    sz_size_t const wanted = max_source_match_bytes > 1 ? max_source_match_bytes : 1;
    return sz_size_bit_ceil(wanted);
}

#pragma endregion Vocabulary

#pragma region Transition

/** One goto-completed row of the hot tier, one target per byte class. */
SZ_HELPER_AUTO sz_u32_t const *sz_substrings_hot_row(sz_substrings_engine_t const *engine, sz_u32_t state) {
    return engine->hot_rows + (sz_size_t)state * engine->classes_count;
}

/**
 *  @brief Advances one state by one byte. The single definition every CPU and GPU walk shares.
 *
 *  Hot states resolve in one load. Cold states probe the double array and, when the slot is owned by
 *  somebody else, hop to the failure link and retry the same byte. The root is total - every one of its
 *  slots resolves, self-looping where the trie has no edge - which is what terminates the retry loop.
 */
SZ_HELPER_AUTO sz_u32_t sz_substrings_step(sz_substrings_engine_t const *engine, sz_u32_t state, sz_u8_t byte) {
    for (;;) {
        if (state < engine->hot_count) {
            // The class depends on the byte alone, so its load sits off the chain of transitions.
            sz_u32_t const *const row = sz_substrings_hot_row(engine, state);
            return row[engine->byte_to_class[byte]];
        }
        // The probe index stays wide: `base + byte` can exceed the id ceiling on a slot this state does not
        // own, and narrowing first would wrap onto a slot `check` might accept. The arrays carry
        // `SZ_U8_MAX` slots of headroom past the last state for exactly this reach.
        sz_size_t const candidate = (sz_size_t)engine->base[state] + byte;
        if (engine->check[candidate] == state) return (sz_u32_t)candidate;
        if (state == engine->root) return engine->root;
        state = engine->fail[state];
    }
}

/** Whether any needle ends on @p state, answered from one bit rather than from the counts array. */
SZ_HELPER_AUTO sz_bool_t sz_substrings_accepts(sz_substrings_engine_t const *engine, sz_u32_t state) {
    return (sz_bool_t)((engine->accepts_words[state >> 5] >> (state & 31u)) & 1u);
}

/**
 *  @brief Advances @p state by one byte and reports how many needles end on the new state.
 *
 *  The pair every walk repeats: the transition, then the output count that decides whether the walk stops
 *  to enumerate matches.
 */
SZ_HELPER_AUTO sz_u32_t sz_substrings_step_counting(sz_substrings_engine_t const *engine, sz_u32_t *state,
                                                    sz_u8_t byte) {
    *state = sz_substrings_step(engine, *state, byte);
    return engine->outputs_counts[*state];
}

#pragma endregion Transition

#pragma region Folding Filter

/** Most folded bytes one source codepoint can produce: three runes of three bytes each. */
#define SZ_SUBSTRINGS_FOLDED_IMAGE_MAX (9)

/** One folded byte, and everything a walk needs about the codepoint it came from. */
typedef struct sz_substrings_folded_byte_t {
    /** The folded byte itself, which is what the automaton steps on. */
    sz_u8_t byte;
    /** Whether this byte ends a folded rune; a needle is valid UTF-8, so only there can a match end. */
    sz_bool_t rune_end;
    /** Whether this codepoint's fold leaves a folded byte at an offset no source byte owns. */
    sz_bool_t breaks_boundary;
    /** Whether the source byte began no well-formed codepoint, so the walk must resynchronize. */
    sz_bool_t malformed;
    /** Offset just past the source codepoint; every folded byte of it reports the same end. */
    sz_size_t codepoint_end;
    /** Folded bytes of this codepoint still to come, which a backward walk has to step over first. */
    sz_u8_t trailing;
    /** Folded bytes back to this codepoint's previous rune end, zero at its first. */
    sz_u8_t shift;
} sz_substrings_folded_byte_t;

/**
 *  @brief Streams a haystack as folded bytes, one source codepoint at a time and never into a buffer.
 *
 *  The automaton's alphabet is bytes while @c sz_utf8_folded_iter_t yields runes, so one codepoint's runes
 *  are drained and re-encoded into a nine-byte image, then handed out a byte at a time. The ASCII fast
 *  path, the expansion buffering and the one-byte malformed resynchronization all remain the iterator's.
 */
typedef struct sz_substrings_folded_cursor_t {
    /** The rune-level iterator this cursor re-encodes the output of. */
    sz_utf8_folded_iter_t runes;
    /** Where the haystack begins, so a codepoint's end can be reported as an offset. */
    sz_cptr_t origin;
    /** One codepoint's folded bytes, which is the widest image a fold can produce. */
    sz_u8_t image[SZ_SUBSTRINGS_FOLDED_IMAGE_MAX];
    /** Bit @c index marks the byte at @c index as ending a folded rune. */
    sz_u16_t rune_end_mask;
    /** Bytes the current image holds. */
    sz_u8_t image_length;
    /** Bytes of it already handed out. */
    sz_u8_t image_index;
    /** Where this codepoint's previous rune ended inside the image, zero before the first. */
    sz_u8_t previous_rune_end;
    /** Offset just past the source codepoint the image came from. */
    sz_size_t codepoint_end;
    /** Whether this codepoint's fold leaves a byte at an offset no source byte owns. */
    sz_bool_t breaks_boundary;
    /** Whether the source byte began no well-formed codepoint. */
    sz_bool_t malformed;
} sz_substrings_folded_cursor_t;

/** Binds a folding cursor over @p haystack, which it reads in place. */
SZ_HELPER_AUTO void sz_substrings_folded_cursor_init(sz_substrings_folded_cursor_t *cursor, sz_cptr_t haystack,
                                                     sz_size_t length) {
    sz_utf8_folded_iter_init_(&cursor->runes, haystack, length);
    cursor->origin = haystack;
    cursor->image_length = 0;
    cursor->image_index = 0;
    cursor->rune_end_mask = 0;
    cursor->previous_rune_end = 0;
    cursor->codepoint_end = 0;
    cursor->breaks_boundary = sz_false_k;
    cursor->malformed = sz_false_k;
}

/** Next folded byte, or @c sz_false_k once the haystack is spent. */
SZ_HELPER_AUTO sz_bool_t sz_substrings_folded_cursor_next(sz_substrings_folded_cursor_t *cursor,
                                                          sz_substrings_folded_byte_t *folded) {
    if (cursor->image_index == cursor->image_length) {
        // ASCII is its own codepoint and folds with one add, so it never decodes and never consults a table.
        if (cursor->runes.ptr < cursor->runes.end && (sz_u8_t)*cursor->runes.ptr < 0x80) {
            cursor->image[0] = (sz_u8_t)sz_ascii_fold_((sz_u8_t)*cursor->runes.ptr);
            cursor->image_length = 1;
            cursor->rune_end_mask = 1;
            cursor->runes.codepoint_begin = cursor->runes.ptr;
            cursor->runes.codepoint_length = 1;
            ++cursor->runes.ptr;
            cursor->breaks_boundary = sz_false_k;
            cursor->malformed = sz_false_k;
        }
        else {
            // Nothing under this lead byte folds, so the codepoint's own bytes already are its folded image:
            // no rune is assembled, no fold ladder walked, no re-encode. This is every CJK, Arabic, Hebrew,
            // Devanagari and emoji sequence, and it is what holds them at byte-exact speed.
            sz_rune_t rune;
            sz_rune_length_t verbatim = sz_rune_invalid_k;
            if (cursor->runes.ptr < cursor->runes.end && !sz_utf8_lead_may_fold_((sz_u8_t)*cursor->runes.ptr))
                verbatim = sz_rune_decode(cursor->runes.ptr, cursor->runes.end, &rune);

            if (verbatim != sz_rune_invalid_k) {
                sz_size_t index;
                for (index = 0; index < (sz_size_t)verbatim; ++index)
                    cursor->image[index] = (sz_u8_t)cursor->runes.ptr[index];
                cursor->image_length = (sz_u8_t)verbatim;
                cursor->rune_end_mask = (sz_u16_t)(1u << (verbatim - 1));
                cursor->runes.codepoint_begin = cursor->runes.ptr;
                cursor->runes.codepoint_length = (sz_size_t)verbatim;
                cursor->runes.ptr += verbatim;
                cursor->malformed = sz_false_k;
                cursor->breaks_boundary = sz_false_k;
            }
            else {
                sz_size_t rune_count = 0;
                if (!sz_utf8_folded_iter_next_(&cursor->runes, &rune)) return sz_false_k;

                // A malformed byte arrives tagged, outside the scalar range `sz_rune_encode` accepts, so it
                // passes through as the literal byte it was.
                cursor->malformed = (sz_bool_t)((rune & 0x80000000u) != 0);
                cursor->image_length = 0;
                cursor->rune_end_mask = 0;
                for (;;) {
                    if (cursor->malformed) cursor->image[cursor->image_length++] = (sz_u8_t)(rune & 0xFF);
                    else cursor->image_length += (sz_u8_t)sz_rune_encode(rune, cursor->image + cursor->image_length);
                    cursor->rune_end_mask |= (sz_u16_t)(1u << (cursor->image_length - 1));
                    ++rune_count;
                    // More pending runes belong to this codepoint; the iterator refills only once spent.
                    if (cursor->runes.pending_idx >= cursor->runes.pending_count) break;
                    sz_utf8_folded_iter_next_(&cursor->runes, &rune);
                }
                cursor->breaks_boundary = (sz_bool_t)(rune_count != 1 || (sz_size_t)cursor->image_length !=
                                                                             cursor->runes.codepoint_length);
            }
        }

        cursor->codepoint_end = (sz_size_t)(cursor->runes.codepoint_begin - cursor->origin) +
                                cursor->runes.codepoint_length;
        cursor->image_index = 0;
        cursor->previous_rune_end = 0;
    }

    {
        sz_u8_t const index = cursor->image_index++;
        folded->byte = cursor->image[index];
        folded->rune_end = (sz_bool_t)((cursor->rune_end_mask >> index) & 1u);
        folded->breaks_boundary = cursor->breaks_boundary;
        folded->malformed = cursor->malformed;
        folded->codepoint_end = cursor->codepoint_end;
        folded->trailing = (sz_u8_t)(cursor->image_length - cursor->image_index);
        folded->shift = (sz_u8_t)(cursor->image_index - cursor->previous_rune_end);
        if (folded->rune_end) {
            // A codepoint's first rune end has nothing before it to repeat, which a zero shift names.
            if (cursor->previous_rune_end == 0) folded->shift = 0;
            cursor->previous_rune_end = cursor->image_index;
        }
    }
    return sz_true_k;
}

/**
 *  @brief Where a match resolved by walking backwards from the codepoint it ends in landed.
 *
 *  @c repeats marks a span an earlier rune end of that same codepoint already reported: needle "s" ends at
 *  both runes the sharp S folds to, and both spans are the whole codepoint. Equal spans are what makes a
 *  repeat, not equal content.
 */
typedef struct sz_substrings_resolved_match_t {
    /** Where the match starts in the haystack's own bytes. */
    sz_size_t source_offset;
    /** Whether an earlier rune end of the same codepoint already reported this very span. */
    sz_bool_t repeats;
} sz_substrings_resolved_match_t;

/**
 *  @brief Resolves a match's source start, and whether it repeats an earlier rune end's span, in one pass.
 *  @param[in] source_end End of the codepoint the match ends in; every rune end of it shares this.
 *  @param[in] trailing Folded bytes of that codepoint sitting past the match's end.
 *  @param[in] shift Folded bytes back to the previous rune end of the same codepoint, zero at the first.
 *
 *  Reached only when the match starts at or before the last boundary-breaking codepoint, which is the only
 *  place either question is open. Two windows one length apart hold the same bytes exactly when the folded
 *  stream is periodic with that period, so the repeat test rides a @p shift -sized ring that one codepoint's
 *  image bounds, and shares the single backward walk with the start it recovers.
 */
SZ_HELPER_AUTO sz_substrings_resolved_match_t sz_substrings_resolve_match(sz_cptr_t haystack, sz_size_t source_end,
                                                                          sz_size_t trailing,
                                                                          sz_size_t folded_match_bytes,
                                                                          sz_size_t shift) {
    sz_utf8_folded_reverse_iter_t iterator;
    sz_substrings_resolved_match_t resolved;
    sz_u8_t pending[4];
    sz_u8_t ring[SZ_SUBSTRINGS_FOLDED_IMAGE_MAX];
    sz_size_t pending_count = 0, stepped, index;
    sz_cptr_t codepoint_begin = haystack + source_end;
    sz_cptr_t start_here = SZ_NULL, start_earlier = SZ_NULL;
    sz_size_t const wanted = shift != 0 ? folded_match_bytes + shift : folded_match_bytes;
    sz_bool_t periodic = (sz_bool_t)(shift != 0);

    sz_utf8_folded_reverse_iter_init_(&iterator, haystack, haystack + source_end);
    resolved.source_offset = source_end;
    resolved.repeats = sz_false_k;
    for (index = 0; index != SZ_SUBSTRINGS_FOLDED_IMAGE_MAX; ++index) ring[index] = 0;

    for (stepped = 0; stepped < trailing + wanted; ++stepped) {
        sz_u8_t byte;
        sz_size_t taken;
        if (pending_count == 0) {
            sz_rune_t image;
            if (!sz_utf8_folded_reverse_iter_prev_(&iterator, &image)) break;
            // A malformed byte arrives tagged above the encodable range, so it passes through as itself.
            if (image & 0x80000000u) pending[0] = (sz_u8_t)(image & 0xFF), pending_count = 1;
            else pending_count = (sz_size_t)sz_rune_encode(image, pending);
            // The cursor sits on the codepoint's own start as soon as any of its runes is yielded, which is
            // the outward snap a match beginning mid-expansion needs.
            codepoint_begin = iterator.ptr;
        }
        byte = pending[--pending_count];
        if (stepped < trailing) continue; // ? Folded bytes of the ending codepoint past the match's own end

        taken = stepped - trailing + 1;
        if (shift != 0) {
            if (taken > shift && ring[(taken - shift) % SZ_SUBSTRINGS_FOLDED_IMAGE_MAX] != byte) periodic = sz_false_k;
            ring[taken % SZ_SUBSTRINGS_FOLDED_IMAGE_MAX] = byte;
        }
        if (taken == folded_match_bytes) start_here = codepoint_begin;
        if (taken == wanted) start_earlier = codepoint_begin;
        if (!periodic && taken >= folded_match_bytes) break;
    }

    if (start_here != SZ_NULL) resolved.source_offset = (sz_size_t)(start_here - haystack);
    resolved.repeats = (sz_bool_t)(periodic && start_here != SZ_NULL && start_here == start_earlier);
    return resolved;
}

/**
 *  @brief Resolves the source span of one output at the rune end @p step stands on.
 *  @param[in] folded Folded bytes consumed so far, the ending offset the output's length is taken back from.
 *  @param[in] last_break_folded_end Folded end of the last codepoint whose fold broke a boundary.
 *
 *  A match starting at or after the last break lies where folded and source offsets still agree, so its
 *  start is one subtraction; anything earlier pays the backward walk.
 */
SZ_HELPER_AUTO sz_substrings_resolved_match_t sz_substrings_folded_span(sz_cptr_t haystack,
                                                                        sz_substrings_folded_byte_t const *step,
                                                                        sz_size_t folded,
                                                                        sz_size_t last_break_folded_end,
                                                                        sz_size_t folded_match_bytes) {
    if (folded - folded_match_bytes >= last_break_folded_end) {
        sz_substrings_resolved_match_t resolved;
        resolved.source_offset = step->codepoint_end - folded_match_bytes;
        resolved.repeats = sz_false_k;
        return resolved;
    }
    return sz_substrings_resolve_match(haystack, step->codepoint_end, step->trailing, folded_match_bytes, step->shift);
}

#pragma endregion Folding Filter

#pragma region Construction

/**
 *  @brief One trie state under construction: its place in its parent's sibling list, and what ends on it.
 *
 *  A tree gives every state exactly one incoming edge, so the byte of that edge is a field of the state
 *  rather than of a separate edge record - which is what lets the whole trie live in one growable array
 *  with no index beside it.
 */
typedef struct sz_substrings_trie_node_t {
    /** Lowest-numbered child, or @ref SZ_SUBSTRINGS_NO_STATE. */
    sz_u32_t first_child;
    /** Next child of this node's own parent, or @ref SZ_SUBSTRINGS_NO_STATE. */
    sz_u32_t next_sibling;
    /** Needle ending exactly here, heading a list threaded through @c needle_next. */
    sz_u32_t output_head;
    /** Needles ending exactly here, before failure-chain inheritance. */
    sz_u32_t output_own_count;
    /** Matches ending here once the failure chain has merged them in. */
    sz_u32_t output_total_count;
    /** The failure state, always strictly shallower, so depth order finishes it first. */
    sz_u32_t failure;
    /** Published double-array slot, or @ref SZ_SUBSTRINGS_NO_STATE before packing places it. */
    sz_u32_t published;
    /** Byte on the edge into this state; the root's is never read. */
    sz_u8_t parent_byte;
    /** Into the published output pool: own matches followed by the failure state's whole run. */
    sz_size_t output_total_offset;
} sz_substrings_trie_node_t;

/** The construction state one build threads through its phases, none of which survives the call. */
typedef struct sz_substrings_builder_t {
    /** The trie, growing by doubling as insertion mints states. */
    sz_substrings_trie_node_t *nodes;
    /** States @c nodes can hold before it has to grow. */
    sz_size_t nodes_capacity;
    /** States the trie holds. */
    sz_size_t nodes_count;

    /** Next needle in the list of needles ending on one state, @b [needles_count]. */
    sz_u32_t *needle_next;
    /** Folded bytes each needle spans, @b [needles_count]. */
    sz_u32_t *needle_folded_bytes;
    /** One needle's canonical folded bytes, reused across insertions; uncased mode only. */
    sz_u8_t *fold_scratch;
    /** Bytes @c fold_scratch holds. */
    sz_size_t fold_scratch_bytes;

    /** States in depth-band, out-degree-descending order: the input to the published numbering. */
    sz_u32_t *order;
    /** Scratch the band sort permutes through, since a band is not a contiguous id range. */
    sz_u32_t *order_scratch;
    /** The root's goto-completed row, so a failure chase that falls all the way back ends in one lookup. */
    sz_u32_t *root_row;

    /** Double-array arena: transition target for @c state on @c byte is @c base[state] @c + @c byte. */
    sz_u32_t *base;
    /** Double-array arena: owner of each slot. */
    sz_u32_t *check;
    /** Which trie state each published slot names, or @ref SZ_SUBSTRINGS_NO_STATE. */
    sz_u32_t *state_of_slot;
    /** One bit per slot; the only record of what the packing search has claimed. */
    sz_u64_t *occupied;
    /** Slots the arena can hold. */
    sz_size_t slots_capacity;
    /** Lowest slot that could still be free; packing only fills forward, so it never moves back. */
    sz_size_t lowest_free_cursor;
    /** One past the highest claimed slot, so every slot at or above it is free by construction. */
    sz_size_t arena_frontier;

    /** Needles inserted, which is also the next needle's own index. */
    sz_size_t needles_count;
    /** Worst-case haystack span of one match; a fold contracting three bytes into one widens it. */
    sz_u32_t max_source_match_bytes;
    /** The mirror bound, in haystack bytes, of the shortest match any needle can make. */
    sz_u32_t min_source_match_bytes;
    /** Most outputs any one state carries once failure links have merged them. */
    sz_u32_t max_outputs_per_state;
    /** Whether this vocabulary matches byte-exact or case-folded. */
    sz_substrings_case_sensitivity_t case_sensitivity;
    /** Each byte's hot-row column, derived from the edge labels once the trie is complete. */
    sz_u8_t byte_to_class[SZ_U8_MAX + 1];
    /** Columns of a hot row. */
    sz_size_t classes_count;
    /** States kept in the dense hot rows. */
    sz_size_t hot_count;
    /** The allocator every buffer above came from, and the one they go back to. */
    sz_memory_allocator_t *alloc;
} sz_substrings_builder_t;

/** Hands every buffer the build took back to its allocator, leaving the builder empty. */
SZ_API_COMPTIME void sz_substrings_builder_free_(sz_substrings_builder_t *builder) {
    sz_memory_allocator_t *const alloc = builder->alloc;
    if (builder->nodes)
        alloc->free(builder->nodes, builder->nodes_capacity * sizeof(sz_substrings_trie_node_t), alloc->handle);
    if (builder->needle_next)
        alloc->free(builder->needle_next, builder->needles_count * sizeof(sz_u32_t), alloc->handle);
    if (builder->needle_folded_bytes)
        alloc->free(builder->needle_folded_bytes, builder->needles_count * sizeof(sz_u32_t), alloc->handle);
    if (builder->fold_scratch) alloc->free(builder->fold_scratch, builder->fold_scratch_bytes, alloc->handle);
    if (builder->order) alloc->free(builder->order, builder->nodes_count * sizeof(sz_u32_t), alloc->handle);
    if (builder->order_scratch)
        alloc->free(builder->order_scratch, builder->nodes_count * sizeof(sz_u32_t), alloc->handle);
    if (builder->root_row) alloc->free(builder->root_row, (SZ_U8_MAX + 1) * sizeof(sz_u32_t), alloc->handle);
    if (builder->base) alloc->free(builder->base, builder->slots_capacity * sizeof(sz_u32_t), alloc->handle);
    if (builder->check) alloc->free(builder->check, builder->slots_capacity * sizeof(sz_u32_t), alloc->handle);
    if (builder->state_of_slot)
        alloc->free(builder->state_of_slot, builder->slots_capacity * sizeof(sz_u32_t), alloc->handle);
    if (builder->occupied)
        alloc->free(builder->occupied, ((builder->slots_capacity >> 6) + 8) * sizeof(sz_u64_t), alloc->handle);
    builder->nodes = SZ_NULL, builder->needle_next = SZ_NULL, builder->needle_folded_bytes = SZ_NULL;
    builder->fold_scratch = SZ_NULL, builder->order = SZ_NULL, builder->order_scratch = SZ_NULL;
    builder->root_row = SZ_NULL, builder->base = SZ_NULL, builder->check = SZ_NULL;
    builder->state_of_slot = SZ_NULL, builder->occupied = SZ_NULL;
}

/** Grows the trie to hold one more state, doubling so insertion stays amortized linear. */
SZ_API_COMPTIME sz_status_t sz_substrings_builder_grow_nodes_(sz_substrings_builder_t *builder) {
    sz_memory_allocator_t *const alloc = builder->alloc;
    sz_size_t const old_capacity = builder->nodes_capacity;
    sz_size_t const new_capacity = old_capacity ? old_capacity * 2 : 1024;
    sz_substrings_trie_node_t *grown;
    if (builder->nodes_count < old_capacity) return sz_success_k;
    if (new_capacity >= (sz_size_t)SZ_SUBSTRINGS_NO_STATE) return sz_overflow_risk_k;

    grown = (sz_substrings_trie_node_t *)alloc->allocate(new_capacity * sizeof(sz_substrings_trie_node_t),
                                                         alloc->handle);
    if (!grown) return sz_bad_alloc_k;
    if (builder->nodes) {
        sz_copy((sz_ptr_t)grown, (sz_cptr_t)builder->nodes, old_capacity * sizeof(sz_substrings_trie_node_t));
        alloc->free(builder->nodes, old_capacity * sizeof(sz_substrings_trie_node_t), alloc->handle);
    }
    builder->nodes = grown;
    builder->nodes_capacity = new_capacity;
    return sz_success_k;
}

/** Mints one fresh state with no children, no outputs and no failure link yet. */
SZ_API_COMPTIME sz_status_t sz_substrings_builder_mint_(sz_substrings_builder_t *builder, sz_u32_t *minted) {
    sz_substrings_trie_node_t *node;
    sz_status_t const grown = sz_substrings_builder_grow_nodes_(builder);
    if (grown != sz_success_k) return grown;
    node = builder->nodes + builder->nodes_count;
    node->first_child = SZ_SUBSTRINGS_NO_STATE;
    node->next_sibling = SZ_SUBSTRINGS_NO_STATE;
    node->output_head = SZ_SUBSTRINGS_NO_STATE;
    node->output_own_count = 0;
    node->output_total_count = 0;
    node->failure = 0;
    node->published = SZ_SUBSTRINGS_NO_STATE;
    node->parent_byte = 0;
    node->output_total_offset = 0;
    *minted = (sz_u32_t)builder->nodes_count;
    ++builder->nodes_count;
    return sz_success_k;
}

/** Child of @p parent on @p byte among its literal edges, or @ref SZ_SUBSTRINGS_NO_STATE if none. */
SZ_API_COMPTIME sz_u32_t sz_substrings_builder_child_(sz_substrings_builder_t const *builder, sz_u32_t parent,
                                                      sz_u8_t byte) {
    sz_u32_t child = builder->nodes[parent].first_child;
    while (child != SZ_SUBSTRINGS_NO_STATE) {
        if (builder->nodes[child].parent_byte == byte) return child;
        child = builder->nodes[child].next_sibling;
    }
    return SZ_SUBSTRINGS_NO_STATE;
}

/** Follows @p parent's @p byte edge, minting a state and threading it onto the sibling list when missing. */
SZ_API_COMPTIME sz_status_t sz_substrings_builder_follow_(sz_substrings_builder_t *builder, sz_u32_t parent,
                                                          sz_u8_t byte, sz_u32_t *child) {
    sz_u32_t minted;
    sz_status_t status;
    sz_u32_t const existing = sz_substrings_builder_child_(builder, parent, byte);
    if (existing != SZ_SUBSTRINGS_NO_STATE) {
        *child = existing;
        return sz_success_k;
    }
    status = sz_substrings_builder_mint_(builder, &minted);
    if (status != sz_success_k) return status;
    builder->nodes[minted].parent_byte = byte;
    builder->nodes[minted].next_sibling = builder->nodes[parent].first_child;
    builder->nodes[parent].first_child = minted;
    *child = minted;
    return sz_success_k;
}

/**
 *  @brief Records that @p needle_index ends on @p state, spanning @p folded_match_bytes folded bytes.
 *
 *  One folded byte can stand for up to @c sz_utf8_fold_max_contraction_k source bytes, and one source byte
 *  for up to @c sz_utf8_fold_max_expansion_k folded ones, so a folded length brackets rather than fixes the
 *  source span. Cased needles fold to themselves, so their bounds stay exact.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_builder_add_output_(sz_substrings_builder_t *builder, sz_u32_t state,
                                                              sz_u32_t needle_index, sz_size_t folded_match_bytes) {
    sz_size_t const contraction = builder->case_sensitivity == sz_substrings_uncased_k
                                      ? (sz_size_t)sz_utf8_fold_max_contraction_k
                                      : 1;
    sz_size_t const expansion = builder->case_sensitivity == sz_substrings_uncased_k
                                    ? (sz_size_t)sz_utf8_fold_max_expansion_k
                                    : 1;
    sz_size_t const source_ceiling = folded_match_bytes * contraction;
    sz_size_t const source_floor = (folded_match_bytes + expansion - 1) / expansion;
    if (folded_match_bytes > (sz_size_t)SZ_SUBSTRINGS_NO_STATE) return sz_overflow_risk_k;
    // A folded walk snaps both ends of a match outward to whole codepoints, so a reported source span
    // reaches one rune past this ceiling. Refusing that much earlier keeps every staged length in 32 bits.
    if (source_ceiling + (sz_size_t)sz_rune_4bytes_k > (sz_size_t)SZ_SUBSTRINGS_NO_STATE) return sz_overflow_risk_k;

    builder->needle_folded_bytes[needle_index] = (sz_u32_t)folded_match_bytes;
    builder->needle_next[needle_index] = builder->nodes[state].output_head;
    builder->nodes[state].output_head = needle_index;
    ++builder->nodes[state].output_own_count;

    builder->max_source_match_bytes = sz_max_of_two(builder->max_source_match_bytes, (sz_u32_t)source_ceiling);
    builder->min_source_match_bytes = builder->min_source_match_bytes
                                          ? sz_min_of_two(builder->min_source_match_bytes, (sz_u32_t)source_floor)
                                          : (sz_u32_t)source_floor;
    return sz_success_k;
}

/** Byte-exact trie insertion: a follow-or-create walk, one state per byte consumed. */
SZ_API_COMPTIME sz_status_t sz_substrings_builder_insert_cased_(sz_substrings_builder_t *builder, sz_u8_t const *bytes,
                                                                sz_size_t length, sz_u32_t needle_index) {
    sz_u32_t state = 0;
    sz_size_t offset;
    for (offset = 0; offset < length; ++offset) {
        sz_status_t const status = sz_substrings_builder_follow_(builder, state, bytes[offset], &state);
        if (status != sz_success_k) return status;
    }
    return sz_substrings_builder_add_output_(builder, state, needle_index, length);
}

/**
 *  @brief Case-folded trie insertion: fold the needle once, then insert those bytes literally.
 *
 *  The haystack is folded as the walk consumes it, so both sides meet in one canonical byte stream and the
 *  trie has nothing case-specific left in it - which is what keeps it a tree with single-valued failure
 *  links. A needle that folds to the same bytes as an earlier one simply shares its path.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_builder_insert_uncased_(sz_substrings_builder_t *builder, sz_cptr_t needle,
                                                                  sz_size_t length, sz_u32_t needle_index) {
    sz_cptr_t const needle_end = needle + length;
    sz_cptr_t cursor = needle;
    sz_size_t written = 0;
    while (cursor != needle_end) {
        sz_rune_t rune, images[3];
        sz_size_t runes, index;
        sz_rune_length_t const consumed = sz_rune_decode(cursor, needle_end, &rune);
        // Malformed UTF-8 is refused outright: the walk resets to the root on a malformed haystack byte, so
        // a needle carrying one could never match, and accepting it would only hide the caller's mistake.
        if (consumed == sz_rune_invalid_k) return sz_invalid_utf8_k;
        runes = sz_unicode_fold_codepoint_(rune, images);
        for (index = 0; index < runes; ++index)
            written += (sz_size_t)sz_rune_encode(images[index], builder->fold_scratch + written);
        cursor += consumed;
    }
    if (written == 0) return sz_success_k;
    return sz_substrings_builder_insert_cased_(builder, builder->fold_scratch, written, needle_index);
}

/**
 *  @brief Reorders one depth band of @c order by out-degree descending, so shallow high-fan-out states -
 *         the ones text keeps returning to - land in the hot tier regardless of vocabulary content.
 *  @note A band is the position range @c [band_first, @c band_last) within @c order, not an id range.
 */
SZ_API_COMPTIME void sz_substrings_builder_order_band_(sz_substrings_builder_t *builder, sz_size_t band_first,
                                                       sz_size_t band_last) {
    sz_u32_t histogram[SZ_U8_MAX + 2]; // ? Out-degrees span zero through `SZ_U8_MAX + 1`
    sz_u32_t running = (sz_u32_t)band_first;
    sz_size_t index, degree;
    if (band_last - band_first < 2) return;

    for (degree = 0; degree <= SZ_U8_MAX + 1; ++degree) histogram[degree] = 0;
    for (index = band_first; index < band_last; ++index) {
        sz_u32_t child = builder->nodes[builder->order[index]].first_child;
        sz_size_t out_degree = 0;
        for (; child != SZ_SUBSTRINGS_NO_STATE; child = builder->nodes[child].next_sibling) ++out_degree;
        ++histogram[out_degree];
    }
    // Suffix-summed, so the highest degree claims the lowest positions in the band.
    for (degree = SZ_U8_MAX + 2; degree-- > 0;) {
        sz_u32_t const count = histogram[degree];
        histogram[degree] = running;
        running += count;
    }
    // Scattered through scratch rather than in place: the source is the band itself, so writing a position
    // before reading it would overwrite a state still waiting to be placed.
    for (index = band_first; index < band_last; ++index) builder->order_scratch[index] = builder->order[index];
    for (index = band_first; index < band_last; ++index) {
        sz_u32_t const state = builder->order_scratch[index];
        sz_u32_t child = builder->nodes[state].first_child;
        sz_size_t out_degree = 0;
        for (; child != SZ_SUBSTRINGS_NO_STATE; child = builder->nodes[child].next_sibling) ++out_degree;
        builder->order[histogram[out_degree]++] = state;
    }
}

/** Goto-completed target for @p state on @p byte; the root answers from its dense row. */
SZ_API_COMPTIME sz_u32_t sz_substrings_builder_chase_(sz_substrings_builder_t const *builder, sz_u32_t state,
                                                      sz_u8_t byte) {
    sz_u32_t current = state;
    for (;;) {
        sz_u32_t child;
        if (current == 0) return builder->root_row[byte];
        child = sz_substrings_builder_child_(builder, current, byte);
        if (child != SZ_SUBSTRINGS_NO_STATE) return child;
        current = builder->nodes[current].failure;
    }
}

/**
 *  @brief Assigns every state's failure link and lays @c order out depth band by depth band.
 *
 *  The classic Aho-Corasick construction: a child's failure link is found by chasing its parent's, and a
 *  failure link is always strictly shallower, so one shallow-to-deep pass finishes with no fixpoint. The
 *  trie is a tree, so each state is reached by exactly one edge and this visits each exactly once.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_builder_link_failures_(sz_substrings_builder_t *builder) {
    sz_memory_allocator_t *const alloc = builder->alloc;
    sz_size_t const states = builder->nodes_count;
    sz_size_t band_first, band_last, discovered, byte;
    sz_u32_t child;

    builder->order = (sz_u32_t *)alloc->allocate(states * sizeof(sz_u32_t), alloc->handle);
    builder->order_scratch = (sz_u32_t *)alloc->allocate(states * sizeof(sz_u32_t), alloc->handle);
    builder->root_row = (sz_u32_t *)alloc->allocate((SZ_U8_MAX + 1) * sizeof(sz_u32_t), alloc->handle);
    if (!builder->order || !builder->order_scratch || !builder->root_row) return sz_bad_alloc_k;

    // Every chase ends at the root's dense row, so it has to exist before the first of them.
    for (byte = 0; byte != SZ_U8_MAX + 1; ++byte) builder->root_row[byte] = 0;
    for (child = builder->nodes[0].first_child; child != SZ_SUBSTRINGS_NO_STATE;
         child = builder->nodes[child].next_sibling)
        builder->root_row[builder->nodes[child].parent_byte] = child;

    builder->order[0] = 0; // ? The root fails to itself, which minting already defaulted to.
    discovered = 1;
    for (band_first = 0, band_last = 1; band_first != band_last;) {
        sz_size_t index;
        for (index = band_first; index < band_last; ++index) {
            sz_u32_t const parent = builder->order[index];
            sz_u32_t const parent_failure = builder->nodes[parent].failure;
            for (child = builder->nodes[parent].first_child; child != SZ_SUBSTRINGS_NO_STATE;
                 child = builder->nodes[child].next_sibling) {
                // A depth-one state fails to the root; anything deeper chases its parent's failure link.
                builder->nodes[child].failure = parent == 0
                                                    ? 0
                                                    : sz_substrings_builder_chase_(builder, parent_failure,
                                                                                   builder->nodes[child].parent_byte);
                builder->order[discovered++] = child;
            }
        }
        sz_substrings_builder_order_band_(builder, band_first, band_last);
        band_first = band_last, band_last = discovered;
    }
    // Every state is reachable from the root by construction, so a short walk means the trie is malformed.
    sz_assert_(discovered == states && "A tree trie reaches every state exactly once");
    sz_unused_(states);
    return sz_success_k;
}

/** Grows every slot-indexed array to hold @p minimum slots, clearing whatever the growth exposed. */
SZ_API_COMPTIME sz_status_t sz_substrings_builder_reserve_slots_(sz_substrings_builder_t *builder, sz_size_t minimum) {
    sz_memory_allocator_t *const alloc = builder->alloc;
    sz_size_t const old_capacity = builder->slots_capacity;
    sz_size_t const new_capacity = sz_size_bit_ceil(minimum);
    sz_size_t const old_words = old_capacity ? (old_capacity >> 6) + 8 : 0;
    sz_size_t const new_words = (new_capacity >> 6) + 8;
    sz_u32_t *base, *check, *state_of_slot;
    sz_u64_t *occupied;
    sz_size_t slot, word;
    if (minimum <= old_capacity) return sz_success_k;
    if (new_capacity >= (sz_size_t)SZ_SUBSTRINGS_NO_STATE) return sz_overflow_risk_k;

    base = (sz_u32_t *)alloc->allocate(new_capacity * sizeof(sz_u32_t), alloc->handle);
    check = (sz_u32_t *)alloc->allocate(new_capacity * sizeof(sz_u32_t), alloc->handle);
    state_of_slot = (sz_u32_t *)alloc->allocate(new_capacity * sizeof(sz_u32_t), alloc->handle);
    // Four words of headroom past the capacity, so a 256-bit feasibility window never reads off the end.
    occupied = (sz_u64_t *)alloc->allocate(new_words * sizeof(sz_u64_t), alloc->handle);
    if (!base || !check || !state_of_slot || !occupied) {
        if (base) alloc->free(base, new_capacity * sizeof(sz_u32_t), alloc->handle);
        if (check) alloc->free(check, new_capacity * sizeof(sz_u32_t), alloc->handle);
        if (state_of_slot) alloc->free(state_of_slot, new_capacity * sizeof(sz_u32_t), alloc->handle);
        if (occupied) alloc->free(occupied, new_words * sizeof(sz_u64_t), alloc->handle);
        return sz_bad_alloc_k;
    }

    if (old_capacity) {
        sz_copy((sz_ptr_t)base, (sz_cptr_t)builder->base, old_capacity * sizeof(sz_u32_t));
        sz_copy((sz_ptr_t)check, (sz_cptr_t)builder->check, old_capacity * sizeof(sz_u32_t));
        sz_copy((sz_ptr_t)state_of_slot, (sz_cptr_t)builder->state_of_slot, old_capacity * sizeof(sz_u32_t));
        sz_copy((sz_ptr_t)occupied, (sz_cptr_t)builder->occupied, old_words * sizeof(sz_u64_t));
        alloc->free(builder->base, old_capacity * sizeof(sz_u32_t), alloc->handle);
        alloc->free(builder->check, old_capacity * sizeof(sz_u32_t), alloc->handle);
        alloc->free(builder->state_of_slot, old_capacity * sizeof(sz_u32_t), alloc->handle);
        alloc->free(builder->occupied, old_words * sizeof(sz_u64_t), alloc->handle);
    }
    // The bitmap is the only record of what is claimed, so a stale set bit would hide a free slot.
    for (word = old_words; word < new_words; ++word) occupied[word] = 0;
    for (slot = old_capacity; slot < new_capacity; ++slot)
        base[slot] = 0, check[slot] = SZ_SUBSTRINGS_NO_STATE, state_of_slot[slot] = SZ_SUBSTRINGS_NO_STATE;

    builder->base = base, builder->check = check, builder->state_of_slot = state_of_slot;
    builder->occupied = occupied, builder->slots_capacity = new_capacity;
    return sz_success_k;
}

/** Marks @p slot taken and carries the frontier past it, which is what keeps the frontier a valid bound. */
SZ_API_COMPTIME void sz_substrings_builder_claim_(sz_substrings_builder_t *builder, sz_size_t slot) {
    builder->occupied[slot >> 6] |= (sz_u64_t)1 << (slot & 63);
    builder->arena_frontier = sz_max_of_two(builder->arena_frontier, slot + 1);
}

/**
 *  @brief First free slot at or after @p from, growing the arena when the search runs off the end.
 *
 *  Packing only ever fills forward, so @c lowest_free_cursor never moves back and the whole walk across one
 *  build is amortized linear in the slot count - but only while every packing phase advances it.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_builder_next_free_(sz_substrings_builder_t *builder, sz_size_t from,
                                                             sz_size_t *found) {
    sz_size_t word;
    for (word = from >> 6;; ++word) {
        sz_u64_t vacancies;
        if ((word << 6) >= builder->slots_capacity) {
            sz_size_t const wanted = builder->slots_capacity ? builder->slots_capacity * 2 : 1024;
            sz_status_t const grown = sz_substrings_builder_reserve_slots_(builder, wanted);
            if (grown != sz_success_k) return grown;
        }
        vacancies = ~builder->occupied[word];
        // Slots before `from` are not candidates, even when the word says they are free.
        if (word == (from >> 6)) vacancies &= ~(sz_u64_t)0 << (from & 63);
        if (vacancies == 0) continue;
        *found = (word << 6) + (sz_size_t)sz_u64_ctz(vacancies);
        return sz_success_k;
    }
}

/** Whether every byte set in @p wanted lands on a currently-free slot at @p base. */
SZ_API_COMPTIME sz_bool_t sz_substrings_builder_row_fits_(sz_substrings_builder_t const *builder, sz_size_t base,
                                                          sz_byteset_t const *wanted) {
    sz_size_t const word = base >> 6, shift = base & 63;
    sz_size_t quarter;
    for (quarter = 0; quarter < 4; ++quarter) {
        sz_u64_t const low = builder->occupied[word + quarter];
        sz_u64_t const high = builder->occupied[word + quarter + 1];
        sz_u64_t const window = shift == 0 ? low : (low >> shift) | (high << (64 - shift));
        if (window & wanted->_u64s[quarter]) return sz_false_k;
    }
    return sz_true_k;
}

/**
 *  @brief Places a hot parent's children on whatever free slots come next.
 *
 *  A hot row addresses every child unconditionally, so nothing has to verify ownership through @c check for
 *  them and they need no shared base. The slot stays unowned: a cold state's probe can land on it, and an
 *  owner would answer that probe as an edge that does not exist.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_builder_pack_hot_children_(sz_substrings_builder_t *builder,
                                                                     sz_u32_t parent) {
    sz_u32_t child;
    for (child = builder->nodes[parent].first_child; child != SZ_SUBSTRINGS_NO_STATE;
         child = builder->nodes[child].next_sibling) {
        sz_size_t assigned;
        sz_status_t status;
        if (builder->nodes[child].published != SZ_SUBSTRINGS_NO_STATE) continue;
        status = sz_substrings_builder_next_free_(builder, builder->lowest_free_cursor, &assigned);
        if (status != sz_success_k) return status;
        sz_substrings_builder_claim_(builder, assigned);
        builder->state_of_slot[assigned] = child;
        builder->nodes[child].published = (sz_u32_t)assigned;
        builder->lowest_free_cursor = assigned + 1;
    }
    return sz_success_k;
}

/**
 *  @brief Places a cold parent's children on one shared base, so @c base[parent] @c + @c byte addresses each.
 *
 *  Candidates are scanned out of the occupancy bitmap anchored on the parent's smallest child byte, and the
 *  whole row is tested at once rather than probed child by child. After
 *  @ref SZ_SUBSTRINGS_MAX_INTERIOR_PROBES rejections the row settles on the arena frontier instead: the
 *  vacancies a packed arena leaves behind are mostly singletons no multi-byte row can ever cover, and a
 *  search that keeps re-walking them is quadratic in the states it places rather than linear.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_builder_pack_cold_children_(sz_substrings_builder_t *builder,
                                                                      sz_u32_t parent) {
    sz_u32_t child_of_byte[SZ_U8_MAX + 1];
    sz_byteset_t child_mask;
    sz_u8_t anchor_byte = 0;
    sz_size_t first_free, candidate, rejected = 0, quarter;
    sz_status_t status;
    sz_u32_t child = builder->nodes[parent].first_child;
    if (child == SZ_SUBSTRINGS_NO_STATE) return sz_success_k;

    sz_byteset_init(&child_mask);
    for (; child != SZ_SUBSTRINGS_NO_STATE; child = builder->nodes[child].next_sibling) {
        sz_byteset_add_u8(&child_mask, builder->nodes[child].parent_byte);
        child_of_byte[builder->nodes[child].parent_byte] = child;
    }
    for (quarter = 0; quarter < 4; ++quarter)
        if (child_mask._u64s[quarter]) {
            anchor_byte = (sz_u8_t)(quarter * 64 + sz_u64_ctz(child_mask._u64s[quarter]));
            break;
        }

    // The arena's lowest vacancy, and the cursor this call publishes: a rejected row leaves it free.
    first_free = builder->lowest_free_cursor;
    status = sz_substrings_builder_next_free_(builder, first_free, &first_free);
    if (status != sz_success_k) return status;

    for (candidate = first_free;;) {
        sz_size_t candidate_base;
        status = sz_substrings_builder_next_free_(builder, candidate, &candidate);
        if (status != sz_success_k) return status;
        if (candidate < anchor_byte) {
            ++candidate;
            continue;
        }
        candidate_base = candidate - anchor_byte;
        status = sz_substrings_builder_reserve_slots_(builder, candidate_base + SZ_U8_MAX + 1);
        if (status != sz_success_k) return status;
        if (!sz_substrings_builder_row_fits_(builder, candidate_base, &child_mask)) {
            // Landing on the frontier itself, rather than an anchor byte past it, is what keeps the fallback
            // free: every slot from there up is unclaimed, so the row strands nothing behind it.
            if (++rejected >= SZ_SUBSTRINGS_MAX_INTERIOR_PROBES) candidate = builder->arena_frontier;
            else ++candidate;
            continue;
        }

        builder->base[builder->nodes[parent].published] = (sz_u32_t)candidate_base;
        for (quarter = 0; quarter < 4; ++quarter) {
            sz_u64_t bits;
            for (bits = child_mask._u64s[quarter]; bits; bits &= bits - 1) {
                sz_u8_t const byte = (sz_u8_t)(quarter * 64 + sz_u64_ctz(bits));
                sz_size_t const slot = candidate_base + byte;
                sz_u32_t const placed = child_of_byte[byte];
                sz_substrings_builder_claim_(builder, slot);
                builder->check[slot] = builder->nodes[parent].published;
                builder->state_of_slot[slot] = placed;
                // A tree gives every state exactly one parent edge, so no state is ever claimed twice.
                sz_assert_(builder->nodes[placed].published == SZ_SUBSTRINGS_NO_STATE && "One parent edge per state");
                builder->nodes[placed].published = (sz_u32_t)slot;
            }
        }
        // `candidate` is the lowest slot just claimed, so only a row landing on `first_free` frees it.
        builder->lowest_free_cursor = first_free == candidate ? first_free + 1 : first_free;
        return sz_success_k;
    }
}

/**
 *  @brief Assigns every live state its published id: the hot tier takes @c [0, @c hot_count) in frequency
 *         order, and the rest pack into the double array in the same depth-ascending order.
 *
 *  Every state has exactly one parent edge, so every state is placed exactly once and each published slot
 *  names a distinct state.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_builder_pack_(sz_substrings_builder_t *builder) {
    sz_size_t index;
    sz_status_t status;
    builder->lowest_free_cursor = builder->hot_count; // ? Hot states own the low ids outright.
    status = sz_substrings_builder_reserve_slots_(builder, builder->hot_count + SZ_U8_MAX + 1);
    if (status != sz_success_k) return status;

    for (index = 0; index < builder->hot_count; ++index) {
        sz_u32_t const state = builder->order[index];
        builder->nodes[state].published = (sz_u32_t)index;
        builder->state_of_slot[index] = state;
        sz_substrings_builder_claim_(builder, index);
    }

    if (builder->hot_count == 0) { // ? The root has no parent to assign it a cold id.
        sz_size_t assigned;
        status = sz_substrings_builder_next_free_(builder, builder->lowest_free_cursor, &assigned);
        if (status != sz_success_k) return status;
        sz_substrings_builder_claim_(builder, assigned);
        builder->state_of_slot[assigned] = 0;
        builder->nodes[0].published = (sz_u32_t)assigned;
        builder->lowest_free_cursor = assigned + 1;
    }

    for (index = 0; index < builder->nodes_count; ++index) {
        sz_u32_t const parent = builder->order[index];
        status = index < builder->hot_count ? sz_substrings_builder_pack_hot_children_(builder, parent)
                                            : sz_substrings_builder_pack_cold_children_(builder, parent);
        if (status != sz_success_k) return status;
    }
    return sz_success_k;
}

/**
 *  @brief Sizes every state's merged output run, and reports the pool those runs need.
 *
 *  A state's matches are its own plus its failure state's complete set, and depth-band order finishes the
 *  failure state first, so one pass settles every offset.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_builder_size_outputs_(sz_substrings_builder_t *builder,
                                                                sz_size_t *outputs_total) {
    sz_size_t running = 0, index;
    for (index = 0; index < builder->nodes_count; ++index) {
        sz_u32_t const state = builder->order[index];
        sz_substrings_trie_node_t *const node = builder->nodes + state;
        sz_size_t const inherited = state == 0 ? 0 : (sz_size_t)builder->nodes[node->failure].output_total_count;
        sz_size_t const total = (sz_size_t)node->output_own_count + inherited;
        if (total > (sz_size_t)SZ_SUBSTRINGS_NO_STATE) return sz_overflow_risk_k;
        node->output_total_count = (sz_u32_t)total;
        node->output_total_offset = running;
        running += total;
        builder->max_outputs_per_state = sz_max_of_two(builder->max_outputs_per_state, node->output_total_count);
    }
    *outputs_total = running;
    return sz_success_k;
}

#pragma endregion Construction

#pragma region Publication

/** Byte offsets of the one block a published automaton owns. @sa `sz_substrings_publish_layout_`. */
typedef struct sz_substrings_layout_t {
    /** Exclusive prefix sums of the output counts, the widest member and so the first. */
    sz_size_t outputs_offsets;
    /** The merged output pool. */
    sz_size_t outputs;
    /** The dense goto-completed rows. */
    sz_size_t hot_rows;
    /** The byte-to-column map, the narrowest member and so the last. */
    sz_size_t byte_to_class;
    /** Double-array bases. */
    sz_size_t base;
    /** Double-array ownership. */
    sz_size_t check;
    /** Failure links. */
    sz_size_t fail;
    /** Outputs per slot. */
    sz_size_t outputs_counts;
    /** One acceptance bit per slot. */
    sz_size_t accepts_words;
    /** Bytes the whole block needs. */
    sz_size_t total;
} sz_substrings_layout_t;

/** Carves one block into every published array, widest alignment first so nothing needs padding. */
SZ_API_COMPTIME sz_substrings_layout_t sz_substrings_publish_layout_(sz_size_t hot_count, sz_size_t classes_count,
                                                                     sz_size_t slots_count, sz_size_t outputs_total) {
    sz_size_t const accepts_words = sz_size_divide_round_up(slots_count, 32);
    sz_substrings_layout_t layout;
    sz_size_t amount = 0;
    layout.outputs_offsets = amount, amount += slots_count * sizeof(sz_size_t);
    layout.outputs = amount, amount += outputs_total * sizeof(sz_substrings_output_t);
    layout.hot_rows = amount, amount += hot_count * classes_count * sizeof(sz_u32_t);
    layout.base = amount, amount += slots_count * sizeof(sz_u32_t);
    layout.check = amount, amount += slots_count * sizeof(sz_u32_t);
    layout.fail = amount, amount += slots_count * sizeof(sz_u32_t);
    layout.outputs_counts = amount, amount += slots_count * sizeof(sz_u32_t);
    layout.accepts_words = amount, amount += accepts_words * sizeof(sz_u32_t);
    layout.byte_to_class = amount, amount += (SZ_U8_MAX + 1) * sizeof(sz_u8_t);
    layout.total = amount;
    return layout;
}

/**
 *  @brief Fills the published output pool from the per-state lists the insertion threaded.
 *
 *  A state's run is its own matches in insertion order, followed by its failure state's whole run, which
 *  depth-band order has already finished.
 */
SZ_API_COMPTIME void sz_substrings_publish_outputs_(sz_substrings_builder_t const *builder,
                                                    sz_substrings_output_t *outputs) {
    sz_size_t index;
    for (index = 0; index < builder->nodes_count; ++index) {
        sz_u32_t const state = builder->order[index];
        sz_substrings_trie_node_t const *const node = builder->nodes + state;
        sz_substrings_output_t *const destination = outputs + node->output_total_offset;
        sz_size_t written = node->output_own_count, position;
        sz_u32_t needle = node->output_head;
        // The per-state list is most-recent-first, so filling it backwards restores insertion order.
        for (; needle != SZ_SUBSTRINGS_NO_STATE; needle = builder->needle_next[needle]) {
            --written;
            destination[written].needle_index = needle;
            destination[written].folded_match_bytes = builder->needle_folded_bytes[needle];
        }
        if (state == 0) continue;
        {
            sz_substrings_trie_node_t const *const inherited = builder->nodes + node->failure;
            for (position = 0; position < inherited->output_total_count; ++position)
                destination[node->output_own_count + position] = outputs[inherited->output_total_offset + position];
        }
    }
}

/**
 *  @brief Gives every byte some needle spells its own class, and every other byte one shared class.
 *
 *  Bytes no edge carries behave identically from every state, falling through the failure links to the
 *  root, so one column answers for all of them, and a row is only as wide as the vocabulary's alphabet.
 */
SZ_API_COMPTIME void sz_substrings_builder_classify_(sz_substrings_builder_t *builder) {
    sz_bool_t labelled[SZ_U8_MAX + 1];
    sz_size_t byte, state, shared_class = SZ_U8_MAX + 1;
    for (byte = 0; byte != SZ_U8_MAX + 1; ++byte) labelled[byte] = sz_false_k;
    for (state = 1; state < builder->nodes_count; ++state) labelled[builder->nodes[state].parent_byte] = sz_true_k;
    builder->classes_count = 0;
    for (byte = 0; byte != SZ_U8_MAX + 1; ++byte) {
        if (!labelled[byte]) continue;
        builder->byte_to_class[byte] = (sz_u8_t)builder->classes_count++;
    }
    for (byte = 0; byte != SZ_U8_MAX + 1; ++byte) {
        if (labelled[byte]) continue;
        if (shared_class == SZ_U8_MAX + 1) shared_class = builder->classes_count++;
        builder->byte_to_class[byte] = (sz_u8_t)shared_class;
    }
}

/**
 *  @brief Materializes the hot tier's goto-completed rows by inheritance.
 *
 *  Goto completion means @c goto(state, @c byte) @c == @c goto(fail(state), @c byte) wherever @c state has
 *  no literal edge on @c byte. The depth-primary ordering puts @c fail(state) at a strictly smaller index,
 *  so a hot state's failure state is always hot and always already materialized - one row copy plus one
 *  store per literal edge, instead of a failure chase per cell.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_publish_hot_rows_(sz_substrings_builder_t const *builder,
                                                            sz_u32_t *hot_rows) {
    sz_size_t hot_index, column;
    sz_u32_t child;
    if (builder->hot_count == 0) return sz_success_k;

    for (column = 0; column != builder->classes_count; ++column) hot_rows[column] = builder->nodes[0].published;
    for (child = builder->nodes[0].first_child; child != SZ_SUBSTRINGS_NO_STATE;
         child = builder->nodes[child].next_sibling)
        hot_rows[builder->byte_to_class[builder->nodes[child].parent_byte]] = builder->nodes[child].published;

    for (hot_index = 1; hot_index < builder->hot_count; ++hot_index) {
        sz_u32_t const state = builder->order[hot_index];
        sz_size_t const inherited_index = (sz_size_t)builder->nodes[builder->nodes[state].failure].published;
        sz_u32_t const *inherited;
        sz_u32_t *row;
        // A failure state is strictly shallower, so depth-primary order places it earlier and its row is
        // already final. A real return rather than an assert: violated in a release build this would read an
        // unwritten row and bake wrong transitions into the published automaton, silently.
        if (inherited_index >= hot_index) return sz_unexpected_dimensions_k;
        inherited = hot_rows + inherited_index * builder->classes_count;
        row = hot_rows + hot_index * builder->classes_count;
        for (column = 0; column != builder->classes_count; ++column) row[column] = inherited[column];
        for (child = builder->nodes[state].first_child; child != SZ_SUBSTRINGS_NO_STATE;
             child = builder->nodes[child].next_sibling)
            row[builder->byte_to_class[builder->nodes[child].parent_byte]] = builder->nodes[child].published;
    }
    return sz_success_k;
}

#pragma endregion Publication

#pragma region Building

SZ_API_COMPTIME void sz_substrings_engine_free_(sz_substrings_engine_t *engine) {
    sz_memory_allocator_t *const alloc = &engine->alloc;
    if (engine->memory) alloc->free(engine->memory, engine->memory_bytes, alloc->handle);
    if (engine->scratch) alloc->free(engine->scratch, engine->scratch_bytes, alloc->handle);
    engine->memory = SZ_NULL, engine->memory_bytes = 0;
    engine->scratch = SZ_NULL, engine->scratch_bytes = 0;
    engine->hot_rows = SZ_NULL, engine->base = SZ_NULL, engine->check = SZ_NULL, engine->fail = SZ_NULL;
    engine->byte_to_class = SZ_NULL, engine->classes_count = 0;
    engine->accepts_words = SZ_NULL, engine->outputs = SZ_NULL;
    engine->outputs_counts = SZ_NULL, engine->outputs_offsets = SZ_NULL;
    engine->outputs_total = 0, engine->slots_count = 0;
    engine->hot_count = 0, engine->state_count = 0, engine->root = 0, engine->needles_count = 0;
    engine->max_source_match_bytes = 0, engine->min_source_match_bytes = 0, engine->max_outputs_per_state = 0;
    engine->matches_budget = 0, engine->chunk_budget = 0;
    engine->report = SZ_NULL, engine->capability = sz_cap_serial_k;
}

/**
 *  @brief Compiles @p needles into @p engine 's first block, leaving the round's arena for a tier to size.
 *
 *  The one place the vocabulary is read: every tier's arena is a function of the bounds this settles, so a
 *  tier sizes its own scratch afterwards rather than passing the vocabulary around twice.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_engine_compile_(sz_sequence_t const *needles,
                                                          sz_substrings_case_sensitivity_t case_sensitivity,
                                                          sz_substrings_overlap_policy_t overlap_policy,
                                                          sz_size_t hot_states, sz_size_t matches_budget,
                                                          sz_capability_t capability,
                                                          sz_memory_allocator_t const *allocator,
                                                          sz_substrings_engine_t *engine) {
    sz_substrings_builder_t builder;
    sz_substrings_layout_t layout;
    sz_memory_allocator_t resolved;
    sz_memory_allocator_t *const alloc = &resolved;
    sz_size_t needle_index, state_index, longest_needle = 0, state_count_published, slots_count;
    sz_size_t outputs_total = 0, slot;
    sz_status_t status;
    sz_u32_t minted_root;
    sz_ptr_t block;

    if (allocator) resolved = *allocator;
    else sz_memory_allocator_init_default(&resolved);
    builder.nodes = SZ_NULL, builder.nodes_capacity = 0, builder.nodes_count = 0;
    builder.needle_next = SZ_NULL, builder.needle_folded_bytes = SZ_NULL;
    builder.fold_scratch = SZ_NULL, builder.fold_scratch_bytes = 0;
    builder.order = SZ_NULL, builder.order_scratch = SZ_NULL, builder.root_row = SZ_NULL;
    builder.base = SZ_NULL, builder.check = SZ_NULL, builder.state_of_slot = SZ_NULL, builder.occupied = SZ_NULL;
    builder.slots_capacity = 0, builder.lowest_free_cursor = 0, builder.arena_frontier = 0;
    builder.needles_count = needles->count;
    builder.max_source_match_bytes = 0, builder.min_source_match_bytes = 0, builder.max_outputs_per_state = 0;
    builder.case_sensitivity = case_sensitivity, builder.hot_count = 0, builder.alloc = alloc;
    if (!needles->count) return sz_unexpected_dimensions_k;

    for (needle_index = 0; needle_index != needles->count; ++needle_index) {
        sz_size_t const length = needles->get_length(needles->handle, needle_index);
        if (!length) return sz_unexpected_dimensions_k;
        longest_needle = sz_max_of_two(longest_needle, length);
    }
    if (needles->count > (sz_size_t)SZ_SUBSTRINGS_NO_STATE) return sz_overflow_risk_k;

    builder.needle_next = (sz_u32_t *)alloc->allocate(needles->count * sizeof(sz_u32_t), alloc->handle);
    builder.needle_folded_bytes = (sz_u32_t *)alloc->allocate(needles->count * sizeof(sz_u32_t), alloc->handle);
    if (!builder.needle_next || !builder.needle_folded_bytes) {
        sz_substrings_builder_free_(&builder);
        return sz_bad_alloc_k;
    }
    // One source byte folds to at most `sz_utf8_fold_max_expansion_k` bytes, which is what bounds the image
    // of the longest needle; a cased vocabulary folds to itself and needs none of it.
    if (case_sensitivity == sz_substrings_uncased_k) {
        builder.fold_scratch_bytes = longest_needle * (sz_size_t)sz_utf8_fold_max_expansion_k;
        builder.fold_scratch = (sz_u8_t *)alloc->allocate(builder.fold_scratch_bytes, alloc->handle);
        if (!builder.fold_scratch) {
            sz_substrings_builder_free_(&builder);
            return sz_bad_alloc_k;
        }
    }

    status = sz_substrings_builder_mint_(&builder, &minted_root); // ? The root, which is always state zero
    sz_unused_(minted_root);
    for (needle_index = 0; needle_index != needles->count && status == sz_success_k; ++needle_index) {
        sz_cptr_t const start = needles->get_start(needles->handle, needle_index);
        sz_size_t const length = needles->get_length(needles->handle, needle_index);
        builder.needle_next[needle_index] = SZ_SUBSTRINGS_NO_STATE;
        builder.needle_folded_bytes[needle_index] = 0;
        status = case_sensitivity == sz_substrings_uncased_k
                     ? sz_substrings_builder_insert_uncased_(&builder, start, length, (sz_u32_t)needle_index)
                     : sz_substrings_builder_insert_cased_(&builder, (sz_u8_t const *)start, length,
                                                           (sz_u32_t)needle_index);
    }
    if (status == sz_success_k) status = sz_substrings_builder_link_failures_(&builder);
    if (status != sz_success_k) {
        sz_substrings_builder_free_(&builder);
        return status;
    }

    sz_substrings_builder_classify_(&builder);
    builder.hot_count = hot_states == SZ_SUBSTRINGS_HOT_STATES_AUTO
                            ? SZ_SUBSTRINGS_HOT_BYTES_DEFAULT / (builder.classes_count * sizeof(sz_u32_t))
                            : hot_states;
    builder.hot_count = sz_min_of_two(builder.hot_count, builder.nodes_count);

    status = sz_substrings_builder_pack_(&builder);
    if (status == sz_success_k) status = sz_substrings_builder_size_outputs_(&builder, &outputs_total);
    if (status != sz_success_k) {
        sz_substrings_builder_free_(&builder);
        return status;
    }

    // The exclusive published bound: not the hot count plus the cold-state count, since a packed child's id
    // is address arithmetic and can skip past slots no state ever ended up owning.
    state_count_published = builder.hot_count;
    for (state_index = 0; state_index != builder.nodes_count; ++state_index)
        state_count_published = sz_max_of_two(state_count_published,
                                              (sz_size_t)builder.nodes[state_index].published + 1);
    slots_count = state_count_published + SZ_U8_MAX;

    layout = sz_substrings_publish_layout_(builder.hot_count, builder.classes_count, slots_count, outputs_total);
    block = (sz_ptr_t)alloc->allocate(layout.total, alloc->handle);
    if (!block) {
        sz_substrings_builder_free_(&builder);
        return sz_bad_alloc_k;
    }

    {
        sz_size_t *const outputs_offsets = (sz_size_t *)(block + layout.outputs_offsets);
        sz_substrings_output_t *const outputs = (sz_substrings_output_t *)(block + layout.outputs);
        sz_u32_t *const hot_rows = (sz_u32_t *)(block + layout.hot_rows);
        sz_u32_t *const base = (sz_u32_t *)(block + layout.base);
        sz_u32_t *const check = (sz_u32_t *)(block + layout.check);
        sz_u32_t *const fail = (sz_u32_t *)(block + layout.fail);
        sz_u32_t *const outputs_counts = (sz_u32_t *)(block + layout.outputs_counts);
        sz_u32_t *const accepts_words = (sz_u32_t *)(block + layout.accepts_words);
        sz_u8_t *const byte_to_class = (sz_u8_t *)(block + layout.byte_to_class);
        sz_size_t const accepts_count = sz_size_divide_round_up(slots_count, 32);

        for (slot = 0; slot != SZ_U8_MAX + 1; ++slot) byte_to_class[slot] = builder.byte_to_class[slot];

        for (slot = 0; slot != accepts_count; ++slot) accepts_words[slot] = 0;
        for (slot = 0; slot != slots_count; ++slot) {
            // The packing arena can be narrower than the published bound, and every slot past it is free by
            // construction - no base, no owner, and a failure link back to the root.
            sz_u32_t const state = slot < builder.slots_capacity ? builder.state_of_slot[slot] : SZ_SUBSTRINGS_NO_STATE;
            base[slot] = slot < builder.slots_capacity ? builder.base[slot] : 0;
            check[slot] = slot < builder.slots_capacity ? builder.check[slot] : SZ_SUBSTRINGS_NO_STATE;
            if (state == SZ_SUBSTRINGS_NO_STATE) {
                outputs_counts[slot] = 0, outputs_offsets[slot] = 0;
                fail[slot] = 0;
                continue;
            }
            outputs_counts[slot] = builder.nodes[state].output_total_count;
            outputs_offsets[slot] = builder.nodes[state].output_total_offset;
            if (outputs_counts[slot]) accepts_words[slot >> 5] |= (sz_u32_t)1 << (slot & 31u);
            // Each state owns exactly one slot, so a slot's failure link is simply its state's, published.
            sz_assert_(builder.nodes[state].published == (sz_u32_t)slot && "One published slot per state");
            fail[slot] = builder.nodes[builder.nodes[state].failure].published;
        }

        sz_substrings_publish_outputs_(&builder, outputs);
        status = sz_substrings_publish_hot_rows_(&builder, hot_rows);
        if (status != sz_success_k) {
            alloc->free(block, layout.total, alloc->handle);
            sz_substrings_builder_free_(&builder);
            return status;
        }

        engine->hot_rows = hot_rows;
        engine->byte_to_class = byte_to_class;
        engine->base = base;
        engine->check = check;
        engine->fail = fail;
        engine->accepts_words = accepts_words;
        engine->outputs = outputs;
        engine->outputs_counts = outputs_counts;
        engine->outputs_offsets = outputs_offsets;
    }

    engine->outputs_total = outputs_total;
    engine->slots_count = slots_count;
    engine->hot_count = (sz_u32_t)builder.hot_count;
    engine->classes_count = (sz_u32_t)builder.classes_count;
    engine->state_count = (sz_u32_t)state_count_published;
    engine->root = builder.nodes[0].published;
    engine->needles_count = (sz_u32_t)needles->count;
    engine->max_source_match_bytes = builder.max_source_match_bytes;
    engine->min_source_match_bytes = builder.min_source_match_bytes;
    engine->max_outputs_per_state = builder.max_outputs_per_state;
    engine->case_sensitivity = case_sensitivity;
    engine->overlap_policy = overlap_policy;
    engine->matches_budget = matches_budget;
    engine->chunk_budget = 0;
    engine->report = SZ_NULL;
    engine->capability = capability;
    engine->alloc = resolved;
    engine->memory = block;
    engine->memory_bytes = layout.total;
    engine->scratch = SZ_NULL;
    engine->scratch_bytes = 0;
    sz_assert_(engine->root == 0 && "The root is the unique shallowest state, so it always sorts first");
    {
        sz_size_t byte;
        sz_byteset_init(&engine->root_live);
        for (byte = 0; byte != SZ_U8_MAX + 1; ++byte)
            if (sz_substrings_step(engine, engine->root, (sz_u8_t)byte) != engine->root)
                sz_byteset_add_u8(&engine->root_live, (sz_u8_t)byte);
    }
    sz_substrings_builder_free_(&builder);
    return sz_success_k;
}

#pragma endregion Building

#pragma region Matching

/** Bytes one chain of an ordered walk covers per round, so its buffered ends fit on the stack. */
#define SZ_SUBSTRINGS_ORDERED_WINDOW (256)

/** Bytes a byte-exact walk primes a slice with, which is one short of the longest match. */
SZ_HELPER_AUTO sz_size_t sz_substrings_bytes_warm_up_(sz_substrings_engine_t const *engine) {
    return engine->max_source_match_bytes > 0 ? (sz_size_t)engine->max_source_match_bytes - 1 : 0;
}

/**
 *  @brief Counts every match in @p haystack, byte for byte, without enumerating a single output run.
 *
 *  The counts ride the transitions, so no output is ever read. @ref SZ_SUBSTRINGS_CHAINS disjoint slices
 *  step at once, each primed by the bytes before it: a state is the longest suffix read so far that spells
 *  a needle prefix, so once the longest match is behind it a chain cannot remember anything earlier, and
 *  the byte it first reports on is one of them. A haystack whose slices would be shorter than that priming
 *  walks on one chain instead.
 */
SZ_API_COMPTIME sz_size_t sz_substrings_count_bytes_serial(sz_substrings_engine_t const *engine,
                                                           sz_cptr_t haystack, sz_size_t length) {
    sz_size_t const warm_up = sz_substrings_bytes_warm_up_(engine);
    sz_size_t const share = length / SZ_SUBSTRINGS_CHAINS, remainder = length % SZ_SUBSTRINGS_CHAINS;
    sz_u8_t const *const bytes = (sz_u8_t const *)haystack;
    sz_u8_t const *slices[SZ_SUBSTRINGS_CHAINS];
    sz_u32_t states[SZ_SUBSTRINGS_CHAINS];
    sz_size_t total = 0, chain, delta, primed;

    if (share <= warm_up) {
        sz_u32_t state = engine->root;
        for (delta = 0; delta != length; ++delta) total += sz_substrings_step_counting(engine, &state, bytes[delta]);
        return total;
    }

    // The fair split hands the first slices one byte more than the last ones, and never two.
    for (chain = 0; chain != SZ_SUBSTRINGS_CHAINS; ++chain) {
        sz_size_t const first = chain * share + sz_min_of_two(chain, remainder);
        slices[chain] = bytes + first, states[chain] = engine->root;
    }
    // Priming reports nothing, so once it ends the report test is gone from the round rather than being
    // re-asked per byte. The first slice starts where a whole-haystack walk starts and primes nothing.
    for (primed = 0; primed != warm_up; ++primed)
        for (chain = 1; chain != SZ_SUBSTRINGS_CHAINS; ++chain)
            states[chain] = sz_substrings_step(engine, states[chain], *(slices[chain] - warm_up + primed));

    for (delta = 0; delta != share; ++delta)
        for (chain = 0; chain != SZ_SUBSTRINGS_CHAINS; ++chain)
            total += sz_substrings_step_counting(engine, states + chain, slices[chain][delta]);
    for (chain = 0; chain != remainder; ++chain)
        total += sz_substrings_step_counting(engine, states + chain, slices[chain][share]);
    return total;
}

/** Reports every needle ending on @p state at @p end_offset, or stops the walk. */
SZ_HELPER_AUTO sz_substrings_walk_t sz_substrings_report_outputs_(sz_substrings_engine_t const *engine,
                                                                  sz_u32_t state, sz_u32_t output_count,
                                                                  sz_size_t end_offset,
                                                                  sz_substrings_reporter_t reporter, void *context) {
    sz_size_t const output_offset = engine->outputs_offsets[state];
    sz_size_t index;
    for (index = 0; index != output_count; ++index) {
        sz_substrings_output_t const output = engine->outputs[output_offset + index];
        sz_size_t const match_length = output.folded_match_bytes;
        // Tested by addition rather than by subtracting the length: a chain primed from the bytes before
        // its slice can spell a match reaching behind that slice, and a subtraction would wrap.
        if (end_offset + 1 < match_length) continue;
        if (reporter(context, output.needle_index, end_offset + 1 - match_length, match_length) == sz_substrings_stop_k)
            return sz_substrings_stop_k;
    }
    return sz_substrings_continue_k;
}

/** One accepting position an ordered round found, held until the chains before it have reported. */
typedef struct sz_substrings_pending_end_t {
    /** The state the chain stood on, whose outputs the flush enumerates. */
    sz_u32_t state;
    /** Where it stood, relative to the chain's window. */
    sz_u32_t delta;
} sz_substrings_pending_end_t;

/**
 *  @brief Reports every match in @p haystack in ascending end order, stepping several chains at once.
 *
 *  Each round cuts @ref SZ_SUBSTRINGS_CHAINS consecutive windows, primes every chain from the bytes before
 *  its window, and buffers the positions where it accepts. Windows are disjoint and each match belongs to
 *  the window its end falls in, so flushing the buffers in window order reproduces one chain's stream. The
 *  last chain ends exactly where the next round begins, so the tail continues from its state unprimed.
 */
SZ_API_COMPTIME void sz_substrings_find_ascending_(sz_substrings_engine_t const *engine, sz_cptr_t haystack,
                                                   sz_size_t length, sz_substrings_reporter_t reporter, void *context) {
    sz_size_t const warm_up = sz_substrings_bytes_warm_up_(engine);
    sz_size_t const round_bytes = SZ_SUBSTRINGS_CHAINS * SZ_SUBSTRINGS_ORDERED_WINDOW;
    sz_u8_t const *const bytes = (sz_u8_t const *)haystack;
    sz_substrings_pending_end_t pending[SZ_SUBSTRINGS_CHAINS][SZ_SUBSTRINGS_ORDERED_WINDOW];
    sz_size_t pending_counts[SZ_SUBSTRINGS_CHAINS];
    sz_u32_t states[SZ_SUBSTRINGS_CHAINS];
    sz_u32_t state = engine->root;
    sz_size_t round = 0, chain, delta, primed, index;

    for (; round + round_bytes <= length; round += round_bytes) {
        sz_u8_t const *const first = bytes + round;
        // Chain zero continues from the state the previous round's last chain ended on, so it primes nothing.
        states[0] = state;
        for (chain = 1; chain != SZ_SUBSTRINGS_CHAINS; ++chain) states[chain] = engine->root;
        for (primed = 0; primed != warm_up; ++primed)
            for (chain = 1; chain != SZ_SUBSTRINGS_CHAINS; ++chain)
                states[chain] = sz_substrings_step(engine, states[chain],
                                                   first[chain * SZ_SUBSTRINGS_ORDERED_WINDOW - warm_up + primed]);

        for (chain = 0; chain != SZ_SUBSTRINGS_CHAINS; ++chain) pending_counts[chain] = 0;
        for (delta = 0; delta != SZ_SUBSTRINGS_ORDERED_WINDOW; ++delta)
            for (chain = 0; chain != SZ_SUBSTRINGS_CHAINS; ++chain) {
                sz_u32_t const output_count = sz_substrings_step_counting(
                    engine, states + chain, first[chain * SZ_SUBSTRINGS_ORDERED_WINDOW + delta]);
                if (output_count == 0) continue;
                pending[chain][pending_counts[chain]].state = states[chain];
                pending[chain][pending_counts[chain]].delta = (sz_u32_t)delta;
                ++pending_counts[chain];
            }

        for (chain = 0; chain != SZ_SUBSTRINGS_CHAINS; ++chain)
            for (index = 0; index != pending_counts[chain]; ++index) {
                sz_substrings_pending_end_t const end = pending[chain][index];
                sz_size_t const end_offset = round + chain * SZ_SUBSTRINGS_ORDERED_WINDOW + end.delta;
                if (sz_substrings_report_outputs_(engine, end.state, engine->outputs_counts[end.state],
                                                  end_offset, reporter, context) == sz_substrings_stop_k)
                    return;
            }
        state = states[SZ_SUBSTRINGS_CHAINS - 1];
    }

    for (delta = round; delta != length; ++delta) {
        sz_u32_t const output_count = sz_substrings_step_counting(engine, &state, bytes[delta]);
        if (output_count == 0) continue;
        if (sz_substrings_report_outputs_(engine, state, output_count, delta, reporter, context) ==
            sz_substrings_stop_k)
            return;
    }
}

/**
 *  @brief Reports every match in @p haystack, byte for byte, in whichever order @p order asks for.
 *
 *  A transition is one data-dependent load, so a single chain leaves the load ports idle for that whole
 *  latency. An unordered consumer gets @ref SZ_SUBSTRINGS_CHAINS disjoint slices stepped at once, each
 *  primed by the bytes before it; an ordered one gets them in rounds of windows it can buffer. Either way
 *  a haystack too short to amortize the priming walks on one chain.
 *
 *  Acceptance rides the output count rather than the automaton's bit, because a state that accepts is
 *  about to have its count read anyway and both arrays are cache-resident on a host.
 */
SZ_API_COMPTIME void sz_substrings_find_bytes_serial(sz_substrings_engine_t const *engine, sz_cptr_t haystack,
                                                     sz_size_t length, sz_substrings_report_order_t order,
                                                     sz_substrings_reporter_t reporter, void *context) {
    sz_size_t const warm_up = sz_substrings_bytes_warm_up_(engine);
    sz_u8_t const *const bytes = (sz_u8_t const *)haystack;
    sz_u8_t const *slices[SZ_SUBSTRINGS_CHAINS];
    sz_u32_t states[SZ_SUBSTRINGS_CHAINS];
    sz_size_t share, remainder, chain, delta, primed;

    // Rounds re-prime every window, which pays only while the priming is a small share of it.
    if (order == sz_substrings_ascending_ends_k && warm_up * 4 <= SZ_SUBSTRINGS_ORDERED_WINDOW) {
        sz_substrings_find_ascending_(engine, haystack, length, reporter, context);
        return;
    }

    // One chain keeps its state in a register, which an array indexed by a runtime chain count cannot.
    if (order == sz_substrings_ascending_ends_k || length / SZ_SUBSTRINGS_CHAINS <= warm_up) {
        sz_u32_t state = engine->root;
        for (delta = 0; delta != length; ++delta) {
            sz_u32_t const output_count = sz_substrings_step_counting(engine, &state, bytes[delta]);
            if (output_count == 0) continue;
            if (sz_substrings_report_outputs_(engine, state, output_count, delta, reporter, context) ==
                sz_substrings_stop_k)
                return;
        }
        return;
    }

    // The fair split hands the first slices one byte more than the last ones, and never two.
    share = length / SZ_SUBSTRINGS_CHAINS, remainder = length % SZ_SUBSTRINGS_CHAINS;
    for (chain = 0; chain != SZ_SUBSTRINGS_CHAINS; ++chain) {
        sz_size_t const first = chain * share + sz_min_of_two(chain, remainder);
        slices[chain] = bytes + first, states[chain] = engine->root;
    }
    // Priming reports nothing, so once it ends the report test is gone from the round rather than being
    // re-asked per byte. The first slice starts where a whole-haystack walk starts and primes nothing.
    for (primed = 0; primed != warm_up; ++primed)
        for (chain = 1; chain != SZ_SUBSTRINGS_CHAINS; ++chain)
            states[chain] = sz_substrings_step(engine, states[chain], *(slices[chain] - warm_up + primed));

    for (delta = 0; delta != share; ++delta)
        for (chain = 0; chain != SZ_SUBSTRINGS_CHAINS; ++chain) {
            sz_u32_t const output_count = sz_substrings_step_counting(engine, states + chain, slices[chain][delta]);
            if (output_count == 0) continue;
            if (sz_substrings_report_outputs_(engine, states[chain], output_count,
                                              (sz_size_t)(slices[chain] - bytes) + delta, reporter,
                                              context) == sz_substrings_stop_k)
                return;
        }
    for (chain = 0; chain != remainder; ++chain) {
        sz_u32_t const output_count = sz_substrings_step_counting(engine, states + chain, slices[chain][share]);
        if (output_count == 0) continue;
        if (sz_substrings_report_outputs_(engine, states[chain], output_count,
                                          (sz_size_t)(slices[chain] - bytes) + share, reporter,
                                          context) == sz_substrings_stop_k)
            return;
    }
}

/**
 *  @brief Whether skipping from the root to the next live byte pays over @p haystack.
 *
 *  A byte search call costs about as much as eight transitions, so the skip pays once fewer than one byte
 *  in eight leaves the root. The density is a property of the text as much as of the vocabulary, so it is
 *  sampled here, at 64 evenly spaced bytes, rather than decided once per automaton.
 */
SZ_API_COMPTIME sz_bool_t sz_substrings_skipping_pays_(sz_substrings_engine_t const *engine, sz_cptr_t haystack,
                                                       sz_size_t length) {
    sz_size_t const samples = 64;
    sz_size_t const stride = length > samples ? length / samples : 1;
    sz_size_t position, taken = 0, live = 0;
    for (position = 0; position < length && taken != samples; position += stride, ++taken)
        live += sz_byteset_contains_u8(&engine->root_live, (sz_u8_t)haystack[position]);
    return (sz_bool_t)(live * 8 < taken);
}

/**
 *  @brief Counts every match in @p haystack on one chain that jumps from the root to the next live byte.
 *  @param[in] find_live The tier's byte search, handed @c root_live.
 *
 *  The root carries no outputs and leaves itself on every dead byte, so a run of them is skipped whole.
 */
SZ_API_COMPTIME sz_size_t sz_substrings_count_skipping_(sz_substrings_engine_t const *engine, sz_cptr_t haystack,
                                                        sz_size_t length, sz_find_byteset_t find_live) {
    sz_u8_t const *const bytes = (sz_u8_t const *)haystack;
    sz_u32_t state = engine->root;
    sz_size_t position = 0, total = 0;
    while (position != length) {
        if (state == engine->root) {
            sz_cptr_t const live = find_live(haystack + position, length - position, &engine->root_live);
            if (!live) break;
            position = (sz_size_t)(live - haystack);
        }
        total += sz_substrings_step_counting(engine, &state, bytes[position]);
        ++position;
    }
    return total;
}

/** Reports every match in @p haystack in ascending end order, skipping from the root to the next live byte. */
SZ_API_COMPTIME void sz_substrings_find_skipping_(sz_substrings_engine_t const *engine, sz_cptr_t haystack,
                                                  sz_size_t length, sz_substrings_reporter_t reporter, void *context,
                                                  sz_find_byteset_t find_live) {
    sz_u8_t const *const bytes = (sz_u8_t const *)haystack;
    sz_u32_t state = engine->root;
    sz_size_t position = 0;
    while (position != length) {
        sz_u32_t output_count;
        if (state == engine->root) {
            sz_cptr_t const live = find_live(haystack + position, length - position, &engine->root_live);
            if (!live) return;
            position = (sz_size_t)(live - haystack);
        }
        output_count = sz_substrings_step_counting(engine, &state, bytes[position]);
        if (output_count != 0 && sz_substrings_report_outputs_(engine, state, output_count, position, reporter,
                                                               context) == sz_substrings_stop_k)
            return;
        ++position;
    }
}

/**
 *  @brief The stages a tier may replace, handed to every verb so one tier's walk reaches all of them.
 *
 *  Byte-exact walks answer for a cased vocabulary; a folded one takes the cursor, which folds as it walks.
 */
typedef struct sz_substrings_walks_t {
    /** Overlapping count of one haystack walked byte for byte. */
    sz_size_t (*count_bytes)(sz_substrings_engine_t const *, sz_cptr_t, sz_size_t);
    /** Overlapping reports of one haystack walked byte for byte, in the order asked. */
    void (*find_bytes)(sz_substrings_engine_t const *, sz_cptr_t, sz_size_t, sz_substrings_report_order_t,
                       sz_substrings_reporter_t, void *);
} sz_substrings_walks_t;

/** The serial stages, which every tier starts from. */
SZ_API_COMPTIME sz_substrings_walks_t sz_substrings_walks_serial_(void) {
    sz_substrings_walks_t walks;
    walks.count_bytes = &sz_substrings_count_bytes_serial;
    walks.find_bytes = &sz_substrings_find_bytes_serial;
    return walks;
}

/** Whether this vocabulary's haystacks are walked byte for byte, rather than folded as they are walked. */
SZ_API_COMPTIME sz_bool_t sz_substrings_walks_bytes_(sz_substrings_engine_t const *engine) {
    return (sz_bool_t)(engine->case_sensitivity == sz_substrings_cased_k);
}

/**
 *  @brief Reports every match in @p haystack, folding it one codepoint at a time.
 *
 *  Only a byte ending a folded rune can end a match, so a reported end is always a whole codepoint's.
 */
SZ_API_COMPTIME void sz_substrings_find_uncased_(sz_substrings_engine_t const *engine, sz_cptr_t haystack,
                                                 sz_size_t length, sz_substrings_reporter_t reporter, void *context) {
    sz_substrings_folded_cursor_t cursor;
    sz_substrings_folded_byte_t step;
    sz_u32_t state = engine->root;
    sz_size_t folded = 0, last_break_folded_end = 0;

    sz_substrings_folded_cursor_init(&cursor, haystack, length);
    while (sz_substrings_folded_cursor_next(&cursor, &step)) {
        sz_size_t output_offset, output_count, index;
        ++folded;
        if (step.malformed) {
            // A malformed byte can never sit inside a match, so the walk drops back to the root and
            // resynchronizes one byte at a time, exactly as the folded iterators do.
            state = engine->root;
            continue;
        }

        state = sz_substrings_step(engine, state, step.byte);
        if (!step.rune_end) continue;
        // Claimed before this rune end reports: a match ending inside a boundary-breaking codepoint starts
        // inside it too, and subtracting its folded length would land mid-codepoint.
        if (step.breaks_boundary) last_break_folded_end = folded + step.trailing;

        output_count = engine->outputs_counts[state];
        if (output_count == 0) continue;
        output_offset = engine->outputs_offsets[state];
        for (index = 0; index != output_count; ++index) {
            sz_substrings_output_t const output = engine->outputs[output_offset + index];
            sz_size_t const folded_length = output.folded_match_bytes;
            sz_substrings_resolved_match_t resolved;
            if (folded < folded_length) continue;
            resolved = sz_substrings_folded_span(haystack, &step, folded, last_break_folded_end, folded_length);
            if (resolved.repeats) continue;
            if (reporter(context, output.needle_index, resolved.source_offset,
                         step.codepoint_end - resolved.source_offset) == sz_substrings_stop_k)
                return;
        }
    }
}

/**
 *  @brief Reports every match of @p haystack, folding it when the vocabulary and the haystack ask.
 *  @note A folded walk keeps one chain whatever @p order names, since a fold consumes a variable number of
 *        source bytes per step and so cannot be indexed in lockstep.
 */
SZ_API_COMPTIME void sz_substrings_find_all_(sz_substrings_engine_t const *engine,
                                             sz_substrings_walks_t const *walks, sz_cptr_t haystack, sz_size_t length,
                                             sz_substrings_report_order_t order, sz_substrings_reporter_t reporter,
                                             void *context) {
    if (sz_substrings_walks_bytes_(engine)) walks->find_bytes(engine, haystack, length, order, reporter, context);
    else sz_substrings_find_uncased_(engine, haystack, length, reporter, context);
}

/**
 *  @brief The undecided starts of a leftmost walk, and which of them any match has claimed.
 *
 *  Every claimed start lies within @c width of the drain position, so the ring holds them keyed by start
 *  modulo @c width, and the bitmap lets draining jump between claims rather than visit every byte.
 */
typedef struct sz_substrings_ring_t {
    /** The @b [width] best match per start, zero where no match has claimed it. */
    sz_substrings_pending_start_t *starts;
    /** One bit per entry of @c starts, set exactly where that entry is claimed. */
    sz_u64_t *claimed;
    /** Entries the ring holds, a power of two so the slot lookup is a mask. */
    sz_size_t width;
} sz_substrings_ring_t;

/** Words the claimed bitmap of a ring @p width entries wide takes. */
SZ_HELPER_AUTO sz_size_t sz_substrings_ring_words_(sz_size_t width) { return sz_size_divide_round_up(width, 64); }

/** The first claimed start in @c [from, @c limit), or @p limit when there is none. */
SZ_API_COMPTIME sz_size_t sz_substrings_ring_next_claimed_(sz_substrings_ring_t const *ring, sz_size_t from,
                                                           sz_size_t limit) {
    // Every claim lies within one width of `from`, so nothing past that can be claimed.
    sz_size_t const end = sz_min_of_two(limit, from + ring->width);
    sz_size_t position = from;
    while (position < end) {
        sz_size_t const slot = position & (ring->width - 1), bit = slot & 63;
        sz_size_t const span = sz_min_of_two(sz_min_of_two((sz_size_t)64 - bit, ring->width - slot), end - position);
        sz_u64_t const window = span == 64 ? ~(sz_u64_t)0 : (((sz_u64_t)1 << span) - 1);
        sz_u64_t const bits = (ring->claimed[slot >> 6] >> bit) & window;
        if (bits) return position + (sz_size_t)sz_u64_ctz(bits);
        position += span;
    }
    return limit;
}

/** What a leftmost walk carries between the overlapping walk beneath it and the cover it is deciding. */
typedef struct sz_substrings_leftmost_context_t {
    /** The undecided starts, empty on entry. */
    sz_substrings_ring_t *ring;
    /** Which cover the ties resolve under. */
    sz_substrings_overlap_policy_t policy;
    /** Where the consumer's own reports go. */
    sz_substrings_reporter_t reporter;
    /** Whatever the consumer bound to that reporter. */
    void *context;
    /** First byte no accepted match has claimed, which is what makes the cover non-overlapping. */
    sz_size_t cursor;
    /** Starts already drained. */
    sz_size_t settled;
    /** One past the last start any match claimed, which is where draining stops. */
    sz_size_t undrained_end;
    /** Whether the consumer is still listening. */
    sz_substrings_walk_t walk;
} sz_substrings_leftmost_context_t;

/** Drains one claimed start, reporting its incumbent when nothing accepted has already covered it. */
SZ_API_COMPTIME void sz_substrings_leftmost_accept_(sz_substrings_leftmost_context_t *leftmost, sz_size_t start) {
    sz_substrings_ring_t *const ring = leftmost->ring;
    sz_size_t const slot_index = start & (ring->width - 1);
    sz_substrings_pending_start_t *const slot = ring->starts + slot_index;
    if (start >= leftmost->cursor) {
        leftmost->walk = leftmost->reporter(leftmost->context, slot->needle_index, start, slot->source_match_bytes);
        leftmost->cursor = start + slot->source_match_bytes;
    }
    slot->needle_index = 0, slot->source_match_bytes = 0;
    ring->claimed[slot_index >> 6] &= ~((sz_u64_t)1 << (slot_index & 63));
}

/** Drains every claimed start before @p limit, jumping between claims. */
SZ_API_COMPTIME void sz_substrings_leftmost_drain_(sz_substrings_leftmost_context_t *leftmost, sz_size_t limit) {
    while (leftmost->settled < limit && leftmost->walk == sz_substrings_continue_k) {
        sz_size_t const next = sz_substrings_ring_next_claimed_(leftmost->ring, leftmost->settled, limit);
        if (next == limit) {
            leftmost->settled = limit;
            break;
        }
        sz_substrings_leftmost_accept_(leftmost, next);
        leftmost->settled = next + 1;
    }
}

/** Takes one overlapping match into the ring, draining whatever it settles on the way. */
SZ_API_COMPTIME sz_substrings_walk_t sz_substrings_leftmost_report_(void *context, sz_size_t needle_index,
                                                                    sz_size_t byte_offset, sz_size_t byte_length) {
    sz_substrings_leftmost_context_t *const leftmost = (sz_substrings_leftmost_context_t *)context;
    sz_substrings_ring_t *const ring = leftmost->ring;
    sz_size_t const settles_before = byte_offset + byte_length > ring->width ? byte_offset + byte_length - ring->width
                                                                             : 0;
    sz_size_t const slot_index = byte_offset & (ring->width - 1);
    sz_substrings_pending_start_t challenger;
    // The walk beneath reports in non-decreasing end order, so the starts drain from the end each match
    // reaches - no second walk, and the cursor test waits until a start can no longer be outbid.
    sz_substrings_leftmost_drain_(leftmost, settles_before);
    if (leftmost->walk == sz_substrings_stop_k) return sz_substrings_stop_k;

    challenger.needle_index = (sz_u32_t)needle_index, challenger.source_match_bytes = (sz_u32_t)byte_length;
    if (sz_substrings_leftmost_wins(challenger, ring->starts[slot_index], leftmost->policy)) {
        ring->starts[slot_index] = challenger;
        ring->claimed[slot_index >> 6] |= (sz_u64_t)1 << (slot_index & 63);
    }
    leftmost->undrained_end = sz_max_of_two(leftmost->undrained_end, byte_offset + 1);
    return sz_substrings_continue_k;
}

/**
 *  @brief Reports the matches of @p haystack that share no bytes, one per accepted start position.
 *  @param[in] ring Empty on entry, and empty again on return.
 *
 *  Matches surface at their end, so the earliest start is not the first seen: over "abcd" against
 *  {"bc", "abcd"}, "bc" completes first and "abcd" starts before it. A start settles only once the walk is
 *  @c max_source_match_bytes past it, which is what the ring holds.
 */
SZ_API_COMPTIME void sz_substrings_find_leftmost_(sz_substrings_engine_t const *engine,
                                                  sz_substrings_walks_t const *walks, sz_cptr_t haystack,
                                                  sz_size_t length, sz_substrings_ring_t *ring,
                                                  sz_substrings_overlap_policy_t policy,
                                                  sz_substrings_reporter_t reporter, void *context) {
    sz_substrings_leftmost_context_t leftmost;
    leftmost.ring = ring, leftmost.policy = policy;
    leftmost.reporter = reporter, leftmost.context = context;
    leftmost.cursor = 0, leftmost.settled = 0, leftmost.undrained_end = 0;
    leftmost.walk = sz_substrings_continue_k;
    sz_assert_(policy != sz_substrings_overlapping_k && "Overlapping matches are reported through `find_all`");
    sz_assert_((ring->width & (ring->width - 1)) == 0 && "A power-of-two width turns the lookup into a mask");

    sz_substrings_find_all_(engine, walks, haystack, length, sz_substrings_ascending_ends_k,
                            &sz_substrings_leftmost_report_, &leftmost);
    // Draining stops at the last start any match claimed rather than at the haystack's end.
    sz_substrings_leftmost_drain_(&leftmost, leftmost.undrained_end);
    // A consumer that stopped the walk left its undecided starts behind, so they are cleared here.
    if (leftmost.walk == sz_substrings_stop_k) {
        sz_size_t slot;
        for (slot = 0; slot != ring->width; ++slot) {
            ring->starts[slot].needle_index = 0;
            ring->starts[slot].source_match_bytes = 0;
        }
        for (slot = 0; slot != sz_substrings_ring_words_(ring->width); ++slot) ring->claimed[slot] = 0;
    }
}

/** Reports every match of @p haystack in the order @p policy names. */
SZ_API_COMPTIME void sz_substrings_visit_(sz_substrings_engine_t const *engine,
                                          sz_substrings_walks_t const *walks, sz_cptr_t haystack, sz_size_t length,
                                          sz_substrings_ring_t *ring, sz_substrings_overlap_policy_t policy,
                                          sz_substrings_report_order_t order, sz_substrings_reporter_t reporter,
                                          void *context) {
    if (policy == sz_substrings_overlapping_k)
        sz_substrings_find_all_(engine, walks, haystack, length, order, reporter, context);
    else sz_substrings_find_leftmost_(engine, walks, haystack, length, ring, policy, reporter, context);
}

#pragma endregion Matching

#pragma region Serial Backends

/** Tallies reported matches, which is what a leftmost count and a rewrite's sizing pass both need. */
typedef struct sz_substrings_tally_t {
    /** Matches reported so far. */
    sz_size_t count;
} sz_substrings_tally_t;

SZ_API_COMPTIME sz_substrings_walk_t sz_substrings_tally_report_(void *context, sz_size_t needle_index,
                                                                 sz_size_t byte_offset, sz_size_t byte_length) {
    sz_unused_(needle_index), sz_unused_(byte_offset), sz_unused_(byte_length);
    ++((sz_substrings_tally_t *)context)->count;
    return sz_substrings_continue_k;
}

/** Collects reported matches into the caller's array, tallying past its end rather than stopping there. */
typedef struct sz_substrings_collector_t {
    /** Where the matches land, or @c SZ_NULL for a pure size query. */
    sz_substrings_match_t *matches;
    /** Entries @c matches holds. */
    sz_size_t capacity;
    /** Matches reported so far, which is the true total whether or not they fit. */
    sz_size_t count;
    /** Which haystack the current walk is over. */
    sz_size_t haystack_index;
} sz_substrings_collector_t;

SZ_API_COMPTIME sz_substrings_walk_t sz_substrings_collect_report_(void *context, sz_size_t needle_index,
                                                                   sz_size_t byte_offset, sz_size_t byte_length) {
    sz_substrings_collector_t *const collector = (sz_substrings_collector_t *)context;
    if (collector->count < collector->capacity) {
        sz_substrings_match_t *const match = collector->matches + collector->count;
        match->haystack_index = collector->haystack_index;
        match->needle_index = needle_index;
        match->byte_offset = byte_offset;
        match->byte_length = byte_length;
    }
    ++collector->count;
    return sz_substrings_continue_k;
}

/** Splices replacements over one haystack, tallying the true size whether or not the output holds it. */
typedef struct sz_substrings_rewriter_t {
    /** The haystack being rewritten, read in place. */
    sz_cptr_t haystack;
    /** Bytes it holds. */
    sz_size_t haystack_length;
    /** One replacement per needle, indexed by needle. */
    sz_sequence_t const *replacements;
    /** Where the rewrite lands, or @c SZ_NULL for a pure size query. */
    sz_ptr_t output;
    /** Bytes @c output holds. */
    sz_size_t output_capacity;
    /** Source bytes consumed, which is where the next gap begins. */
    sz_size_t cursor;
    /** Output bytes written. */
    sz_size_t written;
    /** Source bytes the accepted matches cover. */
    sz_size_t removed;
    /** Replacement bytes they bring in. */
    sz_size_t added;
} sz_substrings_rewriter_t;

/** Copies one stretch when the output still has room for all of it, and skips it whole when it does not. */
SZ_API_COMPTIME void sz_substrings_rewriter_emit_(sz_substrings_rewriter_t *rewriter, sz_cptr_t source,
                                                  sz_size_t bytes) {
    if (bytes == 0 || rewriter->written + bytes > rewriter->output_capacity) return;
    sz_copy(rewriter->output + rewriter->written, source, bytes);
    rewriter->written += bytes;
}

SZ_API_COMPTIME sz_substrings_walk_t sz_substrings_rewrite_report_(void *context, sz_size_t needle_index,
                                                                   sz_size_t byte_offset, sz_size_t byte_length) {
    sz_substrings_rewriter_t *const rewriter = (sz_substrings_rewriter_t *)context;
    sz_sequence_t const *const replacements = rewriter->replacements;
    sz_cptr_t const replacement = replacements->get_start(replacements->handle, needle_index);
    sz_size_t const replacement_length = replacements->get_length(replacements->handle, needle_index);
    sz_substrings_rewriter_emit_(rewriter, rewriter->haystack + rewriter->cursor, byte_offset - rewriter->cursor);
    sz_substrings_rewriter_emit_(rewriter, replacement, replacement_length);
    rewriter->cursor = byte_offset + byte_length;
    rewriter->removed += byte_length, rewriter->added += replacement_length;
    return sz_substrings_continue_k;
}

/**
 *  @brief Rewrites one haystack, reporting the bytes it produces whether or not they fit.
 *  @return Bytes the rewrite produces, which is the true count for every haystack and from the first byte.
 *
 *  Sizing and splicing are one walk: the copies run while there is room and the tally runs to the end
 *  whatever happens, so the size never depends on what the output could hold.
 */
SZ_API_COMPTIME sz_size_t sz_substrings_rewrite_(sz_substrings_engine_t const *engine,
                                                 sz_substrings_walks_t const *walks, sz_cptr_t haystack,
                                                 sz_size_t length, sz_sequence_t const *replacements,
                                                 sz_substrings_ring_t *ring, sz_substrings_overlap_policy_t policy,
                                                 sz_ptr_t output, sz_size_t output_capacity) {
    sz_substrings_rewriter_t rewriter;
    rewriter.haystack = haystack, rewriter.haystack_length = length, rewriter.replacements = replacements;
    rewriter.output = output, rewriter.output_capacity = output_capacity;
    rewriter.cursor = 0, rewriter.written = 0, rewriter.removed = 0, rewriter.added = 0;
    sz_substrings_find_leftmost_(engine, walks, haystack, length, ring, policy, &sz_substrings_rewrite_report_,
                                 &rewriter);
    sz_substrings_rewriter_emit_(&rewriter, haystack + rewriter.cursor, length - rewriter.cursor);
    // Accumulated apart rather than netted per match, so a shrinking rewrite never wraps the unsigned sum.
    return length - rewriter.removed + rewriter.added;
}

/** Bytes one block holding a ring @p width entries wide takes: its starts, then its bitmap. */
SZ_HELPER_AUTO sz_size_t sz_substrings_ring_bytes_(sz_size_t width) {
    return width * sizeof(sz_substrings_pending_start_t) + sz_substrings_ring_words_(width) * sizeof(sz_u64_t);
}

/**
 *  @brief Byte offsets of the one arena every host tier's round runs out of.
 *
 *  Everything a round needs is a function of the vocabulary and the policy, both settled at construction, so
 *  the arena is sized once and no compute verb ever reaches for an allocator.
 */
typedef struct sz_substrings_host_arena_t {
    /** Offset of the round's report, which is the first thing every verb writes. */
    sz_size_t report;
    /** Offset of the leftmost ring's starts, equal to @c bm25_counts when the policy claims no ring. */
    sz_size_t ring;
    /** Offset of the @b [2 * needles] BM25 counters: the per-needle tallies, then the touched list. */
    sz_size_t bm25_counts;
    /** Bytes the whole arena takes. */
    sz_size_t total;
} sz_substrings_host_arena_t;

/** Lays the host arena out for one vocabulary under one policy. */
SZ_API_COMPTIME sz_substrings_host_arena_t sz_substrings_host_arena_(sz_size_t needles_count,
                                                                     sz_size_t max_source_match_bytes,
                                                                     sz_substrings_overlap_policy_t overlap_policy) {
    sz_size_t const ring_width = sz_substrings_pending_starts_width(max_source_match_bytes);
    sz_size_t const ring_bytes =
        overlap_policy == sz_substrings_overlapping_k ? 0 : sz_substrings_ring_bytes_(ring_width);
    sz_substrings_host_arena_t arena;
    arena.report = 0;
    arena.ring = sizeof(sz_substrings_report_t);
    arena.bm25_counts = arena.ring + ring_bytes;
    arena.total = arena.bm25_counts + 2 * needles_count * sizeof(sz_u32_t);
    return arena;
}

/** Binds the leftmost ring onto the engine's arena, empty, or leaves it unbound under an overlapping policy. */
SZ_API_COMPTIME void sz_substrings_ring_bind_(sz_substrings_engine_t const *engine, sz_substrings_ring_t *ring) {
    sz_substrings_host_arena_t const arena =
        sz_substrings_host_arena_(engine->needles_count, engine->max_source_match_bytes, engine->overlap_policy);
    ring->starts = SZ_NULL, ring->claimed = SZ_NULL, ring->width = 0;
    if (engine->overlap_policy == sz_substrings_overlapping_k) return;
    ring->width = sz_substrings_pending_starts_width(engine->max_source_match_bytes);
    ring->starts = (sz_substrings_pending_start_t *)((sz_u8_t *)engine->scratch + arena.ring);
    ring->claimed = (sz_u64_t *)((sz_u8_t *)ring->starts + ring->width * sizeof(sz_substrings_pending_start_t));
}

/** The round's report, which each tier's own arena places and @c _init_* binds. */
SZ_HELPER_AUTO sz_substrings_report_t *sz_substrings_report_(sz_substrings_engine_t const *engine) {
    return engine->report;
}

/** Leaves the report empty, which is what every verb starts its round from. */
SZ_API_COMPTIME void sz_substrings_report_clear_(sz_substrings_report_t *report) {
    report->matches_emitted = 0, report->matches_stored = 0, report->tape_bytes = 0, report->shortfall = 0;
}

/** Allocates and zeroes the host arena, which is the second and last block an engine owns. */
SZ_API_COMPTIME sz_status_t sz_substrings_engine_arena_host_(sz_substrings_engine_t *engine) {
    sz_substrings_host_arena_t const arena =
        sz_substrings_host_arena_(engine->needles_count, engine->max_source_match_bytes, engine->overlap_policy);
    sz_memory_allocator_t *const alloc = &engine->alloc;
    sz_u8_t *block = (sz_u8_t *)alloc->allocate(arena.total, alloc->handle);
    sz_size_t index;
    if (!block) return sz_bad_alloc_k;
    for (index = 0; index != arena.total; ++index) block[index] = 0;
    engine->scratch = block, engine->scratch_bytes = arena.total;
    engine->report = (sz_substrings_report_t *)block;
    return sz_success_k;
}

/**
 *  @brief Compiles @p needles and sizes the host arena beside it, which is @ref sz_substrings_engine_init_cpu.
 *
 *  Split from the public verb only so a device tier can reuse the compilation without the host arena.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_engine_build_(sz_sequence_t const *needles,
                                                        sz_substrings_case_sensitivity_t case_sensitivity,
                                                        sz_substrings_overlap_policy_t overlap_policy,
                                                        sz_size_t hot_states, sz_size_t matches_budget,
                                                        sz_capability_t capability,
                                                        sz_memory_allocator_t *alloc, sz_substrings_engine_t *engine) {
    sz_status_t status = sz_substrings_engine_compile_(needles, case_sensitivity, overlap_policy, hot_states,
                                                       matches_budget, capability, alloc, engine);
    if (status != sz_success_k) return status;
    status = sz_substrings_engine_arena_host_(engine);
    if (status != sz_success_k) sz_substrings_engine_free_(engine);
    return status;
}

/** Refuses an output stride that cannot address one entry per haystack. */
SZ_HELPER_AUTO sz_status_t sz_substrings_stride_check_(sz_size_t stride) {
    return stride == 0 ? sz_unexpected_dimensions_k : sz_success_k;
}

/** Per-haystack counts through @p walks, which is every CPU tier's counting verb. */
SZ_API_COMPTIME sz_status_t sz_substrings_counts_with_(sz_substrings_engine_t *engine,
                                                       sz_substrings_walks_t const *walks,
                                                       sz_sequence_t const *haystacks, sz_size_t *counts,
                                                       sz_size_t counts_stride) {
    sz_substrings_overlap_policy_t const overlap_policy = engine->overlap_policy;
    sz_substrings_report_t *const report = sz_substrings_report_(engine);
    sz_substrings_ring_t ring;
    sz_size_t haystack_index, total = 0;
    sz_status_t const checked = sz_substrings_stride_check_(counts_stride);
    if (checked != sz_success_k) return checked;
    sz_substrings_ring_bind_(engine, &ring);
    sz_substrings_report_clear_(report);

    for (haystack_index = 0; haystack_index != haystacks->count; ++haystack_index) {
        sz_cptr_t const haystack = haystacks->get_start(haystacks->handle, haystack_index);
        sz_size_t const length = haystacks->get_length(haystacks->handle, haystack_index);
        sz_size_t counted;
        // Only an overlapping count over a byte-exact walk can ride the transitions alone; every other shape
        // has to see the matches themselves, either to fold spans together or to decide a cover between them.
        if (overlap_policy == sz_substrings_overlapping_k && sz_substrings_walks_bytes_(engine))
            counted = walks->count_bytes(engine, haystack, length);
        else {
            sz_substrings_tally_t tally;
            tally.count = 0;
            sz_substrings_visit_(engine, walks, haystack, length, &ring, overlap_policy, sz_substrings_unordered_k,
                                 &sz_substrings_tally_report_, &tally);
            counted = tally.count;
        }
        counts[haystack_index * counts_stride] = counted;
        total += counted;
    }
    report->matches_emitted = total, report->matches_stored = total;
    return sz_success_k;
}

SZ_API_COMPTIME sz_status_t sz_substrings_counts_serial(sz_substrings_engine_t *engine,
                                                        sz_sequence_t const *haystacks, sz_size_t *counts,
                                                        sz_size_t counts_stride) {
    sz_substrings_walks_t const walks = sz_substrings_walks_serial_();
    return sz_substrings_counts_with_(engine, &walks, haystacks, counts, counts_stride);
}

/** Every match through @p walks, which is every CPU tier's locating verb. */
SZ_API_COMPTIME sz_status_t sz_substrings_find_with_(sz_substrings_engine_t *engine,
                                                     sz_substrings_walks_t const *walks, sz_sequence_t const *haystacks,
                                                     sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                     sz_size_t *matches_offsets) {
    sz_substrings_report_t *const report = sz_substrings_report_(engine);
    sz_substrings_collector_t collector;
    sz_substrings_ring_t ring;
    sz_size_t haystack_index;
    sz_substrings_ring_bind_(engine, &ring);
    sz_substrings_report_clear_(report);

    collector.matches = matches, collector.capacity = matches_capacity;
    collector.count = 0, collector.haystack_index = 0;
    for (haystack_index = 0; haystack_index != haystacks->count; ++haystack_index) {
        sz_cptr_t const haystack = haystacks->get_start(haystacks->handle, haystack_index);
        sz_size_t const length = haystacks->get_length(haystacks->handle, haystack_index);
        matches_offsets[haystack_index] = collector.count;
        collector.haystack_index = haystack_index;
        sz_substrings_visit_(engine, walks, haystack, length, &ring, engine->overlap_policy,
                             sz_substrings_unordered_k, &sz_substrings_collect_report_, &collector);
    }
    matches_offsets[haystacks->count] = collector.count;
    report->matches_emitted = collector.count;
    report->matches_stored = sz_min_of_two(collector.count, matches_capacity);
    report->shortfall = collector.count - report->matches_stored;
    return sz_success_k;
}

SZ_API_COMPTIME sz_status_t sz_substrings_find_serial(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                      sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                      sz_size_t *matches_offsets) {
    sz_substrings_walks_t const walks = sz_substrings_walks_serial_();
    return sz_substrings_find_with_(engine, &walks, haystacks, matches, matches_capacity, matches_offsets);
}

/** The rewrite through @p walks, which is every CPU tier's rewriting verb. */
SZ_API_COMPTIME sz_status_t sz_substrings_replace_with_(sz_substrings_engine_t *engine,
                                                        sz_substrings_walks_t const *walks,
                                                        sz_sequence_t const *haystacks,
                                                        sz_sequence_t const *replacements, sz_ptr_t tape,
                                                        sz_size_t tape_capacity, sz_size_t *offsets) {
    sz_substrings_report_t *const report = sz_substrings_report_(engine);
    sz_substrings_ring_t ring;
    sz_size_t haystack_index, running = 0;
    // A substitution over matches that share bytes is not a function, so there is no cover to apply.
    if (engine->overlap_policy == sz_substrings_overlapping_k) return sz_status_unknown_k;
    if (replacements->count != engine->needles_count) return sz_unexpected_dimensions_k;
    sz_substrings_ring_bind_(engine, &ring);
    sz_substrings_report_clear_(report);

    // One walk per haystack, splicing straight into the tape at the offset the walk before it settled.
    // The rewrite reports the bytes it produces whether or not they fit, so a second sizing pass would
    // walk the same automaton for an answer this one already has.
    for (haystack_index = 0; haystack_index != haystacks->count; ++haystack_index) {
        sz_cptr_t const haystack = haystacks->get_start(haystacks->handle, haystack_index);
        sz_size_t const length = haystacks->get_length(haystacks->handle, haystack_index);
        sz_size_t const room = tape && running < tape_capacity ? tape_capacity - running : 0;
        offsets[haystack_index] = running;
        running += sz_substrings_rewrite_(engine, walks, haystack, length, replacements, &ring,
                                          engine->overlap_policy, room ? tape + running : SZ_NULL, room);
    }
    offsets[haystacks->count] = running;
    report->tape_bytes = running;
    report->shortfall = running > tape_capacity ? running - tape_capacity : 0;
    return sz_success_k;
}

SZ_API_COMPTIME sz_status_t sz_substrings_replace_serial(sz_substrings_engine_t *engine,
                                                         sz_sequence_t const *haystacks,
                                                         sz_sequence_t const *replacements, sz_ptr_t tape,
                                                         sz_size_t tape_capacity, sz_size_t *offsets) {
    sz_substrings_walks_t const walks = sz_substrings_walks_serial_();
    return sz_substrings_replace_with_(engine, &walks, haystacks, replacements, tape, tape_capacity, offsets);
}

/** Refuses the BM25 arguments no backend can score, before any walk or allocation. */
SZ_HELPER_AUTO sz_status_t sz_substrings_bm25_check(sz_substrings_bm25_t const *parameters,
                                                    sz_f32_t const *needle_weights) {
    if (!parameters || !needle_weights) return sz_unexpected_dimensions_k;
    if (parameters->length_normalization > 0 && !(parameters->average_document_length > 0))
        return sz_unexpected_dimensions_k;
    return sz_success_k;
}

/** The factor every term of one document shares: 1 without length normalization, otherwise `1 - b + b·len/avg`. */
SZ_HELPER_AUTO sz_f64_t sz_substrings_bm25_norm(sz_substrings_bm25_t const *parameters, sz_f64_t document_length) {
    sz_f64_t const normalization = parameters->length_normalization;
    if (!(normalization > 0)) return 1;
    return 1 - normalization + normalization * document_length / parameters->average_document_length;
}

/** One needle's contribution: its weight times the saturated frequency `tf·(k1 + 1) / (tf + k1·norm)`. */
SZ_HELPER_AUTO sz_f64_t sz_substrings_bm25_term(sz_substrings_bm25_t const *parameters, sz_f64_t norm, sz_f32_t weight,
                                                sz_size_t term_frequency) {
    sz_f64_t const saturation = parameters->term_frequency_saturation;
    sz_f64_t const frequency = (sz_f64_t)term_frequency;
    return weight * frequency * (saturation + 1) / (frequency + saturation * norm);
}

/** Per-needle counters for one document, plus the needles it touched in first-occurrence order. */
typedef struct sz_substrings_frequencies_t {
    /** The @b [needles] occurrences of each needle so far, zero for every needle not in @c touched. */
    sz_u32_t *counts;
    /** The @b [needles] list whose head holds the needles with a nonzero count, its tail spare room. */
    sz_u32_t *touched;
    /** Entries at the head of @c touched. */
    sz_size_t touched_count;
    /** Needles in the vocabulary, the length of both arrays. */
    sz_size_t needles_count;
} sz_substrings_frequencies_t;

SZ_API_COMPTIME sz_substrings_walk_t sz_substrings_frequencies_report_(void *context, sz_size_t needle_index,
                                                                       sz_size_t byte_offset, sz_size_t byte_length) {
    sz_substrings_frequencies_t *const frequencies = (sz_substrings_frequencies_t *)context;
    sz_unused_(byte_offset), sz_unused_(byte_length);
    if (frequencies->counts[needle_index]++ == 0)
        frequencies->touched[frequencies->touched_count++] = (sz_u32_t)needle_index;
    return sz_substrings_continue_k;
}

/**
 *  @brief Adds every overlapping occurrence in @p haystack to @p frequencies.
 *
 *  Split from @ref sz_substrings_bm25_total_ so slices of one long document can be counted into separate
 *  rows and merged by addition, since integer frequencies do not depend on who counted them.
 */
SZ_API_COMPTIME void sz_substrings_bm25_count_(sz_substrings_engine_t const *engine,
                                               sz_substrings_walks_t const *walks, sz_cptr_t haystack, sz_size_t length,
                                               sz_substrings_frequencies_t *frequencies) {
    // Raw overlapping frequencies: a leftmost cover would hide a needle nested inside another.
    sz_substrings_find_all_(engine, walks, haystack, length, sz_substrings_unordered_k,
                            &sz_substrings_frequencies_report_, frequencies);
}

/**
 *  @brief Sorts the touched needles ascending with a byte-wise LSD radix sort into the list's own tail.
 *  @return Whichever half holds the sorted order.
 *  @pre `touched_count * 2 <= needles_count`, so the tail is at least as long as the head.
 */
SZ_API_COMPTIME sz_u32_t *sz_substrings_sort_needles_(sz_u32_t *keys, sz_size_t count, sz_size_t needles_count) {
    sz_u32_t *spare = keys + count;
    sz_size_t buckets[SZ_U8_MAX + 1];
    sz_size_t shift, index, bucket, running;
    for (shift = 0; shift < 32 && (needles_count - 1) >> shift; shift += 8) {
        for (bucket = 0; bucket != SZ_U8_MAX + 1; ++bucket) buckets[bucket] = 0;
        for (index = 0; index != count; ++index) ++buckets[(keys[index] >> shift) & SZ_U8_MAX];
        for (bucket = 0, running = 0; bucket != SZ_U8_MAX + 1; ++bucket) {
            sz_size_t const size = buckets[bucket];
            buckets[bucket] = running, running += size;
        }
        for (index = 0; index != count; ++index) spare[buckets[(keys[index] >> shift) & SZ_U8_MAX]++] = keys[index];
        {
            sz_u32_t *const sorted = spare;
            spare = keys, keys = sorted;
        }
    }
    return keys;
}

/**
 *  @brief Scores one counted document in ascending needle order, leaving @p frequencies empty again.
 *
 *  Float addition is not associative, so the order is part of the answer: fixing it makes the score
 *  independent of the order any walk reported in.
 */
SZ_API_COMPTIME sz_f64_t sz_substrings_bm25_total_(sz_substrings_bm25_t const *parameters, sz_f64_t norm,
                                                   sz_f32_t const *needle_weights,
                                                   sz_substrings_frequencies_t *frequencies) {
    sz_size_t const touched_count = frequencies->touched_count, needles_count = frequencies->needles_count;
    sz_u32_t *ordered = frequencies->touched;
    sz_f64_t score = 0;
    sz_size_t index;
    // Past half the vocabulary the row itself is the cheaper index, and arrives already sorted.
    if (touched_count * 2 <= needles_count)
        ordered = sz_substrings_sort_needles_(frequencies->touched, touched_count, needles_count);
    else {
        sz_size_t written = 0, needle;
        for (needle = 0; needle != needles_count; ++needle)
            if (frequencies->counts[needle]) ordered[written++] = (sz_u32_t)needle;
    }
    for (index = 0; index != touched_count; ++index) {
        sz_u32_t const needle = ordered[index];
        score += sz_substrings_bm25_term(parameters, norm, needle_weights[needle], frequencies->counts[needle]);
        frequencies->counts[needle] = 0;
    }
    frequencies->touched_count = 0;
    return score;
}

/** BM25 scores through @p walks, which is every CPU tier's scoring verb. */
SZ_API_COMPTIME sz_status_t sz_substrings_bm25_scores_with_(sz_substrings_engine_t *engine,
                                                            sz_substrings_walks_t const *walks,
                                                            sz_sequence_t const *haystacks,
                                                            sz_f32_t const *document_lengths,
                                                            sz_substrings_bm25_t const *parameters,
                                                            sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                            sz_size_t scores_stride) {
    sz_substrings_host_arena_t const arena =
        sz_substrings_host_arena_(engine->needles_count, engine->max_source_match_bytes, engine->overlap_policy);
    sz_size_t const needles_count = engine->needles_count;
    sz_u32_t *const counters = (sz_u32_t *)((sz_u8_t *)engine->scratch + arena.bm25_counts);
    sz_substrings_frequencies_t frequencies;
    sz_size_t haystack_index, needle_index;
    sz_status_t checked = sz_substrings_bm25_check(parameters, needle_weights);
    if (checked == sz_success_k) checked = sz_substrings_stride_check_(scores_stride);
    if (checked != sz_success_k) return checked;
    sz_substrings_report_clear_(sz_substrings_report_(engine));
    if (needles_count == 0) {
        for (haystack_index = 0; haystack_index != haystacks->count; ++haystack_index)
            scores[haystack_index * scores_stride] = 0;
        return sz_success_k;
    }
    for (needle_index = 0; needle_index != needles_count; ++needle_index) counters[needle_index] = 0;
    frequencies.counts = counters, frequencies.touched = counters + needles_count;
    frequencies.touched_count = 0, frequencies.needles_count = needles_count;

    for (haystack_index = 0; haystack_index != haystacks->count; ++haystack_index) {
        sz_cptr_t const haystack = haystacks->get_start(haystacks->handle, haystack_index);
        sz_size_t const length = haystacks->get_length(haystacks->handle, haystack_index);
        sz_f64_t const norm = sz_substrings_bm25_norm(
            parameters, document_lengths ? (sz_f64_t)document_lengths[haystack_index] : (sz_f64_t)length);
        sz_substrings_bm25_count_(engine, walks, haystack, length, &frequencies);
        scores[haystack_index * scores_stride] =
            (sz_f32_t)sz_substrings_bm25_total_(parameters, norm, needle_weights, &frequencies);
    }
    return sz_success_k;
}

SZ_API_COMPTIME sz_status_t sz_substrings_bm25_scores_serial(sz_substrings_engine_t *engine,
                                                             sz_sequence_t const *haystacks,
                                                             sz_f32_t const *document_lengths,
                                                             sz_substrings_bm25_t const *parameters,
                                                             sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                             sz_size_t scores_stride) {
    sz_substrings_walks_t const walks = sz_substrings_walks_serial_();
    return sz_substrings_bm25_scores_with_(engine, &walks, haystacks, document_lengths, parameters, needle_weights,
                                           scores, scores_stride);
}

#pragma endregion Serial Backends

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_SUBSTRINGS_SERIAL_H_
