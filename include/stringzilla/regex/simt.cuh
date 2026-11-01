/**
 *  @file include/stringzilla/regex/simt.cuh
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The walks CUDA and ROCm share for regular expressions: one thread per chunk of the
 *      corpus marks the starts and walks the ends, and pointer doubling settles every chain that
 *      crosses a chunk's edge.
 *
 *  A haystack owns one slot per byte and one more ahead of them: a haystack with first slot @c V
 *  marks a match starting at byte @c p in slot <tt>V + p</tt>, which is also the slot of byte <tt>p - 1</tt>. The
 *  corpus spans the tape's bytes plus its count in slots, chunks cut that span evenly, so one long
 *  haystack and a million short ones fill the device alike, and a slot finds its haystack by binary
 *  search over the tape offsets. A round covers one segment of slots at a time, which bounds the
 *  candidates it holds by the segment's width. Where every match starts with a few literal bytes, a
 *  thread per slot marks the candidates from their bucket tables instead of the start-finding walk.
 *
 *  Three things cross a chunk's edge, and each crosses it by pointer doubling rather than by one
 *  thread walking on: what the start-finding walk holds where a chunk ends, when priming does not
 *  collapse within a few bytes; the end walks that reach the edge; and the leftmost-first chain,
 *  whose first candidate in every chunk is found from the chunk before it. A link that outlives one
 *  thread's share of bytes is walked on by a whole warp, each lane a piece, which takes its
 *  cross-lane steps as parameters each vendor's kernel binds, as <tt>levenshtein/simt.cuh</tt> does.
 *
 *  A DFA, cased or folded, settles a chunk's edge as a choice among a few states. The position
 *  automaton that replaces an exploding start-finding DFA, and the lookaround passes that fill the
 *  table of where lookarounds hold, step sets whose step distributes over union, so a chunk's walk
 *  maps the set handed to it affinely: between the sets it hands on from the least and from every
 *  input, the undecided elements are a boolean matrix of the undecided elements handed to it, and
 *  those matrices compose by pointer doubling too, as do the sets of states a lookaround body's
 *  own DFA steps. One-byte lookarounds need no pass, as the byte beside a slot decides them. Chunk
 *  edges sit on unit boundaries wherever walks step whole codepoints. A chunk that no reach
 *  settles is wide, and one walk settles the run of unsettled chunks around it from the nearest
 *  exact edge inside the segment.
 *
 *  Both DFAs test up to six lookarounds through blocks of columns, a step picking its block from
 *  the table of where they hold, a ring of the segment and the next. A DFA end walk whose next step
 *  would read past that window pauses there, grouped with the others in its state, and the host
 *  slides the window on until the groups end, so no DFA walk is cut short. The PikeVM that replaces
 *  an exploding end-finding DFA walks every candidate to its end without links, the fallback kept
 *  exact rather than fast, and refuses the call rather than read past the window; its lists and the
 *  sets live in the round block, one share per thread of a launch sized to them.
 *
 *  StringZilla's own kernels over the steps and walks here live in <tt>regex/kernels.cuh</tt>, which
 *  <tt>regex/cuda.cuh</tt> and <tt>regex/rocm.cuh</tt> size and launch; a product including only this header
 *  compiles none of them.
 *
 *  @sa c/stringzilla/regex.h
 *  @sa include/stringzilla/regex/metal.metal, the Metal sibling these mirror kernel for kernel
 */
#ifndef STRINGZILLA_REGEX_SIMT_CUH_
#define STRINGZILLA_REGEX_SIMT_CUH_

#include "stringzilla/types.cuh"

#include "stringzilla/regex/serial.h"
#include "stringzilla/utf8_uncased_fold/simt.cuh" // `sz_utf8_fold_next_simt_`, `sz_utf8_fold_prev_simt_`

#ifdef __cplusplus
extern "C" {
#endif

/*  Plans size a launch from the engine alone, as the host does before it: the shapes a round's
 *  buffers take, and the bytes a block stages and its passes carve. */
#pragma region Plans

/** Threads every kernel here launches with but the carry, which runs on one. */
enum { sz_regex_threads_per_block_simt_k = 256, sz_regex_carry_threads_per_block_simt_k = 1 };

/** Slots one chunk spans, one word of start bits. */
enum { sz_regex_chunk_slots_simt_k = 64 };

/** Round-block bytes per segment slot PikeVMs and position sets share out, one share per thread. */
enum { sz_regex_walk_slot_bytes_simt_k = 16 };

/** States a priming set holds; a column whose image is wider starts none. */
enum { sz_regex_priming_states_simt_k = 16 };

/** Undecided elements a chunk's edge set holds, one bit each of a composition's rows. */
enum { sz_regex_undecided_simt_k = 64 };

/** Bytes priming walks at most before a chunk settles its entry by composition instead. */
enum { sz_regex_priming_bytes_simt_k = 64 };

/** Bytes priming or a set flow reaches upstream of a chunk at most, beyond which one walk settles
 *  its whole piece. */
enum { sz_regex_widening_bytes_simt_k = 4096 };

/** Bytes a link walks on its own before a warp walks the rest of it together. */
enum { sz_regex_link_bytes_simt_k = 1024 };

/** Bytes each lane of a warp walking a link together takes per round. */
enum { sz_regex_piece_bytes_simt_k = 128 };

/** Hops per round of pointer doubling, so each round reaches this many times as far as the last;
 *  many, as each round costs a launch while most chains end within a few hops. */
enum { sz_regex_round_hops_simt_k = 32 };

/** Slots between the checkpoints where end walks leave their state for later walks to stop at. */
enum { sz_regex_checkpoint_slots_simt_k = 16 };

/** Bytes a run loads ahead together, so their latency overlaps the walk over the ones before. */
enum { sz_regex_run_bytes_simt_k = 16 };

/** Cells of a transition table a walk reads from shared memory before it reads global memory. */
enum { sz_regex_staged_cells_simt_k = 4096 };

/** States a body's DFA flow tells apart at most, a bit each of its sets. */
enum { sz_regex_body_states_simt_k = 256 };

/** The pattern, link, candidate and chunk id nothing takes. */
static sz_u32_t const sz_regex_none_simt_k = 0xFFFFFFFFu;

/** The bit marking a walk's id as a group of paused walks rather than a link. */
static sz_u32_t const sz_regex_group_simt_k = 0x80000000u;

/** The round's counters, as indices: links listed, groups opened, walks handed to warps, walks
 *  paused at the window's end, and the flag a walk that cannot pause sets where it would read the
 *  lookaround table past it. */
enum {
    sz_regex_links_counter_simt_k = 0,
    sz_regex_groups_counter_simt_k = 1,
    sz_regex_handed_counter_simt_k = 2,
    sz_regex_paused_counter_simt_k = 3,
    sz_regex_refused_counter_simt_k = 4,
};

/** What a walk in some state at a checkpoint ends with: a link to follow, and the match
 *  it found past the checkpoint, its end a slot offset into the segment. A zero state
 *  marks a free checkpoint. */
typedef struct sz_regex_checkpoint_simt_t {
    sz_u32_t state, link, pattern, end;
} sz_regex_checkpoint_simt_t;

/** An end walk's outcome, a link's or a group's: the link or group whose match wins when it has
 *  one, else this match, its end an absolute slot. A link or group holds the end-finding state it
 *  resumes in until it walks. */
typedef struct sz_regex_result_simt_t {
    sz_u32_t link, pattern;
    sz_u64_t end;
} sz_regex_result_simt_t;

/** A link or group handed to a warp or paused at the end of the lookaround table's window: its id,
 *  the state it holds at the boundary in @c slot, and a zero state once it ended. */
typedef struct sz_regex_continuation_simt_t {
    sz_u32_t link, state;
    sz_u64_t slot;
} sz_regex_continuation_simt_t;

/** The states the start-finding walk may hold at a chunk's right edge when priming did not
 *  collapse, and the state each would leave the chunk in. */
typedef struct sz_regex_frontier_simt_t {
    sz_u32_t count;
    sz_u32_t states[sz_regex_priming_states_simt_k];
    sz_u32_t exits[sz_regex_priming_states_simt_k];
} sz_regex_frontier_simt_t;

/** Which of its frontier's states a chunk enters with: the @c known index, or the frontier of chunk
 *  @c target together with each of that frontier's indices mapped into this one's. */
typedef struct sz_regex_composition_simt_t {
    sz_u32_t target, known;
    sz_u8_t indices[sz_regex_priming_states_simt_k];
} sz_regex_composition_simt_t;

/** The walk a round's set kernels run: the position automaton right to left, or one level's
 *  lookbehind pass left to right or lookahead pass right to left. */
typedef enum sz_regex_flow_simt_t {
    sz_regex_flow_positions_simt_k = 0,
    sz_regex_flow_behind_simt_k = 1,
    sz_regex_flow_ahead_simt_k = 2,
} sz_regex_flow_simt_t;

/** Elements of the set a chunk hands on its walk alone leaves undecided, by index into the set. */
typedef struct sz_regex_undecided_simt_t {
    sz_u32_t count;
    sz_u32_t indices[sz_regex_undecided_simt_k];
} sz_regex_undecided_simt_t;

/** Which undecided elements a chunk hands on, as an affine map of those chunk @c target hands
 *  it: per bit of those, the bits it sets, and the bits set either way; without a target,
 *  @c fixed is the answer. */
typedef struct sz_regex_affine_simt_t {
    sz_u32_t target, rows_count;
    sz_u64_t fixed;
    sz_u64_t rows[sz_regex_undecided_simt_k];
} sz_regex_affine_simt_t;

/** Where lookarounds hold, one entry of look bits per slot of the window <tt>[begin, end)</tt>, from a
 *  ring of two segments' halves indexed by <tt>slot & mask</tt>; null without any. A walk that cannot
 *  pause past the window raises @c refused instead, which fails the call. */
typedef struct sz_regex_looks_simt_t {
    sz_u32_t *table, *refused;
    sz_size_t begin, end, mask;
} sz_regex_looks_simt_t;

/** What one segment hands the next: where the haystack straddling them resumes, whether an empty
 *  match there is refused, the matches kept so far, the last kept match, the drift a rewrite or the
 *  rows an extraction made so far, the kept matches of the extracted pattern so far, how many of
 *  them preceded the straddling haystack, and the contains bits of the partial words at either end
 *  of the bitmap, the first word's in the low half and the last word's in the high half. */
typedef struct sz_regex_carry_simt_t {
    sz_u64_t cursor, refusing, running, last_start, last_end, summed, counted, straddle, contained_edges;
} sz_regex_carry_simt_t;

/**
 *  @brief What one round's kernels read and write, every pointer one the device addresses.
 *
 *  The outputs of the verb a call serves are set and the others null: contained bits, matches,
 *  counts, a rewrite or rows. The set kernels read which flow they run, the lookarounds it walks,
 *  the lookaround whose body's DFA it steps instead of the NFA, and the place its sets take in the
 *  share of the walk.
 *
 *  Groups of walks paused at the window's end live in the checkpoints' two halves, dead once the
 *  links settle, and are found through the leaders, one per forward state; the last slide's start
 *  at @c groups_begin.
 */
typedef struct sz_regex_round_simt_t {
    sz_u32_t const *reverse_images, *forward_images;
    sz_u32_t *entries, *edges, *counters, *flags, *leaders;
    sz_regex_carry_simt_t *carry;
    sz_size_t *scan_chain, *chunk_offsets, *chunk_kept, *chunk_sums, *chunk_covers, *chunk_haystacks;
    sz_u64_t *scan_states;
    sz_regex_frontier_simt_t *frontiers;
    sz_regex_composition_simt_t *compositions, *compositions_next;
    sz_u64_t *starts, *handoffs, *handoff_edges;
    sz_regex_undecided_simt_t *undecided;
    sz_regex_affine_simt_t *affines, *affines_next;
    sz_regex_looks_simt_t looks;
    sz_regex_checkpoint_simt_t *checkpoints;
    sz_u32_t *slots, *links_list, *nexts, *exits, *exits_next, *keep, *kept;
    sz_regex_result_simt_t *results, *links, *links_next, *groups, *groups_next;
    sz_size_t *drifts;
    sz_u8_t *walks;
    sz_u1x8_t *contained;
    sz_size_t *matches_offsets, *counts, *offsets, *rows_offsets;
    sz_match_t *matches, *rows;
    sz_ptr_t target;
    sz_strings_view_t replacements;
    sz_size_t matches_capacity, rows_capacity, target_capacity, counts_stride, pattern_index;
    sz_size_t slots_count, segment, segment_begin, segment_end, chunk_shift, chunks_count, round, rounds;
    sz_size_t walks_count, walk_bytes, walk_sets, pike_slots;
    sz_size_t set_words, bodies_begin, bodies_count;
    sz_u32_t forward_cells, reverse_cells, staged_cells, storing;
    sz_u32_t flow, members, negated, body, body_states;
    sz_u64_t body_reach[sz_regex_body_states_simt_k / 64];
    sz_u32_t groups_begin, groups_capacity, scans;
} sz_regex_round_simt_t;

/** One automaton's transitions as a walk reads them: the class map, the first column of each block
 *  of lookarounds and the rows' head staged in shared memory, the rest of the rows in global
 *  memory, the round's lookaround table whose entries pick the blocks, null without lookarounds,
 *  and the automaton's column images where a walk primes, else null. The table is reached through
 *  the round, as NVVM takes the pointers of a copy held beside staged ones for shared memory. */
typedef struct sz_regex_table_simt_t {
    sz_u8_t const *classes;
    sz_u32_t const *blocks, *head, *rows, *images;
    sz_regex_looks_simt_t const *looks;
    sz_u32_t head_cells, edge_column;
} sz_regex_table_simt_t;

/** Blocks of lookaround columns a row holds at most, one per set of the columned lookarounds, each
 *  staged by its first column. */
enum { sz_regex_blocks_simt_k = 1 << sz_regex_columned_lookarounds_k };

/** Bytes of shared memory the class map and the blocks take ahead of the staged rows, as the
 *  host sizes it. */
enum { sz_regex_staged_bytes_simt_k = 256 + sz_regex_blocks_simt_k * sizeof(sz_u32_t) };

/** Bytes of dynamic shared memory @ref sz_regex_stage_table_simt_ fills for @p cells of an
 *  automaton's rows: the class map and the blocks, then the rows' head. */
STRINGZILLA_INLINE sz_size_t sz_regex_staged_bytes_simt_(sz_regex_engine_t const *engine, sz_u32_t cells) {
    sz_unused_(engine);
    return sz_regex_staged_bytes_simt_k + (sz_size_t)cells * sizeof(sz_u32_t);
}

/** Bytes of one automaton's column images: a count per column of its rows, then a priming set per
 *  column, rounded up to 16 bytes; none without either automaton. */
STRINGZILLA_INLINE sz_size_t sz_regex_images_bytes_simt_(sz_regex_engine_t const *engine) {
    sz_size_t const bytes = sz_regex_stride_(engine->classes_count, engine->lookarounds_count) *
                            (1 + sz_regex_priming_states_simt_k) * sizeof(sz_u32_t);
    if (!engine->forward_rows && !engine->reverse_rows) return 0;
    return sz_size_divide_round_up(bytes, 16) * 16;
}

/** Where a run of chunks leaves the start-finding walk, and whether a match starts inside it, for
 *  each state it may enter with: a priming frontier and each state's exit, or, with no entries,
 *  the one exit whatever the entry, a bit of @c founds per entry. */
typedef struct sz_regex_exits_simt_t {
    sz_u32_t count, founds;
    sz_u32_t entries[sz_regex_priming_states_simt_k], exits[sz_regex_priming_states_simt_k];
} sz_regex_exits_simt_t;

/** Bytes of text a block stages for one tile of chunks: the slots of its chunks, a byte before them
 *  and the alignment of its first granule, in rows of 64 bytes padded by one word, so lanes a chunk
 *  apart read distinct banks. */
enum {
    sz_regex_tile_bytes_simt_k = (int)sz_regex_threads_per_block_simt_k * (int)sz_regex_chunk_slots_simt_k + 17,
    sz_regex_staged_text_bytes_simt_k = (sz_regex_tile_bytes_simt_k + 63) / 64 * 68,
};

/** One haystack, as a walk reads it: its bytes, its length, and its first slot. */
typedef struct sz_regex_haystack_simt_t {
    sz_u8_t const *text;
    sz_size_t length, first;
} sz_regex_haystack_simt_t;

/** A PikeVM over the forward NFA carved out of one share of the round block's walks, as
 *  @c sz_regex_pike_t is, reading the round's lookaround table for the haystack whose first
 *  slot is @c first. */
typedef struct sz_regex_pike_simt_t {
    sz_u32_t *dense, *sparse, *entered;
    sz_size_t *dense_slots, *entered_slots, *current, *matched, *frames;
    sz_size_t count, entered_count, slots_count, end, first;
    sz_regex_looks_simt_t const *looks;
    sz_u32_t pattern;
    sz_bool_t found;
} sz_regex_pike_simt_t;

/** Bytes one PikeVM keeping @p slots_count capture slots takes of the walks, which
 *  @ref sz_regex_pike_carve_simt_ lays out. */
STRINGZILLA_INLINE sz_size_t sz_regex_pike_bytes_simt_(sz_regex_engine_t const *engine, sz_size_t slots_count) {
    sz_size_t const states = engine->nfa_states_count, entered = engine->nfa_links_count + 1;
    sz_size_t const bytes = ((states + entered + 2) * slots_count + 2 * (states + entered)) * sizeof(sz_size_t) +
                            (2 * states + entered) * sizeof(sz_u32_t);
    return (bytes + 15) / 16 * 16;
}

/** Where the lookaround bodies begin among the NFA states: past the last pattern's match state, as
 *  the patterns compile ahead of the bodies and each closes with its own. */
STRINGZILLA_INLINE sz_size_t sz_regex_bodies_begin_simt_(sz_regex_engine_t const *engine) {
    sz_size_t state, matches = 0;
    for (state = 0; state != engine->nfa_states_count; ++state)
        if (engine->nfa_states[state].kind == sz_regex_nfa_match_k && ++matches == engine->patterns_count)
            return state + 1;
    return engine->nfa_states_count;
}

/** Levels of nested lookarounds, each a pass behind and a pass ahead over the levels
 *  below; zero without any. */
STRINGZILLA_INLINE sz_u32_t sz_regex_levels_simt_(sz_regex_engine_t const *engine) {
    sz_u32_t levels = 0, index;
    for (index = 0; index != engine->lookarounds_count; ++index)
        levels = sz_max_of_two(levels, engine->lookarounds[index].level);
    return levels;
}

/** Words of a lookaround pass's set: a bit per body state, then a bit per lookaround to record. */
STRINGZILLA_INLINE sz_size_t sz_regex_looks_words_simt_(sz_regex_engine_t const *engine) {
    if (!engine->lookarounds_count) return 0;
    return (engine->nfa_states_count - sz_regex_bodies_begin_simt_(engine) + engine->lookarounds_count + 63) / 64;
}

/** The look bits of the lookarounds of @p level whose bodies look behind when @p behind, else
 *  ahead, and in @p negated those of them that are negated. */
STRINGZILLA_INLINE sz_u32_t sz_regex_members_simt_(sz_regex_engine_t const *engine, sz_u32_t level, sz_bool_t behind,
                                                   sz_u32_t *negated) {
    sz_u32_t members = 0, index;
    *negated = 0;
    for (index = 0; index != engine->lookarounds_count; ++index) {
        sz_regex_lookaround_t const lookaround = engine->lookarounds[index];
        if (lookaround.level != level || (sz_bool_t)(lookaround.kind >= sz_regex_lookbehind_k) != behind) continue;
        members |= (sz_u32_t)sz_regex_look_lookaround_k << index;
        if (lookaround.kind & 1) *negated |= (sz_u32_t)sz_regex_look_lookaround_k << index;
    }
    return members;
}

/** The look bits of the lookarounds the byte beside a position decides, which the singles kernel
 *  writes before any pass and no pass walks. */
STRINGZILLA_INLINE sz_u32_t sz_regex_singles_simt_(sz_regex_engine_t const *engine) {
    return engine->looks_bytes ? engine->looks_bytes[0] : 0;
}

/**
 *  @brief The states the body DFA of lookaround @p index holds over bytes, from its start on, which
 *      a flow stepping it tells apart.
 *  @param[out] reach A bit per such state, or null.
 *  @return One past the highest, or zero where flows step the NFA instead: for a body without rows,
 *      of an uncased engine, or past @ref sz_regex_body_states_simt_k states.
 */
STRINGZILLA_INLINE sz_u32_t sz_regex_body_reach_simt_(sz_regex_engine_t const *engine, sz_u32_t index,
                                                      sz_u64_t *reach) {
    sz_regex_body_t const *const body = engine->bodies ? engine->bodies + index : STRINGZILLA_NULL;
    sz_u32_t const columns = engine->classes_count + 1;
    sz_u64_t reached[sz_regex_body_states_simt_k / 64] = {0};
    sz_u32_t stack[sz_regex_body_states_simt_k];
    sz_u32_t top = 0, highest = 0, column, word;
    if (!body || !body->rows || engine->case_sensitivity != sz_cased_k) return 0;
    stack[top++] = body->start / columns,
    reached[body->start / columns / 64] |= (sz_u64_t)1 << (body->start / columns % 64);
    // The edge column ends a walk, so the states past it are never stepped on.
    while (top) {
        sz_u32_t const state = stack[--top];
        highest = sz_max_of_two(highest, state);
        for (column = 0; column + 1 != columns; ++column) {
            sz_u32_t const next = body->rows[state * columns + column] / columns;
            if (next >= sz_regex_body_states_simt_k) return 0;
            if (reached[next / 64] >> (next % 64) & 1) continue;
            reached[next / 64] |= (sz_u64_t)1 << (next % 64), stack[top++] = next;
        }
    }
    for (word = 0; reach && word != sz_regex_body_states_simt_k / 64; ++word) reach[word] = reached[word];
    return highest + 1;
}

/** The look bits of the lookarounds whose bodies flows step as DFAs. */
STRINGZILLA_INLINE sz_u32_t sz_regex_body_dfas_simt_(sz_regex_engine_t const *engine) {
    sz_u32_t dfas = 0, index;
    for (index = 0; index != engine->lookarounds_count; ++index)
        if (sz_regex_body_reach_simt_(engine, index, STRINGZILLA_NULL))
            dfas |= (sz_u32_t)sz_regex_look_lookaround_k << index;
    return dfas;
}

/** Slots each thread of the prefixes mark kernel tests per tile, so a tile's haystack searches and
 *  barriers spread over many slots. */
enum { sz_regex_prefix_slots_simt_k = 8 };

/** Chunks of one tile of the prefixes mark kernel. */
enum {
    sz_regex_prefix_chunks_simt_k = (int)sz_regex_threads_per_block_simt_k * (int)sz_regex_prefix_slots_simt_k /
        (int)sz_regex_chunk_slots_simt_k,
};

/** Writes column @p column of @p rows' images: the distinct live states any of its @p states_count
 *  states steps to on it, or one more than a priming set holds when they do not fit, every block's
 *  columns alike; the host and a product's kernel alike fill them column by column. */
STRINGZILLA_CONSTEXPR void sz_regex_column_image_simt_(sz_regex_engine_t const *engine, sz_u32_t const *rows,
                                                       sz_size_t states_count, sz_size_t column, sz_u32_t *images) {
    sz_size_t const stride = sz_regex_stride_(engine->classes_count, engine->lookarounds_count);
    sz_u32_t *const states = images + stride + column * sz_regex_priming_states_simt_k;
    sz_u32_t count = 0;
    sz_size_t state;
    if (column % (engine->classes_count + 1) != engine->classes_count)
        for (state = 1; state != states_count && count <= sz_regex_priming_states_simt_k; ++state) {
            sz_u32_t const next = rows[state * stride + column];
            sz_u32_t index = 0;
            while (index != count && states[index] != next) ++index;
            if (!next || index != count) continue;
            if (count < sz_regex_priming_states_simt_k) states[count] = next;
            ++count;
        }
    images[column] = count;
}

/**
 *  @brief A copy of @p engine reading its tables from @p device_block, for a product that keeps
 *      them in device memory it owns.
 *  @param[in] device_block The @c engine->memory_bytes bytes of @c engine->memory, copied.
 *
 *  A kernel takes the copy by value, as it takes a compiled engine whose tables it reads in place;
 *  @c scratch stays where it is, and each lookaround body's rows, reached through @c bodies, keep
 *  pointing into @c engine->memory, as no walk reads them. The group names move as the u32 tape the
 *  engine always compiles them into.
 */
STRINGZILLA_INLINE sz_regex_engine_t sz_regex_engine_rebase_simt_(sz_regex_engine_t const *engine, void *device_block) {
    sz_regex_engine_t moved = *engine;
    moved.byte_to_class = (sz_u8_t const *)sz_pointer_rebased_(engine->byte_to_class, engine->memory,
                                                               engine->memory_bytes, device_block);
    moved.reverse_rows = (sz_u32_t const *)sz_pointer_rebased_(engine->reverse_rows, engine->memory,
                                                               engine->memory_bytes, device_block);
    moved.forward_rows = (sz_u32_t const *)sz_pointer_rebased_(engine->forward_rows, engine->memory,
                                                               engine->memory_bytes, device_block);
    moved.forward_patterns = (sz_u32_t const *)sz_pointer_rebased_(engine->forward_patterns, engine->memory,
                                                                   engine->memory_bytes, device_block);
    moved.forward_escapes = (sz_byteset_t const *)sz_pointer_rebased_(engine->forward_escapes, engine->memory,
                                                                      engine->memory_bytes, device_block);
    moved.prefixes = (sz_regex_prefixes_t const *)sz_pointer_rebased_(engine->prefixes, engine->memory,
                                                                      engine->memory_bytes, device_block);
    moved.positions_reach = (sz_u64_t const *)sz_pointer_rebased_(engine->positions_reach, engine->memory,
                                                                  engine->memory_bytes, device_block);
    moved.positions_shifts = (sz_u64_t const *)sz_pointer_rebased_(engine->positions_shifts, engine->memory,
                                                                   engine->memory_bytes, device_block);
    moved.positions_groups = (sz_u64_t const *)sz_pointer_rebased_(engine->positions_groups, engine->memory,
                                                                   engine->memory_bytes, device_block);
    moved.positions_all = (sz_u64_t const *)sz_pointer_rebased_(engine->positions_all, engine->memory,
                                                                engine->memory_bytes, device_block);
    moved.positions_initial = (sz_u64_t const *)sz_pointer_rebased_(engine->positions_initial, engine->memory,
                                                                    engine->memory_bytes, device_block);
    moved.positions_ascii = (sz_u32_t const *)sz_pointer_rebased_(engine->positions_ascii, engine->memory,
                                                                  engine->memory_bytes, device_block);
    moved.positions_ranges = (sz_u32_t const *)sz_pointer_rebased_(engine->positions_ranges, engine->memory,
                                                                   engine->memory_bytes, device_block);
    moved.nfa_states = (sz_regex_nfa_state_t const *)sz_pointer_rebased_(engine->nfa_states, engine->memory,
                                                                         engine->memory_bytes, device_block);
    moved.nfa_links = (sz_regex_nfa_link_t const *)sz_pointer_rebased_(engine->nfa_links, engine->memory,
                                                                       engine->memory_bytes, device_block);
    moved.patterns_starts = (sz_u32_t const *)sz_pointer_rebased_(engine->patterns_starts, engine->memory,
                                                                  engine->memory_bytes, device_block);
    moved.patterns_groups = (sz_u32_t const *)sz_pointer_rebased_(engine->patterns_groups, engine->memory,
                                                                  engine->memory_bytes, device_block);
    moved.lookarounds = (sz_regex_lookaround_t const *)sz_pointer_rebased_(engine->lookarounds, engine->memory,
                                                                           engine->memory_bytes, device_block);
    moved.bodies = (sz_regex_body_t const *)sz_pointer_rebased_(engine->bodies, engine->memory, engine->memory_bytes,
                                                                device_block);
    moved.looks_bytes = (sz_u32_t const *)sz_pointer_rebased_(engine->looks_bytes, engine->memory, engine->memory_bytes,
                                                              device_block);
    sz_assert_(engine->groups_names.layout == sz_strings_u32_tape_k);
    moved.groups_names.data.u32_tape.data = (sz_cptr_t)sz_pointer_rebased_(
        engine->groups_names.data.u32_tape.data, engine->memory, engine->memory_bytes, device_block);
    moved.groups_names.data.u32_tape.offsets = (sz_u32_t const *)sz_pointer_rebased_(
        engine->groups_names.data.u32_tape.offsets, engine->memory, engine->memory_bytes, device_block);
    moved.memory = device_block;
    return moved;
}

#pragma endregion Plans

#if STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

/*  Steps advance one byte, run, unit or set, or stage what a block shares, for the walks and the
 *  kernels below. The round's own passes, which read its buffers, stay steps rather than walks. */
#pragma region Steps

/** The lookaround table of a walk with none, which every block lookup reads as absent. */
STRINGZILLA_DEVICE_CONSTANT sz_regex_looks_simt_t const sz_regex_no_looks_simt_ = {STRINGZILLA_NULL, STRINGZILLA_NULL,
                                                                                   0, 0, 0};

/** Stages @p cells of @p rows, the class map and the blocks into dynamic shared memory, for walks
 *  with no lookarounds that prime from @p images, or null; every thread of the block takes part,
 *  so a kernel calls it before any thread can leave. */
STRINGZILLA_DEVICE sz_regex_table_simt_t sz_regex_stage_table_simt_(sz_regex_engine_t const *engine,
                                                                    sz_u32_t const *rows, sz_u32_t const *images,
                                                                    sz_u32_t cells) {
    extern __shared__ sz_u8_t sz_regex_staged_simt_[];
    sz_u32_t *const blocks = (sz_u32_t *)(sz_regex_staged_simt_ + 256);
    sz_u32_t *const head = (sz_u32_t *)(sz_regex_staged_simt_ + sz_regex_staged_bytes_simt_k);
    sz_regex_table_simt_t table;
    sz_u32_t index;
    table.classes = sz_regex_staged_simt_, table.blocks = blocks, table.head = head;
    table.rows = rows, table.images = images, table.head_cells = cells;
    table.looks = &sz_regex_no_looks_simt_, table.edge_column = engine->classes_count;
    for (index = threadIdx.x; index < 256; index += blockDim.x)
        sz_regex_staged_simt_[index] = engine->byte_to_class[index];
    for (index = threadIdx.x; index < sz_regex_blocks_simt_k; index += blockDim.x)
        blocks[index] = (sz_u32_t)sz_regex_looks_block_(engine->classes_count, engine->lookarounds_count,
                                                        index * sz_regex_look_lookaround_k);
    for (index = threadIdx.x; index < table.head_cells; index += blockDim.x) head[index] = rows[index];
    __syncthreads();
    return table;
}

/** Stages a round's @p cells of @p rows as @ref sz_regex_stage_table_simt_ does, at most the
 *  round's own staged cells, its blocks picked by the round's lookaround table. */
STRINGZILLA_DEVICE sz_regex_table_simt_t sz_regex_stage_simt_(sz_regex_engine_t const *engine,
                                                              sz_regex_round_simt_t const *round, sz_u32_t const *rows,
                                                              sz_u32_t cells) {
    sz_regex_table_simt_t table = sz_regex_stage_table_simt_(engine, rows, STRINGZILLA_NULL,
                                                             sz_min_of_two(cells, round->staged_cells));
    table.looks = &round->looks;
    return table;
}

/** A table staging only the class map and the blocks, for the kernels that walk the forward
 *  automaton rarely. */
STRINGZILLA_DEVICE sz_regex_table_simt_t sz_regex_direct_simt_(sz_regex_engine_t const *engine,
                                                               sz_regex_round_simt_t const *round) {
    return sz_regex_stage_simt_(engine, round, engine->forward_rows, 0);
}

/** The state @p state moves to in @p column. */
STRINGZILLA_DEVICE sz_u32_t sz_regex_move_simt_(sz_regex_table_simt_t const *table, sz_u32_t state, sz_u32_t column) {
    sz_u32_t const cell = state + column;
    return cell < table->head_cells ? table->head[cell] : __ldg(table->rows + cell);
}

/** The first column of the block the lookarounds at @p slot select, zero without lookarounds. */
STRINGZILLA_DEVICE sz_u32_t sz_regex_block_at_simt_(sz_regex_table_simt_t const *table, sz_size_t slot) {
    return table->looks->table ? table->blocks[table->looks->table[slot & table->looks->mask] /
                                               sz_regex_look_lookaround_k % sz_regex_blocks_simt_k]
                               : 0;
}

/** Whether a walk would read the lookarounds at @p slot past the table's window, where
 *  it pauses or refuses. */
STRINGZILLA_DEVICE sz_bool_t sz_regex_past_simt_(sz_regex_looks_simt_t const *looks, sz_size_t slot) {
    return (sz_bool_t)(looks->table && slot >= looks->end);
}

/** The state @p state moves to over the haystack-edge column at the boundary in @p slot. */
STRINGZILLA_DEVICE sz_u32_t sz_regex_step_edge_simt_(sz_regex_table_simt_t const *table, sz_u32_t state,
                                                     sz_size_t slot) {
    return sz_regex_move_simt_(table, state, sz_regex_block_at_simt_(table, slot) + table->edge_column);
}

/**
 *  @brief Walks @p state forward over the @p count bytes at @p text, at most a run, stopping where
 *      it dies; the byte at @p text crosses the boundary in @p slot.
 *  @return The bytes consumed; @p last and @p last_offset take the run's last accepting state and
 *      the offset of the byte that reached it.
 */
STRINGZILLA_DEVICE sz_u32_t sz_regex_run_simt_(sz_regex_table_simt_t const *table, sz_u32_t *state, sz_u8_t const *text,
                                               sz_size_t slot, sz_u32_t count, sz_u32_t accepts, sz_u32_t *last,
                                               sz_u32_t *last_offset) {
    sz_u32_t columns[sz_regex_run_bytes_simt_k];
    sz_u32_t index;
#pragma unroll
    for (index = 0; index != sz_regex_run_bytes_simt_k; ++index)
        columns[index] = index < count ? table->classes[text[index]] + sz_regex_block_at_simt_(table, slot + index) : 0;
#pragma unroll
    for (index = 0; index != sz_regex_run_bytes_simt_k; ++index) {
        if (index == count) break;
        *state = sz_regex_move_simt_(table, *state, columns[index]);
        if (*state >= accepts) *last = *state, *last_offset = index;
        if (*state == 0) return index + 1;
    }
    return count;
}

/**
 *  @brief Steps @p state over the first @p count of a run's @p columns, taken right to left.
 *  @return A bit per column whose step left the walk at or past @p accepts, bit 0 the last.
 */
STRINGZILLA_DEVICE sz_u32_t sz_regex_steps_back_simt_(sz_regex_table_simt_t const *table, sz_u32_t *state,
                                                      sz_u32_t const *columns, sz_u32_t count, sz_u32_t accepts) {
    sz_u32_t mask = 0, index;
#pragma unroll
    for (index = 0; index != sz_regex_run_bytes_simt_k; ++index) {
        if (index == count) break;
        *state = sz_regex_move_simt_(table, *state, columns[index]);
        if (*state >= accepts) mask |= 1u << (count - 1 - index);
    }
    return mask;
}

/**
 *  @brief Walks @p state right to left over the @p count bytes before @p text_end, at most a run,
 *      the last of them crossing the boundary in @p slot.
 *  @return A bit per byte whose step left the walk at or past @p accepts, bit 0 the leftmost byte.
 */
STRINGZILLA_DEVICE sz_u32_t sz_regex_run_back_simt_(sz_regex_table_simt_t const *table, sz_u32_t *state,
                                                    sz_u8_t const *text_end, sz_size_t slot, sz_u32_t count,
                                                    sz_u32_t accepts) {
    sz_u32_t columns[sz_regex_run_bytes_simt_k];
    sz_u32_t index;
#pragma unroll
    for (index = 0; index != sz_regex_run_bytes_simt_k; ++index)
        columns[index] = index < count
                             ? table->classes[text_end[-1 - (int)index]] + sz_regex_block_at_simt_(table, slot - index)
                             : 0;
    return sz_regex_steps_back_simt_(table, state, columns, count, accepts);
}

/** Byte @p position of the text @ref sz_regex_stage_text_simt_ staged. */
STRINGZILLA_DEVICE sz_u8_t sz_regex_staged_byte_simt_(sz_u8_t const *staged, sz_size_t position) {
    return staged[position + position / 64 * 4];
}

/**
 *  @brief Copies the bytes from @p begin to @p end of a tape spanning @p tape to @p tape_end into
 *      @p staged, every thread of the block taking 16-byte granules in turn, so loads coalesce.
 *  @return The address staged position zero stands for, @p begin rounded down to a granule.
 */
STRINGZILLA_DEVICE sz_u8_t const *sz_regex_stage_text_simt_(sz_u8_t *staged, sz_u8_t const *begin, sz_u8_t const *end,
                                                            sz_u8_t const *tape, sz_u8_t const *tape_end) {
    sz_u8_t const *const origin = (sz_u8_t const *)((sz_size_t)begin & ~(sz_size_t)15);
    sz_size_t const bytes = (sz_size_t)(end - origin);
    sz_size_t granule, byte;
    for (granule = (sz_size_t)threadIdx.x * 16; granule < bytes; granule += (sz_size_t)blockDim.x * 16) {
        sz_u32_t *const target = (sz_u32_t *)(staged + granule + granule / 64 * 4);
        if (origin + granule >= tape && origin + granule + 16 <= tape_end) {
            uint4 const loaded = *(uint4 const *)(origin + granule);
            target[0] = loaded.x, target[1] = loaded.y, target[2] = loaded.z, target[3] = loaded.w;
        }
        else
            for (byte = 0; byte != 16; ++byte)
                if (origin + granule + byte >= tape && origin + granule + byte < tape_end)
                    staged[granule + granule / 64 * 4 + byte] = origin[granule + byte];
    }
    return origin;
}

/** @ref sz_regex_run_back_simt_ over text @ref sz_regex_stage_text_simt_ staged, @p end being the
 *  staged position of the byte past the run. */
STRINGZILLA_DEVICE sz_u32_t sz_regex_run_back_staged_simt_(sz_regex_table_simt_t const *table, sz_u32_t *state,
                                                           sz_u8_t const *staged, sz_size_t end, sz_size_t slot,
                                                           sz_u32_t count, sz_u32_t accepts) {
    sz_u32_t columns[sz_regex_run_bytes_simt_k];
    sz_u32_t index;
#pragma unroll
    for (index = 0; index != sz_regex_run_bytes_simt_k; ++index)
        columns[index] = index < count ? table->classes[sz_regex_staged_byte_simt_(staged, end - 1 - index)] +
                                             sz_regex_block_at_simt_(table, slot - index)
                                       : 0;
    return sz_regex_steps_back_simt_(table, state, columns, count, accepts);
}

/** The pattern a forward accepting @p state reports, as @c sz_regex_walk_end_serial_ finds it. */
STRINGZILLA_DEVICE sz_u32_t sz_regex_pattern_simt_(sz_regex_engine_t const *engine, sz_u32_t state) {
    return engine->forward_patterns[(state - engine->forward_accepts_begin) /
                                    sz_regex_stride_(engine->classes_count, engine->lookarounds_count)];
}

/** The forward start column for the byte before a match: a line feed, a word byte, or another. */
STRINGZILLA_DEVICE sz_u32_t sz_regex_byte_kind_simt_(sz_u8_t byte) {
    sz_bool_t const word = (sz_bool_t)((byte >= '0' && byte <= '9') || (byte >= 'A' && byte <= 'Z') ||
                                       (byte >= 'a' && byte <= 'z') || byte == '_');
    return byte == '\n' ? 1 : word ? 2 : 3;
}

/** The lookarounds holding at @p slot, from the ring whose halves hold the window's two segments,
 *  none outside the window or without a table. */
STRINGZILLA_DEVICE sz_u32_t sz_regex_looks_at_simt_(sz_regex_looks_simt_t const *looks, sz_size_t slot) {
    return looks->table && slot >= looks->begin && slot < looks->end ? looks->table[slot & looks->mask] : 0;
}

/** Walks @p state right to left from byte @p at of @p haystack to byte @p stop, marking nothing. */
STRINGZILLA_DEVICE sz_u32_t sz_regex_walk_back_simt_(sz_regex_table_simt_t const *table, sz_u32_t state,
                                                     sz_regex_haystack_simt_t const *haystack, sz_size_t at,
                                                     sz_size_t stop) {
    while (at != stop) {
        sz_u32_t const count = (sz_u32_t)sz_min_of_two(at - stop, (sz_size_t)sz_regex_run_bytes_simt_k);
        sz_regex_run_back_simt_(table, &state, haystack->text + at, haystack->first + at, count, sz_regex_none_simt_k);
        at -= count;
    }
    return state;
}

/** The slot haystack @p index starts at, the one its empty-prefix match is marked in. */
STRINGZILLA_DEVICE sz_size_t sz_regex_first_slot_simt_(sz_strings_view_t haystacks, sz_size_t index) {
    return (sz_size_t)(sz_strings_start_simt_(haystacks, index) - sz_strings_start_simt_(haystacks, 0)) + index;
}

STRINGZILLA_DEVICE sz_regex_haystack_simt_t sz_regex_haystack_simt_(sz_strings_view_t haystacks, sz_size_t index) {
    sz_regex_haystack_simt_t haystack;
    haystack.text = (sz_u8_t const *)sz_strings_start_simt_(haystacks, index);
    haystack.length = sz_strings_length_simt_(haystacks, index);
    haystack.first = sz_regex_first_slot_simt_(haystacks, index);
    return haystack;
}

/** The haystack holding @p slot, galloping up from @p low, a haystack starting at or before it. */
STRINGZILLA_DEVICE sz_size_t sz_regex_haystack_at_simt_(sz_strings_view_t haystacks, sz_size_t slot, sz_size_t low) {
    sz_size_t step = 1, high;
    while (low + step < haystacks.count && sz_regex_first_slot_simt_(haystacks, low + step) <= slot)
        low += step, step *= 2;
    high = sz_min_of_two(low + step, haystacks.count);
    while (low + 1 < high) {
        sz_size_t const middle = low + (high - low) / 2;
        if (sz_regex_first_slot_simt_(haystacks, middle) <= slot) low = middle;
        else high = middle;
    }
    return low;
}

/** The first of @p count ascending slots at or past @p wanted, galloping up from @p low; @p count
 *  when none is. */
STRINGZILLA_DEVICE sz_u32_t sz_regex_lower_bound_simt_(sz_u32_t const *slots, sz_u32_t low, sz_u32_t count,
                                                       sz_size_t wanted) {
    sz_u32_t step = 1, high;
    if (low == count || slots[low] >= wanted) return low;
    while (low + step < count && slots[low + step] < wanted) low += step, step *= 2;
    high = sz_min_of_two(low + step, count);
    while (low + 1 < high) {
        sz_u32_t const middle = low + (high - low) / 2;
        if (slots[middle] < wanted) low = middle;
        else high = middle;
    }
    return high;
}

/** The index of the lowest set bit of @p bits, which is not zero. */
STRINGZILLA_DEVICE sz_u32_t sz_regex_lowest_simt_(sz_u64_t bits) {
    return 63 - (sz_u32_t)__clzll((long long)(bits & (0 - bits)));
}

/** Index of @p state among the first @p count of @p states, or 0xFF. */
STRINGZILLA_DEVICE sz_u8_t sz_regex_index_of_simt_(sz_u32_t const *states, sz_u32_t count, sz_u32_t state) {
    sz_u32_t index;
    for (index = 0; index != count; ++index)
        if (states[index] == state) return (sz_u8_t)index;
    return 0xFF;
}

/**
 *  @brief The start-finding state the walk from the haystack's end holds once it has consumed
 *      every byte from @p edge on, found without that walk, or zero when @p reach bytes did
 *      not settle it.
 *
 *  As @c sz_regex_prime_metal_ : the set starts as the image of one column, holding the true
 *  state; a dying state is dropped; past @p reach the set left at the edge is the answer, or,
 *  with no column narrowing to one in @ref sz_regex_widening_bytes_simt_k, an empty set marking
 *  the chunk wide.
 */
STRINGZILLA_DEVICE sz_u32_t sz_regex_prime_simt_(sz_regex_engine_t const *engine, sz_regex_table_simt_t const *table,
                                                 sz_u32_t const *images, sz_regex_haystack_simt_t const *haystack,
                                                 sz_size_t edge, sz_size_t reach, sz_u32_t *set, sz_u32_t *count) {
    sz_size_t const length = haystack->length,
                    columns = sz_regex_stride_(engine->classes_count, engine->lookarounds_count);
    sz_u8_t const *const text = haystack->text;
    sz_size_t distance;
    for (distance = 32;; distance *= 2) {
        sz_size_t const far = length - edge > distance ? edge + distance : length;
        sz_size_t position = far;
        *count = 0;
        if (far == length) set[0] = engine->reverse_starts[0], *count = 1;
        while (*count == 0 && position > edge) {
            sz_u32_t const column = sz_regex_block_at_simt_(table, haystack->first + position) +
                                    table->classes[text[position - 1]];
            sz_u32_t const image = images[column];
            sz_u32_t index;
            --position;
            if (image > sz_regex_priming_states_simt_k) continue;
            for (index = 0; index != image; ++index)
                set[index] = images[columns + column * sz_regex_priming_states_simt_k + index];
            *count = image;
        }
        while (*count > 1 && position > edge) {
            sz_u32_t const column = sz_regex_block_at_simt_(table, haystack->first + position) +
                                    table->classes[text[position - 1]];
            sz_u32_t kept = 0, index;
            --position;
            for (index = 0; index != *count; ++index) {
                sz_u32_t const next = sz_regex_move_simt_(table, set[index], column);
                if (next != 0 && sz_regex_index_of_simt_(set, kept, next) == 0xFF) set[kept++] = next;
            }
            *count = kept;
        }
        if (*count == 1) return sz_regex_walk_back_simt_(table, set[0], haystack, position, edge);
        if (*count > 1 && distance >= reach) return 0;
        // A stretch where no column narrows to a frontier is left to the walk of its whole piece.
        if (distance >= sz_regex_widening_bytes_simt_k) return *count = 0;
    }
}

/** The result candidate @p index settles to once its link is followed. */
STRINGZILLA_DEVICE sz_regex_result_simt_t sz_regex_linked_simt_(sz_regex_round_simt_t const *round, sz_u32_t index) {
    sz_regex_result_simt_t result = round->results[index];
    if (result.link != sz_regex_none_simt_k) {
        sz_regex_result_simt_t const linked = round->links[result.link];
        if (linked.pattern != sz_regex_none_simt_k) result.pattern = linked.pattern, result.end = linked.end;
    }
    return result;
}

/** The match a search takes at candidate @p index of @p haystack, its link followed, with no
 *  pattern when the search passes it: a walk that never accepted, or an empty match before a
 *  UTF-8 continuation byte. */
STRINGZILLA_DEVICE sz_regex_result_simt_t sz_regex_settle_simt_(sz_regex_round_simt_t const *round, sz_u32_t index,
                                                                sz_size_t mark,
                                                                sz_regex_haystack_simt_t const *haystack) {
    sz_regex_result_simt_t result = sz_regex_linked_simt_(round, index);
    sz_size_t const start = mark - haystack->first;
    if (result.pattern != sz_regex_none_simt_k && result.end == mark && start < haystack->length &&
        (haystack->text[start] & 0xC0) == 0x80)
        result.pattern = sz_regex_none_simt_k;
    return result;
}

/** Writes where haystack @p index begins among the kept matches: into the matches offsets of a
 *  find, or, for a count, as the total before it into the count of the haystack before it, which
 *  the counts kernels difference. */
STRINGZILLA_DEVICE void sz_regex_boundary_store_simt_(sz_regex_round_simt_t const *round, sz_size_t index,
                                                      sz_size_t value) {
    if (round->counts) {
        if (index) round->counts[(index - 1) * round->counts_stride] = value;
    }
    else if (round->matches_offsets) round->matches_offsets[index] = value;
}

/** Sets bit @p index of a contains call's bitmap of @p count haystacks: an atomic OR into the
 *  aligned word holding it where that word lies inside the bitmap, else into the carry's edges. */
STRINGZILLA_DEVICE void sz_regex_contained_set_simt_(sz_regex_round_simt_t const *round, sz_size_t count,
                                                     sz_size_t index) {
    sz_size_t const begin = (sz_size_t)round->contained, end = begin + sz_size_divide_round_up(count, 8);
    sz_size_t const address = begin + index / 8, word = address & ~(sz_size_t)3;
    sz_u32_t const bit = 1u << (address % 4 * 8 + index % 8);
    if (word >= begin && word + 4 <= end) atomicOr((unsigned *)word, bit);
    else
        atomicOr((unsigned long long *)&round->carry->contained_edges,
                 (unsigned long long)bit << (word < begin ? 0 : 32));
}

/** ORs the carry's edges into the bytes of the partial words at either end of a contains call's
 *  bitmap of @p count haystacks, which no other thread writes. */
STRINGZILLA_DEVICE void sz_regex_contained_edges_simt_(sz_regex_round_simt_t const *round, sz_size_t count) {
    sz_size_t const begin = (sz_size_t)round->contained, end = begin + sz_size_divide_round_up(count, 8);
    sz_u64_t const edges = round->carry->contained_edges;
    sz_size_t address;
    for (address = begin; address != end && address % 4; ++address)
        round->contained[address - begin] |= (sz_u1x8_t)(edges >> (address % 4 * 8));
    for (address = sz_max_of_two(end & ~(sz_size_t)3, address); address != end; ++address)
        round->contained[address - begin] |= (sz_u1x8_t)(edges >> (32 + address % 4 * 8));
}

/** Byte @p index of @p image. */
STRINGZILLA_DEVICE sz_u8_t sz_regex_image_byte_simt_(sz_utf8_folded_image_t const *image, sz_u32_t index) {
    return (sz_u8_t)(image->bytes >> (8 * index));
}

/** The unit starting at @p position as walks step it, the byte itself unless @p folded, else the
 *  folds of its codepoint or one malformed byte. @return Where the unit ends. */
STRINGZILLA_DEVICE sz_size_t sz_regex_unit_after_simt_(sz_bool_t folded, sz_u8_t const *text, sz_size_t length,
                                                       sz_size_t position, sz_utf8_folded_image_t *image) {
    if (folded) return position + sz_utf8_fold_next_simt_(text + position, text + length, image);
    image->bytes = text[position], image->length = 1, image->rune_ends = 1;
    return position + 1;
}

/** The unit ending at @p end, which is past zero, as @ref sz_regex_unit_after_simt_ steps
 *  it. @return Where it begins. */
STRINGZILLA_DEVICE sz_size_t sz_regex_unit_before_simt_(sz_bool_t folded, sz_u8_t const *text, sz_size_t end,
                                                        sz_utf8_folded_image_t *image) {
    if (folded) return end - sz_utf8_fold_prev_simt_(text, text + end, image);
    image->bytes = text[end - 1], image->length = 1, image->rune_ends = 1;
    return end - 1;
}

/** The rune of @p image whose bytes end at index @p last, a malformed byte tagged as
 *  @c sz_rune_malformed_byte_ tags it. @return The index its bytes begin at. */
STRINGZILLA_DEVICE sz_u32_t sz_regex_image_rune_simt_(sz_utf8_folded_image_t const *image, sz_u32_t last,
                                                      sz_rune_t *rune) {
    sz_u32_t first = last, index;
    sz_u8_t lead;
    if (!image->rune_ends) {
        *rune = sz_rune_malformed_byte_(sz_regex_image_byte_simt_(image, 0));
        return 0;
    }
    while (first && !(image->rune_ends >> (first - 1) & 1)) --first;
    lead = sz_regex_image_byte_simt_(image, first);
    *rune = first == last ? lead : lead & (0x7Fu >> (last - first + 1));
    for (index = first + 1; index <= last; ++index)
        *rune = *rune << 6 | (sz_regex_image_byte_simt_(image, index) & 0x3Fu);
    return first;
}

/** The first unit boundary at or past @p position, as @c sz_regex_boundary_ finds it. */
STRINGZILLA_DEVICE sz_size_t sz_regex_boundary_simt_(sz_u8_t const *text, sz_size_t length, sz_size_t position) {
    sz_size_t back = 1;
    sz_rune_t rune;
    sz_rune_length_t rune_length;
    if (position == length || !sz_utf8_is_continuation_(text[position])) return position;
    while (back != 4 && back <= position && sz_utf8_is_continuation_(text[position - back])) ++back;
    if (back == 4 || back > position) return position;
    rune_length = sz_rune_decode((sz_cptr_t)text + position - back, (sz_cptr_t)text + length, &rune);
    return (sz_size_t)rune_length > back ? position - back + (sz_size_t)rune_length : position;
}

/** Walks @p state right to left over the folds of the @p haystack units ending in <tt>(stop, at]</tt>. */
STRINGZILLA_DEVICE sz_u32_t sz_regex_walk_folded_back_simt_(sz_regex_table_simt_t const *table, sz_u32_t state,
                                                            sz_regex_haystack_simt_t const *haystack, sz_size_t at,
                                                            sz_size_t stop) {
    sz_utf8_folded_image_t image;
    sz_u32_t byte, block;
    while (at > stop) {
        block = sz_regex_block_at_simt_(table, haystack->first + at);
        at = sz_regex_unit_before_simt_(sz_true_k, haystack->text, at, &image);
        for (byte = image.length; byte--; block = 0)
            state = sz_regex_move_simt_(table, state, block + table->classes[sz_regex_image_byte_simt_(&image, byte)]);
    }
    return state;
}

/**
 *  @brief The start-finding state a folded walk holds at unit boundary @p edge, found as
 *      @ref sz_regex_prime_simt_ finds it over the folds of whole units.
 *
 *  It is zero when @p reach bytes did not settle it, leaving the uncollapsed set at the edge, or an
 *  empty one marking the chunk wide.
 */
STRINGZILLA_DEVICE sz_u32_t sz_regex_prime_folded_simt_(sz_regex_engine_t const *engine,
                                                        sz_regex_table_simt_t const *table, sz_u32_t const *images,
                                                        sz_regex_haystack_simt_t const *haystack, sz_size_t edge,
                                                        sz_size_t reach, sz_u32_t *set, sz_u32_t *count) {
    sz_size_t const length = haystack->length,
                    columns = sz_regex_stride_(engine->classes_count, engine->lookarounds_count);
    sz_u8_t const *const text = haystack->text;
    sz_size_t distance;
    for (distance = 32;; distance *= 2) {
        sz_size_t const far = length - edge > distance ? sz_regex_boundary_simt_(text, length, edge + distance)
                                                       : length;
        sz_size_t position = far;
        sz_utf8_folded_image_t image;
        sz_u32_t byte = 0, block = 0, index, state;
        *count = 0;
        if (far == length) set[0] = engine->reverse_starts[0], *count = 1;
        // Folded bytes step right to left, `byte` counting those of the unit at `position` to go.
        while (*count != 1 && (byte || position > edge)) {
            sz_u32_t column, kept = 0;
            if (!byte)
                block = sz_regex_block_at_simt_(table, haystack->first + position),
                position = sz_regex_unit_before_simt_(sz_true_k, text, position, &image), byte = image.length;
            column = block + table->classes[sz_regex_image_byte_simt_(&image, --byte)];
            block = 0;
            if (*count == 0) {
                if (images[column] > sz_regex_priming_states_simt_k) continue;
                for (index = 0; index != images[column]; ++index)
                    set[index] = images[columns + column * sz_regex_priming_states_simt_k + index];
                *count = images[column];
                continue;
            }
            for (index = 0; index != *count; ++index) {
                sz_u32_t const next = sz_regex_move_simt_(table, set[index], column);
                if (next != 0 && sz_regex_index_of_simt_(set, kept, next) == 0xFF) set[kept++] = next;
            }
            *count = kept;
        }
        if (*count > 1 && distance >= reach) return 0;
        if (*count != 1 && distance >= sz_regex_widening_bytes_simt_k) return *count = 0;
        if (*count != 1) continue;
        state = set[0];
        while (byte)
            state = sz_regex_move_simt_(table, state, table->classes[sz_regex_image_byte_simt_(&image, --byte)]);
        return sz_regex_walk_folded_back_simt_(table, state, haystack, position, edge);
    }
}

/**
 *  @brief Walks the end-finding automaton over folded bytes from @p offset bytes into the folds of
 *      the unit at @p start until it dies, as @c sz_regex_walk_end_folded_ does without
 *      checkpoints, raising the refusal rather than reading the lookaround table past its window.
 *  @return Whether a match ends there, an empty one counting only at the unit's own start.
 */
STRINGZILLA_DEVICE sz_bool_t sz_regex_walk_folded_simt_(sz_regex_engine_t const *engine,
                                                        sz_regex_table_simt_t const *table,
                                                        sz_regex_haystack_simt_t const *haystack, sz_size_t start,
                                                        sz_size_t offset, sz_size_t *end, sz_u32_t *pattern) {
    sz_u32_t const accepts = engine->forward_accepts_begin;
    sz_u8_t const *const text = haystack->text;
    sz_size_t const length = haystack->length;
    sz_utf8_folded_image_t image;
    sz_size_t position = start, unit_end = start, byte = offset, kind = 0, consumed = 0, last_consumed = 0;
    sz_size_t last_end = STRINGZILLA_SIZE_MAX;
    sz_u32_t state, last = 0;
    if (start != length) unit_end = sz_regex_unit_after_simt_(sz_true_k, text, length, start, &image);
    if (offset) kind = sz_regex_byte_kind_(sz_regex_image_byte_simt_(&image, (sz_u32_t)offset - 1));
    else if (start) {
        sz_utf8_folded_image_t previous;
        sz_regex_unit_before_simt_(sz_true_k, text, start, &previous);
        kind = sz_regex_byte_kind_(sz_regex_image_byte_simt_(&previous, previous.length - 1u));
    }
    state = engine->forward_starts[kind];
    while (state != 0) {
        sz_u32_t block = 0;
        if (byte == 0 && sz_regex_past_simt_(table->looks, haystack->first + position)) {
            *table->looks->refused = 1;
            return sz_false_k;
        }
        if (byte == 0) block = sz_regex_block_at_simt_(table, haystack->first + position);
        if (position == length) {
            if ((state = sz_regex_move_simt_(table, state, block + table->edge_column)) >= accepts)
                last = state, last_end = length, last_consumed = consumed;
            break;
        }
        state = sz_regex_move_simt_(table, state,
                                    block + table->classes[sz_regex_image_byte_simt_(&image, (sz_u32_t)byte)]);
        if (state >= accepts) last = state, last_end = byte ? unit_end : position, last_consumed = consumed;
        ++consumed;
        if (++byte != image.length) continue;
        position = unit_end, byte = 0;
        if (position != length) unit_end = sz_regex_unit_after_simt_(sz_true_k, text, length, position, &image);
    }
    if (last_end == STRINGZILLA_SIZE_MAX || (offset && !last_consumed)) return sz_false_k;
    *end = last_end, *pattern = sz_regex_pattern_simt_(engine, last);
    return sz_true_k;
}

/** The unit class of folded @p rune, zero if malformed, as @c sz_regex_rune_class_ finds it. */
STRINGZILLA_DEVICE sz_u32_t sz_regex_rune_class_simt_(sz_regex_engine_t const *engine, sz_rune_t rune) {
    sz_size_t low = 0, high = engine->positions_ranges_count;
    if (rune < 0x80) return engine->positions_ascii[rune];
    if (rune & 0x80000000u) return 0;
    while (high - low > 1) {
        sz_size_t const middle = (low + high) / 2;
        if (engine->positions_ranges[middle * 2] <= rune) low = middle;
        else high = middle;
    }
    return engine->positions_ranges[low * 2 + 1];
}

/** The unit ending at @p end, an ASCII byte, a whole codepoint or one malformed byte, and its class
 *  in @p unit_class, as @c sz_regex_unit_before_ finds them. @return Where it begins. */
STRINGZILLA_DEVICE sz_size_t sz_regex_class_before_simt_(sz_regex_engine_t const *engine, sz_u8_t const *text,
                                                         sz_size_t end, sz_u32_t *unit_class) {
    sz_u8_t const last = text[end - 1];
    sz_size_t back = 2;
    sz_rune_t rune;
    if (last < 0x80) {
        *unit_class = engine->positions_ascii[last];
        return end - 1;
    }
    *unit_class = 0;
    if (!sz_utf8_is_continuation_(last)) return end - 1;
    while (back != 5 && back <= end && sz_utf8_is_continuation_(text[end - back])) ++back;
    if (back == 5 || back > end ||
        (sz_size_t)sz_rune_decode((sz_cptr_t)text + end - back, (sz_cptr_t)text + end, &rune) != back)
        return end - 1;
    *unit_class = sz_regex_rune_class_simt_(engine, rune);
    return end - back;
}

/** Steps the position set @p state over one unit of @p unit_class at a boundary of @p kind where
 *  the lookarounds of @p looks hold into @p next, as @c sz_regex_positions_step_ does. */
STRINGZILLA_DEVICE void sz_regex_positions_step_simt_(sz_regex_engine_t const *engine, sz_u64_t const *state,
                                                      sz_u64_t *next, sz_size_t unit_class, sz_size_t kind,
                                                      sz_u32_t looks) {
    sz_size_t const words = engine->positions_words;
    sz_u64_t const *record = engine->positions_shifts;
    sz_u64_t const *const reach = engine->positions_reach + unit_class * words;
    sz_u64_t const *const initial = engine->positions_initial + kind * words;
    sz_size_t word, shift, group;
    for (word = 0; word != words; ++word) next[word] = initial[word];
    for (shift = 0; shift != engine->positions_shifts_count; ++shift, record += 1 + words) {
        sz_i64_t const distance = (sz_i64_t)record[0];
        sz_size_t const span = (sz_size_t)(distance < 0 ? -distance : distance), offset = span / 64, bits = span % 64;
        sz_u64_t const *const mask = record + 1;
        for (word = offset; word < words; ++word) {
            sz_size_t const low = word - offset;
            if (distance >= 0) {
                next[word] |= (state[low] & mask[low]) << bits;
                if (bits && low) next[word] |= (state[low - 1] & mask[low - 1]) >> (64 - bits);
            }
            else {
                next[low] |= (state[word] & mask[word]) >> bits;
                if (bits && word + 1 != words) next[low] |= (state[word + 1] & mask[word + 1]) << (64 - bits);
            }
        }
    }
    record = engine->positions_groups;
    for (group = 0; group != engine->positions_groups_count; ++group, record += 1 + 2 * words) {
        sz_u64_t any = 0, enabled;
        for (word = 0; word != words; ++word) any |= state[word] & record[1 + word];
        enabled = (sz_u64_t)0 -
                  (sz_u64_t)((any != 0) & (record[0] >> kind & 1) & ((record[0] >> 16 & ~(sz_u64_t)looks) == 0));
        for (word = 0; word != words; ++word) next[word] |= record[1 + words + word] & enabled;
    }
    for (word = 0; word != words; ++word) next[word] &= reach[word];
}

/**
 *  @brief Steps the position set at @p state right to left over the units of @p haystack ending in
 *      <tt>(stop, from]</tt>, marking in @p bits every start it finds in slots <tt>[low, high)</tt>.
 *
 *  It steps them as @c sz_regex_mark_positions_serial_ does, or over folded runes when @p folded,
 *  as @c sz_regex_mark_positions_folded_ does.
 *
 *  @param[in] from A unit boundary or the haystack's length, where @p state holds what the units
 *      right of it left.
 *  @param[out] bits The start bits of slot @p low on, or null to mark nothing.
 *  @return The set holding the result, @p state or @p next.
 */
STRINGZILLA_DEVICE sz_u64_t *sz_regex_positions_walk_simt_(sz_regex_engine_t const *engine,
                                                           sz_regex_round_simt_t const *round,
                                                           sz_regex_haystack_simt_t const *haystack, sz_bool_t folded,
                                                           sz_size_t from, sz_size_t stop, sz_u64_t *state,
                                                           sz_u64_t *next, sz_u64_t *bits, sz_size_t low,
                                                           sz_size_t high) {
    sz_u8_t const *const text = haystack->text;
    sz_size_t const length = haystack->length;
    sz_size_t at = from, right_kind = 0;
    sz_utf8_folded_image_t image;
    if (from != length) {
        sz_regex_unit_after_simt_(folded, text, length, from, &image);
        right_kind = sz_regex_byte_kind_(sz_regex_image_byte_simt_(&image, 0));
    }
    while (at > stop) {
        sz_u32_t const looks = sz_regex_looks_at_simt_(&round->looks, haystack->first + at);
        sz_u32_t unit_class, end, rune_start = 0;
        sz_rune_t rune = 0;
        sz_size_t const begin = folded ? sz_regex_unit_before_simt_(sz_true_k, text, at, &image)
                                       : sz_regex_class_before_simt_(engine, text, at, &unit_class);
        // Units step once and folded ones once per rune, right to left, the last rune reading
        // the unit's lookarounds.
        for (end = folded ? image.length : 1; end; end = rune_start) {
            sz_size_t left_kind = sz_regex_byte_kind_(text[at - 1]), slot = haystack->first + at;
            sz_u64_t *const swapped = state;
            if (folded) {
                rune_start = sz_regex_image_rune_simt_(&image, end - 1, &rune);
                unit_class = sz_regex_rune_class_simt_(engine, rune), left_kind = sz_regex_rune_kind_(rune);
                if (end != image.length) slot = haystack->first + begin;
            }
            sz_regex_positions_step_simt_(engine, state, next, unit_class, left_kind * 4 + right_kind,
                                          slot == haystack->first + at ? looks : 0);
            state = next, next = swapped;
            if (bits && slot >= low && slot < high && (state[0] >> 1 & 1))
                bits[(slot - low) / 64] |= (sz_u64_t)1 << ((slot - low) % 64);
            right_kind = left_kind;
        }
        at = begin;
    }
    return state;
}

/** The PikeVM of the share of @p thread among the round's walks, laid out for its capture slots. */
STRINGZILLA_DEVICE sz_regex_pike_simt_t sz_regex_pike_carve_simt_(sz_regex_round_simt_t const *round, sz_size_t thread,
                                                                  sz_regex_engine_t const *engine) {
    sz_u8_t *const walk = round->walks + thread * round->walk_bytes;
    sz_size_t const states = engine->nfa_states_count, entered = engine->nfa_links_count + 1;
    sz_size_t const slots_count = round->pike_slots;
    sz_regex_pike_simt_t pike;
    pike.dense_slots = (sz_size_t *)walk;
    pike.entered_slots = pike.dense_slots + states * slots_count;
    pike.current = pike.entered_slots + entered * slots_count;
    pike.matched = pike.current + slots_count;
    pike.frames = pike.matched + slots_count;
    pike.dense = (sz_u32_t *)(pike.frames + 2 * (states + entered));
    pike.sparse = pike.dense + states;
    pike.entered = pike.sparse + states;
    pike.count = 0, pike.entered_count = 0, pike.slots_count = slots_count, pike.end = 0, pike.first = 0;
    pike.looks = &round->looks, pike.pattern = 0, pike.found = sz_false_k;
    return pike;
}

/** The assertions that hold between two units, as @c sz_regex_looks_between_ finds them. */
STRINGZILLA_DEVICE sz_u32_t sz_regex_looks_between_simt_(sz_u32_t before, sz_u32_t after) {
    sz_u32_t have = sz_regex_is_word_(before) == sz_regex_is_word_(after) ? (sz_u32_t)sz_regex_look_word_inside_k
                                                                          : (sz_u32_t)sz_regex_look_word_boundary_k;
    if (before == sz_regex_end_of_input_k) have |= sz_regex_look_text_start_k | sz_regex_look_line_start_k;
    else if (before == '\n') have |= sz_regex_look_line_start_k;
    if (after == sz_regex_end_of_input_k) have |= sz_regex_look_text_end_k | sz_regex_look_line_end_k;
    else if (after == '\n') have |= sz_regex_look_line_end_k;
    return have;
}

/** Adds every state @p state reaches without consuming to the position's threads, as
 *  @c sz_regex_pike_close_ does. */
STRINGZILLA_DEVICE void sz_regex_pike_close_simt_(sz_regex_engine_t const *engine, sz_regex_pike_simt_t *pike,
                                                  sz_u32_t state, sz_size_t const *slots, sz_u32_t have, sz_size_t open,
                                                  sz_size_t close) {
    sz_regex_nfa_state_t const *const states = engine->nfa_states;
    sz_regex_nfa_link_t const *const links = engine->nfa_links;
    sz_size_t const states_count = engine->nfa_states_count, slots_count = pike->slots_count;
    sz_size_t *const frames = pike->frames, *const current = pike->current;
    sz_size_t top = 0, index;
    for (index = 0; index != slots_count; ++index) current[index] = slots[index];
    frames[top++] = state, frames[top++] = 0;
    while (top) {
        sz_size_t const value = frames[--top], tag = frames[--top];
        if (tag >= states_count) {
            current[tag - states_count] = value;
            continue;
        }
        for (state = (sz_u32_t)tag;;) {
            sz_regex_nfa_state_t const node = states[state];
            sz_u32_t const held = pike->sparse[state];
            sz_u32_t link, other;
            sz_size_t base, low, high;
            if (held < pike->count && pike->dense[held] == state) break;
            pike->sparse[state] = (sz_u32_t)pike->count, pike->dense[pike->count] = state;
            if (node.kind == sz_regex_nfa_ranges_k || node.kind == sz_regex_nfa_match_k)
                for (index = 0; index != slots_count; ++index)
                    pike->dense_slots[pike->count * slots_count + index] = current[index];
            ++pike->count;
            if (node.kind == sz_regex_nfa_ranges_k || node.kind == sz_regex_nfa_match_k) break;
            if (node.kind == sz_regex_nfa_look_k && !(have & node.look)) break;
            if (node.kind == sz_regex_nfa_empty_k && node.slot < slots_count) {
                frames[top++] = states_count + node.slot, frames[top++] = current[node.slot];
                current[node.slot] = node.slot & 1 ? close : open;
            }
            if (node.kind == sz_regex_nfa_look_k || node.kind == sz_regex_nfa_empty_k) {
                state = node.target;
                continue;
            }
            link = node.first_link;
            if (link == sz_regex_none_k) break;
            base = top;
            for (other = links[link].next; other != sz_regex_none_k; other = links[other].next)
                frames[top++] = links[other].target, frames[top++] = 0;
            // Alternatives pushed in priority order pop in reverse, so the pairs are flipped.
            for (low = base, high = top; low + 2 < high; low += 2, high -= 2) {
                sz_size_t const swapped = frames[low];
                frames[low] = frames[high - 2], frames[high - 2] = swapped;
            }
            state = links[link].target;
        }
    }
}

/**
 *  @brief Runs the PikeVM at one position where the lookarounds of @p looks hold, as
 *      @c sz_regex_pike_step_ does.
 *  @return Whether any thread entered the next.
 */
STRINGZILLA_DEVICE sz_bool_t sz_regex_pike_step_simt_(sz_regex_engine_t const *engine, sz_regex_pike_simt_t *pike,
                                                      sz_u32_t before, sz_u32_t after, sz_u32_t looks, sz_size_t open,
                                                      sz_size_t close) {
    sz_regex_nfa_state_t const *const states = engine->nfa_states;
    sz_regex_nfa_link_t const *const links = engine->nfa_links;
    sz_size_t const slots_count = pike->slots_count;
    sz_u32_t const have = sz_regex_looks_between_simt_(before, after) | looks;
    sz_size_t index, slot;
    sz_u32_t link;
    pike->count = 0, pike->found = sz_false_k;
    for (index = 0; index != pike->entered_count; ++index)
        sz_regex_pike_close_simt_(engine, pike, pike->entered[index], pike->entered_slots + index * slots_count, have,
                                  open, close);
    pike->entered_count = 0;
    for (index = 0; index != pike->count; ++index) {
        sz_regex_nfa_state_t const node = states[pike->dense[index]];
        if (node.kind == sz_regex_nfa_match_k) {
            pike->end = close, pike->pattern = node.target, pike->found = sz_true_k;
            for (slot = 0; slot != slots_count; ++slot)
                pike->matched[slot] = pike->dense_slots[index * slots_count + slot];
            break;
        }
        if (node.kind != sz_regex_nfa_ranges_k || after == sz_regex_end_of_input_k) continue;
        for (link = node.first_link; link != sz_regex_none_k; link = links[link].next) {
            if (after < links[link].low || after > links[link].high) continue;
            pike->entered[pike->entered_count] = links[link].target;
            for (slot = 0; slot != slots_count; ++slot)
                pike->entered_slots[pike->entered_count * slots_count + slot] =
                    pike->dense_slots[index * slots_count + slot];
            ++pike->entered_count;
        }
    }
    return (sz_bool_t)(pike->entered_count != 0);
}

/**
 *  @brief Runs the PikeVM from the NFA state @p entry at @p offset bytes into the unit at @p start,
 *      until its threads die or it passes the unit boundary @p bound, as @c sz_regex_walk_pike_
 *      does, raising the refusal rather than reading the lookaround table past its window.
 *  @return Whether a match ends there, an empty one counting only at the unit's own start.
 */
STRINGZILLA_DEVICE sz_bool_t sz_regex_walk_pike_simt_(sz_regex_engine_t const *engine, sz_regex_pike_simt_t *pike,
                                                      sz_u8_t const *text, sz_size_t length, sz_size_t start,
                                                      sz_size_t offset, sz_u32_t entry, sz_size_t bound, sz_size_t *end,
                                                      sz_u32_t *pattern) {
    sz_bool_t const folded = (sz_bool_t)(engine->case_sensitivity != sz_cased_k);
    sz_utf8_folded_image_t image;
    sz_size_t position = start, unit_end = start, byte = offset, last_step = 0, step, index;
    sz_bool_t matched = sz_false_k;
    sz_u32_t before = sz_regex_end_of_input_k;
    image.length = 1;
    if (start != length) unit_end = sz_regex_unit_after_simt_(folded, text, length, start, &image);
    if (offset) before = sz_regex_image_byte_simt_(&image, (sz_u32_t)offset - 1);
    else if (start) {
        sz_utf8_folded_image_t previous;
        sz_regex_unit_before_simt_(folded, text, start, &previous);
        before = sz_regex_image_byte_simt_(&previous, previous.length - 1u);
    }
    pike->entered[0] = entry, pike->entered_count = 1;
    for (index = 0; index != pike->slots_count; ++index)
        pike->entered_slots[index] = STRINGZILLA_SIZE_MAX, pike->matched[index] = STRINGZILLA_SIZE_MAX;
    for (step = 0;; ++step) {
        sz_u32_t const after = position == length ? (sz_u32_t)sz_regex_end_of_input_k
                                                  : sz_regex_image_byte_simt_(&image, (sz_u32_t)byte);
        sz_u32_t held;
        sz_bool_t alive;
        if (byte == 0 && sz_regex_past_simt_(pike->looks, pike->first + position)) {
            *pike->looks->refused = 1;
            return sz_false_k;
        }
        held = byte ? 0 : sz_regex_looks_at_simt_(pike->looks, pike->first + position);
        alive = sz_regex_pike_step_simt_(engine, pike, before, after, held, position, byte ? unit_end : position);
        if (pike->found) matched = sz_true_k, last_step = step;
        if (!alive || (position == bound && byte == 0)) break;
        before = after;
        if (++byte != image.length) continue;
        position = unit_end, byte = 0;
        if (position != length) unit_end = sz_regex_unit_after_simt_(folded, text, length, position, &image);
    }
    if (!matched || (offset && last_step == 0)) return sz_false_k;
    *end = pike->end, *pattern = pike->pattern;
    return sz_true_k;
}

/**
 *  @brief The match a search takes at candidate @p start of @p haystack without links: each rune
 *      its unit folds to tried in turn, as @c sz_regex_search_serial_ tries them, by @p pike when
 *      there is one, else by the folded DFA of @p table.
 *  @param[in,out] offset The folded bytes into the unit the first try starts at, then those the
 *      match starts at.
 */
STRINGZILLA_DEVICE sz_bool_t sz_regex_match_at_simt_(sz_regex_engine_t const *engine,
                                                     sz_regex_table_simt_t const *table, sz_regex_pike_simt_t *pike,
                                                     sz_regex_haystack_simt_t const *haystack, sz_size_t start,
                                                     sz_size_t *end, sz_u32_t *pattern, sz_size_t *offset) {
    sz_u8_t const *const text = haystack->text;
    sz_size_t const length = haystack->length;
    sz_utf8_folded_image_t image;
    sz_u32_t rune_start = (sz_u32_t)*offset, rune_end;
    image.length = 1, image.rune_ends = 1;
    if (engine->case_sensitivity != sz_cased_k && start != length)
        sz_regex_unit_after_simt_(sz_true_k, text, length, start, &image);
    if (pike) pike->first = haystack->first;
    for (;;) {
        sz_bool_t const found = pike ? sz_regex_walk_pike_simt_(engine, pike, text, length, start, rune_start,
                                                                engine->nfa_start, length, end, pattern)
                                     : sz_regex_walk_folded_simt_(engine, table, haystack, start, rune_start, end,
                                                                  pattern);
        if (found && !(*end == start && start < length && (text[start] & 0xC0) == 0x80)) {
            *offset = rune_start;
            return sz_true_k;
        }
        for (rune_end = rune_start; rune_end < image.length && !(image.rune_ends >> rune_end & 1);) ++rune_end;
        rune_start = rune_end + 1;
        if (rune_start >= image.length) return sz_false_k;
    }
}

/** Fills the matched slots of @p pike with the groups of the match of @p pattern over
 *  <tt>[start, end)</tt> of @p haystack, starting inside the folds of its first unit wherever the
 *  search started it. */
STRINGZILLA_DEVICE void sz_regex_groups_simt_(sz_regex_engine_t const *engine, sz_regex_table_simt_t const *table,
                                              sz_regex_pike_simt_t *pike, sz_regex_haystack_simt_t const *haystack,
                                              sz_size_t start, sz_size_t end, sz_u32_t pattern, sz_size_t slots_count) {
    sz_size_t offset = 0, found_end;
    sz_u32_t found_pattern;
    if (engine->case_sensitivity != sz_cased_k) {
        pike->slots_count = 0;
        sz_regex_match_at_simt_(engine, table, engine->forward_rows ? STRINGZILLA_NULL : pike, haystack, start,
                                &found_end, &found_pattern, &offset);
    }
    pike->slots_count = slots_count, pike->first = haystack->first;
    sz_regex_walk_pike_simt_(engine, pike, haystack->text, haystack->length, start, offset,
                             engine->patterns_starts[pattern], end, &found_end, &found_pattern);
}

/**
 *  @brief Appends one match's replacement at @p written of @p target, as @c sz_regex_expand_
 *      does, or only measures it for a null @p target; @p pike reads the groups a reference
 *      names, once per match.
 *  @return The bytes the replacement takes, whether or not they fit below @p capacity.
 */
STRINGZILLA_DEVICE sz_size_t sz_regex_expand_simt_(sz_regex_engine_t const *engine, sz_regex_table_simt_t const *table,
                                                   sz_regex_pike_simt_t *pike, sz_regex_haystack_simt_t const *haystack,
                                                   sz_u8_t const *replacement, sz_size_t replacement_length,
                                                   sz_size_t start, sz_size_t end, sz_u32_t pattern,
                                                   sz_size_t slots_count, sz_ptr_t target, sz_size_t capacity,
                                                   sz_size_t written) {
    sz_u8_t const *const text = haystack->text;
    sz_u8_t const *const names = (sz_u8_t const *)engine->groups_names.data.u32_tape.data;
    sz_u32_t const *const names_offsets = engine->groups_names.data.u32_tape.offsets;
    sz_size_t const first_group = engine->patterns_groups[pattern];
    sz_size_t const groups = engine->patterns_groups[pattern + 1] - first_group;
    sz_size_t cursor = 0, appended = 0, copy;
    sz_bool_t walked = sz_false_k;
    while (cursor != replacement_length) {
        sz_size_t next = cursor + 1, group = STRINGZILLA_SIZE_MAX, number = 0, copied = 1;
        sz_u8_t const *source = replacement + cursor;
        sz_bool_t referenced = sz_false_k;
        if (replacement[cursor] == '$' && next != replacement_length) {
            sz_size_t name_begin = next, name_end = next, digit, named, index;
            if (replacement[next] == '$') source = replacement + next++;
            else if (replacement[next] == '{') {
                name_begin = name_end = next + 1;
                while (name_end != replacement_length && replacement[name_end] != '}') ++name_end;
                if (name_end != replacement_length && name_end != name_begin)
                    referenced = sz_true_k, next = name_end + 1;
            }
            else {
                while (name_end != replacement_length && replacement[name_end] >= '0' && replacement[name_end] <= '9')
                    ++name_end;
                if (name_end != name_begin) referenced = sz_true_k, next = name_end;
            }
            if (referenced) {
                for (digit = name_begin; digit != name_end && replacement[digit] >= '0' && replacement[digit] <= '9';
                     ++digit)
                    number = number > groups ? number : number * 10 + (sz_size_t)(replacement[digit] - '0');
                if (digit == name_end && number <= groups) group = number;
                for (named = 0; digit != name_end && named != groups; ++named) {
                    sz_size_t const name_offset = names_offsets[first_group + named];
                    sz_bool_t same = (sz_bool_t)(names_offsets[first_group + named + 1] - name_offset ==
                                                 name_end - name_begin);
                    for (index = 0; same && index != name_end - name_begin; ++index)
                        same = (sz_bool_t)(names[name_offset + index] == replacement[name_begin + index]);
                    if (same) group = named + 1;
                }
            }
        }
        while (!referenced && next != replacement_length && replacement[next] != '$') ++next, ++copied;
        if (referenced) {
            source = text + start, copied = group == 0 ? end - start : 0;
            if (group != 0 && group != STRINGZILLA_SIZE_MAX) {
                if (!walked) sz_regex_groups_simt_(engine, table, pike, haystack, start, end, pattern, slots_count);
                walked = sz_true_k;
                if (pike->matched[2 * group] != STRINGZILLA_SIZE_MAX &&
                    pike->matched[2 * group + 1] != STRINGZILLA_SIZE_MAX)
                    source = text + pike->matched[2 * group],
                    copied = pike->matched[2 * group + 1] - pike->matched[2 * group];
            }
        }
        if (target)
            for (copy = 0; copy != copied && written + appended + copy < capacity; ++copy)
                target[written + appended + copy] = (char)source[copy];
        appended += copied, cursor = next;
    }
    return appended;
}

/**
 *  @brief Writes the rows of a haystack's first match of @p pattern from @p written on, as
 *      @c sz_regex_extract_with_ does: the whole match, then each group that took part, those past
 *      @p capacity counted only.
 *  @return The rows.
 */
STRINGZILLA_DEVICE sz_size_t sz_regex_rows_write_simt_(sz_regex_engine_t const *engine,
                                                       sz_regex_table_simt_t const *table, sz_regex_pike_simt_t *pike,
                                                       sz_regex_haystack_simt_t const *haystack,
                                                       sz_size_t haystack_index, sz_size_t start, sz_size_t end,
                                                       sz_u32_t pattern, sz_match_t *rows, sz_size_t capacity,
                                                       sz_size_t written) {
    sz_size_t const groups = engine->patterns_groups[pattern + 1] - engine->patterns_groups[pattern];
    sz_size_t count = 0, group;
    if (groups) sz_regex_groups_simt_(engine, table, pike, haystack, start, end, pattern, 2 * groups + 2);
    for (group = 0; group <= groups; ++group) {
        sz_size_t const *const slots = pike->matched + 2 * group;
        if (group && (slots[0] == STRINGZILLA_SIZE_MAX || slots[1] == STRINGZILLA_SIZE_MAX)) continue;
        if (written + count < capacity) {
            sz_match_t *const row = rows + written + count;
            row->haystack_index = haystack_index, row->needle_index = group;
            row->byte_offset = group ? slots[0] : start;
            row->byte_length = group ? slots[1] - slots[0] : end - start;
        }
        ++count;
    }
    return count;
}

/** Kept matches of the extracted pattern in this segment before @p slot: those of the chunks before
 *  its own, scanned, and those of its own chunk before it. */
STRINGZILLA_DEVICE sz_size_t sz_regex_counted_before_simt_(sz_regex_round_simt_t const *round, sz_size_t slot) {
    sz_size_t const begin = round->segment_begin, chunk = (slot - begin) >> round->chunk_shift;
    sz_size_t counted = round->chunk_sums[chunk];
    sz_u32_t index;
    for (index = (sz_u32_t)round->chunk_offsets[chunk];
         index != (sz_u32_t)round->chunk_offsets[chunk + 1] && begin + round->slots[index] < slot; ++index)
        counted += round->kept[index] && sz_regex_linked_simt_(round, index).pattern == round->pattern_index;
    return counted;
}

/** The lookarounds a pass's @p set holds as matched inside the folds of the unit before, which hold
 *  at the next boundary, kept past its body states. */
STRINGZILLA_DEVICE sz_u32_t sz_regex_pending_simt_(sz_regex_engine_t const *engine, sz_regex_round_simt_t const *round,
                                                   sz_u64_t const *set) {
    sz_u32_t pending = 0, lookaround;
    for (lookaround = 0; lookaround != engine->lookarounds_count; ++lookaround) {
        sz_size_t const index = round->bodies_count + lookaround;
        if (set[index / 64] >> (index % 64) & 1) pending |= (sz_u32_t)sz_regex_look_lookaround_k << lookaround;
    }
    return pending;
}

/** Adds @p pending lookarounds to a pass's @p set, as @ref sz_regex_pending_simt_ reads them. */
STRINGZILLA_DEVICE void sz_regex_pend_simt_(sz_regex_engine_t const *engine, sz_regex_round_simt_t const *round,
                                            sz_u64_t *set, sz_u32_t pending) {
    sz_u32_t lookaround;
    for (lookaround = 0; lookaround != engine->lookarounds_count; ++lookaround) {
        sz_size_t const index = round->bodies_count + lookaround;
        if (pending & (sz_u32_t)sz_regex_look_lookaround_k << lookaround)
            set[index / 64] |= (sz_u64_t)1 << (index % 64);
    }
}

/** Records at @p slot of the round's table the members of its pass that hold there: those
 *  @p matched names, or the negated ones it does not. */
STRINGZILLA_DEVICE void sz_regex_looks_record_simt_(sz_regex_round_simt_t const *round, sz_size_t slot,
                                                    sz_u32_t matched) {
    if (slot >= round->looks.begin && slot < round->looks.end)
        round->looks.table[slot & round->looks.mask] |= (matched ^ round->negated) & round->members;
}

/**
 *  @brief Closes the threads of a lookaround pass at one boundary under @p have, those of the body
 *      states of @p state and every member's body starting there, and steps them over @p after into
 *      @p next, as @c sz_regex_looks_step_ does; @p next holds nothing pending.
 *  @return The look bits of the members whose bodies match there.
 */
STRINGZILLA_DEVICE sz_u32_t sz_regex_looks_step_simt_(sz_regex_engine_t const *engine,
                                                      sz_regex_round_simt_t const *round, sz_regex_pike_simt_t *pike,
                                                      sz_u64_t const *state, sz_u64_t *next, sz_u32_t have,
                                                      sz_u32_t after) {
    sz_regex_nfa_state_t const *const states = engine->nfa_states;
    sz_regex_nfa_link_t const *const links = engine->nfa_links;
    sz_size_t const bodies = round->bodies_begin, count = round->bodies_count;
    sz_size_t const words = (count + engine->lookarounds_count + 63) / 64;
    sz_u32_t matched = 0, lookaround, link;
    sz_size_t word, index;
    sz_u64_t bits;
    pike->count = 0;
    for (word = 0; word != words; ++word)
        for (bits = state[word]; bits; bits &= bits - 1) {
            index = word * 64 + sz_regex_lowest_simt_(bits);
            if (index < count)
                sz_regex_pike_close_simt_(engine, pike, (sz_u32_t)(bodies + index), STRINGZILLA_NULL, have, 0, 0);
        }
    for (lookaround = 0; lookaround != engine->lookarounds_count; ++lookaround)
        if (round->members & (sz_u32_t)sz_regex_look_lookaround_k << lookaround)
            sz_regex_pike_close_simt_(engine, pike, engine->lookarounds[lookaround].start, STRINGZILLA_NULL, have, 0,
                                      0);
    for (word = 0; word != words; ++word) next[word] = 0;
    for (index = 0; index != pike->count; ++index) {
        sz_regex_nfa_state_t const node = states[pike->dense[index]];
        if (node.kind == sz_regex_nfa_match_k) matched |= (sz_u32_t)sz_regex_look_lookaround_k << node.target;
        if (node.kind != sz_regex_nfa_ranges_k || after == sz_regex_end_of_input_k) continue;
        for (link = node.first_link; link != sz_regex_none_k; link = links[link].next)
            if (links[link].low <= after && after <= links[link].high) {
                sz_size_t const target = links[link].target - bodies;
                next[target / 64] |= (sz_u64_t)1 << (target % 64);
            }
    }
    return matched;
}

/**
 *  @brief Walks one level's lookbehind pass over the unit boundaries of @p haystack in
 *      <tt>[from, to)</tt>, and its end when @p to passes it, as @c sz_regex_looks_serial_ walks it.
 *
 *  It starts from the set at @p state arriving at @p from, recording where its members hold
 *  when @p recording.
 *
 *  @return The set arriving at @p to, @p state or @p next.
 */
STRINGZILLA_DEVICE sz_u64_t *sz_regex_behind_walk_simt_(sz_regex_engine_t const *engine,
                                                        sz_regex_round_simt_t const *round, sz_regex_pike_simt_t *pike,
                                                        sz_regex_haystack_simt_t const *haystack, sz_size_t from,
                                                        sz_size_t to, sz_u64_t *state, sz_u64_t *next,
                                                        sz_bool_t recording) {
    sz_bool_t const folded = (sz_bool_t)(engine->case_sensitivity != sz_cased_k);
    sz_u8_t const *const text = haystack->text;
    sz_size_t const length = haystack->length;
    sz_size_t position = from, unit_end, byte;
    sz_u32_t before = sz_regex_end_of_input_k, after, matched, pending = sz_regex_pending_simt_(engine, round, state);
    sz_utf8_folded_image_t image;
    sz_u64_t *swapped;
    if (from) {
        sz_regex_unit_before_simt_(folded, text, from, &image);
        before = sz_regex_image_byte_simt_(&image, image.length - 1u);
    }
    while (position < to && position <= length) {
        after = sz_regex_end_of_input_k, unit_end = length + 1;
        if (position != length)
            unit_end = sz_regex_unit_after_simt_(folded, text, length, position, &image),
            after = sz_regex_image_byte_simt_(&image, 0);
        matched = sz_regex_looks_step_simt_(engine, round, pike, state, next,
                                            sz_regex_looks_between_simt_(before, after) |
                                                sz_regex_looks_at_simt_(&round->looks, haystack->first + position),
                                            after);
        if (recording) sz_regex_looks_record_simt_(round, haystack->first + position, matched | pending);
        swapped = state, state = next, next = swapped, pending = 0;
        for (byte = 1; position != length && byte < image.length; ++byte) {
            before = after, after = sz_regex_image_byte_simt_(&image, (sz_u32_t)byte);
            pending |= sz_regex_looks_step_simt_(engine, round, pike, state, next,
                                                 sz_regex_looks_between_simt_(before, after), after);
            swapped = state, state = next, next = swapped;
        }
        before = after, position = unit_end;
    }
    sz_regex_pend_simt_(engine, round, state, pending);
    return state;
}

/**
 *  @brief Walks one level's lookahead pass right to left over the unit boundaries of @p haystack in
 *      <tt>[to, from)</tt>, as @c sz_regex_looks_serial_ walks it.
 *
 *  It starts from the set at @p state past @p from, or from the haystack's end when @p from passes
 *  it, and records where its members hold when @p recording.
 *
 *  @return The set past @p to, @p state or @p next.
 */
STRINGZILLA_DEVICE sz_u64_t *sz_regex_ahead_walk_simt_(sz_regex_engine_t const *engine,
                                                       sz_regex_round_simt_t const *round, sz_regex_pike_simt_t *pike,
                                                       sz_regex_haystack_simt_t const *haystack, sz_size_t from,
                                                       sz_size_t to, sz_u64_t *state, sz_u64_t *next,
                                                       sz_bool_t recording) {
    sz_bool_t const folded = (sz_bool_t)(engine->case_sensitivity != sz_cased_k);
    sz_u8_t const *const text = haystack->text;
    sz_size_t const length = haystack->length;
    sz_size_t position = length, unit_begin, byte;
    sz_u32_t before = sz_regex_end_of_input_k, after, matched, pending = sz_regex_pending_simt_(engine, round, state);
    sz_utf8_folded_image_t image;
    sz_u64_t *swapped;
    if (from <= length) {
        if (from == 0) return state;
        position = sz_regex_unit_before_simt_(folded, text, from, &image);
        before = sz_regex_image_byte_simt_(&image, 0);
    }
    while (position >= to) {
        after = sz_regex_end_of_input_k, unit_begin = position;
        if (position)
            unit_begin = sz_regex_unit_before_simt_(folded, text, position, &image),
            after = sz_regex_image_byte_simt_(&image, image.length - 1u);
        matched = sz_regex_looks_step_simt_(engine, round, pike, state, next,
                                            sz_regex_looks_between_simt_(before, after) |
                                                sz_regex_looks_at_simt_(&round->looks, haystack->first + position),
                                            after);
        if (recording) sz_regex_looks_record_simt_(round, haystack->first + position, matched | pending);
        swapped = state, state = next, next = swapped, pending = 0;
        if (!position) break;
        for (byte = 1; byte < image.length; ++byte) {
            before = after, after = sz_regex_image_byte_simt_(&image, image.length - 1u - (sz_u32_t)byte);
            pending |= sz_regex_looks_step_simt_(engine, round, pike, state, next,
                                                 sz_regex_looks_between_simt_(before, after), after);
            swapped = state, state = next, next = swapped;
        }
        before = after, position = unit_begin;
    }
    sz_regex_pend_simt_(engine, round, state, pending);
    return state;
}

/** Steps every state of the body DFA set @p held over @p column of the body's staged @p table into
 *  @p next, words indexed by constants so both stay in registers. @return Whether a state it lands
 *  on reaches @p accepts. */
STRINGZILLA_DEVICE sz_bool_t sz_regex_body_step_simt_(sz_regex_table_simt_t const *table, sz_u32_t accepts,
                                                      sz_u32_t columns, sz_u64_t const *held, sz_u64_t *next,
                                                      sz_u32_t column) {
    sz_u32_t accepted = 0, word, other;
    sz_u64_t bits;
#pragma unroll
    for (word = 0; word != sz_regex_body_states_simt_k / 64; ++word) next[word] = 0;
#pragma unroll
    for (word = 0; word != sz_regex_body_states_simt_k / 64; ++word)
        for (bits = held[word]; bits; bits &= bits - 1) {
            sz_u32_t const target = sz_regex_move_simt_(table, (word * 64 + sz_regex_lowest_simt_(bits)) * columns,
                                                        column);
            sz_u32_t const index = target / columns;
#pragma unroll
            for (other = 0; other != sz_regex_body_states_simt_k / 64; ++other)
                next[other] |= index / 64 == other ? (sz_u64_t)1 << (index % 64) : 0;
            accepted |= target >= accepts;
        }
    return (sz_bool_t)accepted;
}

/**
 *  @brief Walks the DFA of the body the round's flow names over @p haystack, a set of its states at
 *      a time, from the set handed across @p from to the one it hands across @p to, as
 *      @c sz_regex_looks_bodies_ steps it, recording where its lookaround holds when @p recording;
 *      @p table stages the body's rows.
 *
 *  A step landing on an accepting state marks the boundary it crossed one byte late: behind,
 *  position @c p steps the byte at it, ahead the byte before it, and the haystack's far edge takes
 *  the edge column, so the sets handed across stand where the NFA passes hand theirs.
 *
 *  @return @p state, holding the set handed on.
 */
STRINGZILLA_DEVICE sz_u64_t *sz_regex_body_walk_simt_(sz_regex_engine_t const *engine,
                                                      sz_regex_round_simt_t const *round,
                                                      sz_regex_table_simt_t const *table,
                                                      sz_regex_haystack_simt_t const *haystack, sz_size_t from,
                                                      sz_size_t to, sz_u64_t *state, sz_bool_t recording) {
    sz_u32_t const accepts = engine->bodies[round->body].accepts_begin;
    sz_u32_t const columns = engine->classes_count + 1, words = (round->body_states + 63) / 64;
    sz_bool_t const behind = (sz_bool_t)(round->flow == sz_regex_flow_behind_simt_k);
    sz_u8_t const *const text = haystack->text, *const classes = table->classes;
    sz_size_t const length = haystack->length;
    sz_u64_t held[sz_regex_body_states_simt_k / 64], next[sz_regex_body_states_simt_k / 64], any = 0;
    sz_u32_t word;
    sz_size_t position = from;
#pragma unroll
    for (word = 0; word != sz_regex_body_states_simt_k / 64; ++word)
        held[word] = word < words ? state[word] : 0, any |= held[word];
    // A walk holding no state stays empty, and only one that records reads anything on the way.
    if (!recording && !any) return state;
    if (!behind && from <= length) {
        if (!from) return state;
        position = from - 1;
    }
    else if (!behind) position = length;
    while (behind ? position < to && position <= length : position >= to) {
        sz_u32_t const column = behind ? (position == length ? columns - 1 : classes[text[position]])
                                       : (position ? classes[text[position - 1]] : columns - 1);
        sz_bool_t const accepted = sz_regex_body_step_simt_(table, accepts, columns, held, next, column);
        if (recording) sz_regex_looks_record_simt_(round, haystack->first + position, accepted ? round->members : 0);
#pragma unroll
        for (word = 0; word != sz_regex_body_states_simt_k / 64; ++word) held[word] = next[word];
        if (behind) ++position;
        else if (position) --position;
        else break;
    }
#pragma unroll
    for (word = 0; word != sz_regex_body_states_simt_k / 64; ++word)
        if (word < words) state[word] = held[word];
    return state;
}

/** Words of the sets the round's flow walks. */
STRINGZILLA_DEVICE sz_size_t sz_regex_flow_words_simt_(sz_regex_engine_t const *engine,
                                                       sz_regex_round_simt_t const *round) {
    if (round->flow == sz_regex_flow_positions_simt_k) return engine->positions_words;
    if (round->body != sz_regex_none_simt_k) return (round->body_states + 63) / 64;
    return (round->bodies_count + engine->lookarounds_count + 63) / 64;
}

/** The first position at or past @p position where the round's flow hands sets across: a unit
 *  boundary where its walks step whole codepoints, any byte otherwise, and the haystack's end or
 *  past it as itself. */
STRINGZILLA_DEVICE sz_size_t sz_regex_flow_align_simt_(sz_regex_engine_t const *engine,
                                                       sz_regex_round_simt_t const *round,
                                                       sz_regex_haystack_simt_t const *haystack, sz_size_t position) {
    sz_bool_t const units = (sz_bool_t)(round->flow == sz_regex_flow_positions_simt_k ||
                                        engine->case_sensitivity != sz_cased_k);
    return units && position < haystack->length ? sz_regex_boundary_simt_(haystack->text, haystack->length, position)
                                                : position;
}

/** Where @p haystack 's piece of the chunk of slots <tt>[low, high)</tt> hands sets across: @p begin,
 *  zero where the haystack starts in the chunk, and @p end, past the haystack's length where it
 *  ends in the chunk. */
STRINGZILLA_DEVICE void sz_regex_flow_piece_simt_(sz_regex_engine_t const *engine, sz_regex_round_simt_t const *round,
                                                  sz_regex_haystack_simt_t const *haystack, sz_size_t low,
                                                  sz_size_t high, sz_size_t *begin, sz_size_t *end) {
    *begin = low > haystack->first ? sz_regex_flow_align_simt_(engine, round, haystack, low - haystack->first) : 0;
    *end = sz_regex_flow_align_simt_(engine, round, haystack,
                                     sz_min_of_two(high - haystack->first, haystack->length + 1));
}

/** Fills @p set with the least the round's flow holds anywhere, which is also what it starts from
 *  at a haystack's edge: the loop alone for positions, nothing for a lookaround pass. */
STRINGZILLA_DEVICE void sz_regex_flow_least_simt_(sz_regex_engine_t const *engine, sz_regex_round_simt_t const *round,
                                                  sz_u64_t *set) {
    sz_size_t const words = sz_regex_flow_words_simt_(engine, round);
    sz_size_t word;
    for (word = 0; word != words; ++word) set[word] = 0;
    if (round->flow == sz_regex_flow_positions_simt_k) set[0] = 1;
}

/** Fills @p set with what the round's flow holds at a haystack's edge, where its walks start: the
 *  least, and for a body's DFA its start state. */
STRINGZILLA_DEVICE void sz_regex_flow_start_simt_(sz_regex_engine_t const *engine, sz_regex_round_simt_t const *round,
                                                  sz_u64_t *set) {
    sz_regex_flow_least_simt_(engine, round, set);
    if (round->body != sz_regex_none_simt_k) {
        sz_u32_t const start = engine->bodies[round->body].start / (engine->classes_count + 1);
        set[start / 64] |= (sz_u64_t)1 << (start % 64);
    }
}

/** Fills @p set with everything the round's flow may hold: every position, every state a body's DFA
 *  reaches over bytes, or every state of a member's body and every member pending. */
STRINGZILLA_DEVICE void sz_regex_flow_every_simt_(sz_regex_engine_t const *engine, sz_regex_round_simt_t const *round,
                                                  sz_u64_t *set) {
    sz_size_t const words = sz_regex_flow_words_simt_(engine, round);
    sz_size_t word, state, body = 0, index;
    sz_u32_t lookaround;
    if (round->flow == sz_regex_flow_positions_simt_k) {
        for (word = 0; word != words; ++word) set[word] = engine->positions_all[word];
        return;
    }
    if (round->body != sz_regex_none_simt_k) {
        for (word = 0; word != words; ++word) set[word] = round->body_reach[word];
        return;
    }
    for (word = 0; word != words; ++word) set[word] = 0;
    // The bodies follow one another, each closing with the match state naming its lookaround.
    for (state = 0; state != round->bodies_count; ++state) {
        sz_regex_nfa_state_t const node = engine->nfa_states[round->bodies_begin + state];
        if (node.kind != sz_regex_nfa_match_k) continue;
        if (round->members & (sz_u32_t)sz_regex_look_lookaround_k << node.target)
            for (index = body; index <= state; ++index) set[index / 64] |= (sz_u64_t)1 << (index % 64);
        body = state + 1;
    }
    for (lookaround = 0; lookaround != engine->lookarounds_count; ++lookaround)
        if (round->members & (sz_u32_t)sz_regex_look_lookaround_k << lookaround) {
            index = round->bodies_count + lookaround;
            set[index / 64] |= (sz_u64_t)1 << (index % 64);
        }
}

/**
 *  @brief Walks the round's flow over @p haystack from the set at @p state handed across @p from to
 *      the set it hands across @p to, recording where its lookarounds hold when @p recording.
 *  @param[in] table The rows of the body whose DFA the flow steps when @p dfa, a constant each
 *      kernel compiles for alone.
 *  @return The set holding the result, @p state or @p next.
 */
STRINGZILLA_DEVICE sz_u64_t *sz_regex_flow_walk_simt_(sz_regex_engine_t const *engine,
                                                      sz_regex_round_simt_t const *round, sz_regex_pike_simt_t *pike,
                                                      sz_regex_table_simt_t const *table,
                                                      sz_regex_haystack_simt_t const *haystack, sz_size_t from,
                                                      sz_size_t to, sz_u64_t *state, sz_u64_t *next,
                                                      sz_bool_t recording, sz_bool_t dfa) {
    if (dfa) return sz_regex_body_walk_simt_(engine, round, table, haystack, from, to, state, recording);
    if (round->flow == sz_regex_flow_behind_simt_k)
        return sz_regex_behind_walk_simt_(engine, round, pike, haystack, from, to, state, next, recording);
    if (round->flow == sz_regex_flow_ahead_simt_k)
        return sz_regex_ahead_walk_simt_(engine, round, pike, haystack, from, to, state, next, recording);
    return sz_regex_positions_walk_simt_(engine, round, haystack, (sz_bool_t)(engine->case_sensitivity != sz_cased_k),
                                         sz_min_of_two(from, haystack->length), to, state, next, STRINGZILLA_NULL, 0,
                                         0);
}

/** The undecided elements of @p undecided that @p set holds, a bit each. */
STRINGZILLA_DEVICE sz_u64_t sz_regex_decided_simt_(sz_regex_undecided_simt_t const *undecided, sz_u64_t const *set) {
    sz_u64_t bits = 0;
    sz_u32_t index;
    for (index = 0; index != undecided->count; ++index)
        bits |= (set[undecided->indices[index] / 64] >> (undecided->indices[index] % 64) & 1) << index;
    return bits;
}

/** The rows of @p rows that the bits of @p bits pick, united. */
STRINGZILLA_DEVICE sz_u64_t sz_regex_apply_simt_(sz_u64_t const *rows, sz_u64_t bits) {
    sz_u64_t united = 0;
    for (; bits; bits &= bits - 1) united |= rows[sz_regex_lowest_simt_(bits)];
    return united;
}

/** The haystack whose set the chunk of slots <tt>[low, high)</tt> hands downstream, and whether it hands
 *  one: walks right to left hand on the haystack that started before the chunk, lookbehind passes
 *  the one running past it. */
STRINGZILLA_DEVICE sz_bool_t sz_regex_flow_handing_simt_(sz_strings_view_t haystacks,
                                                         sz_regex_round_simt_t const *round, sz_size_t low,
                                                         sz_size_t high, sz_regex_haystack_simt_t *haystack) {
    sz_bool_t const backward = (sz_bool_t)(round->flow != sz_regex_flow_behind_simt_k);
    *haystack = sz_regex_haystack_simt_(haystacks, sz_regex_haystack_at_simt_(haystacks, backward ? low : high - 1, 0));
    return (sz_bool_t)(backward ? haystack->first < low : haystack->first + haystack->length >= high);
}

/**
 *  @brief Where the round's flow over @p haystack starts exactly within the segment, in @p edge:
 *      the haystack's own edge upstream, or the segment's where the haystack runs past it.
 *  @return The set the segment edge keeps, or null at the haystack's edge, where the flow starts
 *      from its least.
 */
STRINGZILLA_DEVICE sz_u64_t const *sz_regex_flow_edge_simt_(sz_regex_engine_t const *engine,
                                                            sz_regex_round_simt_t const *round,
                                                            sz_regex_haystack_simt_t const *haystack, sz_size_t *edge) {
    sz_bool_t const backward = (sz_bool_t)(round->flow != sz_regex_flow_behind_simt_k);
    if (backward && haystack->first + haystack->length >= round->segment_end) {
        *edge = sz_regex_flow_align_simt_(engine, round, haystack, round->segment_end - haystack->first);
        return round->handoff_edges + (round->segment + 1) * round->set_words;
    }
    if (!backward && haystack->first < round->segment_begin) {
        *edge = sz_regex_flow_align_simt_(engine, round, haystack, round->segment_begin - haystack->first);
        return round->handoff_edges + round->segment * round->set_words;
    }
    *edge = backward ? haystack->length + 1 : 0;
    return STRINGZILLA_NULL;
}

/** Whether a chunk's priming left a frontier to compose, not its entry or a wide stretch. */
STRINGZILLA_DEVICE sz_bool_t sz_regex_composing_simt_(sz_u32_t count) {
    return (sz_bool_t)(count - 1 < sz_regex_priming_states_simt_k);
}

/** Where a priming walk over @p haystack stands at the right edge of a chunk ending at @p high. */
STRINGZILLA_DEVICE sz_size_t sz_regex_prime_edge_simt_(sz_bool_t folded, sz_regex_haystack_simt_t const *haystack,
                                                       sz_size_t high) {
    return folded ? sz_regex_boundary_simt_(haystack->text, haystack->length, high - haystack->first)
                  : high - haystack->first - 1;
}

/** The 16-slot mark whose checkpoint stands at the unit boundary of slot @p slot, reached over
 *  a unit of @p stepped bytes, its checkpoint being the first unit boundary at or past it;
 *  zero for none. */
STRINGZILLA_DEVICE sz_size_t sz_regex_mark_at_simt_(sz_size_t slot, sz_size_t stepped) {
    sz_size_t const phase = slot % sz_regex_checkpoint_slots_simt_k;
    return phase < stepped ? slot - phase : 0;
}

/**
 *  @brief Steps @p state forward from unit boundary @p at of @p haystack, short of @p stop, to
 *      the next boundary a checkpoint may stand at: the next 16-slot mark over bytes, the next
 *      unit over folds.
 *  @return Where the walk stands; @p stepped takes the bytes of its last unit, @p last its last
 *      accepting state, @p last_end the boundary that match ends at, and @p last_at the boundary of
 *      the unit whose step reached it.
 */
STRINGZILLA_DEVICE sz_size_t sz_regex_advance_simt_(sz_regex_table_simt_t const *table, sz_bool_t folded,
                                                    sz_regex_haystack_simt_t const *haystack, sz_size_t at,
                                                    sz_size_t stop, sz_u32_t accepts, sz_u32_t *state, sz_u32_t *last,
                                                    sz_size_t *last_end, sz_size_t *last_at, sz_size_t *stepped) {
    sz_utf8_folded_image_t image;
    sz_size_t unit_end;
    sz_u32_t block, byte;
    if (!folded) {
        sz_size_t const phase = (haystack->first + 1 + at) % sz_regex_checkpoint_slots_simt_k;
        sz_u32_t const count = (sz_u32_t)sz_min_of_two((sz_size_t)sz_regex_checkpoint_slots_simt_k - phase,
                                                       sz_min_of_two(haystack->length, stop) - at);
        sz_u32_t run_last = 0, run_offset = 0;
        sz_u32_t const walked = sz_regex_run_simt_(table, state, haystack->text + at, haystack->first + at, count,
                                                   accepts, &run_last, &run_offset);
        if (run_last) *last = run_last, *last_end = *last_at = at + run_offset;
        *stepped = 1;
        return at + walked;
    }
    unit_end = sz_regex_unit_after_simt_(sz_true_k, haystack->text, haystack->length, at, &image);
    block = sz_regex_block_at_simt_(table, haystack->first + at);
    for (byte = 0; byte != image.length && *state != 0; ++byte, block = 0) {
        *state = sz_regex_move_simt_(table, *state, block + table->classes[sz_regex_image_byte_simt_(&image, byte)]);
        if (*state >= accepts) *last = *state, *last_end = byte ? unit_end : at, *last_at = at;
    }
    *stepped = unit_end - at;
    return unit_end;
}

/** The outcome the walk of @p id settles: its link's, or its group's when @p id marks a group. */
STRINGZILLA_DEVICE sz_regex_result_simt_t *sz_regex_walked_simt_(sz_regex_round_simt_t const *round, sz_u32_t id) {
    return id & sz_regex_group_simt_k ? round->groups + (id & ~sz_regex_group_simt_k) : round->links + id;
}

/** Hands the walk of @p id on to a warp in @p state at the boundary in @p slot, which walks it on,
 *  or pauses it there when its next step would read the lookaround table past its window. */
STRINGZILLA_DEVICE void sz_regex_hand_simt_(sz_regex_round_simt_t const *round, sz_u32_t id, sz_u32_t state,
                                            sz_size_t slot) {
    sz_regex_continuation_simt_t *const continuations = (sz_regex_continuation_simt_t *)round->links_next;
    sz_u32_t const handed = atomicAdd(round->counters + sz_regex_handed_counter_simt_k, 1u);
    continuations[handed].link = id, continuations[handed].state = state, continuations[handed].slot = slot;
}

/**
 *  @brief Walks the link or group @p id on from unit boundary @p at of @p haystack in @p state
 *      until it dies, ends its haystack, or meets a checkpoint of a later chunk of the segment in
 *      the same state.
 *
 *  Otherwise it hands the walk to a warp once it outlives its share of bytes or would read the
 *  lookaround table past its window. Over the folds of whole units when @p folded, it meets
 *  checkpoints where the end walks leave them.
 */
STRINGZILLA_DEVICE void sz_regex_link_walk_simt_(sz_regex_engine_t const *engine, sz_regex_round_simt_t const *round,
                                                 sz_regex_table_simt_t const *table, sz_bool_t folded, sz_u32_t id,
                                                 sz_u32_t state, sz_regex_haystack_simt_t const *haystack,
                                                 sz_size_t at) {
    sz_size_t const begin = round->segment_begin, end = round->segment_end, edge = haystack->first + 1 + at;
    sz_size_t const stop = table->looks->table ? table->looks->end - haystack->first : haystack->length;
    sz_size_t const handover = at + sz_regex_link_bytes_simt_k;
    sz_u32_t const accepts = engine->forward_accepts_begin;
    sz_u32_t last = 0, inherited = sz_regex_none_simt_k;
    sz_size_t last_end = 0, last_at = 0, stepped = 1;
    sz_regex_result_simt_t result;
    result.link = sz_regex_none_simt_k, result.pattern = sz_regex_none_simt_k, result.end = 0;
    while (state != 0) {
        sz_size_t due;
        if (sz_regex_past_simt_(table->looks, haystack->first + at)) {
            sz_regex_hand_simt_(round, id, state, haystack->first + at);
            break;
        }
        if (at == haystack->length) {
            state = sz_regex_step_edge_simt_(table, state, haystack->first + at);
            if (state >= accepts) last = state, last_end = at;
            break;
        }
        due = sz_regex_mark_at_simt_(haystack->first + 1 + at, stepped);
        if (due > edge && due <= end) {
            sz_size_t const owner = (due - 1 - begin) >> round->chunk_shift;
            sz_regex_checkpoint_simt_t const entry =
                round->checkpoints[(due - begin) / sz_regex_checkpoint_slots_simt_k];
            if (round->chunk_offsets[owner + 1] != round->chunk_offsets[owner] && entry.state == state) {
                result.link = entry.link;
                if (entry.pattern != sz_regex_none_simt_k) inherited = entry.pattern, result.end = begin + entry.end;
                break;
            }
        }
        if (due && at >= handover) {
            sz_regex_hand_simt_(round, id, state, haystack->first + at);
            break;
        }
        at = sz_regex_advance_simt_(table, folded, haystack, at, stop, accepts, &state, &last, &last_end, &last_at,
                                    &stepped);
    }
    if (inherited != sz_regex_none_simt_k) result.pattern = inherited;
    else if (last) result.pattern = sz_regex_pattern_simt_(engine, last), result.end = haystack->first + last_end;
    *sz_regex_walked_simt_(round, id) = result;
}

/** Where piece @p index of those a warp walks from @p at begins, at most @p stop: @p at itself
 *  for the first, else its share of bytes on, moved to the first unit boundary at or past it
 *  when @p folded. */
STRINGZILLA_DEVICE sz_size_t sz_regex_piece_simt_(sz_bool_t folded, sz_regex_haystack_simt_t const *haystack,
                                                  sz_size_t at, sz_size_t index, sz_size_t stop) {
    sz_size_t const piece = at + index * sz_regex_piece_bytes_simt_k;
    if (!index) return at;
    if (piece >= stop) return stop;
    return folded ? sz_min_of_two(sz_regex_boundary_simt_(haystack->text, haystack->length, piece), stop) : piece;
}

/** Follows one round's hops of pointer doubling from @p result over @p source, links or groups
 *  alike, the match a later hop reaches winning. */
STRINGZILLA_DEVICE sz_regex_result_simt_t sz_regex_hops_simt_(sz_regex_result_simt_t const *source,
                                                              sz_regex_result_simt_t result) {
    sz_u32_t hop;
    for (hop = 1; hop != sz_regex_round_hops_simt_k && result.link != sz_regex_none_simt_k; ++hop) {
        sz_regex_result_simt_t const next = source[result.link & ~sz_regex_group_simt_k];
        result.link = next.link;
        if (next.pattern != sz_regex_none_simt_k) result.pattern = next.pattern, result.end = next.end;
    }
    return result;
}

/** Composes @p left, a run of chunks, with @p right, the run right after it, into @p left: each
 *  state @p right may enter with leaves where @p left leaves from @p right's exit, a match starting
 *  in either run. */
STRINGZILLA_DEVICE void sz_regex_compose_exits_simt_(sz_regex_exits_simt_t *left, sz_regex_exits_simt_t const *right) {
    sz_regex_exits_simt_t composed;
    sz_u32_t index, known;
    composed.count = right->count, composed.founds = 0;
#pragma unroll
    for (index = 0; index != sz_regex_priming_states_simt_k; ++index) {
        sz_u32_t const middle = right->exits[index];
        sz_u32_t exit = left->exits[0], found = left->founds & 1u;
        if (left->count) exit = 0, found = 0;
#pragma unroll
        for (known = 0; known != sz_regex_priming_states_simt_k; ++known)
            if (known < left->count && left->entries[known] == middle)
                exit = left->exits[known], found = (left->founds >> known) & 1u;
        composed.entries[index] = right->entries[index];
        composed.exits[index] = exit;
        composed.founds |= (found | ((right->founds >> index) & 1u)) << index;
    }
    *left = composed;
}

#pragma endregion Steps

/*  A walk answers for one value at one granularity: a lane walking it alone, a @c sz_simt_group_t
 *  walking it together, or a lane walking one chunk of it with the start-finding state carried from
 *  the chunk after it. A walk reads no @c blockIdx and declares no @c __shared__, so a product
 *  calls it from a kernel of its own under any launch, having staged @c reverse_rows with
 *  @c sz_regex_stage_table_simt_ into @c sz_regex_staged_bytes_simt_ bytes. Its engine is one
 *  @c sz_regex_engine_init_serial compiled into memory the host writes and the device reads, taken
 *  by value with its tables read where they were compiled, or rebased onto a device block with
 *  @c sz_regex_engine_rebase_simt_; the @c first walks take a byte-exact engine whose
 *  @c reverse_rows exist and which holds no lookarounds. */
#pragma region Walks

/**
 *  @brief Walks bytes @p chunk_begin to @p chunk_end of a byte-exact @p text right to left from the
 *      start-finding state @p carried, answering whether a match starts inside them.
 *  @param[in] table The start-finding automaton, staged from @c reverse_rows with no lookarounds.
 *  @param[in] length Bytes of @p text, past which the chunk is cut.
 *  @param[in,out] carried The state after every byte from @p chunk_end on, which
 *      @c reverse_starts[0] is at the text's end; the state at @p chunk_begin on return.
 *
 *  Chunks walked from the text's end towards its start, each carrying the last one's state, answer
 *  exactly as one walk over all of it; the chunk at the text's start also takes its edge column, so
 *  a match starting at the first byte counts.
 */
STRINGZILLA_DEVICE sz_bool_t sz_regex_first_chunk_simt_(sz_regex_engine_t const *engine,
                                                        sz_regex_table_simt_t const *table, sz_cptr_t text,
                                                        sz_size_t length, sz_size_t chunk_begin, sz_size_t chunk_end,
                                                        sz_u32_t *carried) {
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    sz_u32_t const accepts = engine->reverse_accepts_begin;
    sz_size_t at = sz_min_of_two(chunk_end, length);
    sz_u32_t state = *carried;
    sz_bool_t found = sz_false_k;
    while (at > chunk_begin && !found) {
        sz_u32_t const count = (sz_u32_t)sz_min_of_two(at - chunk_begin, (sz_size_t)sz_regex_run_bytes_simt_k);
        found = (sz_bool_t)(sz_regex_run_back_simt_(table, &state, bytes + at, at, count, accepts) != 0);
        at -= count;
    }
    *carried = state;
    if (!found && chunk_begin == 0) found = (sz_bool_t)(sz_regex_step_edge_simt_(table, state, 0) >= accepts);
    return found;
}

/** Whether any pattern of a byte-exact @p engine with no lookarounds matches in @p text, one lane
 *  walking all of it from its end through the start-finding automaton @p table. */
STRINGZILLA_DEVICE sz_bool_t sz_regex_first_thread_simt_(sz_regex_engine_t const *engine,
                                                         sz_regex_table_simt_t const *table, sz_cptr_t text,
                                                         sz_size_t length) {
    sz_u32_t carried = engine->reverse_starts[0];
    return sz_regex_first_chunk_simt_(engine, table, text, length, 0, length, &carried);
}

/**
 *  @brief Whether any pattern of a byte-exact @p engine with no lookarounds matches in @p text, the
 *      lanes of @p group walking it together through the start-finding automaton @p table, staged
 *      with its column images, and every lane answering the same.
 *
 *  Lane @c l takes chunk @c l of @p text cut into @c group->lanes and primes the state at its
 *  right edge with @ref sz_regex_prime_simt_. Where priming leaves a frontier, the lane walks its
 *  chunk from each of its states, and @ref sz_regex_compose_exits_simt_ chains the lanes' exits
 *  from the text's end in log2 of the lanes' rounds of shuffles. Where any chunk is too wide to
 *  prime, every lane walks the whole text in order from its end instead.
 */
STRINGZILLA_DEVICE sz_bool_t sz_regex_first_group_simt_(sz_regex_engine_t const *engine,
                                                        sz_regex_table_simt_t const *table, sz_cptr_t text,
                                                        sz_size_t length, sz_simt_group_t const *group) {
    sz_size_t const chunk_bytes = sz_size_divide_round_up(length, (sz_size_t)group->lanes);
    sz_size_t const chunk_begin = sz_min_of_two((sz_size_t)group->lane * chunk_bytes, length);
    sz_size_t const chunk_end = sz_min_of_two(chunk_begin + chunk_bytes, length);
    sz_regex_haystack_simt_t const haystack = {(sz_u8_t const *)text, length, 0};
    sz_regex_exits_simt_t exits;
    sz_u32_t set[sz_regex_priming_states_simt_k], count = 0, index, distance;
    sz_u32_t const primed = sz_regex_prime_simt_(engine, table, table->images, &haystack, chunk_end,
                                                 sz_regex_priming_bytes_simt_k, set, &count);
    if (group->any(!primed && !count)) return sz_regex_first_thread_simt_(engine, table, text, length);
    if (primed) set[0] = primed, count = 1;
    exits.count = primed ? 0 : count, exits.founds = 0;
#pragma unroll
    for (index = 0; index != sz_regex_priming_states_simt_k; ++index) {
        sz_u32_t carried = set[index];
        if (index >= count) continue;
        exits.entries[index] = carried;
        exits.founds |=
            (sz_u32_t)sz_regex_first_chunk_simt_(engine, table, text, length, chunk_begin, chunk_end, &carried)
            << index;
        exits.exits[index] = carried;
    }
    // Every lane shuffles every round, as shuffles are collective; one past the last keeps its own.
    for (distance = 1; distance < group->lanes; distance *= 2) {
        sz_u32_t const source = group->lane + distance;
        sz_regex_exits_simt_t right;
        right.count = group->shuffle(exits.count, source), right.founds = group->shuffle(exits.founds, source);
#pragma unroll
        for (index = 0; index != sz_regex_priming_states_simt_k; ++index)
            right.entries[index] = group->shuffle(exits.entries[index], source),
            right.exits[index] = group->shuffle(exits.exits[index], source);
        if (source < group->lanes) sz_regex_compose_exits_simt_(&exits, &right);
    }
    return group->any((int)(exits.founds & 1u)) ? sz_true_k : sz_false_k;
}

#pragma endregion Walks

#endif // STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_REGEX_SIMT_CUH_
