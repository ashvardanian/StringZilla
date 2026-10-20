/**
 *  @file include/stringzilla/regex/simt.cuh
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The kernels CUDA and ROCm share for regular expressions: one thread per chunk of the
 *      corpus marks the starts and walks the ends, and pointer doubling settles every chain that
 *      crosses a chunk's edge.
 *
 *  A haystack owns one slot per byte and one more ahead of them: a haystack with first slot @c V
 *  marks a match starting at byte @c p in slot `V + p`, which is also the slot of byte `p - 1`. The
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
 *  cross-lane steps as parameters each vendor's kernel binds, as `levenshtein/simt.cuh` does.
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
 *  Each vendor's host side, in `regex/cuda.cuh` and `regex/rocm.cuh`, sizes and launches these.
 *
 *  @sa include/stringzilla/regex.h
 *  @sa include/stringzilla/regex/metal.metal, the Metal sibling these mirror kernel for kernel
 */
#ifndef STRINGZILLA_REGEX_SIMT_CUH_
#define STRINGZILLA_REGEX_SIMT_CUH_

#include "stringzilla/types.cuh"

#include "stringzilla/regex/serial.h"
#include "stringzilla/utf8_uncased_fold/simt.cuh" // `sz_utf8_fold_next_simt_`, `sz_utf8_fold_prev_simt_`

#if STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Shapes

/** Threads every kernel here launches with. */
enum { sz_regex_threads_per_block_simt_k = 256 };

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

/** Where lookarounds hold, one entry of look bits per slot of the window `[begin, end)`, from a
 *  ring of two segments' halves indexed by `slot & mask`; null without any. A walk that cannot
 *  pause past the window raises @c refused instead, which fails the call. */
typedef struct sz_regex_looks_simt_t {
    sz_u32_t *table, *refused;
    sz_size_t begin, end, mask;
} sz_regex_looks_simt_t;

/** What one segment hands the next: where the haystack straddling them resumes, whether an empty
 *  match there is refused, the matches kept so far, the last kept match, the drift a rewrite or the
 *  rows an extraction made so far, the kept matches of the extracted pattern so far, and how many
 *  of them preceded the straddling haystack. */
typedef struct sz_regex_carry_simt_t {
    sz_u64_t cursor, refusing, running, last_start, last_end, summed, counted, straddle;
} sz_regex_carry_simt_t;

/**
 *  @brief What one round's kernels read and write, every pointer one the device addresses.
 *
 *  The outputs of the verb a call serves are set and the others null: matches, counts, a rewrite or
 *  rows. The set kernels read which flow they run, the lookarounds it walks, the lookaround whose
 *  body's DFA it steps instead of the NFA, and where in a walk's share its sets sit.
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

#pragma endregion Shapes

#pragma region Device Helpers

/** One automaton's transitions as a walk reads them: the class map, the first column of each block
 *  of lookarounds and the rows' head staged in shared memory, the rest of the rows in global
 *  memory, and the round's lookaround table whose entries pick the blocks, null without
 *  lookarounds. The table is reached through the round, as NVVM takes the pointers of a copy held
 *  beside staged ones for shared memory. */
typedef struct sz_regex_table_simt_t {
    sz_u8_t const *classes;
    sz_u32_t const *blocks, *head, *rows;
    sz_regex_looks_simt_t const *looks;
    sz_u32_t head_cells, edge_column;
} sz_regex_table_simt_t;

/** Blocks of lookaround columns a row holds at most, one per set of the columned lookarounds, each
 *  staged by its first column. */
enum { sz_regex_blocks_simt_k = 1 << sz_regex_columned_lookarounds_k };

/** Bytes of shared memory the class map and the blocks take ahead of the staged rows, as the
 *  host sizes it. */
enum { sz_regex_staged_bytes_simt_k = 256 + sz_regex_blocks_simt_k * sizeof(sz_u32_t) };

/** Stages @p cells of @p rows, the class map and the blocks into dynamic shared memory; every
 *  thread of the block takes part, so a kernel calls it before any thread can leave. */
STRINGZILLA_DEVICE sz_regex_table_simt_t sz_regex_stage_simt_(sz_regex_engine_t const *engine,
                                                              sz_regex_round_simt_t const *round, sz_u32_t const *rows,
                                                              sz_u32_t cells) {
    extern __shared__ sz_u8_t sz_regex_staged_simt_[];
    sz_u32_t *const blocks = (sz_u32_t *)(sz_regex_staged_simt_ + 256);
    sz_u32_t *const head = (sz_u32_t *)(sz_regex_staged_simt_ + sz_regex_staged_bytes_simt_k);
    sz_regex_table_simt_t table;
    sz_u32_t index;
    table.classes = sz_regex_staged_simt_, table.blocks = blocks, table.head = head;
    table.rows = rows, table.head_cells = sz_min_of_two(cells, round->staged_cells);
    table.looks = &round->looks, table.edge_column = engine->classes_count;
    for (index = threadIdx.x; index < 256; index += blockDim.x)
        sz_regex_staged_simt_[index] = engine->byte_to_class[index];
    for (index = threadIdx.x; index < sz_regex_blocks_simt_k; index += blockDim.x)
        blocks[index] = (sz_u32_t)sz_regex_looks_block_(engine->classes_count, engine->lookarounds_count,
                                                        index * sz_regex_look_lookaround_k);
    for (index = threadIdx.x; index < table.head_cells; index += blockDim.x) head[index] = rows[index];
    __syncthreads();
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
#pragma unroll sz_regex_run_bytes_simt_k
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

/** Bytes of text a block stages for one tile of chunks: the slots of its chunks, a byte before them
 *  and the alignment of its first granule, in rows of 64 bytes padded by one word, so lanes a chunk
 *  apart read distinct banks. */
enum {
    sz_regex_tile_bytes_simt_k = (int)sz_regex_threads_per_block_simt_k * (int)sz_regex_chunk_slots_simt_k + 17,
    sz_regex_staged_text_bytes_simt_k = (sz_regex_tile_bytes_simt_k + 63) / 64 * 68,
};

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

/** One haystack, as a walk reads it: its bytes, its length, and its first slot. */
typedef struct sz_regex_haystack_simt_t {
    sz_u8_t const *text;
    sz_size_t length, first;
} sz_regex_haystack_simt_t;

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

#pragma endregion Device Helpers

#pragma region Folded Units

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

/** Walks @p state right to left over the folds of the @p haystack units ending in `(stop, at]`. */
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

#pragma endregion Folded Units

#pragma region Position Sets

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
 *      `(stop, from]`, marking in @p bits every start it finds in slots `[low, high)`.
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

#pragma endregion Position Sets

#pragma region PikeVM

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
 *  @param[inout] offset The folded bytes into the unit the first try starts at, then those the
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
 *  `[start, end)` of @p haystack, starting inside the folds of its first unit wherever the
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

#pragma endregion PikeVM

#pragma region Lookaround Passes

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
 *      `[from, to)`, and its end when @p to passes it, as @c sz_regex_looks_serial_ walks it.
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
 *      `[to, from)`, as @c sz_regex_looks_serial_ walks it.
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

#pragma endregion Lookaround Passes

#pragma region Scan Kernels

/** Scans one ticket's own tile of @p values in place, as @c sz_substrings_scan_tile_simt_ does,
 *  seeded by the totals the tiles before it publish in @p states, so one launch makes a whole scan;
 *  it clears @p stale, the states the next scan takes. */
static __global__ void sz_regex_scan_simt_kernel_(sz_size_t *values, sz_size_t count, sz_size_t elements_per_tile,
                                                  sz_size_t *chain, sz_u64_t *states, sz_u64_t *stale) {
    __shared__ sz_size_t shared[sz_regex_threads_per_block_simt_k];
    sz_size_t const tiles = sz_size_divide_round_up(count, elements_per_tile);
    sz_size_t tile, begin, end, running = 0, first;
    if (blockIdx.x >= tiles) return;
    tile = sz_chain_ticket_simt_(chain);
    begin = tile * elements_per_tile;
    end = sz_min_of_two(begin + elements_per_tile, count);
    for (first = tile * blockDim.x + threadIdx.x; first < sz_chain_tiles_max_k; first += tiles * blockDim.x)
        stale[first] = 0;
    for (first = begin; first < end; first += blockDim.x) {
        sz_size_t const index = first + threadIdx.x;
        sz_size_t total = 0;
        sz_block_scan_simt_(index < end ? values[index] : 0, shared, &total);
        running += total;
        __syncthreads();
    }
    running = sz_chain_look_back_simt_(states, tile, running);
    if (tile + 1 == tiles && threadIdx.x == 0) *chain = 0;
    for (first = begin; first < end; first += blockDim.x) {
        sz_size_t const index = first + threadIdx.x;
        sz_size_t const value = index < end ? values[index] : 0;
        sz_size_t total = 0;
        sz_size_t const exclusive = sz_block_scan_simt_(value, shared, &total);
        if (index < end) values[index] = running + exclusive;
        running += total;
        __syncthreads();
    }
}

#pragma endregion Scan Kernels

#pragma region Set Kernels

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

/** Where @p haystack 's piece of the chunk of slots `[low, high)` hands sets across: @p begin,
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

/** The haystack whose set the chunk of slots `[low, high)` hands downstream, and whether it hands
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

/**
 *  @brief Bounds the set each chunk's walk of the round's flow hands downstream, one thread per
 *      chunk in each thread's share of the walks.
 *
 *  The bound is exact where the walk starts at its haystack's edge or at the segment's, from the
 *  set the segment edge keeps, else the set handed on from the least input, kept in @c handoffs,
 *  and the undecided elements the set handed on from every input adds.
 *
 *  Downstream is left for positions and lookahead passes, right for lookbehind passes. While more
 *  elements stay undecided than a composition holds, the walk starts further upstream, never past
 *  the segment's edge; past @ref sz_regex_widening_bytes_simt_k it marks the chunk wide, one past a
 *  composition, for the pieces kernel.
 */
STRINGZILLA_DEVICE void sz_regex_bound_simt_(sz_regex_engine_t const *engine, sz_strings_view_t haystacks,
                                             sz_regex_round_simt_t const *round, sz_bool_t dfa) {
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const width = (sz_size_t)1 << round->chunk_shift, stride = round->set_words;
    sz_bool_t const backward = (sz_bool_t)(round->flow != sz_regex_flow_behind_simt_k);
    sz_size_t const words = sz_regex_flow_words_simt_(engine, round);
    sz_regex_pike_simt_t pike;
    sz_regex_table_simt_t table;
    sz_u64_t *sets;
    sz_size_t chunk;
    // Every thread stages the body's rows before any leaves.
    if (dfa)
        table = sz_regex_stage_simt_(engine, round, engine->bodies[round->body].rows,
                                     round->body_states * (engine->classes_count + 1));
    if (thread >= round->walks_count) return;
    pike = sz_regex_pike_carve_simt_(round, thread, engine), pike.slots_count = 0;
    sets = (sz_u64_t *)(round->walks + thread * round->walk_bytes + round->walk_sets);
    for (chunk = thread; chunk < round->chunks_count; chunk += round->walks_count) {
        sz_size_t const low = round->segment_begin + chunk * width;
        sz_size_t const high = sz_min_of_two(low + width, round->segment_end);
        sz_regex_undecided_simt_t *const undecided = round->undecided + chunk;
        sz_u64_t *const handoff = round->handoffs + chunk * stride;
        sz_u64_t const *kept;
        sz_regex_haystack_simt_t haystack;
        sz_size_t begin, end, from, to, edge, span, word, count;
        sz_u64_t *result, bits;
        undecided->count = 0;
        if (!sz_regex_flow_handing_simt_(haystacks, round, low, high, &haystack)) continue;
        sz_regex_flow_piece_simt_(engine, round, &haystack, low, high, &begin, &end);
        from = backward ? end : begin, to = backward ? begin : end;
        kept = sz_regex_flow_edge_simt_(engine, round, &haystack, &edge);
        for (span = width;; span *= 2) {
            sz_bool_t const exact = (sz_bool_t)(from == edge);
            if (exact && kept)
                for (word = 0; word != words; ++word) sets[word] = kept[word];
            else if (exact) sz_regex_flow_start_simt_(engine, round, sets);
            else sz_regex_flow_least_simt_(engine, round, sets);
            result = sz_regex_flow_walk_simt_(engine, round, &pike, &table, &haystack, from, to, sets, sets + stride,
                                              sz_false_k, dfa);
            for (word = 0; word != words; ++word) handoff[word] = result[word];
            if (exact) break;
            sz_regex_flow_every_simt_(engine, round, sets);
            result = sz_regex_flow_walk_simt_(engine, round, &pike, &table, &haystack, from, to, sets, sets + stride,
                                              sz_false_k, dfa);
            for (word = 0, count = 0; word != words; ++word)
                count += (sz_size_t)__popcll(result[word] & ~handoff[word]);
            // A body's DFA holds one state, so one it reaches from every state is the answer.
            if (round->body != sz_regex_none_simt_k && count == 1) {
                for (word = 0; word != words; ++word) handoff[word] = result[word];
                break;
            }
            if (count <= sz_regex_undecided_simt_k) {
                for (word = 0, count = 0; word != words; ++word)
                    for (bits = result[word] & ~handoff[word]; bits; bits &= bits - 1)
                        undecided->indices[count++] = (sz_u32_t)(word * 64 + sz_regex_lowest_simt_(bits));
                undecided->count = (sz_u32_t)count;
                break;
            }
            if (span >= sz_regex_widening_bytes_simt_k) {
                undecided->count = sz_regex_undecided_simt_k + 1;
                break;
            }
            // More elements stay undecided than a composition holds, so the walk starts further up.
            if (backward)
                from = edge - from > span ? sz_regex_flow_align_simt_(engine, round, &haystack, from + span) : edge;
            else from = from - edge > span ? sz_regex_flow_align_simt_(engine, round, &haystack, from - span) : edge;
        }
    }
}

/** @ref sz_regex_bound_simt_ over the pass's NFA or positions. */
static __global__ void sz_regex_bound_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                   sz_regex_round_simt_t round) {
    sz_regex_bound_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_bound_simt_ over a body's DFA. */
static __global__ void sz_regex_bound_dfa_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                       sz_regex_round_simt_t round) {
    sz_regex_bound_simt_(&engine, haystacks, &round, sz_true_k);
}

/**
 *  @brief Settles every wide chunk of the round's flow as @ref sz_regex_prime_pieces_simt_kernel_
 *      settles priming, one thread per run of unsettled chunks holding one in each thread's share
 *      of the walks.
 *
 *  The run's most upstream wide chunk walks its haystack downstream from the nearest exact set
 *  upstream of it: a settled chunk's handoff, or the set the segment edge keeps, so no walk leaves
 *  its segment. It writes the exact handoff of every chunk it passes down to the run's most
 *  downstream wide one, which then holds nothing undecided, and the chunks past them compose onto
 *  those handoffs as onto any settled chunk's.
 */
STRINGZILLA_DEVICE void sz_regex_bound_pieces_simt_(sz_regex_engine_t const *engine, sz_strings_view_t haystacks,
                                                    sz_regex_round_simt_t const *round, sz_bool_t dfa) {
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const width = (sz_size_t)1 << round->chunk_shift, stride = round->set_words;
    sz_bool_t const backward = (sz_bool_t)(round->flow != sz_regex_flow_behind_simt_k);
    sz_size_t const words = sz_regex_flow_words_simt_(engine, round);
    sz_regex_pike_simt_t pike;
    sz_regex_table_simt_t table;
    sz_u64_t *sets;
    sz_size_t chunk;
    // Every thread stages the body's rows before any leaves.
    if (dfa)
        table = sz_regex_stage_simt_(engine, round, engine->bodies[round->body].rows,
                                     round->body_states * (engine->classes_count + 1));
    if (thread >= round->walks_count) return;
    pike = sz_regex_pike_carve_simt_(round, thread, engine), pike.slots_count = 0;
    sets = (sz_u64_t *)(round->walks + thread * round->walk_bytes + round->walk_sets);
    for (chunk = thread; chunk < round->chunks_count; chunk += round->walks_count) {
        sz_size_t const low = round->segment_begin + chunk * width;
        sz_size_t const high = sz_min_of_two(low + width, round->segment_end);
        sz_u64_t const *anchor = STRINGZILLA_NULL;
        sz_u64_t *state = sets, *other = sets + stride, *result;
        sz_regex_haystack_simt_t haystack;
        sz_size_t next, last, previous, walked, begin, end, word;
        sz_bool_t owner = sz_true_k;
        if (round->undecided[chunk].count <= sz_regex_undecided_simt_k) continue;
        sz_regex_flow_handing_simt_(haystacks, round, low, high, &haystack);

        // The upstream anchor, past undecided chunks only, as a wide chunk further up owns the run.
        for (next = backward ? chunk + 1 : chunk - 1;; next = backward ? next + 1 : next - 1) {
            sz_u32_t count;
            if (next == (backward ? round->chunks_count : STRINGZILLA_SIZE_MAX)) {
                anchor = round->handoff_edges + (backward ? round->segment + 1 : round->segment) * stride;
                break;
            }
            count = round->undecided[next].count;
            if (count > sz_regex_undecided_simt_k) {
                owner = sz_false_k;
                break;
            }
            if (count) continue;
            anchor = round->handoffs + next * stride;
            break;
        }
        if (!owner) continue;

        // The run's most downstream wide chunk, where a settled chunk or a haystack edge ends it.
        for (last = chunk, previous = chunk; backward ? previous != 0 : previous + 1 != round->chunks_count;) {
            sz_size_t previous_low, previous_high;
            sz_u32_t count;
            previous = backward ? previous - 1 : previous + 1;
            previous_low = round->segment_begin + previous * width;
            previous_high = sz_min_of_two(previous_low + width, round->segment_end);
            count = round->undecided[previous].count;
            if (!count ||
                (backward ? haystack.first >= previous_low : haystack.first + haystack.length < previous_high))
                break;
            if (count > sz_regex_undecided_simt_k) last = previous;
        }
        for (word = 0; word != words; ++word) state[word] = anchor[word];
        for (walked = next; walked != last;) {
            sz_size_t walked_low, walked_high;
            walked = backward ? walked - 1 : walked + 1;
            walked_low = round->segment_begin + walked * width;
            walked_high = sz_min_of_two(walked_low + width, round->segment_end);
            sz_regex_flow_piece_simt_(engine, round, &haystack, walked_low, walked_high, &begin, &end);
            result = sz_regex_flow_walk_simt_(engine, round, &pike, &table, &haystack, backward ? end : begin,
                                              backward ? begin : end, state, other, sz_false_k, dfa);
            other = result == state ? other : state, state = result;
            for (word = 0; word != words; ++word) round->handoffs[walked * stride + word] = state[word];
            round->undecided[walked].count = 0;
        }
    }
}

/** @ref sz_regex_bound_pieces_simt_ over the pass's NFA or positions. */
static __global__ void sz_regex_bound_pieces_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                          sz_regex_round_simt_t round) {
    sz_regex_bound_pieces_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_bound_pieces_simt_ over a body's DFA. */
static __global__ void sz_regex_bound_pieces_dfa_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                              sz_regex_round_simt_t round) {
    sz_regex_bound_pieces_simt_(&engine, haystacks, &round, sz_true_k);
}

/**
 *  @brief Maps the undecided elements each chunk hands on from those its upstream neighbour hands
 *      it, one thread per chunk in each thread's share of the walks.
 *
 *  The map is affine while the neighbour's stay undecided, else the answer, the neighbour past the
 *  segment's edge handing on the set the segment there kept.
 */
STRINGZILLA_DEVICE void sz_regex_affine_simt_(sz_regex_engine_t const *engine, sz_strings_view_t haystacks,
                                              sz_regex_round_simt_t const *round, sz_bool_t dfa) {
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const width = (sz_size_t)1 << round->chunk_shift, stride = round->set_words;
    sz_bool_t const backward = (sz_bool_t)(round->flow != sz_regex_flow_behind_simt_k);
    sz_size_t const words = sz_regex_flow_words_simt_(engine, round);
    sz_regex_pike_simt_t pike;
    sz_regex_table_simt_t table;
    sz_u64_t *sets;
    sz_size_t chunk;
    // Every thread stages the body's rows before any leaves.
    if (dfa)
        table = sz_regex_stage_simt_(engine, round, engine->bodies[round->body].rows,
                                     round->body_states * (engine->classes_count + 1));
    if (thread >= round->walks_count) return;
    pike = sz_regex_pike_carve_simt_(round, thread, engine), pike.slots_count = 0;
    sets = (sz_u64_t *)(round->walks + thread * round->walk_bytes + round->walk_sets);
    for (chunk = thread; chunk < round->chunks_count; chunk += round->walks_count) {
        sz_size_t const low = round->segment_begin + chunk * width;
        sz_size_t const high = sz_min_of_two(low + width, round->segment_end);
        sz_regex_undecided_simt_t const *const undecided = round->undecided + chunk;
        sz_regex_affine_simt_t *const affine = round->affines + chunk;
        sz_bool_t const edge = (sz_bool_t)(backward ? chunk + 1 == round->chunks_count : chunk == 0);
        sz_size_t const upstream = backward ? chunk + 1 : chunk - 1;
        sz_u64_t const *known = STRINGZILLA_NULL;
        sz_regex_haystack_simt_t haystack;
        sz_size_t begin, end, from, to, word;
        sz_u32_t row, index;
        sz_u64_t *result;
        if (!undecided->count) continue;
        sz_regex_flow_handing_simt_(haystacks, round, low, high, &haystack);
        sz_regex_flow_piece_simt_(engine, round, &haystack, low, high, &begin, &end);
        from = backward ? end : begin, to = backward ? begin : end;
        if (edge) known = round->handoff_edges + (backward ? round->segment + 1 : round->segment) * stride;
        else if (!round->undecided[upstream].count) known = round->handoffs + upstream * stride;
        for (word = 0; word != words; ++word)
            sets[word] = known ? known[word] : round->handoffs[upstream * stride + word];
        result = sz_regex_flow_walk_simt_(engine, round, &pike, &table, &haystack, from, to, sets, sets + stride,
                                          sz_false_k, dfa);
        affine->fixed = sz_regex_decided_simt_(undecided, result);
        affine->target = known ? sz_regex_none_simt_k : (sz_u32_t)upstream;
        affine->rows_count = known ? 0 : round->undecided[upstream].count;
        for (row = 0; row != affine->rows_count; ++row) {
            index = round->undecided[upstream].indices[row];
            sz_regex_flow_least_simt_(engine, round, sets);
            sets[index / 64] |= (sz_u64_t)1 << (index % 64);
            result = sz_regex_flow_walk_simt_(engine, round, &pike, &table, &haystack, from, to, sets, sets + stride,
                                              sz_false_k, dfa);
            affine->rows[row] = sz_regex_decided_simt_(undecided, result);
        }
    }
}

/** @ref sz_regex_affine_simt_ over the pass's NFA or positions. */
static __global__ void sz_regex_affine_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                    sz_regex_round_simt_t round) {
    sz_regex_affine_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_affine_simt_ over a body's DFA. */
static __global__ void sz_regex_affine_dfa_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                        sz_regex_round_simt_t round) {
    sz_regex_affine_simt_(&engine, haystacks, &round, sz_true_k);
}

/**
 *  @brief One round of pointer doubling over the chunks' affine maps: each composes its own with
 *      the maps a round's hops upstream until one is answered.
 *
 *  The last round adds the answer to the chunk's handoff and keeps the handoff of the segment's
 *  last chunk downstream for the segment beyond.
 */
static __global__ void sz_regex_affine_double_simt_kernel_(sz_regex_round_simt_t round) {
    sz_bool_t const even = (sz_bool_t)(round.round % 2 == 0), last = (sz_bool_t)(round.round + 1 == round.rounds);
    sz_bool_t const backward = (sz_bool_t)(round.flow != sz_regex_flow_behind_simt_k);
    sz_regex_affine_simt_t const *source = even ? round.affines : round.affines_next;
    sz_regex_affine_simt_t *target = even ? round.affines_next : round.affines;
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_regex_undecided_simt_t const *const undecided = round.undecided + chunk;
        sz_u64_t *const handoff = round.handoffs + chunk * round.set_words;
        sz_regex_affine_simt_t affine;
        sz_u64_t rows[sz_regex_undecided_simt_k];
        sz_u32_t hop, row, index;
        sz_size_t word;
        if (undecided->count) {
            affine = source[chunk];
            for (hop = 1; hop != sz_regex_round_hops_simt_k && affine.target != sz_regex_none_simt_k; ++hop) {
                sz_regex_affine_simt_t const *const next = source + affine.target;
                affine.fixed |= sz_regex_apply_simt_(affine.rows, next->fixed);
                for (row = 0; next->target != sz_regex_none_simt_k && row != next->rows_count; ++row)
                    rows[row] = sz_regex_apply_simt_(affine.rows, next->rows[row]);
                for (row = 0; next->target != sz_regex_none_simt_k && row != next->rows_count; ++row)
                    affine.rows[row] = rows[row];
                affine.target = next->target, affine.rows_count = next->rows_count;
            }
            target[chunk] = affine;
            if (last)
                for (index = 0; index != undecided->count; ++index)
                    if (affine.fixed >> index & 1)
                        handoff[undecided->indices[index] / 64] |= (sz_u64_t)1 << (undecided->indices[index] % 64);
        }
        if (last && chunk == (backward ? 0 : round.chunks_count - 1))
            for (word = 0; word != round.set_words; ++word)
                round.handoff_edges[(backward ? round.segment : round.segment + 1) * round.set_words + word] =
                    handoff[word];
    }
}

/** Walks each chunk's pieces from the sets handed to them, one thread per chunk in each thread's
 *  share of the walks, recording where the lookarounds of the round's pass hold. */
STRINGZILLA_DEVICE void sz_regex_looks_simt_(sz_regex_engine_t const *engine, sz_strings_view_t haystacks,
                                             sz_regex_round_simt_t const *round, sz_bool_t dfa) {
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const width = (sz_size_t)1 << round->chunk_shift, stride = round->set_words;
    sz_bool_t const backward = (sz_bool_t)(round->flow != sz_regex_flow_behind_simt_k);
    sz_size_t const words = sz_regex_flow_words_simt_(engine, round);
    sz_regex_pike_simt_t pike;
    sz_regex_table_simt_t table;
    sz_u64_t *sets;
    sz_size_t chunk;
    // Every thread stages the body's rows before any leaves.
    if (dfa)
        table = sz_regex_stage_simt_(engine, round, engine->bodies[round->body].rows,
                                     round->body_states * (engine->classes_count + 1));
    if (thread >= round->walks_count) return;
    pike = sz_regex_pike_carve_simt_(round, thread, engine), pike.slots_count = 0;
    sets = (sz_u64_t *)(round->walks + thread * round->walk_bytes + round->walk_sets);
    for (chunk = thread; chunk < round->chunks_count; chunk += round->walks_count) {
        sz_size_t const low = round->segment_begin + chunk * width;
        sz_size_t const high = sz_min_of_two(low + width, round->segment_end);
        sz_size_t index = sz_regex_haystack_at_simt_(haystacks, low, 0), begin, end, word;
        for (; index < haystacks.count; ++index) {
            sz_regex_haystack_simt_t const haystack = sz_regex_haystack_simt_(haystacks, index);
            sz_u64_t const *handed = STRINGZILLA_NULL;
            if (haystack.first >= high) break;
            sz_regex_flow_piece_simt_(engine, round, &haystack, low, high, &begin, &end);
            if (backward && end <= haystack.length)
                handed = chunk + 1 == round->chunks_count ? round->handoff_edges + (round->segment + 1) * stride
                                                          : round->handoffs + (chunk + 1) * stride;
            else if (!backward && begin)
                handed = chunk == 0 ? round->handoff_edges + round->segment * stride
                                    : round->handoffs + (chunk - 1) * stride;
            if (handed)
                for (word = 0; word != words; ++word) sets[word] = handed[word];
            else sz_regex_flow_start_simt_(engine, round, sets);
            sz_regex_flow_walk_simt_(engine, round, &pike, &table, &haystack, backward ? end : begin,
                                     backward ? begin : end, sets, sets + stride, sz_true_k, dfa);
        }
    }
}

/** @ref sz_regex_looks_simt_ over the pass's NFA or positions. */
static __global__ void sz_regex_looks_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                   sz_regex_round_simt_t round) {
    sz_regex_looks_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_looks_simt_ over a body's DFA. */
static __global__ void sz_regex_looks_dfa_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                       sz_regex_round_simt_t round) {
    sz_regex_looks_simt_(&engine, haystacks, &round, sz_true_k);
}

/**
 *  @brief Writes the one-byte lookarounds holding at every slot of the round's segment into its
 *      half of the table, negated ones flipped, as @c sz_regex_fill_looks_serial_ does, which the
 *      passes then add their own bits to.
 *
 *  Each block takes a run of slots a tile at a time, its first thread galloping to the haystack of
 *  the tile's first slot from the last tile's, and the threads past that haystack on from it.
 */
static __global__ void sz_regex_singles_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                     sz_regex_round_simt_t round) {
    __shared__ sz_u32_t tables[512];
    __shared__ sz_size_t tile_index;
    __shared__ sz_regex_haystack_simt_t tile_haystack;
    sz_size_t const share = sz_size_divide_round_up(
                                sz_size_divide_round_up(round.segment_end - round.segment_begin, gridDim.x),
                                blockDim.x) *
                            blockDim.x;
    sz_size_t const low = round.segment_begin + blockIdx.x * share;
    sz_size_t const high = sz_min_of_two(low + share, round.segment_end);
    sz_u32_t flipped = 0, index;
    sz_size_t tile;
    for (index = threadIdx.x; index < 512; index += blockDim.x) tables[index] = engine.looks_bytes[1 + index];
    for (index = 0; index != engine.lookarounds_count; ++index)
        if (engine.lookarounds[index].kind & 1) flipped |= (sz_u32_t)sz_regex_look_lookaround_k << index;
    flipped &= engine.looks_bytes[0];
    if (threadIdx.x == 0) tile_index = 0;
    for (tile = low; tile < high; tile += blockDim.x) {
        sz_size_t const slot = tile + threadIdx.x;
        sz_regex_haystack_simt_t haystack;
        sz_size_t position;
        __syncthreads();
        if (threadIdx.x == 0)
            tile_index = sz_regex_haystack_at_simt_(haystacks, tile, tile_index),
            tile_haystack = sz_regex_haystack_simt_(haystacks, tile_index);
        __syncthreads();
        if (slot >= high) continue;
        haystack = tile_haystack;
        if (slot > haystack.first + haystack.length)
            haystack = sz_regex_haystack_simt_(haystacks, sz_regex_haystack_at_simt_(haystacks, slot, tile_index + 1));
        position = slot - haystack.first;
        round.looks.table[slot & round.looks.mask] =
            ((position ? tables[haystack.text[position - 1]] : 0) |
             (position != haystack.length ? tables[256 + haystack.text[position]] : 0)) ^
            flipped;
    }
}

#pragma endregion Set Kernels

#pragma region Priming Kernels

/**
 *  @brief Primes every chunk whose last haystack runs past its right edge, one thread per chunk, as
 *      the Metal kernel does.
 *
 *  It leaves the state in @c entries, or, when priming does not collapse within its reach, the
 *  uncollapsed set and where each of its states would leave the chunk, or, when no column narrows
 *  within its reach, a frontier one wider than a priming set holds for the pieces kernel. The first
 *  chunk's entry also lands in the segment's edge, which priming the segment before reads.
 *
 *  A folded walk's chunk edges move to the next unit boundary, where it steps the
 *  folds of whole units.
 */
static __global__ void sz_regex_prime_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                   sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.reverse_rows, round.reverse_cells);
    sz_size_t const width = (sz_size_t)1 << round.chunk_shift, stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_bool_t const folded = (sz_bool_t)(engine.case_sensitivity != sz_cased_k);
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_size_t const low = round.segment_begin + chunk * width;
        sz_size_t const high = sz_min_of_two(low + width, round.segment_end);
        sz_regex_frontier_simt_t *const frontier = round.frontiers + chunk;
        sz_regex_haystack_simt_t haystack;
        sz_u32_t set[sz_regex_priming_states_simt_k], count, state, index, byte, block;
        sz_size_t edge, stop, at;
        frontier->count = 0;
        if (high == round.slots_count) continue;
        haystack = sz_regex_haystack_simt_(haystacks, sz_regex_haystack_at_simt_(haystacks, high - 1, 0));
        if (haystack.first + haystack.length < high) continue;
        edge = folded ? sz_regex_boundary_simt_(haystack.text, haystack.length, high - haystack.first)
                      : high - haystack.first - 1;
        state = folded ? sz_regex_prime_folded_simt_(&engine, &table, round.reverse_images, &haystack, edge,
                                                     sz_regex_priming_bytes_simt_k, set, &count)
                       : sz_regex_prime_simt_(&engine, &table, round.reverse_images, &haystack, edge,
                                              sz_regex_priming_bytes_simt_k, set, &count);
        round.entries[chunk] = state;
        if (chunk == 0) round.edges[round.segment] = state;
        if (state) continue;
        frontier->count = count ? count : sz_regex_priming_states_simt_k + 1;
        if (!count) continue;

        // Each state walks the chunk alone, as the chunk left of it depends on where each leaves.
        for (index = 0; index != count; ++index) frontier->states[index] = set[index];
        if (haystack.first < low && folded)
            for (at = edge, stop = sz_regex_boundary_simt_(haystack.text, haystack.length, low - haystack.first);
                 at > stop;) {
                sz_utf8_folded_image_t image;
                block = sz_regex_block_at_simt_(&table, haystack.first + at);
                at = sz_regex_unit_before_simt_(sz_true_k, haystack.text, at, &image);
                for (byte = image.length; byte--; block = 0) {
                    sz_u32_t const column = block + table.classes[sz_regex_image_byte_simt_(&image, byte)];
                    for (index = 0; index != count; ++index)
                        if (set[index]) set[index] = sz_regex_move_simt_(&table, set[index], column);
                }
            }
        else if (haystack.first < low)
            for (at = edge; at != low - haystack.first - 1; --at) {
                sz_u32_t const column = sz_regex_block_at_simt_(&table, haystack.first + at) +
                                        table.classes[haystack.text[at - 1]];
#pragma unroll
                for (index = 0; index != sz_regex_priming_states_simt_k; ++index)
                    if (index < count && set[index]) set[index] = sz_regex_move_simt_(&table, set[index], column);
            }
        for (index = 0; index != count; ++index) frontier->exits[index] = set[index];
    }
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

/**
 *  @brief Settles every wide chunk, one thread per run of unsettled chunks holding one, as
 *      @c sz_regex_prime_pieces_metal_kernel_ does.
 *
 *  The run's rightmost wide chunk walks its haystack right to left from the nearest exact state
 *  right of it: a settled chunk's entry, the reverse start where the haystack ends, or the next
 *  segment's edge, so no walk leaves its segment. It writes the entry of every chunk it passes down
 *  to the run's leftmost wide one, and the frontiers between them compose onto those entries as
 *  onto any settled chunk's.
 */
static __global__ void sz_regex_prime_pieces_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                          sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.reverse_rows, round.reverse_cells);
    sz_size_t const begin = round.segment_begin, width = (sz_size_t)1 << round.chunk_shift;
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_bool_t const folded = (sz_bool_t)(engine.case_sensitivity != sz_cased_k);
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_size_t high, last, next, at = 0, leftmost, previous, walked;
        sz_regex_haystack_simt_t haystack;
        sz_u32_t state = 0;
        sz_bool_t owner = sz_true_k;
        if (round.frontiers[chunk].count <= sz_regex_priming_states_simt_k) continue;
        high = sz_min_of_two(begin + (chunk + 1) * width, round.segment_end);
        haystack = sz_regex_haystack_simt_(haystacks, sz_regex_haystack_at_simt_(haystacks, high - 1, 0));
        last = haystack.first + haystack.length;

        // The anchor right of the chunk, past frontiers only, as a wide chunk past it owns the run.
        for (next = chunk + 1;; ++next) {
            sz_size_t const next_high = sz_min_of_two(begin + (next + 1) * width, round.slots_count);
            sz_u32_t const count = next == round.chunks_count ? 0 : round.frontiers[next].count;
            if (last < next_high) {
                state = engine.reverse_starts[0], at = haystack.length;
                break;
            }
            if (count > sz_regex_priming_states_simt_k) {
                owner = sz_false_k;
                break;
            }
            if (count) continue;
            state = next == round.chunks_count ? round.edges[round.segment + 1] : round.entries[next];
            at = sz_regex_prime_edge_simt_(folded, &haystack, next_high);
            break;
        }
        if (!owner) continue;

        // The run's leftmost wide chunk, where a settled chunk or the haystack's start ends it.
        for (leftmost = chunk, previous = chunk; previous-- != 0;) {
            sz_u32_t const count = round.frontiers[previous].count;
            if (!count || begin + (previous + 1) * width <= haystack.first) break;
            if (count > sz_regex_priming_states_simt_k) leftmost = previous;
        }
        for (walked = next; walked-- != leftmost;) {
            sz_size_t const edge = sz_regex_prime_edge_simt_(folded, &haystack, begin + (walked + 1) * width);
            state = folded ? sz_regex_walk_folded_back_simt_(&table, state, &haystack, at, edge)
                           : sz_regex_walk_back_simt_(&table, state, &haystack, at, edge);
            round.entries[walked] = state, at = edge;
            if (walked == 0) round.edges[round.segment] = state;
        }
    }
}

/** Points every unsettled chunk at what settles it, as @c sz_regex_compose_metal_kernel_ does; the
 *  last chunk's right neighbour is the next segment's first, whose entry its edge holds. */
static __global__ void sz_regex_compose_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                     sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.reverse_rows, round.reverse_cells);
    sz_size_t const width = (sz_size_t)1 << round.chunk_shift, stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_bool_t const folded = (sz_bool_t)(engine.case_sensitivity != sz_cased_k);
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_u32_t const count = round.frontiers[chunk].count;
        sz_u32_t states[sz_regex_priming_states_simt_k], index;
        sz_regex_composition_simt_t composition;
        sz_size_t next, low, high;
        sz_regex_haystack_simt_t haystack;
        sz_bool_t ends;
        if (!sz_regex_composing_simt_(count)) continue;
        for (index = 0; index != count; ++index) states[index] = round.frontiers[chunk].states[index];
        composition.target = sz_regex_none_simt_k, composition.known = sz_regex_none_simt_k;
        next = chunk + 1, low = round.segment_begin + next * width;
        high = sz_min_of_two(low + width, round.slots_count);
        haystack = sz_regex_haystack_simt_(haystacks, sz_regex_haystack_at_simt_(haystacks, low, 0));
        // The haystack either ends inside the next chunk or spans it, which then is settled or not.
        ends = (sz_bool_t)(haystack.first + haystack.length < high);
        if (!ends && next != round.chunks_count && sz_regex_composing_simt_(round.frontiers[next].count)) {
            composition.target = (sz_u32_t)next;
            for (index = 0; index != round.frontiers[next].count; ++index)
                composition.indices[index] = sz_regex_index_of_simt_(states, count, round.frontiers[next].exits[index]);
        }
        else {
            sz_u32_t const entry = ends                         ? engine.reverse_starts[0]
                                   : next == round.chunks_count ? round.edges[round.segment + 1]
                                                                : round.entries[next];
            sz_size_t const at = ends ? haystack.length : sz_regex_prime_edge_simt_(folded, &haystack, high);
            sz_size_t const edge = sz_regex_prime_edge_simt_(folded, &haystack, low);
            sz_u32_t const state = folded ? sz_regex_walk_folded_back_simt_(&table, entry, &haystack, at, edge)
                                          : sz_regex_walk_back_simt_(&table, entry, &haystack, at, edge);
            composition.known = sz_regex_index_of_simt_(states, count, state);
        }
        round.compositions[chunk] = composition;
    }
}

/** One round of pointer doubling over the unsettled chunks; the last round writes the entry. */
static __global__ void sz_regex_compose_double_simt_kernel_(sz_regex_round_simt_t round) {
    sz_bool_t const even = (sz_bool_t)(round.round % 2 == 0);
    sz_regex_composition_simt_t const *source = even ? round.compositions : round.compositions_next;
    sz_regex_composition_simt_t *target = even ? round.compositions_next : round.compositions;
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_regex_composition_simt_t composition;
        sz_u32_t hop, index;
        if (!sz_regex_composing_simt_(round.frontiers[chunk].count)) continue;
        composition = source[chunk];
        for (hop = 1; hop != sz_regex_round_hops_simt_k && composition.target != sz_regex_none_simt_k; ++hop) {
            sz_regex_composition_simt_t const next = source[composition.target];
            sz_u8_t indices[sz_regex_priming_states_simt_k];
            sz_u32_t const next_count = round.frontiers[next.target == sz_regex_none_simt_k ? 0 : next.target].count;
            if (next.known != sz_regex_none_simt_k) {
                composition.known = composition.indices[next.known];
                composition.target = sz_regex_none_simt_k;
                break;
            }
            for (index = 0; index != next_count; ++index)
                indices[index] = next.indices[index] == 0xFF ? 0xFF : composition.indices[next.indices[index]];
            for (index = 0; index != next_count; ++index) composition.indices[index] = indices[index];
            composition.target = next.target;
        }
        target[chunk] = composition;
        if (round.round + 1 != round.rounds) continue;
        round.entries[chunk] = round.frontiers[chunk].states[composition.known];
        if (chunk == 0) round.edges[round.segment] = round.entries[chunk];
    }
}

#pragma endregion Priming Kernels

#pragma region Walk Kernels

/** Marks every slot of one chunk where a match starts, one thread per chunk, as the Metal kernel
 *  does, each block first staging its tile's text, which its threads read a chunk apart. */
static __global__ void sz_regex_mark_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                  sz_regex_round_simt_t round) {
    extern __shared__ sz_u8_t sz_regex_staged_simt_[];
    __shared__ sz_size_t tile_haystack;
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.reverse_rows, round.reverse_cells);
    sz_u8_t *const staged = sz_regex_staged_simt_ + sz_regex_staged_bytes_simt_k +
                            round.staged_cells * sizeof(sz_u32_t);
    sz_u8_t const *const tape = (sz_u8_t const *)sz_strings_start_simt_(haystacks, 0);
    sz_u8_t const *const tape_end = (sz_u8_t const *)sz_strings_start_simt_(haystacks, haystacks.count);
    sz_size_t const width = (sz_size_t)1 << round.chunk_shift, tile_slots = width * blockDim.x;
    sz_u32_t const accepts = engine.reverse_accepts_begin;
    sz_size_t tile;
    // The scan reads one entry past the chunks, which ends up holding the candidates' total.
    if (blockIdx.x == 0 && threadIdx.x == 0) round.chunk_offsets[round.chunks_count] = 0;
    for (tile = blockIdx.x; tile * blockDim.x < round.chunks_count; tile += gridDim.x) {
        sz_size_t const chunk = tile * blockDim.x + threadIdx.x, tile_low = round.segment_begin + tile * tile_slots;
        sz_size_t const low = round.segment_begin + chunk * width;
        sz_size_t const high = sz_min_of_two(low + width, round.segment_end);
        sz_size_t index, word, first_byte;
        sz_u64_t *const words = round.starts + (low - round.segment_begin) / 64;
        sz_u8_t const *origin;
        sz_u32_t marked = 0;
        // The last tile's walks are done with its text and its haystack before either is replaced.
        __syncthreads();
        if (threadIdx.x == 0) tile_haystack = sz_regex_haystack_at_simt_(haystacks, tile_low, 0);
        __syncthreads();
        // A tile's bytes run from the byte before its first slot, as each slot past the first of a
        // haystack stands for the byte before it, and span at most its slots.
        first_byte = tile_low > tile_haystack ? tile_low - tile_haystack - 1 : 0;
        origin = sz_regex_stage_text_simt_(
            staged, tape + first_byte, tape + sz_min_of_two(first_byte + tile_slots + 1, (sz_size_t)(tape_end - tape)),
            tape, tape_end);
        __syncthreads();
        if (chunk >= round.chunks_count) continue;
        index = sz_regex_haystack_at_simt_(haystacks, low, tile_haystack);
        round.chunk_haystacks[chunk] = index;
        for (word = 0; word != width / 64; ++word) words[word] = 0;
        for (; index < haystacks.count; ++index) {
            sz_regex_haystack_simt_t const haystack = sz_regex_haystack_simt_(haystacks, index);
            sz_size_t begin_byte, end_byte, at;
            sz_u32_t state, bit;
            if (haystack.first >= high) break;
            begin_byte = low > haystack.first ? low - haystack.first - 1 : 0;
            end_byte = sz_min_of_two(haystack.length, high - haystack.first - 1);
            state = end_byte == haystack.length ? engine.reverse_starts[0] : round.entries[chunk];
            for (at = end_byte; at != begin_byte;) {
                sz_u32_t const count = (sz_u32_t)sz_min_of_two(at - begin_byte, (sz_size_t)sz_regex_run_bytes_simt_k);
                sz_u32_t const mask = sz_regex_run_back_staged_simt_(&table, &state, staged,
                                                                     (sz_size_t)(haystack.text + at - origin),
                                                                     haystack.first + at, count, accepts);
                at -= count;
                if (!mask) continue;
                // A byte's own slot marks the match starting right after it.
                bit = (sz_u32_t)(haystack.first + 1 + at - low);
                words[bit / 64] |= (sz_u64_t)mask << (bit % 64);
                if (bit % 64 + count > 64) words[bit / 64 + 1] |= (sz_u64_t)mask >> (64 - bit % 64);
                marked += (sz_u32_t)__popc(mask);
            }
            if (haystack.first < low || sz_regex_step_edge_simt_(&table, state, haystack.first) < accepts) continue;
            bit = (sz_u32_t)(haystack.first - low);
            words[bit / 64] |= (sz_u64_t)1 << (bit % 64), ++marked;
        }
        round.chunk_offsets[chunk] = marked;
    }
}

/** Slots each thread of the prefixes mark kernel tests per tile, so a tile's haystack searches and
 *  barriers spread over many slots. */
enum { sz_regex_prefix_slots_simt_k = 8 };

/** Chunks of one tile of the prefixes mark kernel. */
enum {
    sz_regex_prefix_chunks_simt_k = (int)sz_regex_threads_per_block_simt_k * (int)sz_regex_prefix_slots_simt_k /
                                    (int)sz_regex_chunk_slots_simt_k,
};

/**
 *  @brief Marks every slot whose bytes keep some bucket of the engine's prefixes at every offset
 *      inside its haystack, a slot per thread at a time, as @c sz_regex_find_prefixes_serial_ finds
 *      them, in place of the start-finding walk and its priming; the end walks settle each one.
 *
 *  Each block takes a run of whole chunks a tile at a time, where a thread per chunk gallops to the
 *  haystack of its first slot from the last tile's last, and the slots past that haystack on from
 *  it. A haystack's last slot, past its bytes, is never marked, as no match is empty.
 */
static __global__ void sz_regex_mark_prefixes_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                           sz_regex_round_simt_t round) {
    __shared__ sz_u8_t buckets[sz_regex_prefixes_depth_k][256];
    __shared__ sz_u64_t words[sz_regex_prefix_chunks_simt_k];
    __shared__ sz_size_t indices[sz_regex_prefix_chunks_simt_k];
    __shared__ sz_regex_haystack_simt_t heads[sz_regex_prefix_chunks_simt_k];
    sz_size_t const tile_slots = (sz_size_t)sz_regex_prefix_chunks_simt_k * sz_regex_chunk_slots_simt_k;
    sz_size_t const chunks = sz_size_divide_round_up(round.chunks_count, gridDim.x);
    sz_size_t const share = sz_size_divide_round_up(chunks * sz_regex_chunk_slots_simt_k, tile_slots) * tile_slots;
    sz_size_t const low = round.segment_begin + blockIdx.x * share;
    sz_size_t const high = sz_min_of_two(low + share, round.segment_end);
    sz_u32_t const depth = engine.prefixes->depth;
    sz_size_t tile, index, last = 0;
    for (index = threadIdx.x; index < sz_regex_prefixes_depth_k * 256; index += blockDim.x)
        buckets[index / 256][index % 256] = engine.prefixes->buckets[index / 256][index % 256];
    if (blockIdx.x == 0 && threadIdx.x == 0) round.chunk_offsets[round.chunks_count] = 0;
    for (tile = low; tile < high; tile += tile_slots) {
        __syncthreads();
        if (threadIdx.x < sz_regex_prefix_chunks_simt_k) {
            sz_size_t const first = tile + threadIdx.x * sz_regex_chunk_slots_simt_k;
            words[threadIdx.x] = 0;
            if (first < high) {
                indices[threadIdx.x] = sz_regex_haystack_at_simt_(haystacks, first, last);
                heads[threadIdx.x] = sz_regex_haystack_simt_(haystacks, indices[threadIdx.x]);
                round.chunk_haystacks[(first - round.segment_begin) / sz_regex_chunk_slots_simt_k] =
                    indices[threadIdx.x];
            }
        }
        __syncthreads();
        last = indices[sz_min_of_two((high - tile - 1) / sz_regex_chunk_slots_simt_k,
                                     (sz_size_t)sz_regex_prefix_chunks_simt_k - 1)];
        // Each thread tests a run of consecutive slots, so the haystack it holds serves them all.
        {
            sz_size_t const run = tile + threadIdx.x * sz_regex_prefix_slots_simt_k;
            sz_size_t const chunk = threadIdx.x * sz_regex_prefix_slots_simt_k / sz_regex_chunk_slots_simt_k;
            sz_regex_haystack_simt_t haystack = heads[chunk];
            sz_size_t haystack_index = indices[chunk], slot, position, offset;
            sz_u64_t marks = 0;
            for (slot = run; slot != run + sz_regex_prefix_slots_simt_k && slot < high; ++slot) {
                sz_u8_t held = 0;
                if (slot > haystack.first + haystack.length)
                    haystack_index = sz_regex_haystack_at_simt_(haystacks, slot, haystack_index + 1),
                    haystack = sz_regex_haystack_simt_(haystacks, haystack_index);
                position = slot - haystack.first;
                if (position != haystack.length) held = buckets[0][haystack.text[position]];
                for (offset = 1; held && offset != depth && position + offset != haystack.length; ++offset)
                    held &= buckets[offset][haystack.text[position + offset]];
                marks |= (sz_u64_t)(held != 0) << (slot - run);
            }
            if (marks)
                atomicOr((unsigned long long *)words + chunk,
                         (unsigned long long)(marks << (run - tile) % sz_regex_chunk_slots_simt_k));
        }
        __syncthreads();
        for (index = threadIdx.x; index < sz_regex_prefix_chunks_simt_k && tile + index * 64 < high;
             index += blockDim.x) {
            sz_size_t const chunk = (tile - round.segment_begin) / sz_regex_chunk_slots_simt_k + index;
            round.starts[chunk] = words[index];
            round.chunk_offsets[chunk] = (sz_size_t)__popcll(words[index]);
        }
    }
}

/**
 *  @brief Marks every slot of one chunk where a match of a folded walk starts, one thread per
 *      chunk, as the Metal kernel does.
 *
 *  A start inside the folds of one codepoint snaps outward to the codepoint's own start, as
 *  @c sz_regex_mark_folded_ snaps it, so the walk begins past the unit holding the chunk's last
 *  slot, where priming left its entry.
 */
static __global__ void sz_regex_mark_folded_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                         sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.reverse_rows, round.reverse_cells);
    sz_size_t const width = (sz_size_t)1 << round.chunk_shift, stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_u32_t const accepts = engine.reverse_accepts_begin;
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk == 0) round.chunk_offsets[round.chunks_count] = 0;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_size_t const low = round.segment_begin + chunk * width;
        sz_size_t const high = sz_min_of_two(low + width, round.segment_end);
        sz_size_t index = sz_regex_haystack_at_simt_(haystacks, low, 0), word;
        sz_u64_t *const words = round.starts + (low - round.segment_begin) / 64;
        sz_u32_t marked = 0;
        round.chunk_haystacks[chunk] = index;
        for (word = 0; word != width / 64; ++word) words[word] = 0;
        for (; index < haystacks.count; ++index) {
            sz_regex_haystack_simt_t const haystack = sz_regex_haystack_simt_(haystacks, index);
            sz_size_t first, last, at, stop;
            sz_u32_t state;
            if (haystack.first >= high) break;
            first = low > haystack.first ? low - haystack.first : 0;
            last = sz_min_of_two(haystack.length, high - haystack.first - 1);
            at = last == haystack.length ? last : sz_regex_boundary_simt_(haystack.text, haystack.length, last + 1);
            state = at == haystack.length ? engine.reverse_starts[0] : round.entries[chunk];
            stop = first ? first - 1 : 0;
            while (at > stop) {
                sz_utf8_folded_image_t image;
                sz_u32_t block = sz_regex_block_at_simt_(&table, haystack.first + at), byte;
                sz_size_t const begin = sz_regex_unit_before_simt_(sz_true_k, haystack.text, at, &image);
                for (byte = image.length; byte--; block = 0) {
                    sz_size_t const start = byte + 1u == image.length ? at : begin;
                    state = sz_regex_move_simt_(&table, state,
                                                block + table.classes[sz_regex_image_byte_simt_(&image, byte)]);
                    if (state >= accepts && start >= first && start <= last)
                        words[(haystack.first + start - low) / 64] |= (sz_u64_t)1
                                                                      << ((haystack.first + start - low) % 64);
                }
                at = begin;
            }
            if (first == 0 && sz_regex_step_edge_simt_(&table, state, haystack.first) >= accepts)
                words[(haystack.first - low) / 64] |= (sz_u64_t)1 << ((haystack.first - low) % 64);
        }
        for (word = 0; word != width / 64; ++word) marked += (sz_u32_t)__popcll(words[word]);
        round.chunk_offsets[chunk] = marked;
    }
}

/**
 *  @brief Marks every unit boundary of one chunk where a match starts, stepping position sets right
 *      to left from the set handed to each piece, as @c sz_regex_mark_positions_serial_ does, or
 *      over folded runes when @p folded, as @c sz_regex_mark_positions_folded_ does.
 *
 *  Each thread keeps two sets in its share of the walks, so a launch takes as many threads as
 *  shares fit and each walks every chunk that many apart.
 */
STRINGZILLA_DEVICE void sz_regex_mark_positions_simt_(sz_regex_engine_t const *engine, sz_strings_view_t haystacks,
                                                      sz_regex_round_simt_t const *round, sz_bool_t folded) {
    sz_size_t const width = (sz_size_t)1 << round->chunk_shift, words = engine->positions_words;
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x, stride = round->set_words;
    sz_u64_t *const sets = (sz_u64_t *)(round->walks + thread * round->walk_bytes);
    sz_size_t chunk;
    if (thread == 0) round->chunk_offsets[round->chunks_count] = 0;
    if (thread >= round->walks_count) return;
    for (chunk = thread; chunk < round->chunks_count; chunk += round->walks_count) {
        sz_size_t const low = round->segment_begin + chunk * width;
        sz_size_t const high = sz_min_of_two(low + width, round->segment_end);
        sz_size_t index = sz_regex_haystack_at_simt_(haystacks, low, 0), word;
        sz_u64_t *const bits = round->starts + (low - round->segment_begin) / 64;
        sz_u32_t marked = 0;
        round->chunk_haystacks[chunk] = index;
        for (word = 0; word != width / 64; ++word) bits[word] = 0;
        for (; index < haystacks.count; ++index) {
            sz_regex_haystack_simt_t const haystack = sz_regex_haystack_simt_(haystacks, index);
            sz_u8_t const *const text = haystack.text;
            sz_size_t const length = haystack.length;
            sz_u64_t const *handed = STRINGZILLA_NULL;
            sz_size_t begin, end;
            sz_u64_t *state, *other;
            sz_utf8_folded_image_t image;
            if (haystack.first >= high) break;
            begin = low > haystack.first ? low - haystack.first : 0;
            end = sz_min_of_two(high - haystack.first, length + 1);
            // A piece running past the chunk takes the set the chunk right of it hands on,
            // past its unit boundary.
            if (end <= length)
                handed = chunk + 1 == round->chunks_count ? round->handoff_edges + (round->segment + 1) * stride
                                                          : round->handoffs + (chunk + 1) * stride;
            for (word = 0; word != words; ++word) sets[word] = handed ? handed[word] : word == 0;
            state = sz_regex_positions_walk_simt_(engine, round, &haystack, folded,
                                                  end > length ? length : sz_regex_boundary_simt_(text, length, end),
                                                  begin ? begin - 1 : 0, sets, sets + words, bits, low, high);
            if (begin) continue;
            // The start of the haystack takes one more step, over the edge class.
            other = state == sets ? sets + words : sets;
            if (length) sz_regex_unit_after_simt_(folded, text, length, 0, &image);
            sz_regex_positions_step_simt_(engine, state, other, 1,
                                          length ? sz_regex_byte_kind_(sz_regex_image_byte_simt_(&image, 0)) : 0,
                                          sz_regex_looks_at_simt_(&round->looks, haystack.first));
            if (other[0] >> 1 & 1) bits[(haystack.first - low) / 64] |= (sz_u64_t)1 << ((haystack.first - low) % 64);
        }
        for (word = 0; word != width / 64; ++word) marked += (sz_u32_t)__popcll(bits[word]);
        round->chunk_offsets[chunk] = marked;
    }
}

/** @ref sz_regex_mark_positions_simt_ over bytes. */
static __global__ void sz_regex_mark_positions_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                            sz_regex_round_simt_t round) {
    sz_regex_mark_positions_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_mark_positions_simt_ over folded runes. */
static __global__ void sz_regex_mark_positions_folded_simt_kernel_(sz_regex_engine_t engine,
                                                                   sz_strings_view_t haystacks,
                                                                   sz_regex_round_simt_t round) {
    sz_regex_mark_positions_simt_(&engine, haystacks, &round, sz_true_k);
}

/** Lists the candidates of one chunk in slot order, one thread per chunk, for the end walks taking
 *  one candidate per thread. */
static __global__ void sz_regex_list_simt_kernel_(sz_regex_round_simt_t round) {
    sz_size_t const width = (sz_size_t)1 << round.chunk_shift, stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_size_t const offset = chunk * width;
        sz_u32_t index = (sz_u32_t)round.chunk_offsets[chunk];
        sz_size_t word;
        for (word = 0; word != width / 64; ++word) {
            sz_u64_t bits = round.starts[offset / 64 + word];
            for (; bits; bits &= bits - 1)
                round.slots[index++] = (sz_u32_t)(offset + word * 64 + sz_regex_lowest_simt_(bits));
        }
    }
}

/** Walks every candidate to its end with a PikeVM in each thread's share of the
 *  walks, linking nothing. */
static __global__ void sz_regex_ends_pike_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                       sz_regex_round_simt_t round) {
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const count = round.chunk_offsets[round.chunks_count];
    sz_regex_pike_simt_t pike;
    sz_size_t item;
    if (thread >= round.walks_count) return;
    pike = sz_regex_pike_carve_simt_(&round, thread, &engine);
    pike.slots_count = 0;
    for (item = thread; item < count; item += round.walks_count) {
        sz_size_t const mark = round.segment_begin + round.slots[item];
        sz_regex_haystack_simt_t const haystack = sz_regex_haystack_simt_(
            haystacks,
            sz_regex_haystack_at_simt_(haystacks, mark, round.chunk_haystacks[round.slots[item] >> round.chunk_shift]));
        sz_regex_result_simt_t result;
        sz_size_t end, offset = 0;
        sz_u32_t pattern;
        result.link = sz_regex_none_simt_k, result.pattern = sz_regex_none_simt_k, result.end = 0;
        if (sz_regex_match_at_simt_(&engine, STRINGZILLA_NULL, &pike, &haystack, mark - haystack.first, &end, &pattern,
                                    &offset))
            result.pattern = pattern, result.end = haystack.first + end;
        round.results[item] = result;
    }
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

/**
 *  @brief Walks every candidate of one chunk to its end, right to left, one thread per chunk, which
 *      lists the candidates in slot order as it goes, as @c sz_regex_ends_metal_kernel_ does.
 *
 *  A walk leaves its state at the checkpoint of every 16-slot mark it passes, the first unit
 *  boundary at or past the mark, and a later walk, starting further left, that holds the same state
 *  at the same checkpoint stops there and takes that walk's outcome. A walk reaching the chunk's
 *  edge leaves a link there instead: the state it would walk on in, which the links kernel follows.
 *  A folded walk starts at the first rune its unit folds to, and once the links settle
 *  @ref sz_regex_retry_folded_simt_kernel_ tries the others.
 */
static __global__ void sz_regex_ends_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                  sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.forward_rows, round.forward_cells);
    sz_size_t const width = (sz_size_t)1 << round.chunk_shift, stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t const begin = round.segment_begin;
    sz_bool_t const folded = (sz_bool_t)(engine.case_sensitivity != sz_cased_k);
    sz_u32_t const accepts = engine.forward_accepts_begin;
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_u32_t index = (sz_u32_t)round.chunk_offsets[chunk + 1];
        sz_size_t low, high, checkpoint, haystack_index, word;
        sz_regex_haystack_simt_t haystack;
        if (index == round.chunk_offsets[chunk]) continue;
        low = begin + chunk * width, high = sz_min_of_two(low + width, round.segment_end);
        // Only a chunk with candidates has its checkpoints read, so only such a chunk clears them.
        for (checkpoint = (low - begin) / sz_regex_checkpoint_slots_simt_k + 1;
             checkpoint <= (high - begin) / sz_regex_checkpoint_slots_simt_k; ++checkpoint)
            round.checkpoints[checkpoint].state = 0;
        haystack_index = sz_regex_haystack_at_simt_(haystacks, high - 1, round.chunk_haystacks[chunk]);
        haystack = sz_regex_haystack_simt_(haystacks, haystack_index);

        for (word = (high - low + 63) / 64; word-- != 0;) {
            sz_u64_t bits = round.starts[(low - begin) / 64 + word];
            while (bits) {
                sz_u32_t const bit = 63 - (sz_u32_t)__clzll((long long)bits);
                sz_size_t const mark = low + word * 64 + bit;
                sz_size_t start, at, stepped = 1, kind = 0, last_end = 0, last_at = 0, inherited_end = 0;
                sz_size_t recorded_low = 0, recorded_high = 0;
                sz_u32_t state, last = 0, link = sz_regex_none_simt_k, inherited = sz_regex_none_simt_k;
                sz_regex_result_simt_t result;
                bits ^= (sz_u64_t)1 << bit;
                while (mark < haystack.first) haystack = sz_regex_haystack_simt_(haystacks, --haystack_index);
                round.slots[--index] = (sz_u32_t)(mark - begin);

                start = mark - haystack.first;
                if (start && folded) {
                    sz_utf8_folded_image_t previous;
                    stepped = start - sz_regex_unit_before_simt_(sz_true_k, haystack.text, start, &previous);
                    kind = sz_regex_byte_kind_(sz_regex_image_byte_simt_(&previous, previous.length - 1u));
                }
                else if (start) kind = sz_regex_byte_kind_simt_(haystack.text[start - 1]);
                state = engine.forward_starts[kind];
                for (at = start; state != 0;) {
                    sz_size_t due;
                    if (at == haystack.length) {
                        state = sz_regex_step_edge_simt_(&table, state, haystack.first + at);
                        if (state >= accepts) last = state, last_end = last_at = at;
                        break;
                    }
                    // The mark at the chunk's own start is the edge of the chunk before,
                    // which owns its checkpoint.
                    due = sz_regex_mark_at_simt_(haystack.first + 1 + at, stepped);
                    if (due > low) {
                        sz_size_t const visited = (due - begin) / sz_regex_checkpoint_slots_simt_k;
                        sz_regex_checkpoint_simt_t *const entry = round.checkpoints + visited;
                        if (entry->state == state) {
                            link = entry->link;
                            if (entry->pattern != sz_regex_none_simt_k)
                                inherited = entry->pattern, inherited_end = begin + entry->end, last_at = at;
                            break;
                        }
                        entry->state = state;
                        if (due == high) {
                            link = index;
                            entry->link = index, entry->pattern = sz_regex_none_simt_k;
                            round.links[index].link = state;
                            round.links_list[atomicAdd(round.counters + sz_regex_links_counter_simt_k, 1u)] = index;
                            break;
                        }
                        if (recorded_high == 0) recorded_low = visited;
                        recorded_high = visited + 1;
                    }
                    at = sz_regex_advance_simt_(&table, folded, &haystack, at, haystack.length, accepts, &state, &last,
                                                &last_end, &last_at, &stepped);
                }
                result.link = link;
                result.end = inherited != sz_regex_none_simt_k ? inherited_end : haystack.first + last_end;
                result.pattern = inherited != sz_regex_none_simt_k ? inherited
                                 : last                            ? sz_regex_pattern_simt_(&engine, last)
                                                                   : sz_regex_none_simt_k;
                // The checkpoints passed learn the outcome, keeping the match only when the step
                // reaching it was past them, which a match ending inside the folds of the unit
                // before a checkpoint was not.
                for (checkpoint = recorded_low; checkpoint < recorded_high; ++checkpoint) {
                    sz_bool_t const past = (sz_bool_t)(result.pattern != sz_regex_none_simt_k &&
                                                       haystack.first + 1 + last_at >=
                                                           begin + checkpoint * sz_regex_checkpoint_slots_simt_k);
                    round.checkpoints[checkpoint].link = result.link;
                    round.checkpoints[checkpoint].pattern = past ? result.pattern : sz_regex_none_simt_k;
                    round.checkpoints[checkpoint].end = (sz_u32_t)(result.end - begin);
                }
                round.results[index] = result;
            }
        }
    }
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

/** Walks every link on from its chunk's edge, at the first unit boundary past it for a folded walk,
 *  as @ref sz_regex_link_walk_simt_ walks it. */
static __global__ void sz_regex_links_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                   sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.forward_rows, round.forward_cells);
    sz_size_t const begin = round.segment_begin, end = round.segment_end;
    sz_bool_t const folded = (sz_bool_t)(engine.case_sensitivity != sz_cased_k);
    sz_u32_t const links_count = round.counters[sz_regex_links_counter_simt_k];
    sz_u32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    for (; item < links_count; item += gridDim.x * blockDim.x) {
        sz_u32_t const link = round.links_list[item];
        sz_size_t const chunk = round.slots[link] >> round.chunk_shift;
        sz_size_t const edge = sz_min_of_two(begin + ((chunk + 1) << round.chunk_shift), end);
        sz_regex_haystack_simt_t const haystack = sz_regex_haystack_simt_(
            haystacks, sz_regex_haystack_at_simt_(haystacks, begin + round.slots[link], round.chunk_haystacks[chunk]));
        sz_size_t const at = edge - haystack.first - 1;
        sz_regex_link_walk_simt_(&engine, &round, &table, folded, link, round.links[link].link, &haystack,
                                 folded ? sz_regex_boundary_simt_(haystack.text, haystack.length, at) : at);
    }
}

/** Walks every group the last slide made on from where its walks paused, as the links kernel walks
 *  links, the window having moved two segments on from its end there. */
static __global__ void sz_regex_resume_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                    sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.forward_rows, round.forward_cells);
    sz_size_t const paused = round.looks.begin;
    sz_bool_t const folded = (sz_bool_t)(engine.case_sensitivity != sz_cased_k);
    sz_regex_haystack_simt_t const haystack = sz_regex_haystack_simt_(haystacks,
                                                                      sz_regex_haystack_at_simt_(haystacks, paused, 0));
    sz_size_t const at = folded ? sz_regex_boundary_simt_(haystack.text, haystack.length, paused - haystack.first)
                                : paused - haystack.first;
    sz_u32_t const groups_count = round.counters[sz_regex_groups_counter_simt_k];
    sz_u32_t group = round.groups_begin + blockIdx.x * blockDim.x + threadIdx.x;
    for (; group < groups_count; group += gridDim.x * blockDim.x)
        sz_regex_link_walk_simt_(&engine, &round, &table, folded, sz_regex_group_simt_k | group,
                                 round.groups[group].link, &haystack, at);
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

/**
 *  @brief Walks every link or group handed to a warp on to its end, one warp each, each lane taking
 *      the next piece of the haystack in turn, as @c sz_regex_continue_metal_kernel_ does.
 *
 *  The pieces are of whole units when @p folded. A walk whose next step would read the
 *  lookaround table past its window pauses there, which its continuation keeps, counted; any
 *  other ends with a zero state.
 *
 *  Lane @c turn 's answers reach every lane by a shuffle down to lane zero and a broadcast, the two
 *  cross-lane steps each vendor binds.
 */
STRINGZILLA_DEVICE void sz_regex_continue_simt_(sz_regex_engine_t const *engine, sz_strings_view_t haystacks,
                                                sz_regex_round_simt_t const *round, sz_bool_t folded,
                                                int (*shuffle_down)(int, unsigned), sz_u32_t (*broadcast)(sz_u32_t),
                                                int (*any)(int)) {
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(engine, round, engine->forward_rows, round->forward_cells);
    sz_regex_continuation_simt_t *const continuations = (sz_regex_continuation_simt_t *)round->links_next;
    sz_size_t const columns = sz_regex_stride_(engine->classes_count, engine->lookarounds_count);
    sz_u32_t const count = round->counters[sz_regex_handed_counter_simt_k], accepts = engine->forward_accepts_begin;
    sz_u32_t const lane = threadIdx.x % 32, warps = gridDim.x * (blockDim.x / 32);
    sz_u32_t item = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    for (; item < count; item += warps) {
        sz_regex_continuation_simt_t const continuation = continuations[item];
        sz_regex_haystack_simt_t const haystack = sz_regex_haystack_simt_(
            haystacks, sz_regex_haystack_at_simt_(haystacks, continuation.slot, 0));
        sz_u8_t const *const text = haystack.text;
        sz_size_t const length = haystack.length;
        // The pieces end where the window does, at the first unit boundary at or past it.
        sz_size_t const stop = !sz_regex_past_simt_(table.looks, haystack.first + length) ? length
                               : folded ? sz_regex_boundary_simt_(text, length, table.looks->end - haystack.first)
                                        : table.looks->end - haystack.first;
        sz_u32_t state = continuation.state, last = 0;
        sz_size_t at = continuation.slot - haystack.first, last_end = 0, last_at, stepped;
        sz_bool_t paused = sz_false_k;
        while (state != 0) {
            sz_u32_t domain[sz_regex_priming_states_simt_k], set[sz_regex_priming_states_simt_k];
            sz_u32_t lasts[sz_regex_priming_states_simt_k], lasts_offsets[sz_regex_priming_states_simt_k];
            sz_u32_t column = 0, domain_count = 0, index, turn, byte;
            sz_size_t piece, piece_end, position, unit_end, round_at;
            sz_utf8_folded_image_t image;
            int wide = 0;
            if (sz_regex_past_simt_(table.looks, haystack.first + at)) {
                paused = sz_true_k;
                break;
            }
            if (at == length) {
                state = sz_regex_step_edge_simt_(&table, state, haystack.first + at);
                if (state >= accepts) last = state, last_end = length;
                break;
            }
            piece = sz_regex_piece_simt_(folded, &haystack, at, lane, stop);
            piece_end = sz_regex_piece_simt_(folded, &haystack, at, lane + 1, stop), unit_end = piece;
            if (piece < piece_end) {
                unit_end = sz_regex_unit_after_simt_(folded, text, length, piece, &image);
                column = sz_regex_block_at_simt_(&table, haystack.first + piece) +
                         table.classes[sz_regex_image_byte_simt_(&image, 0)];
            }
            if (piece < piece_end && lane == 0) {
                domain[0] = sz_regex_move_simt_(&table, state, column);
                domain_count = domain[0] != 0;
            }
            else if (piece < piece_end) {
                domain_count = round->forward_images[column];
                wide = domain_count > sz_regex_priming_states_simt_k;
                for (index = 0; !wide && index != domain_count; ++index)
                    domain[index] = round->forward_images[columns + column * sz_regex_priming_states_simt_k + index];
            }
            if (any(wide)) {
                sz_size_t const reach = sz_regex_piece_simt_(folded, &haystack, at, 32, stop);
                while (lane == 0 && state != 0 && at != reach)
                    at = sz_regex_advance_simt_(&table, folded, &haystack, at, reach, accepts, &state, &last, &last_end,
                                                &last_at, &stepped);
                state = broadcast(state), last = broadcast(last);
                last_end = (sz_size_t)broadcast((sz_u32_t)last_end) |
                           ((sz_size_t)broadcast((sz_u32_t)(last_end >> 32)) << 32);
                at = state ? reach : at;
                continue;
            }

            // Every state of the domain walks the piece on its own, remembering its last accepting
            // step, the first folded byte already stepped into the domain.
#pragma unroll
            for (index = 0; index != sz_regex_priming_states_simt_k; ++index) {
                set[index] = index < domain_count ? domain[index] : 0, lasts_offsets[index] = 0;
                lasts[index] = set[index] >= accepts ? set[index] : 0;
            }
            for (position = piece, byte = 1; position < piece_end; ++byte) {
                sz_u32_t byte_column;
                sz_bool_t alive = sz_false_k;
                if (byte == image.length) {
                    position = unit_end, byte = 0;
                    if (position == piece_end) break;
                    unit_end = sz_regex_unit_after_simt_(folded, text, length, position, &image);
                }
                byte_column = (byte ? 0 : sz_regex_block_at_simt_(&table, haystack.first + position)) +
                              table.classes[sz_regex_image_byte_simt_(&image, byte)];
#pragma unroll
                for (index = 0; index != sz_regex_priming_states_simt_k; ++index) {
                    if (index >= domain_count || !set[index]) continue;
                    set[index] = sz_regex_move_simt_(&table, set[index], byte_column);
                    if (set[index] >= accepts)
                        lasts[index] = set[index],
                        lasts_offsets[index] = (sz_u32_t)((byte ? unit_end : position) - piece);
                    alive = (sz_bool_t)(alive || set[index]);
                }
                if (!alive) break;
            }

            // The lanes settle in turn, each from the state the one before it left.
            round_at = at;
            for (turn = 0; turn != 32 && state != 0; ++turn) {
                sz_size_t const turn_piece = sz_regex_piece_simt_(folded, &haystack, round_at, turn, stop);
                sz_size_t const turn_end = sz_regex_piece_simt_(folded, &haystack, round_at, turn + 1, stop);
                sz_u32_t exit = 0, turn_last = 0, turn_offset = 0;
                if (turn_piece >= turn_end) break;
                if (lane == turn) {
                    sz_u32_t const entered = sz_regex_move_simt_(&table, state, column);
#pragma unroll
                    for (index = 0; index != sz_regex_priming_states_simt_k; ++index)
                        if (index < domain_count && entered && domain[index] == entered)
                            exit = set[index], turn_last = lasts[index], turn_offset = lasts_offsets[index];
                }
                state = broadcast((sz_u32_t)shuffle_down((int)exit, turn));
                turn_last = broadcast((sz_u32_t)shuffle_down((int)turn_last, turn));
                turn_offset = broadcast((sz_u32_t)shuffle_down((int)turn_offset, turn));
                if (turn_last) last = turn_last, last_end = turn_piece + turn_offset;
                at = turn_end;
            }
        }
        if (lane != 0) continue;
        if (last) {
            sz_regex_result_simt_t *const walked = sz_regex_walked_simt_(round, continuation.link);
            walked->pattern = sz_regex_pattern_simt_(engine, last), walked->end = haystack.first + last_end;
        }
        continuations[item].state = paused ? state : 0;
        if (paused)
            continuations[item].slot = haystack.first + at,
            atomicAdd(round->counters + sz_regex_paused_counter_simt_k, 1u);
    }
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

/** One round of pointer doubling over the links, each taking the deepest match its hops reach. */
static __global__ void sz_regex_links_double_simt_kernel_(sz_regex_round_simt_t round) {
    sz_bool_t const even = (sz_bool_t)(round.round % 2 == 0);
    sz_regex_result_simt_t const *source = even ? round.links : round.links_next;
    sz_regex_result_simt_t *target = even ? round.links_next : round.links;
    sz_u32_t const links_count = round.counters[sz_regex_links_counter_simt_k];
    sz_u32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    for (; item < links_count; item += gridDim.x * blockDim.x) {
        sz_u32_t const link = round.links_list[item];
        target[link] = sz_regex_hops_simt_(source, source[link]);
    }
}

/**
 *  @brief Gives each distinct state among the walks paused at the end of the lookaround table's
 *      window one group, one thread per walk handed on.
 *
 *  The first walk to take its state's leader allocates the group and publishes it there, and one
 *  past the groups' capacity raises the refusal instead.
 */
static __global__ void sz_regex_group_simt_kernel_(sz_regex_engine_t engine, sz_regex_round_simt_t round) {
    sz_regex_continuation_simt_t const *const continuations = (sz_regex_continuation_simt_t const *)round.links_next;
    sz_size_t const columns = sz_regex_stride_(engine.classes_count, engine.lookarounds_count);
    sz_u32_t const count = round.counters[sz_regex_handed_counter_simt_k];
    sz_u32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    for (; item < count; item += gridDim.x * blockDim.x) {
        sz_u32_t const state = continuations[item].state;
        sz_u32_t *const leader = round.leaders + state / columns;
        sz_u32_t group;
        if (!state || atomicCAS(leader, sz_regex_none_simt_k, item) != sz_regex_none_simt_k) continue;
        group = atomicAdd(round.counters + sz_regex_groups_counter_simt_k, 1u);
        if (group >= round.groups_capacity) {
            *round.looks.refused = 1;
            continue;
        }
        round.groups[group].link = state, round.groups[group].pattern = sz_regex_none_simt_k;
        round.groups[group].end = 0;
        *leader = sz_regex_group_simt_k | group;
    }
}

/** Points every walk paused at the end of the lookaround table's window at its state's group, which
 *  a link or a group of the window's last end then follows as the link whose match wins. */
static __global__ void sz_regex_point_simt_kernel_(sz_regex_engine_t engine, sz_regex_round_simt_t round) {
    sz_regex_continuation_simt_t const *const continuations = (sz_regex_continuation_simt_t const *)round.links_next;
    sz_size_t const columns = sz_regex_stride_(engine.classes_count, engine.lookarounds_count);
    sz_u32_t const count = round.counters[sz_regex_handed_counter_simt_k];
    sz_u32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    for (; item < count; item += gridDim.x * blockDim.x) {
        sz_regex_continuation_simt_t const continuation = continuations[item];
        sz_u32_t group;
        if (!continuation.state) continue;
        group = round.leaders[continuation.state / columns];
        if (group & sz_regex_group_simt_k) sz_regex_walked_simt_(&round, continuation.link)->link = group;
    }
}

/** One round of pointer doubling over the groups, as over the links, each taking the deepest match
 *  a round's hops ahead among the groups of later window ends. */
static __global__ void sz_regex_groups_double_simt_kernel_(sz_regex_round_simt_t round) {
    sz_bool_t const even = (sz_bool_t)(round.round % 2 == 0);
    sz_regex_result_simt_t const *source = even ? round.groups : round.groups_next;
    sz_regex_result_simt_t *target = even ? round.groups_next : round.groups;
    sz_u32_t const count = round.counters[sz_regex_groups_counter_simt_k];
    sz_u32_t group = blockIdx.x * blockDim.x + threadIdx.x;
    for (; group < count; group += gridDim.x * blockDim.x) target[group] = sz_regex_hops_simt_(source, source[group]);
}

/** Gives every link that paused at the end of the lookaround table's window its settled group's
 *  match, when it has one, and unlinks it, so the links double as though it never paused. */
static __global__ void sz_regex_take_simt_kernel_(sz_regex_round_simt_t round) {
    sz_u32_t const links_count = round.counters[sz_regex_links_counter_simt_k];
    sz_u32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    for (; item < links_count; item += gridDim.x * blockDim.x) {
        sz_regex_result_simt_t *const link = round.links + round.links_list[item];
        sz_regex_result_simt_t group;
        if (link->link == sz_regex_none_simt_k || !(link->link & sz_regex_group_simt_k)) continue;
        group = round.groups[link->link & ~sz_regex_group_simt_k];
        link->link = sz_regex_none_simt_k;
        if (group.pattern != sz_regex_none_simt_k) link->pattern = group.pattern, link->end = group.end;
    }
}

/**
 *  @brief Tries the later runes of every folded candidate whose unit folds to several and whose
 *      first rune found no match once the links settled, as @ref sz_regex_match_at_simt_ does.
 *
 *  It takes one thread per candidate, each walked to its end without links, raising the refusal
 *  rather than reading the lookaround table past its window.
 */
static __global__ void sz_regex_retry_folded_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                          sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.forward_rows, round.forward_cells);
    sz_size_t const count = round.chunk_offsets[round.chunks_count], stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t item = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; item < count; item += stride) {
        sz_size_t const mark = round.segment_begin + round.slots[item];
        sz_regex_haystack_simt_t const haystack = sz_regex_haystack_simt_(
            haystacks,
            sz_regex_haystack_at_simt_(haystacks, mark, round.chunk_haystacks[round.slots[item] >> round.chunk_shift]));
        sz_size_t const start = mark - haystack.first;
        sz_utf8_folded_image_t image;
        sz_size_t end, offset;
        sz_u32_t pattern;
        if (start == haystack.length ||
            sz_regex_settle_simt_(&round, (sz_u32_t)item, mark, &haystack).pattern != sz_regex_none_simt_k)
            continue;
        sz_regex_unit_after_simt_(sz_true_k, haystack.text, haystack.length, start, &image);
        if (!(image.rune_ends & (image.rune_ends - 1))) continue;
        offset = sz_regex_lowest_simt_(image.rune_ends) + 1;
        if (sz_regex_match_at_simt_(&engine, &table, STRINGZILLA_NULL, &haystack, start, &end, &pattern, &offset))
            round.results[item].link = sz_regex_none_simt_k, round.results[item].pattern = pattern,
            round.results[item].end = haystack.first + end;
    }
}

#pragma endregion Walk Kernels

#pragma region Chain Kernels

/** Follows the search through one chunk from every candidate as an entry, right to left, one thread
 *  per chunk, and seeds each haystack's chain, as @c sz_regex_chain_metal_kernel_ does. */
static __global__ void sz_regex_chain_simt_kernel_(sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_size_t const begin = round.segment_begin, cursor = round.carry->cursor;
    sz_bool_t const refusing = (sz_bool_t)(round.carry->refusing != 0);
    sz_u32_t const count = (sz_u32_t)round.chunk_offsets[round.chunks_count];
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_u32_t const first = (sz_u32_t)round.chunk_offsets[chunk], end = (sz_u32_t)round.chunk_offsets[chunk + 1];
        sz_size_t haystack_index;
        sz_regex_haystack_simt_t haystack;
        sz_bool_t leaves = sz_false_k;
        sz_u32_t index;
        if (first == end) continue;
        haystack_index = sz_regex_haystack_at_simt_(haystacks, begin + round.slots[end - 1],
                                                    round.chunk_haystacks[chunk]);
        haystack = sz_regex_haystack_simt_(haystacks, haystack_index);
        for (index = end; index-- != first;) {
            sz_size_t const mark = begin + round.slots[index];
            sz_size_t last;
            sz_regex_result_simt_t result;
            sz_u32_t next = index + 1, jump;
            sz_bool_t seed;
            while (mark < haystack.first) haystack = sz_regex_haystack_simt_(haystacks, --haystack_index);
            last = haystack.first + haystack.length;
            result = sz_regex_settle_simt_(&round, index, mark, &haystack);
            if (result.pattern != sz_regex_none_simt_k) {
                next = result.end >= round.segment_end
                           ? count
                           : sz_regex_lower_bound_simt_(round.slots, index, count, result.end - begin);
                if (next < count && begin + round.slots[next] == result.end) {
                    sz_regex_result_simt_t const after = sz_regex_settle_simt_(&round, next, result.end, &haystack);
                    if (after.pattern != sz_regex_none_simt_k && after.end == result.end) ++next;
                }
            }
            jump = next < count && begin + round.slots[next] <= last ? next : sz_regex_none_simt_k;
            round.nexts[index] = jump;
            round.exits[index] = jump == sz_regex_none_simt_k || jump >= end ? jump : round.exits[jump];
            leaves = (sz_bool_t)(leaves || round.exits[index] != sz_regex_none_simt_k);

            // A haystack started in this segment is seeded at its first candidate; the one
            // straddling from the last segment at its first candidate past the cursor, unless that
            // is a refused empty match.
            if (haystack.first >= begin)
                seed = (sz_bool_t)(index == 0 || begin + round.slots[index - 1] < haystack.first);
            else {
                sz_bool_t const first_past = (sz_bool_t)(mark >= cursor &&
                                                         (index == 0 || begin + round.slots[index - 1] < cursor));
                sz_bool_t const refused = (sz_bool_t)(first_past && refusing && mark == cursor &&
                                                      result.pattern != sz_regex_none_simt_k && result.end == mark);
                seed = (sz_bool_t)(first_past && !refused);
                if (index != 0 && !first_past && mark > cursor) {
                    sz_size_t const previous = begin + round.slots[index - 1];
                    sz_bool_t const previous_first =
                        (sz_bool_t)(previous >= cursor && (index == 1 || begin + round.slots[index - 2] < cursor));
                    if (previous_first && refusing && previous == cursor) {
                        sz_regex_result_simt_t const before = sz_regex_settle_simt_(&round, index - 1, previous,
                                                                                    &haystack);
                        seed = (sz_bool_t)(before.pattern != sz_regex_none_simt_k && before.end == previous);
                    }
                }
            }
            round.keep[index] = seed;
        }
        if (leaves) round.flags[0] = 1;
    }
}

/** One round of pointer doubling over the chunk exits: every kept candidate keeps those its next
 *  hops reach, and jumps that far again; a round after one that left nothing to follow retires. */
static __global__ void sz_regex_double_simt_kernel_(sz_regex_round_simt_t round) {
    sz_bool_t const even = (sz_bool_t)(round.round % 2 == 0);
    sz_u32_t const *source = even ? round.exits : round.exits_next;
    sz_u32_t *target = even ? round.exits_next : round.exits;
    sz_u32_t const count = (sz_u32_t)round.chunk_offsets[round.chunks_count];
    sz_u32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (!round.flags[round.round]) return;
    for (; index < count; index += gridDim.x * blockDim.x) {
        // A mark set in this round only reaches candidates on the chain anyway.
        sz_bool_t const kept = (sz_bool_t)(round.keep[index] != 0);
        sz_u32_t jump = source[index], hop;
        for (hop = 1; hop != sz_regex_round_hops_simt_k && jump != sz_regex_none_simt_k; ++hop) {
            if (kept) round.keep[jump] = 1;
            jump = source[jump];
        }
        if (kept && jump != sz_regex_none_simt_k) round.keep[jump] = 1;
        target[index] = jump == sz_regex_none_simt_k ? jump : source[jump];
        if (target[index] != sz_regex_none_simt_k) round.flags[round.round + 1] = 1;
    }
}

/**
 *  @brief Follows the chain through one chunk from the candidates the doubling kept, left to right,
 *      one thread per chunk, as the Metal kernel does.
 *
 *  It counts what it keeps, flagging each kept candidate and counting those of the extracted
 *  pattern, or, once the counts are scanned, stores what the call asks.
 */
static __global__ void sz_regex_sweep_simt_kernel_(sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_size_t const begin = round.segment_begin, width = (sz_size_t)1 << round.chunk_shift;
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk == 0 && !round.storing)
        round.chunk_kept[round.chunks_count] = 0, round.chunk_sums[round.chunks_count] = 0;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_size_t const low = begin + chunk * width, high = sz_min_of_two(low + width, round.segment_end);
        sz_u32_t const first = (sz_u32_t)round.chunk_offsets[chunk], end = (sz_u32_t)round.chunk_offsets[chunk + 1];
        sz_size_t const base = round.storing ? round.carry->running + round.chunk_kept[chunk] : 0;
        sz_size_t haystack_index = round.chunk_haystacks[chunk], pending = haystack_index;
        sz_size_t last_start = 0, last_end = 0, kept_pattern = 0;
        sz_regex_haystack_simt_t haystack = sz_regex_haystack_simt_(haystacks, haystack_index);
        sz_u32_t expected = sz_regex_none_simt_k, kept = 0, index;
        if (haystack.first < low) ++pending;
        for (index = first; index != end; ++index) {
            sz_size_t const mark = begin + round.slots[index];
            sz_regex_result_simt_t result;
            if (round.storing)
                for (; pending < haystacks.count && sz_regex_first_slot_simt_(haystacks, pending) <= mark; ++pending)
                    sz_regex_boundary_store_simt_(&round, pending, base + kept);
            while (haystack_index + 1 < haystacks.count &&
                   sz_regex_first_slot_simt_(haystacks, haystack_index + 1) <= mark)
                haystack = sz_regex_haystack_simt_(haystacks, ++haystack_index);
            if (!round.storing) round.kept[index] = 0;
            if (!round.keep[index] && index != expected) continue;
            expected = round.nexts[index];
            result = sz_regex_settle_simt_(&round, index, mark, &haystack);
            if (result.pattern == sz_regex_none_simt_k) continue;
            if (round.storing && base + kept < round.matches_capacity) {
                sz_match_t match;
                match.haystack_index = haystack_index, match.needle_index = result.pattern;
                match.byte_offset = mark - haystack.first, match.byte_length = result.end - mark;
                round.matches[base + kept] = match;
            }
            if (!round.storing) round.kept[index] = 1, kept_pattern += result.pattern == round.pattern_index;
            last_start = mark, last_end = result.end, ++kept;
        }
        if (!round.storing) {
            round.chunk_kept[chunk] = kept, round.chunk_sums[chunk] = kept_pattern;
            continue;
        }
        for (; pending < haystacks.count && sz_regex_first_slot_simt_(haystacks, pending) < high; ++pending)
            sz_regex_boundary_store_simt_(&round, pending, base + kept);
        // The chunk with the segment's last kept match tells the carry where its haystack resumes.
        if (kept && round.chunk_kept[chunk] + kept == round.chunk_kept[round.chunks_count])
            round.carry->last_start = last_start, round.carry->last_end = last_end;
    }
}

/** Hands the next segment its carry, launched on one thread, as the Metal kernel does, and the last
 *  boundary of whatever the call writes. */
static __global__ void sz_regex_carry_simt_kernel_(sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_carry_simt_t *const carry = round.carry;
    sz_size_t const kept = round.chunk_kept[round.chunks_count];
    if (round.segment_end < round.slots_count) {
        sz_size_t const first = sz_regex_first_slot_simt_(haystacks,
                                                          sz_regex_haystack_at_simt_(haystacks, round.segment_end, 0));
        if (kept && carry->last_start >= first) carry->cursor = carry->last_end, carry->refusing = 1;
        else if (first >= round.segment_begin) carry->cursor = first, carry->refusing = 0;
        if (round.rows_offsets && first >= round.segment_begin && first < round.segment_end)
            carry->straddle = carry->counted + sz_regex_counted_before_simt_(&round, first);
    }
    carry->running += kept;
    sz_regex_boundary_store_simt_(&round, haystacks.count, carry->running);
    if (round.offsets) {
        carry->summed += round.chunk_sums[round.chunks_count];
        round.offsets[haystacks.count] = round.slots_count - haystacks.count + carry->summed;
    }
    if (round.rows_offsets) {
        carry->counted += round.chunk_sums[round.chunks_count];
        carry->summed += round.chunk_covers[round.chunks_count];
        round.rows_offsets[haystacks.count] = carry->summed;
    }
}

/** Saves the total each thread's range of counts follows, before the differences overwrite it. */
static __global__ void sz_regex_counts_edges_simt_kernel_(sz_size_t const *counts, sz_size_t counts_stride,
                                                          sz_size_t count, sz_size_t *edges) {
    sz_size_t const threads = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const begin = sz_min_of_two(thread * sz_size_divide_round_up(count, threads), count);
    edges[thread] = begin ? counts[(begin - 1) * counts_stride] : 0;
}

/** Turns the running totals the sweep left in @p counts into each haystack's own count, every
 *  thread over its own range right to left, so it reads each total before overwriting it. */
static __global__ void sz_regex_counts_simt_kernel_(sz_size_t *counts, sz_size_t counts_stride, sz_size_t count,
                                                    sz_size_t const *edges) {
    sz_size_t const threads = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const per_thread = sz_size_divide_round_up(count, threads);
    sz_size_t const begin = sz_min_of_two(thread * per_thread, count), end = sz_min_of_two(begin + per_thread, count);
    sz_size_t index;
    for (index = end; index-- != begin;)
        counts[index * counts_stride] -= index == begin ? edges[thread] : counts[(index - 1) * counts_stride];
}

#pragma endregion Chain Kernels

#pragma region Rewrite Kernels

/** Measures every kept match's replacement, one thread per candidate in each thread's share of the
 *  walks, leaving how far it moves what follows. */
static __global__ void sz_regex_expand_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                    sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_direct_simt_(&engine, &round);
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const count = round.chunk_offsets[round.chunks_count];
    sz_regex_pike_simt_t pike;
    sz_size_t item;
    if (thread >= round.walks_count) return;
    pike = sz_regex_pike_carve_simt_(&round, thread, &engine);
    for (item = thread; item < count; item += round.walks_count) {
        sz_size_t const mark = round.segment_begin + round.slots[item];
        sz_regex_haystack_simt_t haystack;
        sz_regex_result_simt_t result;
        sz_size_t expanded;
        round.drifts[item] = 0;
        if (!round.kept[item]) continue;
        haystack = sz_regex_haystack_simt_(
            haystacks,
            sz_regex_haystack_at_simt_(haystacks, mark, round.chunk_haystacks[round.slots[item] >> round.chunk_shift]));
        result = sz_regex_linked_simt_(&round, (sz_u32_t)item);
        expanded = sz_regex_expand_simt_(&engine, &table, &pike, &haystack,
                                         (sz_u8_t const *)sz_strings_start_simt_(round.replacements, result.pattern),
                                         sz_strings_length_simt_(round.replacements, result.pattern),
                                         mark - haystack.first, result.end - haystack.first, result.pattern,
                                         round.pike_slots, STRINGZILLA_NULL, 0, 0);
        round.drifts[item] = expanded - (result.end - mark);
    }
}

/** Sums the drift of every chunk's kept matches, one thread per chunk, and tells the chunks the
 *  last of them reaches into that they start covered. */
static __global__ void sz_regex_drift_simt_kernel_(sz_regex_round_simt_t round) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk == 0) round.chunk_sums[round.chunks_count] = 0;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_size_t drift = 0, covered = 0, next;
        sz_u32_t index;
        for (index = (sz_u32_t)round.chunk_offsets[chunk]; index != (sz_u32_t)round.chunk_offsets[chunk + 1]; ++index)
            if (round.kept[index]) drift += round.drifts[index], covered = sz_regex_linked_simt_(&round, index).end;
        round.chunk_sums[chunk] = drift;
        for (next = chunk + 1;
             next < round.chunks_count && round.segment_begin + (next << round.chunk_shift) <= covered; ++next)
            round.chunk_covers[next] = covered;
    }
}

/**
 *  @brief Writes one chunk's share of the rewritten haystacks, one thread per chunk in each
 *      thread's share of the walks: every byte no kept match covers, every kept match's
 *      replacement, and where each haystack starting in the chunk begins.
 *
 *  Byte @c b of haystack @c h, in slot @c s, lands at `s - h - 1` plus the drift of the kept
 *  matches before it, and a match starting in slot @c s at `s - h` plus the same.
 */
static __global__ void sz_regex_rewrite_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                     sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_direct_simt_(&engine, &round);
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const begin = round.segment_begin, width = (sz_size_t)1 << round.chunk_shift;
    sz_regex_pike_simt_t pike;
    sz_size_t chunk;
    if (thread >= round.walks_count) return;
    pike = sz_regex_pike_carve_simt_(&round, thread, &engine);
    for (chunk = thread; chunk < round.chunks_count; chunk += round.walks_count) {
        sz_size_t const low = begin + chunk * width, high = sz_min_of_two(low + width, round.segment_end);
        sz_u32_t index = (sz_u32_t)round.chunk_offsets[chunk], end = (sz_u32_t)round.chunk_offsets[chunk + 1];
        sz_size_t drift = round.carry->summed + round.chunk_sums[chunk];
        sz_size_t covered = sz_max_of_two(round.carry->last_end, round.chunk_covers[chunk]);
        sz_size_t haystack_index = round.chunk_haystacks[chunk], slot;
        sz_regex_haystack_simt_t haystack = sz_regex_haystack_simt_(haystacks, haystack_index);
        for (slot = low; slot != high; ++slot) {
            if (haystack_index + 1 < haystacks.count &&
                sz_regex_first_slot_simt_(haystacks, haystack_index + 1) == slot)
                haystack = sz_regex_haystack_simt_(haystacks, ++haystack_index);
            if (slot == haystack.first) round.offsets[haystack_index] = slot - haystack_index + drift;
            else if (slot > covered && slot - haystack_index - 1 + drift < round.target_capacity)
                round.target[slot - haystack_index - 1 + drift] = (char)haystack.text[slot - haystack.first - 1];
            if (index == end || begin + round.slots[index] != slot) continue;
            if (round.kept[index]) {
                sz_regex_result_simt_t const result = sz_regex_linked_simt_(&round, index);
                sz_regex_expand_simt_(&engine, &table, &pike, &haystack,
                                      (sz_u8_t const *)sz_strings_start_simt_(round.replacements, result.pattern),
                                      sz_strings_length_simt_(round.replacements, result.pattern),
                                      slot - haystack.first, result.end - haystack.first, result.pattern,
                                      round.pike_slots, round.target, round.target_capacity,
                                      slot - haystack_index + drift);
                drift += round.drifts[index], covered = result.end;
            }
            ++index;
        }
    }
}

#pragma endregion Rewrite Kernels

#pragma region Extract Kernels

/** Finds every haystack's first kept match of the extracted pattern and counts its rows, one thread
 *  per chunk in each thread's share of the walks: a match is the first when as many kept matches of
 *  the pattern precede it as precede its haystack. */
static __global__ void sz_regex_firsts_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                    sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_direct_simt_(&engine, &round);
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const begin = round.segment_begin;
    sz_regex_pike_simt_t pike;
    sz_size_t chunk;
    if (thread == 0) round.chunk_covers[round.chunks_count] = 0;
    if (thread >= round.walks_count) return;
    pike = sz_regex_pike_carve_simt_(&round, thread, &engine);
    for (chunk = thread; chunk < round.chunks_count; chunk += round.walks_count) {
        sz_size_t ranked = round.carry->counted + round.chunk_sums[chunk], preceding = 0, rows = 0;
        sz_size_t haystack_index = round.chunk_haystacks[chunk], ranked_haystack = STRINGZILLA_SIZE_MAX;
        sz_regex_haystack_simt_t haystack = sz_regex_haystack_simt_(haystacks, haystack_index);
        sz_u32_t index;
        for (index = (sz_u32_t)round.chunk_offsets[chunk]; index != (sz_u32_t)round.chunk_offsets[chunk + 1]; ++index) {
            sz_size_t const mark = begin + round.slots[index];
            sz_regex_result_simt_t result;
            round.exits[index] = 0;
            while (haystack_index + 1 < haystacks.count &&
                   sz_regex_first_slot_simt_(haystacks, haystack_index + 1) <= mark)
                haystack = sz_regex_haystack_simt_(haystacks, ++haystack_index);
            if (!round.kept[index]) continue;
            result = sz_regex_linked_simt_(&round, index);
            if (result.pattern != round.pattern_index) continue;
            if (ranked_haystack != haystack_index)
                ranked_haystack = haystack_index,
                preceding = haystack.first >= begin
                                ? round.carry->counted + sz_regex_counted_before_simt_(&round, haystack.first)
                                : round.carry->straddle;
            if (ranked++ != preceding) continue;
            round.exits[index] = (sz_u32_t)sz_regex_rows_write_simt_(&engine, &table, &pike, &haystack, haystack_index,
                                                                     mark - haystack.first, result.end - haystack.first,
                                                                     result.pattern, STRINGZILLA_NULL, 0, 0);
            rows += round.exits[index];
        }
        round.chunk_covers[chunk] = rows;
    }
}

/** Writes the rows the firsts kernel counted and where each haystack starting in a chunk begins
 *  among them, one thread per chunk in each thread's share of the walks. */
static __global__ void sz_regex_rows_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                  sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_direct_simt_(&engine, &round);
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const begin = round.segment_begin, width = (sz_size_t)1 << round.chunk_shift;
    sz_regex_pike_simt_t pike;
    sz_size_t chunk;
    if (thread >= round.walks_count) return;
    pike = sz_regex_pike_carve_simt_(&round, thread, &engine);
    for (chunk = thread; chunk < round.chunks_count; chunk += round.walks_count) {
        sz_size_t const low = begin + chunk * width, high = sz_min_of_two(low + width, round.segment_end);
        sz_size_t const base = round.carry->summed + round.chunk_covers[chunk];
        sz_size_t haystack_index = round.chunk_haystacks[chunk], pending = haystack_index, written = 0;
        sz_regex_haystack_simt_t haystack = sz_regex_haystack_simt_(haystacks, haystack_index);
        sz_u32_t index;
        if (haystack.first < low) ++pending;
        for (index = (sz_u32_t)round.chunk_offsets[chunk]; index != (sz_u32_t)round.chunk_offsets[chunk + 1]; ++index) {
            sz_size_t const mark = begin + round.slots[index];
            sz_regex_result_simt_t result;
            for (; pending < haystacks.count && sz_regex_first_slot_simt_(haystacks, pending) <= mark; ++pending)
                round.rows_offsets[pending] = base + written;
            while (haystack_index + 1 < haystacks.count &&
                   sz_regex_first_slot_simt_(haystacks, haystack_index + 1) <= mark)
                haystack = sz_regex_haystack_simt_(haystacks, ++haystack_index);
            if (!round.exits[index]) continue;
            result = sz_regex_linked_simt_(&round, index);
            sz_regex_rows_write_simt_(&engine, &table, &pike, &haystack, haystack_index, mark - haystack.first,
                                      result.end - haystack.first, result.pattern, round.rows, round.rows_capacity,
                                      base + written);
            written += round.exits[index];
        }
        for (; pending < haystacks.count && sz_regex_first_slot_simt_(haystacks, pending) < high; ++pending)
            round.rows_offsets[pending] = base + written;
    }
}

#pragma endregion Extract Kernels

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_REGEX_SIMT_CUH_
