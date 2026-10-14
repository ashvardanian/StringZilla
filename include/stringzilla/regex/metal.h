/**
 *  @file include/stringzilla/regex/metal.h
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Metal backend for regular expressions on Apple7 and newer: the vocabulary compiled on
 *      the host into one shared block, and one command buffer per call, the CUDA tier's
 *      launches as dispatches.
 *
 *  @sa include/stringzilla/regex/metal.metal, the kernels this embeds
 *  @sa include/stringzilla/regex/simt.cuh, the CUDA sibling
 *
 *  The verbs keep the serial tier's shape, with an @c id<MTLCommandQueue> as the stream. Every
 *  block the kernels read must come from @ref sz_allocator_init_unified_metal on that stream's
 *  device, and haystacks and replacements must be tapes, as @ref sz_strings_to_tape_best writes
 *  them with that allocator. The round block is taken once at init, and a call reads the tape's
 *  offsets on the host to cut its corpus into segments the block holds.
 *
 *  A call walks its corpus in segments of slots, each one round of dispatches: mark the starts,
 *  walk the ends, follow the links, settle the chain, and store what it kept. The segment's width
 *  bounds every candidate array, so the round block stays sized by the device rather than by the
 *  matches. Lookarounds hold in a table over a window of two segments; only where a haystack runs
 *  past that window does a call commit what it encoded, wait, read the round's counters back and go
 *  on, which is how its walks slide past the window's end. Otherwise nothing is read back until the
 *  caller synchronizes.
 */
#ifndef STRINGZILLA_REGEX_METAL_H_
#define STRINGZILLA_REGEX_METAL_H_

#include "stringzilla/metal.h"
#include "stringzilla/regex/serial.h"

#if STRINGZILLA_ARCH_METAL_

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_TARGET_METAL

#pragma region Metal

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"

/** The MSL source of the regex kernels, compiled once per device behind the shared prelude. */
static char const sz_regex_source_metal_[] = {
#embed "stringzilla/utf8_uncased_fold/tables.h"
    ,
#embed "stringzilla/utf8_uncased_fold/metal.metal"
    ,
#embed "stringzilla/regex/metal.metal"
    , 0};
#pragma clang diagnostic pop

/** Threads every threadgroup runs, as in the kernels. */
enum { sz_regex_threads_per_group_metal_k = 256 };

/** Threadgroups a grid-strided dispatch takes at most, past which its threads stride instead. */
enum { sz_regex_groups_max_metal_k = 1024 };

/** States a priming set holds, as in the kernels. */
enum { sz_regex_priming_states_metal_k = 16 };

/** Slots between the checkpoints end walks leave, as in the kernels. */
enum { sz_regex_checkpoint_slots_metal_k = 16 };

/** Slots one chunk spans, one word of start bits. */
enum { sz_regex_chunk_slots_metal_k = 64 };

/** Chunks a segment is cut into, enough threads to fill an Apple GPU. */
enum { sz_regex_chunks_metal_k = 1 << 16 };

/** Round-block bytes per slot of a segment that PikeVMs and position sets share out,
 *  one share per thread. */
enum { sz_regex_walk_slot_bytes_metal_k = 16 };

/** Cells of a transition table a walk reads from threadgroup memory,
 *  past which it reads device memory. */
enum { sz_regex_staged_cells_metal_k = 4096 };

/** Hops one round of pointer doubling follows, as in the kernels. */
enum { sz_regex_round_hops_metal_k = 8 };

/** Tiles a scan reduces, one threadgroup each. */
enum { sz_regex_scan_tiles_metal_k = 1024 };

/** Bytes of threadgroup memory the class map and the first column of each block of lookarounds take
 *  ahead of the staged rows, as in the kernels. */
enum { sz_regex_staged_bytes_metal_k = 256 + 16 * sizeof(sz_u32_t) };

/** The round's counters, as the kernels number them: links listed, groups of paused walks opened,
 *  walks handed to SIMD groups, walks paused at the lookaround table's end, and the flag a walk
 *  that cannot pause there sets. */
typedef enum sz_regex_counter_metal_t {
    sz_regex_links_counter_metal_k = 0,
    sz_regex_groups_counter_metal_k = 1,
    sz_regex_handed_counter_metal_k = 2,
    sz_regex_paused_counter_metal_k = 3,
    sz_regex_refused_counter_metal_k = 4,
} sz_regex_counter_metal_t;

/** The buffers a call binds, as the kernels number them, the launch record after the last. */
enum {
    sz_regex_memory_buffer_metal_k,
    sz_regex_scratch_buffer_metal_k,
    sz_regex_haystacks_buffer_metal_k,
    sz_regex_haystacks_offsets_buffer_metal_k,
    sz_regex_matches_buffer_metal_k,
    sz_regex_matches_offsets_buffer_metal_k,
    sz_regex_counts_buffer_metal_k,
    sz_regex_replacements_buffer_metal_k,
    sz_regex_replacements_offsets_buffer_metal_k,
    sz_regex_target_buffer_metal_k,
    sz_regex_buffers_metal_k,
};

/** What a call writes, which the kernels read to tell its outputs apart. */
typedef enum sz_regex_verb_metal_t {

    /** Matches and their boundaries, into the matches and their offsets. */
    sz_regex_finding_metal_k = 0,

    /** Each haystack's count, into the counts. */
    sz_regex_counting_metal_k = 1,

    /** The rewritten haystacks and their boundaries, into the target and the matches' offsets. */
    sz_regex_replacing_metal_k = 2,

    /** Rows of groups and their boundaries, into the matches and their offsets. */
    sz_regex_extracting_metal_k = 3,
} sz_regex_verb_metal_t;

/** Every kernel of `metal.metal`, built at init so no call compiles one. */
static char const *const sz_regex_kernels_metal_[] = {
    "sz_regex_clear_metal_kernel_",
    "sz_regex_scan_reduce_metal_kernel_",
    "sz_regex_scan_carry_metal_kernel_",
    "sz_regex_scan_apply_metal_kernel_",
    "sz_regex_bound_metal_kernel_",
    "sz_regex_flow_pieces_metal_kernel_",
    "sz_regex_affine_metal_kernel_",
    "sz_regex_affine_double_metal_kernel_",
    "sz_regex_looks_metal_kernel_",
    "sz_regex_prime_metal_kernel_",
    "sz_regex_prime_pieces_metal_kernel_",
    "sz_regex_compose_metal_kernel_",
    "sz_regex_compose_double_metal_kernel_",
    "sz_regex_mark_metal_kernel_",
    "sz_regex_mark_folded_metal_kernel_",
    "sz_regex_mark_positions_metal_kernel_",
    "sz_regex_mark_positions_folded_metal_kernel_",
    "sz_regex_list_metal_kernel_",
    "sz_regex_ends_metal_kernel_",
    "sz_regex_ends_pike_metal_kernel_",
    "sz_regex_links_metal_kernel_",
    "sz_regex_resume_metal_kernel_",
    "sz_regex_continue_metal_kernel_",
    "sz_regex_group_metal_kernel_",
    "sz_regex_point_metal_kernel_",
    "sz_regex_groups_double_metal_kernel_",
    "sz_regex_take_metal_kernel_",
    "sz_regex_links_double_metal_kernel_",
    "sz_regex_retry_folded_metal_kernel_",
    "sz_regex_chain_metal_kernel_",
    "sz_regex_double_metal_kernel_",
    "sz_regex_sweep_metal_kernel_",
    "sz_regex_carry_metal_kernel_",
    "sz_regex_counts_edges_metal_kernel_",
    "sz_regex_counts_metal_kernel_",
    "sz_regex_expand_metal_kernel_",
    "sz_regex_drift_metal_kernel_",
    "sz_regex_rewrite_metal_kernel_",
    "sz_regex_firsts_metal_kernel_",
    "sz_regex_rows_metal_kernel_",
};

/** The walk a dispatch's set kernels run, as the kernels number them: the position automaton right
 *  to left, or one level's lookbehind pass left to right or lookahead pass right to left. */
typedef enum sz_regex_flow_metal_t {
    sz_regex_flow_positions_metal_k = 0,
    sz_regex_flow_behind_metal_k = 1,
    sz_regex_flow_ahead_metal_k = 2,
} sz_regex_flow_metal_t;

/** The launch record, laid out as the kernels' record of the same name. */
typedef struct {
    sz_u64_t engine_host, scratch_host;
    sz_u64_t byte_to_class, reverse_rows, forward_rows, forward_patterns;
    sz_u64_t nfa_states, nfa_links, patterns_starts, patterns_groups, names, names_offsets, lookarounds;
    sz_u64_t positions_reach, positions_shifts, positions_groups, positions_all, positions_initial;
    sz_u64_t positions_ascii, positions_ranges;
    sz_u64_t reverse_images, forward_images, entries, edges, carry, counters, flags, tile_sums;
    sz_u64_t chunk_offsets, chunk_kept, chunk_sums, chunk_covers, chunk_haystacks;
    sz_u64_t frontiers, compositions, compositions_next, handoffs, handoff_edges, undecided, affines, affines_next;
    sz_u64_t looks, starts, checkpoints;
    sz_u64_t slots, results, links, links_next, links_list, nexts, exits, exits_next, keep, walks;
    sz_u64_t haystacks_count, haystacks_layout, replacements_layout, slots_count;
    sz_u64_t matches_capacity, target_capacity, counts_stride, pattern_index, verb;
    sz_u64_t segment, segment_begin, segment_end, chunk_width, chunk_shift, chunks_count, round, rounds, storing;
    sz_u64_t walks_count, walk_bytes, walk_sets, pike_slots, scan_count, scan_tiles, target, clear_words;
    sz_u64_t looks_begin, looks_end, looks_mask, set_words, bodies_begin, bodies_count;
    sz_u64_t groups_first, groups_capacity;
    sz_u32_t forward_starts[4], reverse_starts[4];
    sz_u32_t forward_accepts_begin, reverse_accepts_begin, classes_count, case_sensitivity;
    sz_u32_t forward_cells, reverse_cells, staged_cells, nfa_states_count;
    sz_u32_t nfa_links_count, nfa_start, positions_words, positions_shifts_count;
    sz_u32_t positions_groups_count, positions_ranges_count, patterns_count, lookarounds_count;
    sz_u32_t flow, members, negated, stride;
} sz_regex_arguments_metal_t;

/** Bytes of a chunk's uncollapsed priming set and its exits, as the kernels lay them out. */
enum { sz_regex_frontier_bytes_metal_k = (1 + 2 * sz_regex_priming_states_metal_k) * sizeof(sz_u32_t) };

/** Bytes of a chunk's composition, as the kernels lay it out. */
enum { sz_regex_composition_bytes_metal_k = 2 * sizeof(sz_u32_t) + sz_regex_priming_states_metal_k };

/** Undecided elements a chunk's edge set holds, as in the kernels. */
enum { sz_regex_undecided_metal_k = 64 };

/** Bytes of a chunk's undecided elements and of its affine map of them,
 *  as the kernels lay them out. */
enum {
    sz_regex_undecided_bytes_metal_k = (1 + sz_regex_undecided_metal_k) * sizeof(sz_u32_t),
    sz_regex_affine_bytes_metal_k = 2 * sizeof(sz_u32_t) + (1 + sz_regex_undecided_metal_k) * sizeof(sz_u64_t),
};

/** Bytes of the carry between segments, as the kernels lay it out. */
enum { sz_regex_carry_bytes_metal_k = 8 * sizeof(sz_u64_t) };

/** The segment geometry of an engine's round block, and the byte offsets of everything in it. */
typedef struct sz_regex_arena_metal_t {
    sz_size_t segment_slots, pike_slots, pike_bytes, positions_bytes, passes_bytes, set_words;
    sz_size_t entries, edges, carry, counters, flags, tile_sums;
    sz_size_t chunk_offsets, chunk_kept, chunk_sums, chunk_covers, chunk_haystacks;
    sz_size_t frontiers, compositions, compositions_next, handoffs, handoff_edges, undecided, affines, affines_next;
    sz_size_t looks, starts, checkpoints;
    sz_size_t slots, results, links, links_next, links_list, nexts, exits, exits_next, keep, walks, total;
} sz_regex_arena_metal_t;

/** Rounds up to a multiple of 16 bytes, which every array of the round block starts at. */
STRINGZILLA_INLINE sz_size_t sz_regex_aligned_metal_(sz_size_t bytes) { return (bytes + 15) / 16 * 16; }

/** Cells of a row of either DFA, as @c sz_regex_stride_ counts them,
 *  or one block for a vocabulary with neither. */
STRINGZILLA_INLINE sz_size_t sz_regex_stride_metal_(sz_regex_engine_t const *engine) {
    if (!engine->forward_rows && !engine->reverse_rows) return engine->classes_count + 1;
    return sz_regex_stride_(engine->classes_count, engine->lookarounds_count);
}

/** Bytes of one automaton's column images, a count and a priming set for every column of a row;
 *  both sit at the head of the round block, which every call keeps. */
STRINGZILLA_INLINE sz_size_t sz_regex_images_bytes_metal_(sz_regex_engine_t const *engine) {
    return sz_regex_aligned_metal_((1 + sz_regex_priming_states_metal_k) * sz_regex_stride_metal_(engine) *
                                   sizeof(sz_u32_t));
}

/** Rounds of pointer doubling a chain across @p chunks chunks takes, an even count, which leaves
 *  the compositions in their first buffer. */
STRINGZILLA_INLINE sz_size_t sz_regex_rounds_count_metal_(sz_size_t chunks) {
    sz_size_t rounds = 0;
    for (sz_size_t reach = 1; reach <= chunks; reach *= sz_regex_round_hops_metal_k) ++rounds;
    return rounds + rounds % 2;
}

/** Bytes one PikeVM keeping @p slots_count capture slots takes of the walks,
 *  as the kernels carve it. */
STRINGZILLA_INLINE sz_size_t sz_regex_pike_bytes_metal_(sz_regex_engine_t const *engine, sz_size_t slots_count) {
    sz_size_t const states = engine->nfa_states_count, entered = engine->nfa_links_count + 1;
    return sz_regex_aligned_metal_(((states + entered + 2) * slots_count + 2 * (states + entered)) * sizeof(sz_u64_t) +
                                   (2 * states + entered) * sizeof(sz_u32_t));
}

/** Where the lookaround bodies begin among the NFA states,
 *  as @c sz_regex_bodies_begin_simt_ finds it. */
STRINGZILLA_INLINE sz_size_t sz_regex_bodies_begin_metal_(sz_regex_engine_t const *engine) {
    sz_size_t matches = 0;
    for (sz_size_t state = 0; state != engine->nfa_states_count; ++state)
        if (engine->nfa_states[state].kind == sz_regex_nfa_match_k && ++matches == engine->patterns_count)
            return state + 1;
    return engine->nfa_states_count;
}

/** Levels of nested lookarounds, each a pass behind and a pass ahead over the levels below;
 *  zero without any. */
STRINGZILLA_INLINE sz_u32_t sz_regex_levels_metal_(sz_regex_engine_t const *engine) {
    sz_u32_t levels = 0;
    for (sz_u32_t index = 0; index != engine->lookarounds_count; ++index)
        levels = sz_max_of_two(levels, engine->lookarounds[index].level);
    return levels;
}

/** Words of a lookaround pass's set: a bit per body state,
 *  then a bit per lookaround still to record. */
STRINGZILLA_INLINE sz_size_t sz_regex_looks_words_metal_(sz_regex_engine_t const *engine) {
    if (!engine->lookarounds_count) return 0;
    return (engine->nfa_states_count - sz_regex_bodies_begin_metal_(engine) + engine->lookarounds_count + 63) / 64;
}

/** The look bits of the lookarounds of @p level whose bodies look behind when @p behind, else
 *  ahead, and in @p negated those of them that are negated. */
STRINGZILLA_INLINE sz_u32_t sz_regex_members_metal_(sz_regex_engine_t const *engine, sz_u32_t level, sz_bool_t behind,
                                                    sz_u32_t *negated) {
    sz_u32_t members = 0;
    *negated = 0;
    for (sz_u32_t index = 0; index != engine->lookarounds_count; ++index) {
        sz_regex_lookaround_t const lookaround = engine->lookarounds[index];
        if (lookaround.level != level || (sz_bool_t)(lookaround.kind >= sz_regex_lookbehind_k) != behind) continue;
        members |= (sz_u32_t)sz_regex_look_lookaround_k << index;
        if (lookaround.kind & 1) *negated |= (sz_u32_t)sz_regex_look_lookaround_k << index;
    }
    return members;
}

/**
 *  @brief Lays the round block of @p engine out, after the column images.
 *
 *  The segment's own arrays come first; the walks follow, which hold one PikeVM, two position sets,
 *  or a PikeVM and two lookaround sets per thread of a dispatch, at least a threadgroup's worth.
 *  Each segment's edge, the entry of its first chunk, has a slot of its own, so a call spans at
 *  most as many segments as a segment has chunks; so does every set handed across it, per flow:
 *  positions, then each level behind and ahead. The lookaround table is a ring whose two halves
 *  each hold a segment.
 */
STRINGZILLA_INLINE sz_regex_arena_metal_t sz_regex_arena_metal_(sz_regex_engine_t const *engine) {
    sz_size_t const chunks = sz_regex_chunks_metal_k, slots = chunks * sz_regex_chunk_slots_metal_k;
    sz_size_t const looks_words = sz_regex_looks_words_metal_(engine);
    sz_size_t const set_words = sz_max_of_two((sz_size_t)engine->positions_words, looks_words);
    sz_size_t const sets = set_words ? chunks : 0, flows = 1 + 2 * (sz_size_t)sz_regex_levels_metal_(engine);
    sz_regex_arena_metal_t arena;
    arena.segment_slots = slots, arena.set_words = set_words;
    arena.pike_slots = 2 * sz_regex_groups_most_(engine) + 2;
    arena.pike_bytes = sz_regex_pike_bytes_metal_(engine, arena.pike_slots);
    arena.positions_bytes = sz_regex_aligned_metal_(2 * (sz_size_t)engine->positions_words * sizeof(sz_u64_t));
    arena.passes_bytes = looks_words ? arena.pike_bytes + 2 * looks_words * sizeof(sz_u64_t) : 0;
    arena.entries = 2 * sz_regex_images_bytes_metal_(engine);
    arena.edges = arena.entries + sz_regex_aligned_metal_(chunks * sizeof(sz_u32_t));
    arena.carry = arena.edges + sz_regex_aligned_metal_(chunks * sizeof(sz_u32_t));
    arena.counters = arena.carry + sz_regex_aligned_metal_(sz_regex_carry_bytes_metal_k);
    arena.flags = arena.counters + 4 * sizeof(sz_u64_t);
    arena.tile_sums = arena.flags +
                      sz_regex_aligned_metal_((sz_regex_rounds_count_metal_(chunks) + 2) * sizeof(sz_u32_t));
    arena.chunk_offsets = arena.tile_sums + sz_regex_scan_tiles_metal_k * sizeof(sz_u64_t);
    arena.chunk_kept = arena.chunk_offsets + sz_regex_aligned_metal_((chunks + 1) * sizeof(sz_u64_t));
    arena.chunk_sums = arena.chunk_kept + sz_regex_aligned_metal_((chunks + 1) * sizeof(sz_u64_t));
    arena.chunk_covers = arena.chunk_sums + sz_regex_aligned_metal_((chunks + 1) * sizeof(sz_u64_t));
    arena.chunk_haystacks = arena.chunk_covers + sz_regex_aligned_metal_((chunks + 1) * sizeof(sz_u64_t));
    arena.frontiers = arena.chunk_haystacks + sz_regex_aligned_metal_(chunks * sizeof(sz_u64_t));
    arena.compositions = arena.frontiers + sz_regex_aligned_metal_(chunks * sz_regex_frontier_bytes_metal_k);
    arena.compositions_next = arena.compositions + sz_regex_aligned_metal_(chunks * sz_regex_composition_bytes_metal_k);
    arena.handoffs = arena.compositions_next + sz_regex_aligned_metal_(chunks * sz_regex_composition_bytes_metal_k);
    arena.handoff_edges = arena.handoffs + sets * set_words * sizeof(sz_u64_t);
    arena.undecided = arena.handoff_edges +
                      sz_regex_aligned_metal_(set_words ? (chunks + 1) * flows * set_words * sizeof(sz_u64_t) : 0);
    arena.affines = arena.undecided + sz_regex_aligned_metal_(sets * sz_regex_undecided_bytes_metal_k);
    arena.affines_next = arena.affines + sets * sz_regex_affine_bytes_metal_k;
    arena.looks = arena.affines_next + sets * sz_regex_affine_bytes_metal_k;
    arena.starts = arena.looks + (engine->lookarounds_count ? 2 * slots * sizeof(sz_u32_t) : 0);
    arena.checkpoints = arena.starts + slots / 8;
    arena.slots = arena.checkpoints + (slots / sz_regex_checkpoint_slots_metal_k + 1) * 16;
    arena.results = arena.slots + sz_regex_aligned_metal_(slots * sizeof(sz_u32_t));
    arena.links = arena.results + slots * 16;
    arena.links_next = arena.links + slots * 16;
    // The chain's arrays live in the links' second buffer, dead once links settle into the first.
    arena.nexts = arena.links_next;
    arena.exits = arena.nexts + sz_regex_aligned_metal_(slots * sizeof(sz_u32_t));
    arena.exits_next = arena.exits + sz_regex_aligned_metal_(slots * sizeof(sz_u32_t));
    arena.keep = arena.exits_next + sz_regex_aligned_metal_(slots * sizeof(sz_u32_t));
    arena.links_list = arena.links_next + (slots + 1) * 16;
    arena.walks = arena.links_list + sz_regex_aligned_metal_(slots * sizeof(sz_u32_t));
    arena.total = arena.walks + sz_max_of_two(slots * sz_regex_walk_slot_bytes_metal_k,
                                              sz_regex_threads_per_group_metal_k *
                                                  sz_max_of_two(sz_max_of_two(arena.pike_bytes, arena.positions_bytes),
                                                                arena.passes_bytes));
    return arena;
}

/** Writes each column's image into @p images: the distinct live states any state of @p rows steps
 *  to on it, or one more than a priming set holds when they do not fit; an edge column of any
 *  block has none. */
STRINGZILLA_INLINE void sz_regex_images_metal_(sz_regex_engine_t const *engine, sz_u32_t const *rows,
                                               sz_size_t states_count, sz_u32_t *images) {
    sz_size_t const stride = sz_regex_stride_metal_(engine);
    for (sz_size_t column = 0; column != stride; ++column) {
        sz_u32_t *const states = images + stride + column * sz_regex_priming_states_metal_k;
        sz_u32_t count = 0;
        if (column % (engine->classes_count + 1) != engine->classes_count)
            for (sz_size_t state = 1; state != states_count && count <= sz_regex_priming_states_metal_k; ++state) {
                sz_u32_t const next = rows[state * stride + column];
                sz_u32_t index = 0;
                while (index != count && states[index] != next) ++index;
                if (!next || index != count) continue;
                if (count < sz_regex_priming_states_metal_k) states[count] = next;
                ++count;
            }
        images[column] = count;
    }
}

/** Threadgroups of the tier's width a grid-strided dispatch over @p items takes. */
STRINGZILLA_INLINE sz_metal_size_t sz_regex_groups_metal_(sz_size_t items) {
    sz_size_t const groups = sz_size_divide_round_up(items, sz_regex_threads_per_group_metal_k);
    sz_metal_size_t const grid = {groups ? sz_min_of_two(groups, (sz_size_t)sz_regex_groups_max_metal_k) : 1, 1, 1};
    return grid;
}

/** Encodes kernel @p name over @p groups of the tier's width, with the call's @p buffers and room
 *  for the walks' staged class map, blocks and rows. */
STRINGZILLA_INLINE sz_status_t sz_regex_encode_metal_(sz_metal_call_t *call, sz_metal_bound_t const *buffers,
                                                      char const *name, sz_regex_arguments_metal_t const *arguments,
                                                      sz_metal_size_t groups) {
    sz_metal_size_t const threads = {sz_regex_threads_per_group_metal_k, 1, 1};
    sz_metal_threadgroup_memory_(
        call->encoder, sz_regex_staged_bytes_metal_k + sz_size_divide_round_up(arguments->staged_cells, 4) * 16, 0);
    return sz_metal_encode_(call, sz_regex_source_metal_, name, buffers, sz_regex_buffers_metal_k, arguments,
                            sizeof(*arguments), groups, threads);
}

/** Encodes kernel @p name over just enough threadgroups for each of the @c walks_count threads its
 *  record names to own a share of the walks. */
STRINGZILLA_INLINE sz_status_t sz_regex_encode_walks_metal_(sz_metal_call_t *call, sz_metal_bound_t const *buffers,
                                                            char const *name,
                                                            sz_regex_arguments_metal_t const *arguments) {
    sz_metal_size_t const groups = {sz_size_divide_round_up(arguments->walks_count, sz_regex_threads_per_group_metal_k),
                                    1, 1};
    return sz_regex_encode_metal_(call, buffers, name, arguments, groups);
}

/** Encodes a single-thread kernel, the shape the carry between segments takes. */
STRINGZILLA_INLINE sz_status_t sz_regex_encode_one_metal_(sz_metal_call_t *call, sz_metal_bound_t const *buffers,
                                                          char const *name,
                                                          sz_regex_arguments_metal_t const *arguments) {
    sz_metal_size_t const one = {1, 1, 1};
    return sz_metal_encode_(call, sz_regex_source_metal_, name, buffers, sz_regex_buffers_metal_k, arguments,
                            sizeof(*arguments), one, one);
}

/** Zeroes @p bytes of the round block, a whole number of words, from @p offset. */
STRINGZILLA_INLINE sz_status_t sz_regex_clear_metal_(sz_metal_call_t *call, sz_metal_bound_t const *buffers,
                                                     sz_regex_arguments_metal_t *arguments, sz_size_t offset,
                                                     sz_size_t bytes) {
    arguments->target = arguments->scratch_host + offset, arguments->clear_words = bytes / 8;
    return sz_regex_encode_metal_(call, buffers, "sz_regex_clear_metal_kernel_", arguments,
                                  sz_regex_groups_metal_(bytes / 8));
}

/** Turns the @p count 64-bit values at @p offset of the round block into their own exclusive prefix
 *  sum, in three dispatches: each tile reduced, the tile totals carried on one threadgroup, each
 *  tile's base added back. */
STRINGZILLA_INLINE sz_status_t sz_regex_scan_metal_(sz_metal_call_t *call, sz_metal_bound_t const *buffers,
                                                    sz_regex_arguments_metal_t *arguments, sz_size_t offset,
                                                    sz_size_t count) {
    sz_size_t const tiles = sz_min_of_two(sz_size_divide_round_up(count, sz_regex_threads_per_group_metal_k),
                                          (sz_size_t)sz_regex_scan_tiles_metal_k);
    sz_metal_size_t const tile_groups = {tiles, 1, 1}, one = {1, 1, 1};
    sz_status_t status;
    arguments->target = arguments->scratch_host + offset, arguments->scan_tiles = tiles, arguments->scan_count = count;
    status = sz_regex_encode_metal_(call, buffers, "sz_regex_scan_reduce_metal_kernel_", arguments, tile_groups);
    if (status == sz_success_k)
        status = sz_regex_encode_metal_(call, buffers, "sz_regex_scan_carry_metal_kernel_", arguments, one);
    if (status == sz_success_k)
        status = sz_regex_encode_metal_(call, buffers, "sz_regex_scan_apply_metal_kernel_", arguments, tile_groups);
    return status;
}

/** Binds what every verb reads, the engine's two blocks and the haystacks' tape, the first thing
 *  every verb checks. */
STRINGZILLA_INLINE sz_status_t sz_regex_resident_metal_(sz_regex_engine_t const *engine, sz_metal_call_t *call,
                                                        sz_strings_view_t const *haystacks, sz_metal_bound_t *buffers) {
    if (!sz_metal_resolve_call_(call, engine->memory, engine->memory_bytes, buffers + sz_regex_memory_buffer_metal_k) ||
        !sz_metal_resolve_call_(call, engine->scratch, engine->scratch_bytes,
                                buffers + sz_regex_scratch_buffer_metal_k))
        return sz_device_memory_mismatch_k;
    return sz_metal_tape_call_(call, haystacks, buffers + sz_regex_haystacks_buffer_metal_k,
                               buffers + sz_regex_haystacks_offsets_buffer_metal_k);
}

/** The slot haystack @p index starts at, one per byte and one per haystack before it, read on the
 *  host; past the last haystack, the slots the whole tape spans. */
STRINGZILLA_INLINE sz_size_t sz_regex_first_slot_metal_(sz_strings_view_t const *haystacks, sz_size_t index) {
    if (haystacks->layout == sz_strings_u64_tape_k)
        return (sz_size_t)(haystacks->data.u64_tape.offsets[index] - haystacks->data.u64_tape.offsets[0]) + index;
    return (sz_size_t)(haystacks->data.u32_tape.offsets[index] - haystacks->data.u32_tape.offsets[0]) + index;
}

/** Zeroes @p arguments and fills what every dispatch of a call reads: both blocks' host addresses,
 *  the engine's arrays and scalars, the round block's layout, and what @p verb writes. */
STRINGZILLA_INLINE void sz_regex_arguments_metal_(sz_regex_engine_t const *engine, sz_strings_view_t const *haystacks,
                                                  sz_regex_verb_metal_t verb, sz_regex_arguments_metal_t *arguments) {
    sz_regex_arena_metal_t const arena = sz_regex_arena_metal_(engine);
    sz_u64_t const host = (sz_u64_t)engine->scratch;
    memset(arguments, 0, sizeof(*arguments));
    arguments->engine_host = (sz_u64_t)engine->memory, arguments->scratch_host = host;
    arguments->byte_to_class = (sz_u64_t)engine->byte_to_class;
    arguments->reverse_rows = (sz_u64_t)engine->reverse_rows, arguments->forward_rows = (sz_u64_t)engine->forward_rows;
    arguments->forward_patterns = (sz_u64_t)engine->forward_patterns;
    arguments->nfa_states = (sz_u64_t)engine->nfa_states, arguments->nfa_links = (sz_u64_t)engine->nfa_links;
    arguments->patterns_starts = (sz_u64_t)engine->patterns_starts;
    arguments->patterns_groups = (sz_u64_t)engine->patterns_groups;
    arguments->names = (sz_u64_t)engine->groups_names.data.u32_tape.data;
    arguments->names_offsets = (sz_u64_t)engine->groups_names.data.u32_tape.offsets;
    arguments->lookarounds = (sz_u64_t)engine->lookarounds;
    arguments->positions_reach = (sz_u64_t)engine->positions_reach;
    arguments->positions_shifts = (sz_u64_t)engine->positions_shifts;
    arguments->positions_groups = (sz_u64_t)engine->positions_groups;
    arguments->positions_all = (sz_u64_t)engine->positions_all;
    arguments->positions_initial = (sz_u64_t)engine->positions_initial;
    arguments->positions_ascii = (sz_u64_t)engine->positions_ascii;
    arguments->positions_ranges = (sz_u64_t)engine->positions_ranges;
    arguments->reverse_images = host, arguments->forward_images = host + sz_regex_images_bytes_metal_(engine);
    arguments->entries = host + arena.entries, arguments->edges = host + arena.edges;
    arguments->carry = host + arena.carry, arguments->counters = host + arena.counters;
    arguments->flags = host + arena.flags, arguments->tile_sums = host + arena.tile_sums;
    arguments->chunk_offsets = host + arena.chunk_offsets, arguments->chunk_kept = host + arena.chunk_kept;
    arguments->chunk_sums = host + arena.chunk_sums, arguments->chunk_covers = host + arena.chunk_covers;
    arguments->chunk_haystacks = host + arena.chunk_haystacks;
    arguments->frontiers = host + arena.frontiers, arguments->compositions = host + arena.compositions;
    arguments->compositions_next = host + arena.compositions_next;
    arguments->handoffs = host + arena.handoffs, arguments->handoff_edges = host + arena.handoff_edges;
    arguments->undecided = host + arena.undecided, arguments->affines = host + arena.affines;
    arguments->affines_next = host + arena.affines_next, arguments->looks = host + arena.looks;
    arguments->set_words = arena.set_words, arguments->bodies_begin = sz_regex_bodies_begin_metal_(engine);
    arguments->bodies_count = engine->nfa_states_count - arguments->bodies_begin;
    arguments->starts = host + arena.starts, arguments->checkpoints = host + arena.checkpoints;
    arguments->slots = host + arena.slots, arguments->results = host + arena.results;
    arguments->links = host + arena.links, arguments->links_next = host + arena.links_next;
    arguments->links_list = host + arena.links_list, arguments->nexts = host + arena.nexts;
    arguments->exits = host + arena.exits, arguments->exits_next = host + arena.exits_next;
    arguments->keep = host + arena.keep, arguments->walks = host + arena.walks;
    arguments->looks_mask = 2 * arena.segment_slots - 1;
    // Groups of paused walks take the checkpoints' first half and their doubling the second.
    arguments->groups_capacity = (arena.segment_slots / sz_regex_checkpoint_slots_metal_k + 1) / 2;
    arguments->haystacks_count = haystacks->count, arguments->haystacks_layout = (sz_u64_t)haystacks->layout;
    arguments->slots_count = sz_regex_first_slot_metal_(haystacks, haystacks->count);
    arguments->verb = (sz_u64_t)verb;
    arguments->pattern_index = STRINGZILLA_SIZE_MAX, arguments->pike_slots = arena.pike_slots;
    arguments->chunk_width = sz_regex_chunk_slots_metal_k;
    arguments->chunk_shift = sz_size_log2i_nonzero(sz_regex_chunk_slots_metal_k);
    for (sz_size_t kind = 0; kind != 4; ++kind)
        arguments->forward_starts[kind] = engine->forward_starts[kind],
        arguments->reverse_starts[kind] = engine->reverse_starts[kind];
    arguments->forward_accepts_begin = engine->forward_accepts_begin;
    arguments->reverse_accepts_begin = engine->reverse_accepts_begin;
    arguments->classes_count = engine->classes_count, arguments->case_sensitivity = engine->case_sensitivity;
    arguments->stride = (sz_u32_t)sz_regex_stride_metal_(engine);
    arguments->forward_cells = engine->forward_states_count * arguments->stride;
    arguments->reverse_cells = engine->reverse_states_count * arguments->stride;
    arguments->staged_cells = (sz_u32_t)sz_min_of_two(sz_max_of_two(arguments->forward_cells, arguments->reverse_cells),
                                                      (sz_u32_t)sz_regex_staged_cells_metal_k);
    arguments->nfa_states_count = engine->nfa_states_count, arguments->nfa_links_count = engine->nfa_links_count;
    arguments->nfa_start = engine->nfa_start, arguments->positions_words = engine->positions_words;
    arguments->positions_shifts_count = engine->positions_shifts_count;
    arguments->positions_groups_count = engine->positions_groups_count;
    arguments->positions_ranges_count = engine->positions_ranges_count;
    arguments->patterns_count = engine->patterns_count, arguments->lookarounds_count = engine->lookarounds_count;
}

/** Points @p arguments at segment @p segment of its corpus, cut into segments of @p slots slots. */
STRINGZILLA_INLINE void sz_regex_segment_metal_(sz_regex_arguments_metal_t *arguments, sz_size_t segment,
                                                sz_size_t slots) {
    arguments->segment = segment, arguments->segment_begin = segment * slots;
    arguments->segment_end = sz_min_of_two(arguments->segment_begin + slots, arguments->slots_count);
    arguments->chunks_count = sz_size_divide_round_up(arguments->segment_end - arguments->segment_begin,
                                                      sz_regex_chunk_slots_metal_k);
}

/**
 *  @brief Encodes one flow of sets over the segment @p arguments points at.
 *
 *  It encodes the bounds of what each chunk hands on, the walks of the pieces where they stayed
 *  wide, their affine maps and the doubling that answers them, then, when @p recording, the walk
 *  that records where the lookarounds of the pass hold.
 *
 *  @param[in] level The pass's level, for a lookaround pass.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_flow_metal_(sz_regex_engine_t const *engine, sz_metal_call_t *call,
                                                    sz_metal_bound_t const *buffers,
                                                    sz_regex_arguments_metal_t *arguments,
                                                    sz_regex_arena_metal_t const *arena, sz_regex_flow_metal_t flow,
                                                    sz_u32_t level, sz_bool_t recording) {
    sz_bool_t const positions = (sz_bool_t)(flow == sz_regex_flow_positions_metal_k);
    sz_size_t const set = positions ? 0 : 2 * level - (flow == sz_regex_flow_behind_metal_k);
    sz_status_t status;
    arguments->flow = flow, arguments->negated = 0;
    arguments->members = positions
                             ? 0
                             : sz_regex_members_metal_(engine, level, (sz_bool_t)(flow == sz_regex_flow_behind_metal_k),
                                                       &arguments->negated);
    arguments->handoff_edges = arguments->scratch_host + arena->handoff_edges +
                               set * (sz_regex_chunks_metal_k + 1) * arena->set_words * sizeof(sz_u64_t);
    arguments->walk_bytes = positions ? arena->positions_bytes : arena->passes_bytes;
    arguments->walk_sets = positions ? 0 : arena->pike_bytes;
    arguments->walks_count = sz_min_of_two((arena->total - arena->walks) / arguments->walk_bytes,
                                           (sz_size_t)sz_regex_chunks_metal_k);
    status = sz_regex_encode_walks_metal_(call, buffers, "sz_regex_bound_metal_kernel_", arguments);
    if (status == sz_success_k)
        status = sz_regex_encode_walks_metal_(call, buffers, "sz_regex_flow_pieces_metal_kernel_", arguments);
    if (status == sz_success_k)
        status = sz_regex_encode_walks_metal_(call, buffers, "sz_regex_affine_metal_kernel_", arguments);
    for (sz_size_t round = 0; status == sz_success_k && round != arguments->rounds; ++round) {
        arguments->round = round;
        status = sz_regex_encode_metal_(call, buffers, "sz_regex_affine_double_metal_kernel_", arguments,
                                        sz_regex_groups_metal_(arguments->chunks_count));
    }
    if (status == sz_success_k && recording)
        status = sz_regex_encode_walks_metal_(call, buffers, "sz_regex_looks_metal_kernel_", arguments);
    return status;
}

/** Encodes filling segment @p segment 's half of the lookaround ring, the window then starting at
 *  segment @p first, with every level's passes up to @p levels, each level over the ones below
 *  it: the lookbehinds left to right and the lookaheads right to left, handed the sets kept at
 *  the segment's edges. */
STRINGZILLA_INLINE sz_status_t sz_regex_looks_metal_(sz_regex_engine_t const *engine, sz_metal_call_t *call,
                                                     sz_metal_bound_t const *buffers,
                                                     sz_regex_arguments_metal_t *arguments,
                                                     sz_regex_arena_metal_t const *arena, sz_size_t segment,
                                                     sz_size_t first, sz_u32_t levels) {
    sz_size_t const slots = arena->segment_slots;
    sz_size_t const entries = sz_min_of_two(slots, arguments->slots_count - segment * slots);
    sz_u32_t negated;
    sz_status_t status;
    arguments->looks_begin = first * slots, arguments->looks_end = arguments->looks_begin + 2 * slots;
    // Clearing takes whole words, an odd entry count clearing one more of the half's own.
    status = sz_regex_clear_metal_(call, buffers, arguments, arena->looks + segment % 2 * slots * sizeof(sz_u32_t),
                                   (entries + 1) / 2 * sizeof(sz_u64_t));
    sz_regex_segment_metal_(arguments, segment, slots);
    for (sz_u32_t level = 1; status == sz_success_k && level <= levels; ++level) {
        if (sz_regex_members_metal_(engine, level, sz_true_k, &negated))
            status = sz_regex_flow_metal_(engine, call, buffers, arguments, arena, sz_regex_flow_behind_metal_k, level,
                                          sz_true_k);
        if (status == sz_success_k && sz_regex_members_metal_(engine, level, sz_false_k, &negated))
            status = sz_regex_flow_metal_(engine, call, buffers, arguments, arena, sz_regex_flow_ahead_metal_k, level,
                                          sz_true_k);
    }
    return status;
}

/** Encodes priming the start-finding walk over the segment @p arguments points at: a DFA by its
 *  frontiers and the walks of its wide pieces, positions by their flow; either keeps what the
 *  segment's first chunk is handed for the segment before it. */
STRINGZILLA_INLINE sz_status_t sz_regex_prime_segment_metal_(sz_regex_engine_t const *engine, sz_metal_call_t *call,
                                                             sz_metal_bound_t const *buffers,
                                                             sz_regex_arguments_metal_t *arguments,
                                                             sz_regex_arena_metal_t const *arena) {
    sz_metal_size_t const groups = sz_regex_groups_metal_(arguments->chunks_count);
    sz_status_t status;
    if (!engine->reverse_rows)
        return sz_regex_flow_metal_(engine, call, buffers, arguments, arena, sz_regex_flow_positions_metal_k, 0,
                                    sz_false_k);
    status = sz_regex_encode_metal_(call, buffers, "sz_regex_prime_metal_kernel_", arguments, groups);
    if (status == sz_success_k)
        status = sz_regex_encode_metal_(call, buffers, "sz_regex_prime_pieces_metal_kernel_", arguments, groups);
    if (status == sz_success_k)
        status = sz_regex_encode_metal_(call, buffers, "sz_regex_compose_metal_kernel_", arguments, groups);
    for (sz_size_t round = 0; status == sz_success_k && round != arguments->rounds; ++round) {
        arguments->round = round;
        status = sz_regex_encode_metal_(call, buffers, "sz_regex_compose_double_metal_kernel_", arguments, groups);
    }
    return status;
}

/** Whether the haystack holding the last slot of the segment @p arguments points at runs past the
 *  end of the lookaround table's window, the one haystack whose walks from the segment can reach
 *  it, found on the host. */
STRINGZILLA_INLINE sz_bool_t sz_regex_runs_past_metal_(sz_strings_view_t const *haystacks,
                                                       sz_regex_arguments_metal_t const *arguments) {
    sz_size_t low = 0, high = haystacks->count;
    while (high - low > 1) {
        sz_size_t const middle = low + (high - low) / 2;
        if (sz_regex_first_slot_metal_(haystacks, middle) < arguments->segment_end) low = middle;
        else high = middle;
    }
    return (sz_bool_t)(sz_regex_first_slot_metal_(haystacks, low + 1) > arguments->looks_end);
}

/** Commits the open @p call, waits for its stream and opens the call again, so the host reads what
 *  the kernels wrote to the unified round block. */
STRINGZILLA_INLINE sz_status_t sz_regex_synchronize_metal_(sz_metal_call_t *call, sz_stream_t stream) {
    sz_status_t status = sz_metal_commit_(call, sz_success_k);
    if (status == sz_success_k) status = sz_stream_synchronize_metal_(stream);
    if (status == sz_success_k) status = sz_device_enter_metal_(stream, call);
    return status;
}

/**
 *  @brief Settles the links of the segment @p arguments points at that paused at the end of the
 *      lookaround table's window, the one place a regex verb joins its stream.
 *
 *  While walks are paused, the host groups them by state, refills the ring two segments on and
 *  resumes each group as one walk, until none pauses again. The groups then settle by pointer
 *  doubling, each paused link takes its group's match, and the ring is refilled for the segment and
 *  its neighbour. Every join commits @p call, synchronizes @p stream and opens the call again; a
 *  call needing more groups than the checkpoints hold fails with @c sz_unexpected_dimensions_k.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_slides_metal_(sz_regex_engine_t const *engine, sz_metal_call_t *call,
                                                      sz_metal_bound_t const *buffers,
                                                      sz_regex_arguments_metal_t *arguments,
                                                      sz_regex_arena_metal_t const *arena, sz_u32_t levels,
                                                      sz_stream_t stream) {
    sz_size_t const slots = arena->segment_slots, segment = arguments->segment;
    sz_size_t const segments = sz_size_divide_round_up(arguments->slots_count, slots);
    sz_u32_t const *const counters = (sz_u32_t const *)((char const *)engine->scratch + arena->counters);
    sz_size_t windows = 0;
    sz_status_t status;
    for (;;) {
        status = sz_regex_synchronize_metal_(call, stream);
        if (status != sz_success_k) return status;
        if (counters[sz_regex_refused_counter_metal_k]) return sz_unexpected_dimensions_k;
        if (!counters[sz_regex_paused_counter_metal_k]) break;
        sz_metal_size_t const listed = sz_regex_groups_metal_(counters[sz_regex_handed_counter_metal_k]);
        sz_metal_size_t const paused = sz_regex_groups_metal_(counters[sz_regex_paused_counter_metal_k]);
        sz_size_t const first = segment + 2 * (windows + 1);
        arguments->groups_first = counters[sz_regex_groups_counter_metal_k];
        status = sz_regex_clear_metal_(call, buffers, arguments, arena->frontiers,
                                       (engine->forward_states_count + 1) / 2 * sizeof(sz_u64_t));
        if (status == sz_success_k)
            status = sz_regex_encode_metal_(call, buffers, "sz_regex_group_metal_kernel_", arguments, listed);
        if (status == sz_success_k)
            status = sz_regex_encode_metal_(call, buffers, "sz_regex_point_metal_kernel_", arguments, listed);
        // The walks handed on and paused are counted afresh for the window the groups resume in.
        if (status == sz_success_k)
            status = sz_regex_clear_metal_(call, buffers, arguments, arena->counters + 2 * sizeof(sz_u32_t),
                                           sizeof(sz_u64_t));
        for (sz_size_t filled = first; status == sz_success_k && filled != first + 2 && filled < segments; ++filled)
            status = sz_regex_looks_metal_(engine, call, buffers, arguments, arena, filled, first, levels);
        sz_regex_segment_metal_(arguments, segment, slots);
        if (status == sz_success_k)
            status = sz_regex_encode_metal_(call, buffers, "sz_regex_resume_metal_kernel_", arguments, paused);
        if (status == sz_success_k)
            status = sz_regex_encode_metal_(call, buffers, "sz_regex_continue_metal_kernel_", arguments,
                                            sz_regex_groups_metal_(arguments->chunks_count));
        if (status != sz_success_k) return status;
        ++windows;
    }
    if (!windows) return status;

    // Every group's chain spans at most the windows walked, which the doubling's rounds outreach.
    for (sz_size_t round = 0; status == sz_success_k && round != sz_regex_rounds_count_metal_(windows); ++round) {
        arguments->round = round;
        status = sz_regex_encode_metal_(call, buffers, "sz_regex_groups_double_metal_kernel_", arguments,
                                        sz_regex_groups_metal_(counters[sz_regex_groups_counter_metal_k]));
    }
    if (status == sz_success_k)
        status = sz_regex_encode_metal_(call, buffers, "sz_regex_take_metal_kernel_", arguments,
                                        sz_regex_groups_metal_(counters[sz_regex_links_counter_metal_k]));
    for (sz_size_t filled = segment; status == sz_success_k && filled != segment + 2 && filled < segments; ++filled)
        status = sz_regex_looks_metal_(engine, call, buffers, arguments, arena, filled, segment, levels);
    sz_regex_segment_metal_(arguments, segment, slots);
    return status;
}

/**
 *  @brief Encodes the whole call into the open @p call: the passes that hand lookaround sets across
 *      segments, the priming segments right to left, then the walking ones left to right, each
 *      storing what the verb of @p arguments asks.
 *
 *  A start-finding walk primes each chunk from the one right of it, so ahead of the walk the
 *  segments past the first are primed right to left once, leaving each segment's edge for the one
 *  before it. Lookaround passes run the same way per level, lookbehinds left to right, ahead of
 *  everything that reads their table, and every sweep fills each segment's half of the ring once.
 *  End walks without links take one candidate per thread, listed first.
 *
 *  A segment whose last haystack runs past the ring's window may have walks reading the table past
 *  it, which only there joins the stream: DFA links pause and slide on, and the PikeVM, the folded
 *  walks of later runes and the walks of groups fail the call with @c sz_unexpected_dimensions_k.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_rounds_metal_(sz_regex_engine_t const *engine, sz_metal_call_t *call,
                                                      sz_metal_bound_t const *buffers,
                                                      sz_strings_view_t const *haystacks,
                                                      sz_regex_arguments_metal_t *arguments, sz_stream_t stream) {
    sz_regex_arena_metal_t const arena = sz_regex_arena_metal_(engine);
    sz_size_t const slots = arena.segment_slots, slots_count = arguments->slots_count;
    sz_size_t const segments = sz_size_divide_round_up(slots_count, slots);
    sz_bool_t const folded = (sz_bool_t)(engine->case_sensitivity != sz_cased_k);
    sz_u32_t const levels = sz_regex_levels_metal_(engine);
    sz_size_t const walks_bytes = arena.total - arena.walks;
    sz_size_t const pike_walks = sz_min_of_two(walks_bytes / arena.pike_bytes, (sz_size_t)sz_regex_chunks_metal_k);
    sz_size_t const positions_walks = arena.positions_bytes ? sz_min_of_two(walks_bytes / arena.positions_bytes,
                                                                            (sz_size_t)sz_regex_chunks_metal_k)
                                                            : sz_regex_chunks_metal_k;
    sz_size_t const replacing = arguments->verb == sz_regex_replacing_metal_k;
    sz_size_t const extracting = arguments->verb == sz_regex_extracting_metal_k;
    // Walks that cannot pause past the ring's window run after the slides,
    // so their refusals are read at the end.
    sz_bool_t const refusing = (sz_bool_t)(!engine->forward_rows || folded || replacing || extracting);
    sz_u32_t const *const counters = (sz_u32_t const *)((char const *)engine->scratch + arena.counters);
    sz_u32_t negated;
    sz_status_t status = sz_success_k;
    if (segments > sz_regex_chunks_metal_k) return sz_unexpected_dimensions_k;
    arguments->rounds = sz_regex_rounds_count_metal_(
        sz_size_divide_round_up(sz_min_of_two(slots_count, slots), sz_regex_chunk_slots_metal_k));

    // Each level's passes hand sets across segments over lower levels' tables, so they sweep first.
    for (sz_u32_t level = 1; status == sz_success_k && segments > 1 && level <= levels; ++level) {
        if (sz_regex_members_metal_(engine, level, sz_true_k, &negated))
            for (sz_size_t segment = 0; status == sz_success_k && segment != segments; ++segment) {
                status = sz_regex_looks_metal_(engine, call, buffers, arguments, &arena, segment, segment, level - 1);
                if (status == sz_success_k)
                    status = sz_regex_flow_metal_(engine, call, buffers, arguments, &arena,
                                                  sz_regex_flow_behind_metal_k, level, sz_false_k);
            }
        if (sz_regex_members_metal_(engine, level, sz_false_k, &negated))
            for (sz_size_t segment = segments; status == sz_success_k && segment-- != 0;) {
                status = sz_regex_looks_metal_(engine, call, buffers, arguments, &arena, segment, segment, level - 1);
                if (status == sz_success_k)
                    status = sz_regex_flow_metal_(engine, call, buffers, arguments, &arena, sz_regex_flow_ahead_metal_k,
                                                  level, sz_false_k);
            }
    }

    // Priming reads each segment's right neighbour, so the segments past the first prime right to
    // left first, each filling its own half of the ring beside its neighbour's.
    for (sz_size_t segment = segments; status == sz_success_k && segment-- > 1;) {
        if (levels) status = sz_regex_looks_metal_(engine, call, buffers, arguments, &arena, segment, segment, levels);
        sz_regex_segment_metal_(arguments, segment, slots);
        if (status == sz_success_k) status = sz_regex_prime_segment_metal_(engine, call, buffers, arguments, &arena);
    }
    for (sz_size_t segment = 0; status == sz_success_k && segment != segments; ++segment) {
        sz_size_t chunks;
        sz_bool_t past;
        // The walks read the table as far as the next segment, whose half alone is new; the priming
        // sweep ended filling the second segment's half, so the first segment fills only its own.
        if (levels && segment == 0)
            status = sz_regex_looks_metal_(engine, call, buffers, arguments, &arena, segment, segment, levels);
        else if (levels && segment + 1 < segments)
            status = sz_regex_looks_metal_(engine, call, buffers, arguments, &arena, segment + 1, segment, levels);
        sz_regex_segment_metal_(arguments, segment, slots);
        chunks = arguments->chunks_count;
        past = (sz_bool_t)(levels && sz_regex_runs_past_metal_(haystacks, arguments));
        if (status == sz_success_k && segment == 0)
            status = sz_regex_clear_metal_(call, buffers, arguments, arena.carry, sz_regex_carry_bytes_metal_k);
        if (status == sz_success_k) status = sz_regex_prime_segment_metal_(engine, call, buffers, arguments, &arena);

        // The counters and the doubling flags, which a segment reads before writing.
        if (status == sz_success_k)
            status = sz_regex_clear_metal_(call, buffers, arguments, arena.counters, arena.tile_sums - arena.counters);
        if (status == sz_success_k && !engine->reverse_rows) {
            arguments->walks_count = positions_walks, arguments->walk_bytes = arena.positions_bytes;
            status = sz_regex_encode_walks_metal_(
                call, buffers,
                folded ? "sz_regex_mark_positions_folded_metal_kernel_" : "sz_regex_mark_positions_metal_kernel_",
                arguments);
        }
        else if (status == sz_success_k)
            status = sz_regex_encode_metal_(
                call, buffers, folded ? "sz_regex_mark_folded_metal_kernel_" : "sz_regex_mark_metal_kernel_", arguments,
                sz_regex_groups_metal_(chunks));
        if (status == sz_success_k)
            status = sz_regex_scan_metal_(call, buffers, arguments, arena.chunk_offsets, chunks + 1);
        if (status == sz_success_k && !engine->forward_rows) {
            status = sz_regex_encode_metal_(call, buffers, "sz_regex_list_metal_kernel_", arguments,
                                            sz_regex_groups_metal_(chunks));
            arguments->walks_count = pike_walks, arguments->walk_bytes = arena.pike_bytes;
            if (status == sz_success_k)
                status = sz_regex_encode_walks_metal_(call, buffers, "sz_regex_ends_pike_metal_kernel_", arguments);
        }
        else if (status == sz_success_k) {
            status = sz_regex_encode_metal_(call, buffers, "sz_regex_ends_metal_kernel_", arguments,
                                            sz_regex_groups_metal_(chunks));
            if (status == sz_success_k)
                status = sz_regex_encode_metal_(call, buffers, "sz_regex_links_metal_kernel_", arguments,
                                                sz_regex_groups_metal_(chunks));
            if (status == sz_success_k)
                status = sz_regex_encode_metal_(call, buffers, "sz_regex_continue_metal_kernel_", arguments,
                                                sz_regex_groups_metal_(chunks));
            if (status == sz_success_k && past)
                status = sz_regex_slides_metal_(engine, call, buffers, arguments, &arena, levels, stream);
            for (sz_size_t round = 0; status == sz_success_k && round != arguments->rounds; ++round) {
                arguments->round = round;
                status = sz_regex_encode_metal_(call, buffers, "sz_regex_links_double_metal_kernel_", arguments,
                                                sz_regex_groups_metal_(chunks));
            }
            if (status == sz_success_k && folded)
                status = sz_regex_encode_metal_(call, buffers, "sz_regex_retry_folded_metal_kernel_", arguments,
                                                sz_regex_groups_metal_(slots));
        }
        if (status == sz_success_k)
            status = sz_regex_encode_metal_(call, buffers, "sz_regex_chain_metal_kernel_", arguments,
                                            sz_regex_groups_metal_(chunks));
        for (sz_size_t round = 0; status == sz_success_k && round != arguments->rounds; ++round) {
            arguments->round = round;
            status = sz_regex_encode_metal_(call, buffers, "sz_regex_double_metal_kernel_", arguments,
                                            sz_regex_groups_metal_(slots));
        }
        arguments->storing = 0;
        if (status == sz_success_k)
            status = sz_regex_encode_metal_(call, buffers, "sz_regex_sweep_metal_kernel_", arguments,
                                            sz_regex_groups_metal_(chunks));
        if (status == sz_success_k)
            status = sz_regex_scan_metal_(call, buffers, arguments, arena.chunk_kept, chunks + 1);

        // A rewrite or an extraction runs on the walks between the counting sweep and the storing
        // one, which moves the carry's last match on.
        arguments->walks_count = pike_walks, arguments->walk_bytes = arena.pike_bytes;
        if (status == sz_success_k && replacing) {
            status = sz_regex_encode_walks_metal_(call, buffers, "sz_regex_expand_metal_kernel_", arguments);
            if (status == sz_success_k)
                status = sz_regex_clear_metal_(call, buffers, arguments, arena.chunk_covers,
                                               (chunks + 1) * sizeof(sz_u64_t));
            if (status == sz_success_k)
                status = sz_regex_encode_metal_(call, buffers, "sz_regex_drift_metal_kernel_", arguments,
                                                sz_regex_groups_metal_(chunks));
            if (status == sz_success_k)
                status = sz_regex_scan_metal_(call, buffers, arguments, arena.chunk_sums, chunks + 1);
            if (status == sz_success_k)
                status = sz_regex_encode_walks_metal_(call, buffers, "sz_regex_rewrite_metal_kernel_", arguments);
        }
        if (status == sz_success_k && extracting) {
            status = sz_regex_scan_metal_(call, buffers, arguments, arena.chunk_sums, chunks + 1);
            if (status == sz_success_k)
                status = sz_regex_encode_walks_metal_(call, buffers, "sz_regex_firsts_metal_kernel_", arguments);
            if (status == sz_success_k)
                status = sz_regex_scan_metal_(call, buffers, arguments, arena.chunk_covers, chunks + 1);
            if (status == sz_success_k)
                status = sz_regex_encode_walks_metal_(call, buffers, "sz_regex_rows_metal_kernel_", arguments);
        }
        arguments->storing = 1;
        if (status == sz_success_k)
            status = sz_regex_encode_metal_(call, buffers, "sz_regex_sweep_metal_kernel_", arguments,
                                            sz_regex_groups_metal_(chunks));
        if (status == sz_success_k)
            status = sz_regex_encode_one_metal_(call, buffers, "sz_regex_carry_metal_kernel_", arguments);
        if (status == sz_success_k && past && refusing) status = sz_regex_synchronize_metal_(call, stream);
        if (status == sz_success_k && past && refusing && counters[sz_regex_refused_counter_metal_k])
            status = sz_unexpected_dimensions_k;
    }
    return status;
}

/** Takes the one round block every call of @p engine runs out of and writes the column images at
 *  its head, so no compute verb ever allocates; the host writes the images, so the block must be
 *  one the open @p call resolves. */
STRINGZILLA_INLINE sz_status_t sz_regex_arena_reserve_metal_(sz_regex_engine_t *engine, sz_metal_call_t *call,
                                                             sz_stream_t stream) {
    sz_size_t const scratch_bytes = sz_regex_arena_metal_(engine).total;
    sz_metal_bound_t bound;
    engine->scratch = engine->allocator.allocate(scratch_bytes, engine->allocator.handle, stream);
    if (!engine->scratch) return sz_bad_alloc_k;
    engine->scratch_bytes = scratch_bytes;
    if (!sz_metal_resolve_call_(call, engine->memory, engine->memory_bytes, &bound) ||
        !sz_metal_resolve_call_(call, engine->scratch, engine->scratch_bytes, &bound))
        return sz_device_memory_mismatch_k;
    if (engine->reverse_rows)
        sz_regex_images_metal_(engine, engine->reverse_rows, engine->reverse_states_count, (sz_u32_t *)engine->scratch);
    if (engine->forward_rows)
        sz_regex_images_metal_(engine, engine->forward_rows, engine->forward_states_count,
                               (sz_u32_t *)((char *)engine->scratch + sz_regex_images_bytes_metal_(engine)));
    return sz_success_k;
}

/** Points every buffer a call leaves unbound at the round block,
 *  which the kernels naming it never touch. */
STRINGZILLA_INLINE void sz_regex_bind_unused_metal_(sz_metal_bound_t *buffers) {
    for (sz_size_t index = 0; index != sz_regex_buffers_metal_k; ++index)
        if (!buffers[index].buffer) buffers[index] = buffers[sz_regex_scratch_buffer_metal_k];
}

STRINGZILLA_API sz_status_t sz_regex_engine_init_metal(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                       sz_case_sensitivity_t case_sensitivity,
                                                       sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                       sz_stream_t stream) {
    sz_allocator_t unified;
    sz_metal_call_t call;
    sz_status_t status = sz_device_enter_metal_(stream, &call);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    if (allocator) unified = *allocator;
    else sz_allocator_init_unified_metal(&unified);
    status = sz_regex_engine_compile_(engine, patterns, case_sensitivity, malformed_index, sz_cap_metal_k, &unified,
                                      sz_regex_reverse_budget_k, sz_regex_forward_budget_k, stream);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);

    status = sz_regex_arena_reserve_metal_(engine, &call, stream);
    // Building every pipeline here keeps the compiles out of the calls.
    for (sz_size_t index = 0;
         status == sz_success_k && index != sizeof(sz_regex_kernels_metal_) / sizeof(sz_regex_kernels_metal_[0]);
         ++index)
        if (!sz_metal_pipeline_(call.context, sz_regex_source_metal_, sz_regex_kernels_metal_[index]))
            status = sz_device_code_mismatch_k;
    if (status != sz_success_k) {
        sz_regex_engine_free_(engine, stream);
        return sz_metal_commit_(&call, status);
    }
    return sz_metal_commit_(&call, sz_success_k);
}

STRINGZILLA_API sz_status_t sz_regex_counts_metal(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                  sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream) {
    sz_metal_bound_t buffers[sz_regex_buffers_metal_k] = {{STRINGZILLA_NULL, 0}};
    sz_regex_arguments_metal_t arguments;
    sz_metal_call_t call;
    sz_status_t status;
    if (!counts_stride) return sz_unexpected_dimensions_k;
    if (!engine->memory) return sz_missing_kernel_k;
    if (!haystacks->count) return sz_success_k;
    if (haystacks->layout == sz_strings_accessor_k) return sz_device_memory_mismatch_k;
    status = sz_device_enter_metal_(stream, &call);
    if (status == sz_success_k) status = sz_regex_resident_metal_(engine, &call, haystacks, buffers);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    if (!sz_metal_resolve_call_(&call, counts, ((haystacks->count - 1) * counts_stride + 1) * sizeof(sz_size_t),
                                buffers + sz_regex_counts_buffer_metal_k))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    sz_regex_bind_unused_metal_(buffers);

    sz_regex_arguments_metal_(engine, haystacks, sz_regex_counting_metal_k, &arguments);
    arguments.counts_stride = counts_stride;
    status = sz_regex_rounds_metal_(engine, &call, buffers, haystacks, &arguments, stream);
    // The sweeps left running totals, which two dispatches of one grid difference,
    // the results being dead by now.
    if (status == sz_success_k)
        status = sz_regex_encode_metal_(&call, buffers, "sz_regex_counts_edges_metal_kernel_", &arguments,
                                        sz_regex_groups_metal_(haystacks->count));
    if (status == sz_success_k)
        status = sz_regex_encode_metal_(&call, buffers, "sz_regex_counts_metal_kernel_", &arguments,
                                        sz_regex_groups_metal_(haystacks->count));
    return sz_metal_commit_(&call, status);
}

STRINGZILLA_API sz_status_t sz_regex_find_metal(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                sz_match_t *matches, sz_size_t matches_capacity,
                                                sz_size_t *matches_offsets, sz_stream_t stream) {
    sz_metal_bound_t buffers[sz_regex_buffers_metal_k] = {{STRINGZILLA_NULL, 0}};
    sz_regex_arguments_metal_t arguments;
    sz_metal_call_t call;
    sz_status_t status;
    if (!engine->memory) return sz_missing_kernel_k;
    if (!haystacks->count) return sz_success_k;
    if (haystacks->layout == sz_strings_accessor_k) return sz_device_memory_mismatch_k;
    status = sz_device_enter_metal_(stream, &call);
    if (status == sz_success_k) status = sz_regex_resident_metal_(engine, &call, haystacks, buffers);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    if (!sz_metal_resolve_call_(&call, matches_offsets, (haystacks->count + 1) * sizeof(sz_size_t),
                                buffers + sz_regex_matches_offsets_buffer_metal_k))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    if (matches_capacity && !sz_metal_resolve_call_(&call, matches, matches_capacity * sizeof(sz_match_t),
                                                    buffers + sz_regex_matches_buffer_metal_k))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    sz_regex_bind_unused_metal_(buffers);

    sz_regex_arguments_metal_(engine, haystacks, sz_regex_finding_metal_k, &arguments);
    arguments.matches_capacity = matches_capacity;
    status = sz_regex_rounds_metal_(engine, &call, buffers, haystacks, &arguments, stream);
    return sz_metal_commit_(&call, status);
}

STRINGZILLA_API sz_status_t sz_regex_replace_metal(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                   sz_strings_view_t const *replacements, sz_ptr_t target,
                                                   sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream) {
    sz_metal_bound_t buffers[sz_regex_buffers_metal_k] = {{STRINGZILLA_NULL, 0}};
    sz_regex_arguments_metal_t arguments;
    sz_metal_call_t call;
    sz_status_t status;
    if (!engine->memory) return sz_missing_kernel_k;
    if (replacements->count != engine->patterns_count) return sz_unexpected_dimensions_k;
    if (!haystacks->count) return sz_success_k;
    if (haystacks->layout == sz_strings_accessor_k || replacements->layout == sz_strings_accessor_k)
        return sz_device_memory_mismatch_k;
    status = sz_device_enter_metal_(stream, &call);
    if (status == sz_success_k) status = sz_regex_resident_metal_(engine, &call, haystacks, buffers);
    if (status == sz_success_k)
        status = sz_metal_tape_call_(&call, replacements, buffers + sz_regex_replacements_buffer_metal_k,
                                     buffers + sz_regex_replacements_offsets_buffer_metal_k);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    if (!sz_metal_resolve_call_(&call, offsets, (haystacks->count + 1) * sizeof(sz_size_t),
                                buffers + sz_regex_matches_offsets_buffer_metal_k))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    if (target_capacity &&
        !sz_metal_resolve_call_(&call, target, target_capacity, buffers + sz_regex_target_buffer_metal_k))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    sz_regex_bind_unused_metal_(buffers);

    sz_regex_arguments_metal_(engine, haystacks, sz_regex_replacing_metal_k, &arguments);
    arguments.target_capacity = target_capacity, arguments.replacements_layout = (sz_u64_t)replacements->layout;
    status = sz_regex_rounds_metal_(engine, &call, buffers, haystacks, &arguments, stream);
    return sz_metal_commit_(&call, status);
}

STRINGZILLA_API sz_status_t sz_regex_extract_metal(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                   sz_size_t pattern_index, sz_match_t *groups,
                                                   sz_size_t groups_capacity, sz_size_t *groups_offsets,
                                                   sz_stream_t stream) {
    sz_metal_bound_t buffers[sz_regex_buffers_metal_k] = {{STRINGZILLA_NULL, 0}};
    sz_regex_arguments_metal_t arguments;
    sz_metal_call_t call;
    sz_status_t status;
    if (!engine->memory) return sz_missing_kernel_k;
    if (pattern_index >= engine->patterns_count) return sz_unexpected_dimensions_k;
    if (!haystacks->count) return sz_success_k;
    if (haystacks->layout == sz_strings_accessor_k) return sz_device_memory_mismatch_k;
    status = sz_device_enter_metal_(stream, &call);
    if (status == sz_success_k) status = sz_regex_resident_metal_(engine, &call, haystacks, buffers);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    if (!sz_metal_resolve_call_(&call, groups_offsets, (haystacks->count + 1) * sizeof(sz_size_t),
                                buffers + sz_regex_matches_offsets_buffer_metal_k))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    if (groups_capacity && !sz_metal_resolve_call_(&call, groups, groups_capacity * sizeof(sz_match_t),
                                                   buffers + sz_regex_matches_buffer_metal_k))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    sz_regex_bind_unused_metal_(buffers);

    sz_regex_arguments_metal_(engine, haystacks, sz_regex_extracting_metal_k, &arguments);
    arguments.pattern_index = pattern_index, arguments.matches_capacity = groups_capacity;
    status = sz_regex_rounds_metal_(engine, &call, buffers, haystacks, &arguments, stream);
    return sz_metal_commit_(&call, status);
}

#pragma endregion Metal

#endif // STRINGZILLA_TARGET_METAL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_ARCH_METAL_
#endif // STRINGZILLA_REGEX_METAL_H_
