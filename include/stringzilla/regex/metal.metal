/**
 *  @file include/stringzilla/regex/metal.metal
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Regular expressions on Apple GPUs of Metal family 7, M1 and newer: one thread per chunk
 *      of the corpus marks the starts and walks the ends, and pointer doubling settles every chain
 *      that crosses a chunk's edge.
 *
 *  @sa include/stringzilla/regex/metal.h, which embeds and launches this source
 *  @sa include/stringzilla/regex/simt.cuh, the CUDA sibling this mirrors kernel for kernel
 *  @sa include/stringzilla/types.metal, the prelude compiled ahead of this source
 *  @sa include/stringzilla/utf8_uncased_fold/metal.metal, the folding steps embedded ahead of it
 *
 *  A haystack owns one slot per byte and one more ahead of them: a haystack with first slot @c V
 *  marks a match starting at byte @c p in slot <tt>V + p</tt>, which is also the slot of byte <tt>p - 1</tt>. The
 *  corpus spans the tape's bytes plus its count in slots, chunks cut that span evenly, so one long
 *  haystack and a million short ones fill the device alike, and a slot finds its haystack by binary
 *  search over the tape offsets. A round covers one segment of slots at a time, which bounds the
 *  candidates it holds by the segment's width.
 *
 *  Three things cross a chunk's edge, and each crosses it by pointer doubling rather than by one
 *  thread walking on: what the start-finding walk holds where a chunk ends, when priming does not
 *  collapse within a few bytes; the end walks that reach the edge; and the leftmost-first chain,
 *  whose first candidate in every chunk is found from the chunk before it. Position sets and
 *  lookaround passes hand sets across chunk edges as affine maps of the undecided elements, which
 *  compose by pointer doubling too, as <tt>regex/simt.cuh</tt> describes. The PikeVM walks every candidate
 *  without links instead. A chunk whose priming or flow stays wide past a few kilobytes is settled
 *  by one walk from the nearest exact state upstream, never past its segment.
 *
 *  Up to six lookarounds become columns of both DFAs, a block of columns per set of them that may
 *  hold, picked from a table of what holds at every slot. The table is a ring over a window of two
 *  segments; DFA walks reaching its end pause there, grouped by state, and resume once the host has
 *  slid the window on, while walks that cannot pause flag the call refused.
 *
 *  The automata and the round block each point inside themselves by host address, turned into the
 *  kernel's own by @ref sz_reach_metal_ against the block's host address. Candidate indices, slots
 *  and checkpoint ends are 32-bit offsets into the segment, whose width the host keeps below 2^31.
 */

/** Threads every threadgroup here runs. */
constant uint sz_regex_threads_metal_k = 256;

/** States a priming set holds; a column whose image is wider starts none. */
constant uint sz_regex_priming_states_metal_k = 16;

/** Undecided elements a chunk's edge set holds, one bit each of a composition's rows. */
constant uint sz_regex_undecided_metal_k = 64;

/** Bytes priming walks at most before a chunk settles its entry by composition instead. */
constant ulong sz_regex_priming_bytes_metal_k = 64;

/** Bytes priming or a set flow reaches upstream of a chunk at most,
 *  past which one walk settles its whole piece. */
constant ulong sz_regex_widening_bytes_metal_k = 4096;

/** Bytes a link walks on its own before a SIMD group walks the rest of it together. */
constant ulong sz_regex_link_bytes_metal_k = 1024;

/** Bytes each lane of a SIMD group walking a link together takes per round. */
constant uint sz_regex_piece_bytes_metal_k = 128;

/** Hops one round of pointer doubling follows, so a round reaches this many times
 *  further than the last. */
constant uint sz_regex_round_hops_metal_k = 8;

/** Slots between the checkpoints where end walks leave their state for later walks to stop at. */
constant ulong sz_regex_checkpoint_slots_metal_k = 16;

/** Bytes a run loads ahead together, so their latency overlaps the walk over the ones before. */
constant uint sz_regex_run_bytes_metal_k = 16;

/** Words of the sets a body's DFA flow holds, a bit per state, as @c sz_regex_body_states_simt_k
 *  bounds them. */
constant uint sz_regex_body_words_metal_k = 4;

/** The pattern, link, candidate and chunk id nothing takes. */
constant uint sz_regex_none_metal_k = 0xFFFFFFFFu;

/** The bit marking a walk's id as a group of paused walks rather than a link. */
constant uint sz_regex_group_metal_k = 0x80000000u;

/** The round's counters, as indices: links listed, groups opened, walks handed to SIMD groups,
 *  walks paused at the window's end, and the flag a walk that cannot pause sets where it would read
 *  the lookaround table past it. */
constant uint sz_regex_links_counter_metal_k = 0, sz_regex_groups_counter_metal_k = 1,
              sz_regex_handed_counter_metal_k = 2, sz_regex_paused_counter_metal_k = 3,
              sz_regex_refused_counter_metal_k = 4;

/** An unset capture slot or match end. */
constant ulong sz_regex_unset_metal_k = 0xFFFFFFFFFFFFFFFFul;

/** The unit past a haystack's last byte or before its first,
 *  as @c sz_regex_end_of_input_k numbers it. */
constant uint sz_regex_end_of_input_metal_k = 256;

/** NFA state kinds, as @c sz_regex_nfa_kind_t numbers them. */
constant uint sz_regex_nfa_ranges_metal_k = 0, sz_regex_nfa_empty_metal_k = 3, sz_regex_nfa_look_metal_k = 4,
              sz_regex_nfa_match_metal_k = 5;

/** Assertions, as @c sz_regex_look_t numbers them, the first lookaround's bit last. */
constant uint sz_regex_look_text_start_metal_k = 1, sz_regex_look_text_end_metal_k = 2,
              sz_regex_look_line_start_metal_k = 4, sz_regex_look_line_end_metal_k = 8,
              sz_regex_look_word_boundary_metal_k = 16, sz_regex_look_word_inside_metal_k = 32,
              sz_regex_look_lookaround_metal_k = 64;

/** The walk a round's set kernels run, as @c sz_regex_flow_simt_t numbers them. */
constant uint sz_regex_flow_positions_metal_k = 0, sz_regex_flow_behind_metal_k = 1, sz_regex_flow_ahead_metal_k = 2;

/** What a call writes, as @c sz_regex_verb_metal_t numbers it. */
constant ulong sz_regex_finding_metal_k = 0, sz_regex_counting_metal_k = 1, sz_regex_replacing_metal_k = 2,
               sz_regex_extracting_metal_k = 3;

/** How a walk reads its text, a byte at a time or by the folds of whole units, as
 *  @c sz_case_sensitivity_t numbers it. */
enum sz_regex_folding_metal_t : uint {
    sz_regex_cased_metal_k = 0,
    sz_regex_folded_metal_k = 1,
};

/** Whether a positions walk marks the starts it finds. */
enum sz_regex_marking_metal_t : uint {
    sz_regex_unmarked_metal_k = 0,
    sz_regex_marked_metal_k = 1,
};

/** Which automaton walks a candidate to its end without links: the folded DFA or the PikeVM. */
enum sz_regex_end_walk_metal_t : uint {
    sz_regex_folded_dfa_metal_k = 0,
    sz_regex_pike_vm_metal_k = 1,
};

/** Whether a lookaround pass's walk records where its lookarounds hold. */
enum sz_regex_recording_metal_t : uint {
    sz_regex_unrecorded_metal_k = 0,
    sz_regex_recorded_metal_k = 1,
};

/** One match, laid out as @c sz_match_t. */
struct sz_match_metal_t {
    ulong haystack_index, needle_index, byte_offset, byte_length;
};

/** One NFA state, laid out as @c sz_regex_nfa_state_t. */
struct sz_regex_nfa_state_metal_t {
    uint kind, look, target, slot, first_link, last_link;
};

/** One NFA link, laid out as @c sz_regex_nfa_link_t. */
struct sz_regex_nfa_link_metal_t {
    uint target, next;
    uchar low, high;
};

/** One lookaround, laid out as @c sz_regex_lookaround_t. */
struct sz_regex_lookaround_metal_t {
    uint start, kind, level;
};

/** The undecided elements of the set a chunk hands on,
 *  as @c sz_regex_undecided_simt_t lays them out. */
struct sz_regex_undecided_metal_t {
    uint count;
    uint indices[sz_regex_undecided_metal_k];
};

/** Which undecided elements a chunk hands on, as @c sz_regex_affine_simt_t lays the map out. */
struct sz_regex_affine_metal_t {
    uint target, rows_count;
    ulong fixed;
    ulong rows[sz_regex_undecided_metal_k];
};

/** What a walk in some state at a checkpoint ends with: a link to follow, and the match it
 *  found past the checkpoint, its end a slot offset into the segment. A zero state marks a
 *  checkpoint as free. */
struct sz_regex_checkpoint_metal_t {
    uint state, link, pattern, end;
};

/** An end walk's outcome, or a link's: the link whose match wins when it has one, else this
 *  match, its end an absolute slot. A link holds the end-finding state at its chunk edge
 *  until it walks. */
struct sz_regex_result_metal_t {
    uint link, pattern;
    ulong end;
};

/** A walk handed to a SIMD group: its link or group id, and the state it holds at the boundary
 *  in @c slot. A walk paused at the window's end keeps its state there, and any other ends with
 *  a zero one. */
struct sz_regex_continuation_metal_t {
    uint link, state;
    ulong slot;
};

/** The states the start-finding walk may hold at a chunk's right edge when priming did not collapse
 *  within a chunk's width, and the state each would leave the chunk in. */
struct sz_regex_frontier_metal_t {
    uint count;
    uint states[sz_regex_priming_states_metal_k];
    uint exits[sz_regex_priming_states_metal_k];
};

/** Which of its frontier's states a chunk enters with: the @c known index, or the frontier of chunk
 *  @c target together with each of that frontier's indices mapped into this one's. */
struct sz_regex_composition_metal_t {
    uint target, known;
    uchar indices[sz_regex_priming_states_metal_k];
};

/** What one segment hands the next, as @c sz_regex_carry_simt_t lays it out. */
struct sz_regex_carry_metal_t {
    ulong cursor, refusing, running, last_start, last_end, summed, counted, straddle;
};

/** The launch record the host copies in: host addresses in both blocks, and the round's shape. */
struct sz_regex_arguments_metal_t {
    ulong engine_host, scratch_host;
    ulong byte_to_class, reverse_rows, forward_rows, forward_patterns;
    ulong nfa_states, nfa_links, patterns_starts, patterns_groups, names, names_offsets, lookarounds;
    ulong positions_reach, positions_shifts, positions_groups, positions_all, positions_initial;
    ulong positions_ascii, positions_ranges;
    ulong reverse_images, forward_images, entries, edges, carry, counters, flags, tile_sums;
    ulong chunk_offsets, chunk_kept, chunk_sums, chunk_covers, chunk_haystacks;
    ulong frontiers, compositions, compositions_next, handoffs, handoff_edges, undecided, affines, affines_next;
    ulong looks, starts, checkpoints;
    ulong slots, results, links, links_next, links_list, nexts, exits, exits_next, keep, walks;
    ulong haystacks_count, haystacks_layout, replacements_layout, slots_count;
    ulong matches_capacity, target_capacity, counts_stride, pattern_index, verb;
    ulong segment, segment_begin, segment_end, chunk_width, chunk_shift, chunks_count, round, rounds, storing;
    ulong walks_count, walk_bytes, walk_sets, pike_slots, scan_count, scan_tiles, target, clear_words;
    ulong looks_begin, looks_end, looks_mask, set_words, bodies_begin, bodies_count;
    ulong groups_first, groups_capacity;
    ulong prefixes, looks_bytes, body_rows;
    ulong body_reach[sz_regex_body_words_metal_k];
    uint forward_starts[4], reverse_starts[4];
    uint forward_accepts_begin, reverse_accepts_begin, classes_count, case_sensitivity;
    uint forward_cells, reverse_cells, staged_cells, nfa_states_count;
    uint nfa_links_count, nfa_start, positions_words, positions_shifts_count;
    uint positions_groups_count, positions_ranges_count, patterns_count, lookarounds_count;
    uint flow, members, negated, stride;
    uint body, body_states, body_start, body_accepts;
};

/** The engine's arrays a kernel reads beyond its transition tables,
 *  reached from their host addresses. */
struct sz_regex_automata_metal_t {
    device uchar const *byte_to_class, *names;
    device uint const *forward_patterns, *patterns_starts, *patterns_groups, *names_offsets;
    device uint const *positions_ascii, *positions_ranges;
    device ulong const *positions_reach, *positions_shifts, *positions_groups, *positions_all, *positions_initial;
    device sz_regex_nfa_state_metal_t const *states;
    device sz_regex_nfa_link_metal_t const *links;
    device sz_regex_lookaround_metal_t const *lookarounds;
    device uint const *body_rows;
};

inline sz_regex_automata_metal_t sz_regex_automata_metal_(device uchar *engine,
                                                          constant sz_regex_arguments_metal_t &arguments) {
    ulong const host = arguments.engine_host;
    sz_regex_automata_metal_t automata;
    automata.byte_to_class = sz_reach_metal_<uchar>(engine, host, arguments.byte_to_class);
    automata.names = sz_reach_metal_<uchar>(engine, host, arguments.names);
    automata.forward_patterns = sz_reach_metal_<uint>(engine, host, arguments.forward_patterns);
    automata.patterns_starts = sz_reach_metal_<uint>(engine, host, arguments.patterns_starts);
    automata.patterns_groups = sz_reach_metal_<uint>(engine, host, arguments.patterns_groups);
    automata.names_offsets = sz_reach_metal_<uint>(engine, host, arguments.names_offsets);
    automata.positions_ascii = sz_reach_metal_<uint>(engine, host, arguments.positions_ascii);
    automata.positions_ranges = sz_reach_metal_<uint>(engine, host, arguments.positions_ranges);
    automata.positions_reach = sz_reach_metal_<ulong>(engine, host, arguments.positions_reach);
    automata.positions_shifts = sz_reach_metal_<ulong>(engine, host, arguments.positions_shifts);
    automata.positions_groups = sz_reach_metal_<ulong>(engine, host, arguments.positions_groups);
    automata.positions_all = sz_reach_metal_<ulong>(engine, host, arguments.positions_all);
    automata.positions_initial = sz_reach_metal_<ulong>(engine, host, arguments.positions_initial);
    automata.states = sz_reach_metal_<sz_regex_nfa_state_metal_t>(engine, host, arguments.nfa_states);
    automata.links = sz_reach_metal_<sz_regex_nfa_link_metal_t>(engine, host, arguments.nfa_links);
    automata.lookarounds = sz_reach_metal_<sz_regex_lookaround_metal_t>(engine, host, arguments.lookarounds);
    automata.body_rows = sz_reach_metal_<uint>(engine, host, arguments.body_rows);
    return automata;
}

/** Every array of the round block, reached from its host address. */
struct sz_regex_round_metal_t {
    device uint const *reverse_images, *forward_images;
    device uint *entries, *edges, *flags, *slots, *links_list, *nexts, *exits, *exits_next, *keep, *kept, *looks;
    device atomic_uint *counters;
    device ulong *tile_sums, *chunk_offsets, *chunk_kept, *chunk_sums, *chunk_covers, *chunk_haystacks, *starts,
        *drifts, *handoffs, *handoff_edges;
    device sz_regex_carry_metal_t *carry;
    device sz_regex_frontier_metal_t *frontiers;
    device sz_regex_composition_metal_t *compositions, *compositions_next;
    device sz_regex_undecided_metal_t *undecided;
    device sz_regex_affine_metal_t *affines, *affines_next;
    device sz_regex_checkpoint_metal_t *checkpoints;
    device sz_regex_result_metal_t *results, *links, *links_next, *groups, *groups_next;
    device atomic_uint *leaders;
    device uchar *walks;
};

inline sz_regex_round_metal_t sz_regex_round_metal_(device uchar *scratch,
                                                    constant sz_regex_arguments_metal_t &arguments) {
    ulong const host = arguments.scratch_host;
    sz_regex_round_metal_t round;
    round.reverse_images = sz_reach_metal_<uint>(scratch, host, arguments.reverse_images);
    round.forward_images = sz_reach_metal_<uint>(scratch, host, arguments.forward_images);
    round.entries = sz_reach_metal_<uint>(scratch, host, arguments.entries);
    round.edges = sz_reach_metal_<uint>(scratch, host, arguments.edges);
    round.flags = sz_reach_metal_<uint>(scratch, host, arguments.flags);
    round.slots = sz_reach_metal_<uint>(scratch, host, arguments.slots);
    round.links_list = sz_reach_metal_<uint>(scratch, host, arguments.links_list);
    round.nexts = sz_reach_metal_<uint>(scratch, host, arguments.nexts);
    round.exits = sz_reach_metal_<uint>(scratch, host, arguments.exits);
    round.exits_next = sz_reach_metal_<uint>(scratch, host, arguments.exits_next);
    round.keep = sz_reach_metal_<uint>(scratch, host, arguments.keep);
    // Kept flags reuse the links' list, and drifts the exits, both dead once the chain is doubled.
    round.kept = round.links_list;
    round.drifts = (device ulong *)round.exits;
    round.counters = sz_reach_metal_<atomic_uint>(scratch, host, arguments.counters);
    round.tile_sums = sz_reach_metal_<ulong>(scratch, host, arguments.tile_sums);
    round.chunk_offsets = sz_reach_metal_<ulong>(scratch, host, arguments.chunk_offsets);
    round.chunk_kept = sz_reach_metal_<ulong>(scratch, host, arguments.chunk_kept);
    round.chunk_sums = sz_reach_metal_<ulong>(scratch, host, arguments.chunk_sums);
    round.chunk_covers = sz_reach_metal_<ulong>(scratch, host, arguments.chunk_covers);
    round.chunk_haystacks = sz_reach_metal_<ulong>(scratch, host, arguments.chunk_haystacks);
    round.starts = sz_reach_metal_<ulong>(scratch, host, arguments.starts);
    round.carry = sz_reach_metal_<sz_regex_carry_metal_t>(scratch, host, arguments.carry);
    round.frontiers = sz_reach_metal_<sz_regex_frontier_metal_t>(scratch, host, arguments.frontiers);
    round.compositions = sz_reach_metal_<sz_regex_composition_metal_t>(scratch, host, arguments.compositions);
    round.compositions_next = sz_reach_metal_<sz_regex_composition_metal_t>(scratch, host, arguments.compositions_next);
    round.handoffs = sz_reach_metal_<ulong>(scratch, host, arguments.handoffs);
    round.handoff_edges = sz_reach_metal_<ulong>(scratch, host, arguments.handoff_edges);
    round.undecided = sz_reach_metal_<sz_regex_undecided_metal_t>(scratch, host, arguments.undecided);
    round.affines = sz_reach_metal_<sz_regex_affine_metal_t>(scratch, host, arguments.affines);
    round.affines_next = sz_reach_metal_<sz_regex_affine_metal_t>(scratch, host, arguments.affines_next);
    round.looks = sz_reach_metal_<uint>(scratch, host, arguments.looks);
    round.checkpoints = sz_reach_metal_<sz_regex_checkpoint_metal_t>(scratch, host, arguments.checkpoints);
    round.results = sz_reach_metal_<sz_regex_result_metal_t>(scratch, host, arguments.results);
    round.links = sz_reach_metal_<sz_regex_result_metal_t>(scratch, host, arguments.links);
    round.links_next = sz_reach_metal_<sz_regex_result_metal_t>(scratch, host, arguments.links_next);
    round.walks = sz_reach_metal_<uchar>(scratch, host, arguments.walks);
    // Groups of paused walks fill the checkpoints' halves and their leaders the frontiers,
    // dead while walks slide.
    round.groups = (device sz_regex_result_metal_t *)round.checkpoints;
    round.groups_next = round.groups + arguments.groups_capacity;
    round.leaders = (device atomic_uint *)round.frontiers;
    return round;
}

/** One automaton's transitions as a walk reads them: the class map, the first column of each block
 *  of lookarounds and the rows' head staged in threadgroup memory, the rest of the rows in device
 *  memory, and the ring of the lookaround table whose entries pick the blocks, null without
 *  lookarounds, with the end of its window and the flag a walk that cannot pause there sets. */
struct sz_regex_table_metal_t {
    threadgroup uchar const *classes;
    threadgroup uint const *blocks, *head;
    device uint const *rows, *looks;
    device atomic_uint *refused;
    ulong looks_mask, looks_end;
    uint head_cells, edge_column;
};

/** Blocks of lookaround columns a row holds at most, one per set of the six columned lookarounds,
 *  as @c sz_regex_columned_lookarounds_k bounds them, each staged by its first column. */
constant uint sz_regex_blocks_metal_k = 64;

/** Bytes of threadgroup memory the class map and the blocks take ahead of the staged rows,
 *  as the host sizes it. */
constant uint sz_regex_staged_bytes_metal_k = 256 + sz_regex_blocks_metal_k * sizeof(uint);

/** The first column of the block a lookaround table entry's @p looks select,
 *  as @c sz_regex_looks_block_ finds it. */
inline uint sz_regex_looks_block_metal_(constant sz_regex_arguments_metal_t &arguments, uint looks) {
    return (looks / sz_regex_look_lookaround_metal_k & ((1u << arguments.lookarounds_count) - 1)) *
           (arguments.classes_count + 1);
}

/** Stages @p cells of @p rows, the class map and the blocks into @p staged, with every thread of
 *  the threadgroup taking part, so a kernel calls it before any thread can leave. */
inline sz_regex_table_metal_t sz_regex_stage_metal_(device uchar *engine, device uchar *scratch,
                                                    constant sz_regex_arguments_metal_t &arguments, ulong rows_host,
                                                    uint cells, threadgroup uchar *staged, uint lane, uint width) {
    sz_regex_table_metal_t table;
    device uchar const *byte_to_class = sz_reach_metal_<uchar>(engine, arguments.engine_host, arguments.byte_to_class);
    threadgroup uint *blocks = (threadgroup uint *)(staged + 256);
    threadgroup uint *head = (threadgroup uint *)(staged + sz_regex_staged_bytes_metal_k);
    table.rows = sz_reach_metal_<uint>(engine, arguments.engine_host, rows_host);
    table.looks = arguments.lookarounds_count ? sz_reach_metal_<uint>(scratch, arguments.scratch_host, arguments.looks)
                                              : nullptr;
    table.refused = sz_reach_metal_<atomic_uint>(scratch, arguments.scratch_host, arguments.counters) +
                    sz_regex_refused_counter_metal_k;
    table.looks_mask = arguments.looks_mask, table.looks_end = arguments.looks_end;
    table.classes = staged, table.blocks = blocks, table.head = head;
    table.head_cells = min(cells, arguments.staged_cells), table.edge_column = arguments.classes_count;
    for (uint index = lane; index < 256; index += width) staged[index] = byte_to_class[index];
    for (uint index = lane; index < sz_regex_blocks_metal_k; index += width)
        blocks[index] = sz_regex_looks_block_metal_(arguments, index * sz_regex_look_lookaround_metal_k);
    for (uint index = lane; index < table.head_cells; index += width) head[index] = table.rows[index];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return table;
}

/** The state @p state moves to in @p column. */
inline uint sz_regex_move_metal_(thread sz_regex_table_metal_t const &table, uint state, uint column) {
    uint const cell = state + column;
    return cell < table.head_cells ? table.head[cell] : table.rows[cell];
}

/** The first column of the block the lookarounds holding at @p slot select,
 *  zero without lookarounds. */
inline uint sz_regex_block_at_metal_(thread sz_regex_table_metal_t const &table, ulong slot) {
    return table.looks ? table.blocks[table.looks[slot & table.looks_mask] / sz_regex_look_lookaround_metal_k %
                                      sz_regex_blocks_metal_k]
                       : 0;
}

/** Whether a walk would read the lookarounds at @p slot past the table's window,
 *  where it pauses or refuses. */
inline bool sz_regex_past_metal_(thread sz_regex_table_metal_t const &table, ulong slot) {
    return table.looks && slot >= table.looks_end;
}

/** Advances @p state over the haystack-edge column, at the haystack boundary in @p slot. */
inline uint sz_regex_step_edge_metal_(thread sz_regex_table_metal_t const &table, uint state, ulong slot) {
    return sz_regex_move_metal_(table, state, sz_regex_block_at_metal_(table, slot) + table.edge_column);
}

/** One automaton's transitions read straight from device memory, for the kernels that stage
 *  nothing, its class map and blocks in the threadgroup memory every dispatch carries. */
inline sz_regex_table_metal_t sz_regex_direct_metal_(device uchar *engine, device uchar *scratch,
                                                     constant sz_regex_arguments_metal_t &arguments, ulong rows_host,
                                                     threadgroup uchar *staged, uint lane, uint width) {
    return sz_regex_stage_metal_(engine, scratch, arguments, rows_host, 0, staged, lane, width);
}

/**
 *  @brief Walks @p state forward over the @p count bytes at @p text, at most a run, stopping where
 *      it dies; the byte at @p text crosses the boundary in @p slot.
 *  @return The bytes consumed; @p last and @p last_offset take the run's last accepting state and
 *      the offset of the byte that reached it.
 */
inline uint sz_regex_run_metal_(thread sz_regex_table_metal_t const &table, thread uint &state,
                                device uchar const *text, ulong slot, uint count, uint accepts, thread uint &last,
                                thread uint &last_offset) {
    uint columns[sz_regex_run_bytes_metal_k];
#pragma unroll
    for (uint index = 0; index != sz_regex_run_bytes_metal_k; ++index)
        columns[index] = index < count ? table.classes[text[index]] + sz_regex_block_at_metal_(table, slot + index) : 0;
#pragma unroll
    for (uint index = 0; index != sz_regex_run_bytes_metal_k; ++index) {
        if (index == count) break;
        state = sz_regex_move_metal_(table, state, columns[index]);
        if (state >= accepts) last = state, last_offset = index;
        if (state == 0) return index + 1;
    }
    return count;
}

/**
 *  @brief Walks @p state right to left over the @p count bytes before @p text_end, at most a run,
 *      the last of them crossing the boundary in @p slot.
 *  @return A bit per byte whose step left the walk at or past @p accepts, bit 0 the leftmost byte.
 */
inline uint sz_regex_run_back_metal_(thread sz_regex_table_metal_t const &table, thread uint &state,
                                     device uchar const *text_end, ulong slot, uint count, uint accepts) {
    uint columns[sz_regex_run_bytes_metal_k];
    uint mask = 0;
#pragma unroll
    for (uint index = 0; index != sz_regex_run_bytes_metal_k; ++index)
        columns[index] = index < count
                             ? table.classes[text_end[-1 - int(index)]] + sz_regex_block_at_metal_(table, slot - index)
                             : 0;
#pragma unroll
    for (uint index = 0; index != sz_regex_run_bytes_metal_k; ++index) {
        if (index == count) break;
        state = sz_regex_move_metal_(table, state, columns[index]);
        if (state >= accepts) mask |= 1u << (count - 1 - index);
    }
    return mask;
}

/** Walks @p state right to left from byte @p at of @p text down to byte @p stop, marking nothing;
 *  the text's first boundary is in @p first. */
inline uint sz_regex_walk_back_metal_(thread sz_regex_table_metal_t const &table, uint state, device uchar const *text,
                                      ulong first, ulong at, ulong stop) {
    while (at != stop) {
        uint const count = uint(min(at - stop, (ulong)sz_regex_run_bytes_metal_k));
        sz_regex_run_back_metal_(table, state, text + at, first + at, count, sz_regex_none_metal_k);
        at -= count;
    }
    return state;
}

/** The pattern a forward accepting @p state reports, as @c sz_regex_walk_end_serial_ finds it. */
inline uint sz_regex_pattern_metal_(thread sz_regex_automata_metal_t const &automata,
                                    constant sz_regex_arguments_metal_t &arguments, uint state) {
    return automata.forward_patterns[(state - arguments.forward_accepts_begin) / arguments.stride];
}

/** Whether @p unit, a byte or the end of input, is an ASCII word byte, which <tt>\\b</tt> and <tt>\\B</tt> test. */
inline bool sz_regex_is_word_metal_(uint unit) {
    return (unit >= '0' && unit <= '9') || (unit >= 'A' && unit <= 'Z') || (unit >= 'a' && unit <= 'z') || unit == '_';
}

/** The boundary kind of a byte: a line feed, a word byte, or another. */
inline uint sz_regex_byte_kind_metal_(uchar byte) { return byte == '\n' ? 1 : sz_regex_is_word_metal_(byte) ? 2 : 3; }

/** The boundary kind of a folded rune, as @c sz_regex_rune_kind_ reads it. */
inline uint sz_regex_rune_kind_metal_(uint rune) { return rune < 0x80 ? sz_regex_byte_kind_metal_(uchar(rune)) : 3; }

/** The lookarounds holding at @p slot, from the ring of @p table whose halves hold the window's two
 *  segments, none outside the window the record names; walks pause or refuse at its end, so none
 *  reads past it. */
inline uint sz_regex_looks_at_metal_(device uint const *table, constant sz_regex_arguments_metal_t &arguments,
                                     ulong slot) {
    return slot >= arguments.looks_begin && slot < arguments.looks_end ? table[slot & arguments.looks_mask] : 0;
}

/** The haystacks' tape, and the offset its slots count from. */
struct sz_regex_corpus_metal_t {
    sz_strings_tape_metal_t tape;
    ulong base, count;
};

inline sz_regex_corpus_metal_t sz_regex_corpus_metal_(device uchar const *data, device uchar const *offsets,
                                                      constant sz_regex_arguments_metal_t &arguments) {
    sz_regex_corpus_metal_t corpus;
    corpus.tape = {data, offsets, arguments.haystacks_layout};
    corpus.base = sz_strings_offset_metal_(corpus.tape, 0);
    corpus.count = arguments.haystacks_count;
    return corpus;
}

/** The slot haystack @p index starts at, the one its empty-prefix match is marked in. */
inline ulong sz_regex_first_slot_metal_(sz_regex_corpus_metal_t corpus, ulong index) {
    return sz_strings_offset_metal_(corpus.tape, index) - corpus.base + index;
}

/** The haystack holding @p slot, galloping up from @p low, a haystack starting at or before it. */
inline ulong sz_regex_haystack_at_metal_(sz_regex_corpus_metal_t corpus, ulong slot, ulong low) {
    ulong step = 1;
    while (low + step < corpus.count && sz_regex_first_slot_metal_(corpus, low + step) <= slot) low += step, step *= 2;
    ulong high = min(low + step, corpus.count);
    while (low + 1 < high) {
        ulong const middle = low + (high - low) / 2;
        if (sz_regex_first_slot_metal_(corpus, middle) <= slot) low = middle;
        else high = middle;
    }
    return low;
}

/** One haystack, as a walk reads it: its bytes, its length, and its first slot. */
struct sz_regex_haystack_metal_t {
    device uchar const *text;
    ulong length, first;
};

inline sz_regex_haystack_metal_t sz_regex_haystack_metal_(sz_regex_corpus_metal_t corpus, ulong index) {
    sz_regex_haystack_metal_t haystack;
    haystack.text = sz_strings_start_metal_(corpus.tape, index);
    haystack.length = sz_strings_length_metal_(corpus.tape, index);
    haystack.first = sz_regex_first_slot_metal_(corpus, index);
    return haystack;
}

/** The first of @p count ascending slots at or past @p wanted, galloping up from @p low;
 *  @p count when none is. */
inline uint sz_regex_lower_bound_metal_(device uint const *slots, uint low, uint count, ulong wanted) {
    uint step = 1;
    if (low == count || slots[low] >= wanted) return low;
    while (low + step < count && slots[low + step] < wanted) low += step, step *= 2;
    uint high = min(low + step, count);
    while (low + 1 < high) {
        uint const middle = low + (high - low) / 2;
        if (slots[middle] < wanted) low = middle;
        else high = middle;
    }
    return high;
}

/** Index of @p state among the first @p count of @p states, or 0xFF. */
inline uchar sz_regex_index_of_metal_(thread uint const *states, uint count, uint state) {
    for (uint index = 0; index != count; ++index)
        if (states[index] == state) return uchar(index);
    return 0xFF;
}

/**
 *  @brief The start-finding state the walk from the haystack's end holds once it has consumed
 *      every byte from @p edge on, found without that walk, or zero when @p reach bytes did
 *      not settle it.
 *
 *  Every state walks the bytes right of the edge toward it together, deduplicated, until one is
 *  left, as @c sz_regex_prime_serial_ does. The set starts as the image of one column, the states
 *  any state steps to on that byte, which holds the true one; a state that dies is dropped, as the
 *  true walk never dies inside a haystack. A distance that ends uncollapsed doubles up to @p reach,
 *  past which the set left at the edge is the answer, or, with no column narrowing to one in
 *  @ref sz_regex_widening_bytes_metal_k, an empty set marking the chunk wide; a distance that
 *  reaches the haystack's end starts from the true start state.
 */
inline uint sz_regex_prime_metal_(thread sz_regex_table_metal_t const &table,
                                  constant sz_regex_arguments_metal_t &arguments, device uint const *images,
                                  device uchar const *text, ulong first, ulong length, ulong edge, ulong reach,
                                  thread uint *set, thread uint &count) {
    for (ulong distance = 32;; distance *= 2) {
        ulong const far = length - edge > distance ? edge + distance : length;
        ulong position = far;
        count = 0;
        if (far == length) set[0] = arguments.reverse_starts[0], count = 1;
        while (count == 0 && position > edge) {
            uint const column = sz_regex_block_at_metal_(table, first + position) + table.classes[text[position - 1]];
            uint const image = images[column];
            --position;
            if (image > sz_regex_priming_states_metal_k) continue;
            for (uint index = 0; index != image; ++index)
                set[index] = images[arguments.stride + column * sz_regex_priming_states_metal_k + index];
            count = image;
        }
        while (count > 1 && position > edge) {
            uint const column = sz_regex_block_at_metal_(table, first + position) + table.classes[text[position - 1]];
            uint kept = 0;
            --position;
            for (uint index = 0; index != count; ++index) {
                uint const next = sz_regex_move_metal_(table, set[index], column);
                if (next != 0 && sz_regex_index_of_metal_(set, kept, next) == 0xFF) set[kept++] = next;
            }
            count = kept;
        }
        if (count == 1) return sz_regex_walk_back_metal_(table, set[0], text, first, position, edge);
        if (count > 1 && distance >= reach) return 0;
        // A stretch where no column narrows to a frontier is left to the walk of its whole piece.
        if (distance >= sz_regex_widening_bytes_metal_k) return count = 0;
    }
}

/** The result candidate @p index settles to once its link is followed. */
inline sz_regex_result_metal_t sz_regex_linked_metal_(thread sz_regex_round_metal_t const &round, uint index) {
    sz_regex_result_metal_t result = round.results[index];
    if (result.link != sz_regex_none_metal_k) {
        sz_regex_result_metal_t const linked = round.links[result.link];
        if (linked.pattern != sz_regex_none_metal_k) result.pattern = linked.pattern, result.end = linked.end;
    }
    return result;
}

/** The match a search takes at candidate @p index of @p haystack, its link followed, with no
 *  pattern when the search passes it: a walk that never accepted, or an empty match before a
 *  UTF-8 continuation byte. */
inline sz_regex_result_metal_t sz_regex_settle_metal_(thread sz_regex_round_metal_t const &round, uint index,
                                                      ulong mark, thread sz_regex_haystack_metal_t const &haystack) {
    sz_regex_result_metal_t result = sz_regex_linked_metal_(round, index);
    ulong const start = mark - haystack.first;
    if (result.pattern != sz_regex_none_metal_k && result.end == mark && start < haystack.length &&
        (haystack.text[start] & 0xC0) == 0x80)
        result.pattern = sz_regex_none_metal_k;
    return result;
}

/** Writes where haystack @p index begins among the kept matches: into the matches offsets of a
 *  find, or, for a count, as the total before it into the count of the haystack before it, which
 *  the counts kernels difference. */
inline void sz_regex_boundary_store_metal_(constant sz_regex_arguments_metal_t &arguments,
                                           device ulong *matches_offsets, device ulong *counts, ulong index,
                                           ulong value) {
    if (arguments.verb == sz_regex_counting_metal_k) {
        if (index) counts[(index - 1) * arguments.counts_stride] = value;
    }
    else if (arguments.verb == sz_regex_finding_metal_k) matches_offsets[index] = value;
}

/** Byte @p index of @p image. */
inline uchar sz_regex_image_byte_metal_(thread sz_utf8_folded_image_metal_t const &image, uint index) {
    return uchar(image.bytes >> (8 * index));
}

/** The unit starting at @p position as walks step it, the byte itself unless @p folded, else the
 *  folds of its codepoint or one malformed byte. @return Where the unit ends. */
inline ulong sz_regex_unit_after_metal_(sz_regex_folding_metal_t folded, device uchar const *text, ulong length,
                                        ulong position, thread sz_utf8_folded_image_metal_t &image) {
    if (folded) return position + sz_utf8_fold_next_metal_(text + position, text + length, image);
    image.bytes = text[position], image.length = 1, image.rune_ends = 1;
    return position + 1;
}

/**
 *  @brief The unit ending at @p end, past zero, as @ref sz_regex_unit_after_metal_ steps it.
 *  @return Where it begins.
 */
inline ulong sz_regex_unit_before_metal_(sz_regex_folding_metal_t folded, device uchar const *text, ulong end,
                                         thread sz_utf8_folded_image_metal_t &image) {
    if (folded) return end - sz_utf8_fold_prev_metal_(text, text + end, image);
    image.bytes = text[end - 1], image.length = 1, image.rune_ends = 1;
    return end - 1;
}

/** The rune of @p image whose bytes end at index @p last, a malformed byte tagged as
 *  @c sz_rune_malformed_byte_ tags it. @return The index its bytes begin at. */
inline uint sz_regex_image_rune_metal_(thread sz_utf8_folded_image_metal_t const &image, uint last, thread uint &rune) {
    if (!image.rune_ends) {
        rune = 0x80000000u | sz_regex_image_byte_metal_(image, 0);
        return 0;
    }
    uint first = last;
    while (first && !(image.rune_ends >> (first - 1) & 1)) --first;
    uchar const lead = sz_regex_image_byte_metal_(image, first);
    rune = first == last ? lead : lead & (0x7Fu >> (last - first + 1));
    for (uint index = first + 1; index <= last; ++index)
        rune = rune << 6 | (sz_regex_image_byte_metal_(image, index) & 0x3Fu);
    return first;
}

/** The first unit boundary at or past @p position, as @c sz_regex_boundary_ finds it. */
inline ulong sz_regex_boundary_metal_(device uchar const *text, ulong length, ulong position) {
    if (position == length || (text[position] & 0xC0) != 0x80) return position;
    ulong back = 1;
    while (back != 4 && back <= position && (text[position - back] & 0xC0) == 0x80) ++back;
    if (back == 4 || back > position) return position;
    uint rune;
    ulong const rune_length = sz_utf8_decode_rune_metal_(text, length, position - back, rune);
    return rune_length > back ? position - back + rune_length : position;
}

/** Walks @p state right to left over the folds of the units of @p text ending in
 *  <tt>(stop, at]</tt>, the text's first boundary in @p first; only the last fold of a unit crosses
 *  its boundary's lookarounds. */
inline uint sz_regex_walk_folded_back_metal_(thread sz_regex_table_metal_t const &table, uint state,
                                             device uchar const *text, ulong first, ulong at, ulong stop) {
    sz_utf8_folded_image_metal_t image;
    while (at > stop) {
        uint block = sz_regex_block_at_metal_(table, first + at);
        at = sz_regex_unit_before_metal_(sz_regex_folded_metal_k, text, at, image);
        for (uint byte = image.length; byte--; block = 0)
            state = sz_regex_move_metal_(table, state, block + table.classes[sz_regex_image_byte_metal_(image, byte)]);
    }
    return state;
}

/** The start-finding state a folded walk holds at unit boundary @p edge, or zero when @p reach
 *  bytes did not settle it, leaving the uncollapsed set at the edge, or an empty one for a wide
 *  chunk, as @c sz_regex_prime_folded_simt_ finds it. */
inline uint sz_regex_prime_folded_metal_(thread sz_regex_table_metal_t const &table,
                                         constant sz_regex_arguments_metal_t &arguments, device uint const *images,
                                         device uchar const *text, ulong first, ulong length, ulong edge, ulong reach,
                                         thread uint *set, thread uint &count) {
    for (ulong distance = 32;; distance *= 2) {
        ulong const far = length - edge > distance ? sz_regex_boundary_metal_(text, length, edge + distance) : length;
        ulong position = far;
        sz_utf8_folded_image_metal_t image;
        uint byte = 0, block = 0;
        count = 0;
        if (far == length) set[0] = arguments.reverse_starts[0], count = 1;
        // The folded bytes step right to left, `byte` counting
        // those of the unit at `position` still to step.
        while (count != 1 && (byte || position > edge)) {
            if (!byte)
                block = sz_regex_block_at_metal_(table, first + position),
                position = sz_regex_unit_before_metal_(sz_regex_folded_metal_k, text, position, image),
                byte = image.length;
            uint const column = block + table.classes[sz_regex_image_byte_metal_(image, --byte)];
            block = 0;
            if (count == 0) {
                if (images[column] > sz_regex_priming_states_metal_k) continue;
                for (uint index = 0; index != images[column]; ++index)
                    set[index] = images[arguments.stride + column * sz_regex_priming_states_metal_k + index];
                count = images[column];
                continue;
            }
            uint kept = 0;
            for (uint index = 0; index != count; ++index) {
                uint const next = sz_regex_move_metal_(table, set[index], column);
                if (next != 0 && sz_regex_index_of_metal_(set, kept, next) == 0xFF) set[kept++] = next;
            }
            count = kept;
        }
        if (count > 1 && distance >= reach) return 0;
        if (count != 1 && distance >= sz_regex_widening_bytes_metal_k) return count = 0;
        if (count != 1) continue;
        uint state = set[0];
        while (byte)
            state = sz_regex_move_metal_(table, state, table.classes[sz_regex_image_byte_metal_(image, --byte)]);
        return sz_regex_walk_folded_back_metal_(table, state, text, first, position, edge);
    }
}

/**
 *  @brief Walks the end-finding automaton over folded bytes, as @c sz_regex_walk_folded_simt_
 *      does, without links.
 *  @return Whether a match ends there, an empty one counting only at the unit's own start; none
 *      where the walk would read the lookaround table past its window, which it cannot pause at, so
 *      it flags the call refused.
 */
inline bool sz_regex_walk_folded_metal_(thread sz_regex_table_metal_t const &table,
                                        thread sz_regex_automata_metal_t const &automata,
                                        constant sz_regex_arguments_metal_t &arguments, device uchar const *text,
                                        ulong first, ulong length, ulong start, ulong offset, thread ulong &end,
                                        thread uint &pattern) {
    uint const accepts = arguments.forward_accepts_begin;
    sz_utf8_folded_image_metal_t image;
    ulong position = start, unit_end = start, byte = offset, consumed = 0, last_consumed = 0;
    ulong last_end = sz_regex_unset_metal_k;
    uint kind = 0, last = 0;
    if (start != length) unit_end = sz_regex_unit_after_metal_(sz_regex_folded_metal_k, text, length, start, image);
    if (offset) kind = sz_regex_byte_kind_metal_(sz_regex_image_byte_metal_(image, uint(offset) - 1));
    else if (start) {
        sz_utf8_folded_image_metal_t previous;
        sz_regex_unit_before_metal_(sz_regex_folded_metal_k, text, start, previous);
        kind = sz_regex_byte_kind_metal_(sz_regex_image_byte_metal_(previous, previous.length - 1u));
    }
    uint state = arguments.forward_starts[kind];
    while (state != 0) {
        if (!byte && sz_regex_past_metal_(table, first + position)) {
            atomic_store_explicit(table.refused, 1u, memory_order_relaxed);
            return false;
        }
        if (position == length) break;
        uint const block = byte ? 0 : sz_regex_block_at_metal_(table, first + position);
        state = sz_regex_move_metal_(table, state,
                                     block + table.classes[sz_regex_image_byte_metal_(image, uint(byte))]);
        if (state >= accepts) last = state, last_end = byte ? unit_end : position, last_consumed = consumed;
        ++consumed;
        if (++byte != image.length) continue;
        position = unit_end, byte = 0;
        if (position != length)
            unit_end = sz_regex_unit_after_metal_(sz_regex_folded_metal_k, text, length, position, image);
    }
    if (state != 0 && (state = sz_regex_step_edge_metal_(table, state, first + length)) >= accepts)
        last = state, last_end = length, last_consumed = consumed;
    if (last_end == sz_regex_unset_metal_k || (offset && !last_consumed)) return false;
    end = last_end, pattern = sz_regex_pattern_metal_(automata, arguments, last);
    return true;
}

/** The unit class of a folded @p rune, zero for a malformed byte,
 *  as @c sz_regex_rune_class_ finds it. */
inline uint sz_regex_rune_class_metal_(thread sz_regex_automata_metal_t const &automata,
                                       constant sz_regex_arguments_metal_t &arguments, uint rune) {
    if (rune < 0x80) return automata.positions_ascii[rune];
    if (rune & 0x80000000u) return 0;
    ulong low = 0, high = arguments.positions_ranges_count;
    while (high - low > 1) {
        ulong const middle = (low + high) / 2;
        if (automata.positions_ranges[middle * 2] <= rune) low = middle;
        else high = middle;
    }
    return automata.positions_ranges[low * 2 + 1];
}

/**
 *  @brief The unit ending at @p end and its class in @p unit_class, as
 *      @c sz_regex_unit_before_ finds them.
 *  @return Where it begins.
 */
inline ulong sz_regex_class_before_metal_(thread sz_regex_automata_metal_t const &automata,
                                          constant sz_regex_arguments_metal_t &arguments, device uchar const *text,
                                          ulong end, thread uint &unit_class) {
    uchar const last = text[end - 1];
    if (last < 0x80) {
        unit_class = automata.positions_ascii[last];
        return end - 1;
    }
    unit_class = 0;
    if ((last & 0xC0) != 0x80) return end - 1;
    ulong back = 2;
    while (back != 5 && back <= end && (text[end - back] & 0xC0) == 0x80) ++back;
    uint rune;
    if (back == 5 || back > end || sz_utf8_decode_rune_metal_(text, end, end - back, rune) != back) return end - 1;
    unit_class = sz_regex_rune_class_metal_(automata, arguments, rune);
    return end - back;
}

/** Steps the position set @p state over one unit of @p unit_class at a boundary of @p kind where
 *  the lookarounds of @p looks hold into @p next, as @c sz_regex_positions_step_ does. */
inline void sz_regex_positions_step_metal_(thread sz_regex_automata_metal_t const &automata,
                                           constant sz_regex_arguments_metal_t &arguments, device ulong const *state,
                                           device ulong *next, ulong unit_class, ulong kind, uint looks) {
    ulong const words = arguments.positions_words;
    device ulong const *record = automata.positions_shifts;
    device ulong const *reach = automata.positions_reach + unit_class * words;
    device ulong const *initial = automata.positions_initial + kind * words;
    for (ulong word = 0; word != words; ++word) next[word] = initial[word];
    for (uint shift = 0; shift != arguments.positions_shifts_count; ++shift, record += 1 + words) {
        long const distance = long(record[0]);
        ulong const span = ulong(distance < 0 ? -distance : distance), offset = span / 64, bits = span % 64;
        device ulong const *mask = record + 1;
        for (ulong word = offset; word < words; ++word) {
            ulong const low = word - offset;
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
    record = automata.positions_groups;
    for (uint group = 0; group != arguments.positions_groups_count; ++group, record += 1 + 2 * words) {
        ulong any = 0;
        for (ulong word = 0; word != words; ++word) any |= state[word] & record[1 + word];
        ulong const enabled = 0ul -
                              ulong((any != 0) & (record[0] >> kind & 1) & ((record[0] >> 16 & ~ulong(looks)) == 0));
        for (ulong word = 0; word != words; ++word) next[word] |= record[1 + words + word] & enabled;
    }
    for (ulong word = 0; word != words; ++word) next[word] &= reach[word];
}

/**
 *  @brief Steps the position set at @p state over the units of @p haystack that end in
 *      <tt>(stop, from]</tt>, right to left, as @c sz_regex_positions_walk_simt_ does, marking in @p bits
 *      every start it finds in slots <tt>[low, high)</tt> when @p marking.
 *  @return The set holding the result, @p state or @p next.
 */
inline device ulong *sz_regex_positions_walk_metal_(
    thread sz_regex_automata_metal_t const &automata, constant sz_regex_arguments_metal_t &arguments,
    device uint const *looks, thread sz_regex_haystack_metal_t const &haystack, sz_regex_folding_metal_t folded,
    ulong from, ulong stop, device ulong *state, device ulong *next, sz_regex_marking_metal_t marking,
    device ulong *bits, ulong low, ulong high) {
    device uchar const *text = haystack.text;
    ulong const length = haystack.length, first = haystack.first;
    ulong at = from, right_kind = 0;
    sz_utf8_folded_image_metal_t image;
    if (from != length) {
        sz_regex_unit_after_metal_(folded, text, length, from, image);
        right_kind = sz_regex_byte_kind_metal_(sz_regex_image_byte_metal_(image, 0));
    }
    while (at > stop) {
        uint const unit_looks = sz_regex_looks_at_metal_(looks, arguments, first + at);
        uint unit_class = 0, rune = 0, rune_start = 0;
        ulong const begin = folded ? sz_regex_unit_before_metal_(sz_regex_folded_metal_k, text, at, image)
                                   : sz_regex_class_before_metal_(automata, arguments, text, at, unit_class);
        // Units step once and folded ones once per rune, right to left,
        // the last rune reading the unit's lookarounds.
        for (uint end = folded ? image.length : 1; end; end = rune_start) {
            ulong left_kind = sz_regex_byte_kind_metal_(text[at - 1]), slot = first + at;
            if (folded) {
                rune_start = sz_regex_image_rune_metal_(image, end - 1, rune);
                unit_class = sz_regex_rune_class_metal_(automata, arguments, rune);
                left_kind = sz_regex_rune_kind_metal_(rune);
                if (end != image.length) slot = first + begin;
            }
            sz_regex_positions_step_metal_(automata, arguments, state, next, unit_class, left_kind * 4 + right_kind,
                                           slot == first + at ? unit_looks : 0);
            device ulong *const swapped = state;
            state = next, next = swapped;
            if (marking && slot >= low && slot < high && (state[0] >> 1 & 1))
                bits[(slot - low) / 64] |= 1ul << ((slot - low) % 64);
            right_kind = left_kind;
        }
        at = begin;
    }
    return state;
}

/** A PikeVM over the forward NFA carved out of one share of the round block's walks, as
 *  @c sz_regex_pike_t is, and the flag it sets where it would read the lookaround table
 *  past its window. */
struct sz_regex_pike_metal_t {
    device uint *dense, *sparse, *entered;
    device ulong *dense_slots, *entered_slots, *current, *matched, *frames;
    device uint const *looks;
    device atomic_uint *refused;
    ulong count, entered_count, slots_count, end, first;
    uint pattern;
    bool found;
};

/** The PikeVM of the share of thread @p position among the round's walks, laid out for its capture
 *  slots, as <tt>regex/metal.h</tt> sizes it, reading the round's lookaround table for the haystack whose
 *  first slot is @c first. */
inline sz_regex_pike_metal_t sz_regex_pike_carve_metal_(thread sz_regex_round_metal_t const &round,
                                                        constant sz_regex_arguments_metal_t &arguments,
                                                        ulong position) {
    device uchar *walk = round.walks + position * arguments.walk_bytes;
    ulong const states = arguments.nfa_states_count, entered = arguments.nfa_links_count + 1;
    ulong const slots_count = arguments.pike_slots;
    sz_regex_pike_metal_t pike;
    pike.dense_slots = (device ulong *)walk;
    pike.entered_slots = pike.dense_slots + states * slots_count;
    pike.current = pike.entered_slots + entered * slots_count;
    pike.matched = pike.current + slots_count;
    pike.frames = pike.matched + slots_count;
    pike.dense = (device uint *)(pike.frames + 2 * (states + entered));
    pike.sparse = pike.dense + states;
    pike.entered = pike.sparse + states;
    pike.count = 0, pike.entered_count = 0, pike.slots_count = slots_count, pike.end = 0, pike.first = 0;
    pike.looks = round.looks, pike.refused = round.counters + sz_regex_refused_counter_metal_k;
    pike.pattern = 0, pike.found = false;
    return pike;
}

/** The assertions that hold between two units, as @c sz_regex_looks_between_ finds them. */
inline uint sz_regex_looks_between_metal_(uint before, uint after) {
    uint have = sz_regex_is_word_metal_(before) == sz_regex_is_word_metal_(after) ? sz_regex_look_word_inside_metal_k
                                                                                  : sz_regex_look_word_boundary_metal_k;
    if (before == sz_regex_end_of_input_metal_k)
        have |= sz_regex_look_text_start_metal_k | sz_regex_look_line_start_metal_k;
    else if (before == '\n') have |= sz_regex_look_line_start_metal_k;
    if (after == sz_regex_end_of_input_metal_k) have |= sz_regex_look_text_end_metal_k | sz_regex_look_line_end_metal_k;
    else if (after == '\n') have |= sz_regex_look_line_end_metal_k;
    return have;
}

/** Adds every state @p state reaches without consuming to the position's threads, as
 *  @c sz_regex_pike_close_ does. */
inline void sz_regex_pike_close_metal_(thread sz_regex_automata_metal_t const &automata,
                                       constant sz_regex_arguments_metal_t &arguments,
                                       thread sz_regex_pike_metal_t &pike, uint state, device ulong const *slots,
                                       uint have, ulong open, ulong close) {
    ulong const states_count = arguments.nfa_states_count, slots_count = pike.slots_count;
    device ulong *frames = pike.frames, *current = pike.current;
    ulong top = 0;
    for (ulong index = 0; index != slots_count; ++index) current[index] = slots[index];
    frames[top++] = state, frames[top++] = 0;
    while (top) {
        ulong const value = frames[--top], tag = frames[--top];
        if (tag >= states_count) {
            current[tag - states_count] = value;
            continue;
        }
        for (state = uint(tag);;) {
            sz_regex_nfa_state_metal_t const node = automata.states[state];
            uint const held = pike.sparse[state];
            if (held < pike.count && pike.dense[held] == state) break;
            pike.sparse[state] = uint(pike.count), pike.dense[pike.count] = state;
            if (node.kind == sz_regex_nfa_ranges_metal_k || node.kind == sz_regex_nfa_match_metal_k)
                for (ulong index = 0; index != slots_count; ++index)
                    pike.dense_slots[pike.count * slots_count + index] = current[index];
            ++pike.count;
            if (node.kind == sz_regex_nfa_ranges_metal_k || node.kind == sz_regex_nfa_match_metal_k) break;
            if (node.kind == sz_regex_nfa_look_metal_k && !(have & node.look)) break;
            if (node.kind == sz_regex_nfa_empty_metal_k && node.slot < slots_count) {
                frames[top++] = states_count + node.slot, frames[top++] = current[node.slot];
                current[node.slot] = node.slot & 1 ? close : open;
            }
            if (node.kind == sz_regex_nfa_look_metal_k || node.kind == sz_regex_nfa_empty_metal_k) {
                state = node.target;
                continue;
            }
            uint const link = node.first_link;
            if (link == sz_regex_none_metal_k) break;
            ulong const base = top;
            for (uint other = automata.links[link].next; other != sz_regex_none_metal_k;
                 other = automata.links[other].next)
                frames[top++] = automata.links[other].target, frames[top++] = 0;
            // Alternatives pushed in priority order pop in reverse, so the pairs are flipped.
            for (ulong low = base, high = top; low + 2 < high; low += 2, high -= 2) {
                ulong const swapped = frames[low];
                frames[low] = frames[high - 2], frames[high - 2] = swapped;
            }
            state = automata.links[link].target;
        }
    }
}

/**
 *  @brief Runs the PikeVM at one position where the lookarounds of @p looks hold, as
 *      @c sz_regex_pike_step_ does.
 *  @return Whether any thread entered the next.
 */
inline bool sz_regex_pike_step_metal_(thread sz_regex_automata_metal_t const &automata,
                                      constant sz_regex_arguments_metal_t &arguments,
                                      thread sz_regex_pike_metal_t &pike, uint before, uint after, uint looks,
                                      ulong open, ulong close) {
    ulong const slots_count = pike.slots_count;
    uint const have = sz_regex_looks_between_metal_(before, after) | looks;
    pike.count = 0, pike.found = false;
    for (ulong index = 0; index != pike.entered_count; ++index)
        sz_regex_pike_close_metal_(automata, arguments, pike, pike.entered[index],
                                   pike.entered_slots + index * slots_count, have, open, close);
    pike.entered_count = 0;
    for (ulong index = 0; index != pike.count; ++index) {
        sz_regex_nfa_state_metal_t const node = automata.states[pike.dense[index]];
        if (node.kind == sz_regex_nfa_match_metal_k) {
            pike.end = close, pike.pattern = node.target, pike.found = true;
            for (ulong slot = 0; slot != slots_count; ++slot)
                pike.matched[slot] = pike.dense_slots[index * slots_count + slot];
            break;
        }
        if (node.kind != sz_regex_nfa_ranges_metal_k || after == sz_regex_end_of_input_metal_k) continue;
        for (uint link = node.first_link; link != sz_regex_none_metal_k; link = automata.links[link].next) {
            if (after < automata.links[link].low || after > automata.links[link].high) continue;
            pike.entered[pike.entered_count] = automata.links[link].target;
            for (ulong slot = 0; slot != slots_count; ++slot)
                pike.entered_slots[pike.entered_count * slots_count + slot] =
                    pike.dense_slots[index * slots_count + slot];
            ++pike.entered_count;
        }
    }
    return pike.entered_count != 0;
}

/**
 *  @brief Runs the PikeVM from the NFA state @p entry, as @c sz_regex_walk_pike_ does.
 *  @return Whether a match ends there, an empty one counting only at the unit's own start; none
 *      where the walk would read the lookaround table past its window, which it cannot pause at, so
 *      it flags the call refused.
 */
inline bool sz_regex_walk_pike_metal_(thread sz_regex_automata_metal_t const &automata,
                                      constant sz_regex_arguments_metal_t &arguments,
                                      thread sz_regex_pike_metal_t &pike, device uchar const *text, ulong length,
                                      ulong start, ulong offset, uint entry, ulong bound, thread ulong &end,
                                      thread uint &pattern) {
    sz_regex_folding_metal_t const folded = sz_regex_folding_metal_t(arguments.case_sensitivity != 0);
    sz_utf8_folded_image_metal_t image;
    ulong position = start, unit_end = start, byte = offset, last_step = 0;
    bool matched = false;
    uint before = sz_regex_end_of_input_metal_k;
    image.length = 1;
    if (start != length) unit_end = sz_regex_unit_after_metal_(folded, text, length, start, image);
    if (offset) before = sz_regex_image_byte_metal_(image, uint(offset) - 1);
    else if (start) {
        sz_utf8_folded_image_metal_t previous;
        sz_regex_unit_before_metal_(folded, text, start, previous);
        before = sz_regex_image_byte_metal_(previous, previous.length - 1u);
    }
    pike.entered[0] = entry, pike.entered_count = 1;
    for (ulong index = 0; index != pike.slots_count; ++index)
        pike.entered_slots[index] = sz_regex_unset_metal_k, pike.matched[index] = sz_regex_unset_metal_k;
    for (ulong step = 0;; ++step) {
        uint const after = position == length ? sz_regex_end_of_input_metal_k
                                              : uint(sz_regex_image_byte_metal_(image, uint(byte)));
        if (!byte && arguments.lookarounds_count && pike.first + position >= arguments.looks_end) {
            atomic_store_explicit(pike.refused, 1u, memory_order_relaxed);
            return false;
        }
        uint const held = byte ? 0 : sz_regex_looks_at_metal_(pike.looks, arguments, pike.first + position);
        bool const alive = sz_regex_pike_step_metal_(automata, arguments, pike, before, after, held, position,
                                                     byte ? unit_end : position);
        if (pike.found) matched = true, last_step = step;
        if (!alive || (position == bound && byte == 0)) break;
        before = after;
        if (++byte != image.length) continue;
        position = unit_end, byte = 0;
        if (position != length) unit_end = sz_regex_unit_after_metal_(folded, text, length, position, image);
    }
    if (!matched || (offset && last_step == 0)) return false;
    end = pike.end, pattern = pike.pattern;
    return true;
}

/** The match a search takes at candidate @p start without links, each rune its unit folds to tried
 *  in turn from the one @p offset bytes into its folds, which takes the offset found, by the PikeVM
 *  when @p piked, else by the folded DFA, as @c sz_regex_match_at_simt_ does. */
inline bool sz_regex_match_at_metal_(thread sz_regex_table_metal_t const &table,
                                     thread sz_regex_automata_metal_t const &automata,
                                     constant sz_regex_arguments_metal_t &arguments, thread sz_regex_pike_metal_t &pike,
                                     sz_regex_end_walk_metal_t piked, device uchar const *text, ulong length,
                                     ulong start, thread ulong &end, thread uint &pattern, thread ulong &offset) {
    sz_utf8_folded_image_metal_t image;
    uint rune_start = uint(offset);
    image.length = 1, image.rune_ends = 1;
    if (arguments.case_sensitivity != 0 && start != length)
        sz_regex_unit_after_metal_(sz_regex_folded_metal_k, text, length, start, image);
    for (;;) {
        bool const found = piked ? sz_regex_walk_pike_metal_(automata, arguments, pike, text, length, start, rune_start,
                                                             arguments.nfa_start, length, end, pattern)
                                 : sz_regex_walk_folded_metal_(table, automata, arguments, text, pike.first, length,
                                                               start, rune_start, end, pattern);
        if (found && !(end == start && start < length && (text[start] & 0xC0) == 0x80)) {
            offset = rune_start;
            return true;
        }
        uint rune_end = rune_start;
        while (rune_end < image.length && !(image.rune_ends >> rune_end & 1)) ++rune_end;
        rune_start = rune_end + 1;
        if (rune_start >= image.length) return false;
    }
}

/** Fills the matched slots of @p pike with the groups of the match of @p pattern over
 *  <tt>[start, end)</tt>, as @c sz_regex_groups_simt_ does. */
inline void sz_regex_groups_metal_(thread sz_regex_table_metal_t const &table,
                                   thread sz_regex_automata_metal_t const &automata,
                                   constant sz_regex_arguments_metal_t &arguments, thread sz_regex_pike_metal_t &pike,
                                   device uchar const *text, ulong length, ulong start, ulong end, uint pattern,
                                   ulong slots_count) {
    ulong offset = 0, found_end;
    uint found_pattern;
    if (arguments.case_sensitivity != 0) {
        pike.slots_count = 0;
        sz_regex_match_at_metal_(table, automata, arguments, pike,
                                 sz_regex_end_walk_metal_t(arguments.forward_rows == 0), text, length, start, found_end,
                                 found_pattern, offset);
    }
    pike.slots_count = slots_count;
    sz_regex_walk_pike_metal_(automata, arguments, pike, text, length, start, offset, automata.patterns_starts[pattern],
                              end, found_end, found_pattern);
}

/**
 *  @brief Appends one match's replacement, as @c sz_regex_expand_simt_ does, or only measures it
 *      for a null @p target.
 *  @return The bytes the replacement takes, whether or not they fit below @p capacity.
 */
inline ulong sz_regex_expand_metal_(thread sz_regex_table_metal_t const &table,
                                    thread sz_regex_automata_metal_t const &automata,
                                    constant sz_regex_arguments_metal_t &arguments, thread sz_regex_pike_metal_t &pike,
                                    device uchar const *text, ulong length, device uchar const *replacement,
                                    ulong replacement_length, ulong start, ulong end, uint pattern,
                                    device uchar *target, ulong capacity, ulong written) {
    ulong const first_group = automata.patterns_groups[pattern];
    ulong const groups = automata.patterns_groups[pattern + 1] - first_group;
    ulong cursor = 0, appended = 0;
    bool walked = false;
    while (cursor != replacement_length) {
        ulong next = cursor + 1, group = sz_regex_unset_metal_k, number = 0, copied = 1;
        device uchar const *source = replacement + cursor;
        bool referenced = false;
        if (replacement[cursor] == '$' && next != replacement_length) {
            ulong name_begin = next, name_end = next;
            if (replacement[next] == '$') source = replacement + next++;
            else if (replacement[next] == '{') {
                name_begin = name_end = next + 1;
                while (name_end != replacement_length && replacement[name_end] != '}') ++name_end;
                if (name_end != replacement_length && name_end != name_begin) referenced = true, next = name_end + 1;
            }
            else {
                while (name_end != replacement_length && replacement[name_end] >= '0' && replacement[name_end] <= '9')
                    ++name_end;
                if (name_end != name_begin) referenced = true, next = name_end;
            }
            if (referenced) {
                ulong digit = name_begin;
                for (; digit != name_end && replacement[digit] >= '0' && replacement[digit] <= '9'; ++digit)
                    number = number > groups ? number : number * 10 + ulong(replacement[digit] - '0');
                if (digit == name_end && number <= groups) group = number;
                for (ulong named = 0; digit != name_end && named != groups; ++named) {
                    ulong const name_offset = automata.names_offsets[first_group + named];
                    bool same = automata.names_offsets[first_group + named + 1] - name_offset == name_end - name_begin;
                    for (ulong index = 0; same && index != name_end - name_begin; ++index)
                        same = automata.names[name_offset + index] == replacement[name_begin + index];
                    if (same) group = named + 1;
                }
            }
        }
        while (!referenced && next != replacement_length && replacement[next] != '$') ++next, ++copied;
        if (referenced) {
            source = text + start, copied = group == 0 ? end - start : 0;
            if (group != 0 && group != sz_regex_unset_metal_k) {
                if (!walked)
                    sz_regex_groups_metal_(table, automata, arguments, pike, text, length, start, end, pattern,
                                           arguments.pike_slots);
                walked = true;
                if (pike.matched[2 * group] != sz_regex_unset_metal_k &&
                    pike.matched[2 * group + 1] != sz_regex_unset_metal_k)
                    source = text + pike.matched[2 * group],
                    copied = pike.matched[2 * group + 1] - pike.matched[2 * group];
            }
        }
        if (target)
            for (ulong copy = 0; copy != copied && written + appended + copy < capacity; ++copy)
                target[written + appended + copy] = source[copy];
        appended += copied, cursor = next;
    }
    return appended;
}

/** Writes the rows of a haystack's first match of @p pattern from @p written on, as
 *  @c sz_regex_rows_write_simt_ does, or only counts them for a null @p rows. @return The rows. */
inline ulong sz_regex_rows_write_metal_(thread sz_regex_table_metal_t const &table,
                                        thread sz_regex_automata_metal_t const &automata,
                                        constant sz_regex_arguments_metal_t &arguments,
                                        thread sz_regex_pike_metal_t &pike, device uchar const *text, ulong length,
                                        ulong haystack_index, ulong start, ulong end, uint pattern,
                                        device sz_match_metal_t *rows, ulong capacity, ulong written) {
    ulong const groups = automata.patterns_groups[pattern + 1] - automata.patterns_groups[pattern];
    ulong count = 0;
    if (groups)
        sz_regex_groups_metal_(table, automata, arguments, pike, text, length, start, end, pattern, 2 * groups + 2);
    for (ulong group = 0; group <= groups; ++group) {
        device ulong const *slots = pike.matched + 2 * group;
        if (group && (slots[0] == sz_regex_unset_metal_k || slots[1] == sz_regex_unset_metal_k)) continue;
        if (rows && written + count < capacity)
            rows[written + count] = {haystack_index, group, group ? slots[0] : start,
                                     group ? slots[1] - slots[0] : end - start};
        ++count;
    }
    return count;
}

/** Kept matches of the extracted pattern in this segment before @p slot, as
 *  @c sz_regex_counted_before_simt_ counts them. */
inline ulong sz_regex_counted_before_metal_(thread sz_regex_round_metal_t const &round,
                                            constant sz_regex_arguments_metal_t &arguments, ulong slot) {
    ulong const begin = arguments.segment_begin, chunk = (slot - begin) >> arguments.chunk_shift;
    ulong counted = round.chunk_sums[chunk];
    for (uint index = uint(round.chunk_offsets[chunk]);
         index != uint(round.chunk_offsets[chunk + 1]) && begin + round.slots[index] < slot; ++index)
        counted += round.kept[index] && sz_regex_linked_metal_(round, index).pattern == arguments.pattern_index;
    return counted;
}

/** The lookarounds a pass's @p set holds as matched inside the folds of the unit before, as
 *  @c sz_regex_pending_simt_ reads them. */
inline uint sz_regex_pending_metal_(constant sz_regex_arguments_metal_t &arguments, device ulong const *set) {
    uint pending = 0;
    for (uint lookaround = 0; lookaround != arguments.lookarounds_count; ++lookaround) {
        ulong const index = arguments.bodies_count + lookaround;
        if (set[index / 64] >> (index % 64) & 1) pending |= sz_regex_look_lookaround_metal_k << lookaround;
    }
    return pending;
}

/** Adds the lookarounds of @p pending to a pass's @p set,
 *  as @ref sz_regex_pending_metal_ reads them. */
inline void sz_regex_pend_metal_(constant sz_regex_arguments_metal_t &arguments, device ulong *set, uint pending) {
    for (uint lookaround = 0; lookaround != arguments.lookarounds_count; ++lookaround) {
        ulong const index = arguments.bodies_count + lookaround;
        if (pending & sz_regex_look_lookaround_metal_k << lookaround) set[index / 64] |= 1ul << (index % 64);
    }
}

/** Records at @p slot of @p table the members of the pass that hold there,
 *  as @c sz_regex_looks_record_simt_ does. */
inline void sz_regex_looks_record_metal_(device uint *table, constant sz_regex_arguments_metal_t &arguments, ulong slot,
                                         uint matched) {
    if (slot >= arguments.looks_begin && slot < arguments.looks_end)
        table[slot & arguments.looks_mask] |= (matched ^ arguments.negated) & arguments.members;
}

/** Closes the threads of a lookaround pass at one boundary and steps them over @p after into
 *  @p next, as @c sz_regex_looks_step_simt_ does. @return The look bits of the members whose
 *  bodies match there. */
inline uint sz_regex_looks_step_metal_(thread sz_regex_automata_metal_t const &automata,
                                       constant sz_regex_arguments_metal_t &arguments,
                                       thread sz_regex_pike_metal_t &pike, device ulong const *state,
                                       device ulong *next, uint have, uint after) {
    ulong const bodies = arguments.bodies_begin, count = arguments.bodies_count;
    ulong const words = (count + arguments.lookarounds_count + 63) / 64;
    uint matched = 0;
    pike.count = 0;
    for (ulong word = 0; word != words; ++word)
        for (ulong bits = state[word]; bits; bits &= bits - 1) {
            ulong const index = word * 64 + ctz(bits);
            if (index < count)
                sz_regex_pike_close_metal_(automata, arguments, pike, uint(bodies + index), pike.matched, have, 0, 0);
        }
    for (uint lookaround = 0; lookaround != arguments.lookarounds_count; ++lookaround)
        if (arguments.members & sz_regex_look_lookaround_metal_k << lookaround)
            sz_regex_pike_close_metal_(automata, arguments, pike, automata.lookarounds[lookaround].start, pike.matched,
                                       have, 0, 0);
    for (ulong word = 0; word != words; ++word) next[word] = 0;
    for (ulong index = 0; index != pike.count; ++index) {
        sz_regex_nfa_state_metal_t const node = automata.states[pike.dense[index]];
        if (node.kind == sz_regex_nfa_match_metal_k) matched |= sz_regex_look_lookaround_metal_k << node.target;
        if (node.kind != sz_regex_nfa_ranges_metal_k || after == sz_regex_end_of_input_metal_k) continue;
        for (uint link = node.first_link; link != sz_regex_none_metal_k; link = automata.links[link].next)
            if (automata.links[link].low <= after && after <= automata.links[link].high) {
                ulong const target = automata.links[link].target - bodies;
                next[target / 64] |= 1ul << (target % 64);
            }
    }
    return matched;
}

/** Walks one level's lookbehind pass over the unit boundaries of @p haystack in <tt>[from, to)</tt>, and
 *  its end when @p to passes it, as @c sz_regex_behind_walk_simt_ does. @return The set arriving at
 *  @p to, @p state or @p next. */
inline device ulong *sz_regex_behind_walk_metal_(thread sz_regex_automata_metal_t const &automata,
                                                 constant sz_regex_arguments_metal_t &arguments, device uint *looks,
                                                 thread sz_regex_pike_metal_t &pike,
                                                 thread sz_regex_haystack_metal_t const &haystack, ulong from, ulong to,
                                                 device ulong *state, device ulong *next,
                                                 sz_regex_recording_metal_t recording) {
    sz_regex_folding_metal_t const folded = sz_regex_folding_metal_t(arguments.case_sensitivity != 0);
    device uchar const *text = haystack.text;
    ulong const length = haystack.length;
    ulong position = from;
    uint before = sz_regex_end_of_input_metal_k, pending = sz_regex_pending_metal_(arguments, state);
    sz_utf8_folded_image_metal_t image;
    if (from) {
        sz_regex_unit_before_metal_(folded, text, from, image);
        before = sz_regex_image_byte_metal_(image, image.length - 1u);
    }
    while (position < to && position <= length) {
        uint after = sz_regex_end_of_input_metal_k;
        ulong unit_end = length + 1;
        if (position != length)
            unit_end = sz_regex_unit_after_metal_(folded, text, length, position, image),
            after = sz_regex_image_byte_metal_(image, 0);
        uint const matched = sz_regex_looks_step_metal_(
            automata, arguments, pike, state, next,
            sz_regex_looks_between_metal_(before, after) |
                sz_regex_looks_at_metal_(looks, arguments, haystack.first + position),
            after);
        if (recording) sz_regex_looks_record_metal_(looks, arguments, haystack.first + position, matched | pending);
        device ulong *swapped = state;
        state = next, next = swapped, pending = 0;
        for (uint byte = 1; position != length && byte < image.length; ++byte) {
            before = after, after = sz_regex_image_byte_metal_(image, byte);
            pending |= sz_regex_looks_step_metal_(automata, arguments, pike, state, next,
                                                  sz_regex_looks_between_metal_(before, after), after);
            swapped = state, state = next, next = swapped;
        }
        before = after, position = unit_end;
    }
    sz_regex_pend_metal_(arguments, state, pending);
    return state;
}

/** Walks one level's lookahead pass over the unit boundaries of @p haystack in <tt>[to, from)</tt>,
 *  right to left, as @c sz_regex_ahead_walk_simt_ does. @return The set past @p to, @p state
 *  or @p next. */
inline device ulong *sz_regex_ahead_walk_metal_(thread sz_regex_automata_metal_t const &automata,
                                                constant sz_regex_arguments_metal_t &arguments, device uint *looks,
                                                thread sz_regex_pike_metal_t &pike,
                                                thread sz_regex_haystack_metal_t const &haystack, ulong from, ulong to,
                                                device ulong *state, device ulong *next,
                                                sz_regex_recording_metal_t recording) {
    sz_regex_folding_metal_t const folded = sz_regex_folding_metal_t(arguments.case_sensitivity != 0);
    device uchar const *text = haystack.text;
    ulong const length = haystack.length;
    ulong position = length;
    uint before = sz_regex_end_of_input_metal_k, pending = sz_regex_pending_metal_(arguments, state);
    sz_utf8_folded_image_metal_t image;
    if (from <= length) {
        if (from == 0) return state;
        position = sz_regex_unit_before_metal_(folded, text, from, image);
        before = sz_regex_image_byte_metal_(image, 0);
    }
    while (position >= to) {
        uint after = sz_regex_end_of_input_metal_k;
        ulong unit_begin = position;
        if (position)
            unit_begin = sz_regex_unit_before_metal_(folded, text, position, image),
            after = sz_regex_image_byte_metal_(image, image.length - 1u);
        uint const matched = sz_regex_looks_step_metal_(
            automata, arguments, pike, state, next,
            sz_regex_looks_between_metal_(before, after) |
                sz_regex_looks_at_metal_(looks, arguments, haystack.first + position),
            after);
        if (recording) sz_regex_looks_record_metal_(looks, arguments, haystack.first + position, matched | pending);
        device ulong *swapped = state;
        state = next, next = swapped, pending = 0;
        if (!position) break;
        for (uint byte = 1; byte < image.length; ++byte) {
            before = after, after = sz_regex_image_byte_metal_(image, image.length - 1u - byte);
            pending |= sz_regex_looks_step_metal_(automata, arguments, pike, state, next,
                                                  sz_regex_looks_between_metal_(before, after), after);
            swapped = state, state = next, next = swapped;
        }
        before = after, position = unit_begin;
    }
    sz_regex_pend_metal_(arguments, state, pending);
    return state;
}

/** Words of the sets the round's flow walks. */
inline ulong sz_regex_flow_words_metal_(constant sz_regex_arguments_metal_t &arguments) {
    if (arguments.flow == sz_regex_flow_positions_metal_k) return arguments.positions_words;
    if (arguments.body != sz_regex_none_metal_k) return (arguments.body_states + 63) / 64;
    return (arguments.bodies_count + arguments.lookarounds_count + 63) / 64;
}

/** The first position at or past @p position where the round's flow hands sets across, as
 *  @c sz_regex_flow_align_simt_ finds it. */
inline ulong sz_regex_flow_align_metal_(constant sz_regex_arguments_metal_t &arguments,
                                        thread sz_regex_haystack_metal_t const &haystack, ulong position) {
    bool const units = arguments.flow == sz_regex_flow_positions_metal_k || arguments.case_sensitivity != 0;
    return units && position < haystack.length ? sz_regex_boundary_metal_(haystack.text, haystack.length, position)
                                               : position;
}

/** Where @p haystack 's piece of the chunk of slots <tt>[low, high)</tt> hands sets across, as
 *  @c sz_regex_flow_piece_simt_ finds it. */
inline void sz_regex_flow_piece_metal_(constant sz_regex_arguments_metal_t &arguments,
                                       thread sz_regex_haystack_metal_t const &haystack, ulong low, ulong high,
                                       thread ulong &begin, thread ulong &end) {
    begin = low > haystack.first ? sz_regex_flow_align_metal_(arguments, haystack, low - haystack.first) : 0;
    end = sz_regex_flow_align_metal_(arguments, haystack, min(high - haystack.first, haystack.length + 1));
}

/** Fills @p set with the least the round's flow holds anywhere,
 *  as @c sz_regex_flow_least_simt_ does. */
inline void sz_regex_flow_least_metal_(constant sz_regex_arguments_metal_t &arguments, device ulong *set) {
    ulong const words = sz_regex_flow_words_metal_(arguments);
    for (ulong word = 0; word != words; ++word) set[word] = 0;
    if (arguments.flow == sz_regex_flow_positions_metal_k) set[0] = 1;
}

/** Fills @p set with what the round's flow holds at a haystack's edge, as
 *  @c sz_regex_flow_start_simt_ does. */
inline void sz_regex_flow_start_metal_(constant sz_regex_arguments_metal_t &arguments, device ulong *set) {
    sz_regex_flow_least_metal_(arguments, set);
    if (arguments.body != sz_regex_none_metal_k) {
        uint const start = arguments.body_start / (arguments.classes_count + 1);
        set[start / 64] |= 1ul << (start % 64);
    }
}

/** Fills @p set with everything the round's flow may hold, as @c sz_regex_flow_every_simt_ does. */
inline void sz_regex_flow_every_metal_(thread sz_regex_automata_metal_t const &automata,
                                       constant sz_regex_arguments_metal_t &arguments, device ulong *set) {
    ulong const words = sz_regex_flow_words_metal_(arguments);
    if (arguments.flow == sz_regex_flow_positions_metal_k) {
        for (ulong word = 0; word != words; ++word) set[word] = automata.positions_all[word];
        return;
    }
    if (arguments.body != sz_regex_none_metal_k) {
        for (ulong word = 0; word != words; ++word) set[word] = arguments.body_reach[word];
        return;
    }
    for (ulong word = 0; word != words; ++word) set[word] = 0;
    // The bodies follow one another, each closing with the match state naming its lookaround.
    for (ulong state = 0, body = 0; state != arguments.bodies_count; ++state) {
        sz_regex_nfa_state_metal_t const node = automata.states[arguments.bodies_begin + state];
        if (node.kind != sz_regex_nfa_match_metal_k) continue;
        if (arguments.members & sz_regex_look_lookaround_metal_k << node.target)
            for (ulong index = body; index <= state; ++index) set[index / 64] |= 1ul << (index % 64);
        body = state + 1;
    }
    for (uint lookaround = 0; lookaround != arguments.lookarounds_count; ++lookaround)
        if (arguments.members & sz_regex_look_lookaround_metal_k << lookaround) {
            ulong const index = arguments.bodies_count + lookaround;
            set[index / 64] |= 1ul << (index % 64);
        }
}

/** Walks the DFA of the body the round's flow names over @p haystack, a set of its states at a
 *  time, from the set handed across @p from to the one it hands across @p to, as
 *  @c sz_regex_body_walk_simt_ does. @return @p state, holding the set handed on. */
inline device ulong *sz_regex_body_walk_metal_(thread sz_regex_automata_metal_t const &automata,
                                               constant sz_regex_arguments_metal_t &arguments, device uint *looks,
                                               thread sz_regex_haystack_metal_t const &haystack, ulong from, ulong to,
                                               device ulong *state, sz_regex_recording_metal_t recording) {
    uint const columns = arguments.classes_count + 1, words = (arguments.body_states + 63) / 64;
    bool const behind = arguments.flow == sz_regex_flow_behind_metal_k;
    device uchar const *text = haystack.text;
    ulong const length = haystack.length;
    ulong held[sz_regex_body_words_metal_k], next[sz_regex_body_words_metal_k], any = 0;
    ulong position = from;
    for (uint word = 0; word != sz_regex_body_words_metal_k; ++word)
        held[word] = word < words ? state[word] : 0, any |= held[word];
    // A walk holding no state stays empty, and only one that records reads anything on the way.
    if (recording == sz_regex_unrecorded_metal_k && !any) return state;
    if (!behind && from <= length) {
        if (!from) return state;
        position = from - 1;
    }
    else if (!behind) position = length;
    while (behind ? position < to && position <= length : position >= to) {
        uint const column = behind ? (position == length ? columns - 1 : uint(automata.byte_to_class[text[position]]))
                                   : (position ? uint(automata.byte_to_class[text[position - 1]]) : columns - 1);
        bool accepted = false;
        for (uint word = 0; word != sz_regex_body_words_metal_k; ++word) next[word] = 0;
        for (uint word = 0; word != words; ++word)
            for (ulong bits = held[word]; bits; bits &= bits - 1) {
                uint const target = automata.body_rows[(word * 64 + uint(ctz(bits))) * columns + column];
                uint const index = target / columns;
                next[index / 64] |= 1ul << (index % 64);
                accepted = accepted || target >= arguments.body_accepts;
            }
        if (recording == sz_regex_recorded_metal_k)
            sz_regex_looks_record_metal_(looks, arguments, haystack.first + position, accepted ? arguments.members : 0);
        for (uint word = 0; word != sz_regex_body_words_metal_k; ++word) held[word] = next[word];
        if (behind) ++position;
        else if (position) --position;
        else break;
    }
    for (uint word = 0; word != words; ++word) state[word] = held[word];
    return state;
}

/** Walks the round's flow over @p haystack from the set at @p state handed across @p from to
 *  the set it hands across @p to, as @c sz_regex_flow_walk_simt_ does. @return The set
 *  holding the result. */
inline device ulong *sz_regex_flow_walk_metal_(thread sz_regex_automata_metal_t const &automata,
                                               constant sz_regex_arguments_metal_t &arguments, device uint *looks,
                                               thread sz_regex_pike_metal_t &pike,
                                               thread sz_regex_haystack_metal_t const &haystack, ulong from, ulong to,
                                               device ulong *state, device ulong *next,
                                               sz_regex_recording_metal_t recording) {
    if (arguments.body != sz_regex_none_metal_k)
        return sz_regex_body_walk_metal_(automata, arguments, looks, haystack, from, to, state, recording);
    if (arguments.flow == sz_regex_flow_behind_metal_k)
        return sz_regex_behind_walk_metal_(automata, arguments, looks, pike, haystack, from, to, state, next,
                                           recording);
    if (arguments.flow == sz_regex_flow_ahead_metal_k)
        return sz_regex_ahead_walk_metal_(automata, arguments, looks, pike, haystack, from, to, state, next, recording);
    return sz_regex_positions_walk_metal_(
        automata, arguments, looks, haystack, sz_regex_folding_metal_t(arguments.case_sensitivity != 0),
        min(from, haystack.length), to, state, next, sz_regex_unmarked_metal_k, state, 0, 0);
}

/** The undecided elements of @p undecided that @p set holds, a bit each. */
inline ulong sz_regex_decided_metal_(device sz_regex_undecided_metal_t const *undecided, device ulong const *set) {
    ulong bits = 0;
    for (uint index = 0; index != undecided->count; ++index)
        bits |= (set[undecided->indices[index] / 64] >> (undecided->indices[index] % 64) & 1) << index;
    return bits;
}

/** The rows of @p rows that the bits of @p bits pick, united. */
inline ulong sz_regex_apply_metal_(thread ulong const *rows, ulong bits) {
    ulong united = 0;
    for (; bits; bits &= bits - 1) united |= rows[ctz(bits)];
    return united;
}

/** The haystack whose set a chunk hands downstream, and whether it hands one, as
 *  @c sz_regex_flow_handing_simt_ finds it. */
inline bool sz_regex_flow_handing_metal_(sz_regex_corpus_metal_t corpus, constant sz_regex_arguments_metal_t &arguments,
                                         ulong low, ulong high, thread sz_regex_haystack_metal_t &haystack) {
    bool const backward = arguments.flow != sz_regex_flow_behind_metal_k;
    haystack = sz_regex_haystack_metal_(corpus, sz_regex_haystack_at_metal_(corpus, backward ? low : high - 1, 0));
    return backward ? haystack.first < low : haystack.first + haystack.length >= high;
}

/** The threadgroup's exclusive prefix sum of @p value, its own total left in @p total, over
 *  @p shared of one entry per thread. */
inline ulong sz_regex_block_scan_metal_(ulong value, threadgroup ulong *shared, uint lane, uint width,
                                        thread ulong &total) {
    shared[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 1; offset < width; offset *= 2) {
        ulong const addend = lane >= offset ? shared[lane - offset] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[lane] += addend;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    ulong const inclusive = shared[lane];
    total = shared[width - 1];
    threadgroup_barrier(mem_flags::mem_threadgroup); // ! The caller reuses `shared` for the next tile.
    return inclusive - value;
}

/** Zeroes @c clear_words words at @c target, which is how a round clears what it reads first. */
kernel void sz_regex_clear_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                         constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                         uint position [[thread_position_in_grid]], uint threads [[threads_per_grid]]) {
    device ulong *words = sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.target);
    for (ulong index = position; index < arguments.clear_words; index += threads) words[index] = 0;
}

/** Reduces one threadgroup's own contiguous tile of the scanned array into @c tile_sums. */
kernel void sz_regex_scan_reduce_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                               constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                               uint group [[threadgroup_position_in_grid]],
                                               uint lane [[thread_index_in_threadgroup]],
                                               uint width [[threads_per_threadgroup]]) {
    threadgroup ulong shared[sz_regex_threads_metal_k];
    device ulong const *values = sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.target);
    device ulong *tile_sums = sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.tile_sums);
    ulong const count = arguments.scan_count;
    ulong const per_tile = sz_size_divide_round_up_metal_<ulong>(count, arguments.scan_tiles);
    ulong const begin = min((ulong)group * per_tile, count), end = min(begin + per_tile, count);
    ulong running = 0, total;
    for (ulong first = begin; first < end; first += width) {
        ulong const index = first + lane;
        sz_regex_block_scan_metal_(index < end ? values[index] : 0, shared, lane, width, total);
        running += total;
    }
    if (lane == 0) tile_sums[group] = running;
}

/** Scans @c tile_sums in place on one threadgroup. */
kernel void sz_regex_scan_carry_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                              constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                              uint lane [[thread_index_in_threadgroup]],
                                              uint width [[threads_per_threadgroup]]) {
    threadgroup ulong shared[sz_regex_threads_metal_k];
    device ulong *tile_sums = sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.tile_sums);
    ulong carry = 0, total;
    for (ulong first = 0; first < arguments.scan_tiles; first += width) {
        ulong const index = first + lane;
        ulong const exclusive = sz_regex_block_scan_metal_(index < arguments.scan_tiles ? tile_sums[index] : 0, shared,
                                                           lane, width, total);
        if (index < arguments.scan_tiles) tile_sums[index] = carry + exclusive;
        carry += total;
    }
}

/** Scans one threadgroup's tile of the scanned array in place, seeded by the base the carry set. */
kernel void sz_regex_scan_apply_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                              constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                              uint group [[threadgroup_position_in_grid]],
                                              uint lane [[thread_index_in_threadgroup]],
                                              uint width [[threads_per_threadgroup]]) {
    threadgroup ulong shared[sz_regex_threads_metal_k];
    device ulong *values = sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.target);
    device ulong const *tile_sums = sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.tile_sums);
    ulong const count = arguments.scan_count;
    ulong const per_tile = sz_size_divide_round_up_metal_<ulong>(count, arguments.scan_tiles);
    ulong const begin = min((ulong)group * per_tile, count), end = min(begin + per_tile, count);
    ulong running = tile_sums[group], total;
    for (ulong first = begin; first < end; first += width) {
        ulong const index = first + lane;
        ulong const exclusive = sz_regex_block_scan_metal_(index < end ? values[index] : 0, shared, lane, width, total);
        if (index < end) values[index] = running + exclusive;
        running += total;
    }
}

/** Whether a chunk's flow left elements undecided that an affine map answers,
 *  rather than none or too many. */
inline bool sz_regex_mapping_metal_(uint count) { return count - 1 < sz_regex_undecided_metal_k; }

/**
 *  @brief Where the round's flow over @p haystack starts exactly within the segment, in @p edge:
 *      the haystack's own edge upstream, or the segment's where the haystack runs past it, as
 *      @c sz_regex_flow_edge_simt_ finds it.
 *  @return The set the segment edge keeps, or null at the haystack's edge, where the flow starts
 *      from its least.
 */
inline device ulong const *sz_regex_flow_edge_metal_(thread sz_regex_round_metal_t const &round,
                                                     constant sz_regex_arguments_metal_t &arguments,
                                                     thread sz_regex_haystack_metal_t const &haystack,
                                                     thread ulong &edge) {
    bool const backward = arguments.flow != sz_regex_flow_behind_metal_k;
    if (backward && haystack.first + haystack.length >= arguments.segment_end) {
        edge = sz_regex_flow_align_metal_(arguments, haystack, arguments.segment_end - haystack.first);
        return round.handoff_edges + (arguments.segment + 1) * arguments.set_words;
    }
    if (!backward && haystack.first < arguments.segment_begin) {
        edge = sz_regex_flow_align_metal_(arguments, haystack, arguments.segment_begin - haystack.first);
        return round.handoff_edges + arguments.segment * arguments.set_words;
    }
    edge = backward ? haystack.length + 1 : 0;
    return nullptr;
}

/**
 *  @brief Bounds the set each chunk's walk of the round's flow hands downstream, one thread per
 *      chunk in each thread's share of the walks, as @c sz_regex_bound_simt_kernel_ does.
 *
 *  A walk widening upstream stops at the haystack's edge or the segment's, where the set it starts
 *  from is known, so no walk leaves its segment; one still undecided past
 *  @ref sz_regex_widening_bytes_metal_k marks the chunk wide, one element more than a map holds,
 *  for the pieces kernel.
 */
kernel void sz_regex_bound_metal_kernel_(device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
                                         device uchar const *haystacks [[buffer(2)]],
                                         device uchar const *haystacks_offsets [[buffer(3)]],
                                         constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                         uint position [[thread_position_in_grid]]) {
    sz_regex_automata_metal_t const automata = sz_regex_automata_metal_(engine, arguments);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    ulong const width = arguments.chunk_width, stride = arguments.set_words;
    ulong const words = sz_regex_flow_words_metal_(arguments);
    bool const backward = arguments.flow != sz_regex_flow_behind_metal_k;
    if (position >= arguments.walks_count) return;
    sz_regex_pike_metal_t pike = sz_regex_pike_carve_metal_(round, arguments, position);
    device ulong *sets = (device ulong *)(round.walks + position * arguments.walk_bytes + arguments.walk_sets);
    pike.slots_count = 0;
    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += arguments.walks_count) {
        ulong const low = arguments.segment_begin + chunk * width, high = min(low + width, arguments.segment_end);
        device sz_regex_undecided_metal_t *undecided = round.undecided + chunk;
        device ulong *handoff = round.handoffs + chunk * stride;
        sz_regex_haystack_metal_t haystack;
        undecided->count = 0;
        if (!sz_regex_flow_handing_metal_(corpus, arguments, low, high, haystack)) continue;
        ulong begin, end, edge;
        sz_regex_flow_piece_metal_(arguments, haystack, low, high, begin, end);
        device ulong const *kept = sz_regex_flow_edge_metal_(round, arguments, haystack, edge);
        ulong from = backward ? end : begin, to = backward ? begin : end;
        for (ulong span = width;; span *= 2) {
            if (from == edge && kept)
                for (ulong word = 0; word != words; ++word) sets[word] = kept[word];
            else if (from == edge) sz_regex_flow_start_metal_(arguments, sets);
            else sz_regex_flow_least_metal_(arguments, sets);
            device ulong *result = sz_regex_flow_walk_metal_(automata, arguments, round.looks, pike, haystack, from, to,
                                                             sets, sets + stride, sz_regex_unrecorded_metal_k);
            for (ulong word = 0; word != words; ++word) handoff[word] = result[word];
            if (from == edge) break;
            if (span >= sz_regex_widening_bytes_metal_k) {
                undecided->count = sz_regex_undecided_metal_k + 1;
                break;
            }
            sz_regex_flow_every_metal_(automata, arguments, sets);
            result = sz_regex_flow_walk_metal_(automata, arguments, round.looks, pike, haystack, from, to, sets,
                                               sets + stride, sz_regex_unrecorded_metal_k);
            ulong count = 0;
            for (ulong word = 0; word != words; ++word) count += popcount(result[word] & ~handoff[word]);
            // A body's DFA holds one state, so one it reaches from every state is the answer.
            if (arguments.body != sz_regex_none_metal_k && count == 1) {
                for (ulong word = 0; word != words; ++word) handoff[word] = result[word];
                break;
            }
            if (count <= sz_regex_undecided_metal_k) {
                count = 0;
                for (ulong word = 0; word != words; ++word)
                    for (ulong bits = result[word] & ~handoff[word]; bits; bits &= bits - 1)
                        undecided->indices[count++] = uint(word * 64 + ctz(bits));
                undecided->count = uint(count);
                break;
            }
            // More elements stay undecided than a composition holds,
            // so the walk starts further upstream.
            if (backward) from = sz_regex_flow_align_metal_(arguments, haystack, min(from + span, edge));
            else from = from > edge + span ? sz_regex_flow_align_metal_(arguments, haystack, from - span) : edge;
        }
    }
}

/**
 *  @brief Settles every wide chunk of the round's flow, one thread per run of unsettled chunks
 *      holding one in each thread's share of the walks, as
 *      @c sz_regex_flow_pieces_simt_kernel_ does.
 *
 *  The run's upstream-most wide chunk walks its haystack downstream from the nearest exact set
 *  upstream of it: a decided chunk's handoff or the set the segment's edge keeps, so no walk leaves
 *  its segment. It writes the handoff of every chunk it passes up to the run's downstream-most wide
 *  one, and the maps between them read those as they read any decided chunk's.
 */
kernel void sz_regex_flow_pieces_metal_kernel_(device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
                                               device uchar const *haystacks [[buffer(2)]],
                                               device uchar const *haystacks_offsets [[buffer(3)]],
                                               constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                               uint position [[thread_position_in_grid]]) {
    sz_regex_automata_metal_t const automata = sz_regex_automata_metal_(engine, arguments);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    ulong const width = arguments.chunk_width, stride = arguments.set_words;
    ulong const words = sz_regex_flow_words_metal_(arguments);
    bool const backward = arguments.flow != sz_regex_flow_behind_metal_k;
    if (position >= arguments.walks_count) return;
    sz_regex_pike_metal_t pike = sz_regex_pike_carve_metal_(round, arguments, position);
    device ulong *sets = (device ulong *)(round.walks + position * arguments.walk_bytes + arguments.walk_sets);
    pike.slots_count = 0;
    long const step = backward ? -1 : 1, chunks = long(arguments.chunks_count);
    for (long chunk = long(position); chunk < chunks; chunk += long(arguments.walks_count)) {
        if (round.undecided[chunk].count <= sz_regex_undecided_metal_k) continue;
        ulong const low = arguments.segment_begin + ulong(chunk) * width;
        sz_regex_haystack_metal_t haystack;
        sz_regex_flow_handing_metal_(corpus, arguments, low, min(low + width, arguments.segment_end), haystack);

        // Upstream past maps only, as a wide chunk further up owns the run;
        // past the segment's chunks lies its edge.
        long anchor = chunk;
        bool owner = true;
        for (;;) {
            anchor -= step;
            if (anchor < 0 || anchor == chunks) break;
            uint const count = round.undecided[anchor].count;
            owner = count <= sz_regex_undecided_metal_k;
            if (!sz_regex_mapping_metal_(count)) break;
        }
        if (!owner) continue;
        ulong edge;
        device ulong const *known = anchor < 0 || anchor == chunks
                                        ? sz_regex_flow_edge_metal_(round, arguments, haystack, edge)
                                        : round.handoffs + ulong(anchor) * stride;

        // Downstream to the run's last wide chunk, before a decided chunk
        // or one handing another haystack's set.
        long last = chunk;
        for (long next = chunk + step; next >= 0 && next < chunks; next += step) {
            ulong const next_low = arguments.segment_begin + ulong(next) * width;
            bool const handing = backward ? haystack.first < next_low
                                          : haystack.first + haystack.length >= next_low + width;
            uint const count = round.undecided[next].count;
            if (!handing || !count) break;
            if (count > sz_regex_undecided_metal_k) last = next;
        }

        // Each chunk's piece walks on from where its upstream neighbour hands its set across.
        device ulong *state = sets, *spare = sets + stride;
        for (ulong word = 0; word != words; ++word) state[word] = known[word];
        ulong from = 0;
        for (long walked = anchor + step;; walked += step) {
            ulong const walked_low = arguments.segment_begin + ulong(walked) * width;
            ulong begin, end;
            sz_regex_flow_piece_metal_(arguments, haystack, walked_low, min(walked_low + width, arguments.segment_end),
                                       begin, end);
            if (walked == anchor + step) from = backward ? end : begin;
            device ulong *result = sz_regex_flow_walk_metal_(automata, arguments, round.looks, pike, haystack, from,
                                                             backward ? begin : end, state, spare,
                                                             sz_regex_unrecorded_metal_k);
            spare = result == state ? spare : state, state = result;
            for (ulong word = 0; word != words; ++word) round.handoffs[ulong(walked) * stride + word] = state[word];
            from = backward ? begin : end;
            if (walked == last) break;
        }
    }
}

/** Maps the undecided elements each chunk hands on from those its upstream neighbour hands it,
 *  one thread per chunk in each thread's share of the walks, as
 *  @c sz_regex_affine_simt_kernel_ does. */
kernel void sz_regex_affine_metal_kernel_(device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
                                          device uchar const *haystacks [[buffer(2)]],
                                          device uchar const *haystacks_offsets [[buffer(3)]],
                                          constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                          uint position [[thread_position_in_grid]]) {
    sz_regex_automata_metal_t const automata = sz_regex_automata_metal_(engine, arguments);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    ulong const width = arguments.chunk_width, stride = arguments.set_words;
    ulong const words = sz_regex_flow_words_metal_(arguments);
    bool const backward = arguments.flow != sz_regex_flow_behind_metal_k;
    if (position >= arguments.walks_count) return;
    sz_regex_pike_metal_t pike = sz_regex_pike_carve_metal_(round, arguments, position);
    device ulong *sets = (device ulong *)(round.walks + position * arguments.walk_bytes + arguments.walk_sets);
    pike.slots_count = 0;
    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += arguments.walks_count) {
        device sz_regex_undecided_metal_t const *undecided = round.undecided + chunk;
        if (!sz_regex_mapping_metal_(undecided->count)) continue;
        ulong const low = arguments.segment_begin + chunk * width, high = min(low + width, arguments.segment_end);
        bool const edge = backward ? chunk + 1 == arguments.chunks_count : chunk == 0;
        ulong const upstream = backward ? chunk + 1 : chunk - 1;
        device sz_regex_affine_metal_t *affine = round.affines + chunk;
        device ulong const *known = nullptr;
        sz_regex_haystack_metal_t haystack;
        sz_regex_flow_handing_metal_(corpus, arguments, low, high, haystack);
        ulong begin, end;
        sz_regex_flow_piece_metal_(arguments, haystack, low, high, begin, end);
        ulong const from = backward ? end : begin, to = backward ? begin : end;
        // A wide upstream chunk is known too, the pieces kernel having walked its set exactly.
        if (edge) known = round.handoff_edges + (backward ? arguments.segment + 1 : arguments.segment) * stride;
        else if (!sz_regex_mapping_metal_(round.undecided[upstream].count)) known = round.handoffs + upstream * stride;
        for (ulong word = 0; word != words; ++word)
            sets[word] = known ? known[word] : round.handoffs[upstream * stride + word];
        device ulong *result = sz_regex_flow_walk_metal_(automata, arguments, round.looks, pike, haystack, from, to,
                                                         sets, sets + stride, sz_regex_unrecorded_metal_k);
        affine->fixed = sz_regex_decided_metal_(undecided, result);
        affine->target = known ? sz_regex_none_metal_k : uint(upstream);
        affine->rows_count = known ? 0 : round.undecided[upstream].count;
        for (uint row = 0; row != affine->rows_count; ++row) {
            uint const index = round.undecided[upstream].indices[row];
            sz_regex_flow_least_metal_(arguments, sets);
            sets[index / 64] |= 1ul << (index % 64);
            result = sz_regex_flow_walk_metal_(automata, arguments, round.looks, pike, haystack, from, to, sets,
                                               sets + stride, sz_regex_unrecorded_metal_k);
            affine->rows[row] = sz_regex_decided_metal_(undecided, result);
        }
    }
}

/** One round of pointer doubling over the chunks' affine maps,
 *  as @c sz_regex_affine_double_simt_kernel_ does. */
kernel void sz_regex_affine_double_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                 constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                                 uint position [[thread_position_in_grid]],
                                                 uint threads [[threads_per_grid]]) {
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    bool const even = arguments.round % 2 == 0, last = arguments.round + 1 == arguments.rounds;
    bool const backward = arguments.flow != sz_regex_flow_behind_metal_k;
    device sz_regex_affine_metal_t const *source = even ? round.affines : round.affines_next;
    device sz_regex_affine_metal_t *target = even ? round.affines_next : round.affines;
    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += threads) {
        device sz_regex_undecided_metal_t const *undecided = round.undecided + chunk;
        device ulong *handoff = round.handoffs + chunk * arguments.set_words;
        if (sz_regex_mapping_metal_(undecided->count)) {
            sz_regex_affine_metal_t affine = source[chunk];
            ulong rows[sz_regex_undecided_metal_k];
            for (uint hop = 1; hop != sz_regex_round_hops_metal_k && affine.target != sz_regex_none_metal_k; ++hop) {
                device sz_regex_affine_metal_t const *next = source + affine.target;
                affine.fixed |= sz_regex_apply_metal_(affine.rows, next->fixed);
                for (uint row = 0; next->target != sz_regex_none_metal_k && row != next->rows_count; ++row)
                    rows[row] = sz_regex_apply_metal_(affine.rows, next->rows[row]);
                for (uint row = 0; next->target != sz_regex_none_metal_k && row != next->rows_count; ++row)
                    affine.rows[row] = rows[row];
                affine.target = next->target, affine.rows_count = next->rows_count;
            }
            target[chunk] = affine;
            if (last)
                for (uint index = 0; index != undecided->count; ++index)
                    if (affine.fixed >> index & 1)
                        handoff[undecided->indices[index] / 64] |= 1ul << (undecided->indices[index] % 64);
        }
        if (last && chunk == (backward ? 0 : arguments.chunks_count - 1))
            for (ulong word = 0; word != arguments.set_words; ++word)
                round.handoff_edges[(backward ? arguments.segment : arguments.segment + 1) * arguments.set_words +
                                    word] = handoff[word];
    }
}

/** Walks each chunk's pieces from the sets handed to them, recording where the lookarounds of the
 *  round's pass hold, one thread per chunk in each thread's share of the walks, as
 *  @c sz_regex_looks_simt_kernel_ does. */
kernel void sz_regex_looks_metal_kernel_(device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
                                         device uchar const *haystacks [[buffer(2)]],
                                         device uchar const *haystacks_offsets [[buffer(3)]],
                                         constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                         uint position [[thread_position_in_grid]]) {
    sz_regex_automata_metal_t const automata = sz_regex_automata_metal_(engine, arguments);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    ulong const width = arguments.chunk_width, stride = arguments.set_words;
    ulong const words = sz_regex_flow_words_metal_(arguments);
    bool const backward = arguments.flow != sz_regex_flow_behind_metal_k;
    if (position >= arguments.walks_count) return;
    sz_regex_pike_metal_t pike = sz_regex_pike_carve_metal_(round, arguments, position);
    device ulong *sets = (device ulong *)(round.walks + position * arguments.walk_bytes + arguments.walk_sets);
    pike.slots_count = 0;
    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += arguments.walks_count) {
        ulong const low = arguments.segment_begin + chunk * width, high = min(low + width, arguments.segment_end);
        for (ulong index = sz_regex_haystack_at_metal_(corpus, low, 0); index < corpus.count; ++index) {
            sz_regex_haystack_metal_t const haystack = sz_regex_haystack_metal_(corpus, index);
            if (haystack.first >= high) break;
            ulong begin, end;
            sz_regex_flow_piece_metal_(arguments, haystack, low, high, begin, end);
            device ulong const *handed = nullptr;
            if (backward && end <= haystack.length)
                handed = chunk + 1 == arguments.chunks_count ? round.handoff_edges + (arguments.segment + 1) * stride
                                                             : round.handoffs + (chunk + 1) * stride;
            else if (!backward && begin)
                handed = chunk == 0 ? round.handoff_edges + arguments.segment * stride
                                    : round.handoffs + (chunk - 1) * stride;
            if (handed)
                for (ulong word = 0; word != words; ++word) sets[word] = handed[word];
            else sz_regex_flow_start_metal_(arguments, sets);
            sz_regex_flow_walk_metal_(automata, arguments, round.looks, pike, haystack, backward ? end : begin,
                                      backward ? begin : end, sets, sets + stride, sz_regex_recorded_metal_k);
        }
    }
}

/** Writes the one-byte lookarounds holding at every slot of one chunk into the segment's half of
 *  the table, negated ones flipped, one thread per chunk, as @c sz_regex_singles_simt_kernel_
 *  does, for the passes to add to. */
kernel void sz_regex_singles_metal_kernel_(device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
                                           device uchar const *haystacks [[buffer(2)]],
                                           device uchar const *haystacks_offsets [[buffer(3)]],
                                           constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                           uint position [[thread_position_in_grid]],
                                           uint threads [[threads_per_grid]]) {
    sz_regex_automata_metal_t const automata = sz_regex_automata_metal_(engine, arguments);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    device uint const *singles = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.looks_bytes);
    device uint const *after = singles + 1, *before = singles + 257;
    uint flipped = 0;
    for (uint index = 0; index != arguments.lookarounds_count; ++index)
        if (automata.lookarounds[index].kind & 1) flipped |= sz_regex_look_lookaround_metal_k << index;
    flipped &= singles[0];
    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += threads) {
        ulong const low = arguments.segment_begin + chunk * arguments.chunk_width;
        ulong const high = min(low + arguments.chunk_width, arguments.segment_end);
        ulong index = sz_regex_haystack_at_metal_(corpus, low, 0);
        sz_regex_haystack_metal_t haystack = sz_regex_haystack_metal_(corpus, index);
        for (ulong slot = low; slot != high; ++slot) {
            if (slot > haystack.first + haystack.length) haystack = sz_regex_haystack_metal_(corpus, ++index);
            ulong const at = slot - haystack.first;
            uint const held = (at ? after[haystack.text[at - 1]] : 0) |
                              (at != haystack.length ? before[haystack.text[at]] : 0);
            round.looks[slot & arguments.looks_mask] = held ^ flipped;
        }
    }
}

/** Primes every chunk whose last haystack runs past its right edge, one thread per chunk, leaving
 *  the state in @c entries, or, when a chunk's width of priming does not collapse, the uncollapsed
 *  set and where each of its states would leave the chunk, or, when no column narrows within its
 *  reach, a frontier one wider than a priming set holds for the pieces kernel. The first chunk's
 *  entry also lands in the segment's edge, which priming the segment before reads. */
kernel void sz_regex_prime_metal_kernel_(device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
                                         device uchar const *haystacks [[buffer(2)]],
                                         device uchar const *haystacks_offsets [[buffer(3)]],
                                         constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                         uint position [[thread_position_in_grid]], uint threads [[threads_per_grid]],
                                         threadgroup uchar *staged [[threadgroup(0)]],
                                         uint lane [[thread_index_in_threadgroup]],
                                         uint group_width [[threads_per_threadgroup]]) {
    sz_regex_table_metal_t const table = sz_regex_stage_metal_(engine, scratch, arguments, arguments.reverse_rows,
                                                               arguments.reverse_cells, staged, lane, group_width);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    ulong const begin = arguments.segment_begin, width = arguments.chunk_width;
    bool const folded = arguments.case_sensitivity != 0;

    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += threads) {
        ulong const low = begin + chunk * width, high = min(low + width, arguments.segment_end);
        round.frontiers[chunk].count = 0;
        if (high == arguments.slots_count) continue;
        sz_regex_haystack_metal_t const haystack = sz_regex_haystack_metal_(
            corpus, sz_regex_haystack_at_metal_(corpus, high - 1, 0));
        if (haystack.first + haystack.length < high) continue;
        uint set[sz_regex_priming_states_metal_k], count;
        // A folded walk's chunk edges move to the next unit boundary,
        // where it steps the folds of whole units.
        ulong const edge = folded ? sz_regex_boundary_metal_(haystack.text, haystack.length, high - haystack.first)
                                  : high - haystack.first - 1;
        uint const state = folded ? sz_regex_prime_folded_metal_(table, arguments, round.reverse_images, haystack.text,
                                                                 haystack.first, haystack.length, edge,
                                                                 sz_regex_priming_bytes_metal_k, set, count)
                                  : sz_regex_prime_metal_(table, arguments, round.reverse_images, haystack.text,
                                                          haystack.first, haystack.length, edge,
                                                          sz_regex_priming_bytes_metal_k, set, count);
        round.entries[chunk] = state;
        if (chunk == 0) round.edges[arguments.segment] = state;
        if (state) continue;

        device sz_regex_frontier_metal_t *frontier = round.frontiers + chunk;
        frontier->count = count ? count : sz_regex_priming_states_metal_k + 1;
        if (!count) continue;

        // Every state of the set walks the chunk on its own, as the chunk left of it
        // depends on where each leaves.
        for (uint index = 0; index != count; ++index) frontier->states[index] = set[index];
        if (haystack.first < low && folded)
            for (ulong at = edge, stop = sz_regex_boundary_metal_(haystack.text, haystack.length, low - haystack.first);
                 at > stop;) {
                sz_utf8_folded_image_metal_t image;
                uint block = sz_regex_block_at_metal_(table, haystack.first + at);
                at = sz_regex_unit_before_metal_(sz_regex_folded_metal_k, haystack.text, at, image);
                for (uint byte = image.length; byte--; block = 0) {
                    uint const column = block + table.classes[sz_regex_image_byte_metal_(image, byte)];
                    for (uint index = 0; index != count; ++index)
                        if (set[index]) set[index] = sz_regex_move_metal_(table, set[index], column);
                }
            }
        else if (haystack.first < low)
            for (ulong at = edge; at != low - haystack.first - 1; --at) {
                uint const column = sz_regex_block_at_metal_(table, haystack.first + at) +
                                    table.classes[haystack.text[at - 1]];
#pragma unroll
                for (uint index = 0; index != sz_regex_priming_states_metal_k; ++index)
                    if (index < count && set[index]) set[index] = sz_regex_move_metal_(table, set[index], column);
            }
        for (uint index = 0; index != count; ++index) frontier->exits[index] = set[index];
    }
}

/** Whether a chunk's priming left a frontier to compose, not its entry or a wide stretch. */
inline bool sz_regex_composing_metal_(uint count) { return count - 1 < sz_regex_priming_states_metal_k; }

/** Where a priming walk over @p haystack stands at the right edge
 *  of the chunk ending at slot @p high. */
inline ulong sz_regex_prime_edge_metal_(sz_regex_folding_metal_t folded,
                                        thread sz_regex_haystack_metal_t const &haystack, ulong high) {
    return folded ? sz_regex_boundary_metal_(haystack.text, haystack.length, high - haystack.first)
                  : high - haystack.first - 1;
}

/**
 *  @brief Settles every wide chunk, one thread per run of unsettled chunks holding one, as
 *      @c sz_regex_prime_pieces_simt_kernel_ does.
 *
 *  The run's rightmost wide chunk walks its haystack right to left from the nearest exact state
 *  right of it: a settled chunk's entry, the reverse start where the haystack ends, or the next
 *  segment's edge, so no walk leaves its segment. It writes the entry of every chunk it passes down
 *  to the run's leftmost wide one, and the frontiers between them compose onto those entries as
 *  onto any settled chunk's.
 */
kernel void sz_regex_prime_pieces_metal_kernel_(
    device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
    device uchar const *haystacks [[buffer(2)]], device uchar const *haystacks_offsets [[buffer(3)]],
    constant sz_regex_arguments_metal_t &arguments [[buffer(10)]], uint position [[thread_position_in_grid]],
    uint threads [[threads_per_grid]], threadgroup uchar *staged [[threadgroup(0)]],
    uint lane [[thread_index_in_threadgroup]], uint group_width [[threads_per_threadgroup]]) {
    sz_regex_table_metal_t const table = sz_regex_stage_metal_(engine, scratch, arguments, arguments.reverse_rows,
                                                               arguments.reverse_cells, staged, lane, group_width);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    ulong const begin = arguments.segment_begin, width = arguments.chunk_width;
    sz_regex_folding_metal_t const folded = sz_regex_folding_metal_t(arguments.case_sensitivity != 0);

    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += threads) {
        if (round.frontiers[chunk].count <= sz_regex_priming_states_metal_k) continue;
        ulong const high = min(begin + (chunk + 1) * width, arguments.segment_end);
        sz_regex_haystack_metal_t const haystack = sz_regex_haystack_metal_(
            corpus, sz_regex_haystack_at_metal_(corpus, high - 1, 0));
        ulong const last = haystack.first + haystack.length;

        // The anchor right of the chunk, past frontiers only,
        // as a wide chunk further right owns the run.
        ulong next = chunk + 1, at = 0;
        uint state = 0;
        bool owner = true;
        for (;; ++next) {
            ulong const next_high = min(begin + (next + 1) * width, arguments.slots_count);
            uint const count = next == arguments.chunks_count ? 0 : round.frontiers[next].count;
            if (last < next_high) {
                state = arguments.reverse_starts[0], at = haystack.length;
                break;
            }
            if (count > sz_regex_priming_states_metal_k) {
                owner = false;
                break;
            }
            if (count) continue;
            state = next == arguments.chunks_count ? round.edges[arguments.segment + 1] : round.entries[next];
            at = sz_regex_prime_edge_metal_(folded, haystack, next_high);
            break;
        }
        if (!owner) continue;

        // The leftmost wide chunk of the run, which ends where a settled chunk
        // or the haystack's start is.
        ulong leftmost = chunk;
        for (ulong previous = chunk; previous-- != 0;) {
            uint const count = round.frontiers[previous].count;
            if (!count || begin + (previous + 1) * width <= haystack.first) break;
            if (count > sz_regex_priming_states_metal_k) leftmost = previous;
        }
        for (ulong walked = next; walked-- != leftmost;) {
            ulong const edge = sz_regex_prime_edge_metal_(folded, haystack, begin + (walked + 1) * width);
            state = folded ? sz_regex_walk_folded_back_metal_(table, state, haystack.text, haystack.first, at, edge)
                           : sz_regex_walk_back_metal_(table, state, haystack.text, haystack.first, at, edge);
            round.entries[walked] = state, at = edge;
            if (walked == 0) round.edges[arguments.segment] = state;
        }
    }
}

/** Points every unsettled chunk at what settles it: the chunk right of it when that one is
 *  unsettled too, mapping its frontier into this one's, or else the state it enters with, found by
 *  walking the settled chunk right of it from its own entry, the next segment's edge past the last.
 *  Wide chunks count as settled, the pieces kernel having written their entries. */
kernel void sz_regex_compose_metal_kernel_(device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
                                           device uchar const *haystacks [[buffer(2)]],
                                           device uchar const *haystacks_offsets [[buffer(3)]],
                                           constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                           uint position [[thread_position_in_grid]], uint threads [[threads_per_grid]],
                                           threadgroup uchar *staged [[threadgroup(0)]],
                                           uint lane [[thread_index_in_threadgroup]],
                                           uint group_width [[threads_per_threadgroup]]) {
    sz_regex_table_metal_t const table = sz_regex_stage_metal_(engine, scratch, arguments, arguments.reverse_rows,
                                                               arguments.reverse_cells, staged, lane, group_width);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    ulong const begin = arguments.segment_begin, width = arguments.chunk_width;
    sz_regex_folding_metal_t const folded = sz_regex_folding_metal_t(arguments.case_sensitivity != 0);

    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += threads) {
        uint const count = round.frontiers[chunk].count;
        if (!sz_regex_composing_metal_(count)) continue;
        uint states[sz_regex_priming_states_metal_k];
        for (uint index = 0; index != count; ++index) states[index] = round.frontiers[chunk].states[index];
        sz_regex_composition_metal_t composition;
        composition.target = sz_regex_none_metal_k, composition.known = sz_regex_none_metal_k;
        ulong const next = chunk + 1, low = begin + next * width, high = min(low + width, arguments.slots_count);
        sz_regex_haystack_metal_t const haystack = sz_regex_haystack_metal_(
            corpus, sz_regex_haystack_at_metal_(corpus, low, 0));
        // The haystack either ends inside the next chunk or spans it, which then is settled or not.
        bool const ends = haystack.first + haystack.length < high;
        if (!ends && next != arguments.chunks_count && sz_regex_composing_metal_(round.frontiers[next].count)) {
            composition.target = uint(next);
            for (uint index = 0; index != round.frontiers[next].count; ++index)
                composition.indices[index] = sz_regex_index_of_metal_(states, count,
                                                                      round.frontiers[next].exits[index]);
        }
        else {
            uint const entry = ends                             ? arguments.reverse_starts[0]
                               : next == arguments.chunks_count ? round.edges[arguments.segment + 1]
                                                                : round.entries[next];
            device uchar const *text = haystack.text;
            ulong const at = ends ? haystack.length : sz_regex_prime_edge_metal_(folded, haystack, high);
            ulong const edge = sz_regex_prime_edge_metal_(folded, haystack, low);
            uint const state = folded ? sz_regex_walk_folded_back_metal_(table, entry, text, haystack.first, at, edge)
                                      : sz_regex_walk_back_metal_(table, entry, text, haystack.first, at, edge);
            composition.known = sz_regex_index_of_metal_(states, count, state);
        }
        round.compositions[chunk] = composition;
    }
}

/** One round of pointer doubling over the unsettled chunks: each composes its map with the
 *  targets' a round's hops ahead, or takes its index once one is known, and the last round
 *  writes the entry. */
kernel void sz_regex_compose_double_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                  constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                                  uint position [[thread_position_in_grid]],
                                                  uint threads [[threads_per_grid]]) {
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    bool const even = arguments.round % 2 == 0;
    device sz_regex_composition_metal_t const *source = even ? round.compositions : round.compositions_next;
    device sz_regex_composition_metal_t *target = even ? round.compositions_next : round.compositions;

    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += threads) {
        if (!sz_regex_composing_metal_(round.frontiers[chunk].count)) continue;
        sz_regex_composition_metal_t composition = source[chunk];
        for (uint hop = 1; hop != sz_regex_round_hops_metal_k && composition.target != sz_regex_none_metal_k; ++hop) {
            sz_regex_composition_metal_t const next = source[composition.target];
            if (next.known != sz_regex_none_metal_k) {
                composition.known = composition.indices[next.known];
                composition.target = sz_regex_none_metal_k;
                break;
            }
            uchar indices[sz_regex_priming_states_metal_k];
            for (uint index = 0; index != round.frontiers[next.target].count; ++index)
                indices[index] = next.indices[index] == 0xFF ? 0xFF : composition.indices[next.indices[index]];
            for (uint index = 0; index != round.frontiers[next.target].count; ++index)
                composition.indices[index] = indices[index];
            composition.target = next.target;
        }
        target[chunk] = composition;
        if (arguments.round + 1 != arguments.rounds) continue;
        round.entries[chunk] = round.frontiers[chunk].states[composition.known];
        if (chunk == 0) round.edges[arguments.segment] = round.entries[chunk];
    }
}

/**
 *  @brief Marks every slot of one chunk where a match starts, one thread per chunk, counting them
 *      into the chunk's offset for the scan that follows.
 *
 *  Each haystack piece walks the start-finding automaton right to left from the state the walk from
 *  its haystack's end holds at the piece's right edge: the start state when the piece ends the
 *  haystack, the primed entry otherwise. The chunk's own words of start bits are cleared first, and
 *  its first haystack is left for the later kernels to search from.
 */
kernel void sz_regex_mark_metal_kernel_(device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
                                        device uchar const *haystacks [[buffer(2)]],
                                        device uchar const *haystacks_offsets [[buffer(3)]],
                                        constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                        uint position [[thread_position_in_grid]], uint threads [[threads_per_grid]],
                                        threadgroup uchar *staged [[threadgroup(0)]],
                                        uint lane [[thread_index_in_threadgroup]],
                                        uint group_width [[threads_per_threadgroup]]) {
    sz_regex_table_metal_t const table = sz_regex_stage_metal_(engine, scratch, arguments, arguments.reverse_rows,
                                                               arguments.reverse_cells, staged, lane, group_width);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    ulong const begin = arguments.segment_begin, width = arguments.chunk_width;
    uint const accepts = arguments.reverse_accepts_begin;
    // The scan reads one entry past the chunks, which ends up holding the candidates' total.
    if (position == 0) round.chunk_offsets[arguments.chunks_count] = 0;

    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += threads) {
        ulong const low = begin + chunk * width, high = min(low + width, arguments.segment_end);
        ulong index = sz_regex_haystack_at_metal_(corpus, low, 0);
        round.chunk_haystacks[chunk] = index;
        device ulong *words = round.starts + (low - begin) / 64;
        for (ulong word = 0; word != width / 64; ++word) words[word] = 0;
        uint marked = 0;
        for (; index < corpus.count; ++index) {
            sz_regex_haystack_metal_t const haystack = sz_regex_haystack_metal_(corpus, index);
            if (haystack.first >= high) break;
            ulong const begin_byte = low > haystack.first ? low - haystack.first - 1 : 0;
            ulong const end_byte = min(haystack.length, high - haystack.first - 1);
            uint state = end_byte == haystack.length ? arguments.reverse_starts[0] : round.entries[chunk];
            for (ulong at = end_byte; at != begin_byte;) {
                uint const count = uint(min(at - begin_byte, (ulong)sz_regex_run_bytes_metal_k));
                uint const mask = sz_regex_run_back_metal_(table, state, haystack.text + at, haystack.first + at, count,
                                                           accepts);
                at -= count;
                if (!mask) continue;
                // A byte's own slot marks the match starting right after it.
                uint const bit = uint(haystack.first + 1 + at - low);
                words[bit / 64] |= ulong(mask) << (bit % 64);
                if (bit % 64 + count > 64) words[bit / 64 + 1] |= ulong(mask) >> (64 - bit % 64);
                marked += popcount(mask);
            }
            if (haystack.first < low || sz_regex_step_edge_metal_(table, state, haystack.first) < accepts) continue;
            uint const bit = uint(haystack.first - low);
            words[bit / 64] |= 1ul << (bit % 64), ++marked;
        }
        round.chunk_offsets[chunk] = marked;
    }
}

/** Bytes from a candidate start the literal prefilter tests, as @c sz_regex_prefixes_depth_k. */
constant uint sz_regex_prefixes_depth_metal_k = 3;

/** Marks every slot of one chunk whose bytes keep some bucket of the engine's prefixes at every
 *  offset inside its haystack, one thread per chunk, as @c sz_regex_mark_prefixes_simt_kernel_
 *  does, in place of the start-finding walk and its priming; a haystack's last slot never is. */
kernel void sz_regex_mark_prefixes_metal_kernel_(device uchar *engine [[buffer(0)]],
                                                 device uchar *scratch [[buffer(1)]],
                                                 device uchar const *haystacks [[buffer(2)]],
                                                 device uchar const *haystacks_offsets [[buffer(3)]],
                                                 constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                                 uint position [[thread_position_in_grid]],
                                                 uint threads [[threads_per_grid]]) {
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    // The bucket tables, the nibble tables, then the depth, as `sz_regex_prefixes_t` lays them out.
    device uchar const *buckets = sz_reach_metal_<uchar>(engine, arguments.engine_host, arguments.prefixes);
    uint const depth = *(device uint const *)(buckets + sz_regex_prefixes_depth_metal_k * (256 + 32));
    if (position == 0) round.chunk_offsets[arguments.chunks_count] = 0;
    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += threads) {
        ulong const low = arguments.segment_begin + chunk * arguments.chunk_width;
        ulong const high = min(low + arguments.chunk_width, arguments.segment_end);
        ulong index = sz_regex_haystack_at_metal_(corpus, low, 0);
        sz_regex_haystack_metal_t haystack = sz_regex_haystack_metal_(corpus, index);
        ulong marks = 0;
        round.chunk_haystacks[chunk] = index;
        for (ulong slot = low; slot != high; ++slot) {
            if (slot > haystack.first + haystack.length) haystack = sz_regex_haystack_metal_(corpus, ++index);
            ulong const at = slot - haystack.first;
            uchar held = at != haystack.length ? buckets[haystack.text[at]] : 0;
            for (ulong offset = 1; held && offset != depth && at + offset != haystack.length; ++offset)
                held &= buckets[offset * 256 + haystack.text[at + offset]];
            marks |= ulong(held != 0) << (slot - low);
        }
        round.starts[(low - arguments.segment_begin) / 64] = marks;
        round.chunk_offsets[chunk] = popcount(marks);
    }
}

/** Marks every slot of one chunk where a match of a folded walk starts, one thread per chunk, from
 *  the entry priming left, as @c sz_regex_mark_folded_simt_kernel_ does. */
kernel void sz_regex_mark_folded_metal_kernel_(
    device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
    device uchar const *haystacks [[buffer(2)]], device uchar const *haystacks_offsets [[buffer(3)]],
    constant sz_regex_arguments_metal_t &arguments [[buffer(10)]], uint position [[thread_position_in_grid]],
    uint threads [[threads_per_grid]], threadgroup uchar *staged [[threadgroup(0)]],
    uint lane [[thread_index_in_threadgroup]], uint group_width [[threads_per_threadgroup]]) {
    sz_regex_table_metal_t const table = sz_regex_stage_metal_(engine, scratch, arguments, arguments.reverse_rows,
                                                               arguments.reverse_cells, staged, lane, group_width);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    ulong const begin = arguments.segment_begin, width = arguments.chunk_width;
    uint const accepts = arguments.reverse_accepts_begin;
    if (position == 0) round.chunk_offsets[arguments.chunks_count] = 0;

    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += threads) {
        ulong const low = begin + chunk * width, high = min(low + width, arguments.segment_end);
        ulong index = sz_regex_haystack_at_metal_(corpus, low, 0);
        round.chunk_haystacks[chunk] = index;
        device ulong *words = round.starts + (low - begin) / 64;
        for (ulong word = 0; word != width / 64; ++word) words[word] = 0;
        for (; index < corpus.count; ++index) {
            sz_regex_haystack_metal_t const haystack = sz_regex_haystack_metal_(corpus, index);
            if (haystack.first >= high) break;
            ulong const first = low > haystack.first ? low - haystack.first : 0;
            ulong const last = min(haystack.length, high - haystack.first - 1);
            ulong at = last == haystack.length ? last
                                               : sz_regex_boundary_metal_(haystack.text, haystack.length, last + 1);
            uint state = at == haystack.length ? arguments.reverse_starts[0] : round.entries[chunk];
            ulong const stop = first ? first - 1 : 0;
            while (at > stop) {
                sz_utf8_folded_image_metal_t image;
                uint block = sz_regex_block_at_metal_(table, haystack.first + at);
                ulong const unit_begin = sz_regex_unit_before_metal_(sz_regex_folded_metal_k, haystack.text, at, image);
                for (uint byte = image.length; byte--; block = 0) {
                    ulong const start = byte + 1u == image.length ? at : unit_begin;
                    state = sz_regex_move_metal_(table, state,
                                                 block + table.classes[sz_regex_image_byte_metal_(image, byte)]);
                    if (state >= accepts && start >= first && start <= last)
                        words[(haystack.first + start - low) / 64] |= 1ul << ((haystack.first + start - low) % 64);
                }
                at = unit_begin;
            }
            if (first == 0 && sz_regex_step_edge_metal_(table, state, haystack.first) >= accepts)
                words[(haystack.first - low) / 64] |= 1ul << ((haystack.first - low) % 64);
        }
        uint marked = 0;
        for (ulong word = 0; word != width / 64; ++word) marked += popcount(words[word]);
        round.chunk_offsets[chunk] = marked;
    }
}

/** Marks every unit boundary of one chunk where a match starts, stepping position sets in each
 *  thread's share of the walks from the set handed to each piece, as
 *  @c sz_regex_mark_positions_simt_ does. */
inline void sz_regex_mark_positions_metal_(device uchar *engine, device uchar *scratch, device uchar const *haystacks,
                                           device uchar const *haystacks_offsets,
                                           constant sz_regex_arguments_metal_t &arguments, uint position,
                                           sz_regex_folding_metal_t folded) {
    sz_regex_automata_metal_t const automata = sz_regex_automata_metal_(engine, arguments);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    ulong const begin = arguments.segment_begin, width = arguments.chunk_width, words = arguments.positions_words;
    ulong const stride = arguments.set_words;
    device ulong *sets = (device ulong *)(round.walks + position * arguments.walk_bytes);
    if (position == 0) round.chunk_offsets[arguments.chunks_count] = 0;
    if (position >= arguments.walks_count) return;

    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += arguments.walks_count) {
        ulong const low = begin + chunk * width, high = min(low + width, arguments.segment_end);
        ulong index = sz_regex_haystack_at_metal_(corpus, low, 0);
        round.chunk_haystacks[chunk] = index;
        device ulong *bits = round.starts + (low - begin) / 64;
        for (ulong word = 0; word != width / 64; ++word) bits[word] = 0;
        for (; index < corpus.count; ++index) {
            sz_regex_haystack_metal_t const haystack = sz_regex_haystack_metal_(corpus, index);
            if (haystack.first >= high) break;
            device uchar const *text = haystack.text;
            ulong const length = haystack.length;
            ulong const first = low > haystack.first ? low - haystack.first : 0;
            ulong const end = min(high - haystack.first, length + 1);
            // A piece running past the chunk takes the set the chunk right of it hands on,
            // past its unit boundary.
            device ulong const *handed = nullptr;
            if (end <= length)
                handed = chunk + 1 == arguments.chunks_count ? round.handoff_edges + (arguments.segment + 1) * stride
                                                             : round.handoffs + (chunk + 1) * stride;
            for (ulong word = 0; word != words; ++word) sets[word] = handed ? handed[word] : word == 0;
            device ulong *state = sz_regex_positions_walk_metal_(
                automata, arguments, round.looks, haystack, folded,
                end > length ? length : sz_regex_boundary_metal_(text, length, end), first ? first - 1 : 0, sets,
                sets + words, sz_regex_marked_metal_k, bits, low, high);
            if (first) continue;
            // The start of the haystack takes one more step, over the edge class.
            device ulong *other = state == sets ? sets + words : sets;
            sz_utf8_folded_image_metal_t image;
            if (length) sz_regex_unit_after_metal_(folded, text, length, 0, image);
            sz_regex_positions_step_metal_(automata, arguments, state, other, 1,
                                           length ? sz_regex_byte_kind_metal_(sz_regex_image_byte_metal_(image, 0)) : 0,
                                           sz_regex_looks_at_metal_(round.looks, arguments, haystack.first));
            if (other[0] >> 1 & 1) bits[(haystack.first - low) / 64] |= 1ul << ((haystack.first - low) % 64);
        }
        uint marked = 0;
        for (ulong word = 0; word != width / 64; ++word) marked += popcount(bits[word]);
        round.chunk_offsets[chunk] = marked;
    }
}

/** @ref sz_regex_mark_positions_metal_ over bytes. */
kernel void sz_regex_mark_positions_metal_kernel_(device uchar *engine [[buffer(0)]],
                                                  device uchar *scratch [[buffer(1)]],
                                                  device uchar const *haystacks [[buffer(2)]],
                                                  device uchar const *haystacks_offsets [[buffer(3)]],
                                                  constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                                  uint position [[thread_position_in_grid]]) {
    sz_regex_mark_positions_metal_(engine, scratch, haystacks, haystacks_offsets, arguments, position,
                                   sz_regex_cased_metal_k);
}

/** @ref sz_regex_mark_positions_metal_ over folded runes. */
kernel void sz_regex_mark_positions_folded_metal_kernel_(device uchar *engine [[buffer(0)]],
                                                         device uchar *scratch [[buffer(1)]],
                                                         device uchar const *haystacks [[buffer(2)]],
                                                         device uchar const *haystacks_offsets [[buffer(3)]],
                                                         constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                                         uint position [[thread_position_in_grid]]) {
    sz_regex_mark_positions_metal_(engine, scratch, haystacks, haystacks_offsets, arguments, position,
                                   sz_regex_folded_metal_k);
}

/** Lists the candidates of one chunk in slot order, one thread per chunk, for the end walks taking
 *  one candidate per thread. */
kernel void sz_regex_list_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                        constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                        uint position [[thread_position_in_grid]], uint threads [[threads_per_grid]]) {
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    ulong const width = arguments.chunk_width;
    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += threads) {
        uint index = uint(round.chunk_offsets[chunk]);
        for (ulong word = 0; word != width / 64; ++word)
            for (ulong bits = round.starts[chunk * width / 64 + word]; bits; bits &= bits - 1)
                round.slots[index++] = uint(chunk * width + word * 64 + ctz(bits));
    }
}

/** Walks every candidate to its end with a PikeVM, linking nothing,
 *  in each thread's share of the walks. */
kernel void sz_regex_ends_pike_metal_kernel_(device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
                                             device uchar const *haystacks [[buffer(2)]],
                                             device uchar const *haystacks_offsets [[buffer(3)]],
                                             constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                             uint position [[thread_position_in_grid]],
                                             threadgroup uchar *staged [[threadgroup(0)]],
                                             uint lane [[thread_index_in_threadgroup]],
                                             uint group_width [[threads_per_threadgroup]]) {
    sz_regex_table_metal_t const table = sz_regex_direct_metal_(engine, scratch, arguments, arguments.forward_rows,
                                                                staged, lane, group_width);
    sz_regex_automata_metal_t const automata = sz_regex_automata_metal_(engine, arguments);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    ulong const count = round.chunk_offsets[arguments.chunks_count];
    if (position >= arguments.walks_count) return;
    sz_regex_pike_metal_t pike = sz_regex_pike_carve_metal_(round, arguments, position);
    pike.slots_count = 0;
    for (ulong item = position; item < count; item += arguments.walks_count) {
        ulong const mark = arguments.segment_begin + round.slots[item];
        sz_regex_haystack_metal_t const haystack = sz_regex_haystack_metal_(
            corpus, sz_regex_haystack_at_metal_(corpus, mark,
                                                round.chunk_haystacks[round.slots[item] >> arguments.chunk_shift]));
        sz_regex_result_metal_t result = {sz_regex_none_metal_k, sz_regex_none_metal_k, 0};
        ulong end, offset = 0;
        uint pattern;
        pike.first = haystack.first;
        if (sz_regex_match_at_metal_(table, automata, arguments, pike, sz_regex_pike_vm_metal_k, haystack.text,
                                     haystack.length, mark - haystack.first, end, pattern, offset))
            result.pattern = pattern, result.end = haystack.first + end;
        round.results[item] = result;
    }
}

/** The 16-slot mark whose checkpoint stands at the unit boundary of byte slot @p slot, reached
 *  over a unit of @p stepped bytes, its checkpoint being the first unit boundary at or past it;
 *  zero for none. */
inline ulong sz_regex_mark_at_metal_(ulong slot, ulong stepped) {
    ulong const phase = slot % sz_regex_checkpoint_slots_metal_k;
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
inline ulong sz_regex_advance_metal_(thread sz_regex_table_metal_t const &table, sz_regex_folding_metal_t folded,
                                     thread sz_regex_haystack_metal_t const &haystack, ulong at, ulong stop,
                                     uint accepts, thread uint &state, thread uint &last, thread ulong &last_end,
                                     thread ulong &last_at, thread ulong &stepped) {
    if (!folded) {
        ulong const phase = (haystack.first + 1 + at) % sz_regex_checkpoint_slots_metal_k;
        uint const count = uint(min(sz_regex_checkpoint_slots_metal_k - phase, min(haystack.length, stop) - at));
        uint run_last = 0, run_offset = 0;
        uint const walked = sz_regex_run_metal_(table, state, haystack.text + at, haystack.first + at, count, accepts,
                                                run_last, run_offset);
        if (run_last) last = run_last, last_end = last_at = at + run_offset;
        stepped = 1;
        return at + walked;
    }
    sz_utf8_folded_image_metal_t image;
    ulong const unit_end = sz_regex_unit_after_metal_(sz_regex_folded_metal_k, haystack.text, haystack.length, at,
                                                      image);
    uint block = sz_regex_block_at_metal_(table, haystack.first + at);
    for (uint byte = 0; byte != image.length && state != 0; ++byte, block = 0) {
        state = sz_regex_move_metal_(table, state, block + table.classes[sz_regex_image_byte_metal_(image, byte)]);
        if (state >= accepts) last = state, last_end = byte ? unit_end : at, last_at = at;
    }
    stepped = unit_end - at;
    return unit_end;
}

/**
 *  @brief Walks every candidate of one chunk to its end, right to left, one thread per chunk, which
 *      lists the candidates in slot order as it goes.
 *
 *  A walk leaves its state at the checkpoint of every 16-slot mark it passes, the first unit
 *  boundary at or past the mark, and a later walk, starting further left, that holds the same state
 *  at the same checkpoint stops there and takes that walk's outcome, so the chunk's bytes are
 *  walked about once whatever the candidates. A walk reaching the chunk's edge leaves a link there
 *  instead: the state it would walk on in, which the links kernel follows. A folded walk starts at
 *  the first rune its unit folds to, @ref sz_regex_retry_folded_metal_kernel_ trying the others
 *  once the links settle.
 */
kernel void sz_regex_ends_metal_kernel_(device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
                                        device uchar const *haystacks [[buffer(2)]],
                                        device uchar const *haystacks_offsets [[buffer(3)]],
                                        constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                        uint position [[thread_position_in_grid]], uint threads [[threads_per_grid]],
                                        threadgroup uchar *staged [[threadgroup(0)]],
                                        uint lane [[thread_index_in_threadgroup]],
                                        uint group_width [[threads_per_threadgroup]]) {
    sz_regex_table_metal_t const table = sz_regex_stage_metal_(engine, scratch, arguments, arguments.forward_rows,
                                                               arguments.forward_cells, staged, lane, group_width);
    sz_regex_automata_metal_t const automata = sz_regex_automata_metal_(engine, arguments);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    ulong const begin = arguments.segment_begin, width = arguments.chunk_width;
    uint const accepts = arguments.forward_accepts_begin;
    sz_regex_folding_metal_t const folded = sz_regex_folding_metal_t(arguments.case_sensitivity != 0);

    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += threads) {
        uint index = uint(round.chunk_offsets[chunk + 1]);
        if (index == round.chunk_offsets[chunk]) continue;
        ulong const low = begin + chunk * width, high = min(low + width, arguments.segment_end);
        // Only a chunk with candidates has its checkpoints read, so only such a chunk clears them.
        for (ulong checkpoint = (low - begin) / sz_regex_checkpoint_slots_metal_k + 1;
             checkpoint <= (high - begin) / sz_regex_checkpoint_slots_metal_k; ++checkpoint)
            round.checkpoints[checkpoint].state = 0;
        ulong haystack_index = sz_regex_haystack_at_metal_(corpus, high - 1, round.chunk_haystacks[chunk]);
        sz_regex_haystack_metal_t haystack = sz_regex_haystack_metal_(corpus, haystack_index);

        for (ulong word = (high - low + 63) / 64; word-- != 0;) {
            ulong bits = round.starts[(low - begin) / 64 + word];
            while (bits) {
                uint const bit = 63 - uint(clz(bits));
                bits ^= 1ul << bit;
                ulong const mark = low + word * 64 + bit;
                while (mark < haystack.first) haystack = sz_regex_haystack_metal_(corpus, --haystack_index);
                round.slots[--index] = uint(mark - begin);

                ulong const start = mark - haystack.first;
                ulong stepped = 1;
                uint kind = 0;
                if (start && folded) {
                    sz_utf8_folded_image_metal_t previous;
                    stepped = start -
                              sz_regex_unit_before_metal_(sz_regex_folded_metal_k, haystack.text, start, previous);
                    kind = sz_regex_byte_kind_metal_(sz_regex_image_byte_metal_(previous, previous.length - 1u));
                }
                else if (start) kind = sz_regex_byte_kind_metal_(haystack.text[start - 1]);
                uint state = arguments.forward_starts[kind];
                uint last = 0, link = sz_regex_none_metal_k, inherited = sz_regex_none_metal_k;
                ulong last_end = 0, last_at = 0, inherited_end = 0, recorded_low = 0, recorded_high = 0;
                for (ulong at = start; state != 0;) {
                    if (at == haystack.length) {
                        state = sz_regex_step_edge_metal_(table, state, haystack.first + at);
                        if (state >= accepts) last = state, last_end = last_at = at;
                        break;
                    }
                    // The mark at the chunk's own start is the edge of the chunk before,
                    // which owns its checkpoint.
                    ulong const slot = sz_regex_mark_at_metal_(haystack.first + 1 + at, stepped);
                    if (slot > low) {
                        ulong const checkpoint = (slot - begin) / sz_regex_checkpoint_slots_metal_k;
                        device sz_regex_checkpoint_metal_t *entry = round.checkpoints + checkpoint;
                        if (entry->state == state) {
                            link = entry->link;
                            if (entry->pattern != sz_regex_none_metal_k)
                                inherited = entry->pattern, inherited_end = begin + entry->end, last_at = at;
                            break;
                        }
                        entry->state = state;
                        if (slot == high) {
                            link = index;
                            entry->link = index, entry->pattern = sz_regex_none_metal_k;
                            round.links[index].link = state;
                            round.links_list[atomic_fetch_add_explicit(round.counters + sz_regex_links_counter_metal_k,
                                                                       1u, memory_order_relaxed)] = index;
                            break;
                        }
                        if (recorded_high == 0) recorded_low = checkpoint;
                        recorded_high = checkpoint + 1;
                    }
                    at = sz_regex_advance_metal_(table, folded, haystack, at, haystack.length, accepts, state, last,
                                                 last_end, last_at, stepped);
                }
                sz_regex_result_metal_t result;
                result.link = link;
                result.end = inherited != sz_regex_none_metal_k ? inherited_end : haystack.first + last_end;
                result.pattern = inherited != sz_regex_none_metal_k ? inherited
                                 : last                             ? sz_regex_pattern_metal_(automata, arguments, last)
                                                                    : sz_regex_none_metal_k;
                // The checkpoints passed learn the outcome, keeping the match only when the step
                // reaching it was past them, which a match ending inside the folds of the unit
                // before a checkpoint was not.
                for (ulong checkpoint = recorded_low; checkpoint < recorded_high; ++checkpoint) {
                    ulong const slot = begin + checkpoint * sz_regex_checkpoint_slots_metal_k;
                    bool const past = result.pattern != sz_regex_none_metal_k && haystack.first + 1 + last_at >= slot;
                    round.checkpoints[checkpoint].link = result.link;
                    round.checkpoints[checkpoint].pattern = past ? result.pattern : sz_regex_none_metal_k;
                    round.checkpoints[checkpoint].end = uint(result.end - begin);
                }
                round.results[index] = result;
            }
        }
    }
}

/** The outcome the walk of @p id settles: its link's, or its group's when @p id marks a group. */
inline device sz_regex_result_metal_t *sz_regex_walked_metal_(thread sz_regex_round_metal_t const &round, uint id) {
    return id & sz_regex_group_metal_k ? round.groups + (id & ~sz_regex_group_metal_k) : round.links + id;
}

/** Hands the walk of @p id on to a SIMD group in @p state at the boundary in @p slot, which walks
 *  it on, or pauses it there when its next step would read the lookaround table past its window. */
inline void sz_regex_hand_metal_(thread sz_regex_round_metal_t const &round, uint id, uint state, ulong slot) {
    device sz_regex_continuation_metal_t *continuations = (device sz_regex_continuation_metal_t *)round.links_next;
    uint const handed = atomic_fetch_add_explicit(round.counters + sz_regex_handed_counter_metal_k, 1u,
                                                  memory_order_relaxed);
    continuations[handed].link = id, continuations[handed].state = state, continuations[handed].slot = slot;
}

/** Walks the link or group @p id on from unit boundary @p at of @p haystack in @p state until it
 *  dies, ends its haystack, or meets a checkpoint of a later chunk of the segment in the same
 *  state, else hands it to a SIMD group once it outlives its share of bytes or would read the
 *  lookaround table past its window; over the folds of whole units when @p folded, meeting
 *  checkpoints where the end walks leave them. */
inline void sz_regex_link_walk_metal_(thread sz_regex_table_metal_t const &table,
                                      thread sz_regex_automata_metal_t const &automata,
                                      constant sz_regex_arguments_metal_t &arguments,
                                      thread sz_regex_round_metal_t const &round, sz_regex_folding_metal_t folded,
                                      uint id, uint state, thread sz_regex_haystack_metal_t const &haystack, ulong at) {
    ulong const begin = arguments.segment_begin, end = arguments.segment_end, edge = haystack.first + 1 + at;
    ulong const stop = table.looks ? table.looks_end - haystack.first : haystack.length;
    ulong const handover = at + sz_regex_link_bytes_metal_k;
    uint const accepts = arguments.forward_accepts_begin;
    uint last = 0, inherited = sz_regex_none_metal_k;
    ulong last_end = 0, last_at = 0, stepped = 1;
    sz_regex_result_metal_t result = {sz_regex_none_metal_k, sz_regex_none_metal_k, 0};
    while (state != 0) {
        if (sz_regex_past_metal_(table, haystack.first + at)) {
            sz_regex_hand_metal_(round, id, state, haystack.first + at);
            break;
        }
        if (at == haystack.length) {
            state = sz_regex_step_edge_metal_(table, state, haystack.first + at);
            if (state >= accepts) last = state, last_end = at;
            break;
        }
        ulong const slot = sz_regex_mark_at_metal_(haystack.first + 1 + at, stepped);
        if (slot > edge && slot <= end) {
            ulong const owner = (slot - 1 - begin) >> arguments.chunk_shift;
            sz_regex_checkpoint_metal_t const entry =
                round.checkpoints[(slot - begin) / sz_regex_checkpoint_slots_metal_k];
            if (round.chunk_offsets[owner + 1] != round.chunk_offsets[owner] && entry.state == state) {
                result.link = entry.link;
                if (entry.pattern != sz_regex_none_metal_k) inherited = entry.pattern, result.end = begin + entry.end;
                break;
            }
        }
        if (slot && at >= handover) {
            sz_regex_hand_metal_(round, id, state, haystack.first + at);
            break;
        }
        at = sz_regex_advance_metal_(table, folded, haystack, at, stop, accepts, state, last, last_end, last_at,
                                     stepped);
    }
    if (inherited != sz_regex_none_metal_k) result.pattern = inherited;
    else if (last)
        result.pattern = sz_regex_pattern_metal_(automata, arguments, last), result.end = haystack.first + last_end;
    *sz_regex_walked_metal_(round, id) = result;
}

/** Walks every link on from its chunk's edge, at the first unit boundary past it for a folded walk,
 *  as @ref sz_regex_link_walk_metal_ walks it. */
kernel void sz_regex_links_metal_kernel_(device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
                                         device uchar const *haystacks [[buffer(2)]],
                                         device uchar const *haystacks_offsets [[buffer(3)]],
                                         constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                         uint position [[thread_position_in_grid]], uint threads [[threads_per_grid]],
                                         threadgroup uchar *staged [[threadgroup(0)]],
                                         uint lane [[thread_index_in_threadgroup]],
                                         uint group_width [[threads_per_threadgroup]]) {
    sz_regex_table_metal_t const table = sz_regex_stage_metal_(engine, scratch, arguments, arguments.forward_rows,
                                                               arguments.forward_cells, staged, lane, group_width);
    sz_regex_automata_metal_t const automata = sz_regex_automata_metal_(engine, arguments);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    sz_regex_folding_metal_t const folded = sz_regex_folding_metal_t(arguments.case_sensitivity != 0);
    uint const links_count = atomic_load_explicit(round.counters + sz_regex_links_counter_metal_k,
                                                  memory_order_relaxed);
    ulong const begin = arguments.segment_begin, end = arguments.segment_end;

    for (uint item = position; item < links_count; item += threads) {
        uint const link = round.links_list[item];
        ulong const chunk = round.slots[link] >> arguments.chunk_shift;
        ulong const edge = min(begin + ((chunk + 1) << arguments.chunk_shift), end);
        sz_regex_haystack_metal_t const haystack = sz_regex_haystack_metal_(
            corpus, sz_regex_haystack_at_metal_(corpus, begin + round.slots[link], round.chunk_haystacks[chunk]));
        ulong const at = edge - haystack.first - 1;
        sz_regex_link_walk_metal_(table, automata, arguments, round, folded, link, round.links[link].link, haystack,
                                  folded ? sz_regex_boundary_metal_(haystack.text, haystack.length, at) : at);
    }
}

/** Walks every group the last slide opened on from where its walks paused, the window's start, as
 *  the links kernel walks links, the window having moved on two segments from its end there. */
kernel void sz_regex_resume_metal_kernel_(device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
                                          device uchar const *haystacks [[buffer(2)]],
                                          device uchar const *haystacks_offsets [[buffer(3)]],
                                          constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                          uint position [[thread_position_in_grid]], uint threads [[threads_per_grid]],
                                          threadgroup uchar *staged [[threadgroup(0)]],
                                          uint lane [[thread_index_in_threadgroup]],
                                          uint group_width [[threads_per_threadgroup]]) {
    sz_regex_table_metal_t const table = sz_regex_stage_metal_(engine, scratch, arguments, arguments.forward_rows,
                                                               arguments.forward_cells, staged, lane, group_width);
    sz_regex_automata_metal_t const automata = sz_regex_automata_metal_(engine, arguments);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    sz_regex_folding_metal_t const folded = sz_regex_folding_metal_t(arguments.case_sensitivity != 0);
    ulong const paused = arguments.looks_begin;
    sz_regex_haystack_metal_t const haystack = sz_regex_haystack_metal_(corpus,
                                                                        sz_regex_haystack_at_metal_(corpus, paused, 0));
    ulong const at = folded ? sz_regex_boundary_metal_(haystack.text, haystack.length, paused - haystack.first)
                            : paused - haystack.first;
    uint const groups_count = atomic_load_explicit(round.counters + sz_regex_groups_counter_metal_k,
                                                   memory_order_relaxed);
    for (uint group = uint(arguments.groups_first) + position; group < groups_count; group += threads)
        sz_regex_link_walk_metal_(table, automata, arguments, round, folded, sz_regex_group_metal_k | group,
                                  round.groups[group].link, haystack, at);
}

/** Where piece @p index of those a SIMD group walks from @p at begins, at most @p stop: @p at
 *  itself for the first, else its share of bytes on, moved to the first unit boundary at or past it
 *  when @p folded. */
inline ulong sz_regex_piece_metal_(sz_regex_folding_metal_t folded, thread sz_regex_haystack_metal_t const &haystack,
                                   ulong at, ulong index, ulong stop) {
    ulong const piece = at + index * sz_regex_piece_bytes_metal_k;
    if (!index) return at;
    if (piece >= stop) return stop;
    return folded ? min(sz_regex_boundary_metal_(haystack.text, haystack.length, piece), stop) : piece;
}

/**
 *  @brief Walks every link or group handed to a SIMD group on to its end, one SIMD group each.
 *
 *  Each lane takes the next piece of the haystack in turn, the pieces of whole units over folds. A
 *  walk whose next step would read the lookaround table past its window pauses there, which its
 *  continuation keeps, counted; any other ends with a zero state.
 *
 *  Only the first lane knows the state its piece starts in. Every other lane walks its piece from
 *  each state its first byte can lead to, the image of that byte's column, so once the lanes before
 *  it have settled, its own outcome is a lookup. A piece whose image is too wide to walk leaves the
 *  round to the first lane alone.
 */
kernel void sz_regex_continue_metal_kernel_(
    device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
    device uchar const *haystacks [[buffer(2)]], device uchar const *haystacks_offsets [[buffer(3)]],
    constant sz_regex_arguments_metal_t &arguments [[buffer(10)]], threadgroup uchar *staged [[threadgroup(0)]],
    uint position [[thread_index_in_threadgroup]], uint group_width [[threads_per_threadgroup]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint simdgroups [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint lanes [[threads_per_simdgroup]]) {
    sz_regex_table_metal_t const table = sz_regex_stage_metal_(engine, scratch, arguments, arguments.forward_rows,
                                                               arguments.forward_cells, staged, position, group_width);
    sz_regex_automata_metal_t const automata = sz_regex_automata_metal_(engine, arguments);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    device sz_regex_continuation_metal_t *continuations = (device sz_regex_continuation_metal_t *)round.links_next;
    uint const handed = atomic_load_explicit(round.counters + sz_regex_handed_counter_metal_k, memory_order_relaxed);
    uint const accepts = arguments.forward_accepts_begin;
    device uint const *images = round.forward_images;
    sz_regex_folding_metal_t const folded = sz_regex_folding_metal_t(arguments.case_sensitivity != 0);

    for (uint item = group * simdgroups + simdgroup; item < handed; item += groups * simdgroups) {
        sz_regex_continuation_metal_t const continuation = continuations[item];
        sz_regex_haystack_metal_t const haystack = sz_regex_haystack_metal_(
            corpus, sz_regex_haystack_at_metal_(corpus, continuation.slot, 0));
        // The pieces end where the window does, at the first unit boundary at or past it.
        ulong const stop = !sz_regex_past_metal_(table, haystack.first + haystack.length) ? haystack.length
                           : folded ? sz_regex_boundary_metal_(haystack.text, haystack.length,
                                                               table.looks_end - haystack.first)
                                    : table.looks_end - haystack.first;
        uint state = continuation.state, last = 0;
        ulong at = continuation.slot - haystack.first, last_end = 0, last_at, stepped;
        bool paused = false;
        while (state != 0) {
            if (sz_regex_past_metal_(table, haystack.first + at)) {
                paused = true;
                break;
            }
            if (at == haystack.length) {
                state = sz_regex_step_edge_metal_(table, state, haystack.first + at);
                if (state >= accepts) last = state, last_end = haystack.length;
                break;
            }
            ulong const piece = sz_regex_piece_metal_(folded, haystack, at, lane, stop);
            ulong const piece_end = sz_regex_piece_metal_(folded, haystack, at, lane + 1, stop);
            sz_utf8_folded_image_metal_t image;
            ulong unit_end = piece;
            uint column = 0, domain[sz_regex_priming_states_metal_k], count = 0;
            bool wide = false;
            if (piece < piece_end) {
                unit_end = sz_regex_unit_after_metal_(folded, haystack.text, haystack.length, piece, image);
                column = sz_regex_block_at_metal_(table, haystack.first + piece) +
                         table.classes[sz_regex_image_byte_metal_(image, 0)];
            }
            if (piece < piece_end && lane == 0) {
                domain[0] = sz_regex_move_metal_(table, state, column);
                count = domain[0] != 0;
            }
            else if (piece < piece_end) {
                count = images[column];
                wide = count > sz_regex_priming_states_metal_k;
                for (uint index = 0; !wide && index != count; ++index)
                    domain[index] = images[arguments.stride + column * sz_regex_priming_states_metal_k + index];
            }
            if (simd_any(wide)) {
                ulong const round_end = sz_regex_piece_metal_(folded, haystack, at, lanes, stop);
                while (lane == 0 && state != 0 && at != round_end)
                    at = sz_regex_advance_metal_(table, folded, haystack, at, round_end, accepts, state, last, last_end,
                                                 last_at, stepped);
                state = simd_shuffle(state, 0), last = simd_shuffle(last, 0);
                last_end = (ulong(simd_shuffle(uint(last_end >> 32), 0)) << 32) | simd_shuffle(uint(last_end), 0);
                at = state ? round_end : at;
                continue;
            }

            // Every state of the domain walks the rest of the piece on its own, remembering its
            // last accepting step, the folds after a unit's first ending a match at the unit's end.
            uint set[sz_regex_priming_states_metal_k], lasts[sz_regex_priming_states_metal_k];
            uint lasts_offsets[sz_regex_priming_states_metal_k];
#pragma unroll
            for (uint index = 0; index != sz_regex_priming_states_metal_k; ++index) {
                set[index] = index < count ? domain[index] : 0, lasts_offsets[index] = 0;
                lasts[index] = set[index] >= accepts ? set[index] : 0;
            }
            for (ulong unit = piece, byte = 1; piece < piece_end; ++byte) {
                if (byte == image.length) {
                    unit = unit_end, byte = 0;
                    if (unit >= piece_end) break;
                    unit_end = sz_regex_unit_after_metal_(folded, haystack.text, haystack.length, unit, image);
                }
                uint const byte_column = (byte ? 0 : sz_regex_block_at_metal_(table, haystack.first + unit)) +
                                         table.classes[sz_regex_image_byte_metal_(image, uint(byte))];
                uint const ended = uint((byte ? unit_end : unit) - piece);
                bool alive = false;
#pragma unroll
                for (uint index = 0; index != sz_regex_priming_states_metal_k; ++index) {
                    if (index >= count || !set[index]) continue;
                    set[index] = sz_regex_move_metal_(table, set[index], byte_column);
                    if (set[index] >= accepts) lasts[index] = set[index], lasts_offsets[index] = ended;
                    alive = alive || set[index];
                }
                if (!alive) break;
            }

            // The lanes settle in turn, each from the state the one before it left.
            ulong const round_at = at;
            for (uint turn = 0; turn != lanes && state != 0; ++turn) {
                ulong const turn_piece = sz_regex_piece_metal_(folded, haystack, round_at, turn, stop);
                ulong const turn_end = sz_regex_piece_metal_(folded, haystack, round_at, turn + 1, stop);
                if (turn_piece >= turn_end) break;
                uint exit = 0, turn_last = 0, turn_offset = 0;
                if (lane == turn) {
                    uint const entered = sz_regex_move_metal_(table, state, column);
#pragma unroll
                    for (uint index = 0; index != sz_regex_priming_states_metal_k; ++index)
                        if (index < count && entered && domain[index] == entered)
                            exit = set[index], turn_last = lasts[index], turn_offset = lasts_offsets[index];
                }
                state = simd_shuffle(exit, turn), turn_last = simd_shuffle(turn_last, turn);
                turn_offset = simd_shuffle(turn_offset, turn);
                if (turn_last) last = turn_last, last_end = turn_piece + turn_offset;
                at = turn_end;
            }
        }
        if (lane != 0) continue;
        if (last) {
            device sz_regex_result_metal_t *walked = sz_regex_walked_metal_(round, continuation.link);
            walked->pattern = sz_regex_pattern_metal_(automata, arguments, last),
            walked->end = haystack.first + last_end;
        }
        continuations[item].state = paused ? state : 0;
        if (paused)
            continuations[item].slot = haystack.first + at,
            atomic_fetch_add_explicit(round.counters + sz_regex_paused_counter_metal_k, 1u, memory_order_relaxed);
    }
}

/** Follows one round's hops of pointer doubling from @p result over @p source, links or groups
 *  alike, the match a later hop reaches winning. */
inline sz_regex_result_metal_t sz_regex_hops_metal_(device sz_regex_result_metal_t const *source,
                                                    sz_regex_result_metal_t result) {
    for (uint hop = 1; hop != sz_regex_round_hops_metal_k && result.link != sz_regex_none_metal_k; ++hop) {
        sz_regex_result_metal_t const next = source[result.link & ~sz_regex_group_metal_k];
        result.link = next.link;
        if (next.pattern != sz_regex_none_metal_k) result.pattern = next.pattern, result.end = next.end;
    }
    return result;
}

/** One round of pointer doubling over the links: each takes the deepest match a round's hops ahead,
 *  and jumps that much further. */
kernel void sz_regex_links_double_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                                uint position [[thread_position_in_grid]],
                                                uint threads [[threads_per_grid]]) {
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    bool const even = arguments.round % 2 == 0;
    device sz_regex_result_metal_t const *source = even ? round.links : round.links_next;
    device sz_regex_result_metal_t *target = even ? round.links_next : round.links;
    uint const links_count = atomic_load_explicit(round.counters + sz_regex_links_counter_metal_k,
                                                  memory_order_relaxed);
    for (uint item = position; item < links_count; item += threads) {
        uint const link = round.links_list[item];
        target[link] = sz_regex_hops_metal_(source, source[link]);
    }
}

/** Gives each distinct state among the walks paused at the end of the lookaround table's window one
 *  group, one thread per walk handed on: the first to take its state's leader allocates the group
 *  and publishes it there, and one past the groups' capacity raises the refusal instead. */
kernel void sz_regex_group_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                         constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                         uint position [[thread_position_in_grid]], uint threads [[threads_per_grid]]) {
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    device sz_regex_continuation_metal_t const *continuations =
        (device sz_regex_continuation_metal_t const *)round.links_next;
    uint const handed = atomic_load_explicit(round.counters + sz_regex_handed_counter_metal_k, memory_order_relaxed);
    for (uint item = position; item < handed; item += threads) {
        uint const state = continuations[item].state;
        if (!state) continue;
        // The leaders start cleared, and a weak exchange may fail with none taken,
        // so only a taken one stops it.
        device atomic_uint *leader = round.leaders + state / arguments.stride;
        uint taken = 0;
        while (!atomic_compare_exchange_weak_explicit(leader, &taken, item + 1, memory_order_relaxed,
                                                      memory_order_relaxed) &&
               !taken) {}
        if (taken) continue;
        uint const group = atomic_fetch_add_explicit(round.counters + sz_regex_groups_counter_metal_k, 1u,
                                                     memory_order_relaxed);
        if (group >= arguments.groups_capacity) {
            atomic_store_explicit(round.counters + sz_regex_refused_counter_metal_k, 1u, memory_order_relaxed);
            continue;
        }
        round.groups[group].link = state, round.groups[group].pattern = sz_regex_none_metal_k;
        round.groups[group].end = 0;
        atomic_store_explicit(leader, sz_regex_group_metal_k | group, memory_order_relaxed);
    }
}

/** Points every walk paused at the end of the lookaround table's window at its state's group, which
 *  a link or a group of the window's last end then follows as the link whose match wins. */
kernel void sz_regex_point_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                         constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                         uint position [[thread_position_in_grid]], uint threads [[threads_per_grid]]) {
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    device sz_regex_continuation_metal_t const *continuations =
        (device sz_regex_continuation_metal_t const *)round.links_next;
    uint const handed = atomic_load_explicit(round.counters + sz_regex_handed_counter_metal_k, memory_order_relaxed);
    for (uint item = position; item < handed; item += threads) {
        sz_regex_continuation_metal_t const continuation = continuations[item];
        if (!continuation.state) continue;
        uint const group = atomic_load_explicit(round.leaders + continuation.state / arguments.stride,
                                                memory_order_relaxed);
        if (group & sz_regex_group_metal_k) sz_regex_walked_metal_(round, continuation.link)->link = group;
    }
}

/** One round of pointer doubling over the groups, as over the links, each taking the deepest match
 *  a round's hops ahead among the groups of later window ends. */
kernel void sz_regex_groups_double_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                 constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                                 uint position [[thread_position_in_grid]],
                                                 uint threads [[threads_per_grid]]) {
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    bool const even = arguments.round % 2 == 0;
    device sz_regex_result_metal_t const *source = even ? round.groups : round.groups_next;
    device sz_regex_result_metal_t *target = even ? round.groups_next : round.groups;
    uint const count = atomic_load_explicit(round.counters + sz_regex_groups_counter_metal_k, memory_order_relaxed);
    for (uint group = position; group < count; group += threads)
        target[group] = sz_regex_hops_metal_(source, source[group]);
}

/** Gives every link that paused at the end of the lookaround table's window its settled group's
 *  match, when it has one, and unlinks it, so the links double as though it never paused. */
kernel void sz_regex_take_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                        constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                        uint position [[thread_position_in_grid]], uint threads [[threads_per_grid]]) {
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    uint const links_count = atomic_load_explicit(round.counters + sz_regex_links_counter_metal_k,
                                                  memory_order_relaxed);
    for (uint item = position; item < links_count; item += threads) {
        device sz_regex_result_metal_t *link = round.links + round.links_list[item];
        if (link->link == sz_regex_none_metal_k || !(link->link & sz_regex_group_metal_k)) continue;
        sz_regex_result_metal_t const group = round.groups[link->link & ~sz_regex_group_metal_k];
        link->link = sz_regex_none_metal_k;
        if (group.pattern != sz_regex_none_metal_k) link->pattern = group.pattern, link->end = group.end;
    }
}

/** Settles every folded candidate whose first rune found no match once the links settled, one
 *  thread per candidate: one whose unit folds to several runes tries the later ones, as
 *  @ref sz_regex_match_at_metal_ tries them, each walked to its end without links, as
 *  @c sz_regex_retry_folded_simt_kernel_ does. */
kernel void sz_regex_retry_folded_metal_kernel_(
    device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
    device uchar const *haystacks [[buffer(2)]], device uchar const *haystacks_offsets [[buffer(3)]],
    constant sz_regex_arguments_metal_t &arguments [[buffer(10)]], uint position [[thread_position_in_grid]],
    uint threads [[threads_per_grid]], threadgroup uchar *staged [[threadgroup(0)]],
    uint lane [[thread_index_in_threadgroup]], uint group_width [[threads_per_threadgroup]]) {
    sz_regex_table_metal_t const table = sz_regex_stage_metal_(engine, scratch, arguments, arguments.forward_rows,
                                                               arguments.forward_cells, staged, lane, group_width);
    sz_regex_automata_metal_t const automata = sz_regex_automata_metal_(engine, arguments);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    ulong const count = round.chunk_offsets[arguments.chunks_count];
    sz_regex_pike_metal_t pike;
    for (ulong item = position; item < count; item += threads) {
        ulong const mark = arguments.segment_begin + round.slots[item];
        sz_regex_haystack_metal_t const haystack = sz_regex_haystack_metal_(
            corpus, sz_regex_haystack_at_metal_(corpus, mark,
                                                round.chunk_haystacks[round.slots[item] >> arguments.chunk_shift]));
        ulong const start = mark - haystack.first;
        if (start == haystack.length ||
            sz_regex_settle_metal_(round, uint(item), mark, haystack).pattern != sz_regex_none_metal_k)
            continue;
        sz_utf8_folded_image_metal_t image;
        sz_regex_unit_after_metal_(sz_regex_folded_metal_k, haystack.text, haystack.length, start, image);
        uint second = 0;
        while (second < image.length && !(image.rune_ends >> second & 1)) ++second;
        if (++second >= image.length) continue;
        ulong end, offset = second;
        uint pattern;
        pike.first = haystack.first;
        if (sz_regex_match_at_metal_(table, automata, arguments, pike, sz_regex_folded_dfa_metal_k, haystack.text,
                                     haystack.length, start, end, pattern, offset))
            round.results[item].link = sz_regex_none_metal_k, round.results[item].pattern = pattern,
            round.results[item].end = haystack.first + end;
    }
}

/**
 *  @brief Follows the search through one chunk from every candidate as an entry, right to left, one
 *      thread per chunk: where it goes next, and the first candidate past the chunk it reaches.
 *
 *  After a match the search takes the first candidate at or past its end, refusing an empty
 *  match right there; after a candidate it passes, the next one. Each haystack's chain is
 *  seeded at its first candidate, or past the cursor the previous segment carried for the
 *  haystack straddling them.
 */
kernel void sz_regex_chain_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                         device uchar const *haystacks [[buffer(2)]],
                                         device uchar const *haystacks_offsets [[buffer(3)]],
                                         constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                         uint position [[thread_position_in_grid]], uint threads [[threads_per_grid]]) {
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    uint const count = uint(round.chunk_offsets[arguments.chunks_count]);
    ulong const begin = arguments.segment_begin;
    ulong const cursor = round.carry->cursor;
    bool const refusing = round.carry->refusing != 0;

    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += threads) {
        uint const first = uint(round.chunk_offsets[chunk]), end = uint(round.chunk_offsets[chunk + 1]);
        if (first == end) continue;
        ulong haystack_index = sz_regex_haystack_at_metal_(corpus, begin + round.slots[end - 1],
                                                           round.chunk_haystacks[chunk]);
        sz_regex_haystack_metal_t haystack = sz_regex_haystack_metal_(corpus, haystack_index);
        bool leaves = false;
        for (uint index = end; index-- != first;) {
            ulong const mark = begin + round.slots[index];
            while (mark < haystack.first) haystack = sz_regex_haystack_metal_(corpus, --haystack_index);
            ulong const last = haystack.first + haystack.length;
            sz_regex_result_metal_t const result = sz_regex_settle_metal_(round, index, mark, haystack);
            uint next = index + 1;
            if (result.pattern != sz_regex_none_metal_k) {
                next = result.end >= arguments.segment_end
                           ? count
                           : sz_regex_lower_bound_metal_(round.slots, index, count, result.end - begin);
                if (next < count && begin + round.slots[next] == result.end) {
                    sz_regex_result_metal_t const after = sz_regex_settle_metal_(round, next, result.end, haystack);
                    if (after.pattern != sz_regex_none_metal_k && after.end == result.end) ++next;
                }
            }
            uint const jump = next < count && begin + round.slots[next] <= last ? next : sz_regex_none_metal_k;
            round.nexts[index] = jump;
            round.exits[index] = jump == sz_regex_none_metal_k || jump >= end ? jump : round.exits[jump];
            leaves = leaves || round.exits[index] != sz_regex_none_metal_k;

            // A haystack started in this segment is seeded at its first candidate; the one
            // straddling from the last segment at its first candidate past the cursor, unless that
            // is a refused empty match.
            bool seed;
            if (haystack.first >= begin) seed = index == 0 || begin + round.slots[index - 1] < haystack.first;
            else {
                bool const first_past = mark >= cursor && (index == 0 || begin + round.slots[index - 1] < cursor);
                bool const refused = first_past && refusing && mark == cursor &&
                                     result.pattern != sz_regex_none_metal_k && result.end == mark;
                seed = first_past && !refused;
                if (index != 0 && !first_past && mark > cursor) {
                    ulong const previous = begin + round.slots[index - 1];
                    bool const previous_first = previous >= cursor &&
                                                (index == 1 || begin + round.slots[index - 2] < cursor);
                    if (previous_first && refusing && previous == cursor) {
                        sz_regex_result_metal_t const before = sz_regex_settle_metal_(round, index - 1, previous,
                                                                                      haystack);
                        seed = before.pattern != sz_regex_none_metal_k && before.end == previous;
                    }
                }
            }
            round.keep[index] = seed;
        }
        if (leaves) round.flags[0] = 1;
    }
}

/** One round of pointer doubling over the chunk exits: every kept candidate keeps the ones its
 *  next hops reach, and jumps that much further; a round after one that left nothing to follow
 *  retires at once. */
kernel void sz_regex_double_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                          constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                          uint position [[thread_position_in_grid]],
                                          uint threads [[threads_per_grid]]) {
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    if (!round.flags[arguments.round]) return;
    bool const even = arguments.round % 2 == 0;
    device uint const *source = even ? round.exits : round.exits_next;
    device uint *target = even ? round.exits_next : round.exits;
    uint const count = uint(round.chunk_offsets[arguments.chunks_count]);
    for (uint index = position; index < count; index += threads) {
        // A mark set in this round only reaches candidates on the chain anyway.
        bool const kept = round.keep[index] != 0;
        uint jump = source[index];
        for (uint hop = 1; hop != sz_regex_round_hops_metal_k && jump != sz_regex_none_metal_k; ++hop) {
            if (kept) round.keep[jump] = 1;
            jump = source[jump];
        }
        if (kept && jump != sz_regex_none_metal_k) round.keep[jump] = 1;
        target[index] = jump == sz_regex_none_metal_k ? jump : source[jump];
        if (target[index] != sz_regex_none_metal_k) round.flags[arguments.round + 1] = 1;
    }
}

/** Follows the chain through one chunk from the candidates the doubling kept, left to right, one
 *  thread per chunk: counting the matches it keeps, flagging each and counting those of the
 *  extracted pattern, or, once their offsets are scanned, storing what the call asks. */
kernel void sz_regex_sweep_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                         device uchar const *haystacks [[buffer(2)]],
                                         device uchar const *haystacks_offsets [[buffer(3)]],
                                         device sz_match_metal_t *matches [[buffer(4)]],
                                         device ulong *matches_offsets [[buffer(5)]],
                                         device ulong *counts [[buffer(6)]],
                                         constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                         uint position [[thread_position_in_grid]], uint threads [[threads_per_grid]]) {
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    ulong const begin = arguments.segment_begin, width = arguments.chunk_width;
    bool const storing = arguments.storing != 0;
    if (position == 0 && !storing)
        round.chunk_kept[arguments.chunks_count] = 0, round.chunk_sums[arguments.chunks_count] = 0;

    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += threads) {
        ulong const low = begin + chunk * width, high = min(low + width, arguments.segment_end);
        uint const first = uint(round.chunk_offsets[chunk]), end = uint(round.chunk_offsets[chunk + 1]);
        ulong const base = storing ? round.carry->running + round.chunk_kept[chunk] : 0;
        ulong haystack_index = round.chunk_haystacks[chunk], pending = haystack_index;
        sz_regex_haystack_metal_t haystack = sz_regex_haystack_metal_(corpus, haystack_index);
        if (haystack.first < low) ++pending;
        uint expected = sz_regex_none_metal_k, kept = 0;
        ulong last_start = 0, last_end = 0, kept_pattern = 0;
        for (uint index = first; index != end; ++index) {
            ulong const mark = begin + round.slots[index];
            if (storing)
                for (; pending < corpus.count && sz_regex_first_slot_metal_(corpus, pending) <= mark; ++pending)
                    sz_regex_boundary_store_metal_(arguments, matches_offsets, counts, pending, base + kept);
            while (haystack_index + 1 < corpus.count && sz_regex_first_slot_metal_(corpus, haystack_index + 1) <= mark)
                haystack = sz_regex_haystack_metal_(corpus, ++haystack_index);
            if (!storing) round.kept[index] = 0;
            if (!round.keep[index] && index != expected) continue;
            expected = round.nexts[index];
            sz_regex_result_metal_t const result = sz_regex_settle_metal_(round, index, mark, haystack);
            if (result.pattern == sz_regex_none_metal_k) continue;
            if (storing && arguments.verb == sz_regex_finding_metal_k && base + kept < arguments.matches_capacity)
                matches[base + kept] = {haystack_index, result.pattern, mark - haystack.first, result.end - mark};
            if (!storing) round.kept[index] = 1, kept_pattern += result.pattern == arguments.pattern_index;
            last_start = mark, last_end = result.end, ++kept;
        }
        if (!storing) {
            round.chunk_kept[chunk] = kept, round.chunk_sums[chunk] = kept_pattern;
            continue;
        }
        for (; pending < corpus.count && sz_regex_first_slot_metal_(corpus, pending) < high; ++pending)
            sz_regex_boundary_store_metal_(arguments, matches_offsets, counts, pending, base + kept);
        // The chunk with the segment's last kept match tells the carry where its haystack resumes.
        if (kept && round.chunk_kept[chunk] + kept == round.chunk_kept[arguments.chunks_count])
            round.carry->last_start = last_start, round.carry->last_end = last_end;
    }
}

/** Hands the next segment its carry, on one thread: the running total of kept matches, and where
 *  the haystack straddling the two resumes, which is past its last kept match here, and the last
 *  boundary of whatever the call writes. */
kernel void sz_regex_carry_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                         device uchar const *haystacks [[buffer(2)]],
                                         device uchar const *haystacks_offsets [[buffer(3)]],
                                         device ulong *matches_offsets [[buffer(5)]],
                                         device ulong *counts [[buffer(6)]],
                                         constant sz_regex_arguments_metal_t &arguments [[buffer(10)]]) {
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    device sz_regex_carry_metal_t *carry = round.carry;
    ulong const kept = round.chunk_kept[arguments.chunks_count];
    ulong const begin = arguments.segment_begin, end = arguments.segment_end;
    bool const extracting = arguments.verb == sz_regex_extracting_metal_k;
    if (end < arguments.slots_count) {
        ulong const first = sz_regex_first_slot_metal_(corpus, sz_regex_haystack_at_metal_(corpus, end, 0));
        if (kept && carry->last_start >= first) carry->cursor = carry->last_end, carry->refusing = 1;
        else if (first >= begin) carry->cursor = first, carry->refusing = 0;
        if (extracting && first >= begin && first < end)
            carry->straddle = carry->counted + sz_regex_counted_before_metal_(round, arguments, first);
    }
    carry->running += kept;
    sz_regex_boundary_store_metal_(arguments, matches_offsets, counts, arguments.haystacks_count, carry->running);
    if (arguments.verb == sz_regex_replacing_metal_k) {
        carry->summed += round.chunk_sums[arguments.chunks_count];
        matches_offsets[arguments.haystacks_count] = arguments.slots_count - arguments.haystacks_count + carry->summed;
    }
    if (extracting) {
        carry->counted += round.chunk_sums[arguments.chunks_count];
        carry->summed += round.chunk_covers[arguments.chunks_count];
        matches_offsets[arguments.haystacks_count] = carry->summed;
    }
}

/** Saves the total each thread's range of the counts follows, before differences overwrite it. */
kernel void sz_regex_counts_edges_metal_kernel_(device uchar *scratch [[buffer(1)]], device ulong *counts [[buffer(6)]],
                                                constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                                uint position [[thread_position_in_grid]],
                                                uint threads [[threads_per_grid]]) {
    device ulong *edges = sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.results);
    ulong const count = arguments.haystacks_count;
    ulong const begin = min(ulong(position) * sz_size_divide_round_up_metal_<ulong>(count, threads), count);
    edges[position] = begin ? counts[(begin - 1) * arguments.counts_stride] : 0;
}

/** Turns the running totals the sweep left in the counts into each haystack's own count, every
 *  thread over its own range right to left, so it reads each total before overwriting it. */
kernel void sz_regex_counts_metal_kernel_(device uchar *scratch [[buffer(1)]], device ulong *counts [[buffer(6)]],
                                          constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                          uint position [[thread_position_in_grid]],
                                          uint threads [[threads_per_grid]]) {
    device ulong const *edges = sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.results);
    ulong const count = arguments.haystacks_count, stride = arguments.counts_stride;
    ulong const per_thread = sz_size_divide_round_up_metal_<ulong>(count, threads);
    ulong const begin = min(ulong(position) * per_thread, count), end = min(begin + per_thread, count);
    for (ulong index = end; index-- != begin;)
        counts[index * stride] -= index == begin ? edges[position] : counts[(index - 1) * stride];
}

/** Measures every kept match's replacement, one thread per candidate in each thread's share of the
 *  walks, leaving how far it moves what follows. */
kernel void sz_regex_expand_metal_kernel_(
    device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
    device uchar const *haystacks [[buffer(2)]], device uchar const *haystacks_offsets [[buffer(3)]],
    device uchar const *replacements [[buffer(7)]], device uchar const *replacements_offsets [[buffer(8)]],
    constant sz_regex_arguments_metal_t &arguments [[buffer(10)]], uint position [[thread_position_in_grid]],
    threadgroup uchar *staged [[threadgroup(0)]], uint lane [[thread_index_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]]) {
    sz_regex_table_metal_t const table = sz_regex_direct_metal_(engine, scratch, arguments, arguments.forward_rows,
                                                                staged, lane, group_width);
    sz_regex_automata_metal_t const automata = sz_regex_automata_metal_(engine, arguments);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    sz_strings_tape_metal_t const templates = {replacements, replacements_offsets, arguments.replacements_layout};
    ulong const count = round.chunk_offsets[arguments.chunks_count];
    if (position >= arguments.walks_count) return;
    sz_regex_pike_metal_t pike = sz_regex_pike_carve_metal_(round, arguments, position);
    for (ulong item = position; item < count; item += arguments.walks_count) {
        round.drifts[item] = 0;
        if (!round.kept[item]) continue;
        ulong const mark = arguments.segment_begin + round.slots[item];
        sz_regex_haystack_metal_t const haystack = sz_regex_haystack_metal_(
            corpus, sz_regex_haystack_at_metal_(corpus, mark,
                                                round.chunk_haystacks[round.slots[item] >> arguments.chunk_shift]));
        sz_regex_result_metal_t const result = sz_regex_linked_metal_(round, uint(item));
        pike.first = haystack.first;
        ulong const expanded = sz_regex_expand_metal_(
            table, automata, arguments, pike, haystack.text, haystack.length,
            sz_strings_start_metal_(templates, result.pattern), sz_strings_length_metal_(templates, result.pattern),
            mark - haystack.first, result.end - haystack.first, result.pattern, nullptr, 0, 0);
        round.drifts[item] = expanded - (result.end - mark);
    }
}

/** Sums the drift of every chunk's kept matches, one thread per chunk, and tells the chunks the
 *  last of them reaches into that they start covered. */
kernel void sz_regex_drift_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                         constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                         uint position [[thread_position_in_grid]], uint threads [[threads_per_grid]]) {
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    if (position == 0) round.chunk_sums[arguments.chunks_count] = 0;
    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += threads) {
        ulong drift = 0, covered = 0;
        for (uint index = uint(round.chunk_offsets[chunk]); index != uint(round.chunk_offsets[chunk + 1]); ++index)
            if (round.kept[index]) drift += round.drifts[index], covered = sz_regex_linked_metal_(round, index).end;
        round.chunk_sums[chunk] = drift;
        for (ulong next = chunk + 1;
             next < arguments.chunks_count && arguments.segment_begin + (next << arguments.chunk_shift) <= covered;
             ++next)
            round.chunk_covers[next] = covered;
    }
}

/** Writes one chunk's share of the rewritten haystacks, one thread per chunk in each thread's share
 *  of the walks, as @c sz_regex_rewrite_simt_kernel_ does. */
kernel void sz_regex_rewrite_metal_kernel_(
    device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
    device uchar const *haystacks [[buffer(2)]], device uchar const *haystacks_offsets [[buffer(3)]],
    device ulong *offsets [[buffer(5)]], device uchar const *replacements [[buffer(7)]],
    device uchar const *replacements_offsets [[buffer(8)]], device uchar *target [[buffer(9)]],
    constant sz_regex_arguments_metal_t &arguments [[buffer(10)]], uint position [[thread_position_in_grid]],
    threadgroup uchar *staged [[threadgroup(0)]], uint lane [[thread_index_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]]) {
    sz_regex_table_metal_t const table = sz_regex_direct_metal_(engine, scratch, arguments, arguments.forward_rows,
                                                                staged, lane, group_width);
    sz_regex_automata_metal_t const automata = sz_regex_automata_metal_(engine, arguments);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    sz_strings_tape_metal_t const templates = {replacements, replacements_offsets, arguments.replacements_layout};
    ulong const begin = arguments.segment_begin, width = arguments.chunk_width, capacity = arguments.target_capacity;
    if (position >= arguments.walks_count) return;
    sz_regex_pike_metal_t pike = sz_regex_pike_carve_metal_(round, arguments, position);
    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += arguments.walks_count) {
        ulong const low = begin + chunk * width, high = min(low + width, arguments.segment_end);
        uint index = uint(round.chunk_offsets[chunk]), end = uint(round.chunk_offsets[chunk + 1]);
        ulong drift = round.carry->summed + round.chunk_sums[chunk];
        ulong covered = max(round.carry->last_end, round.chunk_covers[chunk]);
        ulong haystack_index = round.chunk_haystacks[chunk];
        sz_regex_haystack_metal_t haystack = sz_regex_haystack_metal_(corpus, haystack_index);
        for (ulong slot = low; slot != high; ++slot) {
            if (haystack_index + 1 < corpus.count && sz_regex_first_slot_metal_(corpus, haystack_index + 1) == slot)
                haystack = sz_regex_haystack_metal_(corpus, ++haystack_index);
            if (slot == haystack.first) offsets[haystack_index] = slot - haystack_index + drift;
            else if (slot > covered && slot - haystack_index - 1 + drift < capacity)
                target[slot - haystack_index - 1 + drift] = haystack.text[slot - haystack.first - 1];
            if (index == end || begin + round.slots[index] != slot) continue;
            if (round.kept[index]) {
                sz_regex_result_metal_t const result = sz_regex_linked_metal_(round, index);
                pike.first = haystack.first;
                sz_regex_expand_metal_(table, automata, arguments, pike, haystack.text, haystack.length,
                                       sz_strings_start_metal_(templates, result.pattern),
                                       sz_strings_length_metal_(templates, result.pattern), slot - haystack.first,
                                       result.end - haystack.first, result.pattern, target, capacity,
                                       slot - haystack_index + drift);
                drift += round.drifts[index], covered = result.end;
            }
            ++index;
        }
    }
}

/** Finds every haystack's first kept match of the extracted pattern and counts its rows, one thread
 *  per chunk in each thread's share of the walks, as @c sz_regex_firsts_simt_kernel_ does. */
kernel void sz_regex_firsts_metal_kernel_(device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
                                          device uchar const *haystacks [[buffer(2)]],
                                          device uchar const *haystacks_offsets [[buffer(3)]],
                                          constant sz_regex_arguments_metal_t &arguments [[buffer(10)]],
                                          uint position [[thread_position_in_grid]],
                                          threadgroup uchar *staged [[threadgroup(0)]],
                                          uint lane [[thread_index_in_threadgroup]],
                                          uint group_width [[threads_per_threadgroup]]) {
    sz_regex_table_metal_t const table = sz_regex_direct_metal_(engine, scratch, arguments, arguments.forward_rows,
                                                                staged, lane, group_width);
    sz_regex_automata_metal_t const automata = sz_regex_automata_metal_(engine, arguments);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    ulong const begin = arguments.segment_begin;
    if (position == 0) round.chunk_covers[arguments.chunks_count] = 0;
    if (position >= arguments.walks_count) return;
    sz_regex_pike_metal_t pike = sz_regex_pike_carve_metal_(round, arguments, position);
    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += arguments.walks_count) {
        ulong ranked = round.carry->counted + round.chunk_sums[chunk], preceding = 0, rows = 0;
        ulong haystack_index = round.chunk_haystacks[chunk], ranked_haystack = sz_regex_unset_metal_k;
        sz_regex_haystack_metal_t haystack = sz_regex_haystack_metal_(corpus, haystack_index);
        for (uint index = uint(round.chunk_offsets[chunk]); index != uint(round.chunk_offsets[chunk + 1]); ++index) {
            ulong const mark = begin + round.slots[index];
            round.exits[index] = 0;
            while (haystack_index + 1 < corpus.count && sz_regex_first_slot_metal_(corpus, haystack_index + 1) <= mark)
                haystack = sz_regex_haystack_metal_(corpus, ++haystack_index);
            if (!round.kept[index]) continue;
            sz_regex_result_metal_t const result = sz_regex_linked_metal_(round, index);
            if (result.pattern != arguments.pattern_index) continue;
            if (ranked_haystack != haystack_index)
                ranked_haystack = haystack_index,
                preceding = haystack.first >= begin ? round.carry->counted + sz_regex_counted_before_metal_(
                                                                                 round, arguments, haystack.first)
                                                    : round.carry->straddle;
            if (ranked++ != preceding) continue;
            pike.first = haystack.first;
            round.exits[index] = uint(sz_regex_rows_write_metal_(
                table, automata, arguments, pike, haystack.text, haystack.length, haystack_index, mark - haystack.first,
                result.end - haystack.first, result.pattern, nullptr, 0, 0));
            rows += round.exits[index];
        }
        round.chunk_covers[chunk] = rows;
    }
}

/** Writes the rows the firsts kernel counted and where each haystack starting in a chunk begins
 *  among them, one thread per chunk in each thread's share of the walks. */
kernel void sz_regex_rows_metal_kernel_(
    device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
    device uchar const *haystacks [[buffer(2)]], device uchar const *haystacks_offsets [[buffer(3)]],
    device sz_match_metal_t *rows [[buffer(4)]], device ulong *rows_offsets [[buffer(5)]],
    constant sz_regex_arguments_metal_t &arguments [[buffer(10)]], uint position [[thread_position_in_grid]],
    threadgroup uchar *staged [[threadgroup(0)]], uint lane [[thread_index_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]]) {
    sz_regex_table_metal_t const table = sz_regex_direct_metal_(engine, scratch, arguments, arguments.forward_rows,
                                                                staged, lane, group_width);
    sz_regex_automata_metal_t const automata = sz_regex_automata_metal_(engine, arguments);
    sz_regex_corpus_metal_t const corpus = sz_regex_corpus_metal_(haystacks, haystacks_offsets, arguments);
    sz_regex_round_metal_t const round = sz_regex_round_metal_(scratch, arguments);
    ulong const begin = arguments.segment_begin, width = arguments.chunk_width;
    if (position >= arguments.walks_count) return;
    sz_regex_pike_metal_t pike = sz_regex_pike_carve_metal_(round, arguments, position);
    for (ulong chunk = position; chunk < arguments.chunks_count; chunk += arguments.walks_count) {
        ulong const low = begin + chunk * width, high = min(low + width, arguments.segment_end);
        ulong const base = round.carry->summed + round.chunk_covers[chunk];
        ulong haystack_index = round.chunk_haystacks[chunk], pending = haystack_index, written = 0;
        sz_regex_haystack_metal_t haystack = sz_regex_haystack_metal_(corpus, haystack_index);
        if (haystack.first < low) ++pending;
        for (uint index = uint(round.chunk_offsets[chunk]); index != uint(round.chunk_offsets[chunk + 1]); ++index) {
            ulong const mark = begin + round.slots[index];
            for (; pending < corpus.count && sz_regex_first_slot_metal_(corpus, pending) <= mark; ++pending)
                rows_offsets[pending] = base + written;
            while (haystack_index + 1 < corpus.count && sz_regex_first_slot_metal_(corpus, haystack_index + 1) <= mark)
                haystack = sz_regex_haystack_metal_(corpus, ++haystack_index);
            if (!round.exits[index]) continue;
            sz_regex_result_metal_t const result = sz_regex_linked_metal_(round, index);
            pike.first = haystack.first;
            sz_regex_rows_write_metal_(table, automata, arguments, pike, haystack.text, haystack.length, haystack_index,
                                       mark - haystack.first, result.end - haystack.first, result.pattern, rows,
                                       arguments.matches_capacity, base + written);
            written += round.exits[index];
        }
        for (; pending < corpus.count && sz_regex_first_slot_metal_(corpus, pending) < high; ++pending)
            rows_offsets[pending] = base + written;
    }
}
