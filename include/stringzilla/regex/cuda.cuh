/**
 *  @file include/stringzilla/regex/cuda.cuh
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief CUDA host side of regular expressions: the round block a call runs out of, the
 *      launches that prime, mark, walk, settle and store, and the @c _cuda exports, over the
 *      kernels of `regex/simt.cuh`.
 *
 *  No compute verb allocates. The round block is taken once at init, sized from the device's
 *  resident threads, and a call reads the tape's offsets on the host to cut its corpus into
 *  segments the block holds, leaving everything else to the device: lookaround passes sweep the
 *  segments first, priming runs right to left over them, then marking, walking and the chain left
 *  to right, each segment's carry handing the next where its straddling haystack resumes. The
 *  lookaround table is a ring of two segments, each sweep filling each segment once.
 *
 *  A verb joins its stream only where a vocabulary with lookarounds meets a haystack running past
 *  the ring's window: there it reads back whether end walks paused at the window's end, which it
 *  then slides on until they finish, or refused, which fails the call.
 *
 *  @sa include/stringzilla/regex/simt.cuh
 *  @sa include/stringzilla/regex/rocm.cuh
 *  @sa include/stringzilla/regex/metal.h, whose round this mirrors launch for dispatch
 */
#ifndef STRINGZILLA_REGEX_CUDA_CUH_
#define STRINGZILLA_REGEX_CUDA_CUH_

#include "stringzilla/cuda.cuh"
#include "stringzilla/regex/simt.cuh"

#if STRINGZILLA_ARCH_CUDA_

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Host Plumbing

/** Blocks per multiprocessor a grid-strided launch covers the device with. */
enum { sz_regex_blocks_per_multiprocessor_cuda_k = 8 };

/** The warp walk of the links that outlived their first thread, with CUDA's cross-lane steps. */
static __global__ void sz_regex_continue_cuda_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                      sz_regex_round_simt_t round) {
    sz_regex_continue_simt_(&engine, haystacks, &round, sz_false_k, sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_,
                            sz_lanes_any_cuda_);
}

/** The warp walk of folded links that outlived their first thread, with CUDA's cross-lane steps. */
static __global__ void sz_regex_continue_folded_cuda_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                             sz_regex_round_simt_t round) {
    sz_regex_continue_simt_(&engine, haystacks, &round, sz_true_k, sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_,
                            sz_lanes_any_cuda_);
}

/** The segment geometry of an engine's round block, and the byte offsets of everything in it. */
typedef struct sz_regex_arena_cuda_t {
    sz_size_t chunks, segment_slots, pike_slots, pike_bytes, positions_bytes, passes_bytes, set_words, images_bytes;
    sz_size_t entries, edges, carry, counters, flags, scan_chain, scan_states;
    sz_size_t chunk_offsets, chunk_kept, chunk_sums, chunk_covers, chunk_haystacks;
    sz_size_t frontiers, compositions, compositions_next, handoffs, handoff_edges, undecided, affines, affines_next;
    sz_size_t looks, starts, checkpoints;
    sz_size_t slots, results, links, links_next, links_list, nexts, exits, exits_next, keep, walks, total;
} sz_regex_arena_cuda_t;

/** Rounds up to a multiple of 16 bytes, which every array of the round block starts at. */
STRINGZILLA_INLINE sz_size_t sz_regex_aligned_cuda_(sz_size_t bytes) { return (bytes + 15) / 16 * 16; }

/** Bytes of one automaton's column images, a count per column of its rows and then a priming
 *  set per column, none without either DFA; both sit at the head of the round block, which
 *  every call keeps. */
STRINGZILLA_INLINE sz_size_t sz_regex_images_bytes_cuda_(sz_regex_engine_t const *engine) {
    if (!engine->forward_rows && !engine->reverse_rows) return 0;
    return sz_regex_aligned_cuda_(sz_regex_stride_(engine->classes_count, engine->lookarounds_count) *
                                  (1 + sz_regex_priming_states_simt_k) * sizeof(sz_u32_t));
}

/** Rounds of pointer doubling a chain across @p chunks chunks takes, an even count, which leaves
 *  the compositions in their first buffer. */
STRINGZILLA_INLINE sz_size_t sz_regex_rounds_count_cuda_(sz_size_t chunks) {
    sz_size_t rounds = 0, reach;
    for (reach = 1; reach <= chunks; reach *= sz_regex_round_hops_simt_k) ++rounds;
    return rounds + rounds % 2;
}

/**
 *  @brief Lays the round block of @p engine out for segments of @p chunks chunks,
 *      after the column images.
 *
 *  The segment's own arrays come first; the walks follow, which hold one PikeVM, two position sets,
 *  or a PikeVM and two lookaround sets per thread of a launch, at least a block's worth. Each
 *  segment's edge, the entry of its first chunk, has a slot of its own, so a call spans at most as
 *  many segments as a segment has chunks; so does every set handed across it, per flow: positions,
 *  then each level behind and ahead, then each body a DFA walks. The lookaround table is a ring of
 *  two segments. The leaders of paused walks' groups live in the frontiers and the groups in the
 *  checkpoints, which priming and the links kernel are done with by then.
 */
STRINGZILLA_INLINE sz_regex_arena_cuda_t sz_regex_arena_cuda_(sz_regex_engine_t const *engine, sz_size_t chunks) {
    sz_size_t const slots = chunks * sz_regex_chunk_slots_simt_k, looks_words = sz_regex_looks_words_simt_(engine);
    sz_u32_t const dfas = sz_regex_body_dfas_simt_(engine);
    sz_size_t set_words = sz_max_of_two((sz_size_t)engine->positions_words, looks_words), sets, flows, index;
    sz_regex_arena_cuda_t arena;
    for (index = 0; index != engine->lookarounds_count; ++index)
        if (dfas & (sz_u32_t)sz_regex_look_lookaround_k << index)
            set_words = sz_max_of_two(
                set_words, (sz_size_t)(sz_regex_body_reach_simt_(engine, (sz_u32_t)index, STRINGZILLA_NULL) + 63) / 64);
    sets = set_words ? chunks : 0;
    flows = 1 + 2 * (sz_size_t)sz_regex_levels_simt_(engine) + (sz_size_t)sz_u32_popcount(dfas);
    arena.chunks = chunks, arena.segment_slots = slots, arena.set_words = set_words;
    arena.images_bytes = sz_regex_images_bytes_cuda_(engine);
    arena.pike_slots = 2 * sz_regex_groups_most_(engine) + 2;
    arena.pike_bytes = sz_regex_pike_bytes_simt_(engine, arena.pike_slots);
    arena.positions_bytes = sz_regex_aligned_cuda_(2 * (sz_size_t)engine->positions_words * sizeof(sz_u64_t));
    arena.passes_bytes = looks_words ? arena.pike_bytes + 2 * looks_words * sizeof(sz_u64_t) : 0;
    arena.entries = 2 * arena.images_bytes;
    arena.edges = arena.entries + sz_regex_aligned_cuda_(chunks * sizeof(sz_u32_t));
    arena.carry = arena.edges + sz_regex_aligned_cuda_(chunks * sizeof(sz_u32_t));
    arena.counters = arena.carry + sz_regex_aligned_cuda_(sizeof(sz_regex_carry_simt_t));
    arena.flags = arena.counters + sz_regex_aligned_cuda_((sz_regex_refused_counter_simt_k + 1) * sizeof(sz_u32_t));
    arena.scan_chain = arena.flags +
                       sz_regex_aligned_cuda_((sz_regex_rounds_count_cuda_(chunks) + 2) * sizeof(sz_u32_t));
    arena.scan_states = arena.scan_chain + 2 * sizeof(sz_u64_t);
    arena.chunk_offsets = arena.scan_states + 2 * sz_chain_tiles_max_k * sizeof(sz_u64_t);
    arena.chunk_kept = arena.chunk_offsets + sz_regex_aligned_cuda_((chunks + 1) * sizeof(sz_size_t));
    arena.chunk_sums = arena.chunk_kept + sz_regex_aligned_cuda_((chunks + 1) * sizeof(sz_size_t));
    arena.chunk_covers = arena.chunk_sums + sz_regex_aligned_cuda_((chunks + 1) * sizeof(sz_size_t));
    arena.chunk_haystacks = arena.chunk_covers + sz_regex_aligned_cuda_((chunks + 1) * sizeof(sz_size_t));
    arena.frontiers = arena.chunk_haystacks + sz_regex_aligned_cuda_(chunks * sizeof(sz_size_t));
    arena.compositions = arena.frontiers +
                         sz_regex_aligned_cuda_(sz_max_of_two(chunks * sizeof(sz_regex_frontier_simt_t),
                                                              engine->forward_states_count * sizeof(sz_u32_t)));
    arena.compositions_next = arena.compositions + sz_regex_aligned_cuda_(chunks * sizeof(sz_regex_composition_simt_t));
    arena.handoffs = arena.compositions_next + sz_regex_aligned_cuda_(chunks * sizeof(sz_regex_composition_simt_t));
    arena.handoff_edges = arena.handoffs + sets * set_words * sizeof(sz_u64_t);
    arena.undecided = arena.handoff_edges +
                      sz_regex_aligned_cuda_(set_words ? (chunks + 1) * flows * set_words * sizeof(sz_u64_t) : 0);
    arena.affines = arena.undecided + sz_regex_aligned_cuda_(sets * sizeof(sz_regex_undecided_simt_t));
    arena.affines_next = arena.affines + sets * sizeof(sz_regex_affine_simt_t);
    arena.looks = arena.affines_next + sets * sizeof(sz_regex_affine_simt_t);
    arena.starts = arena.looks + (engine->lookarounds_count ? 2 * slots * sizeof(sz_u32_t) : 0);
    arena.checkpoints = arena.starts + slots / 8;
    arena.slots = arena.checkpoints +
                  (slots / sz_regex_checkpoint_slots_simt_k + 1) * sizeof(sz_regex_checkpoint_simt_t);
    arena.results = arena.slots + sz_regex_aligned_cuda_(slots * sizeof(sz_u32_t));
    arena.links = arena.results + slots * sizeof(sz_regex_result_simt_t);
    arena.links_next = arena.links + slots * sizeof(sz_regex_result_simt_t);
    // The chain's arrays live in the links' second buffer, dead once links settle into the first.
    arena.nexts = arena.links_next;
    arena.exits = arena.nexts + sz_regex_aligned_cuda_(slots * sizeof(sz_u32_t));
    arena.exits_next = arena.exits + sz_regex_aligned_cuda_(slots * sizeof(sz_u32_t));
    arena.keep = arena.exits_next + sz_regex_aligned_cuda_(slots * sizeof(sz_u32_t));
    arena.links_list = arena.links_next + (slots + 1) * sizeof(sz_regex_result_simt_t);
    arena.walks = arena.links_list + sz_regex_aligned_cuda_(slots * sizeof(sz_u32_t));
    arena.total = arena.walks + sz_max_of_two(slots * sz_regex_walk_slot_bytes_simt_k,
                                              sz_regex_threads_per_block_simt_k *
                                                  sz_max_of_two(sz_max_of_two(arena.pike_bytes, arena.positions_bytes),
                                                                arena.passes_bytes));
    return arena;
}

/** The arena of @p engine 's round block: the widest power of two of chunks whose layout fits. */
STRINGZILLA_INLINE sz_regex_arena_cuda_t sz_regex_arena_of_cuda_(sz_regex_engine_t const *engine) {
    sz_size_t chunks = 1;
    while (sz_regex_arena_cuda_(engine, chunks * 2).total <= engine->scratch_bytes) chunks *= 2;
    return sz_regex_arena_cuda_(engine, chunks);
}

/** Writes each column's image into @p images: the distinct live states any state of @p rows steps
 *  to on it, or one more than a priming set holds when they do not fit, every block's columns
 *  alike, as @c sz_regex_images_metal_ does. */
STRINGZILLA_INLINE void sz_regex_images_cuda_(sz_regex_engine_t const *engine, sz_u32_t const *rows,
                                              sz_size_t states_count, sz_u32_t *images) {
    sz_size_t const stride = sz_regex_stride_(engine->classes_count, engine->lookarounds_count);
    sz_size_t column, state;
    for (column = 0; column != stride; ++column) {
        sz_u32_t *const states = images + stride + column * sz_regex_priming_states_simt_k;
        sz_u32_t count = 0;
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
}

/** Takes the one round block every call of @p engine runs out of, a segment's chunks being
 *  the device's resident threads, and writes the column images at its head, so no compute
 *  verb ever allocates. */
STRINGZILLA_INLINE sz_status_t sz_regex_arena_reserve_cuda_(sz_regex_engine_t *engine, sz_stream_t stream) {
    sz_size_t const resident = sz_device_multiprocessors_cuda_() * sz_device_threads_per_multiprocessor_cuda_();
    sz_size_t const chunks = (sz_size_t)1 << sz_size_log2i_nonzero(
                                 sz_max_of_two(resident, (sz_size_t)sz_regex_threads_per_block_simt_k));
    sz_regex_arena_cuda_t const arena = sz_regex_arena_cuda_(engine, chunks);
    sz_allocator_t *const allocator = &engine->allocator;
    void *const block = allocator->allocate(arena.total, allocator->handle, stream);
    if (!block) return sz_bad_alloc_k;
    // The host writes the images in place, so a device-only allocation cannot hold them.
    if (!sz_memory_accessible_cuda_(block)) {
        allocator->free(block, arena.total, allocator->handle, stream);
        return sz_device_memory_mismatch_k;
    }
    engine->scratch = block, engine->scratch_bytes = arena.total;
    if (engine->reverse_rows)
        sz_regex_images_cuda_(engine, engine->reverse_rows, engine->reverse_states_count, (sz_u32_t *)block);
    if (engine->forward_rows)
        sz_regex_images_cuda_(engine, engine->forward_rows, engine->forward_states_count,
                              (sz_u32_t *)((sz_ptr_t)block + arena.images_bytes));
    return sz_success_k;
}

/** Grid for a grid-strided kernel over @p items, covering the device without exceeding the work. */
STRINGZILLA_INLINE unsigned sz_regex_grid_cuda_(sz_size_t items) {
    sz_size_t blocks = sz_size_divide_round_up(items, sz_regex_threads_per_block_simt_k);
    sz_size_t const covering = sz_max_of_two(sz_device_multiprocessors_cuda_(), (sz_size_t)1) *
                               sz_regex_blocks_per_multiprocessor_cuda_k;
    if (blocks == 0) blocks = 1;
    return (unsigned)sz_min_of_two(blocks, covering);
}

/** Launches @p kernel over @p blocks blocks of the tier's fixed block size, on @p stream. */
STRINGZILLA_INLINE sz_status_t sz_regex_launch_cuda_(void const *kernel, unsigned blocks, void **arguments,
                                                     sz_size_t shared_bytes, sz_stream_t stream) {
    dim3 grid, block;
    grid.x = blocks, grid.y = 1, grid.z = 1;
    block.x = sz_regex_threads_per_block_simt_k, block.y = 1, block.z = 1;
    return sz_launch_cuda_(kernel, grid, block, arguments, shared_bytes, stream);
}

/** Turns @p count values at @p values into their exclusive prefix sum in one launch, each scan of
 *  @p round taking the other of its two buffers of tile states and clearing the one it leaves. */
STRINGZILLA_INLINE sz_status_t sz_regex_scan_cuda_(sz_regex_round_simt_t *round, sz_size_t *values, sz_size_t count,
                                                   sz_stream_t stream) {
    sz_size_t const tiles = sz_min_of_two(sz_size_divide_round_up(count, sz_regex_threads_per_block_simt_k),
                                          (sz_size_t)sz_chain_tiles_max_k);
    sz_size_t elements_per_tile = sz_size_divide_round_up(count, tiles);
    sz_u64_t *states = round->scan_states + (round->scans % 2) * sz_chain_tiles_max_k;
    sz_u64_t *stale = round->scan_states + (++round->scans % 2) * sz_chain_tiles_max_k;
    void *arguments[6];
    arguments[0] = &values, arguments[1] = &count, arguments[2] = &elements_per_tile, arguments[3] = &round->scan_chain;
    arguments[4] = &states, arguments[5] = &stale;
    return sz_regex_launch_cuda_((void const *)sz_regex_scan_simt_kernel_, (unsigned)tiles, arguments, 0, stream);
}

/** The first slot of haystack @p index of a tape, one per byte and one per haystack before it, read
 *  on the host; past the last haystack, the slots the tape spans. */
STRINGZILLA_INLINE sz_size_t sz_regex_first_slot_cuda_(sz_strings_view_t const *haystacks, sz_size_t index) {
    if (haystacks->layout == sz_strings_u64_tape_k)
        return (sz_size_t)(haystacks->data.u64_tape.offsets[index] - haystacks->data.u64_tape.offsets[0]) + index;
    return (sz_size_t)(haystacks->data.u32_tape.offsets[index] - haystacks->data.u32_tape.offsets[0]) + index;
}

/** The slots a tape of haystacks spans: one per byte and one per haystack, read on the host. */
STRINGZILLA_INLINE sz_size_t sz_regex_slots_count_cuda_(sz_strings_view_t const *haystacks) {
    return sz_regex_first_slot_cuda_(haystacks, haystacks->count);
}

/** Whether the haystack holding the last slot of the segment @p round points at runs to the end of
 *  the lookaround table's window or past it, found by binary search over the tape's offsets on the
 *  host: only there can a walk pause or refuse. */
STRINGZILLA_INLINE sz_bool_t sz_regex_runs_past_cuda_(sz_strings_view_t const *haystacks,
                                                      sz_regex_round_simt_t const *round) {
    sz_size_t low = 0, high = haystacks->count;
    while (high - low > 1) {
        sz_size_t const middle = low + (high - low) / 2;
        if (sz_regex_first_slot_cuda_(haystacks, middle) < round->segment_end) low = middle;
        else high = middle;
    }
    return (sz_bool_t)(sz_regex_first_slot_cuda_(haystacks, low + 1) - 1 >= round->looks.end);
}

/** Points @p round at segment @p segment of its corpus, cut into segments of @p slots slots. */
STRINGZILLA_INLINE void sz_regex_segment_cuda_(sz_regex_round_simt_t *round, sz_size_t segment, sz_size_t slots) {
    round->segment = segment, round->segment_begin = segment * slots;
    round->segment_end = sz_min_of_two(round->segment_begin + slots, round->slots_count);
    round->chunks_count = sz_size_divide_round_up(round->segment_end - round->segment_begin,
                                                  sz_regex_chunk_slots_simt_k);
}

/**
 *  @brief Launches one flow of sets over the segment @p round points at.
 *
 *  It launches the bounds of what each chunk hands on, the walks that settle the wide ones, their
 *  affine maps and the doubling that answers them, then, when @p recording, the walk that records
 *  where the lookarounds of the pass hold.
 *
 *  @param[in] level The pass's level, for a lookaround pass.
 *  @param[in] body The lookaround whose body's DFA the flow steps, or none for the pass's NFA over
 *      the members no table fills and no DFA steps.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_flow_cuda_(sz_regex_engine_t const *engine, sz_strings_view_t const *haystacks,
                                                   sz_regex_round_simt_t *round, sz_regex_arena_cuda_t const *arena,
                                                   sz_regex_flow_simt_t flow, sz_u32_t level, sz_u32_t body,
                                                   sz_bool_t recording, sz_stream_t stream) {
    sz_bool_t const positions = (sz_bool_t)(flow == sz_regex_flow_positions_simt_k);
    sz_bool_t const dfa = (sz_bool_t)(body != sz_regex_none_simt_k);
    sz_u32_t const dfas = sz_regex_body_dfas_simt_(engine);
    // A body's DFA steps rows staged as the automata's are.
    sz_size_t const shared_bytes =
        dfa ? sz_regex_staged_bytes_simt_k + (sz_size_t)round->staged_cells * sizeof(sz_u32_t) : 0;
    sz_size_t set = positions ? 0 : 2 * level - (flow == sz_regex_flow_behind_simt_k);
    sz_regex_engine_t launched_engine = *engine;
    sz_strings_view_t launched_haystacks = *haystacks;
    void *walk_arguments[3], *round_arguments[1];
    sz_status_t status;
    sz_size_t step;
    unsigned walks_grid;
    round->flow = flow, round->negated = 0, round->body = body, round->body_states = 0;
    round->members = positions ? 0
                               : sz_regex_members_simt_(engine, level, (sz_bool_t)(flow == sz_regex_flow_behind_simt_k),
                                                        &round->negated);
    if (dfa) {
        round->members = (sz_u32_t)sz_regex_look_lookaround_k << body;
        round->body_states = sz_regex_body_reach_simt_(engine, body, round->body_reach);
        set = 1 + 2 * (sz_size_t)sz_regex_levels_simt_(engine) +
              (sz_size_t)sz_u32_popcount(dfas & (round->members - 1));
    }
    else round->members &= ~dfas & ~sz_regex_singles_simt_(engine);
    round->negated &= round->members;
    round->handoff_edges = (sz_u64_t *)((sz_ptr_t)engine->scratch + arena->handoff_edges) +
                           set * (arena->chunks + 1) * arena->set_words;
    // A body's DFA holds its sets in registers and steps no PikeVM, so its shares hold two sets.
    round->walk_bytes = positions ? arena->positions_bytes
                        : dfa     ? 2 * arena->set_words * sizeof(sz_u64_t)
                                  : arena->passes_bytes;
    round->walk_sets = positions || dfa ? 0 : arena->pike_bytes;
    round->walks_count = sz_min_of_two((arena->total - arena->walks) / round->walk_bytes, arena->chunks);
    walks_grid = (unsigned)sz_size_divide_round_up(round->walks_count, sz_regex_threads_per_block_simt_k);
    walk_arguments[0] = &launched_engine, walk_arguments[1] = &launched_haystacks, walk_arguments[2] = round;
    round_arguments[0] = round;
    status = sz_regex_launch_cuda_(
        dfa ? (void const *)sz_regex_bound_dfa_simt_kernel_ : (void const *)sz_regex_bound_simt_kernel_, walks_grid,
        walk_arguments, shared_bytes, stream);
    if (status == sz_success_k)
        status = sz_regex_launch_cuda_(dfa ? (void const *)sz_regex_bound_pieces_dfa_simt_kernel_
                                           : (void const *)sz_regex_bound_pieces_simt_kernel_,
                                       walks_grid, walk_arguments, shared_bytes, stream);
    if (status == sz_success_k)
        status = sz_regex_launch_cuda_(
            dfa ? (void const *)sz_regex_affine_dfa_simt_kernel_ : (void const *)sz_regex_affine_simt_kernel_,
            walks_grid, walk_arguments, shared_bytes, stream);
    for (step = 0; status == sz_success_k && step != round->rounds; ++step) {
        round->round = step;
        status = sz_regex_launch_cuda_((void const *)sz_regex_affine_double_simt_kernel_,
                                       sz_regex_grid_cuda_(round->chunks_count), round_arguments, 0, stream);
    }
    if (status == sz_success_k && recording)
        status = sz_regex_launch_cuda_(
            dfa ? (void const *)sz_regex_looks_dfa_simt_kernel_ : (void const *)sz_regex_looks_simt_kernel_, walks_grid,
            walk_arguments, shared_bytes, stream);
    return status;
}

/** Launches one level's lookbehind or lookahead pass over the segment @p round points at: the NFA
 *  flow over the members no table fills and no DFA steps, then a flow per body DFA. */
STRINGZILLA_INLINE sz_status_t sz_regex_passes_cuda_(sz_regex_engine_t const *engine,
                                                     sz_strings_view_t const *haystacks, sz_regex_round_simt_t *round,
                                                     sz_regex_arena_cuda_t const *arena, sz_u32_t level,
                                                     sz_bool_t behind, sz_bool_t recording, sz_stream_t stream) {
    sz_regex_flow_simt_t const flow = behind ? sz_regex_flow_behind_simt_k : sz_regex_flow_ahead_simt_k;
    sz_u32_t negated, members = sz_regex_members_simt_(engine, level, behind, &negated), index;
    sz_u32_t const dfas = members & sz_regex_body_dfas_simt_(engine);
    sz_status_t status = sz_success_k;
    if (members & ~dfas & ~sz_regex_singles_simt_(engine))
        status = sz_regex_flow_cuda_(engine, haystacks, round, arena, flow, level, sz_regex_none_simt_k, recording,
                                     stream);
    for (index = 0; status == sz_success_k && index != engine->lookarounds_count; ++index)
        if (dfas & (sz_u32_t)sz_regex_look_lookaround_k << index)
            status = sz_regex_flow_cuda_(engine, haystacks, round, arena, flow, level, index, recording, stream);
    return status;
}

/** Fills the half of the lookaround ring that segment @p segment takes with every level's passes
 *  up to @p levels, each level over the ones below it and each handed the sets kept at the
 *  segment's edges, and moves the window to start at segment @p first; a segment past the corpus
 *  only moves the window. */
STRINGZILLA_INLINE sz_status_t sz_regex_looks_cuda_(sz_regex_engine_t const *engine, sz_strings_view_t const *haystacks,
                                                    sz_regex_round_simt_t *round, sz_regex_arena_cuda_t const *arena,
                                                    sz_size_t segment, sz_size_t first, sz_u32_t levels,
                                                    sz_stream_t stream) {
    sz_size_t const slots = arena->segment_slots, begin = segment * slots;
    sz_regex_engine_t launched_engine = *engine;
    sz_strings_view_t launched_haystacks = *haystacks;
    void *arguments[3];
    sz_u32_t level;
    sz_status_t status;
    round->looks.begin = first * slots, round->looks.end = round->looks.begin + 2 * slots;
    if (begin >= round->slots_count) return sz_success_k;
    sz_regex_segment_cuda_(round, segment, slots);
    arguments[0] = &launched_engine, arguments[1] = &launched_haystacks, arguments[2] = round;
    // The one-byte lookarounds fill the half outright, which the passes then add to.
    if (sz_regex_singles_simt_(engine))
        status = sz_regex_launch_cuda_((void const *)sz_regex_singles_simt_kernel_,
                                       sz_regex_grid_cuda_(round->segment_end - round->segment_begin), arguments, 0,
                                       stream);
    else
        status = sz_fill_cuda_(round->looks.table + (begin & round->looks.mask),
                               (round->segment_end - round->segment_begin) * sizeof(sz_u32_t), 0, stream);
    for (level = 1; status == sz_success_k && level <= levels; ++level) {
        status = sz_regex_passes_cuda_(engine, haystacks, round, arena, level, sz_true_k, sz_true_k, stream);
        if (status == sz_success_k)
            status = sz_regex_passes_cuda_(engine, haystacks, round, arena, level, sz_false_k, sz_true_k, stream);
    }
    return status;
}

/** Primes the start-finding walk over the segment @p round points at: a DFA by its frontiers,
 *  positions by their flow; either keeps what the segment's first chunk is handed for the
 *  segment before it. */
STRINGZILLA_INLINE sz_status_t sz_regex_prime_segment_cuda_(sz_regex_engine_t const *engine,
                                                            sz_strings_view_t const *haystacks,
                                                            sz_regex_round_simt_t *round,
                                                            sz_regex_arena_cuda_t const *arena, sz_stream_t stream) {
    sz_regex_engine_t launched_engine = *engine;
    sz_strings_view_t launched_haystacks = *haystacks;
    sz_size_t const shared_bytes = sz_regex_staged_bytes_simt_k + (sz_size_t)round->staged_cells * sizeof(sz_u32_t);
    unsigned const grid = sz_regex_grid_cuda_(round->chunks_count);
    void *walk_arguments[3], *round_arguments[1];
    sz_status_t status;
    sz_size_t step;
    if (!engine->reverse_rows)
        return sz_regex_flow_cuda_(engine, haystacks, round, arena, sz_regex_flow_positions_simt_k, 0,
                                   sz_regex_none_simt_k, sz_false_k, stream);
    walk_arguments[0] = &launched_engine, walk_arguments[1] = &launched_haystacks, walk_arguments[2] = round;
    round_arguments[0] = round;
    status = sz_regex_launch_cuda_((void const *)sz_regex_prime_simt_kernel_, grid, walk_arguments, shared_bytes,
                                   stream);
    if (status == sz_success_k)
        status = sz_regex_launch_cuda_((void const *)sz_regex_prime_pieces_simt_kernel_, grid, walk_arguments,
                                       shared_bytes, stream);
    if (status == sz_success_k)
        status = sz_regex_launch_cuda_((void const *)sz_regex_compose_simt_kernel_, grid, walk_arguments, shared_bytes,
                                       stream);
    for (step = 0; status == sz_success_k && step != round->rounds; ++step) {
        round->round = step;
        status = sz_regex_launch_cuda_((void const *)sz_regex_compose_double_simt_kernel_, grid, round_arguments, 0,
                                       stream);
    }
    return status;
}

/** Reads the round's counters back into @p counters, joining @p stream, and fails the call
 *  when a walk refused. */
STRINGZILLA_INLINE sz_status_t sz_regex_counters_cuda_(sz_regex_round_simt_t const *round, sz_u32_t *counters,
                                                       sz_stream_t stream) {
    sz_status_t status = sz_copy_cuda_(counters, round->counters,
                                       (sz_regex_refused_counter_simt_k + 1) * sizeof(sz_u32_t), stream);
    if (status == sz_success_k) status = sz_stream_synchronize_cuda_(stream);
    if (status == sz_success_k && counters[sz_regex_refused_counter_simt_k]) status = sz_unexpected_dimensions_k;
    return status;
}

/**
 *  @brief Reads back whether the end walks of the segment @p round points at paused at the end of
 *      the lookaround table's window or refused, and walks the paused ones on to their ends.
 *
 *  While walks are paused, it groups them by state, slides the window a segment on, and walks
 *  every group on as a link walks; then the groups settle by pointer doubling, each paused link
 *  takes its group's match, and the window returns to the segment and the next, for the walks that
 *  read it later.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_slides_cuda_(sz_regex_engine_t const *engine,
                                                     sz_strings_view_t const *haystacks, sz_regex_round_simt_t *round,
                                                     sz_regex_arena_cuda_t const *arena, sz_u32_t levels,
                                                     sz_stream_t stream) {
    sz_regex_engine_t launched_engine = *engine;
    sz_strings_view_t launched_haystacks = *haystacks;
    sz_size_t const segment = round->segment, slots = arena->segment_slots, rounds = round->rounds;
    sz_size_t const shared_bytes = sz_regex_staged_bytes_simt_k + (sz_size_t)round->staged_cells * sizeof(sz_u32_t);
    sz_bool_t const folded = (sz_bool_t)(engine->case_sensitivity != sz_cased_k);
    unsigned const grid = sz_regex_grid_cuda_(slots);
    void *walk_arguments[3], *group_arguments[2], *round_arguments[1];
    sz_u32_t counters[sz_regex_refused_counter_simt_k + 1];
    sz_size_t ends = 0, step;
    sz_status_t status = sz_regex_counters_cuda_(round, counters, stream);
    walk_arguments[0] = &launched_engine, walk_arguments[1] = &launched_haystacks, walk_arguments[2] = round;
    group_arguments[0] = &launched_engine, group_arguments[1] = round, round_arguments[0] = round;
    while (status == sz_success_k && counters[sz_regex_paused_counter_simt_k]) {
        status = sz_fill_cuda_(round->leaders, engine->forward_states_count * sizeof(sz_u32_t), 0xFF, stream);
        if (status == sz_success_k)
            status = sz_regex_launch_cuda_((void const *)sz_regex_group_simt_kernel_, grid, group_arguments, 0, stream);
        if (status == sz_success_k)
            status = sz_regex_launch_cuda_((void const *)sz_regex_point_simt_kernel_, grid, group_arguments, 0, stream);
        if (status == sz_success_k)
            status = sz_fill_cuda_(round->counters + sz_regex_handed_counter_simt_k, 2 * sizeof(sz_u32_t), 0, stream);
        // The window moves a segment on, the half it leaves taking the segment past the other.
        if (status == sz_success_k)
            status = sz_regex_looks_cuda_(engine, haystacks, round, arena, segment + ends + 2, segment + ends + 1,
                                          levels, stream);
        sz_regex_segment_cuda_(round, segment, slots);
        round->groups_begin = counters[sz_regex_groups_counter_simt_k];
        if (status == sz_success_k)
            status = sz_regex_launch_cuda_((void const *)sz_regex_resume_simt_kernel_, grid, walk_arguments,
                                           shared_bytes, stream);
        if (status == sz_success_k)
            status = sz_regex_launch_cuda_(folded ? (void const *)sz_regex_continue_folded_cuda_kernel_
                                                  : (void const *)sz_regex_continue_cuda_kernel_,
                                           grid, walk_arguments, shared_bytes, stream);
        if (status == sz_success_k) status = sz_regex_counters_cuda_(round, counters, stream);
        ++ends;
    }
    if (status != sz_success_k || !ends) return status;
    round->rounds = sz_regex_rounds_count_cuda_(ends);
    for (step = 0; status == sz_success_k && step != round->rounds; ++step) {
        round->round = step;
        status = sz_regex_launch_cuda_((void const *)sz_regex_groups_double_simt_kernel_, grid, round_arguments, 0,
                                       stream);
    }
    round->rounds = rounds;
    if (status == sz_success_k)
        status = sz_regex_launch_cuda_((void const *)sz_regex_take_simt_kernel_, grid, round_arguments, 0, stream);
    if (status == sz_success_k)
        status = sz_regex_looks_cuda_(engine, haystacks, round, arena, segment, segment, levels, stream);
    if (status == sz_success_k)
        status = sz_regex_looks_cuda_(engine, haystacks, round, arena, segment + 1, segment, levels, stream);
    sz_regex_segment_cuda_(round, segment, slots);
    return status;
}

/**
 *  @brief Launches the whole call on @p stream: the passes that hand lookaround sets across
 *      segments, the priming segments right to left, then the walking ones left to right, each
 *      storing what @p round asks, the outputs its caller set and nulled.
 *
 *  A start-finding walk primes each chunk from the one right of it, so ahead of the walk the
 *  segments past the first are primed right to left once, leaving each segment's edge for the one
 *  before it. Lookaround passes run the same way per level, lookbehinds left to right, ahead of
 *  everything that reads their table, whose ring each later sweep fills a segment at a time. End
 *  walks without links take one candidate per thread, listed first.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_rounds_cuda_(sz_regex_engine_t const *engine,
                                                     sz_strings_view_t const *haystacks, sz_regex_round_simt_t *round,
                                                     sz_stream_t stream) {
    sz_regex_arena_cuda_t const arena = sz_regex_arena_of_cuda_(engine);
    sz_ptr_t const block = (sz_ptr_t)engine->scratch;
    sz_size_t const slots_count = sz_regex_slots_count_cuda_(haystacks), slots = arena.segment_slots;
    sz_size_t const segments = sz_size_divide_round_up(slots_count, slots);
    sz_bool_t const folded = (sz_bool_t)(engine->case_sensitivity != sz_cased_k);
    sz_u32_t const levels = sz_regex_levels_simt_(engine);
    sz_size_t const walks_bytes = arena.total - arena.walks;
    sz_size_t const pike_walks = sz_min_of_two(walks_bytes / arena.pike_bytes, arena.chunks);
    sz_size_t const positions_walks = arena.positions_bytes
                                          ? sz_min_of_two(walks_bytes / arena.positions_bytes, arena.chunks)
                                          : arena.chunks;
    sz_size_t const groups_capacity = (slots / sz_regex_checkpoint_slots_simt_k + 1) / 2;
    sz_regex_engine_t launched_engine = *engine;
    sz_strings_view_t launched_haystacks = *haystacks;
    sz_size_t segment, step, shared_bytes, *chunk_offsets, *chunk_kept, *chunk_sums, *chunk_covers;
    void *round_arguments[1], *walk_arguments[3], *chain_arguments[2];
    sz_u32_t level, lookaround, negated, counters[sz_regex_refused_counter_simt_k + 1];
    sz_status_t status = sz_success_k;
    if (segments > arena.chunks) return sz_unexpected_dimensions_k;

    round->reverse_images = (sz_u32_t const *)block;
    round->forward_images = (sz_u32_t const *)(block + arena.images_bytes);
    round->entries = (sz_u32_t *)(block + arena.entries), round->edges = (sz_u32_t *)(block + arena.edges);
    round->carry = (sz_regex_carry_simt_t *)(block + arena.carry);
    round->counters = (sz_u32_t *)(block + arena.counters), round->flags = (sz_u32_t *)(block + arena.flags);
    round->leaders = (sz_u32_t *)(block + arena.frontiers);
    round->scan_chain = (sz_size_t *)(block + arena.scan_chain);
    round->scan_states = (sz_u64_t *)(block + arena.scan_states), round->scans = 0;
    round->chunk_offsets = (sz_size_t *)(block + arena.chunk_offsets);
    round->chunk_kept = (sz_size_t *)(block + arena.chunk_kept);
    round->chunk_sums = (sz_size_t *)(block + arena.chunk_sums);
    round->chunk_covers = (sz_size_t *)(block + arena.chunk_covers);
    round->chunk_haystacks = (sz_size_t *)(block + arena.chunk_haystacks);
    round->frontiers = (sz_regex_frontier_simt_t *)(block + arena.frontiers);
    round->compositions = (sz_regex_composition_simt_t *)(block + arena.compositions);
    round->compositions_next = (sz_regex_composition_simt_t *)(block + arena.compositions_next);
    round->starts = (sz_u64_t *)(block + arena.starts), round->handoffs = (sz_u64_t *)(block + arena.handoffs);
    round->undecided = (sz_regex_undecided_simt_t *)(block + arena.undecided);
    round->affines = (sz_regex_affine_simt_t *)(block + arena.affines);
    round->affines_next = (sz_regex_affine_simt_t *)(block + arena.affines_next);
    round->looks.table = engine->lookarounds_count ? (sz_u32_t *)(block + arena.looks) : STRINGZILLA_NULL;
    round->looks.refused = round->counters + sz_regex_refused_counter_simt_k, round->looks.mask = 2 * slots - 1;
    round->looks.begin = 0, round->looks.end = 0, round->set_words = arena.set_words,
    round->body = sz_regex_none_simt_k;
    round->bodies_begin = sz_regex_bodies_begin_simt_(engine);
    round->bodies_count = engine->nfa_states_count - round->bodies_begin;
    round->checkpoints = (sz_regex_checkpoint_simt_t *)(block + arena.checkpoints);
    round->groups = (sz_regex_result_simt_t *)round->checkpoints, round->groups_next = round->groups + groups_capacity;
    round->groups_begin = 0, round->groups_capacity = (sz_u32_t)groups_capacity;
    round->slots = (sz_u32_t *)(block + arena.slots), round->links_list = (sz_u32_t *)(block + arena.links_list);
    round->nexts = (sz_u32_t *)(block + arena.nexts), round->exits = (sz_u32_t *)(block + arena.exits);
    round->exits_next = (sz_u32_t *)(block + arena.exits_next), round->keep = (sz_u32_t *)(block + arena.keep);
    // Kept flags reuse the links' list, and drifts the exits, both dead once the chain is doubled.
    round->kept = round->links_list, round->drifts = (sz_size_t *)round->exits;
    round->results = (sz_regex_result_simt_t *)(block + arena.results);
    round->links = (sz_regex_result_simt_t *)(block + arena.links);
    round->links_next = (sz_regex_result_simt_t *)(block + arena.links_next);
    round->walks = (sz_u8_t *)(block + arena.walks), round->pike_slots = arena.pike_slots;
    round->slots_count = slots_count, round->chunk_shift = sz_size_log2i_nonzero(sz_regex_chunk_slots_simt_k);
    round->rounds = sz_regex_rounds_count_cuda_(
        sz_size_divide_round_up(sz_min_of_two(slots_count, slots), sz_regex_chunk_slots_simt_k));
    round->round = 0, round->storing = 0;
    round->forward_cells = engine->forward_states_count *
                           (sz_u32_t)sz_regex_stride_(engine->classes_count, engine->lookarounds_count);
    round->reverse_cells = engine->reverse_states_count *
                           (sz_u32_t)sz_regex_stride_(engine->classes_count, engine->lookarounds_count);
    round->staged_cells = sz_max_of_two(round->forward_cells, round->reverse_cells);
    for (lookaround = 0; lookaround != engine->lookarounds_count; ++lookaround)
        round->staged_cells = sz_max_of_two(
            round->staged_cells,
            sz_regex_body_reach_simt_(engine, lookaround, STRINGZILLA_NULL) * (engine->classes_count + 1));
    round->staged_cells = sz_min_of_two(round->staged_cells, (sz_u32_t)sz_regex_staged_cells_simt_k);
    shared_bytes = sz_regex_staged_bytes_simt_k + (sz_size_t)round->staged_cells * sizeof(sz_u32_t);
    chunk_offsets = round->chunk_offsets, chunk_kept = round->chunk_kept;
    chunk_sums = round->chunk_sums, chunk_covers = round->chunk_covers;
    walk_arguments[0] = &launched_engine, walk_arguments[1] = &launched_haystacks, walk_arguments[2] = round;
    chain_arguments[0] = &launched_haystacks, chain_arguments[1] = round;
    round_arguments[0] = round;

    // The ticket counter and both buffers of tile states start clear.
    status = sz_fill_cuda_(round->scan_chain, arena.chunk_offsets - arena.scan_chain, 0, stream);

    // Each level's passes hand sets across segments over lower levels' tables, so they sweep first.
    for (level = 1; status == sz_success_k && segments > 1 && level <= levels; ++level) {
        if (sz_regex_members_simt_(engine, level, sz_true_k, &negated) & ~sz_regex_singles_simt_(engine))
            for (segment = 0; status == sz_success_k && segment != segments; ++segment) {
                status = sz_regex_looks_cuda_(engine, haystacks, round, &arena, segment, segment, level - 1, stream);
                sz_regex_segment_cuda_(round, segment, slots);
                if (status == sz_success_k)
                    status = sz_regex_passes_cuda_(engine, haystacks, round, &arena, level, sz_true_k, sz_false_k,
                                                   stream);
            }
        if (sz_regex_members_simt_(engine, level, sz_false_k, &negated) & ~sz_regex_singles_simt_(engine))
            for (segment = segments; status == sz_success_k && segment-- != 0;) {
                status = sz_regex_looks_cuda_(engine, haystacks, round, &arena, segment, segment, level - 1, stream);
                sz_regex_segment_cuda_(round, segment, slots);
                if (status == sz_success_k)
                    status = sz_regex_passes_cuda_(engine, haystacks, round, &arena, level, sz_false_k, sz_false_k,
                                                   stream);
            }
    }

    // Priming reads each segment's right neighbour, so the segments past the first prime right to
    // left first, each filling its own half of the table beside the half its neighbour filled;
    // literal prefixes mark candidates without priming.
    for (segment = segments; status == sz_success_k && !engine->prefixes && segment-- > 1;) {
        if (levels) status = sz_regex_looks_cuda_(engine, haystacks, round, &arena, segment, segment, levels, stream);
        sz_regex_segment_cuda_(round, segment, slots);
        if (status == sz_success_k) status = sz_regex_prime_segment_cuda_(engine, haystacks, round, &arena, stream);
    }
    for (segment = 0; status == sz_success_k && segment != segments; ++segment) {
        sz_bool_t past;
        unsigned grid;
        // The walks read the table up to the next segment, whose half alone is new past the first,
        // unless no priming filled it.
        if (levels && engine->prefixes && !segment)
            status = sz_regex_looks_cuda_(engine, haystacks, round, &arena, 1, 0, levels, stream);
        if (status == sz_success_k && levels)
            status = sz_regex_looks_cuda_(engine, haystacks, round, &arena, segment ? segment + 1 : 0, segment, levels,
                                          stream);
        sz_regex_segment_cuda_(round, segment, slots);
        grid = sz_regex_grid_cuda_(round->chunks_count);
        past = (sz_bool_t)(levels && sz_regex_runs_past_cuda_(haystacks, round));
        if (status == sz_success_k && segment == 0)
            status = sz_fill_cuda_(round->carry, sizeof(sz_regex_carry_simt_t), 0, stream);
        if (status == sz_success_k && !engine->prefixes)
            status = sz_regex_prime_segment_cuda_(engine, haystacks, round, &arena, stream);

        // The links' counters and doubling flags, the two things a segment reads before writing.
        if (status == sz_success_k)
            status = sz_fill_cuda_(round->counters, arena.scan_chain - arena.counters, 0, stream);
        if (status == sz_success_k && !engine->reverse_rows) {
            round->walks_count = positions_walks, round->walk_bytes = arena.positions_bytes;
            status = sz_regex_launch_cuda_(
                folded ? (void const *)sz_regex_mark_positions_folded_simt_kernel_
                       : (void const *)sz_regex_mark_positions_simt_kernel_,
                (unsigned)sz_size_divide_round_up(positions_walks, sz_regex_threads_per_block_simt_k), walk_arguments,
                0, stream);
        }
        else if (status == sz_success_k && engine->prefixes)
            status = sz_regex_launch_cuda_((void const *)sz_regex_mark_prefixes_simt_kernel_, grid, walk_arguments, 0,
                                           stream);
        else if (status == sz_success_k)
            status = sz_regex_launch_cuda_(
                folded ? (void const *)sz_regex_mark_folded_simt_kernel_ : (void const *)sz_regex_mark_simt_kernel_,
                grid, walk_arguments, shared_bytes + (folded ? 0 : sz_regex_staged_text_bytes_simt_k), stream);
        if (status == sz_success_k) status = sz_regex_scan_cuda_(round, chunk_offsets, round->chunks_count + 1, stream);
        if (status == sz_success_k && !engine->forward_rows) {
            round->walks_count = pike_walks, round->walk_bytes = arena.pike_bytes;
            status = sz_regex_launch_cuda_((void const *)sz_regex_list_simt_kernel_, grid, round_arguments, 0, stream);
            if (status == sz_success_k)
                status = sz_regex_launch_cuda_(
                    (void const *)sz_regex_ends_pike_simt_kernel_,
                    (unsigned)sz_size_divide_round_up(pike_walks, sz_regex_threads_per_block_simt_k), walk_arguments, 0,
                    stream);
        }
        else if (status == sz_success_k) {
            status = sz_regex_launch_cuda_((void const *)sz_regex_ends_simt_kernel_, grid, walk_arguments, shared_bytes,
                                           stream);
            if (status == sz_success_k)
                status = sz_regex_launch_cuda_((void const *)sz_regex_links_simt_kernel_, grid, walk_arguments,
                                               shared_bytes, stream);
            if (status == sz_success_k)
                status = sz_regex_launch_cuda_(folded ? (void const *)sz_regex_continue_folded_cuda_kernel_
                                                      : (void const *)sz_regex_continue_cuda_kernel_,
                                               grid, walk_arguments, shared_bytes, stream);
        }
        // Only a haystack running past the table can leave walks paused or refused there,
        // so only then is one read.
        if (status == sz_success_k && past)
            status = sz_regex_slides_cuda_(engine, haystacks, round, &arena, levels, stream);
        for (step = 0; status == sz_success_k && engine->forward_rows && step != round->rounds; ++step) {
            round->round = step;
            status = sz_regex_launch_cuda_((void const *)sz_regex_links_double_simt_kernel_, grid, round_arguments, 0,
                                           stream);
        }
        if (status == sz_success_k && engine->forward_rows && folded)
            status = sz_regex_launch_cuda_((void const *)sz_regex_retry_folded_simt_kernel_, sz_regex_grid_cuda_(slots),
                                           walk_arguments, shared_bytes, stream);
        if (status == sz_success_k)
            status = sz_regex_launch_cuda_((void const *)sz_regex_chain_simt_kernel_, grid, chain_arguments, 0, stream);
        for (step = 0; status == sz_success_k && step != round->rounds; ++step) {
            round->round = step;
            status = sz_regex_launch_cuda_((void const *)sz_regex_double_simt_kernel_, sz_regex_grid_cuda_(slots),
                                           round_arguments, 0, stream);
        }
        round->storing = 0;
        if (status == sz_success_k)
            status = sz_regex_launch_cuda_((void const *)sz_regex_sweep_simt_kernel_, grid, chain_arguments, 0, stream);
        if (status == sz_success_k) status = sz_regex_scan_cuda_(round, chunk_kept, round->chunks_count + 1, stream);

        // A rewrite or an extraction runs on the walks between the counting sweep and the storing
        // one, which moves the carry's last match on.
        round->walks_count = pike_walks, round->walk_bytes = arena.pike_bytes;
        if (status == sz_success_k && round->offsets) {
            unsigned const walks_grid = (unsigned)sz_size_divide_round_up(pike_walks,
                                                                          sz_regex_threads_per_block_simt_k);
            status = sz_regex_launch_cuda_((void const *)sz_regex_expand_simt_kernel_, walks_grid, walk_arguments,
                                           sz_regex_staged_bytes_simt_k, stream);
            if (status == sz_success_k)
                status = sz_fill_cuda_(chunk_covers, (round->chunks_count + 1) * sizeof(sz_size_t), 0, stream);
            if (status == sz_success_k)
                status = sz_regex_launch_cuda_((void const *)sz_regex_drift_simt_kernel_, grid, round_arguments, 0,
                                               stream);
            if (status == sz_success_k)
                status = sz_regex_scan_cuda_(round, chunk_sums, round->chunks_count + 1, stream);
            if (status == sz_success_k)
                status = sz_regex_launch_cuda_((void const *)sz_regex_rewrite_simt_kernel_, walks_grid, walk_arguments,
                                               sz_regex_staged_bytes_simt_k, stream);
        }
        if (status == sz_success_k && round->rows_offsets) {
            unsigned const walks_grid = (unsigned)sz_size_divide_round_up(pike_walks,
                                                                          sz_regex_threads_per_block_simt_k);
            status = sz_regex_scan_cuda_(round, chunk_sums, round->chunks_count + 1, stream);
            if (status == sz_success_k)
                status = sz_regex_launch_cuda_((void const *)sz_regex_firsts_simt_kernel_, walks_grid, walk_arguments,
                                               sz_regex_staged_bytes_simt_k, stream);
            if (status == sz_success_k)
                status = sz_regex_scan_cuda_(round, chunk_covers, round->chunks_count + 1, stream);
            if (status == sz_success_k)
                status = sz_regex_launch_cuda_((void const *)sz_regex_rows_simt_kernel_, walks_grid, walk_arguments,
                                               sz_regex_staged_bytes_simt_k, stream);
        }
        // Retries and group walks read the table too, so where walks may run past it
        // they are read back once more.
        if (status == sz_success_k && past &&
            ((folded && engine->forward_rows) || round->offsets || round->rows_offsets))
            status = sz_regex_counters_cuda_(round, counters, stream);
        round->storing = 1;
        if (status == sz_success_k)
            status = sz_regex_launch_cuda_((void const *)sz_regex_sweep_simt_kernel_, grid, chain_arguments, 0, stream);
        if (status == sz_success_k) {
            dim3 one;
            one.x = 1, one.y = 1, one.z = 1;
            status = sz_launch_cuda_((void const *)sz_regex_carry_simt_kernel_, one, one, chain_arguments, 0, stream);
        }
    }
    return status;
}

/** Whether every argument the device verbs read or write is one a kernel can address. */
STRINGZILLA_INLINE sz_bool_t sz_regex_resident_cuda_(sz_regex_engine_t const *engine,
                                                     sz_strings_view_t const *haystacks) {
    return (sz_bool_t)(sz_memory_accessible_cuda_(engine->memory) && sz_memory_accessible_cuda_(engine->scratch) &&
                       sz_strings_accessible_cuda_(haystacks));
}

#pragma endregion Host Plumbing

#pragma region Construction

/** Compiles @p patterns on the caller's current device, as @ref sz_regex_engine_init_cuda does. */
STRINGZILLA_INLINE sz_status_t sz_regex_engine_init_scoped_cuda_(sz_regex_engine_t *engine,
                                                                 sz_strings_view_t const *patterns,
                                                                 sz_case_sensitivity_t case_sensitivity,
                                                                 sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                                 sz_stream_t stream) {
    sz_allocator_t unified;
    sz_status_t status;
    if (!sz_device_multiprocessors_cuda_()) return sz_missing_gpu_k;
    if (allocator) unified = *allocator;
    else sz_allocator_init_unified_cuda_(&unified);
    status = sz_regex_engine_compile_(engine, patterns, case_sensitivity, malformed_index, sz_cap_cuda_k, &unified,
                                      sz_regex_reverse_budget_k, sz_regex_forward_budget_k, stream);
    if (status != sz_success_k) return status;
    // The host builder writes the block in place, so a device-only allocation cannot serve
    // as the vocabulary.
    if (!sz_memory_accessible_cuda_(engine->memory)) {
        sz_regex_engine_free_(engine, stream);
        return sz_device_memory_mismatch_k;
    }
    status = sz_regex_arena_reserve_cuda_(engine, stream);
    if (status != sz_success_k) sz_regex_engine_free_(engine, stream);
    return status;
}

#pragma endregion Construction

#pragma region Verbs

/** Clears a round's outputs, which each verb then sets for itself. */
STRINGZILLA_INLINE void sz_regex_round_init_cuda_(sz_regex_round_simt_t *round) {
    memset(round, 0, sizeof(*round));
    round->pattern_index = STRINGZILLA_SIZE_MAX;
}

STRINGZILLA_INLINE sz_status_t sz_regex_counts_scoped_cuda_(sz_regex_engine_t *engine,
                                                            sz_strings_view_t const *haystacks, sz_size_t *counts,
                                                            sz_size_t counts_stride, sz_stream_t stream) {
    sz_regex_round_simt_t round;
    sz_size_t haystacks_count = haystacks->count, *edges;
    unsigned grid;
    void *arguments[4];
    sz_status_t status;
    if (!counts_stride) return sz_unexpected_dimensions_k;
    if (!engine->memory) return sz_missing_kernel_k;
    if (!haystacks->count) return sz_success_k;
    if (haystacks->layout == sz_strings_accessor_k || !sz_regex_resident_cuda_(engine, haystacks) ||
        !sz_memory_accessible_cuda_(counts))
        return sz_device_memory_mismatch_k;
    sz_regex_round_init_cuda_(&round);
    round.counts = counts, round.counts_stride = counts_stride;
    status = sz_regex_rounds_cuda_(engine, haystacks, &round, stream);
    if (status != sz_success_k) return status;
    // The sweeps left running totals, which two launches of one grid difference,
    // the results being dead by now.
    grid = sz_regex_grid_cuda_(haystacks_count), edges = (sz_size_t *)round.results;
    arguments[0] = &counts, arguments[1] = &counts_stride, arguments[2] = &haystacks_count, arguments[3] = &edges;
    status = sz_regex_launch_cuda_((void const *)sz_regex_counts_edges_simt_kernel_, grid, arguments, 0, stream);
    if (status != sz_success_k) return status;
    return sz_regex_launch_cuda_((void const *)sz_regex_counts_simt_kernel_, grid, arguments, 0, stream);
}

STRINGZILLA_INLINE sz_status_t sz_regex_find_scoped_cuda_(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                          sz_match_t *matches, sz_size_t matches_capacity,
                                                          sz_size_t *matches_offsets, sz_stream_t stream) {
    sz_regex_round_simt_t round;
    if (!engine->memory) return sz_missing_kernel_k;
    if (!haystacks->count) return sz_success_k;
    if (haystacks->layout == sz_strings_accessor_k || !sz_regex_resident_cuda_(engine, haystacks) ||
        !sz_memory_accessible_cuda_(matches_offsets) || (matches_capacity && !sz_memory_accessible_cuda_(matches)))
        return sz_device_memory_mismatch_k;
    sz_regex_round_init_cuda_(&round);
    round.matches = matches, round.matches_capacity = matches_capacity, round.matches_offsets = matches_offsets;
    return sz_regex_rounds_cuda_(engine, haystacks, &round, stream);
}

STRINGZILLA_INLINE sz_status_t sz_regex_replace_scoped_cuda_(sz_regex_engine_t *engine,
                                                             sz_strings_view_t const *haystacks,
                                                             sz_strings_view_t const *replacements, sz_ptr_t target,
                                                             sz_size_t target_capacity, sz_size_t *offsets,
                                                             sz_stream_t stream) {
    sz_regex_round_simt_t round;
    if (!engine->memory) return sz_missing_kernel_k;
    if (replacements->count != engine->patterns_count) return sz_unexpected_dimensions_k;
    if (!haystacks->count) return sz_success_k;
    if (haystacks->layout == sz_strings_accessor_k || replacements->layout == sz_strings_accessor_k ||
        !sz_regex_resident_cuda_(engine, haystacks) || !sz_strings_accessible_cuda_(replacements) ||
        !sz_memory_accessible_cuda_(offsets) || (target_capacity && !sz_memory_accessible_cuda_(target)))
        return sz_device_memory_mismatch_k;
    sz_regex_round_init_cuda_(&round);
    round.replacements = *replacements, round.target = target, round.target_capacity = target_capacity;
    round.offsets = offsets;
    return sz_regex_rounds_cuda_(engine, haystacks, &round, stream);
}

STRINGZILLA_INLINE sz_status_t sz_regex_extract_scoped_cuda_(sz_regex_engine_t *engine,
                                                             sz_strings_view_t const *haystacks,
                                                             sz_size_t pattern_index, sz_match_t *groups,
                                                             sz_size_t groups_capacity, sz_size_t *groups_offsets,
                                                             sz_stream_t stream) {
    sz_regex_round_simt_t round;
    if (!engine->memory) return sz_missing_kernel_k;
    if (pattern_index >= engine->patterns_count) return sz_unexpected_dimensions_k;
    if (!haystacks->count) return sz_success_k;
    if (haystacks->layout == sz_strings_accessor_k || !sz_regex_resident_cuda_(engine, haystacks) ||
        !sz_memory_accessible_cuda_(groups_offsets) || (groups_capacity && !sz_memory_accessible_cuda_(groups)))
        return sz_device_memory_mismatch_k;
    sz_regex_round_init_cuda_(&round);
    round.pattern_index = pattern_index, round.rows = groups, round.rows_capacity = groups_capacity;
    round.rows_offsets = groups_offsets;
    return sz_regex_rounds_cuda_(engine, haystacks, &round, stream);
}

#pragma endregion Verbs

/*  Each export makes the stream's device current around its helper, and restores the caller's. */
#if STRINGZILLA_TARGET_CUDA

STRINGZILLA_API sz_status_t sz_regex_engine_init_cuda(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                      sz_case_sensitivity_t case_sensitivity,
                                                      sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                      sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_regex_engine_init_scoped_cuda_(engine, patterns, case_sensitivity, malformed_index, allocator, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_regex_counts_cuda(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                 sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_regex_counts_scoped_cuda_(engine, haystacks, counts, counts_stride, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_regex_find_cuda(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                               sz_match_t *matches, sz_size_t matches_capacity,
                                               sz_size_t *matches_offsets, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_regex_find_scoped_cuda_(engine, haystacks, matches, matches_capacity, matches_offsets, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_regex_replace_cuda(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                  sz_strings_view_t const *replacements, sz_ptr_t target,
                                                  sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_regex_replace_scoped_cuda_(engine, haystacks, replacements, target, target_capacity, offsets, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_regex_extract_cuda(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                  sz_size_t pattern_index, sz_match_t *groups,
                                                  sz_size_t groups_capacity, sz_size_t *groups_offsets,
                                                  sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_regex_extract_scoped_cuda_(engine, haystacks, pattern_index, groups, groups_capacity, groups_offsets,
                                           stream);
    sz_device_leave_cuda_(caller);
    return status;
}

#endif // STRINGZILLA_TARGET_CUDA

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_REGEX_CUDA_CUH_
