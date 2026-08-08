/**
 *  @brief  CUDA backend for multi-pattern exact and case-folded substring search.
 *  @file   include/stringzillas/find_many/cuda.cuh
 *  @author Ash Vardanian
 *  @sa     include/stringzillas/find_many/serial.hpp
 *
 *  `try_build` uploads a host-built `aho_corasick_view`, so this backend consumes the published contract
 *  rather than the builder's internals.
 *
 *  @section Thread Per Chunk, Haystack-Aware
 *
 *  One thread owns one contiguous slice of the input tape, sized from the tape's total length rather than
 *  from the haystack count, so occupancy does not depend on how the tape is split. Chunks never cross a
 *  haystack boundary: each haystack is divided independently against one global `chunk_bytes` target, so no
 *  needle can straddle the seam between two documents.
 *
 *  A thread starts its walk `max_match_bytes - 1` bytes before its chunk, clamped to its own haystack's
 *  start. Aho-Corasick is self-synchronizing, so that warm-up makes chunking exact rather than approximate:
 *  the automaton reaches the same state at the chunk boundary wherever the walk began, as long as it began
 *  early enough. Every walk starts at `view.root`, and a match's start offset is checked against its
 *  haystack's start as a second, independent guard.
 *
 *  @section Shared Memory Hot Tier
 *
 *  A warp runs in lockstep, so it pays the maximum failure-chase depth across its lanes, not the mean.
 *  Staging a prefix of the already frequency-ordered hot tier into shared memory keeps the common
 *  transitions divergence-free and immune to eviction by cold-tier traffic. That prefix is sized against
 *  `target_blocks_per_multiprocessor_` rather than the opt-in per-block ceiling, which would leave one
 *  block resident per SM; states above it but below `hot_count` still resolve through device memory.
 */
#ifndef STRINGZILLAS_FIND_MANY_CUDA_CUH_
#define STRINGZILLAS_FIND_MANY_CUDA_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include "stringzillas/types.cuh"            // `unified_alloc_t`, `cuda_status_t`
#include "stringzillas/find_many/serial.hpp" // `aho_corasick_view`, the host-built contract

namespace ashvardanian {
namespace stringzillas {

// Per-symbol: a using-directive re-exports our `memcpy` and nvcc then finds the call ambiguous.
using ashvardanian::stringzilla::byte_t;
using ashvardanian::stringzilla::size_t;
using ashvardanian::stringzilla::span;
using ashvardanian::stringzilla::status_t;

#pragma region Device Types

/**
 *  @brief One match reported against the flat input tape: which needle, how many bytes, where it ends, and
 *         which haystack it belongs to.
 */
template <typename state_id_type_>
struct find_many_cuda_match {
    using state_id_t = state_id_type_;

    size_t tape_end_offset {};
    size_t haystack_index {};
    state_id_t needle_index {};
    state_id_t match_bytes {};
};

#pragma endregion Device Types

#pragma region Device Kernels

/** @brief Block size every `find_many_cuda` kernel launches with; occupancy is shared-memory-bound, not
 *         thread-count-bound, so a modest fixed block keeps the launch geometry simple. */
static constexpr unsigned find_many_threads_per_block_k = 256;

/**
 *  @brief Selects a chunk walk's pass: `counting_k` only tallies matches so the caller can size and scan the
 *         output, `scattering_k` writes each match at its chunk's precomputed offset.
 *
 *  A `bool`-backed scoped enum consumed with `if constexpr`, matching `tile_march_t` in `stringzillas/types.cuh`.
 */
enum class find_many_pass_t : bool { counting_k = false, scattering_k = true };

/**
 *  @brief Cooperatively fills @p shared_hot_rows from the head of the hot tier, once per block. Frequency
 *         ordering means this prefix is already the hottest slice - no selection logic needed.
 */
template <typename state_id_type_>
SZ_DEVICE_INLINE void find_many_stage_hot_rows_(aho_corasick_view<state_id_type_> const &view,
                                                span<state_id_type_> shared_hot_rows) noexcept {
    for (size_t cell = threadIdx.x; cell < shared_hot_rows.size(); cell += blockDim.x)
        shared_hot_rows[cell] = view.hot_rows[cell];
    __syncthreads();
}

/**
 *  @brief One byte's transition, staged-shared-memory-first: the staged prefix resolves branch-free from
 *         shared memory, and everything else - hot tier beyond the prefix, and the whole cold tier - defers to
 *         @ref aho_corasick_step, the single transition definition every backend shares. Neither the double-
 *         array probe nor the failure retry is reimplemented here.
 *
 *  A single cold lane still makes the whole warp pay that lane's failure-chase depth, since
 *  @ref aho_corasick_step 's retry loop runs to the slowest lane's completion before the warp reconverges.
 *  That is the cost model the shared-memory staging above exists to shrink.
 */
template <typename state_id_type_>
SZ_DEVICE_INLINE state_id_type_ find_many_step_device_(aho_corasick_view<state_id_type_> const &view,
                                                       span<state_id_type_ const> shared_hot_rows, state_id_type_ state,
                                                       u8_t byte) noexcept {
    // Row-major, the same shape as `aho_corasick_view::hot_row`, and computed in 32 bits: the staged prefix
    // is a few hundred rows of shared memory, so a 64-bit cell index would only widen the address math.
    u32_t const shared_rows_count = (u32_t)(shared_hot_rows.size() / find_many_alphabet_size_k);
    if ((u32_t)state < shared_rows_count) return shared_hot_rows[(u32_t)state * find_many_alphabet_size_k + byte];
    return aho_corasick_step(view, state, byte);
}

/**
 *  @brief Finds, via binary search, which haystack owns global chunk @p chunk_index, given the exclusive
 *         prefix sum of chunk counts per haystack (`haystack_chunk_offsets[haystack_count]` is the grand
 *         total chunk count, mirroring how `outputs_offsets` bounds `outputs_counts`).
 */
SZ_DEVICE_INLINE size_t find_many_resolve_haystack_(span<size_t const> haystack_chunk_offsets,
                                                    size_t chunk_index) noexcept {
    size_t low = 0, high = haystack_chunk_offsets.size() - 1;
    while (low + 1 < high) {
        size_t const mid = low + (high - low) / 2;
        if (haystack_chunk_offsets[mid] <= chunk_index) low = mid;
        else high = mid;
    }
    return low;
}

/**
 *  @brief Walks one chunk's transitions, warming up `max_match_bytes - 1` bytes before @p chunk_begin -
 *         clamped to @p haystack_begin, never earlier - so a match ending inside the chunk is found regardless
 *         of where its needle started, without ever reading into the previous haystack. Counts or writes
 *         every match ending in `[chunk_begin, chunk_end)` whose start offset is still within this haystack,
 *         per @p pass_.
 *  @return The number of matches found in the chunk.
 */
template <typename state_id_type_, find_many_pass_t pass_>
SZ_DEVICE_INLINE size_t find_many_walk_chunk_( //
    aho_corasick_view<state_id_type_> const &view, span<state_id_type_ const> shared_hot_rows, span<byte_t const> tape,
    size_t haystack_begin, size_t chunk_begin, size_t chunk_end, size_t haystack_index, size_t output_base_offset,
    span<find_many_cuda_match<state_id_type_>> matches_out) noexcept {

    size_t const warm_up_bytes = view.max_match_bytes > 0 ? (size_t)view.max_match_bytes - 1 : 0;
    size_t const walk_begin = chunk_begin >= haystack_begin + warm_up_bytes ? chunk_begin - warm_up_bytes
                                                                            : haystack_begin;

    state_id_type_ state = view.root; // ? Fresh at walk_begin - no state ever crosses a haystack boundary.
    size_t matches_found = 0;

    // The warm-up prefix reports nothing, and its length is fixed before the walk starts, so it runs as its
    // own loop rather than as a per-byte test every resident thread re-evaluates to the end of its chunk.
    for (size_t position = walk_begin; position < chunk_begin; ++position)
        state = find_many_step_device_(view, shared_hot_rows, state, (u8_t)tape[position]);

    for (size_t position = chunk_begin; position < chunk_end; ++position) {
        u8_t const byte = (u8_t)tape[position];
        state = find_many_step_device_(view, shared_hot_rows, state, byte);

        // Each local takes the width of the array it reads: counts ride the state id, offsets index a pool
        // that is O(states squared).
        state_id_type_ const output_count = view.outputs_counts[state];
        if (output_count == 0) continue;
        size_t const output_offset = view.outputs_offsets[state];
        for (state_id_type_ output_index = 0; output_index < output_count; ++output_index) {
            find_many_output<state_id_type_> const &output = view.outputs[output_offset + output_index];
            // Independent of the warm-up clamp above: a state's merged outputs never span more bytes than
            // the walk has taken since `walk_begin`, so this never triggers. Tested by addition rather than
            // by subtracting the length from the position, because a subtraction would wrap and read as
            // in-bounds in exactly the case the guard exists to catch.
            if (position + 1 < haystack_begin + (size_t)output.match_bytes) continue;
            if constexpr (pass_ == find_many_pass_t::scattering_k) {
                matches_out[output_base_offset + matches_found] = find_many_cuda_match<state_id_type_> {
                    position + 1, haystack_index, output.needle_index, output.match_bytes};
            }
            ++matches_found;
        }
    }
    return matches_found;
}

/**
 *  @brief How many equal-sized chunks of `chunk_bytes` a haystack of @p haystack_length bytes needs - at
 *         least one, so even an empty or shorter-than-a-chunk haystack still gets a thread.
 */
constexpr size_t find_many_chunks_for_haystack_(size_t haystack_length, size_t chunk_bytes) noexcept {
    return haystack_length == 0 ? (size_t)1 : divide_round_up(haystack_length, chunk_bytes);
}

/**
 *  @brief Writes each haystack's chunk count, feeding the exclusive scan that turns this into
 *         `haystack_chunk_offsets` (chunk-index range per haystack) - the input the two chunk kernels below
 *         binary-search to resolve a global chunk index back to its owning haystack.
 */
template <typename state_id_type_>
__global__ void find_many_count_chunks_per_haystack_(span<size_t const> haystack_offsets, size_t chunk_bytes,
                                                     span<size_t> chunks_per_haystack) {
    size_t const haystack_count = haystack_offsets.size() - 1;
    for (size_t haystack_index = (size_t)blockIdx.x * blockDim.x + threadIdx.x; haystack_index < haystack_count;
         haystack_index += (size_t)gridDim.x * blockDim.x) {
        size_t const haystack_length = haystack_offsets[haystack_index + 1] - haystack_offsets[haystack_index];
        chunks_per_haystack[haystack_index] = find_many_chunks_for_haystack_(haystack_length, chunk_bytes);
    }
}

/**
 *  @brief Walks every chunk in the tape, one thread per chunk, in whichever pass @p pass_ names.
 *
 *  The counting pass writes each chunk's match count into @p chunk_match_counts and touches no output; the
 *  scattering pass reads each chunk's exclusive offset out of @p chunk_match_offsets and writes its matches
 *  there. Every chunk owns a private, non-overlapping output range already sized by the counting pass and
 *  turned into offsets by @ref exclusive_sum_across_cuda_device_, so placing a write needs no atomics.
 *
 *  One kernel rather than two: the two passes differ only in that final statement, and splitting them
 *  duplicated the staging, the grid stride, and the haystack resolution verbatim.
 */
template <typename state_id_type_, find_many_pass_t pass_>
__global__ void find_many_walk_per_cuda_chunk_(aho_corasick_view<state_id_type_> view, state_id_type_ shared_rows_count,
                                               span<byte_t const> tape, span<size_t const> haystack_offsets,
                                               span<size_t const> haystack_chunk_offsets, size_t chunk_bytes,
                                               size_t chunk_count, span<size_t> chunk_match_counts,
                                               span<size_t const> chunk_match_offsets,
                                               span<find_many_cuda_match<state_id_type_>> matches_out) {
    extern __shared__ unsigned char find_many_shared_bytes_[];
    span<state_id_type_> const shared_hot_rows {reinterpret_cast<state_id_type_ *>(find_many_shared_bytes_),
                                                (size_t)shared_rows_count * find_many_alphabet_size_k};
    find_many_stage_hot_rows_(view, shared_hot_rows);

    for (size_t chunk_index = (size_t)blockIdx.x * blockDim.x + threadIdx.x; chunk_index < chunk_count;
         chunk_index += (size_t)gridDim.x * blockDim.x) {
        size_t const haystack_index = find_many_resolve_haystack_(haystack_chunk_offsets, chunk_index);
        size_t const haystack_begin = haystack_offsets[haystack_index];
        size_t const haystack_end = haystack_offsets[haystack_index + 1];
        size_t const local_chunk_index = chunk_index - haystack_chunk_offsets[haystack_index];
        size_t const chunk_begin = haystack_begin + local_chunk_index * chunk_bytes;
        size_t const chunk_end = sz_min_of_two(chunk_begin + chunk_bytes, haystack_end);

        size_t const output_base_offset = pass_ == find_many_pass_t::scattering_k ? chunk_match_offsets[chunk_index]
                                                                                  : (size_t)0;
        size_t const matches_in_chunk = find_many_walk_chunk_<state_id_type_, pass_>( //
            view, shared_hot_rows, tape, haystack_begin, chunk_begin, chunk_end, haystack_index, output_base_offset,
            matches_out);
        if constexpr (pass_ == find_many_pass_t::counting_k) chunk_match_counts[chunk_index] = matches_in_chunk;
        else sz_unused_(matches_in_chunk);
    }
}

#pragma endregion Device Kernels

#pragma region Engine

/**
 *  @brief Aho-Corasick-based @b GPU multi-pattern exact/case-folded substring search.
 *  @tparam state_id_type_ The dictionary's state ID type - must match whatever built the uploaded view.
 *  @tparam allocator_type_ The allocator backing this engine's device-resident scratch; unified memory by
 *          default, so the host can read match totals straight back after a stream synchronize.
 *  @tparam capability_ Any capability including `sz_cap_cuda_k` - the kernels need no generation-specific
 *          instructions, so every combination shares this specialization.
 *
 *  Move-only and owns its scratch: the uploaded automaton arrays, and the per-call chunk-planning buffers. A
 *  moved-from engine holds no device memory and must not be used before another `try_build`.
 */
template <                                       //
    typename state_id_type_ = u32_t,             //
    typename allocator_type_ = unified_alloc_t,  //
    sz_capability_t capability_ = sz_cap_cuda_k, //
    typename enable_ = void                      //
    >
struct find_many_cuda;

template <typename state_id_type_, typename allocator_type_, sz_capability_t capability_>
struct find_many_cuda<state_id_type_, allocator_type_, capability_,
                      std::enable_if_t<(capability_ & sz_cap_cuda_k) != 0>> {

    using state_id_t = state_id_type_;
    using allocator_t = allocator_type_;
    using view_t = aho_corasick_view<state_id_t>;
    using match_t = find_many_cuda_match<state_id_t>;
    static constexpr sz_capability_t capability_k = capability_;

    /**
     *  @brief How far beyond `state_count` the cold tier's `base`/`check`/`fail`/`outputs_counts`/
     *         `outputs_offsets` arrays must extend.
     *
     *  A cold transition's target is `base[state] + byte` for `byte` in `[0, 256)`; the builder's contract is
     *  that `base[state] < state_count` for every state - the double array reuses the compact state-id space
     *  as its own slot address space rather than a separate range - so the highest slot ever addressed is
     *  `(state_count - 1) + 255 = state_count + 254`. `state_count + find_many_cold_slot_headroom_k` is
     *  therefore the minimum array length that keeps every such index in bounds.
     */
    static constexpr size_t find_many_cold_slot_headroom_k = find_many_alphabet_size_k - 1;

  private:
    using allocator_traits_t = std::allocator_traits<allocator_t>;
    using state_allocator_t = typename allocator_traits_t::template rebind_alloc<state_id_t>;
    using output_allocator_t = typename allocator_traits_t::template rebind_alloc<find_many_output<state_id_t>>;
    using count_allocator_t = typename allocator_traits_t::template rebind_alloc<u32_t>;
    /** @brief Rebinds to `size_t`, for the output CSR and the tape offsets, neither of which has a ceiling. */
    using offset_allocator_t = typename allocator_traits_t::template rebind_alloc<size_t>;

    /** @brief Device-resident copies of the dictionary's arrays; this engine's owned scratch. */
    safe_vector<state_id_t, state_allocator_t> device_hot_rows_ {};
    safe_vector<state_id_t, state_allocator_t> device_base_ {};
    safe_vector<state_id_t, state_allocator_t> device_check_ {};
    safe_vector<state_id_t, state_allocator_t> device_fail_ {};
    safe_vector<find_many_output<state_id_t>, output_allocator_t> device_outputs_ {};
    safe_vector<state_id_t, state_allocator_t> device_outputs_counts_ {};
    safe_vector<size_t, offset_allocator_t> device_outputs_offsets_ {};

    /** @brief Per-call scratch: chunk count per haystack, then - in place - the exclusive chunk-index offset
     *         per haystack, with the grand total chunk count trailing at `[haystack_count]`. */
    safe_vector<size_t, offset_allocator_t> haystack_chunk_offsets_ {};
    /** @brief Per-call scratch: holds per-chunk counts, then - in place - per-chunk exclusive offsets, with the
     *         grand total in the trailing slot. Grown as needed, reused across calls. */
    safe_vector<size_t, offset_allocator_t> chunk_match_offsets_ {};

    /** @brief Device pointers into the buffers above, ready to pass by value into every kernel. */
    view_t view_ {};
    /** @brief Rows of the hot tier staged into shared memory at block start; sized once in `try_build`. */
    state_id_t shared_rows_ {};
    /** @brief Resident blocks per multiprocessor the hot tier is budgeted against in `try_build`; overridable
     *         through the accessor below. */
    size_t target_blocks_per_multiprocessor_ = 8;

    allocator_t alloc_ {};
    cuda_timer_t timer_ {};

  public:
    find_many_cuda() noexcept = default;
    find_many_cuda(find_many_cuda const &) = delete;
    find_many_cuda &operator=(find_many_cuda const &) = delete;
    find_many_cuda(find_many_cuda &&) noexcept = default;
    find_many_cuda &operator=(find_many_cuda &&) noexcept = default;

    /** @brief Releases every device-resident buffer this engine owns; a fresh `try_build` is required after. */
    void reset() noexcept {
        device_hot_rows_.reset();
        device_base_.reset();
        device_check_.reset();
        device_fail_.reset();
        device_outputs_.reset();
        device_outputs_counts_.reset();
        device_outputs_offsets_.reset();
        haystack_chunk_offsets_.reset();
        chunk_match_offsets_.reset();
        view_ = view_t {};
        shared_rows_ = state_id_t {};
    }

    /** @brief Overrides the resident-blocks-per-multiprocessor target the hot tier is budgeted against in
     *         `try_build`, instead of the default of `8`; call before `try_build`. */
    void target_blocks_per_multiprocessor(size_t desired) noexcept {
        target_blocks_per_multiprocessor_ = desired > 0 ? desired : 1;
    }
    size_t target_blocks_per_multiprocessor() const noexcept { return target_blocks_per_multiprocessor_; }

#pragma region Kernel Table

    struct kernels_t {
        kernel_shape_t chunks_per_haystack;
        kernel_shape_t count_chunk;
        kernel_shape_t scatter_chunk;
        kernel_shape_t exclusive_sum;
    };

    /** @brief Resolves every kernel handle for @p device_id into @p table, raising the dynamic shared-memory
     *         ceiling on the two chunk kernels to the device's opt-in maximum - the actual per-launch shared
     *         allocation depends on the dictionary's hot-tier size, so occupancy is queried per launch instead
     *         of precomputed here. */
    static cuda_status_t resolve_kernels_(kernels_t &table, int device_id) noexcept {
        CUdevice const device = static_cast<CUdevice>(device_id);
        int shared_memory_ceiling = 0;
        cuDeviceGetAttribute(&shared_memory_ceiling, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device);

        cuda_status_t status = resolve_kernel_shape(
            table.chunks_per_haystack, (void const *)&find_many_count_chunks_per_haystack_<state_id_t>, 0, 0, false);
        if (status.status != status_t::success_k) return status;

        status = resolve_kernel_shape(
            table.count_chunk, (void const *)&find_many_walk_per_cuda_chunk_<state_id_t, find_many_pass_t::counting_k>,
            0, (unsigned)shared_memory_ceiling, false);
        if (status.status != status_t::success_k) return status;

        status = resolve_kernel_shape(
            table.scatter_chunk,
            (void const *)&find_many_walk_per_cuda_chunk_<state_id_t, find_many_pass_t::scattering_k>, 0,
            (unsigned)shared_memory_ceiling, false);
        if (status.status != status_t::success_k) return status;

        status = resolve_kernel_shape(table.exclusive_sum, (void const *)&exclusive_sum_across_cuda_device_<size_t>, 0,
                                      0, false);
        return status;
    }

    /** @brief This device's kernel table, resolved on first use. Check the status before reading it - a full
     *         cache hands back an unresolved table. @sa cuda_device_kernels */
    static expected<kernels_t const &, cuda_status_t> kernels(int device_id) noexcept {
        static cuda_device_kernels<kernels_t> per_device;
        auto *entry = per_device.acquire(device_id);
        if (!entry) return {per_device.unusable(), {status_t::missing_gpu_k, cudaSuccess, CUDA_ERROR_INVALID_DEVICE}};
        if (!entry->resolved) {
            cuda_status_t const status = resolve_kernels_(entry->table, device_id);
            if (status.status != status_t::success_k) {
                per_device.release();
                return {per_device.unusable(), status};
            }
            entry->resolved = true;
        }
        per_device.release();
        return {entry->table, {}};
    }

#pragma endregion Kernel Table

    /**
     *  @brief Uploads a host-built automaton view to device memory, owned by this engine as its scratch, and
     *         sizes the shared-memory hot-tier prefix for @p executor 's device.
     *
     *  Uploads `base`, `check`, `fail`, `outputs_counts`, and `outputs_offsets` at
     *  `host_view.state_count + find_many_cold_slot_headroom_k` elements each - see that constant's doc for
     *  the exact bound the builder producing @p host_view must respect. The CSR `outputs` array's length
     *  comes straight from `outputs_total`, which the builder already knows; a cold state's id is not
     *  contiguous with `state_count`, so it could not be inferred from any one state's run.
     */
    cuda_status_t try_build(view_t const &host_view, cuda_executor_t const &executor) noexcept {
        cuda_status_t const current_status = executor.ensure_current();
        if (current_status.status != status_t::success_k) return current_status;

        size_t const hot_cells = (size_t)host_view.hot_count * find_many_alphabet_size_k;
        size_t const state_capacity = (size_t)host_view.state_count + find_many_cold_slot_headroom_k;
        size_t const total_outputs = host_view.outputs_total;

        if (device_hot_rows_.try_assign({host_view.hot_rows, hot_cells}) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};
        if (device_base_.try_assign({host_view.base, state_capacity}) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};
        if (device_check_.try_assign({host_view.check, state_capacity}) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};
        if (device_fail_.try_assign({host_view.fail, state_capacity}) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};
        if (device_outputs_.try_assign({host_view.outputs, total_outputs}) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};
        if (device_outputs_counts_.try_assign({host_view.outputs_counts, state_capacity}) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};
        if (device_outputs_offsets_.try_assign({host_view.outputs_offsets, state_capacity}) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        view_.hot_rows = device_hot_rows_.data();
        view_.base = device_base_.data();
        view_.check = device_check_.data();
        view_.fail = device_fail_.data();
        view_.outputs = device_outputs_.data();
        view_.outputs_counts = device_outputs_counts_.data();
        view_.outputs_offsets = device_outputs_offsets_.data();
        view_.hot_count = host_view.hot_count;
        view_.state_count = host_view.state_count;
        view_.root = host_view.root;
        view_.max_match_bytes = host_view.max_match_bytes;

        // Budgeted against `target_blocks_per_multiprocessor_` resident blocks, not the device's opt-in
        // per-block ceiling: a hot tier sized to the whole ceiling leaves only one block resident per SM,
        // which starves the thread-per-chunk design of the occupancy it needs.
        int shared_memory_per_multiprocessor = 0;
        cuDeviceGetAttribute(&shared_memory_per_multiprocessor,
                             CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, (CUdevice)executor.device_id());
        size_t const bytes_per_hot_row = find_many_alphabet_size_k * sizeof(state_id_t);
        size_t const shared_memory_budget = (size_t)shared_memory_per_multiprocessor /
                                            target_blocks_per_multiprocessor_;
        size_t const rows_that_fit = shared_memory_budget / bytes_per_hot_row;
        shared_rows_ = (state_id_t)sz_min_of_two((size_t)host_view.hot_count, rows_that_fit);

        return {status_t::success_k, cudaSuccess};
    }

  private:
    /** @brief Everything the counting pass establishes that a following scatter pass still needs. */
    struct planned_pass_t {
        kernels_t kernel_table {};
        unsigned shared_memory_bytes = 0;
        unsigned blocks_per_grid = 0;
        size_t chunk_bytes = 0;
        size_t chunk_count = 0;
        /** @brief False when the tape is empty, so the caller returns success without launching anything. */
        bool has_work = false;
    };

    /**
     *  @brief Runs everything `try_count` and `try_find` share: the guards, the kernel table, the timer's
     *         start, the chunk plan, and the counting pass whose exclusive scan both then read.
     */
    cuda_status_t plan_and_count_(span<byte_t const> tape, span<size_t const> haystack_offsets,
                                  cuda_executor_t const &executor, gpu_specs_t const &specs,
                                  planned_pass_t &pass) noexcept {
        cuda_status_t const current_status = executor.ensure_current();
        if (current_status.status != status_t::success_k) return current_status;
        if (haystack_offsets.size() < 2) return {status_t::success_k, cudaSuccess};
        if (haystack_offsets[haystack_offsets.size() - 1] == 0) return {status_t::success_k, cudaSuccess};

        auto [kernel_table, kernels_status] = kernels(executor.device_id());
        if (kernels_status.status != status_t::success_k) return kernels_status;
        pass.kernel_table = kernel_table;

        CUresult const timer_error = timer_.ensure_created(executor.device_id());
        if (timer_error != CUDA_SUCCESS) return make_cuda_status(timer_error);
        CUresult const start_error = timer_.record_start(executor.stream());
        if (start_error != CUDA_SUCCESS) return make_cuda_status(start_error);

        cuda_status_t const plan_status = plan_haystack_chunks_(haystack_offsets, executor, specs, pass.kernel_table,
                                                                pass.shared_memory_bytes, pass.blocks_per_grid,
                                                                pass.chunk_bytes, pass.chunk_count);
        if (plan_status.status != status_t::success_k) return plan_status;

        cuda_status_t const count_status = count_into_offsets_(tape, haystack_offsets, executor, pass.kernel_table,
                                                               pass.shared_memory_bytes, pass.blocks_per_grid,
                                                               pass.chunk_bytes, pass.chunk_count);
        if (count_status.status != status_t::success_k) return count_status;

        pass.has_work = true;
        return {status_t::success_k, cudaSuccess};
    }

    /**
     *  @brief Sizes the chunk grid from the counting kernel's occupancy under the dictionary's actual
     *         shared-memory footprint, then lays out every haystack's chunks against that target: counts
     *         chunks per haystack on the device, exclusive-scans them into `haystack_chunk_offsets_`, and
     *         syncs once to read the grand total chunk count back - the one host round-trip a per-haystack
     *         grid can't avoid, since the two chunk kernels below need that total as their loop bound.
     */
    cuda_status_t plan_haystack_chunks_(span<size_t const> haystack_offsets, cuda_executor_t const &executor,
                                        gpu_specs_t const &specs, kernels_t const &kernel_table,
                                        unsigned &shared_memory_bytes, unsigned &blocks_per_grid, size_t &chunk_bytes,
                                        size_t &chunk_count) noexcept {
        size_t const haystack_count = haystack_offsets.size() - 1;
        size_t const tape_length = (size_t)haystack_offsets[haystack_count];

        shared_memory_bytes = static_cast<unsigned>((size_t)shared_rows_ * find_many_alphabet_size_k *
                                                    sizeof(state_id_t));
        cuda_status_t const occupancy_status = occupancy_grid_for(blocks_per_grid, kernel_table.count_chunk.function,
                                                                  find_many_threads_per_block_k, shared_memory_bytes,
                                                                  specs);
        if (occupancy_status.status != status_t::success_k) return occupancy_status;

        size_t const target_threads = sz_max_of_two((size_t)blocks_per_grid * find_many_threads_per_block_k, (size_t)1);
        chunk_bytes = sz_max_of_two(divide_round_up(tape_length, target_threads), (size_t)1);

        if (haystack_chunk_offsets_.try_resize_uninitialized(haystack_count + 1) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        span<size_t const> haystack_offsets_argument = haystack_offsets;
        size_t chunk_bytes_argument = chunk_bytes;
        span<size_t> chunks_per_haystack_argument {haystack_chunk_offsets_.data(), haystack_count};
        void *chunks_per_haystack_arguments[3] = {&haystack_offsets_argument, &chunk_bytes_argument,
                                                  &chunks_per_haystack_argument};
        CUresult const chunks_error = cuda_launch_t {}
                                          .grid(blocks_per_grid)
                                          .block(find_many_threads_per_block_k)
                                          .shared(0)
                                          .stream(executor.stream())
                                          .launch(kernel_table.chunks_per_haystack.function,
                                                  chunks_per_haystack_arguments);
        if (chunks_error != CUDA_SUCCESS) return make_cuda_status(chunks_error);

        cuda_status_t const scan_status = cuda_launch_exclusive_sum_(
            kernel_table.exclusive_sum, (size_t const *)haystack_chunk_offsets_.data(), haystack_count,
            haystack_chunk_offsets_.data(), executor.stream());
        if (scan_status.status != status_t::success_k) return scan_status;

        CUresult const sync_error = timer_.synchronize(executor.stream());
        if (sync_error != CUDA_SUCCESS) return make_cuda_status(sync_error);
        chunk_count = (size_t)haystack_chunk_offsets_[haystack_count];
        return {status_t::success_k, cudaSuccess};
    }

    /** @brief Runs the counting pass, then the in-place exclusive scan, so `chunk_match_offsets_` holds every
     *         chunk's write offset with the grand total trailing at `[chunk_count]` - the shared core of
     *         `try_count` and `try_find`. */
    cuda_status_t count_into_offsets_(span<byte_t const> tape, span<size_t const> haystack_offsets,
                                      cuda_executor_t const &executor, kernels_t const &kernel_table,
                                      unsigned shared_memory_bytes, unsigned blocks_per_grid, size_t chunk_bytes,
                                      size_t chunk_count) noexcept {
        if (chunk_match_offsets_.try_resize_uninitialized(chunk_count + 1) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        view_t view_argument = view_;
        state_id_t shared_rows_argument = shared_rows_;
        span<byte_t const> tape_argument = tape;
        span<size_t const> haystack_offsets_argument = haystack_offsets;
        span<size_t const> haystack_chunk_offsets_argument {haystack_chunk_offsets_.data(),
                                                            haystack_chunk_offsets_.size()};
        size_t chunk_bytes_argument = chunk_bytes;
        size_t chunk_count_argument = chunk_count;
        span<size_t> chunk_counts_argument {chunk_match_offsets_.data(), chunk_match_offsets_.size()};
        span<size_t const> no_chunk_offsets_argument;
        span<find_many_cuda_match<state_id_t>> no_matches_argument;
        void *count_arguments[10] = {&view_argument,
                                     &shared_rows_argument,
                                     &tape_argument,
                                     &haystack_offsets_argument,
                                     &haystack_chunk_offsets_argument,
                                     &chunk_bytes_argument,
                                     &chunk_count_argument,
                                     &chunk_counts_argument,
                                     &no_chunk_offsets_argument,
                                     &no_matches_argument};
        CUresult const count_error = cuda_launch_t {}
                                         .grid(blocks_per_grid)
                                         .block(find_many_threads_per_block_k)
                                         .shared(shared_memory_bytes)
                                         .stream(executor.stream())
                                         .launch(kernel_table.count_chunk.function, count_arguments);
        if (count_error != CUDA_SUCCESS) return make_cuda_status(count_error);

        // In place: each chunk's slot holds its raw count going in and its exclusive offset coming out - the
        // scan kernel reads `input[i]` into a register before any thread writes `output[i]`, so reusing one
        // buffer for both is safe and skips a second chunk_count-sized allocation.
        return cuda_launch_exclusive_sum_(kernel_table.exclusive_sum, (size_t const *)chunk_match_offsets_.data(),
                                          chunk_count, chunk_match_offsets_.data(), executor.stream());
    }

  public:
    /**
     *  @brief Counts the number of occurrences of all needles across every haystack in @p tape.
     *  @param[in] tape The concatenated input bytes to search.
     *  @param[in] haystack_offsets `(N + 1)` sorted tape offsets for `N` haystacks - `haystack_offsets[i]` is
     *             haystack `i`'s first byte, `haystack_offsets[N]` is the tape's total length. Mirrors
     *             `sz_sequence_u32tape_t`'s convention. No needle is ever reported straddling two haystacks.
     */
    cuda_status_t try_count(span<byte_t const> tape, span<size_t const> haystack_offsets, size_t &matches_total,
                            cuda_executor_t const &executor, gpu_specs_t const &specs) noexcept {
        matches_total = 0;
        planned_pass_t pass;
        cuda_status_t const pass_status = plan_and_count_(tape, haystack_offsets, executor, specs, pass);
        if (pass_status.status != status_t::success_k || !pass.has_work) return pass_status;

        CUresult const stop_error = timer_.record_stop(executor.stream());
        if (stop_error != CUDA_SUCCESS) return make_cuda_status(stop_error);
        CUresult const sync_error = timer_.synchronize(executor.stream());
        if (sync_error != CUDA_SUCCESS) return make_cuda_status(sync_error);

        matches_total = (size_t)chunk_match_offsets_[pass.chunk_count];
        return {status_t::success_k, cudaSuccess, CUDA_SUCCESS, timer_.elapsed_milliseconds()};
    }

    /**
     *  @brief Finds all occurrences of all needles across every haystack in @p tape, via two real device
     *         passes: count then scatter, writing every match at its chunk's precomputed offset without
     *         atomics. See `try_count` for the `haystack_offsets` contract.
     *  @param[out] matches_out Caller-owned output buffer of at least `matches_capacity` entries.
     *  @retval `status_t::unexpected_dimensions_k` More matches exist than `matches_capacity` allows; nothing
     *          is written to `matches_out` in this case.
     */
    cuda_status_t try_find(span<byte_t const> tape, span<size_t const> haystack_offsets, span<match_t> matches_out,
                           size_t &matches_total, cuda_executor_t const &executor, gpu_specs_t const &specs) noexcept {
        size_t const matches_capacity = matches_out.size();
        matches_total = 0;
        planned_pass_t pass;
        cuda_status_t const pass_status = plan_and_count_(tape, haystack_offsets, executor, specs, pass);
        if (pass_status.status != status_t::success_k || !pass.has_work) return pass_status;
        auto const &kernel_table = pass.kernel_table;
        unsigned const shared_memory_bytes = pass.shared_memory_bytes, blocks_per_grid = pass.blocks_per_grid;
        size_t const chunk_bytes = pass.chunk_bytes, chunk_count = pass.chunk_count;

        // The grand total is only known once the scan has run, and only host-visible after a fence - this is
        // the one synchronous stall `try_find` cannot avoid, since the capacity check below must precede the
        // scatter launch.
        CUresult const mid_sync_error = timer_.synchronize(executor.stream());
        if (mid_sync_error != CUDA_SUCCESS) return make_cuda_status(mid_sync_error);
        size_t const matches_found = (size_t)chunk_match_offsets_[chunk_count];
        if (matches_found > matches_capacity) return {status_t::unexpected_dimensions_k, cudaSuccess};

        view_t view_argument = view_;
        state_id_t shared_rows_argument = shared_rows_;
        span<byte_t const> tape_argument = tape;
        span<size_t const> haystack_offsets_argument = haystack_offsets;
        span<size_t const> haystack_chunk_offsets_argument {haystack_chunk_offsets_.data(),
                                                            haystack_chunk_offsets_.size()};
        size_t chunk_bytes_argument = chunk_bytes;
        size_t chunk_count_argument = chunk_count;
        span<size_t> no_chunk_counts_argument;
        span<size_t const> chunk_offsets_argument {chunk_match_offsets_.data(), chunk_match_offsets_.size()};
        // Bounded to what the counting pass actually found, not to the caller's capacity, so a debug index
        // assert inside the kernel catches an over-write rather than merely staying inside the allocation.
        span<match_t> matches_out_argument {matches_out.data(), matches_found};
        void *scatter_arguments[10] = {&view_argument,
                                       &shared_rows_argument,
                                       &tape_argument,
                                       &haystack_offsets_argument,
                                       &haystack_chunk_offsets_argument,
                                       &chunk_bytes_argument,
                                       &chunk_count_argument,
                                       &no_chunk_counts_argument,
                                       &chunk_offsets_argument,
                                       &matches_out_argument};
        CUresult const scatter_error = cuda_launch_t {}
                                           .grid(blocks_per_grid)
                                           .block(find_many_threads_per_block_k)
                                           .shared(shared_memory_bytes)
                                           .stream(executor.stream())
                                           .launch(kernel_table.scatter_chunk.function, scatter_arguments);
        if (scatter_error != CUDA_SUCCESS) return make_cuda_status(scatter_error);

        CUresult const stop_error = timer_.record_stop(executor.stream());
        if (stop_error != CUDA_SUCCESS) return make_cuda_status(stop_error);
        CUresult const sync_error = timer_.synchronize(executor.stream());
        if (sync_error != CUDA_SUCCESS) return make_cuda_status(sync_error);

        matches_total = matches_found;
        return {status_t::success_k, cudaSuccess, CUDA_SUCCESS, timer_.elapsed_milliseconds()};
    }
};

using find_many_u16_cuda_t = find_many_cuda<u16_t, unified_alloc_t, sz_cap_cuda_k>;
using find_many_u32_cuda_t = find_many_cuda<u32_t, unified_alloc_t, sz_cap_cuda_k>;

#pragma endregion Engine

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_FIND_MANY_CUDA_CUH_
