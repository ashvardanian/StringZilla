/**
 *  @brief  CUDA backend for multi-pattern exact and case-folded substring search.
 *  @file   include/stringzillas/substrings/cuda.cuh
 *  @author Ash Vardanian
 *  @sa     include/stringzillas/substrings/serial.hpp
 *
 *  `try_build` uploads a host-built `aho_corasick_view`, so this backend consumes the published contract
 *  rather than the builder's internals.
 *
 *  Each haystack travels as its own pointer-and-length descriptor, so no layout is assumed: one packed
 *  tape and scattered device allocations chunk identically. One thread owns one contiguous slice of one
 *  haystack, sized against the corpus's total byte count rather than the haystack count, so occupancy does
 *  not depend on how the corpus is split, and no needle can straddle the seam between two documents.
 *
 *  A thread starts its walk `max_match_bytes - 1` bytes before its chunk, clamped to its own haystack's
 *  start. Aho-Corasick is self-synchronizing, so that warm-up makes chunking exact rather than approximate:
 *  the automaton reaches the same state at the chunk boundary wherever the walk began.
 *
 *  A transition is one data-dependent load, so the walk is latency-bound rather than bandwidth-bound and
 *  resident warps are what hides it. The automaton is therefore staged into shared memory only when the
 *  whole of it fits without costing a resident block, and read through the cache hierarchy otherwise.
 */
#ifndef STRINGZILLAS_SUBSTRINGS_CUDA_CUH_
#define STRINGZILLAS_SUBSTRINGS_CUDA_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include "stringzillas/types.cuh"             // `unified_alloc_t`, `cuda_status_t`
#include "stringzillas/substrings/serial.hpp" // `aho_corasick_view`, the host-built contract

namespace ashvardanian {
namespace stringzillas {

// Per-symbol: a using-directive re-exports our `memcpy` and nvcc then finds the call ambiguous.
using ashvardanian::stringzilla::byte_t;
using ashvardanian::stringzilla::size_t;
using ashvardanian::stringzilla::small_size_t;
using ashvardanian::stringzilla::span;
using ashvardanian::stringzilla::status_t;
using ashvardanian::stringzilla::to_bytes_view;

#pragma region Device Kernels

/** @brief Block size every `substrings_cuda` kernel launches with; occupancy is shared-memory-bound, not
 *         thread-count-bound, so a modest fixed block keeps the launch geometry simple. */
static constexpr unsigned substrings_threads_per_block_k = 256;

/**
 *  @brief Selects a chunk walk's pass: `counting_k` only tallies matches so the caller can size and scan the
 *         output, `scattering_k` writes each match at its chunk's precomputed offset.
 *
 *  A `bool`-backed scoped enum consumed with `if constexpr`, matching `tile_march_t` in `stringzillas/types.cuh`.
 */
enum class substrings_pass_t : bool { counting_k = false, scattering_k = true };

/**
 *  @brief Cooperatively fills @p shared_hot_rows from the head of the hot tier and @p shared_accepts_words
 *         from the acceptance bitmap, once per block. The hot tier's out-degree ordering makes its head the
 *         best prefix to stage; the bitmap span is empty when `try_build` budgeted it out of shared memory.
 */
template <typename state_id_type_>
SZ_DEVICE_INLINE void substrings_stage_automaton_(aho_corasick_view<state_id_type_> const &view,
                                                  span<state_id_type_> shared_hot_rows,
                                                  span<u32_t const> accepts_words,
                                                  span<u32_t> shared_accepts_words) noexcept {
    for (size_t cell = threadIdx.x; cell < shared_hot_rows.size(); cell += blockDim.x)
        shared_hot_rows[cell] = view.hot_rows[cell];
    for (size_t word = threadIdx.x; word < shared_accepts_words.size(); word += blockDim.x)
        shared_accepts_words[word] = accepts_words[word];
    __syncthreads();
}

/**
 *  @brief One byte's transition, staged-shared-memory-first: the staged prefix resolves branch-free from
 *         shared memory, and everything else - hot tier beyond the prefix, and the whole cold tier - defers
 *         to @ref aho_corasick_step, the single transition definition every backend shares.
 *
 *  A single cold lane still makes the whole warp pay that lane's failure-chase depth, which is the cost the
 *  shared-memory staging exists to shrink.
 */
template <typename state_id_type_>
SZ_DEVICE_INLINE state_id_type_ substrings_step_device_(aho_corasick_view<state_id_type_> const &view,
                                                        span<state_id_type_ const> shared_hot_rows,
                                                        small_size_t shared_rows_count, state_id_type_ state,
                                                        u8_t byte) noexcept {
    if (static_cast<small_size_t>(state) < shared_rows_count)
        return hot_row_of<small_size_t>(shared_hot_rows, state)[byte];
    return aho_corasick_step(view, state, byte);
}

/**
 *  @brief Finds, via binary search, which haystack owns global chunk @p chunk_index, given the exclusive
 *         prefix sum of chunk counts per haystack (`haystack_chunk_offsets[haystack_count]` is the grand
 *         total chunk count, mirroring how `outputs_offsets` bounds `outputs_counts`).
 */
SZ_DEVICE_INLINE size_t substrings_resolve_haystack_(span<size_t const> haystack_chunk_offsets,
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
 *         clamped to the haystack's own start, never earlier - so a match ending inside the chunk is found
 *         regardless of where its needle started, without reading another haystack. Counts or writes
 *         every match ending in `[chunk_begin, chunk_end)` whose start offset is still within this haystack,
 *         per @p pass_.
 *  @return The number of matches found in the chunk.
 */
template <typename state_id_type_, substrings_pass_t pass_>
SZ_DEVICE_INLINE size_t substrings_walk_chunk_( //
    aho_corasick_view<state_id_type_> const &view, span<state_id_type_ const> shared_hot_rows,
    span<u32_t const> accepts_words, span<byte_t const> haystack, size_t chunk_begin, size_t chunk_end,
    size_t haystack_index, size_t output_base_offset, span<substrings_match_t> matches_out) noexcept {

    // Offsets are relative to this haystack, so the warm-up clamps against its own start at zero.
    size_t const warm_up_bytes = view.max_match_bytes > 0 ? (size_t)view.max_match_bytes - 1 : 0;
    size_t const walk_begin = chunk_begin >= warm_up_bytes ? chunk_begin - warm_up_bytes : 0;

    // Every 64-bit quantity is resolved here, once, and the per-byte loops below ride 32-bit deltas from it.
    byte_t const *const walk_base = haystack.data() + walk_begin;
    substrings_match_t *const matches_at_chunk = matches_out.data() + output_base_offset;
    small_size_t const walk_span = static_cast<small_size_t>(chunk_end - walk_begin);
    small_size_t const emit_from = static_cast<small_size_t>(chunk_begin - walk_begin);
    sz_assert_(shared_hot_rows.size() / substrings_alphabet_size_k <= std::numeric_limits<small_size_t>::max() &&
               "The staged prefix is budgeted against one multiprocessor's shared memory in `try_build`");
    small_size_t const shared_rows_count = static_cast<small_size_t>(shared_hot_rows.size() /
                                                                     substrings_alphabet_size_k);

    state_id_type_ state = view.root; // ? Fresh at walk_begin - no state ever crosses a haystack boundary.
    small_size_t matches_found = 0;

    // Every match ending at `position` under the state just entered. The bit answers "does anything end
    // here" - the common no-match byte never touches the global counts array, which at scale costs nearly
    // as much as the tape read itself. Counts ride the state id; offsets index a pool that is O(states
    // squared) and so stays 64-bit, but only as a base hoisted out of the inner loop.
    auto const emit_matches_at = [&](small_size_t position) {
        if (((accepts_words[state >> 5] >> (state & 31u)) & 1u) == 0) return;
        state_id_type_ const output_count = view.outputs_counts[state];
        substrings_output<state_id_type_> const *const outputs_at_state = view.outputs + view.outputs_offsets[state];
        for (state_id_type_ output_index = 0; output_index < output_count; ++output_index) {
            substrings_output<state_id_type_> const &output = outputs_at_state[output_index];
            // `walk_begin` is clamped to the haystack's own start, so underflowing the walk and underflowing
            // the haystack are the same test - and this one needs no absolute offset.
            if (position + 1 < static_cast<small_size_t>(output.match_bytes)) continue;
            if constexpr (pass_ == substrings_pass_t::scattering_k) {
                size_t const match_end = walk_begin + position + 1;
                matches_at_chunk[matches_found] = substrings_match_t {haystack_index, (size_t)output.needle_index,
                                                                      match_end - output.match_bytes,
                                                                      (size_t)output.match_bytes};
            }
            ++matches_found;
        }
    };

    // The warm-up primes the state and reports nothing, so once it ends the emit test vanishes from the
    // loop rather than being re-asked on every byte.
    small_size_t delta = 0;
    for (; delta < emit_from; ++delta)
        state = substrings_step_device_(view, shared_hot_rows, shared_rows_count, state, walk_base[delta]);

    // Peeled to the load's own alignment, so the body pays one 4-byte load per four transitions - the
    // transition chain stays strictly serial; only the tape reads widen.
    for (; delta < walk_span && ((size_t)(walk_base + delta) & 3u) != 0; ++delta) {
        state = substrings_step_device_(view, shared_hot_rows, shared_rows_count, state, walk_base[delta]);
        emit_matches_at(delta);
    }
    for (; delta + 4 <= walk_span; delta += 4) {
        u32_vec_t const quad = sz_u32_load_aligned(walk_base + delta);
#pragma unroll
        for (small_size_t lane = 0; lane < 4; ++lane) {
            state = substrings_step_device_(view, shared_hot_rows, shared_rows_count, state,
                                            static_cast<u8_t>(quad.u32 >> (lane * 8)));
            emit_matches_at(delta + lane);
        }
    }
    for (; delta < walk_span; ++delta) {
        state = substrings_step_device_(view, shared_hot_rows, shared_rows_count, state, walk_base[delta]);
        emit_matches_at(delta);
    }
    return matches_found;
}

/**
 *  @brief How many equal-sized chunks of `chunk_bytes` a haystack of @p haystack_length bytes needs - at
 *         least one, so even an empty or shorter-than-a-chunk haystack still gets a thread.
 */
constexpr size_t substrings_chunks_for_haystack_(size_t haystack_length, size_t chunk_bytes) noexcept {
    return haystack_length == 0 ? (size_t)1 : divide_round_up(haystack_length, chunk_bytes);
}

/**
 *  @brief Writes each haystack's chunk count, feeding the exclusive scan that turns this into
 *         `haystack_chunk_offsets` (chunk-index range per haystack) - the input the two chunk kernels below
 *         binary-search to resolve a global chunk index back to its owning haystack.
 */
template <typename state_id_type_>
__global__ void substrings_count_chunks_per_haystack_(span<span<byte_t const> const> haystacks, size_t chunk_bytes,
                                                      span<size_t> chunks_per_haystack) {
    for (size_t haystack_index = (size_t)blockIdx.x * blockDim.x + threadIdx.x; haystack_index < haystacks.size();
         haystack_index += (size_t)gridDim.x * blockDim.x)
        chunks_per_haystack[haystack_index] = substrings_chunks_for_haystack_(haystacks[haystack_index].size(),
                                                                              chunk_bytes);
}

/**
 *  @brief Walks every chunk of every haystack, one thread per chunk, in whichever pass @p pass_ names.
 *
 *  Both passes share one @p chunk_match_slots buffer, because the host's in-place exclusive scan already
 *  makes them the same allocation: the counting pass writes each chunk's match count into its slot, and the
 *  scattering pass reads the exclusive offset the scan left there. Every chunk owns a private,
 *  non-overlapping output range, so placing a write needs no atomics.
 */
template <typename state_id_type_, substrings_pass_t pass_>
__global__ void substrings_walk_per_cuda_chunk_(
    aho_corasick_view<state_id_type_> view, state_id_type_ shared_rows_count, span<u32_t const> accepts_words,
    u32_t staged_accepts_words, span<span<byte_t const> const> haystacks, span<size_t const> haystack_chunk_offsets,
    size_t chunk_bytes, size_t chunk_count, span<size_t> chunk_match_slots, span<substrings_match_t> matches_out) {
    extern __shared__ unsigned char substrings_shared_bytes_[];
    span<state_id_type_> const shared_hot_rows {reinterpret_cast<state_id_type_ *>(substrings_shared_bytes_),
                                                (size_t)shared_rows_count * substrings_alphabet_size_k};
    // The bitmap words land right after the rows, whose byte count is a multiple of four at either id width.
    span<u32_t> const shared_accepts_words {
        reinterpret_cast<u32_t *>(substrings_shared_bytes_ + shared_hot_rows.size() * sizeof(state_id_type_)),
        staged_accepts_words};
    substrings_stage_automaton_(view, shared_hot_rows, accepts_words, shared_accepts_words);
    // Resolved once per block: the staged copy when `try_build` budgeted it in, the global array otherwise.
    span<u32_t const> const accepts = staged_accepts_words
                                          ? span<u32_t const> {shared_accepts_words.data(), accepts_words.size()}
                                          : accepts_words;

    for (size_t chunk_index = (size_t)blockIdx.x * blockDim.x + threadIdx.x; chunk_index < chunk_count;
         chunk_index += (size_t)gridDim.x * blockDim.x) {
        size_t const haystack_index = substrings_resolve_haystack_(haystack_chunk_offsets, chunk_index);
        span<byte_t const> const haystack = haystacks[haystack_index];
        size_t const local_chunk_index = chunk_index - haystack_chunk_offsets[haystack_index];
        size_t const chunk_begin = local_chunk_index * chunk_bytes;
        size_t const chunk_end = sz_min_of_two(chunk_begin + chunk_bytes, haystack.size());

        size_t const output_base_offset = pass_ == substrings_pass_t::scattering_k ? chunk_match_slots[chunk_index]
                                                                                   : (size_t)0;
        size_t const matches_in_chunk = substrings_walk_chunk_<state_id_type_, pass_>( //
            view, shared_hot_rows, accepts, haystack, chunk_begin, chunk_end, haystack_index, output_base_offset,
            matches_out);
        if constexpr (pass_ == substrings_pass_t::counting_k) chunk_match_slots[chunk_index] = matches_in_chunk;
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
struct substrings_cuda;

template <typename state_id_type_, typename allocator_type_, sz_capability_t capability_>
struct substrings_cuda<state_id_type_, allocator_type_, capability_,
                       std::enable_if_t<(capability_ & sz_cap_cuda_k) != 0>> {

    using state_id_t = state_id_type_;
    using allocator_t = allocator_type_;
    using dictionary_t = aho_corasick_dictionary<state_id_t, allocator_t>;
    using view_t = aho_corasick_view<state_id_t>;
    using match_t = substrings_match_t;
    static constexpr sz_capability_t capability_k = capability_;

    /**
     *  @brief How far beyond `state_count` the cold tier's `base`/`check`/`fail`/`outputs_counts`/
     *         `outputs_offsets` arrays must extend.
     *
     *  A cold transition's target is `base[state] + byte` for `byte` in `[0, 256)`, and the builder guarantees
     *  `base[state] < state_count`, so the highest slot ever addressed is `state_count + 254`.
     */
    static constexpr size_t substrings_cold_slot_headroom_k = substrings_alphabet_size_k - 1;

  private:
    using allocator_traits_t = std::allocator_traits<allocator_t>;
    using state_allocator_t = typename allocator_traits_t::template rebind_alloc<state_id_t>;
    using output_allocator_t = typename allocator_traits_t::template rebind_alloc<substrings_output<state_id_t>>;
    /** @brief Rebinds to `size_t`, for the output CSR and the chunk offsets, neither of which has a ceiling. */
    using offset_allocator_t = typename allocator_traits_t::template rebind_alloc<size_t>;
    using descriptor_allocator_t = typename allocator_traits_t::template rebind_alloc<span<byte_t const>>;
    using match_allocator_t = typename allocator_traits_t::template rebind_alloc<substrings_match_t>;
    using word_allocator_t = typename allocator_traits_t::template rebind_alloc<u32_t>;

    /** @brief Device-resident copies of the dictionary's arrays; this engine's owned scratch. */
    safe_vector<state_id_t, state_allocator_t> device_hot_rows_ {};
    safe_vector<state_id_t, state_allocator_t> device_base_ {};
    safe_vector<state_id_t, state_allocator_t> device_check_ {};
    safe_vector<state_id_t, state_allocator_t> device_fail_ {};
    safe_vector<substrings_output<state_id_t>, output_allocator_t> device_outputs_ {};
    safe_vector<state_id_t, state_allocator_t> device_outputs_counts_ {};
    safe_vector<size_t, offset_allocator_t> device_outputs_offsets_ {};
    /**
     *  @brief Dense acceptance bitmap: bit `state` is set when some needle ends at that state, hot or cold.
     *
     *  Derived from `outputs_counts` at upload, so the per-byte walk gate never touches that 32x larger
     *  array. Words are 32-bit because that is one shared-memory bank: the gate reads exactly one bank-wide
     *  word, and lanes clustered near the root share it as a broadcast rather than a conflict.
     */
    safe_vector<u32_t, word_allocator_t> device_accepts_words_ {};

    /** @brief One descriptor per haystack. Unified, as the host writes them and every chunk thread reads them. */
    safe_vector<span<byte_t const>, descriptor_allocator_t> haystack_descriptors_ {};
    /** @brief Per-call scratch: chunk count per haystack, then - in place - the exclusive chunk-index offset
     *         per haystack, with the grand total chunk count trailing at `[haystack_count]`. */
    safe_vector<size_t, offset_allocator_t> haystack_chunk_offsets_ {};
    /** @brief Per-call scratch: holds per-chunk counts, then - in place - per-chunk exclusive offsets, with the
     *         grand total in the trailing slot. Grown as needed, reused across calls. */
    safe_vector<size_t, offset_allocator_t> chunk_match_offsets_ {};
    /** @brief Unified staging `try_find` scatters into when the caller's output span is host memory a kernel
     *         cannot reach. Mirrors `cuda_cross_buffers::results_staging_` in the similarities engines. */
    safe_vector<substrings_match_t, match_allocator_t> matches_staging_ {};

    /** @brief The host-built automaton this engine compiles from its needles and uploads at `try_build`. */
    dictionary_t dictionary_ {};

    /** @brief The device twin of `aho_corasick_dictionary::view`: scalars from the host build, pointers from
     *         the uploaded buffers. Assembled on demand, so no member can go stale; no driver calls. */
    view_t device_view_() const noexcept {
        view_t view = dictionary_.view();
        view.hot_rows = device_hot_rows_.data();
        view.base = device_base_.data();
        view.check = device_check_.data();
        view.fail = device_fail_.data();
        view.outputs = device_outputs_.data();
        view.outputs_counts = device_outputs_counts_.data();
        view.outputs_offsets = device_outputs_offsets_.data();
        return view;
    }
    /** @brief Hot rows staged into shared memory at block start - all of them, or zero when they would cost
     *         a resident block and the kernels read them through the cache instead. */
    state_id_t shared_rows_ {};
    /** @brief Acceptance bitmap words staged alongside `shared_rows_`, under the same all-or-nothing rule. */
    u32_t staged_accepts_words_ {};
    /** @brief Resident blocks per multiprocessor to budget shared memory against; zero derives it from the
     *         occupancy the kernel reaches with none, which is the default the accessor below overrides. */
    unsigned target_blocks_per_multiprocessor_ = 0;

    allocator_t alloc_ {};
    cuda_timer_t timer_ {};

  public:
    substrings_cuda() noexcept = default;
    substrings_cuda(substrings_cuda const &) = delete;
    substrings_cuda &operator=(substrings_cuda const &) = delete;
    substrings_cuda(substrings_cuda &&) noexcept = default;
    substrings_cuda &operator=(substrings_cuda &&) noexcept = default;

    /** @brief Releases the dictionary and every device-resident buffer this engine owns; a fresh
     *         `try_build` is required after. */
    void reset() noexcept {
        dictionary_.reset();
        device_hot_rows_.reset();
        device_base_.reset();
        device_check_.reset();
        device_fail_.reset();
        device_outputs_.reset();
        device_outputs_counts_.reset();
        device_outputs_offsets_.reset();
        haystack_chunk_offsets_.reset();
        chunk_match_offsets_.reset();
        matches_staging_.reset();
        shared_rows_ = state_id_t {};
    }

    /** @brief Overrides the resident-blocks-per-multiprocessor target the hot tier is budgeted against in
     *         `try_build`; call before `try_build`, with zero restoring the automatic choice. */
    void target_blocks_per_multiprocessor(unsigned desired) noexcept { target_blocks_per_multiprocessor_ = desired; }
    unsigned target_blocks_per_multiprocessor() const noexcept { return target_blocks_per_multiprocessor_; }

#pragma region Kernel Table

    struct kernels_t {
        kernel_shape_t chunks_per_haystack;
        kernel_shape_t count_chunk;
        kernel_shape_t scatter_chunk;
        kernel_shape_t exclusive_sum;
    };

    /** @brief Resolves every kernel handle for @p device_id into @p table, raising the dynamic shared-memory
     *         ceiling on the two chunk kernels to the device's opt-in maximum. The per-launch allocation
     *         depends on the dictionary's hot-tier size, so occupancy is queried per launch. */
    static cuda_status_t resolve_kernels_(kernels_t &table, int device_id) noexcept {
        CUdevice const device = device_id;
        int shared_memory_ceiling = 0;
        cuDeviceGetAttribute(&shared_memory_ceiling, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device);

        cuda_status_t status = resolve_kernel_shape(
            table.chunks_per_haystack,
            reinterpret_cast<void const *>(&substrings_count_chunks_per_haystack_<state_id_t>), 0, 0, false);
        if (status.status != status_t::success_k) return status;

        status = resolve_kernel_shape(
            table.count_chunk,
            reinterpret_cast<void const *>(&substrings_walk_per_cuda_chunk_<state_id_t, substrings_pass_t::counting_k>),
            0, static_cast<unsigned>(shared_memory_ceiling), false);
        if (status.status != status_t::success_k) return status;

        status = resolve_kernel_shape(
            table.scatter_chunk,
            reinterpret_cast<void const *>(
                &substrings_walk_per_cuda_chunk_<state_id_t, substrings_pass_t::scattering_k>),
            0, static_cast<unsigned>(shared_memory_ceiling), false);
        if (status.status != status_t::success_k) return status;

        status = resolve_kernel_shape(table.exclusive_sum,
                                      reinterpret_cast<void const *>(&exclusive_sum_across_cuda_device_<size_t>), 0, 0,
                                      false);
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
     *  @brief Indexes all of the @p needles strings into the FSM and uploads it to @p executor 's device.
     *  @param[in] specs Sizes the host dictionary's hot tier from the last-level cache.
     *  @note Before reusing, please `reset` the FSM.
     *  @sa `aho_corasick_dictionary::try_insert` for the status codes this forwards.
     */
    template <typename needles_type_>
    cuda_status_t try_build(needles_type_ const &needles,
                            substrings_case_sensitivity_t case_sensitivity = substrings_cased_k,
                            cuda_executor_t const &executor = {}, cpu_specs_t const &specs = {}) noexcept {
        dictionary_.case_sensitivity(case_sensitivity);
        for (auto const &needle : needles) {
            status_t const status = dictionary_.try_insert(to_bytes_view(needle));
            if (status != status_t::success_k) return {status, cudaSuccess};
        }
        status_t const built = dictionary_.try_build(specs);
        if (built != status_t::success_k) return {built, cudaSuccess};
        return try_upload_(dictionary_.view(), executor);
    }

    dictionary_t const &dictionary() const noexcept { return dictionary_; }

  private:
    /**
     *  @brief Uploads the host-built automaton view to device memory, owned by this engine as its scratch,
     *         and sizes the shared-memory hot-tier prefix for @p executor 's device.
     *
     *  Uploads `base`, `check`, `fail`, `outputs_counts`, and `outputs_offsets` at
     *  `host_view.state_count + substrings_cold_slot_headroom_k` elements each. The CSR `outputs` length comes
     *  from `outputs_total`, which no single state's run could imply.
     */
    cuda_status_t try_upload_(view_t const &host_view, cuda_executor_t const &executor) noexcept {
        cuda_status_t const current_status = executor.ensure_current();
        if (current_status.status != status_t::success_k) return current_status;

        size_t const hot_cells = (size_t)host_view.hot_count * substrings_alphabet_size_k;
        size_t const state_capacity = (size_t)host_view.state_count + substrings_cold_slot_headroom_k;
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

        // One bit per published slot, over the same capacity the counts upload covers. `try_resize` leaves
        // trivial words uninitialized, so every word is written before any bit is set.
        size_t const accepts_words_count = divide_round_up<size_t>(state_capacity, 32);
        if (device_accepts_words_.try_resize(accepts_words_count) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};
        for (size_t word = 0; word < accepts_words_count; ++word) device_accepts_words_[word] = 0;
        for (size_t slot = 0; slot < state_capacity; ++slot)
            if (host_view.outputs_counts[slot] != 0)
                device_accepts_words_[slot >> 5] |= u32_t(1) << (slot & 31u);

        auto [kernel_table, kernels_status] = kernels(executor.device_id());
        if (kernels_status.status != status_t::success_k) return kernels_status;
        CUfunction const walk_function = kernel_table.count_chunk.function; // ? The scatter kernel shares its shape

        // Occupancy first, staging only out of what is left over. The walk chases a data-dependent transition
        // load, so resident warps are the only thing hiding its latency, while the rows it would stage are
        // cache-resident already - which makes a block traded away for shared memory a straight loss.
        unsigned target_blocks = target_blocks_per_multiprocessor_;
        if (target_blocks == 0) {
            int blocks_without_staging = 0;
            CUresult const occupancy_error = cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks_without_staging, walk_function, (int)substrings_threads_per_block_k, 0);
            if (occupancy_error != CUDA_SUCCESS) return make_cuda_status(occupancy_error);
            target_blocks = (unsigned)sz_max_of_two(blocks_without_staging, 1);
        }

        size_t shared_memory_budget = 0;
        cuda_status_t const budget_status = shared_memory_budget_for_resident_blocks(
            shared_memory_budget, walk_function, substrings_threads_per_block_k, target_blocks,
            executor.device_id());
        if (budget_status.status != status_t::success_k) return budget_status;

        // Staging is all-or-nothing: a partial prefix still pays the copy per block and the bounds test per
        // byte, while most transitions miss it and fall through to memory anyway.
        size_t const bytes_per_hot_row = substrings_alphabet_size_k * sizeof(state_id_t);
        size_t const accepts_bytes = accepts_words_count * sizeof(u32_t);
        size_t const whole_automaton_bytes = (size_t)host_view.hot_count * bytes_per_hot_row + accepts_bytes;
        bool const stages_whole_automaton = whole_automaton_bytes <= shared_memory_budget;
        shared_rows_ = stages_whole_automaton ? static_cast<state_id_t>(host_view.hot_count) : state_id_t {0};
        staged_accepts_words_ = stages_whole_automaton ? static_cast<u32_t>(accepts_words_count) : u32_t {0};

        return {status_t::success_k, cudaSuccess};
    }

    /** @brief Everything the counting pass establishes that a following scatter pass still needs. */
    struct planned_pass_t {
        kernels_t kernel_table {};
        unsigned shared_memory_bytes = 0;
        unsigned blocks_per_grid = 0;
        size_t chunk_bytes = 0;
        size_t chunk_count = 0;
        /** @brief False when the corpus is empty, so the caller returns success without launching anything. */
        bool has_work = false;
    };

    /**
     *  @brief Records one pointer-and-length descriptor per haystack and sums their bytes.
     *
     *  No layout is assumed: haystacks may sit in one tape or in separate allocations, and the input bytes
     *  are validated rather than copied, as in every other CUDA engine here.
     */
    template <typename haystacks_type_>
    cuda_status_t describe_haystacks_(haystacks_type_ const &haystacks, size_t &total_bytes) noexcept {
        if (haystack_descriptors_.try_resize_uninitialized(haystacks.size()) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        total_bytes = 0;
        bool probed = false;
        for (size_t haystack_index = 0; haystack_index < haystacks.size(); ++haystack_index) {
            span<byte_t const> const haystack = to_bytes_view(haystacks[haystack_index]);
            haystack_descriptors_[haystack_index] = haystack;
            total_bytes += haystack.size();
            // The probe is a driver round-trip, so one non-empty element decides for the whole batch.
            if (!probed && haystack.size() != 0) {
                if (!is_device_accessible_memory((void const *)haystack.data()))
                    return {status_t::device_memory_mismatch_k, cudaSuccess};
                probed = true;
            }
        }
        return {status_t::success_k, cudaSuccess};
    }

    /**
     *  @brief Runs everything `try_count` and `try_find` share: the guards, the kernel table, the timer's
     *         start, the chunk plan, and the counting pass whose exclusive scan both then read.
     */
    cuda_status_t plan_and_count_(size_t total_bytes, cuda_executor_t const &executor, gpu_specs_t const &specs,
                                  planned_pass_t &pass) noexcept {
        cuda_status_t const current_status = executor.ensure_current();
        if (current_status.status != status_t::success_k) return current_status;
        if (haystack_descriptors_.size() == 0 || total_bytes == 0) return {status_t::success_k, cudaSuccess};

        auto [kernel_table, kernels_status] = kernels(executor.device_id());
        if (kernels_status.status != status_t::success_k) return kernels_status;
        pass.kernel_table = kernel_table;

        CUresult const timer_error = timer_.ensure_created(executor.device_id());
        if (timer_error != CUDA_SUCCESS) return make_cuda_status(timer_error);
        CUresult const start_error = timer_.record_start(executor.stream());
        if (start_error != CUDA_SUCCESS) return make_cuda_status(start_error);

        cuda_status_t const plan_status = plan_haystack_chunks_(total_bytes, executor, specs, pass.kernel_table,
                                                                pass.shared_memory_bytes, pass.blocks_per_grid,
                                                                pass.chunk_bytes, pass.chunk_count);
        if (plan_status.status != status_t::success_k) return plan_status;

        cuda_status_t const count_status = count_into_offsets_(executor, pass.kernel_table, pass.shared_memory_bytes,
                                                               pass.blocks_per_grid, pass.chunk_bytes,
                                                               pass.chunk_count);
        if (count_status.status != status_t::success_k) return count_status;

        pass.has_work = true;
        return {status_t::success_k, cudaSuccess};
    }

    /**
     *  @brief Sizes the chunk grid from the counting kernel's occupancy under the dictionary's actual
     *         shared-memory footprint, then lays out every haystack's chunks against that target: counts
     *         chunks per haystack on the device, exclusive-scans them into `haystack_chunk_offsets_`, and
     *         syncs once to read the grand total chunk count back, which the two chunk kernels need as their
     *         loop bound.
     */
    cuda_status_t plan_haystack_chunks_(size_t total_bytes, cuda_executor_t const &executor, gpu_specs_t const &specs,
                                        kernels_t const &kernel_table, unsigned &shared_memory_bytes,
                                        unsigned &blocks_per_grid, size_t &chunk_bytes, size_t &chunk_count) noexcept {
        size_t const haystack_count = haystack_descriptors_.size();

        size_t const staged_bytes = (size_t)shared_rows_ * substrings_alphabet_size_k * sizeof(state_id_t) +
                                    (size_t)staged_accepts_words_ * sizeof(u32_t);
        sz_assert_(staged_bytes <= std::numeric_limits<unsigned>::max() &&
                   "The staged prefix is budgeted against one multiprocessor's shared memory in `try_build`");
        shared_memory_bytes = static_cast<unsigned>(staged_bytes);
        cuda_status_t const occupancy_status = occupancy_grid_for(blocks_per_grid, kernel_table.count_chunk.function,
                                                                  substrings_threads_per_block_k, shared_memory_bytes,
                                                                  specs);
        if (occupancy_status.status != status_t::success_k) return occupancy_status;

        size_t const target_threads = sz_max_of_two((size_t)blocks_per_grid * substrings_threads_per_block_k,
                                                    (size_t)1);
        chunk_bytes = sz_max_of_two(divide_round_up(total_bytes, target_threads), (size_t)1);

        // The kernel's 32-bit deltas span one thread's warm-up prefix plus its chunk. This bounds that reach,
        // not the input - haystack offsets themselves stay 64-bit.
        view_t const automaton = dictionary_.view();
        size_t const warm_up_bytes = automaton.max_match_bytes > 0 ? (size_t)automaton.max_match_bytes - 1 : 0;
        if (chunk_bytes + warm_up_bytes > (size_t)std::numeric_limits<small_size_t>::max())
            return {status_t::overflow_risk_k, cudaSuccess};

        // A chunk's match count rides the same narrow type, worst case being a repeated byte against a
        // nested-suffix dictionary, where every position emits `max_outputs_per_state` merged outputs.
        size_t const worst_case_matches_per_chunk = chunk_bytes * (size_t)automaton.max_outputs_per_state;
        if (automaton.max_outputs_per_state != 0 &&
            (worst_case_matches_per_chunk / (size_t)automaton.max_outputs_per_state != chunk_bytes ||
             worst_case_matches_per_chunk > (size_t)std::numeric_limits<small_size_t>::max()))
            return {status_t::overflow_risk_k, cudaSuccess};

        if (haystack_chunk_offsets_.try_resize_uninitialized(haystack_count + 1) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        span<span<byte_t const> const> haystacks_argument {haystack_descriptors_.data(), haystack_descriptors_.size()};
        size_t chunk_bytes_argument = chunk_bytes;
        span<size_t> chunks_per_haystack_argument {haystack_chunk_offsets_.data(), haystack_count};
        void *chunks_per_haystack_arguments[3] = {&haystacks_argument, &chunk_bytes_argument,
                                                  &chunks_per_haystack_argument};
        CUresult const chunks_error = cuda_launch_t {}
                                          .grid(blocks_per_grid)
                                          .block(substrings_threads_per_block_k)
                                          .shared(0)
                                          .stream(executor.stream())
                                          .launch(kernel_table.chunks_per_haystack.function,
                                                  chunks_per_haystack_arguments);
        if (chunks_error != CUDA_SUCCESS) return make_cuda_status(chunks_error);

        cuda_status_t const scan_status = cuda_launch_exclusive_sum_(kernel_table.exclusive_sum,
                                                                     haystack_chunk_offsets_.data(), haystack_count,
                                                                     haystack_chunk_offsets_.data(), executor.stream());
        if (scan_status.status != status_t::success_k) return scan_status;

        CUresult const sync_error = timer_.synchronize(executor.stream());
        if (sync_error != CUDA_SUCCESS) return make_cuda_status(sync_error);
        chunk_count = haystack_chunk_offsets_[haystack_count];
        return {status_t::success_k, cudaSuccess};
    }

    /** @brief Runs the counting pass, then the in-place exclusive scan, so `chunk_match_offsets_` holds every
     *         chunk's write offset with the grand total trailing at `[chunk_count]` - the shared core of
     *         `try_count` and `try_find`. */
    cuda_status_t count_into_offsets_(cuda_executor_t const &executor, kernels_t const &kernel_table,
                                      unsigned shared_memory_bytes, unsigned blocks_per_grid, size_t chunk_bytes,
                                      size_t chunk_count) noexcept {
        if (chunk_match_offsets_.try_resize_uninitialized(chunk_count + 1) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        view_t view_argument = device_view_();
        state_id_t shared_rows_argument = shared_rows_;
        span<u32_t const> accepts_words_argument {device_accepts_words_.data(), device_accepts_words_.size()};
        u32_t staged_accepts_words_argument = staged_accepts_words_;
        span<span<byte_t const> const> haystacks_argument {haystack_descriptors_.data(), haystack_descriptors_.size()};
        span<size_t const> haystack_chunk_offsets_argument {haystack_chunk_offsets_.data(),
                                                            haystack_chunk_offsets_.size()};
        size_t chunk_bytes_argument = chunk_bytes;
        size_t chunk_count_argument = chunk_count;
        span<size_t> chunk_match_slots_argument {chunk_match_offsets_.data(), chunk_match_offsets_.size()};
        span<substrings_match_t> no_matches_argument;
        void *count_arguments[10] = {&view_argument,
                                     &shared_rows_argument,
                                     &accepts_words_argument,
                                     &staged_accepts_words_argument,
                                     &haystacks_argument,
                                     &haystack_chunk_offsets_argument,
                                     &chunk_bytes_argument,
                                     &chunk_count_argument,
                                     &chunk_match_slots_argument,
                                     &no_matches_argument};
        CUresult const count_error = cuda_launch_t {}
                                         .grid(blocks_per_grid)
                                         .block(substrings_threads_per_block_k)
                                         .shared(shared_memory_bytes)
                                         .stream(executor.stream())
                                         .launch(kernel_table.count_chunk.function, count_arguments);
        if (count_error != CUDA_SUCCESS) return make_cuda_status(count_error);

        // In place: each chunk's slot holds its raw count going in and its exclusive offset coming out - the
        // scan kernel reads `input[i]` into a register before any thread writes `output[i]`, so reusing one
        // buffer for both is safe and skips a second chunk_count-sized allocation.
        return cuda_launch_exclusive_sum_(kernel_table.exclusive_sum, chunk_match_offsets_.data(), chunk_count,
                                          chunk_match_offsets_.data(), executor.stream());
    }

  public:
    /**
     *  @brief Occurrences of all needles in each of the @p haystacks, for filtering and ranking.
     *  @param[in] haystacks Device-accessible, contiguously laid out; no needle is ever reported straddling
     *             two of them.
     *  @param[out] matches_total Sum of @p counts_per_haystack, which is what sizes a later `try_find` buffer.
     *
     *  Strictly cheaper than `try_find`: the plan and the counting pass, and no scatter. Chunks never cross a
     *  haystack boundary, so the per-haystack breakdown is a subtraction over the counting pass's offsets.
     */
    template <typename haystacks_type_>
    cuda_status_t try_count(haystacks_type_ const &haystacks, span<size_t> counts_per_haystack, size_t &matches_total,
                            cuda_executor_t const &executor = {}, gpu_specs_t specs = {}) noexcept {
        matches_total = 0;
        sz_assert_(counts_per_haystack.size() == haystacks.size());
        for (size_t index = 0; index < counts_per_haystack.size(); ++index) counts_per_haystack[index] = 0;

        size_t total_bytes = 0;
        cuda_status_t const describe_status = describe_haystacks_(haystacks, total_bytes);
        if (describe_status.status != status_t::success_k) return describe_status;

        planned_pass_t pass;
        cuda_status_t const pass_status = plan_and_count_(total_bytes, executor, specs, pass);
        if (pass_status.status != status_t::success_k || !pass.has_work) return pass_status;

        CUresult const stop_error = timer_.record_stop(executor.stream());
        if (stop_error != CUDA_SUCCESS) return make_cuda_status(stop_error);
        CUresult const sync_error = timer_.synchronize(executor.stream());
        if (sync_error != CUDA_SUCCESS) return make_cuda_status(sync_error);

        matches_total = chunk_match_offsets_[pass.chunk_count];
        for (size_t haystack_index = 0; haystack_index < haystacks.size(); ++haystack_index)
            counts_per_haystack[haystack_index] = chunk_match_offsets_[haystack_chunk_offsets_[haystack_index + 1]] -
                                                  chunk_match_offsets_[haystack_chunk_offsets_[haystack_index]];
        return {status_t::success_k, cudaSuccess, CUDA_SUCCESS, timer_.elapsed_milliseconds()};
    }

    /**
     *  @brief Finds all occurrences of all needles in all the @p haystacks, via two real device passes: count
     *         then scatter, writing every match at its chunk's precomputed offset without atomics.
     *  @param[out] matches_found Matches written, in ascending haystack order.
     *  @retval `status_t::unexpected_dimensions_k` @p matches_out is too small; nothing is written in that
     *          case. See `try_count` for the @p haystacks contract.
     */
    template <typename haystacks_type_>
    cuda_status_t try_find(haystacks_type_ const &haystacks, span<match_t> matches_out, size_t &matches_found,
                           cuda_executor_t const &executor = {}, gpu_specs_t specs = {}) noexcept {
        size_t const matches_capacity = matches_out.size();
        matches_found = 0;

        size_t total_bytes = 0;
        cuda_status_t const describe_status = describe_haystacks_(haystacks, total_bytes);
        if (describe_status.status != status_t::success_k) return describe_status;

        planned_pass_t pass;
        cuda_status_t const pass_status = plan_and_count_(total_bytes, executor, specs, pass);
        if (pass_status.status != status_t::success_k || !pass.has_work) return pass_status;
        auto const &kernel_table = pass.kernel_table;
        unsigned const shared_memory_bytes = pass.shared_memory_bytes, blocks_per_grid = pass.blocks_per_grid;
        size_t const chunk_bytes = pass.chunk_bytes, chunk_count = pass.chunk_count;

        // The grand total is only known once the scan has run, and only host-visible after a fence - this is
        // the one synchronous stall `try_find` cannot avoid, since the capacity check below must precede the
        // scatter launch.
        CUresult const mid_sync_error = timer_.synchronize(executor.stream());
        if (mid_sync_error != CUDA_SUCCESS) return make_cuda_status(mid_sync_error);
        size_t const matches_in_batch = chunk_match_offsets_[chunk_count];
        if (matches_in_batch > matches_capacity) return {status_t::unexpected_dimensions_k, cudaSuccess};

        // A unified output buffer stays zero-copy; host memory a kernel can't reach is staged and drained
        // with one copy after the sync.
        bool const output_is_device_accessible = matches_in_batch == 0 ||
                                                 is_device_accessible_memory((void const *)matches_out.data());
        span<match_t> scatter_target {matches_out.data(), matches_in_batch};
        if (!output_is_device_accessible) {
            if (matches_staging_.try_resize_uninitialized(matches_in_batch) != status_t::success_k)
                return {status_t::bad_alloc_k, cudaSuccess};
            scatter_target = {matches_staging_.data(), matches_in_batch};
        }

        view_t view_argument = device_view_();
        state_id_t shared_rows_argument = shared_rows_;
        span<u32_t const> accepts_words_argument {device_accepts_words_.data(), device_accepts_words_.size()};
        u32_t staged_accepts_words_argument = staged_accepts_words_;
        span<span<byte_t const> const> haystacks_argument {haystack_descriptors_.data(), haystack_descriptors_.size()};
        span<size_t const> haystack_chunk_offsets_argument {haystack_chunk_offsets_.data(),
                                                            haystack_chunk_offsets_.size()};
        size_t chunk_bytes_argument = chunk_bytes;
        size_t chunk_count_argument = chunk_count;
        span<size_t> chunk_match_slots_argument {chunk_match_offsets_.data(), chunk_match_offsets_.size()};
        // Bounded to what the counting pass actually found, not to the caller's capacity, so a debug index
        // assert inside the kernel catches an over-write rather than merely staying inside the allocation.
        span<match_t> matches_out_argument = scatter_target;
        void *scatter_arguments[10] = {&view_argument,
                                       &shared_rows_argument,
                                       &accepts_words_argument,
                                       &staged_accepts_words_argument,
                                       &haystacks_argument,
                                       &haystack_chunk_offsets_argument,
                                       &chunk_bytes_argument,
                                       &chunk_count_argument,
                                       &chunk_match_slots_argument,
                                       &matches_out_argument};
        CUresult const scatter_error = cuda_launch_t {}
                                           .grid(blocks_per_grid)
                                           .block(substrings_threads_per_block_k)
                                           .shared(shared_memory_bytes)
                                           .stream(executor.stream())
                                           .launch(kernel_table.scatter_chunk.function, scatter_arguments);
        if (scatter_error != CUDA_SUCCESS) return make_cuda_status(scatter_error);

        CUresult const stop_error = timer_.record_stop(executor.stream());
        if (stop_error != CUDA_SUCCESS) return make_cuda_status(stop_error);

        // The drain is stream-ordered after the scatter and after the timer's stop event, so the kernel time
        // excludes it and the one synchronize below covers it - the same driver-copy idiom as the strided
        // `cuMemcpy2DAsync` draining `results_staging_` in the similarities engines.
        if (!output_is_device_accessible) {
            CUresult const drain_error = cuMemcpyAsync(
                (CUdeviceptr)matches_out.data(), (CUdeviceptr)matches_staging_.data(),
                matches_in_batch * sizeof(substrings_match_t), executor.stream());
            if (drain_error != CUDA_SUCCESS) return make_cuda_status(drain_error);
        }

        CUresult const sync_error = timer_.synchronize(executor.stream());
        if (sync_error != CUDA_SUCCESS) return make_cuda_status(sync_error);

        matches_found = matches_in_batch;
        return {status_t::success_k, cudaSuccess, CUDA_SUCCESS, timer_.elapsed_milliseconds()};
    }
};

using substrings_u16_cuda_t = substrings_cuda<u16_t, unified_alloc_t, sz_cap_cuda_k>;
using substrings_u32_cuda_t = substrings_cuda<u32_t, unified_alloc_t, sz_cap_cuda_k>;

#pragma endregion Engine

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SUBSTRINGS_CUDA_CUH_
