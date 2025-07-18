/**
 *  @brief  Hardware-accelerated multi-pattern exact substring search on CUDA-capable GPUs.
 *  @file   find_many.cuh
 *  @author Ash Vardanian
 *
 *  @section External Memory
 *
 *  When performing multi-pattern search, we assume that the set of needles must fit in VRAM (~ 50 GB),
 *  may fit into the Shared Memory (~ 50 MB), and, in rare cases, may fit into the Constant Memory (~ 50 KB).
 *  The haystacks, however, may be huge in size and can be fetched from external memory (e.g., NVMe SSDs).
 *
 *  That means we may be inclined to compress the FSM into a smaller representation, so that it can fit into
 *  the Shared Memory (as constant memory is too slow), but we will then likely increase the number of individual
 *  loads... and the problem will resurface again.
 *
 *  @see How slow is constant memory? https://leimao.github.io/blog/CUDA-Constant-Memory/
 *
 *  @section Future Development
 *
 *  Current kernels oversimplify scheduling. They assume either a huge number of shorter haystacks or just a few
 *  really long ones. Reality is often fuzzier, with a mix of both. A better scheduling approach may be to treat
 *  all haystacks as a single tape, regrouping into sub-haystack-level and haystack-level "tasks".
 */
#ifndef STRINGZILLAS_FIND_MANY_CUH_
#define STRINGZILLAS_FIND_MANY_CUH_

#include "stringzillas/types.cuh"
#include "stringzillas/find_many.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/atomic>

namespace ashvardanian {
namespace stringzillas {

#pragma region - General Purpose CUDA Backend

/**
 *  @brief A more generic alternative to `__reduce_add_sync`.
 */
template <typename scalar_type_>
__forceinline__ __device__ scalar_type_ _reduce_in_warp(scalar_type_ x) noexcept {
    // The `__shfl_down_sync` replaces `__shfl_down`
    // https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
    x += __shfl_down_sync(0xffffffff, x, 16);
    x += __shfl_down_sync(0xffffffff, x, 8);
    x += __shfl_down_sync(0xffffffff, x, 4);
    x += __shfl_down_sync(0xffffffff, x, 2);
    x += __shfl_down_sync(0xffffffff, x, 1);
    return x;
}

/**
 *  @brief Wraps a single task for the CUDA-based @b byte-level multi-needle "count" kernels.
 *  @note Used to allow sorting/grouping inputs to differentiate device-wide and warp-wide tasks.
 */
struct cuda_count_many_task_t {
    span<byte_t const> haystack {};
    size_t task_index {0};
    size_t result {0};
};

/**
 *  @brief Wraps a single task for the CUDA-based @b byte-level multi-needle "find" kernels.
 *  @note Used to allow sorting/grouping inputs to differentiate device-wide and warp-wide tasks.
 */
struct cuda_find_task_t {
    span<byte_t const> haystack;
    size_t task_index {0};
    span<size_t> result_needle_ids {};
    span<size_t> result_offsets {};
    span<size_t> result_lengths {};
};

/**
 *  Each warp receives a unique haystack. All threads in a warp take continuous overlapping slices of the haystack.
 *  Overlapping match counts are reported and later aggregated in the calling function, accounting for the overlaps.
 *  It's expected, that the length of the longest needle is smaller than the length of a haystack slice.
 *
 *  @tparam small_size_type_ Helps us avoid 64-bit arithmetic in favor of smaller 16- or 32-bit offsets/lengths.
 */
template <typename small_size_type_, typename state_id_type_>
__device__ _count_short_needle_matches_in_one_part_t _count_short_needle_matches_in_one_part_per_warp_thread( //
    span<byte_t const> const &haystack,                                                                       //
    aho_corasick_dictionary_view<state_id_type_> const &dict,                                                 //
    size_t const thread_index, size_t const thread_pool_size) noexcept {

    using state_id_t = state_id_type_;
    using small_size_t = small_size_type_;

    small_size_t const bytes_per_thread_optimal = divide_round_up(haystack.size(), thread_pool_size);

    // We may have a case of a thread receiving no data at all
    byte_t const *optimal_start = std::min(haystack.data() + thread_index * bytes_per_thread_optimal, haystack.end());
    byte_t const *const prefix_end = std::min(optimal_start + dict.max_needle_length, haystack.end());
    byte_t const *const overlapping_end =
        std::min(optimal_start + bytes_per_thread_optimal + dict.max_needle_length, haystack.end());

    // Reimplement the serial `aho_corasick_dictionary::count` keeping track of the matches,
    // entirely fitting in the prefix
    state_id_t current_state = 0;
    small_size_t result_total = 0;
    small_size_t result_prefix = 0;
    for (; optimal_start != overlapping_end; ++optimal_start) {
        current_state = dict.transitions[current_state][*optimal_start];
        small_size_t const outputs_count = static_cast<small_size_t>(dict.outputs_counts[current_state]);
        result_total += outputs_count;
        result_prefix += non_zero_if<small_size_t>(outputs_count, optimal_start < prefix_end);
    }

    // Re-package into larger output types:
    _count_short_needle_matches_in_one_part_t result;
    result.total = result_total;
    result.prefix = result_prefix;
    return result;
}

/**
 *  Each warp receives a unique haystack. All threads in a warp take continuous @b heavily overlapping slices of the
 *  haystack. The length of the longest needle may be larger than many slices combined, or match the entire haystack.
 *
 *  @note In the worst case this kernel is highly inefficient and should be reconsidered in the future.
 *  There are countless divergent branches in this solution, that depending on the vocabulary can result
 *  in extremely low performance.
 */
template <typename small_size_type_, typename state_id_type_>
__device__ small_size_type_ _count_needle_matches_in_one_part_per_warp_thread( //
    span<byte_t const> const &haystack,                                        //
    aho_corasick_dictionary_view<state_id_type_> const &dict,                  //
    size_t const thread_index, size_t const thread_pool_size) noexcept {

    using state_id_t = state_id_type_;
    using small_size_t = small_size_type_;

    byte_t const *const haystack_data = haystack.data();
    small_size_t const haystack_bytes_length = static_cast<small_size_t>(haystack.size());
    byte_t const *const haystack_end = haystack_data + haystack_bytes_length;

    small_size_t const bytes_per_thread_optimal =
        divide_round_up<small_size_t>(haystack_bytes_length, thread_pool_size);

    // We may have a case of a thread receiving no data at all
    byte_t const *optimal_start = std::min(haystack_data + thread_index * bytes_per_thread_optimal, haystack_end);
    byte_t const *const optimal_end = std::min(optimal_start + bytes_per_thread_optimal, haystack_end);
    byte_t const *const overlapping_end =
        std::min(optimal_start + bytes_per_thread_optimal + dict.max_needle_length, haystack_end);

    // Reimplement the serial `aho_corasick_dictionary::count` keeping track of the matches,
    // entirely fitting in the prefix
    state_id_t current_state = 0;
    small_size_t result_total = 0;
    for (; optimal_start != overlapping_end; ++optimal_start) {
        current_state = dict.transitions[current_state][*optimal_start];
        small_size_t const outputs_count = static_cast<small_size_t>(dict.outputs_counts[current_state]);
        if (outputs_count == 0) continue;

        // In a small & diverse vocabulary, the following loop generally does just 1 iteration
        size_t const outputs_offset = dict.outputs_offsets[current_state];
        for (size_t output_index = 0; output_index < outputs_count; ++output_index) {
            size_t needle_id = dict.outputs[outputs_offset + output_index];
            size_t match_length = dict.needles_lengths[needle_id];
            byte_t const *match_ptr = optimal_start + 1 - match_length;
            result_total += match_ptr < optimal_end;
        }
    }

    return result_total;
}

/**
 *  @brief  Multi-pattern exact substring search on CUDA-capable GPUs, assigning just one warp per haystack.
 *
 *  Nothing smart here. Each warp takes its own haystack from @p `haystacks`.
 *  Different threads in a warp take different continuous slices of a shared haystack.
 *  This works best for fairly short needles and a large quantity of haystacks.
 */
template <typename state_id_type_, sz_capability_t capability_ = sz_cap_cuda_k>
__global__ void _count_matches_with_haystack_per_warp( //
    span<cuda_count_many_task_t> tasks, aho_corasick_dictionary_view<state_id_type_> dict) {

    // We only use this kernel for small haystacks, where a smaller integer type is enough for size.
    using small_size_t = uint;
    using state_id_t = state_id_type_;

    // We may have multiple warps operating in the same block.
    uint const warp_size = warpSize;
    size_t const global_thread_index = static_cast<uint>(blockIdx.x * blockDim.x + threadIdx.x);
    size_t const global_warp_index = static_cast<uint>(global_thread_index / warp_size);
    size_t const warps_per_block = static_cast<uint>(blockDim.x / warp_size);
    size_t const warps_per_device = static_cast<uint>(gridDim.x * warps_per_block);
    uint const warp_thread_index = static_cast<uint>(global_thread_index % warp_size);

    for (size_t task_index = global_warp_index; task_index < tasks.size(); task_index += warps_per_device) {
        // Each warp is assigned to a single task.
        auto &task = tasks[task_index];
        span<byte_t const> const &haystack = task.haystack;
        small_size_t const haystack_length = static_cast<small_size_t>(haystack.size());

        // We shouldn't even consider needles longer than the haystack
        small_size_t const max_needle_length =
            std::min(static_cast<small_size_t>(dict.max_needle_length), haystack_length);
        bool const longest_needle_fits_on_one_thread = max_needle_length * warp_size < haystack_length;
        small_size_t results_per_thread = 0;
        if (longest_needle_fits_on_one_thread) {
            _count_short_needle_matches_in_one_part_t partial_result =
                _count_short_needle_matches_in_one_part_per_warp_thread<small_size_t, state_id_t>( //
                    haystack, dict, warp_thread_index, warp_size);
            results_per_thread =
                partial_result.total - non_zero_if<small_size_t>(partial_result.prefix, warp_thread_index != 0);
        }
        else {
            results_per_thread = _count_needle_matches_in_one_part_per_warp_thread<small_size_t, state_id_t>( //
                haystack, dict, warp_thread_index, warp_size);
        }

        small_size_t results_across_warp = _reduce_in_warp(results_per_thread);
        if (warp_thread_index == 0) task.result = results_across_warp;
    }
}

/**
 *  @brief  Multi-pattern exact substring search on CUDA-capable GPUs, assigning the entire device to one haystack.
 *
 *  Nothing smart here. Each warp takes its own part of a single @p `haystack`.
 *  Different threads in a warp take different continuous slices of a shared haystack.
 *  This works best for fairly short needles and a large quantity of haystacks.
 */
template < //
    typename state_id_type_,
    sz_capability_t capability_ = sz_cap_cuda_k //
    >
__global__ void _count_matches_with_haystack_per_device( //
    span<byte_t const> haystack, aho_corasick_dictionary_view<state_id_type_> dict, size_t *count_for_haystack_ptr) {

    // We only use this kernel for small haystacks, where a smaller integer type is enough for size.
    using state_id_t = state_id_type_;

    // We may have multiple warps operating in the same block.
    size_t const warp_size = warpSize;
    size_t const global_thread_index = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    size_t const threads_per_device = static_cast<size_t>(blockDim.x * gridDim.x);
    size_t const warp_thread_index = static_cast<size_t>(global_thread_index % warp_size);

    size_t const haystack_length = haystack.size();
    size_t const chars_per_core_optimal = divide_round_up<size_t>(haystack_length, threads_per_device);

    // We shouldn't even consider needles longer than the haystack
    size_t const max_needle_length = std::min(static_cast<size_t>(dict.max_needle_length), haystack_length);
    bool const longest_needle_fits_on_one_thread = max_needle_length * threads_per_device < haystack_length;
    size_t results_per_thread = 0;
    if (longest_needle_fits_on_one_thread) {
        _count_short_needle_matches_in_one_part_t partial_result =
            _count_short_needle_matches_in_one_part_per_warp_thread<size_t, state_id_t>( //
                haystack, dict, global_thread_index, threads_per_device);
        results_per_thread =
            partial_result.total - non_zero_if<size_t>(partial_result.prefix, global_thread_index != 0);
    }
    else {
        results_per_thread = _count_needle_matches_in_one_part_per_warp_thread<size_t, state_id_t>( //
            haystack, dict, global_thread_index, threads_per_device);
    }

    // Instead of the efficient tree-like shared-memory reductions with subsequent writes, we simply use atomic
    // references to global memory. Benchmarks suggest that modern GPUs are great at pipelining relaxed increments.
    // To slightly reduce the traffic, we can aggregate within the warp first.
    small_size_t results_across_warp = _reduce_in_warp(results_per_thread);
    if (warp_thread_index == 0) {
        cuda::atomic_ref<size_t> count_for_haystack(*count_for_haystack_ptr);
        count_for_haystack.fetch_add(results_across_warp, cuda::std::memory_order_relaxed);
    }
}

/**
 *  @brief  Multi-pattern exact substring search on CUDA-capable GPUs, assigning just one warp per haystack.
 *
 *  The serial Aho-Corasick algorithm's super-power is looking at each symbol of the haystack just once.
 *  If we have a warp of @b (WS=32) threads, we have several strategies to enumerate the haystack:
 *
 *  - Simple algorithm: split each haystack into WS continuous parts and assign each part to a thread.
 *    That works great until the length of the longest needle is much smaller than the (haystack.size() / WS).
 *  - Advanced algorithm: WS threads are walking through the haystack 2xWS symbols at a time, combining SIMT and
 *    SIMD-style processing.
 *
 *  The problem with the "simple" solution is - imagine a haystack of 1 MB and a collection of 100 short needles
 *  and just 1 long needle almost 1 MB in size. In the worst-case scenario, the first of WS=32 threads will immediately
 *  start matching the longest needle. The (WS-1=31) will finish early, while 1 thread will have a WS longer runtime.
 *  Assuming all the WS threads share a scheduler, our algorithm will be at least (WS-1) times slower than it can be.
 *
 *  The problem with the "advanced" solution is - with frequent failure links reaching back to the root, the threads
 *  within the warp will be effectively observing the same paths once they receive the next character. So despite being
 *  much more hardware-friendly with only sequential coalesced memory access, it directly harms the AC algorithm logic.
 *
 *  A hybrid, however, may be interesting! We can process the majority of the content in a "simple" fashion, with each
 *  thread taking care of its own chunk privately, afterwards using the "advanced" solution to process overlaps and
 *  filter them by needle length?
 *
 *  The most common case is having more input characters per thread in warp than the length of the longest needle.
 *  In that case we can avoid the complex
 */

/**
 *  @brief Aho-Corasick-based @b SIMT multi-pattern exact substring search.
 *  @tparam state_id_type_ The type of the state ID. Default is `sz_u32_t`.
 *  @tparam allocator_type_ The type of the allocator. Default is `dummy_alloc_t`.
 *  @tparam capability_ The capability of the dictionary. Default is `sz_cap_serial_k`.
 */
template <typename state_id_type_, typename allocator_type_, typename enable_>
struct find_many<state_id_type_, allocator_type_, sz_cap_cuda_k, enable_> {

    using dictionary_t = aho_corasick_dictionary<state_id_type_, allocator_type_>;
    using state_id_t = typename dictionary_t::state_id_t;
    using allocator_t = typename dictionary_t::allocator_t;
    using match_t = typename dictionary_t::match_t;
    using state_transitions_t = typename dictionary_t::state_transitions_t;

    static constexpr state_id_t alphabet_size_k = dictionary_t::alphabet_size_k;

    find_many(allocator_t alloc = allocator_t()) noexcept : dict_(alloc) {}
    void reset() noexcept { dict_.reset(); }
    dictionary_t const &dictionary() const noexcept { return dict_; }

    template <typename other_allocator_type_>
    status_t try_build(aho_corasick_dictionary<state_id_t, other_allocator_type_> const &other) noexcept {
        return dict_.try_assign(other);
    }

    /**
     *  @brief Indexes all of the @p needles strings into the FSM.
     *  @retval `status_t::success_k` The needle was successfully added.
     *  @retval `status_t::bad_alloc_k` Memory allocation failed.
     *  @retval `status_t::overflow_risk_k` Too many needles for the current state ID type.
     *  @retval `status_t::contains_duplicates_k` The needle is already in the vocabulary.
     *  @note Before reusing, please `reset` the FSM.
     */
    template <typename needles_type_>
    status_t try_build(needles_type_ &&needles) noexcept {
        for (auto const &needle : needles)
            if (status_t status = dict_.try_insert(needle); status != status_t::success_k) return status;
        return dict_.try_build();
    }

    /**
     *  @brief Counts the number of occurrences of all needles in all @p haystacks. Relevant for filtering and ranking.
     *  @param[in] haystacks The input strings to search in.
     *  @param[in] counts The output buffer for the counts of all needles in each haystack.
     *  @return The total number of occurrences found.
     */
    template <typename haystacks_type_>
    cuda_status_t try_count(                              //
        haystacks_type_ &&haystacks, span<size_t> counts, //
        cuda_executor_t executor = {}, gpu_specs_t const &specs = {}) const noexcept {

        sz_assert_(counts.size() == haystacks.size());

        using haystacks_t = typename std::remove_reference_t<haystacks_type_>;
        using haystack_t = typename haystacks_t::value_type;
        using char_t = typename haystack_t::value_type;
        static_assert(std::is_nothrow_copy_constructible_v<haystacks_t>,
                      "Haystack type must be nothrow copy constructible");

        // Preallocate the events for GPU timing.
        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event, cudaEventBlockingSync);
        cudaEventCreate(&stop_event, cudaEventBlockingSync);

        // Record the start event
        cudaError_t start_event_error = cudaEventRecord(start_event, executor.stream);
        if (start_event_error != cudaSuccess) return {status_t::unknown_k, start_event_error};

        // Allocate GPU memory buffer using safe_vector
        using task_t = cuda_count_many_task_t;
        using task_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<task_t>;
        safe_vector<task_t, task_allocator_t> tasks_buffer(dict_.allocator());
        if (tasks_buffer.try_resize(counts.size()) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaErrorMemoryAllocation};

        // Populate the tasks buffer with haystacks
        for (size_t i = 0; i < haystacks.size(); ++i) {
            auto &haystack = haystacks[i];
            auto haystack_bytes = span<char_t const>(haystack.data(), haystack.size()).template cast<byte_t const>();
            tasks_buffer[i].haystack = haystack_bytes;
            tasks_buffer[i].task_index = i;
            tasks_buffer[i].result = 0; // Initialize result to zero
        }

        // Sort the tasks by size of the haystack and isolate the ones that should be processed across the device.
        std::sort(tasks_buffer.begin(), tasks_buffer.end(),
                  [](task_t const &a, task_t const &b) { return a.haystack.size() < b.haystack.size(); });

        // Calculate optimal thread and block configuration
        uint const threads_per_block = specs.warp_size * 4;               // 4 warps per block
        uint const blocks_per_grid = specs.streaming_multiprocessors * 2; // 2 blocks per SM
        uint const threads_per_device = blocks_per_grid * threads_per_block;
        uint const warps_per_device = threads_per_device / specs.warp_size;

        // Our warp-wide matchers are more efficient if we have enough haystacks to saturate the device.
        // The weird corner case is having many short haystacks and just a couple of very long ones.
        // TODO: Processing such inputs would be extremely inefficient.
        size_t const min_length_for_device_wide_processing =
            round_up_to_multiple<size_t>(dict_.max_needle_length(), 128) * threads_per_device;
        size_t const haystacks_with_device_wide_processing = std::count_if(
            tasks_buffer.begin(), tasks_buffer.end(),
            [&](task_t const &task) { return task.haystack.size() >= min_length_for_device_wide_processing; });

        // We can't move the dictionary to the GPU, but we can pass a view
        aho_corasick_dictionary_view<state_id_t> dict_view(dict_);

        // Launch the kernel for warp-wide processing of haystacks.
        auto kernel = &_count_matches_with_haystack_per_warp<state_id_t, sz_cap_cuda_k>;
        kernel<<<blocks_per_grid, threads_per_block, 0, executor.stream>>>(
            {tasks_buffer.data(), tasks_buffer.size() - haystacks_with_device_wide_processing}, dict_view);

        // Handle the last haystacks that are too long for warp-wide processing.
        for (size_t i = tasks_buffer.size() - haystacks_with_device_wide_processing; i < tasks_buffer.size(); ++i) {
            auto &task = tasks_buffer[i];
            auto device_kernel = &_count_matches_with_haystack_per_device<state_id_t, sz_cap_cuda_k>;

            // Launch the device-wide kernel for this large haystack, passing pointer to task result
            device_kernel<<<blocks_per_grid, threads_per_block, 0, executor.stream>>>(task.haystack, dict_view,
                                                                                      &task.result);
        }

        // Check for kernel launch errors
        cudaError_t launch_error = cudaGetLastError();
        if (launch_error != cudaSuccess) return {status_t::unknown_k, launch_error};

        // Wait for kernel completion
        cudaError_t execution_error = cudaStreamSynchronize(executor.stream);
        if (execution_error != cudaSuccess) return {status_t::unknown_k, execution_error};

        // Copy results back to host - extract results from tasks and put them in the correct order
        for (size_t i = 0; i < tasks_buffer.size(); ++i) {
            auto &task = tasks_buffer[i];
            counts[task.task_index] = task.result;
        }

        // Record stop event and calculate timing
        cudaError_t stop_event_error = cudaEventRecord(stop_event, executor.stream);
        if (stop_event_error != cudaSuccess) return {status_t::unknown_k, stop_event_error};

        float execution_milliseconds = 0;
        cudaEventElapsedTime(&execution_milliseconds, start_event, stop_event);

        // Clean up events
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);

        return {status_t::success_k, cudaSuccess, execution_milliseconds};
    }

    /**
     *  @brief Finds all occurrences of all needles in all the @p haystacks.
     *  @param[in] haystacks The input strings to search in, with support for random access iterators.
     *  @param[in] matches The output buffer for the matches, with support for random access iterators.
     *  @param[out] matches_count The number of matches found.
     *  @return The number of matches found across all the @p haystacks.
     *  @note The @p matches reference objects should be assignable from @b `match_t`.
     */
    template <typename haystacks_type_, typename output_matches_type_>
    status_t try_find(haystacks_type_ &&haystacks, span<size_t const>, output_matches_type_ &&matches,
                      cuda_executor_t executor = {}, gpu_specs_t const &specs = {}) const noexcept {
        size_t count_found = 0, count_allowed = matches.size();
        for (auto it = haystacks.begin(); it != haystacks.end() && count_found != count_allowed; ++it)
            dict_.find(*it, [&](match_t match) {
                match.haystack_index = static_cast<size_t>(it - haystacks.begin());
                matches[count_found] = match;
                count_found++;
                return count_found < count_allowed;
            });
        return status_t::success_k;
    }

  private:
    dictionary_t dict_;
};

#pragma endregion // General Purpose CUDA Backend

using find_many_u32_cuda_t = find_many<u32_t, unified_alloc<char>, sz_cap_cuda_k>;

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_FIND_MANY_CUH_
