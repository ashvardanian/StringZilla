/**
 *  @brief  CUDA-accelerated fingerprinting utilities for string collections.
 *  @file   fingerprints.cuh
 *  @author Ash Vardanian
 *
 *  CUDA specialization of the `floating_rolling_hashers` template for GPU-accelerated count-min-sketching.
 *  Unlike the CPU variants, this implementation focuses on batch-processing of large collections of strings,
 *  assigning warps to process multiple strings in parallel.
 */
#ifndef STRINGZILLAS_FINGERPRINTS_CUH_
#define STRINGZILLAS_FINGERPRINTS_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "stringzillas/types.cuh"
#include "stringzillas/fingerprints.hpp"

namespace ashvardanian {
namespace stringzillas {

#pragma region - CUDA Device Helpers

/**
 *  @brief Wraps a single task for the CUDA-based @b byte-level "fingerprint" kernels.
 *  @note Used to allow sorting/grouping inputs to differentiate device-wide and warp-wide tasks.
 */
template <typename char_type_>
struct cuda_floating_fingerprint_task_ {
    using char_t = char_type_;

    char_t const *text_ptr = nullptr;
    size_t text_length = 0;
    size_t original_index = 0;
    u32_t *min_hashes = nullptr;
    u32_t *min_counts = nullptr;
    warp_tasks_density_t density = warps_working_together_k; // ? Worst case, we have to sync final writes
};

__device__ __forceinline__ f64_t barrett_mod_cuda_(f64_t x, f64_t modulo, f64_t inverse_modulo) noexcept {
    f64_t q = floor(x * inverse_modulo);
    f64_t result = fma(-q, modulo, x);

    if (result < 0.0) result += modulo;
    if (result >= modulo) result -= modulo;
    return result;
}

#pragma endregion - CUDA Device Helpers

#pragma region - CUDA Kernels

/**
 *  Each warp takes in an individual document from @p `tasks` and computes many rolling hashes for it.
 *  Each thread computes an independent rolling hash for a specific dimension, so you should have a multiple
 *  of warp-size dimensions per fingerprint.
 *
 *  Unlike the `basic_rolling_hashers_kernel_` basic variant, all hashers have the same window width.
 *  This greatly simplifies the memory access patterns. Assuming each thread in a warp can issue an independent
 *  read for consecutive elements, and easily loads 32 bits at a time, this kernel is suited to loading
 *  4x bytes and computing 4x (warp_size_) rolling hashes per thread in the inner loop.
 */
template <                                                                     //
    unsigned window_width_, unsigned dimensions_, sz_capability_t capability_, //
    typename char_type_ = byte_t, warp_size_t warp_size_ = warp_size_nvidia_k, //
    warp_tasks_density_t density_ = four_warps_per_multiprocessor_k            //
    >
__global__ void floating_rolling_hashers_on_each_cuda_warp_(                            //
    cuda_floating_fingerprint_task_<char_type_> const *tasks, size_t const tasks_count, //
    floating_rolling_hasher<f64_t> const *hashers, size_t const hashers_count) {

    //
    using task_t = cuda_floating_fingerprint_task_<char_type_>;
    using hasher_t = floating_rolling_hasher<f64_t>;
    constexpr warp_size_t warp_size_k = warp_size_;
    constexpr warp_tasks_density_t density_k = density_;
    constexpr unsigned window_width_k = window_width_;
    constexpr unsigned dimensions_k = dimensions_;
    constexpr unsigned dimensions_per_thread_k = dimensions_k / warp_size_k;
    constexpr f64_t skipped_rolling_state_k = basic_rolling_hashers<hasher_t>::skipped_rolling_state_k;
    constexpr u32_t max_hash_k = basic_rolling_hashers<hasher_t>::max_hash_k;
    static_assert(dimensions_k % warp_size_k == 0, "Dimensions must be a multiple of warp size");

    // We don't use too much shared memory in these algorithms to allow scaling to very long windows,
    // and large number of blocks per SM. The consecutive aligned reads should be very performant.
    __shared__ byte_t discarding_text_chunk[density_k][warp_size_k];
    __shared__ byte_t incoming_text_chunk[density_k][warp_size_k];

    // We may have multiple warps operating in the same block.
    unsigned const warp_size = warpSize;
    sz_assert_(warp_size == warp_size_k && "Warp size mismatch in kernel");
    unsigned const global_thread_index = static_cast<unsigned>(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned const global_warp_index = static_cast<unsigned>(global_thread_index / warp_size_k);
    unsigned const warps_per_block = static_cast<unsigned>(blockDim.x / warp_size_k);
    sz_assert_(warps_per_block == density_k && "Block size mismatch in kernel");
    unsigned const warps_per_device = static_cast<unsigned>(gridDim.x * warps_per_block);
    unsigned const thread_in_warp_index = static_cast<unsigned>(global_thread_index % warp_size_k);
    unsigned const warp_in_block_index = static_cast<unsigned>(global_warp_index % density_k);

    // Load the hashers states per thread.
    f64_t multipliers[dimensions_per_thread_k];
    f64_t negative_discarding_multipliers[dimensions_per_thread_k];
    f64_t modulos[dimensions_per_thread_k];
    f64_t inverse_modulos[dimensions_per_thread_k];
#pragma unroll
    for (unsigned dim_within_thread = 0; dim_within_thread < dimensions_per_thread_k; ++dim_within_thread) {
        unsigned const dim = thread_in_warp_index * dimensions_per_thread_k + dim_within_thread;
        hasher_t const &hasher = hashers[dim];
        if (dim >= hashers_count) continue; // ? Avoid out-of-bounds access
        multipliers[dim_within_thread] = hasher.multiplier();
        negative_discarding_multipliers[dim_within_thread] = hasher.negative_discarding_multiplier();
        modulos[dim_within_thread] = hasher.modulo();
        inverse_modulos[dim_within_thread] = hasher.inverse_modulo();
    }

    // We are computing N edit distances for N pairs of strings. Not a cartesian product!
    // Each block/warp may end up receiving a different number of strings.
    for (size_t task_index = global_warp_index; task_index < tasks_count; task_index += warps_per_device) {
        task_t const task = tasks[task_index];

        // For each state we need to reset the local state
        f64_t rolling_states[dimensions_per_thread_k];
        f64_t rolling_minimums[dimensions_per_thread_k];
        u32_t rolling_counts[dimensions_per_thread_k];
        for (auto &rolling_state : rolling_states) rolling_state = 0.0;
        for (auto &rolling_minimum : rolling_minimums) rolling_minimum = skipped_rolling_state_k;
        for (auto &rolling_count : rolling_counts) rolling_count = 0;

        // Until we reach the `window_width_k`, we don't need to discard any symbols and can keep the code simpler
        size_t const prefix_length = std::min<size_t>(task.text_length, window_width_k);
        size_t new_char_offset = 0;
        for (; new_char_offset < prefix_length; ++new_char_offset) {
            byte_t const new_char = task.text_ptr[new_char_offset]; // ? Hardware may auto-broadcast this
            f64_t const new_term = static_cast<f64_t>(new_char) + 1.0;

#pragma unroll
            for (unsigned dim_within_thread = 0; dim_within_thread < dimensions_per_thread_k; ++dim_within_thread) {
                f64_t &rolling_state = rolling_states[dim_within_thread];
                f64_t const multiplier = multipliers[dim_within_thread];
                f64_t const modulo = modulos[dim_within_thread];
                f64_t const inverse_modulo = inverse_modulos[dim_within_thread];
                rolling_state = fma(rolling_state, multiplier, new_term);
                rolling_state = barrett_mod_cuda_(rolling_state, modulo, inverse_modulo);
            }
        }

        // We now have our first minimum hashes
        if (new_char_offset == window_width_k) {
#pragma unroll
            for (unsigned dim_within_thread = 0; dim_within_thread < dimensions_per_thread_k; ++dim_within_thread) {
                rolling_minimums[dim_within_thread] = rolling_states[dim_within_thread];
                rolling_counts[dim_within_thread] = 1;
            }
        }

        // Now the main massive unrolled, coalescing reads & writes via `discarding_text_chunk` & `incoming_text_chunk`,
        // practically performing a (`warp_size_k` by `warp_size_k`) hash-calculating operation unrolling the loop
        // nested inside of this one.
        for (; new_char_offset + warp_size_k <= task.text_length; new_char_offset += warp_size_k) {

            // Load the next chunk of characters into shared memory
            byte_t const *incoming_bytes = task.text_ptr + new_char_offset;
            byte_t const *discarding_bytes = task.text_ptr + new_char_offset - window_width_k;
            incoming_text_chunk[warp_in_block_index][thread_in_warp_index] = incoming_bytes[thread_in_warp_index];
            discarding_text_chunk[warp_in_block_index][thread_in_warp_index] = discarding_bytes[thread_in_warp_index];

            // Make sure the shared memory is fully loaded.
            __syncwarp();

#pragma unroll
            for (unsigned char_within_step = 0; char_within_step < warp_size_k; ++char_within_step) {
                byte_t const new_char = incoming_text_chunk[warp_in_block_index][char_within_step];
                byte_t const old_char = discarding_text_chunk[warp_in_block_index][char_within_step];
                f64_t const new_term = static_cast<f64_t>(new_char) + 1.0;
                f64_t const old_term = static_cast<f64_t>(old_char) + 1.0;

#pragma unroll
                for (unsigned dim_within_thread = 0; dim_within_thread < dimensions_per_thread_k; ++dim_within_thread) {
                    f64_t &rolling_state = rolling_states[dim_within_thread];
                    f64_t const multiplier = multipliers[dim_within_thread];
                    f64_t const negative_discarding_multiplier = negative_discarding_multipliers[dim_within_thread];
                    f64_t const modulo = modulos[dim_within_thread];
                    f64_t const inverse_modulo = inverse_modulos[dim_within_thread];
                    rolling_state = fma(negative_discarding_multiplier, old_term, rolling_state);
                    rolling_state = barrett_mod_cuda_(rolling_state, modulo, inverse_modulo);
                    rolling_state = fma(rolling_state, multiplier, new_term);
                    rolling_state = barrett_mod_cuda_(rolling_state, modulo, inverse_modulo);

                    // Update the minimums and counts
                    f64_t &rolling_minimum = rolling_minimums[dim_within_thread];
                    u32_t &min_count = rolling_counts[dim_within_thread];
                    min_count *= rolling_state >= rolling_minimum; // ? Discard `min_count` to 0 for new extremums
                    min_count += rolling_state <= rolling_minimum; // ? Increments by 1 for new & old minimums
                    rolling_minimum = (std::min)(rolling_minimum, rolling_state);
                }
            }
        }

        // Roll until the end of the text
        for (; new_char_offset < task.text_length; ++new_char_offset) {
            byte_t const new_char = task.text_ptr[new_char_offset]; // ? Hardware may auto-broadcast this
            byte_t const old_char = task.text_ptr[new_char_offset - window_width_k];
            f64_t const new_term = static_cast<f64_t>(new_char) + 1.0;
            f64_t const old_term = static_cast<f64_t>(old_char) + 1.0;

#pragma unroll
            for (unsigned dim_within_thread = 0; dim_within_thread < dimensions_per_thread_k; ++dim_within_thread) {
                f64_t &rolling_state = rolling_states[dim_within_thread];
                f64_t const multiplier = multipliers[dim_within_thread];
                f64_t const negative_discarding_multiplier = negative_discarding_multipliers[dim_within_thread];
                f64_t const modulo = modulos[dim_within_thread];
                f64_t const inverse_modulo = inverse_modulos[dim_within_thread];
                rolling_state = fma(negative_discarding_multiplier, old_term, rolling_state);
                rolling_state = barrett_mod_cuda_(rolling_state, modulo, inverse_modulo);
                rolling_state = fma(rolling_state, multiplier, new_term);
                rolling_state = barrett_mod_cuda_(rolling_state, modulo, inverse_modulo);

                // Update the minimums and counts
                f64_t &rolling_minimum = rolling_minimums[dim_within_thread];
                u32_t &min_count = rolling_counts[dim_within_thread];
                min_count *= rolling_state >= rolling_minimum; // ? Discard `min_count` to 0 for new extremums
                min_count += rolling_state <= rolling_minimum; // ? Increments by 1 for new & old minimums
                rolling_minimum = (std::min)(rolling_minimum, rolling_state);
            }
        }

        // Finally export the results
#pragma unroll
        for (unsigned dim_within_thread = 0; dim_within_thread < dimensions_per_thread_k; ++dim_within_thread) {
            unsigned const dim = thread_in_warp_index * dimensions_per_thread_k + dim_within_thread;
            if (dim >= hashers_count) continue; // ? Avoid out-of-bounds access
            task.min_counts[dim] = rolling_counts[dim_within_thread];
            task.min_hashes[dim] =
                rolling_minimums[dim_within_thread] == skipped_rolling_state_k
                    ? max_hash_k
                    : static_cast<u32_t>(static_cast<u64_t>(rolling_minimums[dim_within_thread]) & max_hash_k);
        }
    }
}

/**
 *  Each of @p `tasks` is distributed across the entire device, unlike `floating_rolling_hashers_on_each_cuda_warp_`,
 *  where individual warps take care of separate unrelated inputs. The biggest difference is in how the minimum values
 *  are later reduced across the entire device, rather than per-warp.
 */
template <                                                                 //
    size_t window_width_, size_t dimensions_, sz_capability_t capability_, //
    typename char_type_ = byte_t, size_t warp_size_ = 32                   //
    >
__global__ void floating_rolling_hashers_across_cuda_device_(span<cuda_floating_fingerprint_task_<char_type_>> tasks,
                                                             span<floating_rolling_hasher<f64_t> const> hashers) {
    sz_unused_(tasks);
    sz_unused_(hashers);
}

#pragma endregion - CUDA Kernels

/**
 *  @brief CUDA specialization of floating_rolling_hashers for count-min-sketching.
 */
template <size_t window_width_, size_t dimensions_>
struct floating_rolling_hashers<sz_cap_cuda_k, window_width_, dimensions_> {

    using hasher_t = floating_rolling_hasher<f64_t>;
    using rolling_state_t = f64_t;
    using min_hash_t = u32_t;
    using min_count_t = u32_t;
    using allocator_t = unified_alloc<char>;

    using hashers_allocator_t = typename allocator_t::template rebind<hasher_t>::other;
    using hashers_t = safe_vector<hasher_t, hashers_allocator_t>;

    static constexpr size_t window_width_k = window_width_;
    static constexpr size_t dimensions_k = dimensions_;
    static constexpr rolling_state_t skipped_rolling_state_k = std::numeric_limits<rolling_state_t>::max();
    static constexpr min_hash_t max_hash_k = std::numeric_limits<min_hash_t>::max();

    using min_hashes_span_t = span<min_hash_t, dimensions_k>;
    using min_counts_span_t = span<min_count_t, dimensions_k>;

    static constexpr unsigned hashes_per_warp_k = static_cast<unsigned>(warp_size_nvidia_k);
    static constexpr bool has_incomplete_tail_group_k = dimensions_k % hashes_per_warp_k;
    static constexpr size_t aligned_dimensions_k =
        has_incomplete_tail_group_k ? (dimensions_k / hashes_per_warp_k + 1) * hashes_per_warp_k : (dimensions_k);
    static constexpr unsigned groups_count_k = aligned_dimensions_k / hashes_per_warp_k;

  private:
    allocator_t alloc_;
    hashers_t hashers_;

  public:
    floating_rolling_hashers(allocator_t const &alloc = {}) noexcept : alloc_(alloc), hashers_(alloc) {}
    constexpr size_t dimensions() const noexcept { return dimensions_k; }
    constexpr size_t window_width() const noexcept { return window_width_k; }
    constexpr size_t window_width(size_t) const noexcept { return window_width_k; }

    /**
     *  @brief Initializes several rolling hashers with different multipliers and modulos.
     *  @param[in] alphabet_size Size of the alphabet, typically 256 for UTF-8, 4 for DNA, or 20 for proteins.
     */
    SZ_NOINLINE status_t try_seed(size_t alphabet_size = 256) noexcept {
        if (hashers_.try_resize(aligned_dimensions_k) != status_t::success_k) return status_t::bad_alloc_k;
        for (unsigned dim = 0; dim < dimensions_k; ++dim)
            hashers_[dim] = hasher_t(window_width_k, alphabet_size + dim, hasher_t::default_modulo_base_k);
        return status_t::success_k;
    }

    /**
     *  @brief Convenience function to compute the fingerprint of a single @p `text`-ual document.
     *  @param[in] text The input text to hash, typically a UTF-8 encoded string.
     *  @param[out] min_hashes The output fingerprint, a vector of minimum hashes.
     *  @param[out] min_counts The output frequencies of @p `min_hashes` hashes.
     *  @note Unlike the CPU kernels, @b not intended for product use, but rather for testing.
     */
    SZ_NOINLINE cuda_status_t try_fingerprint(span<byte_t const> text, min_hashes_span_t min_hashes,
                                              min_counts_span_t min_counts, gpu_specs_t specs = {},
                                              cuda_executor_t executor = {}) const noexcept {

        using task_t = cuda_floating_fingerprint_task_<byte_t>;
        using tasks_allocator_t = typename allocator_t::template rebind<task_t>::other;

        // Preallocate the events for GPU timing.
        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event, cudaEventBlockingSync);
        cudaEventCreate(&stop_event, cudaEventBlockingSync);

        // Populate the tasks array with a single task for the entire device.
        safe_vector<task_t, tasks_allocator_t> tasks(alloc_);
        if (tasks.try_resize(1) == status_t::bad_alloc_k) return {status_t::bad_alloc_k};

        tasks[0] = task_t {
            .text_ptr = text.data(),
            .text_length = text.size(),
            .original_index = 0,
            .min_hashes = min_hashes.data(),
            .min_counts = min_counts.data(),
            .density = one_warp_per_multiprocessor_k,
        };

        // Record the start event
        cudaError_t start_event_error = cudaEventRecord(start_event, executor.stream);
        if (start_event_error != cudaSuccess) return {status_t::unknown_k, start_event_error};

        void *warp_level_kernel_args[4];
        auto const *tasks_ptr = tasks.data();
        auto const tasks_size = tasks.size();
        auto const *hashers_ptr = hashers_.data();
        auto const hashers_size = (std::min)(dimensions_k, hashers_.size());
        warp_level_kernel_args[0] = (void *)(&tasks_ptr);
        warp_level_kernel_args[1] = (void *)(&tasks_size);
        warp_level_kernel_args[2] = (void *)(&hashers_ptr);
        warp_level_kernel_args[3] = (void *)(&hashers_size);

        auto warp_level_kernel = &floating_rolling_hashers_on_each_cuda_warp_< //
            window_width_k, aligned_dimensions_k, sz_cap_cuda_k, byte_t,       //
            warp_size_nvidia_k, one_warp_per_multiprocessor_k>;

        // TODO: We can be wiser about the dimensions of this grid.
        unsigned const random_block_size = static_cast<unsigned>(warp_size_nvidia_k) * //
                                           static_cast<unsigned>(one_warp_per_multiprocessor_k);
        unsigned const random_blocks_per_multiprocessor = 1;
        cudaError_t launch_error = cudaLaunchCooperativeKernel( //
            reinterpret_cast<void *>(warp_level_kernel),        // Kernel function pointer
            dim3(random_blocks_per_multiprocessor * 1),         // Grid dimensions
            dim3(random_block_size),                            // Block dimensions
            warp_level_kernel_args,                             // Array of kernel argument pointers
            0,                                                  // Shared memory per block (in bytes)
            executor.stream);                                   // CUDA stream
        if (launch_error != cudaSuccess)
            if (launch_error == cudaErrorMemoryAllocation) { return {status_t::bad_alloc_k, launch_error}; }
            else { return {status_t::unknown_k, launch_error}; }

        // Wait until everything completes, as on the next iteration we will update the properties again.
        cudaError_t execution_error = cudaStreamSynchronize(executor.stream);
        if (execution_error != cudaSuccess) { return {status_t::unknown_k, execution_error}; }

        // Calculate the duration:
        cudaError_t stop_event_error = cudaEventRecord(stop_event, executor.stream);
        if (stop_event_error != cudaSuccess) return {status_t::unknown_k, stop_event_error};
        float execution_milliseconds = 0;
        cudaEventElapsedTime(&execution_milliseconds, start_event, stop_event);

        return {status_t::success_k, cudaSuccess, execution_milliseconds};
    }

    template <typename texts_type_, typename min_hashes_per_text_type_, typename min_counts_per_text_type_>
    SZ_NOINLINE cuda_status_t operator()(texts_type_ const &texts, min_hashes_per_text_type_ &&min_hashes_per_text,
                                         min_counts_per_text_type_ &&min_counts_per_text, gpu_specs_t specs = {},
                                         cuda_executor_t executor = {}) const noexcept {

        using texts_t = texts_type_;
        using text_t = typename texts_t::value_type;
        using char_t = typename text_t::value_type;
        using task_t = cuda_floating_fingerprint_task_<char_t>;
        using tasks_allocator_t = typename allocator_t::template rebind<task_t>::other;

        // Preallocate the events for GPU timing.
        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event, cudaEventBlockingSync);
        cudaEventCreate(&stop_event, cudaEventBlockingSync);

        // Populate the tasks for each warp or the entire device, putting it into unified memory.
        safe_vector<task_t, tasks_allocator_t> tasks(alloc_);
        if (tasks.try_resize(texts.size()) == status_t::bad_alloc_k) return {status_t::bad_alloc_k};
        for (size_t task_index = 0; task_index < texts.size(); ++task_index) {
            auto const &text = texts[task_index];
            auto min_hashes = to_span(min_hashes_per_text[task_index]);
            auto min_counts = to_span(min_counts_per_text[task_index]);
            tasks[task_index] = task_t {
                .text_ptr = text.data(),
                .text_length = text.size(),
                .original_index = task_index,
                .min_hashes = min_hashes.data(),
                .min_counts = min_counts.data(),
                .density = four_warps_per_multiprocessor_k,
            };
        }
        // std::partition(tasks.begin(), tasks.end(),
        //                [](task_t const &task) { return task.density == warps_working_together_k; });

        // Record the start event
        cudaError_t start_event_error = cudaEventRecord(start_event, executor.stream);
        if (start_event_error != cudaSuccess) return {status_t::unknown_k, start_event_error};

        void *warp_level_kernel_args[4];
        auto const *tasks_ptr = tasks.data();
        auto const tasks_size = tasks.size();
        auto const *hashers_ptr = hashers_.data();
        auto const hashers_size = (std::min)(dimensions_k, hashers_.size());
        warp_level_kernel_args[0] = (void *)(&tasks_ptr);
        warp_level_kernel_args[1] = (void *)(&tasks_size);
        warp_level_kernel_args[2] = (void *)(&hashers_ptr);
        warp_level_kernel_args[3] = (void *)(&hashers_size);

        static_assert(sizeof(char_t) == sizeof(byte_t), "Characters must be byte-sized");
        auto warp_level_kernel = &floating_rolling_hashers_on_each_cuda_warp_< //
            window_width_k, aligned_dimensions_k, sz_cap_cuda_k, byte_t,       //
            warp_size_nvidia_k, four_warps_per_multiprocessor_k>;

        // TODO: We can be wiser about the dimensions of this grid.
        unsigned const random_block_size = static_cast<unsigned>(warp_size_nvidia_k) * //
                                           static_cast<unsigned>(four_warps_per_multiprocessor_k);
        unsigned const random_blocks_per_multiprocessor = 2;
        cudaError_t launch_error = cudaLaunchCooperativeKernel(                       //
            reinterpret_cast<void *>(warp_level_kernel),                              // Kernel function pointer
            dim3(random_blocks_per_multiprocessor * specs.streaming_multiprocessors), // Grid dimensions
            dim3(random_block_size),                                                  // Block dimensions
            warp_level_kernel_args, // Array of kernel argument pointers
            0,                      // Shared memory per block (in bytes)
            executor.stream);       // CUDA stream
        if (launch_error != cudaSuccess)
            if (launch_error == cudaErrorMemoryAllocation) { return {status_t::bad_alloc_k, launch_error}; }
            else { return {status_t::unknown_k, launch_error}; }

        // Wait until everything completes, as on the next iteration we will update the properties again.
        cudaError_t execution_error = cudaStreamSynchronize(executor.stream);
        if (execution_error != cudaSuccess) { return {status_t::unknown_k, execution_error}; }

        // Calculate the duration:
        cudaError_t stop_event_error = cudaEventRecord(stop_event, executor.stream);
        if (stop_event_error != cudaSuccess) return {status_t::unknown_k, stop_event_error};
        float execution_milliseconds = 0;
        cudaEventElapsedTime(&execution_milliseconds, start_event, stop_event);

        return {status_t::success_k, cudaSuccess, execution_milliseconds};
    }
};

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_FINGERPRINTS_CUH_
