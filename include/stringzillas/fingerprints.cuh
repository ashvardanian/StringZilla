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
template <typename char_type_, typename min_hash_type_ = u32_t, typename min_count_type_ = u32_t>
struct cuda_fingerprint_task_ {
    using char_t = char_type_;
    using min_hash_t = min_hash_type_;
    using min_count_t = min_count_type_;

    char_t const *text_ptr = nullptr;
    size_t text_length = 0;
    size_t original_index = 0;
    min_hash_t *min_hashes = nullptr;
    min_count_t *min_counts = nullptr;
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
 *  @sa This kernel is much slower than `floating_rolling_hashers_on_each_cuda_warp_` and is intended as a fallback.
 *
 *  To avoid dynamically allocated buffers for the @p `hashers`, one should provide a compile-time upper bound
 *  for the number of dimensions, @p `dimensions_upper_bound_`, which is used to allocate registers. 1024 is a good
 *  default. If more dimensions are needed, we can easily call this kernel multiple times.
 */
template <                                                                     //
    unsigned dimensions_upper_bound_,                                          //
    typename hasher_type_,                                                     //
    typename min_hash_type_,                                                   //
    typename min_count_type_,                                                  //
    sz_capability_t capability_,                                               //
    typename char_type_ = byte_t, warp_size_t warp_size_ = warp_size_nvidia_k, //
    warp_tasks_density_t density_ = four_warps_per_multiprocessor_k            //
    >
__global__ void basic_rolling_hashers_kernel_(                                                                  //
    cuda_fingerprint_task_<char_type_, min_hash_type_, min_count_type_> const *tasks, size_t const tasks_count, //
    hasher_type_ const *hashers_global, size_t const hashers_count, size_t const max_window_width) {

    //
    using task_t = cuda_fingerprint_task_<char_type_, min_hash_type_, min_count_type_>;
    using hasher_t = hasher_type_;
    using rolling_state_t = typename hasher_t::state_t;
    using rolling_hash_t = typename hasher_t::hash_t;
    using min_hash_t = min_hash_type_;
    using min_count_t = min_count_type_;
    constexpr warp_size_t warp_size_k = warp_size_;
    constexpr warp_tasks_density_t density_k = density_;
    constexpr unsigned dimensions_k = dimensions_upper_bound_;
    constexpr unsigned dimensions_per_thread_k = dimensions_k / warp_size_k;
    static constexpr rolling_state_t skipped_rolling_state_k = std::numeric_limits<rolling_state_t>::max();
    static constexpr rolling_hash_t skipped_rolling_hash_k = std::numeric_limits<rolling_hash_t>::max();
    static constexpr min_hash_t max_hash_k = std::numeric_limits<min_hash_t>::max();
    static_assert(dimensions_k % warp_size_k == 0, "Dimensions must be a multiple of warp size");
    sz_assert_(hashers_count <= dimensions_k && "We can't have more hashers than the dimensions upper bound");

    // We may have multiple warps operating in the same block.
    unsigned const warp_size = warpSize;
    sz_assert_(warp_size == warp_size_k && "Warp size mismatch in kernel");
    unsigned const global_thread_index = static_cast<unsigned>(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned const global_warp_index = static_cast<unsigned>(global_thread_index / warp_size_k);
    unsigned const warps_per_block = static_cast<unsigned>(blockDim.x / warp_size_k);
    sz_assert_(warps_per_block == density_k && "Block size mismatch in kernel");
    unsigned const warps_per_device = static_cast<unsigned>(gridDim.x * warps_per_block);
    unsigned const thread_in_warp_index = static_cast<unsigned>(global_thread_index % warp_size_k);

    // Load the hashers states per thread in a strided fashion.
    hasher_t hashers[dimensions_per_thread_k];
#pragma unroll
    for (unsigned dim_within_thread = 0; dim_within_thread < dimensions_per_thread_k; ++dim_within_thread) {
        unsigned const dim = dim_within_thread * warp_size_k + thread_in_warp_index;
        hasher_t const &hasher = hashers_global[dim];
        if (dim >= hashers_count) continue; // ? Avoid out-of-bounds access
        hashers[dim_within_thread] = hasher;
    }

    // Each block/warp may end up receiving a different number of strings.
    for (size_t task_index = global_warp_index; task_index < tasks_count; task_index += warps_per_device) {
        task_t const task = tasks[task_index];

        // For each state we need to reset the local state
        rolling_state_t last_states[dimensions_per_thread_k];
        rolling_hash_t rolling_minimums[dimensions_per_thread_k];
        min_count_t rolling_counts[dimensions_per_thread_k];
        for (auto &rolling_state : last_states) rolling_state = rolling_state_t(0);
        for (auto &rolling_minimum : rolling_minimums) rolling_minimum = skipped_rolling_hash_k;
        for (auto &rolling_count : rolling_counts) rolling_count = 0;

        // Until we reach the maximum window length, use a branching code version
        size_t const prefix_length = std::min<size_t>(task.text_length, max_window_width);
        size_t new_char_offset = 0;
        for (; new_char_offset < prefix_length; ++new_char_offset) {
            auto const new_char = task.text_ptr[new_char_offset]; // ? Hardware may auto-broadcast this

#pragma unroll
            for (unsigned dim_within_thread = 0; dim_within_thread < dimensions_per_thread_k; ++dim_within_thread) {
                hasher_t &hasher = hashers[dim_within_thread];
                rolling_state_t &last_state = last_states[dim_within_thread];
                rolling_hash_t &rolling_minimum = rolling_minimums[dim_within_thread];
                min_count_t &min_count = rolling_counts[dim_within_thread];
                if (new_char_offset < hasher.window_width()) {
                    last_state = hasher.push(last_state, new_char);
                    if (hasher.window_width() == (new_char_offset + 1)) {
                        rolling_minimum = (std::min)(rolling_minimum, hasher.digest(last_state));
                        min_count = 1; // First occurrence of this hash
                    }
                    continue;
                }
                auto const old_char = task.text_ptr[new_char_offset - hasher.window_width()];
                last_state = hasher.roll(last_state, old_char, new_char);
                rolling_hash_t new_hash = hasher.digest(last_state);
                min_count *= new_hash >= rolling_minimum; // ? Discard `min_count` to 0 for new extremums
                min_count += new_hash <= rolling_minimum; // ? Increments by 1 for new & old minimums
                rolling_minimum = (std::min)(rolling_minimum, new_hash);
            }
        }

        // Now we can avoid a branch in the nested loop, as we are passed the longest window width
        for (; new_char_offset + warp_size_k <= task.text_length; new_char_offset += warp_size_k) {
            auto const new_char = task.text_ptr[new_char_offset]; // ? Hardware may auto-broadcast this

#pragma unroll
            for (unsigned dim_within_thread = 0; dim_within_thread < dimensions_per_thread_k; ++dim_within_thread) {
                hasher_t &hasher = hashers[dim_within_thread];
                rolling_state_t &last_state = last_states[dim_within_thread];
                rolling_hash_t &rolling_minimum = rolling_minimums[dim_within_thread];
                min_count_t &min_count = rolling_counts[dim_within_thread];
                auto const old_char = task.text_ptr[new_char_offset - hasher.window_width()];
                last_state = hasher.roll(last_state, old_char, new_char);
                rolling_hash_t new_hash = hasher.digest(last_state);
                min_count *= new_hash >= rolling_minimum; // ? Discard `min_count` to 0 for new extremums
                min_count += new_hash <= rolling_minimum; // ? Increments by 1 for new & old minimums
                rolling_minimum = (std::min)(rolling_minimum, new_hash);
            }
        }

        // Finally export the results
#pragma unroll
        for (unsigned dim_within_thread = 0; dim_within_thread < dimensions_per_thread_k; ++dim_within_thread) {
            unsigned const dim = dim_within_thread * warp_size_k + thread_in_warp_index;
            if (dim >= hashers_count) continue; // ? Avoid out-of-bounds access
            rolling_hash_t const &rolling_minimum = rolling_minimums[dim_within_thread];
            task.min_counts[dim] = rolling_minimum == skipped_rolling_state_k
                                       ? 0 // If the rolling minimum is not set, reset to zeros
                                       : rolling_counts[dim_within_thread];
            task.min_hashes[dim] = rolling_minimum == skipped_rolling_state_k
                                       ? max_hash_k // If the rolling minimum is not set, use the maximum hash value
                                       : static_cast<min_hash_t>(rolling_minimum & max_hash_k);
        }
    }
}

/**
 *  Each warp takes in an individual document from @p `tasks` and computes many rolling hashes for it.
 *  Each thread computes an independent rolling hash for a specific dimension, so you should have a multiple
 *  of warp-size dimensions per fingerprint.
 *
 *  Unlike the `basic_rolling_hashers_kernel_` basic variant, all @p `hashers` @b must have the same @p `window_width`.
 *  This greatly simplifies the memory access patterns. Assuming each thread in a warp can issue an independent
 *  read for consecutive elements, and easily loads 32 bits at a time, this kernel is suited to loading
 *  4x bytes and computing 4x (warp_size_) rolling hashes per thread in the inner loop.
 */
template <                                                                     //
    unsigned dimensions_, sz_capability_t capability_,                         //
    typename char_type_ = byte_t, warp_size_t warp_size_ = warp_size_nvidia_k, //
    warp_tasks_density_t density_ = four_warps_per_multiprocessor_k            //
    >
__global__ void floating_rolling_hashers_on_each_cuda_warp_(                   //
    cuda_fingerprint_task_<char_type_> const *tasks, size_t const tasks_count, //
    floating_rolling_hasher<f64_t> const *hashers, size_t const hashers_count, size_t const window_width) {

    //
    using task_t = cuda_fingerprint_task_<char_type_>;
    using hasher_t = floating_rolling_hasher<f64_t>;
    constexpr warp_size_t warp_size_k = warp_size_;
    constexpr warp_tasks_density_t density_k = density_;
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

        // Until we reach the `window_width`, we don't need to discard any symbols and can keep the code simpler
        size_t const prefix_length = std::min<size_t>(task.text_length, window_width);
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
        if (new_char_offset == window_width) {
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
            byte_t const *discarding_bytes = task.text_ptr + new_char_offset - window_width;
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
            byte_t const old_char = task.text_ptr[new_char_offset - window_width];
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
template <                                               //
    size_t dimensions_, sz_capability_t capability_,     //
    typename char_type_ = byte_t, size_t warp_size_ = 32 //
    >
__global__ void floating_rolling_hashers_across_cuda_device_(span<cuda_fingerprint_task_<char_type_>> tasks,
                                                             span<floating_rolling_hasher<f64_t> const> hashers) {
    sz_unused_(tasks);
    sz_unused_(hashers);
}

#pragma endregion - CUDA Kernels

/**
 *  @brief CUDA specialization of `basic_rolling_hashers` for count-min-sketching.
 */
template <typename hasher_type_, typename min_hash_type_, typename min_count_type_>
struct basic_rolling_hashers<hasher_type_, min_hash_type_, min_count_type_, unified_alloc_t, sz_cap_cuda_k> {

    using hasher_t = hasher_type_;
    using rolling_state_t = typename hasher_t::state_t;
    using rolling_hash_t = typename hasher_t::hash_t;

    using min_hash_t = min_hash_type_;
    using min_count_t = min_count_type_;
    using allocator_t = unified_alloc_t;

    using hashers_allocator_t = typename allocator_t::template rebind<hasher_t>::other;
    using hashers_t = safe_vector<hasher_t, hashers_allocator_t>;

    static constexpr sz_capability_t capability_k = sz_cap_cuda_k;
    static constexpr rolling_state_t skipped_rolling_state_k = std::numeric_limits<rolling_state_t>::max();
    static constexpr min_hash_t max_hash_k = std::numeric_limits<min_hash_t>::max();

    using min_hashes_span_t = span<min_hash_t>;
    using min_counts_span_t = span<min_count_t>;

    static constexpr unsigned hashes_per_warp_k = static_cast<unsigned>(warp_size_nvidia_k);
    static constexpr unsigned aligned_dimensions_k = 1024; // ? Must be a multiple of `warp_size_nvidia_k`

  private:
    using allocator_traits_t = std::allocator_traits<allocator_t>;
    using hasher_allocator_t = typename allocator_traits_t::template rebind_alloc<hasher_t>;
    using rolling_states_allocator_t = typename allocator_traits_t::template rebind_alloc<rolling_state_t>;
    using rolling_hashes_allocator_t = typename allocator_traits_t::template rebind_alloc<rolling_hash_t>;
    using min_counts_allocator_t = typename allocator_traits_t::template rebind_alloc<min_count_t>;

    allocator_t allocator_;
    hashers_t hashers_;
    size_t max_window_width_;

  public:
    basic_rolling_hashers(allocator_t const &allocator = {}) noexcept
        : allocator_(allocator), hashers_(allocator), max_window_width_(0) {}

    size_t dimensions() const noexcept { return hashers_.size(); }
    size_t max_window_width() const noexcept { return max_window_width_; }
    size_t window_width(size_t dim) const noexcept { return hashers_[dim].window_width(); }

    /**
     *  @brief Appends multiple new rolling hashers for a given @p window_width.
     *
     *  @param[in] window_width Width of the rolling window, typically 3, 4, 5, 6, or 7.
     *  @param[in] dims Number of hash functions to use, typically 768, 1024, or 1536.
     *  @param[in] alphabet_size Size of the alphabet, typically 256 for UTF-8, 4 for DNA, or 20 for proteins.
     *  @retval status_t::success_k on success, or an error code otherwise.
     *  @retval status_t::bad_alloc_k if the memory allocation fails.
     *
     *  Typical usage of this interface (error handling aside) would be like:
     *
     *  @code{.cpp}
     *  basic_rolling_hashers<rabin_karp_rolling_hasher<u32_t>> hashers;
     *  hashers.try_extend(3, 32); // 32 dims for 3-grams
     *  hashers.try_extend(5, 32); // 32 dims for 5-grams
     *  hashers.try_extend(7, 64); // 64 dims for 7-grams
     *  std::array<u32_t, 128> fingerprint; // 128 total dims
     *  hashers("some text", fingerprint);
     *  @endcode
     */
    SZ_NOINLINE status_t try_extend(size_t window_width, size_t new_dims, size_t alphabet_size = 256) noexcept {
        size_t const old_dims = hashers_.size();
        if (hashers_.try_reserve(old_dims + new_dims) != status_t::success_k) return status_t::bad_alloc_k;
        for (size_t new_dim = 0; new_dim < new_dims; ++new_dim) {
            size_t const dim = old_dims + new_dim;
            status_t status = try_append(hasher_t(window_width, alphabet_size + dim));
            sz_assert_(status == status_t::success_k && "Couldn't fail after the reserve");
        }
        return status_t::success_k;
    }

    /**
     *  @brief Appends a new rolling @p hasher to the collection via `try_append`.
     *  @retval status_t::success_k on success, or an error code otherwise.
     *  @retval status_t::bad_alloc_k if the memory allocation fails.
     */
    SZ_NOINLINE status_t try_append(hasher_t hasher) noexcept {
        auto const new_window_width = hasher.window_width();
        if (hashers_.try_push_back(std::move(hasher)) != status_t::success_k) return status_t::bad_alloc_k;

        max_window_width_ = (std::max)(new_window_width, max_window_width_);
        return status_t::success_k;
    }

    /**
     *  @brief Computes many fingerprints in parallel for input @p texts via an @p executor.
     *  @param[in] texts The input texts to hash, typically a sequential container of UTF-8 encoded strings.
     *  @param[out] min_hashes_per_text The output fingerprints, an array of vectors of minimum hashes.
     *  @param[out] min_counts_per_text The output frequencies of @p `min_hashes_per_text` hashes.
     *  @param[in] executor The device executor to use for parallel processing, defaults to the first GPU.
     *  @param[in] specs The GPU specifications to use, defaults to an empty `gpu_specs_t`.
     *  @retval status_t::success_k on success, or an error code otherwise.
     *  @retval status_t::bad_alloc_k if the memory allocation fails.
     */
    template <typename texts_type_, typename min_hashes_per_text_type_, typename min_counts_per_text_type_>
    SZ_NOINLINE cuda_status_t operator()(                                                                 //
        texts_type_ const &texts,                                                                         //
        min_hashes_per_text_type_ &&min_hashes_per_text, min_counts_per_text_type_ &&min_counts_per_text, //
        cuda_executor_t executor = {}, gpu_specs_t specs = {}) const noexcept {

        using texts_t = texts_type_;
        using text_t = typename texts_t::value_type;
        using char_t = typename text_t::value_type;
        using task_t = cuda_fingerprint_task_<char_t, min_hash_t, min_count_t>;
        using tasks_allocator_t = typename allocator_t::template rebind<task_t>::other;

        // Preallocate the events for GPU timing.
        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event, cudaEventBlockingSync);
        cudaEventCreate(&stop_event, cudaEventBlockingSync);

        // Populate the tasks for each warp or the entire device, putting it into unified memory.
        safe_vector<task_t, tasks_allocator_t> tasks(allocator_);
        if (tasks.try_resize(texts.size()) == status_t::bad_alloc_k) return {status_t::bad_alloc_k};
        for (size_t task_index = 0; task_index < texts.size(); ++task_index) {
            auto const &text = texts[task_index];
            auto min_hashes = to_span(min_hashes_per_text[task_index]);
            auto min_counts = to_span(min_counts_per_text[task_index]);
            // Ensure device-accessible buffers (Unified/Device memory) for inputs and outputs
            if (!is_device_accessible_memory((void const *)text.data()) ||
                !is_device_accessible_memory((void const *)min_hashes.data()) ||
                !is_device_accessible_memory((void const *)min_counts.data()))
                return {status_t::device_memory_mismatch_k, cudaSuccess};
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
        cudaError_t start_event_error = cudaEventRecord(start_event, executor.stream());
        if (start_event_error != cudaSuccess) return {status_t::unknown_k, start_event_error};

        void *warp_level_kernel_args[5];
        auto const *tasks_ptr = tasks.data();
        auto const tasks_size = tasks.size();
        auto const *hashers_ptr = hashers_.data();
        auto const hashers_size = hashers_.size();
        warp_level_kernel_args[0] = (void *)(&tasks_ptr);
        warp_level_kernel_args[1] = (void *)(&tasks_size);
        warp_level_kernel_args[2] = (void *)(&hashers_ptr);
        warp_level_kernel_args[3] = (void *)(&hashers_size);
        warp_level_kernel_args[4] = (void *)(&max_window_width_);

        static_assert(sizeof(char_t) == sizeof(byte_t), "Characters must be byte-sized");
        auto warp_level_kernel = &basic_rolling_hashers_kernel_< //
            aligned_dimensions_k, hasher_t, min_hash_t, min_count_t, sz_cap_cuda_k, byte_t, warp_size_nvidia_k,
            four_warps_per_multiprocessor_k>;

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
            executor.stream());     // CUDA stream
        if (launch_error != cudaSuccess)
            if (launch_error == cudaErrorMemoryAllocation) { return {status_t::bad_alloc_k, launch_error}; }
            else { return {status_t::unknown_k, launch_error}; }

        // Wait until everything completes, as on the next iteration we will update the properties again.
        cudaError_t execution_error = cudaStreamSynchronize(executor.stream());
        if (execution_error != cudaSuccess) { return {status_t::unknown_k, execution_error}; }

        // Calculate the duration:
        cudaError_t stop_event_error = cudaEventRecord(stop_event, executor.stream());
        if (stop_event_error != cudaSuccess) return {status_t::unknown_k, stop_event_error};
        float execution_milliseconds = 0;
        cudaEventElapsedTime(&execution_milliseconds, start_event, stop_event);

        return {status_t::success_k, cudaSuccess, execution_milliseconds};
    }
};

/**
 *  @brief CUDA specialization of `floating_rolling_hashers` for count-min-sketching.
 */
template <size_t dimensions_>
struct floating_rolling_hashers<sz_cap_cuda_k, dimensions_> {

    using hasher_t = floating_rolling_hasher<f64_t>;
    using rolling_state_t = f64_t;
    using min_hash_t = u32_t;
    using min_count_t = u32_t;
    using allocator_t = unified_alloc_t;

    using hashers_allocator_t = typename allocator_t::template rebind<hasher_t>::other;
    using hashers_t = safe_vector<hasher_t, hashers_allocator_t>;

    static constexpr size_t dimensions_k = dimensions_;
    static constexpr sz_capability_t capability_k = sz_cap_cuda_k;
    static constexpr rolling_state_t skipped_rolling_state_k = std::numeric_limits<rolling_state_t>::max();
    static constexpr min_hash_t max_hash_k = std::numeric_limits<min_hash_t>::max();

    using min_hashes_span_t = span<min_hash_t, dimensions_k>;
    using min_counts_span_t = span<min_count_t, dimensions_k>;

    static constexpr unsigned hashes_per_warp_k = static_cast<unsigned>(warp_size_nvidia_k);
    static constexpr bool has_incomplete_tail_group_k = (dimensions_k % hashes_per_warp_k) != 0;
    static constexpr size_t aligned_dimensions_k =
        has_incomplete_tail_group_k ? (dimensions_k / hashes_per_warp_k + 1) * hashes_per_warp_k : (dimensions_k);
    static constexpr unsigned groups_count_k = aligned_dimensions_k / hashes_per_warp_k;

  private:
    allocator_t allocator_;
    hashers_t hashers_;
    size_t window_width_;

  public:
    floating_rolling_hashers(allocator_t const &allocator = {}) noexcept
        : allocator_(allocator), hashers_(allocator), window_width_(0) {}
    constexpr size_t dimensions() const noexcept { return dimensions_k; }
    constexpr size_t window_width() const noexcept { return window_width_; }
    constexpr size_t window_width(size_t) const noexcept { return window_width_; }

    /**
     *  @brief Initializes several rolling hashers with different multipliers and modulos.
     *  @param[in] alphabet_size Size of the alphabet, typically 256 for UTF-8, 4 for DNA, or 20 for proteins.
     *  @param[in] first_dimension_offset The offset for the first dimension within a larger fingerprint, typically 0.
     */
    SZ_NOINLINE status_t try_seed(size_t window_width, size_t alphabet_size = 256,
                                  size_t first_dimension_offset = 0) noexcept {
        if (hashers_.try_resize(aligned_dimensions_k) != status_t::success_k) return status_t::bad_alloc_k;
        for (size_t dim = 0; dim < dimensions_k; ++dim)
            hashers_[dim] =
                hasher_t(window_width, alphabet_size + first_dimension_offset + dim, hasher_t::default_modulo_base_k);
        window_width_ = window_width;
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

        using task_t = cuda_fingerprint_task_<byte_t>;
        using tasks_allocator_t = typename allocator_t::template rebind<task_t>::other;
        sz_unused_(specs);

        // Preallocate the events for GPU timing.
        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event, cudaEventBlockingSync);
        cudaEventCreate(&stop_event, cudaEventBlockingSync);

        // Populate the tasks array with a single task for the entire device.
        safe_vector<task_t, tasks_allocator_t> tasks(allocator_);
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
        cudaError_t start_event_error = cudaEventRecord(start_event, executor.stream());
        if (start_event_error != cudaSuccess) return {status_t::unknown_k, start_event_error};

        void *warp_level_kernel_args[5];
        auto const *tasks_ptr = tasks.data();
        auto const tasks_size = tasks.size();
        auto const *hashers_ptr = hashers_.data();
        auto const hashers_size = (std::min)(dimensions_k, hashers_.size());
        warp_level_kernel_args[0] = (void *)(&tasks_ptr);
        warp_level_kernel_args[1] = (void *)(&tasks_size);
        warp_level_kernel_args[2] = (void *)(&hashers_ptr);
        warp_level_kernel_args[3] = (void *)(&hashers_size);
        warp_level_kernel_args[4] = (void *)(&window_width_);

        auto warp_level_kernel = &floating_rolling_hashers_on_each_cuda_warp_< //
            aligned_dimensions_k, sz_cap_cuda_k, byte_t, warp_size_nvidia_k, one_warp_per_multiprocessor_k>;

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
            executor.stream());                                 // CUDA stream
        if (launch_error != cudaSuccess)
            if (launch_error == cudaErrorMemoryAllocation) { return {status_t::bad_alloc_k, launch_error}; }
            else { return {status_t::unknown_k, launch_error}; }

        // Wait until everything completes, as on the next iteration we will update the properties again.
        cudaError_t execution_error = cudaStreamSynchronize(executor.stream());
        if (execution_error != cudaSuccess) { return {status_t::unknown_k, execution_error}; }

        // Calculate the duration:
        cudaError_t stop_event_error = cudaEventRecord(stop_event, executor.stream());
        if (stop_event_error != cudaSuccess) return {status_t::unknown_k, stop_event_error};
        float execution_milliseconds = 0;
        cudaEventElapsedTime(&execution_milliseconds, start_event, stop_event);

        return {status_t::success_k, cudaSuccess, execution_milliseconds};
    }

    /**
     *  @brief Computes many fingerprints in parallel for input @p texts via an @p executor.
     *  @param[in] texts The input texts to hash, typically a sequential container of UTF-8 encoded strings.
     *  @param[out] min_hashes_per_text The output fingerprints, an array of vectors of minimum hashes.
     *  @param[out] min_counts_per_text The output frequencies of @p `min_hashes_per_text` hashes.
     *  @param[in] executor The device executor to use for parallel processing, defaults to the first GPU.
     *  @param[in] specs The GPU specifications to use, defaults to an empty `gpu_specs_t`.
     *  @retval status_t::success_k on success, or an error code otherwise.
     *  @retval status_t::bad_alloc_k if the memory allocation fails.
     */
    template <typename texts_type_, typename min_hashes_per_text_type_, typename min_counts_per_text_type_>
    SZ_NOINLINE cuda_status_t operator()(texts_type_ const &texts, min_hashes_per_text_type_ &&min_hashes_per_text,
                                         min_counts_per_text_type_ &&min_counts_per_text, cuda_executor_t executor = {},
                                         gpu_specs_t specs = {}) const noexcept {

        using texts_t = texts_type_;
        using text_t = typename texts_t::value_type;
        using char_t = typename text_t::value_type;
        using task_t = cuda_fingerprint_task_<char_t>;
        using tasks_allocator_t = typename allocator_t::template rebind<task_t>::other;

        // Preallocate the events for GPU timing.
        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event, cudaEventBlockingSync);
        cudaEventCreate(&stop_event, cudaEventBlockingSync);

        // Populate the tasks for each warp or the entire device, putting it into unified memory.
        safe_vector<task_t, tasks_allocator_t> tasks(allocator_);
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
        cudaError_t start_event_error = cudaEventRecord(start_event, executor.stream());
        if (start_event_error != cudaSuccess) return {status_t::unknown_k, start_event_error};

        void *warp_level_kernel_args[5];
        auto const *tasks_ptr = tasks.data();
        auto const tasks_size = tasks.size();
        auto const *hashers_ptr = hashers_.data();
        auto const hashers_size = (std::min)(dimensions_k, hashers_.size());
        warp_level_kernel_args[0] = (void *)(&tasks_ptr);
        warp_level_kernel_args[1] = (void *)(&tasks_size);
        warp_level_kernel_args[2] = (void *)(&hashers_ptr);
        warp_level_kernel_args[3] = (void *)(&hashers_size);
        warp_level_kernel_args[4] = (void *)(&window_width_);

        static_assert(sizeof(char_t) == sizeof(byte_t), "Characters must be byte-sized");
        auto warp_level_kernel = &floating_rolling_hashers_on_each_cuda_warp_< //
            aligned_dimensions_k, sz_cap_cuda_k, byte_t, warp_size_nvidia_k, four_warps_per_multiprocessor_k>;

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
            executor.stream());     // CUDA stream
        if (launch_error != cudaSuccess)
            if (launch_error == cudaErrorMemoryAllocation) { return {status_t::bad_alloc_k, launch_error}; }
            else { return {status_t::unknown_k, launch_error}; }

        // Wait until everything completes, as on the next iteration we will update the properties again.
        cudaError_t execution_error = cudaStreamSynchronize(executor.stream());
        if (execution_error != cudaSuccess) { return {status_t::unknown_k, execution_error}; }

        // Calculate the duration:
        cudaError_t stop_event_error = cudaEventRecord(stop_event, executor.stream());
        if (stop_event_error != cudaSuccess) return {status_t::unknown_k, stop_event_error};
        float execution_milliseconds = 0;
        cudaEventElapsedTime(&execution_milliseconds, start_event, stop_event);

        return {status_t::success_k, cudaSuccess, execution_milliseconds};
    }
};

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_FINGERPRINTS_CUH_
