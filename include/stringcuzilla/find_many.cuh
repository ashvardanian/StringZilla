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
 *  That means, the coalesced memory is extremely important. Moreover, assuming we are mostly fetching each
 *  haystack byte only once, we want to make the transfer asynchronous, using @b `cp.async` PTX instructions.
 *
 *
 */
#ifndef STRINGCUZILLA_FIND_MANY_CUH_
#define STRINGCUZILLA_FIND_MANY_CUH_

#include "stringcuzilla/types.cuh"
#include "stringcuzilla/find_many.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

namespace ashvardanian {
namespace stringzilla {

#pragma region - General Purpose CUDA Backend

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
 */
template < //
    typename state_id_type_,
    typename haystacks_strings_type_,           //
    sz_capability_t capability_ = sz_cap_cuda_k //
    >
__global__ void _count_matches_with_haystack_per_warp( //
    haystacks_strings_type_ haystacks, size_t const count_states, state_id_type_ const *transitions,
    state_id_type_ const *count_outputs_per_state) {

    using haystack_t = typename haystacks_strings_type_::value_type;
    using char_t = typename haystack_t::value_type;
    using state_id_t = state_id_type_;

    // We may have multiple warps operating in the same block.
    uint const warp_size = warpSize;
    size_t const global_thread_index = static_cast<uint>(blockIdx.x * blockDim.x + threadIdx.x);
    size_t const global_warp_index = static_cast<uint>(global_thread_index / warp_size);
    size_t const warps_per_block = static_cast<uint>(blockDim.x / warp_size);
    size_t const warps_per_device = static_cast<uint>(gridDim.x * warps_per_block);
    uint const warp_thread_index = static_cast<uint>(global_thread_index % warp_size);
    bool const is_last_in_warp = (warp_thread_index + 1 == warp_size);

    for (size_t haystack_index = global_warp_index; haystack_index < haystacks.size();
         haystack_index += warps_per_device) {
        // Each warp is assigned to a single haystack.
        haystack_t haystack = haystacks[haystack_index];
        size_t const haystack_size = haystack.size();
        size_t const haystack_offset = 0;
        state_id_t current_state = 0;
        state_id_t thread_matches_count = 0;

        // Our text processing is happening left to right.
        // On each cycle we could load 1 char, but we also can prefetch the one that will be used
        // at the next warp-level shuffle. Sadly, there is `__shfl_down_sync` doesn't support circular
        // rotation, so we need to use a reverse processing order for the `next_cycle_char`.
        char_t current_char = haystack[warp_thread_index];
        char_t next_cycle_char = haystack[warp_size + warp_thread_index];

        // Fetch the new state and the number of possible matches ending here.
        state_id_t next_state = transitions[current_state * 256 + static_cast<uint>(current_char)];
        state_id_t current_output_count = count_outputs_per_state[next_state];

        // Aggregate.
        current_state = next_state;
        thread_matches_count += current_output_count;

#pragma unroll
        for (size_t window_offset = 0; window_offset < warp_size; ++window_offset) {
            // Shift down the current character, so all the threads except the last one - step forward.
            current_char = __shfl_down_sync(0xFFFFFFFF, current_char, 1, warp_size);
            // Select what is the end of the logical window observed by all threads in warp at `window_offset`.
            char_t last_char_in_window = __shfl_sync(0xFFFFFFFF, next_cycle_char, window_offset, warp_size);
            // Update the current character for the last thread in the warp.
            current_char = is_last_in_warp ? current_char : last_char_in_window; // ? Hope this is branch-free

            // Fetch the new state and the number of possible matches ending here.
            next_state = transitions[current_state * 256 + static_cast<uint>(current_char)];
            current_output_count = count_outputs_per_state[next_state];

            // Aggregate.
            current_state = next_state;
            thread_matches_count += current_output_count;
        }

        // Now that we've went through `warp_size` starting positions,
    }
}

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

    find_many(allocator_t alloc = allocator_t()) noexcept : dict_(alloc) {}
    void reset() noexcept { dict_.reset(); }

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
    status_t try_count(haystacks_type_ &&haystacks, span<size_t> counts, gpu_specs_t const &specs = {}) const noexcept {
        _sz_assert(counts.size() == haystacks.size());
        for (size_t i = 0; i < counts.size(); ++i) counts[i] = dict_.count(haystacks[i]);
        return status_t::success_k;
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
    status_t try_find(haystacks_type_ &&haystacks, output_matches_type_ &&matches, size_t &matches_count,
                      gpu_specs_t const &specs = {}) const noexcept {
        size_t count_found = 0, count_allowed = matches.size();
        for (auto it = haystacks.begin(); it != haystacks.end() && count_found != count_allowed; ++it)
            dict_.find(*it, [&](match_t match) {
                match.haystack_index = static_cast<size_t>(it - haystacks.begin());
                matches[count_found] = match;
                count_found++;
                return count_found < count_allowed;
            });
        matches_count = count_found;
        return status_t::success_k;
    }

  private:
    dictionary_t dict_;
};

#pragma endregion // General Purpose CUDA Backend

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGCUZILLA_FIND_MANY_CUH_
