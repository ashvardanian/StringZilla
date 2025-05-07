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
 *  That means, the coalesced memory is extremely important. Moreover, assuming we are mostly fetching each
 *  haystack byte only once, we want to make the transfer asynchronous, using @b `cp.async` PTX instructions.
 */
#ifndef STRINGCUZILLA_FIND_MANY_CUH_
#define STRINGCUZILLA_FIND_MANY_CUH_

#include "stringcuzilla/types.cuh"
#include "stringcuzilla/find_many.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

namespace ashvardanian {
namespace stringzilla {

#pragma region - Compressed State Machine

/**
 *  @brief  Reordered bit-unzipped form of @b `aho_corasick_dictionary` for CUDA.
 *
 *  GPUs have many levels of memory hierarchy, and the performance of a kernel heavily
 *  depends on its utilization. Instead of branching the state into 256 next states,
 *
 *  - Constant Memory (~ 50 KB):
 *      - for 256-level branching, and 4 bytes per state, fits ~ 50 states.
 *      - for 256-level branching, and 1 bytes per state, fits ~ 200 states.
 *      - for 2-level branching, and 4 bytes per state, fits ~ 6'000 states.
 *      - for 2-level branching, and 2 bytes per state, fits ~ 12'000 states.
 */
struct compressed_aho_corasick_dictionary_t {

    using u16s_allocator_t = unified_alloc<sz_u16_t>;
    using u32s_allocator_t = unified_alloc<sz_u32_t>;
    using u64s_allocator_t = unified_alloc<sz_u64_t>;

    safe_vector<sz_u16_t, u16s_allocator_t> transitions_u16x2_;
    safe_vector<sz_u32_t, u32s_allocator_t> transitions_u32x2_;
    safe_vector<sz_u64_t, u64s_allocator_t> transitions_u64x2_;

    safe_vector<sz_u32_t, u32s_allocator_t> outputs_counts_;
    safe_vector<sz_u32_t, u32s_allocator_t> needles_lengths_;

    template <typename state_id_type_, typename allocator_type_>
    status_t try_build(aho_corasick_dictionary<state_id_type_, allocator_type_> const &dict) noexcept {
        // We need to reorder the states, so that the most frequently used can be represented
        // as `uint8_t`, next as `uint16_t`, and the rest as `uint32_t` and `uint64_t`.

        //
        return status_t::success_k;
    }
};

#pragma endregion // Compressed State Machine

#pragma region - General Purpose CUDA Backend

/**
 *  @brief  Multi-pattern exact substring search on CUDA-capable GPUs, assigning one or more warps
 *          per haystack string, using atomic writes to global memory for final output.
 */
template < //
    typename state_id_type_,
    typename haystacks_strings_type_,           //
    sz_capability_t capability_ = sz_cap_cuda_k //
    >
__global__ void _count_many_in_cuda_block( //
    haystacks_strings_type_ haystacks, size_t const count_states, state_id_type_ const *transitions,
    state_id_type_ const *count_outputs_per_state) {}

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
