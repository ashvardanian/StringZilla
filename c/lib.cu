/**
 *  @file       lib.cu
 *  @brief      StringParaZilla library for parallel string operations using CUDA C++ and OpenMP backends.
 *  @author     Ash Vardanian
 *  @date       March 23, 2025
 */
#include <fork_union.hpp> // Fork-join scoped thread pool

#include <stringparazilla/find_many.hpp>  // C++ templates for string processing
#include <stringparazilla/similarity.hpp> // C++ templates for string similarity

#if SZ_USE_CUDA
#include <stringparazilla/find_many.cuh>  // Parallel string processing in CUDA
#include <stringparazilla/similarity.cuh> // Parallel string similarity in CUDA
#endif

namespace sz = ashvardanian::stringzilla;
namespace spz = ashvardanian::stringparazilla;

extern "C" {

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_u32tape( //
    sz_cptr_t a_data, sz_u32_t const *a_lengths,         //
    sz_cptr_t b_data, sz_u32_t const *b_lengths,         //
    sz_size_t count,                                     //
    sz_size_t bound,                                     //
    sz_memory_allocator_t *alloc, sz_size_t *results) {

    _sz_unused(bound && alloc);

    using tape_t = sz::arrow_strings_tape<char, sz_u32_t, sz::dummy_alloc_t>;
    sz_size_t const a_total_length = a_lengths[count];
    sz_size_t const b_total_length = b_lengths[count];
    tape_t a({a_data, a_total_length}, {a_lengths, count + 1}, {});
    tape_t b({b_data, b_total_length}, {b_lengths, count + 1}, {});

    sz::status_t result = sz::cuda::levenshtein_distances(a, b, results);
    return (sz_status_t)result;
}

SZ_DYNAMIC sz_status_t sz_needleman_wunsch_scores_u32tape( //
    sz_cptr_t a_data, sz_u32_t const *a_lengths,           //
    sz_cptr_t b_data, sz_u32_t const *b_lengths,           //
    sz_size_t count,                                       //
    sz_error_cost_t const *subs, sz_error_cost_t gap,      //
    sz_memory_allocator_t *alloc, sz_ssize_t *results) {

    _sz_unused(alloc);

    using tape_t = sz::arrow_strings_tape<char, sz_u32_t, sz::dummy_alloc_t>;
    sz_size_t const a_total_length = a_lengths[count];
    sz_size_t const b_total_length = b_lengths[count];
    tape_t a({a_data, a_total_length}, {a_lengths, count + 1}, {});
    tape_t b({b_data, b_total_length}, {b_lengths, count + 1}, {});

    sz::status_t result = sz::cuda::needleman_wunsch_scores(a, b, results, subs, gap);
    return (sz_status_t)result;
}
} // extern "C"