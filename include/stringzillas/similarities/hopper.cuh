/**
 *  @brief Hopper-generation CUDA string-similarity specializations (DPX min/max chains).
 *  @file include/stringzillas/similarities/hopper.cuh
 *  @author Ash Vardanian
 *  @sa include/stringzillas/similarities/serial.hpp
 *  @sa include/stringzillas/similarities/cuda.cuh
 *  @sa include/stringzillas/similarities/kepler.cuh
 */
#ifndef STRINGZILLAS_SIMILARITIES_HOPPER_CUH_
#define STRINGZILLAS_SIMILARITIES_HOPPER_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>        // `cuda::pipeline`
#include <cooperative_groups.h> // `cooperative_groups::this_grid()`

#include "stringzillas/types.cuh"
#include "stringzillas/similarities/serial.hpp" // ISA-agnostic template core (must precede specializations)
#include "stringzillas/similarities/cuda.cuh"   // Base CUDA SIMT tier these specializations refine
#include "stringzillas/similarities/kepler.cuh" // Kepler tier (some Hopper scorers inherit from Kepler scorers)

namespace ashvardanian {
namespace stringzillas {

/*  Hopper-generation optimizations are quite different from Kepler.
 *  Our Kepler optimizations are mostly designed for 8-bit and 16-bit scalars packed as 32-bit words,
 *  while Hopper optimizations are designed for 16-bit and 32-bit scalars, grouping chains of add/min/max
 *  operations using DPX instructions.
 */

#if SZ_USE_HOPPER

template <>
struct tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k> {
    using kepler_warp_scorer_t::tile_scorer; // Make the constructors visible
};

/**
 *  @brief GPU adaptation of the `tile_scorer` - Minimizes Global Levenshtein distance with linear gap costs.
 *  @note Requires Hopper generation GPUs to handle 2x `u16` scores at a time.
 */
template <>
struct tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {
    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible

    SZ_DEVICE_INLINE void operator()(                      //
        char const *first_slice, char const *second_slice, //
        unsigned const tasks_offset, unsigned const tasks_step,
        unsigned const tasks_count,           // ! Unlike CPU, uses `unsigned`
        u16_t const *scores_pre_substitution, //
        u16_t const *scores_pre_insertion,    //
        u16_t const *scores_pre_deletion,     //
        u16_t *scores_new) noexcept {

        u16_t const match_cost = this->substituter_.match;
        u16_t const mismatch_cost = this->substituter_.mismatch;
        u16_t const gap_cost = this->gap_costs_.open_or_extend;

        u32_vec_t match_cost_vec, mismatch_cost_vec, gap_cost_vec, equality_vec;
        match_cost_vec.u32 = match_cost * 0x00010001;       // ! 2x `u16` match costs
        mismatch_cost_vec.u32 = mismatch_cost * 0x00010001; // ! 2x `u16` mismatch costs
        gap_cost_vec.u32 = gap_cost * 0x00010001;           // ! 2x `u16` gap costs

        // The hardest part of this kernel is dealing with unaligned loads!
        // We want to minimize single-byte processing in favor of 2-byte SIMD loads and min/max operations.
        // Assuming we are reading consecutive values from a buffer, in every cycle, most likely, we will be
        // dealing with most values being unaligned!
        u32_vec_t pre_substitution_vec, pre_insertion_vec, pre_deletion_vec;
        u32_vec_t first_vec, second_vec;
        u32_vec_t cost_of_substitution_vec, if_substitution_vec;
        u32_vec_t cell_score_vec;

        // ! As we are processing 2 bytes per loop, and have at least 32 threads per block (32 * 2 = 64),
        // ! and deal with strings only under 64k bytes, this loop will fire at most 1K times per input
        for (unsigned i = tasks_offset * 2; i < tasks_count; i += tasks_step * 2) { // ! it's OK to spill beyond bounds
            pre_substitution_vec.u16s[0] = scores_pre_substitution[i + 0];
            pre_substitution_vec.u16s[1] = scores_pre_substitution[i + 1];
            pre_insertion_vec.u16s[0] = scores_pre_insertion[i + 0];
            pre_insertion_vec.u16s[1] = scores_pre_insertion[i + 1];
            pre_deletion_vec.u16s[0] = scores_pre_deletion[i + 0];
            pre_deletion_vec.u16s[1] = scores_pre_deletion[i + 1];
            // Masked tail: the 2nd lane is a phantom cell on the final odd diagonal element; reading it would
            // spill one char before `first_slice` / one past `second_slice`, so load it only when the cell is real.
            bool const has_second_cell = i + 1 < tasks_count;
            first_vec.u16s[0] = load_immutable_(first_slice + tasks_count - i - 1);
            first_vec.u16s[1] = has_second_cell ? load_immutable_(first_slice + tasks_count - i - 2) : (char)0;
            second_vec.u16s[0] = load_immutable_(second_slice + i + 0);
            second_vec.u16s[1] = has_second_cell ? load_immutable_(second_slice + i + 1) : (char)0;

            // Equality comparison will output 0xFFFF for each matching byte-pair.
            equality_vec.u32 = __vcmpeq2(first_vec.u32, second_vec.u32);
            cost_of_substitution_vec.u32 =                //
                (equality_vec.u32 & match_cost_vec.u32) + //
                (~equality_vec.u32 & mismatch_cost_vec.u32);
            if_substitution_vec.u32 = __vaddus2(pre_substitution_vec.u32, cost_of_substitution_vec.u32);
            // The two gap branches share the same `gap_cost`, so `min(ins, del) + gap == min(ins + gap, del + gap)`.
            // We fold that single add and the final 2-way `min` against the substitution branch into one fused DPX
            // `__viaddmin_u16x2`, replacing the previous two `__vaddus2` adds plus a `__vimin3_u16x2`.
            cell_score_vec.u32 = __viaddmin_u16x2(__vminu2(pre_insertion_vec.u32, pre_deletion_vec.u32),
                                                  gap_cost_vec.u32, if_substitution_vec.u32);

            // When walking through the top-left triangle of the matrix, our output addresses are misaligned.
            scores_new[i + 0] = cell_score_vec.u16s[0];
            scores_new[i + 1] = cell_score_vec.u16s[1];
        }

        // Extract the bottom-right corner of the matrix, which is the result of the global alignment.
        if (tasks_offset == 0) this->final_score_ = scores_new[0];
    }
};

template <>
struct tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
};

template <>
struct tile_scorer<char const *, char const *, u64_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, u64_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
};

template <>
struct tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k> {
    using kepler_warp_scorer_t::tile_scorer; // Make the constructors visible
};

/**
 *  @brief GPU adaptation of the `tile_scorer` - Minimizes Global Levenshtein distance with affine gap costs.
 *  @note Requires Hopper generation GPUs to handle 2x `u8` scores at a time.
 */
template <>
struct tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible

    SZ_DEVICE_INLINE void operator()(                      //
        char const *first_slice, char const *second_slice, //
        unsigned const tasks_offset, unsigned const tasks_step,
        unsigned const tasks_count,             // ! Unlike CPU, uses `unsigned`
        u16_t const *scores_pre_substitution,   //
        u16_t const *scores_pre_insertion,      //
        u16_t const *scores_pre_deletion,       //
        u16_t const *scores_running_insertions, //
        u16_t const *scores_running_deletions,  //
        u16_t *scores_new,                      //
        u16_t *scores_new_insertions,           //
        u16_t *scores_new_deletions) noexcept {

        u16_t const match_cost = this->substituter_.match;
        u16_t const mismatch_cost = this->substituter_.mismatch;
        u16_t const gap_open_cost = this->gap_costs_.open;
        u16_t const gap_extend_cost = this->gap_costs_.extend;

        u32_vec_t match_cost_vec, mismatch_cost_vec, gap_open_cost_vec, gap_extend_cost_vec, equality_vec;
        match_cost_vec.u32 = match_cost * 0x00010001;           // ! 2x `u16` match costs
        mismatch_cost_vec.u32 = mismatch_cost * 0x00010001;     // ! 2x `u16` mismatch costs
        gap_open_cost_vec.u32 = gap_open_cost * 0x00010001;     // ! 2x `u16` gap costs
        gap_extend_cost_vec.u32 = gap_extend_cost * 0x00010001; // ! 2x `u16` gap costs

        // The hardest part of this kernel is dealing with unaligned loads!
        // We want to minimize single-byte processing in favor of 2-byte SIMD loads and min/max operations.
        // Assuming we are reading consecutive values from a buffer, in every cycle, most likely, we will be
        // dealing with most values being unaligned!
        u32_vec_t pre_substitution_vec, pre_insertion_opening_vec, pre_deletion_opening_vec;
        u32_vec_t pre_insertion_expansion_vec, pre_deletion_expansion_vec;
        u32_vec_t first_vec, second_vec;
        u32_vec_t cost_of_substitution_vec, if_substitution_vec, if_insertion_vec, if_deletion_vec;
        u32_vec_t cell_score_vec;

        // ! As we are processing 2 bytes per loop, and have at least 32 threads per block (32 * 2 = 64),
        // ! and deal with strings only under 64k bytes, this loop will fire at most 1K times per input
        for (unsigned i = tasks_offset * 2; i < tasks_count; i += tasks_step * 2) { // ! it's OK to spill beyond bounds
            pre_substitution_vec.u16s[0] = scores_pre_substitution[i + 0];
            pre_substitution_vec.u16s[1] = scores_pre_substitution[i + 1];
            pre_insertion_opening_vec.u16s[0] = scores_pre_insertion[i + 0];
            pre_insertion_opening_vec.u16s[1] = scores_pre_insertion[i + 1];
            pre_deletion_opening_vec.u16s[0] = scores_pre_deletion[i + 0];
            pre_deletion_opening_vec.u16s[1] = scores_pre_deletion[i + 1];
            pre_insertion_expansion_vec.u16s[0] = scores_running_insertions[i + 0];
            pre_insertion_expansion_vec.u16s[1] = scores_running_insertions[i + 1];
            pre_deletion_expansion_vec.u16s[0] = scores_running_deletions[i + 0];
            pre_deletion_expansion_vec.u16s[1] = scores_running_deletions[i + 1];
            // Masked tail: the 2nd lane is a phantom cell on the final odd diagonal element; reading it would
            // spill one char before `first_slice` / one past `second_slice`, so load it only when the cell is real.
            bool const has_second_cell = i + 1 < tasks_count;
            first_vec.u16s[0] = load_immutable_(first_slice + tasks_count - i - 1);
            first_vec.u16s[1] = has_second_cell ? load_immutable_(first_slice + tasks_count - i - 2) : (char)0;
            second_vec.u16s[0] = load_immutable_(second_slice + i + 0);
            second_vec.u16s[1] = has_second_cell ? load_immutable_(second_slice + i + 1) : (char)0;

            // Equality comparison will output 0xFFFF for each matching byte-pair.
            equality_vec.u32 = __vcmpeq2(first_vec.u32, second_vec.u32);
            cost_of_substitution_vec.u32 =                //
                (equality_vec.u32 & match_cost_vec.u32) + //
                (~equality_vec.u32 & mismatch_cost_vec.u32);
            if_substitution_vec.u32 = __vaddus2(pre_substitution_vec.u32, cost_of_substitution_vec.u32);
            if_insertion_vec.u32 = //
                __viaddmin_u16x2(pre_insertion_opening_vec.u32, gap_open_cost_vec.u32,
                                 __vaddus2(pre_insertion_expansion_vec.u32, gap_extend_cost_vec.u32));
            if_deletion_vec.u32 = //
                __viaddmin_u16x2(pre_deletion_opening_vec.u32, gap_open_cost_vec.u32,
                                 __vaddus2(pre_deletion_expansion_vec.u32, gap_extend_cost_vec.u32));
            cell_score_vec.u32 = __vimin3_u16x2(if_substitution_vec.u32, if_insertion_vec.u32, if_deletion_vec.u32);

            // When walking through the top-left triangle of the matrix, our output addresses are misaligned.
            scores_new[i + 0] = cell_score_vec.u16s[0];
            scores_new[i + 1] = cell_score_vec.u16s[1];
            scores_new_insertions[i + 0] = if_insertion_vec.u16s[0];
            scores_new_insertions[i + 1] = if_insertion_vec.u16s[1];
            scores_new_deletions[i + 0] = if_deletion_vec.u16s[0];
            scores_new_deletions[i + 1] = if_deletion_vec.u16s[1];
        }

        // Extract the bottom-right corner of the matrix, which is the result of the global alignment.
        if (tasks_offset == 0) this->final_score_ = scores_new[0];
    }
};

template <>
struct tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
};

template <>
struct tile_scorer<char const *, char const *, u64_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, u64_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
};

#endif

#if SZ_USE_HOPPER

/**
 *  @brief GPU adaptation of the `tile_scorer` - Maximizes Global or Local score with linear gap costs.
 *  @note Requires Hopper generation GPUs to handle 2x `i16` scores at a time.
 */
template <sz_similarity_locality_t locality_>
struct tile_scorer<char const *, char const *, i16_t, error_costs_classes_in_cuda_shared_memory_t, linear_gap_costs_t,
                   sz_maximize_score_k, locality_, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, i16_t, error_costs_classes_in_cuda_shared_memory_t,
                         linear_gap_costs_t, sz_maximize_score_k, locality_, sz_cap_cuda_k> {

    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;

    using tile_scorer<char const *, char const *, i16_t, error_costs_classes_in_cuda_shared_memory_t,
                      linear_gap_costs_t, sz_maximize_score_k, locality_,
                      sz_cap_cuda_k>::tile_scorer; // Make the constructors visible

    SZ_DEVICE_INLINE void operator()(                      //
        char const *first_slice, char const *second_slice, //
        unsigned const tasks_offset, unsigned const tasks_step,
        unsigned const tasks_count,           // ! Unlike CPU, uses `unsigned`
        i16_t const *scores_pre_substitution, //
        i16_t const *scores_pre_insertion,    //
        i16_t const *scores_pre_deletion,     //
        i16_t *scores_new) noexcept {

        error_costs_classes_in_cuda_shared_memory_t const &substituter = this->substituter_;
        i16_t const gap_cost = this->gap_costs_.open_or_extend;
        u32_vec_t gap_cost_vec;
        gap_cost_vec.i16s[0] = gap_cost_vec.i16s[1] = gap_cost;

        // The hardest part of this kernel is dealing with unaligned loads!
        // We want to minimize single-byte processing in favor of 2-byte SIMD loads and min/max operations.
        // Assuming we are reading consecutive values from a buffer, in every cycle, most likely, we will be
        // dealing with most values being unaligned!
        u32_vec_t pre_substitution_vec, pre_insertion_vec, pre_deletion_vec;
        u32_vec_t first_vec, second_vec;
        u32_vec_t cost_of_substitution_vec, if_deletion_or_insertion_vec;
        u32_vec_t cell_score_vec, final_score_vec;
        final_score_vec.i16s[0] = final_score_vec.i16s[1] = 0;

        // ! As we are processing 2 bytes per loop, and have at least 32 threads per block (32 * 2 = 64),
        // ! and deal with strings only under 64k bytes, this loop will fire at most 1K times per input
        for (unsigned i = tasks_offset * 2; i < tasks_count; i += tasks_step * 2) { // ! it's OK to spill beyond bounds
            pre_substitution_vec.i16s[0] = load_last_use_(scores_pre_substitution + i + 0);
            pre_substitution_vec.i16s[1] = load_last_use_(scores_pre_substitution + i + 1);
            pre_insertion_vec.i16s[0] = scores_pre_insertion[i + 0];
            pre_insertion_vec.i16s[1] = scores_pre_insertion[i + 1];
            pre_deletion_vec.i16s[0] = scores_pre_deletion[i + 0];
            pre_deletion_vec.i16s[1] = scores_pre_deletion[i + 1];
            // Masked tail: the 2nd lane is a phantom cell on the final odd diagonal element; reading it would
            // spill one char before `first_slice` / one past `second_slice`, so load it only when the cell is real.
            bool const has_second_cell = i + 1 < tasks_count;
            first_vec.u16s[0] = load_immutable_(first_slice + tasks_count - i - 1);
            first_vec.u16s[1] = has_second_cell ? load_immutable_(first_slice + tasks_count - i - 2) : (char)0;
            second_vec.u16s[0] = load_immutable_(second_slice + i + 0);
            second_vec.u16s[1] = has_second_cell ? load_immutable_(second_slice + i + 1) : (char)0;

            cost_of_substitution_vec.i16s[0] = substituter(first_vec.u16s[0], second_vec.u16s[0]);
            cost_of_substitution_vec.i16s[1] = substituter(first_vec.u16s[1], second_vec.u16s[1]);
            if_deletion_or_insertion_vec.u32 = __vaddss2(__vmaxs2(pre_insertion_vec.u32, pre_deletion_vec.u32),
                                                         gap_cost_vec.u32);

            // For local scoring we should use the ReLU variants of 3-way `max`.
            if constexpr (locality_k == sz_similarity_global_k) {
                cell_score_vec.u32 = __viaddmax_s16x2(pre_substitution_vec.u32, cost_of_substitution_vec.u32,
                                                      if_deletion_or_insertion_vec.u32);
                sz_unused_(final_score_vec);
            }
            else {
                cell_score_vec.u32 = __viaddmax_s16x2_relu(pre_substitution_vec.u32, cost_of_substitution_vec.u32,
                                                           if_deletion_or_insertion_vec.u32);
                // In the last iteration of the loop the second half-word contains noise,
                // so we have to discard it from affecting the final score.
                bool const is_tail = i + 1 == tasks_count;
                final_score_vec.i16s[0] = (std::max)(cell_score_vec.i16s[0], final_score_vec.i16s[0]);
                final_score_vec.i16s[1] = (std::max)(cell_score_vec.i16s[1 - is_tail], final_score_vec.i16s[1]);
            }

            // When walking through the top-left triangle of the matrix, our output addresses are misaligned.
            scores_new[i + 0] = cell_score_vec.i16s[0];
            scores_new[i + 1] = cell_score_vec.i16s[1];
        }

        // Extract the bottom-right corner of the matrix, which is the result of the global alignment.
        if constexpr (locality_k == sz_similarity_global_k) {
            if (tasks_offset == 0) this->final_score_ = scores_new[0];
        }
        else { // Or the best score for local alignment.
            this->final_score_ = __vimax3_s32(this->final_score_, final_score_vec.i16s[0], final_score_vec.i16s[1]);
            // On Hopper we can use specialized warp reductions for up-to 32-bit values:
            // this->final_score_ = pick_best_in_warp_<sz_maximize_score_k>(this->final_score_);
            this->final_score_ = __reduce_max_sync(0xFFFFFFFF, this->final_score_);
        }
    }
};

template <sz_similarity_locality_t locality_>
struct tile_scorer<char const *, char const *, i32_t, error_costs_classes_in_cuda_shared_memory_t, linear_gap_costs_t,
                   sz_maximize_score_k, locality_, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, i32_t, error_costs_classes_in_cuda_shared_memory_t,
                         linear_gap_costs_t, sz_maximize_score_k, locality_, sz_cap_cuda_k> {

    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;

    using tile_scorer<char const *, char const *, i32_t, error_costs_classes_in_cuda_shared_memory_t,
                      linear_gap_costs_t, sz_maximize_score_k, locality_,
                      sz_cap_cuda_k>::tile_scorer; // Make the constructors visible

    SZ_DEVICE_INLINE void operator()(                      //
        char const *first_slice, char const *second_slice, //
        unsigned const tasks_offset, unsigned const tasks_step,
        unsigned const tasks_count,           // ! Unlike CPU, uses `unsigned`
        i32_t const *scores_pre_substitution, //
        i32_t const *scores_pre_insertion,    //
        i32_t const *scores_pre_deletion,     //
        i32_t *scores_new) noexcept {

        // Make sure we are called for an anti-diagonal traversal order
        sz_assert_(scores_pre_insertion + 1 == scores_pre_deletion);
        error_costs_classes_in_cuda_shared_memory_t const &substituter = this->substituter_;
        i32_t const gap_costs = this->gap_costs_.open_or_extend;
        i32_t final_score = 0;

        for (unsigned i = tasks_offset; i < tasks_count; i += tasks_step) {
            i32_t pre_substitution = load_last_use_(scores_pre_substitution + i);
            i32_t pre_insertion = scores_pre_insertion[i];
            i32_t pre_deletion = scores_pre_deletion[i];
            char first_char = load_immutable_(first_slice + tasks_count - i - 1);
            char second_char = load_immutable_(second_slice + i);

            error_cost_t cost_of_substitution = substituter(first_char, second_char);
            i32_t if_deletion_or_insertion = (std::max)(pre_deletion, pre_insertion) + gap_costs;
            i32_t cell_score;

            // For local scoring we should use the ReLU variants of 3-way `max`.
            if constexpr (locality_k == sz_similarity_global_k) {
                cell_score = __viaddmax_s32(pre_substitution, cost_of_substitution, if_deletion_or_insertion);
                sz_unused_(final_score);
            }
            else {
                cell_score = __viaddmax_s32_relu(pre_substitution, cost_of_substitution, if_deletion_or_insertion);
                final_score = (std::max)(cell_score, final_score);
            }

            // When walking through the top-left triangle of the matrix, our output addresses are misaligned.
            scores_new[i] = cell_score;
        }

        // Extract the bottom-right corner of the matrix, which is the result of the global alignment.
        if constexpr (locality_k == sz_similarity_global_k) {
            if (tasks_offset == 0) this->final_score_ = scores_new[0];
        }
        else { // Or the best score for local alignment.
            this->final_score_ = (std::max)(this->final_score_, final_score);
            // On Hopper we can use specialized warp reductions for up-to 32-bit values:
            // this->final_score_ = pick_best_in_warp_<sz_maximize_score_k>(this->final_score_);
            this->final_score_ = __reduce_max_sync(0xFFFFFFFF, this->final_score_);
        }
    }
};

/**
 *  @brief GPU adaptation of the `tile_scorer` - Maximizes Global or Local score with affine gap costs.
 *  @note Requires Hopper generation GPUs to handle 2x `i16` scores at a time.
 */
template <sz_similarity_locality_t locality_>
struct tile_scorer<char const *, char const *, i16_t, error_costs_classes_in_cuda_shared_memory_t, affine_gap_costs_t,
                   sz_maximize_score_k, locality_, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, i16_t, error_costs_classes_in_cuda_shared_memory_t,
                         affine_gap_costs_t, sz_maximize_score_k, locality_, sz_cap_cuda_k> {

    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;

    using tile_scorer<char const *, char const *, i16_t, error_costs_classes_in_cuda_shared_memory_t,
                      affine_gap_costs_t, sz_maximize_score_k, locality_,
                      sz_cap_cuda_k>::tile_scorer; // Make the constructors visible

    SZ_DEVICE_INLINE void operator()(                      //
        char const *first_slice, char const *second_slice, //
        unsigned const tasks_offset, unsigned const tasks_step,
        unsigned const tasks_count,             // ! Unlike CPU, uses `unsigned`
        i16_t const *scores_pre_substitution,   //
        i16_t const *scores_pre_insertion,      //
        i16_t const *scores_pre_deletion,       //
        i16_t const *scores_running_insertions, //
        i16_t const *scores_running_deletions,  //
        i16_t *scores_new,                      //
        i16_t *scores_new_insertions,           //
        i16_t *scores_new_deletions) noexcept {

        error_costs_classes_in_cuda_shared_memory_t const &substituter = this->substituter_;
        i16_t const gap_open_cost = this->gap_costs_.open;
        i16_t const gap_extend_cost = this->gap_costs_.extend;
        u32_vec_t gap_open_cost_vec, gap_extend_cost_vec;
        gap_open_cost_vec.i16s[0] = gap_open_cost_vec.i16s[1] = gap_open_cost;
        gap_extend_cost_vec.i16s[0] = gap_extend_cost_vec.i16s[1] = gap_extend_cost;

        // The hardest part of this kernel is dealing with unaligned loads!
        // We want to minimize single-byte processing in favor of 2-byte SIMD loads and min/max operations.
        // Assuming we are reading consecutive values from a buffer, in every cycle, most likely, we will be
        // dealing with most values being unaligned!
        u32_vec_t pre_substitution_vec, pre_insertion_opening_vec, pre_deletion_opening_vec;
        u32_vec_t pre_insertion_expansion_vec, pre_deletion_expansion_vec;
        u32_vec_t first_vec, second_vec;
        u32_vec_t cost_of_substitution_vec, if_substitution_vec, if_insertion_vec, if_deletion_vec;
        u32_vec_t cell_score_vec, final_score_vec;
        final_score_vec.i16s[0] = final_score_vec.i16s[1] = 0;

        // ! As we are processing 2 bytes per loop, and have at least 32 threads per block (32 * 2 = 64),
        // ! and deal with strings only under 64k bytes, this loop will fire at most 1K times per input
        for (unsigned i = tasks_offset * 2; i < tasks_count; i += tasks_step * 2) { // ! it's OK to spill beyond bounds
            pre_substitution_vec.i16s[0] = load_last_use_(scores_pre_substitution + i + 0);
            pre_substitution_vec.i16s[1] = load_last_use_(scores_pre_substitution + i + 1);
            pre_insertion_opening_vec.i16s[0] = scores_pre_insertion[i + 0];
            pre_insertion_opening_vec.i16s[1] = scores_pre_insertion[i + 1];
            pre_deletion_opening_vec.i16s[0] = scores_pre_deletion[i + 0];
            pre_deletion_opening_vec.i16s[1] = scores_pre_deletion[i + 1];
            pre_insertion_expansion_vec.i16s[0] = scores_running_insertions[i + 0];
            pre_insertion_expansion_vec.i16s[1] = scores_running_insertions[i + 1];
            pre_deletion_expansion_vec.i16s[0] = scores_running_deletions[i + 0];
            pre_deletion_expansion_vec.i16s[1] = scores_running_deletions[i + 1];
            // Masked tail: the 2nd lane is a phantom cell on the final odd diagonal element; reading it would
            // spill one char before `first_slice` / one past `second_slice`, so load it only when the cell is real.
            bool const has_second_cell = i + 1 < tasks_count;
            first_vec.u16s[0] = load_immutable_(first_slice + tasks_count - i - 1);
            first_vec.u16s[1] = has_second_cell ? load_immutable_(first_slice + tasks_count - i - 2) : (char)0;
            second_vec.u16s[0] = load_immutable_(second_slice + i + 0);
            second_vec.u16s[1] = has_second_cell ? load_immutable_(second_slice + i + 1) : (char)0;

            cost_of_substitution_vec.i16s[0] = substituter(first_vec.u16s[0], second_vec.u16s[0]);
            cost_of_substitution_vec.i16s[1] = substituter(first_vec.u16s[1], second_vec.u16s[1]);
            if_substitution_vec.u32 = __vaddss2(pre_substitution_vec.u32, cost_of_substitution_vec.u32);
            if_insertion_vec.u32 = //
                __viaddmax_s16x2(pre_insertion_opening_vec.u32, gap_open_cost_vec.u32,
                                 __vaddss2(pre_insertion_expansion_vec.u32, gap_extend_cost_vec.u32));
            if_deletion_vec.u32 = //
                __viaddmax_s16x2(pre_deletion_opening_vec.u32, gap_open_cost_vec.u32,
                                 __vaddss2(pre_deletion_expansion_vec.u32, gap_extend_cost_vec.u32));

            // For local scoring we should use the ReLU variants of 3-way `max`.
            if constexpr (locality_k == sz_similarity_global_k) {
                cell_score_vec.u32 = __vimax3_s16x2(if_substitution_vec.u32, if_insertion_vec.u32, if_deletion_vec.u32);
                sz_unused_(final_score_vec);
            }
            else {
                cell_score_vec.u32 = __vimax3_s16x2_relu(if_substitution_vec.u32, if_insertion_vec.u32,
                                                         if_deletion_vec.u32);
                // In the last iteration of the loop the second half-word contains noise,
                // so we have to discard it from affecting the final score.
                bool const is_tail = i + 1 == tasks_count;
                final_score_vec.i16s[0] = (std::max)(cell_score_vec.i16s[0], final_score_vec.i16s[0]);
                final_score_vec.i16s[1] = (std::max)(cell_score_vec.i16s[1 - is_tail], final_score_vec.i16s[1]);
            }

            // When walking through the top-left triangle of the matrix, our output addresses are misaligned.
            scores_new[i + 0] = cell_score_vec.i16s[0];
            scores_new[i + 1] = cell_score_vec.i16s[1];
            scores_new_insertions[i + 0] = if_insertion_vec.i16s[0];
            scores_new_insertions[i + 1] = if_insertion_vec.i16s[1];
            scores_new_deletions[i + 0] = if_deletion_vec.i16s[0];
            scores_new_deletions[i + 1] = if_deletion_vec.i16s[1];
        }

        // Extract the bottom-right corner of the matrix, which is the result of the global alignment.
        if constexpr (locality_k == sz_similarity_global_k) {
            if (tasks_offset == 0) this->final_score_ = scores_new[0];
        }
        else { // Or the best score for local alignment.
            this->final_score_ = __vimax3_s32(this->final_score_, final_score_vec.i16s[0], final_score_vec.i16s[1]);
            // On Hopper we can use specialized warp reductions for up-to 32-bit values:
            // this->final_score_ = pick_best_in_warp_<sz_maximize_score_k>(this->final_score_);
            this->final_score_ = __reduce_max_sync(0xFFFFFFFF, this->final_score_);
        }
    }
};

template <sz_similarity_locality_t locality_>
struct tile_scorer<char const *, char const *, i32_t, error_costs_classes_in_cuda_shared_memory_t, affine_gap_costs_t,
                   sz_maximize_score_k, locality_, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, i32_t, error_costs_classes_in_cuda_shared_memory_t,
                         affine_gap_costs_t, sz_maximize_score_k, locality_, sz_cap_cuda_k> {

    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;

    using tile_scorer<char const *, char const *, i32_t, error_costs_classes_in_cuda_shared_memory_t,
                      affine_gap_costs_t, sz_maximize_score_k, locality_,
                      sz_cap_cuda_k>::tile_scorer; // Make the constructors visible

    SZ_DEVICE_INLINE void operator()(                      //
        char const *first_slice, char const *second_slice, //
        unsigned const tasks_offset, unsigned const tasks_step,
        unsigned const tasks_count,             // ! Unlike CPU, uses `unsigned`
        i32_t const *scores_pre_substitution,   //
        i32_t const *scores_pre_insertion,      //
        i32_t const *scores_pre_deletion,       //
        i32_t const *scores_running_insertions, //
        i32_t const *scores_running_deletions,  //
        i32_t *scores_new,                      //
        i32_t *scores_new_insertions,           //
        i32_t *scores_new_deletions) noexcept {

        // Make sure we are called for an anti-diagonal traversal order
        sz_assert_(scores_pre_insertion + 1 == scores_pre_deletion);
        i32_t const gap_open_cost = this->gap_costs_.open;
        i32_t const gap_extend_cost = this->gap_costs_.extend;
        error_costs_classes_in_cuda_shared_memory_t const &substituter = this->substituter_;
        i32_t final_score = 0;

        for (unsigned i = tasks_offset; i < tasks_count; i += tasks_step) {
            i32_t pre_substitution = load_last_use_(scores_pre_substitution + i);
            i32_t pre_insertion_opening = scores_pre_insertion[i];
            i32_t pre_deletion_opening = scores_pre_deletion[i];
            i32_t pre_insertion_expansion = scores_running_insertions[i];
            i32_t pre_deletion_expansion = scores_running_deletions[i];
            char first_char = load_immutable_(first_slice + tasks_count - i - 1);
            char second_char = load_immutable_(second_slice + i);

            error_cost_t cost_of_substitution = substituter(first_char, second_char);
            i32_t if_substitution = pre_substitution + cost_of_substitution;
            i32_t if_insertion = __viaddmax_s32(pre_insertion_opening, gap_open_cost,
                                                pre_insertion_expansion + gap_extend_cost);
            i32_t if_deletion = __viaddmax_s32(pre_deletion_opening, gap_open_cost,
                                               pre_deletion_expansion + gap_extend_cost);
            i32_t cell_score;

            // For local scoring we should use the ReLU variants of 3-way `max`.
            if constexpr (locality_k == sz_similarity_global_k) {
                cell_score = __vimax3_s32(if_substitution, if_insertion, if_deletion);
                sz_unused_(final_score);
            }
            else {
                cell_score = __vimax3_s32_relu(if_substitution, if_insertion, if_deletion);
                final_score = (std::max)(cell_score, final_score);
            }

            // When walking through the top-left triangle of the matrix, our output addresses are misaligned.
            scores_new[i] = cell_score;
            scores_new_insertions[i] = if_insertion;
            scores_new_deletions[i] = if_deletion;
        }

        // Extract the bottom-right corner of the matrix, which is the result of the global alignment.
        if constexpr (locality_k == sz_similarity_global_k) {
            if (tasks_offset == 0) this->final_score_ = scores_new[0];
        }
        else { // Or the best score for local alignment.
            this->final_score_ = (std::max)(this->final_score_, final_score);
            // On Hopper we can use specialized warp reductions for up-to 32-bit values:
            // this->final_score_ = pick_best_in_warp_<sz_maximize_score_k>(this->final_score_);
            this->final_score_ = __reduce_max_sync(0xFFFFFFFF, this->final_score_);
        }
    }
};

template <sz_similarity_locality_t locality_>
struct tile_scorer<char const *, char const *, i64_t, error_costs_classes_in_cuda_shared_memory_t, linear_gap_costs_t,
                   sz_maximize_score_k, locality_, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, i64_t, error_costs_classes_in_cuda_shared_memory_t,
                         linear_gap_costs_t, sz_maximize_score_k, locality_, sz_cap_cuda_k> {

    using tile_scorer<char const *, char const *, i64_t, error_costs_classes_in_cuda_shared_memory_t,
                      linear_gap_costs_t, sz_maximize_score_k, locality_,
                      sz_cap_cuda_k>::tile_scorer; // Make the constructors visible
};

template <sz_similarity_locality_t locality_>
struct tile_scorer<char const *, char const *, i64_t, error_costs_classes_in_cuda_shared_memory_t, affine_gap_costs_t,
                   sz_maximize_score_k, locality_, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, i64_t, error_costs_classes_in_cuda_shared_memory_t,
                         affine_gap_costs_t, sz_maximize_score_k, locality_, sz_cap_cuda_k> {

    using tile_scorer<char const *, char const *, i64_t, error_costs_classes_in_cuda_shared_memory_t,
                      affine_gap_costs_t, sz_maximize_score_k, locality_,
                      sz_cap_cuda_k>::tile_scorer; // Make the constructors visible
};

#pragma region Tiled tier and register tier DPX fold functors

/**
 *  @brief @b Hopper DPX partial specialization of the linear-gap DP cell: fuses `opt(diag+sub, opt(top,left)+gap)`
 *         (with the Smith-Waterman ReLU clamp for local) into one `__viaddmin/max_s32[_relu]`. Bit-exact with the
 *         scalar primary. @sa score_cell (cuda.cuh primary).
 */
template <sz_similarity_objective_t objective_, sz_similarity_locality_t locality_, sz_capability_t capability_,
          typename score_type_>
struct score_cell<objective_, locality_, capability_, score_type_,
                  std::enable_if_t<(capability_ & sz_cap_hopper_k) != 0 && sizeof(score_type_) <= 4>> {
    SZ_DEVICE_INLINE score_type_ operator()(score_type_ diag, score_type_ top, score_type_ left,
                                            score_type_ substitution, score_type_ gap) const noexcept {
        using score_t = score_type_;
        static constexpr bool is_local_k = locality_ == sz_similarity_local_k;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        int const diagonal = static_cast<int>(diag), substitution_cost = static_cast<int>(substitution);
        int const gap_cost = static_cast<int>(gap);
        int const top_cell = static_cast<int>(top), left_cell = static_cast<int>(left);
        if constexpr (objective_ == sz_minimize_distance_k)
            return static_cast<score_t>(
                __viaddmin_s32(diagonal, substitution_cost, min(top_cell, left_cell) + gap_cost));
        else if constexpr (is_local_k)
            return static_cast<score_t>(
                __viaddmax_s32_relu(diagonal, substitution_cost, max(top_cell, left_cell) + gap_cost));
        else
            return static_cast<score_t>(
                __viaddmax_s32(diagonal, substitution_cost, max(top_cell, left_cell) + gap_cost));
#else
        score_t cell = min_or_max<objective_>(
            static_cast<score_t>(diag + substitution),
            min_or_max<objective_>(static_cast<score_t>(top + gap), static_cast<score_t>(left + gap)));
        if constexpr (is_local_k) cell = min_or_max<objective_, score_t>(cell, 0);
        return cell;
#endif
    }
};

/**
 *  @brief @b Hopper DPX partial specialization of the affine-gap (Gotoh) DP cell: each of V, H, and M is a single fused
 *         `__viaddmin/max_s32[_relu]`. Bit-exact with the scalar primary. @sa affine_score_cell (cuda.cuh primary).
 */
template <sz_similarity_objective_t objective_, sz_similarity_locality_t locality_, sz_capability_t capability_,
          typename score_type_>
struct affine_score_cell<objective_, locality_, capability_, score_type_,
                         std::enable_if_t<(capability_ & sz_cap_hopper_k) != 0 && sizeof(score_type_) <= 4>> {
    SZ_DEVICE_INLINE score_type_ operator()( //
        score_type_ diag, score_type_ top_m, score_type_ top_v, score_type_ left_m, score_type_ left_h,
        score_type_ substitution, score_type_ open, score_type_ extend, score_type_ &v_out,
        score_type_ &h_out) const noexcept {
        using score_t = score_type_;
        static constexpr bool is_local_k = locality_ == sz_similarity_local_k;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        int const diagonal = static_cast<int>(diag), substitution_cost = static_cast<int>(substitution);
        int const open_cost = static_cast<int>(open), extend_cost = static_cast<int>(extend);
        if constexpr (objective_ == sz_minimize_distance_k) {
            int const vertical = __viaddmin_s32(static_cast<int>(top_m), open_cost,
                                                static_cast<int>(top_v) + extend_cost);
            int const horizontal = __viaddmin_s32(static_cast<int>(left_m), open_cost,
                                                  static_cast<int>(left_h) + extend_cost);
            v_out = static_cast<score_t>(vertical), h_out = static_cast<score_t>(horizontal);
            return static_cast<score_t>(__viaddmin_s32(diagonal, substitution_cost, min(vertical, horizontal)));
        }
        else {
            int const vertical = __viaddmax_s32(static_cast<int>(top_m), open_cost,
                                                static_cast<int>(top_v) + extend_cost);
            int const horizontal = __viaddmax_s32(static_cast<int>(left_m), open_cost,
                                                  static_cast<int>(left_h) + extend_cost);
            v_out = static_cast<score_t>(vertical), h_out = static_cast<score_t>(horizontal);
            int const gap_best = max(vertical, horizontal);
            return is_local_k ? static_cast<score_t>(__viaddmax_s32_relu(diagonal, substitution_cost, gap_best))
                              : static_cast<score_t>(__viaddmax_s32(diagonal, substitution_cost, gap_best));
        }
#else
        score_t const v = min_or_max<objective_>(static_cast<score_t>(top_m + open),
                                                 static_cast<score_t>(top_v + extend));
        score_t const h = min_or_max<objective_>(static_cast<score_t>(left_m + open),
                                                 static_cast<score_t>(left_h + extend));
        v_out = v, h_out = h;
        score_t if_substitution = static_cast<score_t>(diag + substitution);
        if constexpr (is_local_k) if_substitution = min_or_max<objective_, score_t>(if_substitution, 0);
        return min_or_max<objective_>(min_or_max<objective_>(v, h), if_substitution);
#endif
    }
};

/**
 *  @brief @b Hopper DPX partial specialization of the register-weighted linear-gap horizontal fold: each step is a
 *         fused `__viaddmax_s32(left, gap, cell)`. Bit-exact with the scalar primary. @sa weighted_gap_fold (cuda.cuh
 *         primary).
 */
template <sz_similarity_locality_t locality_, sz_capability_t capability_>
struct weighted_gap_fold<locality_, capability_, std::enable_if_t<(capability_ & sz_cap_hopper_k) != 0>> {
    SZ_DEVICE_INLINE void operator()(i16_t &cell_low, i16_t &cell_high, i16_t left_cell,
                                     i16_t gap_cost) const noexcept {
        static constexpr bool is_local_k = locality_ == sz_similarity_local_k;
        // The two packed cells are a horizontal chain (`cell_high` reads the freshly folded `cell_low`), so they
        // cannot share one packed `__viaddmax_s16x2`; each fused add-max is a scalar 32-bit DPX op (the i16->int
        // sign-extension is free). The local-objective clamp is already branchless (`max` lowers to `IMNMX`).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        int const gap = static_cast<int>(gap_cost);
        cell_low = static_cast<i16_t>(__viaddmax_s32(static_cast<int>(left_cell), gap, static_cast<int>(cell_low)));
        cell_high = static_cast<i16_t>(__viaddmax_s32(static_cast<int>(cell_low), gap, static_cast<int>(cell_high)));
#else
        cell_low = std::max<i16_t>(cell_low, left_cell + gap_cost);
        cell_high = std::max<i16_t>(cell_high, cell_low + gap_cost);
#endif
        if constexpr (is_local_k) cell_low = std::max<i16_t>(cell_low, 0), cell_high = std::max<i16_t>(cell_high, 0);
    }
};

/**
 *  @brief @b Hopper DPX partial specialization of the register-weighted affine deletion-track fold: each
 *         `max(left_m + open, left_h + extend)` is a fused `__viaddmax_s32`. Bit-exact with the scalar primary. @sa
 *         weighted_affine_gap_fold (cuda.cuh primary).
 */
template <sz_similarity_locality_t locality_, sz_capability_t capability_>
struct weighted_affine_gap_fold<locality_, capability_, std::enable_if_t<(capability_ & sz_cap_hopper_k) != 0>> {
    SZ_DEVICE_INLINE void operator()(                                   //
        i16_t match_or_insert_low, i16_t match_or_insert_high,          //
        i16_t left_cell, i16_t left_deletion, i16_t open, i16_t extend, //
        i16_t &deletion_low, i16_t &cell_low, i16_t &deletion_high, i16_t &cell_high) const noexcept {
        static constexpr bool is_local_k = locality_ == sz_similarity_local_k;
        // The high deletion track reads the freshly folded `cell_low`, so the low/high pair is a chain and cannot
        // share one packed `__viaddmax_s16x2`; each deletion is a scalar 32-bit fused add-max (i16->int is free).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        int const open_cost = static_cast<int>(open), extend_cost = static_cast<int>(extend);
        deletion_low = static_cast<i16_t>(
            __viaddmax_s32(static_cast<int>(left_cell), open_cost, static_cast<int>(left_deletion) + extend_cost));
        cell_low = std::max(match_or_insert_low, deletion_low);
        deletion_high = static_cast<i16_t>(
            __viaddmax_s32(static_cast<int>(cell_low), open_cost, static_cast<int>(deletion_low) + extend_cost));
        cell_high = std::max(match_or_insert_high, deletion_high);
#else
        deletion_low = std::max<i16_t>(left_cell + open, left_deletion + extend);
        cell_low = std::max(match_or_insert_low, deletion_low);
        deletion_high = std::max<i16_t>(cell_low + open, deletion_low + extend);
        cell_high = std::max(match_or_insert_high, deletion_high);
#endif
        if constexpr (is_local_k) cell_low = std::max<i16_t>(cell_low, 0), cell_high = std::max<i16_t>(cell_high, 0);
    }
};

#pragma endregion Tiled tier and register tier DPX fold functors

#endif

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SIMILARITIES_HOPPER_CUH_
