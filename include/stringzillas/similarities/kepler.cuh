/**
 *  @brief Kepler-generation CUDA string-similarity specializations (packed 8/16-bit SIMD).
 *  @file include/stringzillas/similarities/kepler.cuh
 *  @author Ash Vardanian
 *  @sa include/stringzillas/similarities/serial.hpp
 *  @sa include/stringzillas/similarities/cuda.cuh
 */
#ifndef STRINGZILLAS_SIMILARITIES_KEPLER_CUH_
#define STRINGZILLAS_SIMILARITIES_KEPLER_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>        // `cuda::pipeline`
#include <cooperative_groups.h> // `cooperative_groups::this_grid()`

#include "stringzillas/types.cuh"
#include "stringzillas/similarities/serial.hpp" // ISA-agnostic template core (must precede specializations)
#include "stringzillas/similarities/cuda.cuh"   // Base CUDA SIMT tier these specializations refine

namespace ashvardanian {
namespace stringzillas {

/*  On Kepler and newer GPUs we benefit from the following:
 *  - processing 4x 8-bit values or 2x 16-bit values at a time, packed as 32-bit words.
 *  - warp-level exchange primitives for fast reduction of the best score.
 */

#if SZ_USE_KEPLER

/**
 *  @brief GPU adaptation of the `tile_scorer` - Minimizes Global Levenshtein distance with linear gap costs.
 *  @note Requires Kepler generation GPUs to handle 4x `u8` scores at a time.
 *
 *  Relies on following instruction families to output 4x @b `u8` scores per call:
 *  - @b `prmt` to shuffle bytes in 32 bit registers.
 *  - @b `vmax4,vmin4,vadd4` video-processing instructions.
 */
template <>
struct tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
    using kepler_warp_scorer_t =
        tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                    sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>;

    __forceinline__ __device__ void operator()(            //
        char const *first_slice, char const *second_slice, //
        unsigned const tasks_offset, unsigned const tasks_step,
        unsigned const tasks_count,          // ! Unlike CPU, uses `unsigned`
        u8_t const *scores_pre_substitution, //
        u8_t const *scores_pre_insertion,    //
        u8_t const *scores_pre_deletion,     //
        u8_t *scores_new) noexcept {

        u8_t const match_cost = this->substituter_.match;
        u8_t const mismatch_cost = this->substituter_.mismatch;
        u8_t const gap_cost = this->gap_costs_.open_or_extend;

        u32_vec_t match_cost_vec, mismatch_cost_vec, gap_cost_vec, equality_vec;
        match_cost_vec.u32 = match_cost * 0x01010101u;       // ! 4x `u8` match costs
        mismatch_cost_vec.u32 = mismatch_cost * 0x01010101u; // ! 4x `u8` mismatch costs
        gap_cost_vec.u32 = gap_cost * 0x01010101u;           // ! 4x `u8` gap costs

        // The hardest part of this kernel is dealing with unaligned loads!
        // We want to minimize single-byte processing in favor of 4-byte SIMD loads and min/max operations.
        // Assuming we are reading consecutive values from a buffer, in every cycle, most likely, we will be
        // dealing with most values being unaligned!
        u32_vec_t pre_substitution_vec, pre_insertion_vec, pre_deletion_vec;
        u32_vec_t first_vec, second_vec;
        u32_vec_t cost_of_substitution_vec, if_substitution_vec, if_deletion_or_insertion_vec;
        u32_vec_t cell_score_vec;

        // ! As we are processing 4 bytes per loop, and have at least 32 threads per block (32 * 4 = 128),
        // ! and deal with strings only under 256 bytes, this loop will fire at most twice per input.
        for (unsigned i = tasks_offset * 4; i < tasks_count; i += tasks_step * 4) { // ! it's OK to spill beyond bounds
            pre_substitution_vec = sz_u32_load_unaligned(scores_pre_substitution + i);
            pre_insertion_vec = sz_u32_load_unaligned(scores_pre_insertion + i);
            pre_deletion_vec = sz_u32_load_unaligned(scores_pre_deletion + i);
            // Masked tail: the window packs 4 cells; phantom cells (`i + lane >= tasks_count`) would spill one
            // char before `first_slice` / past `second_slice`, so use the wide contiguous load only when the
            // whole window is in-bounds, otherwise assemble it lane-by-lane and default the phantom lanes to 0.
            if (i + 4 <= tasks_count) {
                first_vec = sz_u32_load_unaligned(first_slice + tasks_count - i - 4);
                second_vec = sz_u32_load_unaligned(second_slice + i);
                first_vec.u32 = __nv_bswap32(first_vec.u32); // ! reverse the order of bytes in the first vector
            }
            else {
                first_vec.u32 = 0, second_vec.u32 = 0;
                for (unsigned lane = 0; lane < 4 && i + lane < tasks_count; ++lane) {
                    first_vec.u8s[lane] = load_immutable_(first_slice + tasks_count - 1 - i - lane);
                    second_vec.u8s[lane] = load_immutable_(second_slice + i + lane);
                }
            }

            // Equality comparison will output 0xFF for each matching byte.
            equality_vec.u32 = __vcmpeq4(first_vec.u32, second_vec.u32);
            cost_of_substitution_vec.u32 =                //
                (equality_vec.u32 & match_cost_vec.u32) + //
                (~equality_vec.u32 & mismatch_cost_vec.u32);
            if_substitution_vec.u32 = __vaddus4(pre_substitution_vec.u32, cost_of_substitution_vec.u32);
            if_deletion_or_insertion_vec.u32 = __vaddus4(__vminu4(pre_deletion_vec.u32, pre_insertion_vec.u32),
                                                         gap_cost_vec.u32);
            cell_score_vec.u32 = __vminu4(if_deletion_or_insertion_vec.u32, if_substitution_vec.u32);

            // When walking through the top-left triangle of the matrix, our output addresses are misaligned.
            scores_new[i + 0] = cell_score_vec.u8s[0];
            scores_new[i + 1] = cell_score_vec.u8s[1];
            scores_new[i + 2] = cell_score_vec.u8s[2];
            scores_new[i + 3] = cell_score_vec.u8s[3];
        }

        // Extract the bottom-right corner of the matrix, which is the result of the global alignment.
        if (tasks_offset == 0) this->final_score_ = scores_new[0];
    }
};

/**
 *  @brief GPU adaptation of the `tile_scorer` - Minimizes Global Levenshtein distance with linear gap costs.
 *  @note Requires Kepler generation GPUs to handle 2x `u16` scores at a time.
 *
 *  Relies on following instruction families to output 2x @b `u16` scores per call:
 *  - @b `prmt` to shuffle bytes in 32 bit registers.
 *  - @b `vmax2,vmin2,vadd2` video-processing instructions.
 */
template <>
struct tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
    using kepler_warp_scorer_t =
        tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                    sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>;

    __forceinline__ __device__ void operator()(            //
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
        u32_vec_t cost_of_substitution_vec, if_substitution_vec, if_deletion_or_insertion_vec;
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
            if_deletion_or_insertion_vec.u32 = __vaddus2(__vminu2(pre_deletion_vec.u32, pre_insertion_vec.u32),
                                                         gap_cost_vec.u32);
            cell_score_vec.u32 = __vminu2(if_deletion_or_insertion_vec.u32, if_substitution_vec.u32);

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
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
    using kepler_warp_scorer_t =
        tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                    sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>;
};

template <>
struct tile_scorer<char const *, char const *, u64_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, u64_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
    using kepler_warp_scorer_t =
        tile_scorer<char const *, char const *, u64_t, uniform_substitution_costs_t, linear_gap_costs_t,
                    sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>;
};

/**
 *  @brief GPU adaptation of the `tile_scorer` - Minimizes Global Levenshtein distance with affine gap costs.
 *  @note Requires Kepler generation GPUs to handle 4x `u8` scores at a time.
 *
 *  Relies on following instruction families to output 4x @b `u8` scores per call:
 *  - @b `prmt` to shuffle bytes in 32 bit registers.
 *  - @b `vmax4,vmin4,vadd4` video-processing instructions.
 */
template <>
struct tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
    using kepler_warp_scorer_t =
        tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                    sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>;

    __forceinline__ __device__ void operator()(            //
        char const *first_slice, char const *second_slice, //
        unsigned const tasks_offset, unsigned const tasks_step,
        unsigned const tasks_count,            // ! Unlike CPU, uses `unsigned`
        u8_t const *scores_pre_substitution,   //
        u8_t const *scores_pre_insertion,      //
        u8_t const *scores_pre_deletion,       //
        u8_t const *scores_running_insertions, //
        u8_t const *scores_running_deletions,  //
        u8_t *scores_new,                      //
        u8_t *scores_new_insertions,           //
        u8_t *scores_new_deletions) noexcept {

        u8_t const match_cost = this->substituter_.match;
        u8_t const mismatch_cost = this->substituter_.mismatch;
        u8_t const gap_open_cost = this->gap_costs_.open;
        u8_t const gap_extend_cost = this->gap_costs_.extend;
        u32_vec_t match_cost_vec, mismatch_cost_vec, gap_open_cost_vec, gap_extend_cost_vec, equality_vec;
        match_cost_vec.u32 = match_cost * 0x01010101u;           // ! 4x `u8` match costs
        mismatch_cost_vec.u32 = mismatch_cost * 0x01010101u;     // ! 4x `u8` mismatch costs
        gap_open_cost_vec.u32 = gap_open_cost * 0x01010101u;     // ! 4x `u8` gap costs
        gap_extend_cost_vec.u32 = gap_extend_cost * 0x01010101u; // ! 4x `u8` gap costs

        // The hardest part of this kernel is dealing with unaligned loads!
        // We want to minimize single-byte processing in favor of 4-byte SIMD loads and min/max operations.
        // Assuming we are reading consecutive values from a buffer, in every cycle, most likely, we will be
        // dealing with most values being unaligned!
        u32_vec_t pre_substitution_vec, pre_insertion_opening_vec, pre_deletion_opening_vec;
        u32_vec_t pre_insertion_expansion_vec, pre_deletion_expansion_vec;
        u32_vec_t first_vec, second_vec;
        u32_vec_t cost_of_substitution_vec, if_substitution_vec, if_insertion_vec, if_deletion_vec;
        u32_vec_t cell_score_vec;

        // ! As we are processing 4 bytes per loop, and have at least 32 threads per block (32 * 4 = 128),
        // ! and deal with strings only under 256 bytes, this loop will fire at most twice per input.
        for (unsigned i = tasks_offset * 4; i < tasks_count; i += tasks_step * 4) { // ! it's OK to spill beyond bounds
            pre_substitution_vec = sz_u32_load_unaligned(scores_pre_substitution + i);
            pre_insertion_opening_vec = sz_u32_load_unaligned(scores_pre_insertion + i);
            pre_deletion_opening_vec = sz_u32_load_unaligned(scores_pre_deletion + i);
            pre_insertion_expansion_vec = sz_u32_load_unaligned(scores_running_insertions + i);
            pre_deletion_expansion_vec = sz_u32_load_unaligned(scores_running_deletions + i);
            // Masked tail: the window packs 4 cells; phantom cells (`i + lane >= tasks_count`) would spill one
            // char before `first_slice` / past `second_slice`, so use the wide contiguous load only when the
            // whole window is in-bounds, otherwise assemble it lane-by-lane and default the phantom lanes to 0.
            if (i + 4 <= tasks_count) {
                first_vec = sz_u32_load_unaligned(first_slice + tasks_count - i - 4);
                second_vec = sz_u32_load_unaligned(second_slice + i);
                first_vec.u32 = __nv_bswap32(first_vec.u32); // ! reverse the order of bytes in the first vector
            }
            else {
                first_vec.u32 = 0, second_vec.u32 = 0;
                for (unsigned lane = 0; lane < 4 && i + lane < tasks_count; ++lane) {
                    first_vec.u8s[lane] = load_immutable_(first_slice + tasks_count - 1 - i - lane);
                    second_vec.u8s[lane] = load_immutable_(second_slice + i + lane);
                }
            }

            // Equality comparison will output 0xFF for each matching byte.
            equality_vec.u32 = __vcmpeq4(first_vec.u32, second_vec.u32);
            cost_of_substitution_vec.u32 =                //
                (equality_vec.u32 & match_cost_vec.u32) + //
                (~equality_vec.u32 & mismatch_cost_vec.u32);
            if_substitution_vec.u32 = __vaddus4(pre_substitution_vec.u32, cost_of_substitution_vec.u32);
            if_insertion_vec.u32 = __vminu4(__vaddus4(pre_insertion_opening_vec.u32, gap_open_cost_vec.u32),
                                            __vaddus4(pre_insertion_expansion_vec.u32, gap_extend_cost_vec.u32));
            if_deletion_vec.u32 = __vminu4(__vaddus4(pre_deletion_opening_vec.u32, gap_open_cost_vec.u32),
                                           __vaddus4(pre_deletion_expansion_vec.u32, gap_extend_cost_vec.u32));
            cell_score_vec.u32 = __vminu4(if_substitution_vec.u32, __vminu4(if_insertion_vec.u32, if_deletion_vec.u32));

            // When walking through the top-left triangle of the matrix, our output addresses are misaligned.
            scores_new[i + 0] = cell_score_vec.u8s[0];
            scores_new[i + 1] = cell_score_vec.u8s[1];
            scores_new[i + 2] = cell_score_vec.u8s[2];
            scores_new[i + 3] = cell_score_vec.u8s[3];
            scores_new_insertions[i + 0] = if_insertion_vec.u8s[0];
            scores_new_insertions[i + 1] = if_insertion_vec.u8s[1];
            scores_new_insertions[i + 2] = if_insertion_vec.u8s[2];
            scores_new_insertions[i + 3] = if_insertion_vec.u8s[3];
            scores_new_deletions[i + 0] = if_deletion_vec.u8s[0];
            scores_new_deletions[i + 1] = if_deletion_vec.u8s[1];
            scores_new_deletions[i + 2] = if_deletion_vec.u8s[2];
            scores_new_deletions[i + 3] = if_deletion_vec.u8s[3];
        }

        // Extract the bottom-right corner of the matrix, which is the result of the global alignment.
        if (tasks_offset == 0) this->final_score_ = scores_new[0];
    }
};

/**
 *  @brief GPU adaptation of the `tile_scorer` - Minimizes Global Levenshtein distance with affine gap costs.
 *  @note Requires Kepler generation GPUs to handle 2x `u16` scores at a time.
 *
 *  Relies on following instruction families to output 2x @b `u16` scores per call:
 *  - @b `prmt` to shuffle bytes in 32 bit registers.
 *  - @b `vmax2,vmin2,vadd2` video-processing instructions.
 */
template <>
struct tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
    using kepler_warp_scorer_t =
        tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                    sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>;

    __forceinline__ __device__ void operator()(            //
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
            if_insertion_vec.u32 = __vminu2(__vaddus2(pre_insertion_opening_vec.u32, gap_open_cost_vec.u32),
                                            __vaddus2(pre_insertion_expansion_vec.u32, gap_extend_cost_vec.u32));
            if_deletion_vec.u32 = __vminu2(__vaddus2(pre_deletion_opening_vec.u32, gap_open_cost_vec.u32),
                                           __vaddus2(pre_deletion_expansion_vec.u32, gap_extend_cost_vec.u32));
            cell_score_vec.u32 = __vminu2(if_substitution_vec.u32, __vminu2(if_insertion_vec.u32, if_deletion_vec.u32));

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
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {
    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
};

template <>
struct tile_scorer<char const *, char const *, u64_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, u64_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {
    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
};

#endif

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SIMILARITIES_KEPLER_CUH_
