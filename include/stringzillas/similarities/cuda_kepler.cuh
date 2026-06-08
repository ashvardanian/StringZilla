/**
 *  @file include/stringzillas/similarities/cuda_kepler.cuh
 *  @brief Kepler-tier similarity device walkers + explicit entry kernels.
 *  @author Ash Vardanian
 *  @sa include/stringzillas/similarities/cuda_maxwell.cuh
 *
 *  The Kepler packed-SIMD path (`__vminu4`/`__vminu2` over four/two u8/u16 cells per 32-bit register, with
 *  `prmt`/`__vcmpeq4` for the substitution mask) only pays off when each lane owns several contiguous cells
 *  of a diagonal; the warp-per-pair layout here strides one cell per lane, so the packed form is a
 *  follow-on perf body. Until it lands, Kepler reuses the base warp-per-pair walkers for every (op, width)
 *  - they are value-identical. The structure is a separate per-tier body (its own gated entry block calling
 *  the chosen walkers), not a shared template.
 *
 *  Tier acceleration map (this tier):
 *  - levenshtein linear/affine u8/u16/u32 - REUSE base warp walker (packed-SIMD slice is a future perf body)
 *  - needleman_wunsch linear/affine i16/i32 - REUSE base warp walker (Kepler has no gain over base for NW)
 *  - smith_waterman linear/affine i16/i32 - REUSE base warp walker (Kepler has no gain over base for SW)
 */
#ifndef STRINGZILLAS_SIMILARITIES_CUDA_KEPLER_CUH_
#define STRINGZILLAS_SIMILARITIES_CUDA_KEPLER_CUH_

#ifndef SZ_GPU_TIER
#define SZ_GPU_TIER sz_caps_ck_k
#endif

// This tier owns the (single) emitted entry surface when it is the compiled leaf, so suppress the base
// tier's entry kernels (its walkers stay available to reuse).
#define SZS_SUPPRESS_BASE_ENTRIES
#include "stringzillas/similarities/cuda_maxwell.cuh"

namespace ashvardanian {
namespace stringzillas {

#if defined(__CUDACC__)

#pragma region - Kepler Packed-SIMD Warp-Per-Pair Walkers (Levenshtein only)

// The Kepler packed video-SIMD path only pays off for the narrow Levenshtein (minimize / global) widths,
// where several DP cells fit one 32-bit register: 4x u8 via `__vminu4`/`__vaddus4`, 2x u16 via `__vminu2`/
// `__vaddus2`. Each lane owns a CONTIGUOUS group of cells of the anti-diagonal (4 for u8, 2 for u16) strided
// by `warp_size * group` across the diagonal - the multi-cell-per-lane packing for the packed video-SIMD.
// The score-diagonal neighbors and the substitution cost are gathered per element (the strings carry no
// SMEM padding here, so we avoid the original's packed unaligned char loads), then the cell recurrence runs
// packed. Because the host only selects u8/u16 widths when every DP value provably fits the cell type, the
// saturating packed adds never clamp, so the result is bit-identical to the base scalar walker. NW/SW
// (maximize), u32, and all local cases have no Kepler gain and reuse the base warp walkers.
//
// Tier acceleration map (this tier):
//   levenshtein linear/affine u8  - ACCELERATE via packed 4x u8 video-SIMD
//   levenshtein linear/affine u16 - ACCELERATE via packed 2x u16 video-SIMD
//   levenshtein linear/affine u32 - REUSE base warp walker (too wide to pack)
//   needleman_wunsch/smith_waterman linear/affine i16/i32 - REUSE base warp walker (no Kepler gain)

__device__ inline sz_i32_t szs_walk_pair_warp_kepler_u8_linear( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap = costs->gap_open;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;
    unsigned const group = 4; // 4x u8 cells per lane per step

    if (shorter_length == 0 || longer_length == 0) {
        if (is_local) return 0;
        sz_size_t const only_length = shorter_length ? shorter_length : longer_length;
        if (only_length == 0) return 0;
        return (sz_i32_t)((sz_i64_t)gap * (sz_i64_t)only_length);
    }

    unsigned const shorter_dim = (unsigned)shorter_length + 1;
    unsigned const longer_dim = (unsigned)longer_length + 1;
    unsigned const diagonals_count = shorter_dim + longer_dim - 1;
    unsigned const max_diagonal_length = (unsigned)shorter_length + 1;
    unsigned const bytes_per_diagonal = szs_round_up_to_multiple(max_diagonal_length * sizeof(sz_u8_t), 4);

    sz_u8_t *previous_scores = (sz_u8_t *)(shared_memory_for_warp + 0 * bytes_per_diagonal);
    sz_u8_t *current_scores = (sz_u8_t *)(shared_memory_for_warp + 1 * bytes_per_diagonal);
    sz_u8_t *next_scores = (sz_u8_t *)(shared_memory_for_warp + 2 * bytes_per_diagonal);
    char *longer = shared_memory_for_warp + 3 * bytes_per_diagonal;
    char *shorter = longer + longer_length;

    for (unsigned i = lane; i < longer_length; i += warp_size) longer[i] = (char)longer_global[i];
    for (unsigned i = lane; i < shorter_length; i += warp_size) shorter[i] = (char)shorter_global[i];

    sz_u32_vec_t gap_u8x4;
    gap_u8x4.u32 = (sz_u32_t)(sz_u8_t)gap * 0x01010101u;

    if (lane == 0) {
        previous_scores[0] = (sz_u8_t)szs_init_score_linear(0, gap, is_local);
        current_scores[0] = (sz_u8_t)szs_init_score_linear(1, gap, is_local);
        current_scores[1] = (sz_u8_t)szs_init_score_linear(1, gap, is_local);
    }
    __syncwarp();

    sz_i32_t best_score = 0;
    unsigned next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = next_diagonal_index + 1;
        unsigned const cells_count = next_diagonal_length - 2;
        for (unsigned base = lane * group; base < cells_count; base += warp_size * group) {
            sz_u32_vec_t pre_sub_u8x4, pre_ins_u8x4, pre_del_u8x4, cost_u8x4;
            unsigned const lanes = cells_count - base < group ? cells_count - base : group;
            for (unsigned k = 0; k < group; ++k) {
                unsigned const i = base + k;
                pre_sub_u8x4.u8s[k] = k < lanes ? previous_scores[i] : 0;
                pre_ins_u8x4.u8s[k] = k < lanes ? current_scores[i] : 0;
                pre_del_u8x4.u8s[k] = k < lanes ? current_scores[i + 1] : 0;
                cost_u8x4.u8s[k] = k < lanes
                                       ? (sz_u8_t)szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                                (sz_u32_t)(sz_u8_t)longer[i])
                                       : 0;
            }
            sz_u32_vec_t if_sub_u8x4, if_gap_u8x4, cell_u8x4;
            if_sub_u8x4.u32 = __vaddus4(pre_sub_u8x4.u32, cost_u8x4.u32);
            if_gap_u8x4.u32 = __vaddus4(__vminu4(pre_ins_u8x4.u32, pre_del_u8x4.u32), gap_u8x4.u32);
            cell_u8x4.u32 = __vminu4(if_sub_u8x4.u32, if_gap_u8x4.u32);
            for (unsigned k = 0; k < lanes; ++k) {
                next_scores[base + k + 1] = cell_u8x4.u8s[k];
                best_score = szs_pick_best_i32(best_score, (sz_i32_t)cell_u8x4.u8s[k], objective);
            }
        }
        if (lane == 0) {
            next_scores[0] = (sz_u8_t)szs_init_score_linear(next_diagonal_index, gap, is_local);
            next_scores[next_diagonal_length - 1] = (sz_u8_t)szs_init_score_linear(next_diagonal_index, gap, is_local);
        }
        __syncwarp();
        sz_u8_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
    }
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = shorter_dim;
        char const *second_slice = longer + next_diagonal_index - shorter_dim;
        unsigned const cells_count = next_diagonal_length - 1;
        for (unsigned base = lane * group; base < cells_count; base += warp_size * group) {
            sz_u32_vec_t pre_sub_u8x4, pre_ins_u8x4, pre_del_u8x4, cost_u8x4;
            unsigned const lanes = cells_count - base < group ? cells_count - base : group;
            for (unsigned k = 0; k < group; ++k) {
                unsigned const i = base + k;
                pre_sub_u8x4.u8s[k] = k < lanes ? previous_scores[i] : 0;
                pre_ins_u8x4.u8s[k] = k < lanes ? current_scores[i] : 0;
                pre_del_u8x4.u8s[k] = k < lanes ? current_scores[i + 1] : 0;
                cost_u8x4.u8s[k] = k < lanes
                                       ? (sz_u8_t)szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                                (sz_u32_t)(sz_u8_t)second_slice[i])
                                       : 0;
            }
            sz_u32_vec_t if_sub_u8x4, if_gap_u8x4, cell_u8x4;
            if_sub_u8x4.u32 = __vaddus4(pre_sub_u8x4.u32, cost_u8x4.u32);
            if_gap_u8x4.u32 = __vaddus4(__vminu4(pre_ins_u8x4.u32, pre_del_u8x4.u32), gap_u8x4.u32);
            cell_u8x4.u32 = __vminu4(if_sub_u8x4.u32, if_gap_u8x4.u32);
            for (unsigned k = 0; k < lanes; ++k) {
                next_scores[base + k] = cell_u8x4.u8s[k];
                best_score = szs_pick_best_i32(best_score, (sz_i32_t)cell_u8x4.u8s[k], objective);
            }
        }
        if (lane == 0)
            next_scores[next_diagonal_length - 1] = (sz_u8_t)szs_init_score_linear(next_diagonal_index, gap, is_local);
        __syncwarp();
        for (unsigned i = lane; i + 1 < next_diagonal_length; i += warp_size)
            previous_scores[i] = current_scores[i + 1];
        __syncwarp();
        for (unsigned i = lane; i < next_diagonal_length; i += warp_size) current_scores[i] = next_scores[i];
        __syncwarp();
    }
    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {
        unsigned const next_diagonal_length = diagonals_count - next_diagonal_index;
        char const *first_slice = shorter + next_diagonal_index - longer_dim;
        char const *second_slice = longer + next_diagonal_index - shorter_dim;
        for (unsigned base = lane * group; base < next_diagonal_length; base += warp_size * group) {
            sz_u32_vec_t pre_sub_u8x4, pre_ins_u8x4, pre_del_u8x4, cost_u8x4;
            unsigned const lanes = next_diagonal_length - base < group ? next_diagonal_length - base : group;
            for (unsigned k = 0; k < group; ++k) {
                unsigned const i = base + k;
                pre_sub_u8x4.u8s[k] = k < lanes ? previous_scores[i] : 0;
                pre_ins_u8x4.u8s[k] = k < lanes ? current_scores[i] : 0;
                pre_del_u8x4.u8s[k] = k < lanes ? current_scores[i + 1] : 0;
                cost_u8x4.u8s[k] = k < lanes ? (sz_u8_t)szs_subs_cost(
                                                   costs, (sz_u32_t)(sz_u8_t)first_slice[next_diagonal_length - i - 1],
                                                   (sz_u32_t)(sz_u8_t)second_slice[i])
                                             : 0;
            }
            sz_u32_vec_t if_sub_u8x4, if_gap_u8x4, cell_u8x4;
            if_sub_u8x4.u32 = __vaddus4(pre_sub_u8x4.u32, cost_u8x4.u32);
            if_gap_u8x4.u32 = __vaddus4(__vminu4(pre_ins_u8x4.u32, pre_del_u8x4.u32), gap_u8x4.u32);
            cell_u8x4.u32 = __vminu4(if_sub_u8x4.u32, if_gap_u8x4.u32);
            for (unsigned k = 0; k < lanes; ++k) {
                next_scores[base + k] = cell_u8x4.u8s[k];
                best_score = szs_pick_best_i32(best_score, (sz_i32_t)cell_u8x4.u8s[k], objective);
            }
        }
        sz_u8_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        previous_scores++;
        __syncwarp();
    }

    if (is_local) return szs_best_in_warp_i32(best_score, objective);
    sz_i32_t final_score = lane == 0 ? (sz_i32_t)current_scores[0] : 0;
    return __shfl_sync(0xffffffffu, final_score, 0);
}

__device__ inline sz_i32_t szs_walk_pair_warp_kepler_u16_linear( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap = costs->gap_open;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;
    unsigned const group = 2; // 2x u16 cells per lane per step

    if (shorter_length == 0 || longer_length == 0) {
        if (is_local) return 0;
        sz_size_t const only_length = shorter_length ? shorter_length : longer_length;
        if (only_length == 0) return 0;
        return (sz_i32_t)((sz_i64_t)gap * (sz_i64_t)only_length);
    }

    unsigned const shorter_dim = (unsigned)shorter_length + 1;
    unsigned const longer_dim = (unsigned)longer_length + 1;
    unsigned const diagonals_count = shorter_dim + longer_dim - 1;
    unsigned const max_diagonal_length = (unsigned)shorter_length + 1;
    unsigned const bytes_per_diagonal = szs_round_up_to_multiple(max_diagonal_length * sizeof(sz_u16_t), 4);

    sz_u16_t *previous_scores = (sz_u16_t *)(shared_memory_for_warp + 0 * bytes_per_diagonal);
    sz_u16_t *current_scores = (sz_u16_t *)(shared_memory_for_warp + 1 * bytes_per_diagonal);
    sz_u16_t *next_scores = (sz_u16_t *)(shared_memory_for_warp + 2 * bytes_per_diagonal);
    char *longer = shared_memory_for_warp + 3 * bytes_per_diagonal;
    char *shorter = longer + longer_length;

    for (unsigned i = lane; i < longer_length; i += warp_size) longer[i] = (char)longer_global[i];
    for (unsigned i = lane; i < shorter_length; i += warp_size) shorter[i] = (char)shorter_global[i];

    sz_u32_vec_t gap_u16x2;
    gap_u16x2.u32 = (sz_u32_t)(sz_u16_t)gap * 0x00010001u;

    if (lane == 0) {
        previous_scores[0] = (sz_u16_t)szs_init_score_linear(0, gap, is_local);
        current_scores[0] = (sz_u16_t)szs_init_score_linear(1, gap, is_local);
        current_scores[1] = (sz_u16_t)szs_init_score_linear(1, gap, is_local);
    }
    __syncwarp();

    sz_i32_t best_score = 0;
    unsigned next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = next_diagonal_index + 1;
        unsigned const cells_count = next_diagonal_length - 2;
        for (unsigned base = lane * group; base < cells_count; base += warp_size * group) {
            sz_u32_vec_t pre_sub_u16x2, pre_ins_u16x2, pre_del_u16x2, cost_u16x2;
            unsigned const lanes = cells_count - base < group ? cells_count - base : group;
            for (unsigned k = 0; k < group; ++k) {
                unsigned const i = base + k;
                pre_sub_u16x2.u16s[k] = k < lanes ? previous_scores[i] : 0;
                pre_ins_u16x2.u16s[k] = k < lanes ? current_scores[i] : 0;
                pre_del_u16x2.u16s[k] = k < lanes ? current_scores[i + 1] : 0;
                cost_u16x2.u16s[k] = k < lanes ? (sz_u16_t)szs_subs_cost(
                                                     costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                     (sz_u32_t)(sz_u8_t)longer[i])
                                               : 0;
            }
            sz_u32_vec_t if_sub_u16x2, if_gap_u16x2, cell_u16x2;
            if_sub_u16x2.u32 = __vaddus2(pre_sub_u16x2.u32, cost_u16x2.u32);
            if_gap_u16x2.u32 = __vaddus2(__vminu2(pre_ins_u16x2.u32, pre_del_u16x2.u32), gap_u16x2.u32);
            cell_u16x2.u32 = __vminu2(if_sub_u16x2.u32, if_gap_u16x2.u32);
            for (unsigned k = 0; k < lanes; ++k) {
                next_scores[base + k + 1] = cell_u16x2.u16s[k];
                best_score = szs_pick_best_i32(best_score, (sz_i32_t)cell_u16x2.u16s[k], objective);
            }
        }
        if (lane == 0) {
            next_scores[0] = (sz_u16_t)szs_init_score_linear(next_diagonal_index, gap, is_local);
            next_scores[next_diagonal_length - 1] = (sz_u16_t)szs_init_score_linear(next_diagonal_index, gap, is_local);
        }
        __syncwarp();
        sz_u16_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
    }
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = shorter_dim;
        char const *second_slice = longer + next_diagonal_index - shorter_dim;
        unsigned const cells_count = next_diagonal_length - 1;
        for (unsigned base = lane * group; base < cells_count; base += warp_size * group) {
            sz_u32_vec_t pre_sub_u16x2, pre_ins_u16x2, pre_del_u16x2, cost_u16x2;
            unsigned const lanes = cells_count - base < group ? cells_count - base : group;
            for (unsigned k = 0; k < group; ++k) {
                unsigned const i = base + k;
                pre_sub_u16x2.u16s[k] = k < lanes ? previous_scores[i] : 0;
                pre_ins_u16x2.u16s[k] = k < lanes ? current_scores[i] : 0;
                pre_del_u16x2.u16s[k] = k < lanes ? current_scores[i + 1] : 0;
                cost_u16x2.u16s[k] = k < lanes ? (sz_u16_t)szs_subs_cost(
                                                     costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                     (sz_u32_t)(sz_u8_t)second_slice[i])
                                               : 0;
            }
            sz_u32_vec_t if_sub_u16x2, if_gap_u16x2, cell_u16x2;
            if_sub_u16x2.u32 = __vaddus2(pre_sub_u16x2.u32, cost_u16x2.u32);
            if_gap_u16x2.u32 = __vaddus2(__vminu2(pre_ins_u16x2.u32, pre_del_u16x2.u32), gap_u16x2.u32);
            cell_u16x2.u32 = __vminu2(if_sub_u16x2.u32, if_gap_u16x2.u32);
            for (unsigned k = 0; k < lanes; ++k) {
                next_scores[base + k] = cell_u16x2.u16s[k];
                best_score = szs_pick_best_i32(best_score, (sz_i32_t)cell_u16x2.u16s[k], objective);
            }
        }
        if (lane == 0)
            next_scores[next_diagonal_length - 1] = (sz_u16_t)szs_init_score_linear(next_diagonal_index, gap, is_local);
        __syncwarp();
        for (unsigned i = lane; i + 1 < next_diagonal_length; i += warp_size)
            previous_scores[i] = current_scores[i + 1];
        __syncwarp();
        for (unsigned i = lane; i < next_diagonal_length; i += warp_size) current_scores[i] = next_scores[i];
        __syncwarp();
    }
    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {
        unsigned const next_diagonal_length = diagonals_count - next_diagonal_index;
        char const *first_slice = shorter + next_diagonal_index - longer_dim;
        char const *second_slice = longer + next_diagonal_index - shorter_dim;
        for (unsigned base = lane * group; base < next_diagonal_length; base += warp_size * group) {
            sz_u32_vec_t pre_sub_u16x2, pre_ins_u16x2, pre_del_u16x2, cost_u16x2;
            unsigned const lanes = next_diagonal_length - base < group ? next_diagonal_length - base : group;
            for (unsigned k = 0; k < group; ++k) {
                unsigned const i = base + k;
                pre_sub_u16x2.u16s[k] = k < lanes ? previous_scores[i] : 0;
                pre_ins_u16x2.u16s[k] = k < lanes ? current_scores[i] : 0;
                pre_del_u16x2.u16s[k] = k < lanes ? current_scores[i + 1] : 0;
                cost_u16x2.u16s[k] = k < lanes
                                         ? (sz_u16_t)szs_subs_cost(
                                               costs, (sz_u32_t)(sz_u8_t)first_slice[next_diagonal_length - i - 1],
                                               (sz_u32_t)(sz_u8_t)second_slice[i])
                                         : 0;
            }
            sz_u32_vec_t if_sub_u16x2, if_gap_u16x2, cell_u16x2;
            if_sub_u16x2.u32 = __vaddus2(pre_sub_u16x2.u32, cost_u16x2.u32);
            if_gap_u16x2.u32 = __vaddus2(__vminu2(pre_ins_u16x2.u32, pre_del_u16x2.u32), gap_u16x2.u32);
            cell_u16x2.u32 = __vminu2(if_sub_u16x2.u32, if_gap_u16x2.u32);
            for (unsigned k = 0; k < lanes; ++k) {
                next_scores[base + k] = cell_u16x2.u16s[k];
                best_score = szs_pick_best_i32(best_score, (sz_i32_t)cell_u16x2.u16s[k], objective);
            }
        }
        sz_u16_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        previous_scores++;
        __syncwarp();
    }

    if (is_local) return szs_best_in_warp_i32(best_score, objective);
    sz_i32_t final_score = lane == 0 ? (sz_i32_t)current_scores[0] : 0;
    return __shfl_sync(0xffffffffu, final_score, 0);
}

// Affine packed walkers: same multi-cell-per-lane packing, but with the Gotoh gap tracks. Each cell is
// `min(if_substitution, if_insertion, if_deletion)` where the two tracks fold open vs extend per element via
// packed `__vminu*`/`__vaddus*` affine form. Seven rotating diagonals.

__device__ inline sz_i32_t szs_walk_pair_warp_kepler_u8_affine( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap_open = costs->gap_open, gap_extend = costs->gap_extend;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;
    unsigned const group = 4;

    if (shorter_length == 0 || longer_length == 0) {
        if (is_local) return 0;
        sz_size_t const only_length = shorter_length ? shorter_length : longer_length;
        if (only_length == 0) return 0;
        return (sz_i32_t)((sz_i64_t)gap_open + (sz_i64_t)gap_extend * (sz_i64_t)(only_length - 1));
    }

    unsigned const shorter_dim = (unsigned)shorter_length + 1;
    unsigned const longer_dim = (unsigned)longer_length + 1;
    unsigned const diagonals_count = shorter_dim + longer_dim - 1;
    unsigned const max_diagonal_length = (unsigned)shorter_length + 1;
    unsigned const bytes_per_diagonal = szs_round_up_to_multiple(max_diagonal_length * sizeof(sz_u8_t), 4);

    sz_u8_t *previous_scores = (sz_u8_t *)(shared_memory_for_warp + 0 * bytes_per_diagonal);
    sz_u8_t *current_scores = (sz_u8_t *)(shared_memory_for_warp + 1 * bytes_per_diagonal);
    sz_u8_t *next_scores = (sz_u8_t *)(shared_memory_for_warp + 2 * bytes_per_diagonal);
    sz_u8_t *current_inserts = (sz_u8_t *)(shared_memory_for_warp + 3 * bytes_per_diagonal);
    sz_u8_t *next_inserts = (sz_u8_t *)(shared_memory_for_warp + 4 * bytes_per_diagonal);
    sz_u8_t *current_deletes = (sz_u8_t *)(shared_memory_for_warp + 5 * bytes_per_diagonal);
    sz_u8_t *next_deletes = (sz_u8_t *)(shared_memory_for_warp + 6 * bytes_per_diagonal);
    char *longer = shared_memory_for_warp + 7 * bytes_per_diagonal;
    char *shorter = longer + longer_length;

    for (unsigned i = lane; i < longer_length; i += warp_size) longer[i] = (char)longer_global[i];
    for (unsigned i = lane; i < shorter_length; i += warp_size) shorter[i] = (char)shorter_global[i];

    sz_u32_vec_t gap_open_u8x4, gap_extend_u8x4;
    gap_open_u8x4.u32 = (sz_u32_t)(sz_u8_t)gap_open * 0x01010101u;
    gap_extend_u8x4.u32 = (sz_u32_t)(sz_u8_t)gap_extend * 0x01010101u;

    if (lane == 0) {
        previous_scores[0] = (sz_u8_t)szs_init_score_affine(0, gap_open, gap_extend, is_local);
        current_scores[0] = (sz_u8_t)szs_init_score_affine(1, gap_open, gap_extend, is_local);
        current_scores[1] = (sz_u8_t)szs_init_score_affine(1, gap_open, gap_extend, is_local);
        current_inserts[0] = (sz_u8_t)szs_init_gap_affine(1, gap_open, gap_extend, is_local);
        current_deletes[1] = (sz_u8_t)szs_init_gap_affine(1, gap_open, gap_extend, is_local);
    }
    __syncwarp();

    sz_i32_t best_score = 0;
    unsigned next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = next_diagonal_index + 1;
        unsigned const cells_count = next_diagonal_length - 2;
        for (unsigned base = lane * group; base < cells_count; base += warp_size * group) {
            sz_u32_vec_t pre_sub_u8x4, ins_open_u8x4, ins_run_u8x4, del_open_u8x4, del_run_u8x4, cost_u8x4;
            unsigned const lanes = cells_count - base < group ? cells_count - base : group;
            for (unsigned k = 0; k < group; ++k) {
                unsigned const i = base + k;
                pre_sub_u8x4.u8s[k] = k < lanes ? previous_scores[i] : 0;
                ins_open_u8x4.u8s[k] = k < lanes ? current_scores[i] : 0;
                ins_run_u8x4.u8s[k] = k < lanes ? current_inserts[i] : 0;
                del_open_u8x4.u8s[k] = k < lanes ? current_scores[i + 1] : 0;
                del_run_u8x4.u8s[k] = k < lanes ? current_deletes[i + 1] : 0;
                cost_u8x4.u8s[k] = k < lanes
                                       ? (sz_u8_t)szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                                (sz_u32_t)(sz_u8_t)longer[i])
                                       : 0;
            }
            sz_u32_vec_t if_sub_u8x4, if_ins_u8x4, if_del_u8x4, cell_u8x4;
            if_sub_u8x4.u32 = __vaddus4(pre_sub_u8x4.u32, cost_u8x4.u32);
            if_ins_u8x4.u32 = __vminu4(__vaddus4(ins_open_u8x4.u32, gap_open_u8x4.u32),
                                       __vaddus4(ins_run_u8x4.u32, gap_extend_u8x4.u32));
            if_del_u8x4.u32 = __vminu4(__vaddus4(del_open_u8x4.u32, gap_open_u8x4.u32),
                                       __vaddus4(del_run_u8x4.u32, gap_extend_u8x4.u32));
            cell_u8x4.u32 = __vminu4(if_sub_u8x4.u32, __vminu4(if_ins_u8x4.u32, if_del_u8x4.u32));
            for (unsigned k = 0; k < lanes; ++k) {
                next_inserts[base + k + 1] = if_ins_u8x4.u8s[k];
                next_deletes[base + k + 1] = if_del_u8x4.u8s[k];
                next_scores[base + k + 1] = cell_u8x4.u8s[k];
                best_score = szs_pick_best_i32(best_score, (sz_i32_t)cell_u8x4.u8s[k], objective);
            }
        }
        if (lane == 0) {
            next_scores[0] = (sz_u8_t)szs_init_score_affine(next_diagonal_index, gap_open, gap_extend, is_local);
            next_scores[next_diagonal_length - 1] = (sz_u8_t)szs_init_score_affine(next_diagonal_index, gap_open,
                                                                                   gap_extend, is_local);
            next_inserts[0] = (sz_u8_t)szs_init_gap_affine(next_diagonal_index, gap_open, gap_extend, is_local);
            next_deletes[next_diagonal_length - 1] = (sz_u8_t)szs_init_gap_affine(next_diagonal_index, gap_open,
                                                                                  gap_extend, is_local);
        }
        __syncwarp();
        sz_u8_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        sz_u8_t *swap = current_inserts;
        current_inserts = next_inserts;
        next_inserts = swap;
        swap = current_deletes;
        current_deletes = next_deletes;
        next_deletes = swap;
    }
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = shorter_dim;
        char const *second_slice = longer + next_diagonal_index - shorter_dim;
        unsigned const cells_count = next_diagonal_length - 1;
        for (unsigned base = lane * group; base < cells_count; base += warp_size * group) {
            sz_u32_vec_t pre_sub_u8x4, ins_open_u8x4, ins_run_u8x4, del_open_u8x4, del_run_u8x4, cost_u8x4;
            unsigned const lanes = cells_count - base < group ? cells_count - base : group;
            for (unsigned k = 0; k < group; ++k) {
                unsigned const i = base + k;
                pre_sub_u8x4.u8s[k] = k < lanes ? previous_scores[i] : 0;
                ins_open_u8x4.u8s[k] = k < lanes ? current_scores[i] : 0;
                ins_run_u8x4.u8s[k] = k < lanes ? current_inserts[i] : 0;
                del_open_u8x4.u8s[k] = k < lanes ? current_scores[i + 1] : 0;
                del_run_u8x4.u8s[k] = k < lanes ? current_deletes[i + 1] : 0;
                cost_u8x4.u8s[k] = k < lanes
                                       ? (sz_u8_t)szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                                (sz_u32_t)(sz_u8_t)second_slice[i])
                                       : 0;
            }
            sz_u32_vec_t if_sub_u8x4, if_ins_u8x4, if_del_u8x4, cell_u8x4;
            if_sub_u8x4.u32 = __vaddus4(pre_sub_u8x4.u32, cost_u8x4.u32);
            if_ins_u8x4.u32 = __vminu4(__vaddus4(ins_open_u8x4.u32, gap_open_u8x4.u32),
                                       __vaddus4(ins_run_u8x4.u32, gap_extend_u8x4.u32));
            if_del_u8x4.u32 = __vminu4(__vaddus4(del_open_u8x4.u32, gap_open_u8x4.u32),
                                       __vaddus4(del_run_u8x4.u32, gap_extend_u8x4.u32));
            cell_u8x4.u32 = __vminu4(if_sub_u8x4.u32, __vminu4(if_ins_u8x4.u32, if_del_u8x4.u32));
            for (unsigned k = 0; k < lanes; ++k) {
                next_inserts[base + k] = if_ins_u8x4.u8s[k];
                next_deletes[base + k] = if_del_u8x4.u8s[k];
                next_scores[base + k] = cell_u8x4.u8s[k];
                best_score = szs_pick_best_i32(best_score, (sz_i32_t)cell_u8x4.u8s[k], objective);
            }
        }
        if (lane == 0) {
            next_scores[next_diagonal_length - 1] = (sz_u8_t)szs_init_score_affine(next_diagonal_index, gap_open,
                                                                                   gap_extend, is_local);
            next_deletes[next_diagonal_length - 1] = (sz_u8_t)szs_init_gap_affine(next_diagonal_index, gap_open,
                                                                                  gap_extend, is_local);
        }
        sz_u8_t *swap = current_inserts;
        current_inserts = next_inserts;
        next_inserts = swap;
        swap = current_deletes;
        current_deletes = next_deletes;
        next_deletes = swap;
        __syncwarp();
        for (unsigned i = lane; i + 1 < next_diagonal_length; i += warp_size)
            previous_scores[i] = current_scores[i + 1];
        __syncwarp();
        for (unsigned i = lane; i < next_diagonal_length; i += warp_size) current_scores[i] = next_scores[i];
        __syncwarp();
    }
    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {
        unsigned const next_diagonal_length = diagonals_count - next_diagonal_index;
        char const *first_slice = shorter + next_diagonal_index - longer_dim;
        char const *second_slice = longer + next_diagonal_index - shorter_dim;
        for (unsigned base = lane * group; base < next_diagonal_length; base += warp_size * group) {
            sz_u32_vec_t pre_sub_u8x4, ins_open_u8x4, ins_run_u8x4, del_open_u8x4, del_run_u8x4, cost_u8x4;
            unsigned const lanes = next_diagonal_length - base < group ? next_diagonal_length - base : group;
            for (unsigned k = 0; k < group; ++k) {
                unsigned const i = base + k;
                pre_sub_u8x4.u8s[k] = k < lanes ? previous_scores[i] : 0;
                ins_open_u8x4.u8s[k] = k < lanes ? current_scores[i] : 0;
                ins_run_u8x4.u8s[k] = k < lanes ? current_inserts[i] : 0;
                del_open_u8x4.u8s[k] = k < lanes ? current_scores[i + 1] : 0;
                del_run_u8x4.u8s[k] = k < lanes ? current_deletes[i + 1] : 0;
                cost_u8x4.u8s[k] = k < lanes ? (sz_u8_t)szs_subs_cost(
                                                   costs, (sz_u32_t)(sz_u8_t)first_slice[next_diagonal_length - i - 1],
                                                   (sz_u32_t)(sz_u8_t)second_slice[i])
                                             : 0;
            }
            sz_u32_vec_t if_sub_u8x4, if_ins_u8x4, if_del_u8x4, cell_u8x4;
            if_sub_u8x4.u32 = __vaddus4(pre_sub_u8x4.u32, cost_u8x4.u32);
            if_ins_u8x4.u32 = __vminu4(__vaddus4(ins_open_u8x4.u32, gap_open_u8x4.u32),
                                       __vaddus4(ins_run_u8x4.u32, gap_extend_u8x4.u32));
            if_del_u8x4.u32 = __vminu4(__vaddus4(del_open_u8x4.u32, gap_open_u8x4.u32),
                                       __vaddus4(del_run_u8x4.u32, gap_extend_u8x4.u32));
            cell_u8x4.u32 = __vminu4(if_sub_u8x4.u32, __vminu4(if_ins_u8x4.u32, if_del_u8x4.u32));
            for (unsigned k = 0; k < lanes; ++k) {
                next_inserts[base + k] = if_ins_u8x4.u8s[k];
                next_deletes[base + k] = if_del_u8x4.u8s[k];
                next_scores[base + k] = cell_u8x4.u8s[k];
                best_score = szs_pick_best_i32(best_score, (sz_i32_t)cell_u8x4.u8s[k], objective);
            }
        }
        sz_u8_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        sz_u8_t *swap = current_inserts;
        current_inserts = next_inserts;
        next_inserts = swap;
        swap = current_deletes;
        current_deletes = next_deletes;
        next_deletes = swap;
        previous_scores++;
        __syncwarp();
    }

    if (is_local) return szs_best_in_warp_i32(best_score, objective);
    sz_i32_t final_score = lane == 0 ? (sz_i32_t)current_scores[0] : 0;
    return __shfl_sync(0xffffffffu, final_score, 0);
}

__device__ inline sz_i32_t szs_walk_pair_warp_kepler_u16_affine( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap_open = costs->gap_open, gap_extend = costs->gap_extend;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;
    unsigned const group = 2;

    if (shorter_length == 0 || longer_length == 0) {
        if (is_local) return 0;
        sz_size_t const only_length = shorter_length ? shorter_length : longer_length;
        if (only_length == 0) return 0;
        return (sz_i32_t)((sz_i64_t)gap_open + (sz_i64_t)gap_extend * (sz_i64_t)(only_length - 1));
    }

    unsigned const shorter_dim = (unsigned)shorter_length + 1;
    unsigned const longer_dim = (unsigned)longer_length + 1;
    unsigned const diagonals_count = shorter_dim + longer_dim - 1;
    unsigned const max_diagonal_length = (unsigned)shorter_length + 1;
    unsigned const bytes_per_diagonal = szs_round_up_to_multiple(max_diagonal_length * sizeof(sz_u16_t), 4);

    sz_u16_t *previous_scores = (sz_u16_t *)(shared_memory_for_warp + 0 * bytes_per_diagonal);
    sz_u16_t *current_scores = (sz_u16_t *)(shared_memory_for_warp + 1 * bytes_per_diagonal);
    sz_u16_t *next_scores = (sz_u16_t *)(shared_memory_for_warp + 2 * bytes_per_diagonal);
    sz_u16_t *current_inserts = (sz_u16_t *)(shared_memory_for_warp + 3 * bytes_per_diagonal);
    sz_u16_t *next_inserts = (sz_u16_t *)(shared_memory_for_warp + 4 * bytes_per_diagonal);
    sz_u16_t *current_deletes = (sz_u16_t *)(shared_memory_for_warp + 5 * bytes_per_diagonal);
    sz_u16_t *next_deletes = (sz_u16_t *)(shared_memory_for_warp + 6 * bytes_per_diagonal);
    char *longer = shared_memory_for_warp + 7 * bytes_per_diagonal;
    char *shorter = longer + longer_length;

    for (unsigned i = lane; i < longer_length; i += warp_size) longer[i] = (char)longer_global[i];
    for (unsigned i = lane; i < shorter_length; i += warp_size) shorter[i] = (char)shorter_global[i];

    sz_u32_vec_t gap_open_u16x2, gap_extend_u16x2;
    gap_open_u16x2.u32 = (sz_u32_t)(sz_u16_t)gap_open * 0x00010001u;
    gap_extend_u16x2.u32 = (sz_u32_t)(sz_u16_t)gap_extend * 0x00010001u;

    if (lane == 0) {
        previous_scores[0] = (sz_u16_t)szs_init_score_affine(0, gap_open, gap_extend, is_local);
        current_scores[0] = (sz_u16_t)szs_init_score_affine(1, gap_open, gap_extend, is_local);
        current_scores[1] = (sz_u16_t)szs_init_score_affine(1, gap_open, gap_extend, is_local);
        current_inserts[0] = (sz_u16_t)szs_init_gap_affine(1, gap_open, gap_extend, is_local);
        current_deletes[1] = (sz_u16_t)szs_init_gap_affine(1, gap_open, gap_extend, is_local);
    }
    __syncwarp();

    sz_i32_t best_score = 0;
    unsigned next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = next_diagonal_index + 1;
        unsigned const cells_count = next_diagonal_length - 2;
        for (unsigned base = lane * group; base < cells_count; base += warp_size * group) {
            sz_u32_vec_t pre_sub_u16x2, ins_open_u16x2, ins_run_u16x2, del_open_u16x2, del_run_u16x2, cost_u16x2;
            unsigned const lanes = cells_count - base < group ? cells_count - base : group;
            for (unsigned k = 0; k < group; ++k) {
                unsigned const i = base + k;
                pre_sub_u16x2.u16s[k] = k < lanes ? previous_scores[i] : 0;
                ins_open_u16x2.u16s[k] = k < lanes ? current_scores[i] : 0;
                ins_run_u16x2.u16s[k] = k < lanes ? current_inserts[i] : 0;
                del_open_u16x2.u16s[k] = k < lanes ? current_scores[i + 1] : 0;
                del_run_u16x2.u16s[k] = k < lanes ? current_deletes[i + 1] : 0;
                cost_u16x2.u16s[k] = k < lanes ? (sz_u16_t)szs_subs_cost(
                                                     costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                     (sz_u32_t)(sz_u8_t)longer[i])
                                               : 0;
            }
            sz_u32_vec_t if_sub_u16x2, if_ins_u16x2, if_del_u16x2, cell_u16x2;
            if_sub_u16x2.u32 = __vaddus2(pre_sub_u16x2.u32, cost_u16x2.u32);
            if_ins_u16x2.u32 = __vminu2(__vaddus2(ins_open_u16x2.u32, gap_open_u16x2.u32),
                                        __vaddus2(ins_run_u16x2.u32, gap_extend_u16x2.u32));
            if_del_u16x2.u32 = __vminu2(__vaddus2(del_open_u16x2.u32, gap_open_u16x2.u32),
                                        __vaddus2(del_run_u16x2.u32, gap_extend_u16x2.u32));
            cell_u16x2.u32 = __vminu2(if_sub_u16x2.u32, __vminu2(if_ins_u16x2.u32, if_del_u16x2.u32));
            for (unsigned k = 0; k < lanes; ++k) {
                next_inserts[base + k + 1] = if_ins_u16x2.u16s[k];
                next_deletes[base + k + 1] = if_del_u16x2.u16s[k];
                next_scores[base + k + 1] = cell_u16x2.u16s[k];
                best_score = szs_pick_best_i32(best_score, (sz_i32_t)cell_u16x2.u16s[k], objective);
            }
        }
        if (lane == 0) {
            next_scores[0] = (sz_u16_t)szs_init_score_affine(next_diagonal_index, gap_open, gap_extend, is_local);
            next_scores[next_diagonal_length - 1] = (sz_u16_t)szs_init_score_affine(next_diagonal_index, gap_open,
                                                                                    gap_extend, is_local);
            next_inserts[0] = (sz_u16_t)szs_init_gap_affine(next_diagonal_index, gap_open, gap_extend, is_local);
            next_deletes[next_diagonal_length - 1] = (sz_u16_t)szs_init_gap_affine(next_diagonal_index, gap_open,
                                                                                   gap_extend, is_local);
        }
        __syncwarp();
        sz_u16_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        sz_u16_t *swap = current_inserts;
        current_inserts = next_inserts;
        next_inserts = swap;
        swap = current_deletes;
        current_deletes = next_deletes;
        next_deletes = swap;
    }
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = shorter_dim;
        char const *second_slice = longer + next_diagonal_index - shorter_dim;
        unsigned const cells_count = next_diagonal_length - 1;
        for (unsigned base = lane * group; base < cells_count; base += warp_size * group) {
            sz_u32_vec_t pre_sub_u16x2, ins_open_u16x2, ins_run_u16x2, del_open_u16x2, del_run_u16x2, cost_u16x2;
            unsigned const lanes = cells_count - base < group ? cells_count - base : group;
            for (unsigned k = 0; k < group; ++k) {
                unsigned const i = base + k;
                pre_sub_u16x2.u16s[k] = k < lanes ? previous_scores[i] : 0;
                ins_open_u16x2.u16s[k] = k < lanes ? current_scores[i] : 0;
                ins_run_u16x2.u16s[k] = k < lanes ? current_inserts[i] : 0;
                del_open_u16x2.u16s[k] = k < lanes ? current_scores[i + 1] : 0;
                del_run_u16x2.u16s[k] = k < lanes ? current_deletes[i + 1] : 0;
                cost_u16x2.u16s[k] = k < lanes ? (sz_u16_t)szs_subs_cost(
                                                     costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                     (sz_u32_t)(sz_u8_t)second_slice[i])
                                               : 0;
            }
            sz_u32_vec_t if_sub_u16x2, if_ins_u16x2, if_del_u16x2, cell_u16x2;
            if_sub_u16x2.u32 = __vaddus2(pre_sub_u16x2.u32, cost_u16x2.u32);
            if_ins_u16x2.u32 = __vminu2(__vaddus2(ins_open_u16x2.u32, gap_open_u16x2.u32),
                                        __vaddus2(ins_run_u16x2.u32, gap_extend_u16x2.u32));
            if_del_u16x2.u32 = __vminu2(__vaddus2(del_open_u16x2.u32, gap_open_u16x2.u32),
                                        __vaddus2(del_run_u16x2.u32, gap_extend_u16x2.u32));
            cell_u16x2.u32 = __vminu2(if_sub_u16x2.u32, __vminu2(if_ins_u16x2.u32, if_del_u16x2.u32));
            for (unsigned k = 0; k < lanes; ++k) {
                next_inserts[base + k] = if_ins_u16x2.u16s[k];
                next_deletes[base + k] = if_del_u16x2.u16s[k];
                next_scores[base + k] = cell_u16x2.u16s[k];
                best_score = szs_pick_best_i32(best_score, (sz_i32_t)cell_u16x2.u16s[k], objective);
            }
        }
        if (lane == 0) {
            next_scores[next_diagonal_length - 1] = (sz_u16_t)szs_init_score_affine(next_diagonal_index, gap_open,
                                                                                    gap_extend, is_local);
            next_deletes[next_diagonal_length - 1] = (sz_u16_t)szs_init_gap_affine(next_diagonal_index, gap_open,
                                                                                   gap_extend, is_local);
        }
        sz_u16_t *swap = current_inserts;
        current_inserts = next_inserts;
        next_inserts = swap;
        swap = current_deletes;
        current_deletes = next_deletes;
        next_deletes = swap;
        __syncwarp();
        for (unsigned i = lane; i + 1 < next_diagonal_length; i += warp_size)
            previous_scores[i] = current_scores[i + 1];
        __syncwarp();
        for (unsigned i = lane; i < next_diagonal_length; i += warp_size) current_scores[i] = next_scores[i];
        __syncwarp();
    }
    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {
        unsigned const next_diagonal_length = diagonals_count - next_diagonal_index;
        char const *first_slice = shorter + next_diagonal_index - longer_dim;
        char const *second_slice = longer + next_diagonal_index - shorter_dim;
        for (unsigned base = lane * group; base < next_diagonal_length; base += warp_size * group) {
            sz_u32_vec_t pre_sub_u16x2, ins_open_u16x2, ins_run_u16x2, del_open_u16x2, del_run_u16x2, cost_u16x2;
            unsigned const lanes = next_diagonal_length - base < group ? next_diagonal_length - base : group;
            for (unsigned k = 0; k < group; ++k) {
                unsigned const i = base + k;
                pre_sub_u16x2.u16s[k] = k < lanes ? previous_scores[i] : 0;
                ins_open_u16x2.u16s[k] = k < lanes ? current_scores[i] : 0;
                ins_run_u16x2.u16s[k] = k < lanes ? current_inserts[i] : 0;
                del_open_u16x2.u16s[k] = k < lanes ? current_scores[i + 1] : 0;
                del_run_u16x2.u16s[k] = k < lanes ? current_deletes[i + 1] : 0;
                cost_u16x2.u16s[k] = k < lanes
                                         ? (sz_u16_t)szs_subs_cost(
                                               costs, (sz_u32_t)(sz_u8_t)first_slice[next_diagonal_length - i - 1],
                                               (sz_u32_t)(sz_u8_t)second_slice[i])
                                         : 0;
            }
            sz_u32_vec_t if_sub_u16x2, if_ins_u16x2, if_del_u16x2, cell_u16x2;
            if_sub_u16x2.u32 = __vaddus2(pre_sub_u16x2.u32, cost_u16x2.u32);
            if_ins_u16x2.u32 = __vminu2(__vaddus2(ins_open_u16x2.u32, gap_open_u16x2.u32),
                                        __vaddus2(ins_run_u16x2.u32, gap_extend_u16x2.u32));
            if_del_u16x2.u32 = __vminu2(__vaddus2(del_open_u16x2.u32, gap_open_u16x2.u32),
                                        __vaddus2(del_run_u16x2.u32, gap_extend_u16x2.u32));
            cell_u16x2.u32 = __vminu2(if_sub_u16x2.u32, __vminu2(if_ins_u16x2.u32, if_del_u16x2.u32));
            for (unsigned k = 0; k < lanes; ++k) {
                next_inserts[base + k] = if_ins_u16x2.u16s[k];
                next_deletes[base + k] = if_del_u16x2.u16s[k];
                next_scores[base + k] = cell_u16x2.u16s[k];
                best_score = szs_pick_best_i32(best_score, (sz_i32_t)cell_u16x2.u16s[k], objective);
            }
        }
        sz_u16_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        sz_u16_t *swap = current_inserts;
        current_inserts = next_inserts;
        next_inserts = swap;
        swap = current_deletes;
        current_deletes = next_deletes;
        next_deletes = swap;
        previous_scores++;
        __syncwarp();
    }

    if (is_local) return szs_best_in_warp_i32(best_score, objective);
    sz_i32_t final_score = lane == 0 ? (sz_i32_t)current_scores[0] : 0;
    return __shfl_sync(0xffffffffu, final_score, 0);
}

#pragma endregion

#if !defined(SZS_SUPPRESS_KEPLER_ENTRIES)

#pragma region - Kepler Warp-Per-Pair Entry Kernels (emitted only when this is the compiled tier)

// Same 14 (op x gap x width) entry surface the host resolver looks up. Each stages the cost model into
// shared memory via the base helper and calls the base warp walker (no Kepler-specific body landed yet).

extern "C" __global__ void szs_levenshtein_warp_linear_u8(szs_pair_task_t *tasks, unsigned long long tasks_count,
                                                          szs_device_costs_t costs, unsigned shared_memory_per_warp) {
    szs_warp_context_t ctx = szs_make_warp_context(costs, shared_memory_per_warp);
    for (unsigned long long t = ctx.global_warp_index; t < tasks_count; t += ctx.warps_per_device) {
        szs_pair_task_t &task = tasks[t];
        sz_size_t sl = task.a_len, ll = task.b_len;
        sz_u8_t const *sp = task.a_ptr, *lp = task.b_ptr;
        if (sl > ll) {
            sl = task.b_len;
            ll = task.a_len;
            sp = task.b_ptr;
            lp = task.a_ptr;
        }
        sz_i32_t v = szs_walk_pair_warp_kepler_u8_linear(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
        szs_warp_write_back(task, v, ctx.lane);
    }
}

extern "C" __global__ void szs_levenshtein_warp_linear_u16(szs_pair_task_t *tasks, unsigned long long tasks_count,
                                                           szs_device_costs_t costs, unsigned shared_memory_per_warp) {
    szs_warp_context_t ctx = szs_make_warp_context(costs, shared_memory_per_warp);
    for (unsigned long long t = ctx.global_warp_index; t < tasks_count; t += ctx.warps_per_device) {
        szs_pair_task_t &task = tasks[t];
        sz_size_t sl = task.a_len, ll = task.b_len;
        sz_u8_t const *sp = task.a_ptr, *lp = task.b_ptr;
        if (sl > ll) {
            sl = task.b_len;
            ll = task.a_len;
            sp = task.b_ptr;
            lp = task.a_ptr;
        }
        sz_i32_t v = szs_walk_pair_warp_kepler_u16_linear(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
        szs_warp_write_back(task, v, ctx.lane);
    }
}

extern "C" __global__ void szs_levenshtein_warp_linear_u32(szs_pair_task_t *tasks, unsigned long long tasks_count,
                                                           szs_device_costs_t costs, unsigned shared_memory_per_warp) {
    szs_warp_context_t ctx = szs_make_warp_context(costs, shared_memory_per_warp);
    for (unsigned long long t = ctx.global_warp_index; t < tasks_count; t += ctx.warps_per_device) {
        szs_pair_task_t &task = tasks[t];
        sz_size_t sl = task.a_len, ll = task.b_len;
        sz_u8_t const *sp = task.a_ptr, *lp = task.b_ptr;
        if (sl > ll) {
            sl = task.b_len;
            ll = task.a_len;
            sp = task.b_ptr;
            lp = task.a_ptr;
        }
        sz_i32_t v = szs_walk_pair_warp_base_u32(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
        szs_warp_write_back(task, v, ctx.lane);
    }
}

extern "C" __global__ void szs_levenshtein_warp_affine_u8(szs_pair_task_t *tasks, unsigned long long tasks_count,
                                                          szs_device_costs_t costs, unsigned shared_memory_per_warp) {
    szs_warp_context_t ctx = szs_make_warp_context(costs, shared_memory_per_warp);
    for (unsigned long long t = ctx.global_warp_index; t < tasks_count; t += ctx.warps_per_device) {
        szs_pair_task_t &task = tasks[t];
        sz_size_t sl = task.a_len, ll = task.b_len;
        sz_u8_t const *sp = task.a_ptr, *lp = task.b_ptr;
        if (sl > ll) {
            sl = task.b_len;
            ll = task.a_len;
            sp = task.b_ptr;
            lp = task.a_ptr;
        }
        sz_i32_t v = szs_walk_pair_warp_kepler_u8_affine(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
        szs_warp_write_back(task, v, ctx.lane);
    }
}

extern "C" __global__ void szs_levenshtein_warp_affine_u16(szs_pair_task_t *tasks, unsigned long long tasks_count,
                                                           szs_device_costs_t costs, unsigned shared_memory_per_warp) {
    szs_warp_context_t ctx = szs_make_warp_context(costs, shared_memory_per_warp);
    for (unsigned long long t = ctx.global_warp_index; t < tasks_count; t += ctx.warps_per_device) {
        szs_pair_task_t &task = tasks[t];
        sz_size_t sl = task.a_len, ll = task.b_len;
        sz_u8_t const *sp = task.a_ptr, *lp = task.b_ptr;
        if (sl > ll) {
            sl = task.b_len;
            ll = task.a_len;
            sp = task.b_ptr;
            lp = task.a_ptr;
        }
        sz_i32_t v = szs_walk_pair_warp_kepler_u16_affine(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
        szs_warp_write_back(task, v, ctx.lane);
    }
}

extern "C" __global__ void szs_levenshtein_warp_affine_u32(szs_pair_task_t *tasks, unsigned long long tasks_count,
                                                           szs_device_costs_t costs, unsigned shared_memory_per_warp) {
    szs_warp_context_t ctx = szs_make_warp_context(costs, shared_memory_per_warp);
    for (unsigned long long t = ctx.global_warp_index; t < tasks_count; t += ctx.warps_per_device) {
        szs_pair_task_t &task = tasks[t];
        sz_size_t sl = task.a_len, ll = task.b_len;
        sz_u8_t const *sp = task.a_ptr, *lp = task.b_ptr;
        if (sl > ll) {
            sl = task.b_len;
            ll = task.a_len;
            sp = task.b_ptr;
            lp = task.a_ptr;
        }
        sz_i32_t v = szs_walk_pair_warp_base_u32(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
        szs_warp_write_back(task, v, ctx.lane);
    }
}

extern "C" __global__ void szs_needleman_wunsch_warp_linear_i16(szs_pair_task_t *tasks, unsigned long long tasks_count,
                                                                szs_device_costs_t costs, unsigned shared_memory_per_warp) {
    szs_warp_context_t ctx = szs_make_warp_context(costs, shared_memory_per_warp);
    for (unsigned long long t = ctx.global_warp_index; t < tasks_count; t += ctx.warps_per_device) {
        szs_pair_task_t &task = tasks[t];
        sz_size_t sl = task.a_len, ll = task.b_len;
        sz_u8_t const *sp = task.a_ptr, *lp = task.b_ptr;
        if (sl > ll) {
            sl = task.b_len;
            ll = task.a_len;
            sp = task.b_ptr;
            lp = task.a_ptr;
        }
        sz_i32_t v = szs_walk_pair_warp_base_i16(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
        szs_warp_write_back(task, v, ctx.lane);
    }
}

extern "C" __global__ void szs_needleman_wunsch_warp_linear_i32(szs_pair_task_t *tasks, unsigned long long tasks_count,
                                                                szs_device_costs_t costs, unsigned shared_memory_per_warp) {
    szs_warp_context_t ctx = szs_make_warp_context(costs, shared_memory_per_warp);
    for (unsigned long long t = ctx.global_warp_index; t < tasks_count; t += ctx.warps_per_device) {
        szs_pair_task_t &task = tasks[t];
        sz_size_t sl = task.a_len, ll = task.b_len;
        sz_u8_t const *sp = task.a_ptr, *lp = task.b_ptr;
        if (sl > ll) {
            sl = task.b_len;
            ll = task.a_len;
            sp = task.b_ptr;
            lp = task.a_ptr;
        }
        sz_i32_t v = szs_walk_pair_warp_base_i32(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
        szs_warp_write_back(task, v, ctx.lane);
    }
}

extern "C" __global__ void szs_needleman_wunsch_warp_affine_i16(szs_pair_task_t *tasks, unsigned long long tasks_count,
                                                                szs_device_costs_t costs, unsigned shared_memory_per_warp) {
    szs_warp_context_t ctx = szs_make_warp_context(costs, shared_memory_per_warp);
    for (unsigned long long t = ctx.global_warp_index; t < tasks_count; t += ctx.warps_per_device) {
        szs_pair_task_t &task = tasks[t];
        sz_size_t sl = task.a_len, ll = task.b_len;
        sz_u8_t const *sp = task.a_ptr, *lp = task.b_ptr;
        if (sl > ll) {
            sl = task.b_len;
            ll = task.a_len;
            sp = task.b_ptr;
            lp = task.a_ptr;
        }
        sz_i32_t v = szs_walk_pair_warp_base_i16(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
        szs_warp_write_back(task, v, ctx.lane);
    }
}

extern "C" __global__ void szs_needleman_wunsch_warp_affine_i32(szs_pair_task_t *tasks, unsigned long long tasks_count,
                                                                szs_device_costs_t costs, unsigned shared_memory_per_warp) {
    szs_warp_context_t ctx = szs_make_warp_context(costs, shared_memory_per_warp);
    for (unsigned long long t = ctx.global_warp_index; t < tasks_count; t += ctx.warps_per_device) {
        szs_pair_task_t &task = tasks[t];
        sz_size_t sl = task.a_len, ll = task.b_len;
        sz_u8_t const *sp = task.a_ptr, *lp = task.b_ptr;
        if (sl > ll) {
            sl = task.b_len;
            ll = task.a_len;
            sp = task.b_ptr;
            lp = task.a_ptr;
        }
        sz_i32_t v = szs_walk_pair_warp_base_i32(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
        szs_warp_write_back(task, v, ctx.lane);
    }
}

extern "C" __global__ void szs_smith_waterman_warp_linear_i16(szs_pair_task_t *tasks, unsigned long long tasks_count,
                                                              szs_device_costs_t costs, unsigned shared_memory_per_warp) {
    szs_warp_context_t ctx = szs_make_warp_context(costs, shared_memory_per_warp);
    for (unsigned long long t = ctx.global_warp_index; t < tasks_count; t += ctx.warps_per_device) {
        szs_pair_task_t &task = tasks[t];
        sz_size_t sl = task.a_len, ll = task.b_len;
        sz_u8_t const *sp = task.a_ptr, *lp = task.b_ptr;
        if (sl > ll) {
            sl = task.b_len;
            ll = task.a_len;
            sp = task.b_ptr;
            lp = task.a_ptr;
        }
        sz_i32_t v = szs_walk_pair_warp_base_i16(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
        szs_warp_write_back(task, v, ctx.lane);
    }
}

extern "C" __global__ void szs_smith_waterman_warp_linear_i32(szs_pair_task_t *tasks, unsigned long long tasks_count,
                                                              szs_device_costs_t costs, unsigned shared_memory_per_warp) {
    szs_warp_context_t ctx = szs_make_warp_context(costs, shared_memory_per_warp);
    for (unsigned long long t = ctx.global_warp_index; t < tasks_count; t += ctx.warps_per_device) {
        szs_pair_task_t &task = tasks[t];
        sz_size_t sl = task.a_len, ll = task.b_len;
        sz_u8_t const *sp = task.a_ptr, *lp = task.b_ptr;
        if (sl > ll) {
            sl = task.b_len;
            ll = task.a_len;
            sp = task.b_ptr;
            lp = task.a_ptr;
        }
        sz_i32_t v = szs_walk_pair_warp_base_i32(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
        szs_warp_write_back(task, v, ctx.lane);
    }
}

extern "C" __global__ void szs_smith_waterman_warp_affine_i16(szs_pair_task_t *tasks, unsigned long long tasks_count,
                                                              szs_device_costs_t costs, unsigned shared_memory_per_warp) {
    szs_warp_context_t ctx = szs_make_warp_context(costs, shared_memory_per_warp);
    for (unsigned long long t = ctx.global_warp_index; t < tasks_count; t += ctx.warps_per_device) {
        szs_pair_task_t &task = tasks[t];
        sz_size_t sl = task.a_len, ll = task.b_len;
        sz_u8_t const *sp = task.a_ptr, *lp = task.b_ptr;
        if (sl > ll) {
            sl = task.b_len;
            ll = task.a_len;
            sp = task.b_ptr;
            lp = task.a_ptr;
        }
        sz_i32_t v = szs_walk_pair_warp_base_i16(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
        szs_warp_write_back(task, v, ctx.lane);
    }
}

extern "C" __global__ void szs_smith_waterman_warp_affine_i32(szs_pair_task_t *tasks, unsigned long long tasks_count,
                                                              szs_device_costs_t costs, unsigned shared_memory_per_warp) {
    szs_warp_context_t ctx = szs_make_warp_context(costs, shared_memory_per_warp);
    for (unsigned long long t = ctx.global_warp_index; t < tasks_count; t += ctx.warps_per_device) {
        szs_pair_task_t &task = tasks[t];
        sz_size_t sl = task.a_len, ll = task.b_len;
        sz_u8_t const *sp = task.a_ptr, *lp = task.b_ptr;
        if (sl > ll) {
            sl = task.b_len;
            ll = task.a_len;
            sp = task.b_ptr;
            lp = task.a_ptr;
        }
        sz_i32_t v = szs_walk_pair_warp_base_i32(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
        szs_warp_write_back(task, v, ctx.lane);
    }
}

#pragma endregion

#endif // !SZS_SUPPRESS_KEPLER_ENTRIES

#endif // __CUDACC__

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SIMILARITIES_CUDA_KEPLER_CUH_
