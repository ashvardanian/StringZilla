/**
 *  @file include/stringzillas/similarities/cuda_hopper.cuh
 *  @brief Hopper-tier (DPX) similarity device walkers + explicit entry kernels.
 *  @author Ash Vardanian
 *  @sa include/stringzillas/similarities/cuda_maxwell.cuh
 *
 *  Hopper exposes DPX (Dynamic Programming X) integer instructions; the ones that map cleanly onto the DP
 *  recurrence are the 3-input fused add-then-min/max on s32 (`__viaddmin_s32` / `__viaddmax_s32`, computing
 *  `min/max(a + b, c)`) and the 3-way min/max (`__vimin3_s32` / `__vimax3_s32`, with `_relu` clamping to 0
 *  for local). The linear-gap cell is two add-min/max; the affine (Gotoh) cell folds each gap track with
 *  one add-min/max and then takes the 3-way min/max of substitution, insert, delete. So the Hopper tier
 *  ships a DPX linear-cell helper + DPX affine-cell helper and warp-per-pair DPX walkers for both gap models
 *  per width. The result is bit-for-bit the same i32 arithmetic as the base scalar cells, so the golden
 *  values are unchanged. At PTX level the non-relu DPX intrinsics fold into `add`+`min/max`; the relu
 *  variants emit `min/max.s32.relu`; all of them lower to `VIADDMNMX`/`VIMNMX3` SASS on sm_90.
 *
 *  Structurally the warp walkers are the base warp walkers with the inner cell routed through the DPX
 *  helper; the warp cooperation, SMEM layout, `__syncwarp` cadence, and final reduction are identical to
 *  `cuda_maxwell.cuh`.
 *
 *  Tier acceleration map (this tier):
 *  - levenshtein/needleman_wunsch/smith_waterman LINEAR u8/u16/u32/i16/i32 - ACCELERATE via DPX add-min/max
 *  - levenshtein/needleman_wunsch/smith_waterman AFFINE u8/u16/u32/i16/i32 - ACCELERATE via DPX add-min/max + 3-way
 */
#ifndef STRINGZILLAS_SIMILARITIES_CUDA_HOPPER_CUH_
#define STRINGZILLAS_SIMILARITIES_CUDA_HOPPER_CUH_

#ifndef SZ_GPU_TIER
#define SZ_GPU_TIER sz_caps_ckh_k
#endif

// Hopper is the top tier, so it owns the (single) emitted entry surface: suppress both the base and the
// Kepler entry kernels (their walkers stay available; Hopper reuses the base affine warp walkers).
#define SZS_SUPPRESS_BASE_ENTRIES
#define SZS_SUPPRESS_KEPLER_ENTRIES
#include "stringzillas/similarities/cuda_kepler.cuh"

namespace ashvardanian {
namespace stringzillas {

#if defined(__CUDACC__)

#pragma region - Hopper DPX Linear Cell + Warp-Per-Pair Linear Walkers

// One linear-gap DP cell using Hopper DPX fused add-then-min/max. Value-identical to `szs_cell_linear_i32`:
//   minimize: min(pre_sub + sub, pre_ins + gap, pre_del + gap)
//   maximize: max(pre_sub + sub, pre_ins + gap, pre_del + gap)  (clamped to 0 for local on maximize)
__device__ __forceinline__ sz_i32_t szs_cell_linear_dpx_i32( //
    sz_i32_t pre_sub, sz_i32_t pre_ins, sz_i32_t pre_del,    //
    sz_error_cost_t sub_cost, sz_error_cost_t gap,           //
    sz_similarity_objective_t obj, int is_local) {
    sz_i32_t best;
    if (obj == sz_maximize_score_k) {
        sz_i32_t folded = __viaddmax_s32(pre_del, (sz_i32_t)gap, pre_sub + (sz_i32_t)sub_cost);
        best = __viaddmax_s32(pre_ins, (sz_i32_t)gap, folded);
        if (is_local && best < 0) best = 0;
    }
    else {
        sz_i32_t folded = __viaddmin_s32(pre_del, (sz_i32_t)gap, pre_sub + (sz_i32_t)sub_cost);
        best = __viaddmin_s32(pre_ins, (sz_i32_t)gap, folded);
    }
    return best;
}

// One affine-gap (Gotoh) DP cell using Hopper DPX fused add-then-min/max + 3-way min/max. Value-identical
// to `szs_cell_affine_i32`: the insert/delete tracks are `best(open_main + gap_open, run_gap + gap_extend)`
// folded by `__viaddmin/max_s32`, and the cell is the 3-way `__vimin3/__vimax3_s32` of substitution and the
// two tracks (relu variant clamps to 0 for local maximize, matching the base scalar cell). This is a faithful
// affine DPX form (`__viaddmax_s16x2`/`__vimax3_s32` etc.) in the i32 walker.
__device__ __forceinline__ sz_i32_t szs_cell_affine_dpx_i32(                        //
    sz_i32_t pre_sub, sz_i32_t up_main, sz_i32_t up_ins,                            //
    sz_i32_t left_main, sz_i32_t left_del,                                          //
    sz_error_cost_t sub_cost, sz_error_cost_t gap_open, sz_error_cost_t gap_extend, //
    sz_similarity_objective_t obj, int is_local, sz_i32_t *out_ins, sz_i32_t *out_del) {
    sz_i32_t ins, del, best;
    if (obj == sz_maximize_score_k) {
        ins = __viaddmax_s32(up_main, (sz_i32_t)gap_open, up_ins + (sz_i32_t)gap_extend);
        del = __viaddmax_s32(left_main, (sz_i32_t)gap_open, left_del + (sz_i32_t)gap_extend);
        if (is_local) best = __vimax3_s32_relu(pre_sub + (sz_i32_t)sub_cost, ins, del);
        else best = __vimax3_s32(pre_sub + (sz_i32_t)sub_cost, ins, del);
    }
    else {
        ins = __viaddmin_s32(up_main, (sz_i32_t)gap_open, up_ins + (sz_i32_t)gap_extend);
        del = __viaddmin_s32(left_main, (sz_i32_t)gap_open, left_del + (sz_i32_t)gap_extend);
        best = __vimin3_s32(pre_sub + (sz_i32_t)sub_cost, ins, del);
    }
    *out_ins = ins;
    *out_del = del;
    return best;
}

// szs_walk_pair_warp_hopper_<width>_linear: one warp cooperatively scores one LINEAR-gap pair, routing each
// inner cell through the DPX helper. Three rotating score diagonals + both strings in the per-warp SMEM
// slice; cells strided across 32 lanes; `__syncwarp` between diagonals; final reduction matches the base.

__device__ inline sz_i32_t szs_walk_pair_warp_hopper_u8_linear( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap = costs->gap_open;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;

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
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)longer[i]);
            sz_i32_t cell = szs_cell_linear_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            next_scores[i + 1] = (sz_u8_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
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
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t cell = szs_cell_linear_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            next_scores[i] = (sz_u8_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
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
        for (unsigned i = lane; i < next_diagonal_length; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_slice[next_diagonal_length - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t cell = szs_cell_linear_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            next_scores[i] = (sz_u8_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
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

__device__ inline sz_i32_t szs_walk_pair_warp_hopper_u16_linear( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap = costs->gap_open;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;

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
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)longer[i]);
            sz_i32_t cell = szs_cell_linear_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            next_scores[i + 1] = (sz_u16_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
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
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t cell = szs_cell_linear_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            next_scores[i] = (sz_u16_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
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
        for (unsigned i = lane; i < next_diagonal_length; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_slice[next_diagonal_length - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t cell = szs_cell_linear_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            next_scores[i] = (sz_u16_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
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

__device__ inline sz_i32_t szs_walk_pair_warp_hopper_u32_linear( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap = costs->gap_open;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;

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
    unsigned const bytes_per_diagonal = szs_round_up_to_multiple(max_diagonal_length * sizeof(sz_u32_t), 4);

    sz_u32_t *previous_scores = (sz_u32_t *)(shared_memory_for_warp + 0 * bytes_per_diagonal);
    sz_u32_t *current_scores = (sz_u32_t *)(shared_memory_for_warp + 1 * bytes_per_diagonal);
    sz_u32_t *next_scores = (sz_u32_t *)(shared_memory_for_warp + 2 * bytes_per_diagonal);
    char *longer = shared_memory_for_warp + 3 * bytes_per_diagonal;
    char *shorter = longer + longer_length;

    for (unsigned i = lane; i < longer_length; i += warp_size) longer[i] = (char)longer_global[i];
    for (unsigned i = lane; i < shorter_length; i += warp_size) shorter[i] = (char)shorter_global[i];

    if (lane == 0) {
        previous_scores[0] = (sz_u32_t)szs_init_score_linear(0, gap, is_local);
        current_scores[0] = (sz_u32_t)szs_init_score_linear(1, gap, is_local);
        current_scores[1] = (sz_u32_t)szs_init_score_linear(1, gap, is_local);
    }
    __syncwarp();

    sz_i32_t best_score = 0;
    unsigned next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = next_diagonal_index + 1;
        unsigned const cells_count = next_diagonal_length - 2;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)longer[i]);
            sz_i32_t cell = szs_cell_linear_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            next_scores[i + 1] = (sz_u32_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            next_scores[0] = (sz_u32_t)szs_init_score_linear(next_diagonal_index, gap, is_local);
            next_scores[next_diagonal_length - 1] = (sz_u32_t)szs_init_score_linear(next_diagonal_index, gap, is_local);
        }
        __syncwarp();
        sz_u32_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
    }
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = shorter_dim;
        char const *second_slice = longer + next_diagonal_index - shorter_dim;
        unsigned const cells_count = next_diagonal_length - 1;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t cell = szs_cell_linear_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            next_scores[i] = (sz_u32_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0)
            next_scores[next_diagonal_length - 1] = (sz_u32_t)szs_init_score_linear(next_diagonal_index, gap, is_local);
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
        for (unsigned i = lane; i < next_diagonal_length; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_slice[next_diagonal_length - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t cell = szs_cell_linear_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            next_scores[i] = (sz_u32_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        sz_u32_t *rotated = previous_scores;
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

__device__ inline sz_i32_t szs_walk_pair_warp_hopper_i16_linear( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap = costs->gap_open;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;

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
    unsigned const bytes_per_diagonal = szs_round_up_to_multiple(max_diagonal_length * sizeof(sz_i16_t), 4);

    sz_i16_t *previous_scores = (sz_i16_t *)(shared_memory_for_warp + 0 * bytes_per_diagonal);
    sz_i16_t *current_scores = (sz_i16_t *)(shared_memory_for_warp + 1 * bytes_per_diagonal);
    sz_i16_t *next_scores = (sz_i16_t *)(shared_memory_for_warp + 2 * bytes_per_diagonal);
    char *longer = shared_memory_for_warp + 3 * bytes_per_diagonal;
    char *shorter = longer + longer_length;

    for (unsigned i = lane; i < longer_length; i += warp_size) longer[i] = (char)longer_global[i];
    for (unsigned i = lane; i < shorter_length; i += warp_size) shorter[i] = (char)shorter_global[i];

    if (lane == 0) {
        previous_scores[0] = (sz_i16_t)szs_init_score_linear(0, gap, is_local);
        current_scores[0] = (sz_i16_t)szs_init_score_linear(1, gap, is_local);
        current_scores[1] = (sz_i16_t)szs_init_score_linear(1, gap, is_local);
    }
    __syncwarp();

    sz_i32_t best_score = 0;
    unsigned next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = next_diagonal_index + 1;
        unsigned const cells_count = next_diagonal_length - 2;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)longer[i]);
            sz_i32_t cell = szs_cell_linear_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            next_scores[i + 1] = (sz_i16_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            next_scores[0] = (sz_i16_t)szs_init_score_linear(next_diagonal_index, gap, is_local);
            next_scores[next_diagonal_length - 1] = (sz_i16_t)szs_init_score_linear(next_diagonal_index, gap, is_local);
        }
        __syncwarp();
        sz_i16_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
    }
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = shorter_dim;
        char const *second_slice = longer + next_diagonal_index - shorter_dim;
        unsigned const cells_count = next_diagonal_length - 1;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t cell = szs_cell_linear_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            next_scores[i] = (sz_i16_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0)
            next_scores[next_diagonal_length - 1] = (sz_i16_t)szs_init_score_linear(next_diagonal_index, gap, is_local);
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
        for (unsigned i = lane; i < next_diagonal_length; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_slice[next_diagonal_length - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t cell = szs_cell_linear_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            next_scores[i] = (sz_i16_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        sz_i16_t *rotated = previous_scores;
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

__device__ inline sz_i32_t szs_walk_pair_warp_hopper_i32_linear( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap = costs->gap_open;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;

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
    unsigned const bytes_per_diagonal = szs_round_up_to_multiple(max_diagonal_length * sizeof(sz_i32_t), 4);

    sz_i32_t *previous_scores = (sz_i32_t *)(shared_memory_for_warp + 0 * bytes_per_diagonal);
    sz_i32_t *current_scores = (sz_i32_t *)(shared_memory_for_warp + 1 * bytes_per_diagonal);
    sz_i32_t *next_scores = (sz_i32_t *)(shared_memory_for_warp + 2 * bytes_per_diagonal);
    char *longer = shared_memory_for_warp + 3 * bytes_per_diagonal;
    char *shorter = longer + longer_length;

    for (unsigned i = lane; i < longer_length; i += warp_size) longer[i] = (char)longer_global[i];
    for (unsigned i = lane; i < shorter_length; i += warp_size) shorter[i] = (char)shorter_global[i];

    if (lane == 0) {
        previous_scores[0] = szs_init_score_linear(0, gap, is_local);
        current_scores[0] = szs_init_score_linear(1, gap, is_local);
        current_scores[1] = szs_init_score_linear(1, gap, is_local);
    }
    __syncwarp();

    sz_i32_t best_score = 0;
    unsigned next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = next_diagonal_index + 1;
        unsigned const cells_count = next_diagonal_length - 2;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)longer[i]);
            sz_i32_t cell = szs_cell_linear_dpx_i32(previous_scores[i], current_scores[i], current_scores[i + 1], sub,
                                                    gap, objective, is_local);
            next_scores[i + 1] = cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            next_scores[0] = szs_init_score_linear(next_diagonal_index, gap, is_local);
            next_scores[next_diagonal_length - 1] = szs_init_score_linear(next_diagonal_index, gap, is_local);
        }
        __syncwarp();
        sz_i32_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
    }
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = shorter_dim;
        char const *second_slice = longer + next_diagonal_index - shorter_dim;
        unsigned const cells_count = next_diagonal_length - 1;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t cell = szs_cell_linear_dpx_i32(previous_scores[i], current_scores[i], current_scores[i + 1], sub,
                                                    gap, objective, is_local);
            next_scores[i] = cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0)
            next_scores[next_diagonal_length - 1] = szs_init_score_linear(next_diagonal_index, gap, is_local);
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
        for (unsigned i = lane; i < next_diagonal_length; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_slice[next_diagonal_length - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t cell = szs_cell_linear_dpx_i32(previous_scores[i], current_scores[i], current_scores[i + 1], sub,
                                                    gap, objective, is_local);
            next_scores[i] = cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        sz_i32_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        previous_scores++;
        __syncwarp();
    }

    if (is_local) return szs_best_in_warp_i32(best_score, objective);
    sz_i32_t final_score = lane == 0 ? current_scores[0] : 0;
    return __shfl_sync(0xffffffffu, final_score, 0);
}

#pragma endregion

#pragma region - Hopper DPX Affine Cell Warp-Per-Pair Walkers

// szs_walk_pair_warp_hopper_<width>_affine: one warp cooperatively scores one AFFINE-gap (Gotoh) pair,
// routing each inner cell through the DPX affine helper. Seven rotating diagonals (3 scores + 2 insert + 2
// delete tracks) + both strings in the per-warp SMEM slice; cells strided across 32 lanes; `__syncwarp`
// between diagonals. Layout, cadence, and reduction are identical to `szs_walk_pair_warp_base_*` affine -
// only the cell op differs - so the result is value-identical to the base affine walker.

__device__ inline sz_i32_t szs_walk_pair_warp_hopper_u8_affine( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap_open = costs->gap_open, gap_extend = costs->gap_extend;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;

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
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)longer[i]);
            sz_i32_t out_ins, out_del;
            sz_i32_t cell = szs_cell_affine_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                                    (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend,
                                                    objective, is_local, &out_ins, &out_del);
            next_inserts[i + 1] = (sz_u8_t)out_ins;
            next_deletes[i + 1] = (sz_u8_t)out_del;
            next_scores[i + 1] = (sz_u8_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
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
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t out_ins, out_del;
            sz_i32_t cell = szs_cell_affine_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                                    (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend,
                                                    objective, is_local, &out_ins, &out_del);
            next_inserts[i] = (sz_u8_t)out_ins;
            next_deletes[i] = (sz_u8_t)out_del;
            next_scores[i] = (sz_u8_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
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
        for (unsigned i = lane; i < next_diagonal_length; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_slice[next_diagonal_length - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t out_ins, out_del;
            sz_i32_t cell = szs_cell_affine_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                                    (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend,
                                                    objective, is_local, &out_ins, &out_del);
            next_inserts[i] = (sz_u8_t)out_ins;
            next_deletes[i] = (sz_u8_t)out_del;
            next_scores[i] = (sz_u8_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
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

__device__ inline sz_i32_t szs_walk_pair_warp_hopper_u16_affine( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap_open = costs->gap_open, gap_extend = costs->gap_extend;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;

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
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)longer[i]);
            sz_i32_t out_ins, out_del;
            sz_i32_t cell = szs_cell_affine_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                                    (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend,
                                                    objective, is_local, &out_ins, &out_del);
            next_inserts[i + 1] = (sz_u16_t)out_ins;
            next_deletes[i + 1] = (sz_u16_t)out_del;
            next_scores[i + 1] = (sz_u16_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
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
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t out_ins, out_del;
            sz_i32_t cell = szs_cell_affine_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                                    (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend,
                                                    objective, is_local, &out_ins, &out_del);
            next_inserts[i] = (sz_u16_t)out_ins;
            next_deletes[i] = (sz_u16_t)out_del;
            next_scores[i] = (sz_u16_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
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
        for (unsigned i = lane; i < next_diagonal_length; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_slice[next_diagonal_length - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t out_ins, out_del;
            sz_i32_t cell = szs_cell_affine_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                                    (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend,
                                                    objective, is_local, &out_ins, &out_del);
            next_inserts[i] = (sz_u16_t)out_ins;
            next_deletes[i] = (sz_u16_t)out_del;
            next_scores[i] = (sz_u16_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
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

__device__ inline sz_i32_t szs_walk_pair_warp_hopper_u32_affine( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap_open = costs->gap_open, gap_extend = costs->gap_extend;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;

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
    unsigned const bytes_per_diagonal = szs_round_up_to_multiple(max_diagonal_length * sizeof(sz_u32_t), 4);

    sz_u32_t *previous_scores = (sz_u32_t *)(shared_memory_for_warp + 0 * bytes_per_diagonal);
    sz_u32_t *current_scores = (sz_u32_t *)(shared_memory_for_warp + 1 * bytes_per_diagonal);
    sz_u32_t *next_scores = (sz_u32_t *)(shared_memory_for_warp + 2 * bytes_per_diagonal);
    sz_u32_t *current_inserts = (sz_u32_t *)(shared_memory_for_warp + 3 * bytes_per_diagonal);
    sz_u32_t *next_inserts = (sz_u32_t *)(shared_memory_for_warp + 4 * bytes_per_diagonal);
    sz_u32_t *current_deletes = (sz_u32_t *)(shared_memory_for_warp + 5 * bytes_per_diagonal);
    sz_u32_t *next_deletes = (sz_u32_t *)(shared_memory_for_warp + 6 * bytes_per_diagonal);
    char *longer = shared_memory_for_warp + 7 * bytes_per_diagonal;
    char *shorter = longer + longer_length;

    for (unsigned i = lane; i < longer_length; i += warp_size) longer[i] = (char)longer_global[i];
    for (unsigned i = lane; i < shorter_length; i += warp_size) shorter[i] = (char)shorter_global[i];

    if (lane == 0) {
        previous_scores[0] = (sz_u32_t)szs_init_score_affine(0, gap_open, gap_extend, is_local);
        current_scores[0] = (sz_u32_t)szs_init_score_affine(1, gap_open, gap_extend, is_local);
        current_scores[1] = (sz_u32_t)szs_init_score_affine(1, gap_open, gap_extend, is_local);
        current_inserts[0] = (sz_u32_t)szs_init_gap_affine(1, gap_open, gap_extend, is_local);
        current_deletes[1] = (sz_u32_t)szs_init_gap_affine(1, gap_open, gap_extend, is_local);
    }
    __syncwarp();

    sz_i32_t best_score = 0;
    unsigned next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = next_diagonal_index + 1;
        unsigned const cells_count = next_diagonal_length - 2;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)longer[i]);
            sz_i32_t out_ins, out_del;
            sz_i32_t cell = szs_cell_affine_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                                    (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend,
                                                    objective, is_local, &out_ins, &out_del);
            next_inserts[i + 1] = (sz_u32_t)out_ins;
            next_deletes[i + 1] = (sz_u32_t)out_del;
            next_scores[i + 1] = (sz_u32_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            next_scores[0] = (sz_u32_t)szs_init_score_affine(next_diagonal_index, gap_open, gap_extend, is_local);
            next_scores[next_diagonal_length - 1] = (sz_u32_t)szs_init_score_affine(next_diagonal_index, gap_open,
                                                                                    gap_extend, is_local);
            next_inserts[0] = (sz_u32_t)szs_init_gap_affine(next_diagonal_index, gap_open, gap_extend, is_local);
            next_deletes[next_diagonal_length - 1] = (sz_u32_t)szs_init_gap_affine(next_diagonal_index, gap_open,
                                                                                   gap_extend, is_local);
        }
        __syncwarp();
        sz_u32_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        sz_u32_t *swap = current_inserts;
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
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t out_ins, out_del;
            sz_i32_t cell = szs_cell_affine_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                                    (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend,
                                                    objective, is_local, &out_ins, &out_del);
            next_inserts[i] = (sz_u32_t)out_ins;
            next_deletes[i] = (sz_u32_t)out_del;
            next_scores[i] = (sz_u32_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            next_scores[next_diagonal_length - 1] = (sz_u32_t)szs_init_score_affine(next_diagonal_index, gap_open,
                                                                                    gap_extend, is_local);
            next_deletes[next_diagonal_length - 1] = (sz_u32_t)szs_init_gap_affine(next_diagonal_index, gap_open,
                                                                                   gap_extend, is_local);
        }
        sz_u32_t *swap = current_inserts;
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
        for (unsigned i = lane; i < next_diagonal_length; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_slice[next_diagonal_length - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t out_ins, out_del;
            sz_i32_t cell = szs_cell_affine_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                                    (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend,
                                                    objective, is_local, &out_ins, &out_del);
            next_inserts[i] = (sz_u32_t)out_ins;
            next_deletes[i] = (sz_u32_t)out_del;
            next_scores[i] = (sz_u32_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        sz_u32_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        sz_u32_t *swap = current_inserts;
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

__device__ inline sz_i32_t szs_walk_pair_warp_hopper_i16_affine( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap_open = costs->gap_open, gap_extend = costs->gap_extend;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;

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
    unsigned const bytes_per_diagonal = szs_round_up_to_multiple(max_diagonal_length * sizeof(sz_i16_t), 4);

    sz_i16_t *previous_scores = (sz_i16_t *)(shared_memory_for_warp + 0 * bytes_per_diagonal);
    sz_i16_t *current_scores = (sz_i16_t *)(shared_memory_for_warp + 1 * bytes_per_diagonal);
    sz_i16_t *next_scores = (sz_i16_t *)(shared_memory_for_warp + 2 * bytes_per_diagonal);
    sz_i16_t *current_inserts = (sz_i16_t *)(shared_memory_for_warp + 3 * bytes_per_diagonal);
    sz_i16_t *next_inserts = (sz_i16_t *)(shared_memory_for_warp + 4 * bytes_per_diagonal);
    sz_i16_t *current_deletes = (sz_i16_t *)(shared_memory_for_warp + 5 * bytes_per_diagonal);
    sz_i16_t *next_deletes = (sz_i16_t *)(shared_memory_for_warp + 6 * bytes_per_diagonal);
    char *longer = shared_memory_for_warp + 7 * bytes_per_diagonal;
    char *shorter = longer + longer_length;

    for (unsigned i = lane; i < longer_length; i += warp_size) longer[i] = (char)longer_global[i];
    for (unsigned i = lane; i < shorter_length; i += warp_size) shorter[i] = (char)shorter_global[i];

    if (lane == 0) {
        previous_scores[0] = (sz_i16_t)szs_init_score_affine(0, gap_open, gap_extend, is_local);
        current_scores[0] = (sz_i16_t)szs_init_score_affine(1, gap_open, gap_extend, is_local);
        current_scores[1] = (sz_i16_t)szs_init_score_affine(1, gap_open, gap_extend, is_local);
        current_inserts[0] = (sz_i16_t)szs_init_gap_affine(1, gap_open, gap_extend, is_local);
        current_deletes[1] = (sz_i16_t)szs_init_gap_affine(1, gap_open, gap_extend, is_local);
    }
    __syncwarp();

    sz_i32_t best_score = 0;
    unsigned next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = next_diagonal_index + 1;
        unsigned const cells_count = next_diagonal_length - 2;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)longer[i]);
            sz_i32_t out_ins, out_del;
            sz_i32_t cell = szs_cell_affine_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                                    (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend,
                                                    objective, is_local, &out_ins, &out_del);
            next_inserts[i + 1] = (sz_i16_t)out_ins;
            next_deletes[i + 1] = (sz_i16_t)out_del;
            next_scores[i + 1] = (sz_i16_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            next_scores[0] = (sz_i16_t)szs_init_score_affine(next_diagonal_index, gap_open, gap_extend, is_local);
            next_scores[next_diagonal_length - 1] = (sz_i16_t)szs_init_score_affine(next_diagonal_index, gap_open,
                                                                                    gap_extend, is_local);
            next_inserts[0] = (sz_i16_t)szs_init_gap_affine(next_diagonal_index, gap_open, gap_extend, is_local);
            next_deletes[next_diagonal_length - 1] = (sz_i16_t)szs_init_gap_affine(next_diagonal_index, gap_open,
                                                                                   gap_extend, is_local);
        }
        __syncwarp();
        sz_i16_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        sz_i16_t *swap = current_inserts;
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
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t out_ins, out_del;
            sz_i32_t cell = szs_cell_affine_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                                    (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend,
                                                    objective, is_local, &out_ins, &out_del);
            next_inserts[i] = (sz_i16_t)out_ins;
            next_deletes[i] = (sz_i16_t)out_del;
            next_scores[i] = (sz_i16_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            next_scores[next_diagonal_length - 1] = (sz_i16_t)szs_init_score_affine(next_diagonal_index, gap_open,
                                                                                    gap_extend, is_local);
            next_deletes[next_diagonal_length - 1] = (sz_i16_t)szs_init_gap_affine(next_diagonal_index, gap_open,
                                                                                   gap_extend, is_local);
        }
        sz_i16_t *swap = current_inserts;
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
        for (unsigned i = lane; i < next_diagonal_length; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_slice[next_diagonal_length - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t out_ins, out_del;
            sz_i32_t cell = szs_cell_affine_dpx_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                                    (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                                    (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend,
                                                    objective, is_local, &out_ins, &out_del);
            next_inserts[i] = (sz_i16_t)out_ins;
            next_deletes[i] = (sz_i16_t)out_del;
            next_scores[i] = (sz_i16_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        sz_i16_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        sz_i16_t *swap = current_inserts;
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

__device__ inline sz_i32_t szs_walk_pair_warp_hopper_i32_affine( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap_open = costs->gap_open, gap_extend = costs->gap_extend;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;

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
    unsigned const bytes_per_diagonal = szs_round_up_to_multiple(max_diagonal_length * sizeof(sz_i32_t), 4);

    sz_i32_t *previous_scores = (sz_i32_t *)(shared_memory_for_warp + 0 * bytes_per_diagonal);
    sz_i32_t *current_scores = (sz_i32_t *)(shared_memory_for_warp + 1 * bytes_per_diagonal);
    sz_i32_t *next_scores = (sz_i32_t *)(shared_memory_for_warp + 2 * bytes_per_diagonal);
    sz_i32_t *current_inserts = (sz_i32_t *)(shared_memory_for_warp + 3 * bytes_per_diagonal);
    sz_i32_t *next_inserts = (sz_i32_t *)(shared_memory_for_warp + 4 * bytes_per_diagonal);
    sz_i32_t *current_deletes = (sz_i32_t *)(shared_memory_for_warp + 5 * bytes_per_diagonal);
    sz_i32_t *next_deletes = (sz_i32_t *)(shared_memory_for_warp + 6 * bytes_per_diagonal);
    char *longer = shared_memory_for_warp + 7 * bytes_per_diagonal;
    char *shorter = longer + longer_length;

    for (unsigned i = lane; i < longer_length; i += warp_size) longer[i] = (char)longer_global[i];
    for (unsigned i = lane; i < shorter_length; i += warp_size) shorter[i] = (char)shorter_global[i];

    if (lane == 0) {
        previous_scores[0] = szs_init_score_affine(0, gap_open, gap_extend, is_local);
        current_scores[0] = szs_init_score_affine(1, gap_open, gap_extend, is_local);
        current_scores[1] = szs_init_score_affine(1, gap_open, gap_extend, is_local);
        current_inserts[0] = szs_init_gap_affine(1, gap_open, gap_extend, is_local);
        current_deletes[1] = szs_init_gap_affine(1, gap_open, gap_extend, is_local);
    }
    __syncwarp();

    sz_i32_t best_score = 0;
    unsigned next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = next_diagonal_index + 1;
        unsigned const cells_count = next_diagonal_length - 2;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)longer[i]);
            sz_i32_t out_ins, out_del;
            sz_i32_t cell = szs_cell_affine_dpx_i32(previous_scores[i], current_scores[i], current_inserts[i],
                                                    current_scores[i + 1], current_deletes[i + 1], sub, gap_open,
                                                    gap_extend, objective, is_local, &out_ins, &out_del);
            next_inserts[i + 1] = out_ins;
            next_deletes[i + 1] = out_del;
            next_scores[i + 1] = cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            next_scores[0] = szs_init_score_affine(next_diagonal_index, gap_open, gap_extend, is_local);
            next_scores[next_diagonal_length - 1] = szs_init_score_affine(next_diagonal_index, gap_open, gap_extend,
                                                                          is_local);
            next_inserts[0] = szs_init_gap_affine(next_diagonal_index, gap_open, gap_extend, is_local);
            next_deletes[next_diagonal_length - 1] = szs_init_gap_affine(next_diagonal_index, gap_open, gap_extend,
                                                                         is_local);
        }
        __syncwarp();
        sz_i32_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        sz_i32_t *swap = current_inserts;
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
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)shorter[cells_count - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t out_ins, out_del;
            sz_i32_t cell = szs_cell_affine_dpx_i32(previous_scores[i], current_scores[i], current_inserts[i],
                                                    current_scores[i + 1], current_deletes[i + 1], sub, gap_open,
                                                    gap_extend, objective, is_local, &out_ins, &out_del);
            next_inserts[i] = out_ins;
            next_deletes[i] = out_del;
            next_scores[i] = cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            next_scores[next_diagonal_length - 1] = szs_init_score_affine(next_diagonal_index, gap_open, gap_extend,
                                                                          is_local);
            next_deletes[next_diagonal_length - 1] = szs_init_gap_affine(next_diagonal_index, gap_open, gap_extend,
                                                                         is_local);
        }
        sz_i32_t *swap = current_inserts;
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
        for (unsigned i = lane; i < next_diagonal_length; i += warp_size) {
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_slice[next_diagonal_length - i - 1],
                                                (sz_u32_t)(sz_u8_t)second_slice[i]);
            sz_i32_t out_ins, out_del;
            sz_i32_t cell = szs_cell_affine_dpx_i32(previous_scores[i], current_scores[i], current_inserts[i],
                                                    current_scores[i + 1], current_deletes[i + 1], sub, gap_open,
                                                    gap_extend, objective, is_local, &out_ins, &out_del);
            next_inserts[i] = out_ins;
            next_deletes[i] = out_del;
            next_scores[i] = cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        sz_i32_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        sz_i32_t *swap = current_inserts;
        current_inserts = next_inserts;
        next_inserts = swap;
        swap = current_deletes;
        current_deletes = next_deletes;
        next_deletes = swap;
        previous_scores++;
        __syncwarp();
    }

    if (is_local) return szs_best_in_warp_i32(best_score, objective);
    sz_i32_t final_score = lane == 0 ? current_scores[0] : 0;
    return __shfl_sync(0xffffffffu, final_score, 0);
}

#pragma endregion

#pragma region - Hopper Entry Kernels (emitted only when this is the compiled tier)

// LINEAR routes through the DPX linear warp walkers; AFFINE routes through the DPX affine warp walkers. The
// (op x gap x width) surface and the warp-per-pair grid-stride loop match `cuda_maxwell.cuh` exactly.

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
        sz_i32_t v = szs_walk_pair_warp_hopper_u8_linear(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
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
        sz_i32_t v = szs_walk_pair_warp_hopper_u16_linear(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
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
        sz_i32_t v = szs_walk_pair_warp_hopper_u32_linear(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
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
        sz_i32_t v = szs_walk_pair_warp_hopper_u8_affine(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
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
        sz_i32_t v = szs_walk_pair_warp_hopper_u16_affine(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
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
        sz_i32_t v = szs_walk_pair_warp_hopper_u32_affine(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
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
        sz_i32_t v = szs_walk_pair_warp_hopper_i16_linear(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
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
        sz_i32_t v = szs_walk_pair_warp_hopper_i32_linear(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
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
        sz_i32_t v = szs_walk_pair_warp_hopper_i16_affine(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
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
        sz_i32_t v = szs_walk_pair_warp_hopper_i32_affine(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
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
        sz_i32_t v = szs_walk_pair_warp_hopper_i16_linear(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
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
        sz_i32_t v = szs_walk_pair_warp_hopper_i32_linear(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
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
        sz_i32_t v = szs_walk_pair_warp_hopper_i16_affine(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
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
        sz_i32_t v = szs_walk_pair_warp_hopper_i32_affine(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
        szs_warp_write_back(task, v, ctx.lane);
    }
}

#pragma endregion

#endif // __CUDACC__

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SIMILARITIES_CUDA_HOPPER_CUH_
