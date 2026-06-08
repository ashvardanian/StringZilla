/**
 *  @file include/stringzillas/similarities/cuda_maxwell.cuh
 *  @brief Maxwell-tier (sm_50) CUDA baseline: shared device primitives + warp-per-pair walker + entries.
 *  @author Ash Vardanian
 *
 *  The CUDA baseline every higher tier (`cuda_kepler.cuh`, `cuda_hopper.cuh`) includes and builds on. This
 *  file is NVIDIA-only by construction (PTX, warp-synchronous `__shfl_*_sync`/`__syncwarp`, cooperative
 *  groups); AMD/ROCm gets its own `rocm_*` family. It owns two things:
 *
 *  1. The tier-independent device primitives (folded in from the former `cuda_device.cuh`): the scalar DP
 *     cell helpers (mirroring the serial `serial.h` math bit-for-bit), the `szs_device_costs_t` cost model
 *     staged into shared memory, and the `szs_pair_task_t` task POD. These parse on the host too (the
 *     execution-space qualifiers are stubbed) so the Driver-API launcher can size kernel args.
 *  2. The warp-per-pair DP walker - the correctness floor every higher tier reuses where it has no perf
 *     gain. One warp (32 lanes) cooperatively computes one pair: each anti-diagonal's cells are strided
 *     across the lanes, `__syncwarp()` separates diagonals, and the per-warp shared-memory slice holds
 *     three (linear) or seven (affine) score diagonals plus both strings. One explicit walker per DP cell
 *     width (u8/u16/u32 for distance, i16/i32 for score); linear vs affine gaps is a runtime flag.
 *
 *  When the compiling tier is the baseline (`SZ_GPU_TIER == sz_cap_cuda_k`) this header also emits the 14
 *  `extern "C" __global__` warp-per-pair entry kernels plus the device-spanning entries for huge single
 *  pairs. Each entry stages the cost model into shared memory and dispatches the matching walker. Higher
 *  tiers gate this block off and emit their own entries.
 */
#ifndef STRINGZILLAS_SIMILARITIES_CUDA_MAXWELL_CUH_
#define STRINGZILLAS_SIMILARITIES_CUDA_MAXWELL_CUH_

#ifndef SZ_GPU_TIER
#define SZ_GPU_TIER sz_cap_cuda_k
#endif

#include "stringzilla/types.h"  // `sz_i32_t`, `sz_u8_t`, `sz_size_t`, similarity enums
#include "stringzillas/types.h" // `sz_substitution_costs_t`, `SZ_SUBS_MAX_CLASSES`

#if defined(__CUDACC__)
#include <cooperative_groups.h> // `cooperative_groups::this_grid()` for the device-spanning kernels
#endif

// Stub the CUDA execution-space qualifiers when included by a plain host compiler (for the shared PODs);
// nvcc defines `__CUDACC__` and provides the real ones. The device helpers are never called on the host.
#if !defined(__CUDACC__)
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#endif

namespace ashvardanian {
namespace stringzillas {

#pragma region - Shared Device Recurrence Helpers

// Pick min (minimize) or max (maximize) of two signed 32-bit cells per the objective.
__device__ __forceinline__ sz_i32_t szs_pick_best_i32(sz_i32_t a, sz_i32_t b, sz_similarity_objective_t obj) {
    if (obj == sz_maximize_score_k) return a > b ? a : b;
    return a < b ? a : b;
}

// One linear-gap DP cell: best of substitution, insertion-gap, deletion-gap.
__device__ __forceinline__ sz_i32_t szs_cell_linear_i32(  //
    sz_i32_t pre_sub, sz_i32_t pre_ins, sz_i32_t pre_del, //
    sz_error_cost_t sub_cost, sz_error_cost_t gap,        //
    sz_similarity_objective_t obj, int is_local) {
    sz_i32_t best = pre_sub + (sz_i32_t)sub_cost;
    best = szs_pick_best_i32(best, pre_ins + (sz_i32_t)gap, obj);
    best = szs_pick_best_i32(best, pre_del + (sz_i32_t)gap, obj);
    if (is_local && obj == sz_maximize_score_k && best < 0) best = 0;
    return best;
}

// One affine-gap DP cell (Gotoh): updates the insertion & deletion gap tracks and the main cell.
__device__ __forceinline__ sz_i32_t szs_cell_affine_i32(                            //
    sz_i32_t pre_sub, sz_i32_t up_main, sz_i32_t up_ins,                            //
    sz_i32_t left_main, sz_i32_t left_del,                                          //
    sz_error_cost_t sub_cost, sz_error_cost_t gap_open, sz_error_cost_t gap_extend, //
    sz_similarity_objective_t obj, int is_local,                                    //
    sz_i32_t *out_ins, sz_i32_t *out_del) {
    sz_i32_t ins = szs_pick_best_i32(up_main + (sz_i32_t)gap_open, up_ins + (sz_i32_t)gap_extend, obj);
    sz_i32_t del = szs_pick_best_i32(left_main + (sz_i32_t)gap_open, left_del + (sz_i32_t)gap_extend, obj);
    sz_i32_t best = pre_sub + (sz_i32_t)sub_cost;
    best = szs_pick_best_i32(best, ins, obj);
    best = szs_pick_best_i32(best, del, obj);
    if (is_local && obj == sz_maximize_score_k && best < 0) best = 0;
    *out_ins = ins;
    *out_del = del;
    return best;
}

// Boundary value for the linear-gap main matrix at distance `index` from the origin.
__device__ __forceinline__ sz_i32_t szs_init_score_linear(sz_size_t index, sz_error_cost_t gap, int is_local) {
    if (is_local) return 0;
    return (sz_i32_t)index * (sz_i32_t)gap;
}

// Boundary value for the affine-gap main matrix at distance `index` (open then extend).
__device__ __forceinline__ sz_i32_t szs_init_score_affine(sz_size_t index, sz_error_cost_t gap_open,
                                                          sz_error_cost_t gap_extend, int is_local) {
    if (is_local || index == 0) return 0;
    return (sz_i32_t)gap_open + (sz_i32_t)(index - 1) * (sz_i32_t)gap_extend;
}

// Boundary value for an affine gap track (kept saturating-low so it never wins on the edge).
__device__ __forceinline__ sz_i32_t szs_init_gap_affine(sz_size_t index, sz_error_cost_t gap_open,
                                                        sz_error_cost_t gap_extend, int is_local) {
    if (is_local) return 0;
    if (index == 0) return (sz_i32_t)gap_open;
    return (sz_i32_t)gap_open + (sz_i32_t)index * (sz_i32_t)gap_extend;
}

#pragma endregion

#pragma region - Shared Cost Model + Pair Task

// Loop-invariant cost model passed by value to every entry kernel, mirroring `szs_cell_costs_t`. Trivially
// copyable so it can be a by-value kernel argument; the LUT travels by value too so there is no device
// constant-memory symbol and no separable-compilation requirement.
struct szs_device_costs_t {
    int is_uniform;                      // Non-zero => use match/mismatch; else the class LUT.
    sz_error_cost_t match;               // Uniform-cost match (Levenshtein, typically 0).
    sz_error_cost_t mismatch;            // Uniform-cost mismatch (Levenshtein, typically 1).
    sz_error_cost_t gap_open;            // Gap open cost.
    sz_error_cost_t gap_extend;          // Gap extend cost (== open for linear gaps).
    sz_similarity_objective_t objective; // Minimize (Levenshtein) vs maximize (NW/SW).
    int is_local;                        // Smith-Waterman clamp-to-zero on maximize.
    sz_substitution_costs_t lut;         // Class-based substitution LUT (unused when is_uniform).
};

// Cost of substituting element `a` with element `b` under the active cost model.
__device__ __forceinline__ sz_error_cost_t szs_subs_cost(szs_device_costs_t const *costs, sz_u32_t a, sz_u32_t b) {
    if (costs->is_uniform) return a == b ? costs->match : costs->mismatch;
    sz_u8_t class_a = costs->lut.byte_to_class[(sz_u8_t)a];
    sz_u8_t class_b = costs->lut.byte_to_class[(sz_u8_t)b];
    return costs->lut.costs[class_a][class_b];
}

// A single pairwise alignment task in device-accessible memory. Plain POD shared by host and device so the
// host can fill it and the entry kernels read it.
struct szs_pair_task_t {
    sz_u8_t const *a_ptr;
    sz_size_t a_len;
    sz_u8_t const *b_ptr;
    sz_size_t b_len;
    void *scratch;   // Per-task scratch span (sized by the host for the chosen width).
    sz_i64_t result; // Output score / distance (widened to i64 for a uniform task POD).
};

#pragma endregion

#if defined(__CUDACC__)

#pragma region - Warp Reduction + Shared-Memory Layout Helpers

// Warp-wide best (min for distance, max for score) of one per-lane value, mirroring `pick_best_in_warp_`.
__device__ __forceinline__ sz_i32_t szs_best_in_warp_i32(sz_i32_t value, sz_similarity_objective_t obj) {
    value = szs_pick_best_i32(__shfl_down_sync(0xffffffffu, value, 16), value, obj);
    value = szs_pick_best_i32(__shfl_down_sync(0xffffffffu, value, 8), value, obj);
    value = szs_pick_best_i32(__shfl_down_sync(0xffffffffu, value, 4), value, obj);
    value = szs_pick_best_i32(__shfl_down_sync(0xffffffffu, value, 2), value, obj);
    value = szs_pick_best_i32(__shfl_down_sync(0xffffffffu, value, 1), value, obj);
    return value;
}

// Round `value` up to the next multiple of `multiple`.
__device__ __forceinline__ unsigned szs_round_up_to_multiple(unsigned value, unsigned multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

// Per-warp shared-memory byte requirement for a pair, given the diagonal count and cell width. Mirrors
// the host-side `similarity_memory_requirements`: `count_diagonals` diagonals each rounded to 4 bytes,
// plus the two strings. Used by the host to size the dynamic shared-memory partition per warp.
__device__ __forceinline__ unsigned szs_warp_shared_bytes(unsigned shorter_length, unsigned longer_length,
                                                          unsigned bytes_per_cell, unsigned count_diagonals) {
    unsigned const max_diagonal_length = shorter_length + 1;
    unsigned const bytes_per_diagonal = szs_round_up_to_multiple(max_diagonal_length * bytes_per_cell, 4);
    return count_diagonals * bytes_per_diagonal + shorter_length + longer_length;
}

#pragma endregion

#pragma region - Base Warp-Per-Pair Walkers (one explicit body per DP cell width)

// szs_walk_pair_warp_base_{u8,u16,u32,i16,i32}: one warp cooperatively scores one pair. The five bodies
// are identical except for the diagonal cell type; we keep them explicit (no template) so the PTX stays
// readable. Linear (3 diagonals) vs affine (7 diagonals + gap tracks) is the runtime `is_affine` flag.
// `shared_memory_for_warp` is this warp's private SMEM slice (sized by the host launcher); the walker
// stages both strings into it and reuses it for the rotating diagonals. The first string is staged
// un-reversed and reverse-indexed inside the cell loop. Returns the final/best score.

__device__ inline sz_i32_t szs_walk_pair_warp_base_u8( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap_open = costs->gap_open, gap_extend = costs->gap_extend;
    sz_error_cost_t const gap = gap_open;
    int const is_affine = gap_open != gap_extend;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;

    // The caller guarantees the shorter string is in the first argument. Empty handling matches serial.
    if (shorter_length == 0 || longer_length == 0) {
        if (is_local) return 0;
        sz_size_t const only_length = shorter_length ? shorter_length : longer_length;
        if (only_length == 0) return 0;
        if (!is_affine) return (sz_i32_t)((sz_i64_t)gap * (sz_i64_t)only_length);
        return (sz_i32_t)((sz_i64_t)gap_open + (sz_i64_t)gap_extend * (sz_i64_t)(only_length - 1));
    }

    unsigned const shorter_dim = (unsigned)shorter_length + 1;
    unsigned const longer_dim = (unsigned)longer_length + 1;
    unsigned const diagonals_count = shorter_dim + longer_dim - 1;
    unsigned const max_diagonal_length = (unsigned)shorter_length + 1;
    unsigned const count_diagonals = is_affine ? 7 : 3;
    unsigned const bytes_per_diagonal = szs_round_up_to_multiple(max_diagonal_length * sizeof(sz_u8_t), 4);

    sz_u8_t *previous_scores = (sz_u8_t *)(shared_memory_for_warp + 0 * bytes_per_diagonal);
    sz_u8_t *current_scores = (sz_u8_t *)(shared_memory_for_warp + 1 * bytes_per_diagonal);
    sz_u8_t *next_scores = (sz_u8_t *)(shared_memory_for_warp + 2 * bytes_per_diagonal);
    sz_u8_t *current_inserts = (sz_u8_t *)(shared_memory_for_warp + 3 * bytes_per_diagonal);
    sz_u8_t *next_inserts = (sz_u8_t *)(shared_memory_for_warp + 4 * bytes_per_diagonal);
    sz_u8_t *current_deletes = (sz_u8_t *)(shared_memory_for_warp + 5 * bytes_per_diagonal);
    sz_u8_t *next_deletes = (sz_u8_t *)(shared_memory_for_warp + 6 * bytes_per_diagonal);
    char *longer = shared_memory_for_warp + count_diagonals * bytes_per_diagonal;
    char *shorter = longer + longer_length;

    for (unsigned i = lane; i < longer_length; i += warp_size) longer[i] = (char)longer_global[i];
    for (unsigned i = lane; i < shorter_length; i += warp_size) shorter[i] = (char)shorter_global[i];

    if (lane == 0) {
        if (!is_affine) {
            previous_scores[0] = (sz_u8_t)szs_init_score_linear(0, gap, is_local);
            current_scores[0] = (sz_u8_t)szs_init_score_linear(1, gap, is_local);
            current_scores[1] = (sz_u8_t)szs_init_score_linear(1, gap, is_local);
        }
        else {
            previous_scores[0] = (sz_u8_t)szs_init_score_affine(0, gap_open, gap_extend, is_local);
            current_scores[0] = (sz_u8_t)szs_init_score_affine(1, gap_open, gap_extend, is_local);
            current_scores[1] = (sz_u8_t)szs_init_score_affine(1, gap_open, gap_extend, is_local);
            current_inserts[0] = (sz_u8_t)szs_init_gap_affine(1, gap_open, gap_extend, is_local);
            current_deletes[1] = (sz_u8_t)szs_init_gap_affine(1, gap_open, gap_extend, is_local);
        }
    }
    __syncwarp();

    sz_i32_t best_score = 0;
    unsigned next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = next_diagonal_index + 1;
        unsigned const cells_count = next_diagonal_length - 2;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            char first_char = shorter[cells_count - i - 1];
            char second_char = longer[i];
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_char, (sz_u32_t)(sz_u8_t)second_char);
            sz_i32_t cell;
            if (!is_affine) {
                cell = szs_cell_linear_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            }
            else {
                sz_i32_t out_ins, out_del;
                cell = szs_cell_affine_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                           (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend, objective,
                                           is_local, &out_ins, &out_del);
                next_inserts[i + 1] = (sz_u8_t)out_ins;
                next_deletes[i + 1] = (sz_u8_t)out_del;
            }
            next_scores[i + 1] = (sz_u8_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            if (!is_affine) {
                next_scores[0] = (sz_u8_t)szs_init_score_linear(next_diagonal_index, gap, is_local);
                next_scores[next_diagonal_length - 1] = (sz_u8_t)szs_init_score_linear(next_diagonal_index, gap,
                                                                                       is_local);
            }
            else {
                next_scores[0] = (sz_u8_t)szs_init_score_affine(next_diagonal_index, gap_open, gap_extend, is_local);
                next_scores[next_diagonal_length - 1] = (sz_u8_t)szs_init_score_affine(next_diagonal_index, gap_open,
                                                                                       gap_extend, is_local);
                next_inserts[0] = (sz_u8_t)szs_init_gap_affine(next_diagonal_index, gap_open, gap_extend, is_local);
                next_deletes[next_diagonal_length - 1] = (sz_u8_t)szs_init_gap_affine(next_diagonal_index, gap_open,
                                                                                      gap_extend, is_local);
            }
        }
        __syncwarp();
        sz_u8_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        if (is_affine) {
            sz_u8_t *swap = current_inserts;
            current_inserts = next_inserts;
            next_inserts = swap;
            swap = current_deletes;
            current_deletes = next_deletes;
            next_deletes = swap;
        }
    }
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = shorter_dim;
        char const *second_slice = longer + next_diagonal_index - shorter_dim;
        unsigned const cells_count = next_diagonal_length - 1;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            char first_char = shorter[cells_count - i - 1];
            char second_char = second_slice[i];
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_char, (sz_u32_t)(sz_u8_t)second_char);
            sz_i32_t cell;
            if (!is_affine) {
                cell = szs_cell_linear_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            }
            else {
                sz_i32_t out_ins, out_del;
                cell = szs_cell_affine_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                           (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend, objective,
                                           is_local, &out_ins, &out_del);
                next_inserts[i] = (sz_u8_t)out_ins;
                next_deletes[i] = (sz_u8_t)out_del;
            }
            next_scores[i] = (sz_u8_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            if (!is_affine) {
                next_scores[next_diagonal_length - 1] = (sz_u8_t)szs_init_score_linear(next_diagonal_index, gap,
                                                                                       is_local);
            }
            else {
                next_scores[next_diagonal_length - 1] = (sz_u8_t)szs_init_score_affine(next_diagonal_index, gap_open,
                                                                                       gap_extend, is_local);
                next_deletes[next_diagonal_length - 1] = (sz_u8_t)szs_init_gap_affine(next_diagonal_index, gap_open,
                                                                                      gap_extend, is_local);
            }
        }
        if (is_affine) {
            sz_u8_t *swap = current_inserts;
            current_inserts = next_inserts;
            next_inserts = swap;
            swap = current_deletes;
            current_deletes = next_deletes;
            next_deletes = swap;
        }
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
            char first_char = first_slice[next_diagonal_length - i - 1];
            char second_char = second_slice[i];
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_char, (sz_u32_t)(sz_u8_t)second_char);
            sz_i32_t cell;
            if (!is_affine) {
                cell = szs_cell_linear_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            }
            else {
                sz_i32_t out_ins, out_del;
                cell = szs_cell_affine_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                           (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend, objective,
                                           is_local, &out_ins, &out_del);
                next_inserts[i] = (sz_u8_t)out_ins;
                next_deletes[i] = (sz_u8_t)out_del;
            }
            next_scores[i] = (sz_u8_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        sz_u8_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        if (is_affine) {
            sz_u8_t *swap = current_inserts;
            current_inserts = next_inserts;
            next_inserts = swap;
            swap = current_deletes;
            current_deletes = next_deletes;
            next_deletes = swap;
        }
        previous_scores++;
        __syncwarp();
    }

    if (is_local) return szs_best_in_warp_i32(best_score, objective);
    // The global result is the bottom-right corner, held by lane 0 in current_scores[0] after the last rotate.
    sz_i32_t final_score = lane == 0 ? (sz_i32_t)current_scores[0] : 0;
    return __shfl_sync(0xffffffffu, final_score, 0);
}

__device__ inline sz_i32_t szs_walk_pair_warp_base_u16( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap_open = costs->gap_open, gap_extend = costs->gap_extend;
    sz_error_cost_t const gap = gap_open;
    int const is_affine = gap_open != gap_extend;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;

    if (shorter_length == 0 || longer_length == 0) {
        if (is_local) return 0;
        sz_size_t const only_length = shorter_length ? shorter_length : longer_length;
        if (only_length == 0) return 0;
        if (!is_affine) return (sz_i32_t)((sz_i64_t)gap * (sz_i64_t)only_length);
        return (sz_i32_t)((sz_i64_t)gap_open + (sz_i64_t)gap_extend * (sz_i64_t)(only_length - 1));
    }

    unsigned const shorter_dim = (unsigned)shorter_length + 1;
    unsigned const longer_dim = (unsigned)longer_length + 1;
    unsigned const diagonals_count = shorter_dim + longer_dim - 1;
    unsigned const max_diagonal_length = (unsigned)shorter_length + 1;
    unsigned const count_diagonals = is_affine ? 7 : 3;
    unsigned const bytes_per_diagonal = szs_round_up_to_multiple(max_diagonal_length * sizeof(sz_u16_t), 4);

    sz_u16_t *previous_scores = (sz_u16_t *)(shared_memory_for_warp + 0 * bytes_per_diagonal);
    sz_u16_t *current_scores = (sz_u16_t *)(shared_memory_for_warp + 1 * bytes_per_diagonal);
    sz_u16_t *next_scores = (sz_u16_t *)(shared_memory_for_warp + 2 * bytes_per_diagonal);
    sz_u16_t *current_inserts = (sz_u16_t *)(shared_memory_for_warp + 3 * bytes_per_diagonal);
    sz_u16_t *next_inserts = (sz_u16_t *)(shared_memory_for_warp + 4 * bytes_per_diagonal);
    sz_u16_t *current_deletes = (sz_u16_t *)(shared_memory_for_warp + 5 * bytes_per_diagonal);
    sz_u16_t *next_deletes = (sz_u16_t *)(shared_memory_for_warp + 6 * bytes_per_diagonal);
    char *longer = shared_memory_for_warp + count_diagonals * bytes_per_diagonal;
    char *shorter = longer + longer_length;

    for (unsigned i = lane; i < longer_length; i += warp_size) longer[i] = (char)longer_global[i];
    for (unsigned i = lane; i < shorter_length; i += warp_size) shorter[i] = (char)shorter_global[i];

    if (lane == 0) {
        if (!is_affine) {
            previous_scores[0] = (sz_u16_t)szs_init_score_linear(0, gap, is_local);
            current_scores[0] = (sz_u16_t)szs_init_score_linear(1, gap, is_local);
            current_scores[1] = (sz_u16_t)szs_init_score_linear(1, gap, is_local);
        }
        else {
            previous_scores[0] = (sz_u16_t)szs_init_score_affine(0, gap_open, gap_extend, is_local);
            current_scores[0] = (sz_u16_t)szs_init_score_affine(1, gap_open, gap_extend, is_local);
            current_scores[1] = (sz_u16_t)szs_init_score_affine(1, gap_open, gap_extend, is_local);
            current_inserts[0] = (sz_u16_t)szs_init_gap_affine(1, gap_open, gap_extend, is_local);
            current_deletes[1] = (sz_u16_t)szs_init_gap_affine(1, gap_open, gap_extend, is_local);
        }
    }
    __syncwarp();

    sz_i32_t best_score = 0;
    unsigned next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = next_diagonal_index + 1;
        unsigned const cells_count = next_diagonal_length - 2;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            char first_char = shorter[cells_count - i - 1];
            char second_char = longer[i];
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_char, (sz_u32_t)(sz_u8_t)second_char);
            sz_i32_t cell;
            if (!is_affine) {
                cell = szs_cell_linear_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            }
            else {
                sz_i32_t out_ins, out_del;
                cell = szs_cell_affine_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                           (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend, objective,
                                           is_local, &out_ins, &out_del);
                next_inserts[i + 1] = (sz_u16_t)out_ins;
                next_deletes[i + 1] = (sz_u16_t)out_del;
            }
            next_scores[i + 1] = (sz_u16_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            if (!is_affine) {
                next_scores[0] = (sz_u16_t)szs_init_score_linear(next_diagonal_index, gap, is_local);
                next_scores[next_diagonal_length - 1] = (sz_u16_t)szs_init_score_linear(next_diagonal_index, gap,
                                                                                        is_local);
            }
            else {
                next_scores[0] = (sz_u16_t)szs_init_score_affine(next_diagonal_index, gap_open, gap_extend, is_local);
                next_scores[next_diagonal_length - 1] = (sz_u16_t)szs_init_score_affine(next_diagonal_index, gap_open,
                                                                                        gap_extend, is_local);
                next_inserts[0] = (sz_u16_t)szs_init_gap_affine(next_diagonal_index, gap_open, gap_extend, is_local);
                next_deletes[next_diagonal_length - 1] = (sz_u16_t)szs_init_gap_affine(next_diagonal_index, gap_open,
                                                                                       gap_extend, is_local);
            }
        }
        __syncwarp();
        sz_u16_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        if (is_affine) {
            sz_u16_t *swap = current_inserts;
            current_inserts = next_inserts;
            next_inserts = swap;
            swap = current_deletes;
            current_deletes = next_deletes;
            next_deletes = swap;
        }
    }
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = shorter_dim;
        char const *second_slice = longer + next_diagonal_index - shorter_dim;
        unsigned const cells_count = next_diagonal_length - 1;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            char first_char = shorter[cells_count - i - 1];
            char second_char = second_slice[i];
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_char, (sz_u32_t)(sz_u8_t)second_char);
            sz_i32_t cell;
            if (!is_affine) {
                cell = szs_cell_linear_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            }
            else {
                sz_i32_t out_ins, out_del;
                cell = szs_cell_affine_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                           (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend, objective,
                                           is_local, &out_ins, &out_del);
                next_inserts[i] = (sz_u16_t)out_ins;
                next_deletes[i] = (sz_u16_t)out_del;
            }
            next_scores[i] = (sz_u16_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            if (!is_affine) {
                next_scores[next_diagonal_length - 1] = (sz_u16_t)szs_init_score_linear(next_diagonal_index, gap,
                                                                                        is_local);
            }
            else {
                next_scores[next_diagonal_length - 1] = (sz_u16_t)szs_init_score_affine(next_diagonal_index, gap_open,
                                                                                        gap_extend, is_local);
                next_deletes[next_diagonal_length - 1] = (sz_u16_t)szs_init_gap_affine(next_diagonal_index, gap_open,
                                                                                       gap_extend, is_local);
            }
        }
        if (is_affine) {
            sz_u16_t *swap = current_inserts;
            current_inserts = next_inserts;
            next_inserts = swap;
            swap = current_deletes;
            current_deletes = next_deletes;
            next_deletes = swap;
        }
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
            char first_char = first_slice[next_diagonal_length - i - 1];
            char second_char = second_slice[i];
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_char, (sz_u32_t)(sz_u8_t)second_char);
            sz_i32_t cell;
            if (!is_affine) {
                cell = szs_cell_linear_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            }
            else {
                sz_i32_t out_ins, out_del;
                cell = szs_cell_affine_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                           (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend, objective,
                                           is_local, &out_ins, &out_del);
                next_inserts[i] = (sz_u16_t)out_ins;
                next_deletes[i] = (sz_u16_t)out_del;
            }
            next_scores[i] = (sz_u16_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        sz_u16_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        if (is_affine) {
            sz_u16_t *swap = current_inserts;
            current_inserts = next_inserts;
            next_inserts = swap;
            swap = current_deletes;
            current_deletes = next_deletes;
            next_deletes = swap;
        }
        previous_scores++;
        __syncwarp();
    }

    if (is_local) return szs_best_in_warp_i32(best_score, objective);
    sz_i32_t final_score = lane == 0 ? (sz_i32_t)current_scores[0] : 0;
    return __shfl_sync(0xffffffffu, final_score, 0);
}

__device__ inline sz_i32_t szs_walk_pair_warp_base_u32( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap_open = costs->gap_open, gap_extend = costs->gap_extend;
    sz_error_cost_t const gap = gap_open;
    int const is_affine = gap_open != gap_extend;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;

    if (shorter_length == 0 || longer_length == 0) {
        if (is_local) return 0;
        sz_size_t const only_length = shorter_length ? shorter_length : longer_length;
        if (only_length == 0) return 0;
        if (!is_affine) return (sz_i32_t)((sz_i64_t)gap * (sz_i64_t)only_length);
        return (sz_i32_t)((sz_i64_t)gap_open + (sz_i64_t)gap_extend * (sz_i64_t)(only_length - 1));
    }

    unsigned const shorter_dim = (unsigned)shorter_length + 1;
    unsigned const longer_dim = (unsigned)longer_length + 1;
    unsigned const diagonals_count = shorter_dim + longer_dim - 1;
    unsigned const max_diagonal_length = (unsigned)shorter_length + 1;
    unsigned const count_diagonals = is_affine ? 7 : 3;
    unsigned const bytes_per_diagonal = szs_round_up_to_multiple(max_diagonal_length * sizeof(sz_u32_t), 4);

    sz_u32_t *previous_scores = (sz_u32_t *)(shared_memory_for_warp + 0 * bytes_per_diagonal);
    sz_u32_t *current_scores = (sz_u32_t *)(shared_memory_for_warp + 1 * bytes_per_diagonal);
    sz_u32_t *next_scores = (sz_u32_t *)(shared_memory_for_warp + 2 * bytes_per_diagonal);
    sz_u32_t *current_inserts = (sz_u32_t *)(shared_memory_for_warp + 3 * bytes_per_diagonal);
    sz_u32_t *next_inserts = (sz_u32_t *)(shared_memory_for_warp + 4 * bytes_per_diagonal);
    sz_u32_t *current_deletes = (sz_u32_t *)(shared_memory_for_warp + 5 * bytes_per_diagonal);
    sz_u32_t *next_deletes = (sz_u32_t *)(shared_memory_for_warp + 6 * bytes_per_diagonal);
    char *longer = shared_memory_for_warp + count_diagonals * bytes_per_diagonal;
    char *shorter = longer + longer_length;

    for (unsigned i = lane; i < longer_length; i += warp_size) longer[i] = (char)longer_global[i];
    for (unsigned i = lane; i < shorter_length; i += warp_size) shorter[i] = (char)shorter_global[i];

    if (lane == 0) {
        if (!is_affine) {
            previous_scores[0] = (sz_u32_t)szs_init_score_linear(0, gap, is_local);
            current_scores[0] = (sz_u32_t)szs_init_score_linear(1, gap, is_local);
            current_scores[1] = (sz_u32_t)szs_init_score_linear(1, gap, is_local);
        }
        else {
            previous_scores[0] = (sz_u32_t)szs_init_score_affine(0, gap_open, gap_extend, is_local);
            current_scores[0] = (sz_u32_t)szs_init_score_affine(1, gap_open, gap_extend, is_local);
            current_scores[1] = (sz_u32_t)szs_init_score_affine(1, gap_open, gap_extend, is_local);
            current_inserts[0] = (sz_u32_t)szs_init_gap_affine(1, gap_open, gap_extend, is_local);
            current_deletes[1] = (sz_u32_t)szs_init_gap_affine(1, gap_open, gap_extend, is_local);
        }
    }
    __syncwarp();

    sz_i32_t best_score = 0;
    unsigned next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = next_diagonal_index + 1;
        unsigned const cells_count = next_diagonal_length - 2;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            char first_char = shorter[cells_count - i - 1];
            char second_char = longer[i];
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_char, (sz_u32_t)(sz_u8_t)second_char);
            sz_i32_t cell;
            if (!is_affine) {
                cell = szs_cell_linear_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            }
            else {
                sz_i32_t out_ins, out_del;
                cell = szs_cell_affine_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                           (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend, objective,
                                           is_local, &out_ins, &out_del);
                next_inserts[i + 1] = (sz_u32_t)out_ins;
                next_deletes[i + 1] = (sz_u32_t)out_del;
            }
            next_scores[i + 1] = (sz_u32_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            if (!is_affine) {
                next_scores[0] = (sz_u32_t)szs_init_score_linear(next_diagonal_index, gap, is_local);
                next_scores[next_diagonal_length - 1] = (sz_u32_t)szs_init_score_linear(next_diagonal_index, gap,
                                                                                        is_local);
            }
            else {
                next_scores[0] = (sz_u32_t)szs_init_score_affine(next_diagonal_index, gap_open, gap_extend, is_local);
                next_scores[next_diagonal_length - 1] = (sz_u32_t)szs_init_score_affine(next_diagonal_index, gap_open,
                                                                                        gap_extend, is_local);
                next_inserts[0] = (sz_u32_t)szs_init_gap_affine(next_diagonal_index, gap_open, gap_extend, is_local);
                next_deletes[next_diagonal_length - 1] = (sz_u32_t)szs_init_gap_affine(next_diagonal_index, gap_open,
                                                                                       gap_extend, is_local);
            }
        }
        __syncwarp();
        sz_u32_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        if (is_affine) {
            sz_u32_t *swap = current_inserts;
            current_inserts = next_inserts;
            next_inserts = swap;
            swap = current_deletes;
            current_deletes = next_deletes;
            next_deletes = swap;
        }
    }
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = shorter_dim;
        char const *second_slice = longer + next_diagonal_index - shorter_dim;
        unsigned const cells_count = next_diagonal_length - 1;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            char first_char = shorter[cells_count - i - 1];
            char second_char = second_slice[i];
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_char, (sz_u32_t)(sz_u8_t)second_char);
            sz_i32_t cell;
            if (!is_affine) {
                cell = szs_cell_linear_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            }
            else {
                sz_i32_t out_ins, out_del;
                cell = szs_cell_affine_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                           (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend, objective,
                                           is_local, &out_ins, &out_del);
                next_inserts[i] = (sz_u32_t)out_ins;
                next_deletes[i] = (sz_u32_t)out_del;
            }
            next_scores[i] = (sz_u32_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            if (!is_affine) {
                next_scores[next_diagonal_length - 1] = (sz_u32_t)szs_init_score_linear(next_diagonal_index, gap,
                                                                                        is_local);
            }
            else {
                next_scores[next_diagonal_length - 1] = (sz_u32_t)szs_init_score_affine(next_diagonal_index, gap_open,
                                                                                        gap_extend, is_local);
                next_deletes[next_diagonal_length - 1] = (sz_u32_t)szs_init_gap_affine(next_diagonal_index, gap_open,
                                                                                       gap_extend, is_local);
            }
        }
        if (is_affine) {
            sz_u32_t *swap = current_inserts;
            current_inserts = next_inserts;
            next_inserts = swap;
            swap = current_deletes;
            current_deletes = next_deletes;
            next_deletes = swap;
        }
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
            char first_char = first_slice[next_diagonal_length - i - 1];
            char second_char = second_slice[i];
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_char, (sz_u32_t)(sz_u8_t)second_char);
            sz_i32_t cell;
            if (!is_affine) {
                cell = szs_cell_linear_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            }
            else {
                sz_i32_t out_ins, out_del;
                cell = szs_cell_affine_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                           (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend, objective,
                                           is_local, &out_ins, &out_del);
                next_inserts[i] = (sz_u32_t)out_ins;
                next_deletes[i] = (sz_u32_t)out_del;
            }
            next_scores[i] = (sz_u32_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        sz_u32_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        if (is_affine) {
            sz_u32_t *swap = current_inserts;
            current_inserts = next_inserts;
            next_inserts = swap;
            swap = current_deletes;
            current_deletes = next_deletes;
            next_deletes = swap;
        }
        previous_scores++;
        __syncwarp();
    }

    if (is_local) return szs_best_in_warp_i32(best_score, objective);
    sz_i32_t final_score = lane == 0 ? (sz_i32_t)current_scores[0] : 0;
    return __shfl_sync(0xffffffffu, final_score, 0);
}

__device__ inline sz_i32_t szs_walk_pair_warp_base_i16( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap_open = costs->gap_open, gap_extend = costs->gap_extend;
    sz_error_cost_t const gap = gap_open;
    int const is_affine = gap_open != gap_extend;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;

    if (shorter_length == 0 || longer_length == 0) {
        if (is_local) return 0;
        sz_size_t const only_length = shorter_length ? shorter_length : longer_length;
        if (only_length == 0) return 0;
        if (!is_affine) return (sz_i32_t)((sz_i64_t)gap * (sz_i64_t)only_length);
        return (sz_i32_t)((sz_i64_t)gap_open + (sz_i64_t)gap_extend * (sz_i64_t)(only_length - 1));
    }

    unsigned const shorter_dim = (unsigned)shorter_length + 1;
    unsigned const longer_dim = (unsigned)longer_length + 1;
    unsigned const diagonals_count = shorter_dim + longer_dim - 1;
    unsigned const max_diagonal_length = (unsigned)shorter_length + 1;
    unsigned const count_diagonals = is_affine ? 7 : 3;
    unsigned const bytes_per_diagonal = szs_round_up_to_multiple(max_diagonal_length * sizeof(sz_i16_t), 4);

    sz_i16_t *previous_scores = (sz_i16_t *)(shared_memory_for_warp + 0 * bytes_per_diagonal);
    sz_i16_t *current_scores = (sz_i16_t *)(shared_memory_for_warp + 1 * bytes_per_diagonal);
    sz_i16_t *next_scores = (sz_i16_t *)(shared_memory_for_warp + 2 * bytes_per_diagonal);
    sz_i16_t *current_inserts = (sz_i16_t *)(shared_memory_for_warp + 3 * bytes_per_diagonal);
    sz_i16_t *next_inserts = (sz_i16_t *)(shared_memory_for_warp + 4 * bytes_per_diagonal);
    sz_i16_t *current_deletes = (sz_i16_t *)(shared_memory_for_warp + 5 * bytes_per_diagonal);
    sz_i16_t *next_deletes = (sz_i16_t *)(shared_memory_for_warp + 6 * bytes_per_diagonal);
    char *longer = shared_memory_for_warp + count_diagonals * bytes_per_diagonal;
    char *shorter = longer + longer_length;

    for (unsigned i = lane; i < longer_length; i += warp_size) longer[i] = (char)longer_global[i];
    for (unsigned i = lane; i < shorter_length; i += warp_size) shorter[i] = (char)shorter_global[i];

    if (lane == 0) {
        if (!is_affine) {
            previous_scores[0] = (sz_i16_t)szs_init_score_linear(0, gap, is_local);
            current_scores[0] = (sz_i16_t)szs_init_score_linear(1, gap, is_local);
            current_scores[1] = (sz_i16_t)szs_init_score_linear(1, gap, is_local);
        }
        else {
            previous_scores[0] = (sz_i16_t)szs_init_score_affine(0, gap_open, gap_extend, is_local);
            current_scores[0] = (sz_i16_t)szs_init_score_affine(1, gap_open, gap_extend, is_local);
            current_scores[1] = (sz_i16_t)szs_init_score_affine(1, gap_open, gap_extend, is_local);
            current_inserts[0] = (sz_i16_t)szs_init_gap_affine(1, gap_open, gap_extend, is_local);
            current_deletes[1] = (sz_i16_t)szs_init_gap_affine(1, gap_open, gap_extend, is_local);
        }
    }
    __syncwarp();

    sz_i32_t best_score = 0;
    unsigned next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = next_diagonal_index + 1;
        unsigned const cells_count = next_diagonal_length - 2;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            char first_char = shorter[cells_count - i - 1];
            char second_char = longer[i];
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_char, (sz_u32_t)(sz_u8_t)second_char);
            sz_i32_t cell;
            if (!is_affine) {
                cell = szs_cell_linear_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            }
            else {
                sz_i32_t out_ins, out_del;
                cell = szs_cell_affine_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                           (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend, objective,
                                           is_local, &out_ins, &out_del);
                next_inserts[i + 1] = (sz_i16_t)out_ins;
                next_deletes[i + 1] = (sz_i16_t)out_del;
            }
            next_scores[i + 1] = (sz_i16_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            if (!is_affine) {
                next_scores[0] = (sz_i16_t)szs_init_score_linear(next_diagonal_index, gap, is_local);
                next_scores[next_diagonal_length - 1] = (sz_i16_t)szs_init_score_linear(next_diagonal_index, gap,
                                                                                        is_local);
            }
            else {
                next_scores[0] = (sz_i16_t)szs_init_score_affine(next_diagonal_index, gap_open, gap_extend, is_local);
                next_scores[next_diagonal_length - 1] = (sz_i16_t)szs_init_score_affine(next_diagonal_index, gap_open,
                                                                                        gap_extend, is_local);
                next_inserts[0] = (sz_i16_t)szs_init_gap_affine(next_diagonal_index, gap_open, gap_extend, is_local);
                next_deletes[next_diagonal_length - 1] = (sz_i16_t)szs_init_gap_affine(next_diagonal_index, gap_open,
                                                                                       gap_extend, is_local);
            }
        }
        __syncwarp();
        sz_i16_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        if (is_affine) {
            sz_i16_t *swap = current_inserts;
            current_inserts = next_inserts;
            next_inserts = swap;
            swap = current_deletes;
            current_deletes = next_deletes;
            next_deletes = swap;
        }
    }
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = shorter_dim;
        char const *second_slice = longer + next_diagonal_index - shorter_dim;
        unsigned const cells_count = next_diagonal_length - 1;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            char first_char = shorter[cells_count - i - 1];
            char second_char = second_slice[i];
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_char, (sz_u32_t)(sz_u8_t)second_char);
            sz_i32_t cell;
            if (!is_affine) {
                cell = szs_cell_linear_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            }
            else {
                sz_i32_t out_ins, out_del;
                cell = szs_cell_affine_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                           (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend, objective,
                                           is_local, &out_ins, &out_del);
                next_inserts[i] = (sz_i16_t)out_ins;
                next_deletes[i] = (sz_i16_t)out_del;
            }
            next_scores[i] = (sz_i16_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            if (!is_affine) {
                next_scores[next_diagonal_length - 1] = (sz_i16_t)szs_init_score_linear(next_diagonal_index, gap,
                                                                                        is_local);
            }
            else {
                next_scores[next_diagonal_length - 1] = (sz_i16_t)szs_init_score_affine(next_diagonal_index, gap_open,
                                                                                        gap_extend, is_local);
                next_deletes[next_diagonal_length - 1] = (sz_i16_t)szs_init_gap_affine(next_diagonal_index, gap_open,
                                                                                       gap_extend, is_local);
            }
        }
        if (is_affine) {
            sz_i16_t *swap = current_inserts;
            current_inserts = next_inserts;
            next_inserts = swap;
            swap = current_deletes;
            current_deletes = next_deletes;
            next_deletes = swap;
        }
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
            char first_char = first_slice[next_diagonal_length - i - 1];
            char second_char = second_slice[i];
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_char, (sz_u32_t)(sz_u8_t)second_char);
            sz_i32_t cell;
            if (!is_affine) {
                cell = szs_cell_linear_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_scores[i + 1], sub, gap, objective, is_local);
            }
            else {
                sz_i32_t out_ins, out_del;
                cell = szs_cell_affine_i32((sz_i32_t)previous_scores[i], (sz_i32_t)current_scores[i],
                                           (sz_i32_t)current_inserts[i], (sz_i32_t)current_scores[i + 1],
                                           (sz_i32_t)current_deletes[i + 1], sub, gap_open, gap_extend, objective,
                                           is_local, &out_ins, &out_del);
                next_inserts[i] = (sz_i16_t)out_ins;
                next_deletes[i] = (sz_i16_t)out_del;
            }
            next_scores[i] = (sz_i16_t)cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        sz_i16_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        if (is_affine) {
            sz_i16_t *swap = current_inserts;
            current_inserts = next_inserts;
            next_inserts = swap;
            swap = current_deletes;
            current_deletes = next_deletes;
            next_deletes = swap;
        }
        previous_scores++;
        __syncwarp();
    }

    if (is_local) return szs_best_in_warp_i32(best_score, objective);
    sz_i32_t final_score = lane == 0 ? (sz_i32_t)current_scores[0] : 0;
    return __shfl_sync(0xffffffffu, final_score, 0);
}

__device__ inline sz_i32_t szs_walk_pair_warp_base_i32( //
    sz_u8_t const *shorter_global, sz_size_t shorter_length, sz_u8_t const *longer_global, sz_size_t longer_length,
    szs_device_costs_t const *costs, char *shared_memory_for_warp, unsigned lane) {

    sz_error_cost_t const gap_open = costs->gap_open, gap_extend = costs->gap_extend;
    sz_error_cost_t const gap = gap_open;
    int const is_affine = gap_open != gap_extend;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;
    unsigned const warp_size = 32;

    if (shorter_length == 0 || longer_length == 0) {
        if (is_local) return 0;
        sz_size_t const only_length = shorter_length ? shorter_length : longer_length;
        if (only_length == 0) return 0;
        if (!is_affine) return (sz_i32_t)((sz_i64_t)gap * (sz_i64_t)only_length);
        return (sz_i32_t)((sz_i64_t)gap_open + (sz_i64_t)gap_extend * (sz_i64_t)(only_length - 1));
    }

    unsigned const shorter_dim = (unsigned)shorter_length + 1;
    unsigned const longer_dim = (unsigned)longer_length + 1;
    unsigned const diagonals_count = shorter_dim + longer_dim - 1;
    unsigned const max_diagonal_length = (unsigned)shorter_length + 1;
    unsigned const count_diagonals = is_affine ? 7 : 3;
    unsigned const bytes_per_diagonal = szs_round_up_to_multiple(max_diagonal_length * sizeof(sz_i32_t), 4);

    sz_i32_t *previous_scores = (sz_i32_t *)(shared_memory_for_warp + 0 * bytes_per_diagonal);
    sz_i32_t *current_scores = (sz_i32_t *)(shared_memory_for_warp + 1 * bytes_per_diagonal);
    sz_i32_t *next_scores = (sz_i32_t *)(shared_memory_for_warp + 2 * bytes_per_diagonal);
    sz_i32_t *current_inserts = (sz_i32_t *)(shared_memory_for_warp + 3 * bytes_per_diagonal);
    sz_i32_t *next_inserts = (sz_i32_t *)(shared_memory_for_warp + 4 * bytes_per_diagonal);
    sz_i32_t *current_deletes = (sz_i32_t *)(shared_memory_for_warp + 5 * bytes_per_diagonal);
    sz_i32_t *next_deletes = (sz_i32_t *)(shared_memory_for_warp + 6 * bytes_per_diagonal);
    char *longer = shared_memory_for_warp + count_diagonals * bytes_per_diagonal;
    char *shorter = longer + longer_length;

    for (unsigned i = lane; i < longer_length; i += warp_size) longer[i] = (char)longer_global[i];
    for (unsigned i = lane; i < shorter_length; i += warp_size) shorter[i] = (char)shorter_global[i];

    if (lane == 0) {
        if (!is_affine) {
            previous_scores[0] = szs_init_score_linear(0, gap, is_local);
            current_scores[0] = szs_init_score_linear(1, gap, is_local);
            current_scores[1] = szs_init_score_linear(1, gap, is_local);
        }
        else {
            previous_scores[0] = szs_init_score_affine(0, gap_open, gap_extend, is_local);
            current_scores[0] = szs_init_score_affine(1, gap_open, gap_extend, is_local);
            current_scores[1] = szs_init_score_affine(1, gap_open, gap_extend, is_local);
            current_inserts[0] = szs_init_gap_affine(1, gap_open, gap_extend, is_local);
            current_deletes[1] = szs_init_gap_affine(1, gap_open, gap_extend, is_local);
        }
    }
    __syncwarp();

    sz_i32_t best_score = 0;
    unsigned next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = next_diagonal_index + 1;
        unsigned const cells_count = next_diagonal_length - 2;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            char first_char = shorter[cells_count - i - 1];
            char second_char = longer[i];
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_char, (sz_u32_t)(sz_u8_t)second_char);
            sz_i32_t cell;
            if (!is_affine) {
                cell = szs_cell_linear_i32(previous_scores[i], current_scores[i], current_scores[i + 1], sub, gap,
                                           objective, is_local);
            }
            else {
                sz_i32_t out_ins, out_del;
                cell = szs_cell_affine_i32(previous_scores[i], current_scores[i], current_inserts[i],
                                           current_scores[i + 1], current_deletes[i + 1], sub, gap_open, gap_extend,
                                           objective, is_local, &out_ins, &out_del);
                next_inserts[i + 1] = out_ins;
                next_deletes[i + 1] = out_del;
            }
            next_scores[i + 1] = cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            if (!is_affine) {
                next_scores[0] = szs_init_score_linear(next_diagonal_index, gap, is_local);
                next_scores[next_diagonal_length - 1] = szs_init_score_linear(next_diagonal_index, gap, is_local);
            }
            else {
                next_scores[0] = szs_init_score_affine(next_diagonal_index, gap_open, gap_extend, is_local);
                next_scores[next_diagonal_length - 1] = szs_init_score_affine(next_diagonal_index, gap_open, gap_extend,
                                                                              is_local);
                next_inserts[0] = szs_init_gap_affine(next_diagonal_index, gap_open, gap_extend, is_local);
                next_deletes[next_diagonal_length - 1] = szs_init_gap_affine(next_diagonal_index, gap_open, gap_extend,
                                                                             is_local);
            }
        }
        __syncwarp();
        sz_i32_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        if (is_affine) {
            sz_i32_t *swap = current_inserts;
            current_inserts = next_inserts;
            next_inserts = swap;
            swap = current_deletes;
            current_deletes = next_deletes;
            next_deletes = swap;
        }
    }
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = shorter_dim;
        char const *second_slice = longer + next_diagonal_index - shorter_dim;
        unsigned const cells_count = next_diagonal_length - 1;
        for (unsigned i = lane; i < cells_count; i += warp_size) {
            char first_char = shorter[cells_count - i - 1];
            char second_char = second_slice[i];
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_char, (sz_u32_t)(sz_u8_t)second_char);
            sz_i32_t cell;
            if (!is_affine) {
                cell = szs_cell_linear_i32(previous_scores[i], current_scores[i], current_scores[i + 1], sub, gap,
                                           objective, is_local);
            }
            else {
                sz_i32_t out_ins, out_del;
                cell = szs_cell_affine_i32(previous_scores[i], current_scores[i], current_inserts[i],
                                           current_scores[i + 1], current_deletes[i + 1], sub, gap_open, gap_extend,
                                           objective, is_local, &out_ins, &out_del);
                next_inserts[i] = out_ins;
                next_deletes[i] = out_del;
            }
            next_scores[i] = cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (lane == 0) {
            if (!is_affine) {
                next_scores[next_diagonal_length - 1] = szs_init_score_linear(next_diagonal_index, gap, is_local);
            }
            else {
                next_scores[next_diagonal_length - 1] = szs_init_score_affine(next_diagonal_index, gap_open, gap_extend,
                                                                              is_local);
                next_deletes[next_diagonal_length - 1] = szs_init_gap_affine(next_diagonal_index, gap_open, gap_extend,
                                                                             is_local);
            }
        }
        if (is_affine) {
            sz_i32_t *swap = current_inserts;
            current_inserts = next_inserts;
            next_inserts = swap;
            swap = current_deletes;
            current_deletes = next_deletes;
            next_deletes = swap;
        }
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
            char first_char = first_slice[next_diagonal_length - i - 1];
            char second_char = second_slice[i];
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_char, (sz_u32_t)(sz_u8_t)second_char);
            sz_i32_t cell;
            if (!is_affine) {
                cell = szs_cell_linear_i32(previous_scores[i], current_scores[i], current_scores[i + 1], sub, gap,
                                           objective, is_local);
            }
            else {
                sz_i32_t out_ins, out_del;
                cell = szs_cell_affine_i32(previous_scores[i], current_scores[i], current_inserts[i],
                                           current_scores[i + 1], current_deletes[i + 1], sub, gap_open, gap_extend,
                                           objective, is_local, &out_ins, &out_del);
                next_inserts[i] = out_ins;
                next_deletes[i] = out_del;
            }
            next_scores[i] = cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        sz_i32_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        if (is_affine) {
            sz_i32_t *swap = current_inserts;
            current_inserts = next_inserts;
            next_inserts = swap;
            swap = current_deletes;
            current_deletes = next_deletes;
            next_deletes = swap;
        }
        previous_scores++;
        __syncwarp();
    }

    if (is_local) return szs_best_in_warp_i32(best_score, objective);
    sz_i32_t final_score = lane == 0 ? current_scores[0] : 0;
    return __shfl_sync(0xffffffffu, final_score, 0);
}

#pragma endregion

#pragma region - Cost-Model Staging + Warp-Per-Pair Entry Bodies

// Stages the cost model (LUT included) into a block-shared prefix of dynamic SMEM once per block, then
// returns: a pointer to the staged costs, the per-warp SMEM slice for this warp, the lane index, and the
// global warp/stride for the grid-stride task loop. The dynamic SMEM layout per block is:
//   [ szs_device_costs_t prefix (rounded to 16B) ][ warp 0 slice ][ warp 1 slice ] ...
// The host passes `shared_memory_per_warp` so each warp owns `shared_memory_per_warp` bytes after the
// prefix; this keeps the LUT off device constant memory and out of a separate-compilation requirement.
struct szs_warp_context_t {
    szs_device_costs_t *costs;
    char *warp_slice;
    unsigned lane;
    unsigned long long global_warp_index;
    unsigned long long warps_per_device;
};

__device__ inline unsigned szs_costs_prefix_bytes() { return (unsigned)((sizeof(szs_device_costs_t) + 15u) & ~15u); }

__device__ inline szs_warp_context_t szs_make_warp_context(szs_device_costs_t costs, unsigned shared_memory_per_warp) {
    extern __shared__ char szs_shared_block[];
    for (unsigned byte_index = threadIdx.x; byte_index < sizeof(szs_device_costs_t); byte_index += blockDim.x)
        szs_shared_block[byte_index] = reinterpret_cast<char *>(&costs)[byte_index];
    __syncthreads();

    unsigned const warp_size = 32;
    unsigned const warps_per_block = blockDim.x / warp_size;
    unsigned const warp_in_block = threadIdx.x / warp_size;
    char *const warps_base = szs_shared_block + szs_costs_prefix_bytes();

    szs_warp_context_t out;
    out.costs = reinterpret_cast<szs_device_costs_t *>(szs_shared_block);
    out.warp_slice = warps_base + (unsigned long long)warp_in_block * shared_memory_per_warp;
    out.lane = threadIdx.x % warp_size;
    out.global_warp_index = ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) / warp_size;
    out.warps_per_device = (unsigned long long)gridDim.x * warps_per_block;
    return out;
}

// Picks the shorter/longer order (the walker expects the shorter string first) and writes the result.
__device__ inline void szs_warp_write_back(szs_pair_task_t &task, sz_i32_t value, unsigned lane) {
    if (lane == 0) task.result = (sz_i64_t)value;
}

#pragma endregion

#pragma region - Base Warp-Per-Pair Entry Kernels (emitted only when this is the compiled tier)

#ifndef SZS_SUPPRESS_BASE_ENTRIES

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
        sz_i32_t v = szs_walk_pair_warp_base_u8(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
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
        sz_i32_t v = szs_walk_pair_warp_base_u16(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
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
        sz_i32_t v = szs_walk_pair_warp_base_u8(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
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
        sz_i32_t v = szs_walk_pair_warp_base_u16(sp, sl, lp, ll, ctx.costs, ctx.warp_slice, ctx.lane);
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

#endif // SZS_SUPPRESS_BASE_ENTRIES

#pragma endregion

#pragma region - Device-Spanning Walker (one HUGE single pair across the whole grid)

// The whole grid cooperates on one pair, with `cooperative_groups::this_grid().sync()` between
// anti-diagonals; the central-band diagonal rotation is a grid-strided copy through global memory (a
// `cuda::memcpy_async` pipeline would lower to the same data movement but needs sm_70+, so the explicit
// copy keeps the base tier at sm_50). The diagonals live in a caller-provided global scratch
// (`diagonals_ptr`), not shared memory, so an arbitrarily long pair fits. Launched from the host via
// `cudaLaunchCooperativeKernel`. One entry per (op-objective/locality x gap): the host resolver looks
// these up by the `szs_<op>_device_<gap>` names. These live in the base tier only; the higher-tier TUs
// include `cuda_maxwell.cuh` (its header guard prevents re-emission within a single translation unit) and
// inherit them, so every tier's PTX module carries the device-spanning entries.

__device__ inline sz_i32_t szs_span_pair_i32(             //
    sz_u8_t const *shorter_ptr, sz_size_t shorter_length, //
    sz_u8_t const *longer_ptr, sz_size_t longer_length,   //
    szs_device_costs_t const *costs, sz_i32_t *diagonals_ptr) {

    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    sz_error_cost_t const gap_open = costs->gap_open, gap_extend = costs->gap_extend;
    sz_error_cost_t const gap = gap_open;
    int const is_affine = gap_open != gap_extend;
    sz_similarity_objective_t const objective = costs->objective;
    int const is_local = costs->is_local;

    unsigned const shorter_dim = (unsigned)shorter_length + 1;
    unsigned const longer_dim = (unsigned)longer_length + 1;
    unsigned const diagonals_count = shorter_dim + longer_dim - 1;
    unsigned const max_diagonal_length = (unsigned)shorter_length + 1;

    sz_i32_t *previous_scores = diagonals_ptr + 0 * max_diagonal_length;
    sz_i32_t *current_scores = diagonals_ptr + 1 * max_diagonal_length;
    sz_i32_t *next_scores = diagonals_ptr + 2 * max_diagonal_length;
    sz_i32_t *current_inserts = diagonals_ptr + 3 * max_diagonal_length;
    sz_i32_t *next_inserts = diagonals_ptr + 4 * max_diagonal_length;
    sz_i32_t *current_deletes = diagonals_ptr + 5 * max_diagonal_length;
    sz_i32_t *next_deletes = diagonals_ptr + 6 * max_diagonal_length;

    bool const is_main_thread = blockIdx.x == 0 && threadIdx.x == 0;
    unsigned const global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned const global_thread_step = blockDim.x * gridDim.x;

    if (is_main_thread) {
        if (!is_affine) {
            previous_scores[0] = szs_init_score_linear(0, gap, is_local);
            current_scores[0] = szs_init_score_linear(1, gap, is_local);
            current_scores[1] = szs_init_score_linear(1, gap, is_local);
        }
        else {
            previous_scores[0] = szs_init_score_affine(0, gap_open, gap_extend, is_local);
            current_scores[0] = szs_init_score_affine(1, gap_open, gap_extend, is_local);
            current_scores[1] = szs_init_score_affine(1, gap_open, gap_extend, is_local);
            current_inserts[0] = szs_init_gap_affine(1, gap_open, gap_extend, is_local);
            current_deletes[1] = szs_init_gap_affine(1, gap_open, gap_extend, is_local);
        }
    }

    sz_i32_t best_score = 0;
    unsigned next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = next_diagonal_index + 1;
        unsigned const cells_count = next_diagonal_length - 2;
        for (unsigned i = global_thread_index; i < cells_count; i += global_thread_step) {
            char first_char = (char)shorter_ptr[cells_count - i - 1];
            char second_char = (char)longer_ptr[i];
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_char, (sz_u32_t)(sz_u8_t)second_char);
            sz_i32_t cell;
            if (!is_affine) {
                cell = szs_cell_linear_i32(previous_scores[i], current_scores[i], current_scores[i + 1], sub, gap,
                                           objective, is_local);
            }
            else {
                sz_i32_t out_ins, out_del;
                cell = szs_cell_affine_i32(previous_scores[i], current_scores[i], current_inserts[i],
                                           current_scores[i + 1], current_deletes[i + 1], sub, gap_open, gap_extend,
                                           objective, is_local, &out_ins, &out_del);
                next_inserts[i + 1] = out_ins;
                next_deletes[i + 1] = out_del;
            }
            next_scores[i + 1] = cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (is_main_thread) {
            if (!is_affine) {
                next_scores[0] = szs_init_score_linear(next_diagonal_index, gap, is_local);
                next_scores[next_diagonal_length - 1] = szs_init_score_linear(next_diagonal_index, gap, is_local);
            }
            else {
                next_scores[0] = szs_init_score_affine(next_diagonal_index, gap_open, gap_extend, is_local);
                next_scores[next_diagonal_length - 1] = szs_init_score_affine(next_diagonal_index, gap_open, gap_extend,
                                                                              is_local);
                next_inserts[0] = szs_init_gap_affine(next_diagonal_index, gap_open, gap_extend, is_local);
                next_deletes[next_diagonal_length - 1] = szs_init_gap_affine(next_diagonal_index, gap_open, gap_extend,
                                                                             is_local);
            }
        }
        grid.sync();
        sz_i32_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        if (is_affine) {
            sz_i32_t *swap = current_inserts;
            current_inserts = next_inserts;
            next_inserts = swap;
            swap = current_deletes;
            current_deletes = next_deletes;
            next_deletes = swap;
        }
    }
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        unsigned const next_diagonal_length = shorter_dim;
        sz_u8_t const *second_slice = longer_ptr + next_diagonal_index - shorter_dim;
        unsigned const cells_count = next_diagonal_length - 1;
        for (unsigned i = global_thread_index; i < cells_count; i += global_thread_step) {
            char first_char = (char)shorter_ptr[cells_count - i - 1];
            char second_char = (char)second_slice[i];
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_char, (sz_u32_t)(sz_u8_t)second_char);
            sz_i32_t cell;
            if (!is_affine) {
                cell = szs_cell_linear_i32(previous_scores[i], current_scores[i], current_scores[i + 1], sub, gap,
                                           objective, is_local);
            }
            else {
                sz_i32_t out_ins, out_del;
                cell = szs_cell_affine_i32(previous_scores[i], current_scores[i], current_inserts[i],
                                           current_scores[i + 1], current_deletes[i + 1], sub, gap_open, gap_extend,
                                           objective, is_local, &out_ins, &out_del);
                next_inserts[i] = out_ins;
                next_deletes[i] = out_del;
            }
            next_scores[i] = cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        if (is_main_thread) {
            if (!is_affine) {
                next_scores[next_diagonal_length - 1] = szs_init_score_linear(next_diagonal_index, gap, is_local);
            }
            else {
                next_scores[next_diagonal_length - 1] = szs_init_score_affine(next_diagonal_index, gap_open, gap_extend,
                                                                              is_local);
                next_deletes[next_diagonal_length - 1] = szs_init_gap_affine(next_diagonal_index, gap_open, gap_extend,
                                                                             is_local);
            }
        }
        if (is_affine) {
            sz_i32_t *swap = current_inserts;
            current_inserts = next_inserts;
            next_inserts = swap;
            swap = current_deletes;
            current_deletes = next_deletes;
            next_deletes = swap;
        }
        grid.sync();
        // Rotate the central-band score diagonals through global memory (can't alias in place).
        for (unsigned i = global_thread_index; i + 1 < next_diagonal_length; i += global_thread_step)
            previous_scores[i] = current_scores[i + 1];
        grid.sync();
        for (unsigned i = global_thread_index; i < next_diagonal_length; i += global_thread_step)
            current_scores[i] = next_scores[i];
        grid.sync();
    }
    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {
        unsigned const next_diagonal_length = diagonals_count - next_diagonal_index;
        sz_u8_t const *first_slice = shorter_ptr + next_diagonal_index - longer_dim;
        sz_u8_t const *second_slice = longer_ptr + next_diagonal_index - shorter_dim;
        for (unsigned i = global_thread_index; i < next_diagonal_length; i += global_thread_step) {
            char first_char = (char)first_slice[next_diagonal_length - i - 1];
            char second_char = (char)second_slice[i];
            sz_error_cost_t sub = szs_subs_cost(costs, (sz_u32_t)(sz_u8_t)first_char, (sz_u32_t)(sz_u8_t)second_char);
            sz_i32_t cell;
            if (!is_affine) {
                cell = szs_cell_linear_i32(previous_scores[i], current_scores[i], current_scores[i + 1], sub, gap,
                                           objective, is_local);
            }
            else {
                sz_i32_t out_ins, out_del;
                cell = szs_cell_affine_i32(previous_scores[i], current_scores[i], current_inserts[i],
                                           current_scores[i + 1], current_deletes[i + 1], sub, gap_open, gap_extend,
                                           objective, is_local, &out_ins, &out_del);
                next_inserts[i] = out_ins;
                next_deletes[i] = out_del;
            }
            next_scores[i] = cell;
            best_score = szs_pick_best_i32(best_score, cell, objective);
        }
        grid.sync();
        sz_i32_t *rotated = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = rotated;
        if (is_affine) {
            sz_i32_t *swap = current_inserts;
            current_inserts = next_inserts;
            next_inserts = swap;
            swap = current_deletes;
            current_deletes = next_deletes;
            next_deletes = swap;
        }
        previous_scores++;
    }

    if (is_local) return best_score; // The host reduces the per-thread maxima; here it picks lead-thread's.
    return current_scores[0];
}

#pragma endregion

#pragma region - Device-Spanning Entry Kernels

// One global scratch holds the diagonals; `result_ptr` receives the lead-thread score. For Smith-Waterman
// (local) the per-thread best must be reduced grid-wide; we route the local best through an atomic on
// `result_ptr` so the answer is exact regardless of which thread held the global maximum.
extern "C" __global__ void szs_levenshtein_device_linear(szs_pair_task_t *task, szs_device_costs_t costs,
                                                         sz_i32_t *diagonals_ptr) {
    sz_size_t sl = task->a_len, ll = task->b_len;
    sz_u8_t const *sp = task->a_ptr, *lp = task->b_ptr;
    if (sl > ll) {
        sl = task->b_len;
        ll = task->a_len;
        sp = task->b_ptr;
        lp = task->a_ptr;
    }
    sz_i32_t v = szs_span_pair_i32(sp, sl, lp, ll, &costs, diagonals_ptr);
    if (blockIdx.x == 0 && threadIdx.x == 0) task->result = (sz_i64_t)v;
}

extern "C" __global__ void szs_levenshtein_device_affine(szs_pair_task_t *task, szs_device_costs_t costs,
                                                         sz_i32_t *diagonals_ptr) {
    sz_size_t sl = task->a_len, ll = task->b_len;
    sz_u8_t const *sp = task->a_ptr, *lp = task->b_ptr;
    if (sl > ll) {
        sl = task->b_len;
        ll = task->a_len;
        sp = task->b_ptr;
        lp = task->a_ptr;
    }
    sz_i32_t v = szs_span_pair_i32(sp, sl, lp, ll, &costs, diagonals_ptr);
    if (blockIdx.x == 0 && threadIdx.x == 0) task->result = (sz_i64_t)v;
}

// Global alignment (NW): lead thread holds the answer in current_scores[0].
extern "C" __global__ void szs_needleman_wunsch_device_linear(szs_pair_task_t *task, szs_device_costs_t costs,
                                                              sz_i32_t *diagonals_ptr) {
    sz_size_t sl = task->a_len, ll = task->b_len;
    sz_u8_t const *sp = task->a_ptr, *lp = task->b_ptr;
    if (sl > ll) {
        sl = task->b_len;
        ll = task->a_len;
        sp = task->b_ptr;
        lp = task->a_ptr;
    }
    sz_i32_t v = szs_span_pair_i32(sp, sl, lp, ll, &costs, diagonals_ptr);
    if (blockIdx.x == 0 && threadIdx.x == 0) task->result = (sz_i64_t)v;
}

extern "C" __global__ void szs_needleman_wunsch_device_affine(szs_pair_task_t *task, szs_device_costs_t costs,
                                                              sz_i32_t *diagonals_ptr) {
    sz_size_t sl = task->a_len, ll = task->b_len;
    sz_u8_t const *sp = task->a_ptr, *lp = task->b_ptr;
    if (sl > ll) {
        sl = task->b_len;
        ll = task->a_len;
        sp = task->b_ptr;
        lp = task->a_ptr;
    }
    sz_i32_t v = szs_span_pair_i32(sp, sl, lp, ll, &costs, diagonals_ptr);
    if (blockIdx.x == 0 && threadIdx.x == 0) task->result = (sz_i64_t)v;
}

// Local alignment (SW): every thread's per-thread maximum must be folded grid-wide; max via atomic on the
// task result, seeded to 0 by the lead thread before the grid starts the recurrence.
extern "C" __global__ void szs_smith_waterman_device_linear(szs_pair_task_t *task, szs_device_costs_t costs,
                                                            sz_i32_t *diagonals_ptr) {
    sz_size_t sl = task->a_len, ll = task->b_len;
    sz_u8_t const *sp = task->a_ptr, *lp = task->b_ptr;
    if (sl > ll) {
        sl = task->b_len;
        ll = task->a_len;
        sp = task->b_ptr;
        lp = task->a_ptr;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) task->result = 0;
    namespace cg = cooperative_groups;
    cg::this_grid().sync();
    sz_i32_t v = szs_span_pair_i32(sp, sl, lp, ll, &costs, diagonals_ptr);
    if (v > 0) atomicMax((unsigned long long *)&task->result, (unsigned long long)(sz_i64_t)v);
}

extern "C" __global__ void szs_smith_waterman_device_affine(szs_pair_task_t *task, szs_device_costs_t costs,
                                                            sz_i32_t *diagonals_ptr) {
    sz_size_t sl = task->a_len, ll = task->b_len;
    sz_u8_t const *sp = task->a_ptr, *lp = task->b_ptr;
    if (sl > ll) {
        sl = task->b_len;
        ll = task->a_len;
        sp = task->b_ptr;
        lp = task->a_ptr;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) task->result = 0;
    namespace cg = cooperative_groups;
    cg::this_grid().sync();
    sz_i32_t v = szs_span_pair_i32(sp, sl, lp, ll, &costs, diagonals_ptr);
    if (v > 0) atomicMax((unsigned long long *)&task->result, (unsigned long long)(sz_i64_t)v);
}

#pragma endregion

#endif // __CUDACC__

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SIMILARITIES_CUDA_MAXWELL_CUH_
