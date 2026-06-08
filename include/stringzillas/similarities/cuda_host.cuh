/**
 *  @file include/stringzillas/similarities/cuda_host.cuh
 *  @brief Plain-C++ CUDA Driver-API host launchers for the similarity backend.
 *  @author Ash Vardanian
 *
 *  Contains NO `__global__` code - only host code that:
 *  1. Loads ONE PTX blob per tier with `cuModuleLoadData` (the blob is `bin2c`-embedded into the .so),
 *     forcing the CUDART primary context first via `cudaFree(0)` so unified-memory allocs and the
 *     Driver-API module share one context.
 *  2. Resolves every `CUfunction` by name (the 14 warp-per-pair `szs_*` entries + the 6 device-spanning
 *     `szs_*_device_*` entries the base TU emits), so a missing symbol fails loudly at load time.
 *  3. Stages ALL pairs into ONE managed arena (task array + all string bytes + per-task scratch), bins them
 *     by DP cell width, and launches ONCE per (width, mode) group: warp-per-pair with grid ~= `32*SMs`
 *     blocks of several warps (a `warp_tasks_density`-style heuristic sized off the device's shared
 *     memory), or `cudaLaunchCooperativeKernel` device-spanning for a huge single pair.
 *
 *  The public entry points are plain C functions matching the `dispatch.h` typedefs:
 *  `szs_levenshtein_cuda/_kepler/_hopper`, `szs_needleman_wunsch_cuda/_hopper`, `szs_smith_waterman_*`.
 *  Each takes the engine POD + a device scope; the tier is chosen by which PTX module the entry loads.
 */
#ifndef STRINGZILLAS_SIMILARITIES_CUDA_HOST_CUH_
#define STRINGZILLAS_SIMILARITIES_CUDA_HOST_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib> // `malloc`/`free` for the task-permutation scratch

#include "stringzilla/types.h"
#include "stringzillas/types.h"
#include "stringzillas/similarities/serial.h"        // shared cell math (`sz_dp_memory`), input decoding helpers
#include "stringzillas/similarities/cuda_maxwell.cuh" // `szs_pair_task_t`, `szs_device_costs_t`

// The PTX blobs are emitted by `bin2c` from the per-tier `.cu` files and linked into the .so.
// Each is a NUL-terminated C string. They are declared here and defined by the generated `*_ptx.c`.
extern "C" {
extern unsigned char sim_base_ptx[];
extern unsigned char sim_kepler_ptx[];
extern unsigned char sim_hopper_ptx[];
}

namespace ashvardanian {
namespace stringzillas {

#pragma region - Module + Function Resolution

// Holds one loaded similarity PTX module and its resolved entry points (one per kernel symbol).
struct szs_similarity_module_t {
    CUmodule module = nullptr;
    CUfunction szs_levenshtein_warp_linear_u8 = nullptr;
    CUfunction szs_levenshtein_warp_linear_u16 = nullptr;
    CUfunction szs_levenshtein_warp_linear_u32 = nullptr;
    CUfunction szs_levenshtein_warp_affine_u8 = nullptr;
    CUfunction szs_levenshtein_warp_affine_u16 = nullptr;
    CUfunction szs_levenshtein_warp_affine_u32 = nullptr;
    CUfunction szs_needleman_wunsch_warp_linear_i16 = nullptr;
    CUfunction szs_needleman_wunsch_warp_linear_i32 = nullptr;
    CUfunction szs_needleman_wunsch_warp_affine_i16 = nullptr;
    CUfunction szs_needleman_wunsch_warp_affine_i32 = nullptr;
    CUfunction szs_smith_waterman_warp_linear_i16 = nullptr;
    CUfunction szs_smith_waterman_warp_linear_i32 = nullptr;
    CUfunction szs_smith_waterman_warp_affine_i16 = nullptr;
    CUfunction szs_smith_waterman_warp_affine_i32 = nullptr;
    // Device-spanning entries for a huge single pair (resolved from the base PTX; absent => no fallback).
    CUfunction szs_levenshtein_device_linear = nullptr;
    CUfunction szs_levenshtein_device_affine = nullptr;
    CUfunction szs_needleman_wunsch_device_linear = nullptr;
    CUfunction szs_needleman_wunsch_device_affine = nullptr;
    CUfunction szs_smith_waterman_device_linear = nullptr;
    CUfunction szs_smith_waterman_device_affine = nullptr;
    bool ok = false;
};

// Forces the CUDART primary context so Driver-API modules and unified allocs share it.
inline bool szs_ensure_primary_context_() {
    static bool initialized = false;
    if (initialized) return true;
    if (cuInit(0) != CUDA_SUCCESS) return false;
    if (cudaFree(0) != cudaSuccess) return false; // Creates & binds the CUDART primary context.
    initialized = true;
    return true;
}

// Loads a PTX blob into the current context and resolves all similarity entry points by name.
inline szs_similarity_module_t szs_load_similarity_module_(unsigned char const *ptx) {
    szs_similarity_module_t out;
    if (!szs_ensure_primary_context_()) return out;
    if (cuModuleLoadData(&out.module, (void const *)ptx) != CUDA_SUCCESS) return out;
    bool all = true;
    all = all && (cuModuleGetFunction(&out.szs_levenshtein_warp_linear_u8, out.module,
                                      "szs_levenshtein_warp_linear_u8") == CUDA_SUCCESS);
    all = all && (cuModuleGetFunction(&out.szs_levenshtein_warp_linear_u16, out.module,
                                      "szs_levenshtein_warp_linear_u16") == CUDA_SUCCESS);
    all = all && (cuModuleGetFunction(&out.szs_levenshtein_warp_linear_u32, out.module,
                                      "szs_levenshtein_warp_linear_u32") == CUDA_SUCCESS);
    all = all && (cuModuleGetFunction(&out.szs_levenshtein_warp_affine_u8, out.module,
                                      "szs_levenshtein_warp_affine_u8") == CUDA_SUCCESS);
    all = all && (cuModuleGetFunction(&out.szs_levenshtein_warp_affine_u16, out.module,
                                      "szs_levenshtein_warp_affine_u16") == CUDA_SUCCESS);
    all = all && (cuModuleGetFunction(&out.szs_levenshtein_warp_affine_u32, out.module,
                                      "szs_levenshtein_warp_affine_u32") == CUDA_SUCCESS);
    all = all && (cuModuleGetFunction(&out.szs_needleman_wunsch_warp_linear_i16, out.module,
                                      "szs_needleman_wunsch_warp_linear_i16") == CUDA_SUCCESS);
    all = all && (cuModuleGetFunction(&out.szs_needleman_wunsch_warp_linear_i32, out.module,
                                      "szs_needleman_wunsch_warp_linear_i32") == CUDA_SUCCESS);
    all = all && (cuModuleGetFunction(&out.szs_needleman_wunsch_warp_affine_i16, out.module,
                                      "szs_needleman_wunsch_warp_affine_i16") == CUDA_SUCCESS);
    all = all && (cuModuleGetFunction(&out.szs_needleman_wunsch_warp_affine_i32, out.module,
                                      "szs_needleman_wunsch_warp_affine_i32") == CUDA_SUCCESS);
    all = all && (cuModuleGetFunction(&out.szs_smith_waterman_warp_linear_i16, out.module,
                                      "szs_smith_waterman_warp_linear_i16") == CUDA_SUCCESS);
    all = all && (cuModuleGetFunction(&out.szs_smith_waterman_warp_linear_i32, out.module,
                                      "szs_smith_waterman_warp_linear_i32") == CUDA_SUCCESS);
    all = all && (cuModuleGetFunction(&out.szs_smith_waterman_warp_affine_i16, out.module,
                                      "szs_smith_waterman_warp_affine_i16") == CUDA_SUCCESS);
    all = all && (cuModuleGetFunction(&out.szs_smith_waterman_warp_affine_i32, out.module,
                                      "szs_smith_waterman_warp_affine_i32") == CUDA_SUCCESS);
    // Device-spanning entries are best-effort: only the base PTX defines them, so don't gate `ok` on them.
    cuModuleGetFunction(&out.szs_levenshtein_device_linear, out.module, "szs_levenshtein_device_linear");
    cuModuleGetFunction(&out.szs_levenshtein_device_affine, out.module, "szs_levenshtein_device_affine");
    cuModuleGetFunction(&out.szs_needleman_wunsch_device_linear, out.module, "szs_needleman_wunsch_device_linear");
    cuModuleGetFunction(&out.szs_needleman_wunsch_device_affine, out.module, "szs_needleman_wunsch_device_affine");
    cuModuleGetFunction(&out.szs_smith_waterman_device_linear, out.module, "szs_smith_waterman_device_linear");
    cuModuleGetFunction(&out.szs_smith_waterman_device_affine, out.module, "szs_smith_waterman_device_affine");
    out.ok = all;
    return out;
}

#pragma endregion

#pragma region - Cost-Model + Width Helpers

// Magnitude (max absolute cost) of a substitution LUT, for width selection.
inline sz_size_t szs_subs_magnitude_(sz_substitution_costs_t const *lut) {
    sz_size_t magnitude = 0;
    sz_size_t classes = lut->alphabet_size ? lut->alphabet_size : SZ_SUBS_MAX_CLASSES;
    for (sz_size_t r = 0; r < classes; ++r)
        for (sz_size_t c = 0; c < classes; ++c) {
            sz_error_cost_t v = lut->costs[r][c];
            sz_size_t a = (sz_size_t)(v < 0 ? -(sz_i32_t)v : (sz_i32_t)v);
            if (a > magnitude) magnitude = a;
        }
    return magnitude;
}

inline sz_size_t szs_cost_magnitude_(sz_error_cost_t v) { return (sz_size_t)(v < 0 ? -(sz_i32_t)v : (sz_i32_t)v); }

// Picks the DP cell width (1/2/4 bytes) for one pair, clamping u64/i64 down to the widest wave width (4).
inline unsigned szs_bytes_per_cell_(szs_pair_spans_t const &pair, bool is_affine, sz_size_t sub_magnitude,
                                   sz_size_t gap_magnitude, bool is_score) {
    sz_dp_memory_t mem = sz_dp_memory(pair.a_len, pair.b_len, is_affine ? sz_gaps_affine_k : sz_gaps_linear_k,
                                      sub_magnitude, gap_magnitude, 1, 4, is_score ? 1 : 0);
    if (!is_score)
        return mem.bytes_per_cell <= sz_one_byte_per_cell_k ? 1 : mem.bytes_per_cell == sz_two_bytes_per_cell_k ? 2 : 4;
    return mem.bytes_per_cell <= sz_two_bytes_per_cell_k ? 2 : 4;
}

// Per-warp shared-memory bytes the walker needs for one pair at a given launch width. Mirrors the walker's
// layout exactly: `count_diagonals` diagonals, each rounded to 4B, plus both strings (1 byte each).
inline sz_size_t szs_pair_warp_slice_bytes_(sz_size_t a_len, sz_size_t b_len, unsigned bytes_per_cell,
                                           sz_size_t count_diagonals) {
    sz_size_t shorter = sz_min_of_two(a_len, b_len);
    sz_size_t longer = sz_max_of_two(a_len, b_len);
    sz_size_t max_diag = shorter + 1;
    sz_size_t bytes_per_diag = sz_round_up_to_multiple(max_diag * bytes_per_cell, 4);
    return count_diagonals * bytes_per_diag + shorter + longer;
}

#pragma endregion

#pragma region - Device Properties (queried once)

struct szs_device_specs_t {
    int streaming_multiprocessors = 1;
    int warp_size = 32;
    size_t shared_memory_per_multiprocessor = 48u * 1024u; // Conservative default.
    int max_blocks_per_multiprocessor = 16;
    bool ok = false;
};

inline szs_device_specs_t const &szs_query_device_specs_() {
    static szs_device_specs_t specs = [] {
        szs_device_specs_t out;
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            out.streaming_multiprocessors = prop.multiProcessorCount;
            out.warp_size = prop.warpSize;
            out.shared_memory_per_multiprocessor = prop.sharedMemPerMultiprocessor;
            out.max_blocks_per_multiprocessor = prop.maxBlocksPerMultiProcessor ? prop.maxBlocksPerMultiProcessor : 16;
            out.ok = true;
        }
        return out;
    }();
    return specs;
}

// Picks how many warps share one block given the per-warp SMEM requirement, mirroring `warp_tasks_density`:
// the largest power-of-two density whose total block SMEM (+ the per-block cost prefix) fits a multiprocessor.
inline unsigned szs_warps_per_block_(sz_size_t per_warp_bytes, sz_size_t prefix_bytes, szs_device_specs_t const &specs) {
    unsigned const candidates[] = {32, 16, 8, 4, 2, 1};
    for (unsigned density : candidates) {
        if ((int)density > specs.max_blocks_per_multiprocessor * 4) continue; // keep within warp budget
        sz_size_t block_bytes = prefix_bytes + per_warp_bytes * density;
        if (block_bytes < specs.shared_memory_per_multiprocessor) return density;
    }
    return 1;
}

#pragma endregion

#pragma region - Generic Batched Launcher

// Runs one batch of pairs on the GPU in O(#width-groups) launches (NOT one launch per pair). All pairs are
// staged into ONE managed arena; pairs are binned by DP cell width; each (width, mode) bin launches once
// over `32*SMs` blocks of `warps_per_block` warps (warp-per-pair), unless a pair is too large for any
// per-warp shared-memory slice, in which case it routes to the device-spanning cooperative kernel.
template <typename write_back_t_>
inline sz_status_t szs_run_similarity_batch_( //
    szs_similarity_module_t const &mod, szs_device_costs_t costs, bool is_affine, sz_size_t sub_magnitude,
    sz_size_t gap_magnitude, bool is_score, szs_sequence_view_t inputs, sz_size_t count, write_back_t_ write_back,
    char const **error) {

    if (!mod.ok) {
        if (error) *error = "CUDA similarity module not loaded / entries unresolved";
        return sz_status_unknown_k;
    }
    if (count == 0) return sz_success_k;

    szs_device_specs_t const &specs = szs_query_device_specs_();
    sz_size_t const count_diagonals = is_affine ? 7 : 3;
    sz_size_t const costs_prefix = (sizeof(szs_device_costs_t) + 15u) & ~(sz_size_t)15u;
    // The largest per-warp SMEM slice a multiprocessor could ever host (one warp per block). Pairs needing
    // more than this route to the device-spanning path.
    sz_size_t const max_warp_slice = specs.shared_memory_per_multiprocessor > costs_prefix
                                         ? specs.shared_memory_per_multiprocessor - costs_prefix
                                         : 0;

    // 1) Stage all task PODs into one managed allocation.
    szs_pair_task_t *tasks = nullptr;
    if (cudaMallocManaged(&tasks, sizeof(szs_pair_task_t) * count) != cudaSuccess) {
        if (error) *error = "cudaMallocManaged(tasks) failed";
        return sz_bad_alloc_k;
    }

    // 2) Stage all input bytes into one managed arena (a||b back to back per pair).
    sz_size_t total_input_bytes = 0;
    for (sz_size_t i = 0; i < count; ++i) {
        szs_pair_spans_t pair = szs_inputs_pair_(&inputs, i);
        total_input_bytes += pair.a_len + pair.b_len;
    }
    sz_u8_t *inputs_arena = nullptr;
    if (total_input_bytes && cudaMallocManaged(&inputs_arena, total_input_bytes) != cudaSuccess) {
        cudaFree(tasks);
        if (error) *error = "cudaMallocManaged(inputs) failed";
        return sz_bad_alloc_k;
    }

    // 3) Compute per-task scratch sizes (device-spanning diagonals only) + total scratch.
    // Warp-per-pair tasks keep their diagonals in shared memory, so they need no global scratch; only the
    // device-spanning tasks need a global i32 diagonal buffer. We allocate ONE shared scratch big enough
    // for the largest device-spanning pair and reuse it (those launch serially anyway).
    sz_size_t arena_offset = 0;
    sz_size_t max_device_diagonals_cells = 0;
    bool any_device_spanning = false;
    for (sz_size_t i = 0; i < count; ++i) {
        szs_pair_spans_t pair = szs_inputs_pair_(&inputs, i);
        unsigned bytes_per_cell = szs_bytes_per_cell_(pair, is_affine, sub_magnitude, gap_magnitude, is_score);

        sz_u8_t *a_dev = inputs_arena + arena_offset;
        sz_u8_t *b_dev = a_dev + pair.a_len;
        for (sz_size_t k = 0; k < pair.a_len; ++k) a_dev[k] = ((sz_u8_t const *)pair.a_ptr)[k];
        for (sz_size_t k = 0; k < pair.b_len; ++k) b_dev[k] = ((sz_u8_t const *)pair.b_ptr)[k];
        arena_offset += pair.a_len + pair.b_len;

        tasks[i].a_ptr = a_dev;
        tasks[i].a_len = pair.a_len;
        tasks[i].b_ptr = b_dev;
        tasks[i].b_len = pair.b_len;
        tasks[i].scratch = nullptr;
        tasks[i].result = 0;

        // Per-warp SMEM requirement for this pair at its actual launch width. Decide device-spanning here.
        sz_size_t per_warp_need = szs_pair_warp_slice_bytes_(pair.a_len, pair.b_len, bytes_per_cell, count_diagonals);
        bool device_spanning = (pair.a_len && pair.b_len) && (per_warp_need > max_warp_slice);
        if (device_spanning) {
            any_device_spanning = true;
            sz_size_t shorter = sz_min_of_two(pair.a_len, pair.b_len);
            sz_size_t cells = count_diagonals * (shorter + 1);
            if (cells > max_device_diagonals_cells) max_device_diagonals_cells = cells;
        }
        (void)bytes_per_cell;
    }

    sz_i32_t *device_diagonals = nullptr;
    if (any_device_spanning) {
        if (cudaMallocManaged(&device_diagonals, sizeof(sz_i32_t) * max_device_diagonals_cells) != cudaSuccess) {
            cudaFree(inputs_arena);
            cudaFree(tasks);
            if (error) *error = "cudaMallocManaged(device diagonals) failed";
            return sz_bad_alloc_k;
        }
    }

    // 4) Bin warp-per-pair tasks by cell width and launch ONCE per (width) group.
    // We compact same-width warp tasks into a contiguous run of the `tasks` array (stable partition by
    // width: 1, 2, 4). Device-spanning tasks are handled separately, one cooperative launch each.
    // To keep the task array as the kernel argument and launch over a contiguous run, we sort indices.
    // A simple counting bucket over {1,2,4} keeps this O(count) and yields three contiguous launches max.
    sz_size_t *order = (sz_size_t *)malloc(sizeof(sz_size_t) * count);
    if (!order) {
        if (device_diagonals) cudaFree(device_diagonals);
        cudaFree(inputs_arena);
        cudaFree(tasks);
        if (error) *error = "malloc(order) failed";
        return sz_bad_alloc_k;
    }
    // Bucket key per task: 0 => device-spanning, 1/2/4 => warp width.
    auto task_width = [&](sz_size_t i) -> unsigned {
        szs_pair_spans_t pair = szs_inputs_pair_(&inputs, i);
        if (!pair.a_len || !pair.b_len) return is_score ? 2u : 1u; // empty: trivial, any width works
        unsigned bytes_per_cell = szs_bytes_per_cell_(pair, is_affine, sub_magnitude, gap_magnitude, is_score);
        sz_size_t per_warp_need = szs_pair_warp_slice_bytes_(pair.a_len, pair.b_len, bytes_per_cell, count_diagonals);
        if (per_warp_need > max_warp_slice) return 0u; // device-spanning
        return bytes_per_cell;
    };

    // Build a permutation grouped by width key in the order [1,2,4] (warp) then [0] (device-spanning last).
    sz_size_t cursor = 0;
    sz_size_t group_begin[4]; // for keys 1,2,4
    sz_size_t group_end[4];
    unsigned const width_keys[3] = {1, 2, 4};
    for (int g = 0; g < 3; ++g) {
        group_begin[g] = cursor;
        for (sz_size_t i = 0; i < count; ++i)
            if (task_width(i) == width_keys[g]) order[cursor++] = i;
        group_end[g] = cursor;
    }
    sz_size_t device_begin = cursor;
    for (sz_size_t i = 0; i < count; ++i)
        if (task_width(i) == 0u) order[cursor++] = i;
    sz_size_t device_end = cursor;

    // Reorder the managed `tasks` array to match `order` so each group is contiguous for one launch. We do
    // this with a scratch copy to keep it simple and O(count).
    szs_pair_task_t *tasks_sorted = nullptr;
    if (cudaMallocManaged(&tasks_sorted, sizeof(szs_pair_task_t) * count) != cudaSuccess) {
        free(order);
        if (device_diagonals) cudaFree(device_diagonals);
        cudaFree(inputs_arena);
        cudaFree(tasks);
        if (error) *error = "cudaMallocManaged(tasks_sorted) failed";
        return sz_bad_alloc_k;
    }
    for (sz_size_t k = 0; k < count; ++k) tasks_sorted[k] = tasks[order[k]];

    auto pick_warp_kernel = [&](unsigned bytes_per_cell) -> CUfunction {
        if (!is_score)
            return !is_affine ? (bytes_per_cell == 1   ? mod.szs_levenshtein_warp_linear_u8
                                 : bytes_per_cell == 2 ? mod.szs_levenshtein_warp_linear_u16
                                                       : mod.szs_levenshtein_warp_linear_u32)
                              : (bytes_per_cell == 1   ? mod.szs_levenshtein_warp_affine_u8
                                 : bytes_per_cell == 2 ? mod.szs_levenshtein_warp_affine_u16
                                                       : mod.szs_levenshtein_warp_affine_u32);
        if (costs.is_local)
            return !is_affine ? (bytes_per_cell == 2 ? mod.szs_smith_waterman_warp_linear_i16
                                                     : mod.szs_smith_waterman_warp_linear_i32)
                              : (bytes_per_cell == 2 ? mod.szs_smith_waterman_warp_affine_i16
                                                     : mod.szs_smith_waterman_warp_affine_i32);
        return !is_affine ? (bytes_per_cell == 2 ? mod.szs_needleman_wunsch_warp_linear_i16
                                                 : mod.szs_needleman_wunsch_warp_linear_i32)
                          : (bytes_per_cell == 2 ? mod.szs_needleman_wunsch_warp_affine_i16
                                                 : mod.szs_needleman_wunsch_warp_affine_i32);
    };

    sz_status_t status = sz_success_k;

    // One warp-per-pair launch per non-empty width group.
    for (int g = 0; g < 3 && status == sz_success_k; ++g) {
        sz_size_t const begin = group_begin[g], end = group_end[g];
        if (begin == end) continue;
        unsigned const bytes_per_cell = width_keys[g];

        // Largest per-warp SMEM requirement in this group sizes the dynamic shared-memory partition. This
        // MUST match the walker's layout at the group's launch width (`bytes_per_cell`), not `sz_dp_memory`'s
        // own minimal width - the walker stores `bytes_per_cell`-wide diagonals regardless of what width
        // `sz_dp_memory` would have picked for the cell magnitude.
        sz_size_t per_warp_bytes = 0;
        for (sz_size_t k = begin; k < end; ++k) {
            sz_size_t this_warp_bytes = szs_pair_warp_slice_bytes_(tasks_sorted[k].a_len, tasks_sorted[k].b_len,
                                                                  bytes_per_cell, count_diagonals);
            if (this_warp_bytes > per_warp_bytes) per_warp_bytes = this_warp_bytes;
        }
        if (per_warp_bytes == 0) per_warp_bytes = 16;                 // empty pairs still need a (tiny) valid slice
        per_warp_bytes = sz_round_up_to_multiple(per_warp_bytes, 16); // keep per-warp slices 16B-aligned

        unsigned const warps_per_block = szs_warps_per_block_(per_warp_bytes, costs_prefix, specs);
        unsigned const threads_per_block = (unsigned)specs.warp_size * warps_per_block;
        unsigned const shared_per_block = (unsigned)(costs_prefix + per_warp_bytes * warps_per_block);
        unsigned const grid_blocks = (unsigned)specs.streaming_multiprocessors;

        CUfunction fn = pick_warp_kernel(bytes_per_cell);
        // Opt into large dynamic shared memory (H100: up to 227KB/block) for this kernel.
        CUresult ar = cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)shared_per_block);

        szs_pair_task_t *group_tasks = tasks_sorted + begin;
        unsigned long long group_count = (unsigned long long)(end - begin);
        unsigned shared_memory_per_warp = (unsigned)per_warp_bytes;
        void *args[4] = {(void *)&group_tasks, (void *)&group_count, (void *)&costs, (void *)&shared_memory_per_warp};
        (void)ar;
        CUresult lr = cuLaunchKernel(fn, grid_blocks, 1, 1, threads_per_block, 1, 1, shared_per_block, nullptr, args,
                                     nullptr);
        if (lr != CUDA_SUCCESS) {
            if (error) *error = "cuLaunchKernel(similarity warp group) failed";
            status = sz_status_unknown_k;
        }
    }

    // Device-spanning tasks: one cooperative launch each (these are huge single pairs).
    for (sz_size_t k = device_begin; k < device_end && status == sz_success_k; ++k) {
        CUfunction fn = nullptr;
        if (!is_score) fn = !is_affine ? mod.szs_levenshtein_device_linear : mod.szs_levenshtein_device_affine;
        else if (costs.is_local)
            fn = !is_affine ? mod.szs_smith_waterman_device_linear : mod.szs_smith_waterman_device_affine;
        else fn = !is_affine ? mod.szs_needleman_wunsch_device_linear : mod.szs_needleman_wunsch_device_affine;
        if (!fn) {
            if (error) *error = "device-spanning kernel unavailable in this tier";
            status = sz_status_unknown_k;
            break;
        }

        szs_pair_task_t *task_ptr = tasks_sorted + k;
        void *args[3] = {(void *)&task_ptr, (void *)&costs, (void *)&device_diagonals};
        unsigned const block_size = 128;
        // A cooperative launch needs every block co-resident, so the grid may not exceed the device's max
        // active blocks for this kernel. Query the occupancy and clamp the grid to `SMs * maxBlocksPerSM`.
        int max_blocks_per_sm = 0;
        cuOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fn, (int)block_size, 0);
        if (max_blocks_per_sm < 1) max_blocks_per_sm = 1;
        unsigned grid_x = (unsigned)specs.streaming_multiprocessors * (unsigned)max_blocks_per_sm;
        CUresult lr = cuLaunchCooperativeKernel(fn, grid_x, 1, 1, block_size, 1, 1, 0, nullptr, args);
        if (lr != CUDA_SUCCESS) {
            if (error) *error = "cuLaunchCooperativeKernel(device-spanning) failed";
            status = sz_status_unknown_k;
        }
        // Device-spanning launches share `device_diagonals`, so serialize them.
        cudaDeviceSynchronize();
    }

    if (status == sz_success_k && cudaDeviceSynchronize() != cudaSuccess) {
        if (error) *error = "cudaDeviceSynchronize(similarity) failed";
        status = sz_status_unknown_k;
    }

    if (status == sz_success_k)
        for (sz_size_t k = 0; k < count; ++k) write_back(order[k], tasks_sorted[k].result);

    free(order);
    cudaFree(tasks_sorted);
    if (device_diagonals) cudaFree(device_diagonals);
    if (inputs_arena) cudaFree(inputs_arena);
    cudaFree(tasks);
    return status;
}

#pragma endregion

#pragma region - Public C Entries (one per op x tier)

inline sz_status_t szs_levenshtein_entry_(unsigned char const *ptx, szs_levenshtein_engine_t const *engine,
                                         szs_device_scope_impl_t *device, szs_sequence_view_t inputs,
                                         sz_size_t *results, sz_size_t results_stride, char const **error) {
    (void)device;
    szs_similarity_module_t mod = szs_load_similarity_module_(ptx);
    bool const is_affine = engine->gap_mode == sz_gaps_affine_k && engine->gaps.open != engine->gaps.extend;

    szs_device_costs_t costs;
    costs.is_uniform = 1;
    costs.match = engine->substitution.match;
    costs.mismatch = engine->substitution.mismatch;
    costs.gap_open = engine->gaps.open;
    costs.gap_extend = (engine->gap_mode == sz_gaps_affine_k) ? engine->gaps.extend : engine->gaps.open;
    if (!is_affine) costs.gap_extend = costs.gap_open;
    costs.objective = sz_minimize_distance_k;
    costs.is_local = 0;

    sz_size_t const sub_magnitude = sz_max_of_two(szs_cost_magnitude_(costs.match), szs_cost_magnitude_(costs.mismatch));
    sz_size_t const gap_magnitude = sz_max_of_two(szs_cost_magnitude_(costs.gap_open),
                                                  szs_cost_magnitude_(costs.gap_extend));
    sz_size_t const count = szs_inputs_count_(&inputs);

    return szs_run_similarity_batch_(
        mod, costs, is_affine, sub_magnitude, gap_magnitude, /*is_score*/ false, inputs, count,
        [&](sz_size_t i, sz_i64_t v) { *(sz_size_t *)((sz_u8_t *)results + i * results_stride) = (sz_size_t)v; },
        error);
}

// NW & SW carry distinct engine types with an identical field layout; templating on the engine type keeps the
// shared alignment batch path type-clean (no merged `alignment` engine, no type-pun). `is_local` selects the
// global/local recurrence the host need not know beyond seeding the cost POD.
template <typename engine_t_>
inline sz_status_t szs_alignment_entry_(unsigned char const *ptx, engine_t_ const *engine,
                                       szs_device_scope_impl_t *device, szs_sequence_view_t inputs,
                                       sz_ssize_t *results, sz_size_t results_stride, int is_local,
                                       char const **error) {
    (void)device;
    szs_similarity_module_t mod = szs_load_similarity_module_(ptx);
    bool const is_affine = engine->gap_mode == sz_gaps_affine_k;

    szs_device_costs_t costs;
    costs.is_uniform = 0;
    costs.match = 0;
    costs.mismatch = 0;
    costs.gap_open = engine->gaps.open;
    costs.gap_extend = engine->gaps.extend;
    costs.objective = sz_maximize_score_k;
    costs.is_local = is_local;
    costs.lut = engine->substitution;

    sz_size_t const sub_magnitude = szs_subs_magnitude_(&engine->substitution);
    sz_size_t const gap_magnitude = sz_max_of_two(szs_cost_magnitude_(costs.gap_open),
                                                  szs_cost_magnitude_(costs.gap_extend));
    sz_size_t const count = szs_inputs_count_(&inputs);

    return szs_run_similarity_batch_(
        mod, costs, is_affine, sub_magnitude, gap_magnitude, /*is_score*/ true, inputs, count,
        [&](sz_size_t i, sz_i64_t v) { *(sz_ssize_t *)((sz_u8_t *)results + i * results_stride) = (sz_ssize_t)v; },
        error);
}

#pragma endregion

} // namespace stringzillas
} // namespace ashvardanian

#pragma region - Plain-C dispatch.h-matching entry points

extern "C" {

/* Levenshtein: cuda | kepler | hopper tiers (kepler exists only for Levenshtein). */
inline sz_status_t szs_levenshtein_cuda(szs_levenshtein_engine_t const *engine, szs_device_scope_impl_t *device,
                                        szs_sequence_view_t inputs, sz_size_t *r, sz_size_t s, char const **e) {
    return ashvardanian::stringzillas::szs_levenshtein_entry_(sim_base_ptx, engine, device, inputs, r, s, e);
}
inline sz_status_t szs_levenshtein_kepler(szs_levenshtein_engine_t const *engine, szs_device_scope_impl_t *device,
                                          szs_sequence_view_t inputs, sz_size_t *r, sz_size_t s, char const **e) {
    return ashvardanian::stringzillas::szs_levenshtein_entry_(sim_kepler_ptx, engine, device, inputs, r, s, e);
}
inline sz_status_t szs_levenshtein_hopper(szs_levenshtein_engine_t const *engine, szs_device_scope_impl_t *device,
                                          szs_sequence_view_t inputs, sz_size_t *r, sz_size_t s, char const **e) {
    return ashvardanian::stringzillas::szs_levenshtein_entry_(sim_hopper_ptx, engine, device, inputs, r, s, e);
}

/* Needleman-Wunsch: cuda | hopper (no kepler). */
inline sz_status_t szs_needleman_wunsch_cuda(szs_needleman_wunsch_engine_t const *engine,
                                             szs_device_scope_impl_t *device, szs_sequence_view_t inputs,
                                             sz_ssize_t *r, sz_size_t s, char const **e) {
    return ashvardanian::stringzillas::szs_alignment_entry_(sim_base_ptx, engine, device, inputs, r, s, 0, e);
}
inline sz_status_t szs_needleman_wunsch_hopper(szs_needleman_wunsch_engine_t const *engine,
                                               szs_device_scope_impl_t *device, szs_sequence_view_t inputs,
                                               sz_ssize_t *r, sz_size_t s, char const **e) {
    return ashvardanian::stringzillas::szs_alignment_entry_(sim_hopper_ptx, engine, device, inputs, r, s, 0, e);
}

/* Smith-Waterman: cuda | hopper. */
inline sz_status_t szs_smith_waterman_cuda(szs_smith_waterman_engine_t const *engine, szs_device_scope_impl_t *device,
                                           szs_sequence_view_t inputs, sz_ssize_t *r, sz_size_t s, char const **e) {
    return ashvardanian::stringzillas::szs_alignment_entry_(sim_base_ptx, engine, device, inputs, r, s, 1, e);
}
inline sz_status_t szs_smith_waterman_hopper(szs_smith_waterman_engine_t const *engine, szs_device_scope_impl_t *device,
                                             szs_sequence_view_t inputs, sz_ssize_t *r, sz_size_t s, char const **e) {
    return ashvardanian::stringzillas::szs_alignment_entry_(sim_hopper_ptx, engine, device, inputs, r, s, 1, e);
}

} // extern "C"

#endif // STRINGZILLAS_SIMILARITIES_CUDA_HOST_CUH_
