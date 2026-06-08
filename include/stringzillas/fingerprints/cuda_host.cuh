/**
 *  @file include/stringzillas/fingerprints/cuda_host.cuh
 *  @brief Plain-C++ CUDA Driver-API host launcher for the fingerprint backend.
 *  @author Ash Vardanian
 *
 *  Loads the single `cuda`-tier fingerprint PTX blob via `cuModuleLoadData`, resolves the explicit
 *  `szs_fingerprints_warp` entry, stages texts + window widths + output planes into unified memory, and
 *  launches with `cuLaunchKernel` ONCE over the batch. The grid is `warps_per_block` warps per block over
 *  `SM-count` blocks; each warp owns one (text, window-width group) item and grid-strides over the
 *  flattened item space, staging the text into per-warp shared memory (2 * warp_size bytes per warp).
 *  Exposes a plain-C entry `szs_fingerprints_cuda` matching the `dispatch.h` typedef.
 */
#ifndef STRINGZILLAS_FINGERPRINTS_CUDA_HOST_CUH_
#define STRINGZILLAS_FINGERPRINTS_CUDA_HOST_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include "stringzilla/types.h"
#include "stringzillas/types.h"
#include "stringzillas/fingerprints/serial.h" // input decoding helpers + constants
#include "stringzillas/fingerprints/cuda_device.cuh"

extern "C" {
extern unsigned char fp_cuda_ptx[];
}

namespace ashvardanian {
namespace stringzillas {

inline bool szs_fp_ensure_primary_context_() {
    static bool initialized = false;
    if (initialized) return true;
    if (cuInit(0) != CUDA_SUCCESS) return false;
    if (cudaFree(0) != cudaSuccess) return false;
    initialized = true;
    return true;
}

inline sz_status_t szs_fingerprints_run_(unsigned char const *ptx, szs_fingerprints_engine_t const *engine,
                                        szs_device_scope_impl_t *device, szs_sequence_view_t inputs,
                                        sz_u32_t *min_hashes, sz_size_t min_hashes_stride, sz_u32_t *min_counts,
                                        sz_size_t min_counts_stride, char const **error) {
    (void)device;
    szs_fingerprints_engine_t const *eng = engine;
    if (!szs_fp_ensure_primary_context_()) {
        if (error) *error = "cuInit / primary context failed";
        return sz_status_unknown_k;
    }
    CUmodule module = nullptr;
    if (cuModuleLoadData(&module, (void const *)ptx) != CUDA_SUCCESS) {
        if (error) *error = "cuModuleLoadData(fingerprints) failed";
        return sz_status_unknown_k;
    }
    CUfunction fn = nullptr;
    if (cuModuleGetFunction(&fn, module, "szs_fingerprints_warp") != CUDA_SUCCESS) {
        if (error) *error = "szs_fingerprints_warp not found in PTX";
        cuModuleUnload(module);
        return sz_status_unknown_k;
    }

    sz_size_t const count = szs_fingerprints_count_(&inputs);
    sz_size_t const dimensions = eng->dimensions;
    sz_size_t const window_widths_count = eng->window_widths_count;
    sz_size_t const alphabet_size = eng->alphabet_size ? eng->alphabet_size : SZ_FH_DEFAULT_ALPHABET_SIZE_K;

    // Stage window widths into unified memory as u64.
    unsigned long long *widths_dev = nullptr;
    cudaMallocManaged(&widths_dev, sizeof(unsigned long long) * (window_widths_count ? window_widths_count : 1));
    for (sz_size_t g = 0; g < window_widths_count; ++g) widths_dev[g] = (unsigned long long)eng->window_widths[g];

    // Stage tasks + per-text text bytes + per-text output planes into unified memory.
    szs_fingerprint_task_t *tasks = nullptr;
    cudaMallocManaged(&tasks, sizeof(szs_fingerprint_task_t) * (count ? count : 1));
    for (sz_size_t i = 0; i < count; ++i) {
        szs_text_span_t span = szs_fingerprints_text_(&inputs, i);
        sz_byte_t *text_dev = nullptr;
        cudaMallocManaged(&text_dev, span.len ? span.len : 1);
        for (sz_size_t k = 0; k < span.len; ++k) text_dev[k] = ((sz_byte_t const *)span.ptr)[k];
        sz_u32_t *hashes_dev = nullptr;
        sz_u32_t *counts_dev = nullptr;
        cudaMallocManaged(&hashes_dev, sizeof(sz_u32_t) * dimensions);
        cudaMallocManaged(&counts_dev, sizeof(sz_u32_t) * dimensions);
        tasks[i].text = text_dev;
        tasks[i].text_length = span.len;
        tasks[i].min_hashes = hashes_dev;
        tasks[i].min_counts = counts_dev;
    }

    unsigned long long tasks_count_ull = (unsigned long long)count;
    unsigned long long dimensions_ull = (unsigned long long)dimensions;
    unsigned long long widths_count_ull = (unsigned long long)window_widths_count;
    unsigned long long alphabet_ull = (unsigned long long)alphabet_size;
    void *args[6] = {(void *)&tasks,      (void *)&tasks_count_ull, (void *)&dimensions_ull, (void *)&widths_count_ull,
                     (void *)&widths_dev, (void *)&alphabet_ull};

    // Warp-per-(text, group): grid = SM-count blocks of `warps_per_block` warps, each warp owns one item and
    // grid-strides over the `count * window_widths_count` flattened items. Per-warp SMEM stages the text
    // (incoming + discarding `warp_size`-byte chunks). ONE launch covers the whole batch.
    unsigned const warp_size = 32;
    unsigned const warps_per_block = 4;
    unsigned const block = warp_size * warps_per_block;
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    if (sm_count < 1) sm_count = 1;
    unsigned const grid = (unsigned)sm_count;
    unsigned const shared_per_warp = 2u * warp_size; // incoming + discarding chunk
    unsigned const shared_per_block = shared_per_warp * warps_per_block;

    unsigned long long const items_total = (unsigned long long)count * (window_widths_count ? window_widths_count : 1);
    sz_status_t status = sz_success_k;
    if (items_total) {
        CUresult lr = cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, shared_per_block, nullptr, args, nullptr);
        if (lr != CUDA_SUCCESS) {
            if (error) *error = "cuLaunchKernel(fingerprints) failed";
            status = sz_status_unknown_k;
        }
    }
    if (status == sz_success_k && cudaDeviceSynchronize() != cudaSuccess) {
        if (error) *error = "cudaDeviceSynchronize(fingerprints) failed";
        status = sz_status_unknown_k;
    }

    if (status == sz_success_k)
        for (sz_size_t i = 0; i < count; ++i) {
            sz_u32_t *out_hashes = (sz_u32_t *)((sz_byte_t *)min_hashes + i * min_hashes_stride);
            sz_u32_t *out_counts = (sz_u32_t *)((sz_byte_t *)min_counts + i * min_counts_stride);
            for (sz_size_t d = 0; d < dimensions; ++d) {
                out_hashes[d] = tasks[i].min_hashes[d];
                out_counts[d] = tasks[i].min_counts[d];
            }
        }

    for (sz_size_t i = 0; i < count; ++i) {
        if (tasks[i].text) cudaFree((void *)tasks[i].text);
        if (tasks[i].min_hashes) cudaFree(tasks[i].min_hashes);
        if (tasks[i].min_counts) cudaFree(tasks[i].min_counts);
    }
    cudaFree(tasks);
    cudaFree(widths_dev);
    cuModuleUnload(module);
    return status;
}

} // namespace stringzillas
} // namespace ashvardanian

extern "C" {
inline sz_status_t szs_fingerprints_cuda(szs_fingerprints_engine_t const *engine, szs_device_scope_impl_t *device,
                                         szs_sequence_view_t inputs, sz_u32_t *min_hashes, sz_size_t min_hashes_stride,
                                         sz_u32_t *min_counts, sz_size_t min_counts_stride, char const **error) {
    return ashvardanian::stringzillas::szs_fingerprints_run_(fp_cuda_ptx, engine, device, inputs, min_hashes,
                                                            min_hashes_stride, min_counts, min_counts_stride, error);
}
}

#endif // STRINGZILLAS_FINGERPRINTS_CUDA_HOST_CUH_
