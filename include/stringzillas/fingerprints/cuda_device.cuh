/**
 *  @file include/stringzillas/fingerprints/cuda_device.cuh
 *  @brief Shared CUDA device helpers + the explicit fingerprint entry kernel (Min-Hash / Count-Min).
 *  @author Ash Vardanian
 *
 *  Fingerprints keep the single `cuda` tier (no base/kepler/hopper split is meaningful for the rolling
 *  hash). The device math reproduces the serial sweep (`szs_fingerprints_sweep_serial_`) and its `sz_fh_*`
 *  Rabin-Karp / floating-Barrett helpers bit-for-bit (same f64 op order: discard-then-mod, add-then-mod),
 *  so the u32 Min-Hash / Count-Min outputs match the serial reference exactly.
 *
 *  The work is mapped warp-per-(document, window-width group): one warp owns one (text, group) item and
 *  grid-strides over the flattened item space. The warp stages the text into per-warp shared memory ONCE
 *  (lanes strided-load `warp_size` bytes at a time) and the 32 lanes cooperatively roll all of the group's
 *  dimensions across that staged window, so the text is read once per warp per group rather than once per
 *  dimension. Every group carries a single window width (the serial engine slices the dimensions into such
 *  groups), so the discard/incoming staging is uniform across the warp, mirroring the original
 *  `floating_rolling_hashers_on_each_cuda_warp_` kernel. The lanes split the group's dimensions in
 *  contiguous blocks (`dim = lane * dims_per_lane + dim_within_lane`).
 *
 *  A device-spanning entry (`szs_fingerprints_device`, cooperative-groups grid reduction for one very long
 *  document) is DEFERRED: the original `floating_rolling_hashers_across_cuda_device_` was an empty stub with
 *  no reduction math to port, so there is nothing to reproduce bit-for-bit. The warp-per-document path above
 *  already handles arbitrarily long single documents (the staging loop strides the whole text per warp).
 */
#ifndef STRINGZILLAS_FINGERPRINTS_CUDA_DEVICE_CUH_
#define STRINGZILLAS_FINGERPRINTS_CUDA_DEVICE_CUH_

#include "stringzilla/types.h" // `sz_f64_t`, `sz_u32_t`, `sz_u64_t`, `sz_size_t`, `sz_byte_t`

// Stub the CUDA execution-space qualifiers when included by a plain host compiler (for the PODs). nvcc
// defines `__CUDACC__` and provides the real qualifiers.
#if !defined(__CUDACC__)
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#endif

#include <math.h> // `fmod`, `floor`, `fma` for the host-side parse of the device helpers

namespace ashvardanian {
namespace stringzillas {

#pragma region - Rolling-Hash Constants (value-only, no function-like macros)

// The largest integer exactly representable as an IEEE-754 f64 (2^52 - 1).
static const sz_f64_t szs_fh_limit_k = 4503599627370495.0;
// Default modulo base: largest prime under `limit/1000 - 257`.
static const sz_u64_t szs_fh_default_modulo_base_k = 4503599626977ull;
// Sentinel for an unset rolling minimum: the largest representable f64 (matches C++/serial).
static const sz_f64_t szs_fh_skipped_rolling_hash_k = 1.7976931348623157e308;
// Maximum 32-bit Min-Hash value, used both as a mask and the skipped-output value.
static const sz_u32_t szs_fh_max_hash_k = 0xFFFFFFFFu;

#pragma endregion

#pragma region - Device Rolling-Hash Helpers (verbatim from fingerprint_kernels.h)

// Per-dimension rolling-hash coefficients (mirrors `sz_fh_dim_t`).
struct szs_fh_dim_t {
    sz_f64_t multiplier;
    sz_f64_t modulo;
    sz_f64_t inverse_modulo;
    sz_f64_t negative_discarding_multiplier;
};

// Seeds one dimension's coefficients exactly like `sz_fh_seed_dim_`.
__device__ inline szs_fh_dim_t szs_fh_seed_dim(sz_size_t window_width, sz_u64_t multiplier_int, sz_u64_t modulo_int) {
    szs_fh_dim_t dim;
    sz_f64_t const multiplier = (sz_f64_t)multiplier_int;
    sz_f64_t const modulo = (sz_f64_t)modulo_int;
    sz_f64_t discarding = 1.0;
    dim.multiplier = multiplier;
    dim.modulo = modulo;
    dim.inverse_modulo = 1.0 / modulo;
    for (sz_size_t step_index = 0; step_index + 1 < window_width; ++step_index)
        discarding = fmod(discarding * multiplier, modulo);
    dim.negative_discarding_multiplier = -discarding;
    return dim;
}

// Scalar Barrett reduction matching `sz_fh_barrett_mod` (plain `x - q*modulo`, predicate clamps).
__device__ __forceinline__ sz_f64_t szs_fh_barrett_mod(sz_f64_t x, sz_f64_t modulo, sz_f64_t inverse_modulo) {
    sz_f64_t const q = floor(x * inverse_modulo);
    sz_f64_t result = x - q * modulo;
    result += modulo * (sz_f64_t)(result < 0.0);
    result -= modulo * (sz_f64_t)(result >= modulo);
    return result;
}

// Add-head term step: `barrett_mod(fma(state, multiplier, new_term))` (matches `sz_fh_push`).
__device__ __forceinline__ sz_f64_t szs_fh_push(sz_f64_t state, szs_fh_dim_t const *dim, sz_f64_t new_term) {
    sz_f64_t pushed = fma(state, dim->multiplier, new_term);
    return szs_fh_barrett_mod(pushed, dim->modulo, dim->inverse_modulo);
}

// Two-step roll: discard-then-mod, add-then-mod (matches `sz_fh_roll`).
__device__ __forceinline__ sz_f64_t szs_fh_roll(sz_f64_t state, szs_fh_dim_t const *dim, sz_f64_t old_term,
                                                sz_f64_t new_term) {
    sz_f64_t without_old = fma(dim->negative_discarding_multiplier, old_term, state);
    without_old = szs_fh_barrett_mod(without_old, dim->modulo, dim->inverse_modulo);
    sz_f64_t with_new = fma(without_old, dim->multiplier, new_term);
    return szs_fh_barrett_mod(with_new, dim->modulo, dim->inverse_modulo);
}

#pragma endregion

#pragma region - One Task per Text

// A single fingerprint task in device-accessible memory: one text + its two output planes.
struct szs_fingerprint_task_t {
    sz_byte_t const *text;
    sz_size_t text_length;
    sz_u32_t *min_hashes; // Output plane for this text (one entry per dimension).
    sz_u32_t *min_counts; // Output plane for this text (one entry per dimension).
};

// Warp size assumed by the cooperative kernel (NVIDIA). Used to size the per-warp staging buffers and the
// dimension-per-lane mapping.
static const unsigned szs_fingerprints_warp_size_k = 32u;

// Slices a window-width group's dimension range, mirroring `szs_fingerprints_group_geometry_`: the total
// dimensions split into `window_widths_count` groups, the first `remainder` groups get one extra dimension.
__device__ inline void szs_fingerprint_group_geometry( //
    sz_size_t group_index, sz_size_t dimensions, sz_size_t window_widths_count, sz_size_t *out_first_dim,
    sz_size_t *out_group_dims) {
    sz_size_t const base = dimensions / window_widths_count;
    sz_size_t const remainder = dimensions % window_widths_count;
    *out_first_dim = group_index * base + (group_index < remainder ? group_index : remainder);
    *out_group_dims = base + (group_index < remainder ? 1 : 0);
}

#pragma endregion

} // namespace stringzillas
} // namespace ashvardanian

#pragma region - Entry-Kernel Definition

// Warp-per-(document, window-width group) Min-Hash / Count-Min kernel. Each warp grid-strides over the
// flattened (task, group) item space. For its item it resolves the group's dimension slice + single window
// width, then sweeps the text ONCE: the prefix fill primes each lane's states, then the central loop stages
// `warp_size` incoming + `warp_size` discarding bytes into per-warp shared memory (lanes coalesce-load one
// byte each) and the 32 lanes cooperatively roll all of the group's dimensions across the staged window, so
// the text is read once per warp per group rather than once per dimension. A short tail rolls the remaining
// bytes. Each lane owns `dims_per_lane` dimensions in a contiguous block. The f64 op order matches the
// serial `sz_fh_*` recurrence bit-for-bit. C-style `extern "C"` signature with no template parameters keeps
// the PTX symbol name (`szs_fingerprints_warp`) stable for the host resolver. Device code only, under nvcc.
#if defined(__CUDACC__)

extern "C" __global__ void szs_fingerprints_warp(::ashvardanian::stringzillas::szs_fingerprint_task_t *tasks,
                                                 unsigned long long tasks_count, unsigned long long dimensions,
                                                 unsigned long long window_widths_count,
                                                 unsigned long long const *window_widths,
                                                 unsigned long long alphabet_size) {
    using namespace ::ashvardanian::stringzillas;

    // Per-warp staging: two `warp_size`-byte chunks (incoming + discarding) per warp in this block.
    extern __shared__ sz_byte_t szs_fp_shared[];
    unsigned const warp_size = szs_fingerprints_warp_size_k;
    unsigned const lane = threadIdx.x % warp_size;
    unsigned const warp_in_block = threadIdx.x / warp_size;
    unsigned const warps_per_block = blockDim.x / warp_size;
    sz_byte_t *const warp_chunk = szs_fp_shared + (sz_size_t)warp_in_block * 2u * warp_size;
    sz_byte_t *const incoming_chunk = warp_chunk;               // [warp_size] incoming bytes
    sz_byte_t *const discarding_chunk = warp_chunk + warp_size; // [warp_size] discarding bytes

    unsigned long long const global_warp_index = ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) /
                                                 warp_size;
    unsigned long long const warps_per_device = (unsigned long long)gridDim.x * warps_per_block;
    unsigned long long const items_total = tasks_count * window_widths_count;

    // Per-lane register state is bounded by this many dimensions; `dims_per_lane` below never exceeds it for
    // the supported dimension counts (32 lanes * 64 == 2048 dimensions).
    sz_size_t const dims_per_lane_cap = 64;

    for (unsigned long long item = global_warp_index; item < items_total; item += warps_per_device) {
        unsigned long long const task_index = item / window_widths_count;
        unsigned long long const group_index = item % window_widths_count;
        szs_fingerprint_task_t &task = tasks[task_index];

        sz_size_t first_dim, group_dims;
        szs_fingerprint_group_geometry((sz_size_t)group_index, (sz_size_t)dimensions, (sz_size_t)window_widths_count,
                                       &first_dim, &group_dims);
        if (group_dims == 0) continue;
        sz_size_t const window_width = (sz_size_t)window_widths[group_index];

        // This lane's contiguous slice of the group's dimensions: [lane*dims_per_lane, +dims_per_lane).
        sz_size_t const dims_per_lane = (group_dims + warp_size - 1) / warp_size;
        sz_size_t const lane_first = (sz_size_t)lane * dims_per_lane;

        sz_byte_t const *text = task.text;
        sz_size_t const text_length = task.text_length;

        // Short-input shortcut: window never fills, all this lane's dims are skipped.
        if (text_length < window_width) {
            for (sz_size_t d = 0; d < dims_per_lane; ++d) {
                sz_size_t const dim_within_group = lane_first + d;
                if (dim_within_group >= group_dims) break;
                sz_size_t const dim_global = first_dim + dim_within_group;
                task.min_hashes[dim_global] = szs_fh_max_hash_k;
                task.min_counts[dim_global] = 0;
            }
            continue;
        }

        // Seed this lane's dimensions + reset their rolling state (registers, bounded by `dims_per_lane_cap`).
        szs_fh_dim_t dims[64];
        sz_f64_t rolling_states[64];
        sz_f64_t rolling_minimums[64];
        sz_u32_t rolling_counts[64];
        sz_size_t const lane_dims = dims_per_lane < dims_per_lane_cap ? dims_per_lane : dims_per_lane_cap;
        for (sz_size_t d = 0; d < lane_dims; ++d) {
            sz_size_t const dim_within_group = lane_first + d;
            sz_u64_t const multiplier_int = (sz_u64_t)(alphabet_size + first_dim + dim_within_group);
            dims[d] = szs_fh_seed_dim(window_width, multiplier_int, szs_fh_default_modulo_base_k);
            rolling_states[d] = 0.0;
            rolling_minimums[d] = szs_fh_skipped_rolling_hash_k;
            rolling_counts[d] = 0;
        }

        // Prefix fill: push the first `window_width` head terms (no discards yet).
        sz_size_t new_char_offset = 0;
        for (; new_char_offset < window_width; ++new_char_offset) {
            sz_f64_t const new_term = (sz_f64_t)text[new_char_offset] + 1.0;
            for (sz_size_t d = 0; d < lane_dims; ++d)
                rolling_states[d] = szs_fh_push(rolling_states[d], &dims[d], new_term);
        }

        // First minimum is the state once the window is exactly full.
        for (sz_size_t d = 0; d < lane_dims; ++d) {
            rolling_minimums[d] = rolling_states[d];
            rolling_counts[d] = 1;
        }

        // Central loop: stage `warp_size` bytes into shared memory once per warp, then all lanes roll over
        // the staged chunk. The text is read once per warp here, not once per dimension.
        for (; new_char_offset + warp_size <= text_length; new_char_offset += warp_size) {
            incoming_chunk[lane] = text[new_char_offset + lane];
            discarding_chunk[lane] = text[new_char_offset - window_width + lane];
            __syncwarp();
            for (unsigned char_in_step = 0; char_in_step < warp_size; ++char_in_step) {
                sz_f64_t const new_term = (sz_f64_t)incoming_chunk[char_in_step] + 1.0;
                sz_f64_t const old_term = (sz_f64_t)discarding_chunk[char_in_step] + 1.0;
                for (sz_size_t d = 0; d < lane_dims; ++d) {
                    sz_f64_t const state = szs_fh_roll(rolling_states[d], &dims[d], old_term, new_term);
                    sz_f64_t const rolling_minimum = rolling_minimums[d];
                    rolling_states[d] = state;
                    rolling_counts[d] *= (sz_u32_t)(state >= rolling_minimum);
                    rolling_counts[d] += (sz_u32_t)(state <= rolling_minimum);
                    rolling_minimums[d] = state < rolling_minimum ? state : rolling_minimum;
                }
            }
            __syncwarp();
        }

        // Tail: roll the remaining (< warp_size) bytes directly from global memory.
        for (; new_char_offset < text_length; ++new_char_offset) {
            sz_f64_t const new_term = (sz_f64_t)text[new_char_offset] + 1.0;
            sz_f64_t const old_term = (sz_f64_t)text[new_char_offset - window_width] + 1.0;
            for (sz_size_t d = 0; d < lane_dims; ++d) {
                sz_f64_t const state = szs_fh_roll(rolling_states[d], &dims[d], old_term, new_term);
                sz_f64_t const rolling_minimum = rolling_minimums[d];
                rolling_states[d] = state;
                rolling_counts[d] *= (sz_u32_t)(state >= rolling_minimum);
                rolling_counts[d] += (sz_u32_t)(state <= rolling_minimum);
                rolling_minimums[d] = state < rolling_minimum ? state : rolling_minimum;
            }
        }

        // Digest this lane's minimums + counts into the output planes.
        for (sz_size_t d = 0; d < lane_dims; ++d) {
            sz_size_t const dim_within_group = lane_first + d;
            if (dim_within_group >= group_dims) break;
            sz_size_t const dim_global = first_dim + dim_within_group;
            sz_f64_t const rolling_minimum = rolling_minimums[d];
            sz_u64_t const rolling_minimum_as_uint = (sz_u64_t)rolling_minimum;
            task.min_hashes[dim_global] = rolling_minimum == szs_fh_skipped_rolling_hash_k
                                              ? szs_fh_max_hash_k
                                              : (sz_u32_t)(rolling_minimum_as_uint & szs_fh_max_hash_k);
            task.min_counts[dim_global] = rolling_counts[d];
        }
    }
}

#endif // __CUDACC__

#pragma endregion

#endif // STRINGZILLAS_FINGERPRINTS_CUDA_DEVICE_CUH_
