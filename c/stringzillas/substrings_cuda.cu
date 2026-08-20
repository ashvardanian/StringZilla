/**
 *  @file c/stringzillas/substrings_cuda.cu
 *  @brief Base-CUDA-tier instantiations for multi-pattern Aho-Corasick search.
 *  @author Ash Vardanian
 */
#include "stringzillas/substrings.cuh"

namespace ashvardanian {
namespace stringzillas {

/*  The engine's `kernels()` table takes these kernels' addresses from host code, which implicitly instantiates
 *  NVCC's device-stub wrappers mid-file; instantiating the kernels first makes the stubs precede that use,
 *  keeping the generated host code well-formed for host compilers that enforce [temp.expl.spec] ordering
 *  (Clang) rather than tolerating the inversion (GCC). */
template __global__ void substrings_walk_per_cuda_chunk_<u16_t, substrings_pass_t::sizing_k>(
    aho_corasick_view<u16_t>, u16_t, span<u32_t const>, u32_t, span<span<byte_t const> const>, span<size_t const>,
    size_t, size_t, span<size_t>, span<substrings_match_t>);
template __global__ void substrings_walk_per_cuda_chunk_<u32_t, substrings_pass_t::sizing_k>(
    aho_corasick_view<u32_t>, u32_t, span<u32_t const>, u32_t, span<span<byte_t const> const>, span<size_t const>,
    size_t, size_t, span<size_t>, span<substrings_match_t>);
template __global__ void substrings_walk_per_cuda_chunk_<u16_t, substrings_pass_t::writing_k>(
    aho_corasick_view<u16_t>, u16_t, span<u32_t const>, u32_t, span<span<byte_t const> const>, span<size_t const>,
    size_t, size_t, span<size_t>, span<substrings_match_t>);
template __global__ void substrings_walk_per_cuda_chunk_<u32_t, substrings_pass_t::writing_k>(
    aho_corasick_view<u32_t>, u32_t, span<u32_t const>, u32_t, span<span<byte_t const> const>, span<size_t const>,
    size_t, size_t, span<size_t>, span<substrings_match_t>);

template __global__ void exclusive_sum_across_cuda_device_<size_t>( //
    size_t const *, size_t, size_t *);
template __global__ void exclusive_sum_reduce_tiles_across_cuda_device_<size_t>( //
    size_t const *, size_t, size_t, size_t *);
template __global__ void exclusive_sum_apply_tiles_across_cuda_device_<size_t>( //
    size_t const *, size_t, size_t, size_t const *, size_t *);

template __global__ void substrings_score_bm25_per_haystack_<u16_t>( //
    aho_corasick_view<u16_t>, u16_t, span<u32_t const>, u32_t, span<span<byte_t const> const>, span<f32_t const>,
    substrings_bm25_t, span<f32_t const>, span<u32_t>, span<f32_t>);
template __global__ void substrings_score_bm25_per_haystack_<u32_t>( //
    aho_corasick_view<u32_t>, u32_t, span<u32_t const>, u32_t, span<span<byte_t const> const>, span<f32_t const>,
    substrings_bm25_t, span<f32_t const>, span<u32_t>, span<f32_t>);

template struct substrings_cuda<unified_alloc_t, sz_cap_cuda_k>;

} // namespace stringzillas
} // namespace ashvardanian
