/**
 *  @file c/stringzillas/levenshtein_cuda.cu
 *  @brief Base-CUDA-tier instantiations for parallel Levenshtein distances.
 *  @author Ash Vardanian
 *  @date March 23, 2025
 */
#include "stringzillas/similarities.cuh"

namespace ashvardanian {
namespace stringzillas {

/*  The engines' `kernels()` tables take these kernels' addresses from host code, which implicitly
 *  instantiates NVCC's device-stub wrappers mid-file; instantiating the kernels first makes the stubs
 *  precede that use, keeping the generated host code well-formed for host compilers that enforce
 *  [temp.expl.spec] ordering (Clang) rather than tolerating the inversion (GCC). */
template __global__ void score_per_cuda_warp_<                                                    //
    cuda_similarity_task<char>, char, u8_t, u8_t,                                                 //
    uniform_substitution_costs_t, sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k>( //
    cuda_similarity_task<char> *, size_t, uniform_substitution_costs_t const, linear_gap_costs_t const, unsigned const);
template __global__ void score_per_cuda_warp_<                                                    //
    cuda_similarity_task<char>, char, u16_t, u16_t,                                               //
    uniform_substitution_costs_t, sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k>( //
    cuda_similarity_task<char> *, size_t, uniform_substitution_costs_t const, linear_gap_costs_t const, unsigned const);
template __global__ void score_per_cuda_warp_<                                                    //
    cuda_similarity_task<char>, char, u32_t, u32_t,                                               //
    uniform_substitution_costs_t, sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k>( //
    cuda_similarity_task<char> *, size_t, uniform_substitution_costs_t const, linear_gap_costs_t const, unsigned const);
template __global__ void affine_score_per_cuda_warp_<                                             //
    cuda_similarity_task<char>, char, u8_t, u8_t,                                                 //
    uniform_substitution_costs_t, sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k>( //
    cuda_similarity_task<char> *, size_t, uniform_substitution_costs_t const, affine_gap_costs_t const, unsigned const);
template __global__ void affine_score_per_cuda_warp_<                                             //
    cuda_similarity_task<char>, char, u16_t, u16_t,                                               //
    uniform_substitution_costs_t, sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k>( //
    cuda_similarity_task<char> *, size_t, uniform_substitution_costs_t const, affine_gap_costs_t const, unsigned const);
template __global__ void affine_score_per_cuda_warp_<                                             //
    cuda_similarity_task<char>, char, u32_t, u32_t,                                               //
    uniform_substitution_costs_t, sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k>( //
    cuda_similarity_task<char> *, size_t, uniform_substitution_costs_t const, affine_gap_costs_t const, unsigned const);
template __global__ void score_across_cuda_device_<                                             //
    8U, char, u16_t, u64_t, uniform_substitution_costs_t,                                       //
    sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k, cuda_similarity_task<char>>( //
    cuda_similarity_task<char> *, u16_t *, u16_t *, u32_t *, u32_t, u32_t,                      //
    uniform_substitution_costs_t const, linear_gap_costs_t const);
template __global__ void score_across_cuda_device_<                                             //
    8U, char, u32_t, u64_t, uniform_substitution_costs_t,                                       //
    sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k, cuda_similarity_task<char>>( //
    cuda_similarity_task<char> *, u32_t *, u32_t *, u32_t *, u32_t, u32_t,                      //
    uniform_substitution_costs_t const, linear_gap_costs_t const);
template __global__ void score_across_cuda_device_<                                             //
    8U, char, u64_t, u64_t, uniform_substitution_costs_t,                                       //
    sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k, cuda_similarity_task<char>>( //
    cuda_similarity_task<char> *, u64_t *, u64_t *, u32_t *, u32_t, u32_t,                      //
    uniform_substitution_costs_t const, linear_gap_costs_t const);
template __global__ void affine_score_across_cuda_device_<                                      //
    8U, char, u16_t, u64_t, uniform_substitution_costs_t,                                       //
    sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k, cuda_similarity_task<char>>( //
    cuda_similarity_task<char> *, u16_t *, u16_t *, u16_t *, u32_t *, u32_t, u32_t,             //
    uniform_substitution_costs_t const, affine_gap_costs_t const);
template __global__ void affine_score_across_cuda_device_<                                      //
    8U, char, u32_t, u64_t, uniform_substitution_costs_t,                                       //
    sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k, cuda_similarity_task<char>>( //
    cuda_similarity_task<char> *, u32_t *, u32_t *, u32_t *, u32_t *, u32_t, u32_t,             //
    uniform_substitution_costs_t const, affine_gap_costs_t const);
template __global__ void affine_score_across_cuda_device_<                                      //
    8U, char, u64_t, u64_t, uniform_substitution_costs_t,                                       //
    sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k, cuda_similarity_task<char>>( //
    cuda_similarity_task<char> *, u64_t *, u64_t *, u64_t *, u32_t *, u32_t, u32_t,             //
    uniform_substitution_costs_t const, affine_gap_costs_t const);
template __global__ void frontier_init_across_cuda_device_<                                    //
    u16_t, u64_t, sz_minimize_distance_k, sz_similarity_global_k, cuda_similarity_task<char>>( //
    cuda_similarity_task<char> *, u16_t *, u16_t *, u32_t *, u32_t, u32_t, linear_gap_costs_t const);
template __global__ void frontier_init_across_cuda_device_<                                    //
    u32_t, u64_t, sz_minimize_distance_k, sz_similarity_global_k, cuda_similarity_task<char>>( //
    cuda_similarity_task<char> *, u32_t *, u32_t *, u32_t *, u32_t, u32_t, linear_gap_costs_t const);
template __global__ void frontier_init_across_cuda_device_<                                    //
    u64_t, u64_t, sz_minimize_distance_k, sz_similarity_global_k, cuda_similarity_task<char>>( //
    cuda_similarity_task<char> *, u64_t *, u64_t *, u32_t *, u32_t, u32_t, linear_gap_costs_t const);
template __global__ void affine_frontier_init_across_cuda_device_<                             //
    u16_t, u64_t, sz_minimize_distance_k, sz_similarity_global_k, cuda_similarity_task<char>>( //
    cuda_similarity_task<char> *, u16_t *, u16_t *, u16_t *, u32_t *, u32_t, u32_t, affine_gap_costs_t const);
template __global__ void affine_frontier_init_across_cuda_device_<                             //
    u32_t, u64_t, sz_minimize_distance_k, sz_similarity_global_k, cuda_similarity_task<char>>( //
    cuda_similarity_task<char> *, u32_t *, u32_t *, u32_t *, u32_t *, u32_t, u32_t, affine_gap_costs_t const);
template __global__ void affine_frontier_init_across_cuda_device_<                             //
    u64_t, u64_t, sz_minimize_distance_k, sz_similarity_global_k, cuda_similarity_task<char>>( //
    cuda_similarity_task<char> *, u64_t *, u64_t *, u64_t *, u32_t *, u32_t, u32_t, affine_gap_costs_t const);

template __global__ void unit_utf8_score_across_cuda_device_<              //
    8U, u16_t, u64_t, cuda_similarity_task<char>>(                         //
    cuda_similarity_task<char> *, u16_t *, u16_t *, u32_t *, u32_t, u32_t, //
    uniform_substitution_costs_t const, linear_gap_costs_t const);
template __global__ void unit_utf8_score_across_cuda_device_<              //
    8U, u32_t, u64_t, cuda_similarity_task<char>>(                         //
    cuda_similarity_task<char> *, u32_t *, u32_t *, u32_t *, u32_t, u32_t, //
    uniform_substitution_costs_t const, linear_gap_costs_t const);
template __global__ void unit_utf8_score_across_cuda_device_<              //
    8U, u64_t, u64_t, cuda_similarity_task<char>>(                         //
    cuda_similarity_task<char> *, u64_t *, u64_t *, u32_t *, u32_t, u32_t, //
    uniform_substitution_costs_t const, linear_gap_costs_t const);
template __global__ void unit_utf8_frontier_init_across_cuda_device_< //
    u16_t, u64_t, cuda_similarity_task<char>>(                        //
    cuda_similarity_task<char> *, u16_t *, u16_t *, u32_t *, u32_t, u32_t, linear_gap_costs_t const);
template __global__ void unit_utf8_frontier_init_across_cuda_device_< //
    u32_t, u64_t, cuda_similarity_task<char>>(                        //
    cuda_similarity_task<char> *, u32_t *, u32_t *, u32_t *, u32_t, u32_t, linear_gap_costs_t const);
template __global__ void unit_utf8_frontier_init_across_cuda_device_< //
    u64_t, u64_t, cuda_similarity_task<char>>(                        //
    cuda_similarity_task<char> *, u64_t *, u64_t *, u32_t *, u32_t, u32_t, linear_gap_costs_t const);

template struct levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
template struct levenshtein_distances<affine_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
template struct levenshtein_distances_utf8<linear_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
} // namespace stringzillas
} // namespace ashvardanian
