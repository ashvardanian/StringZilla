/**
 *  @file c/stringzillas/levenshtein_cuda.cu
 *  @brief Base-CUDA-tier instantiations for parallel Levenshtein distances.
 *  @author Ash Vardanian
 *  @date March 23, 2025
 */
#include "stringzillas/similarities.cuh"

namespace ashvardanian {
namespace stringzillas {
template struct levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
template struct levenshtein_distances<affine_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
template struct levenshtein_distances_utf8<linear_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
} // namespace stringzillas
} // namespace ashvardanian
