/**
 *  @file c/stringzillas/smith_waterman_cuda.cu
 *  @brief Base-CUDA-tier instantiations for parallel Smith-Waterman local alignment scores.
 *  @author Ash Vardanian
 *  @date March 23, 2025
 */
#include "stringzillas/similarities.cuh"

namespace ashvardanian {
namespace stringzillas {
template struct cuda_weighted_scores<linear_gap_costs_t, ualloc_t, sz_similarity_local_k, sz_cap_cuda_k>;
template struct cuda_weighted_scores<affine_gap_costs_t, ualloc_t, sz_similarity_local_k, sz_cap_cuda_k>;
} // namespace stringzillas
} // namespace ashvardanian
