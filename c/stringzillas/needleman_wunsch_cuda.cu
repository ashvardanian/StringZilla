/**
 *  @file c/stringzillas/needleman_wunsch_cuda.cu
 *  @brief Base-CUDA-tier instantiations for parallel Needleman-Wunsch global alignment scores.
 *  @author Ash Vardanian
 *  @date March 23, 2025
 */
#include "stringzillas/similarities.cuh"

namespace ashvardanian {
namespace stringzillas {
template struct cuda_weighted_scores<linear_gap_costs_t, ualloc_t, sz_similarity_global_k, sz_cap_cuda_k>;
template struct cuda_weighted_scores<affine_gap_costs_t, ualloc_t, sz_similarity_global_k, sz_cap_cuda_k>;
} // namespace stringzillas
} // namespace ashvardanian
